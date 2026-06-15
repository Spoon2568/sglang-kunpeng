import os

import torch
from torch.nn import Parameter

ASYNC_COMPUTE_SO = os.environ.get(
    "KUNPENG_ASYNC_COMPUTE_SO",
    "/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so",
)

_loaded = False


def use_kunpeng_w8a8() -> bool:
    return os.environ.get("SGLANG_USE_KUNPENG_W8A8", "0") == "1"


def load_async_compute():
    global _loaded
    if not _loaded:
        torch.ops.load_library(ASYNC_COMPUTE_SO)
        _loaded = True


def workspace_bytes(m: int, n: int, k: int) -> int:
    return max(m * n * k * 2, 1024)


def process_linear_weight(layer):
    # async_compute.igemm_bdq_out expects weight as [N, K], not [K, N].
    layer.weight = Parameter(layer.weight.data.contiguous(), requires_grad=False)
    layer.weight_scale = Parameter(
        layer.weight_scale.data.reshape(-1).contiguous(), requires_grad=False
    )


def apply_linear(layer, x: torch.Tensor, bias=None):
    load_async_compute()

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])

    if x_2d.dtype != torch.bfloat16:
        x_2d = x_2d.to(torch.bfloat16)

    m, k = x_2d.shape
    n = layer.weight.shape[0]

    x_q = torch.empty((m, k), dtype=torch.int8, device=x_2d.device)
    # quant_out 要求 scale 为 2D (m, 1)；igemm_bdq_out 同样要求 2D。
    x_scale = torch.empty((m, 1), dtype=torch.float32, device=x_2d.device)
    torch.ops.async_compute.quant_out(x_2d, x_q, x_scale)

    weight_packed = torch.empty_like(layer.weight)
    torch.ops.async_compute.igemm_pack_weight_out(layer.weight, m, weight_packed)

    x_q_packed = torch.empty_like(x_q)
    torch.ops.async_compute.igemm_pack_act_out(x_q, weight_packed, x_q_packed)

    out = torch.empty((m, n), dtype=torch.bfloat16, device=x_2d.device)
    workspace = torch.empty(
        workspace_bytes(m, n, k), dtype=torch.uint8, device=x_2d.device
    )

    torch.ops.async_compute.igemm_bdq_out(
        x_q_packed,
        weight_packed,
        layer.weight_scale,
        x_scale,
        out,
        workspace,
    )

    if bias is not None:
        out = out + bias

    return out.reshape(*orig_shape[:-1], n)


# moe

FUSEDMOE_TILEBUF = 64


def _pack_moe_weight(weight: torch.Tensor) -> torch.Tensor:
    packed = torch.empty_like(weight)
    for expert_id in range(weight.shape[0]):
        torch.ops.async_compute.igemm_pack_weight_out(
            weight[expert_id],
            FUSEDMOE_TILEBUF,
            packed[expert_id],
        )
    return packed


def process_moe_weight(layer):
    load_async_compute()

    w13_weight = layer.w13_weight.data.contiguous()
    w2_weight = layer.w2_weight.data.contiguous()

    layer.w13_weight = Parameter(_pack_moe_weight(w13_weight), requires_grad=False)
    layer.w2_weight = Parameter(_pack_moe_weight(w2_weight), requires_grad=False)

    layer.w13_weight_scale = Parameter(
        layer.w13_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
    )
    layer.w2_weight_scale = Parameter(
        layer.w2_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
    )


def apply_moe(layer, dispatch_output, moe_runner_config):
    load_async_compute()

    if moe_runner_config.activation != "silu":
        raise NotImplementedError(
            "Kunpeng W8A8 MoE currently only supports silu activation."
        )
    if moe_runner_config.apply_router_weight_on_input:
        raise NotImplementedError(
            "Kunpeng W8A8 MoE does not support apply_router_weight_on_input yet."
        )
    if moe_runner_config.no_combine:
        raise NotImplementedError("Kunpeng W8A8 MoE does not support no_combine yet.")

    x = dispatch_output.hidden_states
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    topk_output = dispatch_output.topk_output
    if not hasattr(topk_output, "topk_ids"):
        if hasattr(topk_output, "to_standard"):
            topk_output = topk_output.to_standard(moe_runner_config.layer_id)
        else:
            raise RuntimeError("Kunpeng W8A8 MoE requires StandardTopKOutput.")

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights.to(torch.float32)

    num_tokens, hidden_size = x.shape
    num_experts = layer.w13_weight.shape[0]
    gateup_n = layer.w13_weight.shape[1]
    intermediate_size = gateup_n // 2
    topk = topk_ids.shape[1]

    if num_tokens == 0:
        return torch.empty_like(x).reshape(*orig_shape[:-1], hidden_size)

    flat_token_ids = torch.arange(
        num_tokens, device=x.device, dtype=torch.int64
    ).repeat_interleave(topk)
    flat_expert_ids = topk_ids.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1)

    valid = (flat_expert_ids >= 0) & (flat_expert_ids < num_experts)
    if not bool(valid.all()):
        raise RuntimeError(
            "Kunpeng W8A8 MoE currently expects all topk_ids to be local valid "
            f"expert ids in [0, {num_experts})."
        )

    order = torch.argsort(flat_expert_ids, stable=True)
    sorted_token_ids = flat_token_ids[order].contiguous()
    sorted_expert_ids = flat_expert_ids[order].contiguous()
    sorted_weights = flat_weights[order].contiguous()

    routed_tokens = sorted_token_ids.numel()

    counts = torch.bincount(sorted_expert_ids, minlength=num_experts).to(torch.int32)
    experts_offset = torch.empty(num_experts + 1, dtype=torch.int32, device=x.device)
    experts_offset[0] = 0
    experts_offset[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)

    x_q = torch.empty((num_tokens, hidden_size), dtype=torch.int8, device=x.device)
    # quant_out 要求 scale 为 2D (num_tokens, 1)。
    x_scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)
    torch.ops.async_compute.quant_out(x, x_q, x_scale)

    acts_and_scale = torch.empty(
        (num_tokens, hidden_size + 4), dtype=torch.uint8, device=x.device
    )
    torch.ops.async_compute.act_scale_pack_out(x_q, x_scale, acts_and_scale)

    sorted_token_ids_i32 = sorted_token_ids.to(torch.int32).contiguous()

    gateup_out = torch.empty(
        (routed_tokens, gateup_n), dtype=torch.bfloat16, device=x.device
    )
    gateup_pbx = torch.empty(
        FUSEDMOE_TILEBUF * hidden_size, dtype=torch.int8, device=x.device
    )
    gateup_pby = torch.empty(
        FUSEDMOE_TILEBUF * gateup_n * 2, dtype=torch.float32, device=x.device
    )
    pbsc = torch.empty(FUSEDMOE_TILEBUF, dtype=torch.float32, device=x.device)

    torch.ops.async_compute.igemm_fusedmoe_gateup_out(
        acts_and_scale,
        layer.w13_weight,
        layer.w13_weight_scale,
        sorted_token_ids_i32,
        experts_offset,
        gateup_out,
        gateup_pbx,
        gateup_pby,
        pbsc,
    )

    act_q = torch.empty(
        (routed_tokens, intermediate_size), dtype=torch.int8, device=x.device
    )
    act_scale = torch.empty((routed_tokens,), dtype=torch.float32, device=x.device)
    torch.ops.async_compute.silu_mul_quant_out(gateup_out, act_q, act_scale)

    down_out = torch.empty(
        (routed_tokens, hidden_size), dtype=torch.bfloat16, device=x.device
    )
    down_pbx = torch.empty(
        FUSEDMOE_TILEBUF * intermediate_size, dtype=torch.int8, device=x.device
    )
    down_pby = torch.empty(
        FUSEDMOE_TILEBUF * hidden_size, dtype=torch.float32, device=x.device
    )
    torch.ops.async_compute.igemm_fusedmoe_down_out(
        act_q,
        layer.w2_weight,
        act_scale,
        layer.w2_weight_scale,
        sorted_token_ids_i32,
        experts_offset,
        down_out,
        down_pbx,
        down_pby,
        pbsc,
    )

    down_out = down_out * sorted_weights.to(down_out.dtype).view(-1, 1)

    output = torch.zeros(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=x.device
    )
    output.index_add_(0, sorted_token_ids, down_out)

    # routed_scaling_factor 由外层 DeepseekV2MoE.forward_normal() 统一应用，
    # 这里不重复乘，避免 double scaling。
    return output.reshape(*orig_shape[:-1], hidden_size)


# mla int8 bmm
#
# 对照 KunPengDistInfer csrc/model/deepseek.cpp decode 路径：
#
#   uk (q_nope × w_kc):
#     batched_gemm_pack_allthreads(transpose(q_nope, 0, 1), packed_act)
#     batched_gemm_woqs8_allthreads(packed_act, uk, {}, uk_scale, out)
#     → uk         [B, N, K] INT8, uk_scale [B, K, 1] float32 (cscale)
#
#   uv (attn_out × w_vc):
#     batched_gemm_pack_allthreads(transpose(attn_out, 0, 1), packed_act)
#     batched_gemm_woqs8_allthreads(packed_act, uv, uv_scale, {}, out)
#     → uv         [B, V, N] INT8, uv_scale [B, V, 1] float32 (rscale)
#
# batched_gemm_woqs8 的 weight 布局是 [B, out_dim, in_dim]。
# SGLang post_load_weights 存储:
#   w_kc [B, K, N], w_kc_scale [B, K, 1]
#   w_vc [B, N, V], w_vc_scale [B, V, 1]
# 两个都需要 permute 到 [B, out_dim, in_dim] 才能传给 batched_gemm_woqs8。


_BATCHED_GEMM_M_ALIGNMENT = 32


def _pad_batched_gemm_act(act_t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad [B, M, K] activation on M to satisfy KUTACC tile constraints."""
    _, M, K = act_t.shape
    aligned_m = (
        (M + _BATCHED_GEMM_M_ALIGNMENT - 1)
        // _BATCHED_GEMM_M_ALIGNMENT
        * _BATCHED_GEMM_M_ALIGNMENT
    )
    if aligned_m == M:
        return act_t, M

    padded = torch.zeros(
        (act_t.shape[0], aligned_m, K),
        dtype=act_t.dtype,
        device=act_t.device,
    )
    padded[:, :M, :].copy_(act_t)
    return padded, M


def _pack_batched_gemm_tensor(x: torch.Tensor) -> torch.Tensor:
    packed = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    torch.ops.async_compute.batched_gemm_pack_allthreads_out(x, packed)
    return packed


def _batched_gemm_uk(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """q_nope × w_kc: INT8 batched GEMM (对应 KunPengDistInfer 的 uk)。

    Args:
        act: [M, B, K]  BF16, K=qk_nope_head_dim
        weight: [B, K, N]  INT8, N=kv_lora_rank
        weight_scale: [B, K, 1]  float32 (cscale)
    Returns:
        [B, M, N]  BF16
    """
    load_async_compute()

    B = act.shape[1]
    N = weight.shape[-1]

    # act: [M, B, K] → transpose → [B, M, K] → pack
    act_t = act.transpose(0, 1).contiguous()
    if act_t.dtype != torch.bfloat16:
        act_t = act_t.to(torch.bfloat16)
    act_t, orig_M = _pad_batched_gemm_act(act_t)
    packed_act = _pack_batched_gemm_tensor(act_t)

    # weight: [B, K, N] → [B, N, K]  (batched_gemm_woqs8 的 out×in 布局)
    weight_t = weight.permute(0, 2, 1).contiguous()
    packed_weight = _pack_batched_gemm_tensor(weight_t)

    out = torch.empty((B, act_t.shape[1], N), dtype=out_dtype, device=act.device)
    torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
        packed_act, packed_weight, torch.Tensor(), weight_scale, out
    )
    if orig_M != out.shape[1]:
        out = out[:, :orig_M, :].contiguous()
    return out


def _batched_gemm_uv(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """attn_out × w_vc: INT8 batched GEMM (对应 KunPengDistInfer 的 uv)。

    Args:
        act: [M, B, N]  BF16, N=kv_lora_rank
        weight: [B, N, V]  INT8  (post_load_weights 存储的实际布局)
        weight_scale: [B, V, 1]  float32 (rscale)
    Returns:
        [B, M, V]  BF16
    """
    load_async_compute()

    B = act.shape[1]
    V = weight.shape[-1]

    # act: [M, B, N] → transpose → [B, M, N] → pack
    act_t = act.transpose(0, 1).contiguous()
    if act_t.dtype != torch.bfloat16:
        act_t = act_t.to(torch.bfloat16)
    act_t, orig_M = _pad_batched_gemm_act(act_t)
    packed_act = _pack_batched_gemm_tensor(act_t)

    # weight: [B, N, V] → [B, V, N]  (batched_gemm_woqs8 的 out×in 布局)
    weight_t = weight.permute(0, 2, 1).contiguous()
    packed_weight = _pack_batched_gemm_tensor(weight_t)

    out = torch.empty((B, act_t.shape[1], V), dtype=out_dtype, device=act.device)
    torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
        packed_act, packed_weight, weight_scale, torch.Tensor(), out
    )
    if orig_M != out.shape[1]:
        out = out[:, :orig_M, :].contiguous()
    return out
