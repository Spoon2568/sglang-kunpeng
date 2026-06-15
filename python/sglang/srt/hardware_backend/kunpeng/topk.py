"""Grouped TopK for MoE routing using kunpeng async_compute operators."""

from typing import Optional, Tuple

import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)


def grouped_topk_forward(
    router_logits: torch.Tensor,
    topk: int,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
    renormalize: bool,
    scoring_func_sigmoid: bool = True,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped TopK routing for MoE (kunpeng backend).

    This implementation follows KunPengDistInfer architecture where shared experts
    are computed independently and NOT included in topk routing.

    Args:
        router_logits: [M, N_experts] BF16 or FP32
        topk: number of routed experts to select per token (must NOT include shared experts)
        num_expert_group: number of expert groups
        topk_group: number of experts to select per group
        renormalize: whether to renormalize topk weights
        scoring_func_sigmoid: True for sigmoid (DeepSeek V3/R1), False for softmax
        bias: [N_experts] optional correction bias

    Returns:
        token_weights: [M, topk] float32 - weights for routed experts only
        token_ids: [M, topk] int32 - IDs for routed experts only

    Note:
        Kunpeng backend requires --disable-shared-experts-fusion flag.
        Shared experts are computed separately in DeepseekV2MoE.forward_normal().
    """
    assert num_expert_group is not None and topk_group is not None

    load_async_compute()

    M, N = router_logits.shape

    weights = torch.empty((M, topk), dtype=torch.float32, device=router_logits.device)
    # C++ grouped_topk_out requires token_ids to be int16; convert to int32
    # afterwards for downstream consumers (MoE apply expects int32/int64 ids).
    ids_i16 = torch.empty((M, topk), dtype=torch.int16, device=router_logits.device)

    torch.ops.async_compute.grouped_topk_out(
        router_logits,
        weights,
        ids_i16,
        topk,
        num_expert_group,
        topk_group,
        scoring_func_sigmoid,
        renormalize,
        bias,
    )
    return weights, ids_i16.to(torch.int32)
