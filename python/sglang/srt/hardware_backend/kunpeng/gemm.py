"""BF16 GEMM operations using kunpeng async_compute operators."""

import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
    workspace_bytes,
)


def router_gemm_forward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """BF16 router GEMM: hidden_states × weight^T.

    Args:
        hidden_states: [M, H] BF16
        weight: [N_experts, H] BF16

    Returns:
        [M, N_experts] float32 (router logits)
    """
    load_async_compute()

    M, H = hidden_states.shape
    N = weight.shape[0]

    packed_weight = torch.empty_like(weight)
    torch.ops.async_compute.bgemm_pack_weight_out(weight, M, packed_weight)

    packed_act = torch.empty_like(hidden_states)
    torch.ops.async_compute.bgemm_pack_out(hidden_states, packed_weight, packed_act)

    out = torch.empty((M, N), dtype=torch.bfloat16, device=hidden_states.device)
    ws = torch.empty(
        workspace_bytes(M, N, H), dtype=torch.uint8, device=hidden_states.device
    )
    torch.ops.async_compute.bgemm_out(packed_act, packed_weight, out, ws)
    return out
