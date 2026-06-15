"""YaRN RoPE (Rotary Position Embedding) using kunpeng async_compute operators.

YaRN (Yet another RoPE extensioN method) is DeepSeek's method for extending
context length in transformer models. This module provides kunpeng-optimized
implementations of YaRN RoPE operations.

References:
    - YaRN paper: https://arxiv.org/abs/2309.00071
    - DeepSeek-V2: Uses YaRN for extended context (32k -> 128k)
"""

import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)


def yarn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply YaRN RoPE to query and key tensors.

    Args:
        q: Query tensor [total_tokens, num_heads, head_dim], bf16
        k: Key tensor [total_tokens, num_heads, head_dim], bf16
        position_ids: Position indices [batch, seq_len], int64
            Must have position_ids.numel() == total_tokens
        cos_sin_cache: Precomputed cos/sin cache [max_position, head_dim], bf16
            Created by yarn_init_cache_forward()

    Returns:
        (q_out, k_out): Rotated query and key tensors, same shapes as input, bf16

    Notes:
        - Applies YaRN's position-dependent frequency scaling
        - cos_sin_cache must be precomputed with yarn_init_cache_forward
        - All tensors must be on CPU (kunpeng operators are CPU-only)
    """
    load_async_compute()

    # Validate and convert types
    assert q.device.type == "cpu", "yarn_forward requires CPU tensors"
    assert k.device.type == "cpu", "yarn_forward requires CPU tensors"
    assert position_ids.device.type == "cpu", "yarn_forward requires CPU tensors"
    assert cos_sin_cache.device.type == "cpu", "yarn_forward requires CPU tensors"

    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    if k.dtype != torch.bfloat16:
        k = k.to(torch.bfloat16)
    if position_ids.dtype != torch.int64:
        position_ids = position_ids.to(torch.int64)
    if cos_sin_cache.dtype != torch.bfloat16:
        cos_sin_cache = cos_sin_cache.to(torch.bfloat16)

    # Ensure contiguous (C++ requires contiguous memory)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not position_ids.is_contiguous():
        position_ids = position_ids.contiguous()
    if not cos_sin_cache.is_contiguous():
        cos_sin_cache = cos_sin_cache.contiguous()

    # Allocate output tensors
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Call kunpeng yarn_out
    torch.ops.async_compute.yarn_out(q, k, q_out, k_out, position_ids, cos_sin_cache)

    return q_out, k_out


def yarn_init_cache_forward(
    dim: int,
    max_position_embeddings: int,
    base: float,
    scaling_factor: float,
    beta_fast: int = 32,
    beta_slow: int = 1,
    extrapolation_factor: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    attn_factor: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Initialize YaRN cos/sin cache with frequency scaling.

    This computes the YaRN-scaled cos/sin values for all positions up to
    max_position_embeddings. Should be called once during model initialization.

    Args:
        dim: Head dimension (must be positive and even)
        max_position_embeddings: Maximum sequence length to support
        base: RoPE base frequency (e.g., 10000.0 for standard RoPE)
        scaling_factor: YaRN scaling factor (e.g., 4.0 for 4x context extension)
        beta_fast: Fast frequency threshold for YaRN (default: 32)
        beta_slow: Slow frequency threshold for YaRN (default: 1)
        extrapolation_factor: Extrapolation factor (default: 1.0)
        mscale: Magnitude scaling factor (default: 1.0)
        mscale_all_dim: Alternative mscale parameter (default: 0.0)
        attn_factor: Attention factor (default: 1.0)
        device: Device to allocate cache on (must be CPU)

    Returns:
        cos_sin_cache: [max_position_embeddings, dim] bf16 tensor
            Contains interleaved [cos[0], sin[0], cos[1], sin[1], ...]

    Notes:
        - This is called once during model initialization
        - The cache is reused for all forward passes
        - YaRN applies different scaling to different frequency bands
        - DeepSeek-R1 typical params: dim=64, base=10000, scaling_factor=4.0
    """
    load_async_compute()

    assert device.type == "cpu", "yarn_init_cache_forward requires CPU device"
    assert dim > 0 and dim % 2 == 0, "dim must be positive and even"
    assert max_position_embeddings > 0, "max_position_embeddings must be positive"
    assert base > 0.0 and base != 1.0, "base must be positive and not 1.0"
    assert scaling_factor > 0.0, "scaling_factor must be positive"
    assert beta_fast > 0 and beta_slow > 0, "beta_fast and beta_slow must be positive"

    # Allocate cache (bf16, CPU)
    cos_sin_cache = torch.empty(
        (max_position_embeddings, dim),
        dtype=torch.bfloat16,
        device=device,
    )

    # Call kunpeng yarn_init_cache_out (in-place fills cos_sin_cache)
    torch.ops.async_compute.yarn_init_cache_out(
        dim,
        base,
        max_position_embeddings,
        scaling_factor,
        cos_sin_cache,
        beta_fast,
        beta_slow,
        extrapolation_factor,
        mscale,
        mscale_all_dim,
        attn_factor,
    )

    return cos_sin_cache
