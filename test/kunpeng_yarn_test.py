"""Tests for kunpeng YaRN RoPE operators.

Run:
    export KUNPENG_ASYNC_COMPUTE_SO=/path/to/async_compute_op.so
    export SGLANG_USE_KUNPENG_W8A8=1
    python test/kunpeng_yarn_test.py
"""

import os
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)

torch.manual_seed(42)


def _bf16_close(out, ref, atol, rtol, msg=""):
    """Check if bf16 tensors are close (with tolerance)."""
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    rel_diff = diff / (ref.float().abs() + 1e-8)
    max_rel = rel_diff.max().item()

    ok = (diff <= atol + rtol * ref.float().abs()).all().item()
    detail = f"{msg} max_abs={max_abs:.6f} max_rel={max_rel:.6f}"
    return ok, detail


def _compute_reference_yarn_cache(
    dim, max_pos, base, scaling_factor, beta_fast=32, beta_slow=1,
    extrapolation_factor=1.0, mscale=1.0, mscale_all_dim=0.0, attn_factor=1.0
):
    """Compute YaRN cache using PyTorch (matching C++ implementation).

    Returns cache in C++ format: [cos..., sin...] (not interleaved).
    """
    import math

    # YaRN correction range
    low = math.floor(
        dim * math.log(max_pos / (beta_fast * 2 * math.pi)) / (2 * math.log(base))
    )
    high = math.ceil(
        dim * math.log(max_pos / (beta_slow * 2 * math.pi)) / (2 * math.log(base))
    )
    low = max(low, 0.0)
    high = min(high, dim / 2.0 - 1.0)

    # theta_scale: base^(-2/dim)
    theta_scale = 1.0 / (base ** (2.0 / dim))

    # mscale calculation (matching C++ yarn_get_mscale)
    def yarn_get_mscale(scale, msc):
        if scale <= 1.0:
            return 1.0
        return 0.1 * msc * math.log(scale) + 1.0

    real_mscale = (
        yarn_get_mscale(scaling_factor, mscale) /
        yarn_get_mscale(scaling_factor, mscale_all_dim) *
        attn_factor
    )

    # Allocate cache
    cos_sin = torch.empty(max_pos, dim, dtype=torch.float32)

    for pid in range(max_pos):
        interpolation = pid / scaling_factor
        extrapolation = float(pid)

        for i in range(dim // 2):
            # Compute mask (1 - linear_ramp) * extrapolation_factor
            if high > low:
                ramp = (i - low) / (high - low)
            else:
                ramp = 0.0
            ramp = max(0.0, min(1.0, ramp))
            mask = (1.0 - ramp) * extrapolation_factor

            # theta = mix of interpolation and extrapolation
            theta = (1.0 - mask) * interpolation + mask * extrapolation

            # Store cos and sin
            cos_sin[pid, i] = math.cos(theta) * real_mscale
            cos_sin[pid, dim // 2 + i] = math.sin(theta) * real_mscale

            # Update for next frequency
            extrapolation *= theta_scale
            interpolation *= theta_scale

    return cos_sin.to(torch.bfloat16)


def _apply_rotary_emb_reference(q, k, cos_sin_cache, position_ids):
    """Apply RoPE using PyTorch (reference implementation).

    cos_sin_cache format: [cos[0:dim//2], sin[0:dim//2]] (C++ format).
    """
    # q, k: [total_tokens, num_heads, head_dim]
    # position_ids: [batch, seq_len] or [total_tokens]
    # cos_sin_cache: [max_pos, head_dim], first half is cos, second half is sin

    if position_ids.dim() == 2:
        position_ids = position_ids.flatten()

    total_tokens, num_heads, head_dim = q.shape
    assert position_ids.numel() == total_tokens

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    for i in range(total_tokens):
        pos = int(position_ids[i])
        cos_sin = cos_sin_cache[pos].float()  # [head_dim]

        # Extract cos and sin (C++ format: first half cos, second half sin)
        cos = cos_sin[: head_dim // 2]  # [head_dim//2]
        sin = cos_sin[head_dim // 2 :]  # [head_dim//2]

        for h in range(num_heads):
            # Rotate q and k
            q_vec = q[i, h].float()
            k_vec = k[i, h].float()

            # Odd-even grouping: [x0, x1, x2, x3, ...] -> [(x0, x1), (x2, x3), ...]
            q_real = q_vec[0::2]  # [head_dim//2]
            q_imag = q_vec[1::2]  # [head_dim//2]
            k_real = k_vec[0::2]
            k_imag = k_vec[1::2]

            # Complex rotation: (real, imag) * (cos, sin)
            q_out[i, h, 0::2] = (q_real * cos - q_imag * sin).to(torch.bfloat16)
            q_out[i, h, 1::2] = (q_real * sin + q_imag * cos).to(torch.bfloat16)
            k_out[i, h, 0::2] = (k_real * cos - k_imag * sin).to(torch.bfloat16)
            k_out[i, h, 1::2] = (k_real * sin + k_imag * cos).to(torch.bfloat16)

    return q_out, k_out


class TestYaRNInitCache(unittest.TestCase):
    """Test yarn_init_cache_forward correctness."""

    def test_small_cache(self):
        """Small cache: dim=64, max_pos=128, scaling_factor=2.0"""
        from sglang.srt.hardware_backend.kunpeng.rope import yarn_init_cache_forward

        dim = 64
        max_pos = 128
        base = 10000.0
        scaling_factor = 2.0

        cache = yarn_init_cache_forward(
            dim, max_pos, base, scaling_factor, beta_fast=32, beta_slow=1
        )

        # Compute reference
        cache_ref = _compute_reference_yarn_cache(dim, max_pos, base, scaling_factor)

        # Compare
        ok, detail = _bf16_close(cache, cache_ref, atol=0.01, rtol=0.01, msg="small")
        self.assertTrue(ok, detail)

    def test_deepseek_r1_config(self):
        """DeepSeek-R1 config: dim=64, max_pos=163840 (160k), scaling_factor=40.0"""
        from sglang.srt.hardware_backend.kunpeng.rope import yarn_init_cache_forward

        dim = 64
        max_pos = 2048  # Use smaller for testing speed
        base = 10000.0
        scaling_factor = 4.0  # Scaled down from 40.0

        cache = yarn_init_cache_forward(
            dim, max_pos, base, scaling_factor, beta_fast=32, beta_slow=1
        )

        cache_ref = _compute_reference_yarn_cache(dim, max_pos, base, scaling_factor)

        ok, detail = _bf16_close(cache, cache_ref, atol=0.01, rtol=0.01, msg="r1")
        self.assertTrue(ok, detail)

    def test_large_dim(self):
        """Large head_dim: dim=128, max_pos=512"""
        from sglang.srt.hardware_backend.kunpeng.rope import yarn_init_cache_forward

        dim = 128
        max_pos = 512
        base = 10000.0
        scaling_factor = 2.0

        cache = yarn_init_cache_forward(
            dim, max_pos, base, scaling_factor, beta_fast=32, beta_slow=1
        )

        cache_ref = _compute_reference_yarn_cache(dim, max_pos, base, scaling_factor)

        ok, detail = _bf16_close(cache, cache_ref, atol=0.01, rtol=0.01, msg="large")
        self.assertTrue(ok, detail)


class TestYaRNForward(unittest.TestCase):
    """Test yarn_forward correctness."""

    def test_single_token(self):
        """Single token: batch=1, seq_len=1, num_heads=8, head_dim=64"""
        from sglang.srt.hardware_backend.kunpeng.rope import (
            yarn_forward,
            yarn_init_cache_forward,
        )

        num_heads = 8
        head_dim = 64
        max_pos = 128

        # Initialize cache
        cache = yarn_init_cache_forward(
            head_dim, max_pos, base=10000.0, scaling_factor=2.0
        )

        # Single token at position 10
        q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16)
        position_ids = torch.tensor([[10]], dtype=torch.int64)

        # Apply RoPE
        q_out, k_out = yarn_forward(q, k, position_ids, cache)

        # Reference
        q_ref, k_ref = _apply_rotary_emb_reference(q, k, cache, position_ids)

        # Compare
        ok_q, detail_q = _bf16_close(q_out, q_ref, atol=0.02, rtol=0.02, msg="q")
        ok_k, detail_k = _bf16_close(k_out, k_ref, atol=0.02, rtol=0.02, msg="k")
        self.assertTrue(ok_q, detail_q)
        self.assertTrue(ok_k, detail_k)

    def test_batch_sequence(self):
        """Batch with multiple tokens: batch=4, seq_len=16, num_heads=16, head_dim=64"""
        from sglang.srt.hardware_backend.kunpeng.rope import (
            yarn_forward,
            yarn_init_cache_forward,
        )

        batch = 4
        seq_len = 16
        num_heads = 16
        head_dim = 64
        max_pos = 512

        cache = yarn_init_cache_forward(
            head_dim, max_pos, base=10000.0, scaling_factor=2.0
        )

        # Flatten batch
        total_tokens = batch * seq_len
        q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16)

        # Position IDs: [0, 1, ..., 15], [0, 1, ..., 15], ...
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch, 1)  # [4, 16]

        q_out, k_out = yarn_forward(q, k, position_ids, cache)

        # Reference
        q_ref, k_ref = _apply_rotary_emb_reference(q, k, cache, position_ids)

        ok_q, detail_q = _bf16_close(q_out, q_ref, atol=0.02, rtol=0.02, msg="batch_q")
        ok_k, detail_k = _bf16_close(k_out, k_ref, atol=0.02, rtol=0.02, msg="batch_k")
        self.assertTrue(ok_q, detail_q)
        self.assertTrue(ok_k, detail_k)

    def test_long_context(self):
        """Long context: position up to 8192"""
        from sglang.srt.hardware_backend.kunpeng.rope import (
            yarn_forward,
            yarn_init_cache_forward,
        )

        num_heads = 8
        head_dim = 64
        max_pos = 16384
        test_pos = 8192

        cache = yarn_init_cache_forward(
            head_dim, max_pos, base=10000.0, scaling_factor=4.0
        )

        # Single token at high position
        q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16)
        position_ids = torch.tensor([[test_pos]], dtype=torch.int64)

        q_out, k_out = yarn_forward(q, k, position_ids, cache)

        # Reference
        q_ref, k_ref = _apply_rotary_emb_reference(q, k, cache, position_ids)

        ok_q, detail_q = _bf16_close(q_out, q_ref, atol=0.02, rtol=0.02, msg="long_q")
        ok_k, detail_k = _bf16_close(k_out, k_ref, atol=0.02, rtol=0.02, msg="long_k")
        self.assertTrue(ok_q, detail_q)
        self.assertTrue(ok_k, detail_k)


if __name__ == "__main__":
    load_async_compute()
    unittest.main(verbosity=2)
