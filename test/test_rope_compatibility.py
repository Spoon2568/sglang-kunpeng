"""Compatibility test: SGLang DeepseekScalingRotaryEmbedding vs kunpeng yarn ops.

Goal: prove kunpeng YaRN can replace SGLang's RoPE for DeepSeek-R1 inference.

The test is decomposed into two independent concerns so a failure pinpoints
the exact cause:

  Test A (rotation style): feed BOTH implementations the SAME cache, compare
    q_out/k_out. This isolates the neox/gptj rotation math from cache compute.

  Test B (cache compute): build each side's cache with the REAL DeepSeek-R1
    YaRN params and compare element-wise. This isolates the YaRN frequency /
    correction-range math.

DeepSeek-R1 real config (config.json rope_scaling):
    type=yarn, factor=40, beta_fast=32, beta_slow=1,
    mscale=1.0, mscale_all_dim=1.0, original_max_position_embeddings=4096
    qk_rope_head_dim=64, rope_theta=10000
    is_neox_style = not rope_interleave(default True) = False  (GPTJ)

MUST run with SGLANG_USE_CPU_ENGINE=1 so _is_cpu is True at import time,
otherwise RotaryEmbedding.__init__ tries `from vllm._custom_ops import ...`
(a GPU-only path) and fails on ARM CPU.

Run:
    export SGLANG_USE_CPU_ENGINE=1
    export SGLANG_USE_KUNPENG_W8A8=1
    export KUNPENG_ASYNC_COMPUTE_SO=/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so
    python test/test_rope_compatibility.py
"""

import unittest

import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)
from sglang.srt.hardware_backend.kunpeng.rope import (
    yarn_forward,
    yarn_init_cache_forward,
)
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)

# --- DeepSeek-R1 real YaRN config ---
ROTARY_DIM = 64  # qk_rope_head_dim; full-dim rotation (rotary_dim == head_size)
BASE = 10000
SCALING_FACTOR = 40.0
ORIGINAL_MAX_POS = 4096
BETA_FAST = 32
BETA_SLOW = 1
MSCALE = 1.0
MSCALE_ALL_DIM = 1.0  # NOTE: 1.0 (not 0.0) — real config value
IS_NEOX = False  # GPTJ / interleaved


def _stats(out, ref):
    out_f, ref_f = out.float(), ref.float()
    diff = (out_f - ref_f).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref_f.abs() + 1e-6)).max().item()
    return max_abs, max_rel


def _make_sglang_rope(max_position_embeddings):
    """Build the SGLang DeepSeek YaRN rope with real params on CPU."""
    return DeepseekScalingRotaryEmbedding(
        head_size=ROTARY_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=max_position_embeddings,
        base=BASE,
        is_neox_style=IS_NEOX,
        scaling_factor=SCALING_FACTOR,
        dtype=torch.bfloat16,
        extrapolation_factor=1.0,
        attn_factor=1.0,
        beta_fast=BETA_FAST,
        beta_slow=BETA_SLOW,
        mscale=MSCALE,
        mscale_all_dim=MSCALE_ALL_DIM,
        device="cpu",
    )


class TestRotationStyle(unittest.TestCase):
    """Test A: same cache fed to both → isolates rotation (neox vs gptj)."""

    def test_rotation_with_shared_cache(self):
        # Build sglang rope; reuse ITS cache for both sides so only rotation
        # math is under test (cache compute is held identical).
        sglang_rope = _make_sglang_rope(ORIGINAL_MAX_POS)
        shared_cache = sglang_rope.cos_sin_cache.to(torch.bfloat16).contiguous()

        total_tokens = 32
        num_heads = 8
        torch.manual_seed(0)
        q = torch.randn(total_tokens, num_heads, ROTARY_DIM, dtype=torch.bfloat16)
        k = torch.randn(total_tokens, num_heads, ROTARY_DIM, dtype=torch.bfloat16)
        positions = torch.arange(total_tokens, dtype=torch.int64)

        # SGLang rotation (forward_native: pure PyTorch, no vllm/kernel).
        q_sg = q.view(total_tokens, num_heads * ROTARY_DIM).clone()
        k_sg = k.view(total_tokens, num_heads * ROTARY_DIM).clone()
        # forward_native expects [tokens, num_heads, head_size]; pass 3D.
        q_sglang, k_sglang = sglang_rope.forward_native(
            positions,
            q.clone(),
            k.clone(),
        )

        # Kunpeng rotation with the SAME cache.
        q_kp, k_kp = yarn_forward(
            q.clone(), k.clone(), positions.unsqueeze(0), shared_cache
        )

        ma_q, mr_q = _stats(q_kp, q_sglang)
        ma_k, mr_k = _stats(k_kp, k_sglang)
        print("\n[Test A] rotation with shared cache (GPTJ)")
        print(f"  q max_abs={ma_q:.5f} max_rel={mr_q:.5f}")
        print(f"  k max_abs={ma_k:.5f} max_rel={mr_k:.5f}")

        self.assertTrue(
            torch.allclose(q_kp.float(), q_sglang.float(), atol=0.03, rtol=0.03),
            f"Q rotation mismatch: max_abs={ma_q:.5f}",
        )
        self.assertTrue(
            torch.allclose(k_kp.float(), k_sglang.float(), atol=0.03, rtol=0.03),
            f"K rotation mismatch: max_abs={ma_k:.5f}",
        )


class TestCacheCompute(unittest.TestCase):
    """Test B: real-param cache compare → isolates YaRN freq/correction-range."""

    def test_cache_matches_sglang(self):
        # SGLang generates ORIGINAL_MAX_POS * SCALING_FACTOR rows; correction
        # range uses ORIGINAL_MAX_POS internally.
        sglang_rope = _make_sglang_rope(ORIGINAL_MAX_POS)
        sglang_cache = sglang_rope.cos_sin_cache  # [orig*scaling, rotary_dim]
        n_rows = sglang_cache.shape[0]

        # Kunpeng: single max_position_embeddings arg drives BOTH row count and
        # correction range. Pass n_rows to match row count.
        kunpeng_cache = yarn_init_cache_forward(
            dim=ROTARY_DIM,
            max_position_embeddings=n_rows,
            base=float(BASE),
            scaling_factor=SCALING_FACTOR,
            beta_fast=BETA_FAST,
            beta_slow=BETA_SLOW,
            extrapolation_factor=1.0,
            mscale=MSCALE,
            mscale_all_dim=MSCALE_ALL_DIM,
            attn_factor=1.0,
        )

        print("\n[Test B] cache compute with real DeepSeek-R1 params")
        print(f"  sglang cache shape: {tuple(sglang_cache.shape)}")
        print(f"  kunpeng cache shape: {tuple(kunpeng_cache.shape)}")

        # Compare a representative slice (position 0..1023) to keep it fast.
        rows = min(1024, n_rows)
        sg = sglang_cache[:rows].float()
        kp = kunpeng_cache[:rows].float()
        ma, mr = _stats(kp, sg)
        print(f"  cache[:{rows}] max_abs={ma:.5f} max_rel={mr:.5f}")
        print(f"  sglang cache[0,:5]={sg[0,:5].tolist()}")
        print(f"  kunpeng cache[0,:5]={kp[0,:5].tolist()}")
        # Position 1 row, first few freqs — most sensitive to correction range.
        print(f"  sglang cache[1,:5]={sg[1,:5].tolist()}")
        print(f"  kunpeng cache[1,:5]={kp[1,:5].tolist()}")

        self.assertTrue(
            torch.allclose(kp, sg, atol=0.03, rtol=0.03),
            f"Cache mismatch (likely correction-range max_pos conflation): "
            f"max_abs={ma:.5f} max_rel={mr:.5f}",
        )


if __name__ == "__main__":
    load_async_compute()
    unittest.main(verbosity=2)
