"""Unit tests for MLA INT8 batched GEMM (UK / UV projections).

Tests the real production functions _batched_gemm_uk / _batched_gemm_uv from
sglang/srt/hardware_backend/kunpeng/quantization/w8a8_int8.py against a BF16
baseline, using real DeepSeek-R1 MLA shapes.

DeepSeek-R1 MLA dims:
    qk_nope_head_dim = 128   (K for uk)
    kv_lora_rank     = 512   (N for uk, N for uv)
    v_head_dim       = 128   (V for uv)
    num_heads        = 128   (B; with TP=8 -> num_local_heads=16)

Call shapes (from forward_mla.py):
    uk:  act=q_nope     [M, B, K]   weight=w_kc [B, K, N]   scale=w_kc_scale [B, K, 1]
    uv:  act=attn_out   [M, B, N]   weight=w_vc [B, N, V]   scale=w_vc_scale [B, V, 1]

Usage:
    export KUNPENG_ASYNC_COMPUTE_SO=/path/to/async_compute_op.so
    python test/kunpeng_mla_int8_bmm_test.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    _batched_gemm_uk,
    _batched_gemm_uv,
    load_async_compute,
)

# Real DeepSeek-R1 MLA dimensions.
QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
V_HEAD_DIM = 128
# num_local_heads for a few TP configs (128 total heads).
TP8_HEADS = 16
TP4_HEADS = 32


def _quantize_weight_per_out_channel(w_bf16: torch.Tensor):
    """Quantize a [B, out_dim, in_dim] BF16 weight to INT8 with per-out-channel
    scale [B, out_dim, 1], matching channel-wise INT8 used by DeepSeek-R1.
    """
    absmax = w_bf16.abs().amax(dim=-1, keepdim=True)  # [B, out_dim, 1]
    scale = (absmax / 127.0).clamp(min=1e-8)
    w_q = (w_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    return w_q, scale.to(torch.float32)


def _bf16_close(out, ref, atol, rtol, msg=""):
    out_f = out.to(torch.float32)
    ref_f = ref.to(torch.float32)
    max_abs = (out_f - ref_f).abs().max().item()
    denom = ref_f.abs().max().item() + 1e-6
    max_rel = max_abs / denom
    ok = max_abs <= atol or max_rel <= rtol
    return ok, f"{msg} max_abs={max_abs:.4f} max_rel={max_rel:.4f}"


class TestMLAUKProjection(unittest.TestCase):
    """_batched_gemm_uk: q_nope [M,B,K] x w_kc [B,K,N] -> [B,M,N]."""

    def _run(self, M, B, K=QK_NOPE_HEAD_DIM, N=KV_LORA_RANK, atol=2.0, rtol=0.08):
        torch.manual_seed(0)
        # act: [M, B, K] BF16
        act = torch.randn(M, B, K, dtype=torch.bfloat16)

        # weight stored as [B, K, N] (forward_mla layout), quantize over the
        # "in" dim. _batched_gemm_uk permutes to [B, N, K] then quantizes per
        # out-channel N inside KUTACC, so to build a matching BF16 ref we keep a
        # float copy of the same int8 weight.
        w_kc_bf16 = torch.randn(B, K, N, dtype=torch.bfloat16) * 0.1
        # Quantize on the [B, N, K] layout (out=N) to mirror weight_scale [B,K,1]?
        # forward_mla passes w_kc_scale shaped [B, K, 1]; the kernel treats the
        # permuted weight [B,N,K] with cscale over K. Build int8 weight + scale
        # consistent with how the kernel reads them.
        # We quantize the ORIGINAL [B,K,N] per-K-channel to get scale [B,K,1].
        absmax_k = w_kc_bf16.abs().amax(dim=-1, keepdim=True)  # [B, K, 1]
        w_kc_scale = (absmax_k / 127.0).clamp(min=1e-8).to(torch.float32)
        w_kc_q = (w_kc_bf16 / w_kc_scale).round().clamp(-127, 127).to(torch.int8)

        out = _batched_gemm_uk(act, w_kc_q, w_kc_scale)

        # Shape: [B, M, N]
        self.assertEqual(tuple(out.shape), (B, M, N))
        self.assertEqual(out.dtype, torch.bfloat16)

        # BF16 reference: dequantize weight, bmm.
        w_deq = w_kc_q.to(torch.float32) * w_kc_scale  # [B, K, N]
        act_t = act.transpose(0, 1).to(torch.float32)  # [B, M, K]
        ref = torch.bmm(act_t, w_deq)  # [B, M, N]

        ok, detail = _bf16_close(out, ref.to(torch.bfloat16), atol, rtol, msg=f"[M={M},B={B}]")
        self.assertTrue(ok, f"UK projection failed: {detail}")
        print(f"  UK [M={M},B={B},K={K},N={N}]: {detail}")

    def test_uk_decode_tp8(self):
        """Decode: M=1 token, TP=8 (16 heads)."""
        print("\n=== UK: decode M=1, TP=8 ===")
        self._run(M=1, B=TP8_HEADS)

    def test_uk_batch_decode_tp8(self):
        """Batched decode: M=8 tokens, TP=8."""
        print("\n=== UK: batched decode M=8, TP=8 ===")
        self._run(M=8, B=TP8_HEADS)

    def test_uk_tp4(self):
        """Decode: M=4 tokens, TP=4 (32 heads)."""
        print("\n=== UK: M=4, TP=4 ===")
        self._run(M=4, B=TP4_HEADS)


class TestMLAUVProjection(unittest.TestCase):
    """_batched_gemm_uv: attn_out [M,B,N] x w_vc [B,N,V] -> [B,M,V]."""

    def _run(self, M, B, N=KV_LORA_RANK, V=V_HEAD_DIM, atol=2.0, rtol=0.08):
        torch.manual_seed(1)
        # act: [M, B, N] BF16
        act = torch.randn(M, B, N, dtype=torch.bfloat16)

        # weight stored as [B, N, V] (forward_mla layout). w_vc_scale is [B,V,1].
        w_vc_bf16 = torch.randn(B, N, V, dtype=torch.bfloat16) * 0.1
        # Quantize per-V-channel to get scale [B, V, 1]. The kernel permutes
        # weight to [B, V, N] and uses rscale over V.
        w_vc_perm = w_vc_bf16.permute(0, 2, 1).contiguous()  # [B, V, N]
        absmax_v = w_vc_perm.abs().amax(dim=-1, keepdim=True)  # [B, V, 1]
        w_vc_scale = (absmax_v / 127.0).clamp(min=1e-8).to(torch.float32)
        w_vc_perm_q = (w_vc_perm / w_vc_scale).round().clamp(-127, 127).to(torch.int8)
        # Store back in [B, N, V] layout (kernel permutes internally).
        w_vc_q = w_vc_perm_q.permute(0, 2, 1).contiguous()

        out = _batched_gemm_uv(act, w_vc_q, w_vc_scale)

        # Shape: [B, M, V]
        self.assertEqual(tuple(out.shape), (B, M, V))
        self.assertEqual(out.dtype, torch.bfloat16)

        # BF16 reference: dequantize weight on [B,V,N], bmm.
        w_deq = w_vc_perm_q.to(torch.float32) * w_vc_scale  # [B, V, N]
        act_t = act.transpose(0, 1).to(torch.float32)  # [B, M, N]
        ref = torch.bmm(act_t, w_deq.transpose(1, 2))  # [B, M, V]

        ok, detail = _bf16_close(out, ref.to(torch.bfloat16), atol, rtol, msg=f"[M={M},B={B}]")
        self.assertTrue(ok, f"UV projection failed: {detail}")
        print(f"  UV [M={M},B={B},N={N},V={V}]: {detail}")

    def test_uv_decode_tp8(self):
        """Decode: M=1 token, TP=8 (16 heads)."""
        print("\n=== UV: decode M=1, TP=8 ===")
        self._run(M=1, B=TP8_HEADS)

    def test_uv_batch_decode_tp8(self):
        """Batched decode: M=8 tokens, TP=8."""
        print("\n=== UV: batched decode M=8, TP=8 ===")
        self._run(M=8, B=TP8_HEADS)

    def test_uv_tp4(self):
        """Decode: M=4 tokens, TP=4 (32 heads)."""
        print("\n=== UV: M=4, TP=4 ===")
        self._run(M=4, B=TP4_HEADS)


class TestMLAShapeContract(unittest.TestCase):
    """Verify output shapes match the forward_mla.py expectations."""

    def test_uk_output_shape(self):
        print("\n=== UK output shape contract ===")
        M, B = 4, TP8_HEADS
        act = torch.randn(M, B, QK_NOPE_HEAD_DIM, dtype=torch.bfloat16)
        w = torch.randint(-127, 127, (B, QK_NOPE_HEAD_DIM, KV_LORA_RANK), dtype=torch.int8)
        s = torch.ones(B, QK_NOPE_HEAD_DIM, 1, dtype=torch.float32) * 0.01
        out = _batched_gemm_uk(act, w, s)
        # forward_mla does out.transpose(0,1) -> [M, B, N]
        self.assertEqual(tuple(out.shape), (B, M, KV_LORA_RANK))
        self.assertEqual(tuple(out.transpose(0, 1).shape), (M, B, KV_LORA_RANK))
        print(f"  UK out [B,M,N]={tuple(out.shape)}, transposed [M,B,N]={tuple(out.transpose(0,1).shape)} ✓")

    def test_uv_output_shape(self):
        print("\n=== UV output shape contract ===")
        M, B = 4, TP8_HEADS
        act = torch.randn(M, B, KV_LORA_RANK, dtype=torch.bfloat16)
        w = torch.randint(-127, 127, (B, KV_LORA_RANK, V_HEAD_DIM), dtype=torch.int8)
        s = torch.ones(B, V_HEAD_DIM, 1, dtype=torch.float32) * 0.01
        out = _batched_gemm_uv(act, w, s)
        # forward_mla does out.transpose(0,1).flatten(1,2) -> [M, B*V]
        self.assertEqual(tuple(out.shape), (B, M, V_HEAD_DIM))
        flat = out.transpose(0, 1).flatten(1, 2)
        self.assertEqual(tuple(flat.shape), (M, B * V_HEAD_DIM))
        print(f"  UV out [B,M,V]={tuple(out.shape)}, flattened [M,B*V]={tuple(flat.shape)} ✓")


if __name__ == "__main__":
    load_async_compute()
    unittest.main(verbosity=2)
