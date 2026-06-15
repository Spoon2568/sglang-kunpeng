"""Correctness tests for kunpeng async_compute operators used in SGLang.

Covers four replaced operator groups:
  1. Router BF16 GEMM      -> sglang/srt/hardware_backend/kunpeng/gemm.py
  2. Grouped TopK (MoE)    -> sglang/srt/hardware_backend/kunpeng/topk.py
  3. RMSNorm               -> sglang/srt/hardware_backend/kunpeng/norm.py
  4. Argmax (greedy)       -> sglang/srt/hardware_backend/kunpeng/argmax.py

These tests require the kunpeng aarch64 hardware and the compiled
async_compute_op.so. They are NOT part of the CUDA/CPU CI suite.

Run on the compute node:
    ssh cn22863
    source ~/sibow/init.sh
    source /home/share/fengguangnan/sibow/llminfer/sglang-kunpeng/.venv/bin/activate
    export KUNPENG_ASYNC_COMPUTE_SO=/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so
    cd /home/share/fengguangnan/sibow/llminfer/sglang-kunpeng
    python -m pytest test/kunpeng_operators_test.py -v
    # or:  python test/kunpeng_operators_test.py
"""

import os
import unittest

import torch
import torch.nn.functional as F

# Operators only run on bf16-capable aarch64 (kunpeng). The .so path comes from
# KUNPENG_ASYNC_COMPUTE_SO; load_async_compute() reads it lazily.
from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)

torch.manual_seed(0)

# DeepSeek-R1 router config (see KunPengDistInfer context.cpp).
R1_N_ROUTED_EXPERTS = 256
R1_N_GROUPS = 8
R1_TOPK_GROUP = 4
R1_TOPK = 8


def _bf16_close(out, ref, atol, rtol, msg=""):
    """Compare in fp32 with bf16-appropriate tolerances."""
    out_f = out.to(torch.float32)
    ref_f = ref.to(torch.float32)
    max_abs = (out_f - ref_f).abs().max().item()
    denom = ref_f.abs().max().item() + 1e-6
    max_rel = max_abs / denom
    ok = torch.allclose(out_f, ref_f, atol=atol, rtol=rtol)
    detail = f"{msg} max_abs={max_abs:.5f} max_rel={max_rel:.5f}"
    return ok, detail


class TestRouterBF16GEMM(unittest.TestCase):
    """router_gemm_forward: out[M,N] = hidden[M,H] @ weight[N,H]^T (BF16)."""

    def _run_case(self, M, H, N, atol=0.5, rtol=0.05):
        from sglang.srt.hardware_backend.kunpeng.gemm import router_gemm_forward

        hidden = torch.randn(M, H, dtype=torch.bfloat16)
        weight = torch.randn(N, H, dtype=torch.bfloat16)

        out = router_gemm_forward(hidden, weight)

        self.assertEqual(tuple(out.shape), (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)

        ref = hidden.to(torch.float32) @ weight.to(torch.float32).t()
        ok, detail = _bf16_close(out, ref, atol, rtol, msg=f"[M={M},H={H},N={N}]")
        self.assertTrue(ok, detail)

    def test_router_shape_deepseek(self):
        # DeepSeek-R1 router: hidden=7168, 256 experts. M must be %16==0 (bgemm).
        self._run_case(M=16, H=7168, N=R1_N_ROUTED_EXPERTS, atol=1.0, rtol=0.05)

    def test_router_small(self):
        self._run_case(M=16, H=512, N=64)

    def test_router_larger_batch(self):
        self._run_case(M=64, H=2048, N=128, atol=0.8, rtol=0.05)


class _FakeRMSNorm:
    """Minimal stand-in exposing the attributes rmsnorm_forward_kunpeng reads."""

    def __init__(self, hidden_size, eps=1e-6):
        self.weight = torch.nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.variance_epsilon = eps


def _ref_rmsnorm(x, weight, eps, residual=None, post=None):
    """fp32 reference matching RMSNorm.forward_native semantics."""
    xf = x.to(torch.float32)
    res_out = None
    if residual is not None:
        xf = xf + residual.to(torch.float32)
        if post is not None:
            xf = xf + post.to(torch.float32)
        res_out = xf.to(x.dtype)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    out = (xf * weight.to(torch.float32)).to(x.dtype)
    return out, res_out


class TestRMSNorm(unittest.TestCase):
    """rmsnorm_forward_kunpeng vs fp32 forward_native reference."""

    H = 1024
    EPS = 1e-6

    def setUp(self):
        from sglang.srt.hardware_backend.kunpeng.norm import (
            rmsnorm_forward_kunpeng,
        )

        self.fn = rmsnorm_forward_kunpeng
        self.norm = _FakeRMSNorm(self.H, self.EPS)

    def test_no_residual(self):
        x = torch.randn(32, self.H, dtype=torch.bfloat16)
        out = self.fn(self.norm, x.clone())
        ref, _ = _ref_rmsnorm(x, self.norm.weight, self.EPS)
        ok, detail = _bf16_close(out, ref, atol=0.05, rtol=0.02, msg="no_residual")
        self.assertTrue(ok, detail)

    def test_with_residual(self):
        x = torch.randn(32, self.H, dtype=torch.bfloat16)
        residual = torch.randn(32, self.H, dtype=torch.bfloat16)
        out, res_out = self.fn(self.norm, x.clone(), residual.clone())
        ref, ref_res = _ref_rmsnorm(x, self.norm.weight, self.EPS, residual=residual)
        ok, detail = _bf16_close(out, ref, atol=0.05, rtol=0.02, msg="resid_out")
        self.assertTrue(ok, detail)
        ok2, detail2 = _bf16_close(res_out, ref_res, atol=0.05, rtol=0.02, msg="resid_acc")
        self.assertTrue(ok2, detail2)

    def test_with_post_residual_addition(self):
        x = torch.randn(32, self.H, dtype=torch.bfloat16)
        residual = torch.randn(32, self.H, dtype=torch.bfloat16)
        post = torch.randn(32, self.H, dtype=torch.bfloat16)
        out, res_out = self.fn(self.norm, x.clone(), residual.clone(), post.clone())
        ref, ref_res = _ref_rmsnorm(
            x, self.norm.weight, self.EPS, residual=residual, post=post
        )
        ok, detail = _bf16_close(out, ref, atol=0.05, rtol=0.02, msg="post_out")
        self.assertTrue(ok, detail)
        ok2, detail2 = _bf16_close(res_out, ref_res, atol=0.05, rtol=0.02, msg="post_resid")
        self.assertTrue(ok2, detail2)

    def test_non_contiguous_input(self):
        # Guards the bug where a non-contiguous x was copied but the original
        # (unmodified) tensor was returned.
        base = torch.randn(32, self.H * 2, dtype=torch.bfloat16)
        x = base[:, : self.H]  # non-contiguous view
        self.assertFalse(x.is_contiguous())
        out = self.fn(self.norm, x)
        ref, _ = _ref_rmsnorm(x.contiguous(), self.norm.weight, self.EPS)
        ok, detail = _bf16_close(out, ref, atol=0.05, rtol=0.02, msg="non_contig")
        self.assertTrue(ok, detail)


def _ref_grouped_topk(
    router_logits,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
    scoring_func_sigmoid,
    bias=None,
):
    """Pure-torch reference for grouped_topk (matches the C++ algorithm).

    Returns (weights[M,topk] fp32, ids[M,topk] int64) with ids sorted ascending
    per token (the C++ kernel also sorts selected experts ascending).
    """
    logits = router_logits.to(torch.float32)
    M, E = logits.shape
    group_size = E // num_expert_group

    if scoring_func_sigmoid:
        origin = torch.sigmoid(logits)
    else:
        origin = torch.softmax(logits, dim=-1)

    scores = origin.clone()
    if bias is not None:
        scores = scores + bias.to(torch.float32).view(1, -1)

    grouped = scores.view(M, num_expert_group, group_size)
    if bias is not None:
        # group score = top-2 sum within group (C++ uses sorted[0]+sorted[1])
        top2 = grouped.topk(2, dim=-1).values
        group_score = top2.sum(dim=-1)
    else:
        group_score = grouped.max(dim=-1).values

    # select topk_group groups, mask out the rest
    grp_idx = group_score.topk(topk_group, dim=-1).indices  # [M, topk_group]
    group_mask = torch.zeros(M, num_expert_group, dtype=torch.bool)
    group_mask.scatter_(1, grp_idx, True)
    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, group_size).reshape(M, E)

    masked_scores = scores.masked_fill(~expert_mask, float("-inf"))
    sel = masked_scores.topk(topk, dim=-1).indices  # [M, topk]
    sel, _ = torch.sort(sel, dim=-1)  # ascending, matching C++

    weights = torch.gather(origin, 1, sel)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, sel


def _pairs(ids, weights):
    """Set of (expert_id, rounded_weight) per token, order-independent."""
    out = []
    for i in range(ids.shape[0]):
        row = sorted(
            (int(ids[i, j]), round(float(weights[i, j]), 3))
            for j in range(ids.shape[1])
        )
        out.append(row)
    return out


class TestGroupedTopK(unittest.TestCase):
    """grouped_topk_forward vs pure-torch reference."""

    def _run(self, M, scoring_func_sigmoid, use_bias, renormalize):
        from sglang.srt.hardware_backend.kunpeng.topk import grouped_topk_forward

        E = R1_N_ROUTED_EXPERTS
        logits = torch.randn(M, E, dtype=torch.bfloat16)
        bias = (
            torch.randn(E, dtype=torch.float32) * 0.1 if use_bias else None
        )

        weights, ids = grouped_topk_forward(
            router_logits=logits,
            topk=R1_TOPK,
            num_expert_group=R1_N_GROUPS,
            topk_group=R1_TOPK_GROUP,
            renormalize=renormalize,
            scoring_func_sigmoid=scoring_func_sigmoid,
            bias=bias,
        )

        self.assertEqual(tuple(ids.shape), (M, R1_TOPK))
        self.assertEqual(tuple(weights.shape), (M, R1_TOPK))

        ref_w, ref_ids = _ref_grouped_topk(
            logits,
            R1_TOPK,
            R1_N_GROUPS,
            R1_TOPK_GROUP,
            renormalize,
            scoring_func_sigmoid,
            bias,
        )

        # Weights must match for the selected experts. Allow minor expert set
        # differences when softmax scores are very close (C++ softmax_fusion_kernel
        # vs PyTorch softmax can differ in float rounding for tie-breaking).
        for i in range(M):
            got_ids_set = set(int(v) for v in ids[i].tolist())
            exp_ids_set = set(int(v) for v in ref_ids[i].tolist())

            # Check weight sum (should be ~1.0 if renormalized)
            got_sum = sum(float(weights[i, j]) for j in range(R1_TOPK))
            exp_sum = sum(float(ref_w[i, j]) for j in range(R1_TOPK))
            self.assertAlmostEqual(
                got_sum, exp_sum, places=2, msg=f"token {i} weight sum mismatch"
            )

            # Allow up to 2 expert mismatches per token (softmax float precision)
            mismatch_count = len(got_ids_set.symmetric_difference(exp_ids_set))
            if mismatch_count > 2:
                self.fail(
                    f"token {i}: too many expert mismatches ({mismatch_count})\n"
                    f"  got: {sorted(got_ids_set)}\n"
                    f"  exp: {sorted(exp_ids_set)}"
                )

    def test_sigmoid_with_bias(self):
        # DeepSeek-R1 path: sigmoid scoring + correction bias.
        self._run(M=8, scoring_func_sigmoid=True, use_bias=True, renormalize=True)

    def test_sigmoid_no_renorm(self):
        self._run(M=8, scoring_func_sigmoid=True, use_bias=True, renormalize=False)

    def test_softmax_no_bias(self):
        self._run(M=8, scoring_func_sigmoid=False, use_bias=False, renormalize=True)

    def test_ids_dtype_int(self):
        # token_ids must be an integer dtype usable as expert indices downstream.
        from sglang.srt.hardware_backend.kunpeng.topk import grouped_topk_forward

        logits = torch.randn(4, R1_N_ROUTED_EXPERTS, dtype=torch.bfloat16)
        _, ids = grouped_topk_forward(
            router_logits=logits,
            topk=R1_TOPK,
            num_expert_group=R1_N_GROUPS,
            topk_group=R1_TOPK_GROUP,
            renormalize=True,
            scoring_func_sigmoid=True,
            bias=torch.zeros(R1_N_ROUTED_EXPERTS, dtype=torch.float32),
        )
        self.assertIn(ids.dtype, (torch.int16, torch.int32, torch.int64))
        self.assertTrue((ids >= 0).all() and (ids < R1_N_ROUTED_EXPERTS).all())


class TestEmbedding(unittest.TestCase):
    """embedding_forward vs torch.nn.functional.embedding reference."""

    def test_embedding_tp1(self):
        """TP=1: full vocab, no sharding."""
        from sglang.srt.hardware_backend.kunpeng.embedding import embedding_forward

        vocab_size = 1024
        hidden = 512
        n_tokens = 16

        weight = torch.randn(vocab_size, hidden, dtype=torch.bfloat16)
        input_ids = torch.randint(0, vocab_size, (n_tokens,), dtype=torch.int64)

        out = embedding_forward(input_ids, weight, vocab_start=0, vocab_end=vocab_size)
        ref = F.embedding(input_ids, weight)

        ok, detail = _bf16_close(out, ref, atol=0.01, rtol=0.01, msg="tp1")
        self.assertTrue(ok, detail)

    def test_embedding_tp_sharded(self):
        """TP sharded: vocab split across ranks, only local shard accessed."""
        from sglang.srt.hardware_backend.kunpeng.embedding import embedding_forward

        vocab_size = 1024
        shard_size = vocab_size // 4  # simulate TP=4
        hidden = 512
        n_tokens = 16

        # Simulate rank 1's shard: vocab [256, 512)
        vocab_start = 256
        vocab_end = 512
        weight_shard = torch.randn(shard_size, hidden, dtype=torch.bfloat16)

        # Token IDs within this shard's range
        input_ids = torch.randint(vocab_start, vocab_end, (n_tokens,), dtype=torch.int64)

        out = embedding_forward(input_ids, weight_shard, vocab_start, vocab_end)

        # Reference: adjust input_ids to local indices [0, shard_size)
        input_ids_local = input_ids - vocab_start
        ref = F.embedding(input_ids_local, weight_shard)

        ok, detail = _bf16_close(out, ref, atol=0.01, rtol=0.01, msg="tp_sharded")
        self.assertTrue(ok, detail)

    def test_embedding_dtype_conversion(self):
        """Ensure int32 input_ids are converted to int64."""
        from sglang.srt.hardware_backend.kunpeng.embedding import embedding_forward

        vocab_size = 512
        hidden = 256
        weight = torch.randn(vocab_size, hidden, dtype=torch.bfloat16)
        input_ids = torch.randint(0, vocab_size, (8,), dtype=torch.int32)  # int32

        out = embedding_forward(input_ids, weight, 0, vocab_size)
        ref = F.embedding(input_ids.to(torch.int64), weight)

        ok, detail = _bf16_close(out, ref, atol=0.01, rtol=0.01, msg="dtype_conv")
        self.assertTrue(ok, detail)


class TestArgmax(unittest.TestCase):
    """argmax_forward vs torch.argmax reference for greedy sampling."""

    def test_argmax_small_vocab(self):
        """Small vocab size: (8, 32000) - typical greedy sampling batch."""
        from sglang.srt.hardware_backend.kunpeng.argmax import argmax_forward

        batch_size = 8
        vocab_size = 32000

        logits = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16)

        out = argmax_forward(logits)
        ref = torch.argmax(logits, dim=-1)

        self.assertEqual(tuple(out.shape), (batch_size,))
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(torch.equal(out, ref), f"Argmax indices mismatch:\nout={out}\nref={ref}")

    def test_argmax_large_vocab(self):
        """Large vocab size: (16, 256000) - DeepSeek-R1 scale."""
        from sglang.srt.hardware_backend.kunpeng.argmax import argmax_forward

        batch_size = 16
        vocab_size = 256000

        logits = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16)

        out = argmax_forward(logits)
        ref = torch.argmax(logits, dim=-1)

        self.assertEqual(tuple(out.shape), (batch_size,))
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(torch.equal(out, ref), f"Argmax indices mismatch:\nout={out}\nref={ref}")

    def test_argmax_single_batch(self):
        """Single batch element: (1, 50000)."""
        from sglang.srt.hardware_backend.kunpeng.argmax import argmax_forward

        logits = torch.randn(1, 50000, dtype=torch.bfloat16)

        out = argmax_forward(logits)
        ref = torch.argmax(logits, dim=-1)

        self.assertEqual(tuple(out.shape), (1,))
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(torch.equal(out, ref))

    def test_argmax_values_match(self):
        """Verify both index and value are correct."""
        from sglang.srt.hardware_backend.kunpeng.argmax import argmax_forward

        batch_size = 4
        vocab_size = 1024

        logits = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16)

        out_indices = argmax_forward(logits)
        ref_indices = torch.argmax(logits, dim=-1)

        # Check indices match
        self.assertTrue(torch.equal(out_indices, ref_indices))

        # Check that the selected values are indeed the maximum
        for i in range(batch_size):
            idx = out_indices[i].item()
            selected_value = logits[i, idx].item()
            max_value = logits[i].max().item()
            self.assertAlmostEqual(
                selected_value,
                max_value,
                places=3,
                msg=f"Batch {i}: selected value {selected_value} != max {max_value}",
            )

    def test_argmax_dtype_conversion(self):
        """Ensure fp32 input is converted to bf16."""
        from sglang.srt.hardware_backend.kunpeng.argmax import argmax_forward

        logits = torch.randn(4, 1000, dtype=torch.float32)

        out = argmax_forward(logits)
        ref = torch.argmax(logits.to(torch.bfloat16), dim=-1)

        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(torch.equal(out, ref))

    def test_argmax_non_contiguous(self):
        """Handle non-contiguous input tensors."""
        from sglang.srt.hardware_backend.kunpeng.argmax import argmax_forward

        # Create non-contiguous tensor via slicing
        base = torch.randn(8, 10000, dtype=torch.bfloat16)
        logits = base[:, ::2]  # Non-contiguous view
        self.assertFalse(logits.is_contiguous())

        out = argmax_forward(logits)
        ref = torch.argmax(logits, dim=-1)

        self.assertEqual(tuple(out.shape), (8,))
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(torch.equal(out, ref))

    def test_argmax_range_validation(self):
        """Ensure output indices are within valid vocab range."""
        from sglang.srt.hardware_backend.kunpeng.argmax import argmax_forward

        batch_size = 32
        vocab_size = 128000

        logits = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16)

        out = argmax_forward(logits)

        self.assertTrue((out >= 0).all(), "Negative indices found")
        self.assertTrue((out < vocab_size).all(), f"Indices exceed vocab_size {vocab_size}")


class TestMulScalarAdd(unittest.TestCase):
    """mul_scalar_add_forward vs torch reference (x += y * alpha)."""

    def _run_case(self, M, H, alpha, atol=0.05, rtol=0.02):
        from sglang.srt.hardware_backend.kunpeng.elementwise import (
            mul_scalar_add_forward,
        )

        x = torch.randn(M, H, dtype=torch.bfloat16)
        y = torch.randn(M, H, dtype=torch.bfloat16)

        x_test = x.clone()
        x_ref = x.clone()

        # Kunpeng operator: x += y * alpha (in-place)
        out = mul_scalar_add_forward(x_test, y, alpha)

        # Reference: x += y * alpha
        x_ref.add_(y, alpha=alpha)

        # Verify in-place modification
        self.assertTrue(out is x_test, "mul_scalar_add_forward should return x")

        ok, detail = _bf16_close(
            out, x_ref, atol, rtol, msg=f"[M={M},H={H},alpha={alpha}]"
        )
        self.assertTrue(ok, detail)

    def test_small_alpha_0p125(self):
        """DeepSeek-R1 routed_scaling_factor = 0.125"""
        self._run_case(M=16, H=1024, alpha=0.125)

    def test_medium_shape_alpha_0p5(self):
        self._run_case(M=64, H=4096, alpha=0.5)

    def test_large_shape_alpha_1p0(self):
        """alpha=1.0: simple add without scaling"""
        self._run_case(M=128, H=7168, alpha=1.0)

    def test_alpha_2p0(self):
        """alpha > 1.0: scaling up"""
        self._run_case(M=32, H=2048, alpha=2.0)

    def test_deepseek_r1_shape(self):
        """Realistic DeepSeek-R1 shape: hidden=7168, routed_scaling_factor=0.125"""
        self._run_case(M=64, H=7168, alpha=0.125, atol=0.05, rtol=0.02)

    def test_negative_alpha(self):
        """Negative alpha: x += y * (-0.5)"""
        self._run_case(M=16, H=512, alpha=-0.5)

    def test_zero_alpha(self):
        """alpha=0: x should remain unchanged"""
        from sglang.srt.hardware_backend.kunpeng.elementwise import (
            mul_scalar_add_forward,
        )

        M, H = 16, 1024
        x = torch.randn(M, H, dtype=torch.bfloat16)
        y = torch.randn(M, H, dtype=torch.bfloat16)

        x_orig = x.clone()
        out = mul_scalar_add_forward(x, y, alpha=0.0)

        ok, detail = _bf16_close(out, x_orig, atol=0.01, rtol=0.01, msg="alpha=0")
        self.assertTrue(ok, detail)


if __name__ == "__main__":
    load_async_compute()
    unittest.main(verbosity=2)

