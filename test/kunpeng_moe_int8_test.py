"""Unit tests for MoE INT8 quantization (real apply_moe path, DeepSeek-R1 shapes).

This test drives the PRODUCTION code path: sglang's
hardware_backend.kunpeng.quantization.w8a8_int8.apply_moe, exactly as
W8A8Int8MoEMethod.apply calls it. It does NOT re-implement the kernel
sequence by hand — so it protects the path real inference uses.

Shapes follow DeepSeek-R1:
    hidden_size           = 7168
    intermediate_size     = 2048
    num_experts (test)    = 8     (real model: 256; reduced for test speed,
                                   but hidden/intermediate kept real because
                                   those are the dims KUTACC kernels tune for)
    topk                  = 8

Usage:
    export KUNPENG_ASYNC_COMPUTE_SO=/path/to/async_compute_op.so
    python test/kunpeng_moe_int8_test.py
    # or: python -m pytest test/kunpeng_moe_int8_test.py -v
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from torch.nn import Parameter

# Real production code under test.
from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    apply_moe,
    load_async_compute,
    process_moe_weight,
)

# DeepSeek-R1 dims.
HIDDEN = 7168
INTERMEDIATE = 2048
TOPK = 8


def _make_layer(num_experts, hidden, intermediate):
    """Build a layer whose w13/w2 INT8 weights + scales match what
    W8A8Int8MoEMethod.create_weights produces, then run the real
    process_moe_weight to pack them (production prep path)."""
    gateup_n = 2 * intermediate

    w13 = torch.randint(
        -127, 127, (num_experts, gateup_n, hidden), dtype=torch.int8
    )
    w2 = torch.randint(
        -127, 127, (num_experts, hidden, intermediate), dtype=torch.int8
    )
    # Small positive per-channel scales (realistic INT8 dequant magnitude).
    w13_scale = torch.rand(num_experts, gateup_n, 1, dtype=torch.float32) * 0.01 + 1e-3
    w2_scale = torch.rand(num_experts, hidden, 1, dtype=torch.float32) * 0.01 + 1e-3

    layer = SimpleNamespace(
        w13_weight=Parameter(w13, requires_grad=False),
        w2_weight=Parameter(w2, requires_grad=False),
        w13_weight_scale=Parameter(w13_scale, requires_grad=False),
        w2_weight_scale=Parameter(w2_scale, requires_grad=False),
    )
    # Production weight prep: packs weights + squeezes scales in place.
    process_moe_weight(layer)
    return layer


def _make_dispatch_output(x, topk_ids, topk_weights):
    """Construct the StandardDispatchOutput / StandardTopKOutput that
    W8A8Int8MoEMethod.apply hands to apply_moe."""
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )

    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.empty(0),
    )
    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=topk_output,
    )


def _runner_config(num_experts):
    """Minimal MoeRunnerConfig with the fields apply_moe reads."""
    return SimpleNamespace(
        activation="silu",
        apply_router_weight_on_input=False,
        no_combine=False,
        layer_id=0,
        num_experts=num_experts,
        top_k=TOPK,
        routed_scaling_factor=None,
    )


def _ref_moe_bf16(x, layer_raw, topk_ids, topk_weights, num_experts):
    """fp32/bf16 reference: dequantize weights, run dense per-expert MoE.

    layer_raw holds the ORIGINAL (unpacked) int8 weights + scales, since
    process_moe_weight mutates the layer in place. We pass raw tensors here.
    """
    w13, w13_s, w2, w2_s = layer_raw
    num_tokens = x.shape[0]
    out = torch.zeros(num_tokens, x.shape[1], dtype=torch.float32)

    xf = x.to(torch.float32)
    for t in range(num_tokens):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            wgt = float(topk_weights[t, j])
            w13_e = w13[e].to(torch.float32) * w13_s[e]  # [2I, H]
            w2_e = w2[e].to(torch.float32) * w2_s[e]      # [H, I]
            gateup = xf[t] @ w13_e.t()                    # [2I]
            gate, up = gateup.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up     # [I]
            out[t] += wgt * (act @ w2_e.t())              # [H]
    return out


class TestMoEINT8Real(unittest.TestCase):
    def setUp(self):
        load_async_compute()
        torch.manual_seed(42)

    def _run(self, num_experts, num_tokens, check_numeric=False):
        x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16)

        # Keep raw weights for the reference BEFORE process_moe_weight packs them.
        gateup_n = 2 * INTERMEDIATE
        w13 = torch.randint(-127, 127, (num_experts, gateup_n, HIDDEN), dtype=torch.int8)
        w2 = torch.randint(-127, 127, (num_experts, HIDDEN, INTERMEDIATE), dtype=torch.int8)
        w13_s = torch.rand(num_experts, gateup_n, 1, dtype=torch.float32) * 0.01 + 1e-3
        w2_s = torch.rand(num_experts, HIDDEN, 1, dtype=torch.float32) * 0.01 + 1e-3

        layer = SimpleNamespace(
            w13_weight=Parameter(w13.clone(), requires_grad=False),
            w2_weight=Parameter(w2.clone(), requires_grad=False),
            w13_weight_scale=Parameter(w13_s.clone(), requires_grad=False),
            w2_weight_scale=Parameter(w2_s.clone(), requires_grad=False),
        )
        process_moe_weight(layer)

        topk_ids = torch.randint(0, num_experts, (num_tokens, TOPK), dtype=torch.int64)
        topk_weights = torch.rand(num_tokens, TOPK, dtype=torch.float32)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        dispatch = _make_dispatch_output(x, topk_ids, topk_weights)
        cfg = _runner_config(num_experts)

        out = apply_moe(layer, dispatch, cfg)

        self.assertEqual(tuple(out.shape), (num_tokens, HIDDEN))
        self.assertFalse(torch.isnan(out).any(), "output has NaN")
        self.assertFalse(torch.isinf(out).any(), "output has Inf")

        if check_numeric:
            # 保持 scale 为 [E, out_dim, 1]，让 _ref_moe_bf16 里 w_s[e] 是
            # [out_dim, 1]，与 [out_dim, H] 正确广播（squeeze 成 1D 会错误地
            # 去广播最后一维）。
            ref = _ref_moe_bf16(
                x, (w13, w13_s, w2, w2_s),
                topk_ids, topk_weights, num_experts,
            )
            out_f = out.to(torch.float32)
            denom = ref.abs().max().item() + 1e-6
            max_rel = (out_f - ref).abs().max().item() / denom
            print(f"  numeric: max_rel={max_rel:.4f}")
            # INT8 MoE through two GEMMs + silu: loose tolerance.
            self.assertLess(max_rel, 0.15, f"INT8 vs BF16 rel error too high: {max_rel}")
        return out

    def test_small_batch(self):
        """num_tokens < TILEBUF (64): hierarchical/true path."""
        print("\n=== MoE real shape: small batch (16 tokens, 8 experts) ===")
        out = self._run(num_experts=8, num_tokens=16)
        print(f"  output range: [{out.min():.2f}, {out.max():.2f}]")

    def test_large_batch(self):
        """num_tokens >> TILEBUF: staged path, multi-tile per expert."""
        print("\n=== MoE real shape: large batch (256 tokens, 8 experts) ===")
        out = self._run(num_experts=8, num_tokens=256)
        print(f"  output range: [{out.min():.2f}, {out.max():.2f}]")

    def test_numeric_vs_bf16(self):
        """Numeric accuracy vs dequantized BF16 reference (small batch)."""
        print("\n=== MoE real shape: numeric check (8 tokens, 8 experts) ===")
        self._run(num_experts=8, num_tokens=8, check_numeric=True)

    def test_empty_batch(self):
        """num_tokens == 0: apply_moe early-returns empty."""
        print("\n=== MoE real shape: empty batch ===")
        layer = _make_layer(8, HIDDEN, INTERMEDIATE)
        x = torch.empty(0, HIDDEN, dtype=torch.bfloat16)
        topk_ids = torch.empty(0, TOPK, dtype=torch.int64)
        topk_weights = torch.empty(0, TOPK, dtype=torch.float32)
        dispatch = _make_dispatch_output(x, topk_ids, topk_weights)
        out = apply_moe(layer, dispatch, _runner_config(8))
        self.assertEqual(out.shape[0], 0)
        print("  empty batch handled ✓")


if __name__ == "__main__":
    load_async_compute()
    unittest.main(verbosity=2)
