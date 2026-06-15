"""Test MoE TILEBUF size assumption.

Verifies whether FUSEDMOE_TILEBUF=64 is safe for large batch sizes
by checking if KUTACC kernel reuses the temp buffer or needs larger allocation.

Usage:
    python test/kunpeng_moe_tilebuf_test.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

ASYNC_COMPUTE_SO = os.environ.get(
    "KUNPENG_ASYNC_COMPUTE_SO",
    "/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so",
)


def load_async_compute():
    if not os.path.exists(ASYNC_COMPUTE_SO):
        raise FileNotFoundError(f"async_compute_op.so not found: {ASYNC_COMPUTE_SO}")
    torch.ops.load_library(ASYNC_COMPUTE_SO)


def test_moe_with_large_routed_tokens():
    """Test MoE with routed_tokens >> FUSEDMOE_TILEBUF."""
    load_async_compute()

    # Simulate large batch: 某个 expert 处理 256 tokens (>> 64)
    num_tokens = 256
    num_experts = 4
    hidden_size = 512
    intermediate_size = 2048

    # 模拟所有 token 都路由到 expert 0
    sorted_token_ids = torch.arange(num_tokens, dtype=torch.int32)
    experts_offset = torch.tensor([0, num_tokens, num_tokens, num_tokens, num_tokens], dtype=torch.int32)

    # 量化输入
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    x_absmax = x.abs().max(dim=-1, keepdim=True).values
    x_scale = (x_absmax / 127.0).clamp(min=1e-12).to(torch.float32)
    x_q = (x / x_scale).round().clamp(-127, 127).to(torch.int8)

    # Pack act + scale
    acts_and_scale = torch.empty((num_tokens, hidden_size + 4), dtype=torch.uint8)
    torch.ops.async_compute.act_scale_pack_out(x_q, x_scale.view(-1, 1), acts_and_scale)

    # 权重（随机）
    w13_weight = torch.randint(
        -127, 127, (num_experts, 2 * intermediate_size, hidden_size), dtype=torch.int8
    )
    w13_weight_scale = torch.randn(num_experts, 2 * intermediate_size, 1, dtype=torch.float32).abs() * 0.01

    # Output
    gateup_out = torch.empty((num_tokens, 2 * intermediate_size), dtype=torch.bfloat16)

    # 关键：TILEBUF=64，但 routed_tokens=256
    FUSEDMOE_TILEBUF = 64
    gateup_pbx = torch.empty(FUSEDMOE_TILEBUF * hidden_size, dtype=torch.int8)
    gateup_pby = torch.empty(FUSEDMOE_TILEBUF * 2 * intermediate_size * 2, dtype=torch.float32)
    pbsc = torch.empty(FUSEDMOE_TILEBUF, dtype=torch.float32)

    try:
        torch.ops.async_compute.igemm_fusedmoe_gateup_out(
            acts_and_scale,
            w13_weight,
            w13_weight_scale,
            sorted_token_ids,
            experts_offset,
            gateup_out,
            gateup_pbx,
            gateup_pby,
            pbsc,
        )
        print(f"✅ routed_tokens={num_tokens} (4x TILEBUF) passed without crash")
        print(f"   gateup_out range: [{gateup_out.min():.3f}, {gateup_out.max():.3f}]")

        # 检查输出是否合理
        if torch.isnan(gateup_out).any() or torch.isinf(gateup_out).any():
            print("❌ Output contains NaN or Inf")
            return False

        return True
    except Exception as e:
        print(f"❌ FAILED with routed_tokens={num_tokens}: {e}")
        return False


def main():
    print("=" * 60)
    print("MoE TILEBUF Size Test")
    print("=" * 60)
    print(f"\nTesting if FUSEDMOE_TILEBUF=64 is safe when")
    print(f"routed_tokens >> 64...\n")

    passed = test_moe_with_large_routed_tokens()

    print("\n" + "=" * 60)
    if passed:
        print("CONCLUSION: ✅ TILEBUF=64 appears SAFE")
        print("  - KUTACC likely reuses temp buffer per tile")
        print("  - No crash with 256 tokens (4x TILEBUF)")
    else:
        print("CONCLUSION: ❌ TILEBUF=64 may be UNSAFE")
        print("  - Need to increase buffer size or query KUTACC")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
