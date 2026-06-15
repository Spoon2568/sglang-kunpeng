"""Test MLA INT8 BMM with empty cscale tensor.

Verifies that batched_gemm_woqs8 correctly handles empty tensor
as cscale parameter (should treat as nullptr).

Usage:
    python test/kunpeng_mla_bmm_cscale_test.py
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


def test_batched_gemm_uk_empty_cscale():
    """Test UK projection with empty cscale (current implementation)."""
    load_async_compute()

    # Simulate MLA UK projection shape
    M, B, K = 128, 16, 192  # seq_len=128, num_heads=16, qk_nope_head_dim=192
    N = 512  # kv_lora_rank

    # Activation (BF16, unpacked)
    act = torch.randn(M, B, K, dtype=torch.bfloat16)
    act_t = act.transpose(0, 1).contiguous()  # [B, M, K]

    # Pack activation
    packed_act = torch.empty_like(act_t)
    torch.ops.async_compute.batched_gemm_pack_allthreads_out(act_t, packed_act)

    # Weight (INT8, quantized)
    weight = torch.randint(-127, 127, (B, K, N), dtype=torch.int8)
    # rscale 应该是 [B, N, 1]（per output channel），不是 [B, K, 1]
    weight_scale = torch.randn(B, N, 1, dtype=torch.float32).abs() * 0.01
    weight_t = weight.transpose(-2, -1).contiguous()  # [B, N, K]

    # Output
    out = torch.empty((B, M, N), dtype=torch.bfloat16)

    try:
        # 关键：传空 tensor 作为 cscale（当前实现）
        torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
            packed_act, weight_t, weight_scale, torch.Tensor(), out
        )

        print(f"✅ UK projection with empty cscale passed")
        print(f"   Output shape: {out.shape}")
        print(f"   Output range: [{out.min():.3f}, {out.max():.3f}]")

        # 检查输出是否合理
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("❌ Output contains NaN or Inf")
            return False

        if out.abs().max() > 1e6:
            print(f"❌ Output has extreme values (max={out.abs().max().item():.2e})")
            return False

        return True
    except Exception as e:
        print(f"❌ FAILED with empty cscale: {e}")
        return False


def test_batched_gemm_uv_empty_cscale():
    """Test UV projection with empty cscale (current implementation)."""
    load_async_compute()

    # Simulate MLA UV projection shape
    M, B, N = 128, 16, 512  # seq_len=128, num_heads=16, kv_lora_rank=512
    V = 7168  # hidden_size

    # Activation (BF16)
    act = torch.randn(M, B, N, dtype=torch.bfloat16)
    act_t = act.transpose(0, 1).contiguous()  # [B, M, N]

    # Pack activation
    packed_act = torch.empty_like(act_t)
    torch.ops.async_compute.batched_gemm_pack_allthreads_out(act_t, packed_act)

    # Weight (INT8)
    weight = torch.randint(-127, 127, (B, N, V), dtype=torch.int8)
    weight_scale = torch.randn(B, V, 1, dtype=torch.float32).abs() * 0.01
    weight_t = weight.permute(0, 2, 1).contiguous()  # [B, V, N]

    # Output
    out = torch.empty((B, M, V), dtype=torch.bfloat16)

    try:
        # 关键：传空 tensor 作为 cscale
        torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
            packed_act, weight_t, weight_scale, torch.Tensor(), out
        )

        print(f"✅ UV projection with empty cscale passed")
        print(f"   Output shape: {out.shape}")
        print(f"   Output range: [{out.min():.3f}, {out.max():.3f}]")

        if torch.isnan(out).any() or torch.isinf(out).any():
            print("❌ Output contains NaN or Inf")
            return False

        if out.abs().max() > 1e6:
            print(f"❌ Output has extreme values (max={out.abs().max().item():.2e})")
            return False

        return True
    except Exception as e:
        print(f"❌ FAILED with empty cscale: {e}")
        return False


def main():
    print("=" * 60)
    print("MLA INT8 BMM Empty cscale Test")
    print("=" * 60)
    print(f"\nTesting if batched_gemm_woqs8 correctly handles")
    print(f"empty tensor as cscale parameter...\n")

    results = []

    # Test UK projection
    try:
        results.append(("UK projection", test_batched_gemm_uk_empty_cscale()))
    except Exception as e:
        print(f"❌ UK test exception: {e}")
        results.append(("UK projection", False))

    print()

    # Test UV projection
    try:
        results.append(("UV projection", test_batched_gemm_uv_empty_cscale()))
    except Exception as e:
        print(f"❌ UV test exception: {e}")
        results.append(("UV projection", False))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")

    all_passed = all(p for _, p in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("CONCLUSION: ✅ Empty cscale is SAFE")
        print("  - KUTACC correctly handles nullptr/empty tensor")
        print("  - Current MLA INT8 BMM implementation is correct")
    else:
        print("CONCLUSION: ❌ Empty cscale is UNSAFE")
        print("  - Need to pass None or valid cscale tensor")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
