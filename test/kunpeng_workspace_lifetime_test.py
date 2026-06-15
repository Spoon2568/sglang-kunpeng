"""Test workspace lifetime safety for igemm_bdq_out.

This test verifies that the temporary workspace buffer is not prematurely
freed by Python GC before the KUTACC kernel completes. If the kernel is
asynchronous and workspace is freed too early, we'll see random crashes
or data corruption.

Strategy:
1. Call igemm_bdq_out with a temporary workspace (like apply_linear does)
2. Immediately trigger GC to free the workspace
3. Check if results are still correct (would fail if use-after-free)

Usage:
    source ~/sibow/init.sh
    source /home/share/fengguangnan/sibow/llminfer/sglang-kunpeng/.venv/bin/activate
    export KUNPENG_ASYNC_COMPUTE_SO=/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so
    cd /home/share/fengguangnan/sibow/llminfer/sglang-kunpeng
    python test/kunpeng_workspace_lifetime_test.py
"""

import gc
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


def workspace_bytes(m: int, n: int, k: int) -> int:
    return max(m * n * k * 2, 1024)


def test_workspace_immediate_free():
    """Simulates apply_linear's workspace usage pattern."""
    print("Test 1: Immediate workspace free (current apply_linear pattern)")

    M, N, K = 128, 256, 512

    # Quantize input
    x = torch.randn(M, K, dtype=torch.bfloat16)
    x_absmax = x.abs().max(dim=-1, keepdim=True).values
    x_scale = (x_absmax / 127.0).clamp(min=1e-12).to(torch.float32)  # Must be float32
    x_q = (x / x_scale).round().clamp(-127, 127).to(torch.int8)

    # Quantized weight
    weight = torch.randint(-127, 127, (N, K), dtype=torch.int8)
    weight_scale = torch.randn(N, 1, dtype=torch.float32).abs() * 0.01

    # Pack input
    x_packed = torch.empty_like(x_q)
    torch.ops.async_compute.igemm_pack_act_out(x_q, weight, x_packed)

    # Output
    out = torch.empty(M, N, dtype=torch.bfloat16)

    # Critical: workspace is created inside this function scope
    def call_with_temp_workspace():
        workspace = torch.empty(workspace_bytes(M, N, K), dtype=torch.uint8)
        torch.ops.async_compute.igemm_bdq_out(
            x_packed, weight, weight_scale, x_scale, out, workspace
        )
        # workspace goes out of scope here — GC can free it

    call_with_temp_workspace()

    # Force GC immediately
    gc.collect()

    # If kernel is async and workspace was freed, out will be corrupted
    # Check if output contains reasonable values
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("  ❌ FAIL: Output contains NaN or Inf (possible use-after-free)")
        return False

    # Check output range is reasonable (quantized bf16 shouldn't be too large)
    if out.abs().max() > 1e6:
        print(f"  ❌ FAIL: Output has extreme values (max={out.abs().max().item():.2e})")
        return False

    print(f"  ✅ Output looks valid (range: [{out.min().item():.3f}, {out.max().item():.3f}])")
    return True


def test_workspace_persistent():
    """Tests with persistent workspace (safe pattern)."""
    print("\nTest 2: Persistent workspace (safe pattern)")

    M, N, K = 128, 256, 512

    x = torch.randn(M, K, dtype=torch.bfloat16)
    x_absmax = x.abs().max(dim=-1, keepdim=True).values
    x_scale = (x_absmax / 127.0).clamp(min=1e-12).to(torch.float32)
    x_q = (x / x_scale).round().clamp(-127, 127).to(torch.int8)

    weight = torch.randint(-127, 127, (N, K), dtype=torch.int8)
    weight_scale = torch.randn(N, 1, dtype=torch.float32).abs() * 0.01

    x_packed = torch.empty_like(x_q)
    torch.ops.async_compute.igemm_pack_act_out(x_q, weight, x_packed)

    out = torch.empty(M, N, dtype=torch.bfloat16)

    # Safe: workspace persists until after kernel completes
    workspace = torch.empty(workspace_bytes(M, N, K), dtype=torch.uint8)
    torch.ops.async_compute.igemm_bdq_out(
        x_packed, weight, weight_scale, x_scale, out, workspace
    )

    gc.collect()

    if torch.isnan(out).any() or torch.isinf(out).any():
        print("  ❌ FAIL: Output contains NaN or Inf")
        return False

    if out.abs().max() > 1e6:
        print(f"  ❌ FAIL: Output has extreme values (max={out.abs().max().item():.2e})")
        return False

    print(f"  ✅ Output looks valid (range: [{out.min().item():.3f}, {out.max().item():.3f}])")
    return True


def test_stress_repeated_calls():
    """Stress test: many calls with immediate workspace free."""
    print("\nTest 3: Stress test (100 iterations with immediate free)")

    M, N, K = 64, 128, 256

    for i in range(100):
        x = torch.randn(M, K, dtype=torch.bfloat16)
        x_absmax = x.abs().max(dim=-1, keepdim=True).values
        x_scale = (x_absmax / 127.0).clamp(min=1e-12).to(torch.float32)
        x_q = (x / x_scale).round().clamp(-127, 127).to(torch.int8)

        weight = torch.randint(-127, 127, (N, K), dtype=torch.int8)
        weight_scale = torch.randn(N, 1, dtype=torch.float32).abs() * 0.01

        x_packed = torch.empty_like(x_q)
        torch.ops.async_compute.igemm_pack_act_out(x_q, weight, x_packed)

        out = torch.empty(M, N, dtype=torch.bfloat16)

        # Current apply_linear pattern
        workspace = torch.empty(workspace_bytes(M, N, K), dtype=torch.uint8)
        torch.ops.async_compute.igemm_bdq_out(
            x_packed, weight, weight_scale, x_scale, out, workspace
        )

        # Aggressive GC
        if i % 10 == 0:
            gc.collect()

        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"  ❌ FAIL at iteration {i}: NaN or Inf")
            return False

        if out.abs().max() > 1e6:
            print(f"  ❌ FAIL at iteration {i}: extreme values")
            return False

    print("  ✅ All 100 iterations passed")
    return True


def main():
    load_async_compute()

    print("=" * 60)
    print("Workspace Lifetime Safety Test")
    print("=" * 60)
    print("\nGoal: Verify that temporary workspace doesn't cause")
    print("      use-after-free if KUTACC kernel is async.\n")

    results = []

    # Test 1: Current pattern (might fail if async)
    try:
        results.append(("Immediate workspace free", test_workspace_immediate_free()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        results.append(("Immediate workspace free", False))

    # Test 2: Safe pattern
    try:
        results.append(("Persistent workspace", test_workspace_persistent()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        results.append(("Persistent workspace", False))

    # Test 3: Stress test
    try:
        results.append(("Stress test", test_stress_repeated_calls()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        results.append(("Stress test", False))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")

    all_passed = all(p for _, p in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("CONCLUSION: ✅ Workspace lifetime is SAFE")
        print("  - KUTACC kernel is likely SYNCHRONOUS (blocks until done)")
        print("  - Temporary workspace pattern in apply_linear is OK")
    else:
        print("CONCLUSION: ❌ Workspace lifetime is UNSAFE")
        print("  - KUTACC kernel may be ASYNCHRONOUS")
        print("  - Need to fix apply_linear to use persistent workspace")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
