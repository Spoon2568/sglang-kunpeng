"""Test kunpeng grouped_topk_forward against sgl_kernel implementations.

Usage:
    python test/kunpeng_topk_test.py
"""

import os
import sys

os.environ["SGLANG_USE_KUNPENG_W8A8"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import torch.nn.functional as F


def reference_sigmoid_grouped_topk(
    router_logits: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool = True,
    correction_bias: torch.Tensor | None = None,
):
    """
    Pure-PyTorch reference implementation of DeepSeek V3/R1 grouped top-k
    with sigmoid scoring and optional correction_bias.
    """
    M = router_logits.shape[0]
    N = router_logits.shape[1]

    # DeepSeek V3/R1 uses sigmoid scoring
    scores = router_logits.float().sigmoid()
    if correction_bias is not None:
        scores = scores + correction_bias.float().unsqueeze(0)

    # Grouped top-k: reshape to [M, num_expert_group, experts_per_group]
    experts_per_group = N // num_expert_group
    scores_grouped = scores.view(M, num_expert_group, experts_per_group)

    # Select topk_group scores per group
    _, group_topk_indices = torch.topk(scores_grouped, topk_group, dim=-1)
    # Mask: keep only topk_group per group
    mask = torch.zeros_like(scores_grouped).scatter_(-1, group_topk_indices, 1.0)
    masked_scores = scores_grouped * mask

    # Flatten back and select global topk
    scores_masked = masked_scores.view(M, N)
    topk_vals, topk_ids = torch.topk(scores_masked, topk, dim=-1)

    if renormalize:
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        topk_weights = topk_vals / topk_sum
    else:
        topk_weights = topk_vals

    return topk_weights, topk_ids.int()


def reference_softmax_grouped_topk(
    router_logits: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool = True,
):
    """Pure-PyTorch reference: softmax-based grouped top-k."""
    M = router_logits.shape[0]
    N = router_logits.shape[1]

    scores = router_logits.float().softmax(dim=-1)

    experts_per_group = N // num_expert_group
    scores_grouped = scores.view(M, num_expert_group, experts_per_group)

    _, group_topk_indices = torch.topk(scores_grouped, topk_group, dim=-1)
    mask = torch.zeros_like(scores_grouped).scatter_(-1, group_topk_indices, 1.0)
    masked_scores = scores_grouped * mask

    scores_masked = masked_scores.view(M, N)
    topk_vals, topk_ids = torch.topk(scores_masked, topk, dim=-1)

    if renormalize:
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        topk_weights = topk_vals / topk_sum
    else:
        topk_weights = topk_vals

    return topk_weights, topk_ids.int()


def test_kunpeng_vs_reference():
    """Test kunpeng grouped_topk against pure PyTorch reference."""
    from sglang.srt.hardware_backend.kunpeng.topk import grouped_topk_forward

    torch.manual_seed(42)

    M_cases = [1, 4, 128]
    # DeepSeek V3/R1: 256 experts, 8 groups of 32, topk = 8, topk_group = 4
    N = 256
    num_expert_group = 8
    topk_group = 4
    topk = 8

    print("=" * 60)
    print("Test 1: Sigmoid scoring (DeepSeek V3/R1, no bias)")
    print("=" * 60)

    for M in M_cases:
        router_logits = torch.randn(M, N, dtype=torch.bfloat16) * 0.1

        ref_weights, ref_ids = reference_sigmoid_grouped_topk(
            router_logits, topk, num_expert_group, topk_group, renormalize=True
        )

        kunpeng_weights, kunpeng_ids = grouped_topk_forward(
            router_logits=router_logits,
            topk=topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=True,
            scoring_func_sigmoid=True,
            bias=None,
        )

        ids_match = torch.equal(kunpeng_ids, ref_ids)
        weights_close = torch.allclose(
            kunpeng_weights.float(), ref_weights.float(), rtol=1e-3, atol=1e-5
        )

        print(f"  M={M}: ids_match={ids_match}, weights_close={weights_close}")
        if not ids_match or not weights_close:
            mismatch = (kunpeng_ids != ref_ids).sum().item()
            print(f"    id mismatches: {mismatch}")
            print(f"    max weight diff: {(kunpeng_weights.float() - ref_weights.float()).abs().max().item():.6f}")
            return False

    print("  PASSED\n")
    return True


def test_kunpeng_vs_reference_with_bias():
    """Test with correction_bias (DeepSeek V3/R1 noaux_tc routing)."""
    from sglang.srt.hardware_backend.kunpeng.topk import grouped_topk_forward

    torch.manual_seed(42)

    N = 256
    num_expert_group = 8
    topk_group = 4
    topk = 8
    M = 8

    print("=" * 60)
    print("Test 2: Sigmoid scoring with correction_bias")
    print("=" * 60)

    router_logits = torch.randn(M, N, dtype=torch.bfloat16) * 0.1
    correction_bias = torch.randn(N, dtype=torch.float32) * 0.01

    ref_weights, ref_ids = reference_sigmoid_grouped_topk(
        router_logits,
        topk,
        num_expert_group,
        topk_group,
        renormalize=True,
        correction_bias=correction_bias,
    )

    kunpeng_weights, kunpeng_ids = grouped_topk_forward(
        router_logits=router_logits,
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        renormalize=True,
        scoring_func_sigmoid=True,
        bias=correction_bias,
    )

    ids_match = torch.equal(kunpeng_ids, ref_ids)
    weights_close = torch.allclose(
        kunpeng_weights.float(), ref_weights.float(), rtol=1e-3, atol=1e-5
    )

    print(f"  ids_match={ids_match}, weights_close={weights_close}")
    if not ids_match:
        print(f"    id mismatches: {(kunpeng_ids != ref_ids).sum().item()}")
    if not weights_close:
        print(f"    max weight diff: {(kunpeng_weights.float() - ref_weights.float()).abs().max().item():.6f}")

    passed = ids_match and weights_close
    print(f"  {'PASSED' if passed else 'FAILED'}\n")
    return passed


def test_shared_expert_effect():
    """
    Test num_fused_shared_experts equivalence.

    SGLang passes topk=9 and num_fused_shared_experts=1;
    kunpeng passes topk=8 and handles the shared expert separately.
    Both should produce the same TOP-8 routed experts (ignoring the 9th shared expert slot).
    """
    from sglang.srt.hardware_backend.kunpeng.topk import grouped_topk_forward

    torch.manual_seed(42)

    N = 256
    num_expert_group = 8
    topk_group = 4
    M = 8

    print("=" * 60)
    print("Test 3: shared expert routing equivalence")
    print("=" * 60)

    router_logits = torch.randn(M, N, dtype=torch.bfloat16) * 0.1
    correction_bias = torch.randn(N, dtype=torch.float32) * 0.01

    # SGLang: topk=9 with shared expert
    sgl_weights, sgl_ids = reference_sigmoid_grouped_topk(
        router_logits, topk=9, num_expert_group=num_expert_group,
        topk_group=topk_group, renormalize=True, correction_bias=correction_bias,
    )
    # Remove shared expert column (expert 255) — we only compare the routed experts
    # Note: we can't easily identify which column is the "shared" expert in the
    # reference output since it's mixed with routed. The shared expert filling
    # logic in sgl_kernel appends it, but our reference doesn't model that.
    #
    # Instead, we run WITHOUT shared expert (topk=8) on BOTH and compare.
    ref_weights, ref_ids = reference_sigmoid_grouped_topk(
        router_logits, topk=8, num_expert_group=num_expert_group,
        topk_group=topk_group, renormalize=True, correction_bias=correction_bias,
    )

    kunpeng_weights, kunpeng_ids = grouped_topk_forward(
        router_logits=router_logits, topk=8,
        num_expert_group=num_expert_group, topk_group=topk_group,
        renormalize=True, scoring_func_sigmoid=True, bias=correction_bias,
    )

    ids_match = torch.equal(kunpeng_ids, ref_ids)
    weights_close = torch.allclose(
        kunpeng_weights.float(), ref_weights.float(), rtol=1e-3, atol=1e-5
    )

    print(f"  topk=8 (no shared): ids_match={ids_match}, weights_close={weights_close}")

    # Also verify that the SGLang topk=9 first 8 columns contain the same experts
    # as the topk=8 result (shared expert fills the additional slot):
    sgl_routed_ids = sgl_ids[:, :8]
    sgl_routed_weights = sgl_weights[:, :8]

    same_routed = torch.equal(kunpeng_ids, sgl_routed_ids)
    routed_weights_close = torch.allclose(
        kunpeng_weights.float(), sgl_routed_weights.float(), rtol=1e-3, atol=1e-5
    )
    print(f"  routed ids match (sgl topk=9[:8] vs topk=8): {same_routed}")
    print(f"  routed weights close: {routed_weights_close}")
    if not routed_weights_close:
        diff = (kunpeng_weights.float() - sgl_routed_weights.float()).abs().max().item()
        print(f"  max weight diff: {diff:.6f}")

    passed = ids_match and weights_close and same_routed
    print(f"  {'PASSED' if passed else 'FAILED'}\n")
    return passed


def test_softmax_scoring():
    """Test softmax-based grouped topk."""
    from sglang.srt.hardware_backend.kunpeng.topk import grouped_topk_forward

    torch.manual_seed(42)

    N = 64
    num_expert_group = 4
    topk_group = 2
    topk = 4
    M = 8

    print("=" * 60)
    print("Test 4: Softmax scoring")
    print("=" * 60)

    router_logits = torch.randn(M, N, dtype=torch.bfloat16) * 0.1

    ref_weights, ref_ids = reference_softmax_grouped_topk(
        router_logits, topk, num_expert_group, topk_group, renormalize=True
    )

    kunpeng_weights, kunpeng_ids = grouped_topk_forward(
        router_logits=router_logits,
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        renormalize=True,
        scoring_func_sigmoid=False,
    )

    ids_match = torch.equal(kunpeng_ids, ref_ids)
    weights_close = torch.allclose(
        kunpeng_weights.float(), ref_weights.float(), rtol=1e-3, atol=1e-5
    )

    print(f"  ids_match={ids_match}, weights_close={weights_close}")
    if not ids_match or not weights_close:
        print(f"    id mismatches: {(kunpeng_ids != ref_ids).sum().item()}")
        print(f"    max weight diff: {(kunpeng_weights.float() - ref_weights.float()).abs().max().item():.6f}")
    passed = ids_match and weights_close
    print(f"  {'PASSED' if passed else 'FAILED'}\n")
    return passed


def test_renormalize_off():
    """Test without renormalization."""
    from sglang.srt.hardware_backend.kunpeng.topk import grouped_topk_forward

    torch.manual_seed(42)

    N = 256
    num_expert_group = 8
    topk_group = 4
    topk = 8
    M = 8

    print("=" * 60)
    print("Test 5: renormalize=False")
    print("=" * 60)

    router_logits = torch.randn(M, N, dtype=torch.bfloat16) * 0.1

    ref_weights, ref_ids = reference_sigmoid_grouped_topk(
        router_logits, topk, num_expert_group, topk_group, renormalize=False
    )

    kunpeng_weights, kunpeng_ids = grouped_topk_forward(
        router_logits=router_logits,
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        renormalize=False,
        scoring_func_sigmoid=True,
    )

    ids_match = torch.equal(kunpeng_ids, ref_ids)
    weights_close = torch.allclose(
        kunpeng_weights.float(), ref_weights.float(), rtol=1e-3, atol=1e-5
    )

    print(f"  ids_match={ids_match}, weights_close={weights_close}")
    if not ids_match or not weights_close:
        print(f"    max weight diff: {(kunpeng_weights.float() - ref_weights.float()).abs().max().item():.6f}")
    passed = ids_match and weights_close
    print(f"  {'PASSED' if passed else 'FAILED'}\n")
    return passed


if __name__ == "__main__":
    print("Testing kunpeng grouped_topk_forward\n")
    results = []
    results.append(test_kunpeng_vs_reference())
    results.append(test_kunpeng_vs_reference_with_bias())
    results.append(test_shared_expert_effect())
    results.append(test_softmax_scoring())
    results.append(test_renormalize_off())

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        print(f"SOME TESTS FAILED: {total - passed} failures")
        sys.exit(1)
