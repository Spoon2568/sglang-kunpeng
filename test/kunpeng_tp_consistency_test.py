"""Consistency test: kunpeng TP collectives vs torch.distributed (gloo).

Verifies the custom sglang_kupl all_reduce / all_gather produce results
identical to torch.distributed on CPU. The all_gather check is ORDER-SENSITIVE,
so it also confirms the kunpeng KUTACC rank space is aligned with the
gloo/sglang rank space — if ranks are permuted, all_gather blocks land in the
wrong order and the check fails.

Launch with MPI, one process per TP rank (world_size=2 example):

    export SGLANG_USE_KUNPENG_TP=1
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
    mpirun -np 2 python test/kunpeng_tp_consistency_test.py

Exit code 0 = all ranks passed. all_gather FAIL => rank misalignment between
gloo and KUTACC (i.e. you DO need the world_comm rerank fix in kupl_runtime.cpp).
"""

import os
import sys

os.environ.setdefault("SGLANG_USE_KUNPENG_TP", "1")
# KUTACC all_reduce 要求 numel % (8 * OMP_NUM_THREADS) == 0。
# 设为 16 让测试 shape (8,16)=128 能通过（128 = 8*16）。
os.environ.setdefault("OMP_NUM_THREADS", "16")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import torch.distributed as dist


def _env_int(names, default):
    for n in names:
        v = os.environ.get(n)
        if v is not None:
            return int(v)
    return default


def detect_topology():
    rank = _env_int(
        ["RANK", "MV2_COMM_WORLD_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK"], 0
    )
    world = _env_int(
        ["WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"], 1
    )
    local = _env_int(
        ["LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"], rank
    )
    return rank, world, local


def init_dist(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)


def gather_parts(x: torch.Tensor, world: int):
    """Gather each rank's tensor to every rank via gloo (pure data movement,
    dtype-agnostic). Returns a list indexed by rank."""
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x.contiguous())
    return parts


def ref_all_reduce(x: torch.Tensor, world: int) -> torch.Tensor:
    parts = gather_parts(x, world)
    acc = torch.zeros_like(x, dtype=torch.float32)
    for p in parts:
        acc += p.to(torch.float32)
    return acc.to(x.dtype)


def ref_all_gather(x: torch.Tensor, world: int, dim: int) -> torch.Tensor:
    parts = gather_parts(x, world)
    return torch.cat(parts, dim=dim)


def tol_for(dtype):
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    return dict(rtol=1e-5, atol=1e-5)


def make_input(shape, dtype, rank, seed=0):
    """Rank-dependent data so a permuted gather is detectable. Each rank's
    values carry its rank in the magnitude."""
    g = torch.Generator().manual_seed(seed + rank)
    base = torch.randn(shape, generator=g)
    return (base + float(rank)).to(dtype)


# OMP_NUM_THREADS=16 时对齐要求 8*16=128。
# (3,257) numel=771 不满足，会被 tp_all_reduce 拒绝（预期 EXCEPTION）。
SHAPES = [(8, 16), (1, 4096), (128, 7168)]
SHAPES_EXPECTED_FAIL_ALIGN = [(3, 257)]  # 不满足对齐，EXCEPTION 是预期行为
DTYPES = [torch.bfloat16]


def run_checks(rank, world):
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengCommunicator,
    )

    comm = KunpengCommunicator(dist.group.WORLD, rank, world, rank)
    if comm.disabled:
        if rank == 0:
            print("KunpengCommunicator disabled (SGLANG_USE_KUNPENG_TP!=1 or "
                  "world_size==1) — nothing to compare.")
        return True

    comm.barrier()
    passed = True

    for dtype in DTYPES:
        for shape in SHAPES:
            x = make_input(shape, dtype, rank)
            tol = tol_for(dtype)

            # all_reduce
            try:
                got = comm.all_reduce(x.clone())
                exp = ref_all_reduce(x, world)
                ok = torch.allclose(got.float(), exp.float(), **tol)
            except Exception as e:  # noqa: BLE001
                ok, got = False, None
                if rank == 0:
                    print(f"[all_reduce] {dtype} {shape} EXCEPTION: {e}")
            if not ok:
                passed = False
                if rank == 0 and got is not None:
                    md = (got.float() - exp.float()).abs().max().item()
                    print(f"[all_reduce] FAIL {dtype} {shape} max_diff={md:.4g}")
            elif rank == 0:
                print(f"[all_reduce] ok   {dtype} {shape}")

            # all_gather along last dim
            dim = -1
            try:
                got = comm.all_gather(x.clone(), dim)
                exp = ref_all_gather(x, world, dim)
                ok = got.shape == exp.shape and torch.allclose(
                    got.float(), exp.float(), **tol
                )
            except Exception as e:  # noqa: BLE001
                ok, got = False, None
                if rank == 0:
                    print(f"[all_gather] {dtype} {shape} EXCEPTION: {e}")
            if not ok:
                passed = False
                if rank == 0:
                    if got is not None and got.shape == exp.shape:
                        md = (got.float() - exp.float()).abs().max().item()
                        print(f"[all_gather] FAIL {dtype} {shape} max_diff={md:.4g} "
                              "(likely RANK MISALIGNMENT — gather order wrong)")
                    elif got is not None:
                        print(f"[all_gather] FAIL {dtype} {shape} shape {got.shape} "
                              f"!= expected {exp.shape}")
            elif rank == 0:
                print(f"[all_gather] ok   {dtype} {shape}")

    # 测试不满足对齐的 shape：all_reduce 应该 EXCEPTION（预期行为）
    for dtype in DTYPES:
        for shape in SHAPES_EXPECTED_FAIL_ALIGN:
            x = make_input(shape, dtype, rank)
            try:
                got = comm.all_reduce(x.clone())
                # 如果没抛异常，说明对齐检查没生效，这是个问题
                passed = False
                if rank == 0:
                    print(f"[all_reduce] {dtype} {shape} 应该因对齐 EXCEPTION 但成功了 — 检查未生效")
            except Exception as e:  # noqa: BLE001
                # 预期的 EXCEPTION，不算 fail
                if rank == 0:
                    print(f"[all_reduce] {dtype} {shape} 对齐 EXCEPTION（预期）: {str(e)[:80]}")

    comm.barrier()
    comm.destroy()
    return passed


def main():
    rank, world, local = detect_topology()
    init_dist(rank, world)
    if rank == 0:
        print(f"world_size={world}; comparing kunpeng vs torch.distributed(gloo)")

    local_pass = run_checks(rank, world)

    # Reduce pass/fail across ranks so any single failure fails the whole run.
    flag = torch.tensor([0 if local_pass else 1], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    all_pass = flag.item() == 0

    if rank == 0:
        print("RESULT:", "PASS ✅" if all_pass else "FAIL ❌")
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

