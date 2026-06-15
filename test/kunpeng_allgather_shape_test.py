"""Test all_gather shape validation.

Verifies that tp_all_gather correctly rejects mismatched output shapes
to prevent memory corruption.

Usage:
    export SGLANG_USE_KUNPENG_TP=1
    mpirun -np 4 python test/kunpeng_allgather_shape_test.py
"""

import os
import sys

os.environ.setdefault("SGLANG_USE_KUNPENG_TP", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import torch.distributed as dist


def detect_topology():
    rank = int(os.environ.get("RANK", os.environ.get("MV2_COMM_WORLD_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("MV2_COMM_WORLD_SIZE", 1)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("MV2_COMM_WORLD_LOCAL_RANK", rank)))
    return rank, world, local


def test_correct_shape():
    """Test: correct output shape should work."""
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengCommunicator,
    )

    rank, world, local = detect_topology()
    comm = KunpengCommunicator(None, rank, world, local)

    if comm.disabled:
        return True

    input = torch.randn(8, 128, dtype=torch.bfloat16)
    output = torch.empty(8, 128 * world, dtype=torch.bfloat16)  # 正确: batch=8, hidden=128*world
    torch.ops.sglang_kupl.tp_all_gather(comm.handle, output, input)

    comm.destroy()
    return True


def test_wrong_batch_dim():
    """Test: wrong batch dimension should be rejected."""
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengCommunicator,
    )

    rank, world, local = detect_topology()
    comm = KunpengCommunicator(None, rank, world, local)

    if comm.disabled:
        return True

    input = torch.randn(8, 128, dtype=torch.bfloat16)
    output = torch.empty(16, 128 * world, dtype=torch.bfloat16)  # 错误: batch 应该是 8，不是 16

    try:
        torch.ops.sglang_kupl.tp_all_gather(comm.handle, output, input)
        # 如果没抛异常，说明检查失败
        comm.destroy()
        return False
    except RuntimeError as e:
        # 应该包含 "维度 0 不匹配" 的错误信息
        if "维度 0 不匹配" in str(e) or "不匹配" in str(e):
            comm.destroy()
            return True
        else:
            comm.destroy()
            return False


def test_wrong_last_dim():
    """Test: wrong last dimension should be rejected."""
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengCommunicator,
    )

    rank, world, local = detect_topology()
    comm = KunpengCommunicator(None, rank, world, local)

    if comm.disabled:
        return True

    input = torch.randn(8, 128, dtype=torch.bfloat16)
    output = torch.empty(8, 256, dtype=torch.bfloat16)  # 错误: 最后一维���该是 128*world，不是 256

    try:
        torch.ops.sglang_kupl.tp_all_gather(comm.handle, output, input)
        comm.destroy()
        return False
    except RuntimeError as e:
        if "最后一维大小不匹配" in str(e) or "不匹配" in str(e):
            comm.destroy()
            return True
        else:
            comm.destroy()
            return False


def test_wrong_ndim():
    """Test: mismatched number of dimensions should be rejected."""
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengCommunicator,
    )

    rank, world, local = detect_topology()
    comm = KunpengCommunicator(None, rank, world, local)

    if comm.disabled:
        return True

    input = torch.randn(8, 128, dtype=torch.bfloat16)  # 2D
    output = torch.empty(8, 4, 128 * world, dtype=torch.bfloat16)  # 3D - 错误

    try:
        torch.ops.sglang_kupl.tp_all_gather(comm.handle, output, input)
        comm.destroy()
        return False
    except RuntimeError as e:
        if "维度不一致" in str(e):
            comm.destroy()
            return True
        else:
            comm.destroy()
            return False


def main():
    rank, world, local = detect_topology()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29504"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    tests = [
        ("correct shape", test_correct_shape),
        ("wrong batch dim (rejected)", test_wrong_batch_dim),
        ("wrong last dim (rejected)", test_wrong_last_dim),
        ("wrong ndim (rejected)", test_wrong_ndim),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            if test_fn():
                if rank == 0:
                    print(f"✅ {name}")
                passed += 1
            else:
                if rank == 0:
                    print(f"❌ {name}")
        except Exception as e:
            if rank == 0:
                print(f"❌ {name}: {e}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print(f"\nShape validation tests: {passed}/{len(tests)} passed")
        if passed == len(tests):
            print("RESULT: PASS ✅")
        else:
            print("RESULT: FAIL ❌")

    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
