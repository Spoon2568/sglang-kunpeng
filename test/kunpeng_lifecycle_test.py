"""Test runtime lifecycle: init/destroy/re-init cycles.

Verifies that KunpengCommunicator properly cleans up MPI/KUPL/KUTACC resources
and can be re-initialized without leaks or crashes.

Usage:
    export SGLANG_USE_KUNPENG_TP=1
    mpirun -np 2 python test/kunpeng_lifecycle_test.py
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


def test_lifecycle_single_init():
    """Test: single init + destroy."""
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengCommunicator,
    )

    rank, world, local = detect_topology()

    # Create and destroy
    comm = KunpengCommunicator(None, rank, world, local)
    if not comm.disabled:
        x = torch.randn(8, 128, dtype=torch.bfloat16)
        out = comm.all_reduce(x)
        assert out.shape == x.shape

    comm.destroy()
    # Second destroy should be safe (idempotent)
    comm.destroy()

    return True


def test_lifecycle_reinit():
    """Test: init -> destroy -> re-init cycle."""
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengCommunicator,
    )

    rank, world, local = detect_topology()

    # First cycle
    comm1 = KunpengCommunicator(None, rank, world, local)
    if not comm1.disabled:
        x = torch.randn(8, 128, dtype=torch.bfloat16)
        comm1.all_reduce(x)
    comm1.destroy()
    del comm1

    # Second cycle (should work without MPI_Init errors)
    comm2 = KunpengCommunicator(None, rank, world, local)
    if not comm2.disabled:
        x = torch.randn(8, 128, dtype=torch.bfloat16)
        comm2.all_reduce(x)
    comm2.destroy()
    del comm2

    return True


def test_lifecycle_gc():
    """Test: garbage collection triggers __del__."""
    from sglang.srt.distributed.device_communicators.kunpeng_communicator import (
        KunpengCommunicator,
    )

    rank, world, local = detect_topology()

    # Create without explicit destroy — __del__ should clean up
    comm = KunpengCommunicator(None, rank, world, local)
    if not comm.disabled:
        x = torch.randn(8, 128, dtype=torch.bfloat16)
        comm.all_reduce(x)
    # Let it go out of scope; __del__ will be called
    del comm

    # Should be able to create a new one after GC
    comm2 = KunpengCommunicator(None, rank, world, local)
    if not comm2.disabled:
        x = torch.randn(8, 128, dtype=torch.bfloat16)
        comm2.all_reduce(x)
    comm2.destroy()

    return True


def main():
    rank, world, local = detect_topology()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    tests = [
        ("single init/destroy", test_lifecycle_single_init),
        ("re-init after destroy", test_lifecycle_reinit),
        ("GC cleanup (__del__)", test_lifecycle_gc),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            if test_fn():
                if rank == 0:
                    print(f"✅ {name}")
                passed += 1
        except Exception as e:
            if rank == 0:
                print(f"❌ {name}: {e}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print(f"\nLifecycle tests: {passed}/{len(tests)} passed")
        if passed == len(tests):
            print("RESULT: PASS ✅")
        else:
            print("RESULT: FAIL ❌")

    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
