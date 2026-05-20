"""
Tests for Exercise 1: torch.distributed basics.

Normal run (tests your implementation):
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex01.py

Self-test (verifies the test assertions using the reference solution):
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex01.py --self-test
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, device


def ok(name: str, rank: int) -> None:
    if rank == 0:
        print(f"  PASS  {name}")


# ── Reference implementations ─────────────────────────────────────────────────

def _ref_share_from_rank0(value: float, rank: int, device) -> torch.Tensor:
    t = torch.zeros(1, device=device)
    if rank == 0:
        t[0] = float(value)
    dist.broadcast(t, src=0)
    return t


def _ref_distributed_sum(local_value: float, rank: int, device) -> float:
    t = torch.tensor([float(local_value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _ref_distributed_max(local_value: float, rank: int, device) -> float:
    t = torch.tensor([float(local_value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _ref_gather_all(local_tensor: torch.Tensor, world_size: int, device) -> list:
    parts = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(parts, local_tensor.contiguous())
    return parts


# ── Individual tests ──────────────────────────────────────────────────────────

def test_share_from_rank0(fn, rank, world_size, device):
    result = fn(42.0, rank, device)
    assert result.shape == torch.Size([1]), f"expected shape (1,), got {result.shape}"
    got = float(result.item())
    assert abs(got - 42.0) < 1e-6, f"rank {rank}: expected 42.0, got {got}"
    ok("share_from_rank0: both ranks get 42.0", rank)


def test_distributed_sum(fn, rank, world_size, device):
    local_val = 3.0 if rank == 0 else 7.0
    result = fn(local_val, rank, device)
    assert abs(result - 10.0) < 1e-6, f"rank {rank}: expected 10.0, got {result}"
    ok("distributed_sum: 3.0 + 7.0 = 10.0 on both ranks", rank)


def test_distributed_max(fn, rank, world_size, device):
    local_val = 1.0 if rank == 0 else 5.0
    result = fn(local_val, rank, device)
    assert abs(result - 5.0) < 1e-6, f"rank {rank}: expected 5.0, got {result}"
    ok("distributed_max: max(1.0, 5.0) = 5.0 on both ranks", rank)


def test_gather_all(fn, rank, world_size, device):
    local_t = torch.tensor([1.0, 2.0] if rank == 0 else [3.0, 4.0],
                            device=device, dtype=torch.float32)
    result = fn(local_t, world_size, device)

    assert len(result) == world_size, f"expected {world_size} tensors, got {len(result)}"

    expected = [
        torch.tensor([1.0, 2.0], device=device),
        torch.tensor([3.0, 4.0], device=device),
    ]
    for r in range(world_size):
        assert torch.allclose(result[r], expected[r]), (
            f"rank {rank}: result[{r}] = {result[r]}, expected {expected[r]}"
        )
    ok("gather_all: both ranks see [[1,2],[3,4]]", rank)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true",
                        help="Run reference implementations to verify test correctness")
    args = parser.parse_args()

    rank, world_size, device = setup()

    if world_size != 2:
        raise SystemExit(f"These tests expect world_size=2, got {world_size}")

    if args.self_test:
        share_fn = _ref_share_from_rank0
        sum_fn = _ref_distributed_sum
        max_fn = _ref_distributed_max
        gather_fn = _ref_gather_all
        if rank == 0:
            print("=== Self-test mode (reference implementations) ===")
    else:
        from ex01_stubs import (
            share_from_rank0, distributed_sum, distributed_max, gather_all,
        )
        share_fn = share_from_rank0
        sum_fn = distributed_sum
        max_fn = distributed_max
        gather_fn = gather_all
        if rank == 0:
            print("=== Testing your implementation in ex01_stubs.py ===")

    test_share_from_rank0(share_fn, rank, world_size, device)
    test_distributed_sum(sum_fn, rank, world_size, device)
    test_distributed_max(max_fn, rank, world_size, device)
    test_gather_all(gather_fn, rank, world_size, device)

    if rank == 0:
        print("\nAll tests passed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
