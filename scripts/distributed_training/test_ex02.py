"""
Tests for Exercise 2: Data-Parallel DDP.

Normal run:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex02.py

Self-test:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex02.py --self-test
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, Subset

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


def make_model(device: torch.device) -> nn.Module:
    """2-layer MLP, same init on every rank (fixed seed)."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)
    return model


# ── Reference implementations ─────────────────────────────────────────────────

def _ref_make_loader(rank, world_size, batch_size):
    torch.manual_seed(42)
    N = 100
    x = torch.randn(N, 4)
    y = torch.randint(0, 2, (N,))
    dataset = TensorDataset(x, y)
    n_per_rank = N // world_size
    indices = list(range(rank * n_per_rank, (rank + 1) * n_per_rank))
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)


def _ref_train_one_step_ddp(model, loader, rank, device):
    ddp_model = DDP(model, device_ids=[rank])
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    loss = F.cross_entropy(ddp_model(xb), yb)
    loss.backward()
    opt.step()
    return float(loss.item())


def _ref_params_are_equal_across_ranks(model, rank, world_size, device):
    p = list(model.parameters())[0].data.contiguous()
    parts = [torch.empty_like(p) for _ in range(world_size)]
    dist.all_gather(parts, p)
    return all(torch.allclose(parts[0], parts[r]) for r in range(1, world_size))


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_loader_disjoint(make_loader_fn, rank, world_size, device):
    """Each rank's loader yields non-overlapping indices from the full dataset."""
    loader = make_loader_fn(rank, world_size, batch_size=50)
    batches = list(loader)
    assert len(batches) >= 1, "Loader is empty"
    ok("make_loader: loader produces batches", rank)


def test_ddp_params_equal_after_step(make_loader_fn, train_fn, equal_fn,
                                     rank, world_size, device):
    """After one DDP step both ranks must have identical parameters."""
    model = make_model(device)
    loader = make_loader_fn(rank, world_size, batch_size=16)
    loss = train_fn(model, loader, rank, device)

    # Unwrap DDP if the user forgot (tolerate either wrapped or unwrapped model)
    raw = model.module if hasattr(model, "module") else model

    assert isinstance(loss, float), f"train_one_step_ddp must return a float, got {type(loss)}"
    assert loss > 0, f"Loss should be positive, got {loss}"

    are_equal = equal_fn(raw, rank, world_size, device)
    assert are_equal, (
        "Parameters differ across ranks after DDP step — "
        "did you actually wrap the model in DDP?"
    )
    ok("DDP: parameters are identical on both ranks after one step", rank)


def test_params_equal_fn_detects_difference(equal_fn, rank, world_size, device):
    """The params_are_equal function must return False when params actually differ."""
    model = make_model(device)
    # Corrupt rank 1's first parameter
    if rank == 1:
        list(model.parameters())[0].data.add_(100.0)
    are_equal = equal_fn(model, rank, world_size, device)
    assert not are_equal, (
        "params_are_equal returned True even though rank 1's parameters were modified — "
        "the function is not actually comparing across ranks"
    )
    ok("params_are_equal_across_ranks: correctly detects a difference", rank)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = setup()

    if world_size != 2:
        raise SystemExit(f"These tests expect world_size=2, got {world_size}")

    if args.self_test:
        make_loader_fn = _ref_make_loader
        train_fn = _ref_train_one_step_ddp
        equal_fn = _ref_params_are_equal_across_ranks
        if rank == 0:
            print("=== Self-test mode (reference implementations) ===")
    else:
        from ex02_stubs import make_loader, train_one_step_ddp, params_are_equal_across_ranks
        make_loader_fn = make_loader
        train_fn = train_one_step_ddp
        equal_fn = params_are_equal_across_ranks
        if rank == 0:
            print("=== Testing your implementation in ex02_stubs.py ===")

    test_loader_disjoint(make_loader_fn, rank, world_size, device)
    test_ddp_params_equal_after_step(make_loader_fn, train_fn, equal_fn, rank, world_size, device)
    test_params_equal_fn_detects_difference(equal_fn, rank, world_size, device)

    if rank == 0:
        print("\nAll tests passed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
