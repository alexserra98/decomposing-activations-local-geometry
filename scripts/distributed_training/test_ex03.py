"""
Tests for Exercise 3: Component sharding.

Normal run:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex03.py

Self-test:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex03.py --self-test
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
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

def _ref_distributed_logsumexp(local_values: torch.Tensor, dim: int) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.logsumexp(local_values, dim=dim)
    local_max = local_values.max(dim=dim).values.detach()
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    shifted = local_values - global_max.unsqueeze(dim)
    local_sum = shifted.exp().sum(dim=dim)
    global_sum = local_sum.detach().clone()
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    global_sum = global_sum.clamp_min(torch.finfo(global_sum.dtype).tiny)
    return global_max + global_sum.log() + (local_sum - local_sum.detach()) / global_sum


def _ref_sharded_log_prob(x, local_mu, local_log_pi):
    B, D = x.shape
    diff = x[:, None, :] - local_mu[None, :, :]           # (B, K_local, D)
    ll = -0.5 * (D * math.log(2 * math.pi) + (diff ** 2).sum(dim=-1))  # (B, K_local)
    log_num = _ref_distributed_logsumexp(ll + local_log_pi[None, :], dim=1)
    log_den = _ref_distributed_logsumexp(local_log_pi, dim=0)
    return log_num - log_den


def _ref_sync_shared_param_grad(param):
    if param.grad is None:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)


# ── Test A: distributed_logsumexp forward and gradient ────────────────────────

def test_logsumexp_forward(logsumexp_fn, rank, world_size, device):
    """Forward value must match torch.logsumexp on the concatenated tensor."""
    torch.manual_seed(rank)
    K_local = 3
    x = torch.randn(K_local, device=device)

    # Gather full tensor on rank 0 for reference
    all_x = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(all_x, x)

    result = logsumexp_fn(x, dim=0)  # scalar

    if rank == 0:
        ref = torch.logsumexp(torch.cat(all_x, dim=0), dim=0)
        assert torch.allclose(result, ref, atol=1e-5), (
            f"logsumexp forward: got {result.item():.6f}, expected {ref.item():.6f}"
        )
    ok("distributed_logsumexp: forward matches torch.logsumexp", rank)


def test_logsumexp_gradient(logsumexp_fn, rank, world_size, device):
    """Gradient must equal softmax(x_global) for this rank's slice."""
    torch.manual_seed(rank + 10)
    K_local = 3
    x = torch.randn(K_local, device=device, requires_grad=True)

    # Collect all values before the forward pass (using detached copy)
    all_x_det = [torch.empty(K_local, device=device) for _ in range(world_size)]
    dist.all_gather(all_x_det, x.detach())
    x_global = torch.cat(all_x_det, dim=0)  # (K,)
    ref_grad = torch.softmax(x_global, dim=0)  # (K,)
    ref_grad_local = ref_grad[rank * K_local:(rank + 1) * K_local]

    result = logsumexp_fn(x, dim=0)
    result.backward()

    assert x.grad is not None, "x.grad is None after backward — did you implement the correction term?"
    assert torch.allclose(x.grad, ref_grad_local, atol=1e-5), (
        f"rank {rank}: gradient mismatch\n"
        f"  got:      {x.grad.tolist()}\n"
        f"  expected: {ref_grad_local.tolist()}"
    )
    ok("distributed_logsumexp: gradient equals softmax(x_global)[local_slice]", rank)


# ── Test B: sharded_log_prob ──────────────────────────────────────────────────

def test_sharded_log_prob(log_prob_fn, rank, world_size, device):
    """sharded_log_prob must match the full (non-distributed) log_prob on all K components."""
    torch.manual_seed(42)
    K, D, B = 4, 8, 5
    K_local = K // world_size  # 2 per rank

    # Create full model params on rank 0 and broadcast
    mu_full = torch.zeros(K, D, device=device)
    pi_logits_full = torch.zeros(K, device=device)
    if rank == 0:
        mu_full = torch.randn(K, D, device=device)
        pi_logits_full = torch.randn(K, device=device)
    dist.broadcast(mu_full, src=0)
    dist.broadcast(pi_logits_full, src=0)

    # Each rank slices out its components
    start = rank * K_local
    local_mu = mu_full[start:start + K_local].contiguous()
    local_log_pi = pi_logits_full[start:start + K_local].contiguous()

    # Input batch (same on all ranks)
    torch.manual_seed(99)
    x = torch.randn(B, D, device=device)

    result = log_prob_fn(x, local_mu, local_log_pi)  # (B,)

    assert result.shape == (B,), f"expected shape ({B},), got {result.shape}"

    # Reference: compute on rank 0 using all K components
    if rank == 0:
        diff = x[:, None, :] - mu_full[None, :, :]             # (B, K, D)
        ll_full = -0.5 * (D * math.log(2 * math.pi) + (diff ** 2).sum(dim=-1))  # (B, K)
        log_pi = torch.log_softmax(pi_logits_full, dim=0)
        ref = torch.logsumexp(ll_full + log_pi[None, :], dim=1)  # (B,)
        assert torch.allclose(result, ref, atol=1e-4), (
            f"sharded_log_prob mismatch (max diff={( result - ref).abs().max():.2e})\n"
            f"  got:      {result.tolist()}\n"
            f"  expected: {ref.tolist()}"
        )
    ok("sharded_log_prob: matches full (non-distributed) computation", rank)


# ── Test C: sync_shared_param_grad ────────────────────────────────────────────

def test_sync_shared_param_grad(sync_fn, rank, world_size, device):
    """After sync, both ranks must have the same gradient (the sum of both partial grads)."""
    # Each rank computes a different partial gradient
    param = nn.Parameter(torch.zeros(4, device=device))
    partial_grad = torch.full((4,), float(rank + 1), device=device)  # rank 0: 1.0, rank 1: 2.0
    param.grad = partial_grad.clone()

    sync_fn(param)

    expected = 1.0 + 2.0  # sum of partial grads across both ranks = 3.0
    assert torch.allclose(param.grad, torch.full((4,), expected, device=device)), (
        f"rank {rank}: after sync, grad = {param.grad.tolist()}, expected all {expected}"
    )
    ok("sync_shared_param_grad: both ranks have grad = sum of partial grads", rank)


def test_sync_none_grad_is_noop(sync_fn, rank, world_size, device):
    """sync_shared_param_grad must not crash when grad is None."""
    param = nn.Parameter(torch.zeros(4, device=device))
    assert param.grad is None
    sync_fn(param)  # should not raise or hang
    assert param.grad is None, "sync_shared_param_grad should not create a grad when there was none"
    ok("sync_shared_param_grad: no-op when grad is None", rank)


# ── Test D: end-to-end one-step training ─────────────────────────────────────

def test_end_to_end(logsumexp_fn, log_prob_fn, sync_fn, rank, world_size, device):
    """One training step with sharded log_prob + synced grad must match a serial run."""
    torch.manual_seed(7)
    K, D, B = 4, 8, 5
    K_local = K // world_size

    # Shared params (same init on all ranks)
    mu_full = torch.randn(K, D, device=device)
    pi_logits_full = torch.randn(K, device=device)
    psi_shared = torch.randn(D, device=device)  # replicated param

    # Serial reference: all parameters on rank 0, single-process log_prob
    if rank == 0:
        mu_ref = nn.Parameter(mu_full.clone())
        pi_ref = nn.Parameter(pi_logits_full.clone())
        psi_ref = nn.Parameter(psi_shared.clone())

    # Distributed: each rank owns K_local components
    start = rank * K_local
    local_mu = nn.Parameter(mu_full[start:start + K_local].clone())
    local_pi = nn.Parameter(pi_logits_full[start:start + K_local].clone())
    psi = nn.Parameter(psi_shared.clone())  # replicated

    # Same batch on all ranks
    torch.manual_seed(17)
    x = torch.randn(B, D, device=device)

    # Distributed forward + backward
    log_p = log_prob_fn(x, local_mu + 0 * psi.sum(), local_pi)
    loss_dist = -log_p.mean()
    loss_dist.backward()
    sync_fn(psi)

    dist.barrier()

    if rank == 0:
        # Serial forward + backward
        diff = x[:, None, :] - mu_ref[None, :, :]
        ll_full = -0.5 * (D * math.log(2 * math.pi) + (diff ** 2).sum(dim=-1))
        log_pi = torch.log_softmax(pi_ref, dim=0)
        log_p_ref = torch.logsumexp(ll_full + log_pi[None, :], dim=1)
        loss_ref = -log_p_ref.mean()
        loss_ref.backward()

        assert torch.allclose(loss_dist, loss_ref, atol=1e-4), (
            f"loss mismatch: distributed={loss_dist.item():.6f} serial={loss_ref.item():.6f}"
        )

    ok("end-to-end: distributed loss matches serial loss on same batch", rank)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = setup()

    if world_size != 2:
        raise SystemExit(f"These tests expect world_size=2, got {world_size}")

    if args.self_test:
        logsumexp_fn = _ref_distributed_logsumexp
        log_prob_fn = _ref_sharded_log_prob
        sync_fn = _ref_sync_shared_param_grad
        if rank == 0:
            print("=== Self-test mode (reference implementations) ===")
    else:
        from ex03_stubs import distributed_logsumexp, sharded_log_prob, sync_shared_param_grad
        logsumexp_fn = distributed_logsumexp
        log_prob_fn = sharded_log_prob
        sync_fn = sync_shared_param_grad
        if rank == 0:
            print("=== Testing your implementation in ex03_stubs.py ===")

    test_logsumexp_forward(logsumexp_fn, rank, world_size, device)
    test_logsumexp_gradient(logsumexp_fn, rank, world_size, device)
    test_sharded_log_prob(log_prob_fn, rank, world_size, device)
    test_sync_shared_param_grad(sync_fn, rank, world_size, device)
    test_sync_none_grad_is_noop(sync_fn, rank, world_size, device)
    test_end_to_end(logsumexp_fn, log_prob_fn, sync_fn, rank, world_size, device)

    if rank == 0:
        print("\nAll tests passed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
