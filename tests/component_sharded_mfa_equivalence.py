from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from dalg.models.mfa import ComponentShardedMFA, MFA, component_shard_bounds


def _copy_full_slice(full: MFA, shard: ComponentShardedMFA) -> None:
    start, end = shard.component_start, shard.component_end
    with torch.no_grad():
        shard.mu.copy_(full.mu[start:end])
        shard.dir_raw.copy_(full.dir_raw[start:end])
        shard.scale_rho.copy_(full.scale_rho[start:end])
        shard.pi_logits.copy_(full.pi_logits[start:end])
        if full.psi_rho.ndim == 1:
            shard.psi_rho.copy_(full.psi_rho)
        else:
            shard.psi_rho.copy_(full.psi_rho[start:end])


def _gather_cat(x: torch.Tensor) -> torch.Tensor | None:
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x)
    if dist.get_rank() == 0:
        return torch.cat(parts, dim=0)
    return None


def _check_close(name: str, got: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    try:
        torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f"{name} mismatch\n{exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, choices=(None, "gloo", "nccl"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--optimizer", default="adam", choices=("adam", "sgd"))
    args = parser.parse_args()

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    backend = args.backend or ("nccl" if use_cuda else "gloo")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()

    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world != 2:
        raise SystemExit(f"this equivalence test expects world_size=2, got {world}")

    K, D, q, B = 8, 12, 4, 5
    lr = 3e-3
    torch.manual_seed(1234)
    centroids = torch.randn(K, D, device=device)
    batches = [torch.randn(B, D, device=device) for _ in range(args.steps)]

    torch.manual_seed(5678)
    full = MFA(centroids.clone(), rank=q).to(device)
    start, end = component_shard_bounds(K, rank, world)
    shard = ComponentShardedMFA(
        centroids[start:end].clone(),
        rank=q,
        global_K=K,
        component_start=start,
    ).to(device)
    _copy_full_slice(full, shard)

    opt_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
    full_opt = opt_cls(full.parameters(), lr=lr)
    shard_opt = opt_cls(shard.parameters(), lr=lr)

    for step, x in enumerate(batches, 1):
        full_opt.zero_grad(set_to_none=True)
        shard_opt.zero_grad(set_to_none=True)

        full_loss = full.nll(x)
        shard_loss = shard.nll(x)
        _check_close(f"loss step {step}", shard_loss, full_loss, atol=2e-5, rtol=2e-5)

        full_loss.backward()
        shard_loss.backward()
        shard.sync_replicated_grads()

        full_opt.step()
        shard_opt.step()

    final_batch = batches[-1]
    _check_close("final loss", shard.nll(final_batch), full.nll(final_batch), atol=5e-5, rtol=5e-5)

    mu = _gather_cat(shard.mu.detach())
    dir_raw = _gather_cat(shard.dir_raw.detach())
    scale_rho = _gather_cat(shard.scale_rho.detach())
    pi_logits = _gather_cat(shard.pi_logits.detach())

    psi_parts = [torch.empty_like(shard.psi_rho.detach()) for _ in range(world)]
    dist.all_gather(psi_parts, shard.psi_rho.detach())

    if rank == 0:
        _check_close("mu", mu, full.mu.detach(), atol=5e-5, rtol=5e-5)
        _check_close("dir_raw", dir_raw, full.dir_raw.detach(), atol=5e-5, rtol=5e-5)
        _check_close("scale_rho", scale_rho, full.scale_rho.detach(), atol=5e-5, rtol=5e-5)
        _check_close("pi_logits", pi_logits, full.pi_logits.detach(), atol=5e-5, rtol=5e-5)
        for i, psi in enumerate(psi_parts):
            _check_close(f"psi_rho rank {i}", psi, full.psi_rho.detach(), atol=5e-5, rtol=5e-5)
        print(
            f"component-sharded equivalence passed: device={device.type} "
            f"optimizer={args.optimizer} steps={args.steps}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
