"""Single-GPU repro for the MFA + bf16-AMP NaN hypothesis.

Hypothesis under test:
    Under torch.autocast(bfloat16) inside MFA._core, the einsums building
    `v = W^T Psi^{-1} (x - mu)` accumulate bf16 rounding noise that scales
    with q (the MFA rank). The likelihood uses the Woodbury identity:

        quad = quad_Psi - low_rank   where low_rank = v^T M^{-1} v

    is a catastrophic-cancellation expression (both sides positive, difference
    must stay non-negative). For large q, the bf16 noise on v inflates
    low_rank past quad_Psi, making quad negative, which sends ll to +inf
    and the loss to NaN within a few Adam steps.

What this script does:
    Sweeps q ∈ {10, 64, 128, 256, 337, 512} with use_amp ∈ {False, True} on
    one GPU, K small, D=2048 (matches the production setup), runs a short
    Adam loop on synthetic data, and reports whether/when the loss first
    becomes non-finite.

Expected outcome IF the hypothesis is correct:
    - use_amp=False, any q              → finite throughout
    - use_amp=True,  q=10               → finite throughout
    - use_amp=True,  q in {256+, 337+}  → first_nan within ~20-50 steps

Usage:
    PYTHONPATH=src python tests/amp_q_nan_repro.py --device cuda
"""

from __future__ import annotations

import argparse
import math

import torch

from dalg.models.mfa import MFA


def _isfinite(x: float) -> bool:
    return x == x and not math.isinf(x)


def run(
    *,
    K: int,
    D: int,
    q: int,
    B: int,
    steps: int,
    use_amp: bool,
    device: torch.device,
    seed: int,
) -> tuple[list[float], int | None]:
    """Run a short Adam loop on synthetic data, return (per_step_losses, first_nan_step or None)."""
    torch.manual_seed(seed)
    centroids = (torch.randn(K, D, device=device) * 0.5).contiguous()
    model = MFA(centroids, rank=q, use_amp=use_amp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_data = torch.randn(steps, B, D, device=device)

    losses: list[float] = []
    first_nan: int | None = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model.nll(x_data[step])
        loss_val = float(loss.item())
        losses.append(loss_val)
        if not _isfinite(loss_val) and first_nan is None:
            first_nan = step
        loss.backward()
        opt.step()
    return losses, first_nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--B", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qs", type=int, nargs="+", default=[10, 64, 128, 256, 337, 512])
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("requested --device cuda but no CUDA device available")
        torch.set_float32_matmul_precision("high")

    print(f"# device={device} K={args.K} D={args.D} B={args.B} steps={args.steps} seed={args.seed}")
    print(f"# hypothesis: use_amp=True with large q -> NaN within a few Adam steps")
    print()
    header = f"{'q':>5}  {'use_amp':>8}  {'first_nan':>10}  {'final_loss':>14}  {'min_loss':>14}  {'max_loss':>14}"
    print(header)
    print("-" * len(header))

    for q in args.qs:
        for use_amp in (False, True):
            losses, first_nan = run(
                K=args.K,
                D=args.D,
                q=q,
                B=args.B,
                steps=args.steps,
                use_amp=use_amp,
                device=device,
                seed=args.seed,
            )
            finite = [x for x in losses if _isfinite(x)]
            final = losses[-1]
            mn = min(finite) if finite else float("nan")
            mx = max(finite) if finite else float("nan")
            nan_str = str(first_nan) if first_nan is not None else "—"
            print(
                f"{q:>5d}  {str(use_amp):>8}  {nan_str:>10}  "
                f"{final:>14.4f}  {mn:>14.4f}  {mx:>14.4f}"
            )


if __name__ == "__main__":
    main()
