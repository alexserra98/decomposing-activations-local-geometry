"""Empirical probes for MFA overlap, factor rank, and assignment peakiness.

Run directly, for example:

    PYTHONPATH=src python tests/empirical_overlap_assignment_sweep.py

The script writes plots and raw results under
``outputs/experiments/overlap_assignment_sweep`` by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.analysis.cluster_assignments import compute_assignments  # noqa: E402
from dalg.analysis.cluster_overlap import compute_overlap  # noqa: E402
from dalg.models.mfa import MFA, save_mfa  # noqa: E402


D = 100
K = 2
Q_VALUES = [1, 25, 50, 75, 100]
CLOSE_MEAN_DISTANCE = 2.0
DISTANT_MEAN_DISTANCE = 8.0
FLATNESS_LEVELS = [
    ("flat", 0.05),
    ("mild", 0.5),
    ("medium", 2.0),
    ("sharp", 3.5),
]
FLATNESS_COLORS = {
    "flat": "tab:orange",
    "mild": "tab:green",
    "medium": "tab:red",
    "sharp": "tab:purple",
}
DISTANT_COLOR = "tab:blue"


def inv_softplus(x: float) -> float:
    return math.log(math.exp(float(x)) - 1.0)


def basis_directions(q: int, *, mode: str) -> torch.Tensor:
    """Return simple unit directions for the two components."""
    directions = torch.zeros(K, D, q)
    eye = torch.eye(D)

    if mode == "shared":
        directions[0] = eye[:, :q]
        directions[1] = eye[:, :q]
        return directions

    if mode != "orthogonal":
        raise ValueError(f"unknown direction mode: {mode}")

    directions[0] = eye[:, :q]
    if q <= D // 2:
        directions[1] = eye[:, q : 2 * q]
    else:
        # For q > D/2, perfectly disjoint q-dimensional subspaces are
        # impossible in D dimensions. This keeps the non-overlapping part as
        # large as possible and fills the rest with shared axes.
        non_overlap = D - q
        directions[1, :, :non_overlap] = eye[:, q:]
        directions[1, :, non_overlap:] = eye[:, : q - non_overlap]
    return directions


def build_model(
    *,
    q: int,
    mean_distance: float,
    factor_scale: float,
    direction_mode: str,
) -> MFA:
    mu = torch.zeros(K, D)
    mu[0, 2] = -0.5 * mean_distance
    mu[1, 2] = 0.5 * mean_distance

    model = MFA(mu, rank=q, psi_init=1.0, scale_init=max(factor_scale, 1e-4))
    model.mu.data.copy_(mu)
    model.dir_raw.data.copy_(basis_directions(q, mode=direction_mode))
    model.scale_rho.data.fill_(inv_softplus(max(factor_scale, 1e-4)))
    model.psi_rho.data.fill_(inv_softplus(1.0))
    model.pi_logits.data.zero_()
    return model


def sample_from_model(model: MFA, n: int, *, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    comp = torch.randint(0, model.K, (n,), generator=g)
    eps = torch.randn(n, model.D, generator=g)
    z = torch.randn(n, model.q, generator=g)

    with torch.no_grad():
        mu = model.mu[comp]
        W = model.W[comp]
        psi = model._psi()[comp]
        low_rank = torch.einsum("ndq,nq->nd", W, z)
        return mu + low_rank + eps * psi.sqrt()


def sample_peaked_same_means(model: MFA, n: int, *, variance: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    comp = torch.randint(0, model.K, (n,), generator=g)
    return model.mu.detach()[comp] + math.sqrt(variance) * torch.randn(n, model.D, generator=g)


def save_model_to_tmp(model: MFA) -> tuple[tempfile.TemporaryDirectory, Path]:
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "mfa_model.pt"
    save_mfa(model, str(path))
    return tmp, path


def summarize_run(
    *,
    scenario: str,
    q: int,
    flatness: str,
    mean_distance: float,
    factor_scale: float,
    sample_source: str,
    model: MFA,
    x: torch.Tensor,
) -> dict[str, float | int | str]:
    tmp, path = save_model_to_tmp(model)
    try:
        overlap = compute_overlap(path, batch_pairs=16)
        sizes, _assignments, max_resp, peakedness = compute_assignments(
            path,
            [x],
            device="cpu",
            use_inference_cache=True,
        )
    finally:
        tmp.cleanup()

    nonempty = sizes > 0
    entropy = peakedness["entropy"][nonempty].mean().item()
    one_minus_max = peakedness["one_minus_max"][nonempty].mean().item()
    top1_margin = peakedness["top1_minus_top2"][nonempty].mean().item()
    assignment_balance = min(sizes.tolist()) / max(1, int(sizes.sum().item()))

    return {
        "scenario": scenario,
        "q": q,
        "flatness": flatness,
        "mean_distance": mean_distance,
        "factor_scale": factor_scale,
        "sample_source": sample_source,
        "bc": overlap["bc"][0, 1].item(),
        "db": overlap["db"][0, 1].item(),
        "db_mean": overlap["db_mean"][0, 1].item(),
        "db_cov": overlap["db_cov"][0, 1].item(),
        "kl_sym": overlap["kl_sym"][0, 1].item(),
        "entropy": entropy,
        "mean_max_resp": max_resp.mean().item(),
        "one_minus_max": one_minus_max,
        "top1_margin": top1_margin,
        "assignment_balance": assignment_balance,
        "n": int(x.shape[0]),
    }


def run_sweep(n_samples: int, seed: int, close_mean_distance: float) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []

    for q in Q_VALUES:
        model = build_model(
            q=q,
            mean_distance=DISTANT_MEAN_DISTANCE,
            factor_scale=1.0,
            direction_mode="shared",
        )
        x = sample_from_model(model, n_samples, seed=seed + q)
        rows.append(
            summarize_run(
                scenario="distant_means",
                q=q,
                flatness="shared_scale_1",
                mean_distance=DISTANT_MEAN_DISTANCE,
                factor_scale=1.0,
                sample_source="model",
                model=model,
                x=x,
            )
        )

    for q in Q_VALUES:
        for flatness, factor_scale in FLATNESS_LEVELS:
            model = build_model(
                q=q,
                mean_distance=close_mean_distance,
                factor_scale=factor_scale,
                direction_mode="orthogonal",
            )
            x = sample_from_model(model, n_samples, seed=seed + 1000 + q)
            rows.append(
                summarize_run(
                    scenario="close_means_orthogonal",
                    q=q,
                    flatness=flatness,
                    mean_distance=close_mean_distance,
                    factor_scale=factor_scale,
                    sample_source="model",
                    model=model,
                    x=x,
                )
            )

            if flatness == "flat":
                x_peaked = sample_peaked_same_means(
                    model,
                    n_samples,
                    variance=0.05,
                    seed=seed + 2000 + q,
                )
                rows.append(
                    summarize_run(
                        scenario="close_means_orthogonal",
                        q=q,
                        flatness=flatness,
                        mean_distance=close_mean_distance,
                        factor_scale=factor_scale,
                        sample_source="peaked_same_means",
                        model=model,
                        x=x_peaked,
                    )
                )

    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_for(
    rows: list[dict[str, float | int | str]],
    *,
    scenario: str,
    sample_source: str | None = None,
    flatness: str | None = None,
) -> list[dict[str, float | int | str]]:
    out = [r for r in rows if r["scenario"] == scenario]
    if sample_source is not None:
        out = [r for r in out if r["sample_source"] == sample_source]
    if flatness is not None:
        out = [r for r in out if r["flatness"] == flatness]
    return sorted(out, key=lambda r: int(r["q"]))


def save_plots(rows: list[dict[str, float | int | str]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    def format_q_axis(ax) -> None:
        ax.set_xlabel("q")
        ax.set_xticks(Q_VALUES)
        ax.set_xlim(min(Q_VALUES) - 2, max(Q_VALUES) + 2)

    distant = rows_for(rows, scenario="distant_means", sample_source="model")
    close_model = rows_for(
        rows,
        scenario="close_means_orthogonal",
        sample_source="model",
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    q = [r["q"] for r in distant]
    axes[0].plot(q, [r["entropy"] for r in distant], marker="o", linestyle="--", color=DISTANT_COLOR, label="distant means")
    axes[1].plot(q, [r["mean_max_resp"] for r in distant], marker="o", linestyle="--", color=DISTANT_COLOR, label="distant means")
    axes[2].plot(q, [r["top1_margin"] for r in distant], marker="o", linestyle="--", color=DISTANT_COLOR, label="distant means")
    for flatness, _factor_scale in FLATNESS_LEVELS:
        subset = [r for r in close_model if r["flatness"] == flatness]
        q = [r["q"] for r in subset]
        label = f"close {flatness}"
        color = FLATNESS_COLORS[flatness]
        axes[0].plot(q, [r["entropy"] for r in subset], marker="o", color=color, label=label)
        axes[1].plot(q, [r["mean_max_resp"] for r in subset], marker="o", color=color, label=label)
        axes[2].plot(q, [r["top1_margin"] for r in subset], marker="o", color=color, label=label)
    axes[0].set_title("entropy")
    axes[1].set_title("mean max responsibility")
    axes[2].set_title("top1 - top2")
    for ax in axes:
        format_q_axis(ax)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "peakiness_vs_q.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    q = [r["q"] for r in distant]
    ax.plot(q, [r["kl_sym"] for r in distant], marker="o", linestyle="--", color=DISTANT_COLOR, label="distant means")
    for flatness, _factor_scale in FLATNESS_LEVELS:
        subset = [r for r in close_model if r["flatness"] == flatness]
        q = [r["q"] for r in subset]
        label = f"close {flatness}"
        ax.plot(q, [r["kl_sym"] for r in subset], marker="o", color=FLATNESS_COLORS[flatness], label=label)
    ax.set_title("symmetric KL divergence")
    format_q_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_vs_q.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for flatness, _factor_scale in FLATNESS_LEVELS:
        subset = [r for r in close_model if r["flatness"] == flatness]
        q = [r["q"] for r in subset]
        color = FLATNESS_COLORS[flatness]
        axes[0].plot(q, [r["entropy"] for r in subset], marker="o", color=color, label=flatness)
        axes[1].plot(q, [r["kl_sym"] for r in subset], marker="o", color=color, label=flatness)
    axes[0].set_title("close means: entropy")
    axes[1].set_title("close means: symmetric KL")
    for ax in axes:
        format_q_axis(ax)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "flatness_grid.png", dpi=160)
    plt.close(fig)

    actual = rows_for(
        rows,
        scenario="close_means_orthogonal",
        sample_source="model",
        flatness="flat",
    )
    peaked = rows_for(
        rows,
        scenario="close_means_orthogonal",
        sample_source="peaked_same_means",
        flatness="flat",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, subset in [("model samples", actual), ("peaked samples", peaked)]:
        q = [r["q"] for r in subset]
        axes[0].plot(q, [r["entropy"] for r in subset], marker="o", label=label)
        axes[1].plot(q, [r["mean_max_resp"] for r in subset], marker="o", label=label)
    axes[0].set_title("flat variance: entropy")
    axes[1].set_title("flat variance: mean max responsibility")
    for ax in axes:
        format_q_axis(ax)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sample_mismatch_flat_variance.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical MFA overlap/assignment sweep")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/experiments/overlap_assignment_sweep")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mean-distance", type=float, default=CLOSE_MEAN_DISTANCE)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".matplotlib"))
    rows = run_sweep(args.n_samples, args.seed, args.close_mean_distance)

    torch.save(rows, args.out_dir / "results.pt")
    write_csv(rows, args.out_dir / "results.csv")
    if not args.no_plots:
        save_plots(rows, args.out_dir)

    print(f"Saved {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
