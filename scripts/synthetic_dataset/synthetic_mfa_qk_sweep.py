"""Synthetic MFA sweep over fitted component count and factor rank.

This script samples data from a known mixture of low-rank Gaussians, fits MFA
models over a grid of K and q, and records responsibility peakiness plus label
recovery metrics. It is meant as a research probe, not a polished benchmark.

Example:

    PYTHONPATH=src python scripts/synthetic_mfa_qk_sweep.py generate-dataset
    PYTHONPATH=src python scripts/synthetic_mfa_qk_sweep.py fit-one --K-fit 8 --q-fit 100

The default geometry is intentionally closer to the activation setting:
D=500, K_true=8, q_true=100, q_fit in [10, 50, 100, 200, 500].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dalg.models.mfa import MFA, save_mfa  # noqa: E402
from dalg.models.train import train_nll  # noqa: E402


DEFAULT_DATASET_DIR = Path("/orfeo/scratch/dssc/zenocosini/dalg-cache/assets")
DEFAULT_MODEL_ROOT = REPO_ROOT / "dalg-cache/qk_sweep_exploration"


@dataclass
class SweepConfig:
    dataset_path: Path | None = None
    model_root: Path = DEFAULT_MODEL_ROOT
    run_name: str = "default"
    D: int = 500
    K_true: int = 8
    q_true: int = 100
    K_fit: tuple[int, ...] = (4, 6, 8, 10, 12)
    q_fit: tuple[int, ...] = (10, 50, 100, 200, 500)
    n_train: int = 1_000_000
    n_test: int = 50_000
    n_seeds: int = 1
    seed: int = 0
    batch_size: int = 512
    epochs: int = 100
    lr: float = 1e-3
    grad_clip: float | None = 5.0
    early_stop_delta: float = 1e-4
    early_stop_patience: int | None = 5
    early_stop_min_delta: float = 1e-3
    mean_scale: float = 6.0
    factor_scale: float = 1.0
    psi: float = 0.25
    kmeans_max_iter: int = 100
    kmeans_n_init: int = 3
    device: str = "cuda"
    no_plots: bool = False


def default_dataset_path(K_true: int, q_true: int, *, D: int, seed: int) -> Path:
    return DEFAULT_DATASET_DIR / f"synthetic_mfa_Ktrue{K_true}_qtrue{q_true}_D{D}_seed{seed}.pt"


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return items


def _jsonable_config(cfg: SweepConfig) -> dict:
    out = asdict(cfg)
    out["dataset_path"] = str(resolve_dataset_path(cfg))
    out["model_root"] = str(cfg.model_root)
    out["K_fit"] = list(cfg.K_fit)
    out["q_fit"] = list(cfg.q_fit)
    return out


def resolve_dataset_path(cfg: SweepConfig) -> Path:
    if cfg.dataset_path is not None:
        return Path(cfg.dataset_path)
    return default_dataset_path(cfg.K_true, cfg.q_true, D=cfg.D, seed=cfg.seed)


def _resolve_device(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        print("Requested MPS but it is not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


def _orthonormal_columns(D: int, q: int, *, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(D, q, generator=generator)
    q_mat, _ = torch.linalg.qr(raw, mode="reduced")
    return q_mat


def _centroid_directions(D: int, K: int, *, generator: torch.Generator) -> torch.Tensor:
    if K <= D:
        return _orthonormal_columns(D, K, generator=generator).T
    directions = torch.randn(K, D, generator=generator)
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)


def make_ground_truth(cfg: SweepConfig, *, seed: int) -> MFA:
    """Build a ground-truth MFA with separated centroids and rank-q_true factors.

    When K_true <= D, centroids sit on orthonormal directions. For K_true > D,
    centroids use random unit directions because fully orthogonal centroids are
    impossible in the ambient space.
    """
    if cfg.q_true > cfg.D:
        raise ValueError(f"q_true={cfg.q_true} must be <= D={cfg.D}")

    generator = torch.Generator().manual_seed(seed)
    mu_dirs = _centroid_directions(cfg.D, cfg.K_true, generator=generator)
    mu = cfg.mean_scale * mu_dirs

    model = MFA(
        mu,
        rank=cfg.q_true,
        psi_init=cfg.psi,
        scale_init=cfg.factor_scale,
    )
    with torch.no_grad():
        model.mu.copy_(mu)
        directions = torch.stack(
            [
                _orthonormal_columns(cfg.D, cfg.q_true, generator=generator)
                for _ in range(cfg.K_true)
            ],
            dim=0,
        )
        model.dir_raw.copy_(directions)
        # the following control the fattness of the components
        model.scale_rho.fill_(_inv_softplus(cfg.factor_scale)) 
        model.psi_rho.fill_(_inv_softplus(cfg.psi))
        model.pi_logits.zero_()
    return model


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(float(x)))


@torch.no_grad()
def sample_from_mfa(
    model: MFA,
    n: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x and true component labels from an MFA with uniform pi."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, model.K, (n,), generator=generator)
    z = torch.randn(n, model.q, generator=generator)
    eps = torch.randn(n, model.D, generator=generator)

    mu = model.mu.detach().cpu()[labels]
    W = model.W.detach().cpu()[labels]
    psi = model._psi().detach().cpu()[labels]
    low_rank = torch.einsum("ndq,nq->nd", W, z)
    x = mu + low_rank + eps * psi.sqrt()
    return x.float(), labels.long()


def generate_dataset(cfg: SweepConfig, *, seed: int | None = None) -> dict:
    """Generate and save one synthetic train/test dataset."""
    seed = cfg.seed if seed is None else int(seed)
    dataset_path = resolve_dataset_path(cfg)
    torch.manual_seed(seed)
    truth = make_ground_truth(cfg, seed=seed)
    x_train, y_train = sample_from_mfa(truth, cfg.n_train, seed=seed + 10_000)
    x_test, y_test = sample_from_mfa(truth, cfg.n_test, seed=seed + 20_000)

    dataset = {
        "config": _jsonable_config(cfg),
        "seed": seed,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "truth": truth.state_dict(),
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, dataset_path)
    print(f"Saved synthetic dataset to {dataset_path}")
    return dataset


def load_dataset(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_dataset_config(dataset: dict, cfg: SweepConfig) -> None:
    expected = {
        "D": cfg.D,
        "K_true": cfg.K_true,
        "q_true": cfg.q_true,
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "seed": cfg.seed,
    }
    saved = dataset.get("config", {})
    mismatches = [
        f"{key}: dataset={saved.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if saved.get(key) != value
    ]
    if mismatches:
        dataset_path = resolve_dataset_path(cfg)
        details = "\n  ".join(mismatches)
        raise ValueError(
            f"Dataset at {dataset_path} does not match this sweep config:\n"
            f"  {details}\n"
            "Regenerate the dataset or use a different --dataset-path."
        )


def _kmeans_centroids(
    x: torch.Tensor,
    K: int,
    *,
    seed: int,
    max_iter: int,
    n_init: int,
) -> torch.Tensor:
    start = time.time()
    print(
        f"running KMeans: n={x.shape[0]} D={x.shape[1]} K={K} "
        f"max_iter={max_iter} n_init={n_init}",
        flush=True,
    )
    km = KMeans(
        n_clusters=K,
        random_state=seed,
        max_iter=max_iter,
        n_init=n_init,
    )
    km.fit(x.cpu().numpy())
    print(f"KMeans done in {time.time() - start:.1f}s", flush=True)
    return torch.tensor(km.cluster_centers_, dtype=torch.float32)


def _centroid_cache_path(run_dir: Path, cfg: SweepConfig, *, K_fit: int, seed: int) -> Path:
    name = (
        f"K{K_fit:04d}_seed{seed:04d}"
        f"_maxiter{cfg.kmeans_max_iter}_ninit{cfg.kmeans_n_init}.pt"
    )
    return run_dir / "centroids" / name


def _load_or_fit_centroids(
    x: torch.Tensor,
    cfg: SweepConfig,
    *,
    K_fit: int,
    seed: int,
    cache_path: Path | None,
) -> torch.Tensor:
    if cache_path is not None and cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        centroids = saved["centroids"] if isinstance(saved, dict) else saved
        if tuple(centroids.shape) != (K_fit, cfg.D):
            raise ValueError(
                f"Cached centroids at {cache_path} have shape {tuple(centroids.shape)}, "
                f"expected {(K_fit, cfg.D)}"
            )
        print(f"Loaded cached centroids from {cache_path}", flush=True)
        return centroids.float()

    centroids = _kmeans_centroids(
        x,
        K_fit,
        seed=seed,
        max_iter=cfg.kmeans_max_iter,
        n_init=cfg.kmeans_n_init,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        torch.save(
            {
                "centroids": centroids,
                "K_fit": K_fit,
                "seed": seed,
                "D": cfg.D,
                "K_true": cfg.K_true,
                "q_true": cfg.q_true,
                "kmeans_max_iter": cfg.kmeans_max_iter,
                "kmeans_n_init": cfg.kmeans_n_init,
            },
            tmp_path,
        )
        os.replace(tmp_path, cache_path)
        print(f"Saved centroids cache to {cache_path}", flush=True)
    return centroids


def _make_loader(x: torch.Tensor, batch_size: int, *, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


@torch.no_grad()
def _nll(model: MFA, x: torch.Tensor, *, device: torch.device, batch_size: int) -> float:
    model.eval()
    total = 0.0
    total_n = 0
    with model.inference_cache():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            loss = model.nll(xb)
            total += float(loss.item()) * xb.shape[0]
            total_n += xb.shape[0]
    return total / max(total_n, 1)


@torch.no_grad()
def responsibility_metrics(
    model: MFA,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int | list[int] | list[float]]:
    model.eval()
    responsibilities: list[torch.Tensor] = []
    with model.inference_cache():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            responsibilities.append(model.responsibilities(xb).cpu())

    r = torch.cat(responsibilities, dim=0)
    top = r.max(dim=1)
    pred = top.indices
    max_resp = top.values
    entropy = -(r * (r + 1e-8).log()).sum(dim=1)
    margin = _top1_minus_top2(r)
    sizes = torch.bincount(pred, minlength=model.K)
    hist = torch.histc(max_resp, bins=10, min=0.0, max=1.0)

    return {
        "mean_entropy": float(entropy.mean().item()),
        "norm_mean_entropy": float(entropy.mean().item() / math.log(model.K)),
        "mean_max_resp": float(max_resp.mean().item()),
        "mean_top1_minus_top2": float(margin.mean().item()),
        "hungarian_accuracy": _hungarian_accuracy(labels, pred),
        "adjusted_rand": float(adjusted_rand_score(labels.cpu().numpy(), pred.cpu().numpy())),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(labels.cpu().numpy(), pred.cpu().numpy())
        ),
        "empty_clusters": int((sizes == 0).sum().item()),
        "cluster_sizes": [int(v) for v in sizes.tolist()],
        "max_resp_hist": [float(v) for v in hist.tolist()],
    }


def _top1_minus_top2(r: torch.Tensor) -> torch.Tensor:
    if r.shape[1] == 1:
        return r[:, 0]
    top2 = r.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _hungarian_accuracy(true_labels: torch.Tensor, pred_labels: torch.Tensor) -> float:
    counts = _contingency(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment((-counts).numpy())
    matched = counts[row_ind, col_ind].sum().item()
    return float(matched / max(1, true_labels.numel()))


def _contingency(true_labels: torch.Tensor, pred_labels: torch.Tensor) -> torch.Tensor:
    true = true_labels.cpu().long()
    pred = pred_labels.cpu().long()
    n_true = int(true.max().item()) + 1
    n_pred = int(pred.max().item()) + 1
    counts = torch.zeros(n_true, n_pred, dtype=torch.float64)
    for t, p in zip(true.tolist(), pred.tolist()):
        counts[t, p] += 1.0
    return counts


def fit_one(
    cfg: SweepConfig,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    K_fit: int,
    q_fit: int,
    seed: int,
    device: torch.device,
    save_dir: Path | None = None,
    centroid_cache_path: Path | None = None,
) -> dict:
    if q_fit > cfg.D:
        raise ValueError(f"q_fit={q_fit} must be <= D={cfg.D}")

    print(f"\n=== fit K={K_fit} q={q_fit} seed={seed} ===", flush=True)
    centroids = _load_or_fit_centroids(
        x_train,
        cfg,
        K_fit=K_fit,
        seed=seed,
        cache_path=centroid_cache_path,
    )
    model = MFA(centroids, rank=q_fit, psi_init=cfg.psi, scale_init=0.1).to(device)
    train_loader = _make_loader(x_train, cfg.batch_size, seed=seed)

    train_info = train_nll(
        model,
        train_loader,
        val_tensor=x_test,
        epochs=cfg.epochs,
        lr=cfg.lr,
        grad_clip=cfg.grad_clip,
        log_interval=100,
        steps_per_epoch=len(train_loader),
        early_stop_delta=cfg.early_stop_delta,
        early_stop_patience=cfg.early_stop_patience,
        early_stop_min_delta=cfg.early_stop_min_delta,
    )

    print("Evaluating train NLL", flush=True)
    train_nll_value = _nll(model, x_train, device=device, batch_size=cfg.batch_size)
    print("Evaluating test NLL", flush=True)
    test_nll_value = _nll(model, x_test, device=device, batch_size=cfg.batch_size)
    print("Computing test responsibility metrics", flush=True)
    metrics = responsibility_metrics(
        model,
        x_test,
        y_test,
        device=device,
        batch_size=cfg.batch_size,
    )

    row = {
        "seed": seed,
        "K_true": cfg.K_true,
        "q_true": cfg.q_true,
        "D": cfg.D,
        "K_fit": K_fit,
        "q_fit": q_fit,
        "train_nll": train_nll_value,
        "test_nll": test_nll_value,
        "best_epoch": int(train_info["best_epoch"]),
        "best_val_metric": float(train_info["best_metric"]),
        **metrics,
    }
    print("Computing train Hungarian accuracy", flush=True)
    row["train_hungarian_accuracy"] = responsibility_metrics(
        model,
        x_train,
        y_train,
        device=device,
        batch_size=cfg.batch_size,
    )["hungarian_accuracy"]

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving model and metrics to {save_dir}", flush=True)
        save_mfa(
            model,
            str(save_dir / "mfa_model.pt"),
            extra={"synthetic_qk_sweep": row},
        )
        torch.save(row, save_dir / "metrics.pt")
        (save_dir / "metrics.json").write_text(json.dumps(_json_safe(row), indent=2))
        (save_dir / "config.json").write_text(json.dumps(_jsonable_config(cfg), indent=2))
        print(f"Saved model and metrics to {save_dir}")

    return row


def fit_one_from_dataset(
    cfg: SweepConfig,
    *,
    K_fit: int,
    q_fit: int,
    seed: int | None = None,
) -> dict:
    device = _resolve_device(cfg.device)
    seed = cfg.seed if seed is None else int(seed)
    dataset = load_dataset(resolve_dataset_path(cfg))
    validate_dataset_config(dataset, cfg)
    run_dir = cfg.model_root / cfg.run_name
    save_dir = run_dir / f"K{K_fit:04d}_q{q_fit:04d}_seed{seed:04d}"
    centroid_cache_path = _centroid_cache_path(run_dir, cfg, K_fit=K_fit, seed=seed)
    return fit_one(
        cfg,
        x_train=dataset["x_train"],
        y_train=dataset["y_train"],
        x_test=dataset["x_test"],
        y_test=dataset["y_test"],
        K_fit=K_fit,
        q_fit=q_fit,
        seed=seed,
        device=device,
        save_dir=save_dir,
        centroid_cache_path=centroid_cache_path,
    )


def collect_results(cfg: SweepConfig) -> list[dict]:
    run_dir = cfg.model_root / cfg.run_name
    rows = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(run_dir.glob("K*_q*_seed*/metrics.pt"))
    ]
    if rows:
        cfg.K_fit = tuple(sorted({int(row["K_fit"]) for row in rows}))
        cfg.q_fit = tuple(sorted({int(row["q_fit"]) for row in rows}))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, run_dir / "results.csv")
    torch.save(rows, run_dir / "results.pt")
    (run_dir / "config.json").write_text(json.dumps(_jsonable_config(cfg), indent=2))
    if rows and not cfg.no_plots:
        save_plots(rows, run_dir, cfg)
    print(f"Collected {len(rows)} fit results in {run_dir}")
    return rows


def _json_safe(obj):
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_csv(rows: list[dict], path: Path) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: value
                for key, value in row.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
        )
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def save_plots(rows: list[dict], out_dir: Path, cfg: SweepConfig) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    import matplotlib.pyplot as plt

    plot_specs = [
        ("mean_entropy", "Mean responsibility entropy"),
        ("norm_mean_entropy", "Mean entropy / log(K_fit)"),
        ("mean_max_resp", "Mean max responsibility"),
        ("mean_top1_minus_top2", "Mean top1 - top2 responsibility"),
        ("hungarian_accuracy", "Hungarian-matched accuracy"),
        ("adjusted_rand", "Adjusted Rand index"),
        ("normalized_mutual_info", "Normalized mutual information"),
        ("test_nll", "Test NLL"),
    ]
    for metric, title in plot_specs:
        q_values = list(cfg.q_fit)
        q_positions = list(range(len(q_values)))
        fig_width = max(9.0, 0.38 * len(q_values) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        for K in cfg.K_fit:
            values = [
                _mean_metric(rows, metric, K_fit=K, q_fit=q)
                for q in q_values
            ]
            ax.plot(q_positions, values, marker="o", linewidth=1.5, markersize=5, label=f"K={K}")
        if cfg.q_true in q_values:
            ax.axvline(q_values.index(cfg.q_true), color="black", linestyle="--", linewidth=1, label="q true")
        ax.set_title(title)
        ax.set_xlabel("q_fit")
        ax.set_ylabel(metric)
        ax.set_xticks(q_positions)
        ax.set_xticklabels([str(q) for q in q_values], rotation=45, ha="right")
        ax.grid(True, axis="both", alpha=0.25)
        if metric in {
            "hungarian_accuracy",
            "adjusted_rand",
            "normalized_mutual_info",
            "mean_max_resp",
            "mean_top1_minus_top2",
        }:
            ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_vs_q.png", dpi=160)
        plt.close(fig)

        grid = torch.tensor(
            [
                [_mean_metric(rows, metric, K_fit=K, q_fit=q) for q in cfg.q_fit]
                for K in cfg.K_fit
            ],
            dtype=torch.float32,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(grid.numpy(), aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("q_fit")
        ax.set_ylabel("K_fit")
        ax.set_xticks(range(len(cfg.q_fit)))
        ax.set_xticklabels([str(q) for q in cfg.q_fit])
        ax.set_yticks(range(len(cfg.K_fit)))
        ax.set_yticklabels([str(K) for K in cfg.K_fit])
        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_heatmap.png", dpi=160)
        plt.close(fig)


def _mean_metric(rows: list[dict], metric: str, *, K_fit: int, q_fit: int) -> float:
    values = [
        float(row[metric])
        for row in rows
        if int(row["K_fit"]) == int(K_fit) and int(row["q_fit"]) == int(q_fit)
    ]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic MFA K/q responsibility sweep")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-dataset", help="Generate and save the synthetic dataset")
    _add_shared_args(generate)

    fit = subparsers.add_parser("fit-one", help="Fit one MFA for a single K/q setting")
    _add_shared_args(fit)
    fit.add_argument("--K-fit", type=int, required=True)
    fit.add_argument("--q-fit", type=int, required=True)

    collect = subparsers.add_parser("collect-results", help="Merge per-job metrics and write plots")
    _add_shared_args(collect)
    return parser


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--D", type=int, default=500)
    parser.add_argument("--K-true", type=int, default=8)
    parser.add_argument("--q-true", type=int, default=100)
    parser.add_argument("--K-grid", type=parse_int_list, default=(4, 6, 8, 10, 12))
    parser.add_argument("--q-grid", type=parse_int_list, default=(10, 50, 100, 200, 500))
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--mean-scale", type=float, default=6.0)
    parser.add_argument("--factor-scale", type=float, default=1.0)
    parser.add_argument("--psi", type=float, default=0.25)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-n-init", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-plots", action="store_true")


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        dataset_path=args.dataset_path,
        model_root=args.model_root,
        run_name=args.run_name,
        D=args.D,
        K_true=args.K_true,
        q_true=args.q_true,
        K_fit=tuple(args.K_grid),
        q_fit=tuple(args.q_grid),
        n_train=args.n_train,
        n_test=args.n_test,
        n_seeds=args.n_seeds,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        early_stop_delta=args.early_stop_delta,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        mean_scale=args.mean_scale,
        factor_scale=args.factor_scale,
        psi=args.psi,
        kmeans_max_iter=args.kmeans_max_iter,
        kmeans_n_init=args.kmeans_n_init,
        device=args.device,
        no_plots=args.no_plots,
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    if args.command == "generate-dataset":
        generate_dataset(cfg)
    elif args.command == "fit-one":
        fit_one_from_dataset(cfg, K_fit=args.K_fit, q_fit=args.q_fit)
    elif args.command == "collect-results":
        collect_results(cfg)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
