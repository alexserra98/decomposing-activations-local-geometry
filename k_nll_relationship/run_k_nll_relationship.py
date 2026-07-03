"""Synthetic experiment for the relationship between MFA K and optimal NLL.

The experiment creates one ground-truth MFA for each K in
``[10, 50, 100, 500, 1000]`` with fixed ``D=500`` and ``q=20``. It samples a
separate dataset from each ground-truth model, then fits a fresh MFA with the
same K and q to that dataset. The fitted model is initialized independently, so
it has the same architecture but different parameters from the generator.

Example:

    PYTHONPATH=../src uv run python run_k_nll_relationship.py

For a quick smoke test:

    PYTHONPATH=../src uv run python run_k_nll_relationship.py \
      --K-grid 3,5 --D 12 --q 2 --n-train 120 --n-test 60 \
      --epochs 1 --batch-size 32 --device cpu --no-plots
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
from typing import Any

import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dalg.models.mfa import MFA, save_mfa  # noqa: E402
from dalg.models.train import train_nll  # noqa: E402


DEFAULT_K_GRID = (10, 50, 100, 500, 1000)


@dataclass
class ExperimentConfig:
    out_dir: Path = Path(__file__).resolve().parent / "runs"
    K_grid: tuple[int, ...] = DEFAULT_K_GRID
    D: int = 500
    q: int = 20
    n_train: int = 50_000
    n_test: int = 10_000
    seed: int = 0
    batch_size: int = 512
    epochs: int = 25
    lr: float = 1e-3
    grad_clip: float | None = 5.0
    early_stop_patience: int | None = 5
    early_stop_min_delta: float = 1e-3
    early_stop_delta: float = 1e-4
    mean_scale: float = 6.0
    factor_scale: float = 1.0
    psi: float = 0.25
    kmeans_max_iter: int = 100
    kmeans_n_init: int = 3
    device: str = "cuda"
    no_plots: bool = False
    force: bool = False


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return items


def resolve_device(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but CUDA is unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        print("Requested MPS but MPS is unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return requested


def inv_softplus(x: float) -> float:
    return math.log(math.expm1(float(x)))


def orthonormal_columns(D: int, q: int, *, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(D, q, generator=generator)
    q_mat, _ = torch.linalg.qr(raw, mode="reduced")
    return q_mat


def centroid_directions(D: int, K: int, *, generator: torch.Generator) -> torch.Tensor:
    if K <= D:
        return orthonormal_columns(D, K, generator=generator).T
    directions = torch.randn(K, D, generator=generator)
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)


def make_ground_truth(cfg: ExperimentConfig, *, K: int, seed: int) -> MFA:
    if cfg.q > cfg.D:
        raise ValueError(f"q={cfg.q} must be <= D={cfg.D}")

    generator = torch.Generator().manual_seed(seed)
    mu = cfg.mean_scale * centroid_directions(cfg.D, K, generator=generator)
    model = MFA(mu, rank=cfg.q, psi_init=cfg.psi, scale_init=cfg.factor_scale)

    with torch.no_grad():
        model.mu.copy_(mu)
        directions = torch.stack(
            [orthonormal_columns(cfg.D, cfg.q, generator=generator) for _ in range(K)],
            dim=0,
        )
        model.dir_raw.copy_(directions)
        model.scale_rho.fill_(inv_softplus(cfg.factor_scale))
        model.psi_rho.fill_(inv_softplus(cfg.psi))
        model.pi_logits.zero_()
    return model


@torch.no_grad()
def sample_from_mfa(model: MFA, n: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, model.K, (n,), generator=generator)
    z = torch.randn(n, model.q, generator=generator)
    eps = torch.randn(n, model.D, generator=generator)

    mu = model.mu.detach().cpu()[labels]
    W = model.W.detach().cpu()[labels]
    psi = model._psi().detach().cpu()[labels]
    x = mu + torch.einsum("ndq,nq->nd", W, z) + eps * psi.sqrt()
    return x.float(), labels.long()


def generate_dataset(cfg: ExperimentConfig, *, K: int, k_dir: Path) -> dict[str, Any]:
    dataset_path = k_dir / "dataset.pt"
    truth_path = k_dir / "ground_truth_mfa.pt"
    if dataset_path.exists() and truth_path.exists() and not cfg.force:
        print(f"[K={K}] loading existing dataset from {dataset_path}", flush=True)
        return torch.load(dataset_path, map_location="cpu", weights_only=False)

    print(f"[K={K}] creating ground-truth MFA and synthetic dataset", flush=True)
    truth_seed = cfg.seed + K * 10
    truth = make_ground_truth(cfg, K=K, seed=truth_seed)
    x_train, y_train = sample_from_mfa(truth, cfg.n_train, seed=truth_seed + 1)
    x_test, y_test = sample_from_mfa(truth, cfg.n_test, seed=truth_seed + 2)

    dataset = {
        "config": jsonable_config(cfg),
        "K_true": K,
        "q_true": cfg.q,
        "D": cfg.D,
        "truth_seed": truth_seed,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }
    k_dir.mkdir(parents=True, exist_ok=True)
    save_mfa(truth, str(truth_path), extra={"K_true": K, "q_true": cfg.q, "D": cfg.D})
    torch.save(dataset, dataset_path)
    return dataset


def kmeans_centroids(
    x: torch.Tensor,
    *,
    K: int,
    seed: int,
    max_iter: int,
    n_init: int,
) -> torch.Tensor:
    start = time.time()
    print(
        f"[K={K}] KMeans init on n={x.shape[0]} D={x.shape[1]} "
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
    print(f"[K={K}] KMeans finished in {time.time() - start:.1f}s", flush=True)
    return torch.tensor(km.cluster_centers_, dtype=torch.float32)


def load_or_fit_centroids(cfg: ExperimentConfig, x_train: torch.Tensor, *, K: int, k_dir: Path) -> torch.Tensor:
    path = k_dir / "fit_centroids.pt"
    if path.exists() and not cfg.force:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        centroids = saved["centroids"] if isinstance(saved, dict) else saved
        if tuple(centroids.shape) != (K, cfg.D):
            raise ValueError(f"cached centroids have shape {tuple(centroids.shape)}, expected {(K, cfg.D)}")
        print(f"[K={K}] loaded cached fit centroids from {path}", flush=True)
        return centroids.float()

    centroids = kmeans_centroids(
        x_train,
        K=K,
        seed=cfg.seed + K * 100,
        max_iter=cfg.kmeans_max_iter,
        n_init=cfg.kmeans_n_init,
    )
    torch.save({"centroids": centroids, "K": K, "D": cfg.D, "seed": cfg.seed}, path)
    return centroids


def make_loader(x: torch.Tensor, batch_size: int, *, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True, generator=generator)


@torch.no_grad()
def nll(model: MFA, x: torch.Tensor, *, device: torch.device, batch_size: int) -> float:
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


def fit_matched_mfa(
    cfg: ExperimentConfig,
    dataset: dict[str, Any],
    *,
    K: int,
    k_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    metrics_path = k_dir / "metrics.json"
    model_path = k_dir / "fitted_mfa.pt"
    if metrics_path.exists() and model_path.exists() and not cfg.force:
        print(f"[K={K}] loading existing metrics from {metrics_path}", flush=True)
        return json.loads(metrics_path.read_text())

    x_train = dataset["x_train"]
    x_test = dataset["x_test"]
    centroids = load_or_fit_centroids(cfg, x_train, K=K, k_dir=k_dir)

    fit_seed = cfg.seed + K * 1000
    torch.manual_seed(fit_seed)
    model = MFA(centroids, rank=cfg.q, psi_init=cfg.psi, scale_init=0.1).to(device)
    loader = make_loader(x_train, cfg.batch_size, seed=fit_seed)

    print(f"[K={K}] fitting fresh MFA with matched K={K}, q={cfg.q}", flush=True)
    train_info = train_nll(
        model,
        loader,
        val_tensor=x_test,
        epochs=cfg.epochs,
        lr=cfg.lr,
        grad_clip=cfg.grad_clip,
        log_interval=100,
        steps_per_epoch=len(loader),
        early_stop_delta=cfg.early_stop_delta,
        early_stop_patience=cfg.early_stop_patience,
        early_stop_min_delta=cfg.early_stop_min_delta,
    )

    fitted_train_nll = nll(model, x_train, device=device, batch_size=cfg.batch_size)
    fitted_test_nll = nll(model, x_test, device=device, batch_size=cfg.batch_size)
    truth = make_ground_truth(cfg, K=K, seed=int(dataset["truth_seed"])).to(device)
    true_train_nll = nll(truth, x_train, device=device, batch_size=cfg.batch_size)
    true_test_nll = nll(truth, x_test, device=device, batch_size=cfg.batch_size)

    row = {
        "K": K,
        "D": cfg.D,
        "q": cfg.q,
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "fit_seed": fit_seed,
        "train_nll": fitted_train_nll,
        "test_nll": fitted_test_nll,
        "true_train_nll": true_train_nll,
        "true_test_nll": true_test_nll,
        "excess_test_nll": fitted_test_nll - true_test_nll,
        "best_epoch": int(train_info["best_epoch"]),
        "best_val_metric": float(train_info["best_metric"]),
    }

    print(f"[K={K}] saving fitted model and metrics", flush=True)
    save_mfa(model, str(model_path), extra={"k_nll_relationship": row})
    torch.save(row, k_dir / "metrics.pt")
    metrics_path.write_text(json.dumps(json_safe(row), indent=2))
    return row


def linear_fit(rows: list[dict[str, Any]], *, x_key: str, y_key: str) -> dict[str, float]:
    xs = torch.tensor([float(row[x_key]) for row in rows], dtype=torch.float64)
    ys = torch.tensor([float(row[y_key]) for row in rows], dtype=torch.float64)
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if float(denom) == 0.0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    slope = ((xs - x_mean) * (ys - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    pred = intercept + slope * xs
    ss_res = ((ys - pred) ** 2).sum()
    ss_tot = ((ys - y_mean) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if float(ss_tot) > 0.0 else torch.tensor(float("nan"))
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def write_summary(cfg: ExperimentConfig, rows: list[dict[str, Any]]) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["K"]))
    csv_path = cfg.out_dir / "summary.csv"
    fieldnames = sorted({key for row in rows for key, value in row.items() if is_scalar(value)})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])

    summary = {
        "config": jsonable_config(cfg),
        "rows": rows,
        "linear_test_nll_vs_K": linear_fit(rows, x_key="K", y_key="test_nll"),
        "linear_true_test_nll_vs_K": linear_fit(rows, x_key="K", y_key="true_test_nll"),
        "linear_excess_test_nll_vs_K": linear_fit(rows, x_key="K", y_key="excess_test_nll"),
    }
    torch.save(rows, cfg.out_dir / "summary.pt")
    (cfg.out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2))
    (cfg.out_dir / "config.json").write_text(json.dumps(jsonable_config(cfg), indent=2))
    if not cfg.no_plots:
        save_plots(cfg, rows)
    print(f"Wrote summary to {csv_path} and {cfg.out_dir / 'summary.json'}", flush=True)


def save_plots(cfg: ExperimentConfig, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(cfg.out_dir / ".matplotlib"))
    import matplotlib.pyplot as plt

    K = [int(row["K"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(K, [float(row["test_nll"]) for row in rows], marker="o", label="fitted MFA")
    ax.plot(K, [float(row["true_test_nll"]) for row in rows], marker="o", label="ground truth MFA")
    ax.plot(K, [float(row["excess_test_nll"]) for row in rows], marker="o", label="fit - truth")
    ax.set_xlabel("K")
    ax.set_ylabel("NLL")
    ax.set_title("Matched MFA test NLL vs K")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "test_nll_vs_K.png", dpi=160)
    plt.close(fig)


def run_experiment(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Writing experiment artifacts under {cfg.out_dir}", flush=True)
    print(f"Using device: {device}", flush=True)

    rows: list[dict[str, Any]] = []
    for K in cfg.K_grid:
        k_dir = cfg.out_dir / f"K{K:04d}"
        dataset = generate_dataset(cfg, K=K, k_dir=k_dir)
        rows.append(fit_matched_mfa(cfg, dataset, K=K, k_dir=k_dir, device=device))
        write_summary(cfg, rows)
    return rows


def jsonable_config(cfg: ExperimentConfig) -> dict[str, Any]:
    out = asdict(cfg)
    out["out_dir"] = str(cfg.out_dir)
    out["K_grid"] = list(cfg.K_grid)
    return out


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic MFA K vs matched-fit NLL experiment")
    parser.add_argument("--out-dir", type=Path, default=ExperimentConfig.out_dir)
    parser.add_argument("--K-grid", type=parse_int_list, default=DEFAULT_K_GRID)
    parser.add_argument("--D", type=int, default=500)
    parser.add_argument("--q", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    parser.add_argument("--mean-scale", type=float, default=6.0)
    parser.add_argument("--factor-scale", type=float, default=1.0)
    parser.add_argument("--psi", type=float, default=0.25)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-n-init", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite cached datasets, centroids, and fits")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        out_dir=args.out_dir,
        K_grid=tuple(args.K_grid),
        D=args.D,
        q=args.q,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_delta=args.early_stop_delta,
        mean_scale=args.mean_scale,
        factor_scale=args.factor_scale,
        psi=args.psi,
        kmeans_max_iter=args.kmeans_max_iter,
        kmeans_n_init=args.kmeans_n_init,
        device=args.device,
        no_plots=args.no_plots,
        force=args.force,
    )


def main() -> None:
    cfg = config_from_args(build_parser().parse_args())
    run_experiment(cfg)


if __name__ == "__main__":
    main()
