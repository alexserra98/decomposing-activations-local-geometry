r"""Plot held-out coverage for selected saved toy-manifold pipeline runs.

From the repository root::

    PYTHONPATH=src .venv/bin/python scripts/temporary/plot_toy_coverage.py \
        --run kmeans /path/to/kmeans_run --run mfa /path/to/mfa_run \
        --output outputs/coverage/comparison.png

Each run needs config.json, val_indices.json, its model export, and its saved
*_model_assignments.pt. Runs must share the dataset, layer, and training rows.
Uses the independent dataset/test population. No training or benchmark update
is performed. Writes the figure plus .json summaries and .pt plotting data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec
from dalg.evaluation.coverage import evaluate_heldout_distribution_coverage
from dalg.evaluation.toy_manifold_coverage import validate_toy_test_split
from dalg.evaluation.toy_manifold_tiling import _load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_LABELS = {"kmeans": "KMeans+PCA", "mfa": "MFA", "ard": "MFA-ARD", "hddc": "MFA-HDDC"}


def _source_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _load_run(kind, directory):
    if kind not in MODEL_LABELS:
        raise ValueError(f"unsupported model kind {kind!r}; choose {list(MODEL_LABELS)}")
    directory = Path(directory).expanduser().resolve()
    cfg = json.loads((directory / "config.json").read_text())
    saved_kind = cfg.get("model_kind") or {
        "MFA": "mfa", "MFA_ARD": "ard", "MFA_HDDC": "hddc",
    }.get(cfg.get("model", "MFA"))
    if cfg.get("method") == "kmeans":
        saved_kind = "kmeans"
    if kind != saved_kind:
        raise ValueError(f"requested kind {kind!r} differs from saved config: {saved_kind!r}")
    root, suffix = split_shard_dir_spec(cfg["shard_dir"])
    root = _source_path(root)
    subset = cfg.get("subset_spec", suffix)
    if suffix is not None and suffix != subset:
        raise ValueError("run config has conflicting subset specifications")
    layer = int(cfg["layer"])
    shard_cfg = json.loads((root / "config.json").read_text())
    if shard_cfg["window"] != 1 or cfg.get("drop_prefix", 0) != 0:
        raise ValueError("coverage plotting requires one toy activation per row")
    meta = load_meta_index(root, layer)
    positions = resolve_spec_positions(meta, subset, window=1, drop_prefix=0)
    split = json.loads((directory / "val_indices.json").read_text())
    validation = set(split["val_global_rows"])
    train_mask = torch.tensor([meta[p]["global_row"] not in validation for p in positions])
    train = [p for p, keep in zip(positions, train_mask.tolist(), strict=True) if keep]
    if (not train or len(train) != split["train_rows"]
            or len(positions) - len(train) != split["val_rows"]
            or len(validation) != split["val_rows"]):
        raise ValueError(f"saved training/validation split does not match rows: {directory}")

    stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
    model_path = directory / f"{stem}.pt"
    assignment_path = directory / f"{stem}_assignments.pt"
    bundle = torch.load(assignment_path, map_location="cpu", weights_only=True)
    assignments = bundle["assignments"].reshape(-1)
    sizes = bundle["cluster_sizes"].reshape(-1)
    model = _load_model(directory, kind)
    if model.D != shard_cfg["d_model"] or model.D < 2:
        raise ValueError("model dimension does not match the toy dataset")
    if (assignments.dtype != torch.int64 or assignments.numel() != len(positions)
            or bundle.get("subset_spec") != subset or bundle["K"] != model.K
            or bool((assignments < 0).any()) or bool((assignments >= model.K).any())
            or not torch.equal(torch.bincount(assignments, minlength=model.K), sizes)):
        raise ValueError(f"assignment bundle does not match the model/stream: {assignment_path}")
    if "model_path" in bundle and _source_path(bundle["model_path"]) != model_path:
        raise ValueError("assignment model_path differs from the selected model")
    if "source" in bundle:
        source = bundle["source"]
        if (_source_path(source["shard_dir"]) != root or source["layer"] != layer
                or source["drop_prefix"] != 0 or source["num_items"] != len(positions)):
            raise ValueError("assignment source differs from the selected dataset")
    live = torch.bincount(assignments[train_mask], minlength=model.K) > 0
    return {
        "kind": kind, "run_dir": str(directory), "root": root, "layer": layer,
        "train": train, "centroids": model.mu.detach().cpu().float(), "live": live,
        "assignment_path": str(assignment_path),
    }


def _stream(root, layer, batch_size, positions=None):
    return ActivationBatchDataset(
        root, layer=layer, row_subset=positions, batch_size=batch_size,
        drop_prefix=0, dtype=torch.float32,
        shuffle_shards=False, shuffle_within_shard=False,
    )


@torch.no_grad()
def _training_pca(run, batch_size):
    """Accumulate training covariance without materializing the training set."""
    dimension = run["centroids"].shape[1]
    count = 0
    mean = torch.zeros(dimension, dtype=torch.float64)
    scatter = torch.zeros(dimension, dimension, dtype=torch.float64)
    for batch in _stream(run["root"], run["layer"], batch_size, run["train"]):
        batch = batch.double()
        if not torch.isfinite(batch).all():
            raise ValueError("training activations contain non-finite values")
        n = len(batch)
        batch_mean = batch.mean(0)
        centered = batch - batch_mean
        delta = batch_mean - mean
        scatter += centered.T @ centered + torch.outer(delta, delta) * (count * n / (count + n))
        mean += delta * (n / (count + n))
        count += n
    if count != len(run["train"]) or count < 2:
        raise ValueError("training stream does not match the saved split")
    _, vectors = torch.linalg.eigh(scatter / (count - 1))
    return mean, vectors[:, -2:].flip(-1)


def _write_plot(output, test_2d, panels, selected, title):
    """Use the shared-scale layout from run_toy_d128_heldout_coverage.py."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(panels), figsize=(6.75 * len(panels), 5.6),
        sharex=True, sharey=True, layout="constrained", squeeze=False,
    )
    color_max = max(panel["coverage"]["r99"] for panel in panels)
    for axis, panel in zip(axes[0], panels, strict=True):
        summary = panel["coverage"]
        points = test_2d[selected]
        scatter = axis.scatter(
            points[:, 0], points[:, 1], c=panel["distances"][selected],
            cmap="viridis", vmin=0, vmax=color_max, s=3, alpha=0.45,
            linewidths=0, rasterized=True,
        )
        centers = panel["centroids_2d"]
        axis.scatter(
            centers[:, 0], centers[:, 1], c="#e63946", marker="x", s=20,
            linewidths=0.8, label=f"live centroids ({summary['components_live']})",
        )
        axis.set_title(
            f"{panel['label']}\nmean={summary['mean']:.4f}, "
            f"r95={summary['r95']:.4f}, r99={summary['r99']:.4f}"
        )
        axis.set_xlabel("training PCA component 1")
        axis.grid(alpha=0.15)
        axis.legend(loc="best", fontsize=8)
    axes[0, 0].set_ylabel("training PCA component 2")
    figure.colorbar(
        scatter, ax=axes[0].tolist(), shrink=0.88,
        label="ambient distance to nearest live centroid (clipped at largest r99)",
    )
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


@torch.no_grad()
def plot_coverage(runs, output, *, labels=None, device="cpu", batch_size=1024,
                  max_points=0, seed=0, title=None):
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("output must have a .png, .pdf, or .svg suffix")
    for path in (output, output.with_suffix(".json"), output.with_suffix(".pt")):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
    if not runs or batch_size <= 0 or max_points < 0:
        raise ValueError("provide runs, a positive batch size, and max-points >= 0")
    if labels is not None and len(labels) != len(runs):
        raise ValueError("provide exactly one label per run")
    loaded = [_load_run(kind, directory) for kind, directory in runs]
    first = loaded[0]
    for run in loaded[1:]:
        if any(run[key] != first[key] for key in ("root", "layer", "train")):
            raise ValueError("selected runs must share the dataset, layer, and training split")
    source = validate_toy_test_split(first["root"], layer=first["layer"])
    print("Fitting shared PCA on training rows", flush=True)
    mean, basis = _training_pca(first, batch_size)
    test = torch.cat(list(_stream(source["shard_dir"], source["layer"], batch_size)))
    if len(test) != source["num_rows"]:
        raise ValueError("test stream does not match the declared row count")
    test_2d = (test.double() - mean) @ basis
    test = test.to(device)
    panels = []
    for index, run in enumerate(loaded):
        label = labels[index] if labels else f"{MODEL_LABELS[run['kind']]} (K={len(run['live'])})"
        print(f"Computing coverage: {label}", flush=True)
        summary, distances = evaluate_heldout_distribution_coverage(
            test, run["centroids"].to(device), live_components=run["live"].to(device),
            batch_size=batch_size,
        )
        panels.append({
            "label": label, "run_dir": run["run_dir"], "model_kind": run["kind"],
            "assignment_path": run["assignment_path"], "coverage": summary,
            "distances": distances.cpu(), "live": run["live"],
            "centroids_2d": (run["centroids"][run["live"]].double() - mean) @ basis,
        })
    selected = torch.arange(len(test))
    if 0 < max_points < len(test):
        selected = torch.randperm(len(test), generator=torch.Generator().manual_seed(seed))[:max_points]
    if title is None:
        config = json.loads((first["root"] / "config.json").read_text())
        noise = config["generator_config"].get("noise_ratio")
        condition = "Noiseless" if noise is None else f"Noise η={noise:g}"
        title = f"{condition}, D={test.shape[1]} toy manifolds: held-out test split"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_plot(output, test_2d, panels, selected, title)
    report = {
        "source": source, "pca_fit_population": "train", "liveness_split": "train",
        "training_rows": len(first["train"]), "plotted_points": len(selected),
        "plot_seed": seed, "title": title,
        "panels": [{k: v for k, v in panel.items() if not isinstance(v, torch.Tensor)}
                   for panel in panels],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({
        "mean": mean, "basis": basis, "fit_population": "train",
        "train_positions": torch.tensor(first["train"]), "test_2d": test_2d,
        "selected_test_positions": selected, "panels": panels,
    }, output.with_suffix(".pt"))
    print(f"Saved {output} (plus .json and .pt)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("KIND", "DIRECTORY"),
                        help="Repeat per panel; KIND is kmeans, mfa, ard, or hddc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", help="Panel titles, in --run order")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=0,
                        help="Display sample size; 0 plots all. Metrics always use every test point.")
    parser.add_argument("--seed", type=int, default=0, help="Display subsampling seed")
    parser.add_argument("--title")
    args = vars(parser.parse_args())
    args["runs"] = args.pop("run")
    plot_coverage(**args)


if __name__ == "__main__":
    main()
