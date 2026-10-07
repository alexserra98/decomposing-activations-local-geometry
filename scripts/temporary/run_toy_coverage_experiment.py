"""Compare held-out centroid coverage for KMeans+PCA and MFA on noisy spheres."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from dalg.data.manifold_dataset import ToyManifoldConfig, make_toy_manifold_dataset
from dalg.evaluation.coverage import (
    evaluate_heldout_distribution_coverage,
    split_fingerprint,
    stratified_three_way_split,
)
from dalg.evaluation.toy_manifold_metrics import evaluate_toy_manifold_metrics
from dalg.init.centroid_artifact import compute_cluster_pca_directions
from dalg.init.projected_knn import KMeansTorch
from dalg.models.mfa import MFA, load_mfa, save_mfa
from dalg.models.train import train_nll


@torch.no_grad()
def _hard_mfa_assignments(
    model: MFA,
    points: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    assignments = []
    model = model.to(device).eval()
    with model.inference_cache(enabled=True):
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size].to(device)
            assignments.append(model.responsibilities(batch).argmax(dim=1).cpu())
    resolved = torch.cat(assignments)
    return resolved, torch.bincount(resolved, minlength=model.K)


class _KMeansPCAGeometry:
    """Minimal MFA-shaped view used by the existing tangent evaluator."""

    def __init__(self, centroids: torch.Tensor, directions: torch.Tensor) -> None:
        self.mu = centroids.detach().cpu().float()
        self._directions = directions.detach().cpu().float()
        self.K, self.D = map(int, self.mu.shape)
        self.q = int(self._directions.shape[2])
        self.rank_mask = torch.ones(self.K, self.q)

    def _W(self) -> torch.Tensor:
        return self._directions

    def _psi(self) -> torch.Tensor:
        return torch.full((self.K, self.D), 1e-6)


def _fit_kmeans_pca(
    train_points: torch.Tensor,
    *,
    K: int,
    rank: int,
    seed: int,
    device: torch.device,
    n_iter: int,
    restarts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    kmeans = KMeansTorch(
        k=K,
        metric="euclidean",
        n_iter=n_iter,
        restarts=restarts,
        tol=1e-6,
        seed=seed,
        device=device,
    )
    device_points = train_points.to(device)
    centroids = kmeans.fit(device_points)
    assignments = kmeans._assign_streamed(device_points, centroids)
    cluster_sizes = torch.bincount(assignments, minlength=K)
    directions = compute_cluster_pca_directions(
        device_points,
        assignments,
        centroids,
        rank=rank,
    )
    return (
        centroids.detach().cpu(),
        directions.detach().cpu(),
        assignments.detach().cpu(),
        cluster_sizes.detach().cpu(),
        float(kmeans.inertia_),
    )


def _fit_mfa(
    train_points: torch.Tensor,
    validation_points: torch.Tensor,
    *,
    centroids: torch.Tensor,
    directions: torch.Tensor,
    cluster_sizes: torch.Tensor,
    rank: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
) -> tuple[MFA, dict[str, float | int]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    psi_init = max(
        float(((train_points[:, None, :] - centroids[None, :, :]).square().sum(dim=2).min(dim=1).values).mean())
        / train_points.shape[1],
        1e-3,
    )
    model = MFA(
        centroids,
        rank=rank,
        init_directions=directions,
        psi_init=psi_init,
        psi_per_component=True,
    ).to(device)
    with torch.no_grad():
        model.pi_logits.copy_(cluster_sizes.to(device).clamp_min(1).float().log())

    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_points),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    training = train_nll(
        model,
        train_loader,
        val_tensor=validation_points,
        epochs=epochs,
        lr=learning_rate,
        early_stop_delta=0.0,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        epoch_snapshot_every=0,
        log_interval=max(1, math.ceil(len(train_loader) / 4)),
    )
    return model, training


def _alignment_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "association": metrics["association"],
        "tangent_alignment": metrics["tangent_alignment"],
        "tangent_containment": metrics["tangent_containment"],
        "per_manifold": [
            {
                "manifold_id": item["manifold_id"],
                "type_name": item["type_name"],
                "components": item["components"],
                "tangent_alignment": item["tangent_alignment"],
                "tangent_containment": item["tangent_containment"],
            }
            for item in metrics["per_manifold"]
        ],
    }


def _projection(
    train_points: torch.Tensor,
    tensors: list[torch.Tensor],
    *,
    dimensions: int,
) -> list[torch.Tensor]:
    center = train_points.double().mean(dim=0)
    if train_points.shape[1] <= dimensions:
        basis = torch.eye(train_points.shape[1], dtype=torch.float64)
    else:
        _u, _s, vh = torch.linalg.svd(train_points.double() - center, full_matrices=False)
        basis = vh[:dimensions].T
    return [(tensor.double() - center) @ basis for tensor in tensors]


def _metric_title(method: str, result: dict[str, Any]) -> str:
    coverage = result["coverage"]
    alignment = result["geometry"]["tangent_alignment"]
    overlap = alignment["subspace_overlap"]["mean"]
    worst = alignment["worst_direction_cosine"]["mean"]
    return (
        f"{method}<br>mean={coverage['mean']:.3f}, r95={coverage['r95']:.3f}, "
        f"r99={coverage['r99']:.3f}<br>alignment overlap={overlap:.3f}, "
        f"worst cosine={worst:.3f}"
    )


def _write_plotly_html(
    path: Path,
    *,
    data: list[dict[str, Any]],
    layout: dict[str, Any],
) -> None:
    payload = json.dumps({"data": data, "layout": layout})
    path.write_text(
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8">'
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
        "</head><body>"
        '<div id="plot" style="width:100%;height:100vh"></div>'
        f"<script>const figure={payload};"
        "Plotly.newPlot('plot', figure.data, figure.layout, {responsive:true});"
        "</script></body></html>\n"
    )


def _write_embedding_figure(
    path: Path,
    *,
    noise_ratio: float,
    train_points: torch.Tensor,
    test_points: torch.Tensor,
    methods: dict[str, dict[str, Any]],
    dimensions: int,
    max_points: int,
    seed: int,
) -> None:
    names = list(methods)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = torch.randperm(len(test_points), generator=generator)[:max_points]
    projected = _projection(
        train_points,
        [test_points] + [methods[name]["centroids"] for name in names],
        dimensions=dimensions,
    )
    projected_test = projected[0][selected]
    traces: list[dict[str, Any]] = []
    layout: dict[str, Any] = {
        "title": f"Held-out sphere coverage — noise_ratio={noise_ratio:g}",
        "height": 700,
        "template": "plotly_white",
        "annotations": [],
    }
    for column, (name, result, centers) in enumerate(
        zip(names, (methods[name] for name in names), projected[1:], strict=True),
        start=1,
    ):
        distances = result["distances"][selected]
        suffix = "" if column == 1 else str(column)
        x_domain = [0.0, 0.47] if column == 1 else [0.53, 1.0]
        layout["annotations"].append(
            {
                "text": _metric_title(name, result),
                "x": sum(x_domain) / 2.0,
                "y": 1.05,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "align": "center",
            }
        )
        if dimensions == 3:
            scene_name = f"scene{suffix}"
            layout[scene_name] = {
                "domain": {"x": x_domain, "y": [0.0, 0.95]},
                "xaxis": {"title": "projection 1"},
                "yaxis": {"title": "projection 2"},
                "zaxis": {"title": "projection 3"},
                "aspectmode": "data",
            }
            traces.extend(
                [
                    {
                        "type": "scatter3d",
                        "scene": scene_name,
                        "x": projected_test[:, 0].tolist(),
                        "y": projected_test[:, 1].tolist(),
                        "z": projected_test[:, 2].tolist(),
                        "mode": "markers",
                        "name": f"{name} test points",
                        "marker": {
                            "size": 2,
                            "opacity": 0.55,
                            "color": distances.tolist(),
                            "colorscale": "Viridis",
                            "colorbar": {
                                "title": "nearest-centroid distance",
                                "x": x_domain[1],
                                "len": 0.65,
                            },
                        },
                        "showlegend": False,
                    },
                    {
                        "type": "scatter3d",
                        "scene": scene_name,
                        "x": centers[:, 0].tolist(),
                        "y": centers[:, 1].tolist(),
                        "z": centers[:, 2].tolist(),
                        "mode": "markers",
                        "name": f"{name} live centroids",
                        "marker": {"size": 5, "color": "crimson", "symbol": "diamond"},
                    },
                ]
            )
        else:
            xaxis_name = f"xaxis{suffix}"
            yaxis_name = f"yaxis{suffix}"
            trace_xaxis = f"x{suffix}"
            trace_yaxis = f"y{suffix}"
            layout[xaxis_name] = {"domain": x_domain, "title": "projection 1"}
            layout[yaxis_name] = {
                "anchor": trace_xaxis,
                "title": "projection 2",
                "scaleanchor": trace_xaxis,
                "scaleratio": 1,
            }
            traces.extend(
                [
                    {
                        "type": "scattergl",
                        "xaxis": trace_xaxis,
                        "yaxis": trace_yaxis,
                        "x": projected_test[:, 0].tolist(),
                        "y": projected_test[:, 1].tolist(),
                        "mode": "markers",
                        "name": f"{name} test points",
                        "marker": {
                            "size": 2,
                            "opacity": 0.55,
                            "color": distances.tolist(),
                            "colorscale": "Viridis",
                            "colorbar": {
                                "title": "nearest-centroid distance",
                                "x": x_domain[1],
                                "len": 0.65,
                            },
                        },
                        "showlegend": False,
                    },
                    {
                        "type": "scattergl",
                        "xaxis": trace_xaxis,
                        "yaxis": trace_yaxis,
                        "x": centers[:, 0].tolist(),
                        "y": centers[:, 1].tolist(),
                        "mode": "markers",
                        "name": f"{name} live centroids",
                        "marker": {"size": 8, "color": "crimson", "symbol": "diamond"},
                    },
                ]
            )
    _write_plotly_html(path, data=traces, layout=layout)


def _write_embedding_image(
    path: Path,
    *,
    noise_ratio: float,
    train_points: torch.Tensor,
    test_points: torch.Tensor,
    methods: dict[str, dict[str, Any]],
    max_points: int,
    seed: int,
) -> None:
    """Write a static shared-scale 2D projection of held-out coverage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    names = list(methods)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = torch.randperm(len(test_points), generator=generator)[:max_points]
    projected = _projection(
        train_points,
        [test_points] + [methods[name]["centroids"] for name in names],
        dimensions=2,
    )
    projected_test = projected[0][selected]
    projected_centroids = projected[1:]
    color_ceiling = max(float(methods[name]["coverage"]["r99"]) for name in names)
    norm = Normalize(vmin=0.0, vmax=color_ceiling)

    all_x = torch.cat([projected_test[:, 0], *[value[:, 0] for value in projected_centroids]])
    all_y = torch.cat([projected_test[:, 1], *[value[:, 1] for value in projected_centroids]])
    x_margin = max(float(all_x.max() - all_x.min()) * 0.04, 1e-3)
    y_margin = max(float(all_y.max() - all_y.min()) * 0.04, 1e-3)
    x_limits = (float(all_x.min()) - x_margin, float(all_x.max()) + x_margin)
    y_limits = (float(all_y.min()) - y_margin, float(all_y.max()) + y_margin)

    figure, axes = plt.subplots(1, len(names), figsize=(12.5, 5.6), sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]
    point_layer = None
    for axis, name, centers in zip(axes, names, projected_centroids, strict=True):
        result = methods[name]
        coverage = result["coverage"]
        alignment = result["geometry"]["tangent_alignment"]
        distances = result["distances"][selected].clamp_max(color_ceiling)
        point_layer = axis.scatter(
            projected_test[:, 0],
            projected_test[:, 1],
            c=distances,
            s=5,
            cmap="viridis",
            norm=norm,
            alpha=0.58,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            s=54,
            marker="X",
            c="#d62728",
            edgecolors="white",
            linewidths=0.6,
            label="live centroid",
            zorder=3,
        )
        axis.set_title(
            f"{name}\n"
            f"mean={coverage['mean']:.3f}  r95={coverage['r95']:.3f}  r99={coverage['r99']:.3f}\n"
            f"tangent overlap={alignment['subspace_overlap']['mean']:.3f}"
        )
        axis.set_xlabel("training-PCA projection 1")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.legend(loc="lower right", frameon=True, fontsize=8)
    axes[0].set_ylabel("training-PCA projection 2")
    if point_layer is None:
        raise ValueError("at least one method is required")
    colorbar = figure.colorbar(point_layer, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label(
        f"nearest-live-centroid distance (clipped at shared r99={color_ceiling:.3f})"
    )
    figure.suptitle(
        f"Held-out sphere coverage in 2D — noise ratio={noise_ratio:g}",
        fontsize=14,
    )
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.1, top=0.83, wspace=0.12)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def render_existing_2d(
    output_dir: Path,
    *,
    max_points: int,
    seed: int,
    coverage_batch_size: int,
) -> Path:
    """Render 2D images from saved data and fitted models without retraining."""
    output_dir = output_dir.resolve()
    split = torch.load(output_dir / "split.pt", map_location="cpu", weights_only=True)
    for condition_dir in sorted(output_dir.glob("noise_ratio_*")):
        condition_result = json.loads((condition_dir / "metrics.json").read_text())
        dataset = torch.load(
            condition_dir / "dataset.pt", map_location="cpu", weights_only=True
        )
        kmeans = torch.load(
            condition_dir / "kmeans_pca.pt", map_location="cpu", weights_only=True
        )
        mfa_assignments = torch.load(
            condition_dir / "mfa_train_assignments.pt",
            map_location="cpu",
            weights_only=True,
        )
        mfa = load_mfa(condition_dir / "mfa_model.pt", map_location="cpu").eval()
        points = dataset["points"]
        train_points = points[split["train"]]
        test_points = points[split["test"]]
        kmeans_live = kmeans["train_cluster_sizes"] > 0
        mfa_live = mfa_assignments["train_cluster_sizes"] > 0
        _kmeans_summary, kmeans_distances = evaluate_heldout_distribution_coverage(
            test_points,
            kmeans["centroids"],
            live_components=kmeans_live,
            batch_size=coverage_batch_size,
        )
        _mfa_summary, mfa_distances = evaluate_heldout_distribution_coverage(
            test_points,
            mfa.mu.detach().cpu(),
            live_components=mfa_live,
            batch_size=coverage_batch_size,
        )
        figure_methods = {
            "KMeans+PCA": {
                **condition_result["methods"]["KMeans+PCA"],
                "centroids": kmeans["centroids"][kmeans_live],
                "distances": kmeans_distances,
            },
            "MFA": {
                **condition_result["methods"]["MFA"],
                "centroids": mfa.mu.detach().cpu()[mfa_live],
                "distances": mfa_distances,
            },
        }
        _write_embedding_image(
            condition_dir / "coverage_2d.png",
            noise_ratio=float(condition_result["noise_ratio"]),
            train_points=train_points,
            test_points=test_points,
            methods=figure_methods,
            max_points=max_points,
            seed=seed,
        )
    return output_dir


def _write_coverage_curves(path: Path, results: list[dict[str, Any]]) -> None:
    traces = []
    for condition in results:
        for method_name, method in condition["methods"].items():
            curve = method["coverage"]["coverage_curve"]
            traces.append(
                {
                    "type": "scatter",
                    "x": [item["radius"] for item in curve],
                    "y": [item["fraction"] for item in curve],
                    "mode": "lines+markers",
                    "name": f"{method_name}, noise={condition['noise_ratio']:g}",
                }
            )
    _write_plotly_html(
        path,
        data=traces,
        layout={
            "title": "Held-out distribution coverage",
            "xaxis": {"title": "radius ε"},
            "yaxis": {
                "title": "fraction of test points within ε of a live centroid",
                "range": [0.0, 1.01],
            },
            "template": "plotly_white",
        },
    )


def _write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Toy held-out coverage results",
        "",
        "| noise ratio | method | live / K | mean | r95 | r99 | maximum | alignment overlap | worst cosine |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in results:
        for method_name, method in condition["methods"].items():
            coverage = method["coverage"]
            alignment = method["geometry"]["tangent_alignment"]
            lines.append(
                f"| {condition['noise_ratio']:g} | {method_name} | "
                f"{coverage['components_live']} / {coverage['components_total']} | "
                f"{coverage['mean']:.6f} | {coverage['r95']:.6f} | "
                f"{coverage['r99']:.6f} | {coverage['maximum']:.6f} | "
                f"{alignment['subspace_overlap']['mean']:.6f} | "
                f"{alignment['worst_direction_cosine']['mean']:.6f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    thresholds = tuple(float(value) for value in args.coverage_thresholds)
    if thresholds != tuple(sorted(set(thresholds))):
        raise ValueError("coverage thresholds must be sorted and unique")

    base_config = ToyManifoldConfig(
        ambient_dim=args.ambient_dim,
        n_samples=args.samples,
        calibration_size=args.calibration_size,
        manifolds_per_type=1,
        manifold_types=("sphere",),
        offset_radius=0.0,
        noise_ratio=float(args.noise_ratios[0]),
        seed=args.seed,
    )
    base_dataset, _base_metadata = make_toy_manifold_dataset(base_config)
    split = stratified_three_way_split(
        base_dataset.tensors[1],
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    split_artifact = {
        **split,
        "seed": args.split_seed,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
        "fingerprint": split_fingerprint(split),
    }
    torch.save(split_artifact, output_dir / "split.pt")
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output_dir": str(output_dir),
                "split_counts": {name: int(len(split[name])) for name in split},
                "split_fingerprint": split_artifact["fingerprint"],
            },
            indent=2,
        )
    )

    all_results: list[dict[str, Any]] = []
    for noise_ratio in args.noise_ratios:
        condition_dir = output_dir / f"noise_ratio_{noise_ratio:g}"
        condition_dir.mkdir()
        config = ToyManifoldConfig(
            **{
                **asdict(base_config),
                "manifold_types": base_config.manifold_types,
                "noise_ratio": float(noise_ratio),
            }
        )
        dataset, metadata = make_toy_manifold_dataset(config)
        points, labels = dataset.tensors
        if not torch.equal(labels, base_dataset.tensors[1]):
            raise RuntimeError("paired noise conditions produced different row labels")
        torch.save({"points": points, "manifold_ids": labels}, condition_dir / "dataset.pt")
        torch.save(metadata, condition_dir / "manifold_metadata.pt")

        train_points = points[split["train"]]
        validation_points = points[split["validation"]]
        test_points = points[split["test"]]

        centroids, directions, kmeans_train_assignments, kmeans_sizes, inertia = (
            _fit_kmeans_pca(
                train_points,
                K=args.K,
                rank=args.rank,
                seed=args.seed,
                device=device,
                n_iter=args.kmeans_iterations,
                restarts=args.kmeans_restarts,
            )
        )
        kmeans_live = kmeans_sizes > 0
        torch.save(
            {
                "centroids": centroids,
                "principal_components": directions,
                "train_assignments": kmeans_train_assignments,
                "train_cluster_sizes": kmeans_sizes,
                "inertia": inertia,
                "split_fingerprint": split_artifact["fingerprint"],
            },
            condition_dir / "kmeans_pca.pt",
        )

        mfa, training = _fit_mfa(
            train_points,
            validation_points,
            centroids=centroids,
            directions=directions,
            cluster_sizes=kmeans_sizes,
            rank=args.rank,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
        )
        mfa_train_assignments, mfa_sizes = _hard_mfa_assignments(
            mfa,
            train_points,
            batch_size=args.coverage_batch_size,
            device=device,
        )
        mfa_live = mfa_sizes > 0
        save_mfa(
            mfa.cpu(),
            str(condition_dir / "mfa_model.pt"),
            extra={
                "training": training,
                "split_fingerprint": split_artifact["fingerprint"],
            },
        )
        torch.save(
            {
                "train_assignments": mfa_train_assignments,
                "train_cluster_sizes": mfa_sizes,
                "split_fingerprint": split_artifact["fingerprint"],
            },
            condition_dir / "mfa_train_assignments.pt",
        )

        kmeans_coverage, kmeans_distances = evaluate_heldout_distribution_coverage(
            test_points,
            centroids,
            live_components=kmeans_live,
            thresholds=thresholds,
            batch_size=args.coverage_batch_size,
        )
        mfa_coverage, mfa_distances = evaluate_heldout_distribution_coverage(
            test_points,
            mfa.mu.detach().cpu(),
            live_components=mfa_live,
            thresholds=thresholds,
            batch_size=args.coverage_batch_size,
        )
        kmeans_geometry = evaluate_toy_manifold_metrics(
            _KMeansPCAGeometry(centroids, directions),
            metadata,
            kmeans_live,
            rank_threshold=1.0,
        )
        mfa_geometry = evaluate_toy_manifold_metrics(
            mfa.cpu().eval(),
            metadata,
            mfa_live,
            rank_threshold=args.rank_threshold,
        )

        condition_result = {
            "noise_ratio": float(noise_ratio),
            "noise_std": float(metadata["noise_stds"][0]),
            "dataset": {
                "config": asdict(config),
                "split_counts": {name: int(len(split[name])) for name in split},
                "split_fingerprint": split_artifact["fingerprint"],
                "coverage_population": "independent_test_split",
                "alignment_population": "proximity_associated_components",
            },
            "methods": {
                "KMeans+PCA": {
                    "fit_population": "train",
                    "validation_role": "unused_fixed_configuration",
                    "test_role": "final_coverage_only",
                    "coverage": kmeans_coverage,
                    "geometry": _alignment_subset(kmeans_geometry),
                    "kmeans_inertia_train": inertia,
                },
                "MFA": {
                    "fit_population": "train",
                    "validation_role": "early_stopping_and_best_checkpoint_selection",
                    "test_role": "final_coverage_only",
                    "coverage": mfa_coverage,
                    "geometry": _alignment_subset(mfa_geometry),
                    "training": training,
                },
            },
        }
        (condition_dir / "metrics.json").write_text(
            json.dumps(condition_result, indent=2)
        )

        figure_methods = {
            "KMeans+PCA": {
                **condition_result["methods"]["KMeans+PCA"],
                "centroids": centroids[kmeans_live],
                "distances": kmeans_distances,
            },
            "MFA": {
                **condition_result["methods"]["MFA"],
                "centroids": mfa.mu.detach().cpu()[mfa_live],
                "distances": mfa_distances,
            },
        }
        _write_embedding_figure(
            condition_dir / f"coverage_{args.plot_dimensions}d.html",
            noise_ratio=float(noise_ratio),
            train_points=train_points,
            test_points=test_points,
            methods=figure_methods,
            dimensions=args.plot_dimensions,
            max_points=args.plot_points,
            seed=args.seed,
        )
        _write_embedding_image(
            condition_dir / "coverage_2d.png",
            noise_ratio=float(noise_ratio),
            train_points=train_points,
            test_points=test_points,
            methods=figure_methods,
            max_points=args.plot_points,
            seed=args.seed,
        )
        all_results.append(condition_result)

    (output_dir / "metrics.json").write_text(json.dumps(all_results, indent=2))
    _write_summary(output_dir / "README.md", all_results)
    _write_coverage_curves(output_dir / "coverage_curves.html", all_results)
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-ratios", type=float, nargs="+", default=(1000.0, 10.0))
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--calibration-size", type=int, default=20_000)
    parser.add_argument("--ambient-dim", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--rank-threshold", type=float, default=1.0)
    parser.add_argument("--kmeans-iterations", type=int, default=100)
    parser.add_argument("--kmeans-restarts", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--coverage-batch-size", type=int, default=8192)
    parser.add_argument(
        "--coverage-thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--plot-dimensions", type=int, choices=(2, 3), default=3)
    parser.add_argument("--plot-points", type=int, default=5000)
    parser.add_argument(
        "--render-existing-2d",
        action="store_true",
        help="render coverage_2d.png from an existing completed output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.render_existing_2d:
        output_dir = render_existing_2d(
            args.output_dir,
            max_points=args.plot_points,
            seed=args.seed,
            coverage_batch_size=args.coverage_batch_size,
        )
    else:
        output_dir = run(args)
    print(f"Coverage experiment written to {output_dir}")


if __name__ == "__main__":
    main()
