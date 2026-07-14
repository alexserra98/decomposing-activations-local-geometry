#!/usr/bin/env python3
"""Validate per-cluster PCA convergence as the activation sample cap grows."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import torch

from dalg.analysis.cluster_intrinsic_dim import (
    _assignment_subset_spec,
    _collect_sampled_shard_activations,
    _load_assignments,
    _sample_positions_to_shard_requests,
    _validate_assignment_source,
    intrinsic_dim_pca,
    pop_and_save_top_pcs,
)
from dalg.data.shard_activations import load_meta_index
from dalg.data.subset_spec import resolve_spec_positions


PARTITION_NAMES = ("kmeans", "mfa_responsibility")


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "src/dalg").exists():
            return path
    raise RuntimeError(f"Could not find repository root from {start}")


def _path_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], q: float) -> float:
    return float(torch.tensor(values, dtype=torch.float64).quantile(q))


def select_clusters(
    kmeans_sizes: torch.Tensor,
    mfa_sizes: torch.Tensor,
    *,
    num_clusters: int,
    min_size: int,
    seed: int,
) -> torch.Tensor:
    """Select one cluster from each stratum of the joint population ranking."""
    if kmeans_sizes.shape != mfa_sizes.shape:
        raise ValueError(
            f"Partition size shapes differ: {tuple(kmeans_sizes.shape)} vs "
            f"{tuple(mfa_sizes.shape)}"
        )
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")

    joint_sizes = torch.minimum(kmeans_sizes.long(), mfa_sizes.long())
    eligible = torch.nonzero(joint_sizes >= min_size).flatten()
    if eligible.numel() < num_clusters:
        raise ValueError(
            f"Only {eligible.numel()} clusters have at least {min_size:,} members "
            f"in both partitions; cannot select {num_clusters}."
        )

    eligible = eligible[torch.argsort(joint_sizes[eligible], stable=True)]
    rng = torch.Generator().manual_seed(seed)
    chosen: list[int] = []
    for stratum in range(num_clusters):
        lo = stratum * eligible.numel() // num_clusters
        hi = (stratum + 1) * eligible.numel() // num_clusters
        offset = int(torch.randint(hi - lo, (1,), generator=rng).item())
        chosen.append(int(eligible[lo + offset]))
    return torch.tensor(sorted(chosen), dtype=torch.long)


def _choose_nested_positions(
    assignments: torch.Tensor,
    selected_clusters: torch.Tensor,
    *,
    max_samples: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    """Return a random ordered sample per cluster; prefixes form nested samples."""
    K = int(torch.max(assignments).item()) + 1
    lookup = torch.zeros(K, dtype=torch.bool)
    lookup[selected_clusters] = True
    selected_positions = torch.nonzero(lookup[assignments]).flatten()
    selected_labels = assignments[selected_positions]

    rng = torch.Generator().manual_seed(seed)
    ordered: dict[int, torch.Tensor] = {}
    for cluster in selected_clusters.tolist():
        positions = selected_positions[selected_labels == cluster]
        if positions.numel() < max_samples:
            raise ValueError(
                f"Cluster {cluster} has only {positions.numel():,} assignments, "
                f"below the requested maximum sample size {max_samples:,}."
            )
        priority = torch.randperm(positions.numel(), generator=rng)[:max_samples]
        ordered[cluster] = positions[priority]
    return ordered


def _restore_random_order(
    buffers: list[torch.Tensor | None],
    ordered_positions: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Undo stream-order collection so prefixes preserve random sampling priority."""
    ordered_buffers: dict[int, torch.Tensor] = {}
    for cluster, positions in ordered_positions.items():
        buffer = buffers[cluster]
        if buffer is None or buffer.shape[0] != positions.numel():
            observed = None if buffer is None else buffer.shape[0]
            raise RuntimeError(
                f"Collected {observed} rows for cluster {cluster}; "
                f"expected {positions.numel()}."
            )
        stream_order = torch.argsort(positions)
        inverse = torch.empty_like(stream_order)
        inverse[stream_order] = torch.arange(stream_order.numel())
        ordered_buffers[cluster] = buffer[inverse].contiguous()
    return ordered_buffers


def summarize_spectrum(
    variances: torch.Tensor,
    *,
    sample_size: int,
    ambient_dim: int,
) -> tuple[float, float]:
    """Return participation ratio and its finite-sample isotropy normalization."""
    values = variances.double()
    participation_ratio = float(values.sum().square() / values.square().sum())
    isotropic_null = ambient_dim * (sample_size - 1) / (ambient_dim + sample_size)
    if isotropic_null <= 1.0:
        raise ValueError(
            "The finite-sample isotropy normalization requires a sample size "
            "large enough for the isotropic-null participation ratio to exceed one."
        )
    isotropy = (participation_ratio - 1.0) / (isotropic_null - 1.0)
    return participation_ratio, isotropy


def compute_cluster_metrics(
    activations: torch.Tensor,
    *,
    threshold: float,
    pca_device: str,
    top_pcs: int,
) -> dict[str, Any]:
    intrinsic_dim, variances, components = intrinsic_dim_pca(
        activations,
        threshold=threshold,
        device=pca_device,
        top_pcs=top_pcs,
    )
    if components is None:
        raise RuntimeError("intrinsic_dim_pca did not return requested components")
    participation_ratio, isotropy = summarize_spectrum(
        variances,
        sample_size=activations.shape[0],
        ambient_dim=activations.shape[1],
    )
    return {
        "intrinsic_dim": intrinsic_dim,
        "variances": variances,
        "components": components,
        "participation_ratio": participation_ratio,
        "sample_corrected_isotropy": isotropy,
    }


def compare_pc_bases(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Compare two row-wise orthonormal PC bases using principal angles."""
    q = min(left.shape[0], right.shape[0])
    cosines = torch.linalg.svdvals(left[:q] @ right[:q].T).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cosines))
    return {
        "n_pcs_compared": q,
        "pc_mean_cos2": float(cosines.square().mean()),
        "pc_median_angle_deg": float(angles.median()),
        "pc_max_angle_deg": float(angles.max()),
    }


def _save_cap_results(
    out_dir: Path,
    *,
    K: int,
    cluster_sizes: torch.Tensor,
    selected_clusters: torch.Tensor,
    cap: int,
    D: int,
    threshold: float,
    assignments_path: Path,
    assignment_metadata: dict[str, Any],
    metrics: dict[int, dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = torch.zeros(K, dtype=torch.long)
    sample_sizes = torch.zeros(K, dtype=torch.long)
    participation_ratios = torch.full((K,), torch.nan)
    isotropies = torch.full((K,), torch.nan)
    variances: list[torch.Tensor] = [torch.zeros(0) for _ in range(K)]
    components: list[torch.Tensor | None] = [None for _ in range(K)]

    for cluster, values in metrics.items():
        dims[cluster] = values["intrinsic_dim"]
        sample_sizes[cluster] = cap
        participation_ratios[cluster] = values["participation_ratio"]
        isotropies[cluster] = values["sample_corrected_isotropy"]
        variances[cluster] = values["variances"]
        components[cluster] = values["components"]

    results: dict[str, Any] = {
        "intrinsic_dims": dims,
        "cluster_variances": variances,
        "cluster_sizes": cluster_sizes,
        "sample_sizes": sample_sizes,
        "participation_ratios": participation_ratios,
        "sample_corrected_isotropy": isotropies,
        "variance_threshold": threshold,
        "max_samples": cap,
        "assignments_path": str(assignments_path),
        "K": K,
        "D": D,
        "model_kind": "assignments",
        "model_path": None,
        "assignment_metadata": assignment_metadata,
        "selected_clusters": selected_clusters,
        "cluster_top_pcs": components,
        "top_pcs": max(x.shape[0] for x in components if x is not None),
    }
    pcs_path = pop_and_save_top_pcs(results, out_dir)
    torch.save(results, out_dir / "intrinsic_dims.pt")
    print(f"Saved {out_dir / 'intrinsic_dims.pt'}")
    print(f"Saved {pcs_path}")


def _partition_comparisons(
    partition: str,
    all_metrics: dict[int, dict[int, dict[str, Any]]],
    caps: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_cap = max(caps)
    comparisons: list[tuple[str, int, int]] = [
        ("largest_cap", cap, reference_cap) for cap in caps
    ]
    comparisons.extend(
        ("previous_cap", caps[index], caps[index + 1])
        for index in range(len(caps) - 1)
    )

    for comparison, cap, ref_cap in comparisons:
        for cluster, values in all_metrics[cap].items():
            reference = all_metrics[ref_cap][cluster]
            pc = compare_pc_bases(values["components"], reference["components"])
            rows.append(
                {
                    "partition": partition,
                    "comparison": comparison,
                    "cluster": cluster,
                    "sample_cap": cap,
                    "reference_cap": ref_cap,
                    "intrinsic_dim": values["intrinsic_dim"],
                    "reference_intrinsic_dim": reference["intrinsic_dim"],
                    "intrinsic_dim_delta": (
                        values["intrinsic_dim"] - reference["intrinsic_dim"]
                    ),
                    "intrinsic_dim_abs_delta": abs(
                        values["intrinsic_dim"] - reference["intrinsic_dim"]
                    ),
                    "participation_ratio": values["participation_ratio"],
                    "reference_participation_ratio": reference["participation_ratio"],
                    "participation_ratio_relative_error": (
                        values["participation_ratio"]
                        / reference["participation_ratio"]
                        - 1.0
                    ),
                    "participation_ratio_abs_relative_error": abs(
                        values["participation_ratio"]
                        / reference["participation_ratio"]
                        - 1.0
                    ),
                    "sample_corrected_isotropy": values[
                        "sample_corrected_isotropy"
                    ],
                    "reference_sample_corrected_isotropy": reference[
                        "sample_corrected_isotropy"
                    ],
                    "isotropy_delta": (
                        values["sample_corrected_isotropy"]
                        - reference["sample_corrected_isotropy"]
                    ),
                    "isotropy_abs_delta": abs(
                        values["sample_corrected_isotropy"]
                        - reference["sample_corrected_isotropy"]
                    ),
                    **pc,
                }
            )
    return rows


def _summarize_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["partition"],
            row["comparison"],
            row["sample_cap"],
            row["reference_cap"],
        )
        groups.setdefault(key, []).append(row)

    metrics = (
        "intrinsic_dim_abs_delta",
        "participation_ratio_relative_error",
        "participation_ratio_abs_relative_error",
        "isotropy_delta",
        "isotropy_abs_delta",
        "pc_mean_cos2",
        "pc_median_angle_deg",
        "pc_max_angle_deg",
    )
    summaries: list[dict[str, Any]] = []
    for key, group in groups.items():
        summary: dict[str, Any] = {
            "partition": key[0],
            "comparison": key[1],
            "sample_cap": key[2],
            "reference_cap": key[3],
            "n_clusters": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_p10"] = _quantile(values, 0.1)
            summary[f"{metric}_median"] = _quantile(values, 0.5)
            summary[f"{metric}_p90"] = _quantile(values, 0.9)
        summaries.append(summary)
    return summaries


def _centroid_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("centroids", "mu", "means"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Could not find centroid tensor in {path}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--num-clusters", type=int, default=64)
    parser.add_argument("--cluster-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[2000, 5000, 10000, 20000]
    )
    parser.add_argument("--top-pcs", type=int, default=100)
    parser.add_argument("--variance-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-device", default="cuda")
    parser.add_argument(
        "--store-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--kmeans-assignments", type=Path, default=None)
    parser.add_argument("--mfa-assignments", type=Path, default=None)
    parser.add_argument("--kmeans-centroids", type=Path, default=None)
    parser.add_argument("--mfa-init-centroids", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace artifacts in a non-empty output directory in place.",
    )
    parser.add_argument("--skip-centroid-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo = find_repo_root()
    models = repo / "dalg-cache/pile_gemma2b_models"
    mfa_run = models / f"layer{args.layer:02d}_{args.K}_{args.rank}_component_sharded_mfa"
    centroids_dir = models / f"centroids/k{args.K}_L{args.layer:02d}"

    shard_dir = args.shard_dir or repo / "dalg-cache/pile_gemma2b_activations"
    assignment_paths = {
        "kmeans": args.kmeans_assignments
        or centroids_dir / "kmeans_centroid_assignments.pt",
        "mfa_responsibility": args.mfa_assignments
        or mfa_run / "mfa_model_assignments.pt",
    }
    kmeans_centroids = args.kmeans_centroids or centroids_dir / "centroids.pt"
    mfa_init_centroids = args.mfa_init_centroids or mfa_run / "centroids.pt"
    run_name = (
        f"layer{args.layer:02d}_K{args.K}_q{args.rank}_"
        f"c{len(args.cluster_ids) if args.cluster_ids else args.num_clusters}_seed{args.seed}"
    )
    out_dir = args.output_dir or models / "pca_size_validation" / run_name

    caps = sorted(set(args.sample_sizes))
    if not caps or caps[0] < 2:
        raise SystemExit("All --sample-sizes must be at least 2.")
    if args.top_pcs <= 0 or args.top_pcs >= caps[0]:
        raise SystemExit("--top-pcs must be positive and smaller than the lowest sample cap.")
    if args.pca_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for PCA but torch.cuda.is_available() is false.")
    for path in [shard_dir / "config.json", *assignment_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    output_exists = out_dir.exists() and any(out_dir.iterdir())
    if output_exists and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {out_dir}. Choose a fresh --output-dir "
            "or pass --overwrite."
        )
    if output_exists:
        print(f"Overwriting existing validation artifacts in {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_centroid_check:
        if not kmeans_centroids.exists() or not mfa_init_centroids.exists():
            raise FileNotFoundError(
                "Centroid identity check requires both centroid files; pass "
                "--skip-centroid-check only when same-ID comparisons are not needed."
            )
        initial = _centroid_tensor(kmeans_centroids)
        mfa_initial = _centroid_tensor(mfa_init_centroids)
        if initial.shape != mfa_initial.shape or not torch.equal(initial, mfa_initial):
            raise ValueError(
                "KMeans centroids and MFA initialization centroids are not bit-identical; "
                "cluster IDs cannot be compared directly."
            )
        print("Verified bit-identical KMeans and MFA initialization centroids.")

    bundles: dict[str, dict[str, Any]] = {}
    for partition, path in assignment_paths.items():
        assignments, sizes, _peakedness, K, metadata = _load_assignments(path)
        if K != args.K:
            raise ValueError(f"{partition} assignments have K={K}, expected {args.K}")
        bundles[partition] = {
            "assignments": assignments,
            "sizes": sizes,
            "K": K,
            "metadata": metadata,
        }
    if bundles["kmeans"]["assignments"].numel() != bundles[
        "mfa_responsibility"
    ]["assignments"].numel():
        raise ValueError("The two assignment streams have different lengths.")

    max_cap = max(caps)
    if args.cluster_ids:
        selected_clusters = torch.tensor(sorted(set(args.cluster_ids)), dtype=torch.long)
        if selected_clusters.min() < 0 or selected_clusters.max() >= args.K:
            raise ValueError(f"Explicit cluster IDs must lie in [0, {args.K}).")
        joint_sizes = torch.minimum(
            bundles["kmeans"]["sizes"], bundles["mfa_responsibility"]["sizes"]
        )
        underfilled = selected_clusters[joint_sizes[selected_clusters] < max_cap]
        if underfilled.numel():
            raise ValueError(
                f"Explicit clusters below the {max_cap:,} joint population requirement: "
                f"{underfilled.tolist()}"
            )
    else:
        selected_clusters = select_clusters(
            bundles["kmeans"]["sizes"],
            bundles["mfa_responsibility"]["sizes"],
            num_clusters=args.num_clusters,
            min_size=max_cap,
            seed=args.seed,
        )

    selection_rows = [
        {
            "cluster": cluster,
            "kmeans_size": int(bundles["kmeans"]["sizes"][cluster]),
            "mfa_responsibility_size": int(
                bundles["mfa_responsibility"]["sizes"][cluster]
            ),
            "joint_min_size": int(
                min(
                    bundles["kmeans"]["sizes"][cluster],
                    bundles["mfa_responsibility"]["sizes"][cluster],
                )
            ),
        }
        for cluster in selected_clusters.tolist()
    ]
    _write_csv(out_dir / "selected_clusters.csv", selection_rows)
    print(f"Selected clusters: {selected_clusters.tolist()}")

    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    window = int(extract_cfg["window"])
    drop_prefix = int(extract_cfg.get("drop_prefix", 32))
    subset_specs = {
        _assignment_subset_spec(bundle["metadata"]) for bundle in bundles.values()
    }
    if len(subset_specs) != 1:
        raise ValueError(f"Assignment subset specs differ: {subset_specs}")
    subset_spec = subset_specs.pop()
    for bundle in bundles.values():
        _validate_assignment_source(
            bundle["metadata"], layer=args.layer, drop_prefix=drop_prefix
        )

    meta_index = load_meta_index(shard_dir, layer=args.layer)
    if subset_spec:
        keep = resolve_spec_positions(
            meta_index,
            subset_spec,
            window=window,
            drop_prefix=drop_prefix,
        )
        meta_index = [meta_index[index] for index in keep]

    manifest = {
        "status": "running",
        "layer": args.layer,
        "K": args.K,
        "rank": args.rank,
        "sample_sizes": caps,
        "top_pcs": args.top_pcs,
        "variance_threshold": args.variance_threshold,
        "seed": args.seed,
        "pca_device": args.pca_device,
        "store_dtype": args.store_dtype,
        "selected_clusters": selected_clusters.tolist(),
        "subset_spec": subset_spec,
        "shard_dir": str(shard_dir.resolve()),
        "assignments": {
            name: _path_fingerprint(path) for name, path in assignment_paths.items()
        },
        "output_dir": str(out_dir.resolve()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    dtype = getattr(torch, args.store_dtype)
    all_partition_metrics: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    metric_rows: list[dict[str, Any]] = []

    for partition_index, partition in enumerate(PARTITION_NAMES):
        bundle = bundles[partition]
        ordered_positions = _choose_nested_positions(
            bundle["assignments"],
            selected_clusters,
            max_samples=max_cap,
            seed=args.seed + 10_000 * partition_index,
        )
        combined_positions = torch.cat(list(ordered_positions.values()))
        combined_clusters = torch.cat(
            [
                torch.full((positions.numel(),), cluster, dtype=torch.long)
                for cluster, positions in ordered_positions.items()
            ]
        )
        stream_order = torch.argsort(combined_positions)
        combined_positions = combined_positions[stream_order]
        combined_clusters = combined_clusters[stream_order]

        requests = _sample_positions_to_shard_requests(
            combined_positions,
            combined_clusters,
            meta_index,
            window=window,
            drop_prefix=drop_prefix,
            num_expected_items=int(bundle["assignments"].numel()),
        )
        print(
            f"{partition}: loading {combined_positions.numel():,} activations "
            f"from {len(requests):,} shards"
        )
        buffers = _collect_sampled_shard_activations(
            shard_dir,
            args.layer,
            requests,
            K=args.K,
            store_dtype=dtype,
        )
        random_order_buffers = _restore_random_order(buffers, ordered_positions)
        del buffers, requests, combined_positions, combined_clusters, stream_order
        gc.collect()

        cap_metrics: dict[int, dict[int, dict[str, Any]]] = {}
        for cap in caps:
            print(f"{partition}: computing PCA at cap={cap:,}")
            per_cluster: dict[int, dict[str, Any]] = {}
            for index, cluster in enumerate(selected_clusters.tolist(), start=1):
                print(
                    f"  [{index:02d}/{selected_clusters.numel():02d}] cluster {cluster}",
                    flush=True,
                )
                values = compute_cluster_metrics(
                    random_order_buffers[cluster][:cap],
                    threshold=args.variance_threshold,
                    pca_device=args.pca_device,
                    top_pcs=args.top_pcs,
                )
                per_cluster[cluster] = values
                metric_rows.append(
                    {
                        "partition": partition,
                        "cluster": cluster,
                        "sample_cap": cap,
                        "intrinsic_dim": values["intrinsic_dim"],
                        "participation_ratio": values["participation_ratio"],
                        "sample_corrected_isotropy": values[
                            "sample_corrected_isotropy"
                        ],
                    }
                )
            cap_metrics[cap] = per_cluster
            _save_cap_results(
                out_dir / partition / f"n{cap}",
                K=args.K,
                cluster_sizes=bundle["sizes"],
                selected_clusters=selected_clusters,
                cap=cap,
                D=random_order_buffers[selected_clusters[0].item()].shape[1],
                threshold=args.variance_threshold,
                assignments_path=assignment_paths[partition],
                assignment_metadata=bundle["metadata"],
                metrics=per_cluster,
            )
        all_partition_metrics[partition] = cap_metrics
        del random_order_buffers
        gc.collect()
        if args.pca_device.startswith("cuda"):
            torch.cuda.empty_cache()

    _write_csv(out_dir / "per_cluster_metrics.csv", metric_rows)
    comparison_rows: list[dict[str, Any]] = []
    for partition in PARTITION_NAMES:
        comparison_rows.extend(
            _partition_comparisons(
                partition, all_partition_metrics[partition], caps
            )
        )
    _write_csv(out_dir / "convergence_comparisons.csv", comparison_rows)
    summary_rows = _summarize_comparisons(comparison_rows)
    _write_csv(out_dir / "convergence_summary.csv", summary_rows)

    manifest["status"] = "complete"
    manifest["outputs"] = {
        "selected_clusters": "selected_clusters.csv",
        "per_cluster_metrics": "per_cluster_metrics.csv",
        "comparisons": "convergence_comparisons.csv",
        "summary": "convergence_summary.csv",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PCA sample-size validation complete: {out_dir}")


if __name__ == "__main__":
    main()
