"""
CLI for cluster-level metrics on a trained MFA.

Subcommands:
    overlap        Pairwise overlap metrics between MFA components.
    intrinsic-dim  PCA-based intrinsic dimensionality per cluster.
    assignments    Hard cluster assignments. With an MFA, streams activations
                   through responsibilities; with medoids, assigns by nearest
                   Euclidean centroid.
    description-fit
                   Description-vs-context metrics for labeled clusters.
    description-semantics
                   Embedding similarity between cluster descriptions.
    gaussian-group-semantics
                   Label coherence inside groups of nearby MFA Gaussians.

The model-based subcommands expect a trained MFA run directory (``--data-dir``)
containing either ``mfa_model.pt`` or, for component-sharded runs,
``mfa_model_shards.json``. ``--data-dir`` may also be a direct path to a
``.pt`` model file. The description subcommands read ``cluster_labels.json``
from the interpretation pipeline.

Example::

    dalg-run-metrics overlap \
        --data-dir /path/to/layer05_mfa --device cuda --batch-pairs 512

    dalg-run-metrics intrinsic-dim \
        --data-dir /path/to/layer05_mfa --shard-dir /path/to/activations \
        --layer 5 --device cuda --max-samples-per-cluster 2000
"""
import os
import argparse
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


def _resolve_model_path(data_dir: str) -> tuple[Path, Path]:
    """Return ``(model_path, run_dir)`` from a CLI ``--data-dir`` argument.

    Accepts either a directory containing ``mfa_model.pt`` /
    ``mfa_model_shards.json`` or a direct path to a model ``.pt`` file.
    """
    if os.path.isfile(data_dir):
        return Path(data_dir), Path(os.path.dirname(data_dir))
    run_dir = Path(data_dir)
    return run_dir / "mfa_model.pt", run_dir


def cmd_overlap(args) -> None:
    """Compute pairwise overlap metrics between MFA components."""
    from dalg.analysis.cluster_overlap import compute_overlap

    model_path, run_dir = _resolve_model_path(args.data_dir)
    out_dir = Path(args.out_dir or run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = compute_overlap(model_path, device=args.device, batch_pairs=args.batch_pairs)

    save_path = out_dir / "overlap.pt"
    torch.save(results, save_path)
    print(f"Overlap saved to {save_path}")

#TODO remove monolithic option
def cmd_intrinsic_dim(args) -> None:
    """Compute intrinsic dimensionality per cluster.

    Two input layouts are supported:
        (A) monolithic ``--act-dir`` with ``activations.pt`` / ``tokens.pt``;
        (B) sharded ``--shard-dir`` from ``dalg-run-extraction``, with
            ``--layer`` selecting which layer's shards to read.
    """
    from dalg.analysis.cluster_intrinsic_dim import (
        compute_intrinsic_dims, compute_intrinsic_dims_from_shards,
    )
    from dalg.data.subset_spec import split_shard_dir_spec

    model_path, run_dir = _resolve_model_path(args.data_dir)
    out_dir = Path(args.out_dir or run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shard_dir is not None:
        shard_dir, subset_spec = split_shard_dir_spec(args.shard_dir)
        results = compute_intrinsic_dims_from_shards(
            model_path, shard_dir,
            layer=args.layer,
            subset_spec=subset_spec,
            assignments_path=args.assignments_path,
            device=args.device,
            variance_threshold=args.variance_threshold,
            min_population=args.min_population,
            max_samples=args.max_samples_per_cluster,
            pca_device=args.pca_device,
            pca_workers=args.pca_workers,
            seed=(args.seed or 0),
        )
    else:
        act_dir = Path(args.act_dir or run_dir)
        results = compute_intrinsic_dims(
            model_path, act_dir / "activations.pt", act_dir / "tokens.pt",
            assignments_path=args.assignments_path,
            device=args.device,
            batch_size=args.batch_size,
            variance_threshold=args.variance_threshold,
            min_population=args.min_population,
            max_samples=args.max_samples_per_cluster,
            pca_device=args.pca_device,
            pca_workers=args.pca_workers,
            seed=(args.seed or 0),
        )

    save_path = out_dir / "intrinsic_dims.pt"
    torch.save(results, save_path)
    print(f"Intrinsic dims saved to {save_path}")


def cmd_assignments(args) -> None:
    """Compute hard cluster assignments.

    Streams activations from ``--shard-dir`` (at ``--layer``). By default it
    uses MFA responsibilities from ``--data-dir``. If ``--medoids-path`` is
    provided, it assigns by nearest Euclidean medoid instead.

    The default save path is ``<run_dir>/<model_stem>_assignments.pt`` so
    that ``intrinsic-dim`` can pick it up via its ``--assignments-path``
    default of ``<data-dir>/mfa_model_assignments.pt``. For medoids, the
    default save path is next to the medoid file.
    """
    import json
    from torch.utils.data import DataLoader

    from dalg.analysis.cluster_assignments import compute_assignments
    from dalg.analysis.nearest_centroid_assignments import (
        _load_centroids,
        _resolve_device,
        compute_nearest_centroid_assignments,
    )
    from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
    from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec

    device = _resolve_device(args.device)

    shard_dir, subset_spec = split_shard_dir_spec(args.shard_dir)
    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    window = int(extract_cfg["window"])
    drop_prefix = args.drop_prefix
    if drop_prefix is None:
        drop_prefix = int(extract_cfg.get("drop_prefix", 32))

    meta_index = load_meta_index(shard_dir, layer=args.layer)
    positions = resolve_spec_positions(
        meta_index, subset_spec, window=window, drop_prefix=drop_prefix
    )
    print(f"shard_dir={shard_dir}  layer={args.layer}  rows={len(positions):,}"
          + (f"  spec={subset_spec!r}" if subset_spec else ""))

    ds = ActivationBatchDataset(
        shard_dir,
        layer=args.layer,
        row_subset=positions,
        drop_prefix=drop_prefix,
        batch_size=args.batch_size,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=(args.seed or 0),
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    if args.medoids_path is not None:
        medoids_path = Path(args.medoids_path)
        centroids = _load_centroids(medoids_path)
        sizes, assignments, min_distances = compute_nearest_centroid_assignments(
            centroids,
            loader,
            device=device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        payload = {
            "cluster_sizes": sizes,
            "assignments": assignments,
            "min_distances": min_distances,
            "K": int(sizes.numel()),
            "centroids_path": str(medoids_path),
            "subset_spec": subset_spec,
            "source": {
                "shard_dir": str(shard_dir),
                "layer": int(args.layer),
                "drop_prefix": int(drop_prefix),
                "num_items": int(assignments.numel()),
            },
        }
        default_dir = medoids_path.parent
        default_stem = medoids_path.stem
        default_suffix = "_nearest_centroid_assignments.pt" if args.max_batches is None \
            else f"_nearest_centroid_assignments_first{args.max_batches}_batches.pt"
    else:
        model_path, run_dir = _resolve_model_path(args.data_dir)
        sizes, assignments, max_responsibilities, peakedness = compute_assignments(
            model_path,
            loader,
            device=device,
            max_batches=args.max_batches,
            use_inference_cache=args.use_inference_cache,
        )
        payload = {
            "cluster_sizes": sizes,
            "assignments": assignments,
            "max_responsibilities": max_responsibilities,
            "peakedness": peakedness,
            "K": int(sizes.numel()),
            "subset_spec": subset_spec,
        }
        default_dir = run_dir
        default_stem = model_path.stem
        default_suffix = "_assignments.pt" if args.max_batches is None \
            else f"_assignments_first{args.max_batches}_batches.pt"

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else default_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{default_stem}{default_suffix}"

    torch.save(payload, save_path)
    print(f"Assignments saved to {save_path}")


def _description_out_dir(args) -> Path:
    labels_path = Path(args.labels_path)
    out_dir = Path(args.out_dir) if args.out_dir is not None else labels_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def cmd_description_fit(args) -> None:
    """Compute description-fit metrics for labeled cluster interpretations."""
    import json

    from dalg.analysis.cluster_labeling import ORFEO_MODEL
    from dalg.analysis.description_metrics import (
        compute_detection_scores,
        compute_token_embedding_scores,
    )

    out_dir = _description_out_dir(args)
    detection = None
    token_embedding = None

    if not args.skip_detection:
        detection = compute_detection_scores(
            args.labels_path,
            model=(args.judge_model or ORFEO_MODEL),
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            positive_examples=args.positive_examples,
            negative_examples=args.negative_examples,
            cluster_ids=args.clusters,
            max_clusters=args.max_clusters,
            seed=(args.seed or 0),
            max_workers=args.judge_workers,
            show_progress=not args.quiet,
        )

    if not args.skip_token_embedding:
        token_embedding = compute_token_embedding_scores(
            args.labels_path,
            model_name=args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
            target_batch_size=args.target_batch_size,
            positive_examples=args.positive_examples,
            negative_examples=args.negative_examples,
            cluster_ids=args.clusters,
            max_clusters=args.max_clusters,
            seed=(args.seed or 0),
        )

    output = {
        "metadata": {
            "labels_path": str(args.labels_path),
            "positive_examples": int(args.positive_examples),
            "negative_examples": int(args.negative_examples),
            "seed": int(args.seed or 0),
        },
        "detection": detection,
        "token_embedding": token_embedding,
    }
    save_path = out_dir / "description_fit.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Description fit metrics saved to {save_path}")


def cmd_description_semantics(args) -> None:
    """Compute semantic similarity between cluster descriptions."""
    import json

    from dalg.analysis.description_metrics import (
        compute_description_semantics,
        load_labeled_clusters,
    )

    loaded = load_labeled_clusters(args.labels_path)
    all_cluster_ids = sorted(loaded["clusters"])
    selected = [int(k) for k in args.clusters] if args.clusters is not None else all_cluster_ids
    if args.max_clusters is not None:
        selected = selected[: int(args.max_clusters)]

    if args.full_matrix == "always":
        save_full_matrix = True
    elif args.full_matrix == "never":
        save_full_matrix = False
    else:
        save_full_matrix = len(selected) <= int(args.max_full_matrix_clusters)

    dtype = torch.float16 if args.full_matrix_dtype == "float16" else torch.float32
    tensor_output, json_output = compute_description_semantics(
        args.labels_path,
        model_name=args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        similarity_batch_size=args.similarity_batch_size,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        min_group_size=args.min_group_size,
        save_full_matrix=save_full_matrix,
        full_matrix_dtype=dtype,
        cluster_ids=args.clusters,
        max_clusters=args.max_clusters,
    )

    out_dir = _description_out_dir(args)
    tensor_path = out_dir / "description_semantics.pt"
    groups_path = out_dir / "description_semantic_groups.json"
    torch.save(tensor_output, tensor_path)
    groups_path.write_text(json.dumps(json_output, ensure_ascii=False, indent=2))
    if save_full_matrix:
        print(f"Saved full similarity matrix for {len(selected):,} descriptions.")
    else:
        print(
            "Skipped full similarity matrix; use --full-matrix always "
            "if you want a dense heatmap artifact."
        )
    print(f"Description semantics saved to {tensor_path} and {groups_path}")


def cmd_gaussian_group_semantics(args) -> None:
    """Cluster Gaussians by overlap distance and score label coherence."""
    import csv
    import json

    from dalg.analysis.description_metrics import compute_gaussian_group_label_coherence

    out_dir = _description_out_dir(args)
    output = compute_gaussian_group_label_coherence(
        args.labels_path,
        args.overlap_path,
        model_name=args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        distance_key=args.distance_key,
        distance_threshold=args.distance_threshold,
        linkage=args.linkage,
        top_groups=args.top_groups,
    )

    json_path = out_dir / "gaussian_group_label_coherence.json"
    csv_path = out_dir / "gaussian_group_members.csv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    rows = output["member_rows"]
    fieldnames = [
        "group_id",
        "group_size",
        "component_id",
        "label",
        "description",
        "mean_label_cosine",
        "median_label_cosine",
        "mean_distance",
        "median_distance",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Gaussian-group label coherence saved to {json_path} and {csv_path}")


def validate_args(args) -> None:
    if args.command in {"overlap", "intrinsic-dim"}:
        if args.data_dir is None:
            raise SystemExit(f"{args.command}: --data-dir is required")
    if args.command == "intrinsic-dim":
        if args.shard_dir is not None and args.act_dir is not None:
            raise SystemExit("intrinsic-dim: --shard-dir and --act-dir are mutually exclusive")
        if args.shard_dir is not None and args.layer is None:
            raise SystemExit("intrinsic-dim: --layer is required with --shard-dir")
    if args.command == "assignments":
        if args.data_dir is None and args.medoids_path is None:
            raise SystemExit("assignments: pass either --data-dir for MFA or --medoids-path for nearest-medoid assignment")
        if args.data_dir is not None and args.medoids_path is not None:
            raise SystemExit("assignments: --data-dir and --medoids-path are mutually exclusive")
    if args.command == "description-fit":
        if args.skip_detection and args.skip_token_embedding:
            raise SystemExit("description-fit: at least one metric must be enabled")
    if args.command == "gaussian-group-semantics":
        if args.distance_threshold < 0:
            raise SystemExit("gaussian-group-semantics: --distance-threshold must be non-negative")
        if args.top_groups <= 0:
            raise SystemExit("gaussian-group-semantics: --top-groups must be positive")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cluster-level metrics for a trained MFA")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--batch-size", type=int, default=128)
        sp.add_argument("--data-dir", default=None,
                        help="Directory with mfa_model.pt / mfa_model_shards.json, "
                             "or direct path to a .pt model file")
        sp.add_argument("--out-dir", default=None,
                        help="Where to save the result (default: same as --data-dir)")

    sp = sub.add_parser("overlap", help="Compute pairwise overlap metrics")
    add_common(sp)
    sp.add_argument("--batch-pairs", type=int, default=4096,
                    help="Pairs per batch for vectorized overlap")
    sp.set_defaults(func=cmd_overlap)

    sp = sub.add_parser("intrinsic-dim", help="Compute intrinsic dim per cluster")
    add_common(sp)
    sp.add_argument("--act-dir", default=None,
                    help="Monolithic layout: dir with activations.pt/tokens.pt "
                         "(default: same as --data-dir)")
    sp.add_argument("--shard-dir", default=None,
                    help="Shard layout: extraction output dir "
                         "(mutually exclusive with --act-dir)")
    sp.add_argument("--layer", type=int, default=None,
                    help="Layer to read from shards (required with --shard-dir)")
    sp.add_argument("--pca-device", default=None,
                    help="Device for the PCA phase (default: same as --device). "
                         "Set to 'cpu' to free the GPU for other jobs.")
    sp.add_argument("--pca-workers", type=int, default=1,
                    help="Parallel workers for the CPU PCA phase")
    sp.add_argument("--assignments-path", default=None,
                    help="Path to precomputed cluster assignments "
                         "(default: <data-dir>/mfa_model_assignments.pt)")
    sp.add_argument("--variance-threshold", type=float, default=0.90)
    sp.add_argument("--min-population", type=int, default=100)
    sp.add_argument("--max-samples-per-cluster", type=int, default=10000)
    sp.set_defaults(func=cmd_intrinsic_dim)

    sp = sub.add_parser("assignments",
                        help="Compute MFA or nearest-medoid cluster assignments")
    add_common(sp)
    sp.add_argument("--medoids-path", "--centroids-path", dest="medoids_path",
                    default=None,
                    help="Optional .npy/.pt medoids or centroids. If set, "
                         "assign by nearest Euclidean centroid instead of MFA responsibilities.")
    sp.add_argument("--shard-dir", required=True,
                    help="Activation shard directory produced by dalg-run-extraction")
    sp.add_argument("--layer", type=int, required=True,
                    help="Layer index to stream from --shard-dir")
    sp.add_argument("--drop-prefix", type=int, default=None,
                    help="Override the drop_prefix stored in <shard-dir>/config.json")
    sp.add_argument("--max-batches", type=int, default=None,
                    help="Stop after this many batches (for smoke tests)")
    sp.add_argument("--save-path", default=None,
                    help="Explicit save path (overrides --out-dir-based default)")
    sp.add_argument("--no-inference-cache", "--slow-responsibilities",
                    dest="use_inference_cache",
                    action="store_false", default=True,
                    help="Disable the MFA inference cache during responsibilities()")
    sp.set_defaults(func=cmd_assignments)


    sp = sub.add_parser("description-fit",
                        help="Judge whether descriptions fit marked token contexts")
    sp.add_argument("--labels-path", required=True,
                    help="Path to cluster_labels.json from dalg-interpret-mfa or dalg-label-mfa-clusters")
    sp.add_argument("--out-dir", default=None,
                    help="Where to save description_fit.json (default: labels directory)")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--clusters", type=int, nargs="*", default=None,
                    help="Specific cluster ids to score. Defaults to all clusters.")
    sp.add_argument("--max-clusters", type=int, default=None,
                    help="Debug convenience: score only the first N selected clusters")
    sp.add_argument("--positive-examples", type=int, default=8,
                    help="Top examples from the feature to judge as positives")
    sp.add_argument("--negative-examples", type=int, default=8,
                    help="Random examples from other features to judge as negatives")
    sp.add_argument("--skip-detection", action="store_true",
                    help="Skip the LLM activation-detection metric")
    sp.add_argument("--judge-model", default=None,
                    help="Judge model for detection scoring (default: labeling ORFEO_MODEL)")
    sp.add_argument("--judge-temperature", type=float, default=0.0)
    sp.add_argument("--judge-max-tokens", type=int, default=1024)
    sp.add_argument("--judge-workers", type=int, default=1)
    sp.add_argument("--skip-token-embedding", action="store_true",
                    help="Skip the experimental token-context embedding metric")
    sp.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Transformer encoder used for description and target-token embeddings")
    sp.add_argument("--embedding-device", default="cpu")
    sp.add_argument("--embedding-batch-size", type=int, default=32)
    sp.add_argument("--target-batch-size", type=int, default=16)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_description_fit)

    sp = sub.add_parser("description-semantics",
                        help="Embed descriptions and compute semantic similarity")
    sp.add_argument("--labels-path", required=True,
                    help="Path to cluster_labels.json from dalg-interpret-mfa or dalg-label-mfa-clusters")
    sp.add_argument("--out-dir", default=None,
                    help="Where to save description_semantics.pt and groups JSON")
    sp.add_argument("--clusters", type=int, nargs="*", default=None,
                    help="Specific cluster ids to embed. Defaults to all clusters.")
    sp.add_argument("--max-clusters", type=int, default=None,
                    help="Debug convenience: embed only the first N selected clusters")
    sp.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Transformer encoder used for description embeddings")
    sp.add_argument("--embedding-device", default="cpu")
    sp.add_argument("--embedding-batch-size", type=int, default=32)
    sp.add_argument("--similarity-batch-size", type=int, default=1024)
    sp.add_argument("--top-k", type=int, default=25,
                    help="Number of nearest description neighbors to keep per cluster")
    sp.add_argument("--similarity-threshold", type=float, default=0.70,
                    help="Cosine threshold for semantic groups")
    sp.add_argument("--min-group-size", type=int, default=2)
    sp.add_argument("--full-matrix", choices=("auto", "always", "never"), default="auto",
                    help="Whether to save a dense similarity matrix for heatmaps")
    sp.add_argument("--max-full-matrix-clusters", type=int, default=5000,
                    help="With --full-matrix auto, save dense matrix only up to this many clusters")
    sp.add_argument("--full-matrix-dtype", choices=("float32", "float16"), default="float32")
    sp.set_defaults(func=cmd_description_semantics)

    sp = sub.add_parser(
        "gaussian-group-semantics",
        help="Cluster Gaussians by Bhattacharyya distance and score label coherence",
    )
    sp.add_argument("--labels-path", required=True,
                    help="Path to cluster_labels.json from dalg-label-mfa-clusters")
    sp.add_argument("--overlap-path", required=True,
                    help="Path to overlap.pt containing a square Gaussian distance matrix")
    sp.add_argument("--out-dir", default=None,
                    help="Where to save gaussian_group_label_coherence.json and CSV")
    sp.add_argument("--distance-key", default="db",
                    help="Distance matrix key inside overlap.pt, usually db")
    sp.add_argument("--distance-threshold", type=float, required=True,
                    help="Hierarchical clustering distance cutoff")
    sp.add_argument("--linkage", choices=("average", "complete", "single"), default="average")
    sp.add_argument("--top-groups", type=int, default=10,
                    help="Number of largest non-singleton Gaussian-groups to summarize")
    sp.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Transformer encoder used for description embeddings")
    sp.add_argument("--embedding-device", default="cpu")
    sp.add_argument("--embedding-batch-size", type=int, default=32)
    sp.set_defaults(func=cmd_gaussian_group_semantics)

    return p


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
