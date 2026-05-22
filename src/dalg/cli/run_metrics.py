"""
CLI for cluster-level metrics on a trained MFA.

Subcommands:
    overlap        Pairwise overlap metrics between MFA components.
    intrinsic-dim  PCA-based intrinsic dimensionality per cluster.
    assignments    Hard cluster assignments + per-cluster peakedness stats
                   (streams activations through the MFA in one pass).

Both expect a trained MFA run directory (``--data-dir``) containing either
``mfa_model.pt`` or, for component-sharded runs, ``mfa_model_shards.json``.
``--data-dir`` may also be a direct path to a ``.pt`` model file.

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

    model_path, run_dir = _resolve_model_path(args.data_dir)
    out_dir = Path(args.out_dir or run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shard_dir is not None:
        results = compute_intrinsic_dims_from_shards(
            model_path, Path(args.shard_dir),
            layer=args.layer,
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
    """Compute hard cluster assignments and per-cluster peakedness stats.

    Streams activations from ``--shard-dir`` (at ``--layer``) through the
    MFA, takes the argmax of the responsibilities per token, and saves
    cluster sizes, hard assignments, max responsibility per sample, and
    mean per-cluster peakedness metrics to a ``.pt`` file.

    The default save path is ``<run_dir>/<model_stem>_assignments.pt`` so
    that ``intrinsic-dim`` can pick it up via its ``--assignments-path``
    default of ``<data-dir>/mfa_model_assignments.pt``.
    """
    import json
    from torch.utils.data import DataLoader

    from dalg.analysis.cluster_assignments import compute_assignments, _resolve_device
    from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index

    model_path, run_dir = _resolve_model_path(args.data_dir)
    device = _resolve_device(args.device)

    shard_dir = Path(args.shard_dir)
    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    drop_prefix = args.drop_prefix
    if drop_prefix is None:
        drop_prefix = int(extract_cfg.get("drop_prefix", 32))

    meta_index = load_meta_index(shard_dir, layer=args.layer)
    positions = list(range(len(meta_index)))
    print(f"shard_dir={shard_dir}  layer={args.layer}  rows={len(positions):,}")

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

    sizes, assignments, max_responsibilities, peakedness = compute_assignments(
        model_path,
        loader,
        device=device,
        max_batches=args.max_batches,
        use_inference_cache=args.use_inference_cache,
    )

    if args.save_path is not None:
        save_path = Path(args.save_path)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_assignments.pt" if args.max_batches is None \
            else f"_assignments_first{args.max_batches}_batches.pt"
        save_path = out_dir / f"{model_path.stem}{suffix}"

    torch.save({
        "cluster_sizes": sizes,
        "assignments": assignments,
        "max_responsibilities": max_responsibilities,
        "peakedness": peakedness,
        "K": int(sizes.numel()),
    }, save_path)
    print(f"Assignments saved to {save_path}")


def validate_args(args) -> None:
    if args.command == "intrinsic-dim":
        if args.shard_dir is not None and args.act_dir is not None:
            raise SystemExit("intrinsic-dim: --shard-dir and --act-dir are mutually exclusive")
        if args.shard_dir is not None and args.layer is None:
            raise SystemExit("intrinsic-dim: --layer is required with --shard-dir")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cluster-level metrics for a trained MFA")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--batch-size", type=int, default=128)
        sp.add_argument("--data-dir", required=True,
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
                        help="Compute MFA cluster assignments + peakedness stats")
    add_common(sp)
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

    return p


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
