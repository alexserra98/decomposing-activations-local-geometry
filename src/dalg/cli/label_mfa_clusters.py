from __future__ import annotations

import argparse
from pathlib import Path

from dalg.analysis.cluster_labeling import ORFEO_MODEL, label_mfa_clusters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Label MFA clusters with an LLM using precomputed hard assignments, "
            "top responsibilities, and token-window contexts."
        )
    )
    parser.add_argument(
        "--assignments-path",
        type=Path,
        required=True,
        help="Path to mfa_model_assignments.pt.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for top_activations.pt, cluster_examples.json, and cluster_labels.json.",
    )

    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--windows-dataset",
        type=Path,
        default=None,
        help="HF windows dataset. Defaults to shard_dir/config.json['dataset'] when available.",
    )
    parser.add_argument("--tokenizer", default="google/gemma-2b")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--drop-prefix", type=int, default=None)

    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-examples-per-cluster", type=int, default=None)
    parser.add_argument("--pad", type=int, default=10)
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="*",
        default=None,
        help="Specific cluster ids to label. Defaults to all clusters.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Debug convenience: label clusters [0, max_clusters). Ignored if --clusters is passed.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)

    parser.add_argument(
        "--top-index-path",
        type=Path,
        default=None,
        help="Optional cached top activation index path. Defaults to <out-dir>/top_activations.pt.",
    )

    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only build top activations and context examples; do not call Orfeo.",
    )
    parser.add_argument("--llm-model", default=ORFEO_MODEL)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=512)
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = label_mfa_clusters(
        args.assignments_path,
        out_dir=args.out_dir,
        shard_dir=args.shard_dir,
        layer=args.layer,
        windows_dataset=args.windows_dataset,
        tokenizer_name=args.tokenizer,
        window=args.window,
        drop_prefix=args.drop_prefix,
        top_n=args.top_n,
        max_examples_per_cluster=args.max_examples_per_cluster,
        pad=args.pad,
        cluster_ids=args.clusters,
        max_clusters=args.max_clusters,
        chunk_size=args.chunk_size,
        skip_llm=args.skip_llm,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_workers=args.llm_workers,
        top_index_path=args.top_index_path,
        show_progress=not args.quiet,
    )
    out_path = Path(args.out_dir) / "cluster_labels.json"
    print(f"Saved labels for {len(output['clusters'])} clusters to {out_path}")


if __name__ == "__main__":
    main()
