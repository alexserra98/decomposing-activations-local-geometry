"""Temporary PC1-only tile alignment; delete this file to remove the experiment.

Uses saved runs under a dataset root and writes separate JSON reports and a CSV.
Example:
    PYTHONPATH=src .venv/bin/python scripts/temporary/evaluate_tile_alignment_pc1.py \
        dalg-cache/sae-manifold/sae_manifold_temperature_02d_llama31_8b_last \
        --output-dir outputs/evaluations/temperature_tile_alignment_pc1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import torch

from dalg.analysis.pc1_density import leading_components
from dalg.evaluation.common import load_evaluation_data, load_model
from dalg.evaluation.subspaces import effective_component_ranks
from dalg.evaluation.tile_alignment import evaluate_tile_alignment, tile_alignment_report_valid


@torch.no_grad()
def evaluate_pc1(model, live, *, kind, neighbors=5):
    """Cap positive ranks at one for this call without changing model geometry."""
    ranks = effective_component_ranks(model, 1.0)
    # This shared helper uses thin SVD for isotropic HDDC, avoiding D x D eigh.
    geometry = leading_components(model, kind)
    eligible = live.cpu().bool() & (ranks > 0)
    if kind == "kmeans":
        eligible &= model.pca_valid.cpu()
    vectors = geometry["vectors"]
    if not torch.allclose(vectors[eligible].norm(dim=1), torch.ones(
        int(eligible.sum()), dtype=vectors.dtype,
    )):
        raise ValueError("An eligible tile has no unit PC1")
    eigenvalues = torch.stack((geometry["eigenvalues"],
                               geometry["eigenvalues"] - geometry["eigengap"]), dim=1)
    # Scoped to the existing evaluator; restores the function even on failure.
    with patch("dalg.evaluation.tile_alignment.effective_component_ranks",
               return_value=ranks.clamp(max=1)):
        report = evaluate_tile_alignment(
            model, live, neighbors=neighbors,
            eigenspaces=(eigenvalues, vectors.unsqueeze(-1)),
        )
    assert tile_alignment_report_valid(report, components=model.K, neighbors=neighbors)
    for edge in report["comparisons"]:
        expected = float(torch.dot(vectors[edge["source"]], vectors[edge["target"]]).square().clamp(0, 1))
        assert abs(edge["leading_pc_alignment"] - expected) < 1e-10
        assert abs(edge["best_subset_pc_alignment"] - expected) < 1e-10
    report["experimental_override"] = {
        "rank_policy": "cap_positive_learned_ranks_at_one",
        "score": "squared_cosine_between_first_covariance_PCs",
        "original_component_ranks": ranks.tolist(),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.dataset_root.resolve()
    specs = sorted(root.glob("*/*/run_spec.json"))
    if not specs:
        raise ValueError(f"No saved runs under {root}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, spec in enumerate(specs):
        run = json.loads(spec.read_text())
        directory = spec.parent
        kind = run["training"]["model_kind"]
        stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
        protected = [spec, directory / "metrics.json", directory / f"{stem}.pt",
                     directory / f"{stem}_assignments.pt"]
        before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in protected]
        baseline = json.loads((directory / "metrics.json").read_text())
        original = baseline["tile_alignment"]
        model = load_model(directory, kind)
        data = load_evaluation_data(directory, shard_dir=run["dataset"]["shard_dir"],
                                    layer=run["dataset"]["layer"], model=model, model_kind=kind)
        if data.shard_dir.resolve() != root:
            raise ValueError(f"Unexpected dataset in {spec}")
        report = evaluate_pc1(model, data.cluster_sizes > 0, kind=kind,
                              neighbors=original["neighbors"])
        assert [(e["source"], e["target"]) for e in report["comparisons"]] == [
            (e["source"], e["target"]) for e in original["comparisons"]
        ]
        result = dict(run_id=run["run_id"], run_dir=str(directory),
                      model_kind=kind, dataset=data.dataset_summary(run["dataset"]["layer"]),
                      tile_alignment=report, original_tile_alignment=original)
        (args.output_dir / f"{run['run_id']}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n")
        rows.append({
            "run_id": run["run_id"], "model_kind": kind,
            "layer": run["dataset"]["layer"], "K": model.K,
            "surgery_threshold": run["training"]["arguments"]["surgery_threshold"],
            "neighbors": report["neighbors"], "eligible_components": report["eligible_components"],
            "pc1_alignment": report["leading_pc_alignment"]["mean"],
            "original_leading_alignment": original["leading_pc_alignment"]["mean"],
            "original_best_subset_alignment": original["best_subset_pc_alignment"]["mean"],
        })
        assert before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in protected]
        print(f"[{index + 1}/{len(specs)}] {run['run_id']}: "
              f"PC1={rows[-1]['pc1_alignment']}, "
              f"original={rows[-1]['original_leading_alignment']}", flush=True)
    with (args.output_dir / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} reports and {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
