"""Recompute only PC1 density for neutral-year K=50, tau=0.1, layer-4 models."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from dalg.analysis.aggregate_metrics import aggregate_metrics
from dalg.evaluation.common import load_evaluation_data, load_model
from dalg.evaluation.pc1_density import evaluate_pc1_density, pc1_density_report_valid
from dalg.pipeline.common import _write_json_atomic
from dalg.pipeline.completion import _mark_stage
from dalg.pipeline.evaluation import _evaluation_metrics_valid


ROOT = Path("dalg-cache/sae-manifold/sae_manifold_years_neutral_llama31_8b_last")


def main():
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    targets = []
    for path in sorted(ROOT.glob("*/*/run_spec.json")):
        if any("SKIP" in part for part in path.parts):
            continue
        spec = json.loads(path.read_text())
        args = spec["training"]["arguments"]
        if args["K"] == 50 and args["layer"] == 4 and args.get("surgery_threshold") == .1:
            targets.append((path.parent.resolve(), spec))
    assert len(targets) == 2
    assert {s["training"]["model_kind"] for _, s in targets} == {"hddc", "kmeans"}
    output = Path("outputs/evaluations") / f"years_neutral_pc1_pca90_{os.environ['SLURM_JOB_ID']}"
    output.mkdir(parents=True, exist_ok=False)
    summaries = set()
    for directory, spec in targets:
        kind = spec["training"]["model_kind"]
        print(f"Computing {kind}: {directory}", flush=True)
        path = directory / "metrics.json"
        original = path.read_bytes()
        report = json.loads(original)
        backup = output / kind
        backup.mkdir()
        (backup / "metrics_before.json").write_bytes(original)
        (backup / "EVALUATION_COMPLETED_before.json").write_bytes(
            (directory / "EVALUATION_COMPLETED.json").read_bytes()
        )
        cfg = {**report.get("evaluation_config", spec["evaluation"]),
               "pc1_density": True, "pc1_density_space": "pca90"}
        model = load_model(directory, kind)
        data = load_evaluation_data(directory, shard_dir=spec["dataset"]["shard_dir"],
                                    layer=4, model=model, model_kind=kind)
        metric = evaluate_pc1_density(
            model, data, model_kind=kind, layer=4, batch_size=cfg["batch_size"],
            space="pca90", neighbors=cfg.get("pc1_density_neighbors", 20),
            random_directions=cfg.get("pc1_density_random_directions", 32),
            seed=cfg.get("pc1_density_seed", 42),
        )
        assert pc1_density_report_valid(metric, components=model.K, cfg=cfg,
                                        dataset=data.dataset_summary(4))
        report["pc1_density"] = metric
        report["evaluation_config"] = cfg
        report["pc1_density_update"] = {
            "script": str(Path(__file__).resolve()), "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "previous_report": str((backup / "metrics_before.json").resolve()),
            "scope": "pc1_density_only",
        }
        assert _evaluation_metrics_valid({**spec, "evaluation": cfg}, report)
        json.dumps(report, allow_nan=False)
        assert path.read_bytes() == original, "Metrics changed during computation"
        _write_json_atomic(path, report)
        _mark_stage(directory, "evaluation", path)
        result = {"run_dir": str(directory), "projection": metric["projection"],
                  "log_retention": metric["log_retention"], "log_enrichment": metric["log_enrichment"]}
        _write_json_atomic(backup / "result.json", result)
        print(json.dumps(result), flush=True)
        summaries.add(directory.parent)
    summaries.add(ROOT.resolve())
    for directory in sorted(summaries):
        general, manifolds = aggregate_metrics(directory)
        general.to_csv(directory / "general_metrics.csv", index=False)
        manifolds.to_csv(directory / "manifold_metrics.csv", index=False)
        print(f"Refreshed summaries: {directory}", flush=True)


if __name__ == "__main__":
    main()
