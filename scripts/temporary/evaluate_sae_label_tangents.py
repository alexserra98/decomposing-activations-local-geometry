"""Evaluate all matched saved years/temperature models on their held-out test sets."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch

import evaluate_year_tangent_alignment as evaluator


def comparison_plan(roots: list[Path]) -> list[dict]:
    plan = []
    for root in roots:
        root = root.resolve()
        config = json.loads((root / "test/config.json").read_text())
        with (root / "test" / config["source_records"]).open() as handle:
            subset = json.loads(next(handle))["subset"]
        label_field = {"years": "year", "temperature": "fahrenheit"}[subset]
        groups = {}
        for kind, suffix in (("kmeans", "kmeans_pca"), ("hddc", "hddc_em")):
            runs = {}
            for path in sorted(root.glob(f"*_{suffix}/*/config.json")):
                saved = json.loads(path.read_text())
                key = (saved["layer"], saved["K"], saved["surgery_threshold"])
                if key in runs:
                    raise ValueError(f"Multiple {kind} runs match {key} in {root}")
                runs[key] = str(path.parent)
            groups[kind] = runs
        if not groups["kmeans"] or groups["kmeans"].keys() != groups["hddc"].keys():
            raise ValueError(f"Expected matching KMeans/HDDC saved configurations in {root}")
        for layer, k, threshold in sorted(groups["kmeans"]):
            key = (layer, k, threshold)
            plan.append(dict(dataset=subset, root=str(root), layer=layer, K=k, threshold=threshold,
                             label_field=label_field, bin_width=1. if subset == "temperature" else None,
                             kmeans_run=groups["kmeans"][key], hddc_run=groups["hddc"][key],
                             name=f"{subset}_l{layer:02d}_k{k}_threshold{threshold:g}"))
    return plan


def summary_row(item, inputs, report, elapsed):
    reliability = report["reference_agreement"]
    labels, counts = inputs["years"].unique(return_counts=True)
    row = dict(dataset=item["dataset"], layer=item["layer"], K=item["K"], threshold=item["threshold"],
               name=item["name"], test_rows=len(inputs["years"]), observed_labels=len(labels),
               min_label_count=int(counts.min()), max_label_count=int(counts.max()),
               median_label_count=float(counts.double().quantile(.5)),
               possible_consecutive_segments=int(labels[-1] - labels[0]),
               observed_consecutive_segments=int((labels[1:] == labels[:-1] + 1).sum()),
               valid_segments=len(report["segment_years"]),
               reference_agreement_mean=float(reliability.mean()),
               reference_agreement_median=float(reliability.quantile(.5)),
               reference_agreement_q10=float(reliability.quantile(.1)),
               reference_agreement_q90=float(reliability.quantile(.9)),
               reference_agreement_above_half=float((reliability > .5).double().mean()),
               control_segments=len(report["negative_control"]["segment_years"]), elapsed_seconds=elapsed,
               **{k: v for k, v in report["comparison"].items() if k != "summary"})
    for kind, result in report["methods"].items():
        for key in ("score", "conditional_score", "unidentified_weight", "eligible_component_fraction",
                    "negative_control_score", "matched_real_score"):
            row[f"{kind}_{key}"] = result[key]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    plan = comparison_plan(args.roots)
    args.output.mkdir(parents=True, exist_ok=False)
    args.details.mkdir(parents=True, exist_ok=False)
    (args.output / "comparisons").symlink_to(args.details.resolve(), target_is_directory=True)
    (args.output / "comparison_plan.json").write_text(json.dumps(dict(
        seed=42, min_samples=2, atol=1e-12, rtol=1e-10, device=args.device,
        selection="All matched saved layer/K/rank-threshold configurations; no score-based selection",
        label_convention="Exact integer years; floor(Fahrenheit) bins anchored at zero, width 1°F",
        comparisons=plan), indent=2) + "\n")
    print(f"Evaluating {len(plan)} pairs; reports in {args.output.resolve()}", flush=True)
    rows = []
    for index, item in enumerate(plan):
        start = time.monotonic()
        print(f"[{index+1}/{len(plan)}] {item['name']}: load", flush=True)
        inputs = evaluator.load_inputs(Path(item["kmeans_run"]), Path(item["hddc_run"]),
                                       Path(item["root"]) / "test", args.batch_size,
                                       label_field=item["label_field"], bin_width=item["bin_width"])
        print(f"[{index+1}/{len(plan)}] {item['name']}: assign and extract PC1", flush=True)
        geometry = evaluator.model_geometry_and_assignments(inputs["models"], inputs["activations"],
                                                            args.device, args.batch_size)
        report = evaluator.evaluate_crossfit(inputs["activations"], inputs["years"], inputs["templates"], geometry)
        report["provenance"] = inputs["provenance"]
        report["comparison"]["summary"] = report["comparison"]["summary"].replace(
            "year-conditioned", f"{item['label_field']}-conditioned")
        report["prompt_distribution"] = evaluator.prompt_distribution(inputs["records"], report["fold_rows"], inputs["years"])
        evaluator.write_reports(report, args.details / item["name"])
        row = summary_row(item, inputs, report, time.monotonic() - start)
        rows.append(row)
        with (args.output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{index+1}/{len(plan)}] {item['name']}: HDDC={row['hddc_score']:.6f} "
              f"KMeans={row['kmeans_score']:.6f}, reference={row['reference_agreement_median']:.4f}, "
              f"{row['valid_segments']} segments, {row['elapsed_seconds']:.1f}s", flush=True)
        del inputs, geometry, report
        gc.collect()
    (args.output / "complete.json").write_text(json.dumps(dict(comparisons=len(rows)), indent=2) + "\n")


if __name__ == "__main__":
    main()
