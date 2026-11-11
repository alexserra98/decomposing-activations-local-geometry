"""Compare saved models using label-centroid or cross-fitted mean secants."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch

from dalg.analysis import label_tangent_alignment as metric
from dalg.analysis.pc1_density import leading_components
from dalg.data.prompt_activations import load_prompt_activation_index
from dalg.data.shard_activations import ActivationBatchDataset
from dalg.evaluation.common import load_evaluation_data, load_model, resolve_device
from dalg.evaluation.unsupervised import validate_test_shards


def evaluation_labels(records: list[dict], label_field: str, bin_width: float | None = None) -> torch.Tensor:
    """Map exact integer labels or explicit fixed-width numeric bins to integers.

    Bins are [j * width, (j + 1) * width), anchored at zero; negative values
    use floor, not truncation. Float-valued integer years retain their values.
    """
    if bin_width is not None and (not math.isfinite(bin_width) or bin_width <= 0):
        raise ValueError("bin_width must be finite and positive")
    labels = []
    for record in records:
        value = record["labels"].get(label_field)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Every row must carry a finite numerical {label_field} label")
        if bin_width is None and value != int(value):
            raise ValueError("Non-integer labels require an explicit bin_width")
        labels.append(int(value) if bin_width is None else math.floor(value / bin_width))
    return torch.tensor(labels, dtype=torch.int64)


def load_inputs(kmeans_run: Path, hddc_run: Path, test_dir: Path, batch_size: int,
                *, label_field: str = "year", bin_width: float | None = None) -> dict:
    """Load raw last-token data and verify saved development provenance."""
    expected_subset = {"year": "years", "fahrenheit": "temperature"}[label_field]
    runs = {"kmeans": kmeans_run, "hddc": hddc_run}
    configs = {kind: json.loads((path / "config.json").read_text()) for kind, path in runs.items()}
    if configs["hddc"].get("fit_method") != "em":
        raise ValueError("The HDDC checkpoint must have been fitted with EM")
    layer = configs["kmeans"]["layer"]
    root = Path(configs["kmeans"]["shard_dir"])
    if any(c["layer"] != layer or Path(c["shard_dir"]).resolve() != root.resolve()
           for c in configs.values()):
        raise ValueError("Compare runs fitted on the same development shards and layer")
    models, sources = {}, {}
    for kind, run in runs.items():
        models[kind] = load_model(run, kind)
        data = load_evaluation_data(run, shard_dir=root, layer=layer,
                                    model=models[kind], model_kind=kind)
        if data.subset_spec is not None or data.config["window"] != 1 or data.drop_prefix != 0:
            raise ValueError("This runner requires unsliced last-token datasets")
        # Current trainers fit directly in shard space; no visualization PCA applies.
        if models[kind].D != data.config["d_model"]:
            raise ValueError("Checkpoint and raw activation dimensions differ")
        sources[kind] = validate_test_shards(root, test_dir, layer=layer,
                                           model_kind=kind, drop_prefix=0)
    development = load_prompt_activation_index(root, layer)
    records = load_prompt_activation_index(test_dir, layer)
    if any(r["subset"] != expected_subset for r in development + records):
        raise ValueError(f"This comparison requires a {expected_subset}-only dataset")
    if {r["source_row_id"] for r in development} & {r["source_row_id"] for r in records}:
        raise ValueError("Test and development source row IDs overlap")
    if {r["prompt"] for r in development} & {r["prompt"] for r in records}:
        raise ValueError("Test and development contain identical prompts")
    train_config = json.loads((root / "config.json").read_text())
    test_config = json.loads((test_dir / "config.json").read_text())
    for field in ("model", "model_revision", "local_checkpoint_sha256", "tokenizer_revision",
                  "tokenizer_sha256", "representation", "mode", "dtype", "token_selection",
                  "prepend_bos", "append_eos", "chat_template"):
        if train_config[field] != test_config[field]:
            raise ValueError(f"Test representation differs in {field}")
    labels = evaluation_labels(records, label_field, bin_width)
    template_present = ["template_id" in r["labels"] for r in records]
    if any(template_present) and not all(template_present):
        raise ValueError("Template IDs must be present for all rows or absent for all")
    templates = [str(r["labels"]["template_id"]) for r in records] if all(template_present) else None
    dataset = ActivationBatchDataset(test_dir, layer=layer, batch_size=batch_size,
                                     drop_prefix=0, return_metadata=True,
                                     shuffle_shards=False, shuffle_within_shard=False)
    batches, row_ids = [], []
    for values, rows, _ in dataset:
        batches.append(values)
        row_ids.extend(rows.tolist())
    if row_ids != [r["global_row"] for r in records]:
        raise ValueError("Activation and prompt metadata orders differ")
    return dict(models=models, activations=torch.cat(batches), records=records,
                years=labels, templates=templates,
                provenance=dict(runs={k: str(p.resolve()) for k, p in runs.items()},
                                source=sources, representation="raw fitted shard space",
                                label_field=label_field, bin_width=bin_width, bin_origin=0,
                                hddc_assignment_arithmetic="float64 centered cached posterior",
                                development_records_sha256=train_config["source_records_sha256"],
                                test_records_sha256=test_config["source_records_sha256"]))


@torch.no_grad()
def model_geometry_and_assignments(models: dict, activations: torch.Tensor,
                                   device: str, batch_size: int) -> dict:
    """Reuse saved covariance PCs and each model's native assignment rule."""
    result = {}
    for kind, model in models.items():
        geometry = leading_components(model, kind)
        reasons = geometry["exclusion_reasons"][:]
        for k, rank in enumerate(geometry["ranks"]):
            if not reasons[k] and rank == 0:
                reasons[k] = "zero_rank"
            if not reasons[k] and not torch.isfinite(geometry["vectors"][k]).all():
                reasons[k] = "invalid_pc1"
        model = model.to(resolve_device(device)).eval()
        assignments = []
        cache = (model.inference_cache(dtype=torch.float64, component_chunk_size=4)
                 if kind == "hddc" else nullcontext())
        with cache:
            for batch in activations.split(batch_size):
                batch = batch.to(device=model.mu.device, dtype=model.mu.dtype)
                ids = model.predict(batch) if kind == "kmeans" else model.responsibilities(batch).argmax(1)
                assignments.append(ids.cpu())
        result[kind] = dict(pc1=geometry["vectors"], ranks=geometry["ranks"],
                            identifiable=torch.tensor([not r for r in reasons]),
                            exclusion_reasons=reasons, assignments=torch.cat(assignments))
        model.cpu()
    return result


def _aligned_tangents(reference: dict, segments: torch.Tensor) -> torch.Tensor:
    lookup = {int(y): i for i, y in enumerate(reference["segment_years"])}
    return reference["tangents"][[lookup[int(y)] for y in segments]]


def _common_segments(references: list[dict]) -> torch.Tensor:
    common = set(references[0]["segment_years"].tolist())
    for reference in references[1:]:
        common.intersection_update(reference["segment_years"].tolist())
    return torch.tensor(sorted(common), dtype=torch.int64)


def _fold_scores(references, folds, years, geometry, segments):
    scores = []
    for reference, evaluation_rows in zip(references, reversed(folds)):
        weights = metric.segment_assignment_weights(
            metric.label_block_starts(years[evaluation_rows], reference.get("block_size", 1)),
            geometry["assignments"][evaluation_rows],
            segments, len(geometry["pc1"]), label_step=reference["label_step"])
        score = metric.score_pc1_alignment(_aligned_tangents(reference, segments),
                                           geometry["pc1"], geometry["identifiable"], weights)
        diagnostics = metric.component_alignment_diagnostics(
            geometry["assignments"][evaluation_rows], weights, score["alignment"],
            geometry["identifiable"], geometry["ranks"])
        scores.append(dict(**score, components=diagnostics))
    return scores


def evaluate_crossfit(activations, years, templates, geometries, *, seed=42,
                      min_samples=2, label_step=1, atol=1e-12, rtol=1e-10) -> dict:
    """Apply the same folds and valid reference population to both methods."""
    folds = metric.make_crossfit_folds(years, templates, seed=seed)
    if len(folds) != 2 or any(len(fold) == 0 for fold in folds):
        raise ValueError("Cross-fitting requires two nonempty folds")
    if not torch.equal(torch.cat(folds).sort().values, torch.arange(len(years))):
        raise ValueError("Fold indices must partition the test set exactly")
    options = dict(min_samples=min_samples, label_step=label_step, atol=atol, rtol=rtol)
    references = [metric.estimate_label_mean_tangents(activations[rows].double(), years[rows], **options)
                  for rows in folds]
    segments = _common_segments(references)
    if len(segments) == 0:
        raise ValueError("No valid consecutive-year segments shared by both reference folds")
    shuffled = [metric.estimate_label_mean_tangents(
        activations[rows].double(), metric.shuffled_reference_labels(years[rows], seed=seed + 1 + i),
        **options) for i, rows in enumerate(folds)]
    control_segments = _common_segments(references + shuffled)
    report = dict(settings=dict(seed=seed, **options), segment_years=segments,
                  fold_rows=folds, reference=[{k: v for k, v in r.items()
                                              if k not in ("means", "tangents")} for r in references],
                  reference_agreement=metric.reference_agreement(
                      *[_aligned_tangents(r, segments) for r in references]),
                  negative_control=dict(segment_years=control_segments,
                                        excluded_from_primary=len(segments) - len(control_segments),
                                        reference_exclusions=[r["exclusions"] for r in shuffled]),
                  methods={})
    for kind, geometry in geometries.items():
        scores = _fold_scores(references, folds, years, geometry, segments)
        result = dict(**metric.combine_crossfit_scores(scores),
                      eligible_component_fraction=float(geometry["identifiable"].double().mean()),
                      exclusion_reasons=geometry["exclusion_reasons"], folds=scores)
        if len(control_segments):
            control = _fold_scores(shuffled, folds, years, geometry, control_segments)
            matched = _fold_scores(references, folds, years, geometry, control_segments)
            combined_control = metric.combine_crossfit_scores(control)
            result["negative_control_score"] = combined_control["score"]
            result["matched_real_score"] = metric.combine_crossfit_scores(matched)["score"]
            result["negative_control_segment_scores"] = combined_control["segment_scores"]
        else:
            result.update(negative_control_score=None, matched_real_score=None,
                          negative_control_segment_scores=None)
        report["methods"][kind] = result
    report["comparison"] = metric.summarize_paired_alignment(
        segments, report["methods"]["hddc"]["segment_scores"],
        report["methods"]["kmeans"]["segment_scores"])
    return report


def evaluate_template_alignment(activations, years, templates, geometries, *, template_id,
                                seed=42, min_samples=1, block_size=100, atol=1e-12, rtol=1e-10) -> dict:
    """Evaluate successive block centroids and weights using all block prompts.

    The entrypoint and template arguments are retained for notebook calls;
    template selection does not affect centroids, weights, or the control.
    """
    options = dict(min_samples=min_samples, block_size=block_size, atol=atol, rtol=rtol)
    reference = metric.estimate_label_tangents(activations, years, **options)
    segments = reference["segment_years"]
    if len(segments) == 0:
        raise ValueError(f"No valid secants between successive {block_size}-unit label blocks")
    shuffled = metric.estimate_label_tangents(
        activations, metric.shuffled_reference_labels(years, seed=seed + 1), **options)
    control_segments = _common_segments([reference, shuffled])
    populations = (torch.arange(len(years)),)
    report = dict(
        settings=dict(mode="label_centroids", label_step=block_size, seed=seed, reference_population="all_endpoint_prompts",
                      assignment_population="all_endpoint_prompts", **options),
        segment_years=segments, fold_rows=populations, population_names=["all_prompts"],
        reference=[{k: v for k, v in reference.items() if k not in ("means", "tangents")}],
        reference_agreement=None,
        negative_control=dict(segment_years=control_segments,
                              excluded_from_primary=len(segments) - len(control_segments),
                              reference_exclusions=[shuffled["exclusions"]]),
        methods={})
    for kind, geometry in geometries.items():
        score = _fold_scores([reference], populations, years, geometry, segments)[0]
        result = dict(**{k: score[k] for k in (
            "segment_scores", "eligible_mass", "score", "conditional_score", "unidentified_weight")},
            eligible_component_fraction=float(geometry["identifiable"].double().mean()),
            exclusion_reasons=geometry["exclusion_reasons"], folds=[score])
        if len(control_segments):
            control = _fold_scores([shuffled], populations, years, geometry, control_segments)[0]
            matched = _fold_scores([reference], populations, years, geometry, control_segments)[0]
            result.update(negative_control_score=control["score"], matched_real_score=matched["score"],
                          negative_control_segment_scores=control["segment_scores"])
        else:
            result.update(negative_control_score=None, matched_real_score=None,
                          negative_control_segment_scores=None)
        report["methods"][kind] = result
    report["comparison"] = metric.summarize_paired_alignment(
        segments, report["methods"]["hddc"]["segment_scores"],
        report["methods"]["kmeans"]["segment_scores"])
    return report


def prompt_distribution(records, folds, labels=None) -> list[dict]:
    """Count year/template cells separately in each reference population."""
    rows = []
    names = ("all_prompts",) if len(folds) == 1 else ("A", "B")
    for name, indices in zip(names, folds):
        counts = Counter((int(labels[i]) if labels is not None else records[i]["labels"]["year"],
                          str(records[i]["labels"].get("template_id", "unknown")))
                         for i in indices.tolist())
        rows.extend(dict(fold=name, year=y, template=t, count=n)
                    for (y, t), n in sorted(counts.items()))
    return rows


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(report: dict, output: Path) -> None:
    """Write JSON, flat CSV tables, and standalone interactive diagnostic plots."""
    import plotly.express as px
    import plotly.graph_objects as go

    output.mkdir(parents=True, exist_ok=False)
    data = _jsonable(report)
    field = data.get("provenance", {}).get("label_field", "year")
    width = data.get("provenance", {}).get("bin_width") or 1
    label_step = data.get("settings", {}).get("label_step", 1)
    coordinate_name = "Year" if field == "year" else "Temperature bin lower edge (°F)"
    (output / "alignment.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    segments = []
    for i, year in enumerate(data["segment_years"]):
        k = data["methods"]["kmeans"]["segment_scores"][i]
        h = data["methods"]["hddc"]["segment_scores"][i]
        segments.append(dict(segment_start=year * width, segment_end=(year + label_step) * width,
                             kmeans=k, hddc=h, difference=h-k,
                             reference_agreement=(data["reference_agreement"][i]
                                                  if data["reference_agreement"] is not None else None),
                             kmeans_eligible_mass=data["methods"]["kmeans"]["eligible_mass"][i],
                             hddc_eligible_mass=data["methods"]["hddc"]["eligible_mass"][i]))
    _write_csv(output / "segments.csv", segments)
    summary = [dict(method=kind, **{key: result[key] for key in (
        "score", "conditional_score", "unidentified_weight", "eligible_component_fraction",
        "negative_control_score", "matched_real_score")}) for kind, result in data["methods"].items()]
    _write_csv(output / "summary.csv", summary)
    distribution = [dict(fold=r["fold"], label_start=r["year"] * width,
                          template=r["template"], count=r["count"]) for r in data["prompt_distribution"]]
    _write_csv(output / "prompt_distribution.csv", distribution)
    components = []
    for kind, result in data["methods"].items():
        for fold, scores in zip(data.get("population_names", ("A_reference", "B_reference")), result["folds"]):
            diagnostics = scores["components"]
            for k in range(len(result["exclusion_reasons"])):
                components.append(dict(method=kind, fold=fold, component=k,
                                       exclusion_reason=result["exclusion_reasons"][k],
                                       **{key: value[k] for key, value in diagnostics.items()}))
    _write_csv(output / "components.csv", components)
    figure = go.Figure()
    for kind in ("hddc", "kmeans"):
        figure.add_scatter(x=[y * width for y in data["segment_years"]], y=data["methods"][kind]["segment_scores"],
                           mode="markers", name=kind)
    figure.update_layout(xaxis_title=f"{coordinate_name}: segment start (step = {width * label_step:g})",
                         yaxis_title="PC1 alignment", yaxis_range=[0, 1])
    figure.write_html(output / "alignment_by_year.html", include_plotlyjs=True)
    px.histogram(segments, x="difference", labels={"difference": "HDDC − KMeans"}).write_html(
        output / "paired_differences.html", include_plotlyjs=True)
    if data["reference_agreement"] is not None:
        px.histogram(segments, x="reference_agreement", range_x=[0, 1]).write_html(
            output / "reference_reliability.html", include_plotlyjs=True)
    px.scatter(components, x="assignment_mass", y="contribution", color="rank",
               facet_col="method", facet_row="fold", hover_data=["component", "conditional_alignment",
                                                               "identifiable", "exclusion_reason"]).write_html(
        output / "component_diagnostics.html", include_plotlyjs=True)
    selected = data.get("settings", {}).get("mode") == "selected_template"
    target = (f"template-matched prompt activation (template {data['settings']['template_id']})"
              if selected else f"{field}-conditioned mean activation")
    block_size = data.get("settings", {}).get("block_size", 1)
    if block_size > 1:
        target = f"mean activation of successive {block_size * width:g}-unit {field} blocks"
    summary_text = data["comparison"]["summary"].replace("label secants", target)
    (output / "summary.txt").write_text(summary_text + "\n\n"
        f"Alignment measures changes in {target}. It does not establish "
        "global manifold coverage, intrinsic dimension recovery, or statistical significance.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmeans-run", type=Path, required=True)
    parser.add_argument("--hddc-run", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--label-field", choices=("year", "fahrenheit"), default="year")
    parser.add_argument("--bin-width", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-id", help="Legacy notebook mode: use all-label centroids without cross-fitting; ID is ignored")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.batch_size < 1 or args.min_samples < 1:
        parser.error("batch-size and min-samples must be positive")
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error("atol and rtol must be finite and nonnegative")
    if args.output.exists():
        parser.error("output must be a new directory")
    inputs = load_inputs(args.kmeans_run, args.hddc_run, args.test_dir, args.batch_size,
                         label_field=args.label_field, bin_width=args.bin_width)
    geometries = model_geometry_and_assignments(inputs["models"], inputs["activations"],
                                                args.device, args.batch_size)
    evaluate = evaluate_template_alignment if args.template_id is not None else evaluate_crossfit
    options = dict(min_samples=args.min_samples)
    if args.template_id is not None:
        options["template_id"] = args.template_id
        options["block_size"] = args.block_size
    report = evaluate(inputs["activations"], inputs["years"], inputs["templates"],
                      geometries, seed=args.seed, atol=args.atol, rtol=args.rtol, **options)
    report["provenance"] = inputs["provenance"]
    report["prompt_distribution"] = prompt_distribution(inputs["records"], report["fold_rows"], inputs["years"])
    write_reports(report, args.output)


if __name__ == "__main__":
    main()
