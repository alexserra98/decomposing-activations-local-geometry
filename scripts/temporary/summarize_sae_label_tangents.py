"""Render the saved years/temperature alignment sweep without recomputing scores."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    table = pd.read_csv(root / "summary.csv")
    completed = json.loads((root / "complete.json").read_text())["comparisons"]
    assert len(table) == completed
    groups = [(dataset, layer) for dataset in ("years", "temperature") for layer in (4, 28)]
    colors = {"hddc": "#b34b38", "kmeans": "#236e9f"}
    segments = {row["name"]: pd.read_csv(root / "comparisons" / row["name"] / "segments.csv")
                for _, row in table.iterrows()}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (dataset, layer) in zip(axes.flat, groups):
        group = table[(table.dataset == dataset) & (table.layer == layer)]
        for k, marker in ((50, "o"), (100, "s")):
            rows = group[group.K == k].sort_values("threshold")
            for method in ("hddc", "kmeans"):
                ax.plot(rows.threshold, rows[f"{method}_score"], color=colors[method], marker=marker,
                        linestyle="-" if k == 50 else "--", label=f"{method.upper()}, K={k}")
        ax.set(xscale="log", xlabel="Saved rank threshold", ylabel="Primary PC1 alignment",
               title=f"{dataset.title()}, layer {layer}")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(root / "alignment_scores.png", dpi=180)
    fig.savefig(root / "alignment_scores.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    group_rows = []
    for ax, (dataset, layer) in zip(axes.flat, groups):
        group = table[(table.dataset == dataset) & (table.layer == layer)].sort_values(["K", "threshold"])
        first = group.iloc[0]
        reliability = segments[first["name"]].reference_agreement
        for _, row in group.iterrows():
            np.testing.assert_allclose(segments[row["name"]].reference_agreement, reliability)
        ax.hist(reliability, bins=np.linspace(0, 1, 26), color="#5c7860", edgecolor="white")
        ax.axvline(reliability.median(), color="black", linestyle="--", label=f"Median {reliability.median():.3f}")
        ax.set(xlabel="Squared reference tangent agreement", ylabel="Segments", xlim=(0, 1),
               title=f"{dataset.title()}, layer {layer} ({len(reliability)} segments)")
        ax.legend(fontsize=9)
        group_rows.append(dict(dataset=dataset, layer=layer, test_rows=int(first.test_rows),
                               valid_segments=int(first.valid_segments),
                               possible_segments=int(first.possible_consecutive_segments),
                               median_reference_agreement=float(first.reference_agreement_median),
                               mean_reference_agreement=float(first.reference_agreement_mean),
                               hddc_wins=int((group.mean_difference > 1e-12).sum()), comparisons=len(group),
                               hddc_real_above_shuffle=int((group.hddc_matched_real_score > group.hddc_negative_control_score).sum()),
                               kmeans_real_above_shuffle=int((group.kmeans_matched_real_score > group.kmeans_negative_control_score).sum())))
    fig.savefig(root / "reference_reliability.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(group_rows).to_csv(root / "group_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (dataset, layer) in zip(axes.flat, groups):
        group = table[(table.dataset == dataset) & (table.layer == layer)].sort_values(["K", "threshold"])
        ax.boxplot([segments[r["name"]].difference for _, r in group.iterrows()], showfliers=False,
                   tick_labels=[f"K{int(r.K)}\n{r.threshold:g}" for _, r in group.iterrows()])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(ylabel="Per-segment HDDC − KMeans", xlabel="K and saved rank threshold",
               title=f"{dataset.title()}, layer {layer}")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(root / "paired_segment_differences.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (dataset, layer) in zip(axes.flat, groups):
        group = table[(table.dataset == dataset) & (table.layer == layer)]
        maximum = 0.
        for method in ("hddc", "kmeans"):
            for k, marker in ((50, "o"), (100, "s")):
                rows = group[group.K == k]
                ax.scatter(rows[f"{method}_negative_control_score"], rows[f"{method}_matched_real_score"],
                           color=colors[method], marker=marker, label=f"{method.upper()}, K={k}")
                maximum = max(maximum, rows[f"{method}_negative_control_score"].max(), rows[f"{method}_matched_real_score"].max())
        ax.plot([0, maximum * 1.05], [0, maximum * 1.05], "k--", linewidth=.8)
        ax.set(xlabel="Shuffled-reference alignment", ylabel="Real-reference alignment (matched segments)",
               title=f"{dataset.title()}, layer {layer}", xlim=(0, maximum * 1.08), ylim=(0, maximum * 1.08))
        ax.legend(fontsize=8)
    fig.savefig(root / "negative_controls.png", dpi=180)
    plt.close(fig)

    display = table[["dataset", "layer", "K", "threshold", "hddc_score", "kmeans_score", "mean_difference",
                     "positive_fraction", "hddc_negative_control_score", "kmeans_negative_control_score",
                     "hddc_unidentified_weight", "kmeans_unidentified_weight", "name"]].copy()
    display["details"] = display["name"].map(lambda name:
        f'<a href="comparisons/{html.escape(name)}/alignment_by_year.html">segments</a> · '
        f'<a href="comparisons/{html.escape(name)}/component_diagnostics.html">components</a>')
    display = display.drop(columns="name").rename(columns={"positive_fraction": "fraction_segments_hddc_higher"})
    table_html = display.to_html(index=False, float_format=lambda x: f"{x:.5g}", escape=False)
    overview_html = pd.DataFrame(group_rows).to_html(index=False, float_format=lambda x: f"{x:.4f}")
    lines = []
    for row in group_rows:
        lines.append(f"{row['dataset']}, layer {row['layer']}: HDDC higher in {row['hddc_wins']}/{row['comparisons']} "
                     f"saved configurations; {row['valid_segments']}/{row['possible_segments']} possible segments eligible; "
                     f"median reference agreement {row['median_reference_agreement']:.4f}. "
                     f"Real exceeds shuffled reference in {row['hddc_real_above_shuffle']}/{row['comparisons']} HDDC "
                     f"and {row['kmeans_real_above_shuffle']}/{row['comparisons']} KMeans comparisons.")
    controls_note = ("A higher model score alone is insufficient evidence that its PC1 follows label ordering. "
                     "When shuffled references score higher, the real ordering has not demonstrated an advantage "
                     "over this control. In particular, read any raw model wins alongside reference reliability "
                     "and the real-versus-shuffled results.")
    interpretation = " ".join(lines)
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Label-conditioned PC1 alignment: years and temperature</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#17232e}}
h1,h2{{line-height:1.2}}p{{max-width:1000px;line-height:1.6}}table{{border-collapse:collapse;font-size:12px}}
th,td{{padding:8px;border:1px solid #ccd3da;text-align:right}}th{{background:#eef2f6}}img{{max-width:100%}}
.table{{overflow:auto}}a{{color:#236e9f}}</style>
<h1>Label-conditioned PC1 alignment</h1><p>Computed 10 November 2026. All 32 matched saved-model comparisons;
layers 4 and 28, K=50/100, rank thresholds 0.0001/0.001/0.01/0.1. No retraining or checkpoint selection.</p>
<p>{html.escape(interpretation)}</p><p><strong>{controls_note}</strong></p>
<p>Primary score is squared covariance-PC1 alignment, weighted by assignments and equally averaged over valid
segments and the two fold directions. Higher means closer alignment to changes in label-conditioned means.
Unidentified PCs contribute zero while retaining their assignment weight. Conditional scores and excluded
weight are saved in <a href="summary.csv">the complete CSV</a>.</p>
<h2>Populations and reference reliability</h2><p>Years uses exact integer years and 4,004 independent test prompts.
Temperature uses 3,000 test prompts, grouped into the agreed 1°F bins [j,j+1), with zero as bin origin.
The temperature metric therefore measures differences between neighboring bin means. Exact temperature
labels each occur only once and cannot support this cross-fit definition. Both analyses require at least
two observations per endpoint per reference fold; split seed 42. Reference agreement is the squared inner
product of the fold-specific secants. Low agreement limits interpretation independently of model quality.</p>
<div class="table">{overview_html}</div><img src="reference_reliability.png" alt="Reference reliability distributions">
<h2>Primary model comparisons</h2><img src="alignment_scores.png" alt="Primary alignment for every configuration">
<p>KMeans PC1 may be unchanged across rank thresholds because the leading stored cluster PCA direction is
retained. The threshold configurations and neighboring segments are dependent observations; these are
descriptive comparisons, not independent replicates or significance tests.</p>
<h2>Negative control</h2><p>Only reference labels were shuffled, separately inside each fold (seeds 43 and 44).
Evaluation labels, assignments, and fold membership stayed fixed. Real and shuffled scores in this plot
use the same control-eligible segments. Points above the dashed line have greater alignment to the real
ordering. A single permutation is a diagnostic, not a calibrated permutation test.</p>
<img src="negative_controls.png" alt="Real alignment versus shuffled-reference alignment">
<h2>Paired segment differences</h2><p>Boxes summarize segment differences within each configuration; outliers
are hidden only in this overview. The complete segment values and interactive plots are linked below.</p>
<img src="paired_segment_differences.png" alt="Distributions of paired segment score differences">
<h2>All configurations</h2><div class="table">{table_html}</div>
<p>Per-comparison folders contain JSON, component and segment CSVs, prompt/template counts, and interactive
plots. <a href="comparison_plan.json">The frozen comparison plan</a> records source run paths and settings.
Scores concern a mean-activation proxy; they do not establish manifold coverage or recovered intrinsic dimension.</p>
</html>'''
    (root / "report.html").write_text(report)
    (root / "overview.txt").write_text("\n".join(lines) + "\n\n" + controls_note + "\n")
    print("\n".join(lines))
    print(f"Report: {root / 'report.html'}")


if __name__ == "__main__":
    main()
