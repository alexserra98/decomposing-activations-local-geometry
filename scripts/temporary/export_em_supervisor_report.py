#!/usr/bin/env python3
"""Export notebook-derived plots and tables for the October 2026 EM update.

Run from the repository root with .venv/bin/python. Metric selection and
aggregation execute the existing notebook cells; only figure layout is adapted.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "outputs/reports/em_toy_manifolds_2026-10-01"
FIXED_K = 500
FIXED_TAU = 0.1
PLOTS = DEST / "plots"
DATA = DEST / "data"
PLOTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
NOTEBOOK = REPO / "notebooks/toy_manifold_noise_sweep_results.ipynb"
CELLS = json.loads(NOTEBOOK.read_text())["cells"]
MAIN = REPO / "dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
ORIGINAL = REPO / "dalg-cache/deprecated_toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
DIAGNOSTIC = REPO / "dalg-cache/toy_manifold_models_hypersphere_swiss_roll_cylinder_10d_20Keach_noise_sweep_seed0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "axes.titlesize": 8, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.facecolor": "white"})


def execute_cell(index, ns, overrides=None, definitions=False):
    """Execute notebook code, replacing only explicit plot/data selections."""
    tree = ast.parse("".join(CELLS[index]["source"]))
    if definitions:
        tree.body = [node for node in tree.body if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign))]
        # Plotting cells contain figure assignments; stop before they occur.
        cut = next((i for i, node in enumerate(tree.body)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Tuple) for t in node.targets)), len(tree.body))
        tree.body = tree.body[:cut]
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and overrides and target.id in overrides:
                node.value = ast.parse(repr(overrides[target.id]), mode="eval").body
    exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK) + f":cell{index}", "exec"), ns)


def load_dataset(path):
    ns = {"__name__": "report_export"}
    source = ast.parse("".join(CELLS[2]["source"]))
    for node in source.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "DATA_DIR":
            node.value = ast.parse(f"Path({str(path)!r})", mode="eval").body
    exec(compile(ast.fix_missing_locations(source), str(NOTEBOOK) + ":cell2", "exec"), ns)
    return ns


def save(fig, name):
    fig.savefig(PLOTS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(PLOTS / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def compact_sweep(ax, title):
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel(r"Rank threshold $\tau$", fontsize=7, labelpad=1)
    ax.set_ylabel("K", fontsize=7, labelpad=1)
    ax.tick_params(axis="x", labelsize=5.7, rotation=40, pad=1)
    ax.tick_params(axis="y", labelsize=6.3, pad=2)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for text in ax.texts:
        text.set_fontsize(5.8)
    cb = ax.collections[0].colorbar
    if cb:
        cb.set_label("")
        cb.ax.tick_params(labelsize=5.5, length=2, pad=1)


def sweep(ns, name, metrics, noise="noiseless"):
    tables = ns["sweep_tables"][("em", noise)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7, 1.8), layout="constrained")
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        ns["plot_sweep_heatmap"](ax, metric, tables[metric], "em")
        title = {"Avg adjusted alignment": "Adjusted alignment",
                 "qₖ = r (%)": r"Intrinsic-rank matches: $q_k=r$ (%)",
                 "qₖ = m (%)": r"Embedding-rank matches: $q_k=m$ (%)"}[metric]
        compact_sweep(ax, title)
        tables[metric].to_csv(DATA / f"{name}_{ {'Avg adjusted alignment':'adjusted_alignment', 'qₖ = r (%)':'rank_r', 'qₖ = m (%)':'rank_m'}[metric]}.csv")
    save(fig, name)


def compact_tangent(ax, title, show_rows=True):
    # Keep the notebook renderer and scales while tightening its report layout.
    ax.set_title(title, fontsize=8, pad=29, fontweight="normal")
    ax.tick_params(axis="x", labelsize=5.8, pad=1)
    ax.tick_params(axis="y", labelsize=6.3, pad=2, labelleft=show_rows)
    for text in list(ax.texts):
        if text.get_transform() == ax.transAxes:
            text.remove()
        elif text.get_position()[1] == 1.11:
            text.set_fontsize(6.3)
            text.set_text({"noiseless": "No noise"}.get(text.get_text(), text.get_text()))
            text.set_fontweight("normal")
        else:
            text.set_fontsize(5.6)


def tangent(ns, name, titles):
    fig, axes = plt.subplots(1, len(titles), figsize=(7, 2.38), layout="constrained")
    for i, (ax, title) in enumerate(zip(np.atleast_1d(axes), titles)):
        ns["plot_tangent_heatmap"](ax, title, ns["tangent_tables"][title], show_row_labels=i == 0)
        compact_tangent(ax, title, i == 0)
    cb = fig.colorbar(np.atleast_1d(axes)[-1].collections[0], ax=list(np.atleast_1d(axes)), pad=0.008, fraction=0.019)
    cb.ax.tick_params(labelsize=5.7, length=2, pad=1)
    save(fig, name)


def allocation_panel(ns, ax, kind, show_rows=True):
    if kind == "rank":
        values = ns["rank_distance_table"]
        annotations = ns["mean_rank_table"]
        color_args = {"cmap": "viridis_r", "norm": PowerNorm(gamma=0.5, vmin=0, vmax=5, clip=True)}
        fmt, title = ".1f", r"Mean learned rank $q_k$"
    else:
        values = annotations = ns["component_count_table"]
        color_args = {"cmap": "viridis", "vmin": 0, "vmax": float(values.max().max())}
        fmt, title = ".0f", "Associated components"
    sns.heatmap(values, ax=ax, annot=annotations, fmt=fmt, cbar=False,
                linewidths=.5, linecolor="white", annot_kws={"fontsize": 5.6},
                xticklabels=[ns["model_labels"][m] for _, m in values.columns],
                yticklabels=ns["row_labels"], **color_args)
    ax.xaxis.tick_top()
    ax.set(xlabel="", ylabel="")
    ax.set_title(title, fontsize=8, pad=29)
    ax.tick_params(axis="both", labelrotation=0, length=0, pad=1, labelsize=5.8)
    ax.tick_params(axis="y", labelleft=show_rows, labelsize=6.3, pad=2)
    for group, noise in enumerate(ns["NOISE_LEVELS"]):
        ax.text(group * 3 + 1.5, 1.11, "No noise" if noise == "noiseless" else str(noise),
                transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=6.3)
        if group:
            ax.axvline(group * 3, color="white", linewidth=2)
    low_dim = ns["manifold_info"]["intrinsic_dim"].le(2).sum()
    ax.axhline(low_dim, color="white", linewidth=2)
    ax.set_facecolor("#e9edf2")
    for row, col in np.argwhere(values.isna().to_numpy()):
        ax.text(col + .5, row + .5, "—", ha="center", va="center", fontsize=6)
    return title


def allocations(ns, name, kinds):
    fig, axes = plt.subplots(1, len(kinds), figsize=(7, 2.38), layout="constrained")
    for i, (ax, kind) in enumerate(zip(np.atleast_1d(axes), kinds)):
        allocation_panel(ns, ax, kind, i == 0)
        cb = fig.colorbar(ax.collections[0], ax=ax, pad=.008, fraction=.026)
        cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
        if kind == "rank":
            cb.set_ticks([0, 1, 2, 3, 4, 5])
            cb.set_ticklabels(["0", "1", "2", "3", "4", "5+"])
    save(fig, name)


def latex_rank_table(ns):
    summary = ns["summary"].loc[lambda x: x.model_kind.isin(["kmeans", "hddc", "em"])].copy()
    adjusted = ns["manifolds"].groupby("run_id")["tangent_adjusted_alignment.subspace_overlap.mean"].mean()
    summary["adjusted_alignment"] = adjusted
    summary.to_csv(DATA / "main_selected_summary.csv")
    table = summary.loc[summary.model_kind.isin(["kmeans", "em"])]
    table.to_csv(DATA / "rank_table_km_em.csv")
    lines = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
             r"\toprule",
             r"Model & $\rho$ & Mean $\widehat q_k$ & $\widehat q_k=r$ (\%) & $\widehat q_k=m$ (\%) & Val. NLL & Live / total \\",
             r"\midrule"]
    previous_noise = None
    for _, row in table.iterrows():
        noise = row["noise_ratio"]
        if previous_noise is not None and noise != previous_noise:
            lines.append(r"\midrule")
        label = "No noise" if noise == "noiseless" else str(noise)
        model = {"kmeans": "K-means + PCA", "em": "HDDC (EM)"}[row.model_kind]
        nll = "---" if pd.isna(row["nll.validation"]) else f"{row['nll.validation']:.2f}"
        lines.append(f"{model} & {label} & {row['rank.mean_learned']:.2f} & "
                     f"{100*row['rank.exact_match']:.1f} & {100*row['ambient_rank.exact_match']:.1f} & "
                     f"{nll} & {int(row['components.live'])} / {int(row['K'])} " + r"\\")
        previous_noise = noise
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (DATA / "rank_table.tex").write_text("\n".join(lines) + "\n")
    standalone = [
        f"% Corrected ten-manifold benchmark: K={FIXED_K}, tau={FIXED_TAU}.",
        "% Rank statistics are unweighted means over the ten manifold types.",
        "% rho is the curvature-radius / noise-standard-deviation ratio.",
        r"\documentclass[10pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern,amsmath,booktabs}",
        r"\usepackage[paperwidth=210mm,paperheight=72mm,margin=4mm]{geometry}",
        r"\pagestyle{empty}", r"\setlength{\parindent}{0pt}",
        r"\setlength{\tabcolsep}{0pt}", r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{document}", *lines, r"\end{document}",
    ]
    (DEST / "rank_table_km_em.tex").write_text("\n".join(standalone) + "\n")
    print(summary[["model_kind", "noise_ratio", "rank.mean_learned", "rank.exact_match", "ambient_rank.exact_match", "adjusted_alignment", "nll.validation", "components.live"]].to_string())


def swiss_roll_comparison(ns):
    """Keep the historical comparison separate from corrected benchmark plots."""
    columns = ["run_id", "model_kind", "noise_ratio", "K", "surgery_threshold",
               "type_name", "intrinsic_dim", "embedding_dim", "rank.mean_learned",
               "rank.exact_match", "tangent_adjusted_alignment.subspace_overlap.mean"]
    frames = []
    for version, path, source in [("original", ORIGINAL, load_dataset(ORIGINAL)),
                                  ("corrected", MAIN, ns)]:
        rows = source["manifold_df"]
        rows = rows.loc[rows.type_name.eq("swiss_roll") & rows.K.eq(FIXED_K)
                        & rows.surgery_threshold.eq(FIXED_TAU)
                        & rows.model_kind.isin(["kmeans", "hddc", "em"]), columns].copy()
        assert len(rows) == 12
        rows.insert(0, "source_csv", str((path / "manifold_metrics.csv").relative_to(REPO)))
        rows.insert(0, "dataset_version", version)
        frames.append(rows)
    pd.concat(frames, ignore_index=True).to_csv(DATA / "swiss_roll_correction.csv", index=False)


def main():
    ns = load_dataset(MAIN)
    execute_cell(4, ns, {"SWEEP_MODELS": ["em"], "SWEEP_NOISE_LEVELS": ["noiseless", 1000, 100, 10],
                         "SWEEP_K_VALUES": None, "SWEEP_THRESHOLDS": None, "SWEEP_Q_CAPACITY": 32})
    execute_cell(5, ns, definitions=True)
    sweep_metrics = ["Avg adjusted alignment", "qₖ = r (%)"]
    sweep(ns, "main_sweep", sweep_metrics)
    for noise in [1000, 100, 10]:
        sweep(ns, f"main_sweep_noise{noise}", sweep_metrics, noise)
    ns["sweep_summary"].to_csv(DATA / "main_sweep_summary.csv", index=False)
    execute_cell(9, ns, {"K": FIXED_K, "HDDC_THRESHOLD": FIXED_TAU,
                         "EM_THRESHOLD": FIXED_TAU, "KMEANS_THRESHOLD": FIXED_TAU})
    execute_cell(10, ns)
    execute_cell(13, ns, {"NOISE_LEVELS": ["noiseless", 1000, 100, 10],
                          "MODELS": ["kmeans", "hddc", "em"]})
    execute_cell(14, ns, definitions=True)
    ns["model_labels"]["hddc"] = "Ad"
    short_names = {"segment": "Segment", "circle": "Circle", "helix": "Helix",
                   "helix_4d": "Helix 4D", "sphere": "Sphere", "torus": "Torus",
                   "swiss_roll": "Swiss roll", "cylinder": "Cylinder",
                   "hypersphere_10d": "Hypersphere", "product_torus_12d": "Torus 12D"}
    ns["row_labels"] = [f"{short_names[name]} ({row.intrinsic_dim},{row.embedding_dim})" for name, row in ns["manifold_info"].iterrows()]
    tangent(ns, "main_tangent", ["Adjusted alignment", "Tangent containment"])
    tangent(ns, "main_adjusted_alignment", ["Adjusted alignment"])
    tangent(ns, "main_containment", ["Tangent containment"])
    # Reuse the notebook's exact rank and component table expressions.
    for index, names in [(15, {"mean_rank_table", "rank_distance_table"}), (16, {"component_count_table"})]:
        tree = ast.parse("".join(CELLS[index]["source"]))
        tree.body = [node for node in tree.body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
        exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK) + f":cell{index}", "exec"), ns)
    totals = ns["tangent_df"].groupby(["run_id", "K"])["components.associated"].sum()
    assert (totals.to_numpy() == totals.index.get_level_values("K")).all()
    allocations(ns, "main_allocation", ["components", "rank"])
    allocations(ns, "main_components", ["components"])
    allocations(ns, "main_rank", ["rank"])
    for key in ["mean_rank_table", "rank_distance_table", "component_count_table"]:
        ns[key].to_csv(DATA / f"main_{key}.csv")
    for key, table in ns["tangent_tables"].items():
        table.to_csv(DATA / ("main_" + key.lower().replace(" ", "_") + ".csv"))
    ns["tangent_df"].to_csv(DATA / "main_selected_manifolds.csv", index=False)
    latex_rank_table(ns)
    swiss_roll_comparison(ns)
    high = load_dataset(DIAGNOSTIC)
    execute_cell(4, high, {"SWEEP_MODELS": ["em"], "SWEEP_NOISE_LEVELS": ["noiseless", 1000, 100, 10],
                           "SWEEP_K_VALUES": None, "SWEEP_THRESHOLDS": None, "SWEEP_Q_CAPACITY": 32})
    execute_cell(5, high, definitions=True)
    high_metrics = ["Avg adjusted alignment", "qₖ = r (%)", "qₖ = m (%)"]
    sweep(high, "highdim_sweep", high_metrics)
    for noise in [1000, 100, 10]:
        sweep(high, f"highdim_sweep_noise{noise}", high_metrics, noise)
    high["sweep_summary"].to_csv(DATA / "highdim_sweep_summary.csv", index=False)
    (DATA / "provenance.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for root in [MAIN, DIAGNOSTIC, ORIGINAL]
                          for path in [root / "general_metrics.csv", root / "manifold_metrics.csv"]},
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "swiss_roll_comparison_source": str(ORIGINAL.relative_to(REPO)),
        "notebook": str(NOTEBOOK.relative_to(REPO)),
        "notebook_cells_used": [2, 4, 5, 9, 10, 13, 14, 15, 16],
        "main_source": str(MAIN.relative_to(REPO)),
        "diagnostic_source": str(DIAGNOSTIC.relative_to(REPO)),
        "main_selection": {"K": FIXED_K, "tau": FIXED_TAU, "models": ["kmeans", "hddc_adam", "hddc_em"], "noise_ratios": ["noiseless", 1000, 100, 10]},
        "em_rank_capacity": {"main": 32, "diagnostic": 32},
        "metric_weighting": "Unweighted mean over manifold types, as in notebook",
        "sweep_figures": "EM; main_sweep and highdim_sweep are noiseless; named noise companions show other ratios",
    }, indent=2) + "\n")
    print("Plots exported to", PLOTS)


if __name__ == "__main__":
    main()
