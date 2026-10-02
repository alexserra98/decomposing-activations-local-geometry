#!/usr/bin/env python3
"""Export the torus6D appendix using the existing report's notebook selections."""
from __future__ import annotations

import hashlib
import json

from export_em_supervisor_report import (
    DATA, REPO, compact_sweep, execute_cell, load_dataset, plt, save, sns,
)


def main():
    datasets = {}
    for size in ("500K", "5M"):
        source = REPO / f"dalg-cache/toy_product_torus_6d_1each_D128_{size}_noiseless_seed0"
        ns = load_dataset(source)
        execute_cell(4, ns, {"SWEEP_MODELS": ["em"], "SWEEP_NOISE_LEVELS": ["noiseless"],
                             "SWEEP_Q_CAPACITY": 32})
        execute_cell(5, ns, definitions=True)
        assert ns["sweep_manifolds"]["type_name"].eq("product_torus_6d").all()
        assert ns["sweep_manifolds"]["intrinsic_dim"].eq(6).all()
        assert ns["sweep_manifolds"]["embedding_dim"].eq(12).all()
        datasets[size] = (source, ns)

    ks = sorted({k for _, ns in datasets.values() for k in ns["sweep_runs"]["K"]})
    thresholds = sorted({t for _, ns in datasets.values()
                         for t in ns["sweep_runs"]["sweep_value"]})
    provenance = {
        "export_script": "scripts/temporary/export_em_torus_appendix.py",
        "notebook": "notebooks/toy_manifold_noise_sweep_results.ipynb",
        "notebook_cells_used": [2, 4, 5],
        "selection": {"model": "hddc", "fit_method": "em", "noise": "noiseless", "q_capacity": 32},
        "K": ks, "tau": thresholds,
        "metrics": {"adjusted_alignment": "tangent_adjusted_alignment.subspace_overlap.mean",
                    "mean_rank": "rank.mean_learned"},
        "population": "Unweighted associated-component means; undefined tangent scores omitted",
        "color_limits": {"adjusted_alignment": [0, 1], "mean_rank": [0, 12]},
        "missing_cells": "Unavailable saved evaluation; no interpolation",
        "datasets": {},
    }
    for size, (source, ns) in datasets.items():
        name = f"torus6d_{size.lower()}_sweep"
        tables = ns["sweep_tables"][("em", "noiseless")]
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.75), layout="constrained")
        for ax, metric, suffix, title in zip(
            axes, ["Avg adjusted alignment", "Mean qₖ"],
            ["adjusted_alignment", "mean_rank"],
            ["Adjusted tangent alignment", r"Mean learned rank $\bar q$ ($r=6$, $m=12$)"],
        ):
            table = tables[metric].reindex(index=ks, columns=thresholds)
            table.to_csv(DATA / f"{name}_{suffix}.csv")
            if suffix == "mean_rank":
                sns.heatmap(table, ax=ax, annot=True, fmt=".2f", cmap="viridis",
                            vmin=0, vmax=12, linewidths=.5, linecolor="white",
                            xticklabels=[f"{v:g}" for v in thresholds], yticklabels=ks,
                            cbar_kws={"shrink": .9, "ticks": [0, 3, 6, 9, 12]})
                ax.set_facecolor("#e9edf2")
                for i, k in enumerate(ks):
                    for j, t in enumerate(thresholds):
                        if table.loc[k, t] != table.loc[k, t]:
                            ax.text(j + .5, i + .5, "—", ha="center", va="center", color="#626b78")
                ax.tick_params(axis="both", length=0, labelrotation=0)
            else:
                ns["plot_sweep_heatmap"](ax, metric, table, "em")
            compact_sweep(ax, title)
        save(fig, name)
        ns["sweep_summary"].to_csv(DATA / f"{name}_summary.csv", index=False)
        ns["sweep_manifolds"].to_csv(DATA / f"{name}_selected_manifolds.csv", index=False)
        provenance["datasets"][size] = {
            "source": str(source.relative_to(REPO)),
            "selected_runs": len(ns["sweep_runs"]),
            "samples": ns["sweep_runs"]["dataset.selected_rows"].unique().tolist(),
            "source_sha256": {f: hashlib.sha256((source / f).read_bytes()).hexdigest()
                              for f in ["general_metrics.csv", "manifold_metrics.csv"]},
        }
        print(f"{size}: exported {len(ns['sweep_runs'])} runs on a {len(ks)} x {len(thresholds)} grid")
    (DATA / "torus6d_appendix_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
