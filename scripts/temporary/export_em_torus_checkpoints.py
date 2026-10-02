#!/usr/bin/env python3
"""Export likelihood and saved-checkpoint geometry for the torus6D appendix.

Run with PYTHONPATH=src .venv/bin/python from the repository root.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import torch

from dalg.evaluation.toy_manifold_metrics import _component_metrics
from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc
from export_em_supervisor_report import DATA, REPO, compact_sweep, plt, save, sns

EARLY_ITERATION = 2
ROOTS = {size: REPO / f"dalg-cache/toy_product_torus_6d_1each_D128_{size}_noiseless_seed0"
         for size in ("500K", "5M")}


def heatmap(ax, table, title, *, low, high, cmap="viridis", decimals=2, stopped=None):
    annotations = table.map(lambda x: f"{x:.{decimals}f}")
    if stopped is not None:
        annotations = annotations + stopped.map(lambda x: "*" if x else "")
    sns.heatmap(table, ax=ax, annot=annotations, fmt="", cmap=cmap,
                vmin=low, vmax=high, linewidths=.5, linecolor="white",
                xticklabels=[f"{v:g}" for v in table.columns], yticklabels=table.index,
                cbar_kws={"shrink": .9})
    ax.tick_params(axis="both", length=0, labelrotation=0)
    compact_sweep(ax, title)


def main():
    torch.set_num_threads(4)
    frames = {size: pd.read_csv(root / "general_metrics.csv") for size, root in ROOTS.items()}
    for frame in frames.values():
        assert len(frame) == 45 and not frame.duplicated(["K", "surgery_threshold"]).any()
        assert frame["config.training.arguments.fit_method"].eq("em").all()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.75), layout="constrained")
    for ax, (size, frame) in zip(axes, frames.items()):
        table = frame.pivot(index="K", columns="surgery_threshold", values="nll.validation").sort_index().sort_index(axis=1)
        table.to_csv(DATA / f"torus6d_{size.lower()}_validation_nll.csv")
        heatmap(ax, table, f"{size} samples: validation NLL", low=-570, high=-180,
                cmap="viridis_r", decimals=1)
    save(fig, "torus6d_likelihood_comparison")

    # Keep the selected-checkpoint comparison separate from the terminal snapshots.
    comparison = []
    for size, frame in frames.items():
        for _, run in frame.sort_values(["K", "surgery_threshold"]).iterrows():
            directory = ROOTS[size] / run.source_path
            directory = directory.parent
            history = json.loads((directory / "em_history.json").read_text())
            comparison.append({
                "samples": size, "run_id": run.run_id, "K": run.K, "tau": run.surgery_threshold,
                "validation_nll": run["nll.validation"],
                "adjusted_alignment": run["tangent_adjusted_alignment.subspace_overlap.mean"],
                "mean_rank": run["rank.mean_learned"],
                "selected_iteration": history["best_iteration"],
                "last_iteration": history["history"][-1]["iteration"],
                "stop_reason": history["stop_reason"],
            })
    pd.DataFrame(comparison).to_csv(DATA / "torus6d_likelihood_geometry_comparison.csv", index=False)

    metadata_path = REPO / "dalg-cache/assets/toy_product_torus_6d_1each_D128_5M_noiseless_seed0/manifold_metadata.pt"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    rows = []
    for _, run in frames["5M"].sort_values(["K", "surgery_threshold"]).iterrows():
        directory = (ROOTS["5M"] / run.source_path).parent
        history = json.loads((directory / "em_history.json").read_text())
        assert history["stop_reason"] in ("converged", "patience")
        last_iteration = history["history"][-1]["iteration"]
        for stage, iteration in [("early", EARLY_ITERATION), ("terminal", last_iteration)]:
            path = directory / f"epoch_{iteration:04d}/mfa_model.pt"
            model = load_mfa_hddc(path, map_location="cpu").eval()
            with torch.no_grad():
                metrics = _component_metrics(
                    model, metadata, rank_threshold=1., max_mean_to_manifold_distance=None,
                    relative_boundary_eigengap_threshold=1e-6,
                )
            associated = metrics.associations.associated
            valid = associated & metrics.adjusted_alignment_defined
            assert associated.all() and valid.any()
            alignment = float(metrics.adjusted_alignment_overlap[valid].mean())
            rank = float(metrics.effective_ranks[associated].float().mean())
            record = next(h for h in history["history"] if h["iteration"] == iteration)
            assert abs(rank - record["rank_mean"]) < 2e-6
            # Terminal snapshots that were selected must reproduce the published geometry.
            if iteration == history["best_iteration"]:
                assert abs(alignment - run["tangent_adjusted_alignment.subspace_overlap.mean"]) < 1e-6
                assert abs(rank - run["rank.mean_learned"]) < 2e-6
            rows.append({
                "run_id": run.run_id, "K": int(run.K), "tau": float(run.surgery_threshold),
                "stage": stage, "iteration": iteration, "stop_reason": history["stop_reason"],
                "selected_iteration": history["best_iteration"],
                "adjusted_alignment": alignment, "mean_rank": rank,
                "valid_components": int(valid.sum()), "associated_components": int(associated.sum()),
                "validation_nll_history": record["val_nll"], "train_nll_history": record["train_nll"],
                "checkpoint": str(path.relative_to(REPO)),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        # Save progress as a small report artifact; training directories are read-only here.
        pd.DataFrame(rows).to_csv(DATA / "torus6d_5m_checkpoint_geometry.csv", index=False)
        print(f"K={run.K}, tau={run.surgery_threshold}: early {rows[-2]['adjusted_alignment']:.3f}, "
              f"terminal {rows[-1]['adjusted_alignment']:.3f} ({history['stop_reason']}, iteration {last_iteration})", flush=True)

    results = pd.DataFrame(rows)
    for stage in ("early", "terminal"):
        subset = results.loc[results.stage.eq(stage)]
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.75), layout="constrained")
        stopped = subset.pivot(index="K", columns="tau", values="stop_reason").eq("patience") if stage == "terminal" else None
        for ax, metric, title, high in zip(axes, ["adjusted_alignment", "mean_rank"],
            ["Adjusted tangent alignment", r"Mean learned rank $\bar q$ ($r=6$, $m=12$)"], [1, 12]):
            table = subset.pivot(index="K", columns="tau", values=metric).sort_index().sort_index(axis=1)
            table.to_csv(DATA / f"torus6d_5m_{stage}_{metric}.csv")
            heatmap(ax, table, title, low=0, high=high, stopped=stopped)
        save(fig, f"torus6d_5m_{stage}_geometry")
    results.loc[results.stage.eq("terminal")].pivot(index="K", columns="tau", values="iteration").to_csv(DATA / "torus6d_5m_terminal_iterations.csv")
    (DATA / "torus6d_checkpoint_provenance.json").write_text(json.dumps({
        "script": "scripts/temporary/export_em_torus_checkpoints.py",
        "early_iteration": EARLY_ITERATION,
        "terminal_checkpoint": "Last evaluated iteration: 38 converged; 7 validation-patience stops marked *",
        "geometry": "dalg.evaluation.toy_manifold_metrics._component_metrics; unweighted associated components; undefined tangent scores omitted",
        "metadata": str(metadata_path.relative_to(REPO)),
        "likelihood_heatmaps": "nll.validation from general_metrics.csv; selected checkpoint; mean negative log density per sample",
        "checkpoint_likelihood": "em_history.json float64 EM scorer; slightly different numerics from the saved-run evaluator",
        "rank_and_alignment_limits": [[0, 12], [0, 1]],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
