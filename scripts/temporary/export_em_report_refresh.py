"""Recreate the October EM report from saved metrics, without changing runs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dalg.analysis.aggregate_metrics import _report_rows


REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks/toy_manifold_noise_sweep_results.ipynb"
ROOTS = {
    "main": "toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0",
    "torus6d_5m": "toy_product_torus_6d_1each_D128_5M_noiseless_seed0",
    "torus6d_500k": "toy_product_torus_6d_1each_D128_500K_noiseless_seed0",
    "highdim": "toy_manifold_models_hypersphere_swiss_roll_cylinder_10d_20Keach_noise_sweep_seed0",
}
NOISES = ["noiseless", 1000, 100, 10]
MODELS = ["kmeans", "hddc", "em"]
NAMES = {"kmeans": "KM", "mfa": "MFA", "hddc": "Adam", "em": "EM"}
ALIGN = "tangent_adjusted_alignment.subspace_overlap.mean"
RANK = "rank.mean_learned"
EXACT = "rank.exact_match"
AMBIENT = "ambient_rank.exact_match"
CURVE = "heldout_distribution_coverage.coverage_curve"
MACRO = [RANK, EXACT, AMBIENT, ALIGN,
         "tangent_alignment.subspace_overlap.mean", "tangent_containment.subspace_overlap.mean"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def notebook_context():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    context = {"pd": pd, "np": np, "plt": plt, "sns": sns,
               "TANGENT_SCORE": "subspace_overlap"}
    wanted = {"noise_key", "plot_sweep_heatmap", "plot_tangent_heatmap",
              "EM_EXPERIMENT_BY_NOISE"}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            names = {node.name} if isinstance(node, ast.FunctionDef) else set()
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), context)
    return context


def collect(key, output, context, provenance):
    """Use the repository row parser on current top-level run evaluations."""
    root = REPO / "dalg-cache" / ROOTS[key]
    pattern = "*/*/*/metrics.json" if key == "highdim" else "*/models/*/*/metrics.json"
    paths = sorted(root.glob(pattern))
    general_rows, manifold_rows, curve_rows = [], [], []
    excluded, superseded = [], []
    for index, path in enumerate(paths):
        spec_path = path.with_name("run_spec.json")
        spec = json.loads(spec_path.read_text())
        args = spec["training"]["arguments"]
        model = spec["training"]["model_kind"]
        if model == "hddc" and args.get("fit_method") == "em":
            model = "em"
        noise = "noiseless" if path.relative_to(root).parts[0] == "noiseless" else int(path.relative_to(root).parts[0].split("_")[-1])
        experiment = spec["identity"]["experiment"]
        if key == "main" and model == "em" and experiment != context["EM_EXPERIMENT_BY_NOISE"][noise]:
            excluded.append(spec["run_id"])
            continue
        if key == "highdim" and model in {"em", "kmeans"} and spec["evaluation"].get("heldout_distribution_coverage") is not True:
            superseded.append(spec["run_id"])
            continue
        general, manifolds = _report_rows(path, root)
        curve = general.pop(CURVE, None)
        for row in [general, *manifolds]:
            row.update(model_kind=model, noise_ratio=noise)
        selected = args["K"] == 2000 and (model == "mfa" or np.isclose(args.get("surgery_threshold", -1), .05))
        if curve is not None and selected:
            frame = pd.DataFrame(json.loads(curve))
            assert frame.radius.is_monotonic_increasing and frame.fraction.is_monotonic_increasing
            assert np.isclose(frame.fraction.iloc[-1], 1)
            frame.insert(0, "run_id", general["run_id"])
            curve_rows.append(frame)
        general_rows.append(general)
        manifold_rows.extend(manifolds)
        provenance["sources"].append({"path": str(path.relative_to(REPO)),
                                      "sha256": sha256(path),
                                      "spec_sha256": sha256(spec_path)})
        if index % 100 == 0:
            print(key, index, "/", len(paths), flush=True)
    general, manifolds = pd.DataFrame(general_rows), pd.DataFrame(manifold_rows)
    assert general.run_id.is_unique
    assert not manifolds.duplicated(["run_id", "type_name"]).any()
    count = manifolds.type_name.nunique()
    assert manifolds.groupby("run_id").size().reindex(general.run_id).eq(count).all()
    summary = general.merge(manifolds.groupby("run_id")[MACRO].mean(), on="run_id", suffixes=("_global", ""), validate="one_to_one")
    summary.to_csv(output / "data" / f"{key}_summary.csv", index=False)
    manifolds.to_csv(output / "data" / f"{key}_manifolds.csv", index=False)
    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    if not curves.empty:
        curves.to_csv(output / "data" / f"{key}_coverage_curves.csv.gz", index=False)
    provenance["datasets"][key] = {"root": str(root.relative_to(REPO)), "available_evaluations": len(paths),
                                   "selected_evaluations": len(general), "excluded_variant_runs": excluded,
                                   "excluded_superseded_runs": superseded,
                                   "manifold_types": count}
    print(key, "collected", len(general), "runs;", count, "manifolds", flush=True)
    return summary, manifolds, curves


def savefig(fig, output, name):
    for suffix in ("pdf", "png"):
        fig.savefig(output / "plots" / f"{name}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmap(ax, table, title, *, limits=(0, 1), fmt=".2f", cmap="viridis", annot=None):
    sns.heatmap(table, ax=ax, cmap=cmap, vmin=limits[0], vmax=limits[1],
                annot=True if annot is None else annot, fmt=fmt, linewidths=.5,
                annot_kws={"fontsize": 10}, cbar_kws={"shrink": .85},
                xticklabels=[f"{v:g}" for v in table.columns], yticklabels=table.index)
    ax.set_facecolor("#e9edf2")
    for row, col in np.argwhere(table.isna().to_numpy()):
        ax.text(col + .5, row + .5, "—", ha="center", va="center", fontsize=8)
    ax.set(title=title, xlabel=r"Cattell threshold $\tau$", ylabel="K")
    ax.tick_params(axis="both", labelrotation=0, labelsize=8, length=0)


def sweep(summary, output, name, metrics, title, *, model="em", noise="noiseless"):
    rows = summary.loc[summary.model_kind.eq(model) & summary.noise_ratio.eq(noise)]
    assert not rows.duplicated(["K", "surgery_threshold"]).any()
    ks = sorted(summary.K.unique())
    taus = sorted(summary.loc[summary.model_kind.eq(model), "surgery_threshold"].dropna().unique())
    panel_width = 4.8 if len(metrics) == 3 else 5.4
    fig, axes = plt.subplots(1, len(metrics), figsize=(panel_width * len(metrics), 3.3), layout="constrained")
    for ax, (column, label, bounds) in zip(np.atleast_1d(axes), metrics):
        table = rows.pivot(index="K", columns="surgery_threshold", values=column).reindex(index=ks, columns=taus)
        table.to_csv(output / "data" / f"{name}_{column.replace('.', '_')}.csv")
        heatmap(ax, table, label, limits=bounds)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    savefig(fig, output, name)


def fixed_selection(summary):
    return summary.loc[summary.K.eq(2000) & (summary.model_kind.eq("mfa") | np.isclose(summary.surgery_threshold, .05))].copy()


def main_figures(data, output, context):
    summary, manifolds, curves = data["main"]
    selected = fixed_selection(summary)
    assert len(selected) == 16 and not selected.duplicated(["noise_ratio", "model_kind"]).any()
    selected.to_csv(output / "data" / "main_selected_runs.csv", index=False)
    chosen = manifolds.loc[manifolds.run_id.isin(selected.run_id)]
    chosen.to_csv(output / "data" / "main_selected_manifolds.csv", index=False)
    for key in ["main", "highdim"]:
        metrics = [(ALIGN, "Adjusted tangent alignment", (0, 1)), (EXACT, "Exact intrinsic-rank fraction", (0, 1))]
        if key == "highdim":
            metrics.append((AMBIENT, "Exact native-rank fraction", (0, 1)))
        for noise in NOISES:
            suffix = "" if noise == "noiseless" else f"_noise{noise}"
            sweep(data[key][0], output, f"{key}_sweep{suffix}", metrics,
                  f"{'Ten manifolds' if key == 'main' else 'Three 10D manifolds'}: EM, noise ratio = {noise}", noise=noise)
    info = manifolds[["type_name", "intrinsic_dim", "embedding_dim"]].drop_duplicates().sort_values(["intrinsic_dim", "embedding_dim"], kind="stable").set_index("type_name")
    context.update(NOISE_LEVELS=NOISES, MODELS=MODELS, manifold_info=info, model_labels=NAMES,
                   row_labels=[f"{name.replace('_', ' ').capitalize()} ({int(row.intrinsic_dim)},{int(row.embedding_dim)})" for name, row in info.iterrows()])
    columns = pd.MultiIndex.from_product([NOISES, MODELS], names=["noise_ratio", "model_kind"])
    def panel(column):
        return chosen.loc[chosen.model_kind.isin(MODELS)].pivot(index="type_name", columns=["noise_ratio", "model_kind"], values=column).reindex(index=info.index, columns=columns)
    for name, column, title in [("main_adjusted_alignment", ALIGN, "Adjusted tangent alignment"),
                                ("main_alignment", "tangent_alignment.subspace_overlap.mean", "Leading-PC alignment"),
                                ("main_containment", "tangent_containment.subspace_overlap.mean", "Best retained-PC containment")]:
        fig, ax = plt.subplots(figsize=(9.8, 4.6), layout="constrained")
        context["plot_tangent_heatmap"](ax, title, panel(column))
        savefig(fig, output, name)
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 8.8), layout="constrained")
    for ax, column, title in zip(axes, [ALIGN, "tangent_containment.subspace_overlap.mean"], ["Adjusted alignment", "Tangent containment"]):
        context["plot_tangent_heatmap"](ax, title, panel(column))
    savefig(fig, output, "main_tangent")
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 8.2), layout="constrained")
    for ax, column, title in zip(axes, ["components.associated", RANK], ["Associated components", "Mean learned rank"]):
        table = panel(column)
        if column == RANK:
            values = table.sub(info.intrinsic_dim, axis=0).abs()
            sns.heatmap(values, ax=ax, annot=table, fmt=".1f", cmap="viridis_r", vmin=0, vmax=5,
                        cbar=False, linewidths=.5, annot_kws={"fontsize":9})
        else:
            assert chosen.groupby("run_id")[column].sum().eq(2000).all()
            sns.heatmap(table, ax=ax, annot=True, fmt=".0f", cmap="viridis", cbar=False, linewidths=.5,
                        annot_kws={"fontsize":9})
        ax.set_xticklabels([NAMES[m] for _,m in columns], rotation=0)
        ax.set_yticklabels(context["row_labels"], rotation=0)
        ax.set(xlabel="", ylabel="")
        ax.xaxis.tick_top();ax.set_title(title, pad=45)
        for group, noise in enumerate(NOISES):
            ax.text(group*3+1.5, 1.08, str(noise), transform=ax.get_xaxis_transform(), ha="center", fontweight="bold")
            if group: ax.axvline(group*3, color="white", lw=3)
    savefig(fig, output, "main_allocation")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, layout="constrained")
    max_radius = curves.radius.max()*1.02
    colors = dict(zip(NAMES, plt.get_cmap("tab10").colors))
    for ax, noise in zip(axes.flat, NOISES):
        for model in NAMES:
            run = selected.loc[selected.noise_ratio.eq(noise) & selected.model_kind.eq(model)].iloc[0]
            curve = curves.loc[curves.run_id.eq(run.run_id)]
            assert not curve.empty
            ax.step(np.r_[0, curve.radius, max_radius], np.r_[0, curve.fraction, 1], where="post", label=NAMES[model], color=colors[model], lw=1.5)
        ax.set(title=f"Noise ratio: {noise}", xlabel="Euclidean radius", ylabel="Held-out coverage", xlim=(0,max_radius), ylim=(0,1.02))
        ax.grid(alpha=.25);ax.legend(loc="lower right",fontsize=9)
    fig.suptitle(r"Independent test coverage: $K=2000$, $\tau=0.05$", fontweight="bold")
    savefig(fig, output, "main_coverage")
    return selected, chosen


def torus_figures(data, output):
    metrics = [(ALIGN, "Adjusted tangent alignment", (0,1)), (RANK, "Mean learned rank (target: 6)", (0,12))]
    for key in ["torus6d_500k", "torus6d_5m"]:
        sweep(data[key][0], output, key + "_sweep", metrics, f"Isolated noiseless 6D torus: {key.split('_')[-1].upper()} samples, EM")
    sweep(data["torus6d_5m"][0], output, "torus6d_5m_kmeans_sweep", metrics, "Isolated noiseless 6D torus: 5M samples, K-means + PCA", model="kmeans")
    fig, axes = plt.subplots(1, 2, figsize=(10.8,3.3), layout="constrained")
    frames=[data[k][0].loc[data[k][0].model_kind.eq("em")] for k in ["torus6d_500k","torus6d_5m"]]
    bounds=(min(f["nll.validation"].min() for f in frames),max(f["nll.validation"].max() for f in frames))
    for ax, frame, label in zip(axes, frames, ["500K samples", "5M samples"]):
        table=frame.pivot(index="K",columns="surgery_threshold",values="nll.validation").sort_index().sort_index(axis=1)
        heatmap(ax,table,label,limits=bounds,fmt=".0f",cmap="viridis_r")
        table.to_csv(output/"data"/f"torus6d_{label.split()[0].lower()}_validation_nll.csv")
    fig.suptitle("Validation NLL: selected EM checkpoints (lower is better)",fontweight="bold")
    savefig(fig,output,"torus6d_likelihood_comparison")


def checkpoint_geometry(output):
    """Re-evaluate only parameter geometry, using the existing component scorer."""
    import torch
    from dalg.evaluation.toy_manifold_metrics import _component_metrics
    from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc

    torch.set_num_threads(2)
    root = REPO / "dalg-cache" / ROOTS["torus6d_5m"]
    paths = sorted(root.glob("noiseless/models/*_em/*/metrics.json"))
    first = json.loads(paths[0].with_name("run_spec.json").read_text())
    dataset = Path(first["dataset"]["shard_dir"])
    dataset_config = json.loads((dataset/"config.json").read_text())
    metadata_path = dataset/dataset_config["manifold_metadata"]
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True, mmap=True)
    rows = []
    for index, path in enumerate(paths):
        run_dir=path.parent
        spec=json.loads(path.with_name("run_spec.json").read_text())
        history_path=run_dir/"em_history.json"
        history=json.loads(history_path.read_text())
        records={r["iteration"]:r for r in history["history"]}
        assert history["stop_reason"] is not None
        for stage, iteration in [("early",2),("terminal",max(records))]:
            checkpoint=run_dir/f"epoch_{iteration:04d}"/"mfa_model.pt"
            model=load_mfa_hddc(checkpoint,map_location="cpu",device=torch.device("cpu"))
            values=_component_metrics(model,metadata,rank_threshold=1.,max_mean_to_manifold_distance=None,
                                      relative_boundary_eigengap_threshold=1e-6)
            valid=values.adjusted_alignment_defined
            associated=values.associations.associated
            rows.append({"run_id":spec["run_id"],"K":spec["training"]["arguments"]["K"],
                         "surgery_threshold":spec["training"]["arguments"]["surgery_threshold"],
                         "stage":stage,"iteration":iteration,"stop_reason":history["stop_reason"],
                         "selected_iteration":history["best_iteration"],
                         "adjusted_alignment":float(values.adjusted_alignment_overlap[valid].mean()),
                         "mean_rank":float(values.effective_ranks[associated].double().mean()),
                         "valid_components":int(valid.sum()),"associated_components":int(associated.sum()),
                         "validation_nll_history":records[iteration]["val_nll"],
                         "train_nll_history":records[iteration]["train_nll"],
                         "checkpoint":str(checkpoint.relative_to(REPO)),"checkpoint_sha256":sha256(checkpoint),
                         "history_sha256":sha256(history_path)})
            del model, values
        if index%5==0:print("Checkpoint geometry",index+1,"/",len(paths),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/"data/torus6d_5m_checkpoint_geometry.csv",index=False)
    (output/"data/checkpoint_provenance.json").write_text(json.dumps({
        "metadata":str(metadata_path),"metadata_sha256":sha256(metadata_path),
        "scorer":"dalg.evaluation.toy_manifold_metrics._component_metrics",
        "scorer_sha256":sha256(REPO/"src/dalg/evaluation/toy_manifold_metrics.py"),
        "early_iteration":2,"runs":len(paths),"evaluated_checkpoints":len(frame),
        "population":"associated components, independent of assignment-live status; undefined tangent scores omitted",
    },indent=2)+"\n")
    return frame


def checkpoint_figures(frame, output):
    for stage in ["early","terminal"]:
        rows=frame.loc[frame.stage.eq(stage)]
        fig,axes=plt.subplots(1,2,figsize=(10.8,3.3),layout="constrained")
        for ax,column,title,bounds in zip(axes,["adjusted_alignment","mean_rank"],
                ["Adjusted tangent alignment","Mean learned rank (target: 6)"],[(0,1),(0,12)]):
            table=rows.pivot(index="K",columns="surgery_threshold",values=column).sort_index().sort_index(axis=1)
            annotations=table.map(lambda v:f"{v:.2f}" if pd.notna(v) else "")
            if stage=="terminal":
                for row in rows.loc[rows.stop_reason.eq("patience")].itertuples():
                    annotations.loc[row.K,row.surgery_threshold]+="*"
            heatmap(ax,table,title,limits=bounds,fmt="",annot=annotations)
            table.to_csv(output/"data"/f"torus6d_5m_{stage}_{column}.csv")
        fig.suptitle("5M torus: "+("iteration 2" if stage=="early" else "terminal checkpoint (* validation patience)"),fontweight="bold")
        savefig(fig,output,f"torus6d_5m_{stage}_geometry")


def write_report(data, output, checkpoints):
    selected=fixed_selection(data["main"][0])
    order={n:i for i,n in enumerate(NOISES)}
    selected=selected.assign(noise_order=selected.noise_ratio.map(order),
                             model_order=selected.model_kind.map({m:i for i,m in enumerate(NAMES)}))
    selected=selected.sort_values(["noise_order","model_order"])
    def row(model,noise):
        return selected.loc[selected.model_kind.eq(model)&selected.noise_ratio.eq(noise)].iloc[0]
    def number(value,digits=3):
        return "---" if pd.isna(value) else f"{value:.{digits}f}"
    table=[r"\begin{tabular}{llrrrrrr}\toprule",
           r"Noise & Model & Mean $q_k$ & $q_k=r$ (\%) & $q_k=m$ (\%) & Adj. align. & Val. NLL & Live / $K$\\\midrule"]
    coverage=[r"\begin{tabular}{llrrrr}\toprule",
              r"Noise & Model & Mean distance & Median & $r_{95}$ & Training-live\\\midrule"]
    for _,r in selected.iterrows():
        table.append(f"{r.noise_ratio} & {NAMES[r.model_kind]} & {r[RANK]:.2f} & {100*r[EXACT]:.1f} & {100*r[AMBIENT]:.1f} & {r[ALIGN]:.3f} & {number(r['nll.validation'],2)} & {int(r['components.live'])}/2000"+r"\\")
        c="heldout_distribution_coverage."
        coverage.append(f"{r.noise_ratio} & {NAMES[r.model_kind]} & {r[c+'mean']:.3f} & {r[c+'median']:.3f} & {r[c+'r95']:.3f} & {int(r[c+'components_live'])}"+r"\\")
        if r.model_kind=="em" and r.noise_ratio!=10:
            table.append(r"\addlinespace[2pt]");coverage.append(r"\addlinespace[2pt]")
    table.append(r"\bottomrule\end{tabular}");coverage.append(r"\bottomrule\end{tabular}")
    (output/"data/rank_table.tex").write_text("\n".join(table)+"\n")
    (output/"data/coverage_table.tex").write_text("\n".join(coverage)+"\n")
    coverage_columns=["run_id","noise_ratio","model_kind"]+[c for c in selected if c.startswith("heldout_distribution_coverage.")]
    selected[coverage_columns].to_csv(output/"data/main_selected_coverage.csv",index=False)
    for noise in NOISES:
        group=selected.loc[selected.noise_ratio.eq(noise)]
        assert group["heldout_distribution_coverage.source.shard_dir"].nunique()==1
        assert group["heldout_distribution_coverage.reference_points"].eq(30000).all()
        assert group["heldout_distribution_coverage.liveness_split"].eq("train").all()
    torus={}
    for key in ["torus6d_500k","torus6d_5m"]:
        frame=data[key][0]
        torus[key]=frame.loc[frame.model_kind.eq("em")&frame.K.eq(2000)&np.isclose(frame.surgery_threshold,.05)].iloc[0]
    comparison=pd.DataFrame([torus[k] for k in torus]).assign(dataset=list(torus))
    comparison.to_csv(output/"data/torus6d_selected_comparison.csv",index=False)
    example=checkpoints.loc[checkpoints.K.eq(2000)&np.isclose(checkpoints.surgery_threshold,.05)].set_index("stage")
    assert len(example)==2
    early,terminal=example.loc["early"],example.loc["terminal"]
    terminal_rows=checkpoints.loc[checkpoints.stage.eq("terminal")]
    stop_counts=terminal_rows.stop_reason.value_counts()
    high=data["highdim"][0]
    high=high.loc[high.model_kind.eq("em")&high.noise_ratio.eq("noiseless")&np.isclose(high.surgery_threshold,.005)]
    def percent_range(series):
        low,high=round(100*series.min(),1),round(100*series.max(),1)
        return (f"{low:.1f}" if low==high else f"{low:.1f}--{high:.1f}")+r"\%"
    main_manifolds=data["main"][1]
    def manifold(name,noise):
        rid=row("em",noise).run_id
        return main_manifolds.loc[main_manifolds.run_id.eq(rid)&main_manifolds.type_name.eq(name)].iloc[0]
    swiss0,swiss10=manifold("swiss_roll","noiseless"),manifold("swiss_roll",10)
    torus0,torus10=manifold("product_torus_12d","noiseless"),manifold("product_torus_12d",10)
    values={
        "main_result":f"At $K=2000$, $\\tau=0.05$, EM's adjusted tangent alignment is \\textbf{{{row('em','noiseless')[ALIGN]:.3f}}} versus \\textbf{{{row('kmeans','noiseless')[ALIGN]:.3f}}} for K-means without noise (Adam: {row('hddc','noiseless')[ALIGN]:.3f}). At $\\rho=10$, EM reaches \\textbf{{{row('em',10)[ALIGN]:.3f}}} versus \\textbf{{{row('kmeans',10)[ALIGN]:.3f}}} for K-means. EM improves on Adam at all four noise levels; K-means retains the higher score in the three lower-noise conditions.",
        "swiss":f"The current benchmark uses the corrected Swiss-roll geometry. At the selected settings, EM recovers the intrinsic rank for {100*swiss0[EXACT]:.1f}\\% of associated components without noise and {100*swiss10[EXACT]:.1f}\\% at $\\rho=10$; adjusted alignment is {swiss0[ALIGN]:.3f} and {swiss10[ALIGN]:.3f}, respectively. These are current-setting results; the earlier report's before/after comparison used different settings.",
        "likelihood":f"At $\\rho=10$, Adam and EM have similar validation NLL ({row('hddc',10)['nll.validation']:.2f} and {row('em',10)['nll.validation']:.2f}), but adjusted alignment is {row('hddc',10)[ALIGN]:.3f} and {row('em',10)[ALIGN]:.3f}. The notebook's active-BIC adds a reward for training-live components, so changing $K$ also changes the possible reward. Neither likelihood nor this score alone establishes tangent or rank recovery.",
        "high_rank":f"On the 12D product torus in the mixed benchmark, EM's noiseless mean rank is {torus0[RANK]:.2f} and adjusted alignment is {torus0[ALIGN]:.3f}; at $\\rho=10$, mean rank is {torus10[RANK]:.2f} and alignment is {torus10[ALIGN]:.3f} ($r=12$, $m=24$). Both underestimation and overestimation remain possible.",
        "highdim":f"The separate hypersphere, Swiss-roll and cylinder benchmark uses 20,000 samples per manifold, with $r=10$, $m=11$. At $\\tau=0.005$, noiseless EM adjusted alignment ranges from {high[ALIGN].min():.3f} to {high[ALIGN].max():.3f} across the available $K$ values, while native-rank recovery is {percent_range(high[AMBIENT])} and intrinsic-rank recovery is {percent_range(high[EXACT])}.",
        "coverage":f"K-means has the smaller $r_{{95}}$ in all four conditions at these settings. Without noise, its $r_{{95}}$ is \\textbf{{{row('kmeans','noiseless')['heldout_distribution_coverage.r95']:.3f}}} versus \\textbf{{{row('em','noiseless')['heldout_distribution_coverage.r95']:.3f}}} for EM. At $\\rho=10$, the values are \\textbf{{{row('kmeans',10)['heldout_distribution_coverage.r95']:.3f}}} and \\textbf{{{row('em',10)['heldout_distribution_coverage.r95']:.3f}}}. Thus EM's better tangent recovery at high noise does not imply better coverage by Euclidean centroid distances.",
        "torus_comparison":f"At $K=2000$, $\\tau=0.05$, the selected 500K and 5M EM models have adjusted alignment \\textbf{{{torus['torus6d_500k'][ALIGN]:.3f}}} and \\textbf{{{torus['torus6d_5m'][ALIGN]:.3f}}}, mean ranks {torus['torus6d_500k'][RANK]:.2f} and {torus['torus6d_5m'][RANK]:.2f}, and validation NLL {torus['torus6d_500k']['nll.validation']:.2f} and {torus['torus6d_5m']['nll.validation']:.2f}. Each NLL is a per-sample mean over that dataset's own held-out validation split. This is a comparison of two fitted runs, not an isolated causal effect of sample size.",
        "checkpoint_example":f"For the current 5M run at $K=2000$, $\\tau=0.05$, adjusted alignment changes from \\textbf{{{early.adjusted_alignment:.3f}}} at iteration 2 to \\textbf{{{terminal.adjusted_alignment:.3f}}} at iteration {int(terminal.iteration)}. Mean rank changes from {early.mean_rank:.2f} to {terminal.mean_rank:.2f}, and the history's validation NLL changes from {early.validation_nll_history:.2f} to {terminal.validation_nll_history:.2f}. The selected iteration is {int(terminal.selected_iteration)}. History NLL uses the EM scorer and can differ numerically from the saved-run evaluator on the preceding page.",
        "stops":f"The 44 evaluated 5M EM runs comprise {int(stop_counts.get('converged',0))} convergence stops and {int(stop_counts.get('patience',0))} validation-patience stops. An asterisk marks the latter; their terminal checkpoint can differ from the best-validation checkpoint used for the selected-model plots. The unfinished $K=2000$, $\\tau=1$ run is omitted from both checkpoint panels.",
    }
    # Reuse the original report's typography and page geometry.
    original=REPO/"outputs/reports/em_toy_manifolds_2026-10-01/report.tex"
    preamble=original.read_text().split(r"\begin{document}")[0]
    body=REPORT_BODY
    for key,value in values.items():
        body=body.replace("@"+key+"@",value)
    assert "@" not in body
    (output/"report.tex").write_text(preamble+body)
    provenance=json.loads((output/"data/provenance.json").read_text())
    provenance["exporter_sha256"]=sha256(Path(__file__))
    provenance["report_template_source"]={"path":str(original.relative_to(REPO)),"sha256":sha256(original)}
    (output/"data/provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    readme="""EM TOY-MANIFOLD REPORT — 23 OCTOBER 2026

report.pdf is the compiled report; report.tex is the editable LaTeX source.
plots/ contains vector PDFs and PNGs. data/ contains all source summaries,
selected rows, plotted grids, full selected coverage curves (compressed CSV),
and provenance with SHA-256 hashes of the per-run metrics and specifications.

Main selection: K=2000, tau=0.05, rank capacity 32. The EM experiments are the
shared-noise variants selected by notebooks/toy_manifold_noise_sweep_results.ipynb.
The _bk experiments are excluded, with run IDs recorded in provenance.json.
Rank and geometry summaries give equal weight to each manifold type, omit
undefined scores, and include associated assignment-dead components. KMeans
excludes PCA-ineligible components as specified by the saved evaluator.
Coverage uses 30,000 independent test points per noise condition and training-live
centroids. It measures sample-distribution coverage, independently of tangents.

Sources are the four dataset roots recorded in data/provenance.json. Metrics
were read directly from current per-run metrics.json files, because the 500K
torus summaries are absent and the 5M torus summaries omit completed runs.
The high-dimensional diagnostic retains its own available K grid (50--500).
For EM and KMeans it uses the newer run specifications that explicitly enable
held-out coverage, excluding superseded runs at the same K/threshold settings.
The 500K torus has 45 evaluated EM runs. The 5M torus has 44 evaluated EM runs
and 45 evaluated KMeans runs. The missing 5M EM cell is K=2000, tau=1; it has
no final evaluation and is shown as missing. No previous values fill that cell.

Checkpoint geometry is recomputed from current epoch_0002 and terminal model
files using dalg.evaluation.toy_manifold_metrics._component_metrics. Source
checkpoints and histories are hashed in data/torus6d_5m_checkpoint_geometry.csv.
It does not recompute assignments, refit models, or alter any scratch artifact.
The old checkpoint geometry and its historical conclusions are not reused.
Historical tests outside the four requested roots are omitted from this report.

REPRODUCE (from repository root; choose a fresh folder)
PYTHONPATH=src .venv/bin/python scripts/temporary/export_em_report_refresh.py --output outputs/reports/<new-folder>
To rerender the saved snapshot without recollecting metrics/checkpoints:
PYTHONPATH=src .venv/bin/python scripts/temporary/export_em_report_refresh.py --output outputs/reports/<new-folder> --from-data

COMPILE (from the report folder)
tectonic report.tex
or: pdflatex report.tex (twice)
Compilation needs only this folder, not the repository or scratch datasets.

REFERENCES
docs/experiments/evaluation/toy-manifold-tiling.md
docs/evaluation/heldout-distribution-coverage.md
docs/models/mfa-hddc.md
docs/reference/toy-manifold-dataset.md
"""
    (output/"README.txt").write_text(readme)


REPORT_BODY = r"""\begin{document}
{\LARGE\bfseries From SGD to EM: recovering local geometry}\par
{\small Updated 23 October 2026 \hfill Current notebook selection: $K=2000$, $\tau=0.05$}\par
\textbf{Remark:} Previous to this report I always used shared noise among components $b$. For this report on I will use component specific noise $b_k$, mainly to make it more similar to Bouveyron et al. paper and because this change doesn't affect our metrics.
\sectionhead{ID estimation via surgery thresholds}
Since I have changed some details on how ID is estimated, I will report here the full updated procedure in details.
For each gaussian component $k$ we first compute the empirical covariance $C_k$ and its eigenvalues $\{ \lambda_k \}_{k=1}^{k=D}$, with $D$ being the embedding dimension.
\subparagraph{1. Select a candidate rank \(r_k\) with Cattell.}
First we computed the normalised gap between eigevalues as
\begin{equation*}
    \Delta_j = \frac{\lambda_j - \lambda_{j+1}}{\lambda_1}
\end{equation*}
based on the selected surgery threshold $\tau$ we select the \textbf{last} index for which $\Delta_j > \tau$. If no gap qualify $q_k$ is set to a predefined $q_{\text{max}}$.
\[
r_k =
\begin{cases}
\max\{j : \Delta_j > \tau\}, & \text{if a gap qualifies}, \\
q_{\max}, & \text{otherwise}.
\end{cases}
\]

\subparagraph{2. Estimate $b_k$, including the numnerical floor}
After estimating $d_k$ I estimate the noise using the remaning $D-d_k$:
\begin{equation}
    b_k = \max(\frac{\sum_{j=d_{k+1}}^D \lambda_{k,j}}{D-d_k}, b_{\min})
\end{equation}
Where $b_{\min}$ is the numerical floor.

\subparagraph{Remove directions $\lambda_{k,j} \leq b_k$, and adjust $b_k$}
This part, as far as I understand is not implemented in the original paper from Bouveyron et al. When one of the selected direction $\lambda_{k,j}$ (with $j\leq d_k$) is $< b_k$, I decrease $d_k$ by one and recompute $b_k$ with the new eigenvalue. This step is performed when $\lambda_{kj}$ is smaller than numerical floor, or when $b$ is not component specific but shared among all gaussian.


Full-data EM fits adaptive-rank MFA using soft assignments and closed-form updates. This report refreshes the original experiment summary from the current saved evaluations. It compares K-means + PCA (KM), vanilla MFA (MFA), HDDC fitted with Adam (Adam), and HDDC fitted with EM (EM). The selected EM runs use a shared isotropic noise variance and rank capacity 32.

\sectionhead{What the geometry metrics measure}
Let $r$ be intrinsic dimension, $m$ native embedding dimension, and $q_k$ learned rank; the ambient dimension is $D=128$. With true tangent basis $T$ and ordered covariance PCs $P$, adjusted alignment is $\adj_k=\|T^\top P_{1:r}\|_F^2/r$ for $q_k\ge r$, $\adj_k=\|T^\top P_{1:q_k}\|_F^2/r$ for $0<q_k<r$, and zero at rank zero. Containment maximizes overlap over $r$ PCs selected from the first $q_k$ when $q_k\ge r$. Both lie in $[0,1]$; neither directly penalizes excess rank. Rank and geometry summaries give equal weight to manifold types. Undefined geometry is omitted, and PCA-ineligible KMeans components are excluded from rank and tangent scores.

\sectionhead{Ten-manifold noise sweep}
The benchmark contains 30,000 samples per manifold. Noise is indexed by $\rho=R_{\mathrm{curvature}}/\sigma$; smaller values mean stronger noise. $K$ is the number of components and $\tau$ the Cattell threshold.
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=.96\linewidth]{plots/main_sweep.pdf}
\figcaption{Current noiseless EM sweep. Rank and tangent statistics are macro-averages over ten manifold types; all 45 grid combinations have evaluations.}
\end{minipage}\par
@main_result@
\par\begin{minipage}{\linewidth}\centering
{\small\setlength{\tabcolsep}{3pt}\input{data/rank_table.tex}}
\captionof{table}{Fixed $K=2000$, $\tau=0.05$ for KM, Adam and EM; vanilla MFA uses rank capacity 32. NLL is undefined for KMeans. Live counts use the full assignment stream.}
\end{minipage}\par
\clearpage
\sectionhead{Tangent geometry by manifold}
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=.95\linewidth]{plots/main_tangent.pdf}
\figcaption{Per-manifold geometry at the selected settings. Row labels give $(r,m)$; columns group KM, Adam and EM within each noise condition.}
\end{minipage}\par
\sectionhead{Swiss roll at the current settings}
@swiss@
\sectionhead{Likelihood and geometric recovery}
@likelihood@
\sectionhead{High-dimensional rank recovery}
@high_rank@

\clearpage
\sectionhead{Where the components and dimensions go}
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=.95\linewidth]{plots/main_allocation.pdf}
\figcaption{Associated component counts sum to $K=2000$, including assignment-dead components. Rank labels show mean $q_k$; color shows $|\bar q-r|$, with errors above five sharing the darkest color.}
\end{minipage}\par
\sectionhead{A separate 10D benchmark}
@highdim@
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth]{plots/highdim_sweep.pdf}
\figcaption{Noiseless EM on the three 10D manifolds. This benchmark retains its own saved $K$ grid; it has no $K=2000$ runs. Noisy-condition companion sweeps are included in the plots folder.}
\end{minipage}\par
\textbf{Interpretation.} EM improves tangent recovery relative to Adam at these settings, especially under strong noise. Exact intrinsic rank remains a separate challenge. High tangent scores can coexist with excess rank, and the high-dimensional product torus can also suffer severe rank collapse. The isolated-torus results below test the same issue without competition among manifold types.

\clearpage
\sectionhead{Held-out distribution coverage}
For an independent test point $x$, let $d(x)=\min_{k\in C}\|x-\mu_k\|_2$, where $C$ contains centroids with at least one training assignment. The coverage curve is the empirical fraction $\Pr_{\mathrm{test}}[d(x)\le\epsilon]$. Each noise condition uses the same 30,000 independent test points for all four models. Higher coverage at a fixed radius is better; smaller distance summaries are better.
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=.94\linewidth]{plots/main_coverage.pdf}
\figcaption{Full empirical test-coverage curves from the current evaluations, with common axes and the same $K=2000$, $\tau=0.05$ selection as the rank table.}
\end{minipage}\par
@coverage@
\par\begin{minipage}{\linewidth}\centering
{\small\setlength{\tabcolsep}{6pt}\input{data/coverage_table.tex}}
\captionof{table}{Euclidean distance to the nearest training-live centroid. $r_{95}$ is the smallest empirical radius containing at least 95\% of test points. Training-live counts can differ from the full-stream live counts in Table 1.}
\end{minipage}\par
Coverage measures placement of centroids under the test distribution. It does not use covariance, learned directions, or the exact continuous manifold. Ambient noise contributes to distances, and good coverage does not establish correct local tangent geometry.

\clearpage
\appendix
\sectionhead{Appendix A: isolated 6D torus sweeps}
The isolated noiseless product torus has $(r,m)=(6,12)$ in $D=128$. The 500K source has 45 evaluated EM runs. The 5M source has 44 evaluated EM runs and 45 KMeans runs. The 5M EM run at $K=2000$, $\tau=1$ has no saved evaluation and appears as a dash. All figures use current saved-model evaluations; geometry and mean rank are unweighted over associated components.
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=.96\linewidth]{plots/torus6d_500k_sweep.pdf}
\figcaption{500,000 samples, EM. Alignment spans 0--1; mean rank spans 0--12. The intrinsic rank target is six.}
\end{minipage}\par
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=.96\linewidth]{plots/torus6d_5m_sweep.pdf}
\figcaption{5,000,000 samples, EM, on the same geometry scales. Missing results are not replaced with values from the old report.}
\end{minipage}\par
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=.96\linewidth]{plots/torus6d_5m_kmeans_sweep.pdf}
\figcaption{5,000,000 samples, K-means + PCA. These newly available baselines use the same axes and geometry scales as EM.}
\end{minipage}\par
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=.96\linewidth]{plots/torus6d_likelihood_comparison.pdf}
\figcaption{Selected-checkpoint mean validation NLL, with a common scale. Cell labels are rounded to integers; lower is better. Each dataset uses its own held-out validation population.}
\end{minipage}

\clearpage
\sectionhead{Appendix B: current checkpoint trajectories}
@torus_comparison@
\par
@checkpoint_example@
\par
@stops@
\sectionhead{Early checkpoint: iteration 2}
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth]{plots/torus6d_5m_early_geometry.pdf}
\figcaption{Current 5M EM checkpoints at iteration 2, for the 44 runs with final saved evaluations. Geometry is computed with the repository's component evaluator from the actual checkpoint parameters.}
\end{minipage}\par
\sectionhead{Convergence or validation-patience stop}
\par\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth]{plots/torus6d_5m_terminal_geometry.pdf}
\figcaption{Terminal snapshots of the same 44 runs, using the early-panel scales. An asterisk denotes a validation-patience stop. Geometry does not depend on assignment-live status.}
\end{minipage}\par
The checkpoint analysis separates the optimization trajectory from the final checkpoint selection. All values on this page are refreshed from the current runs; the earlier report's checkpoint identities and numerical conclusions are not carried forward. Source metrics, run specifications, model checkpoints and histories are recorded in the accompanying data and provenance files.
\end{document}
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=REPO/"outputs/reports/em_toy_manifolds_2026-10-23")
    parser.add_argument("--from-data", action="store_true", help="Render the already collected report snapshot")
    parser.add_argument("--checkpoints-only", action="store_true", help="Refresh checkpoint geometry in an existing report folder")
    args=parser.parse_args()
    output=args.output.resolve()
    if args.checkpoints_only:
        checkpoint_figures(checkpoint_geometry(output),output)
        return
    if args.from_data:
        context=notebook_context()
        data={}
        for key in ROOTS:
            frames=[pd.read_csv(output/"data"/f"{key}_{suffix}.csv",low_memory=False) for suffix in ["summary","manifolds"]]
            for frame in frames:
                frame["noise_ratio"]=frame["noise_ratio"].map(context["noise_key"])
            curve=output/"data"/f"{key}_coverage_curves.csv.gz"
            data[key]=(*frames,pd.read_csv(curve) if curve.exists() else pd.DataFrame())
        main_figures(data,output,context)
        torus_figures(data,output)
        checkpoint_path=output/"data/torus6d_5m_checkpoint_geometry.csv"
        checkpoints=pd.read_csv(checkpoint_path)
        checkpoint_figures(checkpoints,output)
        write_report(data,output,checkpoints)
        return
    output.mkdir(parents=True,exist_ok=False)
    (output/"data").mkdir();(output/"plots").mkdir()
    context=notebook_context()
    provenance={"report_date":"2026-10-23","K":2000,"tau":.05,
                "notebook":str(NOTEBOOK.relative_to(REPO)),"notebook_sha256":sha256(NOTEBOOK),
                "em_experiments":context["EM_EXPERIMENT_BY_NOISE"],"sources":[],"datasets":{}}
    data={key:collect(key,output,context,provenance) for key in ROOTS}
    (output/"data/provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    main_figures(data,output,context)
    torus_figures(data,output)
    checkpoints=checkpoint_geometry(output)
    checkpoint_figures(checkpoints,output)
    write_report(data,output,checkpoints)
    print("Saved",output,flush=True)


if __name__ == "__main__":
    main()
