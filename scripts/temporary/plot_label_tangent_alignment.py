"""Interactive PCA3 view of data, Gaussian components, and label secants."""

from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2


def make_alignment_figure(activations, years, templates, prompts, reference, comparisons,
                          *, probability=0.5, opacity=0.12, label_field="year", alignment_space="PCA3"):
    """Build a model-selector figure using the evaluator's component diagnostics.

    ``comparisons`` maps descriptive run labels to projected Gaussian geometry,
    evaluated projected ``pc1`` directions, and saved ``components`` diagnostics.
    Optional ``pc1_visible`` masks omit directions with no visible projection
    without changing the alignment diagnostics computed in ``alignment_space``.
    PC1 lines pass through each center and end on the ellipsoid surface; they
    use the evaluated directions, not eigenvectors recomputed from covariance.
    Color and hover alignment use
    conditional_alignment: sum_s(w_sk a_sk) / sum_s(w_sk). Undefined alignment
    is gray, never silently assigned score zero. Ellipsoids contain the stated
    probability under each projected three-dimensional marginal Gaussian.
    """
    points = np.asarray(activations, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("The interactive plot requires finite (N,3) activations")
    if not 0 < probability < 1 or not 0 < opacity <= 1:
        raise ValueError("Require 0 < probability < 1 and 0 < opacity <= 1")
    if not comparisons:
        raise ValueError("At least one model comparison is required")
    if label_field not in ("year", "fahrenheit"):
        raise ValueError("label_field must be year or fahrenheit")
    temperature = label_field == "fahrenheit"
    template_values = [None] * len(points) if templates is None else templates
    hover = []
    for y, t, p in zip(years, template_values, prompts, strict=True):
        label = f"Temperature: {float(y):g} °F" if temperature else f"Year: {int(y)}"
        template = "" if t is None else f"<br>Template: {html.escape(str(t))}"
        hover.append(label + template + f"<br>{html.escape(p)}")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name="All test prompts",
        marker=dict(size=2, color="#888888", opacity=.25), text=hover,
        hovertemplate="%{text}<extra></extra>"))
    centroids = np.asarray(reference["means"])
    block_size = reference.get("block_size", 1)
    def interval(y):
        return (f"[{int(y)}, {int(y) + block_size}) °F" if temperature
                else f"{int(y)}–{int(y) + block_size - 1}")

    population = "All prompts" if templates is None else "All templates"
    centroid_hover = [f"Label centroid: {interval(y)}<br>Prompts: {int(n)}<br>{population}"
                      for y, n in zip(reference["years"], reference["counts"], strict=True)]
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
        mode="markers", name="Label centroids", marker=dict(size=3, color="#2464b4"),
        text=centroid_hover, hovertemplate="%{text}<extra></extra>"))
    pairs = np.asarray(reference["endpoint_indices"], dtype=int)
    lines = np.full((len(pairs), 3, 3), np.nan)
    lines[:, :2] = centroids[pairs]
    lines = lines.reshape(-1, 3)
    segment_hover = [text for a, b in pairs
                     for text in (f"Block secant: {interval(reference['years'][a])} → "
                                  f"{interval(reference['years'][b])}",) * 3]
    fig.add_trace(go.Scatter3d(
        x=lines[:, 0], y=lines[:, 1], z=lines[:, 2], mode="lines", name="Centroid secants",
        line=dict(color="#e87512", width=5), text=segment_hover,
        hovertemplate="%{text}<extra></extra>"))
    common_count = len(fig.data)
    azimuth, polar = np.meshgrid(np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 12))
    sphere = np.stack((np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth),
                       np.cos(polar)), axis=-1)
    radius_squared = chi2.ppf(probability, df=3)
    groups = []
    for label, comparison in comparisons.items():
        start = len(fig.data)
        gaussian, diagnostics = comparison["gaussians"], comparison["components"]
        centers = np.asarray(gaussian["centers"])
        covariances = np.asarray(gaussian["covariances"])
        valid = np.asarray(gaussian["valid"], dtype=bool)
        scores = np.asarray(diagnostics["conditional_alignment"])
        pc1 = np.asarray(comparison["pc1"])
        pc1_visible = np.asarray(comparison.get("pc1_visible", diagnostics["identifiable"]), dtype=bool)
        if pc1.shape != centers.shape:
            raise ValueError("Projected PC1 directions must match the Gaussian centers")
        if pc1_visible.shape != (len(centers),):
            raise ValueError("PC1 visibility must have one entry per Gaussian")
        center_hover = []
        pc1_lines, pc1_hover = [], []
        colorbar_added = False
        for k, (center, covariance) in enumerate(zip(centers, covariances, strict=True)):
            score = scores[k]
            alignment_text = f"{score:.4f}" if np.isfinite(score) else "undefined (no eligible segment weight)"
            detail = (f"{html.escape(label)}<br><b>Component {k}</b> (zero-based ID)"
                      f"<br>Alignment (weighted mean cos²): {alignment_text}"
                      f"<br>Alignment space: {html.escape(alignment_space)}"
                      f"<br>Contribution to total score: {float(diagnostics['contribution'][k]):.5f}"
                      f"<br>Segment weight: {float(diagnostics['segment_mass'][k]):.5f}"
                      f"<br>Test assignment fraction: {float(diagnostics['assignment_mass'][k]):.3f}"
                      f"<br>Saved rank: {int(diagnostics['rank'][k])}"
                      f"<br>PC1 identifiable: {bool(diagnostics['identifiable'][k])}")
            center_hover.append(detail)
            if not valid[k]:
                continue
            eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
            if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -1e-9 * max(abs(eigenvalues).max(), 1):
                raise ValueError("Projected Gaussian covariance must be finite and positive semidefinite")
            transform = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0) * radius_squared)
            vertices = center + sphere @ transform.T
            defined = bool(np.isfinite(score))
            show_scale = defined and not colorbar_added
            fig.add_trace(go.Surface(
                x=vertices[:, :, 0], y=vertices[:, :, 1], z=vertices[:, :, 2],
                surfacecolor=np.full(azimuth.shape, score if defined else 0.), cmin=0, cmax=1,
                colorscale="Viridis" if defined else [[0, "#aaaaaa"], [1, "#aaaaaa"]],
                opacity=opacity, showscale=show_scale,
                colorbar=dict(title="Alignment<br>mean cos²", len=.65),
                name=f"Component {k}", showlegend=False,
                hovertemplate=detail + f"<br>{probability:.0%} marginal Gaussian ellipsoid<extra></extra>",
                visible=not groups))
            colorbar_added |= show_scale
            if bool(diagnostics["identifiable"][k]) and pc1_visible[k]:
                direction = pc1[k]
                if not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1.):
                    raise ValueError("Identifiable projected PC1 directions must be finite unit vectors")
                # Intersect the evaluated direction with this probability ellipsoid.
                precision = direction @ np.linalg.solve(covariance, direction)
                half_length = np.sqrt(radius_squared / precision)
                pc1_lines.extend((center - half_length * direction, center + half_length * direction,
                                  np.full(3, np.nan)))
                pc1_hover.extend([detail + "<br>Evaluated PC1 (projected original direction)"] * 3)
        pc1_lines = np.asarray(pc1_lines).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(
            x=pc1_lines[:, 0], y=pc1_lines[:, 1], z=pc1_lines[:, 2], mode="lines",
            line=dict(color="#b2182b", width=2, dash="dot"), name="Evaluated PC1", text=pc1_hover,
            hovertemplate="%{text}<extra></extra>", visible=not groups))
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="markers",
            marker=dict(size=4, color="#222222", symbol="diamond"), name="Gaussian centers",
            text=center_hover, hovertemplate="%{text}<extra></extra>", visible=not groups))
        groups.append((label, start, len(fig.data)))
    buttons = []
    for label, start, end in groups:
        visible = [i < common_count or start <= i < end for i in range(len(fig.data))]
        buttons.append(dict(label=label, method="update", args=[dict(visible=visible),
                            dict(title=dict(text=f"{label} — PCA3 display; alignment in {html.escape(alignment_space)}"))]))
    fig.update_layout(
        title=dict(text=f"{groups[0][0]} — PCA3 display; alignment in {html.escape(alignment_space)}"),
        scene=dict(xaxis_title="Global PC1", yaxis_title="Global PC2", zaxis_title="Global PC3",
                   aspectmode="data"), height=780, margin=dict(l=0, r=0, b=0, t=110),
        updatemenus=[dict(buttons=buttons, x=0, y=1.1, xanchor="left", yanchor="top")],
        legend=dict(orientation="h", x=0, y=-.05), uirevision="alignment-pca3")
    return fig
