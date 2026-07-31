from pathlib import Path

import matplotlib.pyplot as plt


def find_repo_root() -> Path:
    for path in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Could not locate the repository root")


def add_table(fig, bbox, columns, rows, *, widths=None, font_size=9):
    ax = fig.add_axes(bbox)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        colWidths=widths,
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#E5E7EB")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F9FAFB")
    return table


repo = find_repo_root()
out_path = (
    repo
    / "outputs/experiments/layer05_q10_q394_comparison"
    / "layer05_q10_q394_mfa_comparison_report.pdf"
)
out_path.parent.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
ink = "#111827"
muted = "#4B5563"
accent = "#1D4ED8"

fig.text(
    0.08,
    0.955,
    "Layer 5 MFA: q10 vs q394",
    fontsize=20,
    weight="bold",
    color=ink,
)
fig.text(
    0.08,
    0.925,
    "Concise comparison of fit, hard assignments, and empty-component lineages",
    fontsize=10.5,
    color=muted,
)
fig.add_artist(plt.Line2D([0.08, 0.92], [0.905, 0.905], color=accent, linewidth=1.5))

fig.text(0.08, 0.875, "Main result", fontsize=12, weight="bold", color=ink)
fig.text(
    0.08,
    0.85,
    "q394 achieves a substantially lower validation NLL, but only 622 of its 1,000 components receive\n"
    "assignments. Its 378 empty-component lineages are usually redistributed across several\n"
    "occupied q394 clusters rather than merged cleanly into a single destination.",
    fontsize=10,
    color=ink,
    linespacing=1.35,
    va="top",
)

fig.text(0.08, 0.78, "Model summary", fontsize=12, weight="bold", color=ink)
add_table(
    fig,
    [0.08, 0.675, 0.84, 0.085],
    ["Model", "Rank q", "Best validation NLL", "Occupied clusters"],
    [
        ["q10", "10", "1,726.062", "1,000 / 1,000"],
        ["q394", "394", "1,075.520", "622 / 1,000"],
    ],
    widths=[0.20, 0.16, 0.32, 0.32],
)

fig.text(0.08, 0.64, "Assignment agreement", fontsize=12, weight="bold", color=ink)
add_table(
    fig,
    [0.08, 0.455, 0.84, 0.165],
    ["Metric", "Value", "Interpretation"],
    [
        ["Adjusted Rand index", "0.418", "Moderate pairwise agreement"],
        ["Normalized mutual information", "0.761", "Substantial shared information"],
        ["Completeness", "0.793", "q10 regions mostly remain grouped"],
        ["Homogeneity", "0.730", "q394 regions mix multiple q10 regions"],
        ["Raw exact agreement", "44.62%", "Before relabeling"],
        ["Hungarian exact agreement", "49.98%", "After optimal one-to-one relabeling"],
    ],
    widths=[0.31, 0.16, 0.53],
    font_size=8.6,
)

fig.text(0.08, 0.42, "Empty-component lineage flows", fontsize=12, weight="bold", color=ink)
add_table(
    fig,
    [0.08, 0.235, 0.84, 0.165],
    ["Quantity", "Result"],
    [
        ["Empty q394 components", "378"],
        ["Tokens in their same-ID q10 counterparts", "23.09M (31.33% of all tokens)"],
        ["Median fraction sent to the largest destination", "33.97%"],
        ["Median entropy-effective destinations", "9.30"],
        ["Mean destinations receiving at least one token", "122.97"],
        ["Lineage / partial-Hungarian intersection", "291 / 378 (76.98%)"],
    ],
    widths=[0.62, 0.38],
    font_size=8.8,
)

fig.text(0.08, 0.195, "Interpretation", fontsize=12, weight="bold", color=ink)
fig.text(
    0.08,
    0.17,
    "The q394 model improves density fit while producing many unoccupied components.\n"
    "A typical empty-component lineage has the diversity of a split across about nine\n"
    "balanced destinations. The larger raw mean of 123 counts every destination receiving\n"
    "even one token and is therefore sensitive to small tails. The 77% overlap with the\n"
    "partial-Hungarian set shows that component lineage and partition matching agree\n"
    "substantially, but they are not equivalent definitions.",
    fontsize=9.6,
    color=ink,
    linespacing=1.32,
    va="top",
)

fig.text(
    0.08,
    0.055,
    "Source: layer05_q10_q394_mfa_comparison.ipynb. Both runs use identical ordered initialization centroids.",
    fontsize=7.5,
    color=muted,
)

fig.savefig(out_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print(out_path)
