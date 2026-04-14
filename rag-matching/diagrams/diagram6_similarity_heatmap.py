"""
Diagram 6 — Embedding Space Cosine Similarity Heatmap (poster version)

Layout:
  Left (60%):  15x15 heatmap — rows/cols sorted by hierarchical clustering,
               company row/col shading via axhspan/axvspan (no axis-transform hacks),
               dashed company block outlines, value labels on high-similarity cells
  Right (40%): top panel = within vs cross-company bar chart with jittered points
               bottom panel = ranked table of top-5 cross-company matches

All values computed from real SEC EDGAR extraction outputs (AAPL, META, GOOG).
No synthetic or hard-coded similarity values.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from embedder import load_vectors

# ---------------------------------------------------------------------------
# Load + compute
# ---------------------------------------------------------------------------
CURATOR_DB = Path(__file__).parent.parent / "curator_db"
matrix, metadata = load_vectors(CURATOR_DB)
n = len(metadata)

sim_matrix = (matrix @ matrix.T).astype(float)
np.fill_diagonal(sim_matrix, np.nan)

# Hierarchical clustering — reorder so similar pairs sit adjacent
dist_matrix = 1.0 - np.nan_to_num(sim_matrix, nan=0.5)
np.fill_diagonal(dist_matrix, 0.0)
condensed      = squareform(dist_matrix, checks=False)
linkage_matrix = linkage(condensed, method="average")
order          = leaves_list(linkage_matrix)

sim_sorted  = sim_matrix[np.ix_(order, order)]
meta_sorted = [metadata[i] for i in order]
labels      = [f"{m['ticker']} {m['filing_year']}" for m in meta_sorted]

# Within vs cross-company stats
within_vals, cross_vals, cross_pairs = [], [], []
for i in range(n):
    for j in range(i + 1, n):
        v = sim_matrix[i, j]
        if np.isnan(v):
            continue
        if metadata[i]["ticker"] == metadata[j]["ticker"]:
            within_vals.append(v)
        else:
            cross_vals.append(v)
            cross_pairs.append((v, i, j))

within_mean = float(np.mean(within_vals))
cross_mean  = float(np.mean(cross_vals))
ratio       = within_mean / cross_mean
cross_pairs.sort(reverse=True)
top_pairs = cross_pairs[:5]

# ---------------------------------------------------------------------------
# Colors + colormap
# ---------------------------------------------------------------------------
COMPANY_COLORS = {
    "AAPL": "#1565C0",
    "GOOG": "#2E7D32",
    "META": "#6A1B9A",
}

cmap = LinearSegmentedColormap.from_list(
    "fin_sim",
    ["#1A237E", "#5C6BC0", "#FFFFFF", "#EF9A9A", "#B71C1C"],
    N=256,
)

# ---------------------------------------------------------------------------
# Figure — 1×2 outer grid; right side split into bar + table
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor("#FAFAFA")

outer = gridspec.GridSpec(
    1, 2,
    width_ratios=[1.85, 1],
    wspace=0.28,
    left=0.07, right=0.97,
    top=0.88,  bottom=0.14,
)

# Heatmap panel
ax_heat = fig.add_subplot(outer[0])
ax_heat.set_facecolor("#FAFAFA")

# Right side: bar on top, table on bottom
right_gs = gridspec.GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=outer[1],
    height_ratios=[1.1, 1],
    hspace=0.55,
)
ax_bar   = fig.add_subplot(right_gs[0])
ax_table = fig.add_subplot(right_gs[1])
ax_bar.set_facecolor("#FAFAFA")
ax_table.set_facecolor("#FAFAFA")

# ---------------------------------------------------------------------------
# HEATMAP
# ---------------------------------------------------------------------------
vmin = float(np.nanmin(sim_sorted))
vmax = 1.0
plot_matrix = np.nan_to_num(sim_sorted, nan=1.0)

im = ax_heat.imshow(plot_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

# Colorbar — placed cleanly inside the heatmap axes using inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = inset_axes(ax_heat, width="3%", height="60%", loc="lower right",
                 bbox_to_anchor=(0.08, 0.02, 1, 1),
                 bbox_transform=ax_heat.transAxes, borderpad=0)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Cosine\nSimilarity", fontsize=9, labelpad=6)
cbar.ax.tick_params(labelsize=8)

# Tick labels — colored by company
ax_heat.set_xticks(range(n))
ax_heat.set_yticks(range(n))
ax_heat.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5, fontweight="bold")
ax_heat.set_yticklabels(labels, fontsize=8.5, fontweight="bold")

for i, m in enumerate(meta_sorted):
    color = COMPANY_COLORS[m["ticker"]]
    ax_heat.get_xticklabels()[i].set_color(color)
    ax_heat.get_yticklabels()[i].set_color(color)

# Row/column shading — axhspan/axvspan (purely inside axes, no axis transforms)
for i, m in enumerate(meta_sorted):
    color = COMPANY_COLORS[m["ticker"]]
    ax_heat.axhspan(i - 0.5, i + 0.5, color=color, alpha=0.07, zorder=0)
    ax_heat.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.07, zorder=0)

# Dashed company-block outlines
sorted_groups = {}
for i, m in enumerate(meta_sorted):
    sorted_groups.setdefault(m["ticker"], []).append(i)

for ticker, indices in sorted_groups.items():
    start = indices[0] - 0.5
    size  = len(indices)
    ax_heat.add_patch(plt.Rectangle(
        (start, start), size, size,
        fill=False,
        edgecolor=COMPANY_COLORS[ticker],
        linewidth=2.2, linestyle="--", zorder=6,
    ))

# Value labels — high-similarity cells + top cross-company pairs
top_pair_cells = set()
for v, i, j in top_pairs:
    si = list(order).index(i)
    sj = list(order).index(j)
    top_pair_cells.update([(si, sj), (sj, si)])

for i in range(n):
    for j in range(n):
        v = sim_sorted[i, j]
        if np.isnan(v):
            continue
        if v >= 0.85 or (i, j) in top_pair_cells:
            txt_color = "white" if v > 0.80 else "#212121"
            ax_heat.text(j, i, f"{v:.2f}",
                         ha="center", va="center",
                         fontsize=7, fontweight="bold", color=txt_color)

ax_heat.set_xlim(-0.5, n - 0.5)
ax_heat.set_ylim(n - 0.5, -0.5)

# ---------------------------------------------------------------------------
# BAR CHART — within vs cross-company
# ---------------------------------------------------------------------------
categories = ["Within-\nCompany", "Cross-\nCompany"]
means      = [within_mean, cross_mean]
bar_colors = ["#1565C0", "#78909C"]

bars = ax_bar.bar(categories, means, color=bar_colors,
                  width=0.42, zorder=3, alpha=0.82)

rng = np.random.default_rng(42)
for xi, vals in enumerate([within_vals, cross_vals]):
    jitter = rng.uniform(-0.13, 0.13, size=len(vals))
    ax_bar.scatter(
        [xi + j for j in jitter], vals,
        color=bar_colors[xi], alpha=0.4, s=16, zorder=4,
    )

for bar, mean in zip(bars, means):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                mean + 0.006,
                f"{mean:.3f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=bar.get_facecolor())

# Ratio bracket
bx      = 0.5
by_low  = cross_mean
by_high = within_mean
ax_bar.annotate("", xy=(bx, by_high), xytext=(bx, by_low),
                arrowprops=dict(arrowstyle="<->", color="#E53935", lw=2.0))
ax_bar.text(bx + 0.07, (by_low + by_high) / 2,
            f"{ratio:.2f}x\nhigher",
            ha="left", va="center",
            fontsize=10, fontweight="bold", color="#E53935")

ax_bar.set_ylim(0.30, 1.02)
ax_bar.set_ylabel("Mean Cosine Similarity", fontsize=10, labelpad=6)
ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_color("#CFD8DC")
ax_bar.spines["bottom"].set_color("#CFD8DC")
ax_bar.yaxis.grid(True, color="#ECEFF1", linewidth=0.8, zorder=0)
ax_bar.set_axisbelow(True)
ax_bar.tick_params(axis="x", labelsize=10)
ax_bar.tick_params(axis="y", labelsize=8.5)
ax_bar.set_title("Cluster Separation", fontsize=11,
                 fontweight="bold", pad=8, color="#212121")
ax_bar.text(0, 0.315, f"n={len(within_vals)}", ha="center",
            fontsize=8, color="#546E7A")
ax_bar.text(1, 0.315, f"n={len(cross_vals)}", ha="center",
            fontsize=8, color="#546E7A")

# ---------------------------------------------------------------------------
# TABLE — top-5 cross-company matches
# ---------------------------------------------------------------------------
ax_table.axis("off")
ax_table.set_title("Top Cross-Company Matches", fontsize=11,
                   fontweight="bold", pad=6, color="#212121")

col_headers = ["Rank", "Company A", "Company B", "Sim"]
col_x       = [0.02, 0.16, 0.52, 0.88]
row_height  = 0.155
header_y    = 0.92

# Header row
for hdr, cx in zip(col_headers, col_x):
    ax_table.text(cx, header_y, hdr,
                  ha="left", va="top",
                  fontsize=9, fontweight="bold", color="#455A64",
                  transform=ax_table.transAxes)

# Divider line under header
ax_table.plot([0, 1], [header_y - 0.05, header_y - 0.05],
              color="#CFD8DC", linewidth=1.0,
              transform=ax_table.transAxes, clip_on=False)

for rank, (v, i, j) in enumerate(top_pairs):
    ti = metadata[i]["ticker"]
    tj = metadata[j]["ticker"]
    yi = metadata[i]["filing_year"]
    yj = metadata[j]["filing_year"]
    ry = header_y - 0.10 - rank * row_height

    # Alternating row background
    if rank % 2 == 0:
        ax_table.add_patch(plt.Rectangle(
            (0, ry - 0.04), 1.0, row_height,
            transform=ax_table.transAxes,
            color="#F3E5F5", alpha=0.4, zorder=0,
        ))

    ax_table.text(col_x[0], ry, f"#{rank+1}",
                  ha="left", va="center", fontsize=9,
                  color="#546E7A", fontweight="bold",
                  transform=ax_table.transAxes)
    ax_table.text(col_x[1], ry, f"{ti} {yi}",
                  ha="left", va="center", fontsize=9,
                  color=COMPANY_COLORS[ti], fontweight="bold",
                  transform=ax_table.transAxes)
    ax_table.text(col_x[2], ry, f"{tj} {yj}",
                  ha="left", va="center", fontsize=9,
                  color=COMPANY_COLORS[tj], fontweight="bold",
                  transform=ax_table.transAxes)
    ax_table.text(col_x[3], ry, f"{v:.3f}",
                  ha="left", va="center", fontsize=9,
                  color="#B71C1C", fontweight="bold",
                  transform=ax_table.transAxes)

# ---------------------------------------------------------------------------
# Legend + titles
# ---------------------------------------------------------------------------
legend_patches = [
    mpatches.Patch(color=c, label=t, alpha=0.85)
    for t, c in COMPANY_COLORS.items()
]
fig.legend(
    handles=legend_patches,
    loc="upper right",
    fontsize=10,
    framealpha=0.95,
    edgecolor="#CFD8DC",
    title="Company",
    title_fontsize=10,
    bbox_to_anchor=(0.97, 0.97),
)

fig.suptitle(
    "Embedding Space Cosine Similarity — AAPL, META, GOOG (2021-2026)",
    fontsize=15, fontweight="bold", y=0.97, color="#212121",
)
fig.text(
    0.5, 0.01,
    "Rows/columns sorted by hierarchical clustering (average linkage)  |  "
    "Source: SEC EDGAR 10-K filings  |  "
    "Model: all-MiniLM-L6-v2, 384-dim, cosine similarity",
    ha="center", fontsize=8.5, color="#78909C",
)

out = "rag-matching/diagrams/diagram6_similarity_heatmap.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
