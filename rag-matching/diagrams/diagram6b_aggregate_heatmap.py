"""
Diagram 6b — Aggregate Company-Pair Similarity (poster-compact version)

3x3 block matrix showing mean cosine similarity between each company pair,
paired with a within vs cross-company bar chart.

Much more readable at poster scale than the full 15x15 matrix.
All values computed from real embeddings — no hard-coded numbers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from embedder import load_vectors

# ---------------------------------------------------------------------------
# Load + compute full similarity matrix
# ---------------------------------------------------------------------------
CURATOR_DB = Path(__file__).parent.parent / "curator_db"
matrix, metadata = load_vectors(CURATOR_DB)
n = len(metadata)

sim_matrix = (matrix @ matrix.T).astype(float)
np.fill_diagonal(sim_matrix, np.nan)

# ---------------------------------------------------------------------------
# Aggregate into 3x3 block matrix (mean sim per company pair)
# ---------------------------------------------------------------------------
companies = ["AAPL", "META", "GOOG"]   # fixed order: alphabetical

# Group indices by ticker
groups = {c: [i for i, m in enumerate(metadata) if m["ticker"] == c]
          for c in companies}

block = np.zeros((3, 3))
block_n = np.zeros((3, 3), dtype=int)   # pair counts for annotation

for r, ca in enumerate(companies):
    for c, cb in enumerate(companies):
        vals = []
        for i in groups[ca]:
            for j in groups[cb]:
                if i == j:
                    continue
                v = sim_matrix[i, j]
                if not np.isnan(v):
                    vals.append(v)
        block[r, c] = np.mean(vals) if vals else 0.0
        block_n[r, c] = len(vals)

# Within vs cross stats (for bar chart)
within_vals, cross_vals = [], []
for i in range(n):
    for j in range(i + 1, n):
        v = sim_matrix[i, j]
        if np.isnan(v):
            continue
        if metadata[i]["ticker"] == metadata[j]["ticker"]:
            within_vals.append(v)
        else:
            cross_vals.append(v)

within_mean = float(np.mean(within_vals))
cross_mean  = float(np.mean(cross_vals))
ratio       = within_mean / cross_mean

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COMPANY_COLORS = {
    "AAPL": "#0369A1",   # ocean blue — strong, clearly readable
    "META": "#BE185D",   # deep magenta-pink — very distinct from blue/green
    "GOOG": "#15803D",   # rich forest green — distinct from blue/pink
}

# Sequential cool-to-deep colormap: very light sky → vivid cobalt → deep navy
# High contrast at both ends; cells always readable with white or dark text
cmap = LinearSegmentedColormap.from_list(
    "fin_sim",
    ["#DBEAFE", "#60A5FA", "#2563EB", "#1E3A8A"],
    N=256,
)

# ---------------------------------------------------------------------------
# Figure — side-by-side: 3x3 heatmap | bar chart
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 4.8))
fig.patch.set_facecolor("#FAFAFA")

gs = gridspec.GridSpec(
    1, 2,
    width_ratios=[1, 0.75],
    wspace=0.38,
    left=0.08, right=0.97,
    top=0.82,  bottom=0.16,
)

ax_heat = fig.add_subplot(gs[0])
ax_bar  = fig.add_subplot(gs[1])
ax_heat.set_facecolor("#FAFAFA")
ax_bar.set_facecolor("#FAFAFA")

# ---------------------------------------------------------------------------
# 3x3 HEATMAP
# ---------------------------------------------------------------------------
vmin = 0.45   # anchored so cream = "low similarity" is visually clear
vmax = 1.0

im = ax_heat.imshow(block, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

# Cell annotations — value on top line, sublabel below
# Text is dark on light cells (low sim), white on dark cells (high sim)
for r in range(3):
    for c in range(3):
        val   = block[r, c]
        # Luminance threshold: cream cells are light, amber/red are dark
        light = val < 0.70
        txt_color = "#1E293B" if light else "white"
        sub_color = "#475569" if light else "#BFDBFE"
        diag  = (r == c)

        ax_heat.text(c, r - 0.12, f"{val:.3f}",
                     ha="center", va="center",
                     fontsize=15, fontweight="bold", color=txt_color)
        sublabel = "same company" if diag else "cross-company"
        ax_heat.text(c, r + 0.28, sublabel,
                     ha="center", va="center",
                     fontsize=7.5, style="italic", color=sub_color)

# Diagonal cell highlight — white border to emphasise same-company blocks
for d in range(3):
    ax_heat.add_patch(plt.Rectangle(
        (d - 0.5, d - 0.5), 1.0, 1.0,
        fill=False,
        edgecolor="white", linewidth=3.0, zorder=5,
    ))

# Tick labels — colored by company
ax_heat.set_xticks(range(3))
ax_heat.set_yticks(range(3))
ax_heat.set_xticklabels(companies, fontsize=13, fontweight="bold")
ax_heat.set_yticklabels(companies, fontsize=13, fontweight="bold")

for i, c in enumerate(companies):
    color = COMPANY_COLORS[c]
    ax_heat.get_xticklabels()[i].set_color(color)
    ax_heat.get_yticklabels()[i].set_color(color)

ax_heat.tick_params(length=0)

# Colorbar
cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_label("Mean Cosine\nSimilarity", fontsize=9, labelpad=6)
cbar.ax.tick_params(labelsize=8.5)

ax_heat.set_title(
    "Mean Embedding Similarity\nper Company Pair",
    fontsize=11, fontweight="bold", pad=10, color="#212121",
)

# ---------------------------------------------------------------------------
# BAR CHART — within vs cross-company
# ---------------------------------------------------------------------------
categories = ["Within-\nCompany", "Cross-\nCompany"]
means      = [within_mean, cross_mean]
bar_colors = ["#0369A1", "#D97706"]   # ocean blue (within) vs vivid amber (cross)

bars = ax_bar.bar(categories, means, color=bar_colors,
                  width=0.42, zorder=3, alpha=0.85)

# Jittered individual points
rng = np.random.default_rng(42)
for xi, vals in enumerate([within_vals, cross_vals]):
    jitter = rng.uniform(-0.13, 0.13, size=len(vals))
    ax_bar.scatter(
        [xi + j for j in jitter], vals,
        color=bar_colors[xi], alpha=0.35, s=14, zorder=4,
    )

# Mean value labels
for bar, mean in zip(bars, means):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                mean + 0.006,
                f"{mean:.3f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=bar.get_facecolor())

# Ratio bracket
bx = 0.5
ax_bar.annotate("", xy=(bx, within_mean), xytext=(bx, cross_mean),
                arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=2.5))
ax_bar.text(bx + 0.07, (cross_mean + within_mean) / 2,
            f"{ratio:.2f}x",
            ha="left", va="center",
            fontsize=11, fontweight="bold", color="#DC2626")

ax_bar.set_ylim(0.35, 1.0)
ax_bar.set_ylabel("Mean Cosine Similarity", fontsize=10, labelpad=6)
ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_color("#CFD8DC")
ax_bar.spines["bottom"].set_color("#CFD8DC")
ax_bar.yaxis.grid(True, color="#ECEFF1", linewidth=0.8, zorder=0)
ax_bar.set_axisbelow(True)
ax_bar.tick_params(axis="x", labelsize=10)
ax_bar.tick_params(axis="y", labelsize=9)
ax_bar.set_title("Cluster Separation", fontsize=11,
                 fontweight="bold", pad=10, color="#212121")

# ---------------------------------------------------------------------------
# Titles + footer
# ---------------------------------------------------------------------------
fig.suptitle(
    "RAG Embedding Space — Company Similarity Structure",
    fontsize=13, fontweight="bold", y=0.97, color="#212121",
)
fig.text(
    0.5, 0.01,
    "Source: SEC EDGAR 10-K filings (AAPL, META, GOOG, 2021-2026)  |  "
    "Model: all-MiniLM-L6-v2, 384-dim embeddings",
    ha="center", fontsize=8, color="#78909C",
)

out = "rag-matching/diagrams/diagram6b_aggregate_heatmap.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
