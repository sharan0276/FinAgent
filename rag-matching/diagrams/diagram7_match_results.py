"""
Diagram 7 — RAG Top-K Match Results (Experimental Results)

Query: META 2022 (severe net income decline, flat revenue, high R&D spend)

Left panel  — Horizontal ranked similarity bars for top-5 retrieved matches
Right panel — Financial profile comparison: query vs each match (label heatmap)

All values loaded live from curator_db — no hard-coded numbers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from embedder import load_vectors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QUERY_TICKER = "META"
QUERY_YEAR   = 2022
TOP_K        = 5
CURATOR_DB   = Path(__file__).parent.parent / "curator_db"

METRICS = ["Revenues", "NetIncome", "Cash", "OperatingCashFlow", "ResearchAndDevelopment"]
METRIC_LABELS = ["Revenue", "Net Income", "Cash", "Op. Cash Flow", "R&D"]

LABEL_TO_INT = {
    "severe_decline":   -2,
    "moderate_decline": -1,
    "stable":            0,
    "moderate_growth":  +1,
    "strong_growth":    +2,
}
LABEL_SHORT = {
    "severe_decline":   "sev↓",
    "moderate_decline": "mod↓",
    "stable":           "stbl",
    "moderate_growth":  "mod↑",
    "strong_growth":    "str↑",
}

COMPANY_COLORS = {
    "META": "#7C3AED",   # vivid violet
    "GOOG": "#0F766E",   # deep teal
    "AAPL": "#B45309",   # amber
}

# ---------------------------------------------------------------------------
# Load embeddings + compute matches
# ---------------------------------------------------------------------------
matrix, metadata = load_vectors(CURATOR_DB)
q_idx = next(i for i, m in enumerate(metadata)
             if m["ticker"] == QUERY_TICKER and m["filing_year"] == QUERY_YEAR)
q_vec = matrix[q_idx]

sims = (matrix @ q_vec).tolist()
sims[q_idx] = -1  # exclude self

ranked_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:TOP_K]

# ---------------------------------------------------------------------------
# Load curator data for query + matches
# ---------------------------------------------------------------------------
def load_curator(ticker, year):
    path = CURATOR_DB / f"{ticker.lower()}_{year}.json"
    return json.loads(path.read_text(encoding="utf-8"))

q_data   = load_curator(QUERY_TICKER, QUERY_YEAR)
q_sigs   = {s["signal_type"] for s in q_data["risk_signals"]}

matches = []
for idx in ranked_idx:
    m    = metadata[idx]
    data = load_curator(m["ticker"], m["filing_year"])
    m_sigs  = {s["signal_type"] for s in data["risk_signals"]}
    shared  = sorted(q_sigs & m_sigs)
    matches.append({
        "ticker":      m["ticker"],
        "year":        m["filing_year"],
        "sim":         sims[idx],
        "shared":      shared,
        "deltas":      data["financial_deltas"],
    })

# ---------------------------------------------------------------------------
# Build financial label matrix  shape: (TOP_K+1, len(METRICS))
# Rows: query first, then matches in rank order
# ---------------------------------------------------------------------------
all_rows  = [q_data] + [{"financial_deltas": m["deltas"]} for m in matches]
row_labels = [f"META 2022\n(query)"] + [
    f"{m['ticker']} {m['year']}" for m in matches
]

label_matrix = np.zeros((len(all_rows), len(METRICS)), dtype=float)
text_matrix  = [["" for _ in METRICS] for _ in all_rows]

for ri, row in enumerate(all_rows):
    for ci, metric in enumerate(METRICS):
        entry = row["financial_deltas"].get(metric)
        if entry:
            label_matrix[ri, ci] = LABEL_TO_INT[entry["label"]]
            text_matrix[ri][ci]  = LABEL_SHORT[entry["label"]]
        else:
            label_matrix[ri, ci] = np.nan
            text_matrix[ri][ci]  = "n/a"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 5.2))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    1, 2,
    width_ratios=[1, 1.55],
    wspace=0.32,
    left=0.06, right=0.97,
    top=0.80, bottom=0.13,
)
ax_bars = fig.add_subplot(gs[0])
ax_grid = fig.add_subplot(gs[1])

for ax in (ax_bars, ax_grid):
    ax.set_facecolor("white")

# ===========================================================================
# LEFT — Horizontal ranked similarity bars
# ===========================================================================
bar_h     = 0.52
sim_vals  = [m["sim"] for m in matches]
bar_cols  = [COMPANY_COLORS[m["ticker"]] for m in matches]
y_pos     = list(range(TOP_K - 1, -1, -1))   # top match at top

bars = ax_bars.barh(y_pos, sim_vals, height=bar_h,
                    color=bar_cols, alpha=0.88, zorder=3)

# Similarity value label — inside bar, right-aligned
x_min = min(sim_vals) - 0.04
for yi, (bar, m) in enumerate(zip(bars, matches)):
    bw = bar.get_width()
    ax_bars.text(
        x_min + 0.005, y_pos[yi],
        f"{m['sim']:.4f}",
        va="center", ha="left",
        fontsize=10.5, fontweight="bold",
        color="white",
    )
    # Shared signal count badge on right side
    n_shared = len(m["shared"])
    badge_txt = f"{n_shared} shared signal{'s' if n_shared != 1 else ''}"
    ax_bars.text(
        bw + 0.002, y_pos[yi],
        badge_txt,
        va="center", ha="left",
        fontsize=8.5, color="#111827", style="italic", fontweight="bold",
    )

# Rank labels on y-axis
ytick_labels = [f"#{i+1}  {m['ticker']} {m['year']}" for i, m in enumerate(matches)]
ax_bars.set_yticks(y_pos)
ax_bars.set_yticklabels(ytick_labels[::-1], fontsize=10.5)   # reversed because y_pos reversed

# Color ytick labels by company
for i, m in enumerate(matches):
    # y_pos is reversed: match[0] is at top (y_pos[0] = TOP_K-1)
    ax_bars.get_yticklabels()[TOP_K - 1 - i].set_color(COMPANY_COLORS[m["ticker"]])
    ax_bars.get_yticklabels()[TOP_K - 1 - i].set_fontweight("bold")

x_lo = min(sim_vals) - 0.04
x_hi = max(sim_vals) + 0.08
ax_bars.set_xlim(x_lo, x_hi)
ax_bars.set_xlabel("Cosine Similarity", fontsize=10, labelpad=5)
ax_bars.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax_bars.tick_params(axis="x", labelsize=9)

ax_bars.spines["top"].set_visible(False)
ax_bars.spines["right"].set_visible(False)
ax_bars.spines["left"].set_visible(False)
ax_bars.spines["bottom"].set_color("#D1D5DB")
ax_bars.xaxis.grid(True, color="#F3F4F6", linewidth=0.8, zorder=0)
ax_bars.set_axisbelow(True)
ax_bars.tick_params(left=False)

ax_bars.set_title(
    f"Top-{TOP_K} Retrieved Matches\nQuery: {QUERY_TICKER} {QUERY_YEAR}",
    fontsize=11, fontweight="bold", pad=10, color="#111827",
)

# ===========================================================================
# RIGHT — Financial profile comparison grid (label heatmap)
# ===========================================================================
# Diverging colormap: deep red → white → deep green
grid_cmap = LinearSegmentedColormap.from_list(
    "fin_delta",
    ["#B91C1C", "#FCA5A5", "#F9FAFB", "#86EFAC", "#15803D"],
    N=256,
)

n_rows, n_cols = label_matrix.shape
im = ax_grid.imshow(
    label_matrix,
    cmap=grid_cmap,
    vmin=-2.5, vmax=2.5,
    aspect="auto",
)

# Cell text
for ri in range(n_rows):
    for ci in range(n_cols):
        val = label_matrix[ri, ci]
        txt = text_matrix[ri][ci]
        # Pick text color by background brightness
        dark_bg = abs(val) >= 1.5
        txt_color = "white" if dark_bg else "#1F2937"
        ax_grid.text(
            ci, ri, txt,
            ha="center", va="center",
            fontsize=9, fontweight="bold" if ri == 0 else "normal",
            color=txt_color,
        )

# Highlight query row with a border
ax_grid.add_patch(plt.Rectangle(
    (-0.5, -0.5), n_cols, 1.0,
    fill=False, edgecolor="#7C3AED", linewidth=2.5, zorder=5,
))

# Axes labels
ax_grid.set_xticks(range(n_cols))
ax_grid.set_xticklabels(METRIC_LABELS, fontsize=9.5, fontweight="bold", color="#374151")
ax_grid.set_yticks(range(n_rows))
ax_grid.set_yticklabels(row_labels, fontsize=9.5)

# Color row labels
ax_grid.get_yticklabels()[0].set_color("#7C3AED")
ax_grid.get_yticklabels()[0].set_fontweight("bold")
for ri, m in enumerate(matches, start=1):
    ax_grid.get_yticklabels()[ri].set_color(COMPANY_COLORS[m["ticker"]])
    ax_grid.get_yticklabels()[ri].set_fontweight("bold")

ax_grid.tick_params(length=0)
for spine in ax_grid.spines.values():
    spine.set_visible(False)

ax_grid.set_title(
    "Financial Profile Comparison\n(Query vs Retrieved Matches)",
    fontsize=11, fontweight="bold", pad=10, color="#111827",
)

# Legend for grid colors
legend_items = [
    mpatches.Patch(color="#15803D", label="Strong growth"),
    mpatches.Patch(color="#86EFAC", label="Moderate growth"),
    mpatches.Patch(color="#D1D5DB", label="Stable"),
    mpatches.Patch(color="#FCA5A5", label="Moderate decline"),
    mpatches.Patch(color="#B91C1C", label="Severe decline"),
]
legend = ax_grid.legend(
    handles=legend_items,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=5,
    fontsize=9.5,
    frameon=True,
    handlelength=1.6,
    handleheight=1.2,
    labelcolor="#111827",
    edgecolor="#D1D5DB",
    facecolor="white",
)
legend.get_frame().set_linewidth(1.2)

# ===========================================================================
# Suptitle + footer
# ===========================================================================
fig.suptitle(
    "RAG Retrieval — Experimental Results: META 2022 Query",
    fontsize=13, fontweight="bold", y=0.96, color="#111827",
)
fig.text(
    0.5, 0.01,
    "Source: SEC EDGAR 10-K filings (AAPL, META, GOOG, 2021–2026)  |  "
    "Model: all-MiniLM-L6-v2, 384-dim  |  Index: FAISS flat cosine",
    ha="center", fontsize=8, color="#374151",
)

out = "rag-matching/diagrams/diagram7_match_results.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
