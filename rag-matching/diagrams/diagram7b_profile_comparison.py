"""
Diagram 7b — Financial Profile Comparison: Query vs Retrieved Matches

Query: META 2022
Grid heatmap showing financial trajectory labels (revenue, net income, cash,
operating cash flow, R&D) for the query and its top-5 retrieved matches.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

METRICS       = ["Revenues", "NetIncome", "Cash", "OperatingCashFlow", "ResearchAndDevelopment"]
METRIC_LABELS = ["Revenue", "Net Income", "Cash", "Op. Cash Flow", "R&D"]

LABEL_TO_INT = {
    "severe_decline":   -2,
    "moderate_decline": -1,
    "stable":            0,
    "moderate_growth":  +1,
    "strong_growth":    +2,
}
LABEL_SHORT = {
    "severe_decline":   "sev ↓",
    "moderate_decline": "mod ↓",
    "stable":           "stable",
    "moderate_growth":  "mod ↑",
    "strong_growth":    "str ↑",
}

COMPANY_COLORS = {
    "META": "#7C3AED",
    "GOOG": "#0F766E",
    "AAPL": "#B45309",
}

# ---------------------------------------------------------------------------
# Load + compute matches
# ---------------------------------------------------------------------------
def load_curator(ticker, year):
    path = CURATOR_DB / f"{ticker.lower()}_{year}.json"
    return json.loads(path.read_text(encoding="utf-8"))

matrix, metadata = load_vectors(CURATOR_DB)
q_idx = next(i for i, m in enumerate(metadata)
             if m["ticker"] == QUERY_TICKER and m["filing_year"] == QUERY_YEAR)
q_vec  = matrix[q_idx]
sims   = (matrix @ q_vec).tolist()
sims[q_idx] = -1

ranked_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:TOP_K]

q_data = load_curator(QUERY_TICKER, QUERY_YEAR)
q_sigs = {s["signal_type"] for s in q_data["risk_signals"]}

matches = []
for idx in ranked_idx:
    m    = metadata[idx]
    data = load_curator(m["ticker"], m["filing_year"])
    matches.append({
        "ticker": m["ticker"],
        "year":   m["filing_year"],
        "sim":    sims[idx],
        "deltas": data["financial_deltas"],
    })

# ---------------------------------------------------------------------------
# Build label matrix  rows = query + top-5 matches, cols = metrics
# ---------------------------------------------------------------------------
all_rows   = [q_data] + [{"financial_deltas": m["deltas"]} for m in matches]
row_labels = [f"META 2022  (query)"] + [f"{m['ticker']} {m['year']}" for m in matches]

n_rows = len(all_rows)
n_cols = len(METRICS)

label_matrix = np.full((n_rows, n_cols), np.nan)
text_matrix  = [["n/a"] * n_cols for _ in range(n_rows)]

for ri, row in enumerate(all_rows):
    for ci, metric in enumerate(METRICS):
        entry = row["financial_deltas"].get(metric)
        if entry:
            label_matrix[ri, ci] = LABEL_TO_INT[entry["label"]]
            text_matrix[ri][ci]  = LABEL_SHORT[entry["label"]]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

grid_cmap = LinearSegmentedColormap.from_list(
    "fin_delta",
    ["#B91C1C", "#FCA5A5", "#F3F4F6", "#86EFAC", "#15803D"],
    N=256,
)

ax.imshow(label_matrix, cmap=grid_cmap, vmin=-2.5, vmax=2.5, aspect="auto")

# Cell text
for ri in range(n_rows):
    for ci in range(n_cols):
        val = label_matrix[ri, ci]
        txt = text_matrix[ri][ci]
        dark_bg = abs(val) >= 1.5
        txt_color = "white" if dark_bg else "#111827"
        ax.text(
            ci, ri, txt,
            ha="center", va="center",
            fontsize=9.5,
            fontweight="bold" if ri == 0 else "normal",
            color=txt_color,
        )

# Purple border around query row
ax.add_patch(plt.Rectangle(
    (-0.5, -0.5), n_cols, 1.0,
    fill=False, edgecolor="#7C3AED", linewidth=2.8, zorder=5,
))

# Axes
ax.set_xticks(range(n_cols))
ax.set_xticklabels(METRIC_LABELS, fontsize=10, fontweight="bold", color="#111827")
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=10)

ax.get_yticklabels()[0].set_color("#7C3AED")
ax.get_yticklabels()[0].set_fontweight("bold")
for ri, m in enumerate(matches, start=1):
    ax.get_yticklabels()[ri].set_color(COMPANY_COLORS[m["ticker"]])
    ax.get_yticklabels()[ri].set_fontweight("bold")

ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(
    "Financial Profile — Query vs Retrieved Matches",
    fontsize=12, fontweight="bold", pad=12, color="#111827",
)

# Legend — below the grid, 5 items in one row
legend_items = [
    mpatches.Patch(facecolor="#15803D", edgecolor="#111827", linewidth=0.6, label="Strong growth"),
    mpatches.Patch(facecolor="#86EFAC", edgecolor="#111827", linewidth=0.6, label="Moderate growth"),
    mpatches.Patch(facecolor="#F3F4F6", edgecolor="#111827", linewidth=0.6, label="Stable"),
    mpatches.Patch(facecolor="#FCA5A5", edgecolor="#111827", linewidth=0.6, label="Moderate decline"),
    mpatches.Patch(facecolor="#B91C1C", edgecolor="#111827", linewidth=0.6, label="Severe decline"),
]
legend = ax.legend(
    handles=legend_items,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=5,
    fontsize=9,
    frameon=True,
    handlelength=1.8,
    handleheight=1.1,
    labelcolor="#111827",
    edgecolor="#374151",
    facecolor="white",
)
legend.get_frame().set_linewidth(1.4)

fig.text(
    0.5, 0.01,
    "Source: SEC EDGAR 10-K filings  |  Model: all-MiniLM-L6-v2, 384-dim",
    ha="center", fontsize=8, color="#374151",
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = "rag-matching/diagrams/diagram7b_profile_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
