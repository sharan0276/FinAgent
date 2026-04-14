"""
Diagram 7a — Top-K Retrieved Matches: Ranked Similarity Bars

Query: META 2022
Shows top-5 cosine similarity scores with shared signal count annotations.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    m_sigs = {s["signal_type"] for s in data["risk_signals"]}
    matches.append({
        "ticker":  m["ticker"],
        "year":    m["filing_year"],
        "sim":     sims[idx],
        "shared":  sorted(q_sigs & m_sigs),
    })

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bar_h    = 0.52
sim_vals = [m["sim"] for m in matches]
bar_cols = [COMPANY_COLORS[m["ticker"]] for m in matches]
y_pos    = list(range(TOP_K - 1, -1, -1))   # rank #1 at top

bars = ax.barh(y_pos, sim_vals, height=bar_h,
               color=bar_cols, alpha=0.90, zorder=3)

x_min = min(sim_vals) - 0.04
for yi, (bar, m) in enumerate(zip(bars, matches)):
    bw = bar.get_width()
    # Score inside bar
    ax.text(
        x_min + 0.005, y_pos[yi],
        f"{m['sim']:.4f}",
        va="center", ha="left",
        fontsize=11, fontweight="bold", color="white",
    )
    # Shared signals badge
    n = len(m["shared"])
    ax.text(
        bw + 0.003, y_pos[yi],
        f"{n} shared signal{'s' if n != 1 else ''}",
        va="center", ha="left",
        fontsize=9, fontweight="bold", style="italic", color="#111827",
    )

# Y-axis tick labels colored by company
ytick_labels = [f"#{i+1}  {m['ticker']} {m['year']}" for i, m in enumerate(matches)]
ax.set_yticks(y_pos)
ax.set_yticklabels(ytick_labels[::-1], fontsize=11)
for i, m in enumerate(matches):
    lbl = ax.get_yticklabels()[TOP_K - 1 - i]
    lbl.set_color(COMPANY_COLORS[m["ticker"]])
    lbl.set_fontweight("bold")

ax.set_xlim(min(sim_vals) - 0.04, max(sim_vals) + 0.10)
ax.set_xlabel("Cosine Similarity", fontsize=10, labelpad=6)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(left=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#D1D5DB")
ax.xaxis.grid(True, color="#E5E7EB", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

ax.set_title(
    f"Top-{TOP_K} Retrieved Matches — Query: {QUERY_TICKER} {QUERY_YEAR}",
    fontsize=12, fontweight="bold", pad=12, color="#111827",
)

fig.text(
    0.5, 0.01,
    "Model: all-MiniLM-L6-v2, 384-dim  |  Metric: cosine similarity",
    ha="center", fontsize=8, color="#374151",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = "rag-matching/diagrams/diagram7a_similarity_bars.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
