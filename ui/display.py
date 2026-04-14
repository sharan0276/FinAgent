from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

def format_display_label(value: str | None) -> str:
    if not value:
        return "-"
    if value.isupper():
        return value
    return str(value).replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# Table row builders
# ---------------------------------------------------------------------------

def build_compact_match_rows(matches: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Ticker": match.ticker,
            "Company": match.company or "-",
            "Matched Filing Year": match.matched_filing_year,
            "Similarity": f"{match.similarity:.4f}",
        }
        for match in matches
    ]


def build_peer_snapshot_rows(snapshot: Any) -> list[dict[str, str]]:
    if not snapshot:
        return []
    return [
        {"Category": "Common Strengths", "Details": ", ".join(snapshot.common_strengths) or "-"},
        {"Category": "Common Pressures", "Details": ", ".join(snapshot.common_pressures) or "-"},
        {"Category": "Shared Risk Types", "Details": ", ".join(format_display_label(item) for item in snapshot.shared_risk_types) or "-"},
        {"Category": "Target Differences", "Details": ", ".join(format_display_label(item) for item in snapshot.target_differences) or "-"},
    ]


# ---------------------------------------------------------------------------
# Severity badge HTML
# ---------------------------------------------------------------------------

_SEVERITY_COLORS = {
    "high": ("#d62728", "white"),
    "medium": ("#ff7f0e", "white"),
    "low": ("#2ca02c", "white"),
}


def severity_badge(severity: str) -> str:
    bg, fg = _SEVERITY_COLORS.get(severity.lower(), ("#888", "white"))
    label = format_display_label(severity)
    return f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:3px;font-size:0.8em;font-weight:600">{label}</span>'


# ---------------------------------------------------------------------------
# Diff helpers for Side by Side
# ---------------------------------------------------------------------------

def posture_differs(a: Any, b: Any) -> bool:
    label_a = a.comparison_report.posture.label if a.comparison_report.posture else None
    label_b = b.comparison_report.posture.label if b.comparison_report.posture else None
    return label_a != label_b


def build_risk_diff_rows(agentic: Any, baseline: Any) -> list[dict[str, str]]:
    """Build a unified risk diff table showing which risks each pipeline found."""
    def risk_set(artifact: Any) -> set[str]:
        profile = artifact.comparison_report.target_profile
        if not profile:
            return set()
        return {r.signal_type for r in profile.top_risks}

    agentic_risks = risk_set(agentic)
    baseline_risks = risk_set(baseline)
    all_risks = sorted(agentic_risks | baseline_risks)

    rows = []
    for risk in all_risks:
        rows.append({
            "Risk Type": format_display_label(risk),
            "Agentic": "✓" if risk in agentic_risks else "—",
            "Baseline RAG": "✓" if risk in baseline_risks else "—",
        })
    return rows


def build_risk_overlap_diff_rows(agentic: Any, baseline: Any) -> list[dict[str, str]]:
    """Merge risk overlap rows from both pipelines for comparison."""
    def overlap_map(artifact: Any) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for row in artifact.comparison_report.risk_overlap_rows:
            result[row.group] = row.risk_types
        return result

    a_map = overlap_map(agentic)
    b_map = overlap_map(baseline)
    groups = ["shared_now", "target_only_now", "peer_only_now"]

    rows = []
    for group in groups:
        a_risks = ", ".join(format_display_label(r) for r in a_map.get(group, [])) or "—"
        b_risks = ", ".join(format_display_label(r) for r in b_map.get(group, [])) or "—"
        rows.append({
            "Group": format_display_label(group),
            "Agentic": a_risks,
            "Baseline RAG": b_risks,
        })
    return rows


def build_peer_diff_rows(agentic: Any, baseline: Any) -> list[dict[str, str]]:
    """Compare peer matches found by each pipeline."""
    def peer_set(artifact: Any) -> dict[str, float]:
        return {m.ticker: m.similarity for m in artifact.bundle.matches}

    a_peers = peer_set(agentic)
    b_peers = peer_set(baseline)
    all_tickers = sorted(set(a_peers) | set(b_peers))

    rows = []
    for ticker in all_tickers:
        a_sim = f"{a_peers[ticker]:.4f}" if ticker in a_peers else "—"
        b_sim = f"{b_peers[ticker]:.4f}" if ticker in b_peers else "—"
        rows.append({
            "Peer Ticker": ticker,
            "Agentic Similarity": a_sim,
            "Baseline Similarity": b_sim,
        })
    return rows


# ---------------------------------------------------------------------------
# Chart: Pipeline Quality Radar
# ---------------------------------------------------------------------------

def build_pipeline_radar(agentic: Any, baseline: Any) -> go.Figure:
    """Spider/radar chart comparing pipeline output richness across 6 dimensions."""

    def counts(artifact: Any) -> list[float]:
        report = artifact.comparison_report
        profile = report.target_profile
        return [
            float(len(artifact.bundle.matches)),                            # Peers Found
            float(len(profile.top_risks) if profile else 0),               # Risks Identified
            float(len(profile.positive_deltas) if profile else 0),         # Positive Deltas
            float(len(profile.negative_deltas) if profile else 0),         # Negative Deltas
            float(len(report.narrative_sections)),                          # Narrative Sections
            float(len(report.forward_watchlist)),                           # Watchlist Items
        ]

    categories = [
        "Peers Found",
        "Risks Identified",
        "Positive Deltas",
        "Negative Deltas",
        "Narrative Sections",
        "Watchlist Items",
    ]

    a_vals = counts(agentic)
    b_vals = counts(baseline)

    # Close the polygon
    cats_closed = categories + [categories[0]]
    a_closed = a_vals + [a_vals[0]]
    b_closed = b_vals + [b_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=a_closed,
        theta=cats_closed,
        fill="toself",
        name="Agentic Pipeline",
        line=dict(color="#1f77b4", width=2),
        fillcolor="rgba(31,119,180,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=b_closed,
        theta=cats_closed,
        fill="toself",
        name="Baseline RAG",
        line=dict(color="#ff7f0e", width=2),
        fillcolor="rgba(255,127,14,0.15)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=True, tickfont=dict(size=10))),
        showlegend=True,
        title=dict(text="Pipeline Output Quality Comparison", font=dict(size=14)),
        margin=dict(t=50, b=30, l=30, r=30),
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart: Financial Metrics Bar Chart
# ---------------------------------------------------------------------------

def _extract_latest_value(data: Any) -> Optional[float]:
    """Extract the most recent value from either ingestion format.

    Format A (AAPL-style): {"years": [...], "values": [...], "deltas": [...]}
    Format B (AMZN-style): [{"year": 2024, "value": 123456, ...}, ...]
    Values in format B are raw USD, not millions — divide by 1e6.
    """
    if isinstance(data, dict) and "years" in data:
        years = data.get("years") or []
        vals = data.get("values") or []
        if not years or not vals:
            return None
        val = vals[-1]
        return float(val) if val is not None else None

    if isinstance(data, list) and data:
        # Sort by year descending, take most recent
        rows = sorted(
            [r for r in data if isinstance(r, dict) and r.get("value") is not None],
            key=lambda r: r.get("year", 0),
            reverse=True,
        )
        if not rows:
            return None
        raw = rows[0]["value"]
        # Raw values are full USD — convert to millions
        return float(raw) / 1e6

    return None

def build_financial_metrics_chart(ingestion: dict[str, Any], *, height: int = 430) -> Optional[go.Figure]:
    """Bar chart of key financial metrics for the target company (most recent year)."""
    annual = ingestion.get("financial_data", {}).get("annual", {})
    ticker = ingestion.get("ticker", "")
    company = ingestion.get("company_name", ticker)

    _DISPLAY_METRICS = [
        ("Revenues", "Revenue"),
        ("NetIncome", "Net Income"),
        ("GrossProfit", "Gross Profit"),
        ("OperatingCashFlow", "Operating Cash Flow"),
        ("ResearchAndDevelopment", "R&D"),
    ]

    metrics, values = [], []
    for key, label in _DISPLAY_METRICS:
        if key not in annual:
            continue
        data = annual[key]
        val = _extract_latest_value(data)
        if val is not None:
            metrics.append(label)
            values.append(float(val))

    if not metrics:
        return None

    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]

    fig = go.Figure(go.Bar(
        x=metrics,
        y=values,
        marker_color=colors,
        text=[f"${v:,.0f}M" for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text=f"{ticker} ({company}) — Key Financials (Most Recent Year, USD Millions)", font=dict(size=13)),
        xaxis=dict(title="Metric"),
        yaxis=dict(title="USD Millions"),
        height=height,
        margin=dict(t=60, b=40, l=60, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        uniformtext=dict(mode="hide", minsize=9),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ---------------------------------------------------------------------------
# Chart: Multi-Year Revenue & Net Income Trend
# ---------------------------------------------------------------------------

def _extract_all_years(data: Any) -> list[tuple[int, float]]:
    """Return sorted (year, value_millions) pairs from either ingestion format."""
    pairs: list[tuple[int, float]] = []

    if isinstance(data, dict) and "years" in data:
        years = data.get("years") or []
        vals = data.get("values") or []
        for y, v in zip(years, vals):
            if y is not None and v is not None:
                try:
                    pairs.append((int(y), float(v)))
                except (TypeError, ValueError):
                    pass

    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            y = row.get("year")
            v = row.get("value")
            if y is not None and v is not None:
                try:
                    pairs.append((int(y), float(v) / 1e6))
                except (TypeError, ValueError):
                    pass

    return sorted(pairs, key=lambda x: x[0])


def build_multiyear_trend_chart(ingestion: dict[str, Any], *, height: int = 380) -> Optional[go.Figure]:
    """Line chart of Revenue and Net Income across all available years."""
    annual = ingestion.get("financial_data", {}).get("annual", {})
    ticker = ingestion.get("ticker", "")
    company = ingestion.get("company_name", ticker)

    _TREND_METRICS = [
        ("Revenues", "Revenue", "#1f77b4"),
        ("NetIncome", "Net Income", "#2ca02c"),
        ("GrossProfit", "Gross Profit", "#9467bd"),
        ("OperatingCashFlow", "Operating Cash Flow", "#ff7f0e"),
    ]

    fig = go.Figure()
    has_data = False

    for key, label, color in _TREND_METRICS:
        if key not in annual:
            continue
        pairs = _extract_all_years(annual[key])
        if not pairs:
            continue
        years = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        mode = "lines+markers" if len(pairs) > 1 else "markers"
        fig.add_trace(go.Scatter(
            x=years,
            y=vals,
            mode=mode,
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=9),
        ))
        has_data = True

    if not has_data:
        return None

    fig.update_layout(
        title=dict(text=f"{ticker} ({company}) — Multi-Year Financial Trends (USD Millions)", font=dict(size=13)),
        xaxis=dict(title="Fiscal Year", tickmode="linear", dtick=1),
        yaxis=dict(title="USD Millions"),
        height=height,
        margin=dict(t=60, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ---------------------------------------------------------------------------
# Chart: Risk Severity Distribution (Side by Side comparison)
# ---------------------------------------------------------------------------

def build_risk_severity_chart(agentic: Any, baseline: Any) -> Optional[go.Figure]:
    """Grouped bar chart comparing risk severity counts between pipelines."""

    def severity_counts(artifact: Any) -> dict[str, int]:
        counts = {"high": 0, "medium": 0, "low": 0}
        profile = artifact.comparison_report.target_profile
        if not profile:
            return counts
        for risk in profile.top_risks:
            sev = risk.severity.lower()
            if sev in counts:
                counts[sev] += 1
        return counts

    a_counts = severity_counts(agentic)
    b_counts = severity_counts(baseline)

    # Only render if at least one pipeline has risks
    if sum(a_counts.values()) == 0 and sum(b_counts.values()) == 0:
        return None

    severities = ["High", "Medium", "Low"]
    a_vals = [a_counts["high"], a_counts["medium"], a_counts["low"]]
    b_vals = [b_counts["high"], b_counts["medium"], b_counts["low"]]

    _SEV_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Agentic Pipeline",
        x=severities,
        y=a_vals,
        marker_color="#1f77b4",
        text=a_vals,
        textposition="outside",
        offsetgroup=0,
        width=0.25,
    ))
    fig.add_trace(go.Bar(
        name="Baseline RAG",
        x=severities,
        y=b_vals,
        marker_color="#ff7f0e",
        text=b_vals,
        textposition="outside",
        offsetgroup=1,
        width=0.25,
    ))
    fig.update_layout(
        title=dict(text="Risk Severity Distribution — Agentic vs Baseline RAG", font=dict(size=13)),
        xaxis=dict(title="Severity"),
        yaxis=dict(title="Number of Risks", rangemode="tozero"),
        barmode="group",
        height=460,
        margin=dict(t=60, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def build_export_payload(agentic: Any, baseline: Any) -> str:
    """Serialize both artifacts to a single JSON string for download."""
    payload = {
        "agentic_pipeline": agentic.model_dump(mode="json"),
        "baseline_rag": baseline.model_dump(mode="json") if baseline else None,
    }
    return json.dumps(payload, indent=2)
