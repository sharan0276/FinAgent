from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestration.report_models import OrchestrationArtifact
from ui.display import (
    build_export_payload,
    build_financial_metrics_chart,
    build_multiyear_trend_chart,
    build_peer_diff_rows,
    build_pipeline_radar,
    build_risk_diff_rows,
    build_risk_overlap_diff_rows,
    build_risk_severity_chart,
    build_compact_match_rows,
    build_peer_snapshot_rows,
    format_display_label,
    posture_differs,
    severity_badge,
)
from ui.services import (
    build_company_dataset_for_ui,
    get_available_tickers,
    get_faiss_index_status,
    get_ticker_dataset_status,
    list_saved_report_artifacts,
    load_saved_report_artifact,
    rebuild_faiss_index,
    run_analysis_for_ui,
    run_baseline_rag_for_ui,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIPELINE_COLORS = {
    "orchestration_v1": "#1f77b4",
    "baseline_rag_v1": "#ff7f0e",
}
_PIPELINE_LABELS = {
    "orchestration_v1": "Agentic Pipeline",
    "baseline_rag_v1": "Baseline RAG",
}
_POSTURE_COLORS = {
    "Elevated": "#d62728",
    "Mixed": "#ff7f0e",
    "Stable": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_write(text: str) -> None:
    """Write text escaping $ to prevent LaTeX rendering."""
    if text:
        st.markdown(text.replace("$", r"\$"))


def _df_table(rows: list[dict]) -> None:
    """Render a list of dicts as a full-width dataframe with no index."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True)


def _rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _pipeline_badge(schema_version: str) -> str:
    label = _PIPELINE_LABELS.get(schema_version, schema_version)
    color = _PIPELINE_COLORS.get(schema_version, "#888")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600">{label}</span>'
    )


def _posture_badge(label: Optional[str], *, highlight: bool = False) -> str:
    if not label:
        return "—"
    color = _POSTURE_COLORS.get(label, "#888")
    border = "border:2px solid #333;" if highlight else ""
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-size:0.9em;font-weight:600;{border}">{label}</span>'
    )


def _confidence_badge(confidence: str) -> str:
    colors = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#888"}
    color = colors.get(confidence.lower(), "#888")
    label = format_display_label(confidence)
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:3px;font-size:0.8em;font-weight:600">{label} confidence</span>'
    )


# ---------------------------------------------------------------------------
# Section renderers — each is a standalone callable for aligned side-by-side
# ---------------------------------------------------------------------------

def _section_header(artifact: OrchestrationArtifact, *, artifact_path: Optional[str] = None, show_badge: bool = True) -> None:
    bundle = artifact.bundle
    report = artifact.comparison_report
    if show_badge:
        st.markdown(_pipeline_badge(artifact.schema_version), unsafe_allow_html=True)
        st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ticker", bundle.target.ticker)
        st.metric("Company", bundle.target.company or "—")
    with col2:
        st.metric("Filing Year", bundle.target.latest_filing_year or "Unknown")
        st.metric("Peers Found", len(bundle.matches))
    with col3:
        st.metric("Status", report.status)
        st.metric("Top K", bundle.run_metadata.top_k)
    if artifact_path:
        st.caption(f"Artifact: {artifact_path}")


def _section_posture_summary(artifact: OrchestrationArtifact) -> None:
    report = artifact.comparison_report
    if report.posture:
        st.markdown(
            f"**Posture:** {_posture_badge(report.posture.label)}",
            unsafe_allow_html=True,
        )
        for bullet in report.posture.rationale_bullets:
            _safe_write(f"- {bullet}")
    if artifact.bundle.run_metadata.warnings:
        with st.expander("Warnings", expanded=False):
            for w in artifact.bundle.run_metadata.warnings:
                st.warning(w)
    if report.error:
        st.error(report.error)
    st.markdown("**Summary**")
    _safe_write(report.summary or "No summary available.")
    with st.expander("Step Status", expanded=False):
        rows = [{"Step": k, "Status": v} for k, v in artifact.bundle.run_metadata.status_by_step.items()]
        _df_table(rows)


def _section_financials_chart(ingestion: Optional[dict], *, height: int = 430, show_trend: bool = True) -> None:
    if ingestion:
        fig = build_financial_metrics_chart(ingestion, height=height)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        if show_trend:
            trend = build_multiyear_trend_chart(ingestion)
            if trend:
                st.plotly_chart(trend, use_container_width=True)


def _section_positive_deltas(artifact: OrchestrationArtifact) -> None:
    st.markdown("**Positive Deltas**")
    profile = artifact.comparison_report.target_profile
    rows = (
        [{"Metric": i.metric, "Label": format_display_label(i.label), "Value": i.value} for i in profile.positive_deltas]
        if profile else []
    ) or [{"Metric": "—", "Label": "—", "Value": None}]
    _df_table(rows)


def _section_negative_deltas(artifact: OrchestrationArtifact) -> None:
    st.markdown("**Negative Deltas**")
    profile = artifact.comparison_report.target_profile
    rows = (
        [{"Metric": i.metric, "Label": format_display_label(i.label), "Value": i.value} for i in profile.negative_deltas]
        if profile else []
    ) or [{"Metric": "—", "Label": "—", "Value": None}]
    _df_table(rows)


def _section_top_risks(artifact: OrchestrationArtifact) -> None:
    st.markdown("**Top Risks**")
    profile = artifact.comparison_report.target_profile
    if not profile or not profile.top_risks:
        st.caption("No top risks identified.")
        return
    for item in profile.top_risks:
        badge = severity_badge(item.severity)
        st.markdown(
            f"{badge} &nbsp; **{format_display_label(item.signal_type)}**"
            + (f" &nbsp; <span style='color:#666;font-size:0.85em'>{item.section}</span>" if item.section else ""),
            unsafe_allow_html=True,
        )
        if item.summary:
            _safe_write(item.summary)
        if item.citation:
            with st.expander("Evidence", expanded=False):
                _safe_write(item.citation)


def _section_target_profile(artifact: OrchestrationArtifact, ingestion: Optional[dict] = None) -> None:
    st.markdown("**Target Profile**")
    if not artifact.comparison_report.target_profile:
        st.info("Target profile not available.")
        return
    _section_financials_chart(ingestion)
    _section_positive_deltas(artifact)
    _section_negative_deltas(artifact)
    _section_top_risks(artifact)


def _section_peers(artifact: OrchestrationArtifact) -> None:
    bundle = artifact.bundle
    report = artifact.comparison_report
    st.markdown("**Peer Matches**")
    if bundle.matches:
        for match in bundle.matches:
            col_t, col_s = st.columns([1, 3])
            with col_t:
                st.markdown(f"**{match.ticker}**")
                st.caption(match.company or "")
                st.caption(f"FY{match.matched_filing_year}")
            with col_s:
                st.progress(float(match.similarity), text=f"Similarity: {match.similarity:.4f}")
    else:
        st.info("No peer matches found.")

    if report.peer_snapshot:
        with st.expander("Peer Snapshot", expanded=False):
            _df_table(build_peer_snapshot_rows(report.peer_snapshot))


def _section_risks(artifact: OrchestrationArtifact) -> None:
    report = artifact.comparison_report
    st.markdown("**Risk Overlap**")
    overlap_rows = [
        {
            "Group": format_display_label(row.group),
            "Risk Types": ", ".join(format_display_label(r) for r in row.risk_types) or "—",
        }
        for row in report.risk_overlap_rows
    ] or [{"Group": "—", "Risk Types": "—"}]
    _df_table(overlap_rows)

    st.markdown("**Forward Watchlist**")
    if report.forward_watchlist:
        for item in report.forward_watchlist:
            badge = _confidence_badge(item.confidence)
            st.markdown(
                f"{badge} &nbsp; **{format_display_label(item.watch_risk_type)}**",
                unsafe_allow_html=True,
            )
            _safe_write(item.why_relevant)
            with st.expander("Evidence", expanded=False):
                if item.peer_evidence:
                    for e in item.peer_evidence:
                        _safe_write(f"- {e}")
                else:
                    st.caption("No peer evidence available.")
    else:
        st.caption("No forward watchlist items.")


def _section_narrative(artifact: OrchestrationArtifact) -> None:
    report = artifact.comparison_report
    st.markdown("**Narrative**")
    if report.narrative_sections:
        for section in report.narrative_sections:
            st.markdown(f"**{section.title}**")
            _safe_write(section.content)
            if section.citations:
                with st.expander("Evidence", expanded=False):
                    for c in section.citations:
                        _safe_write(f"- {c}")
    else:
        st.info("No narrative sections available.")


def _section_raw_json(artifact: OrchestrationArtifact) -> None:
    with st.expander("Raw JSON", expanded=False):
        st.json(artifact.model_dump(mode="json"), expanded=False)


# ---------------------------------------------------------------------------
# Full report (single column)
# ---------------------------------------------------------------------------

def _render_report(
    artifact: OrchestrationArtifact,
    *,
    artifact_path: Optional[str] = None,
    ingestion: Optional[dict] = None,
) -> None:
    _section_header(artifact, artifact_path=artifact_path)
    st.markdown("---")
    _section_posture_summary(artifact)
    st.markdown("---")
    _section_target_profile(artifact, ingestion=ingestion)
    st.markdown("---")
    _section_peers(artifact)
    st.markdown("---")
    _section_risks(artifact)
    st.markdown("---")
    _section_narrative(artifact)
    _section_raw_json(artifact)


# ---------------------------------------------------------------------------
# Side by Side — section-by-section for true alignment
# ---------------------------------------------------------------------------

def _render_side_by_side(
    agentic_payload: dict,
    baseline_payload: dict,
    agentic_path: Optional[str],
    agentic_ingestion: Optional[dict],
    baseline_ingestion: Optional[dict],
) -> None:
    ag = OrchestrationArtifact.model_validate(agentic_payload)
    bl = OrchestrationArtifact.model_validate(baseline_payload)

    # --- Radar chart (full width) ---
    st.subheader("Pipeline Quality Comparison")
    st.plotly_chart(build_pipeline_radar(ag, bl), use_container_width=True)

    # --- Risk severity chart ---
    severity_fig = build_risk_severity_chart(ag, bl)
    if severity_fig:
        st.plotly_chart(severity_fig, use_container_width=True)

    # --- Diff tables (full width) ---
    st.subheader("Risk Identification Diff")
    _df_table(build_risk_diff_rows(ag, bl))

    st.subheader("Risk Overlap by Group")
    _df_table(build_risk_overlap_diff_rows(ag, bl))

    st.subheader("Peer Match Comparison")
    peer_rows = build_peer_diff_rows(ag, bl)
    if peer_rows:
        _df_table(peer_rows)
    else:
        st.caption("No peer matches from either pipeline.")

    # --- Posture diff ---
    st.markdown("---")
    differs = posture_differs(ag, bl)
    if differs:
        st.warning("Pipelines disagree on financial posture.")

    # --- Section-by-section parallel rendering ---
    _parallel_section("Overview", ag, bl,
        lambda a: _section_header(a, artifact_path=agentic_path if a is ag else None, show_badge=False),
        lambda a: _section_header(a, show_badge=False))

    _parallel_section("Posture & Summary", ag, bl,
        _section_posture_summary, _section_posture_summary)

    # Target profile — sub-sections paired for alignment
    st.markdown("---\n### Target Profile")
    _parallel_section("Financial Metrics", ag, bl,
        lambda a: _section_financials_chart(agentic_ingestion, height=550, show_trend=False),
        lambda a: _section_financials_chart(baseline_ingestion, height=550, show_trend=False),
        show_divider=False, show_badges=False)

    _parallel_section("Positive Deltas", ag, bl,
        _section_positive_deltas, _section_positive_deltas,
        show_divider=False, show_badges=False)

    _parallel_section("Negative Deltas", ag, bl,
        _section_negative_deltas, _section_negative_deltas,
        show_divider=False, show_badges=False)

    _parallel_section("Top Risks", ag, bl,
        _section_top_risks, _section_top_risks,
        show_divider=False, show_badges=False)

    _parallel_section("Peer Matches", ag, bl,
        _section_peers, _section_peers)

    _parallel_section("Risks & Watchlist", ag, bl,
        _section_risks, _section_risks)

    _parallel_section("Narrative", ag, bl,
        _section_narrative, _section_narrative)

    col_a, col_b = st.columns(2)
    with col_a:
        _section_raw_json(ag)
    with col_b:
        _section_raw_json(bl)


def _parallel_section(title: str, ag, bl, fn_ag, fn_bl, *, show_divider: bool = True, show_badges: bool = True) -> None:
    """Render a named section for both pipelines in aligned columns."""
    if show_divider:
        st.markdown(f"---\n### {title}")
    else:
        st.markdown(f"### {title}")
    col_a, col_b = st.columns(2)
    with col_a:
        if show_badges:
            st.markdown(_pipeline_badge(ag.schema_version), unsafe_allow_html=True)
        fn_ag(ag)
    with col_b:
        if show_badges:
            st.markdown(_pipeline_badge(bl.schema_version), unsafe_allow_html=True)
        fn_bl(bl)


# ---------------------------------------------------------------------------
# Dataset & FAISS status panel
# ---------------------------------------------------------------------------

def _render_path_list(title: str, paths: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not paths:
        st.caption("None")
        return
    for path in paths:
        st.code(path, language="text")


def _render_status_panel(ticker: str) -> None:
    st.subheader("Dataset Status")
    dataset_result = st.session_state.get("dataset_result")
    if dataset_result:
        if dataset_result.get("error"):
            st.error(dataset_result["error"])
        st.json(
            {
                "ticker": dataset_result.get("ticker"),
                "company_name": dataset_result.get("company_name"),
                "latest_filing_year": dataset_result.get("latest_filing_year"),
                "status_by_step": dataset_result.get("status_by_step", {}),
                "warnings": dataset_result.get("warnings", []),
            },
            expanded=True,
        )
        if dataset_result.get("ingestion_path"):
            st.code(dataset_result["ingestion_path"], language="text")
        _render_path_list("Extraction Artifacts", dataset_result.get("extraction_paths", []))
        _render_path_list("Curator Artifacts", dataset_result.get("curator_paths", []))
    else:
        st.caption("No dataset build has been run in this session.")

    st.subheader("Ticker Dataset")
    st.json(get_ticker_dataset_status(ticker), expanded=False)

    st.subheader("FAISS Index")
    index_status = st.session_state.get("index_status") or get_faiss_index_status()
    if not index_status.get("index_exists"):
        st.warning("FAISS index file is missing.")
    st.json(
        {
            "index_exists": index_status.get("index_exists"),
            "metadata_exists": index_status.get("metadata_exists"),
            "entry_count": index_status.get("entry_count"),
            "index_path": index_status.get("index_path"),
        },
        expanded=False,
    )
    with st.expander("Raw FAISS Metadata", expanded=False):
        st.json(index_status.get("metadata") or index_status.get("build_payload") or {}, expanded=False)


# ---------------------------------------------------------------------------
# Ingestion loader for charts
# ---------------------------------------------------------------------------

def _load_ingestion_for_ticker(ticker: str) -> Optional[dict]:
    try:
        path = REPO_ROOT / "data-ingestion" / "outputs" / ticker.upper() / "complete_ingestion.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="FinAgent", layout="wide")
    st.title("FinAgent — Financial Analysis")

    with st.sidebar:
        st.header("Analysis")

        available_tickers = get_available_tickers()
        default_idx = available_tickers.index("AAPL") if "AAPL" in available_tickers else 0
        ticker = st.selectbox("Ticker", options=available_tickers, index=default_idx, key="main_ticker")
        top_k = st.number_input("Top K Peers", min_value=1, max_value=10, value=2, step=1)

        st.markdown("**Agentic Pipeline**")
        if st.button("Run Agentic Pipeline", use_container_width=True):
            with st.spinner(f"Running agentic analysis for {ticker}..."):
                try:
                    artifact, output_path = run_analysis_for_ui(ticker, top_k=int(top_k))
                    st.session_state["agentic_artifact"] = artifact.model_dump(mode="json")
                    st.session_state["agentic_artifact_path"] = str(output_path)
                    st.success("Agentic pipeline complete.")
                except Exception as exc:
                    st.error(f"Agentic pipeline failed: {exc}")

        st.markdown("**Baseline RAG**")
        focus_query = st.text_input(
            "Focus Area (optional)",
            value="",
            placeholder="e.g. margin compression, R&D trends",
            key="focus_query",
        ).strip() or None

        if st.button("Run Baseline RAG", use_container_width=True):
            with st.spinner(f"Running baseline RAG for {ticker}..."):
                try:
                    artifact = run_baseline_rag_for_ui(ticker, top_k=int(top_k), focus_query=focus_query)
                    st.session_state["baseline_artifact"] = artifact.model_dump(mode="json")
                    st.success("Baseline RAG complete.")
                except Exception as exc:
                    st.error(f"Baseline RAG failed: {exc}")

        if st.button("Clear Results", use_container_width=True, type="secondary"):
            for key in ("agentic_artifact", "agentic_artifact_path", "baseline_artifact"):
                st.session_state.pop(key, None)
            _rerun()

        st.divider()
        st.markdown("**Load Saved Report**")
        saved_artifacts = list_saved_report_artifacts()
        options = ["None"] + [item["label"] for item in saved_artifacts]
        selected_label = st.selectbox("Saved Reports", options=options)
        load_target = st.radio("Load into", options=["Agentic", "Baseline"], horizontal=True, key="load_target")
        if selected_label != "None" and st.button("Load Report", use_container_width=True):
            selected = next(item for item in saved_artifacts if item["label"] == selected_label)
            artifact = load_saved_report_artifact(selected["path"])
            key = "agentic_artifact" if load_target == "Agentic" else "baseline_artifact"
            st.session_state[key] = artifact.model_dump(mode="json")
            if load_target == "Agentic":
                st.session_state["agentic_artifact_path"] = selected["path"]

        st.divider()
        st.header("Dataset Management")
        dataset_ticker = st.selectbox(
            "Company Ticker",
            options=[""] + available_tickers,
            index=0,
            key="dataset_ticker",
        ) if available_tickers else st.text_input("Company Ticker", value="", key="dataset_ticker").strip().upper()

        rebuild_after_add = st.checkbox("Rebuild FAISS after add", value=False)
        if st.button("Build Dataset Artifacts", use_container_width=True, disabled=not dataset_ticker):
            with st.spinner(f"Building dataset for {dataset_ticker}..."):
                build_result = build_company_dataset_for_ui(dataset_ticker)
            st.session_state["dataset_result"] = build_result
            if rebuild_after_add and not build_result.get("error"):
                with st.spinner("Rebuilding FAISS index..."):
                    st.session_state["index_status"] = rebuild_faiss_index()

        if st.button("Rebuild FAISS Index", use_container_width=True):
            with st.spinner("Rebuilding FAISS index..."):
                st.session_state["index_status"] = rebuild_faiss_index()

    # ------------------------------------------------------------------
    # Main area
    # ------------------------------------------------------------------
    agentic_payload = st.session_state.get("agentic_artifact")
    baseline_payload = st.session_state.get("baseline_artifact")
    agentic_path = st.session_state.get("agentic_artifact_path")
    both_loaded = agentic_payload is not None and baseline_payload is not None

    agentic_ticker = OrchestrationArtifact.model_validate(agentic_payload).bundle.target.ticker if agentic_payload else ticker
    baseline_ticker = OrchestrationArtifact.model_validate(baseline_payload).bundle.target.ticker if baseline_payload else ticker
    agentic_ingestion = _load_ingestion_for_ticker(agentic_ticker)
    baseline_ingestion = _load_ingestion_for_ticker(baseline_ticker)

    if both_loaded:
        ag = OrchestrationArtifact.model_validate(agentic_payload)
        bl = OrchestrationArtifact.model_validate(baseline_payload)
        st.download_button(
            label="Export Comparison JSON",
            data=build_export_payload(ag, bl),
            file_name=f"{agentic_ticker}_comparison_export.json",
            mime="application/json",
        )

        tab_side, tab_agentic, tab_baseline, tab_status = st.tabs([
            "Side by Side", "Agentic Pipeline", "Baseline RAG", "Dataset Status"
        ])
        with tab_side:
            _render_side_by_side(agentic_payload, baseline_payload, agentic_path, agentic_ingestion, baseline_ingestion)
        with tab_agentic:
            _render_report(ag, artifact_path=agentic_path, ingestion=agentic_ingestion)
        with tab_baseline:
            _render_report(bl, ingestion=baseline_ingestion)
        with tab_status:
            _render_status_panel(ticker or dataset_ticker)

    elif agentic_payload or baseline_payload:
        tab_report, tab_status = st.tabs(["Report", "Dataset Status"])
        with tab_report:
            main_col, status_col = st.columns([2, 1])
            with main_col:
                if agentic_payload:
                    _render_report(OrchestrationArtifact.model_validate(agentic_payload), artifact_path=agentic_path, ingestion=agentic_ingestion)
                else:
                    _render_report(OrchestrationArtifact.model_validate(baseline_payload), ingestion=baseline_ingestion)
            with status_col:
                _render_status_panel(ticker or dataset_ticker)
        with tab_status:
            _render_status_panel(ticker or dataset_ticker)

    else:
        main_col, status_col = st.columns([2, 1])
        with main_col:
            st.info(
                "Select a ticker and run **Agentic Pipeline** and/or **Baseline RAG** from the sidebar. "
                "When both are loaded, a **Side by Side** comparison tab appears automatically."
            )
        with status_col:
            _render_status_panel(ticker or dataset_ticker)


if __name__ == "__main__":
    main()
