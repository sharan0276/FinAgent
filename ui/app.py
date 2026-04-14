from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestration.report_models import OrchestrationArtifact
from ui.display import build_compact_match_rows, build_peer_snapshot_rows, format_display_label
from ui.services import (
    build_company_dataset_for_ui,
    get_faiss_index_status,
    get_ticker_dataset_status,
    list_saved_report_artifacts,
    load_saved_report_artifact,
    rebuild_faiss_index,
    run_analysis_for_ui,
    run_baseline_rag_for_ui,
)

# ---------------------------------------------------------------------------
# Report rendering helpers
# ---------------------------------------------------------------------------

_PIPELINE_COLORS = {
    "orchestration_v1": "#1f77b4",   # blue
    "baseline_rag_v1": "#ff7f0e",    # orange
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


def _pipeline_badge(schema_version: str) -> str:
    label = _PIPELINE_LABELS.get(schema_version, schema_version)
    color = _PIPELINE_COLORS.get(schema_version, "#888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:4px;font-size:0.85em;font-weight:600">{label}</span>'


def _posture_badge(label: Optional[str]) -> str:
    if not label:
        return ""
    color = _POSTURE_COLORS.get(label, "#888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:4px;font-size:0.9em;font-weight:600">{label}</span>'


def _render_path_list(title: str, paths: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not paths:
        st.caption("None")
        return
    for path in paths:
        st.code(path, language="text")


def _render_status_map(status_by_step: dict[str, str]) -> None:
    if not status_by_step:
        st.caption("No step status available.")
        return
    rows = [{"step": step, "status": status} for step, status in status_by_step.items()]
    st.table(rows)


def _render_target_profile(profile: Any) -> None:
    if not profile:
        st.info("Target profile not available.")
        return

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**Positive Deltas**")
        positive_rows = [
            {"Metric": item.metric, "Label": format_display_label(item.label), "Value": item.value}
            for item in profile.positive_deltas
        ] or [{"Metric": "-", "Label": "-", "Value": None}]
        st.table(positive_rows)
    with negatives:
        st.markdown("**Negative Deltas**")
        negative_rows = [
            {"Metric": item.metric, "Label": format_display_label(item.label), "Value": item.value}
            for item in profile.negative_deltas
        ] or [{"Metric": "-", "Label": "-", "Value": None}]
        st.table(negative_rows)

    st.markdown("**Top Risks**")
    if profile.top_risks:
        for item in profile.top_risks:
            st.markdown(
                f"**{format_display_label(item.signal_type)}** | {format_display_label(item.severity)} | {item.section or '-'}"
            )
            if item.summary:
                st.write(item.summary)
            st.caption(f"Occurrences: {item.occurrences}")
            if item.citation:
                with st.expander("View evidence"):
                    st.write(item.citation)
    else:
        st.caption("No top risks available.")


def _render_report(artifact: OrchestrationArtifact, *, artifact_path: Optional[str] = None) -> None:
    bundle = artifact.bundle
    report = artifact.comparison_report
    schema = artifact.schema_version

    # Header with pipeline badge
    st.markdown(_pipeline_badge(schema), unsafe_allow_html=True)
    st.markdown("")

    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ticker", bundle.target.ticker)
        st.metric("Company", bundle.target.company or "—")
    with col2:
        st.metric("Latest Filing Year", bundle.target.latest_filing_year or "Unknown")
        st.metric("Matches Found", len(bundle.matches))
    with col3:
        st.metric("Status", report.status)
        st.metric("Top K", bundle.run_metadata.top_k)

    if artifact_path:
        st.caption(f"Artifact: {artifact_path}")

    # Posture
    if report.posture:
        st.markdown(
            f"**Financial Posture:** {_posture_badge(report.posture.label)}",
            unsafe_allow_html=True,
        )
        for bullet in report.posture.rationale_bullets:
            st.write(f"- {bullet}")

    # Warnings / errors
    if bundle.run_metadata.warnings:
        with st.expander("Warnings", expanded=False):
            for w in bundle.run_metadata.warnings:
                st.warning(w)
    if report.error:
        st.error(report.error)

    # Summary
    st.markdown("**Summary**")
    st.write(report.summary or "No summary available.")

    # Step status
    with st.expander("Step Status", expanded=False):
        _render_status_map(bundle.run_metadata.status_by_step)

    # Target profile
    st.markdown("---")
    st.markdown("**Target Profile**")
    _render_target_profile(report.target_profile)

    # Peer comparison
    st.markdown("---")
    st.markdown("**Peer Matches**")
    if bundle.matches:
        st.markdown("**Matches**")
        st.table(build_compact_match_rows(bundle.matches))
    else:
        st.info("No peer matches available.")

    if report.peer_snapshot:
        st.markdown("**Peer Snapshot**")
        st.table(build_peer_snapshot_rows(report.peer_snapshot))

    # Risk tables
    st.markdown("---")
    st.markdown("**Risk Overlap**")
    overlap_rows = [
        {
            "Group": format_display_label(row.group),
            "Risk Types": ", ".join(format_display_label(item) for item in row.risk_types) or "-",
        }
        for row in report.risk_overlap_rows
    ] or [{"Group": "-", "Risk Types": "-"}]
    st.table(overlap_rows)
    st.markdown("**Forward Watchlist**")
    if report.forward_watchlist:
        for item in report.forward_watchlist:
            st.markdown(f"**{format_display_label(item.watch_risk_type)}** | {format_display_label(item.confidence)} confidence")
            st.write(item.why_relevant)
            with st.expander("View evidence"):
                if item.peer_evidence:
                    for evidence in item.peer_evidence:
                        st.write(f"- {evidence}")
                else:
                    st.caption("No peer evidence available.")
    else:
        st.caption("No forward watchlist items available.")

    st.markdown("**Forward Watchlist**")
    st.table(
        [i.model_dump(mode="json") for i in report.forward_watchlist]
        or [{"watch_risk_type": "-", "why_relevant": "-", "peer_evidence": [], "confidence": "low"}]
    )

    # Narrative
    st.markdown("---")
    st.markdown("**Narrative**")
    if report.narrative_sections:
        for section in report.narrative_sections:
            st.markdown(f"**{section.title}**")
            st.write(section.content)
            if section.citations:
                with st.expander("View evidence"):
                    for citation in section.citations:
                        st.write(f"- {citation}")
    else:
        st.info("No narrative sections available.")

    # Raw JSON
    with st.expander("Raw JSON", expanded=False):
        st.json(artifact.model_dump(mode="json"), expanded=False)


# ---------------------------------------------------------------------------
# Dataset & FAISS status panel
# ---------------------------------------------------------------------------

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
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="FinAgent", layout="wide")
    st.title("FinAgent — Financial Analysis")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Analysis")
        ticker = st.text_input("Ticker", value="AAPL", key="main_ticker").strip().upper()
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

        st.divider()
        st.markdown("**Load Saved Report**")
        saved_artifacts = list_saved_report_artifacts()
        options = ["None"] + [item["label"] for item in saved_artifacts]
        selected_label = st.selectbox("Saved Reports", options=options)
        load_target = st.radio(
            "Load into",
            options=["Agentic", "Baseline"],
            horizontal=True,
            key="load_target",
        )
        if selected_label != "None" and st.button("Load Report", use_container_width=True):
            selected = next(item for item in saved_artifacts if item["label"] == selected_label)
            artifact = load_saved_report_artifact(selected["path"])
            key = "agentic_artifact" if load_target == "Agentic" else "baseline_artifact"
            path_key = "agentic_artifact_path" if load_target == "Agentic" else None
            st.session_state[key] = artifact.model_dump(mode="json")
            if path_key:
                st.session_state[path_key] = selected["path"]

        st.divider()
        st.header("Dataset Management")
        dataset_ticker = st.text_input("Company Ticker", value="", key="dataset_ticker").strip().upper()
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
    # Main area — tabs
    # ------------------------------------------------------------------
    agentic_payload = st.session_state.get("agentic_artifact")
    baseline_payload = st.session_state.get("baseline_artifact")
    agentic_path = st.session_state.get("agentic_artifact_path")

    both_loaded = agentic_payload is not None and baseline_payload is not None

    if both_loaded:
        tab_side, tab_agentic, tab_baseline, tab_status = st.tabs([
            "Side by Side", "Agentic Pipeline", "Baseline RAG", "Dataset Status"
        ])

        with tab_side:
            _render_side_by_side(agentic_payload, baseline_payload, agentic_path)

        with tab_agentic:
            artifact = OrchestrationArtifact.model_validate(agentic_payload)
            _render_report(artifact, artifact_path=agentic_path)

        with tab_baseline:
            artifact = OrchestrationArtifact.model_validate(baseline_payload)
            _render_report(artifact)

        with tab_status:
            _render_status_panel(ticker or dataset_ticker)

    elif agentic_payload or baseline_payload:
        tab_report, tab_status = st.tabs(["Report", "Dataset Status"])

        with tab_report:
            main_col, status_col = st.columns([2, 1])
            with main_col:
                if agentic_payload:
                    artifact = OrchestrationArtifact.model_validate(agentic_payload)
                    _render_report(artifact, artifact_path=agentic_path)
                else:
                    artifact = OrchestrationArtifact.model_validate(baseline_payload)
                    _render_report(artifact)
            with status_col:
                _render_status_panel(ticker or dataset_ticker)

        with tab_status:
            _render_status_panel(ticker or dataset_ticker)

    else:
        main_col, status_col = st.columns([2, 1])
        with main_col:
            st.info(
                "Run **Agentic Pipeline** and/or **Baseline RAG** from the sidebar to view results. "
                "When both are loaded, a **Side by Side** comparison tab will appear."
            )
        with status_col:
            _render_status_panel(ticker or dataset_ticker)


def _render_side_by_side(
    agentic_payload: dict,
    baseline_payload: dict,
    agentic_path: Optional[str],
) -> None:
    """Render both pipeline results in equal-width columns for direct comparison."""
    col_agentic, col_baseline = st.columns(2)

    with col_agentic:
        artifact = OrchestrationArtifact.model_validate(agentic_payload)
        _render_report(artifact, artifact_path=agentic_path)

    with col_baseline:
        artifact = OrchestrationArtifact.model_validate(baseline_payload)
        _render_report(artifact)


if __name__ == "__main__":
    main()
