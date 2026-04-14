from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestration.report_models import OrchestrationArtifact
from ui.services import (
    build_company_dataset_for_ui,
    get_faiss_index_status,
    get_ticker_dataset_status,
    list_saved_report_artifacts,
    load_saved_report_artifact,
    rebuild_faiss_index,
    run_analysis_for_ui,
)


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

    positives, negatives = st.columns(2)
    with positives:
        st.markdown("**Positive Deltas**")
        st.table([item.model_dump(mode="json") for item in profile.positive_deltas] or [{"metric": "-", "label": "-", "value": None}])
    with negatives:
        st.markdown("**Negative Deltas**")
        st.table([item.model_dump(mode="json") for item in profile.negative_deltas] or [{"metric": "-", "label": "-", "value": None}])

    st.markdown("**Top Risks**")
    st.table([item.model_dump(mode="json") for item in profile.top_risks] or [{"signal_type": "-", "severity": "-", "section": None, "summary": None, "occurrences": 0}])


def _render_report(artifact: OrchestrationArtifact, artifact_path: str | None = None) -> None:
    bundle = artifact.bundle
    report = artifact.comparison_report

    st.subheader("Overview")
    overview_left, overview_right = st.columns(2)
    with overview_left:
        st.metric("Ticker", bundle.target.ticker)
        st.metric("Latest Filing Year", bundle.target.latest_filing_year or "Unknown")
        st.metric("Comparison Status", report.status)
    with overview_right:
        st.metric("Top K", bundle.run_metadata.top_k)
        st.metric("Matches", len(bundle.matches))
        st.metric("Run Timestamp", bundle.run_metadata.run_timestamp)

    if artifact_path:
        st.caption(f"Artifact: {artifact_path}")

    st.markdown("**Step Status**")
    _render_status_map(bundle.run_metadata.status_by_step)

    if bundle.run_metadata.warnings:
        st.warning("\n".join(bundle.run_metadata.warnings))
    if report.error:
        st.error(report.error)

    st.subheader("Comparison Summary")
    if report.posture:
        st.markdown(f"**Posture:** {report.posture.label}")
        for bullet in report.posture.rationale_bullets:
            st.write(f"- {bullet}")
    st.write(report.summary or "No summary available.")

    st.subheader("Target Profile")
    _render_target_profile(report.target_profile)

    st.subheader("Peer Comparison")
    if bundle.matches:
        st.markdown("**Matches**")
        st.table([match.model_dump(mode="json") for match in bundle.matches])
    else:
        st.info("No peer matches available.")

    if report.peer_snapshot:
        st.markdown("**Peer Snapshot**")
        st.json(report.peer_snapshot.model_dump(mode="json"), expanded=False)

    st.subheader("Risk Tables")
    st.markdown("**Risk Overlap**")
    st.table([row.model_dump(mode="json") for row in report.risk_overlap_rows] or [{"group": "-", "risk_types": []}])
    st.markdown("**Forward Watchlist**")
    st.table([item.model_dump(mode="json") for item in report.forward_watchlist] or [{"watch_risk_type": "-", "why_relevant": "-", "peer_evidence": [], "confidence": "low"}])

    st.subheader("Narrative")
    if report.narrative_sections:
        for section in report.narrative_sections:
            st.markdown(f"**{section.title}**")
            st.write(section.content)
    else:
        st.info("No narrative sections available.")

    with st.expander("Raw Orchestration JSON"):
        st.json(artifact.model_dump(mode="json"), expanded=False)


def main() -> None:
    st.set_page_config(page_title="FinAgent UI", layout="wide")
    st.title("FinAgent")
    st.caption("Local UI for company comparison, dataset intake, and FAISS index management.")

    with st.sidebar:
        st.header("Analyze Company")
        analyze_ticker = st.text_input("Ticker", value="AAPL", key="analyze_ticker").strip().upper()
        top_k = st.number_input("Top K Matches", min_value=1, max_value=10, value=2, step=1)
        if st.button("Run Analysis", use_container_width=True):
            with st.spinner(f"Running comparison for {analyze_ticker}..."):
                artifact, output_path = run_analysis_for_ui(analyze_ticker, top_k=int(top_k))
            st.session_state["current_artifact"] = artifact.model_dump(mode="json")
            st.session_state["current_artifact_path"] = str(output_path)

        saved_artifacts = list_saved_report_artifacts()
        options = ["None"] + [item["label"] for item in saved_artifacts]
        selected_label = st.selectbox("Browse Saved Reports", options=options)
        if selected_label != "None":
            selected = next(item for item in saved_artifacts if item["label"] == selected_label)
            if st.button("Load Saved Report", use_container_width=True):
                artifact = load_saved_report_artifact(selected["path"])
                st.session_state["current_artifact"] = artifact.model_dump(mode="json")
                st.session_state["current_artifact_path"] = selected["path"]

        st.divider()
        st.header("Add Company to Dataset")
        dataset_ticker = st.text_input("New Company Ticker", value="", key="dataset_ticker").strip().upper()
        rebuild_after_add = st.checkbox("Rebuild FAISS after add", value=False)
        if st.button("Build Dataset Artifacts", use_container_width=True, disabled=not dataset_ticker):
            with st.spinner(f"Building dataset artifacts for {dataset_ticker}..."):
                build_result = build_company_dataset_for_ui(dataset_ticker)
            st.session_state["dataset_result"] = build_result
            if rebuild_after_add and not build_result.get("error"):
                with st.spinner("Rebuilding FAISS index..."):
                    st.session_state["index_status"] = rebuild_faiss_index()

        st.divider()
        st.header("FAISS Index")
        if st.button("Rebuild FAISS Index", use_container_width=True):
            with st.spinner("Rebuilding FAISS index..."):
                st.session_state["index_status"] = rebuild_faiss_index()

    current_artifact_payload = st.session_state.get("current_artifact")
    current_artifact_path = st.session_state.get("current_artifact_path")

    left, right = st.columns([2, 1])

    with left:
        if current_artifact_payload:
            artifact = OrchestrationArtifact.model_validate(current_artifact_payload)
            _render_report(artifact, current_artifact_path)
        else:
            st.info("Run an analysis or load a saved report to view the comparison output.")

    with right:
        st.subheader("Dataset Intake Status")
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

        st.subheader("Current Ticker Dataset")
        ticker_for_status = dataset_ticker or analyze_ticker
        st.json(get_ticker_dataset_status(ticker_for_status), expanded=False)

        st.subheader("FAISS Index Status")
        index_status = st.session_state.get("index_status") or get_faiss_index_status()
        if not index_status.get("index_exists"):
            st.warning("FAISS index file is missing.")
        st.json(
            {
                "index_exists": index_status.get("index_exists"),
                "metadata_exists": index_status.get("metadata_exists"),
                "entry_count": index_status.get("entry_count"),
                "index_path": index_status.get("index_path"),
                "metadata_path": index_status.get("metadata_path"),
            },
            expanded=False,
        )
        with st.expander("Raw FAISS Metadata"):
            st.json(index_status.get("metadata") or index_status.get("build_payload") or {}, expanded=False)


if __name__ == "__main__":
    main()
