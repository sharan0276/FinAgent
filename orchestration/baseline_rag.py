"""Baseline RAG pipeline: single-shot LLM call over ingested financial data.

This is a deliberate contrast to the multi-step agentic orchestration pipeline.
Same model, same output schema — only the retrieval strategy and reasoning depth differ.
This design choice isolates pipeline architecture as the variable, keeping the model constant.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from .artifact_resolver import REPO_ROOT, ingestion_artifact_path
from .openrouter_client import OpenRouterClient
from .report_models import (
    ComparisonBundle,
    ComparisonReportResult,
    ForwardWatchItem,
    MatchContext,
    MetricDeltaItem,
    OrchestrationArtifact,
    PeerSnapshot,
    PostureCard,
    ReportSection,
    RiskItem,
    RiskOverlapRow,
    RunMetadata,
    TargetContext,
    TargetProfile,
)


# ---------------------------------------------------------------------------
# Ingestion data helpers
# ---------------------------------------------------------------------------

def _load_ingestion(ticker: str, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    path = ingestion_artifact_path(ticker, repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(f"No ingestion artifact found for {ticker} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_usd(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.0f}M"


def _format_delta(delta: Optional[float]) -> str:
    if delta is None:
        return ""
    sign = "+" if delta >= 0 else ""
    return f" ({sign}{delta:.1f}% YoY)"


def flatten_ingestion_to_text(ingestion: dict[str, Any]) -> str:
    """Convert ingestion JSON to a compact, LLM-readable text block.

    Preserves the structure of financial metrics across years while being
    token-efficient. Intentionally does not include NLP/sentiment signals —
    that is what the agentic pipeline's extraction and curator steps add.
    """
    ticker = ingestion.get("ticker", "UNKNOWN")
    company = ingestion.get("company_name", ticker)
    annual = ingestion.get("financial_data", {}).get("annual", {})

    lines: list[str] = [f"=== {ticker} ({company}) — Ingested Financial Data ==="]

    for metric, data in annual.items():
        years: list = data.get("years", [])
        values: list = data.get("values", [])
        deltas: list = data.get("deltas", [])
        unit: str = data.get("unit", "USD_millions")

        if not years:
            continue

        lines.append(f"\n{metric}:")
        for i, year in enumerate(years):
            val = values[i] if i < len(values) else None
            delta = deltas[i] if i < len(deltas) else None
            val_str = _format_usd(val) if "USD" in unit else (str(val) if val is not None else "N/A")
            delta_str = _format_delta(delta)
            lines.append(f"  FY{year}: {val_str}{delta_str}")

    return "\n".join(lines)


def _latest_year_from_ingestion(ingestion: dict[str, Any]) -> Optional[int]:
    annual = ingestion.get("financial_data", {}).get("annual", {})
    all_years: list[int] = []
    for data in annual.values():
        all_years.extend(data.get("years", []))
    return max(all_years) if all_years else None


# ---------------------------------------------------------------------------
# Peer retrieval
# ---------------------------------------------------------------------------

def _ensure_rag_path(repo_root: Path) -> None:
    rag_dir = str(repo_root / "rag-matching")
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)


def _find_peers_via_faiss(
    ticker: str,
    top_k: int,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Use existing FAISS index to find semantically similar peers.

    Requires a curator file for the target ticker. Uses the most recent
    curator as the query embedding, consistent with the agentic pipeline.
    """
    _ensure_rag_path(repo_root)
    from matcher import find_matches  # type: ignore[import]

    curator_root = repo_root / "data-extraction" / "outputs" / "curator" / ticker.upper()
    curator_files = sorted(curator_root.glob("*.json")) if curator_root.exists() else []
    if not curator_files:
        return []

    query_file = curator_files[-1]
    artifact_dir = repo_root / "rag-matching" / "index_artifacts"
    result = find_matches(query_file, top_k=top_k, artifact_dir=artifact_dir)
    return result.get("matches", [])


def _find_peers_fallback(
    ticker: str,
    top_k: int,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Fallback peer discovery when no FAISS curator exists for the target.

    Returns available tickers from ingestion outputs, sorted alphabetically.
    Noted as zero-similarity since no embedding comparison was performed.
    """
    ingestion_root = repo_root / "data-ingestion" / "outputs"
    available = [
        d.name
        for d in sorted(ingestion_root.iterdir())
        if d.is_dir()
        and d.name.upper() != ticker.upper()
        and (d / "complete_ingestion.json").exists()
    ]
    return [
        {"ticker": t, "filing_year": None, "similarity": 0.0}
        for t in available[:top_k]
    ]


def find_peers(
    ticker: str,
    top_k: int,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    """Find top-k peer companies. Tries FAISS first, falls back to ingestion scan."""
    peers = _find_peers_via_faiss(ticker, top_k, repo_root=repo_root)
    if not peers:
        peers = _find_peers_fallback(ticker, top_k, repo_root=repo_root)
    return peers[:top_k]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior financial analyst. You will receive structured financial data \
for a target company and its peers, extracted directly from SEC filings.

Your task is to produce a concise but thorough financial comparison report in JSON.

The JSON must exactly match this schema (no extra keys, no missing required keys):
{
  "status": "completed",
  "summary": "<2-3 sentence overall assessment grounded in the numbers>",
  "posture": {
    "label": "Elevated" | "Mixed" | "Stable",
    "rationale_bullets": ["<bullet>", "<bullet>", "<bullet>"]
  },
  "target_profile": {
    "ticker": "<ticker>",
    "company": "<company name>",
    "filing_year": <most recent year as integer>,
    "positive_deltas": [
      {"metric": "<metric name>", "label": "<short label>", "value": <float or null>}
    ],
    "negative_deltas": [
      {"metric": "<metric name>", "label": "<short label>", "value": <float or null>}
    ],
    "top_risks": [
      {
        "signal_type": "<risk category>",
        "severity": "high" | "medium" | "low",
        "section": "<context>",
        "summary": "<one sentence>",
        "occurrences": 1
      }
    ]
  },
  "peer_snapshot": {
    "peer_group": "<description of the peer group>",
    "common_strengths": ["<strength>"],
    "common_pressures": ["<pressure>"],
    "shared_risk_types": ["<risk type>"],
    "target_differences": ["<how target differs from peers>"]
  },
  "risk_overlap_rows": [
    {"group": "shared_now", "risk_types": ["<type>"]},
    {"group": "target_only_now", "risk_types": ["<type>"]},
    {"group": "peer_only_now", "risk_types": ["<type>"]}
  ],
  "forward_watchlist": [
    {
      "watch_risk_type": "<type>",
      "why_relevant": "<reason>",
      "peer_evidence": ["<evidence>"],
      "confidence": "high" | "medium" | "low"
    }
  ],
  "narrative_sections": [
    {"title": "<section title>", "content": "<2-3 paragraph narrative>"}
  ],
  "error": null
}

Rules:
- Base all claims directly on the numbers provided. Do not fabricate figures.
- If data is missing or a field cannot be determined, use null for numeric fields.
- Produce at least 2 narrative sections: "Financial Performance" and "Risk Assessment".
- Posture label meanings: Elevated = significant risks/deterioration, Stable = strong/improving, Mixed = balanced.
"""


def _build_user_prompt(
    target_text: str,
    peer_texts: list[str],
    focus_query: Optional[str],
) -> str:
    focus_block = f"\n**Analyst Focus Area:** {focus_query}\nPrioritize this theme in your analysis.\n" if focus_query else ""
    peer_block = "\n\n".join(peer_texts) if peer_texts else "No peer data available for comparison."

    return f"""{focus_block}
## Target Company — Financial Data
{target_text}

## Peer Companies — Financial Data
{peer_block}

Produce the JSON comparison report now.
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _safe_str(val: Any, default: str = "") -> str:
    return str(val) if val is not None else default


def _parse_posture(raw: Optional[dict]) -> Optional[PostureCard]:
    if not raw:
        return None
    label = raw.get("label", "Mixed")
    if label not in ("Elevated", "Mixed", "Stable"):
        label = "Mixed"
    return PostureCard(label=label, rationale_bullets=raw.get("rationale_bullets", []))


def _parse_metric_items(items: Optional[list]) -> List[MetricDeltaItem]:
    result: list[MetricDeltaItem] = []
    for item in (items or []):
        try:
            result.append(MetricDeltaItem(
                metric=_safe_str(item.get("metric")),
                label=_safe_str(item.get("label")),
                value=float(item["value"]) if item.get("value") is not None else None,
            ))
        except (TypeError, ValueError, KeyError):
            continue
    return result


def _parse_risks(items: Optional[list]) -> List[RiskItem]:
    result: list[RiskItem] = []
    for item in (items or []):
        severity = item.get("severity", "medium")
        if severity not in ("high", "medium", "low"):
            severity = "medium"
        try:
            result.append(RiskItem(
                signal_type=_safe_str(item.get("signal_type")),
                severity=severity,
                section=item.get("section"),
                summary=item.get("summary"),
                occurrences=int(item.get("occurrences", 1)),
            ))
        except (TypeError, ValueError):
            continue
    return result


def _parse_target_profile(
    raw: Optional[dict],
    ticker: str,
    ingestion: dict[str, Any],
) -> Optional[TargetProfile]:
    filing_year = _latest_year_from_ingestion(ingestion) or 0
    if not raw:
        return TargetProfile(ticker=ticker, filing_year=filing_year)
    try:
        return TargetProfile(
            ticker=_safe_str(raw.get("ticker"), ticker),
            company=raw.get("company"),
            filing_year=int(raw.get("filing_year") or filing_year),
            positive_deltas=_parse_metric_items(raw.get("positive_deltas")),
            negative_deltas=_parse_metric_items(raw.get("negative_deltas")),
            top_risks=_parse_risks(raw.get("top_risks")),
        )
    except (TypeError, ValueError):
        return TargetProfile(ticker=ticker, filing_year=filing_year)


def _parse_peer_snapshot(raw: Optional[dict]) -> Optional[PeerSnapshot]:
    if not raw:
        return None
    return PeerSnapshot(
        peer_group=_safe_str(raw.get("peer_group")),
        common_strengths=raw.get("common_strengths") or [],
        common_pressures=raw.get("common_pressures") or [],
        shared_risk_types=raw.get("shared_risk_types") or [],
        target_differences=raw.get("target_differences") or [],
    )


def _parse_risk_overlap(rows: Optional[list]) -> List[RiskOverlapRow]:
    valid_groups = {"shared_now", "target_only_now", "peer_only_now"}
    result: list[RiskOverlapRow] = []
    for row in (rows or []):
        group = row.get("group", "")
        if group not in valid_groups:
            continue
        result.append(RiskOverlapRow(group=group, risk_types=row.get("risk_types") or []))
    return result


def _parse_watchlist(items: Optional[list]) -> List[ForwardWatchItem]:
    result: list[ForwardWatchItem] = []
    for item in (items or []):
        confidence = item.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        try:
            result.append(ForwardWatchItem(
                watch_risk_type=_safe_str(item.get("watch_risk_type")),
                why_relevant=_safe_str(item.get("why_relevant")),
                peer_evidence=item.get("peer_evidence") or [],
                confidence=confidence,
            ))
        except (TypeError, ValueError):
            continue
    return result


def _parse_narrative(sections: Optional[list]) -> List[ReportSection]:
    result: list[ReportSection] = []
    for section in (sections or []):
        try:
            result.append(ReportSection(
                title=_safe_str(section.get("title")),
                content=_safe_str(section.get("content")),
            ))
        except (TypeError, ValueError):
            continue
    return result


def _parse_report(
    raw: dict[str, Any],
    ticker: str,
    ingestion: dict[str, Any],
    model_name: str,
) -> ComparisonReportResult:
    status = raw.get("status", "completed")
    if status not in ("completed", "failed", "skipped"):
        status = "completed"

    return ComparisonReportResult(
        status=status,
        summary=_safe_str(raw.get("summary"), "No summary available."),
        model_name=model_name,
        posture=_parse_posture(raw.get("posture")),
        target_profile=_parse_target_profile(raw.get("target_profile"), ticker, ingestion),
        peer_snapshot=_parse_peer_snapshot(raw.get("peer_snapshot")),
        risk_overlap_rows=_parse_risk_overlap(raw.get("risk_overlap_rows")),
        forward_watchlist=_parse_watchlist(raw.get("forward_watchlist")),
        narrative_sections=_parse_narrative(raw.get("narrative_sections")),
        error=raw.get("error"),
    )


# ---------------------------------------------------------------------------
# Failed artifact helper
# ---------------------------------------------------------------------------

def _make_failed_artifact(
    ticker: str,
    top_k: int,
    run_ts: str,
    error: str,
    status_by_step: dict[str, str],
    warnings: Optional[list[str]] = None,
) -> OrchestrationArtifact:
    return OrchestrationArtifact(
        schema_version="baseline_rag_v1",
        bundle=ComparisonBundle(
            target=TargetContext(ticker=ticker),
            matches=[],
            run_metadata=RunMetadata(
                top_k=top_k,
                run_timestamp=run_ts,
                status_by_step=status_by_step,
                warnings=warnings or [],
            ),
        ),
        comparison_report=ComparisonReportResult(
            status="failed",
            summary="Baseline RAG pipeline failed.",
            error=error,
        ),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_baseline_rag(
    ticker: str,
    *,
    top_k: int = 2,
    focus_query: Optional[str] = None,
    repo_root: Path = REPO_ROOT,
    client: Optional[OpenRouterClient] = None,
) -> OrchestrationArtifact:
    """Run the baseline RAG pipeline for a ticker.

    Strategy:
    1. Load ingested financial data for target (structured JSON, no NLP signals)
    2. Find top-k peers via FAISS (if curator available) or ingestion scan
    3. Flatten all data to structured text
    4. Single LLM call → parse into OrchestrationArtifact

    The schema_version "baseline_rag_v1" distinguishes this from the agentic
    pipeline ("orchestration_v1") so results can be compared in the UI.
    """
    ticker = ticker.strip().upper()
    run_ts = datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []
    status_by_step: dict[str, str] = {}

    # --- Step 1: Load target ingestion ---
    try:
        target_ingestion = _load_ingestion(ticker, repo_root=repo_root)
        status_by_step["load_target"] = "completed"
    except FileNotFoundError as exc:
        status_by_step["load_target"] = "failed"
        return _make_failed_artifact(ticker, top_k, run_ts, str(exc), status_by_step)

    company_name: Optional[str] = target_ingestion.get("company_name")
    latest_year = _latest_year_from_ingestion(target_ingestion)
    target_text = flatten_ingestion_to_text(target_ingestion)
    status_by_step["flatten_target"] = "completed"

    # --- Step 2: Find peers ---
    try:
        peers = find_peers(ticker, top_k, repo_root=repo_root)
        retrieval_method = "faiss" if any(p.get("similarity", 0.0) > 0 for p in peers) else "fallback"
        status_by_step["find_peers"] = "completed"
        if retrieval_method == "fallback":
            warnings.append("No curator file for target — used ingestion scan for peer discovery (no embedding similarity).")
    except Exception as exc:
        warnings.append(f"Peer retrieval failed: {exc}")
        peers = []
        status_by_step["find_peers"] = "partial"

    # --- Step 3: Load and flatten peer data ---
    peer_texts: list[str] = []
    match_contexts: list[MatchContext] = []

    for peer in peers:
        peer_ticker = str(peer.get("ticker", "")).upper()
        if not peer_ticker:
            continue
        try:
            peer_ingestion = _load_ingestion(peer_ticker, repo_root=repo_root)
            peer_texts.append(flatten_ingestion_to_text(peer_ingestion))
            peer_year = _latest_year_from_ingestion(peer_ingestion) or int(peer.get("filing_year") or 0)
            match_contexts.append(MatchContext(
                ticker=peer_ticker,
                company=peer_ingestion.get("company_name"),
                matched_filing_year=peer_year,
                similarity=float(peer.get("similarity", 0.0)),
            ))
        except Exception as exc:
            warnings.append(f"Could not load peer {peer_ticker}: {exc}")

    status_by_step["load_peers"] = "completed" if peer_texts else "partial"

    # --- Step 4: Single LLM call ---
    llm_client = client or OpenRouterClient()
    user_prompt = _build_user_prompt(target_text, peer_texts, focus_query)

    try:
        raw_report, model_name = llm_client.complete_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        status_by_step["llm_call"] = "completed"
    except Exception as exc:
        status_by_step["llm_call"] = "failed"
        return _make_failed_artifact(
            ticker, top_k, run_ts, str(exc), status_by_step, warnings=warnings
        )

    # --- Step 5: Parse into typed report model ---
    try:
        report = _parse_report(raw_report, ticker, target_ingestion, model_name)
        status_by_step["parse_report"] = "completed"
    except Exception as exc:
        warnings.append(f"Report parsing error: {exc}")
        report = ComparisonReportResult(
            status="failed",
            summary="Report parsing failed after LLM call.",
            error=str(exc),
        )
        status_by_step["parse_report"] = "failed"

    bundle = ComparisonBundle(
        target=TargetContext(
            ticker=ticker,
            company=company_name,
            latest_filing_year=latest_year,
            ingestion_path=str(ingestion_artifact_path(ticker, repo_root=repo_root)),
        ),
        matches=match_contexts,
        run_metadata=RunMetadata(
            top_k=top_k,
            run_timestamp=run_ts,
            status_by_step=status_by_step,
            warnings=warnings,
        ),
    )

    return OrchestrationArtifact(
        schema_version="baseline_rag_v1",
        bundle=bundle,
        comparison_report=report,
    )