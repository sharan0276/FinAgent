from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .report_models import (
    ComparisonBundle,
    ComparisonReportResult,
    ForwardWatchItem,
    MetricDeltaItem,
    PeerSnapshot,
    PostureCard,
    ReportSection,
    RiskItem,
    RiskOverlapRow,
    TargetProfile,
)
from .openrouter_client import OpenRouterClient, OpenRouterError


POSITIVE_LABELS = {"strong_growth", "moderate_growth"}
NEGATIVE_LABELS = {"moderate_decline", "severe_decline"}
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_item(metric: str, info: dict[str, Any]) -> MetricDeltaItem:
    return MetricDeltaItem(
        metric=metric,
        label=str(info.get("label", "unknown")),
        value=info.get("value"),
    )


def _unique_risk_items(curator: dict[str, Any], *, limit: int = 5) -> list[RiskItem]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signal in curator.get("risk_signals", []):
        grouped[str(signal.get("signal_type", "unknown"))].append(signal)

    ranked: list[RiskItem] = []
    for signal_type, signals in grouped.items():
        signals = sorted(signals, key=lambda item: SEVERITY_RANK.get(str(item.get("severity")), 3))
        top = signals[0]
        ranked.append(
            RiskItem(
                signal_type=signal_type,
                severity=str(top.get("severity", "unknown")),
                section=top.get("section"),
                summary=top.get("summary"),
                occurrences=len(signals),
            )
        )

    ranked.sort(key=lambda item: (SEVERITY_RANK.get(item.severity, 3), -item.occurrences, item.signal_type))
    return ranked[:limit]


def build_target_profile(curator: dict[str, Any]) -> TargetProfile:
    positives: list[MetricDeltaItem] = []
    negatives: list[MetricDeltaItem] = []
    for metric, info in curator.get("financial_deltas", {}).items():
        label = str(info.get("label"))
        item = _metric_item(metric, info)
        if label in POSITIVE_LABELS:
            positives.append(item)
        elif label in NEGATIVE_LABELS:
            negatives.append(item)

    positives.sort(key=lambda item: (0 if item.label == "strong_growth" else 1, -(item.value or 0), item.metric))
    negatives.sort(key=lambda item: (0 if item.label == "severe_decline" else 1, item.value or 0, item.metric))

    return TargetProfile(
        ticker=str(curator.get("ticker", "UNKNOWN")),
        company=curator.get("company"),
        filing_year=int(curator.get("filing_year", 0)),
        positive_deltas=positives[:4],
        negative_deltas=negatives[:4],
        top_risks=_unique_risk_items(curator),
    )


def _risk_type_set(curator: dict[str, Any]) -> set[str]:
    return {str(signal.get("signal_type")) for signal in curator.get("risk_signals", []) if signal.get("signal_type")}


def _peer_current_curators(bundle: ComparisonBundle) -> list[dict[str, Any]]:
    current_curators: list[dict[str, Any]] = []
    for match in bundle.matches:
        if match.context_curator_paths:
            current_curators.append(_load_json(match.context_curator_paths[0]))
    return current_curators


def build_risk_overlap_rows(target_curator: dict[str, Any], peer_curators: list[dict[str, Any]]) -> list[RiskOverlapRow]:
    target_risks = _risk_type_set(target_curator)
    peer_risks = set().union(*(_risk_type_set(curator) for curator in peer_curators)) if peer_curators else set()
    return [
        RiskOverlapRow(group="shared_now", risk_types=sorted(target_risks & peer_risks)),
        RiskOverlapRow(group="target_only_now", risk_types=sorted(target_risks - peer_risks)),
        RiskOverlapRow(group="peer_only_now", risk_types=sorted(peer_risks - target_risks)),
    ]


def build_peer_snapshot(target_curator: dict[str, Any], peer_curators: list[dict[str, Any]]) -> PeerSnapshot:
    metric_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    peer_risk_counts: Counter[str] = Counter()
    target_risks = _risk_type_set(target_curator)

    for curator in peer_curators:
        for metric, info in curator.get("financial_deltas", {}).items():
            label = str(info.get("label"))
            if label:
                metric_label_counts[str(metric)][label] += 1
        peer_risk_counts.update(_risk_type_set(curator))

    common_strengths = sorted(
        metric for metric, counts in metric_label_counts.items() if sum(counts[label] for label in POSITIVE_LABELS) >= 2
    )
    common_pressures = sorted(
        metric for metric, counts in metric_label_counts.items() if sum(counts[label] for label in NEGATIVE_LABELS) >= 2
    )
    shared_risk_types = sorted(risk_type for risk_type, count in peer_risk_counts.items() if count >= 2)
    target_differences = sorted(target_risks - set(peer_risk_counts))

    return PeerSnapshot(
        peer_group="Top matched peer neighborhood",
        common_strengths=common_strengths,
        common_pressures=common_pressures,
        shared_risk_types=shared_risk_types,
        target_differences=target_differences,
    )


def determine_posture(target_curator: dict[str, Any], risk_overlap_rows: list[RiskOverlapRow]) -> PostureCard:
    financials = target_curator.get("financial_deltas", {})
    negative_count = sum(1 for info in financials.values() if str(info.get("label")) in NEGATIVE_LABELS)
    positive_count = sum(1 for info in financials.values() if str(info.get("label")) in POSITIVE_LABELS)
    high_risk_count = sum(1 for signal in target_curator.get("risk_signals", []) if str(signal.get("severity")) == "high")
    shared_now = next((row.risk_types for row in risk_overlap_rows if row.group == "shared_now"), [])

    if high_risk_count >= 2 or (negative_count >= 2 and len(shared_now) >= 2):
        label = "Elevated"
    elif positive_count >= 2 and negative_count <= 1 and high_risk_count <= 1 and len(shared_now) <= 1:
        label = "Stable"
    else:
        label = "Mixed"

    bullets = [
        f"Current year shows {high_risk_count} high-severity risk signal(s) and {negative_count} negative financial delta(s).",
        f"{len(shared_now)} current risk type(s) overlap with the matched peer neighborhood.",
        f"{positive_count} metric(s) still show growth, which tempers the current pressure picture.",
    ]
    return PostureCard(label=label, rationale_bullets=bullets[:3])


def build_forward_watchlist(bundle: ComparisonBundle, target_curator: dict[str, Any]) -> list[ForwardWatchItem]:
    target_risks = _risk_type_set(target_curator)
    candidates: dict[str, dict[str, Any]] = {}

    for match in bundle.matches:
        if len(match.context_curator_paths) < 2:
            continue
        anchor = _load_json(match.context_curator_paths[0])
        anchor_risks = _risk_type_set(anchor)
        for future_path in match.context_curator_paths[1:]:
            future = _load_json(future_path)
            future_year = int(future.get("filing_year", 0))
            grouped_signals: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for signal in future.get("risk_signals", []):
                signal_type = str(signal.get("signal_type", "unknown"))
                grouped_signals[signal_type].append(signal)

            for risk_type, signals in grouped_signals.items():
                if risk_type in anchor_risks:
                    continue
                top_signal = sorted(signals, key=lambda item: SEVERITY_RANK.get(str(item.get("severity")), 3))[0]
                future_negative_metrics = [
                    metric
                    for metric, info in future.get("financial_deltas", {}).items()
                    if str(info.get("label")) in NEGATIVE_LABELS
                ]
                entry = candidates.setdefault(
                    risk_type,
                    {
                        "score": 0,
                        "peer_evidence": [],
                        "future_mentions": 0,
                    },
                )
                entry["future_mentions"] += 1
                entry["score"] += 2 if risk_type in target_risks else 1
                entry["score"] += 2 if str(top_signal.get("severity")) == "high" else 1
                entry["peer_evidence"].append(
                    f"{match.ticker} {future_year}: {str(top_signal.get('summary', '')).strip()}"
                )
                if future_negative_metrics:
                    entry["peer_evidence"].append(
                        f"{match.ticker} {future_year}: negative delta pressure in {', '.join(sorted(future_negative_metrics)[:3])}."
                    )

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-item[1]["score"], -item[1]["future_mentions"], item[0]),
    )[:5]

    watchlist: list[ForwardWatchItem] = []
    for risk_type, info in ranked:
        confidence = "high" if info["future_mentions"] >= 2 else "medium"
        why = (
            f"Peers similar to the target showed {risk_type} after the matched year, making it a pattern to monitor."
        )
        if risk_type in target_risks:
            why = f"{risk_type} is already adjacent to the target risk profile and later reappears in peer future years."
        watchlist.append(
            ForwardWatchItem(
                watch_risk_type=risk_type,
                why_relevant=why,
                peer_evidence=info["peer_evidence"][:3],
                confidence=confidence,
            )
        )
    return watchlist


def build_deterministic_report(bundle: ComparisonBundle) -> ComparisonReportResult:
    if not bundle.target.curator_path:
        return ComparisonReportResult(
            status="failed",
            summary="Comparison report could not run because the target curator artifact is missing.",
            error="missing_target_curator",
        )

    target_curator = _load_json(str(bundle.target.curator_path))
    peer_curators = _peer_current_curators(bundle)
    target_profile = build_target_profile(target_curator)
    risk_overlap_rows = build_risk_overlap_rows(target_curator, peer_curators)
    peer_snapshot = build_peer_snapshot(target_curator, peer_curators)
    posture = determine_posture(target_curator, risk_overlap_rows)
    forward_watchlist = build_forward_watchlist(bundle, target_curator)

    return ComparisonReportResult(
        status="completed",
        summary=(
            f"{target_profile.company or target_profile.ticker} shows a {posture.label.lower()} current posture "
            f"relative to its matched peer neighborhood."
        ),
        posture=posture,
        target_profile=target_profile,
        peer_snapshot=peer_snapshot,
        risk_overlap_rows=risk_overlap_rows,
        forward_watchlist=forward_watchlist,
        narrative_sections=[
            ReportSection(title="What Looks Similar", content="Peer similarity summary unavailable."),
            ReportSection(title="What To Watch Next", content="Forward watchlist summary unavailable."),
        ],
    )


def _llm_payload(report: ComparisonReportResult) -> dict[str, Any]:
    return {
        "summary": report.summary,
        "posture": {
            "label": report.posture.label if report.posture else None,
            "rationale_bullets": report.posture.rationale_bullets if report.posture else [],
        },
        "target_profile": report.target_profile.model_dump(mode="json") if report.target_profile else None,
        "peer_snapshot": report.peer_snapshot.model_dump(mode="json") if report.peer_snapshot else None,
        "risk_overlap_rows": [row.model_dump(mode="json") for row in report.risk_overlap_rows],
        "forward_watchlist": [item.model_dump(mode="json") for item in report.forward_watchlist],
        "response_schema": {
            "summary": "2-4 sentence executive summary",
            "posture_rationale_bullets": ["bullet 1", "bullet 2", "bullet 3"],
            "narrative_sections": [
                {"title": "What Looks Similar", "content": "short paragraph"},
                {"title": "What To Watch Next", "content": "short paragraph"},
            ],
        },
    }


def _apply_llm_response(report: ComparisonReportResult, payload: dict[str, Any], model_name: str) -> ComparisonReportResult:
    rationale = [str(item).strip() for item in payload.get("posture_rationale_bullets", []) if str(item).strip()]
    if report.posture and rationale:
        report.posture.rationale_bullets = rationale[:3]

    sections = []
    for section in payload.get("narrative_sections", []):
        title = str(section.get("title", "")).strip()
        content = str(section.get("content", "")).strip()
        if title and content:
            sections.append(ReportSection(title=title, content=content))

    report.summary = str(payload.get("summary", "")).strip() or report.summary
    report.model_name = model_name
    if sections:
        report.narrative_sections = sections
    return report


def _build_prompt(report: ComparisonReportResult) -> str:
    return json.dumps(
        {
            "comparison_data": _llm_payload(report),
            "instructions": {
                "goal": "Write concise explanation text for a company-profile-first comparison report.",
                "constraints": [
                    "Use only the supplied comparison data.",
                    "Do not invent any company facts or future predictions.",
                    "Treat future peer years as patterns to watch, not forecasts.",
                    "Keep wording compact and professional.",
                    "Return valid JSON only.",
                ],
            },
        },
        indent=2,
    )


def generate_comparison_report(
    bundle: ComparisonBundle,
    *,
    client: OpenRouterClient | None = None,
) -> ComparisonReportResult:
    report = build_deterministic_report(bundle)
    if report.status == "failed":
        return report

    client = client or OpenRouterClient()
    try:
        payload, model_name = client.complete_json(
            system_prompt=(
                "You are a financial analysis assistant. "
                "Polish the supplied deterministic comparison data into concise summary and narrative text. "
                "Return only valid JSON."
            ),
            user_prompt=_build_prompt(report),
        )
        return _apply_llm_response(report, payload, model_name)
    except OpenRouterError as exc:
        report.status = "failed"
        report.error = str(exc)
        report.summary = "Comparison agent failed; saved the deterministic comparison structure for inspection."
        return report
