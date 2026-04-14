from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from baseline_rag import flatten_ingestion_to_text
from orchestration.artifact_resolver import ingestion_artifact_path
from orchestration.report_models import OrchestrationArtifact

from .models import (
    DeterministicFact,
    EvaluationInput,
    EvaluationOverlapRow,
    EvaluationRiskItem,
    EvaluationSection,
    EvaluationWatchItem,
    EvidenceItem,
)


def load_artifact(path: str | Path) -> OrchestrationArtifact:
    payload = Path(path).read_text(encoding="utf-8")
    return OrchestrationArtifact.model_validate_json(payload)


def discover_artifact_paths(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file():
        return [root_path.resolve()]
    return sorted(path.resolve() for path in root_path.rglob("*.json"))


def _artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _pipeline_name(schema_version: str) -> str:
    return "baseline" if schema_version.startswith("baseline_rag") else "agentic"


def _repo_root_from_ingestion_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).resolve()
    parts = candidate.parts
    try:
        idx = parts.index("data-ingestion")
    except ValueError:
        return None
    return Path(*parts[:idx])


def _load_curator_facts(curator_payload: dict[str, Any] | None) -> tuple[list[DeterministicFact], list[EvidenceItem]]:
    if not curator_payload:
        return [], []

    facts: list[DeterministicFact] = []
    evidence: list[EvidenceItem] = []
    for metric, info in (curator_payload.get("financial_deltas") or {}).items():
        label = str(info.get("label", "unknown"))
        value = info.get("value")
        facts.append(
            DeterministicFact(
                metric=str(metric),
                label=label,
                value=float(value) if value is not None else None,
                source="curator",
            )
        )
        evidence.append(
            EvidenceItem(
                source_type="target_curator_delta",
                text=f"{metric}: {label}" + (f" ({value})" if value is not None else ""),
                metadata={"metric": str(metric), "label": label},
            )
        )

    for signal in curator_payload.get("risk_signals", []):
        summary = str(signal.get("summary", "")).strip()
        if summary:
            evidence.append(
                EvidenceItem(
                    source_type="target_curator_risk",
                    text=summary,
                    citations=[str(signal.get("citation", "")).strip()] if signal.get("citation") else [],
                    metadata={
                        "signal_type": str(signal.get("signal_type", "")),
                        "severity": str(signal.get("severity", "")),
                    },
                )
            )
    return facts, evidence


def _load_ingestion_facts(ingestion_payload: dict[str, Any] | None) -> tuple[list[DeterministicFact], list[EvidenceItem]]:
    if not ingestion_payload:
        return [], []

    facts: list[DeterministicFact] = []
    annual = ingestion_payload.get("financial_data", {}).get("annual", {})
    for metric, payload in annual.items():
        years = payload.get("years") or []
        values = payload.get("values") or []
        deltas = payload.get("deltas") or []
        if not years:
            continue
        idx = len(years) - 1
        value = values[idx] if idx < len(values) else None
        delta = deltas[idx] if idx < len(deltas) else None
        label = "stable"
        if delta is None:
            label = "insufficient_data"
        elif delta >= 20:
            label = "strong_growth"
        elif delta >= 5:
            label = "moderate_growth"
        elif delta <= -20:
            label = "severe_decline"
        elif delta <= -5:
            label = "moderate_decline"
        facts.append(
            DeterministicFact(
                metric=str(metric),
                label=label,
                value=float(delta) if delta is not None else None,
                source="ingestion",
            )
        )
    return facts, [EvidenceItem(source_type="target_ingestion_text", text=flatten_ingestion_to_text(ingestion_payload))]


def build_evaluation_input(path: str | Path) -> EvaluationInput:
    artifact_path = Path(path).resolve()
    artifact = load_artifact(artifact_path)
    pipeline = _pipeline_name(artifact.schema_version)
    report = artifact.comparison_report
    bundle = artifact.bundle

    target_evidence_pool: list[EvidenceItem] = []
    peer_evidence_pool: list[EvidenceItem] = []
    deterministic_facts: list[DeterministicFact] = []

    if pipeline == "agentic":
        curator_payload = _read_json(bundle.target.curator_path)
        facts, curator_evidence = _load_curator_facts(curator_payload)
        deterministic_facts.extend(facts)
        target_evidence_pool.extend(curator_evidence)
    else:
        ingestion_payload = _read_json(bundle.target.ingestion_path)
        facts, ingestion_evidence = _load_ingestion_facts(ingestion_payload)
        deterministic_facts.extend(facts)
        target_evidence_pool.extend(ingestion_evidence)
        repo_root = _repo_root_from_ingestion_path(bundle.target.ingestion_path)
        for match in bundle.matches:
            peer_path = ingestion_artifact_path(match.ticker, repo_root=repo_root) if repo_root else None
            peer_payload = _read_json(peer_path)
            if peer_payload:
                peer_evidence_pool.append(
                    EvidenceItem(
                        source_type="peer_ingestion_text",
                        text=flatten_ingestion_to_text(peer_payload),
                        metadata={"ticker": match.ticker, "similarity": match.similarity},
                    )
                )
            peer_evidence_pool.append(
                EvidenceItem(
                    source_type="peer_match_metadata",
                    text=f"{match.ticker} matched filing year {match.matched_filing_year} at similarity {match.similarity:.3f}",
                    metadata={"ticker": match.ticker, "matched_filing_year": match.matched_filing_year},
                )
            )

    for risk in report.target_profile.top_risks if report.target_profile else []:
        if risk.summary:
            target_evidence_pool.append(
                EvidenceItem(
                    source_type="report_target_risk",
                    text=risk.summary,
                    citations=[risk.citation] if risk.citation else [],
                    metadata={"signal_type": risk.signal_type, "severity": risk.severity},
                )
            )

    for section in report.narrative_sections:
        if section.citations:
            target_evidence_pool.append(
                EvidenceItem(
                    source_type="narrative_citations",
                    text=f"{section.title}: {' | '.join(section.citations)}",
                    citations=section.citations,
                )
            )

    for item in report.forward_watchlist:
        for evidence in item.peer_evidence:
            peer_evidence_pool.append(
                EvidenceItem(
                    source_type="watch_peer_evidence",
                    text=evidence,
                    metadata={"watch_risk_type": item.watch_risk_type, "confidence": item.confidence},
                )
            )

    if report.peer_snapshot:
        for value in report.peer_snapshot.common_strengths:
            peer_evidence_pool.append(EvidenceItem(source_type="peer_common_strength", text=value))
        for value in report.peer_snapshot.common_pressures:
            peer_evidence_pool.append(EvidenceItem(source_type="peer_common_pressure", text=value))
        for value in report.peer_snapshot.shared_risk_types:
            peer_evidence_pool.append(EvidenceItem(source_type="peer_shared_risk", text=value))
        for value in report.peer_snapshot.target_differences:
            peer_evidence_pool.append(EvidenceItem(source_type="peer_target_difference", text=value))

    return EvaluationInput(
        pipeline=pipeline,
        ticker=bundle.target.ticker,
        company=bundle.target.company,
        artifact_path=str(artifact_path),
        artifact_hash=_artifact_hash(artifact_path),
        schema_version=artifact.schema_version,
        report_status=report.status,
        summary_text=report.summary,
        posture_label=report.posture.label if report.posture else None,
        posture_bullets=report.posture.rationale_bullets if report.posture else [],
        target_risks=[
            EvaluationRiskItem(
                signal_type=risk.signal_type,
                severity=risk.severity,
                section=risk.section,
                summary=risk.summary,
                citation=risk.citation,
                occurrences=risk.occurrences,
            )
            for risk in (report.target_profile.top_risks if report.target_profile else [])
        ],
        risk_overlap_rows=[EvaluationOverlapRow(group=row.group, risk_types=row.risk_types) for row in report.risk_overlap_rows],
        forward_watchlist=[
            EvaluationWatchItem(
                watch_risk_type=item.watch_risk_type,
                why_relevant=item.why_relevant,
                peer_evidence=item.peer_evidence,
                confidence=item.confidence,
            )
            for item in report.forward_watchlist
        ],
        narrative_sections=[
            EvaluationSection(title=section.title, content=section.content, citations=section.citations)
            for section in report.narrative_sections
        ],
        target_evidence_pool=target_evidence_pool,
        peer_evidence_pool=peer_evidence_pool,
        deterministic_financial_facts=deterministic_facts,
        common_strengths=report.peer_snapshot.common_strengths if report.peer_snapshot else [],
        common_pressures=report.peer_snapshot.common_pressures if report.peer_snapshot else [],
        shared_risk_types=report.peer_snapshot.shared_risk_types if report.peer_snapshot else [],
        target_differences=report.peer_snapshot.target_differences if report.peer_snapshot else [],
        warnings=list(bundle.run_metadata.warnings),
    )
