from __future__ import annotations

from typing import Any

from .models import EvaluationInput, EvaluationScore


POSITIVE_LABELS = {"strong_growth", "moderate_growth"}
NEGATIVE_LABELS = {"moderate_decline", "severe_decline"}
FORECAST_TERMS = ("will", "expect", "forecast", "project", "likely to", "should")


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, round(value, 4)))


def _shared_row(evaluation_input: EvaluationInput, group: str) -> list[str]:
    for row in evaluation_input.risk_overlap_rows:
        if row.group == group:
            return row.risk_types
    return []


def _score_consistency(evaluation_input: EvaluationInput) -> tuple[float, list[str]]:
    checks: list[bool] = []
    warnings: list[str] = []
    positives = sum(1 for fact in evaluation_input.deterministic_financial_facts if fact.label in POSITIVE_LABELS)
    negatives = sum(1 for fact in evaluation_input.deterministic_financial_facts if fact.label in NEGATIVE_LABELS)
    high_risks = sum(1 for risk in evaluation_input.target_risks if risk.severity == "high")

    checks.append(bool(evaluation_input.summary_text.strip()))
    if not checks[-1]:
        warnings.append("Summary is empty.")

    profile_present = bool(evaluation_input.target_risks or evaluation_input.deterministic_financial_facts)
    checks.append(evaluation_input.report_status != "completed" or profile_present)
    if not checks[-1]:
        warnings.append("Completed report is missing target profile evidence.")

    if evaluation_input.posture_label == "Stable":
        plausible = high_risks == 0 and negatives <= positives
        checks.append(plausible)
        if not plausible:
            warnings.append("Stable posture conflicts with current risks or negative deltas.")
    elif evaluation_input.posture_label == "Elevated":
        plausible = high_risks > 0 or negatives > positives
        checks.append(plausible)
        if not plausible:
            warnings.append("Elevated posture is not supported by the current deterministic facts.")
    elif evaluation_input.posture_label == "Mixed":
        plausible = (positives > 0 and negatives > 0) or high_risks > 0
        checks.append(plausible)
        if not plausible:
            warnings.append("Mixed posture lacks balancing evidence.")

    shared_now = set(_shared_row(evaluation_input, "shared_now"))
    target_only_now = set(_shared_row(evaluation_input, "target_only_now"))
    overlap_ok = not (shared_now & target_only_now)
    checks.append(overlap_ok)
    if not overlap_ok:
        warnings.append("Risk overlap rows contain the same risk types in shared and target-only buckets.")

    shared_alignment = shared_now.issubset(set(evaluation_input.shared_risk_types) | shared_now)
    checks.append(shared_alignment)
    if not shared_alignment:
        warnings.append("Shared risk overlap rows do not align with peer snapshot shared risks.")

    watch_ok = all(item.peer_evidence for item in evaluation_input.forward_watchlist)
    checks.append(watch_ok or not evaluation_input.forward_watchlist)
    if not checks[-1]:
        warnings.append("Watchlist items are missing peer evidence.")

    score = sum(1 for item in checks if item) / len(checks) if checks else 1.0
    return _clamp(score), warnings


def _score_coverage(evaluation_input: EvaluationInput) -> tuple[float, list[str]]:
    warnings: list[str] = []
    section_score = (
        sum(1 for section in evaluation_input.narrative_sections if section.citations) / len(evaluation_input.narrative_sections)
        if evaluation_input.narrative_sections
        else 0.0
    )
    risk_score = (
        sum(1 for risk in evaluation_input.target_risks if risk.citation or risk.summary) / len(evaluation_input.target_risks)
        if evaluation_input.target_risks
        else 0.0
    )
    watch_score = (
        sum(1 for item in evaluation_input.forward_watchlist if item.peer_evidence) / len(evaluation_input.forward_watchlist)
        if evaluation_input.forward_watchlist
        else 1.0
    )
    if section_score == 0.0 and evaluation_input.narrative_sections:
        warnings.append("Narrative sections do not carry explicit citations.")
    score = (section_score + risk_score + watch_score) / 3
    return _clamp(score), warnings


def _score_overreach(evaluation_input: EvaluationInput, *, report_judgement: dict[str, Any] | None = None) -> tuple[float, list[str]]:
    penalty = 0.0
    warnings: list[str] = []

    if any(not item.peer_evidence for item in evaluation_input.forward_watchlist):
        penalty += 0.15
        warnings.append("Watchlist overreaches beyond its evidence.")

    report_text = " ".join(
        [evaluation_input.summary_text, *evaluation_input.posture_bullets, *(section.content for section in evaluation_input.narrative_sections)]
    ).lower()
    if any(term in report_text for term in FORECAST_TERMS) and not evaluation_input.peer_evidence_pool:
        penalty += 0.1
        warnings.append("Forecast-like language appears without supporting peer evidence.")

    if "shared" in report_text and not _shared_row(evaluation_input, "shared_now"):
        penalty += 0.1
        warnings.append("Report claims shared peer patterns without overlap evidence.")

    if report_judgement:
        level = str(report_judgement.get("comparative_usefulness", "")).strip().lower()
        if level == "weak":
            penalty += 0.05
        penalty += 0.05 * len(report_judgement.get("overreach_flags", []) or [])

    return _clamp(penalty, high=0.5), warnings


def score_evaluation_input(
    evaluation_input: EvaluationInput,
    *,
    claim_support: float | None = None,
    comparative_usefulness: float | None = None,
    report_judgement: dict[str, Any] | None = None,
) -> tuple[EvaluationScore, list[str]]:
    consistency, consistency_warnings = _score_consistency(evaluation_input)
    coverage, coverage_warnings = _score_coverage(evaluation_input)
    overreach, overreach_warnings = _score_overreach(evaluation_input, report_judgement=report_judgement)
    warnings = [*evaluation_input.warnings, *consistency_warnings, *coverage_warnings, *overreach_warnings]

    weighted_total = (0.25 * consistency) + (0.20 * coverage)
    weight_sum = 0.45
    if claim_support is not None:
        weighted_total += 0.30 * claim_support
        weight_sum += 0.30
    if comparative_usefulness is not None:
        weighted_total += 0.15 * comparative_usefulness
        weight_sum += 0.15

    overall = (weighted_total / weight_sum) - overreach if weight_sum else 0.0
    score = EvaluationScore(
        deterministic_consistency=consistency,
        evidence_coverage=coverage,
        claim_support=None if claim_support is None else _clamp(claim_support),
        comparative_usefulness=None if comparative_usefulness is None else _clamp(comparative_usefulness),
        overreach_penalty=overreach,
        overall_score=_clamp(overall),
    )
    return score, warnings
