from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PipelineName = Literal["agentic", "baseline"]
ClaimLabel = Literal["supported", "partially_supported", "unsupported"]


class EvidenceItem(BaseModel):
    source_type: str
    text: str
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRiskItem(BaseModel):
    signal_type: str
    severity: str
    section: str | None = None
    summary: str | None = None
    citation: str | None = None
    occurrences: int = 1


class EvaluationOverlapRow(BaseModel):
    group: str
    risk_types: list[str] = Field(default_factory=list)


class EvaluationWatchItem(BaseModel):
    watch_risk_type: str
    why_relevant: str
    peer_evidence: list[str] = Field(default_factory=list)
    confidence: str


class EvaluationSection(BaseModel):
    title: str
    content: str
    citations: list[str] = Field(default_factory=list)


class DeterministicFact(BaseModel):
    metric: str
    label: str
    value: float | None = None
    source: str | None = None


class EvaluationInput(BaseModel):
    pipeline: PipelineName
    ticker: str
    company: str | None = None
    artifact_path: str
    artifact_hash: str
    schema_version: str
    report_status: str
    summary_text: str
    posture_label: str | None = None
    posture_bullets: list[str] = Field(default_factory=list)
    target_risks: list[EvaluationRiskItem] = Field(default_factory=list)
    risk_overlap_rows: list[EvaluationOverlapRow] = Field(default_factory=list)
    forward_watchlist: list[EvaluationWatchItem] = Field(default_factory=list)
    narrative_sections: list[EvaluationSection] = Field(default_factory=list)
    target_evidence_pool: list[EvidenceItem] = Field(default_factory=list)
    peer_evidence_pool: list[EvidenceItem] = Field(default_factory=list)
    deterministic_financial_facts: list[DeterministicFact] = Field(default_factory=list)
    common_strengths: list[str] = Field(default_factory=list)
    common_pressures: list[str] = Field(default_factory=list)
    shared_risk_types: list[str] = Field(default_factory=list)
    target_differences: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ClaimAssessment(BaseModel):
    claim: str
    source: str
    label: ClaimLabel
    evidence_snippets: list[str] = Field(default_factory=list)


class EvaluationScore(BaseModel):
    deterministic_consistency: float
    evidence_coverage: float
    claim_support: float | None = None
    comparative_usefulness: float | None = None
    overreach_penalty: float = 0.0
    overall_score: float = 0.0


class EvaluationResult(BaseModel):
    ticker: str
    pipeline: PipelineName
    schema_version: str = "evaluation_v1"
    artifact_path: str
    artifact_hash: str
    component_scores: EvaluationScore
    warnings: list[str] = Field(default_factory=list)
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    judge_metadata: dict[str, Any] = Field(default_factory=dict)


class PairwiseComparisonResult(BaseModel):
    ticker: str
    agentic_score: float | None = None
    baseline_score: float | None = None
    score_delta: float | None = None
    agentic_wins: list[str] = Field(default_factory=list)
    baseline_wins: list[str] = Field(default_factory=list)
    ties: list[str] = Field(default_factory=list)


class BatchEvaluationOutput(BaseModel):
    schema_version: str = "evaluation_batch_v1"
    run_name: str
    created_at: str
    results: list[EvaluationResult] = Field(default_factory=list)
    comparisons: list[PairwiseComparisonResult] = Field(default_factory=list)
