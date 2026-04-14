from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


StepState = Literal["completed", "failed", "partial", "skipped_existing", "not_needed"]
ReportStatus = Literal["completed", "failed", "skipped"]
PostureLabel = Literal["Elevated", "Mixed", "Stable"]
WatchConfidence = Literal["high", "medium", "low"]


class ReportSection(BaseModel):
    title: str
    content: str
    citations: list[str] = Field(default_factory=list)


class MetricDeltaItem(BaseModel):
    metric: str
    label: str
    value: float | None = None


class RiskItem(BaseModel):
    signal_type: str
    severity: str
    section: str | None = None
    summary: str | None = None
    citation: str | None = None
    occurrences: int = 1


class PostureCard(BaseModel):
    label: PostureLabel
    rationale_bullets: list[str] = Field(default_factory=list)


class TargetProfile(BaseModel):
    ticker: str
    company: str | None = None
    filing_year: int
    positive_deltas: list[MetricDeltaItem] = Field(default_factory=list)
    negative_deltas: list[MetricDeltaItem] = Field(default_factory=list)
    top_risks: list[RiskItem] = Field(default_factory=list)


class PeerSnapshot(BaseModel):
    peer_group: str
    common_strengths: list[str] = Field(default_factory=list)
    common_pressures: list[str] = Field(default_factory=list)
    shared_risk_types: list[str] = Field(default_factory=list)
    target_differences: list[str] = Field(default_factory=list)


class RiskOverlapRow(BaseModel):
    group: Literal["shared_now", "target_only_now", "peer_only_now"]
    risk_types: list[str] = Field(default_factory=list)


class ForwardWatchItem(BaseModel):
    watch_risk_type: str
    why_relevant: str
    peer_evidence: list[str] = Field(default_factory=list)
    confidence: WatchConfidence


class TargetContext(BaseModel):
    ticker: str
    company: str | None = None
    latest_filing_year: int | None = None
    ingestion_path: str | None = None
    extraction_path: str | None = None
    curator_path: str | None = None


class MatchContext(BaseModel):
    ticker: str
    company: str | None = None
    matched_filing_year: int
    similarity: float
    context_curator_paths: list[str] = Field(default_factory=list)


class RunMetadata(BaseModel):
    top_k: int
    run_timestamp: str
    status_by_step: dict[str, StepState] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ComparisonBundle(BaseModel):
    target: TargetContext
    matches: list[MatchContext] = Field(default_factory=list)
    run_metadata: RunMetadata


class ComparisonReportResult(BaseModel):
    status: ReportStatus
    summary: str
    model_name: str | None = None
    posture: PostureCard | None = None
    target_profile: TargetProfile | None = None
    peer_snapshot: PeerSnapshot | None = None
    risk_overlap_rows: list[RiskOverlapRow] = Field(default_factory=list)
    forward_watchlist: list[ForwardWatchItem] = Field(default_factory=list)
    narrative_sections: list[ReportSection] = Field(default_factory=list)
    error: str | None = None


class OrchestrationArtifact(BaseModel):
    schema_version: str = "orchestration_v1"
    bundle: ComparisonBundle
    comparison_report: ComparisonReportResult
