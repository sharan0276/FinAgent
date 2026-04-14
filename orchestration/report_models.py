from __future__ import annotations

from typing import Literal, Optional

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
    value: Optional[float] = None


class RiskItem(BaseModel):
    signal_type: str
    severity: str
    section: Optional[str] = None
    summary: Optional[str] = None
    citation: Optional[str] = None
    occurrences: int = 1


class PostureCard(BaseModel):
    label: PostureLabel
    rationale_bullets: list[str] = Field(default_factory=list)


class TargetProfile(BaseModel):
    ticker: str
    company: Optional[str] = None
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
    company: Optional[str] = None
    latest_filing_year: Optional[int] = None
    ingestion_path: Optional[str] = None
    extraction_path: Optional[str] = None
    curator_path: Optional[str] = None


class MatchContext(BaseModel):
    ticker: str
    company: Optional[str] = None
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
    model_name: Optional[str] = None
    posture: Optional[PostureCard] = None
    target_profile: Optional[TargetProfile] = None
    peer_snapshot: Optional[PeerSnapshot] = None
    risk_overlap_rows: list[RiskOverlapRow] = Field(default_factory=list)
    forward_watchlist: list[ForwardWatchItem] = Field(default_factory=list)
    narrative_sections: list[ReportSection] = Field(default_factory=list)
    error: Optional[str] = None


class OrchestrationArtifact(BaseModel):
    schema_version: str = "orchestration_v1"
    bundle: ComparisonBundle
    comparison_report: ComparisonReportResult
