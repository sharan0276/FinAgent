from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricTrendSummaryV1:
    metric_name: str
    frequency_covered: str
    latest_annual_value: Optional[float]
    latest_annual_period_end: Optional[str]
    latest_quarterly_value: Optional[float]
    latest_quarterly_period_end: Optional[str]
    annual_direction_5y: str
    annual_direction_3y: str
    annual_cagr_5y: Optional[float]
    annual_yoy_latest: Optional[float]
    quarterly_direction_recent: str
    quarterly_volatility_flag: bool
    annual_series: List[Dict[str, Any]] = field(default_factory=list)
    quarterly_series: List[Dict[str, Any]] = field(default_factory=list)
    missing_data_notes: List[str] = field(default_factory=list)
    source_metric_refs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "frequency_covered": self.frequency_covered,
            "annual_series": self.annual_series,
            "quarterly_series": self.quarterly_series,
            "latest_annual_value": self.latest_annual_value,
            "latest_annual_period_end": self.latest_annual_period_end,
            "latest_quarterly_value": self.latest_quarterly_value,
            "latest_quarterly_period_end": self.latest_quarterly_period_end,
            "annual_direction_5y": self.annual_direction_5y,
            "annual_direction_3y": self.annual_direction_3y,
            "annual_cagr_5y": self.annual_cagr_5y,
            "annual_yoy_latest": self.annual_yoy_latest,
            "quarterly_direction_recent": self.quarterly_direction_recent,
            "quarterly_volatility_flag": self.quarterly_volatility_flag,
            "missing_data_notes": self.missing_data_notes,
            "source_metric_refs": self.source_metric_refs,
        }


@dataclass
class RiskSignalV1:
    signal_id: str
    taxonomy_version: str
    filing_year: int
    filing_date: str
    accession: str
    section_id: str
    subheading: str
    risk_types: List[str]
    severity: int
    direction: str
    confidence: float
    evidence_text: str
    evidence_start: int
    evidence_end: int
    judge_rationale: str
    source_rank: int
    keep_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "taxonomy_version": self.taxonomy_version,
            "filing_year": self.filing_year,
            "filing_date": self.filing_date,
            "accession": self.accession,
            "section_id": self.section_id,
            "subheading": self.subheading,
            "risk_types": self.risk_types,
            "severity": self.severity,
            "direction": self.direction,
            "confidence": self.confidence,
            "evidence_text": self.evidence_text,
            "evidence_start": self.evidence_start,
            "evidence_end": self.evidence_end,
            "judge_rationale": self.judge_rationale,
            "source_rank": self.source_rank,
            "keep_score": self.keep_score,
        }


@dataclass
class SectionRiskResultV1:
    section_id: str
    section_name: str
    filing_year: int
    filing_date: str
    accession: str
    parser_mode: Optional[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    processing_error: Optional[str]
    subheading_count: int
    chunk_count: int
    candidate_count: int
    kept_signal_count: int
    notes: List[str]
    signals: List[RiskSignalV1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_name": self.section_name,
            "filing_year": self.filing_year,
            "filing_date": self.filing_date,
            "accession": self.accession,
            "parser_mode": self.parser_mode,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "processing_error": self.processing_error,
            "subheading_count": self.subheading_count,
            "chunk_count": self.chunk_count,
            "candidate_count": self.candidate_count,
            "kept_signal_count": self.kept_signal_count,
            "notes": self.notes,
            "signals": [signal.to_dict() for signal in self.signals],
        }


@dataclass
class SectionRiskRollupV1:
    section_id: str
    section_name: str
    years_covered: List[int]
    recurring_risk_types: List[str]
    representative_signals: List[Dict[str, Any]]
    severity_trend: str
    contributing_years: List[int]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_name": self.section_name,
            "years_covered": self.years_covered,
            "recurring_risk_types": self.recurring_risk_types,
            "representative_signals": self.representative_signals,
            "severity_trend": self.severity_trend,
            "contributing_years": self.contributing_years,
            "notes": self.notes,
        }


@dataclass
class CompanyAnalysisArtifactV1:
    schema_version: str
    taxonomy_version: str
    prompt_version: str
    model_name: str
    run_timestamp: str
    run_id: str
    ticker: str
    cik: str
    company_name: str
    source_artifact_path: str
    source_accessions: List[str]
    numeric_trends: Dict[str, MetricTrendSummaryV1]
    filing_analysis: List[Dict[str, Any]]
    section_rollups: Optional[Dict[str, SectionRiskRollupV1]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "taxonomy_version": self.taxonomy_version,
            "prompt_version": self.prompt_version,
            "model_name": self.model_name,
            "run_timestamp": self.run_timestamp,
            "run_id": self.run_id,
            "ticker": self.ticker,
            "cik": self.cik,
            "company_name": self.company_name,
            "source_artifact_path": self.source_artifact_path,
            "source_accessions": self.source_accessions,
            "numeric_trends": {key: value.to_dict() for key, value in self.numeric_trends.items()},
            "filing_analysis": self.filing_analysis,
        }
        if self.section_rollups is not None:
            data["section_rollups"] = {key: value.to_dict() for key, value in self.section_rollups.items()}
        return data
