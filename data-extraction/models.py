from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FinBertProbabilities(BaseModel):
    positive: float
    neutral: float
    negative: float


class TextCandidate(BaseModel):
    candidate_id: str
    section_id: str
    section_label: str
    sentence_index: int
    sentence_text: str
    previous_sentence: Optional[str] = None
    next_sentence: Optional[str] = None
    risk_score: float
    finbert_probs: FinBertProbabilities


class NumericDelta(BaseModel):
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    current_end_date: Optional[str] = None
    previous_end_date: Optional[str] = None
    current_accession: Optional[str] = None
    previous_accession: Optional[str] = None
    current_tag: Optional[str] = None
    previous_tag: Optional[str] = None
    delta_percent: Optional[float] = None
    label: Optional[str] = None
    reason: Optional[str] = None


class ExtractionArtifact(BaseModel):
    schema_version: str
    model_name: str
    run_timestamp: str
    ticker: str
    company_name: str
    cik: str
    filing_year: int
    filing_date: str
    accession: str
    parser_mode: Optional[str] = None
    source_artifact_path: str
    processed_sections: List[str] = Field(default_factory=list)
    skipped_sections: List[str] = Field(default_factory=list)
    numeric_deltas: Dict[str, NumericDelta]
    text_candidates: List[TextCandidate]
