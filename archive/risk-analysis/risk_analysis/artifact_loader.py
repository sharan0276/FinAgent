from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_TARGET_SECTIONS = [
    "item_1a_risk_factors",
    "item_1c_cybersecurity",
    "item_3_legal",
    "item_7_mda",
    "item_7a_market_risk",
]


class ArtifactValidationError(ValueError):
    """Raised when an ingestion artifact is missing required structure."""


def default_source_path(repo_root: Path, ticker: str) -> Path:
    return repo_root / "data-ingestion" / "outputs" / ticker.upper() / "complete_ingestion.json"


def load_ingestion_artifact(path: Path, required_sections: Optional[Iterable[str]] = None) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Source artifact not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    _validate_top_level(data)
    _validate_financial_data(data["financial_data"])
    _validate_text_data(data["text_data"], list(required_sections or DEFAULT_TARGET_SECTIONS))
    return data


def _validate_top_level(data: Dict) -> None:
    required = ["ticker", "cik", "company_name", "financial_data", "text_data"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ArtifactValidationError(f"Missing top-level keys: {missing}")


def _validate_financial_data(financial_data: Dict) -> None:
    for key in ["annual", "quarterly", "metric_errors"]:
        if key not in financial_data:
            raise ArtifactValidationError(f"Missing financial_data.{key}")
    if not isinstance(financial_data["annual"], dict) or not isinstance(financial_data["quarterly"], dict):
        raise ArtifactValidationError("financial_data annual/quarterly fields must be dictionaries")


def _validate_text_data(text_data: Dict, required_sections: List[str]) -> None:
    if "filings" not in text_data:
        raise ArtifactValidationError("Missing text_data.filings")
    filings = text_data["filings"]
    if not isinstance(filings, list):
        raise ArtifactValidationError("text_data.filings must be a list")
    for filing in filings:
        for key in ["filingDate", "accession", "sections"]:
            if key not in filing:
                raise ArtifactValidationError(f"Missing filing field: {key}")
        sections = filing["sections"]
        if not isinstance(sections, dict):
            raise ArtifactValidationError("filing.sections must be a dictionary")
        for section_id in required_sections:
            if section_id not in sections:
                raise ArtifactValidationError(
                    f"Missing required section '{section_id}' in filing {filing.get('accession', '<unknown>')}"
                )


def collect_source_accessions(data: Dict) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for filing in data["text_data"]["filings"]:
        accession = filing.get("accession")
        if accession and accession not in seen:
            seen.add(accession)
            ordered.append(accession)
    return ordered
