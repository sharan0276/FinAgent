from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from artifact_loader import iter_filings, load_ingestion_artifact
from constants import SCHEMA_VERSION
from models import ExtractionArtifact
from numeric_delta import build_numeric_deltas
from scoring import BaseSentenceScorer
from text_candidates import build_text_candidates


def run_extraction(
    *,
    source_path: Path,
    output_dir: Path,
    scorer: BaseSentenceScorer,
    filing_years: Optional[Iterable[int]] = None,
) -> List[Path]:
    data = load_ingestion_artifact(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []
    for filing in iter_filings(data, filing_years=filing_years):
        filing_date = str(filing["filingDate"])
        filing_year = int(filing_date[:4])
        candidates, processed_sections, skipped_sections = build_text_candidates(
            filing.get("sections", {}),
            scorer=scorer,
        )
        artifact = ExtractionArtifact(
            schema_version=SCHEMA_VERSION,
            model_name=getattr(scorer, "model_name", scorer.__class__.__name__),
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=str(data["ticker"]),
            company_name=str(data["company_name"]),
            cik=str(data["cik"]),
            filing_year=filing_year,
            filing_date=filing_date,
            accession=str(filing.get("accessionNumber", "")),
            parser_mode=filing.get("parser_mode"),
            source_artifact_path=str(source_path),
            processed_sections=processed_sections,
            skipped_sections=skipped_sections,
            numeric_deltas=build_numeric_deltas(data.get("financial_data", {}), filing_year),
            text_candidates=candidates,
        )
        output_path = output_dir / f"{artifact.ticker.lower()}_{artifact.filing_year}_extraction.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(artifact.model_dump(mode="json"), handle, indent=2)
        written_paths.append(output_path)
    return written_paths
