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


def _get_target_year(filing: dict, data: dict) -> int:
    import re
    url = filing.get("url", "")
    match = re.search(r"((?:19|20)\d{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\.[a-zA-Z0-9]+$", url)
    if match:
        return int(match.group(1))
    
    filing_year = int(str(filing["filingDate"])[:4])
    all_years = []
    for info in data.get("financial_data", {}).get("annual", {}).values():
        if isinstance(info, dict) and "years" in info:
            all_years.extend(info["years"])
    max_data_year = max(all_years) if all_years else filing_year
    filings = data.get("text_data", {}).get("filings", [])
    max_filing_year = max([int(str(f["filingDate"])[:4]) for f in filings]) if filings else filing_year
    
    offset = max_filing_year - max_data_year if max_filing_year >= max_data_year else 0
    return filing_year - offset


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
        target_year = _get_target_year(filing, data)
        
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
            accession=str(filing.get("accession", "")),
            parser_mode=filing.get("parser_mode"),
            source_artifact_path=str(source_path),
            processed_sections=processed_sections,
            skipped_sections=skipped_sections,
            numeric_deltas=build_numeric_deltas(
                data.get("financial_data", {}), 
                target_year, 
                accession=str(filing.get("accession", ""))
            ),
            text_candidates=candidates,
        )
        output_path = output_dir / f"{artifact.ticker.lower()}_{artifact.filing_year}_extraction.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(artifact.model_dump(mode="json"), handle, indent=2)
        written_paths.append(output_path)
    return written_paths
