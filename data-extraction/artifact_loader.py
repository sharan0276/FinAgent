from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def default_source_path(repo_root: Path, ticker: str) -> Path:
    return repo_root / "data-ingestion" / "outputs" / ticker.upper() / "complete_ingestion.json"


def load_ingestion_artifact(source_path: Path) -> Dict:
    with Path(source_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_filings(data: Dict, filing_years: Optional[Iterable[int]] = None) -> List[Dict]:
    filings = data.get("text_data", {}).get("filings", [])
    allowed_years = {int(year) for year in filing_years} if filing_years else None
    selected = []
    for filing in filings:
        filing_date = str(filing.get("filingDate", ""))
        if len(filing_date) < 4:
            continue
        filing_year = int(filing_date[:4])
        if allowed_years and filing_year not in allowed_years:
            continue
        selected.append(filing)
    return selected
