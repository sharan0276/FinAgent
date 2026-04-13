from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def ingestion_artifact_path(ticker: str, *, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data-ingestion" / "outputs" / ticker.upper() / "complete_ingestion.json"


def extraction_dir(ticker: str, *, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data-extraction" / "outputs" / ticker.upper()


def extraction_artifact_path(ticker: str, filing_year: int, *, repo_root: Path = REPO_ROOT) -> Path:
    return extraction_dir(ticker, repo_root=repo_root) / f"{ticker.lower()}_{filing_year}_extraction.json"


def curator_dir(ticker: str, *, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data-extraction" / "outputs" / "curator" / ticker.upper()


def curator_artifact_path(ticker: str, filing_year: int, *, repo_root: Path = REPO_ROOT) -> Path:
    return curator_dir(ticker, repo_root=repo_root) / f"{ticker.lower()}_{filing_year}.json"


def final_output_dir(
    ticker: str,
    *,
    output_root: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> Path:
    base = output_root or (repo_root / "orchestration" / "outputs")
    return base / ticker.upper()


def final_output_path(
    ticker: str,
    *,
    output_root: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> Path:
    return final_output_dir(ticker, output_root=output_root, repo_root=repo_root) / (
        f"{ticker.lower()}_comparison_bundle.json"
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_filing_year(ingestion_payload: dict[str, Any]) -> int:
    years: list[int] = []
    filings = ingestion_payload.get("text_data", {}).get("filings", [])
    for filing in filings:
        filing_date = str(filing.get("filingDate", ""))
        if len(filing_date) >= 4 and filing_date[:4].isdigit():
            years.append(int(filing_date[:4]))
    if not years:
        raise ValueError("No filing years found in ingestion artifact.")
    return max(years)


def company_name_from_ingestion(ingestion_payload: dict[str, Any]) -> str | None:
    company_name = ingestion_payload.get("company_name")
    return str(company_name) if company_name else None


def list_curator_years(ticker: str, *, repo_root: Path = REPO_ROOT) -> list[int]:
    years: list[int] = []
    for path in sorted(curator_dir(ticker, repo_root=repo_root).glob("*.json")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            years.append(int(parts[1]))
        except ValueError:
            continue
    return sorted(years)


def matched_context_paths(
    ticker: str,
    anchor_year: int,
    *,
    repo_root: Path = REPO_ROOT,
    limit: int = 3,
) -> list[Path]:
    years = [year for year in list_curator_years(ticker, repo_root=repo_root) if year >= anchor_year]
    return [curator_artifact_path(ticker, year, repo_root=repo_root) for year in years[:limit]]

