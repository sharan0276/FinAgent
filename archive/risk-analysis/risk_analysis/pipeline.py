from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Optional, Sequence
from uuid import uuid4

from .artifact_loader import DEFAULT_TARGET_SECTIONS, collect_source_accessions, load_ingestion_artifact
from .artifact_writer import write_company_analyses
from .models import CompanyAnalysisArtifactV1
from .numeric_analysis import build_numeric_trend_summaries
from .taxonomy import TAXONOMY_VERSION
from .text_risk_extraction import analyze_filings, build_llm_client


SCHEMA_VERSION = "company_analysis_v1"


def run_analysis(
    *,
    source_path: Path,
    output_path: Path,
    provider: str = "openrouter",
    model_name: Optional[str] = None,
    prompt_version: str = "prompt_v1",
    section_ids: Optional[Sequence[str]] = None,
    filing_years: Optional[Sequence[int]] = None,
    verbose: bool = False,
    debug_dir: Optional[Path] = None,
) -> List[Path]:
    section_ids = list(section_ids or DEFAULT_TARGET_SECTIONS)
    data = load_ingestion_artifact(source_path, required_sections=section_ids)
    if filing_years:
        allowed_years = {int(year) for year in filing_years}
        original_count = len(data["text_data"]["filings"])
        data["text_data"]["filings"] = [
            filing
            for filing in data["text_data"]["filings"]
            if len(str(filing.get("filingDate", ""))) >= 4 and int(str(filing.get("filingDate", ""))[:4]) in allowed_years
        ]
        if verbose:
            print(
                f"[risk-analysis] Filing year filter {sorted(allowed_years)} reduced filings "
                f"from {original_count} to {len(data['text_data']['filings'])}"
            )
    numeric_trends = build_numeric_trend_summaries(data["financial_data"])
    llm_client = build_llm_client(
        provider=provider,
        model_name=model_name or os.getenv("MODEL_NAME") or "heuristic-v1",
        debug_dir=debug_dir,
    )
    if verbose:
        print(
            f"[risk-analysis] Running analysis for {data['ticker']} with "
            f"{getattr(llm_client, 'provider_name', provider)}:{getattr(llm_client, 'model_name', model_name)}"
        )
    filing_analysis, rollups = analyze_filings(
        data["text_data"]["filings"],
        section_ids=section_ids,
        llm_client=llm_client,
        verbose=verbose,
    )
    base_artifact = CompanyAnalysisArtifactV1(
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        prompt_version=prompt_version,
        model_name=getattr(llm_client, "model_name", model_name),
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=str(uuid4()),
        ticker=data["ticker"],
        cik=data["cik"],
        company_name=data["company_name"],
        source_artifact_path=str(source_path),
        source_accessions=collect_source_accessions(data),
        numeric_trends=numeric_trends,
        filing_analysis=filing_analysis,
        section_rollups=rollups,
    )
    yearly_artifacts = build_yearly_artifacts(base_artifact)
    output_path = Path(output_path)
    artifact_targets = []
    for artifact in yearly_artifacts:
        filing = artifact.filing_analysis[0]
        filing_year = filing["filing_year"]
        yearly_output_path = output_path.parent / f"{artifact.ticker}_{filing_year}_company_analysis.json"
        artifact_targets.append((artifact, yearly_output_path))
    if verbose:
        print(f"[risk-analysis] Writing {len(artifact_targets)} yearly file(s)")
    return write_company_analyses(artifact_targets)


def build_yearly_artifacts(base_artifact: CompanyAnalysisArtifactV1) -> List[CompanyAnalysisArtifactV1]:
    yearly_artifacts: List[CompanyAnalysisArtifactV1] = []
    for filing in base_artifact.filing_analysis:
        yearly_artifacts.append(
            CompanyAnalysisArtifactV1(
                schema_version=base_artifact.schema_version,
                taxonomy_version=base_artifact.taxonomy_version,
                prompt_version=base_artifact.prompt_version,
                model_name=base_artifact.model_name,
                run_timestamp=base_artifact.run_timestamp,
                run_id=base_artifact.run_id,
                ticker=base_artifact.ticker,
                cik=base_artifact.cik,
                company_name=base_artifact.company_name,
                source_artifact_path=base_artifact.source_artifact_path,
                source_accessions=base_artifact.source_accessions,
                numeric_trends=base_artifact.numeric_trends,
                filing_analysis=[filing],
                section_rollups=None,
            )
        )
    return yearly_artifacts
