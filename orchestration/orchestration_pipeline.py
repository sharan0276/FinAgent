from __future__ import annotations

import json
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifact_resolver import (
    REPO_ROOT,
    company_name_from_ingestion,
    curator_artifact_path,
    extraction_artifact_path,
    final_output_path,
    ingestion_artifact_path,
    latest_filing_year,
    load_json,
    matched_context_paths,
)
from .comparison_agent import generate_comparison_report
from .report_models import (
    ComparisonBundle,
    ComparisonReportResult,
    MatchContext,
    OrchestrationArtifact,
    RunMetadata,
    TargetContext,
)


def _ensure_module_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_ingestion_step(ticker: str, *, repo_root: Path) -> Path:
    ingestion_dir = repo_root / "data-ingestion"
    _ensure_module_path(ingestion_dir)
    from ingestion_pipeline import run_ingestion, save_ingestion_output

    result = run_ingestion(ticker)
    return save_ingestion_output(result).resolve()


def _run_extraction_step(ticker: str, filing_year: int, *, repo_root: Path) -> Path:
    extraction_dir = repo_root / "data-extraction"
    _ensure_module_path(extraction_dir)
    extraction_pipeline = _load_module_from_file("data_extraction_pipeline", extraction_dir / "pipeline.py")
    scoring_module = _load_module_from_file("data_extraction_scoring", extraction_dir / "scoring.py")

    source_path = ingestion_artifact_path(ticker, repo_root=repo_root)
    output_dir = extraction_dir / "outputs" / ticker.upper()
    scorer = scoring_module.FinBERTSentenceScorer()
    written = extraction_pipeline.run_extraction(
        source_path=source_path,
        output_dir=output_dir,
        scorer=scorer,
        filing_years=[filing_year],
    )
    if not written:
        raise RuntimeError(f"Extraction produced no artifact for {ticker} {filing_year}.")
    return Path(written[0]).resolve()


def _run_extraction_steps(ticker: str, filing_years: list[int], *, repo_root: Path) -> list[Path]:
    extraction_dir = repo_root / "data-extraction"
    _ensure_module_path(extraction_dir)
    extraction_pipeline = _load_module_from_file("data_extraction_pipeline_batch", extraction_dir / "pipeline.py")
    scoring_module = _load_module_from_file("data_extraction_scoring_batch", extraction_dir / "scoring.py")

    source_path = ingestion_artifact_path(ticker, repo_root=repo_root)
    output_dir = extraction_dir / "outputs" / ticker.upper()
    scorer = scoring_module.FinBERTSentenceScorer()
    written = extraction_pipeline.run_extraction(
        source_path=source_path,
        output_dir=output_dir,
        scorer=scorer,
        filing_years=filing_years,
    )
    return [Path(path).resolve() for path in written]


def _run_curator_step(extraction_path: Path, *, repo_root: Path) -> Path:
    extraction_dir = repo_root / "data-extraction"
    _ensure_module_path(extraction_dir)
    from curator_agent import run_curator

    extraction_payload = load_json(extraction_path)
    ticker = str(extraction_payload["ticker"]).upper()
    output_dir = extraction_dir / "outputs" / "curator" / ticker
    result = run_curator(str(extraction_path), output_dir=str(output_dir))
    if result is None:
        raise RuntimeError(f"Curator generation returned no result for {extraction_path}.")
    return (output_dir / f"{ticker.lower()}_{int(result.filing_year)}.json").resolve()


def _run_curator_steps(extraction_paths: list[Path], *, repo_root: Path) -> list[Path]:
    extraction_dir = repo_root / "data-extraction"
    _ensure_module_path(extraction_dir)
    from curator_agent import run_curator

    written_paths: list[Path] = []
    for extraction_path in extraction_paths:
        extraction_payload = load_json(extraction_path)
        ticker = str(extraction_payload["ticker"]).upper()
        output_dir = extraction_dir / "outputs" / "curator" / ticker
        result = run_curator(str(extraction_path), output_dir=str(output_dir))
        if result is None:
            raise RuntimeError(f"Curator generation returned no result for {extraction_path}.")
        written_paths.append((output_dir / f"{ticker.lower()}_{int(result.filing_year)}.json").resolve())
    return written_paths


def _run_matcher_step(input_file: Path, top_k: int, *, repo_root: Path) -> dict[str, Any]:
    rag_dir = repo_root / "rag-matching"
    _ensure_module_path(rag_dir)
    from matcher import find_matches

    return find_matches(input_file, top_k=top_k)


@dataclass
class PipelineDependencies:
    run_ingestion: Callable[..., Path] = _run_ingestion_step
    run_extraction: Callable[..., Path] = _run_extraction_step
    run_extractions: Callable[..., list[Path]] = _run_extraction_steps
    run_curator: Callable[..., Path] = _run_curator_step
    run_curators: Callable[..., list[Path]] = _run_curator_steps
    run_matcher: Callable[..., dict[str, Any]] = _run_matcher_step
    run_comparison_agent: Callable[..., ComparisonReportResult] = generate_comparison_report


def _write_artifact(artifact: OrchestrationArtifact, *, ticker: str, output_root: Path, repo_root: Path) -> Path:
    path = final_output_path(ticker, output_root=output_root, repo_root=repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def _build_target_context(
    *,
    ticker: str,
    company: str | None,
    latest_year: int | None,
    repo_root: Path,
) -> TargetContext:
    ingestion_path = ingestion_artifact_path(ticker, repo_root=repo_root)
    extraction_path = (
        extraction_artifact_path(ticker, latest_year, repo_root=repo_root) if latest_year is not None else None
    )
    curator_path = curator_artifact_path(ticker, latest_year, repo_root=repo_root) if latest_year is not None else None
    return TargetContext(
        ticker=ticker.upper(),
        company=company,
        latest_filing_year=latest_year,
        ingestion_path=str(ingestion_path.resolve()) if ingestion_path.exists() else None,
        extraction_path=str(extraction_path.resolve()) if extraction_path and extraction_path.exists() else None,
        curator_path=str(curator_path.resolve()) if curator_path and curator_path.exists() else None,
    )


def _failure_artifact(
    *,
    ticker: str,
    company_name: str | None,
    latest_year: int | None,
    top_k: int,
    warnings: list[str],
    status_by_step: dict[str, str],
    output_root: Path,
    repo_root: Path,
    error: str,
) -> tuple[OrchestrationArtifact, Path]:
    artifact = OrchestrationArtifact(
        bundle=ComparisonBundle(
            target=_build_target_context(
                ticker=ticker,
                company=company_name,
                latest_year=latest_year,
                repo_root=repo_root,
            ),
            matches=[],
            run_metadata=RunMetadata(
                top_k=top_k,
                run_timestamp=datetime.now(timezone.utc).isoformat(),
                status_by_step=status_by_step,
                warnings=warnings,
            ),
        ),
        comparison_report=ComparisonReportResult(
            status="skipped",
            summary="Pipeline stopped before comparison because a prerequisite step failed.",
            error=error,
        ),
    )
    path = _write_artifact(artifact, ticker=ticker, output_root=output_root, repo_root=repo_root)
    return artifact, path


def _filing_years_from_ingestion(ingestion_payload: dict[str, Any]) -> list[int]:
    years: set[int] = set()
    filings = ingestion_payload.get("text_data", {}).get("filings", [])
    for filing in filings:
        filing_date = str(filing.get("filingDate", ""))
        if len(filing_date) >= 4 and filing_date[:4].isdigit():
            years.add(int(filing_date[:4]))
    return sorted(years)


def build_company_dataset(
    ticker: str,
    *,
    repo_root: Path = REPO_ROOT,
    deps: PipelineDependencies | None = None,
) -> dict[str, Any]:
    deps = deps or PipelineDependencies()
    ticker = ticker.upper()
    status_by_step: dict[str, str] = {}
    warnings: list[str] = []
    company_name: str | None = None
    latest_year: int | None = None

    ingestion_path = ingestion_artifact_path(ticker, repo_root=repo_root)
    extraction_paths: list[Path] = []
    curator_paths: list[Path] = []

    try:
        if ingestion_path.exists():
            status_by_step["ingestion"] = "skipped_existing"
        else:
            ingestion_path = deps.run_ingestion(ticker, repo_root=repo_root)
            status_by_step["ingestion"] = "completed"

        ingestion_payload = load_json(ingestion_path)
        company_name = company_name_from_ingestion(ingestion_payload)
        filing_years = _filing_years_from_ingestion(ingestion_payload)
        if not filing_years:
            raise ValueError("No filing years found in ingestion artifact.")
        latest_year = max(filing_years)
    except Exception as exc:
        status_by_step["ingestion"] = "failed"
        return {
            "ticker": ticker,
            "company_name": company_name,
            "latest_filing_year": latest_year,
            "ingestion_path": str(ingestion_path.resolve()) if ingestion_path.exists() else None,
            "extraction_paths": [],
            "curator_paths": [],
            "status_by_step": status_by_step,
            "warnings": warnings,
            "error": str(exc),
        }

    try:
        existing_extractions = []
        missing_years: list[int] = []
        for year in filing_years:
            path = extraction_artifact_path(ticker, year, repo_root=repo_root)
            if path.exists():
                existing_extractions.append(path.resolve())
            else:
                missing_years.append(year)

        written_extractions = deps.run_extractions(ticker, missing_years, repo_root=repo_root) if missing_years else []
        extraction_paths = sorted(existing_extractions + written_extractions)
        if not missing_years:
            status_by_step["extraction"] = "skipped_existing"
        elif existing_extractions:
            status_by_step["extraction"] = "partial"
        else:
            status_by_step["extraction"] = "completed"
    except Exception as exc:
        status_by_step["extraction"] = "failed"
        return {
            "ticker": ticker,
            "company_name": company_name,
            "latest_filing_year": latest_year,
            "ingestion_path": str(ingestion_path.resolve()) if ingestion_path.exists() else None,
            "extraction_paths": [str(path.resolve()) for path in extraction_paths],
            "curator_paths": [],
            "status_by_step": status_by_step,
            "warnings": warnings,
            "error": str(exc),
        }

    try:
        existing_curators = []
        missing_extraction_paths: list[Path] = []
        for extraction_path in extraction_paths:
            extraction_payload = load_json(extraction_path)
            filing_year = int(extraction_payload["filing_year"])
            curator_path = curator_artifact_path(ticker, filing_year, repo_root=repo_root)
            if curator_path.exists():
                existing_curators.append(curator_path.resolve())
            else:
                missing_extraction_paths.append(extraction_path)

        written_curators = deps.run_curators(missing_extraction_paths, repo_root=repo_root) if missing_extraction_paths else []
        curator_paths = sorted(existing_curators + written_curators)
        if not missing_extraction_paths:
            status_by_step["curator"] = "skipped_existing"
        elif existing_curators:
            status_by_step["curator"] = "partial"
        else:
            status_by_step["curator"] = "completed"
    except Exception as exc:
        status_by_step["curator"] = "failed"
        return {
            "ticker": ticker,
            "company_name": company_name,
            "latest_filing_year": latest_year,
            "ingestion_path": str(ingestion_path.resolve()) if ingestion_path.exists() else None,
            "extraction_paths": [str(path.resolve()) for path in extraction_paths],
            "curator_paths": [str(path.resolve()) for path in curator_paths],
            "status_by_step": status_by_step,
            "warnings": warnings,
            "error": str(exc),
        }

    return {
        "ticker": ticker,
        "company_name": company_name,
        "latest_filing_year": latest_year,
        "ingestion_path": str(ingestion_path.resolve()),
        "extraction_paths": [str(path.resolve()) for path in extraction_paths],
        "curator_paths": [str(path.resolve()) for path in curator_paths],
        "status_by_step": status_by_step,
        "warnings": warnings,
        "error": None,
    }


def run_orchestration(
    ticker: str,
    *,
    top_k: int = 2,
    output_root: Path | None = None,
    repo_root: Path = REPO_ROOT,
    deps: PipelineDependencies | None = None,
) -> tuple[OrchestrationArtifact, Path]:
    deps = deps or PipelineDependencies()
    ticker = ticker.upper()
    output_root = output_root or (repo_root / "orchestration" / "outputs")

    status_by_step: dict[str, str] = {}
    warnings: list[str] = []
    latest_year: int | None = None
    company_name: str | None = None

    try:
        ingestion_path = ingestion_artifact_path(ticker, repo_root=repo_root)
        if ingestion_path.exists():
            status_by_step["ingestion"] = "skipped_existing"
        else:
            ingestion_path = deps.run_ingestion(ticker, repo_root=repo_root)
            status_by_step["ingestion"] = "completed"

        ingestion_payload = load_json(ingestion_path)
        latest_year = latest_filing_year(ingestion_payload)
        company_name = company_name_from_ingestion(ingestion_payload)
    except Exception as exc:
        status_by_step["ingestion"] = "failed"
        warnings.append(str(exc))
        return _failure_artifact(
            ticker=ticker,
            company_name=company_name,
            latest_year=latest_year,
            top_k=top_k,
            warnings=warnings,
            status_by_step=status_by_step,
            output_root=output_root,
            repo_root=repo_root,
            error=str(exc),
        )

    try:
        extraction_path = extraction_artifact_path(ticker, latest_year, repo_root=repo_root)
        if extraction_path.exists():
            status_by_step["extraction"] = "skipped_existing"
        else:
            extraction_path = deps.run_extraction(ticker, latest_year, repo_root=repo_root)
            status_by_step["extraction"] = "completed"
    except Exception as exc:
        status_by_step["extraction"] = "failed"
        warnings.append(str(exc))
        return _failure_artifact(
            ticker=ticker,
            company_name=company_name,
            latest_year=latest_year,
            top_k=top_k,
            warnings=warnings,
            status_by_step=status_by_step,
            output_root=output_root,
            repo_root=repo_root,
            error=str(exc),
        )

    try:
        curator_path = curator_artifact_path(ticker, latest_year, repo_root=repo_root)
        if curator_path.exists():
            status_by_step["curator"] = "skipped_existing"
        else:
            curator_path = deps.run_curator(extraction_path, repo_root=repo_root)
            status_by_step["curator"] = "completed"
    except Exception as exc:
        status_by_step["curator"] = "failed"
        warnings.append(str(exc))
        return _failure_artifact(
            ticker=ticker,
            company_name=company_name,
            latest_year=latest_year,
            top_k=top_k,
            warnings=warnings,
            status_by_step=status_by_step,
            output_root=output_root,
            repo_root=repo_root,
            error=str(exc),
        )

    try:
        match_result = deps.run_matcher(curator_path, top_k, repo_root=repo_root)
        raw_matches = match_result.get("matches", [])
        if len(raw_matches) < top_k:
            warnings.append(f"Requested top {top_k} matches but only found {len(raw_matches)}.")
            status_by_step["rag"] = "partial"
        else:
            status_by_step["rag"] = "completed"
    except Exception as exc:
        status_by_step["rag"] = "failed"
        warnings.append(str(exc))
        return _failure_artifact(
            ticker=ticker,
            company_name=company_name,
            latest_year=latest_year,
            top_k=top_k,
            warnings=warnings,
            status_by_step=status_by_step,
            output_root=output_root,
            repo_root=repo_root,
            error=str(exc),
        )

    matches: list[MatchContext] = []
    for match in raw_matches:
        context_paths = matched_context_paths(
            str(match["ticker"]),
            int(match["filing_year"]),
            repo_root=repo_root,
        )
        company = match.get("company")
        if not company and context_paths:
            company = load_json(context_paths[0]).get("company")
        if len(context_paths) < 3:
            warnings.append(
                f"{match['ticker']} has only {len(context_paths)} curator year(s) from {match['filing_year']} onward."
            )
            if status_by_step["rag"] == "completed":
                status_by_step["rag"] = "partial"
        matches.append(
            MatchContext(
                ticker=str(match["ticker"]),
                company=str(company or match["ticker"]),
                matched_filing_year=int(match["filing_year"]),
                similarity=float(match["similarity"]),
                context_curator_paths=[str(path.resolve()) for path in context_paths],
            )
        )

    bundle = ComparisonBundle(
        target=_build_target_context(
            ticker=ticker,
            company=company_name,
            latest_year=latest_year,
            repo_root=repo_root,
        ),
        matches=matches,
        run_metadata=RunMetadata(
            top_k=top_k,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            status_by_step=status_by_step,
            warnings=warnings,
        ),
    )

    comparison_report = deps.run_comparison_agent(bundle)
    comparison_status = "completed" if comparison_report.status == "completed" else "failed"
    status_by_step["comparison"] = comparison_status
    bundle.run_metadata.status_by_step["comparison"] = comparison_status

    artifact = OrchestrationArtifact(bundle=bundle, comparison_report=comparison_report)
    path = _write_artifact(artifact, ticker=ticker, output_root=output_root, repo_root=repo_root)
    return artifact, path
