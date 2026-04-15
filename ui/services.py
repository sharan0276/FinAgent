from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from baseline_rag import run_baseline_rag
from orchestration.artifact_resolver import REPO_ROOT
from orchestration.orchestration_pipeline import PipelineDependencies, build_company_dataset, run_orchestration
from orchestration.report_models import OrchestrationArtifact


def _load_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_module_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def get_available_tickers(*, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return tickers that have an ingestion artifact, sorted alphabetically."""
    ingestion_root = repo_root / "data-ingestion" / "outputs"
    if not ingestion_root.exists():
        return []
    return sorted(
        d.name
        for d in ingestion_root.iterdir()
        if d.is_dir() and (d / "complete_ingestion.json").exists()
    )


def list_saved_report_artifacts(*, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    output_root = repo_root / "orchestration" / "outputs"
    artifacts: list[dict[str, Any]] = []
    if not output_root.exists():
        return artifacts

    for path in sorted(output_root.glob("*/*_comparison_bundle.json")):
        try:
            artifact = load_saved_report_artifact(path)
        except Exception:
            continue
        artifacts.append(
            {
                "label": (
                    f"{artifact.bundle.target.ticker} | "
                    f"{artifact.bundle.target.latest_filing_year or 'unknown'} | "
                    f"{artifact.comparison_report.status}"
                ),
                "path": str(path.resolve()),
                "ticker": artifact.bundle.target.ticker,
                "latest_filing_year": artifact.bundle.target.latest_filing_year,
                "comparison_status": artifact.comparison_report.status,
                "run_timestamp": artifact.bundle.run_metadata.run_timestamp,
            }
        )
    return artifacts


def load_saved_report_artifact(path: str | Path) -> OrchestrationArtifact:
    payload = Path(path).read_text(encoding="utf-8")
    return OrchestrationArtifact.model_validate_json(payload)


def get_ticker_dataset_status(ticker: str, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    ingestion_path = repo_root / "data-ingestion" / "outputs" / ticker / "complete_ingestion.json"
    extraction_dir = repo_root / "data-extraction" / "outputs" / ticker
    curator_dir = repo_root / "data-extraction" / "outputs" / "curator" / ticker

    extraction_paths = sorted(str(path.resolve()) for path in extraction_dir.glob("*_extraction.json")) if extraction_dir.exists() else []
    curator_paths = sorted(str(path.resolve()) for path in curator_dir.glob("*.json")) if curator_dir.exists() else []

    return {
        "ticker": ticker,
        "ingestion_exists": ingestion_path.exists(),
        "ingestion_path": str(ingestion_path.resolve()) if ingestion_path.exists() else None,
        "extraction_count": len(extraction_paths),
        "extraction_paths": extraction_paths,
        "curator_count": len(curator_paths),
        "curator_paths": curator_paths,
    }


def build_company_dataset_for_ui(
    ticker: str,
    *,
    repo_root: Path = REPO_ROOT,
    deps: PipelineDependencies | None = None,
) -> dict[str, Any]:
    return build_company_dataset(normalize_ticker(ticker), repo_root=repo_root, deps=deps)


def run_analysis_for_ui(
    ticker: str,
    *,
    top_k: int = 2,
    repo_root: Path = REPO_ROOT,
    deps: PipelineDependencies | None = None,
) -> tuple[OrchestrationArtifact, Path]:
    return run_orchestration(normalize_ticker(ticker), top_k=top_k, repo_root=repo_root, deps=deps)


def run_baseline_rag_for_ui(
    ticker: str,
    *,
    top_k: int = 2,
    focus_query: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> OrchestrationArtifact:
    """Run the baseline RAG pipeline and return the artifact.

    Unlike run_analysis_for_ui, this does not write an output file —
    the baseline is ephemeral and only stored in session state for comparison.
    """
    return run_baseline_rag(
        normalize_ticker(ticker),
        top_k=top_k,
        focus_query=focus_query or None,
        repo_root=repo_root,
    )


def get_faiss_index_status(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_dir = repo_root / "rag-matching" / "index_artifacts"
    index_path = artifact_dir / "faiss.index"
    metadata_path = artifact_dir / "metadata.json"

    metadata_payload: dict[str, Any] | None = None
    entry_count = 0
    if metadata_path.exists():
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        entry_count = len(metadata_payload.get("entries", []))

    return {
        "artifact_dir": str(artifact_dir.resolve()),
        "index_exists": index_path.exists(),
        "index_path": str(index_path.resolve()),
        "metadata_exists": metadata_path.exists(),
        "metadata_path": str(metadata_path.resolve()),
        "entry_count": entry_count,
        "metadata": metadata_payload,
    }


def rebuild_faiss_index(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    rag_dir = repo_root / "rag-matching"
    _ensure_module_path(rag_dir)
    indexer = _load_module_from_file("ui_rag_indexer", rag_dir / "indexer.py")
    payload = indexer.build_index()
    status = get_faiss_index_status(repo_root=repo_root)
    status["build_payload"] = payload
    return status
