from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from orchestration.artifact_resolver import REPO_ROOT

from .documents import build_ingestion_retrieval_text, latest_year_from_ingestion
from .embeddings import embed_texts


INGESTION_SOURCE_ROOT = REPO_ROOT / "data-ingestion" / "outputs"


def iter_ingestion_files(ingestion_root: Path = INGESTION_SOURCE_ROOT) -> list[Path]:
    return sorted(
        path
        for path in ingestion_root.glob("*/*")
        if path.is_file() and path.name == "complete_ingestion.json"
    )


def load_ingestion_file(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_metadata_row(path: Path, ingestion: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(path).resolve()
    return {
        "ticker": str(ingestion.get("ticker", source_path.parent.name)).upper(),
        "company": ingestion.get("company_name"),
        "latest_filing_year": latest_year_from_ingestion(ingestion),
        "source_path": str(source_path),
        "modified_time": source_path.stat().st_mtime,
    }


def load_vector_matrix(
    ingestion_root: Path = INGESTION_SOURCE_ROOT,
    *,
    vectorizer: Callable[[list[str]], np.ndarray] = embed_texts,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    metadata: list[dict[str, Any]] = []
    texts: list[str] = []

    for path in iter_ingestion_files(ingestion_root):
        ingestion = load_ingestion_file(path)
        texts.append(build_ingestion_retrieval_text(ingestion))
        metadata.append(build_metadata_row(path, ingestion))

    if not texts:
        raise ValueError(f"No ingestion artifacts found under {ingestion_root}")

    return vectorizer(texts), metadata
