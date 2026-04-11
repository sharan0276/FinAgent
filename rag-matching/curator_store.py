from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATOR_SOURCE_ROOT = REPO_ROOT / "data-extraction" / "outputs" / "curator"
CURATOR_EMBEDDING_MODEL = "BAAI/bge-m3"


def iter_curator_files(curator_root: Path = CURATOR_SOURCE_ROOT) -> list[Path]:
    return sorted(path for path in curator_root.rglob("*.json") if path.is_file())


def load_curator_file(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_stored_embedding(curator: dict[str, Any]) -> np.ndarray | None:
    vector = curator.get("embedding_vector")
    if not vector:
        return None
    return np.array(vector, dtype="float32").reshape(1, -1)


def build_metadata_row(path: Path, curator: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": curator["ticker"],
        "company": curator["company"],
        "filing_year": curator["filing_year"],
        "source_path": str(Path(path).resolve()),
    }


def load_vector_matrix(
    curator_root: Path = CURATOR_SOURCE_ROOT,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    metadata: list[dict[str, Any]] = []
    vectors: list[list[float]] = []

    for path in iter_curator_files(curator_root):
        curator = load_curator_file(path)
        vector = curator.get("embedding_vector")
        if not vector:
            print(f"  [warning] {path.name} has no embedding_vector - skipping")
            continue
        vectors.append(vector)
        metadata.append(build_metadata_row(path, curator))

    if not vectors:
        raise ValueError(
            f"No curator files with embedding_vector found under {curator_root}"
        )

    return np.array(vectors, dtype="float32"), metadata
