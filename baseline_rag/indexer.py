"""Build and load the persistent FAISS index for ingestion-only baseline matching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from orchestration.artifact_resolver import REPO_ROOT

from .embeddings import embed_texts
from .ingestion_store import INGESTION_SOURCE_ROOT, iter_ingestion_files, load_vector_matrix
from .runtime_compat import prepare_openmp_runtime


ARTIFACT_DIR = Path(__file__).resolve().parent / "index_artifacts"
INDEX_PATH = ARTIFACT_DIR / "faiss.index"
META_PATH = ARTIFACT_DIR / "metadata.json"


def _import_faiss():
    prepare_openmp_runtime()
    try:
        import faiss  # type: ignore

        return faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FAISS is required for baseline ingestion matching. Install faiss-cpu in the active environment."
        ) from exc


def baseline_artifact_dir(*, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "baseline_rag" / "index_artifacts"


def build_index(
    *,
    ingestion_root: Path = INGESTION_SOURCE_ROOT,
    artifact_dir: Path = ARTIFACT_DIR,
    vectorizer: Callable[[list[str]], np.ndarray] = embed_texts,
) -> dict[str, Any]:
    # Fail fast before loading the large embedding model when FAISS is unavailable.
    faiss = _import_faiss()
    print(f"[baseline_rag] Building ingestion index from {ingestion_root}...")
    matrix, metadata = load_vector_matrix(ingestion_root, vectorizer=vectorizer)
    n, dim = matrix.shape
    print(f"[baseline_rag] Indexed {n} company document(s), dim={dim}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(artifact_dir / INDEX_PATH.name))

    payload = {"entries": metadata}
    (artifact_dir / META_PATH.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_index(*, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    metadata_path = artifact_dir / META_PATH.name
    index_path = artifact_dir / INDEX_PATH.name
    if not metadata_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Baseline index artifacts not found under {artifact_dir}. Build the baseline index first."
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    faiss = _import_faiss()
    index = faiss.read_index(str(index_path))
    return {"index": index, "entries": entries}


def search_index(bundle: dict[str, Any], query_vector: np.ndarray, top_k: int) -> tuple[list[float], list[int]]:
    index = bundle["index"]
    scores, indices = index.search(query_vector, top_k)
    return scores[0].tolist(), indices[0].tolist()


def _metadata_matches_sources(entries: list[dict[str, Any]], ingestion_root: Path) -> bool:
    current_paths = {str(path.resolve()): path.stat().st_mtime for path in iter_ingestion_files(ingestion_root)}
    saved_paths = {str(entry.get("source_path")): entry.get("modified_time") for entry in entries}
    if set(current_paths.keys()) != set(saved_paths.keys()):
        return False
    for path_str, mtime in current_paths.items():
        if saved_paths.get(path_str) != mtime:
            return False
    return True


def ensure_index(
    *,
    repo_root: Path = REPO_ROOT,
    vectorizer: Callable[[list[str]], np.ndarray] = embed_texts,
) -> tuple[dict[str, Any], str]:
    artifact_dir = baseline_artifact_dir(repo_root=repo_root)
    ingestion_root = repo_root / "data-ingestion" / "outputs"

    try:
        bundle = load_index(artifact_dir=artifact_dir)
        if _metadata_matches_sources(bundle.get("entries", []), ingestion_root):
            print("[baseline_rag] Reusing existing ingestion index.")
            return bundle, "reused"
        print("[baseline_rag] Ingestion index is stale; rebuilding.")
    except FileNotFoundError:
        print("[baseline_rag] No ingestion index found; building it now.")

    build_index(ingestion_root=ingestion_root, artifact_dir=artifact_dir, vectorizer=vectorizer)
    return load_index(artifact_dir=artifact_dir), "rebuilt"
