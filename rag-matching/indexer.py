"""Build and load the persistent FAISS index for curator matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from curator_store import CURATOR_SOURCE_ROOT, load_vector_matrix
from runtime_compat import prepare_openmp_runtime

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
            "FAISS is required for rag-matching. Install faiss-cpu in the active environment."
        ) from exc


def build_index(
    *,
    curator_root: Path = CURATOR_SOURCE_ROOT,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, Any]:
    matrix, metadata = load_vector_matrix(curator_root)
    n, dim = matrix.shape
    print(f"Building index: {n} vectors, dim={dim}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    faiss = _import_faiss()
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(artifact_dir / INDEX_PATH.name))

    payload = {"entries": metadata}
    (artifact_dir / META_PATH.name).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("Backend      -> faiss")
    print(f"Index saved  -> {artifact_dir / INDEX_PATH.name}")
    print(f"Metadata     -> {artifact_dir / META_PATH.name}")
    print(f"Total entries: {index.ntotal}")
    return payload


def load_index(*, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    metadata_path = artifact_dir / META_PATH.name
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Index metadata not found at {metadata_path}. Run indexer.py first."
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = payload["entries"]
    faiss = _import_faiss()
    index = faiss.read_index(str(artifact_dir / INDEX_PATH.name))
    return {"index": index, "entries": entries}


def search_index(
    bundle: dict[str, Any],
    query_vector: np.ndarray,
    top_k: int,
) -> tuple[list[float], list[int]]:
    index = bundle["index"]
    scores, indices = index.search(query_vector, top_k)
    return scores[0].tolist(), indices[0].tolist()


def print_info(*, artifact_dir: Path = ARTIFACT_DIR) -> None:
    bundle = load_index(artifact_dir=artifact_dir)
    entries = bundle["entries"]
    print("Index backend: faiss")
    print(f"Index entries: {len(entries)}")
    print(f"\nAll indexed companies:")
    for i, m in enumerate(entries):
        print(f"  [{i:02d}] {m['ticker']:<6} {m['filing_year']}  {m['company']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", action="store_true", help="Print index stats without rebuilding")
    args = parser.parse_args()

    if args.info:
        print_info()
    else:
        build_index()
