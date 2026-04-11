"""
indexer.py

Builds a FAISS flat inner-product index from all embedded curator JSONs.
Since embedder.py uses normalize_embeddings=True, inner product == cosine similarity.

Saves:
  curator_db/faiss.index   — the FAISS index
  curator_db/metadata.json — list of {ticker, company, filing_year, path} in index order

Usage:
    python indexer.py          # build and save index
    python indexer.py --info   # print index stats only (no rebuild)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from curator_store import CURATOR_SOURCE_ROOT, load_vector_matrix
from runtime_compat import prepare_openmp_runtime

ARTIFACT_DIR = Path(__file__).resolve().parent / "index_artifacts"
INDEX_PATH = ARTIFACT_DIR / "faiss.index"
MATRIX_PATH = ARTIFACT_DIR / "matrix.npy"
META_PATH = ARTIFACT_DIR / "metadata.json"


def _try_import_faiss():
    try:
        prepare_openmp_runtime()
        import faiss  # type: ignore

        return faiss
    except ModuleNotFoundError:
        return None


def build_index(
    *,
    curator_root: Path = CURATOR_SOURCE_ROOT,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, Any]:
    matrix, metadata = load_vector_matrix(curator_root)
    n, dim = matrix.shape
    print(f"Building index: {n} vectors, dim={dim}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    faiss = _try_import_faiss()
    backend = "faiss" if faiss is not None else "numpy"

    if faiss is not None:
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        faiss.write_index(index, str(artifact_dir / INDEX_PATH.name))
        total = index.ntotal
    else:
        np.save(artifact_dir / MATRIX_PATH.name, matrix)
        total = int(n)

    payload = {"backend": backend, "entries": metadata}
    (artifact_dir / META_PATH.name).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Backend      -> {backend}")
    if backend == "faiss":
        print(f"Index saved  -> {artifact_dir / INDEX_PATH.name}")
    else:
        print(f"Matrix saved -> {artifact_dir / MATRIX_PATH.name}")
    print(f"Metadata     -> {artifact_dir / META_PATH.name}")
    print(f"Total entries: {total}")
    return payload


def load_index(*, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    metadata_path = artifact_dir / META_PATH.name
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Index metadata not found at {metadata_path}. Run indexer.py first."
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    backend = payload["backend"]
    entries = payload["entries"]

    if backend == "faiss":
        faiss = _try_import_faiss()
        if faiss is None:
            raise RuntimeError(
                "Index was built with FAISS, but faiss is not installed in this environment."
            )
        index = faiss.read_index(str(artifact_dir / INDEX_PATH.name))
    elif backend == "numpy":
        index = np.load(artifact_dir / MATRIX_PATH.name)
    else:
        raise ValueError(f"Unsupported index backend: {backend}")

    return {"backend": backend, "index": index, "entries": entries}


def search_index(
    bundle: dict[str, Any],
    query_vector: np.ndarray,
    top_k: int,
) -> tuple[list[float], list[int]]:
    backend = bundle["backend"]
    index = bundle["index"]

    if backend == "faiss":
        scores, indices = index.search(query_vector, top_k)
        return scores[0].tolist(), indices[0].tolist()

    matrix: np.ndarray = index
    similarities = matrix @ query_vector.reshape(-1)
    order = np.argsort(similarities)[::-1][:top_k]
    scores = [float(similarities[idx]) for idx in order]
    indices = [int(idx) for idx in order]
    return scores, indices


def print_info(*, artifact_dir: Path = ARTIFACT_DIR) -> None:
    bundle = load_index(artifact_dir=artifact_dir)
    entries = bundle["entries"]
    print(f"Index backend: {bundle['backend']}")
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
