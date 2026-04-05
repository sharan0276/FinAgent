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

import faiss
import numpy as np

from embedder import load_vectors

CURATOR_DB = Path(__file__).parent / "curator_db"
INDEX_PATH = CURATOR_DB / "faiss.index"
META_PATH  = CURATOR_DB / "metadata.json"


def build_index() -> None:
    matrix, metadata = load_vectors(CURATOR_DB)
    n, dim = matrix.shape
    print(f"Building index: {n} vectors, dim={dim}")

    # Flat IP index — exact search, cosine similarity (vectors are L2-normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Index saved  -> {INDEX_PATH}")
    print(f"Metadata saved -> {META_PATH}")
    print(f"Total entries: {index.ntotal}")


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load saved index and metadata. Called by matcher.py."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. Run indexer.py first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, metadata


def print_info() -> None:
    index, metadata = load_index()
    print(f"Index entries: {index.ntotal}")
    print(f"\nAll indexed companies:")
    for i, m in enumerate(metadata):
        print(f"  [{i:02d}] {m['ticker']:<6} {m['filing_year']}  {m['company']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", action="store_true", help="Print index stats without rebuilding")
    args = parser.parse_args()

    if args.info:
        print_info()
    else:
        build_index()
