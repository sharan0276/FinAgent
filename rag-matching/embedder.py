"""
embedder.py

Embeds every curator JSON's embedding_text field using a local
sentence-transformers model and writes the vector back into the file.

Model: all-MiniLM-L6-v2  (384-dim, fast, free)
In production swap for OpenAI text-embedding-3-large (1536-dim)
by replacing _embed_batch() — the rest of the pipeline is dimension-agnostic.

Usage:
    python embedder.py               # embed all curator_db/*.json
    python embedder.py --force       # re-embed even if vector already present
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CURATOR_DB = Path(__file__).parent / "curator_db"
MODEL_NAME = "all-MiniLM-L6-v2"


def _load_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def _embed_batch(model, texts: list[str]) -> list[list[float]]:
    """Returns list of float vectors. Swap this function to change embedding provider."""
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_all(force: bool = False) -> None:
    curator_files = sorted(p for p in CURATOR_DB.glob("*.json") if p.name != "metadata.json")
    if not curator_files:
        print("No curator JSON files found. Run synthetic_curator.py first.")
        return

    # Filter to files that need embedding
    to_embed = []
    for path in curator_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if force or not data.get("embedding_vector"):
            to_embed.append(path)
        else:
            print(f"  [skip] {path.name} (already embedded)")

    if not to_embed:
        print("All files already embedded. Use --force to re-embed.")
        return

    model = _load_model()

    # Batch all texts for efficiency
    all_data = []
    all_texts = []
    for path in to_embed:
        data = json.loads(path.read_text(encoding="utf-8"))
        all_data.append((path, data))
        all_texts.append(data["embedding_text"])

    print(f"\nEmbedding {len(all_texts)} curator files...")
    vectors = _embed_batch(model, all_texts)

    # Write vectors back
    for (path, data), vector in zip(all_data, vectors):
        data["embedding_vector"] = vector
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  [embedded] {path.name}  dim={len(vector)}")

    print(f"\nDone. Embedded {len(to_embed)} files.")


def load_vectors(curator_db: Path = CURATOR_DB) -> tuple[np.ndarray, list[dict]]:
    """
    Load all embedded curator JSONs.

    Returns:
        matrix: np.ndarray shape (N, D) — all embedding vectors stacked
        metadata: list of dicts with {ticker, company, filing_year, path}
    """
    metadata = []
    vectors = []

    for path in sorted(p for p in curator_db.glob("*.json") if p.name != "metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        vec = data.get("embedding_vector")
        if not vec:
            print(f"  [warning] {path.name} has no embedding_vector — skipping")
            continue
        vectors.append(vec)
        metadata.append({
            "ticker":      data["ticker"],
            "company":     data["company"],
            "filing_year": data["filing_year"],
            "path":        str(path),
        })

    if not vectors:
        raise ValueError("No embedded curator files found. Run embedder.py first.")

    matrix = np.array(vectors, dtype="float32")
    return matrix, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-embed even if already present")
    args = parser.parse_args()
    embed_all(force=args.force)
