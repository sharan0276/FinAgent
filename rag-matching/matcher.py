"""
matcher.py

Core of Agent 3's matching logic.

Given a query company (ticker + optional year), finds the top-k most
similar companies from the curator database using cosine similarity
on embedding vectors.

Also loads 1-2 lookahead years per match so Agent 3 can answer:
"What happened next?"

Usage:
    python matcher.py SNAP              # match SNAP's most recent year
    python matcher.py META 2022 --top 5 # match META 2022 with top 5 results
    python matcher.py WE 2020 --top 3   # match WeWork 2020

Output:
    Prints a formatted match report showing:
    - Query company financial profile
    - Top-k matches with similarity scores
    - Shared risk themes
    - Lookahead data (what happened 1-2 years after the match year)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from curator_store import (
    CURATOR_EMBEDDING_MODEL,
    CURATOR_SOURCE_ROOT,
    get_stored_embedding,
    load_curator_file,
)
from indexer import build_index, load_index, search_index
from runtime_compat import prepare_openmp_runtime


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def _embed_query_text(text: str) -> np.ndarray:
    try:
        prepare_openmp_runtime()
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Query file is missing embedding_vector and sentence_transformers is not installed."
        ) from exc

    model = SentenceTransformer(CURATOR_EMBEDDING_MODEL)
    vector = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(vector, dtype="float32")


def _load_query(input_file: Path) -> tuple[dict[str, Any], np.ndarray]:
    curator = load_curator_file(input_file)
    vector = get_stored_embedding(curator)
    if vector is None:
        embedding_text = curator.get("embedding_text")
        if not embedding_text:
            raise ValueError(
                f"{input_file} is missing both embedding_vector and embedding_text."
            )
        vector = _embed_query_text(str(embedding_text))
    return curator, vector


def _search_headroom(total_entries: int, top_k: int) -> int:
    return min(total_entries, max(top_k * 10, top_k + 5))


def find_matches(
    input_file: str | Path,
    *,
    top_k: int = 2,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    query_path = Path(input_file).resolve()
    query_curator, query_vector = _load_query(query_path)
    bundle = load_index() if artifact_dir is None else load_index(artifact_dir=artifact_dir)
    entries = bundle["entries"]

    if not entries:
        raise ValueError("The saved index contains no entries.")

    index_dim = int(bundle["index"].d) if bundle["backend"] == "faiss" else int(bundle["index"].shape[1])
    query_dim = int(query_vector.shape[1])
    if query_dim != index_dim:
        raise ValueError(
            f"Query embedding dimension {query_dim} does not match index dimension {index_dim}. "
            f"Expected the query embedding model to match curator generation ({CURATOR_EMBEDDING_MODEL})."
        )

    search_k = _search_headroom(len(entries), top_k)
    scores, indices = search_index(bundle, query_vector, search_k)

    best_by_ticker: dict[str, dict[str, Any]] = {}
    query_ticker = str(query_curator["ticker"]).upper()

    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        entry = entries[idx]
        entry_path = Path(entry["source_path"]).resolve()
        entry_ticker = str(entry["ticker"]).upper()

        if entry_path == query_path:
            continue
        if entry_ticker == query_ticker:
            continue

        candidate = {
            "ticker": entry["ticker"],
            "company": entry["company"],
            "filing_year": entry["filing_year"],
            "source_path": entry["source_path"],
            "similarity": round(float(score), 4),
        }
        current = best_by_ticker.get(entry_ticker)
        if current is None or candidate["similarity"] > current["similarity"]:
            best_by_ticker[entry_ticker] = candidate

    matches = sorted(
        best_by_ticker.values(),
        key=lambda item: item["similarity"],
        reverse=True,
    )[:top_k]

    return {
        "query": {
            "ticker": query_curator["ticker"],
            "company": query_curator["company"],
            "filing_year": query_curator["filing_year"],
            "source_path": str(query_path),
        },
        "matches": [{"ticker": m["ticker"], "similarity": m["similarity"]} for m in matches],
        "match_details": matches,
    }


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_match_report(result: dict[str, Any]) -> None:
    query = result["query"]
    matches = result["matches"]
    print(f"Query: {query['ticker']} {query['filing_year']} ({query['company']})")
    print("Matches:")
    for match in matches:
        print(f"  {match['ticker']}: {match['similarity']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find top distinct company matches for a curator JSON input file."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=CURATOR_SOURCE_ROOT / "AAPL" / "aapl_2022.json",
        help="Path to a curator JSON query file.",
    )
    parser.add_argument("--top", type=int, default=2, help="Number of distinct company matches")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of report")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the persistent index before matching",
    )
    args = parser.parse_args()

    if args.build_index:
        build_index()

    result = find_matches(args.input_file, top_k=args.top)

    if args.json:
        print(json.dumps({"query": result["query"], "matches": result["matches"]}, indent=2))
    else:
        print_match_report(result)


if __name__ == "__main__":
    main()
