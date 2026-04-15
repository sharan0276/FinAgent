from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestration.artifact_resolver import REPO_ROOT, ingestion_artifact_path

from .documents import build_ingestion_retrieval_text
from .embeddings import embed_texts
from .indexer import ensure_index, search_index
from .ingestion_store import load_ingestion_file


def _search_headroom(total_entries: int, top_k: int) -> int:
    return min(total_entries, max(top_k * 10, top_k + 5))


def find_matches_for_ticker(
    ticker: str,
    *,
    top_k: int = 2,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[dict[str, Any]], str]:
    print(f"[baseline_rag] Finding top {top_k} peer match(es) for {ticker.upper()}...")
    bundle, index_status = ensure_index(repo_root=repo_root)
    entries = bundle["entries"]
    if not entries:
        raise ValueError("The saved baseline index contains no entries.")

    query_path = ingestion_artifact_path(ticker, repo_root=repo_root)
    query_ingestion = load_ingestion_file(query_path)
    query_vector = embed_texts([build_ingestion_retrieval_text(query_ingestion)])

    index_dim = int(bundle["index"].d)
    query_dim = int(query_vector.shape[1])
    if query_dim != index_dim:
        raise ValueError(
            f"Query embedding dimension {query_dim} does not match index dimension {index_dim}."
        )

    search_k = _search_headroom(len(entries), top_k)
    scores, indices = search_index(bundle, query_vector, search_k)

    best_by_ticker: dict[str, dict[str, Any]] = {}
    query_ticker = ticker.upper()

    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        entry = entries[idx]
        entry_ticker = str(entry.get("ticker", "")).upper()
        if entry_ticker == query_ticker:
            continue

        candidate = {
            "ticker": entry.get("ticker"),
            "company": entry.get("company"),
            "filing_year": entry.get("latest_filing_year"),
            "source_path": entry.get("source_path"),
            "similarity": round(float(score), 4),
        }
        current = best_by_ticker.get(entry_ticker)
        if current is None or candidate["similarity"] > current["similarity"]:
            best_by_ticker[entry_ticker] = candidate

    matches = sorted(best_by_ticker.values(), key=lambda item: item["similarity"], reverse=True)[:top_k]
    print(f"[baseline_rag] Found {len(matches)} peer match(es) for {ticker.upper()}.")
    return matches, index_status
