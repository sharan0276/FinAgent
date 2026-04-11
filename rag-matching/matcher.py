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

from embedder import load_vectors
from indexer import load_index

CURATOR_DB = Path(__file__).parent / "curator_db"


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def _load_curator(ticker: str, year: int) -> dict | None:
    """Load a curator JSON by ticker + year."""
    path = CURATOR_DB / f"{ticker.lower()}_{year}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_embedding(curator: dict) -> np.ndarray | None:
    vec = curator.get("embedding_vector")
    if not vec:
        return None
    return np.array(vec, dtype="float32").reshape(1, -1)


def _load_lookahead(ticker: str, match_year: int, years_ahead: int = 2) -> list[dict]:
    """Load 1-2 lookahead years for a matched company. Flags if missing."""
    lookaheads = []
    for offset in range(1, years_ahead + 1):
        yr = match_year + offset
        data = _load_curator(ticker, yr)
        lookaheads.append({
            "year": yr,
            "available": data is not None,
            "data": data,
        })
    return lookaheads


def find_matches(
    query_ticker: str,
    query_year: int | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Find top-k matches for a query company.

    Args:
        query_ticker: e.g. "SNAP"
        query_year:   e.g. 2022. If None, uses the most recent year available.
        top_k:        number of matches to return

    Returns dict with:
        query:   curator data for the query company
        matches: list of match dicts (ranked by similarity)
    """
    # Resolve year
    if query_year is None:
        # Find the most recent curator file for this ticker
        available = sorted(CURATOR_DB.glob(f"{query_ticker.lower()}_*.json"))
        if not available:
            raise FileNotFoundError(f"No curator files found for ticker '{query_ticker}'")
        query_year = int(available[-1].stem.split("_")[1])
        print(f"Using most recent year: {query_year}")

    query_curator = _load_curator(query_ticker, query_year)
    if query_curator is None:
        raise FileNotFoundError(
            f"No curator file for {query_ticker} {query_year}. "
            f"Run synthetic_curator.py first."
        )

    query_vec = _get_embedding(query_curator)
    if query_vec is None:
        raise ValueError(
            f"No embedding vector for {query_ticker} {query_year}. "
            f"Run embedder.py first."
        )

    # Load index
    index, metadata = load_index()

    # Search — request top_k + some extra to allow self-exclusion
    search_k = min(top_k + 5, index.ntotal)
    scores, indices = index.search(query_vec, search_k)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    matches = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        meta = metadata[idx]
        # Exclude exact same company-year
        if (meta["ticker"].upper() == query_ticker.upper()
                and meta["filing_year"] == query_year):
            continue
        if len(matches) >= top_k:
            break

        match_curator = _load_curator(meta["ticker"], meta["filing_year"])
        lookahead = _load_lookahead(meta["ticker"], meta["filing_year"])

        # Compute shared risk signal types
        query_signal_types = {s["signal_type"] for s in query_curator.get("risk_signals", [])}
        match_signal_types = {s["signal_type"] for s in (match_curator or {}).get("risk_signals", [])}
        shared_signals = sorted(query_signal_types & match_signal_types)

        matches.append({
            "rank":           len(matches) + 1,
            "ticker":         meta["ticker"],
            "company":        meta["company"],
            "filing_year":    meta["filing_year"],
            "similarity":     round(score, 4),
            "shared_signals": shared_signals,
            "lookahead":      lookahead,
            "curator":        match_curator,
        })

    return {"query": query_curator, "matches": matches}


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def _delta_bar(label: str) -> str:
    bars = {
        "strong_growth":    "+++",
        "moderate_growth":  "++ ",
        "stable":           "-- ",
        "moderate_decline": "vv ",
        "severe_decline":   "vvv",
    }
    return bars.get(label or "", "???")


def print_match_report(result: dict) -> None:
    query = result["query"]
    matches = result["matches"]

    q_ticker = query["ticker"]
    q_year   = query["filing_year"]
    q_company = query["company"]

    print("\n" + "=" * 70)
    print(f"MATCH REPORT — {q_company} ({q_ticker}) {q_year}")
    print("=" * 70)

    # Query financial profile
    print(f"\nQUERY FINANCIAL PROFILE:")
    for metric, data in query.get("financial_deltas", {}).items():
        label = data.get("label", "n/a")
        bar   = _delta_bar(label)
        print(f"  {metric:<26} {bar}  {label}")

    # Query risk signals
    sigs = query.get("risk_signals", [])
    if sigs:
        print(f"\nQUERY RISK SIGNALS ({len(sigs)}):")
        for s in sigs:
            sev_icon = {"high": "[H]", "medium": "[M]", "low": "[L]"}.get(s["severity"], "[ ]")
            print(f"  {sev_icon} [{s['topic']}] {s['signal_type']}")
            print(f"     {s['summary'][:100]}...")

    # Matches
    print(f"\nTOP {len(matches)} MATCHES:")
    print("-" * 70)
    for m in matches:
        print(f"\n  #{m['rank']}  {m['company']} ({m['ticker']}) {m['filing_year']}  "
              f"similarity={m['similarity']:.4f}")

        # Financial profile of match
        match_deltas = (m["curator"] or {}).get("financial_deltas", {})
        if match_deltas:
            metrics_line = "  ".join(
                f"{k[:3]}:{_delta_bar(v.get('label'))}"
                for k, v in match_deltas.items()
            )
            print(f"     Financials: {metrics_line}")

        # Shared signal types
        if m["shared_signals"]:
            print(f"     Shared risk signals: {', '.join(m['shared_signals'])}")
        else:
            print(f"     Shared risk signals: (none)")

        # Lookahead
        print(f"     Lookahead:")
        for la in m["lookahead"]:
            if la["available"]:
                la_data = la["data"]
                la_deltas = la_data.get("financial_deltas", {})
                # Show net income and cash as quick health indicators
                ni    = la_deltas.get("NetIncome", {}).get("label", "?")
                cash  = la_deltas.get("Cash", {}).get("label", "?")
                rev   = la_deltas.get("Revenues", {}).get("label", "?")
                la_sigs = la_data.get("risk_signals", [])
                high_count = sum(1 for s in la_sigs if s.get("severity") == "high")
                print(f"       {la['year']}: Rev={_delta_bar(rev)} NI={_delta_bar(ni)} Cash={_delta_bar(cash)}  high-risk-signals={high_count}")
            else:
                print(f"       {la['year']}: [MISSING] data not available")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find matching companies for a query ticker"
    )
    parser.add_argument("ticker", help="Ticker to query, e.g. SNAP")
    parser.add_argument("year", nargs="?", type=int, help="Filing year (default: most recent)")
    parser.add_argument("--top", type=int, default=5, help="Number of matches (default: 5)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of report")
    args = parser.parse_args()

    result = find_matches(args.ticker, args.year, top_k=args.top)

    if args.json:
        # Remove full curator data to keep output manageable
        for m in result["matches"]:
            m.pop("curator", None)
            for la in m["lookahead"]:
                la.pop("data", None)
        print(json.dumps(result, indent=2))
    else:
        print_match_report(result)


if __name__ == "__main__":
    main()
