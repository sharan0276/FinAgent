from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from artifact_loader import default_source_path
from pipeline import run_extraction
from scoring import FinBERTSentenceScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build unified yearly extraction artifacts from complete_ingestion.json")
    parser.add_argument("ticker", help="Ticker symbol matching an existing ingestion artifact.")
    parser.add_argument("--source", type=Path, help="Override the source complete_ingestion.json path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data-extraction") / "outputs",
        help="Base output directory.",
    )
    parser.add_argument("--year", type=int, default=None, help="Optional filing year filter.")
    parser.add_argument("--years", nargs="*", type=int, default=None, help="Optional filing years filter.")
    parser.add_argument("--model", default="ProsusAI/finbert", help="FinBERT model name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for sentence scoring.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    ticker = args.ticker.upper()
    source_path = args.source or default_source_path(repo_root, ticker)
    output_dir = args.output_dir / ticker
    filing_years = args.years or ([args.year] if args.year is not None else None)
    scorer = FinBERTSentenceScorer(model_name=args.model, batch_size=args.batch_size)
    written = run_extraction(
        source_path=source_path,
        output_dir=output_dir,
        scorer=scorer,
        filing_years=filing_years,
    )
    if len(written) == 1:
        print(f"Wrote extraction artifact to {written[0]}")
    else:
        print(f"Wrote {len(written)} extraction artifacts:")
        for path in written:
            print(f"  {path}")
    return 0
