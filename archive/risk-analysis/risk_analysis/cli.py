from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .artifact_loader import DEFAULT_TARGET_SECTIONS, default_source_path
from .pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build yearly company analysis artifacts from ingestion outputs.")
    parser.add_argument("ticker", help="Ticker symbol matching an existing ingestion artifact.")
    parser.add_argument("--source", type=Path, help="Override the source complete_ingestion.json path.")
    parser.add_argument("--output-dir", type=Path, default=Path("risk-analysis") / "outputs", help="Base output directory.")
    parser.add_argument("--provider", default="openrouter", choices=["heuristic", "openrouter"], help="LLM provider.")
    parser.add_argument("--model", default=None, help="Model name to use. Defaults to MODEL_NAME from .env/environment.")
    parser.add_argument("--prompt-version", default="prompt_v1", help="Prompt version metadata.")
    parser.add_argument("--sections", nargs="*", default=None, help="Optional section ids override for debugging.")
    parser.add_argument("--year", type=int, default=None, help="Optional filing year filter, e.g. 2025.")
    parser.add_argument("--years", nargs="*", type=int, default=None, help="Optional filing years filter, e.g. --years 2024 2025.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress printing.")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Directory for raw model debug request/response dumps.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    ticker = args.ticker.upper()
    source_path = args.source or default_source_path(repo_root, ticker)
    output_path = args.output_dir / ticker / "company_analysis.json"
    debug_dir = args.debug_dir or (args.output_dir / ticker / "debug")
    filing_years = args.years or ([args.year] if args.year is not None else None)
    if not args.quiet:
        print(f"[risk-analysis] Source: {source_path}")
        print(f"[risk-analysis] Output base: {output_path.parent}")
        print(f"[risk-analysis] Provider: {args.provider}")
        print(f"[risk-analysis] Model: {args.model or os.getenv('MODEL_NAME')}")
        print(f"[risk-analysis] Filing years: {filing_years or 'all'}")
        print(f"[risk-analysis] Debug dir: {debug_dir}")
    written_paths = run_analysis(
        source_path=source_path,
        output_path=output_path,
        provider=args.provider,
        model_name=args.model or os.getenv("MODEL_NAME"),
        prompt_version=args.prompt_version,
        section_ids=args.sections or DEFAULT_TARGET_SECTIONS,
        filing_years=filing_years,
        verbose=not args.quiet,
        debug_dir=debug_dir,
    )
    if len(written_paths) == 1:
        print(f"Wrote analysis artifact to {written_paths[0]}")
    else:
        print(f"Wrote {len(written_paths)} analysis artifacts:")
        for path in written_paths:
            print(f"  {path}")
    return 0
