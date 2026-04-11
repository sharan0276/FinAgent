# Batch runner for Agent 2 (Curator).
# Takes a company ticker as input, processes all FinBERT extraction
# JSONs for that company, and writes curator output JSONs to a
# company-specific output folder.
#
# Usage:
#   python run_all.py
#   Enter ticker when prompted: AAPL
#
# Input:  data-extraction/outputs/AAPL/aapl_2021_extraction.json
# Output: data-extraction/outputs/curator/AAPL/aapl_2021.json

import sys
import time
from pathlib import Path
from curator_agent import run_curator

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_ROOT  = "data-extraction/outputs"
OUTPUT_ROOT = "data-extraction/outputs/curator"
TOP_N       = 46
API_DELAY   = 2  # seconds between API calls — prevents OpenRouter rate limiting


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ticker_from_args() -> str:
    """
    Reads the company ticker from the command line arguments.
    Validates that the corresponding input folder exists.
    """
    if len(sys.argv) < 2:
        print("Usage: python company_filing_embedding.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].strip().upper()

    company_dir = Path(INPUT_ROOT) / ticker
    if not company_dir.exists():
        print(f"  Error: Folder not found: {company_dir}")
        print(f"  Check that {ticker} exists under {INPUT_ROOT}/")
        sys.exit(1)

    return ticker


def find_extraction_files(ticker: str) -> list[Path]:
    """
    Finds all FinBERT extraction JSONs for the given ticker.
    Returns files sorted chronologically by filename.
    """
    company_dir = Path(INPUT_ROOT) / ticker
    files = sorted(company_dir.glob("*_extraction.json"))
    return files


# ── Main runner ───────────────────────────────────────────────────────────────

def run_for_company(ticker: str):
    # Output goes to data-extraction/outputs/curator/AAPL/
    output_dir = Path(OUTPUT_ROOT) / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_extraction_files(ticker)

    if not files:
        print(f"No extraction files found for {ticker} under {INPUT_ROOT}/{ticker}/")
        return

    print(f"\nFound {len(files)} file(s) for {ticker}:")
    for f in files:
        print(f"  {f.name}")

    print(f"\nOutput folder: {output_dir}")
    print("=" * 60)

    success_count = 0
    failed_files  = []

    for filepath in files:
        # Derive year from filename — e.g. aapl_2021_extraction → 2021
        parts = filepath.stem.split("_")
        year  = parts[1] if len(parts) >= 2 else "unknown"
        output_path = output_dir / f"{ticker.lower()}_{year}.json"

        # Skip if output already exists
        if output_path.exists():
            print(f"\n  Skipping {filepath.name} — already exists at {output_path.name}")
            success_count += 1
            continue

        print(f"\n  Processing: {filepath.name}")

        try:
            result = run_curator(
                finbert_json_path=str(filepath),
                output_dir=str(output_dir),
                top_n=TOP_N,
            )

            if result is not None:
                success_count += 1
                print(f"  ✓ Saved: {output_path.name}")
            else:
                failed_files.append(filepath.name)
                print(f"  ✗ Failed: {filepath.name} — run_curator returned None")

        except Exception as e:
            failed_files.append(filepath.name)
            print(f"  ✗ Error on {filepath.name}: {e}")

        time.sleep(API_DELAY)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Done: {ticker}")
    print(f"  Successful : {success_count} / {len(files)}")
    print(f"  Failed     : {len(failed_files)}")

    if failed_files:
        print(f"\n  Failed files — rerun manually:")
        for f in failed_files:
            print(f"    python curator_agent.py {INPUT_ROOT}/{ticker}/{f}")
    else:
        print(f"\n  All files written to: {output_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = get_ticker_from_args()
    run_for_company(ticker)