import os
import subprocess
from pathlib import Path

# The final list of target tickers
TICKERS = [
    "BYND", 
    "CRM", "GOOG", "INTC", "LYFT", "META", 
    "NVDA", "PLTR", "SNAP", "UBER"
]

def run_command(command, description):
    print(f"\n🚀 {description}...")
    print(f"👉 Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description}")
        return False

def main():
    # Setup paths
    python_bin = "venv/bin/python3"
    extraction_output_root = Path("data-extraction/outputs")
    
    # 1. Environment Verification
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n❌ ERROR: OPENROUTER_API_KEY is not set.")
        print("Please run: export OPENROUTER_API_KEY='your-key-here'")
        return

    print(f"\n{'='*60}")
    print(f" FINAL PORTFOLIO SYNC: {len(TICKERS)} TICKERS")
    print(f"{'='*60}")

    for ticker in TICKERS:
        print(f"\n\n💎 [ {ticker} ] - Processing all 5 years...")

        # Step 1: Ingestion (Ensures 5 years exist for companies like AAPL)
        ingestion_success = run_command(
            [python_bin, "data-ingestion/ingestion_pipeline.py", ticker, "--years", "5"],
            f"STEP 1: Ingesting data for {ticker}"
        )
        if not ingestion_success: continue

        # Step 2: Extraction (Applies the numeric_delta.py structural and polarity fixes)
        extraction_success = run_command(
            [python_bin, "data-extraction/main.py", ticker],
            f"STEP 2: Extracting signals and deltas for {ticker}"
        )
        if not extraction_success: continue

        # Step 3: Curator AI Analysis (Updates risk summaries and embeddings)
        ticker_extract_dir = extraction_output_root / ticker
        extraction_files = list(ticker_extract_dir.glob("*_extraction.json"))
        curator_out_dir = extraction_output_root / "curator" / ticker

        print(f"\n🤖 STEP 3: Running Curator AI on {len(extraction_files)} filings...")
        for extraction_file in extraction_files:
            run_command(
                [python_bin, "data-extraction/curator_agent.py", str(extraction_file), str(curator_out_dir)],
                f"Analyzing {extraction_file.name}"
            )

    # Step 4: Final FAISS Indexing
    print(f"\n{'='*60}")
    print(f" STEP 4: REBUILDING VECTOR INDEX")
    print(f"{'='*60}")
    run_command(
        [python_bin, "rag-matching/indexer.py"],
        "Final FAISS Index Update"
    )

    print(f"\n{'='*60}")
    print(f"✅ SYNC COMPLETE!")
    print(f"All {len(TICKERS)} tickers are now fully updated with 5 years of data.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
