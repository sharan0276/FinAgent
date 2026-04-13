import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Executes a command and returns the output/error."""
    print(f"\n🚀 {description}...")
    print(f"👉 Running: {' '.join(command)}")
    
    try:
        # Use subprocess.run to isolate the environment for each script
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=False, # Print directly to terminal for real-time progress
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description}")
        print(f"Error Code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Unified FinAgent Pipeline Orchestrator")
    parser.add_argument("ticker", help="Ticker symbol (e.g., META, AAPL, GOOG)")
    parser.add_argument("--years", type=int, default=5, help="Number of years to fetch (default: 5)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    python_bin = "venv/bin/python3"
    
    # Verify environment
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n⚠️  WARNING: OPENROUTER_API_KEY is not set.")
        print("You must run: export OPENROUTER_API_KEY='your-key-here' to use the Curator phase.")

    print(f"\n{'='*60}")
    print(f" FINANCIAL AGENT - FULL PIPELINE: {ticker}")
    print(f"{'='*60}")

    # Step 1: Ingestion
    ingestion_script = "data-ingestion/ingestion_pipeline.py"
    step1_success = run_command(
        [python_bin, ingestion_script, ticker, "--years", str(args.years)],
        f"STEP 1: Ingesting data for {ticker}"
    )
    if not step1_success: return

    # Step 2: Extraction
    # main.py in data-extraction runs the pipeline with the Barcode fix
    extraction_script = "data-extraction/main.py"
    step2_success = run_command(
        [python_bin, extraction_script, ticker],
        f"STEP 2: Extracting signals and deltas for {ticker}"
    )
    if not step2_success: return

    # Step 3: Curator AI Analysis
    # We need to find the extraction JSON files created in Step 2
    extraction_dir = Path("data-extraction/outputs") / ticker
    extraction_files = list(extraction_dir.glob("*_extraction.json"))
    
    if not extraction_files:
        print(f"❌ No extraction files found in {extraction_dir}. Skipping Curator.")
        return

    curator_script = "data-extraction/curator_agent.py"
    curator_out = f"data-extraction/outputs/curator/{ticker}"
    
    print(f"\n🤖 STEP 3: Running Curator AI on {len(extraction_files)} filings...")
    for extraction_file in extraction_files:
        run_command(
            [python_bin, curator_script, str(extraction_file), curator_out],
            f"Analyzing {extraction_file.name}"
        )

    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE FOR {ticker}!")
    print(f"📁 Results: data-extraction/outputs/curator/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
