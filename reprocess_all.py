import os
import subprocess
from pathlib import Path

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
    base_dir = Path("data-extraction/outputs")
    tickers = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "curator"]
    
    # Ensure OPENROUTER_API_KEY is available
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("❌ ERROR: OPENROUTER_API_KEY is not set.")
        return

    python_bin = "venv/bin/python3" # Using the user's venv

    print(f"Found tickers: {', '.join(tickers)}")

    for ticker in tickers:
        print(f"\n{'='*40}")
        print(f" PROCESSING TICKER: {ticker}")
        print(f"{'='*40}")

        # Step 1: Extraction
        extract_success = run_command(
            [python_bin, "data-extraction/main.py", ticker],
            f"Extraction for {ticker}"
        )
        if not extract_success:
            print(f"⚠️ Skipping {ticker} due to extraction failure.")
            continue

        # Step 2: Curation
        ticker_extract_dir = base_dir / ticker
        extraction_files = list(ticker_extract_dir.glob("*_extraction.json"))
        curator_out_dir = base_dir / "curator" / ticker
        curator_out_dir.mkdir(parents=True, exist_ok=True)

        for extraction_file in extraction_files:
            # Check if curator output already exists for this filing
            # Filename format: {ticker.lower()}_{year}.json
            # Extraction filename: {ticker.lower()}_{year}_extraction.json
            output_name = extraction_file.name.replace("_extraction.json", ".json")
            output_path = curator_out_dir / output_name

            force_tickers = {"BYND", "INTC", "GOOG", "LYFT", "META", "NVDA", "PLTR", "SNAP", "UBER"}
            if output_path.exists() and ticker not in force_tickers:
                print(f"⏩ Skipping curation for {extraction_file.name} (Output exists)")
                continue

            run_command(
                [python_bin, "data-extraction/curator_agent.py", str(extraction_file), str(curator_out_dir)],
                f"Curation for {extraction_file.name}"
            )

    # Step 3: Indexing
    print(f"\n{'='*40}")
    print(f" REBUILDING FAISS INDEX")
    print(f"{'='*40}")
    run_command(
        [python_bin, "rag-matching/indexer.py"],
        "Rebuilding FAISS index"
    )

    print("\n✅ All tickers re-processed and index updated!")

if __name__ == "__main__":
    main()
