from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from orchestration.orchestration_pipeline import run_orchestration
else:
    from orchestration.orchestration_pipeline import run_orchestration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end company comparison orchestrator.")
    parser.add_argument("ticker", help="Ticker symbol to analyze, e.g. AAPL")
    parser.add_argument("--top", type=int, default=2, help="Number of distinct company matches to return")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("orchestration") / "outputs",
        help="Base output directory for orchestration artifacts.",
    )
    parser.add_argument("--json", action="store_true", help="Print the final artifact as JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact, output_path = run_orchestration(
        args.ticker,
        top_k=args.top,
        output_root=args.output_dir,
    )
    if args.json:
        print(json.dumps(artifact.model_dump(mode="json"), indent=2))
    else:
        print(
            json.dumps(
                {
                    "ticker": artifact.bundle.target.ticker,
                    "latest_filing_year": artifact.bundle.target.latest_filing_year,
                    "matches": [match.model_dump(mode="json") for match in artifact.bundle.matches],
                    "comparison_status": artifact.comparison_report.status,
                    "output_path": str(output_path),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
