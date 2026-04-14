from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .deterministic import score_evaluation_input
from .judge import EvaluationJudge
from .loaders import build_evaluation_input, discover_artifact_paths
from .models import BatchEvaluationOutput, EvaluationResult, PairwiseComparisonResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved agentic and baseline report artifacts.")
    parser.add_argument("--agentic-dir", type=Path, help="Directory of saved agentic artifacts.")
    parser.add_argument("--baseline-dir", type=Path, help="Directory of saved baseline artifacts.")
    parser.add_argument("--agentic-artifact", type=Path, help="Single saved agentic artifact.")
    parser.add_argument("--baseline-artifact", type=Path, help="Single saved baseline artifact.")
    parser.add_argument("--ticker", action="append", help="Optional ticker filter. Repeatable.")
    parser.add_argument("--judge", action="store_true", help="Enable OpenRouter-backed judge scoring.")
    parser.add_argument("--json", action="store_true", help="Write JSON output artifact.")
    parser.add_argument("--csv", action="store_true", help="Write CSV output artifact.")
    parser.add_argument("--run-name", help="Stable output name. Defaults to a timestamped run id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation") / "outputs",
        help="Directory for saved evaluation outputs.",
    )
    return parser


def _collect_paths(single_path: Path | None, directory: Path | None) -> list[Path]:
    paths: list[Path] = []
    if single_path:
        paths.extend(discover_artifact_paths(single_path))
    if directory:
        paths.extend(discover_artifact_paths(directory))
    return sorted(set(paths))


def _filter_inputs(inputs: list, ticker_filter: set[str]) -> list:
    if not ticker_filter:
        return inputs
    return [item for item in inputs if item.ticker.upper() in ticker_filter]


def _compare_results(results: list[EvaluationResult]) -> list[PairwiseComparisonResult]:
    by_ticker: dict[str, dict[str, EvaluationResult]] = {}
    for result in results:
        by_ticker.setdefault(result.ticker, {})[result.pipeline] = result

    comparisons: list[PairwiseComparisonResult] = []
    dimensions = [
        "deterministic_consistency",
        "evidence_coverage",
        "claim_support",
        "comparative_usefulness",
        "overall_score",
    ]
    for ticker, pair in sorted(by_ticker.items()):
        agentic = pair.get("agentic")
        baseline = pair.get("baseline")
        comparison = PairwiseComparisonResult(
            ticker=ticker,
            agentic_score=agentic.component_scores.overall_score if agentic else None,
            baseline_score=baseline.component_scores.overall_score if baseline else None,
        )
        if agentic and baseline:
            comparison.score_delta = round(agentic.component_scores.overall_score - baseline.component_scores.overall_score, 4)
            for dimension in dimensions:
                agentic_value = getattr(agentic.component_scores, dimension)
                baseline_value = getattr(baseline.component_scores, dimension)
                if agentic_value is None or baseline_value is None:
                    continue
                if agentic_value > baseline_value:
                    comparison.agentic_wins.append(dimension)
                elif baseline_value > agentic_value:
                    comparison.baseline_wins.append(dimension)
                else:
                    comparison.ties.append(dimension)
        comparisons.append(comparison)
    return comparisons


def run_batch(
    *,
    agentic_paths: list[Path],
    baseline_paths: list[Path],
    judge_enabled: bool,
    ticker_filter: set[str] | None = None,
) -> BatchEvaluationOutput:
    ticker_filter = ticker_filter or set()
    inputs = _filter_inputs(
        [build_evaluation_input(path) for path in [*agentic_paths, *baseline_paths]],
        ticker_filter,
    )
    judge = EvaluationJudge() if judge_enabled else None
    results: list[EvaluationResult] = []

    for evaluation_input in inputs:
        claim_assessments = []
        claim_support = None
        report_judgement = {}
        comparative_usefulness = None
        judge_metadata = {}

        if judge is not None:
            claims = judge.extract_claims(evaluation_input)
            claim_assessments, claim_support = judge.assess_claims(evaluation_input, claims)
            report_judgement, comparative_usefulness = judge.assess_report(evaluation_input)
            judge_metadata = {"claims": claims, "report_judgement": report_judgement}

        score, warnings = score_evaluation_input(
            evaluation_input,
            claim_support=claim_support,
            comparative_usefulness=comparative_usefulness,
            report_judgement=report_judgement,
        )
        results.append(
            EvaluationResult(
                ticker=evaluation_input.ticker,
                pipeline=evaluation_input.pipeline,
                artifact_path=evaluation_input.artifact_path,
                artifact_hash=evaluation_input.artifact_hash,
                component_scores=score,
                warnings=warnings,
                claim_assessments=claim_assessments,
                judge_metadata=judge_metadata,
            )
        )

    run_name = datetime.now(timezone.utc).strftime("eval_%Y%m%dT%H%M%SZ")
    return BatchEvaluationOutput(
        run_name=run_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        results=results,
        comparisons=_compare_results(results),
    )


def _write_outputs(output: BatchEvaluationOutput, *, output_dir: Path, run_name: str, write_json: bool, write_csv: bool) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if write_json:
        json_path = output_dir / f"{run_name}.json"
        json_path.write_text(json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8")
        written.append(json_path)
    if write_csv:
        csv_path = output_dir / f"{run_name}.csv"
        rows = []
        for result in output.results:
            rows.append(
                {
                    "ticker": result.ticker,
                    "pipeline": result.pipeline,
                    "overall_score": result.component_scores.overall_score,
                    "deterministic_consistency": result.component_scores.deterministic_consistency,
                    "evidence_coverage": result.component_scores.evidence_coverage,
                    "claim_support": result.component_scores.claim_support,
                    "comparative_usefulness": result.component_scores.comparative_usefulness,
                    "overreach_penalty": result.component_scores.overreach_penalty,
                }
            )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["ticker", "pipeline"])
            writer.writeheader()
            writer.writerows(rows)
        written.append(csv_path)
    return written


def main() -> int:
    args = build_parser().parse_args()
    agentic_paths = _collect_paths(args.agentic_artifact, args.agentic_dir)
    baseline_paths = _collect_paths(args.baseline_artifact, args.baseline_dir)
    output = run_batch(
        agentic_paths=agentic_paths,
        baseline_paths=baseline_paths,
        judge_enabled=args.judge,
        ticker_filter={ticker.upper() for ticker in (args.ticker or [])},
    )
    run_name = args.run_name or output.run_name
    output.run_name = run_name
    written = _write_outputs(
        output,
        output_dir=args.output_dir,
        run_name=run_name,
        write_json=args.json or not args.csv,
        write_csv=args.csv,
    )
    print(json.dumps({"run_name": run_name, "written_paths": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
