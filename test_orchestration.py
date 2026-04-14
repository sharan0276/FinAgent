from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from orchestration.artifact_resolver import matched_context_paths
from orchestration.orchestration_pipeline import PipelineDependencies, run_orchestration
from orchestration.report_models import ComparisonReportResult


class OrchestrationTests(unittest.TestCase):
    def _make_temp_dir(self, name: str) -> Path:
        base = Path(__file__).resolve().parent / ".tmp_orchestration_tests"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{name}_{uuid.uuid4().hex}"
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_matched_context_paths_returns_anchor_plus_future_years(self) -> None:
        goog_paths = matched_context_paths("GOOG", 2024)
        self.assertEqual([path.stem for path in goog_paths], ["goog_2024", "goog_2025", "goog_2026"])

        meta_paths = matched_context_paths("META", 2024)
        self.assertEqual([path.stem for path in meta_paths], ["meta_2024", "meta_2025"])

    def test_orchestration_skips_existing_artifacts_and_writes_bundle(self) -> None:
        output_root = self._make_temp_dir("skip_existing")

        def fake_matcher(_input_file: Path, _top_k: int, *, repo_root: Path):
            return {
                "matches": [
                    {"ticker": "GOOG", "company": "Alphabet Inc.", "filing_year": 2024, "similarity": 0.95},
                    {"ticker": "META", "company": "Meta Platforms, Inc.", "filing_year": 2023, "similarity": 0.91},
                ]
            }

        def fake_report(_bundle):
            return ComparisonReportResult(
                status="completed",
                summary="Comparison complete.",
                model_name="test-model",
            )

        deps = PipelineDependencies(
            run_ingestion=lambda *_args, **_kwargs: self.fail("ingestion should be skipped"),
            run_extraction=lambda *_args, **_kwargs: self.fail("extraction should be skipped"),
            run_curator=lambda *_args, **_kwargs: self.fail("curator should be skipped"),
            run_matcher=fake_matcher,
            run_comparison_agent=fake_report,
        )

        artifact, output_path = run_orchestration("AAPL", output_root=output_root, deps=deps)

        self.assertEqual(artifact.bundle.run_metadata.status_by_step["ingestion"], "skipped_existing")
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["extraction"], "skipped_existing")
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["curator"], "skipped_existing")
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["rag"], "completed")
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["comparison"], "completed")
        self.assertEqual(len(artifact.bundle.matches), 2)
        self.assertTrue(output_path.exists())

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["bundle"]["target"]["ticker"], "AAPL")
        self.assertEqual(payload["comparison_report"]["status"], "completed")

    def test_orchestration_runs_missing_target_steps(self) -> None:
        repo_root = self._make_temp_dir("missing_steps")
        output_root = repo_root / "orchestration-out"
        calls: list[str] = []

        def fake_ingestion(ticker: str, *, repo_root: Path) -> Path:
            calls.append("ingestion")
            path = repo_root / "data-ingestion" / "outputs" / ticker / "complete_ingestion.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "ticker": ticker,
                        "company_name": "Test Corp",
                        "text_data": {"filings": [{"filingDate": "2025-01-31"}]},
                    }
                ),
                encoding="utf-8",
            )
            return path

        def fake_extraction(ticker: str, filing_year: int, *, repo_root: Path) -> Path:
            calls.append("extraction")
            path = repo_root / "data-extraction" / "outputs" / ticker / f"{ticker.lower()}_{filing_year}_extraction.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"ticker": ticker, "filing_year": filing_year}), encoding="utf-8")
            return path

        def fake_curator(extraction_path: Path, *, repo_root: Path) -> Path:
            calls.append("curator")
            path = repo_root / "data-extraction" / "outputs" / "curator" / "TEST" / "test_2025.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "company": "Test Corp",
                        "ticker": "TEST",
                        "filing_year": 2025,
                        "financial_deltas": {"Revenues": {"value": 0.1, "label": "moderate_growth"}},
                        "risk_signals": [],
                        "embedding_text": "test",
                        "embedding_vector": [0.1, 0.2],
                    }
                ),
                encoding="utf-8",
            )
            return path

        def fake_matcher(_input_file: Path, _top_k: int, *, repo_root: Path):
            goog_dir = repo_root / "data-extraction" / "outputs" / "curator" / "GOOG"
            goog_dir.mkdir(parents=True, exist_ok=True)
            for year in (2024, 2025):
                (goog_dir / f"goog_{year}.json").write_text(
                    json.dumps(
                        {
                            "company": "Alphabet Inc.",
                            "ticker": "GOOG",
                            "filing_year": year,
                            "financial_deltas": {},
                            "risk_signals": [],
                            "embedding_text": "x",
                            "embedding_vector": [0.1, 0.2],
                        }
                    ),
                    encoding="utf-8",
                )
            return {"matches": [{"ticker": "GOOG", "company": "Alphabet Inc.", "filing_year": 2024, "similarity": 0.8}]}

        def fake_report(_bundle):
            return ComparisonReportResult(status="failed", summary="No API key.", error="OPENROUTER_API_KEY is not set.")

        deps = PipelineDependencies(
            run_ingestion=fake_ingestion,
            run_extraction=fake_extraction,
            run_curator=fake_curator,
            run_matcher=fake_matcher,
            run_comparison_agent=fake_report,
        )

        artifact, output_path = run_orchestration(
            "TEST",
            top_k=2,
            output_root=output_root,
            repo_root=repo_root,
            deps=deps,
        )

        self.assertEqual(calls, ["ingestion", "extraction", "curator"])
        self.assertEqual(artifact.bundle.target.latest_filing_year, 2025)
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["ingestion"], "completed")
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["extraction"], "completed")
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["curator"], "completed")
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["rag"], "partial")
        self.assertEqual(artifact.bundle.run_metadata.status_by_step["comparison"], "failed")
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
