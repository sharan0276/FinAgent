from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from orchestration.orchestration_pipeline import PipelineDependencies
from orchestration.report_models import ComparisonBundle, ComparisonReportResult, OrchestrationArtifact, RunMetadata, TargetContext
from ui.services import (
    build_company_dataset_for_ui,
    get_faiss_index_status,
    get_ticker_dataset_status,
    list_saved_report_artifacts,
    load_saved_report_artifact,
    rebuild_faiss_index,
    run_baseline_rag_for_ui,
)


class UIServicesTests(unittest.TestCase):
    def _make_temp_dir(self, name: str) -> Path:
        base = Path(__file__).resolve().parent / ".tmp_ui_services_tests"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{name}_{uuid.uuid4().hex}"
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _write_report_artifact(self, repo_root: Path, ticker: str = "TEST") -> Path:
        artifact = OrchestrationArtifact(
            bundle=ComparisonBundle(
                target=TargetContext(
                    ticker=ticker,
                    company="Test Corp",
                    latest_filing_year=2025,
                ),
                matches=[],
                run_metadata=RunMetadata(
                    top_k=2,
                    run_timestamp="2026-04-13T00:00:00+00:00",
                    status_by_step={"comparison": "completed"},
                    warnings=[],
                ),
            ),
            comparison_report=ComparisonReportResult(
                status="completed",
                summary="Ready.",
                model_name="test-model",
            ),
        )
        output_path = repo_root / "orchestration" / "outputs" / ticker / f"{ticker.lower()}_comparison_bundle.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2), encoding="utf-8")
        return output_path

    def test_list_and_load_saved_report_artifacts(self) -> None:
        repo_root = self._make_temp_dir("saved_reports")
        output_path = self._write_report_artifact(repo_root)

        items = list_saved_report_artifacts(repo_root=repo_root)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["ticker"], "TEST")
        loaded = load_saved_report_artifact(output_path)
        self.assertEqual(loaded.comparison_report.summary, "Ready.")

    def test_build_company_dataset_skips_existing_artifacts(self) -> None:
        repo_root = self._make_temp_dir("dataset_existing")
        ticker = "TEST"
        ingestion_path = repo_root / "data-ingestion" / "outputs" / ticker / "complete_ingestion.json"
        ingestion_path.parent.mkdir(parents=True, exist_ok=True)
        ingestion_path.write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "company_name": "Test Corp",
                    "text_data": {
                        "filings": [
                            {"filingDate": "2024-01-31"},
                            {"filingDate": "2025-01-31"},
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        extraction_dir = repo_root / "data-extraction" / "outputs" / ticker
        extraction_dir.mkdir(parents=True, exist_ok=True)
        curator_dir = repo_root / "data-extraction" / "outputs" / "curator" / ticker
        curator_dir.mkdir(parents=True, exist_ok=True)
        for year in (2024, 2025):
            (extraction_dir / f"{ticker.lower()}_{year}_extraction.json").write_text(
                json.dumps({"ticker": ticker, "filing_year": year}),
                encoding="utf-8",
            )
            (curator_dir / f"{ticker.lower()}_{year}.json").write_text(
                json.dumps({"ticker": ticker, "filing_year": year}),
                encoding="utf-8",
            )

        deps = PipelineDependencies(
            run_ingestion=lambda *_args, **_kwargs: self.fail("ingestion should be skipped"),
            run_extraction=lambda *_args, **_kwargs: self.fail("single extraction should not run"),
            run_extractions=lambda *_args, **_kwargs: self.fail("batch extraction should be skipped"),
            run_curator=lambda *_args, **_kwargs: self.fail("single curator should not run"),
            run_curators=lambda *_args, **_kwargs: self.fail("batch curator should be skipped"),
        )

        result = build_company_dataset_for_ui(ticker, repo_root=repo_root, deps=deps)

        self.assertIsNone(result["error"])
        self.assertEqual(result["status_by_step"]["ingestion"], "skipped_existing")
        self.assertEqual(result["status_by_step"]["extraction"], "skipped_existing")
        self.assertEqual(result["status_by_step"]["curator"], "skipped_existing")
        self.assertEqual(len(result["extraction_paths"]), 2)
        self.assertEqual(len(result["curator_paths"]), 2)

    def test_build_company_dataset_runs_missing_extraction_and_curator(self) -> None:
        repo_root = self._make_temp_dir("dataset_missing")
        ticker = "TEST"
        calls: list[str] = []

        ingestion_path = repo_root / "data-ingestion" / "outputs" / ticker / "complete_ingestion.json"
        ingestion_path.parent.mkdir(parents=True, exist_ok=True)
        ingestion_path.write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "company_name": "Test Corp",
                    "text_data": {
                        "filings": [
                            {"filingDate": "2024-01-31"},
                            {"filingDate": "2025-01-31"},
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        def fake_run_extractions(ticker: str, filing_years: list[int], *, repo_root: Path) -> list[Path]:
            calls.append(f"extraction:{','.join(str(year) for year in filing_years)}")
            output_dir = repo_root / "data-extraction" / "outputs" / ticker
            output_dir.mkdir(parents=True, exist_ok=True)
            paths: list[Path] = []
            for year in filing_years:
                path = output_dir / f"{ticker.lower()}_{year}_extraction.json"
                path.write_text(json.dumps({"ticker": ticker, "filing_year": year}), encoding="utf-8")
                paths.append(path)
            return paths

        def fake_run_curators(extraction_paths: list[Path], *, repo_root: Path) -> list[Path]:
            calls.append(f"curator:{len(extraction_paths)}")
            output_dir = repo_root / "data-extraction" / "outputs" / "curator" / ticker
            output_dir.mkdir(parents=True, exist_ok=True)
            written: list[Path] = []
            for extraction_path in extraction_paths:
                year = int(extraction_path.stem.split("_")[1])
                output_path = output_dir / f"{ticker.lower()}_{year}.json"
                output_path.write_text(json.dumps({"ticker": ticker, "filing_year": year}), encoding="utf-8")
                written.append(output_path)
            return written

        deps = PipelineDependencies(
            run_extractions=fake_run_extractions,
            run_curators=fake_run_curators,
        )

        result = build_company_dataset_for_ui(ticker, repo_root=repo_root, deps=deps)

        self.assertIsNone(result["error"])
        self.assertEqual(calls, ["extraction:2024,2025", "curator:2"])
        self.assertEqual(result["status_by_step"]["extraction"], "completed")
        self.assertEqual(result["status_by_step"]["curator"], "completed")
        self.assertEqual(len(result["curator_paths"]), 2)

    def test_build_company_dataset_surfaces_curator_failure(self) -> None:
        repo_root = self._make_temp_dir("dataset_curator_failure")
        ticker = "TEST"
        ingestion_path = repo_root / "data-ingestion" / "outputs" / ticker / "complete_ingestion.json"
        ingestion_path.parent.mkdir(parents=True, exist_ok=True)
        ingestion_path.write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "company_name": "Test Corp",
                    "text_data": {"filings": [{"filingDate": "2025-01-31"}]},
                }
            ),
            encoding="utf-8",
        )

        def fake_run_extractions(ticker: str, filing_years: list[int], *, repo_root: Path) -> list[Path]:
            output_dir = repo_root / "data-extraction" / "outputs" / ticker
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{ticker.lower()}_2025_extraction.json"
            path.write_text(json.dumps({"ticker": ticker, "filing_year": 2025}), encoding="utf-8")
            return [path]

        def fake_run_curators(_paths: list[Path], *, repo_root: Path) -> list[Path]:
            raise RuntimeError("curator failed")

        deps = PipelineDependencies(
            run_extractions=fake_run_extractions,
            run_curators=fake_run_curators,
        )

        result = build_company_dataset_for_ui(ticker, repo_root=repo_root, deps=deps)

        self.assertEqual(result["status_by_step"]["curator"], "failed")
        self.assertEqual(result["error"], "curator failed")

    def test_build_company_dataset_surfaces_invalid_ticker(self) -> None:
        repo_root = self._make_temp_dir("dataset_invalid_ticker")

        def fake_ingestion(_ticker: str, *, repo_root: Path) -> Path:
            raise RuntimeError("ticker not found")

        deps = PipelineDependencies(run_ingestion=fake_ingestion)

        result = build_company_dataset_for_ui("BAD", repo_root=repo_root, deps=deps)

        self.assertEqual(result["status_by_step"]["ingestion"], "failed")
        self.assertEqual(result["error"], "ticker not found")

    def test_get_ticker_dataset_status(self) -> None:
        repo_root = self._make_temp_dir("dataset_status")
        ticker = "TEST"
        ingestion_path = repo_root / "data-ingestion" / "outputs" / ticker / "complete_ingestion.json"
        extraction_path = repo_root / "data-extraction" / "outputs" / ticker / "test_2025_extraction.json"
        curator_path = repo_root / "data-extraction" / "outputs" / "curator" / ticker / "test_2025.json"
        ingestion_path.parent.mkdir(parents=True, exist_ok=True)
        extraction_path.parent.mkdir(parents=True, exist_ok=True)
        curator_path.parent.mkdir(parents=True, exist_ok=True)
        ingestion_path.write_text("{}", encoding="utf-8")
        extraction_path.write_text("{}", encoding="utf-8")
        curator_path.write_text("{}", encoding="utf-8")

        status = get_ticker_dataset_status(ticker, repo_root=repo_root)

        self.assertTrue(status["ingestion_exists"])
        self.assertEqual(status["extraction_count"], 1)
        self.assertEqual(status["curator_count"], 1)

    def test_get_faiss_index_status_reads_metadata(self) -> None:
        repo_root = self._make_temp_dir("faiss_status")
        artifact_dir = repo_root / "rag-matching" / "index_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "faiss.index").write_text("index", encoding="utf-8")
        (artifact_dir / "metadata.json").write_text(
            json.dumps({"entries": [{"ticker": "AAPL"}, {"ticker": "META"}]}),
            encoding="utf-8",
        )

        status = get_faiss_index_status(repo_root=repo_root)

        self.assertTrue(status["index_exists"])
        self.assertEqual(status["entry_count"], 2)

    def test_rebuild_faiss_index_returns_metadata(self) -> None:
        repo_root = self._make_temp_dir("faiss_rebuild")
        rag_dir = repo_root / "rag-matching"
        artifact_dir = rag_dir / "index_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        indexer_path = rag_dir / "indexer.py"
        indexer_path.parent.mkdir(parents=True, exist_ok=True)
        indexer_path.write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "import json",
                    "",
                    "def build_index():",
                    "    artifact_dir = Path(__file__).resolve().parent / 'index_artifacts'",
                    "    artifact_dir.mkdir(parents=True, exist_ok=True)",
                    "    (artifact_dir / 'faiss.index').write_text('index', encoding='utf-8')",
                    "    payload = {'entries': [{'ticker': 'AAPL'}]}",
                    "    (artifact_dir / 'metadata.json').write_text(json.dumps(payload), encoding='utf-8')",
                    "    return payload",
                ]
            ),
            encoding="utf-8",
        )

        status = rebuild_faiss_index(repo_root=repo_root)

        self.assertTrue(status["index_exists"])
        self.assertEqual(status["entry_count"], 1)

    def test_rebuild_faiss_index_surfaces_missing_dependency(self) -> None:
        repo_root = self._make_temp_dir("faiss_missing")
        rag_dir = repo_root / "rag-matching"
        rag_dir.mkdir(parents=True, exist_ok=True)
        (rag_dir / "indexer.py").write_text(
            "\n".join(
                [
                    "def build_index():",
                    "    raise RuntimeError('FAISS is required for rag-matching. Install faiss-cpu in the active environment.')",
                ]
            ),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            rebuild_faiss_index(repo_root=repo_root)

    def test_rebuild_faiss_index_surfaces_no_curator_files(self) -> None:
        repo_root = self._make_temp_dir("faiss_no_curators")
        rag_dir = repo_root / "rag-matching"
        rag_dir.mkdir(parents=True, exist_ok=True)
        (rag_dir / "indexer.py").write_text(
            "\n".join(
                [
                    "def build_index():",
                    "    raise RuntimeError('No curator files with embedding_vector found under data-extraction/outputs/curator')",
                ]
            ),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            rebuild_faiss_index(repo_root=repo_root)

    def test_run_baseline_rag_for_ui_returns_artifact(self) -> None:
        repo_root = self._make_temp_dir("baseline_ui")
        ticker = "TEST"
        ingestion_path = repo_root / "data-ingestion" / "outputs" / ticker / "complete_ingestion.json"
        peer_path = repo_root / "data-ingestion" / "outputs" / "PEER" / "complete_ingestion.json"
        ingestion_path.parent.mkdir(parents=True, exist_ok=True)
        peer_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ticker": ticker,
            "company_name": "Test Corp",
            "financial_data": {
                "annual": {
                    "Revenues": {
                        "years": [2024, 2025],
                        "values": [100.0, 110.0],
                        "deltas": [None, 10.0],
                        "unit": "USD_millions",
                    }
                }
            },
            "text_data": {"filings": [{"filingDate": "2025-10-31", "sections": {"item_7_mda": "Growth"}}]},
        }
        ingestion_path.write_text(json.dumps(payload), encoding="utf-8")
        peer_path.write_text(json.dumps({**payload, "ticker": "PEER", "company_name": "Peer Corp"}), encoding="utf-8")

        class FakeClient:
            def complete_json(self, *, system_prompt: str, user_prompt: str):
                return (
                    {
                        "status": "completed",
                        "summary": "Baseline ready.",
                        "posture": {"label": "Mixed", "rationale_bullets": ["Stable trend"]},
                        "target_profile": {"ticker": ticker, "company": "Test Corp", "filing_year": 2025, "positive_deltas": [], "negative_deltas": [], "top_risks": []},
                        "peer_snapshot": {"peer_group": "Peers", "common_strengths": [], "common_pressures": [], "shared_risk_types": [], "target_differences": []},
                        "risk_overlap_rows": [],
                        "forward_watchlist": [],
                        "narrative_sections": [{"title": "Financial Performance", "content": "OK"}, {"title": "Risk Assessment", "content": "OK"}],
                        "error": None,
                    },
                    "test-model",
                )

        from unittest.mock import patch
        import numpy as np

        with patch("baseline_rag.indexer.load_vector_matrix") as load_vector_matrix, patch(
            "baseline_rag.matcher.embed_texts"
        ) as matcher_embed, patch("ui.services.run_baseline_rag") as service_runner:
            load_vector_matrix.return_value = (
                np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype="float32"),
                [
                    {"ticker": ticker, "company": "Test Corp", "latest_filing_year": 2025, "source_path": str(ingestion_path.resolve()), "modified_time": ingestion_path.stat().st_mtime},
                    {"ticker": "PEER", "company": "Peer Corp", "latest_filing_year": 2025, "source_path": str(peer_path.resolve()), "modified_time": peer_path.stat().st_mtime},
                ],
            )
            matcher_embed.return_value = np.asarray([[1.0, 0.0]], dtype="float32")

            from baseline_rag.pipeline import run_baseline_rag

            service_runner.side_effect = lambda *args, **kwargs: run_baseline_rag(*args, client=FakeClient(), **kwargs)
            artifact = run_baseline_rag_for_ui(ticker, repo_root=repo_root)

        self.assertEqual(artifact.schema_version, "baseline_rag_v1")
        self.assertEqual(artifact.bundle.target.ticker, ticker)
        self.assertEqual(artifact.comparison_report.status, "completed")


if __name__ == "__main__":
    unittest.main()
