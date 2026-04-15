from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from baseline_rag import flatten_ingestion_to_text
from baseline_rag.documents import build_ingestion_retrieval_text
from baseline_rag.indexer import build_index
from baseline_rag.matcher import find_matches_for_ticker
from baseline_rag.pipeline import run_baseline_rag
from orchestration.report_models import OrchestrationArtifact


class BaselineRAGCompatibilityTests(unittest.TestCase):
    def _make_temp_dir(self, name: str) -> Path:
        base = Path(__file__).resolve().parent / ".tmp_rag_tests"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{name}_{uuid.uuid4().hex}"
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _write_ingestion_artifact(
        self,
        repo_root: Path,
        ticker: str,
        *,
        revenue: float,
        year: int = 2025,
        risk_text: str = "Supply chain concentration remains a key risk.",
    ) -> Path:
        path = repo_root / "data-ingestion" / "outputs" / ticker / "complete_ingestion.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ticker": ticker,
            "company_name": f"{ticker} Corp",
            "financial_data": {
                "annual": {
                    "Revenues": {
                        "years": [year - 1, year],
                        "values": [revenue - 10.0, revenue],
                        "deltas": [None, 10.0],
                        "unit": "USD_millions",
                    }
                }
            },
            "text_data": {
                "filings": [
                    {
                        "form": "10-K",
                        "filingDate": f"{year}-10-31",
                        "sections": {
                            "item_1a_risk_factors": risk_text,
                            "item_7_mda": f"{ticker} margin performance improved in {year}.",
                        },
                    }
                ]
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_flatten_ingestion_handles_current_metric_arrays(self) -> None:
        ingestion = {
            "ticker": "TEST",
            "company_name": "Test Corp",
            "financial_data": {
                "annual": {
                    "Revenues": {
                        "years": [2023, 2024],
                        "values": [100.0, 125.0],
                        "deltas": [None, 25.0],
                        "unit": "USD_millions",
                    }
                }
            },
        }

        text = flatten_ingestion_to_text(ingestion)

        self.assertIn("=== TEST (Test Corp) - Ingested Financial Data ===", text)
        self.assertIn("FY2023: $100M", text)
        self.assertIn("FY2024: $125M (+$25.0M YoY change)", text)

    def test_flatten_ingestion_handles_legacy_point_rows(self) -> None:
        ingestion = {
            "ticker": "TEST",
            "company_name": "Test Corp",
            "financial_data": {
                "annual": {
                    "Revenues": [
                        {"year": 2023, "value": 100.0, "delta": None, "unit": "USD_millions"},
                        {"year": 2024, "value": 110.0, "delta": 10.0, "unit": "USD_millions"},
                    ]
                }
            },
        }

        text = flatten_ingestion_to_text(ingestion)

        self.assertIn("FY2023: $100M", text)
        self.assertIn("FY2024: $110M (+$10.0M YoY change)", text)

    def test_build_ingestion_retrieval_text_includes_sections(self) -> None:
        ingestion = {
            "ticker": "TEST",
            "company_name": "Test Corp",
            "financial_data": {
                "annual": {
                    "Revenues": {
                        "years": [2024, 2025],
                        "values": [100.0, 120.0],
                        "deltas": [None, 20.0],
                        "unit": "USD_millions",
                    }
                }
            },
            "text_data": {
                "filings": [
                    {
                        "filingDate": "2025-10-31",
                        "sections": {
                            "item_1a_risk_factors": "Supplier concentration is a material risk.",
                            "item_7_mda": "Revenue grew due to services expansion.",
                        },
                    }
                ]
            },
        }

        text = build_ingestion_retrieval_text(ingestion)

        self.assertIn("Retrieval Profile: TEST (Test Corp)", text)
        self.assertIn("FY2025: $120M (+$20.0M YoY change)", text)
        self.assertIn("Item 1A Risk Factors:", text)
        self.assertIn("Supplier concentration is a material risk.", text)

    @patch("baseline_rag.matcher.ensure_index")
    @patch("baseline_rag.matcher.embed_texts")
    def test_find_matches_uses_ingestion_only_index(self, matcher_embed, ensure_index) -> None:
        repo_root = self._make_temp_dir("baseline_matcher")
        aapl_path = self._write_ingestion_artifact(repo_root, "AAPL", revenue=120.0)
        msft_path = self._write_ingestion_artifact(repo_root, "MSFT", revenue=119.0)
        tsla_path = self._write_ingestion_artifact(repo_root, "TSLA", revenue=10.0)

        import numpy as np

        class FakeIndex:
            d = 2

            def search(self, query_vector, top_k):
                return (
                    np.asarray([[1.0, 0.95, 0.1]], dtype="float32"),
                    np.asarray([[0, 1, 2]], dtype="int64"),
                )

        ensure_index.return_value = (
            {
                "index": FakeIndex(),
                "entries": [
                    {"ticker": "AAPL", "company": "AAPL Corp", "latest_filing_year": 2025, "source_path": str(aapl_path.resolve()), "modified_time": aapl_path.stat().st_mtime},
                    {"ticker": "MSFT", "company": "MSFT Corp", "latest_filing_year": 2025, "source_path": str(msft_path.resolve()), "modified_time": msft_path.stat().st_mtime},
                    {"ticker": "TSLA", "company": "TSLA Corp", "latest_filing_year": 2025, "source_path": str(tsla_path.resolve()), "modified_time": tsla_path.stat().st_mtime},
                ],
            },
            "reused",
        )
        matcher_embed.return_value = np.asarray([[1.0, 0.0]], dtype="float32")

        matches, index_status = find_matches_for_ticker("AAPL", top_k=2, repo_root=repo_root)

        self.assertIn(index_status, {"rebuilt", "reused"})
        self.assertEqual(matches[0]["ticker"], "MSFT")
        self.assertNotEqual(matches[0]["ticker"], "AAPL")

    def test_run_baseline_rag_saves_schema_compatible_artifact(self) -> None:
        repo_root = self._make_temp_dir("baseline_pipeline")
        self._write_ingestion_artifact(repo_root, "AAPL", revenue=120.0)
        self._write_ingestion_artifact(repo_root, "MSFT", revenue=118.0)

        class FakeClient:
            def complete_json(self, *, system_prompt: str, user_prompt: str):
                self.system_prompt = system_prompt
                self.user_prompt = user_prompt
                return (
                    {
                        "status": "completed",
                        "summary": "AAPL shows steady growth with manageable risks.",
                        "posture": {"label": "Mixed", "rationale_bullets": ["Revenue improved."]},
                        "target_profile": {
                            "ticker": "AAPL",
                            "company": "AAPL Corp",
                            "filing_year": 2025,
                            "positive_deltas": [{"metric": "Revenues", "label": "growth", "value": 10.0}],
                            "negative_deltas": [],
                            "top_risks": [],
                        },
                        "peer_snapshot": {
                            "peer_group": "Large-cap tech peers",
                            "common_strengths": ["Scale"],
                            "common_pressures": [],
                            "shared_risk_types": [],
                            "target_differences": ["Higher services mix"],
                        },
                        "risk_overlap_rows": [{"group": "shared_now", "risk_types": ["supply_chain"]}],
                        "forward_watchlist": [],
                        "narrative_sections": [
                            {"title": "Financial Performance", "content": "Strong revenue trend.", "citations": []},
                            {"title": "Risk Assessment", "content": "Risks remain manageable.", "citations": []},
                        ],
                        "error": None,
                    },
                    "test-model",
                )

        with patch("baseline_rag.indexer.load_vector_matrix") as load_vector_matrix, patch(
            "baseline_rag.matcher.embed_texts"
        ) as matcher_embed:
            import numpy as np

            aapl_path = repo_root / "data-ingestion" / "outputs" / "AAPL" / "complete_ingestion.json"
            msft_path = repo_root / "data-ingestion" / "outputs" / "MSFT" / "complete_ingestion.json"
            load_vector_matrix.return_value = (
                np.asarray([[1.0, 0.0], [0.95, 0.05]], dtype="float32"),
                [
                    {"ticker": "AAPL", "company": "AAPL Corp", "latest_filing_year": 2025, "source_path": str(aapl_path.resolve()), "modified_time": aapl_path.stat().st_mtime},
                    {"ticker": "MSFT", "company": "MSFT Corp", "latest_filing_year": 2025, "source_path": str(msft_path.resolve()), "modified_time": msft_path.stat().st_mtime},
                ],
            )
            matcher_embed.return_value = np.asarray([[1.0, 0.0]], dtype="float32")

            artifact = run_baseline_rag("AAPL", repo_root=repo_root, client=FakeClient())

        validated = OrchestrationArtifact.model_validate(artifact.model_dump(mode="json"))
        saved_path = repo_root / "baseline_rag" / "outputs" / "AAPL" / "aapl_comparison_bundle.json"

        self.assertEqual(validated.schema_version, "baseline_rag_v1")
        self.assertEqual(validated.bundle.target.ticker, "AAPL")
        self.assertTrue(saved_path.exists())
        self.assertGreaterEqual(len(validated.bundle.matches), 1)

    @patch("baseline_rag.indexer.load_vector_matrix")
    @patch("baseline_rag.indexer._import_faiss")
    def test_build_index_fails_before_vectorizing_when_faiss_missing(self, import_faiss, load_vector_matrix) -> None:
        import_faiss.side_effect = RuntimeError("FAISS missing")

        with self.assertRaisesRegex(RuntimeError, "FAISS missing"):
            build_index()

        load_vector_matrix.assert_not_called()


if __name__ == "__main__":
    unittest.main()
