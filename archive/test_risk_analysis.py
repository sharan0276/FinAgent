import json
import os
import sys
import unittest
from unittest import mock
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RISK_ANALYSIS_ROOT = PROJECT_ROOT / "risk-analysis"

if str(RISK_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(RISK_ANALYSIS_ROOT))

from risk_analysis.numeric_analysis import build_numeric_trend_summaries
from risk_analysis.pipeline import build_yearly_artifacts, run_analysis
from risk_analysis.text_risk_extraction import (
    OpenRouterRiskLLMClient,
    _candidate_to_risk_types,
    _normalize_candidates_payload,
    _parse_json_content,
    _merge_small_chunks,
    _normalize_keep_source_ranks,
    chunk_section_text,
    segment_section,
)


class NumericTrendTests(unittest.TestCase):
    def test_embeds_original_series_records(self):
        financial_data = {
            "annual": {
                "Revenues": [
                    {"year": 2024, "end_date": "2024-12-31", "value": 100.0, "accession": "a1", "tag": "Revenues"},
                    {"year": 2025, "end_date": "2025-12-31", "value": 120.0, "accession": "a2", "tag": "Revenues"},
                ]
            },
            "quarterly": {
                "Revenues": [
                    {"year": 2025, "quarter": 1, "end_date": "2025-03-31", "value": 25.0, "accession": "q1", "tag": "Revenues"},
                    {"year": 2025, "quarter": 2, "end_date": "2025-06-30", "value": 28.0, "accession": "q2", "tag": "Revenues"},
                ]
            },
            "metric_errors": {},
        }
        summaries = build_numeric_trend_summaries(financial_data)
        self.assertEqual(len(summaries["Revenues"].annual_series), 2)
        self.assertEqual(summaries["Revenues"].annual_series[0]["accession"], "a1")
        self.assertEqual(summaries["Revenues"].quarterly_series[1]["quarter"], 2)

    def test_sparse_annual_data_returns_insufficient(self):
        financial_data = {
            "annual": {"Revenues": [{"year": 2024, "end_date": "2024-12-31", "value": 100.0}, {"year": 2025, "end_date": "2025-12-31", "value": 102.0}]},
            "quarterly": {"Revenues": []},
            "metric_errors": {},
        }
        summaries = build_numeric_trend_summaries(financial_data)
        self.assertEqual(summaries["Revenues"].annual_direction_5y, "insufficient_data")
        self.assertIsNone(summaries["Revenues"].annual_cagr_5y)

    def test_mixed_quarterly_data_flags_volatility(self):
        financial_data = {
            "annual": {"Cash": []},
            "quarterly": {"Cash": [
                {"year": 2024, "quarter": 1, "end_date": "2024-03-31", "value": 100.0},
                {"year": 2024, "quarter": 2, "end_date": "2024-06-30", "value": 200.0},
                {"year": 2024, "quarter": 3, "end_date": "2024-09-30", "value": 80.0},
                {"year": 2024, "quarter": 4, "end_date": "2024-12-31", "value": 220.0},
                {"year": 2025, "quarter": 1, "end_date": "2025-03-31", "value": 70.0},
            ]},
            "metric_errors": {},
        }
        summaries = build_numeric_trend_summaries(financial_data)
        self.assertIn(summaries["Cash"].quarterly_direction_recent, {"mixed", "increasing", "decreasing"})
        self.assertTrue(summaries["Cash"].quarterly_volatility_flag)


class TextSegmentationTests(unittest.TestCase):
    def test_segment_section_prefers_heading_blocks(self):
        text = "\n".join(["Overview", "This section describes risk exposure in detail.", "", "Supply Chain", "Supplier concentration could disrupt production.", "", "Regulation", "New regulation may increase compliance costs."])
        segments = segment_section(text)
        self.assertGreaterEqual(len(segments), 2)

    def test_chunk_fallback_groups_paragraphs(self):
        text = "\n\n".join([
            "Paragraph one with enough detail to remain in the same chunk.",
            "Paragraph two continues the discussion with more details about risk exposure.",
            "Paragraph three adds another point about competition and execution risks.",
            "Paragraph four discusses regulation and compliance changes.",
            "Paragraph five starts a second chunk.",
        ])
        chunks = chunk_section_text(text, max_paragraphs=4)
        self.assertEqual(len(chunks), 2)

    def test_merge_small_chunks_reduces_chunk_count(self):
        chunks = [(f"Heading {idx}", "A" * 300) for idx in range(20)]
        merged = _merge_small_chunks(chunks, section_id="item_7_mda")
        self.assertLessEqual(len(merged), 12)


class NormalizationTests(unittest.TestCase):
    def test_candidate_normalization_accepts_taxonomy_id_alias(self):
        item = {"taxonomy_id": "user_concentration"}
        self.assertEqual(_candidate_to_risk_types(item), ["customer_concentration"])

    def test_candidates_payload_accepts_items_wrapper(self):
        payload = {"candidates": {"items": [{"evidence": "A risk.", "taxonomy_id": "regulatory_legal"}]}}
        normalized = _normalize_candidates_payload(payload)
        self.assertEqual(len(normalized), 1)

    def test_keep_source_ranks_accepts_dict_shapes(self):
        raw = [{"source_rank": 3}, {"rank": "7"}, 9, "11"]
        self.assertEqual(_normalize_keep_source_ranks(raw), [3, 7, 9, 11])


class OpenRouterProviderTests(unittest.TestCase):
    def test_openrouter_client_requires_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OpenRouterRiskLLMClient()

    def test_parse_json_content_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            _parse_json_content("not-json")

    def test_parse_json_content_supports_content_blocks(self):
        parsed = _parse_json_content([{"type": "text", "text": '{"ok": true}'}])
        self.assertEqual(parsed, {"ok": True})


class EndToEndAnalysisTests(unittest.TestCase):
    def test_end_to_end_meta_artifact_with_stubbed_openrouter(self):
        source = PROJECT_ROOT / "data-ingestion" / "outputs" / "META" / "complete_ingestion.json"
        output_dir = PROJECT_ROOT / ".cache" / "risk-analysis-test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "company_analysis.json"

        def fake_post(url, headers=None, json=None, timeout=None):
            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self_inner):
                    messages = json.get("messages", [])
                    user_prompt = messages[-1]["content"] if messages else ""
                    if "Chunk text:" in user_prompt:
                        chunk_text = user_prompt.split("Chunk text:\n", 1)[1]
                        evidence = chunk_text.split(". ", 1)[0].strip()
                        if not evidence.endswith("."):
                            evidence += "."
                        body = {
                            "candidates": [
                                {
                                    "risk_types": ["regulatory_legal"],
                                    "severity": 4,
                                    "direction": "unclear",
                                    "confidence": 0.8,
                                    "evidence_text": evidence,
                                    "rationale": "Stub extraction",
                                }
                            ]
                        }
                    else:
                        body = {"keep_source_ranks": [{"source_rank": 0}], "rationales": {"0": "Stub verification"}}
                    return {"choices": [{"message": {"content": json_module.dumps(body)}}]}

            return FakeResponse()

        json_module = json
        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "MODEL_NAME": "test-model",
            },
            clear=False,
        ):
            with mock.patch("risk_analysis.text_risk_extraction.requests.post", side_effect=fake_post):
                written_paths = run_analysis(source_path=source, output_path=output, provider="openrouter", model_name=None, section_ids=["item_1a_risk_factors"])
        self.assertEqual(len(written_paths), 5)
        first_output = written_paths[0]
        self.assertTrue(first_output.exists())
        data = json.loads(first_output.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "company_analysis_v1")
        self.assertIn("numeric_trends", data)
        self.assertIn("filing_analysis", data)
        self.assertNotIn("section_rollups", data)
        self.assertEqual(len(data["filing_analysis"]), 1)
        revenues = data["numeric_trends"]["Revenues"]
        self.assertIn("annual_series", revenues)
        self.assertIn("quarterly_series", revenues)
        first_section = data["filing_analysis"][0]["section_results"]["item_1a_risk_factors"]
        self.assertEqual(first_section["llm_provider"], "openrouter")
        self.assertEqual(first_section["llm_model"], "test-model")
        self.assertIsNone(first_section["processing_error"])

    def test_openrouter_failure_records_processing_error(self):
        source = PROJECT_ROOT / "data-ingestion" / "outputs" / "META" / "complete_ingestion.json"
        output_dir = PROJECT_ROOT / ".cache" / "risk-analysis-test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "company_analysis_error.json"
        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "MODEL_NAME": "test-model",
            },
            clear=False,
        ):
            with mock.patch("risk_analysis.text_risk_extraction.requests.post", side_effect=RuntimeError("openrouter down")):
                written_paths = run_analysis(source_path=source, output_path=output, provider="openrouter", model_name=None, section_ids=["item_1a_risk_factors"])
        self.assertEqual(len(written_paths), 5)
        data = json.loads(written_paths[0].read_text(encoding="utf-8"))
        first_section = data["filing_analysis"][0]["section_results"]["item_1a_risk_factors"]
        self.assertEqual(first_section["llm_provider"], "openrouter")
        self.assertIn("openrouter down", first_section["processing_error"])
        self.assertEqual(first_section["signals"], [])
        self.assertTrue(data["filing_analysis"][0]["filing_errors"])

    def test_year_filter_limits_processed_filings(self):
        source = PROJECT_ROOT / "data-ingestion" / "outputs" / "META" / "complete_ingestion.json"
        output_dir = PROJECT_ROOT / ".cache" / "risk-analysis-test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "company_analysis_2025.json"

        def fake_post(url, headers=None, json=None, timeout=None):
            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self_inner):
                    messages = json.get("messages", [])
                    user_prompt = messages[-1]["content"] if messages else ""
                    if "Chunk text:" in user_prompt:
                        body = {"candidates": []}
                    else:
                        body = {"keep_source_ranks": [], "rationales": {}}
                    return {"choices": [{"message": {"content": json_module.dumps(body)}}]}

            return FakeResponse()

        json_module = json
        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "MODEL_NAME": "test-model",
            },
            clear=False,
        ):
            with mock.patch("risk_analysis.text_risk_extraction.requests.post", side_effect=fake_post):
                written_paths = run_analysis(
                    source_path=source,
                    output_path=output,
                    provider="openrouter",
                    model_name=None,
                    section_ids=["item_1a_risk_factors"],
                    filing_years=[2025],
                )
        self.assertEqual(len(written_paths), 1)
        self.assertTrue(written_paths[0].name.endswith("META_2025_company_analysis.json"))
        data = json.loads(written_paths[0].read_text(encoding="utf-8"))
        self.assertEqual(len(data["filing_analysis"]), 1)
        self.assertEqual(data["filing_analysis"][0]["filing_year"], 2025)

    def test_yearly_artifacts_drop_rollups_and_preserve_numeric_trends(self):
        artifact = {
            "schema_version": "company_analysis_v1",
            "taxonomy_version": "risk_taxonomy_v1",
            "prompt_version": "prompt_v1",
            "model_name": "test-model",
            "run_timestamp": "2026-01-01T00:00:00Z",
            "run_id": "run-1",
            "ticker": "TEST",
            "cik": "0000000000",
            "company_name": "Test Corp",
            "source_artifact_path": "source.json",
            "source_accessions": ["a1", "a2"],
            "numeric_trends": build_numeric_trend_summaries(
                {
                    "annual": {"Revenues": [{"year": 2024, "end_date": "2024-12-31", "value": 100.0, "accession": "a1", "tag": "Revenues"}]},
                    "quarterly": {"Revenues": [{"year": 2024, "quarter": 4, "end_date": "2024-12-31", "value": 30.0, "accession": "q1", "tag": "Revenues"}]},
                    "metric_errors": {},
                }
            ),
            "filing_analysis": [
                {"filing_year": 2024, "filing_date": "2024-12-31", "accession": "a1", "parser_mode": "sec_parser", "filing_errors": [], "section_results": {}},
                {"filing_year": 2025, "filing_date": "2025-12-31", "accession": "a2", "parser_mode": "sec_parser", "filing_errors": [], "section_results": {}},
            ],
            "section_rollups": {},
        }
        from risk_analysis.models import CompanyAnalysisArtifactV1

        base = CompanyAnalysisArtifactV1(**artifact)
        yearly = build_yearly_artifacts(base)
        self.assertEqual(len(yearly), 2)
        self.assertIsNone(yearly[0].section_rollups)
        self.assertEqual(len(yearly[0].filing_analysis), 1)
        self.assertEqual(yearly[0].numeric_trends["Revenues"].annual_series, yearly[1].numeric_trends["Revenues"].annual_series)


if __name__ == "__main__":
    unittest.main()
