from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Correctly add the data-extraction folder to the path, handling the hyphenated folder name
DATA_EXTRACTION_DIR = Path(__file__).resolve().parent / "data-extraction"
if str(DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_EXTRACTION_DIR))
    
from numeric_delta import build_numeric_deltas
from pipeline import run_extraction
from text_candidates import is_candidate_sentence, split_into_sentences


class FakeScorer:
    model_name = "fake-finbert"

    def score_sentences(self, sentences):
        outputs = []
        for sentence in sentences:
            lowered = sentence.lower()
            negative = 0.15
            for term in ("risk", "adverse", "litigation", "cyber", "volatile", "debt", "competition"):
                if term in lowered:
                    negative += 0.1
            negative = min(0.95, negative)
            neutral = max(0.04, 1.0 - negative - 0.01)
            positive = max(0.01, 1.0 - negative - neutral)
            outputs.append(
                {
                    "positive": round(positive, 6),
                    "negative": round(negative, 6),
                    "neutral": round(neutral, 6),
                }
            )
        return outputs


class DataExtractionTests(unittest.TestCase):
    def test_split_into_sentences_breaks_on_punctuation(self):
        sentences = split_into_sentences("Risk is rising. Competition is intense! Can margins hold?")
        self.assertEqual(sentences, ["Risk is rising.", "Competition is intense!", "Can margins hold?"])

    def test_is_candidate_sentence_filters_noise(self):
        self.assertFalse(is_candidate_sentence("2025 2024 2023 33% 19%"))
        self.assertFalse(is_candidate_sentence("Risk Factors"))
        self.assertTrue(
            is_candidate_sentence("Competitive pressure could materially adversely affect gross margins over time.")
        )

    def test_numeric_delta_bucketing_and_missing_prior(self):
        # Using the standard Dictionary format
        financial_data = {
            "annual": {
                "Revenues": {
                    "tag": "Revenues",
                    "years": [2023, 2024],
                    "values": [100.0, 130.0],
                },
                "NetIncome": {
                    "tag": "NetIncome",
                    "years": [2024],
                    "values": [50.0],
                },
                "Cash": {},
                "Assets": {},
                "LongTermDebt": {},
                "OperatingCashFlow": {},
                "ResearchAndDevelopment": {},
                "GrossProfit": {},
            }
        }
        # Standardized logic finds max year (2024) automatically
        deltas = build_numeric_deltas(financial_data, 2024)
        
        # Scaling by 1,000,000 is now enforced
        self.assertEqual(deltas["Revenues"].current_value, 130_000_000.0)
        self.assertEqual(deltas["Revenues"].delta_percent, 30.0)
        self.assertEqual(deltas["Revenues"].label, "strong_growth")
        self.assertIsNone(deltas["NetIncome"].delta_percent)
        self.assertEqual(deltas["NetIncome"].reason, "missing_previous_year")

    def test_run_extraction_writes_expected_artifact(self):
        # NOTE: This test depends on a dictionary-style ingestion file.
        # Ensure data-ingestion/outputs/AAPL/complete_ingestion.json is re-ingested.
        source_path = Path("data-ingestion/outputs/AAPL/complete_ingestion.json")
        temp_output_dir = Path("data-extraction") / "test-output-unittest"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        written = run_extraction(
            source_path=source_path,
            output_dir=temp_output_dir,
            scorer=FakeScorer(),
            filing_years=[2025],
        )
        self.assertEqual(len(written), 1)
        output_path = written[0]
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["filing_year"], 2025)
        self.assertEqual(payload["schema_version"], "data_extraction_v1")
        self.assertEqual(payload["model_name"], "fake-finbert")
        self.assertEqual(
            set(payload["numeric_deltas"].keys()),
            {
                "Revenues",
                "NetIncome",
                "Cash",
                "Assets",
                "LongTermDebt",
                "OperatingCashFlow",
                "ResearchAndDevelopment",
                "GrossProfit",
            },
        )
        self.assertEqual(
            sorted(payload["processed_sections"] + payload["skipped_sections"]),
            sorted(
                [
                    "item_1a_risk_factors",
                    "item_1c_cybersecurity",
                    "item_3_legal",
                    "item_7_mda",
                    "item_7a_market_risk",
                ]
            ),
        )
        self.assertEqual(len(payload["text_candidates"]), 100)
        self.assertTrue(
            all(candidate["section_id"] in payload["processed_sections"] for candidate in payload["text_candidates"])
        )


if __name__ == "__main__":
    unittest.main()
