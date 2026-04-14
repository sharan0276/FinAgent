from __future__ import annotations

import unittest

from baseline_rag import flatten_ingestion_to_text


class BaselineRAGCompatibilityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
