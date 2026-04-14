from __future__ import annotations

import unittest

from orchestration.report_models import MatchContext
from ui.display import build_compact_match_rows, format_display_label


class UIAppHelpersTests(unittest.TestCase):
    def test_format_display_label_cleans_internal_names(self) -> None:
        self.assertEqual(format_display_label("shared_now"), "Shared Now")
        self.assertEqual(format_display_label("third_party_vendor_dependency"), "Third Party Vendor Dependency")
        self.assertEqual(format_display_label("high-confidence"), "High Confidence")
        self.assertEqual(format_display_label("AAPL"), "AAPL")

    def test_build_compact_match_rows_excludes_internal_fields(self) -> None:
        rows = build_compact_match_rows(
            [
                MatchContext(
                    ticker="GOOG",
                    company="Alphabet Inc.",
                    matched_filing_year=2025,
                    similarity=0.91,
                    context_curator_paths=["d:/fake/path.json"],
                )
            ]
        )

        self.assertEqual(
            rows,
            [
                {
                    "Ticker": "GOOG",
                    "Company": "Alphabet Inc.",
                    "Matched Filing Year": 2025,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
