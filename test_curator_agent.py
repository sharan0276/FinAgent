from __future__ import annotations

import sys
import unittest
from pathlib import Path

DATA_EXTRACTION_DIR = Path(__file__).resolve().parent / "data-extraction"
if str(DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_EXTRACTION_DIR))

from curator_agent import (
    compute_financial_deltas,
    get_top_candidates,
    format_candidates_for_prompt,
    build_embedding_text
)
from curator_models import DeltaLabel, FinancialDelta, RiskSignal, Severity, Topic, SignalType


class CuratorAgentDeterministicTests(unittest.TestCase):
    def test_compute_financial_deltas(self):
        data = {
            "numeric_deltas": {
                "Revenues": {"delta_percent": 25.0},      # > 20: strong_growth
                "NetIncome": {"delta_percent": 10.0},     # > 5: moderate_growth
                "Cash": {"delta_percent": -3.0},          # >= -5: stable
                "Assets": {"delta_percent": -15.0},       # >= -20: moderate_decline
                "LongTermDebt": {"delta_percent": -25.0}, # < -20: severe_decline
                "MissingPrior": {}                        # None: insufficient_data
            }
        }
        
        deltas = compute_financial_deltas(data)
        
        self.assertEqual(deltas["Revenues"].label, DeltaLabel.strong_growth)
        self.assertEqual(deltas["Revenues"].value, 0.25)
        
        self.assertEqual(deltas["NetIncome"].label, DeltaLabel.moderate_growth)
        self.assertEqual(deltas["NetIncome"].value, 0.10)
        
        self.assertEqual(deltas["Cash"].label, DeltaLabel.stable)
        self.assertEqual(deltas["Assets"].label, DeltaLabel.moderate_decline)
        self.assertEqual(deltas["LongTermDebt"].label, DeltaLabel.severe_decline)
        
        self.assertEqual(deltas["MissingPrior"].label, DeltaLabel.insufficient_data)
        self.assertIsNone(deltas["MissingPrior"].value)

    def test_get_top_candidates_allocation(self):
        # Create a mock set of candidates 
        # (15 for item_7, 5 for item_7a, 10 for item_1a, 2 for item_3)
        candidates = []
        
        for i in range(15):
            candidates.append({"section_id": "item_7_mda", "risk_score": 0.9 - (i*0.01)})
        for i in range(5):
            candidates.append({"section_id": "item_7a_market_risk", "risk_score": 0.8 - (i*0.01)})
        for i in range(10):
            candidates.append({"section_id": "item_1a_risk_factors", "risk_score": 0.95 - (i*0.01)})
        for i in range(2):
            candidates.append({"section_id": "item_3_legal", "risk_score": 0.7})
            
        data = {"text_candidates": candidates}
        
        # We expect exact fixed allocations: 12 from Item 7, 5 from Item 7A, 2 from Item 3, 
        # Wait, the allocation logic will fall back to Item 1A to fill missing slots to reach top_n
        top_candidates = get_top_candidates(data, top_n=30)
        
        # dist counts
        dist = {}
        for c in top_candidates:
            dist[c["section_id"]] = dist.get(c["section_id"], 0) + 1
            
        # Target sizes:
        # item_7 wants 12, has 15 -> takes 12
        # item_7a wants 10, has 5 -> takes 5
        # item_3 wants 8, has 2 -> takes 2
        # item_1a wants 10, has 10 -> takes 10
        # total allocated from fixed = 12 + 5 + 2 + 10 = 29
        # Fallback fills up to 30 using Item 1A, but Item 1A only has 10 and we took 10, so it limits at 29
        self.assertEqual(len(top_candidates), 29)
        self.assertEqual(dist["item_7_mda"], 12)
        self.assertEqual(dist["item_7a_market_risk"], 5)
        self.assertEqual(dist["item_3_legal"], 2)
        self.assertEqual(dist["item_1a_risk_factors"], 10)

    def test_format_candidates_for_prompt(self):
        c = [
            {
                "section_label": "Item 7",
                "sentence_text": "Sales dropped.",
                "previous_sentence": "We had issues.",
                "next_sentence": "This impacted margins significantly this year.",
                "risk_score": 0.854
            }
        ]
        
        formatted = format_candidates_for_prompt(c)
        self.assertIn("[1] Item 7 | 0.85", formatted)
        self.assertIn("Sales dropped.", formatted)
        self.assertNotIn("PREV: We had issues.", formatted) # < 20 chars, so ignored
        self.assertIn("NEXT: This impacted margins significantly this year.", formatted) # > 20 chars

    def test_build_embedding_text(self):
        deltas = {
            "Revenues": FinancialDelta(value=0.25, label=DeltaLabel.strong_growth),
            "Cash": FinancialDelta(value=-0.03, label=DeltaLabel.stable)
        }
        signals = [
            RiskSignal(
                signal_id=1,
                topic=Topic.market_risk,
                signal_type=SignalType.foreign_currency_exposure,
                section="Item 7",
                filing_year=2024,
                company="AAPL",
                summary="Currency fluctuations hurt margins.",
                severity=Severity.medium,
                citation="AAPL 10-K 2024"
            )
        ]
        
        text = build_embedding_text("AAPL", 2024, deltas, signals)
        self.assertIn("Revenues: strong_growth", text)
        self.assertIn("Cash: stable", text)
        self.assertIn("foreign_currency_exposure (medium): Currency fluctuations hurt margins.", text)
        self.assertTrue(text.startswith("Company: AAPL, Year: 2024. Financial:"))

if __name__ == "__main__":
    unittest.main()
