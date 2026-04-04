from __future__ import annotations

from typing import Dict, List, Tuple


SCHEMA_VERSION = "data_extraction_v1"
DEFAULT_TOP_K = 100
PER_SECTION_TARGET = 20
MIN_SENTENCE_LENGTH = 20
DEFAULT_BATCH_SIZE = 16

TARGET_SECTIONS: List[Tuple[str, str]] = [
    ("item_1a_risk_factors", "Item 1A"),
    ("item_1c_cybersecurity", "Item 1C"),
    ("item_3_legal_proceedings", "Item 3"),
    ("item_7_mda", "Item 7"),
    ("item_7a_market_risk", "Item 7A"),
]

TARGET_SECTION_IDS = {section_id for section_id, _ in TARGET_SECTIONS}
SECTION_LABELS: Dict[str, str] = dict(TARGET_SECTIONS)

NUMERIC_METRICS: List[str] = [
    "Revenues",
    "NetIncome",
    "Cash",
    "Assets",
    "LongTermDebt",
    "OperatingCashFlow",
    "ResearchAndDevelopment",
    "GrossProfit",
]

DELTA_BUCKETS = [
    (20.0, "strong_growth"),
    (5.0, "moderate_growth"),
    (-5.0, "stable"),
    (-20.0, "moderate_decline"),
]
