import importlib.util
import json
import re
from pathlib import Path


_UTILS_SPEC = importlib.util.spec_from_file_location(
    "sec_parser_utils", Path(__file__).with_name("sec_parser_utils.py")
)
_UTILS = importlib.util.module_from_spec(_UTILS_SPEC)
assert _UTILS_SPEC.loader is not None
_UTILS_SPEC.loader.exec_module(_UTILS)


TICKER = "META"
FORM = "10-K"
OUTPUT_DIR = _UTILS.OUTPUT_DIR
VALIDATION_JSON = OUTPUT_DIR / "meta_sec_parser_validation.json"
HEADING_INDEX = OUTPUT_DIR / "meta_sec_parser_heading_index.txt"
SECTION_DIR = OUTPUT_DIR / "meta_sections"

CORE_SECTIONS = [
    ("part_i", r"\bpart i\b"),
    ("item_1_business", r"\bitem 1\b.*\bbusiness\b"),
    ("item_1a_risk_factors", r"\bitem 1a\b.*\brisk factors\b"),
    (
        "item_7_mda",
        r"\bitem 7\b.*management'?s discussion and analysis of financial condition and results of operations",
    ),
    (
        "item_7a_market_risk",
        r"\bitem 7a\b.*quantitative and qualitative disclosures about market risk",
    ),
    (
        "item_8_financial_statements",
        r"\bitem 8\b.*financial statements and supplementary data",
    ),
    ("part_iii", r"\bpart iii\b"),
    ("item_10", r"\bitem 10\b"),
]

SECTION_EXPORTS = [
    "item_1_business",
    "item_1a_risk_factors",
    "item_7_mda",
    "item_7a_market_risk",
    "item_8_financial_statements",
]

EXPECTED_TERMS = {
    "item_1_business": ["business", "products", "competition", "customers"],
    "item_1a_risk_factors": ["risk", "adverse", "could", "material"],
    "item_7_mda": ["results of operations", "liquidity", "capital resources", "cash flows"],
    "item_7a_market_risk": ["market risk", "interest rate", "foreign exchange", "equity"],
    "item_8_financial_statements": ["balance sheets", "cash flows", "statements", "notes"],
}

MIN_WORD_COUNTS = {
    "item_1_business": 1500,
    "item_1a_risk_factors": 2000,
    "item_7_mda": 1500,
    "item_7a_market_risk": 200,
    "item_8_financial_statements": 2000,
}


def normalize(text: str) -> str:
    lowered = text.lower().replace("\xa0", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


def flatten_nodes(tree):
    rows = []
    for index, node in enumerate(tree.nodes):
        text = getattr(node, "text", "") or ""
        semantic_element = getattr(node, "semantic_element", None)
        element_type = type(semantic_element).__name__ if semantic_element else "Unknown"
        rows.append(
            {
                "index": index,
                "text": text,
                "normalized": normalize(text),
                "element_type": element_type,
                "is_heading": element_type in {"TitleElement", "TopSectionTitle"},
            }
        )
    return rows


def find_anchor(rows, pattern: str):
    compiled = re.compile(pattern)
    for row in rows:
        if row["is_heading"] and compiled.search(row["normalized"]):
            return row["index"]
    return None


def section_bounds(section_positions: dict[str, int | None], key: str):
    start = section_positions.get(key)
    if start is None:
        return None, None
    later = sorted(
        position
        for section_key, position in section_positions.items()
        if position is not None and position > start
    )
    end = later[0] if later else None
    return start, end


def slice_rows(rows, start: int | None, end: int | None):
    if start is None:
        return []
    return rows[start:end] if end is not None else rows[start:]


def section_text(rows_slice) -> str:
    return "\n".join(row["text"] for row in rows_slice if row["text"].strip())


def count_expected_terms(text: str, section_key: str) -> dict[str, bool]:
    normalized = normalize(text)
    return {term: (term in normalized) for term in EXPECTED_TERMS[section_key]}


def xbrl_token_ratio(text: str) -> float:
    total_tokens = max(len(text.split()), 1)
    xbrl_hits = len(re.findall(r"\b(?:us-gaap|xbrli|iso4217|dei|country|srt|meta):", text))
    return round(xbrl_hits / total_tokens, 4)


def replacement_char_count(text: str) -> int:
    return text.count("?") + text.count("\ufffd")


def metadata_blob_detected(rows) -> bool:
    for row in rows[:10]:
        if len(row["text"]) > 800 and ":" in row["text"] and sum(ch.isdigit() for ch in row["text"]) > 50:
            return True
    return False


def build_heading_index(rows) -> str:
    lines = ["Detected heading-like nodes:"]
    for row in rows:
        if row["element_type"] in {"TitleElement", "TopSectionTitle", "TableOfContentsElement"}:
            preview = row["text"].replace("\n", " ").strip()
            lines.append(f"[{row['index']:04d}] {row['element_type']}: {preview[:240]}")
    return "\n".join(lines) + "\n"


def validate() -> tuple[dict, dict[str, str], str]:
    html = _UTILS.fetch_filing_html(TICKER, FORM)
    elements, tree, rendered_tree = _UTILS.parse_filing_html(html)
    rows = flatten_nodes(tree)
    all_text = "\n".join(row["text"] for row in rows if row["text"].strip())

    section_positions = {key: find_anchor(rows, pattern) for key, pattern in CORE_SECTIONS}
    coverage = {key: (position is not None) for key, position in section_positions.items()}

    ordering_pairs = [
        ("part_i", "item_1_business"),
        ("item_1_business", "item_1a_risk_factors"),
        ("item_1a_risk_factors", "item_7_mda"),
        ("item_7_mda", "item_7a_market_risk"),
        ("item_7a_market_risk", "item_8_financial_statements"),
        ("item_8_financial_statements", "part_iii"),
        ("part_iii", "item_10"),
    ]
    ordering = {}
    for left, right in ordering_pairs:
        left_pos = section_positions.get(left)
        right_pos = section_positions.get(right)
        ordering[f"{left}_before_{right}"] = (
            left_pos is not None and right_pos is not None and left_pos < right_pos
        )

    extracted_sections = {}
    section_metrics = {}
    for key in SECTION_EXPORTS:
        start, end = section_bounds(section_positions, key)
        rows_slice = slice_rows(rows, start, end)
        text = section_text(rows_slice)
        extracted_sections[key] = text
        words = tokenize_words(text)
        term_hits = count_expected_terms(text, key) if text else {term: False for term in EXPECTED_TERMS[key]}
        section_metrics[key] = {
            "start_index": start,
            "end_index": end,
            "word_count": len(words),
            "char_count": len(text),
            "table_nodes": sum(1 for row in rows_slice if row["element_type"] == "TableElement"),
            "xbrl_token_ratio": xbrl_token_ratio(text),
            "replacement_char_count": replacement_char_count(text),
            "min_word_count_pass": len(words) >= MIN_WORD_COUNTS[key],
            "expected_terms_present": term_hits,
            "expected_term_hits": sum(term_hits.values()),
        }

    header_window = normalize(" ".join(row["text"] for row in rows[:40]))
    filing_identity = {
        "company_name_found_near_top": "meta platforms, inc" in header_window,
        "form_10k_found_near_top": "form 10-k" in header_window,
        "fiscal_year_2025_found_near_top": "2025" in header_window,
        "html_size_ok": len(html) > 500000,
    }

    noise = {
        "whole_document_xbrl_token_ratio": xbrl_token_ratio(all_text),
        "whole_document_replacement_char_count": replacement_char_count(all_text),
        "metadata_blob_detected_near_top": metadata_blob_detected(rows),
    }

    status = "good"
    if not all(coverage.values()) or not all(ordering.values()):
        status = "bad"
    elif noise["metadata_blob_detected_near_top"] or any(
        metrics["xbrl_token_ratio"] > 0.02 for metrics in section_metrics.values()
    ):
        status = "usable_with_cleanup"
    elif not all(
        metrics["min_word_count_pass"] and metrics["expected_term_hits"] >= 2
        for metrics in section_metrics.values()
    ):
        status = "usable_with_review"

    result = {
        "ticker": TICKER,
        "form": FORM,
        "parser_method": "Edgar10QParser Method 2 with 10-Q-specific top-section steps removed",
        "filing_identity": filing_identity,
        "core_section_coverage": coverage,
        "section_positions": section_positions,
        "section_ordering": ordering,
        "section_metrics": section_metrics,
        "noise_checks": noise,
        "document_metrics": {
            "html_length": len(html),
            "semantic_node_count": len(rows),
            "rendered_tree_line_count": len(rendered_tree.splitlines()),
        },
        "overall_status": status,
    }

    return result, extracted_sections, build_heading_index(rows)


def main() -> None:
    result, extracted_sections, heading_index = validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    VALIDATION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    HEADING_INDEX.write_text(heading_index, encoding="utf-8")

    for key, text in extracted_sections.items():
        output_path = SECTION_DIR / f"{key}.txt"
        output_path.write_text(text, encoding="utf-8")

    print(_UTILS.safe_console(json.dumps(result, indent=2)))
    print(f"\nSaved validation report to: {VALIDATION_JSON}")
    print(f"Saved heading index to: {HEADING_INDEX}")
    print(f"Saved extracted sections to: {SECTION_DIR}")


if __name__ == "__main__":
    main()
