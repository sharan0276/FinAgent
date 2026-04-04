from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from company_facts_cleaner import DataCleaner
from document_fetcher import DocumentFetcher
from sec_client import SECClient

OUTPUT_ROOT = Path("data-ingestion") / "outputs"

CORE_SECTION_PATTERNS: dict[str, str] = {
    "item_1_business": r"\bitem\s+1\.?\s+business\b",
    "item_1a_risk_factors": r"\bitem\s+1a\.?\s+risk factors\b",
    "item_1c_cybersecurity": r"\bitem\s+1c\.?\s+cybersecurity\b", 
    "item_3_legal": r"\bitem\s+3\.?\s+legal proceedings\b", 
    "item_7_mda": r"\bitem\s+7\.?\s+management'?s discussion and analysis.*\b",
    "item_7a_market_risk": r"\bitem\s+7a\.?\s+.*market risk\b"
}

DEFAULT_METRICS = [
    "Revenues",
    "NetIncome",
    "Cash",
    "Assets",
    "LongTermDebt",
    "OperatingCashFlow",
    "ResearchAndDevelopment",
    "GrossProfit",
]


def _normalize(text: str) -> str:
    text = text.lower().replace("\xa0", " ")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def _flatten_tree_rows(tree: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(tree.nodes):
        text = getattr(node, "text", "") or ""
        semantic_element = getattr(node, "semantic_element", None)
        element_type = type(semantic_element).__name__ if semantic_element else "Unknown"
        rows.append(
            {
                "index": index,
                "text": text,
                "normalized": _normalize(text),
                "element_type": element_type,
                "is_heading": element_type in {"TitleElement", "TopSectionTitle"},
            }
        )
    return rows


def _find_toc_anchor(rows: list[dict[str, Any]]) -> int | None:
    """Find the index of the Table of Contents landmark."""
    patterns = [
        r"table of contents",
        r"\bcontents\b",
        r"\bindex\b",
    ]
    for pattern in patterns:
        compiled = re.compile(pattern)
        for row in rows:
            # Table of Contents is often a heading, but can be plain text
            if compiled.search(row["normalized"]):
                # Ensure it's not a tiny match or part of a sentence
                if len(row["normalized"]) < 50:
                    return row["index"]
    return None


def _find_heading_anchor(
    rows: list[dict[str, Any]], 
    pattern: str, 
    start_from: int = 0,
    must_be_heading: bool = True
) -> int | None:
    compiled = re.compile(pattern)
    for i in range(start_from, len(rows)):
        row = rows[i]
        if must_be_heading:
            if row["is_heading"] and compiled.search(row["normalized"]):
                return row["index"]
        else:
            if compiled.search(row["normalized"]):
                return row["index"]
    return None


def _extract_core_sections_from_tree(tree: Any) -> dict[str, str]:
    rows = _flatten_tree_rows(tree)
    
    # 1. Find landmarks to avoid Cover/TOC
    toc_index = _find_toc_anchor(rows) or 0
    part_i_index = _find_heading_anchor(
        rows, r"\bpart i\b", start_from=toc_index, must_be_heading=True
    ) or toc_index

    # 2. Find Item anchors starting AFTER Part I
    positions = {
        section: _find_heading_anchor(rows, pattern, start_from=part_i_index)
        for section, pattern in CORE_SECTION_PATTERNS.items()
    }

    extracted: dict[str, str] = {}
    sorted_anchors = sorted([p for p in positions.values() if p is not None])

    for section, start in positions.items():
        if start is None:
            extracted[section] = ""
            continue

        # Find the end by looking for ANY new "Item" heading
        end = None
        for i in range(start + 1, len(rows)):
            row = rows[i]
            if row["is_heading"]:
                # Matches patterns like "item 2", "item 1b", "item 4"
                if re.match(r"^item\s+\d+[a-z]?\.?\b", row["normalized"]):
                    end = i
                    break
                    
        # Fallback to the next tracked anchor if no generic heading is found
        if end is None:
            later_positions = [p for p in sorted_anchors if p > start]
            end = later_positions[0] if later_positions else None
            
        section_rows = rows[start:end] if end is not None else rows[start:]
        
        extracted[section] = "\n".join(
            row["text"] for row in section_rows if row["text"].strip()
        )

    return extracted


def _extract_core_sections_from_plain_text(html_text: str) -> dict[str, str]:
    import html
    # Simple tag stripping
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = html.unescape(text)
    normalized = _normalize(text)
    
    # Locate landmarks in plain text
    toc_match = re.search(r"table of contents|index|contents", normalized)
    toc_pos = toc_match.start() if toc_match else 0
    
    # TOC contains "Part I" at the top, and then the actual body starts with "Part I" again.
    # By picking the second occurrence, we jump entirely past the TOC list.
    part_i_matches = list(re.finditer(r"\bpart i\b", normalized[toc_pos:]))
    if len(part_i_matches) >= 2:
        start_pos = toc_pos + part_i_matches[1].start()
    elif part_i_matches:
        start_pos = toc_pos + part_i_matches[0].start()
    else:
        start_pos = toc_pos

    extracted: dict[str, str] = {}
    
    # Collect all matches and their positions
    found_sections = []
    for section, pattern in CORE_SECTION_PATTERNS.items():
        # Look for matches only after Part I body
        match = re.search(pattern, normalized[start_pos:])
        if match:
            absolute_start = start_pos + match.start()
            found_sections.append((section, absolute_start))

    # Sort by position to find ends
    found_sections.sort(key=lambda x: x[1])
    
    for i, (section, start) in enumerate(found_sections):
        end = found_sections[i+1][1] if i+1 < len(found_sections) else len(normalized)
        
        # Limit length to 100k chars for safety in fallback mode
        end = min(end, start + 100000)
        
        # To prevent swallowing untracked sections (like Item 1C swallowing Item 2),
        # we look for the next generic "Item N" heading. We use standard SEC titles 
        # in the regex to avoid false positives (e.g., "refer to item 2 below").
        generic_item_pat = r"\bitem\s+\d+[a-z]?\.?\s+(?:business|risk|unresolved|cybersecurity|properties|legal|mine|market|selected|reserved|management|quantitative|financial|changes|controls|other|disclosure|directors|executive|security|certain|principal|exhibits|form)\b"
        
        # Search starting 50 chars ahead so we don't accidentally match our own heading
        search_start = start + 50
        next_item_match = re.search(generic_item_pat, normalized[search_start:end])
        if next_item_match:
            end = search_start + next_item_match.start()
            
        content = normalized[start:end].strip()
        
        # Clean up: if the captured text starts with the heading, keep it
        extracted[section] = content

    # Fill in missing
    for section in CORE_SECTION_PATTERNS:
        if section not in extracted:
            extracted[section] = ""

    return extracted


def run_financial_ingestion(
    client: SECClient,
    cik: str,
    n_years: int,
    metrics: list[str],
) -> dict[str, Any]:
    cleaner = DataCleaner()
    facts = client.get_company_facts(cik)

    annual: dict[str, list[dict[str, Any]]] = {}
    quarterly: dict[str, list[dict[str, Any]]] = {}
    metric_errors: dict[str, str] = {}

    for metric in metrics:
        try:
            result = cleaner.get_all_values(facts, metric, n_years=n_years)
            annual[metric] = result["annual"]
            quarterly[metric] = result["quarterly"]
        except Exception as exc:
            metric_errors[metric] = str(exc)
            annual[metric] = []
            quarterly[metric] = []

    return {
        "annual": annual,
        "quarterly": quarterly,
        "metric_errors": metric_errors,
    }


def run_text_ingestion(client: SECClient, cik: str, n_years: int) -> dict[str, Any]:
    submissions = client.get_submissions(cik)
    fetcher = DocumentFetcher(client)
    filings = fetcher.get_latest_10k(submissions, cik, n_years=n_years) or []

    parse_filing_html = None
    parser_error = None
    try:
        from sec_parser_utils import parse_filing_html as _parse_filing_html

        parse_filing_html = _parse_filing_html
        #parse_filing_html = None  # TEMP FOR TESTING: Force plain text fallback
    except Exception as exc:
        parser_error = str(exc)

    parsed_filings: list[dict[str, Any]] = []
    for filing in filings:
        if parse_filing_html is not None:
            print(f"[{filing['form']} {filing['filingDate']}] Extracting sections using sec_parser...")
            elements, tree, rendered_tree = parse_filing_html(filing["html"])
            sections = _extract_core_sections_from_tree(tree)
            semantic_element_count = len(elements)
            tree_line_count = len(rendered_tree.splitlines())
            parser_mode = "sec_parser"
        else:
            print(f"[{filing['form']} {filing['filingDate']}] Using plain_text_fallback...")
            sections = _extract_core_sections_from_plain_text(filing["html"])
            semantic_element_count = None
            tree_line_count = None
            parser_mode = "plain_text_fallback"

        parsed_filings.append(
            {
                "form": filing["form"],
                "filingDate": filing["filingDate"],
                "accession": filing["accession"],
                "url": filing["url"],
                "html_length": len(filing["html"]),
                "semantic_element_count": semantic_element_count,
                "tree_line_count": tree_line_count,
                "parser_mode": parser_mode,
                "sections": sections,
            }
        )

    return {
        "filings": parsed_filings,
        "parser_error": parser_error,
    }


def run_ingestion(
    ticker: str,
    n_years: int = 5,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    metrics = metrics or DEFAULT_METRICS
    client = SECClient()
    cik = client.get_cik_from_ticker(ticker)
    submissions = client.get_submissions(cik)
    company_name = submissions.get("name") or submissions.get("entityName")

    with ThreadPoolExecutor(max_workers=2) as pool:
        financial_future = pool.submit(
            run_financial_ingestion, client, cik, n_years, metrics
        )
        text_future = pool.submit(run_text_ingestion, client, cik, n_years)

        financial_data = financial_future.result()
        text_data = text_future.result()

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "company_name": company_name,
        "n_years": n_years,
        "metrics": metrics,
        "financial_data": financial_data,
        "text_data": text_data,
    }


def save_ingestion_output(result: dict[str, Any]) -> Path:
    ticker = result["ticker"].upper()
    out_dir = OUTPUT_ROOT / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "complete_ingestion.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete SEC ingestion: financial metrics + parsed 10-K text."
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--years", type=int, default=5, help="Number of years of 10-K data")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Override metric aliases used for Company Facts extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ingestion(args.ticker, n_years=args.years, metrics=args.metrics)
    output_path = save_ingestion_output(result)

    print(
        json.dumps(
            {
                "ticker": result["ticker"],
                "cik": result["cik"],
                "company_name": result["company_name"],
                "filings_processed": len(result["text_data"]["filings"]),
                "metrics_processed": len(result["metrics"]),
                "parser_mode": (
                    result["text_data"]["filings"][0]["parser_mode"]
                    if result["text_data"]["filings"]
                    else None
                ),
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
