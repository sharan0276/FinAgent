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
    "part_i": r"\bpart i\b",
    "item_1_business": r"\bitem 1\b.*\bbusiness\b",
    "item_1a_risk_factors": r"\bitem 1a\b.*\brisk factors\b",
    "item_7_mda": r"\bitem 7\b.*management'?s discussion and analysis",
    "item_7a_market_risk": r"\bitem 7a\b.*market risk",
    "item_8_financial_statements": r"\bitem 8\b.*financial statements",
    # "part_iii": r"\bpart iii\b",
    # "item_10": r"\bitem 10\b",
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


def _find_heading_anchor(rows: list[dict[str, Any]], pattern: str) -> int | None:
    compiled = re.compile(pattern)
    for row in rows:
        if row["is_heading"] and compiled.search(row["normalized"]):
            return row["index"]
    return None


def _extract_core_sections_from_tree(tree: Any) -> dict[str, str]:
    rows = _flatten_tree_rows(tree)
    positions = {
        section: _find_heading_anchor(rows, pattern)
        for section, pattern in CORE_SECTION_PATTERNS.items()
    }

    extracted: dict[str, str] = {}
    for section, start in positions.items():
        if start is None:
            extracted[section] = ""
            continue

        later_positions = sorted(
            p for p in positions.values() if p is not None and p > start
        )
        end = later_positions[0] if later_positions else None
        section_rows = rows[start:end] if end is not None else rows[start:]
        extracted[section] = "\n".join(
            row["text"] for row in section_rows if row["text"].strip()
        )

    return extracted


def _extract_core_sections_from_plain_text(html: str) -> dict[str, str]:
    text = re.sub(r"<[^>]+>", " ", html)
    normalized = _normalize(text)
    extracted: dict[str, str] = {}

    for section, pattern in CORE_SECTION_PATTERNS.items():
        match = re.search(pattern, normalized)
        if not match:
            extracted[section] = ""
            continue

        start = max(0, match.start() - 250)
        end = min(len(normalized), match.start() + 3000)
        extracted[section] = normalized[start:end]

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
    except Exception as exc:
        parser_error = str(exc)

    parsed_filings: list[dict[str, Any]] = []
    for filing in filings:
        if parse_filing_html is not None:
            elements, tree, rendered_tree = parse_filing_html(filing["html"])
            sections = _extract_core_sections_from_tree(tree)
            semantic_element_count = len(elements)
            tree_line_count = len(rendered_tree.splitlines())
            parser_mode = "sec_parser"
        else:
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
