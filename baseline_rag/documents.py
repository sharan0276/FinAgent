from __future__ import annotations

from typing import Any, Optional

from orchestration.artifact_resolver import latest_filing_year


RETRIEVAL_SECTION_IDS = (
    "item_1_business",
    "item_1a_risk_factors",
    "item_1c_cybersecurity",
    "item_3_legal",
    "item_7_mda",
    "item_7a_market_risk",
)

SECTION_LABELS = {
    "item_1_business": "Item 1 Business",
    "item_1a_risk_factors": "Item 1A Risk Factors",
    "item_1c_cybersecurity": "Item 1C Cybersecurity",
    "item_3_legal": "Item 3 Legal Proceedings",
    "item_7_mda": "Item 7 MD&A",
    "item_7a_market_risk": "Item 7A Market Risk",
}


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_row_year(row: dict[str, Any]) -> Optional[int]:
    for key in ("year", "fiscal_year", "fy"):
        year = _coerce_int(row.get(key))
        if year is not None:
            return year

    for key in ("end", "period_end", "date"):
        raw = str(row.get(key, ""))
        if len(raw) >= 4 and raw[:4].isdigit():
            return int(raw[:4])
    return None


def annual_metric_rows(metric_payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(metric_payload, dict) and "years" in metric_payload:
        years = metric_payload.get("years") or []
        values = metric_payload.get("values") or []
        deltas = metric_payload.get("deltas") or []
        unit = metric_payload.get("unit", "USD_millions")
        for index, year in enumerate(years):
            rows.append(
                {
                    "year": _coerce_int(year),
                    "value": _coerce_float(values[index]) if index < len(values) else None,
                    "delta": _coerce_float(deltas[index]) if index < len(deltas) else None,
                    "unit": unit,
                }
            )
        return [row for row in rows if row["year"] is not None]

    source_rows: list[dict[str, Any]] = []
    if isinstance(metric_payload, dict) and isinstance(metric_payload.get("annual"), list):
        source_rows = [row for row in metric_payload.get("annual", []) if isinstance(row, dict)]
    elif isinstance(metric_payload, list):
        source_rows = [row for row in metric_payload if isinstance(row, dict)]

    for row in source_rows:
        rows.append(
            {
                "year": _infer_row_year(row),
                "value": _coerce_float(row.get("value", row.get("val"))),
                "delta": _coerce_float(row.get("delta", row.get("delta_percent"))),
                "unit": row.get("unit", "USD_millions"),
            }
        )

    rows = [row for row in rows if row["year"] is not None]
    rows.sort(key=lambda row: row["year"] or 0)
    return rows


def format_value(value: Optional[float], unit: str) -> str:
    if value is None:
        return "N/A"
    if "usd" in unit.lower():
        return f"${value:,.0f}M"
    return f"{value:,.2f}"


def format_delta(delta: Optional[float], unit: str) -> str:
    if delta is None:
        return ""
    sign = "+" if delta >= 0 else ""
    if "usd" in unit.lower():
        return f" ({sign}${delta:,.1f}M YoY change)"
    return f" ({sign}{delta:,.2f} YoY change)"


def flatten_ingestion_to_text(ingestion: dict[str, Any]) -> str:
    """Convert ingestion JSON to a compact, LLM-readable financial summary."""
    ticker = ingestion.get("ticker", "UNKNOWN")
    company = ingestion.get("company_name", ticker)
    annual = ingestion.get("financial_data", {}).get("annual", {})

    lines: list[str] = [f"=== {ticker} ({company}) - Ingested Financial Data ==="]

    for metric, metric_payload in annual.items():
        rows = annual_metric_rows(metric_payload)
        if not rows:
            continue

        unit = str(rows[0].get("unit") or "USD_millions")
        lines.append(f"\n{metric}:")
        for row in rows:
            year = row["year"]
            val_str = format_value(row["value"], unit)
            delta_str = format_delta(row["delta"], unit)
            lines.append(f"  FY{year}: {val_str}{delta_str}")

    return "\n".join(lines)


def latest_year_from_ingestion(ingestion: dict[str, Any]) -> Optional[int]:
    try:
        return latest_filing_year(ingestion)
    except Exception:
        annual = ingestion.get("financial_data", {}).get("annual", {})
        all_years: list[int] = []
        for metric_payload in annual.values():
            all_years.extend(
                row["year"]
                for row in annual_metric_rows(metric_payload)
                if row.get("year") is not None
            )
        return max(all_years) if all_years else None


def _latest_filing(ingestion: dict[str, Any]) -> dict[str, Any] | None:
    filings = ingestion.get("text_data", {}).get("filings", [])
    candidates = [filing for filing in filings if isinstance(filing, dict)]
    if not candidates:
        return None
    return max(candidates, key=lambda filing: str(filing.get("filingDate", "")))


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_ingestion_retrieval_text(
    ingestion: dict[str, Any],
    *,
    max_chars_per_section: int = 4000,
) -> str:
    """Create a deterministic embedding document from ingestion-only content."""
    ticker = ingestion.get("ticker", "UNKNOWN")
    company = ingestion.get("company_name", ticker)
    latest_year = latest_year_from_ingestion(ingestion)
    filing = _latest_filing(ingestion)

    lines: list[str] = [
        f"=== Retrieval Profile: {ticker} ({company}) ===",
        f"Latest Filing Year: {latest_year if latest_year is not None else 'Unknown'}",
        "",
        flatten_ingestion_to_text(ingestion),
    ]

    if filing:
        filing_date = filing.get("filingDate")
        if filing_date:
            lines.append(f"\nLatest Filing Date: {filing_date}")
        sections = filing.get("sections", {})
        for section_id in RETRIEVAL_SECTION_IDS:
            raw_text = sections.get(section_id) if isinstance(sections, dict) else None
            if not raw_text:
                continue
            normalized = _normalize_text(str(raw_text))
            if not normalized:
                continue
            snippet = normalized[:max_chars_per_section]
            lines.append(f"\n{SECTION_LABELS.get(section_id, section_id)}:")
            lines.append(snippet)

    return "\n".join(lines)
