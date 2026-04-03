"""
Cleans raw XBRL financial data from SEC Company Facts API.
Handles common data issues:
  1. Tag routing: companies switch XBRL tags over time
  2. Sliced pie trap: filter out segment breakdowns, keep consolidated totals
  3. Period filter: separate annual and quarterly values
"""

import re
from datetime import datetime


class DataCleaner:
    """
    Cleans raw XBRL data from SEC Company Facts API into validated annual and
    quarterly metrics.

    Single public orchestrator: get_all_values().
    """

    _TAG_FALLBACK_CHAINS: dict[str, list[str]] = {
        "Revenues": [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ],
        "NetIncome": [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
        "Cash": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsAndShortTermInvestments",
            "Cash",
        ],
        "Assets": ["Assets"],
        "LongTermDebt": [
            "LongTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermNotesPayable",
            "ConvertibleNotesPayable",
        ],
        "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
        "ResearchAndDevelopment": [
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ],
        "GrossProfit": ["GrossProfit"],
    }

    def _resolve_tag(
        self,
        facts: dict,
        metric_alias: str,
        taxonomy: str = "us-gaap",
    ) -> tuple[str, list[dict]]:
        chain = self._TAG_FALLBACK_CHAINS.get(metric_alias, [metric_alias])
        taxonomy_data = facts.get("facts", {}).get(taxonomy, {})

        for tag in chain:
            if tag in taxonomy_data:
                units_dict = taxonomy_data[tag].get("units", {})
                for _, entries in units_dict.items():
                    return tag, entries

        raise KeyError(
            f"No XBRL tag found for metric '{metric_alias}' using chain: {chain}"
        )

    def _filter_consolidated(self, entries: list[dict]) -> list[dict]:
        return [
            e for e in entries if "segment" not in e or e.get("segment") is None
        ]

    def _filter_annual(self, entries: list[dict]) -> list[dict]:
        annual = []
        for e in entries:
            form = e.get("form", "")
            frame = e.get("frame", "")

            if "10-K" in form:
                annual.append(e)
                continue

            if frame and re.match(r"^CY\d{4}$", frame):
                annual.append(e)
                continue

            start = e.get("start")
            end = e.get("end")
            if start and end:
                try:
                    days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
                    if 350 <= days <= 380:
                        annual.append(e)
                except ValueError:
                    pass

        return annual

    def _filter_quarterly(self, entries: list[dict]) -> list[dict]:
        quarterly = []
        for e in entries:
            form = e.get("form", "")
            frame = e.get("frame", "")

            if "10-Q" in form:
                quarterly.append(e)
                continue

            if frame and re.match(r"^CY\d{4}Q\d$", frame):
                quarterly.append(e)
                continue

            start = e.get("start")
            end = e.get("end")
            if start and end:
                try:
                    days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
                    if 80 <= days <= 100:
                        quarterly.append(e)
                except ValueError:
                    pass

        return quarterly

    @staticmethod
    def _to_millions(value) -> float | None:
        if value is None:
            return None
        return round(value / 1_000_000, 2)

    @staticmethod
    def _compute_deltas(values: list) -> list:
        deltas = []
        for i, v in enumerate(values):
            if i == 0 or values[i - 1] is None or v is None:
                deltas.append(None)
            else:
                deltas.append(round(v - values[i - 1], 2))
        return deltas

    def _format_annual(self, entries: list[dict], tag: str) -> dict:
        years = [int(e["end"][:4]) if e.get("end") else None for e in entries]
        values = [self._to_millions(e.get("val")) for e in entries]
        return {
            "tag": tag,
            "unit": "USD_millions",
            "years": years,
            "values": values,
            "deltas": self._compute_deltas(values),
        }

    def _format_quarterly(self, entries: list[dict], tag: str) -> dict:
        periods = []
        values = []
        for e in entries:
            end_date = e.get("end", "")
            frame = e.get("frame", "")
            year = int(end_date[:4]) if end_date else None
            quarter = None
            if frame:
                match = re.search(r"Q(\d)$", frame)
                if match:
                    quarter = int(match.group(1))
            periods.append(f"{year}Q{quarter}" if year and quarter else end_date)
            values.append(self._to_millions(e.get("val")))
        return {
            "tag": tag,
            "unit": "USD_millions",
            "periods": periods,
            "values": values,
            "deltas": self._compute_deltas(values),
        }

    def get_all_values(
        self,
        facts: dict,
        metric_alias: str,
        n_years: int = 5,
        taxonomy: str = "us-gaap",
    ) -> dict[str, dict]:
        n_quarters = n_years * 4

        tag, raw_entries = self._resolve_tag(facts, metric_alias, taxonomy)
        consolidated = self._filter_consolidated(raw_entries)

        annual = self._filter_annual(consolidated)
        quarterly = self._filter_quarterly(consolidated)

        annual.sort(key=lambda e: e.get("end", ""))
        # Deduplicate by fiscal year: keep only the latest filing per year
        # (handles amendments like 10-K/A which would otherwise double-count a year)
        seen_years: dict[int, dict] = {}
        for e in annual:
            year = int(e["end"][:4]) if e.get("end") else None
            if year is not None:
                seen_years[year] = e  # later entries overwrite earlier ones (already sorted)
        annual = sorted(seen_years.values(), key=lambda e: e.get("end", ""))
        annual = annual[-n_years:]

        quarterly.sort(key=lambda e: e.get("end", ""))
        quarterly = quarterly[-n_quarters:]

        return {
            "annual": self._format_annual(annual, tag),
            "quarterly": self._format_quarterly(quarterly, tag),
        }
