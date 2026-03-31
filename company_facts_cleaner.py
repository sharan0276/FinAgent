"""
Cleans raw XBRL financial data from SEC Company Facts API.
Handles the real-world data traps that break naive systems:
  1. Tag routing      — companies switch XBRL tags over time
  2. Sliced pie trap  — filter out segment breakdowns, keep consolidated totals
  3. Period filter    — separate annual and quarterly values

Focused on US technology companies.
"""

import re
from datetime import datetime

class DataCleaner:
    """
    Cleans raw XBRL data from SEC Company Facts API into
    validated annual and quarterly financial metrics ready for Agent 3.

    Single public orchestrator: get_all_values()
    Returns both annual and quarterly in one pass — no duplicate processing.

    Usage:
        cleaner = DataCleaner()
        result  = cleaner.get_all_values(facts, "Revenues")
        annual    = result["annual"]
        quarterly = result["quarterly"]
    """

    # TAG Fall back Chains - ensure that metric names are listed and mentioend for different organizations
    _TAG_FALLBACK_CHAINS: dict[str, list[str]] = {

        # Revenue — Apple switched tags in 2018
        "Revenues": [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ],

        # Net Income
        "NetIncome": [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ],

        # Cash
        "Cash": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsAndShortTermInvestments",
            "Cash",
        ],

        # Total Assets
        "Assets": [
            "Assets",
        ],

        # Long Term Debt — convertible notes common in tech
        "LongTermDebt": [
            "LongTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermNotesPayable",
            "ConvertibleNotesPayable",
        ],

        # Operating Cash Flow — cash burn signal
        "OperatingCashFlow": [
            "NetCashProvidedByUsedInOperatingActivities",
        ],

        # R&D Spend — sudden cuts signal financial stress
        "ResearchAndDevelopment": [
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ],

        # Gross Profit — margin compression signal
        "GrossProfit": [
            "GrossProfit",
        ],
    }

    def _resolve_tag(self, facts:dict, metric_alias: str, taxonomy: str = "us-gaap") -> tuple[str, list]:
        """
        Resolves the actual XBRL tag for a metric alias using fallback chains.
        
        Raises the KeyError if no tag in the TAG_FALLBACK_CHAINS matches the alias.
        """

        chain = self._TAG_FALLBACK_CHAINS.get(metric_alias, [metric_alias])
        taxonomy_data = facts.get("facts", {}).get(taxonomy, {})

        for tag in chain:
            if tag in taxonomy_data:
                units_dict = taxonomy_data[tag].get("units", {})
                for unit_key, enteries in units_dict.items():
                    return tag, enteries
        
        raise KeyError(f"No XBRL tag found for metric '{metric_alias}' using chain: {chain}")


        def _filter_consolidated(self, enteries: list[dict]) -> list[dict]:
            """
            Remove segment/market type breakdowns.
            Keep only consolidated total enteries
            """

            return [
                e for e in enteries if "segment" not in e or e.get("segment") is None
            ]
            
        

    def _filter_annual(self, entries: list[dict]) -> list[dict]:
        """
        Keep only annual 10-K entries.
        Three methods tried in order: form field, frame field, duration check.
        """
        annual = []
        for e in entries:
            form  = e.get("form", "")
            frame = e.get("frame", "")

            if "10-K" in form:
                annual.append(e)
                continue

            if frame and re.match(r"^CY\d{4}$", frame):
                annual.append(e)
                continue

            start = e.get("start")
            end   = e.get("end")
            if start and end:
                try:
                    days = (
                        datetime.fromisoformat(end) -
                        datetime.fromisoformat(start)
                    ).days
                    if 350 <= days <= 380:
                        annual.append(e)
                except ValueError:
                    pass

        return annual

        


    def _filter_quarterly(self, entries: list[dict]) -> list[dict]:
        """
        Keep only quarterly 10-Q entries.
        Three methods tried in order: form field, frame field, duration check.
        """
        quarterly = []
        for e in entries:
            form  = e.get("form", "")
            frame = e.get("frame", "")

            if "10-Q" in form:
                quarterly.append(e)
                continue

            if frame and re.match(r"^CY\d{4}Q\d$", frame):
                quarterly.append(e)
                continue

            start = e.get("start")
            end   = e.get("end")
            if start and end:
                try:
                    days = (
                        datetime.fromisoformat(end) -
                        datetime.fromisoformat(start)
                    ).days
                    if 80 <= days <= 100:
                        quarterly.append(e)
                except ValueError:
                    pass

        return quarterly


    def _format_annual(self, entries: list[dict], tag: str) -> list[dict]:
        """Shape annual entries into clean output dicts."""
        return [
            {
                "year":      int(e["end"][:4]) if e.get("end") else None,
                "end_date":  e.get("end"),
                "value":     e.get("val"),
                "tag":       tag,
                "accession": e.get("accn"),
            }
            for e in entries
        ]

    def _format_quarterly(self, entries: list[dict], tag: str) -> list[dict]:
        """Shape quarterly entries into clean output dicts."""
        results = []
        for e in entries:
            end_date = e.get("end", "")
            frame    = e.get("frame", "")
            quarter  = None
            if frame:
                match = re.search(r"Q(\d)$", frame)
                if match:
                    quarter = int(match.group(1))
            results.append({
                "year":      int(end_date[:4]) if end_date else None,
                "quarter":   quarter,
                "end_date":  end_date,
                "value":     e.get("val"),
                "tag":       tag,
                "accession": e.get("accn"),
            })
        return results



    def get_all_values(self, facts: dict, metric_alias: str, n_years: int = 5, taxonomy: str = "us-gaap",) -> dict[str, list[dict]]:
        """
        Single pass pipeline for one metric:
          1. _resolve_tag()           — find correct XBRL tag
          2. _filter_consolidated()   — remove segment splits
          3. _filter_annual()         — split off annual entries
          4. _filter_quarterly()      — split off quarterly entries
          5. _format_annual()         — shape into clean dicts
          6. _format_quarterly()      — shape into clean dicts

        Args:
            facts:         Raw company facts dict from get_company_facts()
            metric_alias:  Friendly name e.g. "Revenues", "Cash"
            n_years:       Years to return (default 5) — quarters = n_years * 4
            taxonomy:      "us-gaap" (default) or "ifrs-full"

        Returns:
            {
                "annual":    [{ year, end_date, value, tag, accession }, ...],
                "quarterly": [{ year, quarter, end_date, value, tag, accession }, ...]
            }

        Raises:
            KeyError if metric not found for this company
        """
        n_quarters = n_years * 4

        # Step 1 + 2 — run once, shared by both annual and quarterly
        tag, raw_entries = self._resolve_tag(facts, metric_alias, taxonomy)
        consolidated     = self._filter_consolidated(raw_entries)

        # Step 3 + 4 — split consolidated entries into annual and quarterly
        annual    = self._filter_annual(consolidated)
        quarterly = self._filter_quarterly(consolidated)

        # Sort and cap both lists
        annual.sort(key=lambda e: e.get("end", ""))
        annual = annual[-n_years:]

        quarterly.sort(key=lambda e: e.get("end", ""))
        quarterly = quarterly[-n_quarters:]

        # Step 5 + 6 — format and return both
        return {
            "annual":    self._format_annual(annual, tag),
            "quarterly": self._format_quarterly(quarterly, tag),
        }