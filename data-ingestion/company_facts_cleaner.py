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
        "GrossProfit": ["GrossProfit", "GrossMargin"],
    }

    _FLOW_METRICS = {
        "Revenues",
        "NetIncome",
        "OperatingCashFlow",
        "ResearchAndDevelopment",
        "GrossProfit",
    }

    _SNAPSHOT_METRICS = {
        "Cash",
        "Assets",
        "LongTermDebt",
    }

    _QUARTER_FP_MAP = {
        "Q1": 1,
        "Q2": 2,
        "Q3": 3,
        "Q4": 4,
    }

    def _filter_consolidated(self, entries: list[dict]) -> list[dict]:
        return [
            entry for entry in entries if "segment" not in entry or entry.get("segment") is None
        ]

    def _has_duration(self, entry: dict) -> bool:
        return bool(entry.get("start") and entry.get("end"))

    def _duration_days(self, entry: dict) -> int | None:
        start = entry.get("start")
        end = entry.get("end")
        if not (start and end):
            return None

        try:
            return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
        except ValueError:
            return None

    def _infer_year(self, entry: dict) -> int | None:
        end = entry.get("end")
        if not end:
            return None

        try:
            return int(end[:4])
        except ValueError:
            return None

    def _infer_quarter(self, entry: dict) -> int | None:
        fp = (entry.get("fp") or "").upper()
        if fp in self._QUARTER_FP_MAP:
            return self._QUARTER_FP_MAP[fp]

        frame = entry.get("frame", "")
        match = re.search(r"Q(\d)$", frame)
        if match:
            return int(match.group(1))

        form = (entry.get("form") or "").upper()
        if form in {"10-K", "10-K/A"} and not self._has_duration(entry):
            return 4

        return None

    def _entry_rank(self, entry: dict) -> tuple[str, int, int, str, str]:
        filed = entry.get("filed") or ""
        form = (entry.get("form") or "").upper()
        amended = 1 if form.endswith("/A") else 0
        tag_priority = -(entry.get("_tag_rank", 999))
        end = entry.get("end") or ""
        accession = entry.get("accn") or ""
        return (filed, amended, tag_priority, end, accession)

    def _dedupe_latest(self, entries: list[dict], key_builder) -> list[dict]:
        chosen: dict[object, dict] = {}

        for entry in entries:
            key = key_builder(entry)
            if key is None:
                continue

            current = chosen.get(key)
            if current is None or self._entry_rank(entry) > self._entry_rank(current):
                chosen[key] = entry

        return list(chosen.values())

    def _is_annual_entry(self, entry: dict) -> bool:
        form = (entry.get("form") or "").upper()
        fp = (entry.get("fp") or "").upper()
        frame = entry.get("frame", "")
        duration_days = self._duration_days(entry)

        if form in {"10-K", "10-K/A"} and (fp == "FY" or not self._has_duration(entry)):
            return True

        if frame and re.match(r"^CY\d{4}$", frame):
            return True

        return duration_days is not None and 350 <= duration_days <= 380

    def _annual_entries(self, entries: list[dict]) -> list[dict]:
        annual = [entry for entry in entries if self._is_annual_entry(entry)]
        annual = self._dedupe_latest(annual, self._infer_year)
        annual.sort(key=lambda entry: entry.get("end", ""))
        return annual

    def _quarterly_snapshot_entries(self, entries: list[dict]) -> list[dict]:
        snapshot_entries = [
            entry
            for entry in entries
            if not self._has_duration(entry)
        ]
        snapshot_entries = self._dedupe_latest(snapshot_entries, lambda entry: entry.get("end"))
        snapshot_entries.sort(key=lambda entry: entry.get("end", ""))
        return snapshot_entries

    def _quarterly_flow_entries(self, entries: list[dict]) -> list[dict]:
        annual_by_year = {
            self._infer_year(entry): entry
            for entry in self._annual_entries(entries)
            if self._infer_year(entry) is not None
        }

        quarter_candidates: dict[tuple[int, int], dict] = {}
        cumulative_by_year: dict[int, dict[int, dict]] = {}

        for entry in entries:
            quarter = self._infer_quarter(entry)
            year = self._infer_year(entry)
            if quarter is None or year is None:
                continue

            duration_days = self._duration_days(entry)
            if quarter == 4 and year in annual_by_year:
                continue

            if quarter == 1:
                current = quarter_candidates.get((year, 1))
                if current is None or self._entry_rank(entry) > self._entry_rank(current):
                    quarter_candidates[(year, 1)] = entry
                continue

            if duration_days is not None and duration_days <= 100:
                current = quarter_candidates.get((year, quarter))
                if current is None or self._entry_rank(entry) > self._entry_rank(current):
                    quarter_candidates[(year, quarter)] = entry
                continue

            cumulative_by_year.setdefault(year, {})
            current = cumulative_by_year[year].get(quarter)
            if current is None or self._entry_rank(entry) > self._entry_rank(current):
                cumulative_by_year[year][quarter] = entry

        discrete_entries: list[dict] = []

        for year, q1_entry in quarter_candidates.items():
            if year[1] == 1:
                discrete_entries.append(q1_entry)

        for year, quarter_entries in cumulative_by_year.items():
            q1_entry = quarter_candidates.get((year, 1))
            q2_entry = quarter_entries.get(2)
            q3_entry = quarter_entries.get(3)
            annual_entry = annual_by_year.get(year)

            if q2_entry is not None:
                q2_value = q2_entry.get("val")
                q1_value = q1_entry.get("val") if q1_entry is not None else None
                if q2_value is not None:
                    discrete_entries.append(
                        {
                            **q2_entry,
                            "val": q2_value - q1_value if q1_value is not None else q2_value,
                            "quarter": 2,
                        }
                    )

            if q3_entry is not None:
                q3_value = q3_entry.get("val")
                q2_value = quarter_entries.get(2, {}).get("val")
                if q3_value is not None:
                    discrete_entries.append(
                        {
                            **q3_entry,
                            "val": q3_value - q2_value if q2_value is not None else q3_value,
                            "quarter": 3,
                        }
                    )

            if annual_entry is not None:
                annual_value = annual_entry.get("val")
                q3_value = quarter_entries.get(3, {}).get("val")
                if annual_value is not None:
                    discrete_entries.append(
                        {
                            **annual_entry,
                            "val": annual_value - q3_value if q3_value is not None else annual_value,
                            "quarter": 4,
                        }
                    )

        discrete_entries = self._dedupe_latest(
            discrete_entries,
            lambda entry: (entry.get("end"), entry.get("quarter")),
        )
        discrete_entries.sort(key=lambda entry: entry.get("end", ""))
        return discrete_entries

    def _candidate_tag_entries(
        self,
        facts: dict,
        tag: str,
        taxonomy: str,
        tag_rank: int,
    ) -> list[dict]:
        taxonomy_data = facts.get("facts", {}).get(taxonomy, {})
        tag_data = taxonomy_data.get(tag, {})
        units_dict = tag_data.get("units", {})

        preferred_units = ("USD", "USD/shares", "shares", "pure")
        for unit in preferred_units:
            entries = units_dict.get(unit)
            if entries:
                return [
                    {
                        **entry,
                        "_tag": tag,
                        "_tag_rank": tag_rank,
                    }
                    for entry in self._filter_consolidated(entries)
                ]

        for entries in units_dict.values():
            if entries:
                return [
                    {
                        **entry,
                        "_tag": tag,
                        "_tag_rank": tag_rank,
                    }
                    for entry in self._filter_consolidated(entries)
                ]

        return []

    def _resolve_entries(
        self,
        facts: dict,
        metric_alias: str,
        taxonomy: str = "us-gaap",
    ) -> list[dict]:
        chain = self._TAG_FALLBACK_CHAINS.get(metric_alias, [metric_alias])
        combined_entries: list[dict] = []

        for tag_rank, tag in enumerate(chain):
            combined_entries.extend(
                self._candidate_tag_entries(
                    facts,
                    tag,
                    taxonomy,
                    tag_rank=tag_rank,
                )
            )

        if not combined_entries:
            raise KeyError(
                f"No XBRL tag found for metric '{metric_alias}' using chain: {chain}"
            )

        return combined_entries

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

    def _format_annual(self, entries: list[dict]) -> dict:
        tag = entries[0].get("_tag", "") if entries else ""
        years = [self._infer_year(e) for e in entries]
        values = [self._to_millions(e.get("val")) for e in entries]
        return {
            "tag": tag,
            "unit": "USD_millions",
            "years": years,
            "values": values,
            "deltas": self._compute_deltas(values),
        }

    def _format_quarterly(self, entries: list[dict]) -> dict:
        tag = entries[0].get("_tag", "") if entries else ""
        periods = []
        values = []
        for e in entries:
            year = self._infer_year(e)
            quarter = e.get("quarter", self._infer_quarter(e))
            periods.append(f"{year}Q{quarter}" if year and quarter else (e.get("end") or ""))
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

        raw_entries = self._resolve_entries(facts, metric_alias, taxonomy=taxonomy)

        annual = self._annual_entries(raw_entries)[-n_years:]
        if metric_alias in self._FLOW_METRICS:
            quarterly = self._quarterly_flow_entries(raw_entries)[-n_quarters:]
        else:
            quarterly = self._quarterly_snapshot_entries(raw_entries)[-n_quarters:]

        return {
            "annual": self._format_annual(annual),
            "quarterly": self._format_quarterly(quarterly),
        }
