from __future__ import annotations

from statistics import pstdev
from typing import Dict, List, Optional

from .models import MetricTrendSummaryV1


FLAT_THRESHOLD = 0.03


def build_numeric_trend_summaries(financial_data: Dict) -> Dict[str, MetricTrendSummaryV1]:
    annual = financial_data.get("annual", {})
    quarterly = financial_data.get("quarterly", {})
    metrics = sorted(set(annual.keys()) | set(quarterly.keys()))
    return {
        metric: _summarize_metric(
            metric,
            sorted(annual.get(metric, []), key=lambda item: (item.get("year") or -1, item.get("end_date") or "")),
            sorted(
                quarterly.get(metric, []),
                key=lambda item: ((item.get("year") or -1), (item.get("quarter") or -1), item.get("end_date") or ""),
            ),
        )
        for metric in metrics
    }


def _summarize_metric(metric: str, annual_points: List[Dict], quarterly_points: List[Dict]) -> MetricTrendSummaryV1:
    notes: List[str] = []
    annual_values = [point["value"] for point in annual_points if _has_numeric_value(point)]
    quarterly_values = [point["value"] for point in quarterly_points if _has_numeric_value(point)]

    if len(annual_values) < 5:
        notes.append("Less than 5 annual datapoints available for 5-year trend analysis.")
    if len(annual_values) < 3:
        notes.append("Less than 3 annual datapoints available for 3-year trend analysis.")
    if len(quarterly_values) < 4:
        notes.append("Less than 4 quarterly datapoints available for recent quarter trend analysis.")

    latest_annual = annual_points[-1] if annual_points else None
    latest_quarterly = quarterly_points[-1] if quarterly_points else None
    return MetricTrendSummaryV1(
        metric_name=metric,
        frequency_covered=_frequency_covered(annual_points, quarterly_points),
        latest_annual_value=latest_annual.get("value") if latest_annual else None,
        latest_annual_period_end=latest_annual.get("end_date") if latest_annual else None,
        latest_quarterly_value=latest_quarterly.get("value") if latest_quarterly else None,
        latest_quarterly_period_end=latest_quarterly.get("end_date") if latest_quarterly else None,
        annual_direction_5y=_direction_label(annual_values[-5:], min_points=5),
        annual_direction_3y=_direction_label(annual_values[-3:], min_points=3),
        annual_cagr_5y=_compute_cagr(annual_values[-5:]),
        annual_yoy_latest=_compute_latest_yoy(annual_values),
        quarterly_direction_recent=_direction_label(quarterly_values[-4:], min_points=4),
        quarterly_volatility_flag=_quarterly_volatility_flag(quarterly_values[-8:]),
        annual_series=[_annual_series_item(point) for point in annual_points],
        quarterly_series=[_quarterly_series_item(point) for point in quarterly_points],
        missing_data_notes=notes,
        source_metric_refs={"annual": f"financial_data.annual.{metric}", "quarterly": f"financial_data.quarterly.{metric}"},
    )


def _frequency_covered(annual_points: List[Dict], quarterly_points: List[Dict]) -> str:
    if annual_points and quarterly_points:
        return "both"
    if annual_points:
        return "annual"
    if quarterly_points:
        return "quarterly"
    return "none"


def _has_numeric_value(point: Dict) -> bool:
    return isinstance(point.get("value"), (int, float))


def _direction_label(values: List[float], *, min_points: int) -> str:
    clean = [value for value in values if isinstance(value, (int, float))]
    if len(clean) < min_points:
        return "insufficient_data"

    pct_changes = []
    for prev, curr in zip(clean, clean[1:]):
        if prev == 0:
            if curr == 0:
                pct_changes.append(0.0)
            else:
                return "mixed"
        else:
            pct_changes.append((curr - prev) / abs(prev))

    if all(abs(change) <= FLAT_THRESHOLD for change in pct_changes):
        return "flat"
    if all(change >= FLAT_THRESHOLD for change in pct_changes):
        return "increasing"
    if all(change <= -FLAT_THRESHOLD for change in pct_changes):
        return "decreasing"
    return "mixed"


def _compute_cagr(values: List[float]) -> Optional[float]:
    clean = [value for value in values if isinstance(value, (int, float))]
    if len(clean) < 5:
        return None
    start, end = clean[0], clean[-1]
    periods = len(clean) - 1
    if start <= 0 or end <= 0 or periods <= 0:
        return None
    return (end / start) ** (1 / periods) - 1


def _compute_latest_yoy(values: List[float]) -> Optional[float]:
    clean = [value for value in values if isinstance(value, (int, float))]
    if len(clean) < 2:
        return None
    prev, curr = clean[-2], clean[-1]
    if prev == 0:
        return None
    return (curr - prev) / abs(prev)


def _quarterly_volatility_flag(values: List[float]) -> bool:
    clean = [value for value in values if isinstance(value, (int, float))]
    if len(clean) < 4:
        return False
    pct_changes: List[float] = []
    for prev, curr in zip(clean, clean[1:]):
        if prev == 0:
            continue
        pct_changes.append((curr - prev) / abs(prev))
    if len(pct_changes) < 3:
        return False
    return pstdev(pct_changes) > 0.25


def _annual_series_item(point: Dict) -> Dict:
    return {
        "year": point.get("year"),
        "end_date": point.get("end_date"),
        "value": point.get("value"),
        "accession": point.get("accession"),
        "tag": point.get("tag"),
    }


def _quarterly_series_item(point: Dict) -> Dict:
    return {
        "year": point.get("year"),
        "quarter": point.get("quarter"),
        "end_date": point.get("end_date"),
        "value": point.get("value"),
        "accession": point.get("accession"),
        "tag": point.get("tag"),
    }
