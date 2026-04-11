from __future__ import annotations

from typing import Dict, Optional

from constants import DELTA_BUCKETS, NUMERIC_METRICS
from models import NumericDelta


def build_numeric_deltas(financial_data: Dict, filing_year: int) -> Dict[str, NumericDelta]:
    annual = financial_data.get("annual", {})
    deltas: Dict[str, NumericDelta] = {}
    for metric in NUMERIC_METRICS:
        points = annual.get(metric, [])
        current = _find_point(points, filing_year)
        previous = _find_point(points, filing_year - 1)
        delta = NumericDelta(
            current_value=_point_value(current),
            previous_value=_point_value(previous),
            current_end_date=_point_field(current, "end_date"),
            previous_end_date=_point_field(previous, "end_date"),
            current_accession=_point_field(current, "accession"),
            previous_accession=_point_field(previous, "accession"),
            current_tag=_point_field(current, "tag"),
            previous_tag=_point_field(previous, "tag"),
        )
        if current is None:
            delta.reason = "missing_current_year"
        elif previous is None:
            delta.reason = "missing_prior_year"
        else:
            change = _compute_percent_change(delta.current_value, delta.previous_value)
            if change is None:
                delta.reason = "invalid_prior_value"
            else:
                delta.delta_percent = change
                delta.label = _bucket_label(change)
        deltas[metric] = delta
    return deltas


def _find_point(points, year: int) -> Optional[Dict]:
    for point in points:
        if int(point.get("year", -1)) == year:
            return point
    return None


def _point_value(point: Optional[Dict]) -> Optional[float]:
    if point is None:
        return None
    value = point.get("value")
    if value is None:
        return None
    return float(value)


def _point_field(point: Optional[Dict], field: str) -> Optional[str]:
    if point is None:
        return None
    value = point.get(field)
    return None if value is None else str(value)


def _compute_percent_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


def _bucket_label(delta_percent: float) -> str:
    for threshold, label in DELTA_BUCKETS:
        if delta_percent > threshold:
            return label
    return "severe_decline"
