from __future__ import annotations

from typing import Dict, List, Optional
from models import NumericDelta

# The list of metrics we must compute deltas for
NUMERIC_METRICS = [
    "Revenues",
    "NetIncome",
    "Cash",
    "Assets",
    "LongTermDebt",
    "OperatingCashFlow",
    "ResearchAndDevelopment",
    "GrossProfit",
]

# Metrics where a DECREASE year-over-year is financially POSITIVE.
# For these, the delta sign is flipped before bucketing so that
# e.g. LongTermDebt -6% → "moderate_growth" (debt paydown = good).
INVERSE_METRICS = {
    "LongTermDebt",   # paying down debt improves balance sheet strength
}

def build_numeric_deltas(
    financial_data: Dict, 
    target_year: int, 
    accession: str = ""
) -> Dict[str, NumericDelta]:
    """
    Identifies 'current' and 'previous' years for each metric and computes delta.
    
    Handles two data structures:
    1. Dict based (AAPL style): {'tag': '...', 'years': [...], 'values': [...]}
    2. List based (AMZN style): [{'year': 2025, 'value': 100, 'tag': '...', ...}, ...]
    """
    annual = financial_data.get("annual", {})
    deltas: Dict[str, NumericDelta] = {}

    for metric in NUMERIC_METRICS:
        info = annual.get(metric, {})
        if not info:
            continue
        
        # Look up indexes for current (target) and previous (target - 1) years
        current_idx = _find_year_index(info, target_year)
        previous_idx = _find_year_index(info, target_year - 1)
        
        # Extract values
        current_val = _get_value_at(info, current_idx)
        previous_val = _get_value_at(info, previous_idx)

        # Extract tags and metadata safely
        current_tag = _get_meta_at(info, current_idx, "tag")
        previous_tag = _get_meta_at(info, previous_idx, "tag")
        current_end = _get_meta_at(info, current_idx, "end_date")
        previous_end = _get_meta_at(info, previous_idx, "end_date")
        current_acc = _get_meta_at(info, current_idx, "accession")
        previous_acc = _get_meta_at(info, previous_idx, "accession")
        
        delta = NumericDelta(
            current_value=current_val,
            previous_value=previous_val,
            current_end_date=current_end, 
            previous_end_date=previous_end,
            current_accession=current_acc,
            previous_accession=previous_acc,
            current_tag=current_tag,
            previous_tag=previous_tag,
            delta_percent=_compute_percent_change(current_val, previous_val) if current_val and previous_val else None,
            label=_bucket_label(_compute_percent_change(current_val, previous_val), metric) if current_val and previous_val else None,
            reason=_get_reason(current_val, previous_val)
        )
        deltas[metric] = delta

    return deltas

def _find_year_index(info: Any, target_year: int) -> Optional[int]:
    """Returns the index of the target year in the data structure."""
    if isinstance(info, dict):
        years = info.get("years", [])
        try:
            return years.index(target_year)
        except (ValueError, AttributeError):
            return None
    elif isinstance(info, list):
        for i, entry in enumerate(info):
            if isinstance(entry, dict) and entry.get("year") == target_year:
                return i
    return None

def _get_value_at(info: Any, index: Optional[int]) -> Optional[float]:
    """Returns the float value at a specific index, scaled correctly."""
    if index is None: return None
    if isinstance(info, dict):
        values = info.get("values", [])
        try:
            # Dict values are in millions, scale to raw
            return float(values[index]) * 1_000_000
        except (IndexError, TypeError, ValueError):
            return None
    elif isinstance(info, list):
        try:
            # List values are already raw (unscaled)
            return float(info[index].get("value", 0))
        except (IndexError, KeyError, TypeError, ValueError):
            return None
    return None

def _get_meta_at(info: Any, index: Optional[int], key: str) -> Optional[str]:
    """Helper to safely extract metadata from both formats."""
    if index is None: return None
    if isinstance(info, dict):
        return info.get(key)
    elif isinstance(info, list):
        try:
            return info[index].get(key)
        except (IndexError, AttributeError):
            return None
    return None

def _get_reason(current: Optional[float], previous: Optional[float]) -> Optional[str]:
    reasons = []
    if current is None:
        reasons.append("missing_current_year")
    if previous is None:
        reasons.append("missing_previous_year")
    return ", ".join(reasons) if reasons else None

def _compute_percent_change(current: float, previous: float) -> float:
    if not previous: return 0.0
    return ((current - previous) / previous) * 100.0

def _bucket_label(delta_pct: float, metric: str = "") -> str:
    """Buckets a percentage delta into a severity label.

    For INVERSE_METRICS (e.g. LongTermDebt), a negative delta is financially
    positive (debt paydown), so the sign is flipped before bucketing.
    """
    effective = -delta_pct if metric in INVERSE_METRICS else delta_pct
    # Use debt-specific labels so meaning is unambiguous
    if metric in INVERSE_METRICS:
        # Use debt-specific labels so meaning is unambiguous
        if effective > 20:   return "strong_reduction"
        elif effective > 5:  return "moderate_reduction"
        elif effective >= -5: return "stable"
        elif effective >= -20: return "moderate_increase"
        else:                return "severe_increase"
    else:
        if effective > 20:   return "strong_growth"
        elif effective > 5:  return "moderate_growth"
        elif effective >= -5: return "stable"
        elif effective >= -20: return "moderate_decline"
        else:                return "severe_decline"