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

def build_numeric_deltas(
    financial_data: Dict, 
    target_year: int, 
    accession: str = ""
) -> Dict[str, NumericDelta]:
    """
    Identifies 'current' and 'previous' years for each metric and computes delta.
    
    Data Structure (from complete_ingestion.json):
    financial_data['annual']['MetricName'] = {
        'tag': '...',
        'years': [2025, 2024, 2023],
        'values': [100.0, 90.0, 80.0],
        ...
    }
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
        
        # Extract and scale values
        current_val = _get_value_at(info, current_idx)
        previous_val = _get_value_at(info, previous_idx)
        
        delta = NumericDelta(
            current_value=current_val,
            previous_value=previous_val,
            current_end_date=None, 
            previous_end_date=None,
            current_accession=None,
            previous_accession=None,
            current_tag=info.get("tag"),
            previous_tag=info.get("tag"),
            delta_percent=_compute_percent_change(current_val, previous_val) if current_val and previous_val else None,
            label=_bucket_label(_compute_percent_change(current_val, previous_val)) if current_val and previous_val else None,
            reason=_get_reason(current_val, previous_val)
        )
        deltas[metric] = delta

    return deltas

def _find_year_index(info: Dict, target_year: int) -> Optional[int]:
    """Returns the index of the target year in the 'years' list."""
    if not isinstance(info, dict): return None
    years = info.get("years", [])
    try:
        return years.index(target_year)
    except (ValueError, AttributeError):
        return None

def _get_value_at(info: Dict, index: Optional[int]) -> Optional[float]:
    """Returns the raw float value (scaled by 1M) at a specific index."""
    if index is None: return None
    values = info.get("values", [])
    try:
        # Scaling Millions (from ingestion) to Raw (for extraction output)
        return float(values[index]) * 1_000_000
    except (IndexError, TypeError, ValueError):
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

def _bucket_label(delta_pct: float) -> str:
    if delta_pct > 20:
        return "strong_growth"
    elif delta_pct > 5:
        return "moderate_growth"
    elif delta_pct >= -5:
        return "stable"
    elif delta_pct >= -20:
        return "moderate_decline"
    else:
        return "severe_decline"
