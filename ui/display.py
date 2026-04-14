from __future__ import annotations

from typing import Any


def format_display_label(value: str | None) -> str:
    if not value:
        return "-"
    if value.isupper():
        return value
    return str(value).replace("_", " ").replace("-", " ").title()


def build_compact_match_rows(matches: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Ticker": match.ticker,
            "Company": match.company or "-",
            "Matched Filing Year": match.matched_filing_year,
        }
        for match in matches
    ]


def build_peer_snapshot_rows(snapshot: Any) -> list[dict[str, str]]:
    if not snapshot:
        return []
    return [
        {"Category": "Common Strengths", "Details": ", ".join(snapshot.common_strengths) or "-"},
        {"Category": "Common Pressures", "Details": ", ".join(snapshot.common_pressures) or "-"},
        {"Category": "Shared Risk Types", "Details": ", ".join(format_display_label(item) for item in snapshot.shared_risk_types) or "-"},
        {"Category": "Target Differences", "Details": ", ".join(format_display_label(item) for item in snapshot.target_differences) or "-"},
    ]
