"""
synthetic_curator.py

Builds the curator JSON database for RAG matching experiments.

Two sources of data:
  1. REAL companies — converts data-extraction outputs (AAPL, META, GOOG)
     into curator-schema JSONs with risk signals derived from finbert-scored
     text candidates.
  2. SYNTHETIC companies — hand-crafted profiles for SNAP, WeWork, MSFT, NFLX
     representing different risk archetypes (cash-burn, collapse, growth, transition).

Output: curator_db/{ticker_lower}_{year}.json per file

Curator schema (from project signal schema):
  company, ticker, filing_year,
  financial_deltas: {metric: {value: float, label: str}},
  risk_signals: [{signal_id, topic, signal_type, section,
                  filing_year, company, summary, severity, citation}],
  embedding_text: str   <- used for embedding
  embedding_vector: []  <- filled by embedder.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXTRACTION_ROOT = Path(__file__).parent.parent / "data-extraction" / "outputs"
CURATOR_DB = Path(__file__).parent / "curator_db"

# ---------------------------------------------------------------------------
# Keyword → (topic, signal_type) mapping for risk signal derivation
# Each entry: (keywords_list, topic, signal_type, default_severity)
# ---------------------------------------------------------------------------
_SIGNAL_RULES: list[tuple[list[str], str, str, str]] = [
    (["going concern", "substantial doubt"],
     "liquidity_risk", "going_concern_warning", "high"),
    (["cash burn", "operating loss", "cash outflow", "negative cash"],
     "liquidity_risk", "cash_runway_concern", "high"),
    (["covenant", "waiver", "breach"],
     "liquidity_risk", "debt_covenant_risk", "high"),
    (["debt matur", "refinanc"],
     "liquidity_risk", "refinancing_risk", "medium"),
    (["impairment", "write-down", "write-off", "goodwill"],
     "liquidity_risk", "impairment_charges", "medium"),
    (["app store", "apple", "google play", "platform fee", "third-party platform"],
     "market_risk", "platform_dependency", "medium"),
    (["competition", "competitor", "market share", "competitive pressure"],
     "market_risk", "intensifying_competition", "medium"),
    (["user growth", "daily active", "engagement", "monthly active"],
     "market_risk", "network_effect_weakening", "medium"),
    (["advertiser", "advertising revenue", "ad spend"],
     "market_risk", "advertiser_concentration", "medium"),
    (["customer concentration", "one customer", "few customers"],
     "market_risk", "revenue_concentration_risk", "medium"),
    (["antitrust", "monopoly", "forced breakup"],
     "legal_risk", "antitrust_exposure", "high"),
    (["ftc", "sec investigation", "doj", "regulator", "regulatory scrutiny"],
     "legal_risk", "regulatory_investigation", "high"),
    (["gdpr", "ccpa", "privacy", "data protection", "data breach"],
     "legal_risk", "data_privacy_violation", "medium"),
    (["lawsuit", "litigation", "class action", "arbitration"],
     "legal_risk", "active_litigation", "medium"),
    (["cybersecurity", "ransomware", "unauthorized access", "breach", "security incident"],
     "legal_risk", "cybersecurity_incident", "high"),
    (["key personnel", "key executive", "founders", "ceo departure"],
     "operational_risk", "key_person_dependency", "medium"),
    (["supply chain", "sole supplier", "semiconductor", "component shortage"],
     "operational_risk", "supply_chain_concentration", "medium"),
    (["talent", "hiring", "retention", "attrition"],
     "operational_risk", "talent_retention_risk", "low"),
    (["internal control", "material weakness", "significant deficiency"],
     "operational_risk", "internal_controls_weakness", "high"),
    (["interest rate", "floating rate", "variable rate"],
     "macro_risk", "interest_rate_sensitivity", "medium"),
    (["foreign currency", "exchange rate", "currency risk", "fx"],
     "macro_risk", "foreign_currency_exposure", "medium"),
    (["china", "geopolit", "trade restrict", "huawei"],
     "macro_risk", "china_geopolitical_exposure", "medium"),
    (["tariff", "trade war", "import duty"],
     "macro_risk", "trade_tariff_risk", "medium"),
    (["inflation", "wage inflation", "input cost"],
     "macro_risk", "inflation_impact", "low"),
    (["metaverse", "virtual reality", "augmented reality", "vr headset", "unproven"],
     "strategic_risk", "moonshot_unproven_bet", "medium"),
    (["r&d", "research and development", "return on investment", "rd spend"],
     "strategic_risk", "rd_productivity_risk", "low"),
    (["acquisition", "integration challenge", "dilutive", "goodwill impairment"],
     "strategic_risk", "failed_acquisition", "medium"),
    (["business model", "pivot", "transition"],
     "strategic_risk", "business_model_transition_risk", "medium"),
    (["moat", "differentiator", "competitive advantage"],
     "strategic_risk", "competitive_moat_erosion", "low"),
]


def _derive_signal(
    sentence: str,
    section_label: str,
    company: str,
    filing_year: int,
    signal_id: int,
) -> dict[str, Any] | None:
    """Try to match a sentence to a risk signal type. Returns None if no match."""
    text_lower = sentence.lower()
    for keywords, topic, signal_type, severity in _SIGNAL_RULES:
        if any(kw in text_lower for kw in keywords):
            # Truncate to 2 sentences max
            summary = sentence.strip()
            if len(summary) > 250:
                summary = summary[:250].rsplit(" ", 1)[0] + "..."
            return {
                "signal_id": signal_id,
                "topic": topic,
                "signal_type": signal_type,
                "section": section_label,
                "filing_year": filing_year,
                "company": company,
                "summary": summary,
                "severity": severity,
                "citation": f"{company} 10-K {filing_year}, {section_label}",
            }
    return None


def _build_financial_deltas(numeric_deltas: dict) -> dict:
    """Map extraction numeric_deltas format → curator financial_deltas format."""
    result = {}
    for metric, data in numeric_deltas.items():
        if isinstance(data, dict) and data.get("label"):
            result[metric] = {
                "value": round(data["delta_percent"] / 100, 4) if data.get("delta_percent") is not None else None,
                "label": data["label"],
            }
    return result


def _build_embedding_text(
    company: str,
    filing_year: int,
    financial_deltas: dict,
    risk_signals: list[dict],
) -> str:
    parts = [f"Company: {company}, Year: {filing_year}."]

    # Financial trajectory
    for metric, data in financial_deltas.items():
        parts.append(f"{metric}: {data['label']}.")

    # Risk signal types and summaries
    if risk_signals:
        parts.append("Risk signals:")
        for sig in risk_signals[:8]:  # cap at 8 to keep embedding text focused
            parts.append(
                f"{sig['signal_type']} ({sig['severity']}): {sig['summary']}"
            )

    return " ".join(parts)


def _from_extraction(extraction_path: Path) -> dict:
    """Convert one data-extraction JSON into a curator-schema JSON."""
    data = json.loads(extraction_path.read_text(encoding="utf-8"))

    ticker = data["ticker"]
    company = data.get("company_name", ticker)
    filing_year = data["filing_year"]

    financial_deltas = _build_financial_deltas(data.get("numeric_deltas", {}))
    candidates = data.get("text_candidates", [])

    # Sort by risk_score descending, take top 20 for signal derivation
    top_candidates = sorted(candidates, key=lambda c: c.get("risk_score", 0), reverse=True)[:20]

    risk_signals: list[dict] = []
    seen_types: set[str] = set()
    signal_id = 1

    for cand in top_candidates:
        if len(risk_signals) >= 10:
            break
        sentence = cand.get("sentence_text", "")
        section_label = cand.get("section_label", "Item 1A")
        signal = _derive_signal(sentence, section_label, company, filing_year, signal_id)
        if signal and signal["signal_type"] not in seen_types:
            risk_signals.append(signal)
            seen_types.add(signal["signal_type"])
            signal_id += 1

    embedding_text = _build_embedding_text(company, filing_year, financial_deltas, risk_signals)

    return {
        "company": company,
        "ticker": ticker,
        "filing_year": filing_year,
        "financial_deltas": financial_deltas,
        "risk_signals": risk_signals,
        "embedding_text": embedding_text,
        "embedding_vector": [],
    }


# ---------------------------------------------------------------------------
# Synthetic company profiles
# Each profile defines financial_deltas labels and hand-crafted risk signals
# These represent archetypes useful for testing the matcher:
#   - SNAP: social media cash-burn stress
#   - WEWORK: pre-collapse distress
#   - MSFT: healthy strong-growth
#   - NFLX: content-investment transition pressure
# ---------------------------------------------------------------------------

def _synthetic_profile(
    ticker: str,
    company: str,
    filing_year: int,
    deltas: dict[str, tuple[float | None, str]],  # metric -> (value, label)
    signals: list[dict],
) -> dict:
    financial_deltas = {
        m: {"value": v, "label": lbl} for m, (v, lbl) in deltas.items()
    }
    embedding_text = _build_embedding_text(company, filing_year, financial_deltas, signals)
    return {
        "company": company,
        "ticker": ticker,
        "filing_year": filing_year,
        "financial_deltas": financial_deltas,
        "risk_signals": signals,
        "embedding_text": embedding_text,
        "embedding_vector": [],
    }


_SYNTHETIC_PROFILES: list[dict] = [

    # ------------------------------------------------------------------ SNAP
    _synthetic_profile("SNAP", "Snap Inc.", 2022, {
        "Revenues":             (-0.05, "stable"),
        "NetIncome":            (-0.23, "severe_decline"),
        "Cash":                 (-0.41, "severe_decline"),
        "Assets":               (-0.08, "moderate_decline"),
        "LongTermDebt":         (0.12,  "moderate_growth"),
        "OperatingCashFlow":    (-0.67, "severe_decline"),
        "ResearchAndDevelopment":(0.03, "stable"),
        "GrossProfit":          (-0.11, "moderate_decline"),
    }, [
        {"signal_id": 1, "topic": "liquidity_risk",  "signal_type": "cash_runway_concern",
         "section": "Item 7",  "filing_year": 2022, "company": "Snap Inc.",
         "summary": "Operating cash outflows widened year over year with no disclosed path to breakeven.",
         "severity": "high", "citation": "Snap Inc. 10-K 2022, Item 7"},
        {"signal_id": 2, "topic": "market_risk", "signal_type": "network_effect_weakening",
         "section": "Item 1A", "filing_year": 2022, "company": "Snap Inc.",
         "summary": "Daily active user growth decelerated significantly as TikTok and Reels competed for teen attention.",
         "severity": "high", "citation": "Snap Inc. 10-K 2022, Item 1A"},
        {"signal_id": 3, "topic": "market_risk", "signal_type": "platform_dependency",
         "section": "Item 1A", "filing_year": 2022, "company": "Snap Inc.",
         "summary": "Apple ATT privacy changes materially reduced ad targeting effectiveness and revenue.",
         "severity": "high", "citation": "Snap Inc. 10-K 2022, Item 1A"},
        {"signal_id": 4, "topic": "market_risk", "signal_type": "advertiser_concentration",
         "section": "Item 7",  "filing_year": 2022, "company": "Snap Inc.",
         "summary": "Advertising revenue is highly concentrated; macroeconomic pullback in ad spend caused revenue miss.",
         "severity": "medium", "citation": "Snap Inc. 10-K 2022, Item 7"},
    ]),

    _synthetic_profile("SNAP", "Snap Inc.", 2023, {
        "Revenues":             (-0.03, "stable"),
        "NetIncome":            (-0.18, "moderate_decline"),
        "Cash":                 (-0.55, "severe_decline"),
        "Assets":               (-0.12, "moderate_decline"),
        "LongTermDebt":         (0.08,  "stable"),
        "OperatingCashFlow":    (-0.50, "severe_decline"),
        "ResearchAndDevelopment":(-0.10, "moderate_decline"),
        "GrossProfit":          (-0.06, "moderate_decline"),
    }, [
        {"signal_id": 1, "topic": "liquidity_risk",  "signal_type": "cash_runway_concern",
         "section": "Item 7",  "filing_year": 2023, "company": "Snap Inc.",
         "summary": "Cash and equivalents declined 55% year over year; management acknowledged need to reach cash flow breakeven.",
         "severity": "high", "citation": "Snap Inc. 10-K 2023, Item 7"},
        {"signal_id": 2, "topic": "operational_risk", "signal_type": "talent_retention_risk",
         "section": "Item 1A", "filing_year": 2023, "company": "Snap Inc.",
         "summary": "Multiple rounds of layoffs totaling 20% of workforce raised execution and morale risks.",
         "severity": "medium", "citation": "Snap Inc. 10-K 2023, Item 1A"},
        {"signal_id": 3, "topic": "market_risk", "signal_type": "intensifying_competition",
         "section": "Item 1A", "filing_year": 2023, "company": "Snap Inc.",
         "summary": "TikTok, Instagram Reels, and YouTube Shorts all compete directly for the same short-form video audience.",
         "severity": "high", "citation": "Snap Inc. 10-K 2023, Item 1A"},
        {"signal_id": 4, "topic": "strategic_risk", "signal_type": "moonshot_unproven_bet",
         "section": "Item 1",  "filing_year": 2023, "company": "Snap Inc.",
         "summary": "AR glasses and Spectacles hardware remain pre-revenue with significant ongoing investment.",
         "severity": "medium", "citation": "Snap Inc. 10-K 2023, Item 1"},
    ]),

    # --------------------------------------------------------------- WEWORK
    _synthetic_profile("WE", "WeWork Inc.", 2020, {
        "Revenues":             (-0.18, "moderate_decline"),
        "NetIncome":            (-3.20, "severe_decline"),
        "Cash":                 (-0.62, "severe_decline"),
        "Assets":               (-0.08, "moderate_decline"),
        "LongTermDebt":         (0.45,  "strong_growth"),
        "OperatingCashFlow":    (-1.10, "severe_decline"),
        "ResearchAndDevelopment":(0.02, "stable"),
        "GrossProfit":          (-0.35, "severe_decline"),
    }, [
        {"signal_id": 1, "topic": "liquidity_risk",  "signal_type": "going_concern_warning",
         "section": "Item 7",  "filing_year": 2020, "company": "WeWork Inc.",
         "summary": "Auditors issued going concern qualification; management acknowledged substantial doubt about the company's ability to continue as a going concern.",
         "severity": "high", "citation": "WeWork Inc. 10-K 2020, Item 7"},
        {"signal_id": 2, "topic": "liquidity_risk",  "signal_type": "cash_runway_concern",
         "section": "Item 7",  "filing_year": 2020, "company": "WeWork Inc.",
         "summary": "Net loss of $3.2B with cash burn outpacing revenue; company required SoftBank bailout to fund operations.",
         "severity": "high", "citation": "WeWork Inc. 10-K 2020, Item 7"},
        {"signal_id": 3, "topic": "liquidity_risk",  "signal_type": "refinancing_risk",
         "section": "Item 7",  "filing_year": 2020, "company": "WeWork Inc.",
         "summary": "Long-term debt increased 45%; significant maturities approaching with uncertain refinancing path.",
         "severity": "high", "citation": "WeWork Inc. 10-K 2020, Item 7"},
        {"signal_id": 4, "topic": "operational_risk", "signal_type": "key_person_dependency",
         "section": "Item 1A", "filing_year": 2020, "company": "WeWork Inc.",
         "summary": "Departure of founder Adam Neumann and subsequent executive turnover created governance and operational uncertainty.",
         "severity": "high", "citation": "WeWork Inc. 10-K 2020, Item 1A"},
        {"signal_id": 5, "topic": "strategic_risk",  "signal_type": "business_model_transition_risk",
         "section": "Item 1A", "filing_year": 2020, "company": "WeWork Inc.",
         "summary": "Flexible workspace model heavily exposed to COVID-19 office avoidance with multi-year lease obligations creating severe mismatch.",
         "severity": "high", "citation": "WeWork Inc. 10-K 2020, Item 1A"},
    ]),

    _synthetic_profile("WE", "WeWork Inc.", 2021, {
        "Revenues":             (0.04, "stable"),
        "NetIncome":            (-4.60, "severe_decline"),
        "Cash":                 (-0.44, "severe_decline"),
        "Assets":               (-0.14, "moderate_decline"),
        "LongTermDebt":         (0.22,  "moderate_growth"),
        "OperatingCashFlow":    (-0.90, "severe_decline"),
        "ResearchAndDevelopment":(None, "stable"),
        "GrossProfit":          (-0.20, "moderate_decline"),
    }, [
        {"signal_id": 1, "topic": "liquidity_risk",  "signal_type": "going_concern_warning",
         "section": "Item 7",  "filing_year": 2021, "company": "WeWork Inc.",
         "summary": "Going concern language retained despite SPAC-merger proceeds; management cited continued cash burn and uncertain occupancy recovery.",
         "severity": "high", "citation": "WeWork Inc. 10-K 2021, Item 7"},
        {"signal_id": 2, "topic": "liquidity_risk",  "signal_type": "cash_runway_concern",
         "section": "Item 7",  "filing_year": 2021, "company": "WeWork Inc.",
         "summary": "Cash declined another 44% year over year despite IPO proceeds; operations remain deeply loss-making.",
         "severity": "high", "citation": "WeWork Inc. 10-K 2021, Item 7"},
        {"signal_id": 3, "topic": "legal_risk",      "signal_type": "active_litigation",
         "section": "Item 3",  "filing_year": 2021, "company": "WeWork Inc.",
         "summary": "Multiple landlord lawsuits and lease disputes following location closures and renegotiations.",
         "severity": "medium", "citation": "WeWork Inc. 10-K 2021, Item 3"},
        {"signal_id": 4, "topic": "strategic_risk",  "signal_type": "competitive_moat_erosion",
         "section": "Item 1",  "filing_year": 2021, "company": "WeWork Inc.",
         "summary": "Traditional landlords now directly offering flexible workspace, eroding WeWork's premium positioning.",
         "severity": "high", "citation": "WeWork Inc. 10-K 2021, Item 1"},
    ]),

    # ------------------------------------------------------------------ MSFT
    _synthetic_profile("MSFT", "Microsoft Corporation", 2022, {
        "Revenues":             (0.18, "moderate_growth"),
        "NetIncome":            (0.19, "moderate_growth"),
        "Cash":                 (0.11, "moderate_growth"),
        "Assets":               (0.15, "moderate_growth"),
        "LongTermDebt":         (-0.05, "stable"),
        "OperatingCashFlow":    (0.16, "moderate_growth"),
        "ResearchAndDevelopment":(0.22, "strong_growth"),
        "GrossProfit":          (0.19, "moderate_growth"),
    }, [
        {"signal_id": 1, "topic": "legal_risk", "signal_type": "antitrust_exposure",
         "section": "Item 1A", "filing_year": 2022, "company": "Microsoft Corporation",
         "summary": "EU and US regulators scrutinizing Activision Blizzard acquisition; potential for forced divestiture or conduct restrictions.",
         "severity": "medium", "citation": "Microsoft Corporation 10-K 2022, Item 1A"},
        {"signal_id": 2, "topic": "macro_risk",  "signal_type": "foreign_currency_exposure",
         "section": "Item 7A", "filing_year": 2022, "company": "Microsoft Corporation",
         "summary": "Strong US dollar headwind reduced reported revenue by approximately $1.8B compared to constant currency.",
         "severity": "low", "citation": "Microsoft Corporation 10-K 2022, Item 7A"},
        {"signal_id": 3, "topic": "macro_risk",  "signal_type": "china_geopolitical_exposure",
         "section": "Item 1A", "filing_year": 2022, "company": "Microsoft Corporation",
         "summary": "Operations in China face regulatory uncertainty including data localization requirements and potential access restrictions.",
         "severity": "low", "citation": "Microsoft Corporation 10-K 2022, Item 1A"},
    ]),

    _synthetic_profile("MSFT", "Microsoft Corporation", 2023, {
        "Revenues":             (0.07, "moderate_growth"),
        "NetIncome":            (0.26, "strong_growth"),
        "Cash":                 (0.18, "moderate_growth"),
        "Assets":               (0.34, "strong_growth"),
        "LongTermDebt":         (0.55, "strong_growth"),
        "OperatingCashFlow":    (0.14, "moderate_growth"),
        "ResearchAndDevelopment":(0.19, "moderate_growth"),
        "GrossProfit":          (0.09, "moderate_growth"),
    }, [
        {"signal_id": 1, "topic": "strategic_risk", "signal_type": "moonshot_unproven_bet",
         "section": "Item 1",  "filing_year": 2023, "company": "Microsoft Corporation",
         "summary": "Multi-billion dollar OpenAI investment and Copilot integration represent significant unproven bets on generative AI monetization.",
         "severity": "low", "citation": "Microsoft Corporation 10-K 2023, Item 1"},
        {"signal_id": 2, "topic": "legal_risk",     "signal_type": "regulatory_investigation",
         "section": "Item 1A", "filing_year": 2023, "company": "Microsoft Corporation",
         "summary": "FTC lawsuit to block Activision acquisition resolved but ongoing regulatory scrutiny of AI and cloud market power.",
         "severity": "medium", "citation": "Microsoft Corporation 10-K 2023, Item 1A"},
        {"signal_id": 3, "topic": "operational_risk","signal_type": "cybersecurity_vulnerability",
         "section": "Item 1A", "filing_year": 2023, "company": "Microsoft Corporation",
         "summary": "State-sponsored attacks on Microsoft cloud infrastructure disclosed; Secure Future Initiative launched in response.",
         "severity": "high", "citation": "Microsoft Corporation 10-K 2023, Item 1A"},
    ]),

    # ------------------------------------------------------------------ NFLX
    _synthetic_profile("NFLX", "Netflix Inc.", 2022, {
        "Revenues":             (0.07, "moderate_growth"),
        "NetIncome":            (-0.48, "severe_decline"),
        "Cash":                 (-0.21, "moderate_decline"),
        "Assets":               (0.04,  "stable"),
        "LongTermDebt":         (-0.04, "stable"),
        "OperatingCashFlow":    (0.85,  "strong_growth"),
        "ResearchAndDevelopment":(0.10, "moderate_growth"),
        "GrossProfit":          (0.04,  "stable"),
    }, [
        {"signal_id": 1, "topic": "market_risk",   "signal_type": "customer_churn_risk",
         "section": "Item 7",  "filing_year": 2022, "company": "Netflix Inc.",
         "summary": "Subscriber count declined in Q1 and Q2 2022 for the first time in company history, accelerating password-sharing crackdown.",
         "severity": "high", "citation": "Netflix Inc. 10-K 2022, Item 7"},
        {"signal_id": 2, "topic": "market_risk",   "signal_type": "intensifying_competition",
         "section": "Item 1A", "filing_year": 2022, "company": "Netflix Inc.",
         "summary": "Disney+, HBO Max, Peacock and Apple TV+ all competing aggressively with content spend and bundled pricing.",
         "severity": "medium", "citation": "Netflix Inc. 10-K 2022, Item 1A"},
        {"signal_id": 3, "topic": "strategic_risk","signal_type": "business_model_transition_risk",
         "section": "Item 1",  "filing_year": 2022, "company": "Netflix Inc.",
         "summary": "Ad-supported tier launch represents significant business model change with uncertain conversion rates and ad revenue timeline.",
         "severity": "medium", "citation": "Netflix Inc. 10-K 2022, Item 1"},
        {"signal_id": 4, "topic": "macro_risk",    "signal_type": "foreign_currency_exposure",
         "section": "Item 7A", "filing_year": 2022, "company": "Netflix Inc.",
         "summary": "Approximately 60% of revenue from outside US; dollar strength reduced reported revenue growth.",
         "severity": "low", "citation": "Netflix Inc. 10-K 2022, Item 7A"},
    ]),

    _synthetic_profile("NFLX", "Netflix Inc.", 2023, {
        "Revenues":             (0.07,  "moderate_growth"),
        "NetIncome":            (2.90,  "strong_growth"),
        "Cash":                 (0.31,  "strong_growth"),
        "Assets":               (0.08,  "moderate_growth"),
        "LongTermDebt":         (-0.07, "stable"),
        "OperatingCashFlow":    (1.43,  "strong_growth"),
        "ResearchAndDevelopment":(0.06, "moderate_growth"),
        "GrossProfit":          (0.14,  "moderate_growth"),
    }, [
        {"signal_id": 1, "topic": "market_risk",   "signal_type": "intensifying_competition",
         "section": "Item 1A", "filing_year": 2023, "company": "Netflix Inc.",
         "summary": "Streaming market consolidation ongoing; competitors consolidating platforms may increase content bidding competition.",
         "severity": "medium", "citation": "Netflix Inc. 10-K 2023, Item 1A"},
        {"signal_id": 2, "topic": "strategic_risk","signal_type": "rd_productivity_risk",
         "section": "Item 7",  "filing_year": 2023, "company": "Netflix Inc.",
         "summary": "Content spend efficiency improving but live events (sports) expansion carries significant upfront cost risk.",
         "severity": "low", "citation": "Netflix Inc. 10-K 2023, Item 7"},
    ]),
]


def build_all() -> None:
    CURATOR_DB.mkdir(exist_ok=True)
    count = 0

    # 1. Real companies from extraction outputs
    for company_dir in sorted(EXTRACTION_ROOT.iterdir()):
        if not company_dir.is_dir():
            continue
        for extraction_file in sorted(company_dir.glob("*_extraction.json")):
            curator = _from_extraction(extraction_file)
            ticker = curator["ticker"].lower()
            year = curator["filing_year"]
            out_path = CURATOR_DB / f"{ticker}_{year}.json"
            out_path.write_text(json.dumps(curator, indent=2), encoding="utf-8")
            print(f"  [real]      {out_path.name}  "
                  f"({len(curator['financial_deltas'])} metrics, "
                  f"{len(curator['risk_signals'])} signals)")
            count += 1

    # 2. Synthetic profiles
    for profile in _SYNTHETIC_PROFILES:
        ticker = profile["ticker"].lower()
        year = profile["filing_year"]
        out_path = CURATOR_DB / f"{ticker}_{year}.json"
        out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(f"  [synthetic] {out_path.name}  "
              f"({len(profile['financial_deltas'])} metrics, "
              f"{len(profile['risk_signals'])} signals)")
        count += 1

    print(f"\nTotal curator JSONs written: {count}")


if __name__ == "__main__":
    build_all()
