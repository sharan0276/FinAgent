# curator_models.py
# Pydantic models for Agent 2 (Curator) output.
# Every signal Claude returns gets validated against these before saving.
# If validation fails, the retry logic in curator_agent.py triggers.

from pydantic import BaseModel, field_validator
from typing import Optional
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────
# These are the only valid values for topic, severity, and delta label.
# Pydantic will reject anything outside these sets automatically.

class Topic(str, Enum):
    liquidity_risk    = "liquidity_risk"
    market_risk       = "market_risk"
    legal_risk        = "legal_risk"
    operational_risk  = "operational_risk"
    macro_risk        = "macro_risk"
    strategic_risk    = "strategic_risk"


class Severity(str, Enum):
    high   = "high"
    medium = "medium"
    low    = "low"


class DeltaLabel(str, Enum):
    strong_growth     = "strong_growth"      # above +20%
    moderate_growth   = "moderate_growth"    # +5% to +20%
    stable            = "stable"             # -5% to +5%
    moderate_decline  = "moderate_decline"   # -20% to -5%
    severe_decline    = "severe_decline"     # below -20%
    insufficient_data = "insufficient_data"  # null delta (e.g. first year)


# ── Valid signal types ────────────────────────────────────────────────────────
# Full taxonomy from the master doc.
# Stored as a plain set — used in the validator below.
VALID_SIGNAL_TYPES = {
    # Liquidity
    "cash_runway_concern", "going_concern_warning", "debt_covenant_risk",
    "refinancing_risk", "negative_operating_cash_flow", "working_capital_deficit",
    "dividend_buyback_suspension", "impairment_charges", "revenue_concentration_risk",
    "accounts_receivable_deterioration",
    # Market
    "intensifying_competition", "product_obsolescence", "platform_dependency",
    "customer_churn_risk", "pricing_power_erosion", "market_saturation",
    "network_effect_weakening", "advertiser_concentration", "subscription_model_risk",
    "geographic_concentration",
    # Legal
    "active_litigation", "regulatory_investigation", "antitrust_exposure",
    "data_privacy_violation", "cybersecurity_incident", "export_control_risk",
    "ip_infringement_claim", "content_moderation_liability",
    "environmental_compliance", "government_contract_risk",
    # Operational
    "key_person_dependency", "talent_retention_risk", "supply_chain_concentration",
    "infrastructure_single_point_failure", "cybersecurity_vulnerability",
    "third_party_vendor_dependency", "manufacturing_defect_risk", "scaling_risk",
    "disaster_recovery_gaps", "internal_controls_weakness",
    # Macro
    "interest_rate_sensitivity", "foreign_currency_exposure", "inflation_impact",
    "recession_sensitivity", "china_geopolitical_exposure", "trade_tariff_risk",
    "sanctions_exposure", "sovereign_debt_risk",
    # Strategic
    "failed_acquisition", "new_market_entry_failure", "rd_productivity_risk",
    "moonshot_unproven_bet", "business_model_transition_risk",
    "customer_adoption_lag", "monetization_uncertainty",
    "strategic_partnership_dependency", "competitive_moat_erosion",
    "capital_allocation_risk",
}


# ── Models ────────────────────────────────────────────────────────────────────

class RiskSignal(BaseModel):
    """
    One extracted risk signal from a 10-K section.
    Claude must return an array of these. Each field is validated.
    """
    signal_id:   int
    topic:       Topic       # must be one of the 6 valid topics
    signal_type: str         # validated below against VALID_SIGNAL_TYPES
    section:     str         # e.g. "Item 1A", "Item 7"
    filing_year: int
    company:     str
    summary:     str         # 1-2 sentence plain English description
    severity:    Severity    # high / medium / low
    citation:    str         # e.g. "AAPL 10-K 2021, Item 1A"

    @field_validator("topic", mode="before")
    @classmethod
    def map_signal_to_topic(cls, v: str) -> str:
        """
        Sometimes the AI puts a specific signal_type into the broad topic field.
        This maps common mixups back to their correct parent topic automatically.
        """
        mapping = {
            "capital_allocation_risk": "strategic_risk",
            "cash_runway_concern": "liquidity_risk",
            "antitrust_exposure": "legal_risk",
            # Add more aliases here as discovered
        }
        return mapping.get(v, v)

    @field_validator("signal_type")
    @classmethod
    def validate_signal_type(cls, v: str) -> str:
        if v not in VALID_SIGNAL_TYPES:
            raise ValueError(
                f"'{v}' is not a valid signal_type. "
                f"Must be one of: {sorted(VALID_SIGNAL_TYPES)}"
            )
        return v

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary cannot be empty")
        return v.strip()


class FinancialDelta(BaseModel):
    """
    One financial metric with its year-over-year delta and severity label.
    delta_percent is None for the earliest year (no prior year to compare).
    """
    value: Optional[float]   # the raw delta as a decimal, e.g. -0.23 = -23%
    label: DeltaLabel        # severity bucket computed from value


class CuratorOutput(BaseModel):
    """
    The complete output file Agent 2 writes for one company for one year.
    File name: {ticker_lowercase}_{year}.json
    """
    company:           str
    ticker:            str
    filing_year:       int
    financial_deltas:  dict[str, FinancialDelta]  # keyed by metric name
    risk_signals:      list[RiskSignal]
    embedding_text:    str                         # serialized text for Agent 3
    embedding_vector:  list[float]                 # 1536 floats from OpenAI