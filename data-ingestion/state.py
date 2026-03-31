
from typing import Optional
from pydantic import BaseModel, Field
from models import TenKDocument, FinancialDataPoint, QuarterlyDataPoint


# ── Main pipeline state ───────────────────────────────────────────────────────

class FinAgentState(BaseModel):
    """
    Shared state flowing through the entire LangGraph pipeline.
    Each agent receives this, does its work, and returns an updated copy
    using state.model_copy(update={...}).
    """

    # ── Input ──────────────────────────────────────────────────────────────
    ticker: str                      # e.g. "AAPL" — only required input

    # ── Agent 1: Research Agent ─────────────────────────────────────────────
    cik:          Optional[str] = None
    company_name: Optional[str] = None

    # 10-K HTML documents — last 5 years
    filings: list[TenKDocument] = Field(default_factory=list)

    # Annual financial metrics — last 5 years
    financials: dict[str, list[FinancialDataPoint]] = Field(default_factory=dict)

    # Quarterly financial metrics — last 20 quarters (5 years * 4)
    quarterly_financials: dict[str, list[QuarterlyDataPoint]] = Field(default_factory=dict)

    # ── Agent 2: Extraction Agent ───────────────────────────────────────────
    # extracted_sections: dict[str, str]        = Field(default_factory=dict)
    # extracted_tables:   dict[str, list]       = Field(default_factory=dict)

    # ── Agent 3: Rule Engine ────────────────────────────────────────────────
    # risk_flags: list[str]                     = Field(default_factory=list)

    # ── Agent 4: Synthesis Agent ────────────────────────────────────────────
    # summary: Optional[str]                    = None

    # ── Pipeline-wide error log ─────────────────────────────────────────────
    errors: list[str] = Field(default_factory=list)

    def add_error(self, msg: str) -> "FinAgentState":
        """Return a new state with an error appended (immutable update)."""
        return self.model_copy(update={"errors": self.errors + [msg]})