
from typing import Optional
from pydantic import BaseModel


class TenKDocument(BaseModel):
    """Represents a single downloaded 10-K or 10-K/A filing."""
    form:        str            # "10-K" or "10-K/A"
    filing_date: str            # "YYYY-MM-DD"
    accession:   str            # "0000320193-23-000106"
    url:         str            # full SEC archive URL
    html:        str            # raw HTML text of the filing


class FinancialDataPoint(BaseModel):
    """A single annual value for one financial metric."""
    year:      Optional[int]    # e.g. 2023
    end_date:  Optional[str]    # "YYYY-MM-DD"
    value:     Optional[float]  # raw dollar value e.g. 394328000000
    tag:       str              # actual XBRL tag used
    accession: Optional[str]    # links back to the source filing


class QuarterlyDataPoint(BaseModel):
    """A single quarterly value for one financial metric."""
    year:      Optional[int]    # e.g. 2023
    quarter:   Optional[int]    # 1, 2, 3, or 4
    end_date:  Optional[str]    # "YYYY-MM-DD"
    value:     Optional[float]  # raw dollar value
    tag:       str              # actual XBRL tag used
    accession: Optional[str]    # links back to the source filing