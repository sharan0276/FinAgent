"""
Raw SEC EDGAR API calls - no business logic just clean fetching.
Fetches the Submissions, Company Facts and Companyu COncept Datasets.
"""

import time
import requests

import json
from pathlib import Path
from datetime import datetime, timezone

'''# Constants - API links
BASE_URL = "https://data.sec.gov/"
SUBMISSIONS_URL = f"{BASE_URL}/submissions/CIK{{cik}}.json"
COMPANY_FACTS_URL = f"{BASE_URL}/api/xbrl/companyfacts/CIK{{cik}}.json"
COMPANY_CONCEPT_URL = f"{BASE_URL}/api/xbrl/companyconcept/CIK{{cik}}/{{category}}/{{tag}}.json"

# Establishing Headers needed for SEC EDGAR
HEADERS = {
    "User-Agent": "FinAgentTeam9 your_university_email@example.com",  # ← update email
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# SEC rate limit: max 10 requests/second
REQUEST_DELAY = 0.11  # seconds between calls'''

class SECClient:
    """
    A reusable client for SEC EDGAR API calls.

    Handles:
      - User-Agent authentication header
      - Rate limiting (SEC max: 10 requests/second)
      - Connection reuse via requests.Session
      - All three EDGAR endpoints + CIK lookup

    Usage:
        client = SECClient()
        submissions = client.get_submissions("0000320193")
    """

    # SEC endpoint templates 
    _SUBMISSIONS_URL     = "https://data.sec.gov/submissions/CIK{cik}.json"
    _COMPANY_FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    _COMPANY_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
    _TICKERS_URL         = "https://www.sec.gov/files/company_tickers.json"
    


    def __init__(
        self,
        user_name: str = "FinAgentTeam9",
        user_email: str = "sharan.giri.0276@gmail.com",
        request_delay: float = 0.11,     # Rate Limit of 10 requests per second
        request_timeout: float = 10.0,   # Network timeout for SEC servers 
        cache_dir: str = ".cache",
        ticker_cache_ttl_hours: int = 24 # File update frequency is per day
    ):
        self.request_delay = request_delay
        self.request_timeout = request_timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok = True)
        self.ticker_cache_ttl_hours = ticker_cache_ttl_hours
        self._ticker_cache: dict | None = None

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": f"{user_name} {user_email}",
            "Accept-Encoding": "gzip, deflate"
        })


    def _get(self, url:str) -> dict:
        """
        Single rate-limited GET request.
        Raises requests.HTTPError on bad status codes (4xx, 5xx).
        """
        time.sleep(self.request_delay)
        resp = self._session.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        return resp.json()

    
    # Endpoint 1 : Submissions API Call (https://data.sec.gov/submissions/CIK{cik}.json)
    # The meta data for all the filings of a company is stored here 
    def get_submissions(self, cik:str) -> dict:
        """
        Fetch all filings metadata for a company.

        Returns the raw SEC submissions dict including:
          - entityName
          - filings.recent: form, accessionNumber, primaryDocument, filingDate

        Args:
            cik: Zero-padded 10-digit CIK string e.g. "0000320193"
        """
        url = self._SUBMISSIONS_URL.format(cik=cik)
        return self._get(url)

    
    # Endpoint 2 : Company Facts - The Data Dump (https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json)
    # This API call returns the entire XBRL taxonomy for a company. 


    def get_company_facts(self, cik: str) -> dict:
        """
        Fetch ALL structured XBRL facts ever reported by a company.

        Returns nested dict:
          facts → us-gaap / ifrs-full → tag → units → list of values

        Use when you need to explore what tags a company has used
        or pull multiple metrics at once.

        Args:
            cik: Zero-padded 10-digit CIK string
        """
        url = self._COMPANY_FACTS_URL.format(cik=cik)
        return self._get(url)

    
    # Endpoint 3 : Company Concept (https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{category}/{tag}.json)
    def get_company_concept(
        self,
        cik: str,
        tag: str,
        taxonomy: str = "us-gaap",
    ) -> dict:
        """
        Fetch the full history of ONE specific financial metric.
        Lightweight — use for targeted math (e.g. YoY revenue delta).

        Args:
            cik:      Zero-padded 10-digit CIK string
            tag:      XBRL tag e.g. "Cash", "Revenues", "Assets"
            taxonomy: "us-gaap" (default) or "ifrs-full" for foreign filers

        Example:
            client.get_company_concept("0000320193", "Cash")
        """
        url = self._COMPANY_CONCEPT_URL.format(
            cik=cik, taxonomy=taxonomy, tag=tag
        )
        return self._get(url)

    
    # Helper Functions to fetch the Ticker and CIK mapping

    def _build_ticker_index(self, data: dict) -> dict:
        """
        Convert raw SEC ticker list into a fast O(1) lookup dict.
        Built once when data is first loaded, reused for every lookup.

        Raw SEC format:  { "0": { "cik_str": 320193, "ticker": "AAPL", ... }, ... }
        Index format:    { "AAPL": "0000320193", "MSFT": "0000789019", ... }
        """
        return {
            entry["ticker"]: str(entry["cik_str"]).zfill(10)
            for entry in data.values()
        }
    def _ticker_cache_path(self) -> Path:
        return self.cache_dir / "company_tickers.json"
    

    def _is_cache_fresh(self) -> bool:
        cache_path = self._ticker_cache_path()
        if not cache_path.exists():
            return False
        
        # Read teh saved timestamp in the cache file
        with open(cache_path, "r") as f:
            cached = json.load(f)
        
        saved_time = datetime.fromisoformat(cached["saved_at"])
        age_hours = (datetime.now(timezone.utc) - saved_time).total_seconds() / 3600
        return age_hours < self.ticker_cache_ttl_hours
    
    
    def _save_ticker_cache(self, data: dict) -> None:
        """Save ticker data to disk with a timestamp."""
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        with open(self._ticker_cache_path(), "w") as f:
            json.dump(payload, f)
        print(f"[SECClient] Ticker cache saved to {self._ticker_cache_path()}")

    def _load_ticker_cache(self) -> dict:
        """Load ticker data from disk cache."""
        with open(self._ticker_cache_path()) as f:
            cached = json.load(f)
        print(f"[SECClient] Loaded ticker cache from disk (saved: {cached['saved_at']})")
        return cached["data"]

    def get_cik_from_ticker(self, ticker: str) -> str:
        """
        Resolve a stock ticker to a zero-padded 10-digit CIK.

        Cache hierarchy:
          1. In-memory cache  → instant, reused within same run
          2. Disk cache       → loaded if under 24hrs old, skips network call
          3. SEC network      → fresh download, saved to disk for next time

        Returns:
            Zero-padded CIK string e.g. "0000320193"

        Raises:
            ValueError if ticker not found
        """
        # Layer 1: already fetched this run — use in-memory
        if self._ticker_cache is not None:
            print(f"[SECClient] Using in-memory ticker cache")
            index = self._ticker_cache

        # Layer 2: disk cache is fresh — load from file
        elif self._is_cache_fresh():
            data = self._load_ticker_cache()
            index = self._build_ticker_index(data)
            self._ticker_cache = index   # promote to in-memory for this run

        # Layer 3: cache is stale or missing — download fresh
        else:
            print(f"[SECClient] Downloading fresh ticker list from SEC...")
            data = self._get(self._TICKERS_URL)
            self._save_ticker_cache(data)
            index = self._build_ticker_index(data)
            self._ticker_cache = index   # promote to in-memory for this run

        # Lookup for the ticker in the index
        ticker_upper = ticker.upper()
        cik = index.get(ticker_upper)
        if cik is None:
            raise ValueError(f"Ticker '{ticker}' not found in SEC company list.")
        return cik

        