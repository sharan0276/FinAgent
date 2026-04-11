"""
Navigates SEC EDGAR filing directories and downloads 10-K HTML documents.
Injects SECClient to reuse its session, headers, and rate limiting.
"""

import re
import time
from typing import Optional

from sec_client import SECClient


class DocumentFetcher:
    """
    Downloads the actual 10-K HTML document for a given company.

    Handles:
      - Filtering submissions to 10-K and 10-K/A filings only
      - Issuer filter (removes insider/third-party filings)
      - Deduplication by fiscal year (amendment wins over original)
      - URL construction from accession number and document name
      - Raw HTML download via injected SECClient session

    Usage:
        client  = SECClient()
        fetcher = DocumentFetcher(client)
        result  = fetcher.get_latest_10k(submissions, cik)
    """

    def __init__(self, client: SECClient):
        # Reuse the same session, headers, and rate limiting from SECClient
        # No new session created — one connection for the entire agent
        self.client = client

    def _combine_submission_recent_blocks(self, submissions: dict) -> dict:
        """
        Merge the base `filings.recent` block with paginated SEC submissions
        history files so older 10-K filings remain discoverable.
        """
        combined = {
            key: list(values)
            for key, values in submissions.get("filings", {}).get("recent", {}).items()
        }

        for file_info in submissions.get("filings", {}).get("files", []):
            name = file_info.get("name")
            if not name:
                continue

            extra_recent = self.client._get(f"https://data.sec.gov/submissions/{name}")
            for key, values in extra_recent.items():
                combined.setdefault(key, [])
                combined[key].extend(values)

        return combined

    
    # Step 1: Filter and deduplicate filings
    
    def _extract_10k_filings(self, submissions: dict, cik: str) -> list[dict]:
        """
        Filter submissions down to one 10-K per fiscal year.

        - Keeps only 10-K and 10-K/A forms
        - Removes third-party filings via issuer filter
        - SEC returns newest first so first filing seen per year is
          always the most correct (amendment if one exists)

        Returns list sorted newest first:
            [{ "form", "accession", "primaryDoc", "filingDate" }, ...]
        """
        recent = self._combine_submission_recent_blocks(submissions)
        forms      = recent["form"]
        accessions = recent["accessionNumber"]
        docs       = recent["primaryDocument"]
        dates      = recent["filingDate"]

        cik_no_zeros = cik.lstrip("0")

        filings = []
        for form, acc, doc, date in zip(forms, accessions, docs, dates):
            if form not in ("10-K", "10-K/A"):
                continue

            # Issuer filter: accession prefix must match the company's own CIK
            # removes Form 4s and other filings made about the company by others
            acc_prefix = acc.replace("-", "")[:10].lstrip("0")
            if acc_prefix != cik_no_zeros:
                continue

            filings.append({
                "form": form,
                "accession": acc,
                "primaryDoc": doc,
                "filingDate": date,
            })

        # Deduplicate by fiscal year — SEC returns newest first so first
        # filing seen per year is already the most recent and most correct
        seen: dict[str, dict] = {}
        for f in filings:
            year = f["filingDate"][:4]
            if year not in seen:
                seen[year] = f

        return sorted(seen.values(), key=lambda x: x["filingDate"], reverse=True)


    # Step 2: Build the document URL
    
    def _build_doc_url(self, cik: str, accession: str, primary_doc: str) -> str:
        """
        Construct the direct URL to a filing's HTML document.

        URL formula:
            ARCHIVE_BASE / CIK(no leading zeros) / accession(no hyphens) / filename

        Also strips SEC styling prefixes e.g. "xslF345X05/" from filenames
        to get the raw machine-readable file instead of the styled viewer.
        """
        cik_no_zeros   = cik.lstrip("0")
        acc_no_hyphens = accession.replace("-", "")

        # Strip any prefix path like "xslF345X05/" from the document name
        clean_doc = re.sub(r"^[^/]+/", "", primary_doc)

        return (
            f"https://www.sec.gov/Archives/edgar/data"
            f"/{cik_no_zeros}/{acc_no_hyphens}/{clean_doc}"
        )


    #Step 3: Download the HTML

    def _fetch_document_html(self, url: str) -> str:
        """
        Download the raw HTML text of a filing document.
        Reuses SECClient's session and rate limiting.
        Returns full HTML string (can be 1-5MB for a full 10-K).
        """
        time.sleep(self.client.request_delay)
        resp = self.client._session.get(url, timeout=self.client.request_timeout)
        resp.raise_for_status()
        return resp.text


    # Step 4: Orchestrator

    def get_latest_10k(self, submissions: dict, cik: str, n_years: int = 5) -> Optional[dict]:
        """
        Download the last n years of 10-K filings.

        Loops through the top n filings (one per fiscal year),
        downloads each HTML document, and returns them as a list
        sorted newest first.

        Returns:
        [
            {
                "form":        "10-K" or "10-K/A",
                "filingDate":  "YYYY-MM-DD",
                "accession":   "0000320193-23-000106",
                "url":         "https://www.sec.gov/Archives/...",
                "html":        "<full HTML text of the 10-K>"
            },
            up to n_years entries
        ]
        Returns empty list if no 10-K filings found.
        """
        filings = self._extract_10k_filings(submissions, cik)
        if not filings:
            return None

        latest = filings[:n_years]
        results = []

        for i, filing in enumerate(latest, start = 1):
            print(f"[DocumentFetcher] Downloading {i}/{len(latest)}: "
                  f"{filing['form']} {filing['filingDate']}")
            url    = self._build_doc_url(cik, filing["accession"], filing["primaryDoc"])
            html   = self._fetch_document_html(url)

            results.append({
                "form":        filing["form"],
                "filingDate":  filing["filingDate"],
                "accession":   filing["accession"],
                "url":         url,
                "html":        html
            })
        return results
