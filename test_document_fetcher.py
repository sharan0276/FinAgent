"""
Simple test script for document fetcher.
Run with: python test_document_fetcher.py
"""

from sec_client import SECClient
from document_fetcher import DocumentFetcher

client = SECClient(user_email="sharan.giri.0276@gmail.com")
document_fetcher = DocumentFetcher(client)

def check(label: str, condition: bool, detail: str  = ""):
    status = "PASS" if condition else "FAIL"
    if detail: 
        print(f"{label} : {status} | {detail}")


# Fetch submissions once — reused across all tests
print("\nFetching AAPL submissions for tests...")
cik         = client.get_cik_from_ticker("AAPL")
submissions = client.get_submissions(cik)
print(f"CIK: {cik} | Company: {submissions.get('name')}\n")



# ── Test 1: extract_10k_filings ───────────────────────────────────────────────

print("[Test 1] _extract_10k_filings")

filings = document_fetcher._extract_10k_filings(submissions, cik)

check("Returns a list",                   isinstance(filings, list))
check("At least 5 years of filings",      len(filings) >= 5,
      f"found: {len(filings)} filings")

# Check structure of each filing dict
if filings:
    f = filings[0]
    check("Each filing has 'form'",        "form" in f,         f"form: {f.get('form')}")
    check("Each filing has 'accession'",   "accession" in f,    f"accession: {f.get('accession')}")
    check("Each filing has 'primaryDoc'",  "primaryDoc" in f,   f"doc: {f.get('primaryDoc')}")
    check("Each filing has 'filingDate'",  "filingDate" in f,   f"date: {f.get('filingDate')}")

# Check only 10-K and 10-K/A forms present
forms_found = set(f["form"] for f in filings)
check("Only 10-K forms present",          forms_found.issubset({"10-K", "10-K/A"}),
      f"forms found: {forms_found}")

# Check sorted newest first
dates = [f["filingDate"] for f in filings]
check("Sorted newest first",              dates == sorted(dates, reverse=True),
      f"dates: {dates[:3]}...")

# Check one filing per year — no duplicate years
years = [f["filingDate"][:4] for f in filings]
check("One filing per year (no dupes)",   len(years) == len(set(years)),
      f"years: {years}")

# Check issuer filter — all accessions start with company CIK
cik_no_zeros = cik.lstrip("0")
for f in filings:
    acc_prefix = f["accession"].replace("-", "")[:10].lstrip("0")
    if acc_prefix != cik_no_zeros:
        check("Issuer filter working", False, f"foreign accession: {f['accession']}")
        break
else:
    check("Issuer filter working", True)

# Print summary of filings found
print(f"\n  Filings found:")
for f in filings[:5]:
    print(f"    {f['form']} | {f['filingDate']} | {f['primaryDoc']}")


# ── Test 2: build_doc_url ─────────────────────────────────────────────────────

print("\n[Test 2] _build_doc_url")

latest  = filings[0]
url     = document_fetcher._build_doc_url(cik, latest["accession"], latest["primaryDoc"])

check("Returns a string",                 isinstance(url, str))
check("Starts with SEC archive URL",      url.startswith("https://www.sec.gov/Archives/edgar/data/"))
check("CIK has no leading zeros in URL",  f"/{cik.lstrip('0')}/" in url,
      f"url: {url}")
check("Accession has no hyphens in URL",  latest["accession"].replace("-", "") in url,
      f"accession in url: {latest['accession'].replace('-', '')}")
check("No styling prefix in filename",    "xsl" not in url.split("/")[-1],
      f"filename: {url.split('/')[-1]}")
check("URL ends with .htm or .html",      url.endswith(".htm") or url.endswith(".html"),
      f"url: {url}")

print(f"\n  Built URL:\n  {url}")

# Test prefix stripping specifically
prefixed_doc = f"xslF345X05/{latest['primaryDoc']}"
url_stripped = document_fetcher._build_doc_url(cik, latest["accession"], prefixed_doc)
check("Strips xsl prefix correctly",      url == url_stripped,
      f"stripped: {url_stripped.split('/')[-1]}")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Slow test (actual HTML download — runs once)
# ════════════════════════════════════════════════════════════════════════════

print("\n[Test 3] get_latest_10k  (downloading 5 x 10-K...)")
print("  This may take 30-60 seconds...\n")

results = document_fetcher.get_latest_10k(submissions, cik, n_years=5)

check("Returns a list",                   isinstance(results, list))
check("Returns 5 filings",                len(results) == 5,
      f"got: {len(results)} filings")

for i, result in enumerate(results):
    year = result.get("filingDate", "")[:4]
    check(f"Filing {i+1} ({year}) has all fields",
          all(k in result for k in ("form", "filingDate", "accession", "url", "html")))
    check(f"Filing {i+1} ({year}) HTML not empty",
          len(result.get("html", "")) > 1000,
          f"html length: {len(result.get('html', ''))}")
    check(f"Filing {i+1} ({year}) looks like HTML",
          "<html" in result.get("html", "").lower()
          or "<!doctype" in result.get("html", "").lower())

# Check sorted newest first
dates = [r["filingDate"] for r in results]
check("Sorted newest first",              dates == sorted(dates, reverse=True),
      f"dates: {dates}")

print(f"\n  Downloaded filings:")
for r in results:
    print(f"    {r['form']} | {r['filingDate']} | {len(r['html'])//1024} KB")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n─────────────────────────────────────")
print("All tests complete.")
print("If any show FAIL, fix before moving to data_cleaner.py")
print("─────────────────────────────────────\n")