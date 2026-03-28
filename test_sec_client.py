"""
Simple test script for SECClient.
Run with: python test_sec_client.py
"""

from sec_client import SECClient


client = SECClient(user_email="sharan.giri.0276@gmail.com")

def check(label: str, condition: bool, detail: str  = ""):
    status = "PASS" if condition else "FAIL"
    if detail: 
        print(f"{status} | {detail}")


# Test 1 : look up CIK for tickert

print("\n[Test 1] get_cik_from_ticker")

cik = client.get_cik_from_ticker("AAPL")
check("Returns a string",         isinstance(cik, str),      f"got: {cik}")
check("Exactly 10 characters",    len(cik) == 10,            f"length: {len(cik)}")
check("Correct CIK for AAPL",     cik == "0000320193",       f"got: {cik}")
check("Starts with zeros",        cik.startswith("0"),       f"got: {cik}")

# lowercase ticker should also work
cik_lower = client.get_cik_from_ticker("aapl")
check("Lowercase ticker works",   cik_lower == cik,          f"got: {cik_lower}")

# second call should hit in-memory cache
cik2 = client.get_cik_from_ticker("MSFT")
check("MSFT resolves correctly",  len(cik2) == 10,           f"got: {cik2}")

# invalid ticker should raise
try:
    client.get_cik_from_ticker("INVALIDTICKER999")
    check("Invalid ticker raises ValueError", False)
except ValueError as e:
    check("Invalid ticker raises ValueError", True, str(e))


# ── Test 2: Submissions ───────────────────────────────────────────────────────

print("\n[Test 2] get_submissions")

subs = client.get_submissions(cik)
check("Returns a dict",               isinstance(subs, dict))
check("Has 'name' field",             "name" in subs,             f"name: {subs.get('name')}")
check("Name contains Apple",          "Apple" in subs.get("name", ""))
check("Has 'filings' field",          "filings" in subs)
check("Has 'recent' filings",         "recent" in subs.get("filings", {}))

recent = subs["filings"]["recent"]
check("Recent has 'form' list",       "form" in recent)
check("Recent has 'accessionNumber'", "accessionNumber" in recent)
check("Recent has 'primaryDocument'", "primaryDocument" in recent)
check("Recent has 'filingDate'",      "filingDate" in recent)
check("Has multiple filings",         len(recent["form"]) > 10,
      f"filing count: {len(recent['form'])}")


# ── Test 3: Company Facts ─────────────────────────────────────────────────────

print("\n[Test 3] get_company_facts")

facts = client.get_company_facts(cik)
check("Returns a dict",           isinstance(facts, dict))
check("Has 'facts' key",          "facts" in facts)
check("Has 'us-gaap' taxonomy",   "us-gaap" in facts.get("facts", {}))

gaap = facts["facts"]["us-gaap"]
check("Has Assets tag",           "Assets" in gaap)
check("Has multiple tags",        len(gaap) > 50,   f"tag count: {len(gaap)}")

# Check a known tag has actual values
assets_units = gaap["Assets"]["units"]
check("Assets has USD values",    "USD" in assets_units)
check("Assets has entries",       len(assets_units.get("USD", [])) > 0,
      f"entry count: {len(assets_units.get('USD', []))}")


# ── Test 4: Company Concept ───────────────────────────────────────────────────

print("\n[Test 4] get_company_concept")

concept = client.get_company_concept(cik, "Assets")
check("Returns a dict",           isinstance(concept, dict))
check("Has 'tag' field",          "tag" in concept,    f"tag: {concept.get('tag')}")
check("Tag matches requested",    concept.get("tag") == "Assets")
check("Has 'units' field",        "units" in concept)
check("Has USD values",           "USD" in concept.get("units", {}))

entries = concept["units"].get("USD", [])
check("Has multiple year entries", len(entries) > 5,   f"entry count: {len(entries)}")

# Check structure of a single entry
if entries:
    e = entries[0]
    check("Entry has 'val'",      "val" in e,    f"val: {e.get('val')}")
    check("Entry has 'end'",      "end" in e,    f"end: {e.get('end')}")
    check("Entry has 'form'",     "form" in e,   f"form: {e.get('form')}")


# ── Test 5: Cache behavior ────────────────────────────────────────────────────

print("\n[Test 5] Cache behavior")

# In-memory cache should be populated after Test 1
check("In-memory cache populated",  client._ticker_cache is not None)
check("Cache is a dict",            isinstance(client._ticker_cache, dict))
check("Cache has AAPL key",         "AAPL" in client._ticker_cache)
check("Cache value is CIK string",  client._ticker_cache.get("AAPL") == "0000320193")

# Disk cache file should exist
from pathlib import Path
cache_path = Path(".cache/company_tickers.json")
check("Disk cache file exists",     cache_path.exists(),  str(cache_path))

# Cache file should have saved_at timestamp
if cache_path.exists():
    import json
    with open(cache_path) as f:
        cached = json.load(f)
    check("Cache file has saved_at",  "saved_at" in cached)
    check("Cache file has data",      "data" in cached)
    print(f"         saved_at: {cached.get('saved_at')}")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n─────────────────────────────────────")
print("All tests complete.")
print("If any show FAIL, fix before moving to document_fetcher.py")
print("─────────────────────────────────────\n")


