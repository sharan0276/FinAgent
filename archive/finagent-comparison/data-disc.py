import requests
import pandas as pd

HEADERS = {
    "User-Agent": "yourname youremail@example.com"
}

CIK = "0000320193"

# 1) Get Apple submissions history
submissions_url = f"https://data.sec.gov/submissions/CIK{CIK}.json"
submissions = requests.get(submissions_url, headers=HEADERS, timeout=30)
submissions.raise_for_status()
submissions_json = submissions.json()

recent = pd.DataFrame(submissions_json["filings"]["recent"])

# Filter Apple 2025 10-K
apple_2025_10k = recent[
    (recent["form"] == "10-K") &
    (recent["filingDate"] == "2025-10-31")
]

print(apple_2025_10k[[
    "accessionNumber"
]])
print(apple_2025_10k[["filingDate"]])
print(apple_2025_10k[["reportDate"]])
print(apple_2025_10k[["primaryDocument"]])
print(apple_2025_10k[["primaryDocDescription"]])

accession = "0000320193-25-000079"
primary_doc = "aapl-20250927.htm"

accession_nodash = accession.replace("-", "")
filing_url = f"https://www.sec.gov/Archives/edgar/data/320193/{accession_nodash}/{primary_doc}"

print(filing_url)