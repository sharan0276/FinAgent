# Data Ingestion Notes

## Goal

Build a reusable ingestion pipeline for any company ticker that combines:

- structured financial metrics from SEC XBRL APIs
- textual content from 10-K filings
- deterministic processing (no LLM required for ingestion)

---

## Current Architecture (Unified Flow)

The main entrypoint is:

- `data-ingestion/ingestion_pipeline.py`

For a ticker (example: `AAPL`), the pipeline does this:

1. Resolve ticker -> CIK
2. Run two tracks in parallel:
- Financial track (Company Facts API -> cleaned annual/quarterly metrics)
- Filing text track (Submissions -> latest 10-K HTML -> section extraction)
3. Merge both tracks into one consolidated JSON artifact

Output path:

- `data-ingestion/outputs/<TICKER>/complete_ingestion.json`

---

## Financial Track (Numbers)

Primary modules:

- `data-ingestion/sec_client.py`
- `data-ingestion/company_facts_cleaner.py`

### What it does

1. `SECClient` fetches SEC datasets:
- submissions
- company facts
- company concept (optional)
- ticker-to-CIK map with in-memory + disk cache

2. `DataCleaner` normalizes XBRL facts:
- XBRL tag fallback chains (handles tag changes over years)
- consolidated-only filtering (removes segment breakdown rows)
- annual vs quarterly filtering
- standardized output records

### Current metric aliases used

- `Revenues`
- `NetIncome`
- `Cash`
- `Assets`
- `LongTermDebt`
- `OperatingCashFlow`
- `ResearchAndDevelopment`
- `GrossProfit`

---

## Filing Text Track (10-K)

Primary modules:

- `data-ingestion/document_fetcher.py`
- `data-ingestion/sec_parser_utils.py`
- `data-ingestion/ingestion_pipeline.py`

### What it does

1. Filter to issuer-owned `10-K` / `10-K/A` filings
2. Deduplicate by filing year (latest/amended wins)
3. Download last N years of filing HTML
4. Extract core sections from filing text

### Parser behavior

- Preferred mode: `sec-parser` semantic parse path. It is highly accurate as it uses HTML structure bounds to cleanly extract section text.
- Fallback mode: plain-text extraction from HTML if parser dependencies are missing, the library crashes, or the filing is too old/unstructured.

### Recent Extraction Improvements (vs Old Code)
The extraction heuristics were updated to enforce strict boundaries and prevent data contamination:
1. **Curly Quote Normalization**: Filings often use HTML entity curly quotes (`&#8217;`). The old plain-text fallback failed to match regex strings like `management's` because of this. We introduced `html.unescape()` and text normalization to reliably match punctuation.
2. **Table of Contents Evasion**: The old plain-text logic searched for the first available instance of "Part I" to start extraction, which actually began extracting inside the Table of Contents. The new logic targets the *second* occurrence of "Part I", reliably jumping the TOC.
3. **Strict Generic Item Boundaries**: The old parsing logic (in both `sec_parser` and plain-text) only stopped recording when it hit the *next tracked section* (e.g. Item 1C swallowed Item 2 because Item 3 was the next tracked target). The new logic constantly scans ahead and stops recording the moment it hits **ANY** generic "Item X" heading, perfectly partitioning the data.

In output, each filing includes `parser_mode` so it is clear which mode was used.

---

## High-Signal Core Section Targets

We specifically target the following sections to maximize the extraction of actionable intelligence:

- **Item 1 (Operations):** Intellectual Property (IP) strategy and core business dependencies.
- **Item 1A (Threats):** AI disruption, competitive shifts, and talent attrition.
- **Item 1C (Security):** Reliance on third-party cloud/logistics and cyber-vulnerabilities.
- **Item 3 (Litigation):** Real-world IP disputes and regulatory enforcement actions.
- **Item 7 (Narrative):** Management's "why" behind revenue or margin changes.
- **Item 7A (Finance):** Sensitivity to currency, interest rate, and equity market swings.

## Data Models and Shared State

- `data-ingestion/models.py`
- `data-ingestion/state.py`

These define typed structures for filings and financial datapoints, and the shared pipeline state used by agent orchestration.

---

## Running the Pipeline

From project root:

```bash
python data-ingestion/ingestion_pipeline.py AAPL --years 5
```

Optional metric override:

```bash
python data-ingestion/ingestion_pipeline.py AAPL --metrics Revenues NetIncome Cash
```

---

## Tests

- `test_sec_client.py`
- `test_document_fetcher.py`

These currently serve as integration-style smoke tests for SEC connectivity, submissions parsing, URL construction, and multi-year 10-K download behavior.

---

## Dependencies

Root requirements are maintained in:

- `requirements.txt`

Current ingestion-relevant dependencies include:

- `requests`
- `python-dotenv`
- `pydantic`
- `sec-downloader`
- `sec-parser`
- `pypdf`

---

## Legacy Spike Artifacts

The Meta-specific deterministic spike scripts are still present:

- `data-ingestion/meta_sec_parser_spike.py`
- `data-ingestion/meta_sec_parser_validate.py`

And historical outputs under:

- `data-ingestion/outputs/meta_*`
- `data-ingestion/outputs/meta_sections/*`

These are useful references but are no longer the main ingestion entrypoint.

---

## Known Cleanup Items

1. Remove old root compatibility shim files once filesystem locks/permissions are fully resolved.
2. Decide whether to keep or archive legacy Meta spike scripts and outputs.
3. Improve section extraction quality for difficult filings and table-heavy regions.
4. Add stricter automated tests for unified `complete_ingestion.json` schema and content quality.
