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
- preferred unit selection for common numeric concepts
- fallback-chain merging across tags when companies switch XBRL tags over time
- consolidated-only filtering (removes segment breakdown rows)
- annual vs quarterly filtering
- annual deduplication to exactly one canonical datapoint per fiscal year
- latest-entry selection when multiple SEC facts exist for the same period
- discrete quarterly derivation for flow metrics from YTD SEC facts when needed
- quarter inference from `fp`, `frame`, and year-end snapshot fallback
- standardized output records

### Current financial output schema

The active ingestion pipeline currently writes financial metrics in a point-list schema that is directly compatible with `data-extraction/`:

- `financial_data.annual[metric] -> list[dict]`
- `financial_data.quarterly[metric] -> list[dict]`

Annual point shape:

- `year`
- `end_date`
- `value`
- `tag`
- `accession`

Quarterly point shape:

- `year`
- `quarter`
- `end_date`
- `value`
- `tag`
- `accession`

Important note:

- there was a brief compact-format branch using `years/values/deltas` arrays
- the current code has been rolled back to point lists to remain compatible with the active extraction pipeline
- if a company artifact was generated during the compact-format window, rerun ingestion before using it downstream

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
3. Merge paginated SEC submissions history so older annual filings remain discoverable
4. Download last N years of filing HTML
5. Extract core sections from filing text

### Parser behavior

- Preferred mode: `sec-parser` semantic parse path. It is highly accurate as it uses HTML structure bounds to cleanly extract section text.
- Fallback mode: plain-text extraction from HTML if parser dependencies are missing, the library crashes, or the filing is too old/unstructured.
- Runtime behavior is now per-filing: if `sec-parser` fails for a single filing, the pipeline falls back for that filing instead of failing the entire text ingestion pass.
- Quality-gate behavior is also per-filing: if `sec-parser` extraction looks too sparse or malformed, the pipeline compares it with the plain-text path and keeps the stronger result.

### Recent Extraction Improvements (vs Old Code)
The extraction heuristics were updated to enforce strict boundaries and prevent data contamination:
1. **Curly Quote Normalization**: Filings often use HTML entity curly quotes (`&#8217;`). The old plain-text fallback failed to match regex strings like `management's` because of this. We introduced `html.unescape()` and text normalization to reliably match punctuation.
2. **Table of Contents Evasion**: The old plain-text logic searched for the first available instance of "Part I" to start extraction, which actually began extracting inside the Table of Contents. The new logic targets the *second* occurrence of "Part I", reliably jumping the TOC.
3. **Explicit Stop Boundaries**: The extraction path now defines exact stop headings for each tracked section. Example: `Item 1 -> Item 1A`, `Item 1A -> Item 1B/1C`, `Item 1C -> Item 2`, `Item 7 -> Item 7A`, `Item 7A -> Item 8`.
4. **Generic Item Fallback Boundaries**: If an explicit stop heading is missing or malformed in a filing, the parser still falls back to the next generic `Item X` heading to avoid overflow into neighboring sections.
5. **Line-Based Heading Matching**: In the plain-text path, section starts and stops are only allowed to trigger when the heading appears on its own line. References like `see Item 7A ...` inside a paragraph should not split sections.
6. **Case-Insensitive Heading Matching**: Section start and stop matching explicitly handles all-caps filings such as `ITEM 1A. RISK FACTORS` and `ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK`.

In output, each filing includes `parser_mode` so it is clear which mode was used.

### Current Data Quality Guarantees

The current ingestion code is designed to guarantee the following:

- annual numeric output should contain at most one datapoint per fiscal year for each metric
- quarterly output should be internally consistent enough to compare across filings, even when SEC provides a mix of discrete-quarter and YTD values
- text ingestion requests the last `N` years of filings when run with `--years N`
- extracted section text should stop at the intended next section boundary rather than swallowing downstream content
- fresh ingestion artifacts should have a financial schema that can be read directly by `data-extraction/numeric_delta.py`

### Current Working Reference Artifacts

The following generated artifacts are the current working reference set and should be treated as the source of truth for downstream experimentation unless explicitly regenerated:

- `data-ingestion/outputs/AAPL/complete_ingestion.json`
- `data-ingestion/outputs/META/complete_ingestion.json`
- `data-ingestion/outputs/GOOG/complete_ingestion.json`

Interpretation:

- these artifacts are considered sufficiently consistent for the next project phase
- they are inspectable and deterministic, even if some fields remain imperfect
- `metric_errors` and empty section strings should be interpreted as explicit missing-data signals rather than silent failures

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
3. Improve section extraction quality for difficult filings and table-heavy regions, especially when `sec-parser` and plain-text extraction disagree.
4. Revisit fiscal-quarter labeling consistency for non-calendar fiscal years if downstream comparison requires stricter quarter semantics.
5. Add stricter automated tests for unified `complete_ingestion.json` schema and content quality.
