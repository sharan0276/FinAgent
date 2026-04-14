# Project Handoff For Future LLM Context

## Purpose of This Document

This document is meant to give a future LLM enough context to understand:

- what this project is trying to build
- what has been implemented so far
- which files are responsible for which parts of the system
- what decisions have already been made
- what is still incomplete
- where the project is headed next

The goal is that a future LLM can read this file and quickly become useful without having to rediscover the project structure from scratch.

## High-Level Project Goal

This project is building a company financial analyzer focused on SEC EDGAR `10-K` filings.

The intended long-term workflow is:

1. Ingest financial and textual data for a set of companies from SEC EDGAR.
2. Store structured and textual representations in a form that supports comparison.
3. When a user asks for analysis of a new company, ingest that company in the same way.
4. Compare the new company against similar companies already stored in the system.
5. Produce a financial-health or risk-oriented analysis.

## Current Scope

The current implemented scope is narrower than the full vision.

What is implemented now:

- deterministic SEC ingestion
- ticker to CIK resolution
- SEC submissions retrieval
- SEC company facts retrieval
- cleaning of raw XBRL financial data
- fetching of recent `10-K` filings
- parsing of filing HTML with `sec-parser`
- extraction of selected core text sections
- generation of a combined ingestion artifact in JSON
- yearly extraction artifacts with FinBERT-ranked candidates and numeric deltas
- curator artifacts with company-year embeddings
- a standalone FAISS-based company matcher over curator artifacts
- a new `orchestration/` layer that resumes missing target-company artifacts, runs company matching, assembles peer context, and calls a comparison agent through OpenRouter
- a baseline RAG comparison path that works directly from ingestion artifacts for side-by-side comparison against the agentic flow
- a standalone `evaluation/` package that scores saved agentic and baseline report artifacts offline
- a local Streamlit app that can run both pipelines, compare them side by side, and manage dataset / FAISS operations

Current practical status:

- the generated `complete_ingestion.json` artifacts are now the working source of truth for downstream analysis
- the repo now contains a broader active working set beyond the original `AAPL`, `META`, and `GOOG` references, including newer ingested and orchestrated outputs for additional tickers

What is not yet implemented in the current broader system:

- a mature analyst-style financial-health reasoning layer on top of matches
- reliable structured table export from `Item 8`
- robust production-grade section extraction for every edge case filing
- a fully polished end-to-end path from ingestion through retrieval to final report generation

## Design Principle So Far

The current ingestion pipeline is intentionally deterministic.

This means:

- no LLM is used for parsing the SEC filing
- no LLM is used for validation
- parsing and validation are based on code, rules, regex, and library behavior

This decision was made to ensure that:

- ingestion is reproducible
- bugs are easier to diagnose
- outputs can be verified directly against source filings
- LLM usage is postponed until the source data is trustworthy

## Current System Structure

The current codebase is split across six main implemented layers:

1. `data-ingestion/`
2. `data-extraction/`
3. `rag-matching/`
4. `orchestration/`
5. `evaluation/`
6. `ui/`

The major logical parts are:

1. SEC API access
2. XBRL fact cleaning
3. 10-K document retrieval
4. 10-K semantic parsing
5. section extraction
6. unified ingestion output generation
7. yearly extraction artifacts
8. curator company-year artifacts with embeddings
9. nearest-neighbor company retrieval over curator artifacts
10. orchestration and comparison-bundle generation
11. offline report evaluation
12. local comparison UI and operator workflow

## File Responsibilities

### `data-ingestion/ingestion_pipeline.py`

This is the current main entrypoint for the ingestion flow.

Its responsibilities are:

- accept a ticker and optional arguments from the command line
- resolve the ticker to a CIK using `SECClient`
- launch two ingestion tracks in parallel
- merge both tracks into one final JSON object
- save the final result to `data-ingestion/outputs/<TICKER>/complete_ingestion.json`

It runs two parallel tracks:

- financial track
- text track

It also contains helper functions for:

- flattening the parsed semantic tree
- detecting section heading anchors
- extracting core sections from the parsed tree
- falling back to plain-text extraction if `sec-parser` is unavailable

### `data-ingestion/sec_client.py`

This file is the low-level SEC API client.

Its responsibilities are:

- create and maintain a reusable `requests.Session`
- apply SEC-compliant headers
- enforce simple SEC rate limiting
- fetch submissions data
- fetch company facts data
- fetch single company concept data
- resolve ticker symbols to CIK values
- cache the SEC ticker-to-CIK map on disk and in memory

This file deliberately avoids business logic and acts as the transport layer for SEC data.

### `data-ingestion/company_facts_cleaner.py`

This file cleans and normalizes the SEC Company Facts XBRL data.

Its responsibilities are:

- handle tag fallback chains for common financial metrics
- prefer the correct SEC unit set before normalizing values
- merge fallback-tag coverage across years when companies change tags over time
- remove segment-level rows and keep consolidated values
- separate annual values from quarterly values
- deduplicate raw SEC entries
- keep one canonical annual datapoint per fiscal year
- derive discrete quarterly values for flow metrics when SEC reports YTD facts
- infer quarter labels from filing metadata when possible
- format the cleaned metrics into simple standardized records

It is responsible for the "numbers" side of ingestion.

Important detail:

- the active saved output shape for financial metrics is the compact parallel-array form
- annual outputs are expected to look like `years`, `values`, `deltas`, `unit`, and `tag`
- quarterly outputs are expected to look like `periods`, `values`, `deltas`, `unit`, and `tag`
- downstream extraction code is already aligned to this shape

### `data-ingestion/document_fetcher.py`

This file handles the `10-K` retrieval process.

Its responsibilities are:

- filter submissions to `10-K` and `10-K/A`
- apply issuer filtering so the filing belongs to the company itself
- merge paginated SEC submissions history files when `filings.recent` is not deep enough
- deduplicate filings by year
- build the SEC archive document URL
- download the raw filing HTML
- return the latest `n` years of filing documents

It is responsible for the "get the actual filing HTML" part of the text pipeline.

### `data-ingestion/sec_parser_utils.py`

This file contains the reusable parsing utilities built around `sec-parser`.

Its responsibilities are:

- load SEC identity from environment variables
- fetch filing HTML using `sec-downloader` when needed
- configure `sec-parser`
- implement the chosen parser strategy for `10-K` parsing
- parse filing HTML into semantic elements and a semantic tree

Important detail:

The installed `sec-parser` version does not provide a dedicated `Edgar10KParser`.
Instead, the current code uses `Edgar10QParser` with a modified pipeline that removes 10-Q-specific section logic. This choice came from the library documentation and testing done earlier in the project.

### `data-ingestion/models.py`

This file defines typed data models for ingestion artifacts.

Examples:

- `TenKDocument`
- `FinancialDataPoint`
- `QuarterlyDataPoint`

These models make the intended shapes of the ingestion data clearer and help future pipeline work.

### `data-ingestion/state.py`

This file defines `FinAgentState`, which appears to be intended for a future multi-agent or LangGraph-style orchestration layer.

It captures shared state such as:

- ticker
- cik
- company name
- downloaded filings
- annual financials
- quarterly financials
- error tracking

This suggests the system is expected to evolve beyond the current standalone ingestion script into a larger multi-stage workflow.

### `data-ingestion/company_facts_cleaner.py`

This file converts raw Company Facts into cleaned metrics for the unified output artifact.

Although conceptually simple, it is important because SEC XBRL is inconsistent across companies and over time. This file is the main normalization layer for that inconsistency.

### `data-ingestion/outputs/<TICKER>/complete_ingestion.json`

This is not source code, but it is the main product of the current ingestion pipeline.

It is the closest thing to a canonical ingestion artifact right now.

It combines:

- company identity
- selected financial metrics
- parsed filing metadata
- extracted filing sections

At the current stage of the project, these generated artifacts should be treated as the source of truth for future analysis work unless a newer ingestion run intentionally replaces them.

Example currently open in the IDE:

- `data-ingestion/outputs/AAPL/complete_ingestion.json`

### `data-extraction/outputs/<TICKER>/<ticker>_<year>_extraction.json`

This is the yearly extraction artifact produced from `complete_ingestion.json`.

It combines:

- ranked text candidates from selected sections
- annual numeric deltas
- filing metadata

This is the current intermediate artifact between deterministic ingestion and curator generation.

### `data-extraction/outputs/curator/<TICKER>/<ticker>_<year>.json`

This is the current retrieval-ready company-year artifact.

It combines:

- financial delta labels
- curated risk signals
- `embedding_text`
- `embedding_vector`

These files are the canonical input to the active `rag-matching/` retrieval layer.

### `rag-matching/indexer.py` and `rag-matching/matcher.py`

These files implement the current standalone matching flow.

Responsibilities:

- recursively scan `data-extraction/outputs/curator/`
- build a persistent FAISS index under `rag-matching/index_artifacts/`
- accept a single curator JSON file as query input
- retrieve nearest neighbors by cosine similarity
- exclude the query company itself
- deduplicate repeated years using `best year wins`
- return the top distinct matching companies

Important detail:

- this matching layer uses curator embeddings and FAISS
- it does **not** use the older Chroma demo directories
- it assumes curator files already contain valid stored embedding vectors

### `orchestration/orchestration_pipeline.py`, `orchestration/runner.py`, `orchestration/comparison_agent.py`, `orchestration/report_models.py`, and `baseline_rag/pipeline.py`

These files implement the current top-level comparison flow.

Responsibilities:

- accept a user ticker through `orchestration/runner.py`
- check whether ingestion, extraction, and curator artifacts for the target company already exist
- run only the missing target-company steps
- call the FAISS matcher to retrieve the top distinct peer companies
- expand each matched company into `matched year + up to 2 future curator years`
- assemble one structured comparison bundle under `orchestration/outputs/<TICKER>/`
- call a final comparison agent through OpenRouter
- save the returned report content together with the bundle
- provide a separate baseline comparison flow that skips extraction/curation reasoning depth and works from ingested financial data plus peer retrieval

Important detail:

- this is intentionally a hybrid layer, not a planner-style agent runtime
- orchestration is deterministic for data preparation and uses OpenRouter only for the final comparison/report step
- `orchestration/openrouter_client.py` now loads OpenRouter config from the repo `.env`
- the current comparison report is no longer just freeform text sections
- it now returns structured fields including:
  - posture
  - target profile
  - peer snapshot
  - risk overlap rows
  - forward watchlist
  - narrative sections
  - narrative citations
  - per-risk citations in the target profile

Additional detail:

- `baseline_rag/pipeline.py` returns the same typed artifact envelope but marks it with `schema_version = "baseline_rag_v1"`
- the baseline path is intentionally simpler and exists mainly as a comparison baseline against the multi-step agentic pipeline
- the baseline package was split out of `orchestration/` to keep the comparison baseline conceptually separate from the main agentic orchestration layer
- baseline flattening now explicitly adapts to the current ingestion artifact shape so it can consume active financial outputs end to end

### `evaluation/models.py`, `evaluation/loaders.py`, `evaluation/deterministic.py`, `evaluation/judge.py`, and `evaluation/runner.py`

These files implement the current offline evaluation layer for comparing the agentic and baseline report outputs.

Responsibilities:

- load saved report artifacts from disk without changing the original pipeline outputs
- normalize agentic and baseline artifacts into one comparable evaluation input
- preserve evidence fairness by scoring each pipeline against only the evidence available to that pipeline
- run deterministic checks for report consistency, evidence coverage, and overreach
- optionally call OpenRouter through the existing client for claim support and comparative usefulness judging
- cache judge responses on disk for reproducibility and cost control
- write batch evaluation outputs and pairwise comparison summaries under `evaluation/outputs/`

Important detail:

- this layer is intentionally evaluation-only and does not change `OrchestrationArtifact`, `ComparisonReportResult`, or the pipeline contracts
- deterministic checks carry substantial weight so the evaluator is not only another LLM opinion
- the current batch entrypoint is `python -m evaluation.runner`

### `ui/app.py` and `ui/services.py`

These files implement the local operator and demo surface.

Responsibilities:

- run the agentic pipeline from the sidebar
- run the baseline RAG flow from the sidebar
- display results for either pipeline using the same typed artifact reader
- show both reports side by side when both are loaded
- load saved agentic report artifacts into the UI
- build missing dataset artifacts for new companies
- rebuild and inspect the FAISS index

Important detail:

- baseline results are currently session-scoped in the UI rather than written to disk by default
- the current UI renders evidence-aware report sections, compact peer tables, target risk citations, forward-watch evidence expanders, and dataset/index management panels

## Historical / Experimental Context

Earlier in the project, there was a more focused `Meta`-only experiment around `sec-parser` parsing quality and deterministic validation.

That earlier work established:

- `sec-parser` could produce a useful semantic tree for `10-K` filings
- `10-K` parsing required a workaround because the installed library exposed 10-Q-oriented parser logic
- deterministic validation was feasible using section coverage, ordering, length, and noise checks
- extracted `Item 8` content included financial tables, but not in a strongly structured table format

That experimentation influenced the current parser setup in `sec_parser_utils.py`.

## Core Sections Currently Targeted

The ingestion pipeline currently targets a reduced set of `10-K` sections for text extraction.

At the moment, `ingestion_pipeline.py` has active patterns for:

- `PART I`
- `Item 1. Business`
- `Item 1A. Risk Factors`
- `Item 1C. Cybersecurity`
- `Item 3. Legal Proceedings`
- `Item 7. MD&A`
- `Item 7A. Market Risk`

There are commented-out patterns for:

- `Item 8. Financial Statements`
- `PART III`
- `Item 10`

This means the current unified pipeline is narrower than the earlier validator logic.

Interpretation:

- the project has already proven that more sections can be found
- the production-oriented unified pipeline is currently being kept simpler
- section scope may expand again later after stability improves

## What the Current Pipeline Produces

For a ticker like `AAPL`, the current pipeline creates:

- `data-ingestion/outputs/AAPL/complete_ingestion.json`

That JSON contains:

- ticker
- cik
- company name
- selected metric aliases
- annual financial data
- quarterly financial data
- filing metadata
- parser mode
- extracted filing sections

For financial metrics, the expected current schema is:

- `financial_data.annual[metric] -> { "tag", "unit", "years", "values", "deltas" }`
- `financial_data.quarterly[metric] -> { "tag", "unit", "periods", "values", "deltas" }`

Interpretation:

- annual `deltas` are value deltas between adjacent annual points in the stored unit
- quarterly `deltas` are value deltas between adjacent quarterly points in the stored unit
- these are not percentage deltas

This output is the main machine-readable artifact that future stages will build on.

## Current Parser Behavior

For filing text, the pipeline prefers `sec-parser`.

The path is:

1. fetch filing HTML
2. parse HTML to semantic elements and semantic tree
3. flatten tree into rows
4. locate heading anchors with regex
5. apply explicit stop boundaries for each tracked section
6. save extracted section text into the final JSON

If `sec-parser` is unavailable, the code falls back to plain-text extraction from the raw HTML.
If `sec-parser` fails for an individual filing at runtime, the pipeline falls back for that filing and continues processing the remaining filings.
If `sec-parser` produces weak section coverage for an individual filing, the pipeline can also keep the plain-text result instead of the semantic-parse result.

That fallback is less accurate, but it prevents the entire ingestion pipeline from failing.

The output includes `parser_mode` so downstream code can know how trustworthy the extraction is.

## Recent Ingestion Fixes

The most recent ingestion cleanup addressed three practical data-quality issues:

1. Annual financial metrics are now deduplicated to one canonical datapoint per fiscal year.
2. Quarter labels are inferred more reliably from SEC metadata such as `fp` and `frame`, with a year-end snapshot fallback for Q4-style balance-sheet values.
3. Section extraction now uses explicit next-heading boundaries so tracked sections do not overflow into adjacent filing sections.

Additional recent updates:

4. Older 10-K filings are now discovered through SEC paginated submissions history rather than relying only on `filings.recent`.
5. XBRL fallback tags can now be merged across years so companies like `GOOG` do not lose recent or historical coverage when the active tag changes.
6. In the plain-text extractor, section headings are matched line-by-line so references to `Item 1A` or `Item 7A` inside paragraph text do not accidentally terminate a section.
7. Section heading matching is explicitly case-insensitive so all-caps filings are handled more consistently.
8. The active financial output format remains compact parallel arrays, and downstream extraction now works directly against that schema.

Interpretation:

- numeric time series should now look like `2025, 2024, 2023...` rather than mixing repeated or stale annual values
- text extraction for `Item 1`, `1A`, `1C`, `3`, `7`, and `7A` should stop at the intended next section more consistently
- running the pipeline with `--years 5` is intended to produce five years of both financial and filing-text coverage when SEC data is available
- current reference artifacts for `AAPL`, `META`, and `GOOG` are considered stable enough to drive the next analysis phase

## Important Project Decisions Already Made

### 1. Use SEC as the primary source of truth

The project is intentionally built around SEC EDGAR rather than third-party financial data vendors.

### 2. Keep ingestion deterministic

LLMs are not used for parsing.

### 3. Separate numbers and text

The current architecture treats:

- structured XBRL facts
- filing narrative text

as two separate but parallel ingestion tracks that are later merged.

### 4. Use only selected sections first

The project is not attempting to treat the whole filing equally.
It focuses first on the sections most likely to help risk and health analysis.

### 5. Build inspectable outputs

The project has emphasized writing reviewable JSON and text outputs to disk so that results can be inspected manually.

## Known Limitations Right Now

### 1. Section coverage in the main pipeline is narrower than the validator experiment

The main ingestion pipeline currently extracts fewer sections than were explored in earlier validation work.

### 2. Table handling is still weak

Although `sec-parser` detects tables, the current pipeline mainly treats them as flattened text.
This is especially relevant for `Item 8`, where financial statements matter the most.

### 3. Parser support for `10-K` is indirect

The current parser configuration is a workaround built on top of `Edgar10QParser`.
It works, but it is not a perfect native `10-K` parser.

### 4. Retrieval and orchestration are implemented, but still narrow

The project now has a working nearest-neighbor company matcher over curator artifacts, a baseline-vs-agentic comparison setup, and a stronger structured report layout. However, it is still a v1 analyst layer rather than a mature decision-support engine.

### 5. No mature financial-health synthesis layer yet

The system can now assemble target and peer context and call a comparison agent, but it does not yet produce a deeply designed financial-risk or health conclusion.

### 6. Evaluation is present, but still v1

The repo now has a practical offline evaluation layer, but it is intentionally lean. It focuses on saved artifact comparison, not on being a full general-purpose RAG benchmark suite.

## Where the Project Appears To Be Headed

Based on the implemented code and earlier project direction, the likely roadmap is:

1. stabilize ingestion for multiple companies
2. improve section extraction quality
3. improve table extraction, especially for `Item 8`
4. continue refining curator artifact quality
5. compare new companies against a reference set using the matcher
6. keep improving evaluation so agentic-vs-baseline quality differences are measurable
7. add an LLM for explanation, synthesis, and similarity-based reasoning

In other words:

- today: deterministic ingestion
- next: better extraction, orchestration, and storage
- later: stronger similarity-aware LLM interpretation

## How A Future LLM Should Interpret The Current Codebase

If a future LLM is asked to help on this project, it should understand:

- the ingestion system is the most mature part right now
- `complete_ingestion.json` is the main output artifact
- the current repo already contains a small reference set of ingested artifacts in `outputs/`
- `ingestion_pipeline.py` is the current operational entrypoint
- `SECClient`, `DocumentFetcher`, and `DataCleaner` form the core deterministic retrieval and normalization layer
- `sec_parser_utils.py` is the current semantic parsing layer
- section extraction is functional but still evolving
- table extraction is a known weak area
- the long-term goal is company comparison and financial-health analysis, not just filing download
- `orchestration/` is now the current top-level way to wire ingestion, retrieval, and comparison together
- `evaluation/` is now the current way to compare saved agentic and baseline reports on grounded dimensions
- `ui/` is now an important active layer, not just a thin wrapper
- there are now two report-producing paths in active use: the agentic comparison pipeline and the baseline RAG pipeline

The future LLM should avoid assuming:

- that the older Chroma demos are the active retrieval layer
- that table extraction is production-ready
- that `Item 8` is already structured for ratio analysis
- that the current parser is a perfect `10-K` parser
- that every company will use the same XBRL tag or fiscal-quarter labeling conventions

## Recommended Next Steps

The most sensible next development steps are:

1. expand the stable section set in `ingestion_pipeline.py` if needed
2. create structured export for `Item 8` tables
3. define the schema for storing ingested company artifacts for retrieval/comparison
4. decide what features will drive similarity between companies
5. improve the comparison-agent prompt/output design now that the orchestration layer is in place

## Bottom Line

This project has already moved beyond a simple parser experiment.

It now has a working deterministic ingestion pipeline that combines:

- structured SEC financial facts
- recent `10-K` narrative extraction

into one saved artifact per company.

The main unfinished gap is no longer basic wiring.
The main unfinished gap is improving the quality and usefulness of the final comparison-and-reasoning layer on top of the now-wired ingestion, retrieval, and orchestration stack.
