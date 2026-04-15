# Data Extraction Notes

## Goal

Build a reusable downstream extraction pipeline that reads the source-of-truth ingestion artifact:

- `data-ingestion/outputs/<TICKER>/complete_ingestion.json`

and produces one unified JSON per:

- `company + filing_year`

Each yearly extraction artifact combines:

- FinBERT-ranked text candidates from selected 10-K sections
- annual year-over-year numeric deltas for the tracked financial metrics
- filing and source metadata for traceability

This phase does **not** produce final risk taxonomy labels yet.
It is an intermediate extraction layer meant to prepare high-signal text and compact numeric change data for later curation.

---

## Current Architecture

The new package lives in:

- `data-extraction/`

The main entrypoint is:

- `data-extraction/main.py`

The CLI flow is:

1. Load one `complete_ingestion.json`
2. Iterate over `text_data.filings[]`
3. Treat each filing as one yearly output unit
4. Extract ranked text candidates from selected sections
5. Compute annual numeric deltas for the filing year
6. Write one JSON file per filing year

Default output path:

- `data-extraction/outputs/<TICKER>/`

Example output filename:

- `aapl_2025_extraction.json`

---

## Implemented Modules

### `data-extraction/cli.py`

Command-line interface for the new extraction pipeline.

Responsibilities:

- accept ticker input
- optionally override source artifact path
- optionally filter by year or years
- choose output directory
- configure FinBERT model and batch size
- launch the extraction pipeline

### `data-extraction/pipeline.py`

Main orchestration layer for the package.

Responsibilities:

- load the ingestion artifact
- iterate through filing records
- derive filing year from filing date
- build text candidates
- build numeric deltas
- assemble the final schema
- write one output JSON per filing year

### `data-extraction/text_candidates.py`

Text preprocessing and candidate selection layer.

Responsibilities:

- read only the selected filing sections
- split section text into sentences
- filter low-value strings such as:
  - empty or very short sentences
  - heading-like lines
  - numeric or table-like fragments
  - obvious SEC boilerplate references
- score valid sentences through the FinBERT scorer
- rank candidates by negative-class probability
- apply per-section balancing
- keep the final top `100` candidates per filing year

### `data-extraction/scoring.py`

FinBERT scoring adapter.

Responsibilities:

- load `ProsusAI/finbert` through `transformers`
- score sentences in batches
- return class probabilities for:
  - positive
  - negative
  - neutral

Important detail:

- the candidate ranking score is:
  - `risk_score = P(negative)`

FinBERT is only used as a sentence ranker in this phase.
It does **not** assign project risk taxonomy labels.

### `data-extraction/numeric_delta.py`

Numeric extraction layer for annual financial changes.

Responsibilities:

- read only `financial_data.annual`
- locate the current-year datapoint for each tracked metric
- locate the prior-year datapoint
- compute year-over-year percent change
- assign the severity bucket
- preserve traceability fields such as:
  - value
  - end date
  - accession
  - source tag

### `data-extraction/models.py`

Pydantic models for the final output schema.

Examples:

- `FinBertProbabilities`
- `TextCandidate`
- `NumericDelta`
- `ExtractionArtifact`

### `data-extraction/artifact_loader.py`

Helpers for:

- locating the default source artifact
- loading the JSON
- filtering filings by year when requested

### `data-extraction/constants.py`

Central configuration for:

- schema version
- target sections
- numeric metrics
- candidate counts
- delta bucket thresholds

---

## Target Text Sections

The current extraction step reads only:

- `item_1a_risk_factors`
- `item_1c_cybersecurity`
- `item_3_legal`
- `item_7_mda`
- `item_7a_market_risk`

If a section is missing or empty in the ingestion artifact:

- it is added to `skipped_sections`
- it is not treated as a failure for the whole filing

If a section contributes usable candidates:

- it is added to `processed_sections`

---

## Text Candidate Output

For each filing year, the pipeline keeps the top `100` ranked candidates.

Selection behavior:

- target `20` candidates per section
- if a section is sparse, unused slots are reallocated globally by score

Each candidate includes:

- `candidate_id`
- `section_id`
- `section_label`
- `sentence_index`
- `sentence_text`
- `previous_sentence`
- `next_sentence`
- `risk_score`
- `finbert_probs`

The stored probabilities are:

- `positive`
- `negative`
- `neutral`

The output is a ranked candidate list only.
There is no thresholded boolean label and no final structured risk classification yet.

---

## Numeric Delta Output

The pipeline uses the annual financial series already present in the ingestion artifact.

Important assumption:

- the active extraction code supports both the compact ingestion schema and older list-style yearly datapoints
- fresh ingestion runs are expected to use the compact `years` / `values` / `deltas` array form
- older list-style ingestion artifacts can still be read, but they are no longer the primary format

Current metrics:

- `Revenues`
- `NetIncome`
- `Cash`
- `Assets`
- `LongTermDebt`
- `OperatingCashFlow`
- `ResearchAndDevelopment`
- `GrossProfit`

For each filing year and metric, the output stores:

- `current_value`
- `previous_value`
- `current_end_date`
- `previous_end_date`
- `current_accession`
- `previous_accession`
- `current_tag`
- `previous_tag`
- `delta_percent`
- `label`
- `reason`

Current bucket thresholds:

- standard metrics:
  - `> 20%` -> `strong_growth`
  - `5% to 20%` -> `moderate_growth`
  - `-5% to 5%` -> `stable`
  - `-20% to -5%` -> `moderate_decline`
  - `< -20%` -> `severe_decline`
- `LongTermDebt` is treated as an inverse metric:
  - `> 20% effective improvement` -> `strong_reduction`
  - `5% to 20% effective improvement` -> `moderate_reduction`
  - `-5% to 5%` -> `stable`
  - `-20% to -5% effective deterioration` -> `moderate_increase`
  - `< -20% effective deterioration` -> `severe_increase`

If a prior-year datapoint is missing:

- `delta_percent = null`
- `label = null`
- `reason = "missing_prior_year"`

If the current-year datapoint is missing:

- `reason = "missing_current_year"`

If the prior value is invalid for percent-change math:

- `reason = "invalid_prior_value"`

---

## Output Schema

Each yearly extraction file contains:

- `schema_version`
- `model_name`
- `run_timestamp`
- `ticker`
- `company_name`
- `cik`
- `filing_year`
- `filing_date`
- `accession`
- `parser_mode`
- `source_artifact_path`
- `processed_sections`
- `skipped_sections`
- `numeric_deltas`
- `text_candidates`

This is the active downstream artifact for the extraction phase.

---

## Running The Pipeline

From project root, after installing dependencies:

```bash
python data-extraction/main.py AAPL
python data-extraction/main.py GOOG
python data-extraction/main.py META
```

Run only a single filing year:

```bash
python data-extraction/main.py AAPL --year 2025
python data-extraction/main.py GOOG --year 2025
python data-extraction/main.py META --year 2025
```

Override output directory:

```bash
python data-extraction/main.py AAPL --output-dir data-extraction/outputs
```

Override model or batch size:

```bash
python data-extraction/main.py AAPL --model ProsusAI/finbert --batch-size 16
```

---

## Tests

Test file:

- `test_data_extraction.py`

Current checks cover:

- sentence splitting
- candidate filtering
- numeric delta bucketing
- missing-prior-year handling
- integration-style artifact writing for one real AAPL filing year

Run:

```bash
python -m unittest test_data_extraction.py
```

---

## Dependency Note

The implemented FinBERT path depends on:

- `transformers`
- `torch`

These were added to root requirements.

If `torch` or `transformers` is not installed, the CLI will fail early with a clear runtime error from the scorer initialization.

---

## Current Status

What is complete:

- new `data-extraction/` package
- CLI entrypoint
- per-year JSON writing
- FinBERT-based sentence ranking pipeline
- annual numeric delta extraction
- Pydantic output schema
- unittest coverage for the implemented v1 behavior

What is intentionally not included yet:

- final risk taxonomy assignment
- LLM-based risk-signal structuring
- final analyst report generation

Interpretation:

- `data-ingestion/` remains the source-of-truth ingestion layer
- `data-extraction/` is now the active intermediate curation layer
- later phases should build on the extraction outputs rather than re-parsing raw filing text

---

## Phase 2: Curator Agent (NEW)

Building on top of the original structure above, we have introduced the "Curator Agent". This agent acts as a strict processor that transforms the FinBERT text candidates into highly structured, validated risk signals.

**What was added:**
- **`curator_agent.py`**: The agent orchestrator. It applies deterministic constraints (stratified section allocation, fallback fillers) to create highly structured extraction arrays, fetching LLM analysis through OpenRouter.
- **`curator_models.py`**: Strict Pydantic models mapping the project's precise risk taxonomy and delta labels.
- **`company_filing_embedding.py`**: Automatically parses a chronological folder of yearly extractions for a given ticker without interactive prompting.
- **`test_curator_agent.py`**: Dedicated deterministic unit tests safeguarding candidate logic, financial bucketing logic, prompt structures, and embedding metadata formats. 

**Workflow:**
The Curator Agent automatically generates normalized `1024`-dimension local vector embeddings using `BGE-M3` and merges them with the financial deltas into `outputs/curator/<TICKER>/<ticker>_<year>.json`.

These curator files are now the canonical input to the active company-matching retrieval layer in `rag-matching/`.

**Running Phase 2:**
First, ensure your key is valid and without trailing newlines:
Set `OPENROUTER_API_KEY` in your shell or repo `.env`.

Run the batch embedding processor:
```bash
python data-extraction/company_filing_embedding.py AAPL
```

To run Phase 2 tests isolated from API calls:
```bash
python -m unittest test_curator_agent.py
```

---

## Phase 3: Standalone RAG Matching

There is now a standalone retrieval layer in:

- `rag-matching/`

This layer treats the curator output directory as the retrieval database:

- `data-extraction/outputs/curator/<TICKER>/<ticker>_<year>.json`

Current behavior:

1. recursively load all curator JSON files
2. use stored `embedding_vector` values as the reference vectors
3. build one persistent FAISS index under `rag-matching/index_artifacts/`
4. accept a single curator JSON file as query input
5. retrieve nearest neighbors by cosine similarity
6. exclude the query company itself
7. deduplicate repeated years using `best year wins`
8. return the top distinct matching companies

Important detail:

- this retrieval layer uses FAISS and curator embeddings
- it does **not** use `chroma_db/` or `chroma_db_pipeline/`
- it assumes curator files already contain `embedding_vector`

Example:

```bash
python rag-matching/matcher.py --input-file data-extraction/outputs/curator/AAPL/aapl_2022.json --top 2 --json
```

Build or rebuild the persistent index:

```bash
python rag-matching/indexer.py
```

Matcher tests:

```bash
python -m unittest test_rag_matching.py
```

---

## Phase 4: Orchestration Layer

There is now a top-level orchestration package in:

- `orchestration/`

This layer is the current end-to-end controller for user-company comparison.

Current behavior:

1. accept a target ticker
2. reuse existing ingestion, extraction, and curator artifacts when present
3. build only the missing target-company artifacts when needed
4. run RAG matching against the target curator file
5. take the top 2 matched companies
6. expand each match into `matched year + up to 2 future curator years`
7. assemble a structured comparison bundle
8. call a final comparison agent through OpenRouter
9. save the final bundle under `orchestration/outputs/<TICKER>/`

Run:

```bash
python orchestration/runner.py AAPL --json
```

Important details:

- `orchestration/openrouter_client.py` loads OpenRouter config from the repo `.env`
- orchestration does not rebuild the FAISS index
- if FAISS is unavailable, the run stops at the matching step and writes a structured failure artifact
- the current orchestration controller lives in `orchestration/orchestration_pipeline.py`
- the comparison report now returns structured fields rather than just generic prose sections

Current report shape includes:

- `summary`
- `posture`
- `target_profile`
- `peer_snapshot`
- `risk_overlap_rows`
- `forward_watchlist`
- `narrative_sections`

Current report details now also include:

- citation-aware `target_profile.top_risks`
- citation-aware `narrative_sections`
- a local Streamlit rendering path that exposes evidence expanders for cited sections and peer watchlist evidence

---

## UI And Comparison Surfaces

The local Streamlit UI in `ui/` now acts as a comparison workbench rather than just a thin orchestration trigger.

Current behavior:

1. run the agentic pipeline
2. run a separate baseline RAG pipeline from `orchestration/baseline_rag.py`
3. compare both outputs side by side in the UI
4. load saved agentic artifacts
5. manage dataset intake and FAISS index rebuilds

Important detail:

- baseline RAG returns the same top-level typed artifact shape but uses `schema_version = "baseline_rag_v1"`
- baseline runs are currently ephemeral in the UI and are not saved under `orchestration/outputs/` by default
