# FinAgent

FinAgent is a local SEC `10-K` analysis system for building company comparison reports from public filings.

At a high level, the project:

- ingests structured financial data and filing text from SEC EDGAR
- extracts yearly signals from those filings
- builds curator artifacts and embeddings for retrieval
- finds similar peer companies with FAISS
- generates a structured comparison report through either:
  - a multi-step agentic pipeline
  - a simpler baseline RAG pipeline
- optionally evaluates saved reports offline with a shared scorecard

## Project Overview

The repo is organized around a few main layers:

- `data-ingestion/`: deterministic SEC retrieval, financial cleaning, and filing section extraction
- `data-extraction/`: yearly extraction artifacts and curator artifacts with risk signals and embeddings
- `rag-matching/`: FAISS-based peer retrieval over curator artifacts
- `orchestration/`: the main agentic comparison pipeline and report generation
- `baseline_rag/`: the simpler comparison baseline over ingestion artifacts
- `evaluation/`: offline scoring for saved agentic vs baseline report artifacts
- `ui/`: a local Streamlit app for running and comparing pipelines

## Agentic Flow

The agentic flow is the main end-to-end path in this repo. It is hybrid by design: data preparation is deterministic, and OpenRouter is used only at the final report-generation stage.

### Step 1. Ingestion

`data-ingestion/` builds a unified company artifact at:

- `data-ingestion/outputs/<TICKER>/complete_ingestion.json`

This artifact includes:

- company identity
- annual financial metrics
- quarterly financial metrics
- filing metadata
- extracted `10-K` text sections

Run ingestion for one ticker:

```bash
python data-ingestion/ingestion_pipeline.py AAPL --years 5
```

### Step 2. Yearly Extraction

`data-extraction/` reads the ingestion artifact and produces yearly extraction JSON files that combine numeric deltas with ranked text candidates.

Output location:

- `data-extraction/outputs/<TICKER>/<ticker>_<year>_extraction.json`

Run extraction:

```bash
python data-extraction/main.py AAPL
```

### Step 3. Curator Artifacts

The curator step converts yearly extraction output into retrieval-ready artifacts with:

- financial delta labels
- curated risk signals
- embedding text
- embedding vectors

Output location:

- `data-extraction/outputs/curator/<TICKER>/<ticker>_<year>.json`

Run curator generation for a ticker:

```bash
python data-extraction/company_filing_embedding.py AAPL
```

### Step 4. Peer Retrieval

`rag-matching/` builds and queries a FAISS index over curator artifacts so the system can retrieve similar companies.

Important input:

- a curator artifact for the target company and year

Index build command:

```bash
python rag-matching/indexer.py
```

Match command:

```bash
python rag-matching/matcher.py --input-file data-extraction/outputs/curator/AAPL/aapl_2025.json --top 2 --json
```

### Step 5. Agentic Report Generation

`orchestration/` is the main comparison pipeline. It:

- checks whether ingestion, extraction, and curator artifacts already exist
- runs only the missing target-company steps
- retrieves the top peer companies
- expands peer context
- generates a structured comparison report
- saves the final comparison bundle to disk

Output location:

- `orchestration/outputs/<TICKER>/<ticker>_comparison_bundle.json`

Run the full agentic pipeline:

```bash
python orchestration/runner.py AAPL --json
```

## Baseline RAG Flow

The baseline is intentionally simpler than the agentic pipeline. It exists so you can compare a direct, single-shot reporting approach against the richer multi-step flow.

The baseline:

- reads ingestion artifacts directly
- finds peers through the same retrieval path when possible
- skips the extraction and curator reasoning depth
- generates the same top-level report envelope as the agentic flow

Main module:

- `baseline_rag/pipeline.py`

Current note:

- the baseline package is separate from `orchestration/`
- it is designed for side-by-side comparison with the agentic flow
- it is compatible with the current ingestion artifact shape

## Commands By Stage

### Run One Stage At A Time

Ingestion:

```bash
python data-ingestion/ingestion_pipeline.py AAPL --years 5
```

Extraction:

```bash
python data-extraction/main.py AAPL
```

Curator artifact generation:

```bash
python data-extraction/company_filing_embedding.py AAPL
```

Build the FAISS index:

```bash
python rag-matching/indexer.py
```

Retrieve similar companies:

```bash
python rag-matching/matcher.py --input-file data-extraction/outputs/curator/AAPL/aapl_2025.json --top 2 --json
```

Run the full agentic pipeline:

```bash
python orchestration/runner.py AAPL --json
```

Run the local UI:

```bash
streamlit run ui/app.py
```

Run offline evaluation on saved artifacts:

```bash
python -m evaluation.runner --agentic-dir orchestration/outputs --baseline-dir baseline_rag/outputs --json
```

## Using The Project

There are two common ways to use the repo.

### Option 1. Use The Full Agentic Pipeline

Use this when you want the main report-generation path.

Recommended order:

1. ingest a company
2. generate extraction artifacts
3. generate curator artifacts
4. build or refresh the FAISS index
5. run orchestration to generate the final report

### Option 2. Use The UI

Use this when you want a local operator workflow for:

- running the agentic pipeline
- running the baseline RAG pipeline
- comparing both outputs side by side
- loading saved report artifacts
- rebuilding or inspecting the FAISS index

Run:

```bash
streamlit run ui/app.py
```

## Key Output Artifacts

Main outputs you will work with:

- ingestion artifact: `data-ingestion/outputs/<TICKER>/complete_ingestion.json`
- yearly extraction artifacts: `data-extraction/outputs/<TICKER>/`
- curator artifacts: `data-extraction/outputs/curator/<TICKER>/`
- agentic comparison bundles: `orchestration/outputs/<TICKER>/`
- evaluation outputs: `evaluation/outputs/`

## Environment Notes

The report-generation layers use OpenRouter through:

- `orchestration/openrouter_client.py`

Expected environment variables:

- `OPENROUTER_API_KEY`
- optional `OPENROUTER_MODEL`

The project also depends on local FAISS / embedding tooling for the retrieval path.

## Documentation

Additional project context:

- [documentation/overview.md](documentation/overview.md)
- [documentation/project-handoff-for-llm.md](documentation/project-handoff-for-llm.md)
- [documentation/roadmap.md](documentation/roadmap.md)
