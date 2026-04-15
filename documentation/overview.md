# FinAgent Overview

FinAgent is a local SEC `10-K` analysis workflow built around deterministic data preparation, curator-based retrieval, and structured report generation.

## Active Flows

### Agentic Comparison Flow

1. `data-ingestion/` builds `data-ingestion/outputs/<TICKER>/complete_ingestion.json`
2. `data-extraction/` builds yearly extraction artifacts from that ingestion output
3. curator generation writes `data-extraction/outputs/curator/<TICKER>/<ticker>_<year>.json`
4. `rag-matching/` retrieves similar companies from the curator embedding index
5. `orchestration/` resumes missing steps, runs matching, assembles peer context, and generates the structured comparison report

### Baseline RAG Flow

- `baseline_rag/pipeline.py` loads the target ingestion artifact directly
- it retrieves peers through FAISS when possible, otherwise falls back to a simpler ingestion-scan peer set
- it makes a single report-generation call and returns the same top-level artifact shape with `schema_version = "baseline_rag_v1"`
- it now lives outside `orchestration/` as a separate comparison baseline package while still reusing the shared report schema and OpenRouter client

### Offline Evaluation Flow

1. `evaluation/loaders.py` loads saved agentic or baseline report artifacts from disk
2. it normalizes both into a single `EvaluationInput` shape
3. `evaluation/deterministic.py` scores consistency, evidence coverage, and overreach rules
4. `evaluation/judge.py` can optionally run OpenRouter-backed claim and report judging with on-disk caching
5. `evaluation/runner.py` writes batch evaluation outputs and pairwise agentic-vs-baseline comparisons

## Source Of Truth

- Ingestion source of truth: `data-ingestion/outputs/<TICKER>/complete_ingestion.json`
- Retrieval source of truth: `data-extraction/outputs/curator/<TICKER>/`
- Saved agentic comparison artifact: `orchestration/outputs/<TICKER>/<ticker>_comparison_bundle.json`
- Baseline RAG artifacts are currently session-scoped in the Streamlit UI and are not written to disk by default
- Saved evaluation artifact: `evaluation/outputs/<run_name>.json`

## Main Entry Points

- Ingestion: `python data-ingestion/ingestion_pipeline.py AAPL --years 5`
- Extraction: `python data-extraction/main.py AAPL`
- Curator batch run: `python data-extraction/company_filing_embedding.py AAPL`
- Matching: `python rag-matching/matcher.py --input-file data-extraction/outputs/curator/AAPL/aapl_2025.json --top 2 --json`
- Agentic orchestration: `python orchestration/runner.py AAPL --top 2 --json`
- Evaluation: `python -m evaluation.runner --agentic-dir orchestration/outputs --json`
- UI: `streamlit run ui/app.py`

## Current UI Behavior

The local Streamlit app now supports:

- running the agentic pipeline
- running the baseline RAG pipeline
- viewing both outputs side by side
- loading saved report artifacts
- dataset intake for new companies
- FAISS rebuild and index inspection

## Important Design Choices

- Ingestion is deterministic and does not use an LLM for SEC parsing.
- Retrieval uses curator embeddings plus FAISS.
- The active report schema is defined in `orchestration/report_models.py`.
- The baseline comparison path is intentionally simpler than the agentic path, but it is still kept compatible with the current ingestion artifact shape.
- Baseline artifacts are not written to disk by the Streamlit UI by default; evaluation against baseline outputs requires separately saved baseline artifact files.
- The evaluation layer is intentionally offline and comparison-first rather than a general benchmark framework.
- The evaluation layer uses one scorecard but keeps evidence pools separate so the baseline is not judged against hidden agentic-only evidence.
- Final reports are structured rather than freeform and now include evidence-aware fields such as risk citations and cited narrative sections.
- The UI is local-only and uses the existing Python pipeline modules directly.

## Evaluation Scorecard

The current evaluation package scores:

- `deterministic_consistency`
- `evidence_coverage`
- `claim_support`
- `comparative_usefulness`
- `overreach_penalty`

Judge-backed metrics are optional. Deterministic-only mode works from local files without network access.

## Current Gaps

- `Item 8` table extraction is still weak.
- The comparison/report layer is improved but still v1 quality.
- FAISS, transformer, and embedding-model dependencies must exist locally for full end-to-end runs.
