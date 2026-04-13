# FinAgent Overview

FinAgent is a local SEC `10-K` comparison workflow built around deterministic data preparation plus an LLM-based final report step.

## Active Flow

1. `data-ingestion/` builds `data-ingestion/outputs/<TICKER>/complete_ingestion.json`
2. `data-extraction/` builds yearly extraction artifacts from that ingestion output
3. curator generation writes `data-extraction/outputs/curator/<TICKER>/<ticker>_<year>.json`
4. `rag-matching/` builds a FAISS index over curator embeddings and retrieves similar companies
5. `orchestration/` resumes missing steps, runs matching, assembles peer context, and generates the comparison report
6. `ui/` exposes the flow in a local Streamlit app

## Source Of Truth

- Ingestion source of truth: `data-ingestion/outputs/<TICKER>/complete_ingestion.json`
- Retrieval source of truth: `data-extraction/outputs/curator/<TICKER>/`
- Final comparison artifact: `orchestration/outputs/<TICKER>/<ticker>_comparison_bundle.json`

## Main Entry Points

- Ingestion: `python data-ingestion/ingestion_pipeline.py AAPL --years 5`
- Extraction: `python data-extraction/main.py AAPL`
- Matching: `python rag-matching/matcher.py --input-file data-extraction/outputs/curator/AAPL/aapl_2025.json --top 2 --json`
- Orchestration: `python orchestration/runner.py AAPL --json`
- UI: `streamlit run ui/app.py`

## Important Design Choices

- Ingestion is deterministic and does not use an LLM for SEC parsing.
- Comparison uses curator embeddings plus FAISS.
- The current report schema is defined in `orchestration/report_models.py`.
- The UI is local-only and uses the existing Python pipeline modules directly.

## Current Gaps

- `Item 8` table extraction is still weak.
- The comparison/report layer is still v1 quality.
- FAISS and model dependencies must exist locally for full end-to-end runs.
