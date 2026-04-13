# FinAgent

Financial risk screening project built around SEC 10-K ingestion, yearly extraction, curator artifact generation, and company matching.

Current active flow:

1. `data-ingestion/` builds deterministic `complete_ingestion.json`
2. `data-extraction/` builds yearly extraction artifacts
3. curator generation writes `data-extraction/outputs/curator/<TICKER>/<ticker>_<year>.json`
4. `rag-matching/` builds a persistent FAISS index over curator artifacts and retrieves similar companies
5. `orchestration/` resumes missing target-company steps, runs matching, gathers matched future-year context, and calls an OpenRouter-backed comparison agent

Important note:

- the current standalone matcher uses curator embeddings plus FAISS
- it does **not** use the older Chroma demo directories in `chroma_db/` or `chroma_db_pipeline/`
- it assumes curator files already include `embedding_vector`

Orchestration:

- run `python orchestration/runner.py AAPL --json` to execute the hybrid comparison flow
- `orchestration/openrouter_client.py` loads `OPENROUTER_API_KEY` and optional `OPENROUTER_MODEL` from the repo `.env`
- if FAISS is unavailable, orchestration stops at the matching step and writes a structured failure artifact instead of crashing
- the comparison report now returns structured fields for posture, target profile, peer snapshot, risk overlap, forward watchlist, and short narrative sections

UI:

- run `streamlit run ui/app.py` to launch the local demo UI
- the UI supports:
  - running a comparison report for a ticker
  - browsing saved orchestration artifacts under `orchestration/outputs/`
  - adding a company to the local artifact corpus by building ingestion, extraction, and curator outputs
  - rebuilding the FAISS index from the current curator dataset

Ingestion note:

- `data-ingestion/company_facts_cleaner.py` currently emits annual and quarterly financials as point lists again, not compact `years/values/deltas` arrays
- if a ticker was ingested during the compact-format window, rerun ingestion so its `complete_ingestion.json` matches the current extraction-compatible schema
