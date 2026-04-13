# FinAgent

FinAgent is a local SEC `10-K` comparison workflow built around deterministic ingestion, curator embeddings, FAISS matching, and a final comparison report.

## Main Commands

- `python data-ingestion/ingestion_pipeline.py AAPL --years 5`
- `python data-extraction/main.py AAPL`
- `python orchestration/runner.py AAPL --json`
- `streamlit run ui/app.py`

## Active Pipeline

1. `data-ingestion/` builds `complete_ingestion.json`
2. `data-extraction/` builds yearly extraction artifacts
3. curator generation writes curator JSONs with embeddings
4. `rag-matching/` builds and queries the FAISS index
5. `orchestration/` assembles the comparison bundle and report
6. `ui/` provides a local demo/operator interface

## Notes

- The active retrieval path uses curator embeddings plus FAISS.
- `orchestration/openrouter_client.py` loads `OPENROUTER_API_KEY` and optional `OPENROUTER_MODEL` from the repo `.env`.
- If FAISS is unavailable, orchestration writes a structured failure artifact instead of crashing.
- If an old ingestion artifact uses a stale compact financial schema, rerun ingestion before extraction.

## Docs

- [documentation/overview.md](documentation/overview.md)
- [documentation/roadmap.md](documentation/roadmap.md)
