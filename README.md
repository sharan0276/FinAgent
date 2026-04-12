# FinAgent

Financial risk screening project built around SEC 10-K ingestion, yearly extraction, curator artifact generation, and company matching.

Current active flow:

1. `data-ingestion/` builds deterministic `complete_ingestion.json`
2. `data-extraction/` builds yearly extraction artifacts
3. curator generation writes `data-extraction/outputs/curator/<TICKER>/<ticker>_<year>.json`
4. `rag-matching/` builds a persistent FAISS index over curator artifacts and retrieves similar companies

Important note:

- the current standalone matcher uses curator embeddings plus FAISS
- it does **not** use the older Chroma demo directories in `chroma_db/` or `chroma_db_pipeline/`
- it assumes curator files already include `embedding_vector`
