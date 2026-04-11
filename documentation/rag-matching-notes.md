# RAG Matching Notes

## Goal

Provide a standalone retrieval tool that accepts one curator JSON file and returns the top distinct matching companies from the reference curator dataset.

This is the current retrieval layer for company comparison experiments.

---

## Canonical Inputs

The matcher uses curator files under:

- `data-extraction/outputs/curator/<TICKER>/<ticker>_<year>.json`

Each curator file represents one:

- `company + filing_year`

The matcher treats those curator files as the retrieval database.

Important fields used by the matcher:

- `ticker`
- `company`
- `filing_year`
- `embedding_text`
- `embedding_vector`

---

## Current Architecture

The active retrieval code lives in:

- `rag-matching/curator_store.py`
- `rag-matching/indexer.py`
- `rag-matching/matcher.py`
- `rag-matching/runtime_compat.py`

The flow is:

1. Recursively scan `data-extraction/outputs/curator/**/*.json`
2. Load each file's stored `embedding_vector`
3. Build one persistent similarity index
4. Save index artifacts under `rag-matching/index_artifacts/`
5. Accept one curator JSON file as the query input
6. Use the query file's `embedding_vector`
7. Search the index for nearest neighbors by cosine similarity
8. Remove the query company itself
9. Deduplicate repeated years using `best year wins`
10. Return the top `k` distinct companies

---

## Similarity Backend

Primary backend:

- FAISS `IndexFlatIP`

Because curator embeddings are normalized, inner product is equivalent to cosine similarity.

Saved artifacts:

- `rag-matching/index_artifacts/faiss.index`
- `rag-matching/index_artifacts/metadata.json`

The metadata file maps index rows back to:

- `ticker`
- `company`
- `filing_year`
- `source_path`

Practical note:

- if `faiss` is unavailable, the code currently falls back to a numpy matrix search
- the intended primary runtime is still FAISS

---

## Query Behavior

The matcher now accepts a single input file path, for example:

```bash
python rag-matching/matcher.py --input-file data-extraction/outputs/curator/AAPL/aapl_2022.json --top 2 --json
```

Default test target:

- `data-extraction/outputs/curator/AAPL/aapl_2022.json`

Distinct-company rules:

- exclude the exact query file
- exclude all entries with the same ticker as the query
- if multiple filing years from the same matched company appear, keep only the highest-similarity year

Output in JSON mode:

- `query`
- `matches`

Current `matches` payload is intentionally minimal:

- `ticker`
- `similarity`

Internally, the matcher also tracks:

- `company`
- `filing_year`
- `source_path`

for debugging and traceability.

---

## Embedding Model

The curator files were generated using:

- `SentenceTransformer("BAAI/bge-m3")`

That means the stored vectors are:

- `1024` dimensions

Query-time fallback behavior:

- if a query file already has `embedding_vector`, use it directly
- if not, embed `embedding_text` in memory using the same `BAAI/bge-m3` model
- do not rewrite the query file in v1

This model alignment is required so query vectors match the FAISS index dimension.

---

## Windows Runtime Note

On some Windows environments, FAISS and Torch-backed libraries can load conflicting OpenMP runtimes.

To reduce that issue, `rag-matching/runtime_compat.py` sets:

- `KMP_DUPLICATE_LIB_OK=TRUE`

before importing FAISS or `sentence-transformers` in the matcher path.

This is a compatibility workaround, not a modeling decision.

---

## What This Layer Does Not Use

The current matcher does **not** use:

- `chroma_db/`
- `chroma_db_pipeline/`
- `src/rag/baseline_rag.py`
- `src/pipeline.py`

Those belong to older Chroma-based RAG demos and are separate from the active curator matching flow.

---

## Tests

Primary matcher test file:

- `test_rag_matching.py`

Current checks cover:

- recursive curator discovery
- persistent index creation
- distinct-company retrieval
- self-company exclusion
- repeated-query index reuse
- fallback query embedding when `embedding_vector` is missing

Run:

```bash
python -m unittest test_rag_matching.py
```
