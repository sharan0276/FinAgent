# RAG Matching Notes

## Goal

Provide a standalone retrieval tool that accepts one curator JSON file and returns the top distinct matching companies from the reference curator dataset.

This is the current retrieval layer for company comparison experiments.

This matcher is also the retrieval backend used by the new top-level `orchestration/` package.

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

Current `matches` payload is:

- `ticker`
- `filing_year`
- `similarity`

The standalone matcher may also have `company` available internally, but orchestration should treat `ticker`, `filing_year`, and `similarity` as the stable minimum contract.

Internally, the matcher also tracks:

- `company`
- `filing_year`
- `source_path`

for debugging and traceability.

---

## Embedding Assumption

The curator files were generated with `1024`-dimension `BGE-M3` vectors.

The matcher assumes:

- every database curator file has `embedding_vector`
- every query curator file has `embedding_vector`

If a file is missing `embedding_vector`, the matcher raises an error.

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

## Relationship To Orchestration

The new `orchestration/` layer uses this matcher as a deterministic substep:

1. prepare the target company curator artifact
2. call the matcher for top distinct peers
3. use each matched `ticker + filing_year` as the anchor for collecting up to 2 future curator years
4. pass that assembled context to the OpenRouter-backed comparison agent

The matcher itself remains standalone and unchanged; orchestration is the layer that adds report assembly on top of it.

The current comparison layer built on top of matcher output is structured rather than text-only. It uses the matched `ticker + filing_year` anchors to populate:

- peer neighborhood snapshot
- current risk overlap rows
- forward watchlist items based on later peer years

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

Run:

```bash
python -m unittest test_rag_matching.py
```
