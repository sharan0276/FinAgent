# FinAgent Roadmap

## Current Priority

Turn the working ingestion -> extraction -> curator -> matching pipeline into a stronger comparison product by improving signal quality, report quality, and evaluation against the simpler baseline RAG path.

## Near-Term Work

1. Improve risk-signal quality and validation in curator outputs.
2. Improve comparison-agent prompting and output reliability.
3. Improve `Item 8` handling for more structured financial evidence.
4. Expand the reference dataset beyond the current small ticker set.
5. Make the local UI a better operator and demo surface.
6. Use the baseline-vs-agentic setup to evaluate whether the extra pipeline complexity is improving grounded output quality.
7. Keep the separate `baseline_rag/` package aligned with the current ingestion artifact contract as ingestion/extraction evolve.

## Constraints To Preserve

- Keep SEC ingestion deterministic.
- Keep artifacts inspectable on disk.
- Keep retrieval based on curator embeddings and FAISS.
- Avoid reintroducing older prototype paths.
