# Risk Analysis

This package builds yearly company analysis artifacts on top of the source-of-truth ingestion outputs in `data-ingestion/outputs/<TICKER>/complete_ingestion.json`.

It preserves the ingested numeric series and adds:

- deterministic numeric trend summaries
- section-level risk signal extraction for `Item 1A`, `Item 1C`, `Item 3`, `Item 7`, and `Item 7A`
- per-filing-year section outputs
- lightweight 5-year rollups

Run from the project root:

```bash
python risk-analysis/main.py META
python risk-analysis/main.py GOOG
python risk-analysis/main.py AAPL --source data-ingestion/outputs/AAPL/complete_ingestion.json
python risk-analysis/main.py AAPL --provider heuristic
```

Outputs are written to `risk-analysis/outputs/<TICKER>/` as one file per filing year, for example `META_2025_company_analysis.json`.
