# Next Steps: Section-Level Comparable Risk + Numeric Pipeline

## Purpose

This document captures the agreed plan for the next phase of the project so work can continue in a fresh context without losing design decisions.

Primary objective:

- Build a concise, verifiable, and comparable representation of company risk signals by combining:
  - SEC API numeric trends
  - section/subheading-level textual risk signals from 10-K filings

---

## Current Baseline (Already Implemented)

- Deterministic ingestion pipeline exists in `data-ingestion/ingestion_pipeline.py`.
- Financial metrics are pulled from SEC Company Facts API and cleaned.
- Text sections are extracted from filing content for key 10-K sections.
- Current output is written to `data-ingestion/outputs/<TICKER>/complete_ingestion.json`.

---

## Core Design Decisions (Locked)

### 1. Base comparison unit

Use:

- `company + filing + section`

Then aggregate upward as needed.

### 2. Section boundaries for extraction (code-based)

Section extraction uses next heading as stop boundary:

- `Item 1` stops at `Item 1A`
- `Item 1A` stops at `Item 1B`
- `Item 7` stops at `Item 7A`
- `Item 7A` stops at `Item 8`

### 3. Subheading strategy

- Extract all subheadings within each section.
- LLM processing runs subheading-by-subheading.

### 4. Risk taxonomy and signal shape

- Use `20` risk types initially.
- Each signal includes at least:
  - risk type(s) (multi-label allowed)
  - severity
  - direction
  - confidence
  - evidence text/span

Direction values:

- `increasing`
- `stable`
- `decreasing`
- `unclear`

### 5. Signal volume policy

- Extract up to `5` candidate risk sentences per subheading.
- Pool candidates across all subheadings in a section.
- Keep global top `20` signals per section using ranking by:
  - severity
  - confidence

For the four target sections (`Item 1`, `1A`, `7`, `7A`), this yields up to:

- `80` signals total per filing

### 6. Verification architecture (separate flow)

Run in two stages:

1. Extraction LLM pass
2. Verification LLM pass + deterministic checks

Deterministic checks should validate:

- evidence span exists in source text
- section/subheading linkage is valid
- label values are from allowed vocabularies
- score fields are in valid ranges
- duplicate/near-duplicate cleanup

### 7. Trend alignment philosophy

- Focus on pattern/trend alignment, not only strict date alignment.
- Cross-company comparison should detect similar risk/numeric trajectories even across different calendar periods.

### 8. Numeric data compacting decision

Do not store loose repeated records. Store compact time series per metric:

- metric id
- frequency (`annual` / `quarterly`)
- ordered series of `(period_end, value, accession, tag)`

This is the numeric structure that will be fused with textual signals.

### 9. Dataset plan

- Build a verified/labeled reference dataset of ~15 companies.
- Use same ingestion + extraction + verification flow for those companies and for any new user company.
- Compare new company against this reference set.

### 10. Metadata/versioning

Persist run metadata with outputs, including:

- `schema_version`
- `taxonomy_version`
- `prompt_version`
- `model_name`
- `run_timestamp`
- `run_id`

---

## Immediate Next Implementation Steps

1. Define `risk_taxonomy_v1` (20 labels + definitions + examples).
2. Define `signal_schema_v1` for section/subheading outputs.
3. Implement section->subheading extraction boundaries in code.
4. Implement LLM extraction pass (5 signals per subheading).
5. Implement pooled ranking to top-20 per section.
6. Implement verification pass + deterministic validators.
7. Add numeric compact time-series transformation.
8. Build fusion structure for comparison across companies.
9. Run pilot on 2 companies before scaling to all 15.

---

## Non-Goals (For This Phase)

- No immediate reduction of risk types below 20 unless quality/cost issues appear.
- No implementation of final user-facing comparison UX yet.
- No LLM-only extraction without deterministic verification.

---

## Practical Note

If resumed later in a new context, start by reading:

1. `documentation/data-ingestion-notes.md`
2. `documentation/next-steps-risk-comparison-plan.md` (this file)
3. `data-ingestion/ingestion_pipeline.py`

This should restore full project intent and the agreed roadmap.
