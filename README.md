# FinAgent

FinAgent is a local SEC 10-K analysis system that builds structured financial comparison reports from public filings. It implements two pipelines — a multi-step agentic system and a simpler baseline RAG — and provides a side-by-side comparison UI to evaluate the quality difference between them. The project is designed as a research tool for studying whether agentic pipeline architecture produces richer financial analysis than naive single-shot retrieval.

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Available Tickers](#available-tickers)
- [Running the UI](#running-the-ui)
- [Agentic Pipeline](#agentic-pipeline)
- [Baseline RAG Pipeline](#baseline-rag-pipeline)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Authors](#authors)

---

## Architecture

![Architecture Diagram](documentation/Arch%20Diag.drawio.png)

The system is organized into four layers:

1. **Data Preparation** — deterministic SEC ingestion, yearly extraction, and curator artifact generation with embeddings. No LLM is used here.
2. **Retrieval** — FAISS-based peer retrieval over curator embeddings to find similar companies.
3. **Report Generation** — either the multi-step agentic pipeline or the single-shot baseline RAG, both producing the same structured output schema.
4. **Evaluation** — offline scoring of saved report artifacts on consistency, evidence coverage, and comparative usefulness.

**Key design choice:** Both pipelines use the same LLM and the same output schema (`OrchestrationArtifact`). This isolates pipeline architecture as the variable, not model capability.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env in the project root and fill in your keys

# 3. Launch the UI
streamlit run ui/app.py
```

Select a ticker from the dropdown, click **Run Agentic Pipeline** and **Run Baseline RAG**, then open the **Side by Side** tab.

---

## Setup

### Python Version

Python 3.10 or later is required.

### Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `streamlit`, `faiss-cpu`, `sentence-transformers`, `torch`, `pydantic`, `sec-edgar-downloader`, `requests`, `plotly`.

### Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4.1-mini        # optional, this is the default
SEC_API_USER_AGENT=Your Name your@email.com  # required by SEC EDGAR
```

**Getting an OpenRouter API key:**
- Sign up at [openrouter.ai](https://openrouter.ai)
- Go to **API Keys** and create a new key
- Add a small credit ($5 is sufficient for testing)
- The default model `openai/gpt-4.1-mini` is cost-efficient for development

**SEC User Agent:**
The SEC EDGAR API requires a valid user agent string in the format `Name email@address.com`. Use your real name and email — SEC rate-limits by user agent.

---

## Available Tickers

The following 16 companies have pre-ingested data and are available immediately in the UI:

| Ticker | Company |
|--------|---------|
| AAPL | Apple Inc. |
| ADBE | Adobe Inc. |
| AMD | Advanced Micro Devices |
| AMZN | Amazon.com Inc. |
| BYND | Beyond Meat Inc. |
| CRM | Salesforce Inc. |
| GOOG | Alphabet Inc. |
| INTC | Intel Corporation |
| LYFT | Lyft Inc. |
| META | Meta Platforms Inc. |
| MSFT | Microsoft Corporation |
| NVDA | NVIDIA Corporation |
| PLTR | Palantir Technologies |
| QCOM | Qualcomm Incorporated |
| SNAP | Snap Inc. |
| UBER | Uber Technologies |

Many of these tickers also have curator artifacts checked in under `data-extraction/outputs/curator/`, which enables FAISS-based peer matching immediately. If curator artifacts are missing for a ticker, the UI can build them from the **Dataset Management** panel, and the Baseline RAG path can still fall back to ingestion-only peer selection.

To add a new ticker, use the **Dataset Management** panel in the UI or run the pipeline steps manually (see [Agentic Pipeline](#agentic-pipeline)).

---

## Running the UI

```bash
streamlit run ui/app.py
```

The UI runs locally and connects directly to the Python pipeline modules — no server required.

### Sidebar

| Control | Description |
|---------|-------------|
| **Ticker** | Dropdown of available tickers with ingestion data |
| **Top K Peers** | Number of peer companies to retrieve via FAISS |
| **Run Agentic Pipeline** | Runs the full multi-step pipeline for the selected ticker |
| **Focus Area** | Optional text to steer the Baseline RAG analysis (e.g. "margin compression") |
| **Run Baseline RAG** | Runs the single-shot RAG pipeline |
| **Clear Results** | Resets both pipeline results from session |
| **Load Saved Report** | Load a previously saved agentic artifact into either slot |
| **Build Dataset Artifacts** | Run ingestion → extraction → curator for a new ticker |
| **Rebuild FAISS Index** | Rebuild the peer retrieval index from all curator files |

### Main Tabs

**Side by Side** (appears when both pipelines are loaded)
- Pipeline Quality Radar — spider chart comparing output richness across 6 dimensions
- Risk Severity Distribution — grouped bar showing High/Medium/Low risk counts per pipeline
- Risk Identification Diff — which risks each pipeline found
- Risk Overlap by Group — shared/target-only/peer-only risk breakdown for each pipeline
- Peer Match Comparison — similarity scores
- Posture disagreement warning when pipelines reach different conclusions
- Full reports in aligned parallel columns (each section exactly beside its counterpart)

**Agentic Pipeline / Baseline RAG** (individual report tabs)
- Financial Metrics bar chart — key metrics for the most recent fiscal year
- Multi-Year Trend line chart — Revenue, Net Income, Gross Profit, OCF across all available years
- Posture, target profile, peer matches with similarity progress bars
- Risk overlap and forward watchlist with severity/confidence color badges
- Evidence-aware narrative with citations

**Dataset Status**
- Ingestion, extraction, and curator artifact availability for the selected ticker
- FAISS index status and entry count

**Export**
- Download both pipeline results as a single comparison JSON file when both are loaded

---

## Agentic Pipeline

The agentic pipeline is the main end-to-end path. Data preparation is deterministic; OpenRouter is only used at the final report generation stage.

### Step 1 — Ingestion

Fetches structured financial data and 10-K filing sections from SEC EDGAR.

```bash
python data-ingestion/ingestion_pipeline.py AAPL --years 5
```

Output: `data-ingestion/outputs/AAPL/complete_ingestion.json`

Contains: company identity, annual/quarterly financial metrics, filing metadata, extracted 10-K text sections.

### Step 2 — Yearly Extraction

Reads the ingestion artifact and produces yearly extraction files combining numeric deltas with ranked text candidates and FinBERT-scored risk signals.

```bash
python data-extraction/main.py AAPL
```

Output: `data-extraction/outputs/AAPL/aapl_<year>_extraction.json`

### Step 3 — Curator Artifacts

Converts yearly extraction output into retrieval-ready artifacts with financial delta labels, curated risk signals, embedding text, and embedding vectors.

```bash
python data-extraction/company_filing_embedding.py AAPL
```

Output: `data-extraction/outputs/curator/AAPL/aapl_<year>.json`

### Step 4 — Peer Retrieval

Builds a FAISS index over all curator embeddings and retrieves similar companies for a target.

```bash
# Build the index
python rag-matching/indexer.py

# Query for peers
python rag-matching/matcher.py --input-file data-extraction/outputs/curator/AAPL/aapl_2025.json --top 2 --json
```

### Step 5 — Report Generation

Orchestrates all previous steps (skipping any that already exist), retrieves top peer companies, assembles multi-year context, and generates a structured comparison report via a single OpenRouter call.

```bash
python orchestration/runner.py AAPL --top 2 --json
```

Output: `orchestration/outputs/AAPL/aapl_comparison_bundle.json`

---

## Baseline RAG Pipeline

The baseline exists as a deliberate contrast to the agentic pipeline for research comparison purposes.

**What it does differently:**
- Reads ingestion data directly — skips extraction and curator reasoning depth
- No NLP signals, no FinBERT scoring, no multi-year curator context
- Single LLM call with flattened financial metrics as context
- Optional focus query to steer analysis without changing retrieval

**What stays the same:**
- Same LLM and model via OpenRouter
- Same FAISS retrieval when curator files exist for the target
- Same output schema (`OrchestrationArtifact` with `schema_version = "baseline_rag_v1"`)

This design means any quality difference in the Side by Side comparison is attributable to pipeline architecture, not model capability.

Main module: `baseline_rag/pipeline.py`

---

## Evaluation

The evaluation module scores saved report artifacts offline without re-running pipelines.

```bash
python -m evaluation.runner \
  --agentic-dir orchestration/outputs \
  --json
```

Output: `evaluation/outputs/<run_name>.json`

If you have saved baseline artifacts from a separate run, you can add `--baseline-dir <path>` or `--baseline-artifact <path>`. The Streamlit UI keeps baseline results in session state by default and does not write them to disk automatically.

### Scorecard

| Metric | Description |
|--------|-------------|
| `deterministic_consistency` | Internal consistency of structured fields |
| `evidence_coverage` | How well citations cover risk and narrative claims |
| `claim_support` | Whether quantitative claims are grounded in source data |
| `comparative_usefulness` | Quality of peer comparison relative to matched companies |
| `overreach_penalty` | Penalizes fabricated or unsupported claims |

**Judge mode** (requires OpenRouter): adds LLM-backed claim and report quality scoring with on-disk caching.

```bash
python -m evaluation.runner --agentic-dir orchestration/outputs --judge --json
```

Deterministic-only mode works entirely offline from local files.

---

## Project Structure

```
FinAgent/
├── data-ingestion/         # SEC EDGAR ingestion pipeline
│   └── outputs/            # Ingestion artifacts by ticker
├── data-extraction/        # Yearly extraction + curator generation
│   └── outputs/
│       ├── <TICKER>/       # Yearly extraction JSON files
│       └── curator/        # Curator artifacts with embeddings
├── rag-matching/           # FAISS index and peer matcher
│   └── index_artifacts/    # Saved FAISS index and metadata
├── orchestration/          # Agentic comparison pipeline
│   ├── report_models.py    # Shared output schema (Pydantic)
│   ├── openrouter_client.py
│   ├── orchestration_pipeline.py
│   ├── comparison_agent.py
│   └── outputs/            # Saved agentic comparison bundles
├── baseline_rag/           # Single-shot RAG baseline
│   └── pipeline.py
├── evaluation/             # Offline evaluation and scoring
│   └── outputs/
├── ui/                     # Streamlit UI
│   ├── app.py
│   ├── services.py
│   └── display.py          # Chart and table builders
├── documentation/          # Architecture diagrams and notes
├── .env                    # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'faiss'`**
```bash
pip install faiss-cpu
```

**`OPENROUTER_API_KEY is not set`**
Add `OPENROUTER_API_KEY=your_key` to your `.env` file and restart Streamlit. Streamlit caches the environment at startup — a page refresh is not enough, a full restart is required.

**`402 Payment Required` from OpenRouter**
Your OpenRouter account is out of credits. Add credits at [openrouter.ai](https://openrouter.ai) under **Credits**.

**`Port 8501 is already in use`**
Another Streamlit instance is running. Either kill it or use a different port:
```bash
streamlit run ui/app.py --server.port 8502
```

**`SyntaxError` or `TypeError` around `X | None` annotations**
You are likely running Python older than 3.10. This project now uses modern union-type syntax across the active codebase, so upgrade to Python 3.10+ and reinstall dependencies.

**Baseline RAG shows `Matches Found: 0`**
The FAISS index needs to be rebuilt. Click **Rebuild FAISS Index** in the Dataset Management sidebar, or run:
```bash
python rag-matching/indexer.py
```

**Agentic pipeline is slow on first run**
The first run for a ticker downloads and processes 10-K filings from SEC EDGAR and runs FinBERT scoring. Subsequent runs skip completed steps automatically.

---

## Authors

- **Sharan Giri** - giri.sha@northeastern.edu 
- **Sumit Kanu** - kanu.s@northeastern.edu
- **Om Mane** - mane.om@northeastern.edu

*CS6180 — Generative AI, Spring 2026*

---

*Built with [Streamlit](https://streamlit.io), [FAISS](https://github.com/facebookresearch/faiss), [OpenRouter](https://openrouter.ai), and [SEC EDGAR](https://www.sec.gov/edgar).*
