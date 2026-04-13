# curator_agent.py
# Agent 2 (Curator) — reads FinBERT JSON, calls Claude to extract risk signals,
# computes financial deltas, generates BGE-M3 embedding, writes output JSON.

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from curator_models import (
    CuratorOutput, DeltaLabel, FinancialDelta, RiskSignal,
    VALID_SIGNAL_TYPES, Topic
)

# ── Clients ───────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "anthropic/claude-sonnet-4"
MAX_RETRIES        = 3

# BGE-M3 loaded once at module level — downloads ~2GB on first run,
# cached locally after. normalize_embeddings=True required for cosine similarity.
print("Loading BGE-M3 embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-m3")
print("BGE-M3 ready.")


# ── Step 1: Load FinBERT JSON ─────────────────────────────────────────────────

def load_finbert_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Step 2: Section-stratified candidate selection ────────────────────────────
# Investment analysts weight 10-K sections differently:
#   Item 7   — management confirms what actually happened (highest weight)
#   Item 7A  — quantified dollar exposures, most distinctive per slot
#   Item 3   — confirmed active legal proceedings, not speculative
#   Item 1C  — confirmed cybersecurity incidents actually disclosed
#   Item 1A  — forward-looking legal boilerplate, reduced weight
#
# Pure FinBERT score filtering over-represents Item 1A because risk factor
# prose is written to sound alarming. Item 7A quantitative disclosures score
# lower despite being the most distinctive signals for similarity matching.
# Stratified allocation fixes this by guaranteeing cross-section representation.

def get_top_candidates(data: dict, top_n: int = 46) -> list[dict]:
    candidates = data.get("text_candidates", [])

    # Group by section and sort each by risk_score descending
    by_section = defaultdict(list)
    for c in candidates:
        by_section[c.get("section_id", "unknown")].append(c)
    for section in by_section:
        by_section[section].sort(key=lambda x: x["risk_score"], reverse=True)

    # Analytically weighted allocations
    fixed_allocations = {
        "item_7_mda":            12,  # confirmed financial events
        "item_7a_market_risk":   10,  # quantified dollar exposures
        "item_3_legal":           8,  # confirmed legal proceedings
        "item_1c_cybersecurity":  6,  # confirmed cyber incidents
        "item_1a_risk_factors":  10,  # forward-looking, reduced weight
    }

    result = []
    slots_used = 0

    # Fill fixed allocations — skip silently if section not present
    for section, allocation in fixed_allocations.items():
        available = by_section.get(section, [])
        take = min(allocation, len(available))
        result.extend(available[:take])
        slots_used += take

    # Fallback — fill remaining slots from Item 1A with deduplication
    # Triggered when Item 3 or 1C were skipped by FinBERT
    remaining = top_n - slots_used
    if remaining > 0:
        already_included = {id(c) for c in result}
        extras = [
            c for c in by_section.get("item_1a_risk_factors", [])
            if id(c) not in already_included
        ]
        result.extend(extras[:remaining])

    # Print distribution for verification
    dist = defaultdict(int)
    for c in result:
        dist[c.get("section_id", "unknown")] += 1
    # Clean label mapping for readable print output
    section_labels = {
        "item_1a_risk_factors":  "1A",
        "item_7_mda":            "7",
        "item_7a_market_risk":   "7A",
        "item_3_legal":          "3",
        "item_1c_cybersecurity": "1C",
        "unknown":               "unknown",
    }
    print(f"  Candidates: " + ", ".join(
        f"{section_labels.get(s, s)}={n}"
        for s, n in sorted(dist.items())
    ))

    return result


# ── Step 3: Format candidates for prompt ─────────────────────────────────────
# Each candidate includes prev/next sentence for context so Claude can
# write accurate summaries and assign correct severity.

def format_candidates_for_prompt(candidates: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(candidates, start=1):
        section  = c.get("section_label", "Unknown")
        sentence = c.get("sentence_text", "")
        prev     = c.get("previous_sentence") or ""
        nxt      = c.get("next_sentence") or ""
        score    = c.get("risk_score", 0)

        # Only include context if substantive
        context = ""
        if len(prev) > 20:
            context += f" PREV: {prev}"
        if len(nxt) > 20:
            context += f" NEXT: {nxt}"

        blocks.append(f"[{i}] {section} | {score:.2f} | {sentence}{context}")
    return "\n\n".join(blocks)


# ── Step 3: Build extraction prompt ──────────────────────────────────────────
# Concise prompt — every token earns its place.
# Section map, valid values, signal template, and mandatory rules.

def build_extraction_prompt(data: dict, candidates: list[dict]) -> str:
    company     = data.get("company_name", "Unknown")
    ticker      = data.get("ticker", "Unknown")
    filing_year = data.get("filing_year", "Unknown")
    text_block  = format_candidates_for_prompt(candidates)

    section_map = {
        "item_1a_risk_factors":  "Item 1A",
        "item_7_mda":            "Item 7",
        "item_7a_market_risk":   "Item 7A",
        "item_1c_cybersecurity": "Item 1C",
        "item_3_legal":          "Item 3",
        "item_1_business":       "Item 1",
    }
    section_map_str = "\n".join(f"  {k} → {v}" for k, v in section_map.items())
    taxonomy_str    = "\n".join(f"  - {s}" for s in sorted(VALID_SIGNAL_TYPES))
    topic_str       = "\n".join(f"  - {t.value}" for t in Topic)

    signal_template = (
        "SIGNAL TEMPLATE — match exactly:\n"
        "{{\n"
        f'  "signal_id": <integer starting at 1>,\n'
        f'  "topic": <broad category — ONLY one of VALID TOPICS e.g. "market_risk". NEVER put a signal_type value here like "capital_allocation_risk">,\n'
        f'  "signal_type": <specific type — ONLY one of VALID SIGNAL TYPES e.g. "foreign_currency_exposure". NEVER put a topic value here like "market_risk">,\n'
        f'  "section": <"Item 1A" | "Item 7" | "Item 7A" | "Item 1C" | "Item 3">,\n'
        f'  "filing_year": {filing_year},\n'
        f'  "company": "{ticker}",\n'
        f'  "summary": <1 sentence, specific facts only, no boilerplate>,\n'
        f'  "severity": <"high" | "medium" | "low">,\n'
        f'  "citation": "{ticker} 10-K {filing_year}, <section>"\n'
        "}}"
    )

    return f"""You are a financial risk analyst extracting structured risk signals from SEC 10-K filing sentences.

COMPANY: {company} | TICKER: {ticker} | YEAR: {filing_year}

SECTION MAP:
{section_map_str}

VALID TOPICS:
{topic_str}

VALID SIGNAL TYPES:
{taxonomy_str}

SEVERITY:
- high: explicit financial impact, active investigation, going concern, material weakness
- medium: risk acknowledged with potential impact, worsening trend
- low: standard boilerplate with no specific financial impact

SENTENCES:
{text_block}

{signal_template}


MANDATORY RULES:
1. Return ONLY a raw JSON array [ ... ] — no text before or after, no markdown, no code fences
2. PYDANTIC: topic must be exactly one of VALID TOPICS e.g. "market_risk". NEVER use a signal_type value like "capital_allocation_risk" here.
3. PYDANTIC: signal_type must be exactly one of VALID SIGNAL TYPES e.g. "foreign_currency_exposure". NEVER use a topic value like "market_risk" here. They are completely different fields.
4. PYDANTIC: severity must be exactly "high", "medium", or "low" — no capitals, no variations
5. PYDANTIC: summary — 1 sentence, specific facts only. BAD: "could materially adversely affect results" GOOD: "Q4 2021 supplier disruptions caused worldwide sales shortages"
6. Only extract signals grounded in specific language — no inference, no fabrication
7. No duplicate signal_type values — same underlying risk combine at higher severity. Different risks sharing a type use most specific type for each.
8. Signal selection and ordering: ALWAYS include every high severity signal and every signal with a specific dollar amount or percentage. THEN medium signals ordered Item 7 first, Item 7A second, Item 1C third, Item 3 fourth, Item 1A last and only if specific. EXCLUDE low severity unless quantified. Output array in this same order. Do NOT pad. Hard cap 25."""




# ── Step 4: Call Claude with retry ────────────────────────────────────────────
# Each retry sends same token count as attempt 1 — no history passing.
# Only a short error hint is prepended so retries cost the same as attempt 1.

def call_claude_with_retry(
    prompt: str,
    max_retries: int = MAX_RETRIES
) -> list[RiskSignal]:

    last_error = ""

    for attempt in range(1, max_retries + 1):
        print(f"  Claude call attempt {attempt}/{max_retries}...")

        # Prepend short error hint on retry — never pass full history
        current_prompt = prompt if attempt == 1 else (
            f"IMPORTANT: Previous attempt failed — {last_error}\n"
            f"Fix only that issue and return corrected JSON array.\n\n"
            + prompt
        )

        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model":      MODEL,
                "max_tokens": 16000,
                "messages":   [{"role": "user", "content": current_prompt}],
            })
        )

        resp_json = response.json()

        if "error" in resp_json:
            last_error = f"API error: {resp_json['error']}"
            print(f"  OpenRouter error: {resp_json['error']}")
            time.sleep(1)
            continue

        if "choices" not in resp_json:
            last_error = f"Unexpected response shape: {resp_json}"
            print(f"  Unexpected response: {resp_json}")
            time.sleep(1)
            continue

        raw_text = resp_json["choices"][0]["message"]["content"].strip()
        print(f"  Response length: {len(raw_text)} chars")
        print(f"  Last 200 chars: {raw_text[-200:]}")

        # Strip markdown fences if Claude added them
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            print(f"  JSON parse failed: {e}")
            time.sleep(1)
            continue

        signals = []
        validation_errors = []
        for item in parsed:
            try:
                signals.append(RiskSignal(**item))
            except ValidationError as e:
                validation_errors.append(str(e))

        if validation_errors:
            last_error = validation_errors[0][:300]
            print(f"  Pydantic failed. First error: {last_error}")
            time.sleep(1)
            continue

        print(f"  Extracted {len(signals)} valid signals.")
        return signals

    print(f"  WARNING: All {max_retries} attempts failed. Returning empty signals.")
    return []


# ── Step 5: Compute financial deltas ─────────────────────────────────────────
# Reads numeric_deltas from FinBERT JSON and applies severity bucketing.
# delta_percent is null for earliest year (no prior year) → insufficient_data.

def compute_financial_deltas(data: dict) -> dict[str, FinancialDelta]:
    result = {}
    for metric, info in data.get("numeric_deltas", {}).items():
        delta_pct = info.get("delta_percent")

        if delta_pct is None:
            label, value = DeltaLabel.insufficient_data, None
        else:
            value = round(delta_pct / 100, 4)
            if delta_pct > 20:
                label = DeltaLabel.strong_growth
            elif delta_pct > 5:
                label = DeltaLabel.moderate_growth
            elif delta_pct >= -5:
                label = DeltaLabel.stable
            elif delta_pct >= -20:
                label = DeltaLabel.moderate_decline
            else:
                label = DeltaLabel.severe_decline

        result[metric] = FinancialDelta(value=value, label=label)
    return result


# ── Step 6: Build embedding text ──────────────────────────────────────────────
# Combines financial delta labels and signal summaries into one text block.
# Summaries are included so BGE-M3 captures semantic risk context not just
# signal type labels — AAPL supply chain ≠ SNAP supply chain in vector space.
# Signals capped at 20 to keep embedding text at reasonable token length.

def build_embedding_text(
    ticker: str,
    filing_year: int,
    deltas: dict[str, FinancialDelta],
    signals: list[RiskSignal],
) -> str:
    delta_parts = ", ".join(
        f"{metric}: {d.label.value}" for metric, d in deltas.items()
    )
    signal_parts = "; ".join(
        f"{s.signal_type} ({s.severity.value}): {s.summary}"
        for s in signals[:20]
    ) or "no signals extracted"

    return (
        f"Company: {ticker}, Year: {filing_year}. "
        f"Financial: {delta_parts}. "
        f"Risk signals: {signal_parts}."
    )


# ── Step 7: Generate embedding vector ────────────────────────────────────────
# BGE-M3 produces 1024-float vector. normalize_embeddings=True required
# so FAISS dot product equals cosine similarity.
# Tell Om: FAISS index dimension is 1024 → faiss.IndexFlatIP(1024)

def generate_embedding(text: str) -> list[float]:
    return embedding_model.encode(
        text,
        normalize_embeddings=True
    ).tolist()


# ── Step 8: Write output file ─────────────────────────────────────────────────
# Saves complete CuratorOutput as {ticker_lowercase}_{year}.json

def write_output(output: CuratorOutput, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{output.ticker.lower()}_{output.filing_year}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(mode="json"), f, indent=2)
    print(f"  Wrote: {filepath}")
    return filepath


# ── Main entry point ──────────────────────────────────────────────────────────
# Orchestrates all 8 steps for one FinBERT JSON file.

def run_curator(
    finbert_json_path: str,
    output_dir: str = "outputs/curator",
    top_n: int = 46,
) -> Optional[CuratorOutput]:

    print(f"\nRunning Agent 2 on: {finbert_json_path}")

    # Step 1 — load
    data        = load_finbert_json(finbert_json_path)
    ticker      = data.get("ticker", "UNKNOWN")
    company     = data.get("company_name", ticker)
    filing_year = data.get("filing_year", 0)
    print(f"  Company: {company} | Year: {filing_year}")

    # Step 2 — stratified candidate selection
    candidates = get_top_candidates(data, top_n=top_n)

    # Step 3 + 4 — build prompt and call Claude
    prompt  = build_extraction_prompt(data, candidates)
    signals = call_claude_with_retry(prompt)

    # Post-process — sort by section priority then severity
    # Ensures embedding text weights confirmed events over boilerplate
    section_order  = {"Item 7": 0, "Item 7A": 1, "Item 1C": 2, "Item 3": 3, "Item 1A": 4}
    severity_order = {"high": 0, "medium": 1, "low": 2}
    signals = sorted(
        signals,
        key=lambda s: (
            section_order.get(s.section, 5),
            severity_order.get(s.severity.value, 3)
        )
    )
    # Reassign sequential signal_ids after sort
    for i, signal in enumerate(signals, start=1):
        signal.signal_id = i

    # Step 5 — compute financial deltas
    deltas = compute_financial_deltas(data)
    print(f"  Deltas: {len(deltas)} metrics")

    # Step 6 — build embedding text
    embedding_text = build_embedding_text(ticker, filing_year, deltas, signals)

    # Step 7 — generate BGE-M3 embedding vector (1024 floats)
    print("  Generating embedding...")
    embedding_vector = generate_embedding(embedding_text)

    # Assemble and validate full output
    output = CuratorOutput(
        company=company,
        ticker=ticker,
        filing_year=filing_year,
        financial_deltas=deltas,
        risk_signals=signals,
        embedding_text=embedding_text,
        embedding_vector=embedding_vector,
    )

    # Step 8 — write to disk
    write_output(output, output_dir)
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────
# python curator_agent.py <finbert_json_path> [output_dir] [top_n]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python curator_agent.py <finbert_json> [output_dir] [top_n]")
        sys.exit(1)

    path    = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data-extraction/outputs/curator"
    n       = int(sys.argv[3]) if len(sys.argv) > 3 else 46

    run_curator(path, output_dir=out_dir, top_n=n)