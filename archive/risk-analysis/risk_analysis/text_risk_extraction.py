from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from .models import RiskSignalV1, SectionRiskResultV1, SectionRiskRollupV1
from .taxonomy import TAXONOMY_BY_ID, TAXONOMY_VERSION


SECTION_NAMES = {
    "item_1a_risk_factors": "Item 1A. Risk Factors",
    "item_1c_cybersecurity": "Item 1C. Cybersecurity",
    "item_3_legal": "Item 3. Legal Proceedings",
    "item_7_mda": "Item 7. Management's Discussion and Analysis",
    "item_7a_market_risk": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
}


RISK_KEYWORDS = {
    "macroeconomic_demand": ["demand", "advertiser", "spending", "economy", "macroeconomic", "recession", "consumer"],
    "competition_market_share": ["competition", "competitor", "market share", "pricing pressure"],
    "regulatory_legal": ["regulation", "regulatory", "antitrust", "compliance", "law", "government"],
    "cybersecurity_privacy": ["cybersecurity", "security", "privacy", "data breach", "attack", "incident"],
    "technology_platform_change": ["technology", "platform", "artificial intelligence", "transition", "innovation", "ai"],
    "supply_chain_operations": ["supply", "supplier", "shortage", "manufacturing", "operations", "logistics"],
    "customer_concentration": ["customer", "partner", "distribution partner", "concentration"],
    "advertising_monetization": ["advertising", "monetization", "tac", "cost-per-click", "impressions"],
    "product_execution": ["launch", "product", "quality", "delay", "roadmap"],
    "talent_workforce": ["employee", "talent", "hiring", "retention", "workforce"],
    "capital_investment_margin": ["margin", "capital", "investment", "cost", "infrastructure", "capex"],
    "international_fx": ["foreign currency", "exchange rate", "international", "emerging markets", "global"],
    "content_ip": ["intellectual property", "license", "copyright", "patent", "trademark"],
    "reputation_brand_trust": ["trust", "brand", "reputation", "public perception"],
    "dependency_third_party_platform": ["browser", "app store", "platform", "default search", "third-party"],
    "financial_liquidity_balance_sheet": ["debt", "liquidity", "cash", "capital resources", "financial flexibility"],
    "governance_controls": ["internal control", "disclosure controls", "material weakness", "governance"],
    "litigation_contingency": ["litigation", "claim", "lawsuit", "settlement", "contingency"],
    "market_volatility_rates": ["interest rate", "marketable securities", "volatility", "investment"],
    "ai_model_risk": ["artificial intelligence", "model", "responsible ai", "hallucination", "ai"],
}


SEVERITY_TERMS = {
    5: ["materially", "severe", "significant", "substantial", "adverse", "critical"],
    4: ["could harm", "could adversely", "may materially", "material"],
    3: ["may", "could", "might", "risk"],
}


@dataclass
class ExtractedCandidate:
    filing_year: int
    filing_date: str
    accession: str
    section_id: str
    subheading: str
    risk_types: List[str]
    severity: int
    direction: str
    confidence: float
    evidence_text: str
    judge_rationale: str
    source_rank: int


class BaseRiskLLMClient:
    model_name = "heuristic-v1"
    provider_name = "heuristic"

    def extract_candidates(
        self,
        *,
        filing_year: int,
        filing_date: str,
        accession: str,
        section_id: str,
        subheading: str,
        chunk_text: str,
        source_rank_offset: int,
    ) -> List[ExtractedCandidate]:
        raise NotImplementedError

    def verify_candidates(self, *, section_text: str, candidates: Sequence[ExtractedCandidate]) -> List[ExtractedCandidate]:
        raise NotImplementedError

    def debug_summary(self) -> str:
        return f"{self.provider_name}:{self.model_name}"


class HeuristicRiskLLMClient(BaseRiskLLMClient):
    model_name = "heuristic-v1"
    provider_name = "heuristic"

    def extract_candidates(
        self,
        *,
        filing_year: int,
        filing_date: str,
        accession: str,
        section_id: str,
        subheading: str,
        chunk_text: str,
        source_rank_offset: int,
    ) -> List[ExtractedCandidate]:
        sentences = _split_sentences(chunk_text)
        ranked: List[Tuple[float, ExtractedCandidate]] = []
        for idx, sentence in enumerate(sentences):
            if len(sentence) < 60:
                continue
            risk_types = _match_risk_types(section_id, sentence)
            if not risk_types:
                continue
            confidence = _sentence_confidence(sentence, risk_types)
            severity = _sentence_severity(sentence)
            ranked.append(
                (
                    severity + confidence,
                    ExtractedCandidate(
                        filing_year=filing_year,
                        filing_date=filing_date,
                        accession=accession,
                        section_id=section_id,
                        subheading=subheading,
                        risk_types=risk_types,
                        severity=severity,
                        direction=_infer_direction(sentence),
                        confidence=confidence,
                        evidence_text=sentence,
                        judge_rationale="Heuristic extractor matched risk keywords and severity language in the source sentence.",
                        source_rank=source_rank_offset + idx,
                    ),
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1].source_rank))
        return [candidate for _, candidate in ranked[:5]]

    def verify_candidates(self, *, section_text: str, candidates: Sequence[ExtractedCandidate]) -> List[ExtractedCandidate]:
        verified: List[ExtractedCandidate] = []
        seen = []
        for candidate in candidates:
            evidence = candidate.evidence_text.strip()
            if evidence not in section_text:
                continue
            if not candidate.risk_types:
                continue
            if candidate.direction not in {"increasing", "stable", "decreasing", "unclear"}:
                continue
            if any(_near_duplicate(evidence, prior.evidence_text) for prior in seen):
                continue
            verified.append(
                ExtractedCandidate(
                    filing_year=candidate.filing_year,
                    filing_date=candidate.filing_date,
                    accession=candidate.accession,
                    section_id=candidate.section_id,
                    subheading=candidate.subheading,
                    risk_types=candidate.risk_types,
                    severity=candidate.severity,
                    direction=candidate.direction,
                    confidence=round(min(1.0, candidate.confidence + 0.1), 3),
                    evidence_text=evidence,
                    judge_rationale="Heuristic verifier confirmed the evidence span exists in the section and removed duplicates.",
                    source_rank=candidate.source_rank,
                )
            )
            seen.append(candidate)
        return verified


class OpenRouterRiskLLMClient(BaseRiskLLMClient):
    provider_name = "openrouter"

    def __init__(self, model_name: Optional[str] = None, *, debug_dir: Optional[Path] = None):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.model_name = model_name or os.getenv("MODEL_NAME")
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for provider=openrouter")
        if not self.base_url:
            raise ValueError("OPENROUTER_BASE_URL is required for provider=openrouter")
        if not self.model_name:
            raise ValueError("MODEL_NAME is required for provider=openrouter unless --model is provided")

    def extract_candidates(
        self,
        *,
        filing_year: int,
        filing_date: str,
        accession: str,
        section_id: str,
        subheading: str,
        chunk_text: str,
        source_rank_offset: int,
    ) -> List[ExtractedCandidate]:
        payload = self._call_chat_completion(
            system_prompt=_extraction_system_prompt(),
            user_prompt=_build_extraction_prompt(filing_year, section_id, subheading, chunk_text),
            debug_label=f"{accession}_{section_id}_extract_{source_rank_offset}",
        )
        candidates: List[ExtractedCandidate] = []
        for idx, item in enumerate(_normalize_candidates_payload(payload)[:5]):
            risk_types = _candidate_to_risk_types(item)
            if not risk_types:
                continue
            evidence_text = (
                item.get("evidence_text")
                or item.get("evidence")
                or item.get("evidence_span")
                or item.get("sentence")
                or ""
            ).strip()
            if not evidence_text:
                continue
            candidates.append(
                ExtractedCandidate(
                    filing_year=filing_year,
                    filing_date=filing_date,
                    accession=accession,
                    section_id=section_id,
                    subheading=subheading,
                    risk_types=risk_types,
                    severity=_safe_int(item.get("severity"), default=3),
                    direction=str(item.get("direction", "unclear")).strip().lower() or "unclear",
                    confidence=_safe_float(item.get("confidence"), default=0.5),
                    evidence_text=evidence_text,
                    judge_rationale=item.get("rationale", "").strip() or "OpenRouter extraction pass.",
                    source_rank=source_rank_offset + idx,
                )
            )
        return candidates

    def verify_candidates(self, *, section_text: str, candidates: Sequence[ExtractedCandidate]) -> List[ExtractedCandidate]:
        payload = self._call_chat_completion(
            system_prompt=_verification_system_prompt(),
            user_prompt=_build_verification_prompt(section_text, candidates),
            debug_label=f"{candidates[0].accession if candidates else 'no_candidates'}_{candidates[0].section_id if candidates else 'section'}_verify",
        )
        keep_ids = set(_normalize_keep_source_ranks(payload.get("keep_source_ranks", [])))
        rationales = payload.get("rationales", {})
        if not isinstance(rationales, dict):
            rationales = {}
        verified: List[ExtractedCandidate] = []
        for candidate in candidates:
            if candidate.source_rank not in keep_ids:
                continue
            verified.append(
                ExtractedCandidate(
                    filing_year=candidate.filing_year,
                    filing_date=candidate.filing_date,
                    accession=candidate.accession,
                    section_id=candidate.section_id,
                    subheading=candidate.subheading,
                    risk_types=candidate.risk_types,
                    severity=candidate.severity,
                    direction=candidate.direction,
                    confidence=candidate.confidence,
                    evidence_text=candidate.evidence_text,
                    judge_rationale=rationales.get(str(candidate.source_rank), "OpenRouter verification pass."),
                    source_rank=candidate.source_rank,
                )
            )
        return verified

    def _call_chat_completion(self, *, system_prompt: str, user_prompt: str, debug_label: str) -> Dict:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost",
                "X-Title": "FinAgent Risk Analysis",
            },
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        self._dump_debug_file(f"{debug_label}_response.json", payload)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Malformed OpenRouter response shape") from exc
        self._dump_debug_file(f"{debug_label}_content.txt", content)
        return _parse_json_content(content)

    def _dump_debug_file(self, filename: str, content) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / _sanitize_filename(filename)
        if isinstance(content, (dict, list)):
            path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(str(content), encoding="utf-8")


def build_llm_client(provider: str, model_name: Optional[str], *, debug_dir: Optional[Path] = None) -> BaseRiskLLMClient:
    if provider.lower() == "openrouter":
        return OpenRouterRiskLLMClient(model_name=model_name, debug_dir=debug_dir)
    return HeuristicRiskLLMClient()


def analyze_filings(
    filings: Sequence[Dict],
    *,
    section_ids: Sequence[str],
    llm_client: BaseRiskLLMClient,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict[str, SectionRiskRollupV1]]:
    filing_analysis: List[Dict] = []
    for filing in filings:
        year = _filing_year(filing)
        if verbose:
            print(f"[risk-analysis] Filing {year} {filing.get('accession')} with {llm_client.debug_summary()}")
        section_results: Dict[str, SectionRiskResultV1] = {}
        filing_errors: List[Dict[str, str]] = []
        for section_id in section_ids:
            section_text = (filing.get("sections", {}) or {}).get(section_id, "") or ""
            result = _analyze_section(
                filing=filing,
                filing_year=year,
                section_id=section_id,
                section_text=section_text,
                llm_client=llm_client,
                verbose=verbose,
            )
            section_results[section_id] = result
            if result.processing_error:
                filing_errors.append({"section_id": section_id, "error": result.processing_error})
            if verbose:
                print(
                    f"[risk-analysis]   {section_id}: chunks={result.chunk_count} "
                    f"candidates={result.candidate_count} kept={result.kept_signal_count} "
                    f"error={'none' if not result.processing_error else result.processing_error}"
                )
        filing_analysis.append(
            {
                "filing_year": year,
                "filing_date": filing.get("filingDate"),
                "accession": filing.get("accession"),
                "parser_mode": filing.get("parser_mode"),
                "filing_errors": filing_errors,
                "section_results": {key: value.to_dict() for key, value in section_results.items()},
            }
        )
    return filing_analysis, build_section_rollups(filing_analysis, section_ids)


def _analyze_section(
    *,
    filing: Dict,
    filing_year: int,
    section_id: str,
    section_text: str,
    llm_client: BaseRiskLLMClient,
    verbose: bool = False,
) -> SectionRiskResultV1:
    notes: List[str] = []
    if not section_text.strip():
        notes.append("Section text is empty in the source artifact.")
        return SectionRiskResultV1(
            section_id=section_id,
            section_name=SECTION_NAMES[section_id],
            filing_year=filing_year,
            filing_date=filing.get("filingDate", ""),
            accession=filing.get("accession", ""),
            parser_mode=filing.get("parser_mode"),
            llm_provider=getattr(llm_client, "provider_name", None),
            llm_model=getattr(llm_client, "model_name", None),
            processing_error=None,
            subheading_count=0,
            chunk_count=0,
            candidate_count=0,
            kept_signal_count=0,
            notes=notes,
            signals=[],
        )

    subheadings = segment_section(section_text)
    if not subheadings:
        subheadings = chunk_section_text(section_text)
        notes.append("Fell back to paragraph chunking because no stable subheadings were found.")
    else:
        subheadings = _merge_small_chunks(subheadings, section_id=section_id)

    if verbose:
        print(
            f"[risk-analysis]     start {section_id}: chars={len(section_text)} "
            f"subheading_chunks={len(subheadings)}"
        )

    candidates: List[ExtractedCandidate] = []
    source_rank_offset = 0
    processing_error: Optional[str] = None
    rejected_candidate_reasons: List[str] = []
    try:
        for heading, chunk_text in subheadings:
            if verbose:
                print(
                    f"[risk-analysis]       chunk rank={source_rank_offset} heading={heading[:60]!r} chars={len(chunk_text)}"
                )
            extracted = llm_client.extract_candidates(
                filing_year=filing_year,
                filing_date=filing.get("filingDate", ""),
                accession=filing.get("accession", ""),
                section_id=section_id,
                subheading=heading,
                chunk_text=chunk_text,
                source_rank_offset=source_rank_offset,
            )
            candidates.extend(extracted)
            source_rank_offset += max(1, len(_split_sentences(chunk_text)))
        if not candidates:
            notes.append("Extraction returned zero accepted candidates for this section.")
        verified = llm_client.verify_candidates(section_text=section_text, candidates=candidates)
        kept_signals = _finalize_signals(section_text, verified)
    except Exception as exc:
        processing_error = str(exc)
        notes.append("Model-backed section processing failed; no signals were kept for this section.")
        kept_signals = []
    if rejected_candidate_reasons:
        notes.extend(rejected_candidate_reasons[:5])
    return SectionRiskResultV1(
        section_id=section_id,
        section_name=SECTION_NAMES[section_id],
        filing_year=filing_year,
        filing_date=filing.get("filingDate", ""),
        accession=filing.get("accession", ""),
        parser_mode=filing.get("parser_mode"),
        llm_provider=getattr(llm_client, "provider_name", None),
        llm_model=getattr(llm_client, "model_name", None),
        processing_error=processing_error,
        subheading_count=len(subheadings),
        chunk_count=len(subheadings),
        candidate_count=len(candidates),
        kept_signal_count=len(kept_signals),
        notes=notes,
        signals=kept_signals,
    )


def segment_section(section_text: str) -> List[Tuple[str, str]]:
    lines = [line.strip() for line in section_text.splitlines()]
    segments: List[Tuple[str, List[str]]] = []
    current_heading = "Section Overview"
    current_buffer: List[str] = []
    heading_count = 0
    for line in lines:
        if not line:
            if current_buffer and current_buffer[-1] != "":
                current_buffer.append("")
            continue
        if _looks_like_subheading(line):
            if current_buffer:
                segments.append((current_heading, current_buffer))
            current_heading = line
            current_buffer = []
            heading_count += 1
            continue
        current_buffer.append(line)
    if current_buffer:
        segments.append((current_heading, current_buffer))
    if heading_count < 2:
        return []
    return [(heading, _join_lines(buffer)) for heading, buffer in segments if _join_lines(buffer).strip()]


def chunk_section_text(section_text: str, *, max_paragraphs: int = 4) -> List[Tuple[str, str]]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", section_text) if paragraph.strip()]
    if not paragraphs:
        return []
    return [
        (f"Chunk {idx // max_paragraphs + 1}", "\n\n".join(paragraphs[idx : idx + max_paragraphs]))
        for idx in range(0, len(paragraphs), max_paragraphs)
    ]


def _merge_small_chunks(chunks: Sequence[Tuple[str, str]], *, section_id: str) -> List[Tuple[str, str]]:
    if not chunks:
        return []
    max_chunks = 12 if section_id == "item_7_mda" else 10
    min_chars = 1200 if section_id == "item_7_mda" else 900
    max_chars = 5000 if section_id == "item_7_mda" else 4000

    merged: List[Tuple[str, str]] = []
    current_headings: List[str] = []
    current_parts: List[str] = []
    current_chars = 0

    for heading, text in chunks:
        text = text.strip()
        if not text:
            continue
        projected = current_chars + len(text)
        if current_parts and (projected > max_chars or (len(merged) + 1 < max_chunks and current_chars >= min_chars)):
            merged.append((_combined_heading(current_headings), "\n\n".join(current_parts).strip()))
            current_headings = []
            current_parts = []
            current_chars = 0
        current_headings.append(heading)
        current_parts.append(text)
        current_chars += len(text)

    if current_parts:
        merged.append((_combined_heading(current_headings), "\n\n".join(current_parts).strip()))

    while len(merged) > max_chunks:
        merged = _coalesce_adjacent_chunks(merged)
    return merged


def build_section_rollups(filing_analysis: Sequence[Dict], section_ids: Iterable[str]) -> Dict[str, SectionRiskRollupV1]:
    rollups: Dict[str, SectionRiskRollupV1] = {}
    for section_id in section_ids:
        signals: List[RiskSignalV1] = []
        years: List[int] = []
        for filing in filing_analysis:
            section = filing["section_results"][section_id]
            years.append(filing["filing_year"])
            signals.extend(_dict_to_signal(item) for item in section["signals"])
        risk_type_counts = Counter(risk_type for signal in signals for risk_type in signal.risk_types)
        recurring = [risk_type for risk_type, count in risk_type_counts.items() if count >= 2]
        representative = [
            {
                "filing_year": signal.filing_year,
                "risk_types": signal.risk_types,
                "severity": signal.severity,
                "confidence": signal.confidence,
                "evidence_text": signal.evidence_text,
            }
            for signal in sorted(signals, key=lambda item: (-item.severity, -item.confidence, item.source_rank))[:5]
        ]
        notes = []
        if not signals:
            notes.append("No verified signals were kept for this section across the available filings.")
        rollups[section_id] = SectionRiskRollupV1(
            section_id=section_id,
            section_name=SECTION_NAMES[section_id],
            years_covered=sorted(set(years)),
            recurring_risk_types=sorted(recurring),
            representative_signals=representative,
            severity_trend=_severity_trend(signals),
            contributing_years=sorted({signal.filing_year for signal in signals}),
            notes=notes,
        )
    return rollups


def _finalize_signals(section_text: str, candidates: Sequence[ExtractedCandidate]) -> List[RiskSignalV1]:
    final: List[RiskSignalV1] = []
    seen_texts: List[str] = []
    for candidate in sorted(candidates, key=lambda item: (-item.severity, -item.confidence, item.source_rank)):
        evidence = candidate.evidence_text.strip()
        if evidence not in section_text:
            continue
        if any(_near_duplicate(evidence, prior) for prior in seen_texts):
            continue
        evidence_start = section_text.index(evidence)
        evidence_end = evidence_start + len(evidence)
        signal_id = hashlib.sha1(
            f"{candidate.accession}|{candidate.section_id}|{candidate.source_rank}|{evidence}".encode("utf-8")
        ).hexdigest()[:16]
        keep_score = round(min(1.0, (candidate.severity / 5) * 0.6 + candidate.confidence * 0.4), 3)
        final.append(
            RiskSignalV1(
                signal_id=signal_id,
                taxonomy_version=TAXONOMY_VERSION,
                filing_year=candidate.filing_year,
                filing_date=candidate.filing_date,
                accession=candidate.accession,
                section_id=candidate.section_id,
                subheading=candidate.subheading,
                risk_types=[risk_type for risk_type in candidate.risk_types if risk_type in TAXONOMY_BY_ID],
                severity=max(1, min(5, int(candidate.severity))),
                direction=candidate.direction if candidate.direction in {"increasing", "stable", "decreasing", "unclear"} else "unclear",
                confidence=round(max(0.0, min(1.0, candidate.confidence)), 3),
                evidence_text=evidence,
                evidence_start=evidence_start,
                evidence_end=evidence_end,
                judge_rationale=candidate.judge_rationale,
                source_rank=candidate.source_rank,
                keep_score=keep_score,
            )
        )
        seen_texts.append(evidence)
        if len(final) >= 20:
            break
    return sorted(final, key=lambda item: (-item.severity, -item.confidence, -item.keep_score, item.source_rank))


def _build_extraction_prompt(filing_year: int, section_id: str, subheading: str, chunk_text: str) -> str:
    return (
        "Extract up to 5 evidence-backed risk statements from the provided SEC filing text chunk.\n"
        f"Filing year: {filing_year}\n"
        f"Section: {section_id}\n"
        f"Subheading: {subheading}\n"
        f"Allowed taxonomy ids: {', '.join(sorted(TAXONOMY_BY_ID.keys()))}\n"
        "Return exact evidence spans from the text, plus severity 1-5, direction, confidence 0-1, and rationale.\n"
        f"Chunk text:\n{chunk_text}"
    )


def _build_verification_prompt(section_text: str, candidates: Sequence[ExtractedCandidate]) -> str:
    candidate_lines = [
        (
            f"source_rank={candidate.source_rank}; risk_types={candidate.risk_types}; severity={candidate.severity}; "
            f"direction={candidate.direction}; confidence={candidate.confidence}; evidence={candidate.evidence_text}"
        )
        for candidate in candidates
    ]
    return (
        "Review the candidate risk statements. Keep only the statements that are clearly supported by the source section text. "
        "Reject vague, duplicate, or unsupported statements.\n"
        f"Source section text:\n{section_text}\nCandidates:\n" + "\n".join(candidate_lines)
    )


def _extraction_system_prompt() -> str:
    return (
        "You extract risk statements from SEC 10-K text. "
        "Return strict JSON only. "
        "Each candidate must use exact evidence text copied from the provided chunk. "
        "Use only taxonomy ids provided in the prompt. "
        "Severity must be an integer from 1 to 5. "
        "Direction must be one of increasing, stable, decreasing, unclear. "
        "Confidence must be a number from 0 to 1. "
        "Return an object with key 'candidates'."
    )


def _verification_system_prompt() -> str:
    return (
        "You verify candidate risk statements extracted from SEC 10-K text. "
        "Return strict JSON only. "
        "Keep only candidates clearly supported by the source section text. "
        "Reject vague, duplicate, or unsupported candidates. "
        "Return an object with keys 'keep_source_ranks' and 'rationales'."
    )


def _parse_json_content(content) -> Dict:
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ).strip()
    else:
        text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenRouter returned malformed JSON output") from exc
    if not isinstance(parsed, dict):
        raise ValueError("OpenRouter JSON output must be an object")
    return parsed


def _normalize_candidates_payload(payload: Dict) -> List[Dict]:
    candidates = payload.get("candidates", [])
    if isinstance(candidates, dict):
        candidates = candidates.get("items", [])
    if not isinstance(candidates, list):
        return []
    normalized: List[Dict] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        normalized.append(item)
    return normalized


def _candidate_to_risk_types(item: Dict) -> List[str]:
    raw = item.get("risk_types")
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = [raw]
    elif item.get("taxonomy_id"):
        values = [item.get("taxonomy_id")]
    else:
        values = []
    normalized: List[str] = []
    alias_map = {
        "user_concentration": "customer_concentration",
        "privacy_security": "cybersecurity_privacy",
        "legal_regulatory": "regulatory_legal",
        "market_risk": "market_volatility_rates",
    }
    for value in values:
        if not isinstance(value, str):
            continue
        clean = value.strip()
        clean = alias_map.get(clean, clean)
        if clean in TAXONOMY_BY_ID and clean not in normalized:
            normalized.append(clean)
    return normalized


def _normalize_keep_source_ranks(raw_values) -> List[int]:
    if isinstance(raw_values, dict):
        raw_values = raw_values.get("items", [])
    if not isinstance(raw_values, list):
        return []
    normalized: List[int] = []
    for value in raw_values:
        if isinstance(value, int):
            normalized.append(value)
        elif isinstance(value, str) and value.strip().isdigit():
            normalized.append(int(value.strip()))
        elif isinstance(value, dict):
            for key in ("source_rank", "rank", "id", "value"):
                nested = value.get(key)
                if isinstance(nested, int):
                    normalized.append(nested)
                    break
                if isinstance(nested, str) and nested.strip().isdigit():
                    normalized.append(int(nested.strip()))
                    break
    return normalized


def _safe_int(value, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _combined_heading(headings: Sequence[str]) -> str:
    clean = [heading for heading in headings if heading]
    if not clean:
        return "Section Chunk"
    if len(clean) == 1:
        return clean[0]
    return f"{clean[0]} + {len(clean) - 1} more"


def _coalesce_adjacent_chunks(chunks: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if len(chunks) <= 1:
        return list(chunks)
    merged: List[Tuple[str, str]] = []
    idx = 0
    while idx < len(chunks):
        if idx == len(chunks) - 1:
            merged.append(chunks[idx])
            break
        left_heading, left_text = chunks[idx]
        right_heading, right_text = chunks[idx + 1]
        merged.append((_combined_heading([left_heading, right_heading]), f"{left_text}\n\n{right_text}".strip()))
        idx += 2
    return merged


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180]


def _filing_year(filing: Dict) -> int:
    filing_date = filing.get("filingDate", "")
    return int(filing_date[:4]) if len(filing_date) >= 4 and filing_date[:4].isdigit() else 0


def _looks_like_subheading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 90 or stripped.endswith(".") or stripped.endswith(":"):
        return False
    if re.match(r"^(item|part)\s+\w+", stripped, flags=re.IGNORECASE):
        return False
    words = stripped.split()
    if len(words) > 10:
        return False
    if stripped.isupper():
        return True
    title_like_words = sum(1 for word in words if word[:1].isupper())
    return title_like_words >= max(1, len(words) - 1)


def _join_lines(lines: Sequence[str]) -> str:
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", normalized) if part.strip()]


def _match_risk_types(section_id: str, sentence: str) -> List[str]:
    lower = sentence.lower()
    matches = [risk_id for risk_id, keywords in RISK_KEYWORDS.items() if any(keyword in lower for keyword in keywords)]
    section_bias = {
        "item_1c_cybersecurity": "cybersecurity_privacy",
        "item_3_legal": "litigation_contingency",
        "item_7a_market_risk": "market_volatility_rates",
    }
    preferred = section_bias.get(section_id)
    if preferred and preferred not in matches:
        matches.append(preferred)
    return matches[:3]


def _sentence_confidence(sentence: str, risk_types: Sequence[str]) -> float:
    base = 0.45 + min(0.35, len(risk_types) * 0.1)
    if re.search(r"\bmay\b|\bcould\b|\bmight\b", sentence, flags=re.IGNORECASE):
        base += 0.05
    if len(sentence) > 160:
        base += 0.05
    return round(min(base, 0.95), 3)


def _sentence_severity(sentence: str) -> int:
    lower = sentence.lower()
    for severity, markers in SEVERITY_TERMS.items():
        if any(marker in lower for marker in markers):
            return severity
    return 2


def _infer_direction(sentence: str) -> str:
    lower = sentence.lower()
    if any(term in lower for term in ["increase", "increasing", "higher", "grow", "rising"]):
        return "increasing"
    if any(term in lower for term in ["decrease", "decline", "lower", "reduce", "fall"]):
        return "decreasing"
    if any(term in lower for term in ["stable", "remain", "consistent"]):
        return "stable"
    return "unclear"


def _near_duplicate(left: str, right: str) -> bool:
    left_tokens = set(re.findall(r"\w+", left.lower()))
    right_tokens = set(re.findall(r"\w+", right.lower()))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return overlap >= 0.8


def _severity_trend(signals: Sequence[RiskSignalV1]) -> str:
    if len(signals) < 2:
        return "insufficient_data"
    by_year: Dict[int, float] = {}
    for signal in signals:
        by_year[signal.filing_year] = max(by_year.get(signal.filing_year, 0.0), float(signal.severity))
    ordered = [score for _, score in sorted(by_year.items())]
    if len(ordered) < 2:
        return "insufficient_data"
    if ordered[-1] > ordered[0]:
        return "increasing"
    if ordered[-1] < ordered[0]:
        return "decreasing"
    if len(set(ordered)) == 1:
        return "stable"
    return "mixed"


def _dict_to_signal(data: Dict) -> RiskSignalV1:
    return RiskSignalV1(**data)
