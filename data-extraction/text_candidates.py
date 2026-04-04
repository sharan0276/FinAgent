from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from constants import DEFAULT_TOP_K, MIN_SENTENCE_LENGTH, PER_SECTION_TARGET, SECTION_LABELS, TARGET_SECTIONS
from models import FinBertProbabilities, TextCandidate
from scoring import BaseSentenceScorer


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
NUMERIC_FRAGMENT_RE = re.compile(r"^[\d\s,$.%()\-:;/]+$")
SEC_REFERENCE_RE = re.compile(r"(sec\.gov|investor relations|not incorporated by reference)", re.IGNORECASE)


@dataclass
class SentenceRecord:
    section_id: str
    section_label: str
    sentence_index: int
    sentence_text: str
    previous_sentence: str | None
    next_sentence: str | None
    risk_score: float = 0.0
    probs: Dict[str, float] | None = None


def build_text_candidates(
    sections: Dict[str, str],
    scorer: BaseSentenceScorer,
    *,
    top_k: int = DEFAULT_TOP_K,
    per_section_target: int = PER_SECTION_TARGET,
) -> Tuple[List[TextCandidate], List[str], List[str]]:
    section_records: Dict[str, List[SentenceRecord]] = {}
    processed_sections: List[str] = []
    skipped_sections: List[str] = []

    for section_id, label in TARGET_SECTIONS:
        section_text = (sections.get(section_id) or "").strip()
        if not section_text:
            skipped_sections.append(section_id)
            continue
        records = _collect_sentence_records(section_id, label, section_text)
        if not records:
            skipped_sections.append(section_id)
            continue
        scores = scorer.score_sentences([record.sentence_text for record in records])
        for record, probs in zip(records, scores):
            record.probs = probs
            record.risk_score = float(probs["negative"])
        section_records[section_id] = sorted(records, key=lambda item: item.risk_score, reverse=True)
        processed_sections.append(section_id)

    selected = _select_balanced_candidates(section_records, top_k=top_k, per_section_target=per_section_target)
    candidates = []
    for idx, record in enumerate(selected, start=1):
        probs = record.probs or {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        candidates.append(
            TextCandidate(
                candidate_id=f"{record.section_id}_{idx}",
                section_id=record.section_id,
                section_label=record.section_label,
                sentence_index=record.sentence_index,
                sentence_text=record.sentence_text,
                previous_sentence=record.previous_sentence,
                next_sentence=record.next_sentence,
                risk_score=record.risk_score,
                finbert_probs=FinBertProbabilities(**probs),
            )
        )
    return candidates, processed_sections, skipped_sections


def split_into_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()
    if not normalized:
        return []
    chunks = SENTENCE_SPLIT_RE.split(normalized)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def is_candidate_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(stripped) < MIN_SENTENCE_LENGTH:
        return False
    if NUMERIC_FRAGMENT_RE.match(stripped):
        return False
    if SEC_REFERENCE_RE.search(stripped):
        return False
    alpha_count = sum(1 for char in stripped if char.isalpha())
    if alpha_count < 10:
        return False
    words = stripped.split()
    if len(words) <= 8 and all(word[:1].isupper() for word in words if word[:1].isalpha()):
        return False
    return True


def _collect_sentence_records(section_id: str, section_label: str, section_text: str) -> List[SentenceRecord]:
    sentences = split_into_sentences(section_text)
    records: List[SentenceRecord] = []
    for index, sentence in enumerate(sentences):
        if not is_candidate_sentence(sentence):
            continue
        prev_sentence = sentences[index - 1].strip() if index > 0 else None
        next_sentence = sentences[index + 1].strip() if index + 1 < len(sentences) else None
        records.append(
            SentenceRecord(
                section_id=section_id,
                section_label=section_label,
                sentence_index=index,
                sentence_text=sentence.strip(),
                previous_sentence=prev_sentence,
                next_sentence=next_sentence,
            )
        )
    return records


def _select_balanced_candidates(
    section_records: Dict[str, List[SentenceRecord]],
    *,
    top_k: int,
    per_section_target: int,
) -> List[SentenceRecord]:
    selected: List[SentenceRecord] = []
    remainders: List[SentenceRecord] = []
    seen_keys = set()

    for section_id, _ in TARGET_SECTIONS:
        records = list(section_records.get(section_id, []))
        primary = records[:per_section_target]
        overflow = records[per_section_target:]
        for record in primary:
            key = (record.section_id, record.sentence_index)
            if key not in seen_keys:
                seen_keys.add(key)
                selected.append(record)
        remainders.extend(overflow)

    if len(selected) < top_k:
        for record in sorted(remainders, key=lambda item: item.risk_score, reverse=True):
            key = (record.section_id, record.sentence_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(record)
            if len(selected) >= top_k:
                break

    selected.sort(key=lambda item: item.risk_score, reverse=True)
    return selected[:top_k]
