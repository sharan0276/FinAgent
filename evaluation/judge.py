from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from orchestration.openrouter_client import OpenRouterClient

from .models import ClaimAssessment, EvaluationInput


CLAIM_EXTRACTION_PROMPT_VERSION = "v1"
CLAIM_SUPPORT_PROMPT_VERSION = "v1"
REPORT_PROMPT_VERSION = "v1"


def _cache_key(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_claim_extraction_prompt(evaluation_input: EvaluationInput) -> str:
    sections = [
        {
            "title": section.title,
            "content": section.content,
            "claim_cap": 2,
        }
        for section in evaluation_input.narrative_sections
    ]
    payload = {
        "ticker": evaluation_input.ticker,
        "summary": evaluation_input.summary_text,
        "summary_claim_cap": 3,
        "posture_bullets": evaluation_input.posture_bullets[:3],
        "posture_claim_cap": 3,
        "narrative_sections": sections,
        "output_schema": {
            "claims": [
                {"claim": "string", "source": "summary|posture|section:<title>"}
            ]
        },
    }
    return json.dumps(payload, indent=2)


def build_claim_support_prompt(claim: str, evidence_items: list[str]) -> str:
    payload = {
        "claim": claim,
        "evidence_snippets": evidence_items[:10],
        "allowed_labels": ["supported", "partially_supported", "unsupported"],
        "output_schema": {
            "label": "supported|partially_supported|unsupported",
            "evidence_snippets": ["string"],
        },
    }
    return json.dumps(payload, indent=2)


def build_report_judge_prompt(evaluation_input: EvaluationInput) -> str:
    payload = {
        "ticker": evaluation_input.ticker,
        "summary": evaluation_input.summary_text,
        "shared_risk_types": evaluation_input.shared_risk_types,
        "target_differences": evaluation_input.target_differences,
        "watchlist": [item.model_dump(mode="json") for item in evaluation_input.forward_watchlist],
        "peer_evidence": [item.text for item in evaluation_input.peer_evidence_pool[:12]],
        "output_schema": {
            "comparative_usefulness": "strong|partial|weak",
            "overreach_flags": ["string"],
        },
    }
    return json.dumps(payload, indent=2)


class EvaluationJudge:
    def __init__(
        self,
        *,
        client: OpenRouterClient | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.client = client or OpenRouterClient()
        self.cache_dir = Path(cache_dir or (Path(__file__).resolve().parent / "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _complete_json(self, *, cache_key_value: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        cached = self._read_cache(cache_key_value)
        if cached is not None:
            return cached
        payload, model_name = self.client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        wrapped = {"response": payload, "model_name": model_name}
        self._write_cache(cache_key_value, wrapped)
        return wrapped

    def extract_claims(self, evaluation_input: EvaluationInput) -> list[dict[str, str]]:
        prompt = build_claim_extraction_prompt(evaluation_input)
        key = _cache_key(evaluation_input.artifact_hash, CLAIM_EXTRACTION_PROMPT_VERSION, prompt)
        payload = self._complete_json(
            cache_key_value=key,
            system_prompt="Extract a small set of atomic report claims. Return strict JSON only.",
            user_prompt=prompt,
        )
        claims = payload["response"].get("claims", [])
        return [
            {"claim": str(item.get("claim", "")).strip(), "source": str(item.get("source", "")).strip()}
            for item in claims
            if str(item.get("claim", "")).strip()
        ]

    def assess_claims(self, evaluation_input: EvaluationInput, claims: list[dict[str, str]]) -> tuple[list[ClaimAssessment], float]:
        evidence_items = [item.text for item in [*evaluation_input.target_evidence_pool, *evaluation_input.peer_evidence_pool]]
        assessments: list[ClaimAssessment] = []
        score_total = 0.0
        for item in claims:
            prompt = build_claim_support_prompt(item["claim"], evidence_items)
            key = _cache_key(evaluation_input.artifact_hash, CLAIM_SUPPORT_PROMPT_VERSION, item["claim"], prompt)
            payload = self._complete_json(
                cache_key_value=key,
                system_prompt="Classify whether the claim is supported by the supplied evidence. Return strict JSON only.",
                user_prompt=prompt,
            )
            label = str(payload["response"].get("label", "unsupported")).strip()
            if label not in {"supported", "partially_supported", "unsupported"}:
                label = "unsupported"
            assessments.append(
                ClaimAssessment(
                    claim=item["claim"],
                    source=item["source"],
                    label=label,
                    evidence_snippets=[str(v) for v in payload["response"].get("evidence_snippets", [])[:3]],
                )
            )
            score_total += {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}[label]
        average = score_total / len(assessments) if assessments else 0.0
        return assessments, round(average, 4)

    def assess_report(self, evaluation_input: EvaluationInput) -> tuple[dict[str, Any], float]:
        prompt = build_report_judge_prompt(evaluation_input)
        key = _cache_key(evaluation_input.artifact_hash, REPORT_PROMPT_VERSION, prompt)
        payload = self._complete_json(
            cache_key_value=key,
            system_prompt="Judge comparative substance and overreach for this report. Return strict JSON only.",
            user_prompt=prompt,
        )
        response = payload["response"]
        usefulness = str(response.get("comparative_usefulness", "weak")).strip().lower()
        score = {"strong": 1.0, "partial": 0.5, "weak": 0.0}.get(usefulness, 0.0)
        return response, score
