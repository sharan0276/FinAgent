from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from evaluation.deterministic import score_evaluation_input
from evaluation.judge import EvaluationJudge, build_claim_extraction_prompt, build_report_judge_prompt
from evaluation.loaders import build_evaluation_input
from evaluation.runner import run_batch
from orchestration.report_models import (
    ComparisonBundle,
    ComparisonReportResult,
    ForwardWatchItem,
    MatchContext,
    OrchestrationArtifact,
    PeerSnapshot,
    PostureCard,
    ReportSection,
    RiskItem,
    RiskOverlapRow,
    RunMetadata,
    TargetContext,
    TargetProfile,
)


class FakeJudgeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0):
        self.calls += 1
        if "Extract a small set of atomic report claims" in system_prompt:
            return {"claims": [{"claim": "The company faces antitrust exposure.", "source": "summary"}]}, "fake-model"
        if "Classify whether the claim is supported" in system_prompt:
            return {"label": "supported", "evidence_snippets": ["Antitrust case is cited in the report."]}, "fake-model"
        return {"comparative_usefulness": "strong", "overreach_flags": []}, "fake-model"


class EvaluationTests(unittest.TestCase):
    def _make_temp_dir(self, name: str) -> Path:
        base = Path(__file__).resolve().parent / ".tmp_evaluation_tests"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{name}_{uuid.uuid4().hex}"
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_agentic_artifact(self, repo_root: Path) -> Path:
        curator_path = repo_root / "data-extraction" / "outputs" / "curator" / "TEST" / "test_2025.json"
        self._write_json(
            curator_path,
            {
                "company": "Test Corp",
                "ticker": "TEST",
                "filing_year": 2025,
                "financial_deltas": {
                    "Revenue": {"value": 12.0, "label": "moderate_growth"},
                    "OperatingCashFlow": {"value": -9.0, "label": "moderate_decline"},
                },
                "risk_signals": [
                    {
                        "signal_id": 1,
                        "topic": "legal_risk",
                        "signal_type": "antitrust_exposure",
                        "section": "Item 3",
                        "filing_year": 2025,
                        "company": "TEST",
                        "summary": "The company faces an active antitrust proceeding.",
                        "severity": "high",
                        "citation": "TEST 10-K 2025, Item 3",
                    }
                ],
                "embedding_text": "x",
                "embedding_vector": [0.1, 0.2],
            },
        )
        artifact = OrchestrationArtifact(
            bundle=ComparisonBundle(
                target=TargetContext(ticker="TEST", company="Test Corp", latest_filing_year=2025, curator_path=str(curator_path)),
                matches=[
                    MatchContext(
                        ticker="PEER",
                        company="Peer Co",
                        matched_filing_year=2024,
                        similarity=0.92,
                        context_curator_paths=[],
                    )
                ],
                run_metadata=RunMetadata(top_k=1, run_timestamp="2026-04-14T00:00:00Z", status_by_step={}, warnings=[]),
            ),
            comparison_report=ComparisonReportResult(
                status="completed",
                summary="Test Corp has elevated legal pressure but still shows revenue growth.",
                posture=PostureCard(label="Mixed", rationale_bullets=["Revenue is growing.", "Legal risk remains high."]),
                target_profile=TargetProfile(
                    ticker="TEST",
                    company="Test Corp",
                    filing_year=2025,
                    positive_deltas=[],
                    negative_deltas=[],
                    top_risks=[
                        RiskItem(
                            signal_type="antitrust_exposure",
                            severity="high",
                            section="Item 3",
                            summary="The company faces an active antitrust proceeding.",
                            citation="TEST 10-K 2025, Item 3",
                            occurrences=1,
                        )
                    ],
                ),
                peer_snapshot=PeerSnapshot(
                    peer_group="Peers",
                    common_strengths=["Revenue"],
                    common_pressures=["OperatingCashFlow"],
                    shared_risk_types=["antitrust_exposure"],
                    target_differences=["manufacturing_defect_risk"],
                ),
                risk_overlap_rows=[RiskOverlapRow(group="shared_now", risk_types=["antitrust_exposure"])],
                forward_watchlist=[
                    ForwardWatchItem(
                        watch_risk_type="cybersecurity_incident",
                        why_relevant="Peers surfaced it later.",
                        peer_evidence=["PEER 2025: cybersecurity controls weakened."],
                        confidence="medium",
                    )
                ],
                narrative_sections=[
                    ReportSection(
                        title="Current Posture",
                        content="Legal risk is elevated, but revenues improved.",
                        citations=["TEST 10-K 2025, Item 3"],
                    )
                ],
            ),
        )
        artifact_path = repo_root / "orchestration" / "outputs" / "TEST" / "test_comparison_bundle.json"
        self._write_json(artifact_path, artifact.model_dump(mode="json"))
        return artifact_path

    def _write_baseline_artifact(self, repo_root: Path, *, with_watch_evidence: bool = True) -> Path:
        ingestion_path = repo_root / "data-ingestion" / "outputs" / "TEST" / "complete_ingestion.json"
        peer_ingestion_path = repo_root / "data-ingestion" / "outputs" / "PEER" / "complete_ingestion.json"
        ingestion_payload = {
            "ticker": "TEST",
            "company_name": "Test Corp",
            "financial_data": {"annual": {"Revenue": {"years": [2024, 2025], "values": [100.0, 110.0], "deltas": [None, 10.0], "unit": "USD_millions"}}},
            "text_data": {"filings": [{"filingDate": "2025-12-31"}]},
        }
        peer_payload = {
            "ticker": "PEER",
            "company_name": "Peer Co",
            "financial_data": {"annual": {"Revenue": {"years": [2024, 2025], "values": [90.0, 92.0], "deltas": [None, 2.0], "unit": "USD_millions"}}},
            "text_data": {"filings": [{"filingDate": "2025-12-31"}]},
        }
        self._write_json(ingestion_path, ingestion_payload)
        self._write_json(peer_ingestion_path, peer_payload)

        artifact = OrchestrationArtifact(
            schema_version="baseline_rag_v1",
            bundle=ComparisonBundle(
                target=TargetContext(ticker="TEST", company="Test Corp", latest_filing_year=2025, ingestion_path=str(ingestion_path)),
                matches=[MatchContext(ticker="PEER", company="Peer Co", matched_filing_year=2025, similarity=0.88)],
                run_metadata=RunMetadata(top_k=1, run_timestamp="2026-04-14T00:00:00Z", status_by_step={}, warnings=[]),
            ),
            comparison_report=ComparisonReportResult(
                status="completed",
                summary="Test Corp looks stable versus peers.",
                posture=PostureCard(label="Stable", rationale_bullets=["Revenue growth is positive."]),
                target_profile=TargetProfile(
                    ticker="TEST",
                    company="Test Corp",
                    filing_year=2025,
                    top_risks=[
                        RiskItem(
                            signal_type="antitrust_exposure",
                            severity="high",
                            summary="The company faces an active antitrust proceeding.",
                            citation=None,
                            occurrences=1,
                        )
                    ],
                ),
                peer_snapshot=PeerSnapshot(peer_group="Peers", shared_risk_types=[], target_differences=[]),
                risk_overlap_rows=[],
                forward_watchlist=[
                    ForwardWatchItem(
                        watch_risk_type="cybersecurity_incident",
                        why_relevant="Peers mention it.",
                        peer_evidence=["PEER 2025: cyber issue"] if with_watch_evidence else [],
                        confidence="low",
                    )
                ],
                narrative_sections=[ReportSection(title="Financial Performance", content="Revenue improved.", citations=[])],
            ),
        )
        artifact_path = repo_root / "baseline_rag" / "outputs" / "TEST" / "test_baseline_bundle.json"
        self._write_json(artifact_path, artifact.model_dump(mode="json"))
        return artifact_path

    def test_build_evaluation_input_normalizes_agentic_and_baseline(self) -> None:
        repo_root = self._make_temp_dir("normalize")
        agentic_path = self._write_agentic_artifact(repo_root)
        baseline_path = self._write_baseline_artifact(repo_root)

        agentic = build_evaluation_input(agentic_path)
        baseline = build_evaluation_input(baseline_path)

        self.assertEqual(agentic.pipeline, "agentic")
        self.assertTrue(any(item.source_type == "target_curator_delta" for item in agentic.target_evidence_pool))
        self.assertEqual(baseline.pipeline, "baseline")
        self.assertTrue(any(item.source_type == "target_ingestion_text" for item in baseline.target_evidence_pool))
        self.assertTrue(any(item.source_type == "peer_match_metadata" for item in baseline.peer_evidence_pool))

    def test_deterministic_scoring_flags_contradictory_posture(self) -> None:
        repo_root = self._make_temp_dir("contradiction")
        baseline_path = self._write_baseline_artifact(repo_root)
        evaluation_input = build_evaluation_input(baseline_path)

        score, warnings = score_evaluation_input(evaluation_input)

        self.assertLess(score.deterministic_consistency, 1.0)
        self.assertTrue(any("Stable posture conflicts" in warning for warning in warnings))

    def test_deterministic_scoring_penalizes_watchlist_without_evidence(self) -> None:
        repo_root = self._make_temp_dir("watch_penalty")
        baseline_path = self._write_baseline_artifact(repo_root, with_watch_evidence=False)
        evaluation_input = build_evaluation_input(baseline_path)

        score, warnings = score_evaluation_input(evaluation_input)

        self.assertGreater(score.overreach_penalty, 0.0)
        self.assertTrue(any("Watchlist" in warning for warning in warnings))

    def test_evidence_coverage_distinguishes_cited_sections(self) -> None:
        repo_root = self._make_temp_dir("coverage")
        agentic_path = self._write_agentic_artifact(repo_root)
        baseline_path = self._write_baseline_artifact(repo_root)

        agentic_score, _ = score_evaluation_input(build_evaluation_input(agentic_path))
        baseline_score, _ = score_evaluation_input(build_evaluation_input(baseline_path))

        self.assertGreater(agentic_score.evidence_coverage, baseline_score.evidence_coverage)

    def test_judge_prompt_builders_and_cache(self) -> None:
        repo_root = self._make_temp_dir("judge")
        agentic_path = self._write_agentic_artifact(repo_root)
        evaluation_input = build_evaluation_input(agentic_path)
        cache_dir = repo_root / "judge-cache"
        client = FakeJudgeClient()
        judge = EvaluationJudge(client=client, cache_dir=cache_dir)

        extraction_prompt = build_claim_extraction_prompt(evaluation_input)
        report_prompt = build_report_judge_prompt(evaluation_input)
        self.assertIn('"summary_claim_cap": 3', extraction_prompt)
        self.assertIn('"comparative_usefulness": "strong|partial|weak"', report_prompt)

        claims = judge.extract_claims(evaluation_input)
        assessments, claim_support = judge.assess_claims(evaluation_input, claims)
        report_judgement, usefulness = judge.assess_report(evaluation_input)
        first_call_count = client.calls
        judge.extract_claims(evaluation_input)

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].label, "supported")
        self.assertEqual(claim_support, 1.0)
        self.assertEqual(usefulness, 1.0)
        self.assertEqual(report_judgement["comparative_usefulness"], "strong")
        self.assertEqual(client.calls, first_call_count)

    def test_batch_runner_pairs_outputs_by_ticker(self) -> None:
        repo_root = self._make_temp_dir("runner")
        agentic_path = self._write_agentic_artifact(repo_root)
        baseline_path = self._write_baseline_artifact(repo_root)

        output = run_batch(
            agentic_paths=[agentic_path],
            baseline_paths=[baseline_path],
            judge_enabled=False,
        )

        self.assertEqual(len(output.results), 2)
        self.assertEqual(len(output.comparisons), 1)
        self.assertEqual(output.comparisons[0].ticker, "TEST")


if __name__ == "__main__":
    unittest.main()
