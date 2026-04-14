from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from orchestration.comparison_agent import (
    build_deterministic_report,
    build_forward_watchlist,
    build_peer_snapshot,
    build_risk_overlap_rows,
    build_target_profile,
    determine_posture,
    generate_comparison_report,
)
from orchestration.openrouter_client import OpenRouterError
from orchestration.report_models import ComparisonBundle, MatchContext, RunMetadata, TargetContext


class FakeClient:
    def __init__(self, payload=None, error: str | None = None):
        self.payload = payload or {}
        self.error = error

    def complete_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2):
        if self.error:
            raise OpenRouterError(self.error)
        return self.payload, "fake-openrouter-model"


class ComparisonAgentTests(unittest.TestCase):
    def _make_temp_dir(self, name: str) -> Path:
        base = Path(__file__).resolve().parent / ".tmp_comparison_agent_tests"
        base.mkdir(parents=True, exist_ok=True)
        path = base / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _write_curator(
        self,
        path: Path,
        *,
        company: str,
        ticker: str,
        filing_year: int,
        financial_deltas: dict,
        risk_signals: list[dict],
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "company": company,
                    "ticker": ticker,
                    "filing_year": filing_year,
                    "financial_deltas": financial_deltas,
                    "risk_signals": risk_signals,
                    "embedding_text": "test",
                    "embedding_vector": [0.1, 0.2],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _make_bundle(self, root: Path) -> ComparisonBundle:
        target = self._write_curator(
            root / "AAPL" / "aapl_2025.json",
            company="Apple Inc.",
            ticker="AAPL",
            filing_year=2025,
            financial_deltas={
                "Revenues": {"value": 0.11, "label": "moderate_growth"},
                "Cash": {"value": 0.09, "label": "moderate_growth"},
                "LongTermDebt": {"value": -0.12, "label": "moderate_decline"},
                "OperatingCashFlow": {"value": -0.21, "label": "severe_decline"},
            },
            risk_signals=[
                {"signal_type": "foreign_currency_exposure", "severity": "high", "summary": "FX pressure remained material.", "section": "Item 7", "citation": "AAPL 10-K 2025, Item 7"},
                {"signal_type": "supply_chain_concentration", "severity": "high", "summary": "Supplier concentration remains high.", "section": "Item 1A", "citation": "AAPL 10-K 2025, Item 1A"},
                {"signal_type": "trade_tariff_risk", "severity": "medium", "summary": "Tariff changes could hurt margins.", "section": "Item 7", "citation": "AAPL 10-K 2025, Item 7"},
            ],
        )
        goog_2024 = self._write_curator(
            root / "GOOG" / "goog_2024.json",
            company="Alphabet Inc.",
            ticker="GOOG",
            filing_year=2024,
            financial_deltas={
                "Revenues": {"value": 0.14, "label": "moderate_growth"},
                "NetIncome": {"value": 0.35, "label": "strong_growth"},
                "LongTermDebt": {"value": -0.16, "label": "moderate_decline"},
            },
            risk_signals=[
                {"signal_type": "foreign_currency_exposure", "severity": "medium", "summary": "FX affected international revenue.", "section": "Item 7", "citation": "GOOG 10-K 2024, Item 7"},
                {"signal_type": "pricing_power_erosion", "severity": "medium", "summary": "Fee changes pressured platform revenue.", "section": "Item 7", "citation": "GOOG 10-K 2024, Item 7"},
            ],
        )
        goog_2025 = self._write_curator(
            root / "GOOG" / "goog_2025.json",
            company="Alphabet Inc.",
            ticker="GOOG",
            filing_year=2025,
            financial_deltas={
                "Revenues": {"value": -0.08, "label": "moderate_decline"},
                "OperatingCashFlow": {"value": -0.07, "label": "moderate_decline"},
            },
            risk_signals=[
                {"signal_type": "pricing_power_erosion", "severity": "high", "summary": "Monetization pressure deepened.", "section": "Item 7", "citation": "GOOG 10-K 2025, Item 7"},
                {"signal_type": "regulatory_investigation", "severity": "high", "summary": "New regulatory probe expanded.", "section": "Item 3", "citation": "GOOG 10-K 2025, Item 3"},
            ],
        )
        meta_2023 = self._write_curator(
            root / "META" / "meta_2023.json",
            company="Meta Platforms, Inc.",
            ticker="META",
            filing_year=2023,
            financial_deltas={
                "Revenues": {"value": 0.15, "label": "moderate_growth"},
                "NetIncome": {"value": 0.68, "label": "strong_growth"},
                "OperatingCashFlow": {"value": 0.41, "label": "strong_growth"},
            },
            risk_signals=[
                {"signal_type": "foreign_currency_exposure", "severity": "high", "summary": "FX hurt advertising revenue.", "section": "Item 7", "citation": "META 10-K 2023, Item 7"},
                {"signal_type": "pricing_power_erosion", "severity": "high", "summary": "Ad pricing weakened.", "section": "Item 7", "citation": "META 10-K 2023, Item 7"},
            ],
        )
        meta_2024 = self._write_curator(
            root / "META" / "meta_2024.json",
            company="Meta Platforms, Inc.",
            ticker="META",
            filing_year=2024,
            financial_deltas={
                "Revenues": {"value": 0.03, "label": "stable"},
                "Cash": {"value": -0.11, "label": "moderate_decline"},
            },
            risk_signals=[
                {"signal_type": "cybersecurity_incident", "severity": "high", "summary": "Security event response costs increased.", "section": "Item 1C", "citation": "META 10-K 2024, Item 1C"},
            ],
        )

        return ComparisonBundle(
            target=TargetContext(
                ticker="AAPL",
                company="Apple Inc.",
                latest_filing_year=2025,
                curator_path=str(target),
            ),
            matches=[
                MatchContext(
                    ticker="GOOG",
                    company="Alphabet Inc.",
                    matched_filing_year=2024,
                    similarity=0.95,
                    context_curator_paths=[str(goog_2024), str(goog_2025)],
                ),
                MatchContext(
                    ticker="META",
                    company="Meta Platforms, Inc.",
                    matched_filing_year=2023,
                    similarity=0.91,
                    context_curator_paths=[str(meta_2023), str(meta_2024)],
                ),
            ],
            run_metadata=RunMetadata(top_k=2, run_timestamp="2026-04-13T00:00:00Z", status_by_step={}),
        )

    def test_target_profile_is_data_grounded(self) -> None:
        root = self._make_temp_dir("target_profile")
        bundle = self._make_bundle(root)
        target_curator = json.loads(Path(bundle.target.curator_path).read_text(encoding="utf-8"))

        profile = build_target_profile(target_curator)

        self.assertEqual(profile.ticker, "AAPL")
        self.assertEqual(profile.company, "Apple Inc.")
        self.assertEqual(profile.filing_year, 2025)
        self.assertEqual(profile.positive_deltas[0].metric, "Revenues")
        self.assertEqual(profile.negative_deltas[0].metric, "OperatingCashFlow")
        self.assertEqual(profile.top_risks[0].signal_type, "foreign_currency_exposure")
        self.assertEqual(profile.top_risks[0].citation, "AAPL 10-K 2025, Item 7")

    def test_risk_overlap_classification(self) -> None:
        root = self._make_temp_dir("risk_overlap")
        bundle = self._make_bundle(root)
        target_curator = json.loads(Path(bundle.target.curator_path).read_text(encoding="utf-8"))
        peer_curators = [json.loads(Path(match.context_curator_paths[0]).read_text(encoding="utf-8")) for match in bundle.matches]

        rows = build_risk_overlap_rows(target_curator, peer_curators)
        row_map = {row.group: row.risk_types for row in rows}

        self.assertIn("foreign_currency_exposure", row_map["shared_now"])
        self.assertIn("supply_chain_concentration", row_map["target_only_now"])
        self.assertIn("pricing_power_erosion", row_map["peer_only_now"])

    def test_peer_snapshot_aggregates_neighborhood(self) -> None:
        root = self._make_temp_dir("peer_snapshot")
        bundle = self._make_bundle(root)
        target_curator = json.loads(Path(bundle.target.curator_path).read_text(encoding="utf-8"))
        peer_curators = [json.loads(Path(match.context_curator_paths[0]).read_text(encoding="utf-8")) for match in bundle.matches]

        snapshot = build_peer_snapshot(target_curator, peer_curators)

        self.assertEqual(snapshot.peer_group, "Top matched peer neighborhood")
        self.assertIn("Revenues", snapshot.common_strengths)
        self.assertIn("foreign_currency_exposure", snapshot.shared_risk_types)

    def test_posture_assignment_elevated_mixed_stable(self) -> None:
        elevated_curator = {
            "financial_deltas": {
                "Debt": {"label": "moderate_decline"},
                "CashFlow": {"label": "severe_decline"},
            },
            "risk_signals": [
                {"signal_type": "a", "severity": "high"},
                {"signal_type": "b", "severity": "high"},
            ],
        }
        mixed_curator = {
            "financial_deltas": {
                "Revenues": {"label": "moderate_growth"},
                "CashFlow": {"label": "moderate_decline"},
            },
            "risk_signals": [{"signal_type": "a", "severity": "medium"}],
        }
        stable_curator = {
            "financial_deltas": {
                "Revenues": {"label": "moderate_growth"},
                "NetIncome": {"label": "strong_growth"},
            },
            "risk_signals": [{"signal_type": "a", "severity": "medium"}],
        }

        self.assertEqual(determine_posture(elevated_curator, [type("Row", (), {"group": "shared_now", "risk_types": ["x", "y"]})()]).label, "Elevated")
        self.assertEqual(determine_posture(mixed_curator, [type("Row", (), {"group": "shared_now", "risk_types": ["x"]})()]).label, "Mixed")
        self.assertEqual(determine_posture(stable_curator, [type("Row", (), {"group": "shared_now", "risk_types": []})()]).label, "Stable")

    def test_forward_watchlist_uses_future_peer_patterns(self) -> None:
        root = self._make_temp_dir("watchlist")
        bundle = self._make_bundle(root)
        target_curator = json.loads(Path(bundle.target.curator_path).read_text(encoding="utf-8"))

        watchlist = build_forward_watchlist(bundle, target_curator)

        risk_types = [item.watch_risk_type for item in watchlist]
        self.assertIn("regulatory_investigation", risk_types)
        self.assertIn("cybersecurity_incident", risk_types)
        self.assertTrue(all("forecast" not in item.why_relevant.lower() for item in watchlist))

    def test_openrouter_response_maps_into_structured_report(self) -> None:
        root = self._make_temp_dir("llm_success")
        bundle = self._make_bundle(root)
        client = FakeClient(
            payload={
                "summary": "Apple faces a mixed but manageable setup versus peers.",
                "posture_rationale_bullets": [
                    "High-severity current risks remain concentrated in macro and operational areas.",
                    "Peer overlap is strongest around FX and pricing pressure.",
                    "Growth metrics still offset part of the downside picture.",
                ],
                "narrative_sections": [
                    {"title": "Current Posture", "content": "Apple remains under real but manageable pressure based on the target filing's cited risks and current deltas."},
                    {"title": "Peer Comparison", "content": "FX pressure and pricing pressure show up across the peer neighborhood."},
                    {"title": "What To Watch Next", "content": "Regulatory and cybersecurity issues appear in later peer years and are worth monitoring."},
                ],
            }
        )

        report = generate_comparison_report(bundle, client=client)

        self.assertEqual(report.status, "completed")
        self.assertEqual(report.model_name, "fake-openrouter-model")
        self.assertEqual(report.posture.label, "Elevated")
        self.assertEqual(len(report.narrative_sections), 3)
        self.assertTrue(report.narrative_sections[0].citations)
        self.assertIn("mixed but manageable", report.summary.lower())

    def test_deterministic_report_builds_three_grounded_sections(self) -> None:
        root = self._make_temp_dir("deterministic_sections")
        bundle = self._make_bundle(root)

        report = build_deterministic_report(bundle)

        self.assertEqual(len(report.narrative_sections), 3)
        self.assertEqual([section.title for section in report.narrative_sections], ["Current Posture", "Peer Comparison", "What To Watch Next"])
        self.assertTrue(all(section.content.strip() for section in report.narrative_sections))
        self.assertTrue(report.narrative_sections[0].citations)

    def test_failure_fallback_keeps_deterministic_structure(self) -> None:
        root = self._make_temp_dir("llm_failure")
        bundle = self._make_bundle(root)
        report = generate_comparison_report(bundle, client=FakeClient(error="OPENROUTER_API_KEY is not set."))

        self.assertEqual(report.status, "failed")
        self.assertIsNotNone(report.target_profile)
        self.assertIsNotNone(report.peer_snapshot)
        self.assertGreater(len(report.forward_watchlist), 0)
        self.assertEqual(report.error, "OPENROUTER_API_KEY is not set.")
        self.assertIn("deterministic comparison structure", report.summary.lower())


if __name__ == "__main__":
    unittest.main()
