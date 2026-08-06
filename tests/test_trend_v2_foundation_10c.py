from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.trend_v2_foundation import (
    ArtifactKind,
    ArtifactRetentionPolicy,
    ExecutionStatus,
    LocalResultStore,
    StrategyRunManifest,
    StrategyRunSpec,
    evaluate_saved_runs,
    load_evaluation_profiles,
)
from src.trend_v2_foundation.api import ReadOnlyTrendApi
from src.trend_v2_foundation.decision_report import DecisionReportError, DecisionReportService
from src.trend_v2_foundation.registry import SavedRunRegistryBuilder


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "config" / "trend_v2" / "evaluation_profiles"


def strategy_spec() -> StrategyRunSpec:
    return StrategyRunSpec(
        data_snapshot_hash="a" * 64,
        economic_date_range={"start": "2020-01-02", "end": "2025-12-31"},
        universe_specification={"key": "synthetic", "version": "v1"},
        benchmark={"symbol": "SPY", "return": "adjusted_close"},
        trend_filter={"key": "price_above_ma", "version": "v1", "parameters": {"days": 200}},
        signal={"key": "synthetic_breakout", "version": "v1", "parameters": {"threshold": 1.0}},
        entry_rule={"key": "next_open", "version": "v1"},
        initial_stop={"key": "atr_stop", "version": "v1", "parameters": {"multiple": 2.0}},
        trailing_exit={"key": "channel_exit", "version": "v1", "parameters": {"days": 20}},
        position_sizing={"key": "equal_weight", "version": "v1"},
        portfolio_constraints={"max_positions": 10, "max_weight": 0.2},
        transaction_costs={"round_trip_rate": 0.002},
        slippage={"model": "fixed_bps", "basis_points": 5},
        engine_version="synthetic_engine_v1",
    )


def summary_metrics() -> dict:
    return {
        "cagr": 0.18,
        "cagr_spy_ratio": 1.12,
        "maximum_drawdown": -0.17,
        "maximum_drawdown_spy_ratio": 0.6,
        "cdar95_spy_ratio": 0.7,
        "calmar_spy_ratio": 1.1,
        "recovery_duration_days": 100,
        "annual_turnover": 1.0,
    }


class Foundation10CDecisionReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = LocalResultStore(
            self.temp.name, ArtifactRetentionPolicy(5_000_000, 1_000_000, 20, 20)
        )
        self.spec = strategy_spec()
        summary = self.store.put_artifact(
            "summary_metrics", ArtifactKind.SUMMARY_METRICS, summary_metrics(), row_count=1
        ).record
        behavior = self.store.put_artifact(
            "behavior_metadata", ArtifactKind.BEHAVIOR_METADATA,
            {"behavior_group_id": "B001", "normalized_return_hash": "f" * 64}, row_count=1,
        ).record
        self.store.save_strategy_run(StrategyRunManifest.create(
            self.spec, source_code_commit="a" * 40, artifacts=(summary, behavior),
            creation_time="2026-08-06T00:00:00Z", execution_status=ExecutionStatus.SUCCEEDED,
        ))
        self.profile = load_evaluation_profiles(PROFILE_DIR)["research_default"]
        self.evaluation = evaluate_saved_runs(
            self.store, [self.spec.strategy_run_id], self.profile,
            benchmark_data_identity="spy-snapshot", creation_time="2026-08-06T00:01:00Z",
        )
        self.service = DecisionReportService(self.store, source_commit="a" * 40)
        self.payload = {
            "strategy_run_id": self.spec.strategy_run_id,
            "evaluation_run_id": self.evaluation.evaluation_run_id,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_same_evidence_is_deduplicated_and_reference_only(self) -> None:
        first, replayed = self.service.create(self.payload, idempotency_key="first")
        second, replayed_again = self.service.create(self.payload, idempotency_key="second")
        report = first["decision_report"]
        self.assertFalse(replayed)
        self.assertTrue(replayed_again)
        self.assertEqual(report["decision_report_id"], second["decision_report"]["decision_report_id"])
        self.assertEqual(len(self.store.decision_report_history()), 1)
        self.assertNotIn("raw_metrics", report["evidence_references"])
        self.assertNotIn("economic_metrics", report["evidence_references"])
        self.assertIn("candidate_hash", report["evidence_references"]["evaluation"])

    def test_idempotency_conflict_and_registry_rebuild_fail_closed(self) -> None:
        self.service.create(self.payload, idempotency_key="same")
        with self.assertRaises(DecisionReportError) as raised:
            self.service.create({**self.payload, "robustness_plan_id": "other"}, idempotency_key="same")
        self.assertEqual(raised.exception.code, "decision_report_idempotency_conflict")
        rebuilt = SavedRunRegistryBuilder(self.store).rebuild(persist=False)
        self.assertEqual(len(rebuilt.decision_reports), 1)
        self.assertEqual(rebuilt.decision_reports[0].integrity_status.value, "valid")

    def test_api_create_list_and_detail_do_not_need_execution_services(self) -> None:
        api = ReadOnlyTrendApi(self.store, decision_report_service=self.service)
        created = api.dispatch(
            "POST", "/api/v1/decision-reports", headers={"Idempotency-Key": "api-create"}, body=self.payload
        )
        self.assertEqual(created.status_code, 201)
        report_id = created.body["decision_report"]["decision_report_id"]
        listed = api.dispatch("GET", "/api/v1/decision-reports?page_size=10")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual([item["decision_report_id"] for item in listed.body["items"]], [report_id])
        detail = api.dispatch("GET", f"/api/v1/decision-reports/{report_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(detail.body["decision_state"], {"pass", "fail", "incomplete"})
        self.assertIn("개인화된 투자 조언", (ROOT / "src/trend_v2_foundation/ui_assets/app.js").read_text(encoding="utf-8"))
