from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.trend_v2_foundation import (
    API_VERSION,
    ArtifactAvailability,
    ArtifactKind,
    ArtifactRetentionPolicy,
    ApiServerConfig,
    AttemptOperationalStatus,
    AttemptTerminalOutcome,
    ExecutionAttempt,
    ExecutionStatus,
    FileExecutionAttemptRepository,
    IntegrityStatus,
    LocalResultStore,
    ReadOnlyTrendApi,
    SavedRunRegistryBuilder,
    StrategyRunManifest,
    StrategyRunSpec,
    calculate_and_evaluate_saved_runs,
    load_evaluation_profiles,
    load_terminology_source,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "config" / "trend_v2" / "evaluation_profiles"
TERMINOLOGY_PATH = ROOT / "config" / "trend_v2" / "terminology_ko.json"
CREATED_AT = "2026-08-01T00:00:00Z"


def policy() -> ArtifactRetentionPolicy:
    return ArtifactRetentionPolicy(
        max_store_bytes=30_000_000,
        max_artifact_bytes=5_000_000,
        max_strategy_runs=20,
        max_evaluation_runs=20,
    )


def curve(returns: list[float], *, exposure: float = 0.8) -> dict:
    dates = [item.date().isoformat() for item in pd.bdate_range("2024-01-02", periods=len(returns))]
    values = 1_000.0 * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    rows = [
        {
            "economic_date": economic_date,
            "portfolio_value": float(value),
            "daily_return": float(daily_return),
            "gross_exposure": exposure,
            "net_exposure": exposure,
            "cash_weight": 1.0 - exposure,
            "daily_turnover": 0.01,
            "transaction_cost": 0.01,
            "position_count": 3,
        }
        for economic_date, value, daily_return in zip(dates, values, returns)
    ]
    return {
        "schema_version": "daily_portfolio_curve_v1",
        "economic_date_range": {"start": dates[0], "end": dates[-1]},
        "rows": rows,
    }


def robustness() -> dict:
    return {
        "schema_version": "robustness_summary_v1",
        "walk_forward_fold_count": 3,
        "walk_forward_pass_ratio": 2 / 3,
        "walk_forward_worst_fold": -0.01,
        "loyo_case_count": 2,
        "loyo_stability_ratio": 1.0,
        "loyo_reversing_years": [],
        "block_bootstrap_effect": 0.02,
        "bootstrap_confidence_interval": {
            "lower": 0.001,
            "upper": 0.04,
            "confidence_level": 0.95,
        },
        "raw_p_value": 0.02,
        "adjusted_p_value": 0.04,
        "multiple_testing_method": "holm",
        "transaction_cost_stress_survival": 1.0,
        "dominant_asset_group": "equity",
        "dominant_group_share": 0.5,
        "unclassified_group_share": 0.0,
        "method_metadata": {"seeds": [11], "sample_counts": {"bootstrap": 100}},
        "unavailable_reasons": {},
    }


def spec(threshold: float, date_range: dict) -> StrategyRunSpec:
    return StrategyRunSpec(
        data_snapshot_hash=("a" if threshold == 1.0 else "b") * 64,
        economic_date_range=date_range,
        universe_specification={"name": "synthetic"},
        benchmark={"symbol": "SPY", "identity": "synthetic_spy_v1"},
        trend_filter={"family": "price_above_ma", "days": 20},
        signal={"family": "synthetic_breakout", "threshold": threshold},
        entry_rule={"family": "next_open"},
        initial_stop={"family": "atr", "multiple": 2.0},
        trailing_exit={"family": "channel", "days": 10},
        position_sizing={"family": "equal"},
        portfolio_constraints={"max_positions": 5},
        transaction_costs={"bps": 5},
        slippage={"bps": 2},
        engine_version="synthetic_engine_v1",
    )


class Foundation3Fixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = LocalResultStore(self.temporary.name, policy())
        benchmark_returns = [0.0002] * 90
        benchmark_returns[30] = -0.04
        self.benchmark = curve(benchmark_returns, exposure=1.0)
        first_returns = [0.00035] * 90
        first_returns[30] = -0.015
        second_returns = [0.00030] * 90
        second_returns[30] = -0.02
        self.first, self.first_daily_record = self._save_run(1.0, curve(first_returns))
        self.second, self.second_daily_record = self._save_run(1.1, curve(second_returns, exposure=0.7))
        profiles = load_evaluation_profiles(PROFILE_DIR)
        self.research = calculate_and_evaluate_saved_runs(
            self.store,
            (self.first.strategy_run_id, self.second.strategy_run_id),
            profiles["research_default"],
            creation_time=CREATED_AT,
        )
        self.weighted = calculate_and_evaluate_saved_runs(
            self.store,
            (self.first.strategy_run_id, self.second.strategy_run_id),
            profiles["exploratory_weighted_example"],
            creation_time="2026-08-01T00:01:00Z",
        )
        self.attempts = FileExecutionAttemptRepository(
            Path(self.temporary.name) / "execution_attempts"
        )
        self.attempt = ExecutionAttempt.create(
            spec(1.0, self.first.canonical_specification["economic_date_range"]),
            attempt_number=1,
            created_timestamp="2026-08-01T00:00:00Z",
            source_commit="c" * 40,
            engine_version="synthetic_engine_v1",
        )
        self.attempts.save(self.attempt)
        self.attempt = self.attempts.transition(
            self.attempt.execution_attempt_id,
            operational_status=AttemptOperationalStatus.RUNNING,
            started_timestamp="2026-08-01T00:00:01Z",
            current_stage="stored_artifact_validation",
            progress_summary={"completed_units": 1, "total_units": 2},
        )
        self.attempt = self.attempts.transition(
            self.attempt.execution_attempt_id,
            operational_status=AttemptOperationalStatus.COMPLETED,
            terminal_outcome=AttemptTerminalOutcome.SUCCEEDED,
            completed_timestamp="2026-08-01T00:00:02Z",
            current_stage="complete",
            progress_summary={"completed_units": 2, "total_units": 2},
            artifact_references=(
                {"strategy_run_id": self.first.strategy_run_id, "artifact_key": "daily_portfolio_curve"},
            ),
        )
        self.builder = SavedRunRegistryBuilder(self.store, self.attempts)
        self.api = ReadOnlyTrendApi(
            self.store,
            registry_builder=self.builder,
            attempt_repository=self.attempts,
            terminology_source=load_terminology_source(TERMINOLOGY_PATH),
        )

    def _save_run(self, threshold: float, daily: dict):
        daily_record = self.store.put_artifact(
            "daily_portfolio_curve",
            ArtifactKind.DAILY_PORTFOLIO_CURVE,
            daily,
            row_count=len(daily["rows"]),
        ).record
        benchmark_record = self.store.put_artifact(
            "benchmark_daily_portfolio_curve",
            ArtifactKind.DAILY_PORTFOLIO_CURVE,
            self.benchmark,
            row_count=len(self.benchmark["rows"]),
        ).record
        robustness_record = self.store.put_artifact(
            "robustness_summary", ArtifactKind.ROBUSTNESS_SUMMARY, robustness(), row_count=1
        ).record
        trades = {
            "rows": [
                {
                    "symbol": "AAA",
                    "entry_date": daily["rows"][5]["economic_date"],
                    "exit_date": daily["rows"][25]["economic_date"],
                }
            ]
        }
        trade_record = self.store.put_artifact(
            "trade_lifecycles", ArtifactKind.TRADE_LIFECYCLES, trades, row_count=1
        ).record
        manifest = StrategyRunManifest.create(
            spec(threshold, daily["economic_date_range"]),
            source_code_commit="c" * 40,
            artifacts=(daily_record, benchmark_record, robustness_record, trade_record),
            creation_time=(CREATED_AT if threshold == 1.0 else "2026-08-01T00:00:30Z"),
            execution_status=ExecutionStatus.SUCCEEDED,
            warnings=("synthetic_only",),
        )
        self.store.save_strategy_run(manifest)
        return manifest, daily_record

    def close(self) -> None:
        self.temporary.cleanup()


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Foundation3Fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_rebuild_is_deterministic_and_ordered(self) -> None:
        first = self.fixture.builder.rebuild()
        second = self.fixture.builder.rebuild()
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.registry_id, second.registry_id)
        self.assertEqual(
            [item.strategy_run_id for item in first.strategy_runs],
            sorted(item.strategy_run_id for item in first.strategy_runs),
        )
        restored = self.fixture.builder.load_or_rebuild()
        self.assertEqual(restored.to_dict(), first.to_dict())

    def test_duplicate_and_orphan_handling_is_explicit(self) -> None:
        source = (
            Path(self.fixture.temporary.name)
            / "strategy_runs"
            / self.fixture.first.strategy_run_id
            / "manifest.json"
        )
        duplicate = Path(self.fixture.temporary.name) / "strategy_runs" / "duplicate" / "manifest.json"
        duplicate.parent.mkdir(parents=True)
        shutil.copyfile(source, duplicate)
        self.fixture.store.put_artifact(
            "orphan", ArtifactKind.SUMMARY_METRICS, {"value": 1}, row_count=1
        )
        registry = self.fixture.builder.rebuild()
        self.assertEqual(len(registry.strategy_runs), 2)
        self.assertTrue(
            any(item.issue_code == "duplicate_strategy_run_equivalent" for item in registry.issues)
        )
        self.assertEqual(len(registry.orphan_object_hashes), 1)
        self.assertTrue(any(item.issue_code == "orphan_object" for item in registry.issues))

    def test_missing_corrupt_and_pruned_are_distinct(self) -> None:
        path = self.fixture.store.object_path_for_hash(self.fixture.first_daily_record.content_hash)
        path.unlink()
        missing = self.fixture.builder.rebuild()
        missing_run = next(
            item for item in missing.strategy_runs if item.strategy_run_id == self.fixture.first.strategy_run_id
        )
        missing_artifact = next(
            item for item in missing_run.artifacts if item.artifact_key == "daily_portfolio_curve"
        )
        self.assertEqual(missing_artifact.availability, ArtifactAvailability.MISSING)

        second_path = self.fixture.store.object_path_for_hash(self.fixture.second_daily_record.content_hash)
        second_path.write_bytes(b"not-gzip")
        corrupt = self.fixture.builder.rebuild()
        corrupt_run = next(
            item for item in corrupt.strategy_runs if item.strategy_run_id == self.fixture.second.strategy_run_id
        )
        corrupt_artifact = next(
            item for item in corrupt_run.artifacts if item.artifact_key == "daily_portfolio_curve"
        )
        self.assertEqual(corrupt_artifact.availability, ArtifactAvailability.CORRUPT)

        with tempfile.TemporaryDirectory() as directory:
            store = LocalResultStore(directory, policy())
            payload = curve([0.0, 0.01])
            record = store.put_artifact(
                "daily_portfolio_curve",
                ArtifactKind.DAILY_PORTFOLIO_CURVE,
                payload,
                row_count=2,
            ).record
            manifest = StrategyRunManifest.create(
                spec(1.0, payload["economic_date_range"]),
                source_code_commit="d" * 40,
                artifacts=(record,),
                creation_time=CREATED_AT,
            )
            store.save_strategy_run(manifest)
            store.mark_artifact_pruned(
                record.content_hash,
                pruned_at="2026-08-01T01:00:00Z",
                reason="synthetic_retention_test",
            )
            registry = SavedRunRegistryBuilder(store).rebuild()
            artifact = next(
                item
                for item in registry.strategy_runs[0].artifacts
                if item.artifact_key == "daily_portfolio_curve"
            )
            self.assertEqual(artifact.availability, ArtifactAvailability.PRUNED)
            self.assertEqual(registry.strategy_runs[0].retention_status, "pruned")

    def test_registry_connects_attempts_profiles_evaluations_and_provenance(self) -> None:
        registry = self.fixture.builder.rebuild()
        run = next(
            item for item in registry.strategy_runs if item.strategy_run_id == self.fixture.first.strategy_run_id
        )
        self.assertEqual(len(run.evaluation_run_ids), 2)
        self.assertEqual(len(run.evaluation_profile_ids), 2)
        self.assertEqual(run.execution_attempt_ids, (self.fixture.attempt.execution_attempt_id,))
        self.assertEqual(len(run.derived_metric_ids), 1)
        self.assertEqual(run.integrity_status, IntegrityStatus.VALID)
        self.assertTrue(run.benchmark_provenance)
        self.assertTrue(run.calculation_provenance)
        self.assertTrue(all(item["source_artifact_hashes"] for item in run.calculation_provenance))


class ExecutionAttemptTests(unittest.TestCase):
    def test_lifecycle_retry_identity_and_strategy_status_separation(self) -> None:
        payload = curve([0.0, 0.01])
        run_spec = spec(1.0, payload["economic_date_range"])
        first = ExecutionAttempt.create(
            run_spec,
            attempt_number=1,
            created_timestamp=CREATED_AT,
            source_commit="e" * 40,
            engine_version="synthetic_engine_v1",
        )
        retry = ExecutionAttempt.create(
            run_spec,
            attempt_number=2,
            retry_of_execution_attempt_id=first.execution_attempt_id,
            created_timestamp="2026-08-01T00:01:00Z",
            source_commit="e" * 40,
            engine_version="synthetic_engine_v1",
        )
        self.assertNotEqual(first.execution_attempt_id, retry.execution_attempt_id)
        self.assertEqual(first.intended_strategy_run_id, retry.intended_strategy_run_id)
        with self.assertRaisesRegex(ValueError, "must be terminal"):
            StrategyRunManifest.create(
                run_spec,
                source_code_commit="e" * 40,
                artifacts=(),
                creation_time=CREATED_AT,
                execution_status=ExecutionStatus.RUNNING,
            )

    def test_invalid_timestamp_and_state_combinations_fail(self) -> None:
        payload = curve([0.0, 0.01])
        attempt = ExecutionAttempt.create(
            spec(1.0, payload["economic_date_range"]),
            attempt_number=1,
            created_timestamp=CREATED_AT,
            source_commit="e" * 40,
            engine_version="synthetic_engine_v1",
        )
        with self.assertRaisesRegex(ValueError, "require only a start timestamp"):
            replace(attempt, operational_status=AttemptOperationalStatus.RUNNING)
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            replace(
                attempt,
                operational_status=AttemptOperationalStatus.RUNNING,
                started_timestamp="2025-01-01T00:00:00Z",
            )
        with self.assertRaisesRegex(ValueError, "failure code and message"):
            replace(
                attempt,
                operational_status=AttemptOperationalStatus.FAILED,
                terminal_outcome=AttemptTerminalOutcome.FAILED,
                started_timestamp="2026-08-01T00:00:01Z",
                completed_timestamp="2026-08-01T00:00:02Z",
            )

    def test_repository_rejects_invalid_transition_and_terminal_mutation(self) -> None:
        fixture = Foundation3Fixture()
        try:
            with self.assertRaisesRegex(ValueError, "terminal execution attempts are immutable"):
                fixture.attempts.transition(
                    fixture.attempt.execution_attempt_id,
                    current_stage="changed_after_completion",
                )
        finally:
            fixture.close()


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Foundation3Fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def get(self, target: str):
        return self.fixture.api.dispatch("GET", target, headers={"X-Request-ID": "test-request"})

    def test_health_metadata_and_local_defaults(self) -> None:
        health = self.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.body["api_version"], API_VERSION)
        metadata = self.get("/api/v1/metadata")
        self.assertEqual(metadata.status_code, 200)
        self.assertIn("daily_portfolio_curve_v1", metadata.body["supported_artifact_schema_versions"])
        self.assertIn("benchmark_coverage_failure", metadata.body.get("error_codes", []))
        self.assertEqual(ApiServerConfig().host, "127.0.0.1")
        self.assertEqual(ApiServerConfig().cors_origins, ())
        with self.assertRaisesRegex(ValueError, "loopback"):
            ApiServerConfig(host="0.0.0.0")

    def test_run_list_detail_manifest_spec_provenance_status_and_artifacts(self) -> None:
        response = self.get("/api/v1/runs?page_size=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.body["items"]), 1)
        self.assertIsNotNone(response.body["page"]["next_cursor"])
        run_id = self.fixture.first.strategy_run_id
        for suffix in ("", "/manifest", "/specification", "/provenance", "/status", "/artifacts"):
            with self.subTest(suffix=suffix):
                detail = self.get(f"/api/v1/runs/{run_id}{suffix}")
                self.assertEqual(detail.status_code, 200, detail.body)
        provenance = self.get(f"/api/v1/runs/{run_id}/provenance").body
        self.assertEqual(provenance["source_data_snapshot_id"], "a" * 64)
        self.assertTrue(provenance["calculation_provenance"])

    def test_filter_sort_allow_lists_and_invalid_queries(self) -> None:
        first = self.fixture.first
        good = self.get(
            "/api/v1/runs?status=succeeded&engine_version=synthetic_engine_v1"
            "&artifact_key=daily_portfolio_curve&artifact_availability=available"
            "&sort=-creation_time"
        )
        self.assertEqual(good.status_code, 200)
        self.assertEqual(len(good.body["items"]), 2)
        profile_filter = self.get(
            f"/api/v1/runs?profile_id={self.fixture.research.evaluation_run.evaluation_profile_id}"
        )
        self.assertEqual(len(profile_filter.body["items"]), 2)
        snapshot_filter = self.get(f"/api/v1/runs?data_snapshot_id={first.snapshot_hash}")
        self.assertEqual(len(snapshot_filter.body["items"]), 1)
        for target in (
            "/api/v1/runs?unknown=value",
            "/api/v1/runs?sort=arbitrary",
            "/api/v1/runs?status=running",
            "/api/v1/runs?page_size=201",
            "/api/v1/runs?cursor=not-a-cursor",
        ):
            with self.subTest(target=target):
                response = self.get(target)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.body["error"]["code"], "invalid_query")

    def test_artifact_reads_are_bounded_and_do_not_execute_economic_backtests(self) -> None:
        run_id = self.fixture.first.strategy_run_id
        self.assertEqual(set(self.fixture.research.cache_status.values()), {"calculated"})
        self.assertEqual(set(self.fixture.weighted.cache_status.values()), {"reused"})
        history_before = self.fixture.store.derived_metric_history()
        evaluations_before = self.fixture.store.evaluation_history()
        first = self.get(f"/api/v1/runs/{run_id}/curve?page_size=2")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(len(first.body["items"]), 2)
        cursor = first.body["page"]["next_cursor"]
        second = self.get(f"/api/v1/runs/{run_id}/curve?page_size=2&cursor={cursor}")
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(first.body["items"], second.body["items"])
        start = first.body["items"][1]["economic_date"]
        ranged = self.get(f"/api/v1/runs/{run_id}/curve?start_date={start}&page_size=3")
        self.assertTrue(all(item["economic_date"] >= start for item in ranged.body["items"]))
        rolling = self.get(f"/api/v1/runs/{run_id}/rolling-metrics?window_sessions=63&page_size=5")
        self.assertEqual(rolling.status_code, 200)
        derived = self.get(f"/api/v1/runs/{run_id}/derived-metrics")
        self.assertEqual(derived.status_code, 200)
        self.assertEqual(self.fixture.store.derived_metric_history(), history_before)
        self.assertEqual(self.fixture.store.evaluation_history(), evaluations_before)
        too_large = self.get(f"/api/v1/runs/{run_id}/curve?page_size=1001")
        self.assertEqual(too_large.status_code, 400)

    def test_profile_evaluation_outputs_behavior_and_exact_hash(self) -> None:
        profile_id = self.fixture.research.evaluation_run.evaluation_profile_id
        profile = self.get(f"/api/v1/evaluation-profiles/{profile_id}")
        self.assertEqual(profile.status_code, 200)
        evaluations = self.get(
            f"/api/v1/evaluation-runs?strategy_run_id={self.fixture.first.strategy_run_id}"
        )
        self.assertEqual(len(evaluations.body["items"]), 2)
        evaluation_id = self.fixture.research.evaluation_run.evaluation_run_id
        detail = self.get(f"/api/v1/evaluation-runs/{evaluation_id}")
        self.assertEqual(detail.body["profile_hash"], self.fixture.research.evaluation_run.profile_hash)
        outputs = self.get(f"/api/v1/evaluation-runs/{evaluation_id}/outputs")
        self.assertIn("mandatory_gates", outputs.body["items"][0])
        self.assertIn("pareto", outputs.body["items"][0])
        self.assertIn("robustness_vetoes", outputs.body["items"][0])
        self.assertIn("tie_break_order", outputs.body["items"][0])
        self.assertIn("exploratory_weighted", outputs.body["items"][0])
        behavior = self.get(f"/api/v1/evaluation-runs/{evaluation_id}/behavior")
        diagnostics = behavior.body["pairwise_diagnostics"][0]["diagnostics"]
        for key in (
            "daily_return_correlation",
            "active_date_jaccard",
            "entry_date_jaccard",
            "exit_date_jaccard",
            "normalized_path_distance",
        ):
            self.assertIn(key, diagnostics)
        self.assertTrue(behavior.body["simplicity_metadata"])
        self.assertTrue(behavior.body["candidate_clusters"])

    def test_execution_attempts_are_separate_and_filterable(self) -> None:
        response = self.get(
            "/api/v1/execution-attempts?operational_status=completed"
            f"&intended_strategy_run_id={self.fixture.first.strategy_run_id}"
        )
        self.assertEqual(len(response.body["items"]), 1)
        item = response.body["items"][0]
        self.assertEqual(item["execution_attempt_id"], self.fixture.attempt.execution_attempt_id)
        self.assertEqual(item["operational_status"], "completed")
        manifest = self.get(f"/api/v1/runs/{self.fixture.first.strategy_run_id}/manifest")
        self.assertEqual(manifest.body["execution_status"], "succeeded")
        detail = self.get(f"/api/v1/execution-attempts/{self.fixture.attempt.execution_attempt_id}")
        self.assertEqual(detail.status_code, 200)

    def test_missing_corrupt_and_pruned_error_contracts_are_korean_and_path_safe(self) -> None:
        path = self.fixture.store.object_path_for_hash(self.fixture.first_daily_record.content_hash)
        path.unlink()
        missing = self.get(f"/api/v1/runs/{self.fixture.first.strategy_run_id}/curve")
        self.assertEqual(missing.body["error"]["code"], "artifact_missing")
        self.assertTrue(missing.body["error"]["message_ko"])
        self.assertEqual(missing.body["error"]["request_id"], "test-request")
        rendered = json.dumps(missing.body, ensure_ascii=False)
        self.assertNotIn(self.fixture.temporary.name, rendered)
        self.assertNotRegex(rendered, r"[A-Za-z]:\\")

        second_path = self.fixture.store.object_path_for_hash(self.fixture.second_daily_record.content_hash)
        second_path.write_bytes(b"broken")
        corrupt = self.get(f"/api/v1/runs/{self.fixture.second.strategy_run_id}/curve")
        self.assertEqual(corrupt.body["error"]["code"], "artifact_corrupt")

        with tempfile.TemporaryDirectory() as directory:
            store = LocalResultStore(directory, policy())
            payload = curve([0.0, 0.01])
            record = store.put_artifact(
                "daily_portfolio_curve", ArtifactKind.DAILY_PORTFOLIO_CURVE, payload, row_count=2
            ).record
            manifest = StrategyRunManifest.create(
                spec(1.0, payload["economic_date_range"]),
                source_code_commit="f" * 40,
                artifacts=(record,),
                creation_time=CREATED_AT,
            )
            store.save_strategy_run(manifest)
            store.mark_artifact_pruned(
                record.content_hash,
                pruned_at="2026-08-01T01:00:00Z",
                reason="test",
            )
            api = ReadOnlyTrendApi(
                store,
                terminology_source=load_terminology_source(TERMINOLOGY_PATH),
            )
            pruned = api.dispatch("GET", f"/api/v1/runs/{manifest.strategy_run_id}/curve")
            self.assertEqual(pruned.status_code, 410)
            self.assertEqual(pruned.body["error"]["code"], "retention_pruned_artifact")

    def test_path_traversal_write_methods_and_unknown_routes_fail_closed(self) -> None:
        traversal = self.get("/api/v1/runs/..%2Fretention_policy.json")
        self.assertEqual(traversal.status_code, 400)
        self.assertEqual(traversal.body["error"]["code"], "invalid_identifier")
        write = self.fixture.api.dispatch("POST", "/api/v1/runs")
        self.assertEqual(write.status_code, 405)
        self.assertEqual(write.body["error"]["code"], "method_not_allowed")
        unknown = self.get("/api/v1/not-real")
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.body["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
