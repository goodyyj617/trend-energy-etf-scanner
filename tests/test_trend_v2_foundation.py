import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.trend_v2_foundation import (
    ArtifactKind,
    ArtifactRetentionPolicy,
    EvaluationProfile,
    ExecutionStatus,
    LocalResultStore,
    ParetoObjective,
    StrategyRunManifest,
    StrategyRunSpec,
    canonical_json,
    epsilon_pareto,
    evaluate_saved_runs,
    evaluate_strategy_runs,
    load_evaluation_profiles,
    load_terminology_source,
)
from src.trend_v2_foundation.contracts import MetricDirection
from src.trend_v2_foundation.metrics import METRIC_REGISTRY, metrics_from_portfolio_summaries
from src.trend_v2_foundation.terminology import REQUIRED_ENTRY_FIELDS


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "config" / "trend_v2" / "evaluation_profiles"
TERMINOLOGY_PATH = ROOT / "config" / "trend_v2" / "terminology_ko.json"
CREATED_AT = "2026-08-01T00:00:00Z"


def strategy_spec(*, signal_threshold: float = 1.0) -> StrategyRunSpec:
    return StrategyRunSpec(
        data_snapshot_hash="a" * 64,
        economic_date_range={"start": "2020-01-02", "end": "2025-12-31"},
        universe_specification={"key": "synthetic", "version": "v1"},
        benchmark={"symbol": "SPY", "return": "adjusted_close"},
        trend_filter={"key": "price_above_ma", "version": "v1", "parameters": {"days": 200}},
        signal={"key": "synthetic_breakout", "version": "v1", "parameters": {"threshold": signal_threshold}},
        entry_rule={"key": "next_open", "version": "v1"},
        initial_stop={"key": "atr_stop", "version": "v1", "parameters": {"multiple": 2.0}},
        trailing_exit={"key": "channel_exit", "version": "v1", "parameters": {"days": 20}},
        position_sizing={"key": "equal_weight", "version": "v1"},
        portfolio_constraints={"max_positions": 10, "max_weight": 0.2},
        transaction_costs={"round_trip_rate": 0.002},
        slippage={"model": "fixed_bps", "basis_points": 5},
        engine_version="synthetic_engine_v1",
    )


def metrics(
    *,
    cagr_ratio: float = 0.9,
    mdd_ratio: float = 0.6,
    cdar_ratio: float = 0.7,
    calmar_ratio: float = 1.1,
    recovery: int = 100,
    turnover: float = 1.0,
) -> dict:
    return {
        "cagr_spy_ratio": cagr_ratio,
        "maximum_drawdown_spy_ratio": mdd_ratio,
        "cdar95_spy_ratio": cdar_ratio,
        "calmar_spy_ratio": calmar_ratio,
        "recovery_duration_days": recovery,
        "annual_turnover": turnover,
    }


def policy() -> ArtifactRetentionPolicy:
    return ArtifactRetentionPolicy(
        max_store_bytes=2_000_000,
        max_artifact_bytes=200_000,
        max_strategy_runs=20,
        max_evaluation_runs=20,
    )


class CanonicalIdentityTests(unittest.TestCase):
    def test_canonical_serialization_ignores_mapping_insertion_order(self) -> None:
        left = {"z": [3, {"b": 2, "a": 1}], "a": "한글"}
        right = {"a": "한글", "z": [3, {"a": 1, "b": 2}]}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(canonical_json(left), '{"a":"한글","z":[3,{"a":1,"b":2}]}')

    def test_strategy_run_id_is_stable(self) -> None:
        first = strategy_spec()
        second = StrategyRunSpec.from_dict(json.loads(canonical_json(first)))
        self.assertEqual(first.strategy_run_id, second.strategy_run_id)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_economic_rule_change_invalidates_strategy_run_id(self) -> None:
        self.assertNotEqual(
            strategy_spec(signal_threshold=1.0).strategy_run_id,
            strategy_spec(signal_threshold=1.1).strategy_run_id,
        )

    def test_threshold_only_change_does_not_invalidate_strategy_run_id(self) -> None:
        profiles = load_evaluation_profiles(PROFILE_DIR)
        profile = profiles["research_default"]
        changed_gate = replace(profile.mandatory_gates[0], threshold=0.85)
        changed_profile = replace(
            profile, mandatory_gates=(changed_gate,) + profile.mandatory_gates[1:]
        )
        self.assertEqual(strategy_spec().strategy_run_id, strategy_spec().strategy_run_id)
        self.assertNotEqual(profile.evaluation_profile_id, changed_profile.evaluation_profile_id)

    def test_evaluation_profile_id_is_stable_and_sensitive_to_weights(self) -> None:
        profile = load_evaluation_profiles(PROFILE_DIR)["exploratory_weighted_example"]
        clone = EvaluationProfile.from_dict(json.loads(canonical_json(profile)))
        self.assertEqual(profile.profile_hash, clone.profile_hash)
        self.assertEqual(profile.evaluation_profile_id, clone.evaluation_profile_id)
        changed_weights = dict(profile.exploratory_metric_weights)
        changed_weights["annual_turnover"] = 0.01
        changed = replace(profile, exploratory_metric_weights=changed_weights)
        self.assertNotEqual(profile.evaluation_profile_id, changed.evaluation_profile_id)


class EvaluationPipelineTests(unittest.TestCase):
    def test_epsilon_pareto_behavior(self) -> None:
        observations = {
            "a": {"growth": 1.0, "risk": 0.2},
            "b": {"growth": 0.99995, "risk": 0.20005},
            "c": {"growth": 0.8, "risk": 0.4},
        }
        strict, strict_dominated = epsilon_pareto(
            observations,
            (
                ParetoObjective("growth", MetricDirection.MAXIMIZE, 0.0),
                ParetoObjective("risk", MetricDirection.MINIMIZE, 0.0),
            ),
        )
        tolerant, tolerant_dominated = epsilon_pareto(
            observations,
            (
                ParetoObjective("growth", MetricDirection.MAXIMIZE, 0.0001),
                ParetoObjective("risk", MetricDirection.MINIMIZE, 0.0001),
            ),
        )
        self.assertEqual(strict, {"a"})
        self.assertEqual(tolerant, {"a", "b"})
        self.assertEqual(strict_dominated["b"], ("a",))
        self.assertEqual(tolerant_dominated["c"], ("a", "b"))

    def test_weighted_mode_never_overrides_mandatory_gate_failure(self) -> None:
        profile = load_evaluation_profiles(PROFILE_DIR)["exploratory_weighted_example"]
        run = evaluate_strategy_runs(
            profile,
            {
                "fails_gate_but_scores_high": metrics(
                    cagr_ratio=0.79,
                    mdd_ratio=0.1,
                    cdar_ratio=0.1,
                    calmar_ratio=4.0,
                    recovery=5,
                    turnover=0.1,
                ),
                "passes": metrics(cagr_ratio=0.9, mdd_ratio=0.7, cdar_ratio=0.75),
            },
            benchmark_data_identity="synthetic_spy_v1",
            creation_time=CREATED_AT,
        )
        result = {item.strategy_run_id: item for item in run.results}["fails_gate_but_scores_high"]
        self.assertFalse(result.mandatory_gates_passed)
        self.assertIn("mandatory_gate_failed", result.final_labels)
        self.assertNotIn("constraint_pareto_selected", result.final_labels)
        self.assertIsNotNone(result.weighted_view)
        self.assertEqual(result.weighted_view.rank, 1)
        self.assertIn(
            "high_weighted_rank_but_mandatory_gate_failed", result.weighted_view.warnings
        )
        self.assertAlmostEqual(sum(run.normalized_weights.values()), 1.0)
        self.assertIn("baseline", run.ranking_sensitivity["scenarios"])

    def test_evaluation_run_identity_is_input_order_independent(self) -> None:
        profile = load_evaluation_profiles(PROFILE_DIR)["research_default"]
        left = evaluate_strategy_runs(
            profile,
            {"b": metrics(), "a": metrics(cagr_ratio=0.95)},
            benchmark_data_identity="spy_hash",
            creation_time=CREATED_AT,
        )
        right = evaluate_strategy_runs(
            profile,
            {"a": metrics(cagr_ratio=0.95), "b": metrics()},
            benchmark_data_identity="spy_hash",
            creation_time="2026-08-01T01:00:00Z",
        )
        self.assertEqual(left.evaluation_run_id, right.evaluation_run_id)


class ResultStoreTests(unittest.TestCase):
    def _save_run(self, store: LocalResultStore, spec: StrategyRunSpec, summary: dict):
        summary_put = store.put_artifact(
            "summary_metrics", ArtifactKind.SUMMARY_METRICS, summary, row_count=1
        )
        behavior_put = store.put_artifact(
            "behavior_metadata",
            ArtifactKind.BEHAVIOR_METADATA,
            {"behavior_group_id": "B001", "normalized_return_hash": "f" * 64},
            row_count=1,
        )
        manifest = StrategyRunManifest.create(
            spec,
            source_code_commit="b" * 40,
            artifacts=(summary_put.record, behavior_put.record),
            creation_time=CREATED_AT,
            execution_status=ExecutionStatus.SUCCEEDED,
            warnings=("synthetic_test_data",),
            limitations=("not_market_data",),
        )
        store.save_strategy_run(manifest)
        return manifest, summary_put

    def test_result_store_content_deduplication_and_orphan_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalResultStore(directory, policy())
            first = store.put_artifact(
                "yearly_metrics_a", ArtifactKind.YEARLY_METRICS, [{"year": 2024, "cagr": 0.1}], row_count=1
            )
            second = store.put_artifact(
                "yearly_metrics_b", ArtifactKind.YEARLY_METRICS, [{"cagr": 0.1, "year": 2024}], row_count=1
            )
            self.assertEqual(first.record.content_hash, second.record.content_hash)
            self.assertEqual(first.retention_status, "retained")
            self.assertEqual(second.retention_status, "deduplicated")
            self.assertEqual(store.retention_status().orphan_object_count, 1)
            self.assertGreater(store.estimate_artifact_size([{"x": 1}]).stored_bytes, 0)

    def test_artifact_hash_validation_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalResultStore(directory, policy())
            manifest, summary_put = self._save_run(store, strategy_spec(), metrics())
            self.assertTrue(store.validate_manifest(manifest.strategy_run_id).valid)
            object_path = Path(directory) / "objects" / "sha256" / f"{summary_put.record.content_hash}.json.gz"
            object_path.write_bytes(gzip.compress(b'{"tampered":true}', mtime=0))
            validation = store.validate_manifest(manifest.strategy_run_id)
            self.assertFalse(validation.valid)
            self.assertTrue(
                any("mismatch:summary_metrics" in error for error in validation.errors),
                validation.errors,
            )

    def test_saved_runs_are_reevaluated_without_backtest_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalResultStore(directory, policy())
            first, _ = self._save_run(store, strategy_spec(signal_threshold=1.0), metrics())
            second, _ = self._save_run(
                store,
                strategy_spec(signal_threshold=1.1),
                metrics(cagr_ratio=0.95, mdd_ratio=0.55, cdar_ratio=0.65),
            )
            profile = load_evaluation_profiles(PROFILE_DIR)["research_default"]
            run = evaluate_saved_runs(
                store,
                (second.strategy_run_id, first.strategy_run_id),
                profile,
                benchmark_data_identity="synthetic_spy_snapshot_hash",
                creation_time=CREATED_AT,
            )
            restored = store.get_evaluation_run(run.evaluation_run_id)
            self.assertEqual(restored.evaluation_run_id, run.evaluation_run_id)
            self.assertEqual(restored.strategy_run_ids, tuple(sorted((first.strategy_run_id, second.strategy_run_id))))
            self.assertEqual(store.evaluation_history(), (run.evaluation_run_id,))
            self.assertEqual(store.orphan_hashes(), ())


class RegistryAndTerminologyTests(unittest.TestCase):
    def test_registry_connects_existing_reliable_summary_metrics(self) -> None:
        strategy = {
            "cagr": 0.12,
            "annualized_volatility": 0.18,
            "sharpe_ratio": 0.8,
            "sortino_ratio": 1.1,
            "maximum_drawdown": -0.2,
            "conditional_drawdown_at_risk_95": -0.18,
            "calmar_ratio": 0.6,
            "max_drawdown_duration_days": 200,
            "annual_turnover": 1.5,
        }
        benchmark = {
            "cagr": 0.10,
            "maximum_drawdown": -0.4,
            "conditional_drawdown_at_risk_95": -0.3,
            "calmar_ratio": 0.25,
        }
        result = metrics_from_portfolio_summaries(strategy, benchmark)
        self.assertAlmostEqual(result["cagr_spy_ratio"], 1.2)
        self.assertAlmostEqual(result["maximum_drawdown_spy_ratio"], 0.5)
        self.assertAlmostEqual(result["cdar95_spy_ratio"], 0.6)
        self.assertAlmostEqual(result["calmar_spy_ratio"], 2.4)
        self.assertEqual(METRIC_REGISTRY["cdar95"].source_summary_key, "conditional_drawdown_at_risk_95")

    def test_korean_terminology_schema_is_complete(self) -> None:
        source = load_terminology_source(TERMINOLOGY_PATH)
        required_entries = {
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "cdar95",
            "calmar_ratio",
            "recovery_duration",
            "turnover",
            "pareto_dominance",
            "epsilon_dominance",
            "walk_forward",
            "loyo",
            "block_bootstrap",
        }
        self.assertTrue(required_entries <= set(source["entries"]))
        for key, entry in source["entries"].items():
            self.assertEqual(REQUIRED_ENTRY_FIELDS - set(entry), set(), key)
            self.assertTrue(entry["variable_definitions"], key)
            self.assertTrue(entry["worked_numerical_example"], key)


if __name__ == "__main__":
    unittest.main()
