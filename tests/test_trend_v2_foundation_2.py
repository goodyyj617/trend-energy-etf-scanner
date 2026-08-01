from __future__ import annotations

import math
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.portfolio import summarize_portfolio_curve
from src.trend_v2_foundation import (
    ArtifactKind,
    ArtifactRetentionPolicy,
    ArtifactSchemaError,
    ExecutionStatus,
    LocalResultStore,
    StrategyRunManifest,
    StrategyRunSpec,
    behavior_similarity,
    calculate_absolute_metrics,
    calculate_and_evaluate_saved_runs,
    calculation_settings,
    content_hash,
    deduplicate_behaviors,
    exact_common_date_curves,
    generate_behavior_metadata,
    generate_rolling_metrics,
    generate_yearly_metrics,
    load_evaluation_profiles,
    validate_daily_portfolio_curve,
    validate_robustness_summary,
)


CREATED_AT = "2026-08-01T00:00:00Z"
PROFILE_DIR = "config/trend_v2/evaluation_profiles"


def curve(
    returns: list[float],
    *,
    start: str = "2020-01-02",
    dates: list[str] | None = None,
    exposure: float = 0.8,
    gross_returns: list[float] | None = None,
) -> dict:
    if dates is None:
        dates = [value.date().isoformat() for value in pd.bdate_range(start, periods=len(returns))]
    values = (1_000.0 * np.cumprod(1.0 + np.asarray(returns))).tolist()
    gross_values = (
        (1_000.0 * np.cumprod(1.0 + np.asarray(gross_returns))).tolist()
        if gross_returns is not None
        else None
    )
    rows = []
    for index, (economic_date, value, daily_return) in enumerate(zip(dates, values, returns)):
        row = {
            "economic_date": economic_date,
            "portfolio_value": float(value),
            "daily_return": float(daily_return),
            "gross_exposure": exposure,
            "net_exposure": exposure,
            "cash_weight": 1.0 - exposure,
            "daily_turnover": 0.01,
            "transaction_cost": 0.05,
            "position_count": 4,
        }
        if gross_values is not None and gross_returns is not None:
            row["gross_portfolio_value"] = float(gross_values[index])
            row["gross_daily_return"] = float(gross_returns[index])
        rows.append(row)
    return {
        "schema_version": "daily_portfolio_curve_v1",
        "economic_date_range": (
            {"start": dates[0], "end": dates[-1]} if dates else None
        ),
        "rows": rows,
    }


def legacy_curve(payload: dict) -> tuple[pd.DataFrame, float]:
    rows = payload["rows"]
    initial = rows[0]["portfolio_value"] / (1.0 + rows[0]["daily_return"])
    frame = pd.DataFrame(
        {
            "date": [row["economic_date"] for row in rows],
            "observation_type": "trading_session",
            "strategy_key": "legacy",
            "portfolio_equity": [row["portfolio_value"] for row in rows],
            "cash_value": [row["portfolio_value"] * row["cash_weight"] for row in rows],
            "invested_value": [row["portfolio_value"] * row["gross_exposure"] for row in rows],
            "gross_exposure": [row["gross_exposure"] for row in rows],
            "active_position_count": [row["position_count"] for row in rows],
            "daily_portfolio_return": [row["daily_return"] for row in rows],
            "transaction_cost_paid": [row["transaction_cost"] for row in rows],
            "turnover": [row["daily_turnover"] for row in rows],
        }
    )
    equity = frame["portfolio_equity"]
    frame["cumulative_return"] = equity / initial - 1.0
    frame["running_peak_equity"] = equity.cummax().clip(lower=initial)
    frame["drawdown"] = equity / frame["running_peak_equity"] - 1.0
    return frame, initial


def robustness() -> dict:
    return {
        "schema_version": "robustness_summary_v1",
        "walk_forward_fold_count": 5,
        "walk_forward_pass_ratio": 0.8,
        "walk_forward_worst_fold": -0.02,
        "loyo_case_count": 8,
        "loyo_stability_ratio": 0.875,
        "loyo_reversing_years": [2020],
        "block_bootstrap_effect": 0.03,
        "bootstrap_confidence_interval": {
            "lower": 0.01,
            "upper": 0.05,
            "confidence_level": 0.95,
        },
        "raw_p_value": 0.01,
        "adjusted_p_value": 0.03,
        "multiple_testing_method": "holm",
        "transaction_cost_stress_survival": 1.0,
        "dominant_asset_group": "equity",
        "dominant_group_share": 0.4,
        "unclassified_group_share": 0.0,
        "method_metadata": {
            "seeds": [7],
            "sample_counts": {"bootstrap": 1000},
            "source_hashes": ["a" * 64],
        },
        "unavailable_reasons": {},
    }


def spec(threshold: float) -> StrategyRunSpec:
    return StrategyRunSpec(
        data_snapshot_hash="a" * 64,
        economic_date_range={"start": "2020-01-02", "end": "2023-01-25"},
        universe_specification={"name": "synthetic"},
        benchmark={"symbol": "SPY", "identity": "synthetic_spy_v1"},
        trend_filter={"family": "synthetic"},
        signal={"family": "synthetic", "threshold": threshold},
        entry_rule={"family": "next_open"},
        initial_stop={"family": "atr"},
        trailing_exit={"family": "atr"},
        position_sizing={"family": "equal"},
        portfolio_constraints={"max_positions": 5},
        transaction_costs={"bps": 5},
        slippage={"bps": 2},
        engine_version="synthetic_no_backtest_v1",
    )


def policy() -> ArtifactRetentionPolicy:
    return ArtifactRetentionPolicy(
        max_store_bytes=20_000_000,
        max_artifact_bytes=5_000_000,
        max_strategy_runs=20,
        max_evaluation_runs=20,
    )


class StoredArtifactSchemaTests(unittest.TestCase):
    def test_daily_schema_rejects_missing_duplicate_unsorted_nonfinite_and_impossible(self) -> None:
        valid = curve([0.0, 0.01, -0.01])
        validate_daily_portfolio_curve(valid)
        cases = []
        missing = {**valid, "rows": [dict(row) for row in valid["rows"]]}
        del missing["rows"][0]["cash_weight"]
        cases.append((missing, "missing required fields"))
        duplicate = {**valid, "rows": [dict(row) for row in valid["rows"]]}
        duplicate["rows"][1]["economic_date"] = duplicate["rows"][0]["economic_date"]
        duplicate["economic_date_range"] = {
            "start": duplicate["rows"][0]["economic_date"],
            "end": duplicate["rows"][-1]["economic_date"],
        }
        cases.append((duplicate, "duplicate economic dates"))
        unsorted = {**valid, "rows": list(reversed(valid["rows"]))}
        unsorted["economic_date_range"] = {
            "start": unsorted["rows"][0]["economic_date"],
            "end": unsorted["rows"][-1]["economic_date"],
        }
        cases.append((unsorted, "sorted ascending"))
        nonfinite = {**valid, "rows": [dict(row) for row in valid["rows"]]}
        nonfinite["rows"][0]["daily_return"] = math.inf
        cases.append((nonfinite, "finite numeric"))
        impossible = {**valid, "rows": [dict(row) for row in valid["rows"]]}
        impossible["rows"][0]["portfolio_value"] = 0
        cases.append((impossible, "greater than zero"))
        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ArtifactSchemaError, message):
                validate_daily_portfolio_curve(payload)

    def test_robustness_binary_and_missing_fields_fail_clearly(self) -> None:
        valid = robustness()
        validate_robustness_summary(valid)
        bad_binary = {**valid, "transaction_cost_stress_survival": True}
        with self.assertRaisesRegex(ArtifactSchemaError, "binary representation"):
            validate_robustness_summary(bad_binary)
        missing = dict(valid)
        del missing["adjusted_p_value"]
        with self.assertRaisesRegex(ArtifactSchemaError, "missing required fields"):
            validate_robustness_summary(missing)


class MetricParityTests(unittest.TestCase):
    def test_required_legacy_metric_parity_paths(self) -> None:
        irregular_dates = [
            "2020-01-02",
            "2020-01-06",
            "2020-01-07",
            "2020-01-13",
            "2020-01-22",
            "2020-02-03",
        ]
        cases = {
            "normal": curve([0.0, 0.01, -0.02, 0.015, 0.005, -0.003]),
            "flat": curve([0.0] * 6),
            "gain": curve([0.0, 0.01, 0.01, 0.01, 0.01, 0.01]),
            "recovered": curve([0.0, 0.05, -0.1, 0.04, 0.04, 0.04]),
            "unrecovered": curve([0.0, 0.05, -0.1, -0.02, 0.01, 0.01]),
            "irregular": curve([0.0, 0.01, -0.02, 0.015, 0.005, -0.003], dates=irregular_dates),
        }
        mapping = {
            "cagr": "cagr",
            "annualized_volatility": "annualized_volatility",
            "sharpe_ratio": "sharpe_ratio",
            "sortino_ratio": "sortino_ratio",
            "maximum_drawdown": "maximum_drawdown",
            "cdar95": "conditional_drawdown_at_risk_95",
            "calmar_ratio": "calmar_ratio",
            "recovery_duration_days": "max_drawdown_duration_days",
            "annual_turnover": "annual_turnover",
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                rolling = generate_rolling_metrics(payload)
                actual, _ = calculate_absolute_metrics(payload, rolling)
                legacy, initial = legacy_curve(payload)
                expected = summarize_portfolio_curve(
                    "legacy", "Legacy", legacy, initial_capital=initial
                )
                for key, legacy_key in mapping.items():
                    if expected[legacy_key] is None:
                        self.assertIsNone(actual[key], key)
                    else:
                        self.assertAlmostEqual(actual[key], expected[legacy_key], places=12, msg=key)


class GenerationAndBenchmarkTests(unittest.TestCase):
    def test_yearly_and_rolling_generation_is_ordered_and_no_lookahead(self) -> None:
        payload = curve([0.001] * 800, start="2019-08-01")
        yearly = generate_yearly_metrics(payload)
        self.assertFalse(yearly["rows"][0]["complete_year"])
        self.assertFalse(yearly["rows"][-1]["complete_year"])
        self.assertTrue(all(row["complete_year"] for row in yearly["rows"][1:-1]))
        rolling = generate_rolling_metrics(payload)
        self.assertEqual(rolling["configured_windows"], [63, 252, 756])
        first_63 = next(row for row in rolling["rows"] if row["window_sessions"] == 63)
        self.assertEqual(first_63["economic_date"], payload["rows"][62]["economic_date"])
        self.assertEqual(first_63["observation_count"], 63)

    def test_benchmark_alignment_uses_exact_common_dates_and_fails_closed(self) -> None:
        strategy = curve([0.0, 0.01, 0.01], dates=["2020-01-02", "2020-01-03", "2020-01-06"])
        benchmark = curve([0.0, 0.02, 0.02], dates=["2020-01-03", "2020-01-06", "2020-01-07"])
        left, right, alignment = exact_common_date_curves(strategy, benchmark)
        self.assertEqual([row["economic_date"] for row in left["rows"]], ["2020-01-03", "2020-01-06"])
        self.assertEqual([row["economic_date"] for row in right["rows"]], ["2020-01-03", "2020-01-06"])
        self.assertEqual(alignment["dropped_strategy_dates"], ["2020-01-02"])
        self.assertEqual(alignment["dropped_benchmark_dates"], ["2020-01-07"])
        left, right, alignment = exact_common_date_curves(
            strategy, benchmark, min_common_observations=3
        )
        self.assertIsNone(left)
        self.assertIsNone(right)
        self.assertEqual(alignment["status"], "insufficient_common_date_coverage")


class BehaviorTests(unittest.TestCase):
    def test_fingerprints_and_separate_similarity_diagnostics_are_deterministic(self) -> None:
        payload = curve([0.0, 0.01, -0.01, 0.02])
        trades = {
            "rows": [
                {"symbol": "AAA", "entry_date": payload["rows"][1]["economic_date"], "exit_date": payload["rows"][3]["economic_date"]}
            ]
        }
        first = generate_behavior_metadata(
            payload,
            daily_curve_hash=content_hash(payload),
            trade_lifecycles=trades,
            trade_lifecycles_hash=content_hash(trades),
        )
        second = generate_behavior_metadata(
            payload,
            daily_curve_hash=content_hash(payload),
            trade_lifecycles=trades,
            trade_lifecycles_hash=content_hash(trades),
        )
        self.assertEqual(first, second)
        diagnostics = behavior_similarity(first, second)
        self.assertEqual(diagnostics["daily_return_correlation"], 1.0)
        self.assertEqual(diagnostics["active_date_jaccard"], 1.0)
        self.assertEqual(diagnostics["entry_date_jaccard"], 1.0)
        self.assertEqual(diagnostics["exit_date_jaccard"], 1.0)
        self.assertEqual(diagnostics["normalized_path_distance"], 0.0)

    def test_all_configured_similarity_conditions_are_required_and_runs_are_preserved(self) -> None:
        payload = curve([0.0, 0.01, -0.01, 0.02])
        metadata = generate_behavior_metadata(
            payload, daily_curve_hash=content_hash(payload)
        )
        configuration = {
            "enabled": True,
            "required_conditions": [
                {"metric_key": "daily_return_correlation", "operator": ">=", "threshold": 0.999},
                {"metric_key": "active_date_jaccard", "operator": ">=", "threshold": 0.99},
                {"metric_key": "normalized_path_distance", "operator": "<=", "threshold": 0.01},
            ],
            "representative_order": [
                {"field": "complexity_score", "direction": "minimize"}
            ],
        }
        clustered, pairwise = deduplicate_behaviors(
            {"simple": metadata, "complex": metadata},
            configuration,
            simplicity_metadata={
                "simple": {"complexity_score": 1},
                "complex": {"complexity_score": 2},
            },
        )
        self.assertEqual(clustered["simple"]["representative_strategy_run_id"], "simple")
        self.assertTrue(clustered["complex"]["duplicated"])
        self.assertTrue(clustered["complex"]["underlying_strategy_run_preserved"])
        self.assertTrue(next(iter(pairwise.values()))["duplicated"])


class StoredIntegrationTests(unittest.TestCase):
    def _save_run(
        self,
        store: LocalResultStore,
        run_spec: StrategyRunSpec,
        daily: dict,
        benchmark: dict,
        *,
        include_robustness: bool = True,
    ) -> StrategyRunManifest:
        records = [
            store.put_artifact(
                "daily_portfolio_curve",
                ArtifactKind.DAILY_PORTFOLIO_CURVE,
                daily,
                row_count=len(daily["rows"]),
            ).record,
            store.put_artifact(
                "benchmark_daily_portfolio_curve",
                ArtifactKind.DAILY_PORTFOLIO_CURVE,
                benchmark,
                row_count=len(benchmark["rows"]),
            ).record,
        ]
        if include_robustness:
            records.append(
                store.put_artifact(
                    "robustness_summary",
                    ArtifactKind.ROBUSTNESS_SUMMARY,
                    robustness(),
                    row_count=1,
                ).record
            )
        manifest = StrategyRunManifest.create(
            run_spec,
            source_code_commit="b" * 40,
            artifacts=records,
            creation_time=CREATED_AT,
            execution_status=ExecutionStatus.SUCCEEDED,
            warnings=("synthetic_only",),
        )
        store.save_strategy_run(manifest)
        return manifest

    def test_profile_change_reuses_metrics_and_preserves_default_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalResultStore(directory, policy())
            benchmark = curve([0.0002] * 800, start="2020-01-02", exposure=1.0)
            first_curve = curve([0.0003] * 800, start="2020-01-02", gross_returns=[0.00031] * 800)
            second_returns = [0.00028] * 800
            second_returns[300] = -0.01
            second_curve = curve(second_returns, start="2020-01-02", gross_returns=[value + 0.00001 for value in second_returns])
            first = self._save_run(store, spec(1.0), first_curve, benchmark)
            second = self._save_run(store, spec(1.1), second_curve, benchmark)
            profiles = load_evaluation_profiles(PROFILE_DIR)
            research = calculate_and_evaluate_saved_runs(
                store,
                (first.strategy_run_id, second.strategy_run_id),
                profiles["research_default"],
                creation_time=CREATED_AT,
            )
            weighted = calculate_and_evaluate_saved_runs(
                store,
                (second.strategy_run_id, first.strategy_run_id),
                profiles["exploratory_weighted_example"],
                creation_time="2026-08-01T00:01:00Z",
            )
            self.assertEqual(research.derived_metric_ids, weighted.derived_metric_ids)
            self.assertEqual(set(research.cache_status.values()), {"calculated"})
            self.assertEqual(set(weighted.cache_status.values()), {"reused"})
            self.assertEqual(len(store.derived_metric_history()), 2)
            self.assertEqual(len(store.evaluation_history()), 2)
            for left, right in zip(
                research.evaluation_run.results, weighted.evaluation_run.results
            ):
                self.assertEqual(left.mandatory_gates_passed, right.mandatory_gates_passed)
                self.assertEqual(left.pareto_member, right.pareto_member)
                self.assertEqual(left.robustness_passed, right.robustness_passed)
                self.assertIsNone(left.weighted_view)
                self.assertIsNotNone(right.weighted_view)
            self.assertEqual(store.orphan_hashes(), ())

    def test_missing_robustness_gate_fails_closed_with_exact_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalResultStore(directory, policy())
            benchmark_returns = [0.0002] * 800
            benchmark_returns[300] = -0.10
            strategy_returns = [0.0003] * 800
            strategy_returns[300] = -0.03
            benchmark = curve(benchmark_returns)
            daily = curve(strategy_returns)
            manifest = self._save_run(
                store, spec(1.0), daily, benchmark, include_robustness=False
            )
            profile = load_evaluation_profiles(PROFILE_DIR)["final_eligibility_default"]
            result = calculate_and_evaluate_saved_runs(
                store,
                (manifest.strategy_run_id,),
                profile,
                creation_time=CREATED_AT,
            )
            candidate = result.evaluation_run.results[0]
            self.assertFalse(candidate.robustness_passed)
            self.assertTrue(
                all(
                    check.reason == "robustness_summary_artifact_missing"
                    for check in candidate.robustness_results
                )
            )
            self.assertIn("robustness_vetoed", candidate.final_labels)

    def test_manifest_row_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalResultStore(directory, policy())
            payload = curve([0.0, 0.01])
            put = store.put_artifact(
                "daily_portfolio_curve",
                ArtifactKind.DAILY_PORTFOLIO_CURVE,
                payload,
                row_count=1,
            )
            manifest = StrategyRunManifest.create(
                spec(1.0),
                source_code_commit="b" * 40,
                artifacts=(put.record,),
                creation_time=CREATED_AT,
            )
            with self.assertRaisesRegex(ValueError, "row_count_mismatch"):
                store.save_strategy_run(manifest)


if __name__ == "__main__":
    unittest.main()
