import unittest
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from src.features import add_signal_surge_v0
from src.trend_v2 import (
    EntryRule,
    FilterRule,
    InitialStopRule,
    PhaseAComponents,
    PortfolioConstructionRule,
    PositionSizingRule,
    SignalRule,
    TrailingExitRule,
    UniverseEligibilityRule,
    LEGACY_SCANNER_TREND_FILTER_CONTROL_V0,
    PRIMARY_PHASE_A_TREND_FILTER,
    add_phase_a_price_trend_features,
    build_score_breakout_rules,
    build_signal_diagnostics,
    classify_score_breakout,
    decompose_legacy_scanner,
    default_phase_a_components,
    evaluate_signal_observations,
    frequency_matched_random_events,
    make_score_breakout_rule,
    signal_event_counts,
    run_phase_a_signal_comparison,
    simulate_signal_lifecycles,
    within_symbol_shifted_events,
)


def feature_fixture(periods: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
    for symbol_index, symbol in enumerate(("AAA", "BBB", "SPY")):
        for index, date in enumerate(dates):
            close = 100.0 + symbol_index * 7.0 + index * (0.15 + symbol_index * 0.02)
            score = 0.05 + index * 0.001 + 0.004 * np.sin(index / 7.0)
            eligible = symbol != "SPY"
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "eligible_universe": eligible,
                    "te63": 0.20,
                    "te126": 0.10,
                    "surge_ratio": 1.30,
                    "ma50": close - 2.0,
                    "ma150": close - 4.0,
                    "r20": -99.0 if index % 2 else 99.0,
                    "r63": 0.10,
                    "r126": 0.20,
                    "er20": -50.0 if index % 3 else 50.0,
                    "er63": 0.50,
                    "atr20_pct": 0.02,
                    "hhv126_ratio": 0.95,
                    "score": score,
                    "low20": close - 5.0,
                    "stop_distance_pct": -0.05,
                }
            )
    return pd.DataFrame(rows)


class PhaseAArchitectureTests(unittest.TestCase):
    def test_component_boundaries_are_first_class(self) -> None:
        components = default_phase_a_components()
        self.assertIsInstance(components, PhaseAComponents)
        self.assertIsInstance(components.universe, UniverseEligibilityRule)
        self.assertIsInstance(components.trend_filter, FilterRule)
        self.assertIsInstance(components.entry, EntryRule)
        self.assertIsInstance(components.initial_stop, InitialStopRule)
        self.assertIsInstance(components.trailing_exit, TrailingExitRule)
        self.assertIsInstance(components.position_sizing, PositionSizingRule)
        self.assertIsInstance(components.portfolio_construction, PortfolioConstructionRule)
        self.assertEqual(components.trend_filter.key, "price_above_rising_ma200_v0")
        self.assertEqual(components.position_sizing.key, "canonical_equal_weight_active_v1")
        self.assertEqual(components.position_sizing.target_weights(["AAA", "BBB"]), {"AAA": 0.5, "BBB": 0.5})

    def test_primary_trend_filter_is_independent_of_score_and_trend_energy(self) -> None:
        frame = feature_fixture(260)
        prepared = add_phase_a_price_trend_features(frame)
        baseline = PRIMARY_PHASE_A_TREND_FILTER.evaluate(frame)
        changed = frame.copy()
        for column in ("score", "te63", "te126", "r20", "r63", "r126", "er20", "er63"):
            changed[column] = np.linspace(-1_000_000.0, 1_000_000.0, len(changed))
        changed_prepared = add_phase_a_price_trend_features(changed)

        pd.testing.assert_series_equal(
            prepared["phase_a_ma200"], changed_prepared["phase_a_ma200"]
        )
        pd.testing.assert_series_equal(
            prepared["phase_a_ma200_slope_20"],
            changed_prepared["phase_a_ma200_slope_20"],
        )
        pd.testing.assert_series_equal(
            baseline,
            PRIMARY_PHASE_A_TREND_FILTER.evaluate(changed),
        )

    def test_legacy_scanner_decomposition_reconstructs_v1_exactly(self) -> None:
        frame = feature_fixture(5)
        legacy = add_signal_surge_v0(frame)["signal_surge_v0"]
        decomposed = decompose_legacy_scanner(frame)

        pd.testing.assert_series_equal(
            decomposed["v2_legacy_signal_reconstructed"],
            legacy,
            check_names=False,
        )
        self.assertTrue(
            {
                "v2_trend_filter",
                "v2_trigger_signal",
                "v2_continuous_ranking",
                "v2_risk_er63",
                "v2_risk_atr20_pct",
                "v2_risk_stop_distance_pct",
            }.issubset(decomposed.columns)
        )
        self.assertEqual(
            LEGACY_SCANNER_TREND_FILTER_CONTROL_V0.key,
            "legacy_scanner_trend_filter_control_v0",
        )
        pd.testing.assert_series_equal(
            LEGACY_SCANNER_TREND_FILTER_CONTROL_V0.evaluate(frame),
            decomposed["v2_trend_filter"],
            check_names=False,
        )

    def test_score_breakout_has_one_numeric_parameter_and_ignores_r20_er20(self) -> None:
        frame = feature_fixture(25).loc[lambda value: value["symbol"] == "AAA"].copy()
        rule = make_score_breakout_rule(10)
        changed = frame.copy()
        changed["r20"] = -1_000_000.0
        changed["er20"] = -1_000_000.0

        self.assertIsInstance(rule, SignalRule)
        self.assertEqual(rule.params, {"family": "score_breakout", "score_lookback": 10})
        pd.testing.assert_series_equal(rule.evaluate(frame), rule.evaluate(changed))
        self.assertEqual(
            [rule.params["score_lookback"] for rule in build_score_breakout_rules()],
            [10, 20, 40],
        )

    def test_observations_keep_legacy_as_baseline_role(self) -> None:
        frame = feature_fixture(20)
        legacy = SignalRule(
            "fixture_legacy",
            "Fixture legacy",
            "Fixture",
            {"family": "legacy"},
            lambda value: value["signal_surge_v0"],
            role="legacy_baseline",
        )
        decomposed = decompose_legacy_scanner(frame)
        observations = evaluate_signal_observations(
            decomposed,
            [legacy],
            default_phase_a_components(),
        )
        pd.testing.assert_series_equal(
            observations["fixture_legacy"],
            decomposed.loc[observations.index, "signal_surge_v0"],
            check_names=False,
        )
        self.assertIn("legacy_scanner_trend_filter_control_v0", observations)

    def test_unsupported_position_sizing_fails_closed(self) -> None:
        components = default_phase_a_components()
        unsupported = replace(
            components,
            position_sizing=PositionSizingRule(
                "unsupported_volatility_sizing",
                "Unsupported fixture",
                lambda symbols: {str(symbol): 1.0 / len(symbols) for symbol in symbols},
            ),
        )
        frame = feature_fixture(20).loc[lambda value: value["symbol"] == "AAA"].copy()
        events = pd.Series(False, index=frame.index)
        events.iloc[5] = True
        with self.assertRaisesRegex(ValueError, "fixed canonical equal weight"):
            simulate_signal_lifecycles(
                frame,
                events,
                strategy_key="unsupported",
                components=unsupported,
            )


class PhaseASignalControlTests(unittest.TestCase):
    def test_random_events_are_deterministic_and_frequency_matched_by_symbol(self) -> None:
        frame = feature_fixture(15)
        eligible = frame["eligible_universe"]
        target = pd.Series(False, index=frame.index)
        for _, group in frame.loc[eligible].groupby("symbol"):
            target.loc[group.index[:4]] = True

        first = frequency_matched_random_events(frame, target, eligible, seed=1729)
        second = frequency_matched_random_events(frame, target, eligible, seed=1729)
        pd.testing.assert_series_equal(first, second)
        for symbol, group in frame.groupby("symbol"):
            target_counts = signal_event_counts(
                group,
                target.loc[group.index],
                default_phase_a_components().entry,
            )
            random_counts = signal_event_counts(
                group,
                first.loc[group.index],
                default_phase_a_components().entry,
            )
            self.assertEqual(
                random_counts["executable_trigger_count"],
                target_counts["executable_trigger_count"],
            )
            self.assertEqual(
                random_counts["raw_boolean_signal_count"],
                target_counts["executable_trigger_count"],
            )

    def test_random_matching_prevents_adjacent_event_collapse(self) -> None:
        frame = feature_fixture(12).loc[lambda value: value["symbol"] == "AAA"].copy()
        target = pd.Series(False, index=frame.index)
        target.iloc[[0, 1, 3, 4]] = True
        random_events = frequency_matched_random_events(
            frame,
            target,
            pd.Series(True, index=frame.index),
            seed=3253,
        )
        target_counts = signal_event_counts(
            frame, target, default_phase_a_components().entry
        )
        random_counts = signal_event_counts(
            frame, random_events, default_phase_a_components().entry
        )
        self.assertEqual(target_counts["raw_boolean_signal_count"], 4)
        self.assertEqual(target_counts["executable_trigger_count"], 2)
        self.assertEqual(random_counts["raw_boolean_signal_count"], 2)
        self.assertEqual(random_counts["executable_trigger_count"], 2)
        chosen = np.flatnonzero(random_events.to_numpy(dtype=bool))
        self.assertTrue(all(right - left > 1 for left, right in zip(chosen, chosen[1:])))

    def test_random_matching_fails_when_exact_trigger_count_is_impossible(self) -> None:
        frame = feature_fixture(6).loc[lambda value: value["symbol"] == "AAA"].copy()
        target = pd.Series(False, index=frame.index)
        target.iloc[[0, 2, 4]] = True
        eligible = pd.Series(False, index=frame.index)
        eligible.iloc[:3] = True
        with self.assertRaisesRegex(ValueError, "requested 3, maximum isolated eligible dates 2"):
            frequency_matched_random_events(
                frame,
                target,
                eligible,
                seed=5003,
            )

    def test_shift_placebo_never_crosses_symbol_boundaries(self) -> None:
        frame = feature_fixture(8)
        events = pd.Series(False, index=frame.index)
        for _, group in frame.groupby("symbol"):
            events.loc[group.index[0]] = True
        shifted = within_symbol_shifted_events(frame, events, shift_bars=3)
        for _, group in frame.groupby("symbol"):
            ordered = group.sort_values("date")
            self.assertTrue(bool(shifted.loc[ordered.index[3]]))
            self.assertEqual(int(shifted.loc[ordered.index].sum()), 1)

    def test_signal_only_diagnostics_use_forward_paths_not_exit_rules(self) -> None:
        frame = feature_fixture(6).loc[lambda value: value["symbol"] == "AAA"].copy()
        events = pd.Series(False, index=frame.index)
        events.iloc[0] = True
        diagnostics = build_signal_diagnostics(
            frame,
            {"fixture": events},
            horizons=(1, 3),
            excursion_horizon=3,
        )
        row = diagnostics.iloc[0]
        base = frame.iloc[0]["close"]
        self.assertAlmostEqual(row["forward_return_1d"], frame.iloc[1]["close"] / base - 1.0)
        self.assertAlmostEqual(row["forward_return_3d"], frame.iloc[3]["close"] / base - 1.0)
        self.assertAlmostEqual(row["mfe_3d"], frame.iloc[1:4]["high"].max() / base - 1.0)
        self.assertAlmostEqual(row["mae_3d"], frame.iloc[1:4]["low"].min() / base - 1.0)
        self.assertNotIn("exit_date", diagnostics.columns)

    def test_common_lifecycle_has_no_time_exit_or_profit_target(self) -> None:
        frame = feature_fixture(80).loc[lambda value: value["symbol"] == "AAA"].copy()
        events = pd.Series(False, index=frame.index)
        events.iloc[10] = True
        lifecycles = simulate_signal_lifecycles(
            frame,
            events,
            strategy_key="fixture",
            components=default_phase_a_components(),
        )
        self.assertEqual(len(lifecycles), 1)
        self.assertEqual(lifecycles.iloc[0]["exit_reason"], "open_at_end")
        self.assertNotIn("max_holding_days", set(lifecycles["exit_reason"]))
        self.assertFalse(any("target" in column for column in lifecycles.columns))


class PhaseAEndToEndTests(unittest.TestCase):
    @staticmethod
    def _metric(
        key: str,
        variant: str,
        *,
        parent: str | None = None,
        cagr: float,
        mdd: float,
        cdar: float,
        calmar: float,
        recovery: int,
    ) -> dict:
        return {
            "strategy_key": key,
            "variant": variant,
            "parent_signal_key": parent,
            "score_lookback": 20 if variant == "score_breakout" else np.nan,
            "common_years": 10.0,
            "strategy_cagr": cagr,
            "strategy_calmar": calmar,
            "strategy_maximum_drawdown": mdd,
            "strategy_cdar95": cdar,
            "strategy_recovery_duration_days": recovery,
            "strategy_cagr_spy_ratio": 0.90,
            "maximum_drawdown_spy_ratio": 0.70,
            "cdar95_spy_ratio": 0.75,
            "calmar_spy_ratio": 1.10,
            "raw_boolean_signal_count": 200,
            "executable_trigger_count": 120,
            "completed_lifecycle_count": 80,
        }

    @staticmethod
    def _robustness_evidence(
        *,
        improvement_ratio: float = 0.75,
        interval: tuple[float, float] = (-0.001, 0.02),
        adjusted_p_value: float = 0.01,
    ) -> dict:
        return {
            "walk_forward_fold_count": 6,
            "walk_forward_improvement_ratio": improvement_ratio,
            "leave_one_year_out_result_count": 8,
            "leave_one_year_out_stability_ratio": 1.0,
            "date_block_bootstrap_paired_effect_confidence_interval": interval,
            "multiple_testing_adjusted_p_value": adjusted_p_value,
            "asset_group_concentration_diagnostics": {
                "dominant_group": "broad_equity",
                "dominant_effect_share": 0.40,
            },
            "event_count_comparability": True,
            "executable_trigger_count_comparability": True,
        }

    def _classification_fixture(self) -> tuple[dict, list[dict], pd.DataFrame]:
        score = self._metric(
            "score_breakout_l20", "score_breakout", cagr=0.12, mdd=-0.12,
            cdar=-0.08, calmar=1.0, recovery=80,
        )
        weaker = [
            self._metric("trend", "trend_filter_only", cagr=0.10, mdd=-0.16, cdar=-0.11, calmar=0.7, recovery=120),
            self._metric("price", "prior_price_high", cagr=0.09, mdd=-0.15, cdar=-0.10, calmar=0.6, recovery=110),
            self._metric("random", "frequency_matched_random", parent="score_breakout_l20", cagr=0.07, mdd=-0.18, cdar=-0.13, calmar=0.4, recovery=140),
            self._metric("shift", "shifted_placebo", parent="score_breakout_l20", cagr=0.08, mdd=-0.17, cdar=-0.12, calmar=0.5, recovery=130),
        ]
        summary = pd.DataFrame([{"signal_key": "score_breakout_l20", "signal_count": 200}])
        return score, weaker, summary

    def test_classifier_missing_robustness_evidence_is_inconclusive(self) -> None:
        score, weaker, summary = self._classification_fixture()
        result = classify_score_breakout(pd.DataFrame([score, *weaker]), summary)
        self.assertEqual(result["classification"], "Inconclusive")
        self.assertIn("walk_forward_fold_count", result["missing_robustness_evidence"])
        self.assertIn("Missing or invalid robustness evidence", result["reason"])

    def test_classifier_complete_but_nonpassing_evidence_is_inconclusive(self) -> None:
        score, weaker, summary = self._classification_fixture()
        evidence = self._robustness_evidence(adjusted_p_value=0.20)
        result = classify_score_breakout(
            pd.DataFrame([score, *weaker]),
            summary,
            robustness_evidence=evidence,
        )
        self.assertEqual(result["classification"], "Inconclusive")
        self.assertIn("Complete evidence was supplied", result["reason"])

    def test_classifier_passing_robustness_evidence_retains(self) -> None:
        score, weaker, summary = self._classification_fixture()
        retained = classify_score_breakout(
            pd.DataFrame([score, *weaker]),
            summary,
            robustness_evidence=self._robustness_evidence(),
        )
        self.assertEqual(retained["classification"], "Retain")

    def test_classifier_adverse_robustness_evidence_rejects(self) -> None:
        score, weaker, summary = self._classification_fixture()
        dominated_score = {**score, "strategy_cagr": 0.05, "strategy_calmar": 0.3,
                           "strategy_maximum_drawdown": -0.20, "strategy_cdar95": -0.15,
                           "strategy_recovery_duration_days": 180}
        rejected = classify_score_breakout(
            pd.DataFrame([dominated_score, *weaker]),
            summary,
            robustness_evidence=self._robustness_evidence(
                improvement_ratio=0.25,
                interval=(-0.02, 0.001),
            ),
        )
        self.assertEqual(rejected["classification"], "Reject")

    def test_comparison_uses_shared_rules_and_returns_bounded_classification(self) -> None:
        frame = feature_fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = run_phase_a_signal_comparison(
                frame,
                score_lookbacks=(10,),
                random_seeds=(1729, 3253),
            )

        self.assertEqual(result.classification["classification"], "Inconclusive")
        self.assertIn("Missing or invalid robustness evidence", result.classification["reason"])
        self.assertIn("walk_forward_fold_count", result.classification["missing_robustness_evidence"])
        self.assertEqual(result.methodology["trend_filter"], "price_above_rising_ma200_v0")
        self.assertEqual(
            result.methodology["trend_filter_sensitivity_control"],
            "legacy_scanner_trend_filter_control_v0",
        )
        self.assertEqual(result.methodology["score_breakout_numeric_parameter"], "score_lookback")
        self.assertEqual(result.methodology["r20_er20_role"], "diagnostics_and_ranking_only")
        self.assertIsNone(result.methodology["fixed_holding_period_exit"])
        self.assertIsNone(result.methodology["fixed_profit_target"])
        self.assertEqual(
            set(result.portfolio_metrics["variant"]),
            {
                "trend_filter_only",
                "prior_price_high",
                "score_breakout",
                "frequency_matched_random",
                "shifted_placebo",
                "legacy_baseline",
            },
        )
        for column in (
            "strategy_cagr_spy_ratio",
            "maximum_drawdown_spy_ratio",
            "cdar95_spy_ratio",
            "calmar_spy_ratio",
            "strategy_recovery_duration_days",
            "annual_turnover",
            "total_transaction_cost",
            "raw_boolean_signal_count",
            "executable_trigger_count",
            "completed_lifecycle_count",
        ):
            self.assertIn(column, result.portfolio_metrics)
        shifted = result.portfolio_metrics.loc[
            result.portfolio_metrics["variant"] == "shifted_placebo"
        ].iloc[0]
        self.assertGreaterEqual(int(shifted["shift_edge_loss_raw_count"]), 0)
        self.assertIn(
            bool(shifted["executable_trigger_count_matches_parent"]),
            (True, False),
        )
        if not result.lifecycles.empty:
            for column in (
                "entry_rule",
                "initial_stop_rule",
                "trailing_exit_rule",
                "position_sizing_rule",
                "portfolio_construction_rule",
            ):
                self.assertEqual(result.lifecycles[column].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
