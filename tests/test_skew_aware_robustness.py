import math
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from scripts.analyze_skew_aware_robustness import (
    INITIAL_EQUITY,
    _bootstrap_chunk_metrics,
    aggregate_crash_rows,
    behavior_groups,
    conditional_drawdown_at_risk,
    concentration_and_stress_rows,
    construct_spy_crash_episodes,
    cost_sensitivity_rows,
    crash_strategy_episode_row,
    drawdown_episodes,
    drawdown_from_returns,
    equity_from_returns,
    expected_maximum_sharpe,
    expected_shortfall,
    first_threshold_delay,
    load_inputs,
    longest_underwater_observations,
    normalized_return_hash,
    pareto_frontier,
    psr_probability,
    stationary_bootstrap_indices,
    stress_path_metrics,
    validate_inputs,
)


class StationaryBootstrapTest(unittest.TestCase):
    def test_reproducible_indices(self) -> None:
        left = stationary_bootstrap_indices(
            12, 5, 4, np.random.default_rng(20260728)
        )
        right = stationary_bootstrap_indices(
            12, 5, 4, np.random.default_rng(20260728)
        )
        np.testing.assert_array_equal(left, right)

    def test_block_continuation_or_restart(self) -> None:
        indices = stationary_bootstrap_indices(
            11, 20, 5, np.random.default_rng(7)
        )
        self.assertTrue(((indices >= 0) & (indices < 11)).all())
        continuations = (indices[:, 1:] == (indices[:, :-1] + 1) % 11)
        self.assertGreater(int(continuations.sum()), 0)
        self.assertLess(int(continuations.sum()), continuations.size)

    def test_strategy_and_spy_share_indices(self) -> None:
        strategy = np.arange(8, dtype=float) / 100.0
        spy = strategy + 0.5
        indices = stationary_bootstrap_indices(
            8, 4, 3, np.random.default_rng(11)
        )
        np.testing.assert_allclose(spy[indices] - strategy[indices], 0.5)

    def test_bootstrap_equity_reconstruction(self) -> None:
        strategy = np.array([0.01, -0.02, 0.03, 0.0])
        indices = np.array([[0, 1, 2, 3], [2, 2, 1, 0]], dtype=np.int32)
        result = _bootstrap_chunk_metrics(strategy, strategy, indices, 1.0)
        expected = INITIAL_EQUITY * np.prod(1.0 + strategy[indices], axis=1)
        np.testing.assert_allclose(result["ending_equity"], expected)
        np.testing.assert_allclose(result["cagr"], expected / INITIAL_EQUITY - 1.0)


class PathMetricTest(unittest.TestCase):
    def test_equity_and_mdd(self) -> None:
        returns = np.array([0.10, -0.20, 0.05])
        equity, drawdown = drawdown_from_returns(returns)
        np.testing.assert_allclose(equity, [1100.0, 880.0, 924.0])
        self.assertAlmostEqual(float(drawdown.min()), -0.20)
        self.assertAlmostEqual(stress_path_metrics(returns)["maximum_drawdown"], -0.20)

    def test_fixed_tail_cdar(self) -> None:
        values = np.array([0.0] * 18 + [-0.10, -0.20])
        self.assertAlmostEqual(conditional_drawdown_at_risk(values, 0.90), -0.15)
        self.assertAlmostEqual(conditional_drawdown_at_risk(values, 0.95), -0.20)

    def test_expected_shortfall_inclusive_ties(self) -> None:
        values = np.array([-0.10, -0.10] + [0.01] * 18)
        self.assertAlmostEqual(expected_shortfall(values, 0.95), -0.10)

    def test_time_under_water(self) -> None:
        self.assertEqual(
            longest_underwater_observations([0.0, -0.1, -0.2, 0.0, -0.1]),
            2,
        )

    def test_drawdown_episode_separation(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5, tz="UTC")
        returns = np.array([0.10, -0.10, 0.12, -0.05, 0.06])
        episodes = drawdown_episodes(dates, returns)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["peak_date"], "2020-01-01")
        self.assertEqual(episodes[0]["recovery_date"], "2020-01-03")
        self.assertEqual(episodes[1]["recovery_date"], "2020-01-05")


class CrashEpisodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2020-01-01", periods=9, tz="UTC")
        self.equity = np.array([100.0, 105.0, 100.0, 90.0, 85.0, 106.0, 100.0, 80.0, 110.0])

    def test_nonoverlap_and_nested_thresholds(self) -> None:
        episodes = construct_spy_crash_episodes(
            self.dates, self.equity, (0.10, 0.20)
        )
        ten = [row for row in episodes if row["threshold_percent"] == 10]
        twenty = [row for row in episodes if row["threshold_percent"] == 20]
        self.assertEqual(len(ten), 2)
        self.assertEqual(len(twenty), 1)
        self.assertLess(ten[0]["recovery_index"], ten[1]["onset_index"])
        self.assertIn(ten[1]["episode_id"], twenty[0]["nested_within_episode_ids"])

    def test_crash_loss_capture_formula(self) -> None:
        episodes = construct_spy_crash_episodes(
            self.dates, self.equity, (0.10,)
        )
        spy_returns = np.concatenate(
            ([self.equity[0] / 100.0 - 1.0], self.equity[1:] / self.equity[:-1] - 1.0)
        )
        strategy = spy_returns * 0.5
        row = crash_strategy_episode_row(
            "s", self.dates, strategy, spy_returns, episodes[0]
        )
        expected = max(-row["strategy_onset_to_spy_trough_return"], 0.0) / max(
            -row["spy_onset_to_trough_return"], 1e-12
        )
        self.assertAlmostEqual(row["crash_loss_capture"], expected)

    def test_exposure_decay(self) -> None:
        exposure = np.array([1.0, 0.8, 0.7, 0.4, 0.2])
        self.assertEqual(first_threshold_delay(exposure, 1, 4, 0.75), (1.0, "reached_before_episode_end"))
        self.assertEqual(first_threshold_delay(exposure, 2, 4, 0.75), (0.0, "already_below_at_onset"))
        delay, reason = first_threshold_delay(np.ones(5), 1, 4, 0.5)
        self.assertTrue(math.isnan(delay))
        self.assertEqual(reason, "not_reached_before_recovery_or_data_end")


class ConcentrationAndCostTest(unittest.TestCase):
    def test_best_period_and_removal_stress(self) -> None:
        dates = pd.date_range("2018-01-01", periods=800, freq="B", tz="UTC")
        returns = np.full(len(dates), 0.0001)
        returns[100] = 0.10
        episodes = drawdown_episodes(dates, returns)
        concentration, rows = concentration_and_stress_rows(
            "s", dates, returns, episodes
        )
        self.assertGreater(concentration["best_day_contribution_ratio"], 0)
        removed = next(row for row in rows if row["scenario_type"] == "remove_best_day")
        baseline = next(row for row in rows if row["scenario_type"] == "baseline")
        self.assertLess(removed["ending_equity"], baseline["ending_equity"])
        self.assertTrue(
            any(row["scenario_type"] == "leave_one_full_calendar_year_out" for row in rows)
        )

    def test_cost_sensitivity_reconstruction(self) -> None:
        returns = np.array([0.01, 0.0])
        equity = equity_from_returns(returns)
        cost = np.array([1.0, 0.0])
        summary, rows = cost_sensitivity_rows("s", returns, equity, cost)
        by_factor = {row["cost_multiplier"]: row for row in rows}
        self.assertAlmostEqual(by_factor[1.0]["ending_equity"], equity[-1])
        self.assertLess(by_factor[2.0]["ending_equity"], by_factor[1.0]["ending_equity"])
        self.assertGreater(summary["cost_2x_cagr_drop"], 0)


class PsrDsrAndBehaviorTest(unittest.TestCase):
    def test_psr_matches_independent_formula(self) -> None:
        sharpe = 0.8
        benchmark = 0.5
        n = 1000
        skew = 0.4
        excess = 1.2
        daily = sharpe / math.sqrt(252)
        daily_benchmark = benchmark / math.sqrt(252)
        denominator = math.sqrt(
            1 - skew * daily + (((excess + 3) - 1) / 4) * daily**2
        )
        expected = NormalDist().cdf(
            (daily - daily_benchmark) * math.sqrt(n - 1) / denominator
        )
        self.assertAlmostEqual(
            psr_probability(sharpe, benchmark, n, skew, excess), expected
        )

    def test_dsr_raw_trial_count_more_conservative(self) -> None:
        self.assertGreater(
            expected_maximum_sharpe(0.2, 540),
            expected_maximum_sharpe(0.2, 100),
        )
        raw = psr_probability(1.0, expected_maximum_sharpe(0.2, 540), 2000, 0, 0)
        dedup = psr_probability(1.0, expected_maximum_sharpe(0.2, 100), 2000, 0, 0)
        self.assertLess(raw, dedup)

    def test_behavioral_deduplication_and_row_order(self) -> None:
        frame = pd.DataFrame(
            {
                "b": [0.01, 0.02, -0.01],
                "a": [0.01, 0.02, -0.01],
                "c": [0.01, 0.02, -0.0100001],
            }
        )
        left, groups_left = behavior_groups(frame)
        right, groups_right = behavior_groups(frame[["c", "a", "b"]])
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(groups_left, groups_right)
        self.assertEqual(left.set_index("strategy_key").loc["a", "behavior_group_size"], 2)
        self.assertEqual(
            normalized_return_hash(frame["a"]),
            normalized_return_hash(frame["b"]),
        )


class ParetoAndOrderingTest(unittest.TestCase):
    def test_strict_and_tolerance_pareto(self) -> None:
        frame = pd.DataFrame(
            {
                "strategy_key": ["a", "b", "c"],
                "growth": [1.0, 0.99995, 0.8],
                "risk": [0.2, 0.20005, 0.4],
            }
        )
        dimensions = {"growth": "max", "risk": "min"}
        strict, strict_counts = pareto_frontier(frame, dimensions)
        tolerant, tolerant_counts = pareto_frontier(
            frame, dimensions, {"growth": 0.0001, "risk": 0.0001}
        )
        self.assertEqual(strict, {"a"})
        self.assertEqual(tolerant, {"a", "b"})
        self.assertGreater(strict_counts["c"], 0)
        self.assertGreater(tolerant_counts["c"], 0)

    def test_deterministic_table_sort(self) -> None:
        frame = pd.DataFrame(
            {"strategy_key": ["b", "a"], "rank": [1, 1], "value": [2, 10]}
        )
        ordered = frame.sort_values(["rank", "strategy_key"], kind="mergesort")
        self.assertEqual(ordered["strategy_key"].tolist(), ["a", "b"])


class BoundedInputIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_inputs()
        cls.validation = validate_inputs(cls.inputs)

    def test_t0_excluded_and_common_economic_dates(self) -> None:
        self.assertEqual(self.validation["t0_excluded_rows"], 1)
        self.assertEqual(self.validation["economic_observations"], 2332)
        self.assertEqual(self.validation["material_failures"], [])

    def test_exact_current_population(self) -> None:
        self.assertEqual(self.validation["strategy_count"], 540)
        self.assertEqual(self.validation["qualified_count"], 42)
        self.assertEqual(self.validation["manifest_curve_count"], 42)

    def test_portfolio_statistics_do_not_enter_ranking(self) -> None:
        self.assertFalse(self.validation["portfolio_statistics_in_ranking"])


if __name__ == "__main__":
    unittest.main()
