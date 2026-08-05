import csv
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.analyze_skew_aware_robustness import (
    BEHAVIOR_GROUP_OUTPUT,
    BOOTSTRAP_OUTPUT,
    COMPARISON_SCOPE,
    INITIAL_EQUITY,
    NUMERIC_ATOL,
    NUMERIC_RTOL,
    PARETO_OUTPUT,
    REPORT_OUTPUT,
    ROOT,
    SEED,
    SUMMARY_OUTPUT,
    _safe_csv,
    align_matching_snapshot_ending_equity,
    _bootstrap_chunk_metrics,
    aggregate_crash_rows,
    behavior_groups,
    bootstrap_behavior_groups,
    bootstrap_index_matrix_hash,
    common_bootstrap_indices,
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
    group_leader_tags,
    load_inputs,
    longest_underwater_observations,
    normalized_return_hash,
    pareto_frontier,
    psr_probability,
    stationary_bootstrap_indices,
    stationary_bootstrap_summary,
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


class BehaviorPathInvariantMethodTest(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], np.ndarray]:
        matrix = pd.DataFrame(
            {
                "alpha": [0.01, -0.02, 0.03, 0.00, 0.01, -0.01, 0.02, 0.00],
                "alpha_duplicate": [
                    0.01,
                    -0.02,
                    0.03,
                    0.00,
                    0.01,
                    -0.01,
                    0.02,
                    0.00,
                ],
                "beta": [0.00, 0.01, -0.01, 0.02, 0.00, 0.01, -0.02, 0.01],
            }
        )
        behavior, mapping = behavior_groups(matrix)
        spy = np.array([0.005, -0.01, 0.01, 0.00, 0.004, -0.004, 0.01, 0.00])
        return matrix, behavior, mapping, spy

    def test_01_identical_paths_form_one_behavior_group(self) -> None:
        _, behavior, _, _ = self.fixture()
        indexed = behavior.set_index("strategy_key")
        self.assertEqual(
            indexed.loc["alpha", "behavior_group_id"],
            indexed.loc["alpha_duplicate", "behavior_group_id"],
        )
        self.assertEqual(indexed.loc["alpha", "behavior_group_size"], 2)

    def test_02_identical_paths_are_bootstrapped_once(self) -> None:
        matrix, behavior, mapping, spy = self.fixture()
        indices = common_bootstrap_indices(8, 40, 3)
        _, group_rows, computations = bootstrap_behavior_groups(
            matrix,
            behavior,
            mapping,
            matrix.columns,
            spy,
            1.0,
            indices=indices,
            mean_block_length=3,
            bootstrap_scope=COMPARISON_SCOPE,
            pareto_input=True,
        )
        self.assertEqual(computations, 2)
        self.assertEqual(len(group_rows), 2)

    def test_03_identical_paths_receive_exact_serialized_summaries(self) -> None:
        matrix, behavior, mapping, spy = self.fixture()
        labels, _, _ = bootstrap_behavior_groups(
            matrix,
            behavior,
            mapping,
            matrix.columns,
            spy,
            1.0,
            indices=common_bootstrap_indices(8, 40, 3),
            mean_block_length=3,
            bootstrap_scope=COMPARISON_SCOPE,
            pareto_input=True,
        )
        pair = labels.loc[labels["strategy_key"].str.startswith("alpha")].sort_values(
            "strategy_key"
        )
        pd.testing.assert_series_equal(
            pair.iloc[0].drop(labels="strategy_key"),
            pair.iloc[1].drop(labels="strategy_key"),
            check_names=False,
            check_exact=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapped.csv"
            _safe_csv(labels, path, ["strategy_key"])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            alpha_rows = [row for row in rows if row["strategy_key"].startswith("alpha")]
            for field in alpha_rows[0]:
                if field != "strategy_key":
                    self.assertEqual(alpha_rows[0][field], alpha_rows[1][field])

    def test_04_strategy_key_rename_does_not_change_bootstrap(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, 0.00, 0.01, -0.01])
        spy = returns / 2
        indices = common_bootstrap_indices(len(returns), 50, 3)
        left = stationary_bootstrap_summary(
            "old_name", returns, spy, 1.0, indices=indices, mean_block_length=3
        )
        right = stationary_bootstrap_summary(
            "new_name", returns, spy, 1.0, indices=indices, mean_block_length=3
        )
        left.pop("strategy_key")
        right.pop("strategy_key")
        self.assertEqual(left, right)

    def test_05_strategy_column_shuffle_does_not_change_results(self) -> None:
        matrix, behavior, mapping, spy = self.fixture()
        indices = common_bootstrap_indices(8, 40, 3)
        left, _, _ = bootstrap_behavior_groups(
            matrix,
            behavior,
            mapping,
            matrix.columns,
            spy,
            1.0,
            indices=indices,
            mean_block_length=3,
            bootstrap_scope=COMPARISON_SCOPE,
        )
        shuffled = matrix[["beta", "alpha_duplicate", "alpha"]]
        shuffled_behavior, shuffled_mapping = behavior_groups(shuffled)
        right, _, _ = bootstrap_behavior_groups(
            shuffled,
            shuffled_behavior,
            shuffled_mapping,
            shuffled.columns,
            spy,
            1.0,
            indices=indices,
            mean_block_length=3,
            bootstrap_scope=COMPARISON_SCOPE,
        )
        pd.testing.assert_frame_equal(
            left.sort_values("strategy_key").reset_index(drop=True),
            right.sort_values("strategy_key").reset_index(drop=True),
        )

    def test_06_all_groups_use_same_comparison_index_matrix(self) -> None:
        matrix, behavior, mapping, spy = self.fixture()
        labels, _, _ = bootstrap_behavior_groups(
            matrix,
            behavior,
            mapping,
            matrix.columns,
            spy,
            1.0,
            indices=common_bootstrap_indices(8, 40, 3),
            mean_block_length=3,
            bootstrap_scope=COMPARISON_SCOPE,
        )
        self.assertEqual(labels["index_matrix_hash"].nunique(), 1)

    def test_07_comparison_index_hash_is_reproducible(self) -> None:
        left = common_bootstrap_indices(20, 30, 5, seed=SEED)
        right = common_bootstrap_indices(20, 30, 5, seed=SEED)
        self.assertEqual(
            bootstrap_index_matrix_hash(left),
            bootstrap_index_matrix_hash(right),
        )

    def test_08_strategy_and_spy_use_the_exact_same_indices(self) -> None:
        strategy = np.arange(10, dtype=float) / 100
        spy = strategy + 1.0
        indices = common_bootstrap_indices(10, 25, 4)
        np.testing.assert_array_equal(strategy[indices] + 1.0, spy[indices])

    def test_09_primary_prefix_matches_common_comparison_indices(self) -> None:
        comparison = common_bootstrap_indices(12, 50, 4, seed=SEED)
        extra = common_bootstrap_indices(12, 50, 4, seed=SEED + 1)
        detailed = np.vstack([comparison, extra])
        np.testing.assert_array_equal(detailed[: len(comparison)], comparison)

    def test_10_primary_extra_paths_do_not_enter_pareto(self) -> None:
        comparison = pd.DataFrame(
            {
                "strategy_key": ["primary", "other"],
                "growth": [0.9, 0.8],
                "risk": [0.2, 0.3],
                "pareto_input": [True, True],
            }
        )
        detailed = pd.DataFrame(
            {
                "strategy_key": ["primary"],
                "growth": [0.1],
                "risk": [0.9],
                "pareto_input": [False],
            }
        )
        before, _ = pareto_frontier(
            comparison, {"growth": "max", "risk": "min"}
        )
        combined = pd.concat([comparison, detailed], ignore_index=True)
        after, _ = pareto_frontier(
            combined.loc[combined["pareto_input"]],
            {"growth": "max", "risk": "min"},
        )
        self.assertEqual(before, after)

    def test_11_identical_labels_share_strict_pareto_flag(self) -> None:
        groups = pd.DataFrame(
            {
                "behavior_group_id": ["G1", "G2"],
                "growth": [1.0, 0.8],
                "risk": [0.2, 0.4],
            }
        )
        frontier, _ = pareto_frontier(
            groups,
            {"growth": "max", "risk": "min"},
            key_column="behavior_group_id",
        )
        mapped = {"a": "G1" in frontier, "a_duplicate": "G1" in frontier}
        self.assertEqual(mapped["a"], mapped["a_duplicate"])

    def test_12_identical_labels_share_tolerance_pareto_flag(self) -> None:
        groups = pd.DataFrame(
            {
                "behavior_group_id": ["G1", "G2"],
                "growth": [1.0, 0.99995],
                "risk": [0.2, 0.20005],
            }
        )
        frontier, _ = pareto_frontier(
            groups,
            {"growth": "max", "risk": "min"},
            {"growth": 0.0001, "risk": 0.0001},
            key_column="behavior_group_id",
        )
        mapped = {"a": "G1" in frontier, "a_duplicate": "G1" in frontier}
        self.assertEqual(mapped["a"], mapped["a_duplicate"])

    def test_13_labels_in_same_group_cannot_dominate_each_other(self) -> None:
        groups = pd.DataFrame(
            {
                "behavior_group_id": ["G1", "G2"],
                "growth": [1.0, 0.8],
                "risk": [0.2, 0.4],
            }
        )
        _, counts = pareto_frontier(
            groups,
            {"growth": "max", "risk": "min"},
            key_column="behavior_group_id",
        )
        self.assertEqual(counts["G1"], 0)
        self.assertEqual(counts["G2"], 1)

    def test_14_group_frontier_is_invariant_to_duplicate_label_count(self) -> None:
        base = pd.DataFrame(
            {"behavior_group_id": ["G1", "G2"], "growth": [1.0, 0.8]}
        )
        duplicated_labels = pd.concat(
            [base, base.loc[base["behavior_group_id"].eq("G1")]],
            ignore_index=True,
        )
        left, _ = pareto_frontier(
            base, {"growth": "max"}, key_column="behavior_group_id"
        )
        right, _ = pareto_frontier(
            duplicated_labels.drop_duplicates("behavior_group_id"),
            {"growth": "max"},
            key_column="behavior_group_id",
        )
        self.assertEqual(left, right)

    def test_15_adding_duplicate_label_does_not_change_frontier(self) -> None:
        matrix, behavior, _, _ = self.fixture()
        base = pd.DataFrame(
            {
                "strategy_key": ["alpha", "beta"],
                "growth": [1.0, 0.8],
            }
        ).merge(behavior[["strategy_key", "behavior_group_id"]], on="strategy_key")
        with_duplicate = pd.concat(
            [
                base,
                pd.DataFrame(
                    {
                        "strategy_key": ["alpha_duplicate"],
                        "growth": [1.0],
                        "behavior_group_id": [
                            behavior.set_index("strategy_key").loc[
                                "alpha_duplicate", "behavior_group_id"
                            ]
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
        left, _ = pareto_frontier(
            base.drop_duplicates("behavior_group_id"),
            {"growth": "max"},
            key_column="behavior_group_id",
        )
        right, _ = pareto_frontier(
            with_duplicate.drop_duplicates("behavior_group_id"),
            {"growth": "max"},
            key_column="behavior_group_id",
        )
        self.assertEqual(left, right)

    def test_16_leader_tags_are_behavior_group_level(self) -> None:
        groups = pd.DataFrame(
            {
                "behavior_group_id": ["G1", "G2"],
                "growth": [1.0, 0.8],
            }
        )
        tags = group_leader_tags(groups, {"highest_growth": ("growth", "max")})
        self.assertEqual(tags["G1"], ["highest_growth"])
        self.assertNotIn("G2", tags)

    def test_17_fixed_seed_synthetic_analysis_is_reproducible(self) -> None:
        matrix, behavior, mapping, spy = self.fixture()

        def run() -> tuple[pd.DataFrame, pd.DataFrame]:
            return bootstrap_behavior_groups(
                matrix,
                behavior,
                mapping,
                matrix.columns,
                spy,
                1.0,
                indices=common_bootstrap_indices(8, 40, 3, seed=SEED),
                mean_block_length=3,
                bootstrap_scope=COMPARISON_SCOPE,
            )[:2]

        left_labels, left_groups = run()
        right_labels, right_groups = run()
        pd.testing.assert_frame_equal(left_labels, right_labels)
        pd.testing.assert_frame_equal(left_groups, right_groups)

    @staticmethod
    def ending_equity_snapshot_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
        matrix = pd.DataFrame(
            {
                "alpha": [0.10, -0.05, 0.02],
                "beta": [-0.03, 0.04, 0.01],
            }
        )
        ending = INITIAL_EQUITY * (1.0 + matrix).prod(axis=0)
        portfolio = pd.DataFrame(
            {
                "strategy_key": ["beta", "alpha"],
                "ending_equity": [ending["beta"], ending["alpha"]],
            }
        )
        return matrix, portfolio

    def test_18_matching_snapshot_ending_equity_is_preserved(self) -> None:
        matrix, portfolio = self.ending_equity_snapshot_fixture()
        with patch.object(
            pd,
            "read_csv",
            side_effect=AssertionError("matching-snapshot parity must not read live files"),
        ):
            aligned = align_matching_snapshot_ending_equity(matrix, portfolio)

        self.assertEqual(aligned["strategy_key"].tolist(), matrix.columns.tolist())
        np.testing.assert_allclose(
            aligned["research_ending_equity"],
            aligned["canonical_ending_equity"],
            atol=NUMERIC_ATOL,
            rtol=NUMERIC_RTOL,
        )

    def test_18a_matching_snapshot_rejects_missing_or_duplicate_keys(self) -> None:
        matrix, portfolio = self.ending_equity_snapshot_fixture()
        duplicate_matrix = pd.DataFrame(
            [[0.01, 0.02], [0.03, 0.04]],
            columns=["alpha", "alpha"],
        )
        duplicate_portfolio = pd.concat(
            [portfolio, portfolio.loc[portfolio["strategy_key"].eq("alpha")]],
            ignore_index=True,
        )
        cases = [
            (
                "missing portfolio key",
                matrix,
                portfolio.loc[portfolio["strategy_key"].eq("alpha")],
                "missing from portfolio summary: beta",
            ),
            (
                "extra portfolio key",
                matrix,
                pd.concat(
                    [
                        portfolio,
                        pd.DataFrame(
                            {"strategy_key": ["gamma"], "ending_equity": [1000.0]}
                        ),
                    ],
                    ignore_index=True,
                ),
                "missing from return matrix: gamma",
            ),
            (
                "duplicate matrix key",
                duplicate_matrix,
                portfolio.loc[portfolio["strategy_key"].eq("alpha")],
                "duplicate strategy keys in return matrix: alpha",
            ),
            (
                "duplicate portfolio key",
                matrix,
                duplicate_portfolio,
                "duplicate strategy keys in portfolio summary: alpha",
            ),
        ]
        for label, candidate_matrix, candidate_portfolio, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    align_matching_snapshot_ending_equity(
                        candidate_matrix, candidate_portfolio
                    )

    def test_19_generated_csv_ordering_is_deterministic(self) -> None:
        bootstrap = pd.read_csv(BOOTSTRAP_OUTPUT)
        expected_bootstrap = bootstrap.sort_values(
            ["bootstrap_scope", "mean_block_length", "strategy_key"],
            kind="mergesort",
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(bootstrap, expected_bootstrap)
        groups = pd.read_csv(BEHAVIOR_GROUP_OUTPUT)
        self.assertEqual(
            groups["behavior_group_id"].tolist(),
            sorted(groups["behavior_group_id"]),
        )

    def test_20_no_production_output_gate_ui_or_data_workflow_is_modified(self) -> None:
        completed = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "HEAD",
                "--",
                "docs/data",
                "config",
                ":(exclude)config/oos_evaluation_manifest.json",
                ":(exclude)config/oos_evaluation_protocol_v1.json",
                ":(exclude)config/trend_v2/evaluation_profiles/exploratory_weighted_example.json",
                ":(exclude)config/trend_v2/evaluation_profiles/final_eligibility_default.json",
                ":(exclude)config/trend_v2/evaluation_profiles/research_default.json",
                ":(exclude)config/trend_v2/terminology_ko.json",
                "web",
                ".github/workflows/backtest-only.yml",
                ".github/workflows/daily_scan.yml",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "")
        for output in [
            REPORT_OUTPUT,
            SUMMARY_OUTPUT,
            BOOTSTRAP_OUTPUT,
            PARETO_OUTPUT,
            BEHAVIOR_GROUP_OUTPUT,
        ]:
            self.assertTrue(output.is_relative_to(ROOT / "docs" / "tasks"))


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
        self.assertEqual(self.validation["economic_observations"], 2331)
        self.assertEqual(self.validation["material_failures"], [])

    def test_exact_current_population(self) -> None:
        self.assertEqual(self.validation["strategy_count"], 540)
        self.assertEqual(self.validation["qualified_count"], 42)
        self.assertEqual(self.validation["manifest_curve_count"], 42)

    def test_portfolio_statistics_do_not_enter_ranking(self) -> None:
        self.assertFalse(self.validation["portfolio_statistics_in_ranking"])


if __name__ == "__main__":
    unittest.main()
