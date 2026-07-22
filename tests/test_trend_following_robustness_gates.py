import json
import random
import unittest
from pathlib import Path

import pandas as pd

from src.backtest import (
    ANALYSIS_SCHEMA_VERSION,
    SIGNAL_RULES,
    STRATEGY_YEAR_BASIS,
    _attach_parameter_stability,
    _attach_robustness_tiers,
    _attach_time_stability,
    _gate_fields,
    _strategy_year_details,
    rank_strategy_summary,
    validate_strategy_year_summary,
)


ROOT = Path(__file__).parents[1]


def annual_row(
    strategy_key: str,
    year: int,
    *,
    completed: int = 100,
    gross_profit: float = 50.0,
    gross_loss: float = 10.0,
    median: float = -0.01,
    full: bool = True,
) -> dict:
    summed = gross_profit - gross_loss
    return {
        "strategy_key": strategy_key,
        "entry_year": year,
        "year_basis": STRATEGY_YEAR_BASIS,
        "is_partial_year": not full,
        "is_full_calendar_year": full,
        "completed_trades": completed,
        "gross_profit": gross_profit,
        "gross_loss_abs": gross_loss,
        "sum_trade_returns": summed,
        "avg_trade_return": summed / completed,
        "median_trade_return": median,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "win_rate": 0.45,
    }


def overall_metrics(completed: int, profit_factor: float, avg: float, median: float = -0.01) -> dict:
    return {
        "completed_trades": completed,
        "profit_factor": profit_factor,
        "avg_trade_return": avg,
        "median_trade_return": median,
        "trade_win_rate": 0.45,
    }


class ProvisionalGateBoundaryTest(unittest.TestCase):
    def test_sample_gate_boundary_at_99_and_100(self) -> None:
        self.assertFalse(_gate_fields(overall_metrics(99, 1.2, 0.01))["sample_gate_pass"])
        self.assertTrue(_gate_fields(overall_metrics(100, 1.2, 0.01))["sample_gate_pass"])

    def test_edge_profit_factor_boundary(self) -> None:
        self.assertFalse(_gate_fields(overall_metrics(100, 1.1999, 0.01))["edge_gate_pass"])
        self.assertTrue(_gate_fields(overall_metrics(100, 1.2, 0.01))["edge_gate_pass"])
        self.assertTrue(_gate_fields(overall_metrics(100, 1.2001, 0.01))["edge_gate_pass"])

    def test_edge_average_return_boundary(self) -> None:
        self.assertFalse(_gate_fields(overall_metrics(100, 1.2, -0.0001))["edge_gate_pass"])
        self.assertFalse(_gate_fields(overall_metrics(100, 1.2, 0.0))["edge_gate_pass"])
        self.assertTrue(_gate_fields(overall_metrics(100, 1.2, 0.0001))["edge_gate_pass"])

    def test_partial_year_and_annual_trade_minimum_are_excluded(self) -> None:
        key = "candidate"
        rows = [
            annual_row(key, 2017, completed=500, full=False),
            annual_row(key, 2018, completed=99),
            annual_row(key, 2019, completed=100),
        ]
        result = _attach_time_stability(pd.DataFrame({"strategy_key": [key]}), pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result.eligible_years, 1)
        self.assertEqual(result.joint_positive_years, 1)
        self.assertEqual(result.loyo_fold_count, 1)

    def test_joint_positive_ignores_annual_median(self) -> None:
        key = "candidate"
        passing = annual_row(key, 2018, gross_profit=20, gross_loss=10, median=-0.50)
        avg_fail = annual_row(key, 2019, gross_profit=9, gross_loss=10, median=0.50)
        pf_fail = annual_row(key, 2020, gross_profit=10, gross_loss=10, median=0.50)
        details = _strategy_year_details(pd.DataFrame([passing, avg_fail, pf_fail]), None)
        self.assertEqual(details["joint_positive_status"].tolist(), ["Yes", "No", "No"])

    def test_time_gate_eligible_year_and_joint_ratio_boundaries(self) -> None:
        key = "candidate"
        four = pd.DataFrame([annual_row(key, year) for year in range(2018, 2022)])
        result = _attach_time_stability(pd.DataFrame({"strategy_key": [key]}), four).iloc[0]
        self.assertFalse(result.time_gate_pass)
        self.assertEqual(result.time_gate_status, "Insufficient")

        five = pd.DataFrame([
            annual_row(key, 2018, gross_profit=50, gross_loss=10),
            annual_row(key, 2019, gross_profit=50, gross_loss=10),
            annual_row(key, 2020, gross_profit=50, gross_loss=10),
            annual_row(key, 2021, gross_profit=9, gross_loss=10),
            annual_row(key, 2022, gross_profit=9, gross_loss=10),
        ])
        result = _attach_time_stability(pd.DataFrame({"strategy_key": [key]}), five).iloc[0]
        self.assertEqual(result.eligible_years, 5)
        self.assertEqual(result.joint_positive_year_ratio, 0.60)
        self.assertTrue(result.time_gate_pass)

        below = five.copy()
        below.loc[below.entry_year == 2020, ["gross_profit", "sum_trade_returns", "avg_trade_return", "profit_factor"]] = [
            9.0, -1.0, -0.01, 0.9,
        ]
        result = _attach_time_stability(pd.DataFrame({"strategy_key": [key]}), below).iloc[0]
        self.assertEqual(result.joint_positive_year_ratio, 0.40)
        self.assertFalse(result.time_gate_pass)

    def test_loyo_reconstructs_pooled_metrics_and_ratio_boundary(self) -> None:
        key = "candidate"
        at_boundary = pd.DataFrame([
            annual_row(key, 2018, gross_profit=100, gross_loss=10),
            *[annual_row(key, year, gross_profit=0, gross_loss=10) for year in range(2019, 2023)],
        ])
        result = _attach_time_stability(pd.DataFrame({"strategy_key": [key]}), at_boundary).iloc[0]
        self.assertEqual(result.loyo_pass_count, 4)
        self.assertEqual(result.loyo_pass_ratio, 0.80)
        self.assertEqual(result.omitted_year_causing_worst_avg, 2018)
        self.assertEqual(result.omitted_year_causing_worst_pf, 2018)
        self.assertAlmostEqual(result.worst_loyo_avg_trade_return, -0.10)
        self.assertAlmostEqual(result.worst_loyo_profit_factor, 0.0)

        below = pd.DataFrame([
            annual_row(key, 2018, gross_profit=30, gross_loss=10),
            annual_row(key, 2019, gross_profit=30, gross_loss=10),
            *[annual_row(key, year, gross_profit=0, gross_loss=10) for year in range(2020, 2023)],
        ])
        result = _attach_time_stability(pd.DataFrame({"strategy_key": [key]}), below).iloc[0]
        self.assertEqual(result.loyo_pass_ratio, 0.60)


class ParameterAndQualificationTest(unittest.TestCase):
    @staticmethod
    def _signal_key(params: tuple[int, float, float]) -> str:
        for rule in SIGNAL_RULES:
            observed = (
                int(rule.params["score_lookback"]),
                float(rule.params["r20_min"]),
                float(rule.params["er20_min"]),
            )
            if observed == params:
                return rule.key
        raise AssertionError(f"Missing fixture signal params: {params}")

    def _parameter_fixture(self, neighbors: list[tuple[tuple[int, float, float], bool]]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        candidate_params = (20, 0.0, 0.10)
        definitions = [(candidate_params, True), *neighbors]
        summary_rows = []
        annual_rows = []
        for index, (params, edge_pass) in enumerate(definitions):
            signal_key = self._signal_key(params)
            strategy_key = f"{signal_key}__entry__exit"
            pf = 1.3 if edge_pass else 1.1
            avg = 0.01 if edge_pass else -0.01
            summary_rows.append({
                "strategy_key": strategy_key,
                "signal_key": signal_key,
                "entry_key": "entry",
                "exit_key": "exit",
                "signal_params": json.dumps({"score_lookback": params[0], "r20_min": params[1], "er20_min": params[2]}),
                "completed_trades": 100,
                "profit_factor": pf,
                "avg_trade_return": avg,
                "median_trade_return": -0.01 + index * 0.0001,
                "trade_win_rate": 0.4 + index * 0.001,
                "trade_sequence_drawdown": -0.2,
            })
            annual_rows.append(annual_row(strategy_key, 2018, gross_profit=13 + index, gross_loss=10))
        candidate_key = summary_rows[0]["strategy_key"]
        return pd.DataFrame(summary_rows), pd.DataFrame(annual_rows), candidate_key

    def test_parameter_neighbor_minimum_below_and_at_two(self) -> None:
        one_summary, one_annual, key = self._parameter_fixture([((10, 0.0, 0.10), True)])
        row = _attach_parameter_stability(one_summary, one_annual).set_index("strategy_key").loc[key]
        self.assertEqual(row.effective_eligible_neighbors, 1)
        self.assertFalse(row.parameter_gate_pass)

        two_summary, two_annual, key = self._parameter_fixture([
            ((10, 0.0, 0.10), True),
            ((40, 0.0, 0.10), True),
        ])
        row = _attach_parameter_stability(two_summary, two_annual).set_index("strategy_key").loc[key]
        self.assertEqual(row.effective_eligible_neighbors, 2)
        self.assertTrue(row.parameter_gate_pass)

    def test_effective_neighbor_ratio_boundary(self) -> None:
        params = [
            (10, 0.0, 0.10), (40, 0.0, 0.10), (20, -0.02, 0.10),
            (20, 0.02, 0.10), (20, 0.0, 0.05),
        ]
        summary, annual, key = self._parameter_fixture(list(zip(params, [True, True, True, False, False])))
        row = _attach_parameter_stability(summary, annual).set_index("strategy_key").loc[key]
        self.assertEqual(row.effective_eligible_neighbors, 5)
        self.assertEqual(row.effective_neighbor_edge_pass_ratio, 0.60)
        self.assertTrue(row.parameter_gate_pass)

        summary, annual, key = self._parameter_fixture(list(zip(params, [True, True, False, False, False])))
        row = _attach_parameter_stability(summary, annual).set_index("strategy_key").loc[key]
        self.assertEqual(row.effective_neighbor_edge_pass_ratio, 0.40)
        self.assertFalse(row.parameter_gate_pass)

    def test_candidate_identical_neighbor_is_removed_from_effective_breadth(self) -> None:
        summary, annual, key = self._parameter_fixture([((20, -0.02, 0.10), True)])
        metric_columns = [
            "completed_trades", "profit_factor", "avg_trade_return", "median_trade_return", "trade_win_rate",
        ]
        summary.loc[1, metric_columns] = summary.loc[0, metric_columns].to_numpy()
        annual_metric_columns = [
            "completed_trades", "gross_profit", "gross_loss_abs", "sum_trade_returns",
            "avg_trade_return", "median_trade_return", "profit_factor", "win_rate",
        ]
        annual.loc[1, annual_metric_columns] = annual.loc[0, annual_metric_columns].to_numpy()
        row = _attach_parameter_stability(summary, annual).set_index("strategy_key").loc[key]
        self.assertEqual(row.raw_eligible_neighbors, 1)
        self.assertEqual(row.effective_eligible_neighbors, 0)
        self.assertFalse(row.parameter_gate_pass)

    def test_final_qualification_requires_every_gate_and_allows_negative_median(self) -> None:
        row = {
            **overall_metrics(100, 1.2, 0.001, median=-0.20),
            **_gate_fields(overall_metrics(100, 1.2, 0.001, median=-0.20)),
            "time_gate_pass": True,
            "parameter_gate_pass": True,
            "time_gate_unavailable_reason": None,
            "parameter_gate_unavailable_reason": None,
        }
        qualified = _attach_robustness_tiers(pd.DataFrame([row])).iloc[0]
        self.assertTrue(qualified.mandatory_gates_pass)
        self.assertEqual(qualified.qualification_tier, "Qualified")
        self.assertLess(qualified.median_trade_return, 0)

        for gate in ["sample_gate_pass", "edge_gate_pass", "time_gate_pass", "parameter_gate_pass"]:
            failing = dict(row)
            failing[gate] = False
            result = _attach_robustness_tiers(pd.DataFrame([failing])).iloc[0]
            self.assertFalse(result.mandatory_gates_pass, gate)

    def test_missing_or_invalid_annual_data_cannot_qualify(self) -> None:
        summary = pd.DataFrame([{"strategy_key": "candidate"}])
        timed = _attach_time_stability(summary, pd.DataFrame(), validate_strategy_year_summary(pd.DataFrame()))
        self.assertIsNone(timed.iloc[0].time_gate_pass)
        self.assertEqual(timed.iloc[0].time_gate_status, "Not available")

        base = {
            **overall_metrics(100, 1.3, 0.01),
            **_gate_fields(overall_metrics(100, 1.3, 0.01)),
            "time_gate_pass": None,
            "parameter_gate_pass": True,
            "time_gate_unavailable_reason": "Strategy-year aggregate is unavailable.",
            "parameter_gate_unavailable_reason": None,
        }
        result = _attach_robustness_tiers(pd.DataFrame([base])).iloc[0]
        self.assertFalse(result.mandatory_gates_pass)
        self.assertEqual(result.qualification_tier, "Not qualified")
        self.assertEqual(result.mandatory_gate_status, "Not available")

        annual = pd.DataFrame([annual_row("candidate", 2018)])
        overall = pd.DataFrame([{
            "strategy_key": "candidate", "completed_trades": 101, "sum_trade_returns": 40.0,
            "avg_trade_return": 40 / 101, "profit_factor": 5.0,
        }])
        reason = validate_strategy_year_summary(annual, overall)
        self.assertIn("do not reconstruct", reason)


class RankingAndPresentationTest(unittest.TestCase):
    def test_committed_bounded_aggregates_reproduce_approved_42_and_primary_candidate(self) -> None:
        summary = pd.read_csv(ROOT / "docs" / "data" / "backtest_strategy_summary.csv")
        annual = pd.read_csv(ROOT / "docs" / "data" / "backtest_strategy_year_summary.csv")
        gate_fields = pd.DataFrame([
            _gate_fields({
                "completed_trades": row.completed_trades,
                "profit_factor": row.profit_factor,
                "avg_trade_return": row.avg_trade_return,
                "median_trade_return": row.median_trade_return,
                "trade_win_rate": row.trade_win_rate,
            })
            for row in summary.itertuples(index=False)
        ])
        production = pd.concat([summary.reset_index(drop=True), gate_fields], axis=1)
        production = _attach_time_stability(production, annual)
        production = _attach_parameter_stability(production, annual)
        production = _attach_robustness_tiers(production)
        production = rank_strategy_summary(production)

        self.assertEqual(len(production), len(summary))
        required = {
            "sample_gate_pass", "edge_gate_pass", "time_gate_pass", "parameter_gate_pass",
            "mandatory_gates_pass", "qualification_tier", "eligible_years", "joint_positive_years",
            "joint_positive_year_ratio", "median_annual_avg_trade_return", "minimum_annual_avg_trade_return",
            "median_annual_profit_factor", "minimum_annual_profit_factor", "loyo_fold_count",
            "loyo_pass_count", "loyo_pass_ratio", "worst_loyo_avg_trade_return",
            "worst_loyo_profit_factor", "omitted_year_causing_worst_avg", "omitted_year_causing_worst_pf",
            "raw_eligible_neighbors", "raw_neighbor_edge_pass_ratio", "effective_eligible_neighbors",
            "effective_neighbor_edge_pass_ratio", "qualification_rank",
        }
        self.assertTrue(required.issubset(production.columns))
        self.assertEqual(int(production.mandatory_gates_pass.sum()), 42)
        self.assertEqual(
            production.iloc[0].strategy_key,
            "score_bo_l40_rm002_erp010__signal_3d_confirm__ma50",
        )
        self.assertEqual(production.iloc[0].qualification_tier, "Qualified")
        self.assertLess(production.iloc[0].median_trade_return, 0)
        reranked = rank_strategy_summary(production.sample(frac=1, random_state=19))
        self.assertEqual(reranked.strategy_key.tolist(), production.strategy_key.tolist())

    def test_ranking_is_deterministic_and_input_order_independent(self) -> None:
        rows = []
        for key, pf, avg, trades in [("b", 1.4, 0.02, 200), ("a", 1.4, 0.02, 200), ("c", 1.3, 0.03, 300)]:
            rows.append({
                "strategy_key": key,
                "qualification_tier": "Qualified",
                "time_gate_pass": True,
                "parameter_gate_pass": True,
                "loyo_pass_ratio": 1.0,
                "joint_positive_year_ratio": 0.60,
                "effective_neighbor_edge_pass_ratio": 1.0,
                "profit_factor": pf,
                "avg_trade_return": avg,
                "completed_trades": trades,
            })
        expected = rank_strategy_summary(pd.DataFrame(rows))["strategy_key"].tolist()
        self.assertEqual(expected, ["a", "b", "c"])
        random.Random(7).shuffle(rows)
        self.assertEqual(rank_strategy_summary(pd.DataFrame(rows))["strategy_key"].tolist(), expected)

    def test_ranking_uses_typed_values_documented_directions_and_missing_last(self) -> None:
        base = {
            "qualification_tier": "Qualified",
            "time_gate_pass": True,
            "parameter_gate_pass": True,
            "loyo_pass_ratio": "1.0",
            "joint_positive_year_ratio": "0.625",
            "effective_neighbor_edge_pass_ratio": "1.0",
            "profit_factor": "2.0",
            "avg_trade_return": "0.01",
            "completed_trades": "100",
        }

        def assert_first(winner_changes: dict, loser_changes: dict) -> None:
            winner = {**base, "strategy_key": "winner", **winner_changes}
            loser = {**base, "strategy_key": "loser", **loser_changes}
            rows = pd.DataFrame([loser, winner]).sample(frac=1, random_state=23)
            ranked = rank_strategy_summary(rows)
            self.assertEqual(ranked.iloc[0].strategy_key, "winner")
            for column in [
                "loyo_pass_ratio", "joint_positive_year_ratio", "effective_neighbor_edge_pass_ratio",
                "profit_factor", "avg_trade_return", "completed_trades",
            ]:
                self.assertTrue(pd.api.types.is_numeric_dtype(ranked[column]), column)

        assert_first({}, {"qualification_tier": "Not qualified"})
        assert_first({}, {"time_gate_pass": False})
        assert_first({}, {"parameter_gate_pass": False})
        for field in [
            "loyo_pass_ratio", "joint_positive_year_ratio", "effective_neighbor_edge_pass_ratio",
            "profit_factor", "avg_trade_return", "completed_trades",
        ]:
            assert_first({field: "10"}, {field: "2"})
            assert_first({field: "2"}, {field: None})

        tied = pd.DataFrame([
            {**base, "strategy_key": "b"},
            {**base, "strategy_key": "a"},
        ])
        self.assertEqual(rank_strategy_summary(tied).strategy_key.tolist(), ["a", "b"])

    def test_partial_year_display_and_unavailable_ui_contract_are_present(self) -> None:
        key = "candidate"
        details = _strategy_year_details(pd.DataFrame([annual_row(key, 2017, full=False)]), None)
        self.assertEqual(details.iloc[0].joint_positive_status, "Not eligible")
        self.assertTrue(details.iloc[0].is_partial_year)

        dashboard = (ROOT / "docs" / "backtest_dashboard.js").read_text(encoding="utf-8")
        page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Not available", dashboard)
        self.assertIn("joint_positive_status", dashboard)
        self.assertIn("Partial", dashboard)
        self.assertIn("LOYO Pass Ratio", page)

    def test_workflow_ownership_and_raw_output_policy_are_unchanged(self) -> None:
        backtest_workflow = (ROOT / ".github" / "workflows" / "backtest-only.yml").read_text(encoding="utf-8")
        daily_workflow = (ROOT / ".github" / "workflows" / "daily_scan.yml").read_text(encoding="utf-8")
        self.assertIn("git add docs/data/backtest_summary.json docs/data/backtest_strategy_summary.csv docs/data/backtest_strategy_year_summary.csv", backtest_workflow)
        self.assertIn("git restore --worktree -- docs/data/backtest_summary.json docs/data/backtest_strategy_summary.csv", daily_workflow)
        for raw in ["backtest_trades.csv", "signal_diagnostics.csv", "backtest_skipped.csv"]:
            self.assertIn(raw, backtest_workflow)
            self.assertIn(raw, daily_workflow)
            self.assertFalse((ROOT / "docs" / "data" / raw).exists())

    def test_schema_version_is_incremented_for_gate_contract(self) -> None:
        self.assertEqual(ANALYSIS_SCHEMA_VERSION, 3)


if __name__ == "__main__":
    unittest.main()
