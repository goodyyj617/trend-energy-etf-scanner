import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest import (
    STRATEGY_RULES,
    STRATEGY_YEAR_BASIS,
    STRATEGY_YEAR_SUMMARY_COLUMNS,
    _strategy_year_metadata,
    _trade_metric_values,
    build_strategy_year_summary,
    run_backtests,
    summarize_trades,
)


FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-10


class StrategyYearSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy_a = STRATEGY_RULES[0]
        self.strategy_b = STRATEGY_RULES[1]
        rows = [
            self._trade(self.strategy_a, "2020-12-31", "2021-01-05", -0.20, 1, "A"),
            self._trade(self.strategy_a, "2020-12-31", "2020-12-31", 0.00, 2, "B"),
            self._trade(self.strategy_a, "2020-12-31", "2021-01-04", 0.10, 3, "C"),
            self._trade(self.strategy_a, "2020-12-31", "2021-01-08", 0.30, 4, "D"),
            self._trade(self.strategy_a, "2021-06-01", "2021-06-03", -0.10, 3, "E"),
            self._trade(self.strategy_a, "2021-07-01", "2021-07-06", 0.20, 4, "F"),
            self._trade(self.strategy_b, "2022-01-03", "2022-01-04", 0.05, 2, "G"),
            self._trade(self.strategy_b, "2022-01-04", "2022-01-04", 0.00, 1, "H"),
        ]
        self.trades = pd.DataFrame(rows)
        self.skipped = pd.DataFrame()

    @staticmethod
    def _trade(strategy, entry_date: str, exit_date: str, net_return: float, holding_days: int, symbol: str) -> dict:
        return {
            "strategy_key": strategy.key,
            "strategy_label": strategy.label,
            "signal_key": strategy.signal.key,
            "signal_label": strategy.signal.label,
            "signal_params": json.dumps(strategy.signal.params, sort_keys=True),
            "entry_key": strategy.entry.key,
            "entry_label": strategy.entry.label,
            "exit_key": strategy.exit.key,
            "exit_label": strategy.exit.label,
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "net_return": net_return,
            "holding_days": holding_days,
            "exit_reason": "stop_hit",
        }

    def test_reconstructs_overall_counts_returns_average_and_profit_factor(self) -> None:
        annual = build_strategy_year_summary(self.trades)
        for strategy_key, group in self.trades.groupby("strategy_key"):
            rows = annual[annual["strategy_key"] == strategy_key]
            overall = _trade_metric_values(group)
            returns = pd.to_numeric(group["net_return"])

            self.assertEqual(int(rows["completed_trades"].sum()), overall["completed_trades"])
            self.assertEqual(int(rows["winning_trades"].sum()), int((returns > 0).sum()))
            self.assertEqual(int(rows["losing_trades"].sum()), int((returns < 0).sum()))
            self.assertEqual(int(rows["flat_trades"].sum()), int((returns == 0).sum()))
            self.assertEqual(
                int(rows[["winning_trades", "losing_trades", "flat_trades"]].to_numpy().sum()),
                overall["completed_trades"],
            )
            np.testing.assert_allclose(
                rows["sum_trade_returns"].sum(), overall["sum_trade_returns"],
                rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
            )

            expected_gross_profit = float(returns[returns > 0].sum())
            expected_gross_loss_abs = abs(float(returns[returns < 0].sum()))
            np.testing.assert_allclose(rows["gross_profit"].sum(), expected_gross_profit, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
            np.testing.assert_allclose(rows["gross_loss_abs"].sum(), expected_gross_loss_abs, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)

            reconstructed_avg = rows["sum_trade_returns"].sum() / rows["completed_trades"].sum()
            np.testing.assert_allclose(reconstructed_avg, overall["avg_trade_return"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
            if expected_gross_loss_abs > 0:
                reconstructed_pf = expected_gross_profit / expected_gross_loss_abs
                np.testing.assert_allclose(reconstructed_pf, overall["profit_factor"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
            else:
                self.assertTrue(np.isnan(overall["profit_factor"]))

    def test_distribution_metrics_use_existing_completed_return_conventions(self) -> None:
        annual = build_strategy_year_summary(self.trades)
        row = annual[
            (annual["strategy_key"] == self.strategy_a.key)
            & (annual["entry_year"] == 2020)
        ].iloc[0]

        self.assertEqual(row["completed_trades"], 4)
        self.assertEqual(row["winning_trades"], 2)
        self.assertEqual(row["losing_trades"], 1)
        self.assertEqual(row["flat_trades"], 1)
        self.assertAlmostEqual(row["gross_profit"], 0.40)
        self.assertAlmostEqual(row["gross_loss_abs"], 0.20)
        self.assertAlmostEqual(row["sum_trade_returns"], 0.20)
        self.assertAlmostEqual(row["avg_trade_return"], 0.05)
        self.assertAlmostEqual(row["median_trade_return"], 0.05)
        self.assertAlmostEqual(row["profit_factor"], 2.0)
        self.assertAlmostEqual(row["win_rate"], 0.50)
        self.assertAlmostEqual(row["p10_trade_return"], -0.14)
        self.assertAlmostEqual(row["worst_trade_return"], -0.20)
        self.assertAlmostEqual(row["best_trade_return"], 0.30)
        self.assertAlmostEqual(row["avg_holding_days"], 2.5)

    def test_zero_loss_profit_factor_matches_overall_summary(self) -> None:
        annual = build_strategy_year_summary(self.trades)
        row = annual[annual["strategy_key"] == self.strategy_b.key].iloc[0]
        overall = _trade_metric_values(self.trades[self.trades["strategy_key"] == self.strategy_b.key])
        self.assertEqual(row["gross_loss_abs"], 0.0)
        self.assertGreater(row["gross_profit"], 0.0)
        self.assertTrue(np.isnan(row["profit_factor"]))
        self.assertTrue(np.isnan(overall["profit_factor"]))

    def test_cross_year_exit_is_assigned_to_entry_year(self) -> None:
        annual = build_strategy_year_summary(self.trades)
        row = annual[
            (annual["strategy_key"] == self.strategy_a.key)
            & (annual["entry_year"] == 2020)
        ].iloc[0]
        self.assertEqual(row["completed_trades"], 4)
        self.assertEqual(row["last_exit_date"], "2021-01-08")

    def test_partial_and_full_year_flags_use_available_entry_period(self) -> None:
        annual = build_strategy_year_summary(self.trades)
        coverage = annual.drop_duplicates("entry_year").set_index("entry_year")

        self.assertEqual(coverage.loc[2020, "year_period_start"], "2020-12-31")
        self.assertEqual(coverage.loc[2020, "year_period_end"], "2020-12-31")
        self.assertTrue(bool(coverage.loc[2020, "is_partial_year"]))
        self.assertFalse(bool(coverage.loc[2020, "is_full_calendar_year"]))

        self.assertEqual(coverage.loc[2021, "year_period_start"], "2021-01-01")
        self.assertEqual(coverage.loc[2021, "year_period_end"], "2021-12-31")
        self.assertFalse(bool(coverage.loc[2021, "is_partial_year"]))
        self.assertTrue(bool(coverage.loc[2021, "is_full_calendar_year"]))

        self.assertEqual(coverage.loc[2022, "year_period_start"], "2022-01-01")
        self.assertEqual(coverage.loc[2022, "year_period_end"], "2022-01-04")
        self.assertTrue(bool(coverage.loc[2022, "is_partial_year"]))

    def test_empty_strategy_year_combinations_are_not_emitted(self) -> None:
        annual = build_strategy_year_summary(self.trades)
        observed = set(zip(annual["strategy_key"], annual["entry_year"]))
        self.assertEqual(
            observed,
            {
                (self.strategy_a.key, 2020),
                (self.strategy_a.key, 2021),
                (self.strategy_b.key, 2022),
            },
        )
        self.assertTrue((annual["year_basis"] == STRATEGY_YEAR_BASIS).all())

    def test_schema_and_identity_fields_are_stable(self) -> None:
        annual = build_strategy_year_summary(self.trades)
        self.assertEqual(list(annual.columns), STRATEGY_YEAR_SUMMARY_COLUMNS)
        row = annual.iloc[0]
        params = self.strategy_a.signal.params
        self.assertEqual(row["strategy_label"], self.strategy_a.label)
        self.assertEqual(row["score_lookback"], params["score_lookback"])
        self.assertEqual(row["r20_min"], params["r20_min"])
        self.assertEqual(row["er20_min"], params["er20_min"])
        self.assertEqual(row["entry_rule"], self.strategy_a.entry.key)
        self.assertEqual(row["exit_rule"], self.strategy_a.exit.key)

    def test_aggregation_does_not_mutate_existing_records_or_summary(self) -> None:
        trades_before = self.trades.copy(deep=True)
        skipped_before = self.skipped.copy(deep=True)
        summary_before = summarize_trades(self.trades, self.skipped)
        strategy_count_before = len(summary_before)

        annual = build_strategy_year_summary(self.trades)
        summary_after = summarize_trades(self.trades, self.skipped)

        pd.testing.assert_frame_equal(self.trades, trades_before)
        pd.testing.assert_frame_equal(self.skipped, skipped_before)
        pd.testing.assert_frame_equal(summary_after, summary_before)
        self.assertEqual(len(summary_after), strategy_count_before)
        self.assertEqual(len(self.trades), len(trades_before))
        self.assertEqual(len(self.skipped), len(skipped_before))
        self.assertEqual(annual["strategy_key"].nunique(), 2)

    def test_empty_input_emits_only_the_bounded_schema(self) -> None:
        annual = build_strategy_year_summary(pd.DataFrame())
        self.assertTrue(annual.empty)
        self.assertEqual(list(annual.columns), STRATEGY_YEAR_SUMMARY_COLUMNS)

    def test_small_mocked_backtest_writes_bounded_csv_and_metadata(self) -> None:
        features = pd.DataFrame({"symbol": ["TEST"]})
        simulated = (self.trades.to_dict(orient="records"), [], 2, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("src.backtest.build_historical_features", return_value=features),
                patch("src.backtest._simulate_symbol_strategies", return_value=simulated),
            ):
                payload = run_backtests(
                    prices=pd.DataFrame(),
                    universe=pd.DataFrame(),
                    cfg={"backtest_generate_signal_diagnostics": False},
                    data_dir=temp_dir,
                    as_of="2022-01-04",
                )

            output_path = Path(temp_dir) / "backtest_strategy_year_summary.csv"
            self.assertTrue(output_path.exists())
            generated = pd.read_csv(output_path)
            self.assertEqual(len(generated), 3)
            self.assertEqual(payload["strategy_year_summary_row_count"], 3)
            self.assertEqual(payload["strategy_year_min"], 2020)
            self.assertEqual(payload["strategy_year_max"], 2022)
            self.assertEqual(payload["strategy_year_basis"], STRATEGY_YEAR_BASIS)
            self.assertEqual(payload["partial_years"], [2020, 2022])
            self.assertEqual(payload["full_calendar_years"], [2021])
            self.assertGreaterEqual(payload["timing_metadata"]["strategy_year_aggregation_sec"], 0.0)
            self.assertGreaterEqual(payload["timing_metadata"]["gate_analysis_sec"], 0.0)
            self.assertEqual(payload["analysis_schema_version"], 3)
            self.assertEqual(payload["qualification_data_status"], "Available")
            self.assertIn("sample_gate_pass", payload["summary"][0])
            self.assertIn("loyo_pass_ratio", payload["summary"][0])
            self.assertIn("effective_neighbor_edge_pass_ratio", payload["summary"][0])
            self.assertFalse(any(row["mandatory_gates_pass"] for row in payload["summary"]))
            self.assertEqual(
                _strategy_year_metadata(generated),
                {
                    "strategy_year_summary_row_count": 3,
                    "strategy_year_min": 2020,
                    "strategy_year_max": 2022,
                    "strategy_year_basis": STRATEGY_YEAR_BASIS,
                    "partial_years": [2020, 2022],
                    "full_calendar_years": [2021],
                },
            )


if __name__ == "__main__":
    unittest.main()
