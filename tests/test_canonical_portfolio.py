import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio import (
    CDAR_DEFINITION_VERSION,
    INITIALIZATION_OBSERVATION,
    PORTFOLIO_CURVE_SCHEMA_VERSION,
    PORTFOLIO_MODEL_NAME,
    build_portfolio_outputs,
    build_price_panel,
    build_spy_benchmark,
    conditional_drawdown_at_risk,
    economic_curve,
    expected_shortfall,
    select_published_curve_keys,
    simulate_canonical_portfolio,
    summarize_portfolio_curve,
    write_portfolio_outputs,
    write_unavailable_portfolio_outputs,
)


def prices(symbols=("AAA", "BBB", "SPY"), periods=5):
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    rows = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            base = 100 + i * (1 if symbol != "BBB" else 2)
            rows.append({"date": date, "symbol": symbol, "open": base, "close": base, "low": base - 1})
    return pd.DataFrame(rows)


def trade(symbol="AAA", entry="2024-01-02", exit_date="2024-01-04", exit_price=102, reason="stop_hit"):
    return {
        "strategy_key": "s1", "symbol": symbol, "entry_date": entry, "entry_price": 100,
        "exit_date": exit_date, "exit_price": exit_price, "exit_reason": reason, "stop_at_exit": exit_price,
    }


class CanonicalPortfolioTests(unittest.TestCase):
    def test_flat_cash_and_initial_equity(self):
        curve = simulate_canonical_portfolio(pd.DataFrame(), build_price_panel(prices()), round_trip_cost=.002)
        self.assertEqual(curve.iloc[0].observation_type, INITIALIZATION_OBSERVATION)
        self.assertEqual(curve.iloc[1].observation_type, "trading_session")
        self.assertEqual(curve.iloc[1].active_position_count, 0)
        self.assertTrue((curve.portfolio_equity == 1000).all())
        self.assertTrue((curve.cash_value == 1000).all())
        self.assertTrue((curve.gross_exposure == 0).all())

    def test_one_position_cost_fractional_and_reconciliation(self):
        panel = build_price_panel(prices(symbols=("AAA",), periods=3))
        curve = simulate_canonical_portfolio(pd.DataFrame([trade()]), panel, round_trip_cost=.002)
        self.assertEqual(curve.iloc[0].observation_type, INITIALIZATION_OBSERVATION)
        self.assertAlmostEqual(curve.iloc[1].transaction_cost_paid, 1000 / 1.001 * .001, places=6)
        self.assertTrue(np.allclose(curve.cash_value + curve.invested_value, curve.portfolio_equity))
        reconstructed = 1000 * (1 + curve.daily_portfolio_return).prod()
        self.assertAlmostEqual(reconstructed, curve.portfolio_equity.iloc[-1], places=8)
        self.assertGreater(curve.invested_value.iloc[1] / 100, 9.0)
        self.assertGreaterEqual(curve.cash_value.min(), -1e-9)
        self.assertLessEqual(curve.gross_exposure.max(), 1 + 1e-9)

    def test_two_positions_equal_weight_no_duplicate(self):
        frame = pd.DataFrame([trade(), trade("BBB")])
        curve = simulate_canonical_portfolio(frame, build_price_panel(prices(periods=3)), round_trip_cost=.002)
        self.assertEqual(curve.active_position_count.iloc[1], 2)
        self.assertAlmostEqual(curve.invested_value.iloc[1], 1000 / 1.001, places=5)
        duplicate = pd.DataFrame([trade(), trade()])
        dup_curve = simulate_canonical_portfolio(duplicate, build_price_panel(prices(("AAA",), 3)), round_trip_cost=.002)
        self.assertEqual(dup_curve.active_position_count.max(), 1)

    def test_gap_stop_and_intraday_cash_timing(self):
        p = prices(("AAA", "BBB"), 4)
        p.loc[(p.symbol == "AAA") & (p.date == pd.Timestamp("2024-01-04")), "open"] = 90
        rows = [
            trade(exit_price=90),
            trade("BBB", entry="2024-01-04", exit_date=None, exit_price=None),
        ]
        curve = simulate_canonical_portfolio(pd.DataFrame(rows), build_price_panel(p), round_trip_cost=.002)
        # Gap exit precedes the new entry and both active members are determined at the open.
        self.assertEqual(curve.loc[curve.date == "2024-01-04", "active_position_count"].iloc[0], 1)

        intraday = pd.DataFrame([trade(exit_price=95), trade("BBB", entry="2024-01-05", exit_date=None, exit_price=None)])
        icurve = simulate_canonical_portfolio(intraday, build_price_panel(prices(("AAA", "BBB"), 4)), round_trip_cost=.002)
        stop_day = icurve.loc[icurve.date == "2024-01-04"].iloc[0]
        self.assertEqual(stop_day.active_position_count, 0)
        self.assertGreater(stop_day.cash_value, 0)

    def test_max_hold_close_not_used_at_open(self):
        rows = [trade(reason="max_holding_days"), trade("BBB", entry="2024-01-04", exit_date=None, exit_price=None)]
        curve = simulate_canonical_portfolio(pd.DataFrame(rows), build_price_panel(prices(("AAA", "BBB"), 4)), round_trip_cost=.002)
        day = curve.loc[curve.date == "2024-01-04"].iloc[0]
        self.assertEqual(day.active_position_count, 1)
        self.assertGreater(day.cash_value, 0)

    def test_drawdown_recovery_es_and_cdar(self):
        panel = build_price_panel(prices(("AAA",), 4))
        p = panel.closes
        p[:, 0] = [100, 80, 90, 105]
        curve = simulate_canonical_portfolio(
            pd.DataFrame([trade(exit_date=None, exit_price=None)]), panel, round_trip_cost=0
        )
        summary = summarize_portfolio_curve("s1", "S1", curve)
        self.assertAlmostEqual(summary["maximum_drawdown"], -.2)
        self.assertEqual(summary["max_drawdown_recovery_date"], "2024-01-05")
        fixture = pd.Series([-.10, -.05, 0, .02])
        self.assertAlmostEqual(expected_shortfall(fixture, .95), -.10)
        self.assertAlmostEqual(conditional_drawdown_at_risk(fixture.clip(upper=0), .95), -.10)

    def test_spy_normalization_and_selection_cap(self):
        benchmark = build_spy_benchmark(build_price_panel(prices()))
        self.assertEqual(benchmark["status"], "Available")
        self.assertEqual(benchmark["series"][0]["benchmark_equity"], 1000)
        self.assertEqual(benchmark["series"][0]["observation_type"], INITIALIZATION_OBSERVATION)
        self.assertEqual(benchmark["series"][1]["benchmark_equity"], 1000)
        self.assertEqual(benchmark["curve_schema_version"], PORTFOLIO_CURVE_SCHEMA_VERSION)
        rows = pd.DataFrame([
            {"strategy_key": f"s{i}", "qualification_rank": i, "qualification_tier": "Qualified",
             "profit_factor": i, "avg_trade_return": i, "joint_positive_year_ratio": i,
             "loyo_pass_ratio": i, "effective_neighbor_edge_pass_ratio": i, "median_trade_return": i}
            for i in range(1, 121)
        ])
        selected = select_published_curve_keys(rows, cap=100)
        self.assertEqual(len(selected), 100)
        self.assertEqual(selected[0], "s1")
        self.assertEqual(selected[-1], "s100")

    def test_unavailable_outputs_are_safe_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_unavailable_portfolio_outputs(root, "Full run required.")
            self.assertEqual(manifest["status"], "Not available")
            self.assertEqual(manifest["curve_schema_version"], PORTFOLIO_CURVE_SCHEMA_VERSION)
            benchmark = json.loads((root / "backtest_benchmark_spy.json").read_text())
            self.assertEqual(benchmark["series"], [])
            self.assertEqual(benchmark["curve_schema_version"], PORTFOLIO_CURVE_SCHEMA_VERSION)
            self.assertEqual(benchmark["cdar_definition_version"], CDAR_DEFINITION_VERSION)
            first = (root / "backtest_portfolio_daily_returns.csv.gz").read_bytes()
            write_unavailable_portfolio_outputs(root, "Full run required.")
            self.assertEqual(first, (root / "backtest_portfolio_daily_returns.csv.gz").read_bytes())
            self.assertEqual(PORTFOLIO_MODEL_NAME, "canonical_equal_weight_active_v1")

    def test_reduced_output_fixture_publishes_t0_without_changing_economics(self):
        strategy_summary = pd.DataFrame([
            {
                "strategy_key": "s1",
                "strategy_label": "S1",
                "qualification_rank": 1,
                "qualification_tier": "Qualified",
                "profit_factor": 1.3,
                "avg_trade_return": .01,
                "joint_positive_year_ratio": .6,
                "loyo_pass_ratio": .8,
                "effective_neighbor_edge_pass_ratio": .6,
                "median_trade_return": .005,
            },
            {
                "strategy_key": "s2",
                "strategy_label": "S2 flat",
                "qualification_rank": 2,
                "qualification_tier": "Qualified",
                "profit_factor": 1.2,
                "avg_trade_return": .005,
                "joint_positive_year_ratio": .6,
                "loyo_pass_ratio": .8,
                "effective_neighbor_edge_pass_ratio": .6,
                "median_trade_return": .002,
            },
        ])
        outputs = build_portfolio_outputs(
            pd.DataFrame([trade()]),
            pd.DataFrame(),
            prices(),
            strategy_summary,
            round_trip_cost=.002,
        )

        self.assertEqual(outputs["curve_schema_version"], PORTFOLIO_CURVE_SCHEMA_VERSION)
        self.assertEqual(outputs["cdar_definition_version"], CDAR_DEFINITION_VERSION)
        self.assertEqual(len(outputs["daily_returns"]), 6)
        self.assertEqual(outputs["daily_returns"].iloc[0]["s1"], 0)
        self.assertEqual(outputs["daily_returns"].iloc[0]["s2"], 0)
        for key in ("s1", "s2"):
            curve = outputs["curves"][key]
            self.assertEqual(curve.iloc[0].observation_type, INITIALIZATION_OBSERVATION)
            self.assertEqual(curve.iloc[0].portfolio_equity, 1000)

        s1_curve = outputs["curves"]["s1"]
        s1_economic = economic_curve(s1_curve)
        self.assertAlmostEqual(
            1000 * (1 + s1_economic.daily_portfolio_return).prod(),
            s1_curve.iloc[-1].portfolio_equity,
            places=8,
        )
        self.assertAlmostEqual(
            outputs["summary"].set_index("strategy_key").loc["s1", "ending_equity"],
            s1_curve.iloc[-1].portfolio_equity,
            places=12,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_portfolio_outputs(outputs, root)
            self.assertEqual(manifest["curve_schema_version"], PORTFOLIO_CURVE_SCHEMA_VERSION)
            self.assertEqual(manifest["cdar_definition_version"], CDAR_DEFINITION_VERSION)
            self.assertEqual(manifest["published_curve_count"], 2)
            payload = json.loads(
                (root / "backtest_portfolio_curves" / "s1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["series"][0]["observation_type"], INITIALIZATION_OBSERVATION)
            self.assertEqual(payload["series"][0]["portfolio_equity"], 1000)
            benchmark = json.loads(
                (root / "backtest_benchmark_spy.json").read_text(encoding="utf-8")
            )
            self.assertEqual(benchmark["series"][0]["observation_type"], INITIALIZATION_OBSERVATION)

    def test_t0_is_cash_only_and_precedes_first_session(self):
        curve = simulate_canonical_portfolio(
            pd.DataFrame([trade()]), build_price_panel(prices(("AAA",), 3)), round_trip_cost=.002
        )
        t0 = curve.iloc[0]
        self.assertEqual(t0.portfolio_equity, 1000)
        self.assertEqual(t0.cash_value, 1000)
        self.assertEqual(t0.invested_value, 0)
        self.assertEqual(t0.gross_exposure, 0)
        self.assertEqual(t0.active_position_count, 0)
        self.assertEqual(t0.daily_portfolio_return, 0)
        self.assertEqual(t0.cumulative_return, 0)
        self.assertEqual(t0.running_peak_equity, 1000)
        self.assertEqual(t0.drawdown, 0)
        self.assertEqual(t0.transaction_cost_paid, 0)
        self.assertEqual(t0.turnover, 0)
        self.assertLess(pd.Timestamp(t0.date), pd.Timestamp(curve.iloc[1].date, tz="UTC"))

    def test_t0_preserves_earliest_entry_economics_and_multiple_positions(self):
        panel = build_price_panel(prices(("AAA", "BBB"), 3))
        curve = simulate_canonical_portfolio(
            pd.DataFrame([trade(), trade("BBB")]), panel, round_trip_cost=.002
        )
        economic = economic_curve(curve)
        self.assertEqual(economic.iloc[0].active_position_count, 2)
        self.assertGreater(economic.iloc[0].transaction_cost_paid, 0)
        self.assertAlmostEqual(
            1000 * (1 + economic.daily_portfolio_return).prod(),
            curve.iloc[-1].portfolio_equity,
            places=8,
        )
        self.assertEqual(curve.transaction_cost_paid.sum(), economic.transaction_cost_paid.sum())
        self.assertEqual(curve.turnover.sum(), economic.turnover.sum())

    def test_t0_does_not_shift_cagr_economic_day_count(self):
        curve = simulate_canonical_portfolio(
            pd.DataFrame([trade()]), build_price_panel(prices(("AAA",), 5)), round_trip_cost=.002
        )
        with_t0 = summarize_portfolio_curve("s1", "S1", curve)
        without_t0 = summarize_portfolio_curve("s1", "S1", economic_curve(curve))
        self.assertEqual(with_t0["portfolio_start_date"], "2024-01-02")
        self.assertAlmostEqual(with_t0["ending_equity"], without_t0["ending_equity"], places=12)
        self.assertAlmostEqual(with_t0["cagr"], without_t0["cagr"], places=12)


class ConditionalDrawdownAtRiskTests(unittest.TestCase):
    def test_fixed_tail_count_fixtures(self):
        fixtures = json.loads(
            (Path(__file__).parent / "fixtures" / "portfolio_cdar_fixtures.json").read_text(
                encoding="utf-8"
            )
        )
        for fixture in fixtures:
            values = fixture["values"] + [0] * fixture["repeat_zeros"]
            with self.subTest(name=fixture["name"]):
                self.assertAlmostEqual(
                    conditional_drawdown_at_risk(pd.Series(values), .95),
                    fixture["expected"],
                )

    def test_non_finite_values_are_filtered_deterministically(self):
        values = pd.Series([np.nan, np.inf, -np.inf, -.2, 0])
        self.assertAlmostEqual(conditional_drawdown_at_risk(values, .95), -.2)

    def test_invalid_confidence_and_positive_drawdown(self):
        with self.assertRaises(ValueError):
            conditional_drawdown_at_risk(pd.Series([-.1]), 1)
        with self.assertRaises(ValueError):
            conditional_drawdown_at_risk(pd.Series([.01, 0]), .95)
        self.assertEqual(conditional_drawdown_at_risk(pd.Series([5e-10, 0]), .95), 0)

    def test_definition_version_is_stable(self):
        self.assertEqual(CDAR_DEFINITION_VERSION, "negative_drawdown_fixed_tail_count_v1")


if __name__ == "__main__":
    unittest.main()
