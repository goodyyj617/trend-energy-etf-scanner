import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio import (
    PORTFOLIO_MODEL_NAME,
    build_price_panel,
    build_spy_benchmark,
    conditional_drawdown_at_risk,
    expected_shortfall,
    select_published_curve_keys,
    simulate_canonical_portfolio,
    summarize_portfolio_curve,
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
        self.assertTrue((curve.portfolio_equity == 1000).all())
        self.assertTrue((curve.cash_value == 1000).all())
        self.assertTrue((curve.gross_exposure == 0).all())

    def test_one_position_cost_fractional_and_reconciliation(self):
        panel = build_price_panel(prices(symbols=("AAA",), periods=3))
        curve = simulate_canonical_portfolio(pd.DataFrame([trade()]), panel, round_trip_cost=.002)
        self.assertAlmostEqual(curve.iloc[0].transaction_cost_paid, 1000 / 1.001 * .001, places=6)
        self.assertTrue(np.allclose(curve.cash_value + curve.invested_value, curve.portfolio_equity))
        reconstructed = 1000 * (1 + curve.daily_portfolio_return).prod()
        self.assertAlmostEqual(reconstructed, curve.portfolio_equity.iloc[-1], places=8)
        self.assertGreater(curve.invested_value.iloc[0] / 100, 9.0)
        self.assertGreaterEqual(curve.cash_value.min(), -1e-9)
        self.assertLessEqual(curve.gross_exposure.max(), 1 + 1e-9)

    def test_two_positions_equal_weight_no_duplicate(self):
        frame = pd.DataFrame([trade(), trade("BBB")])
        curve = simulate_canonical_portfolio(frame, build_price_panel(prices(periods=3)), round_trip_cost=.002)
        self.assertEqual(curve.active_position_count.iloc[0], 2)
        self.assertAlmostEqual(curve.invested_value.iloc[0], 1000 / 1.001, places=5)
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
        self.assertAlmostEqual(conditional_drawdown_at_risk(fixture, .95), -.10)

    def test_spy_normalization_and_selection_cap(self):
        benchmark = build_spy_benchmark(build_price_panel(prices()))
        self.assertEqual(benchmark["status"], "Available")
        self.assertEqual(benchmark["series"][0]["benchmark_equity"], 1000)
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
            self.assertEqual(json.loads((root / "backtest_benchmark_spy.json").read_text())["series"], [])
            first = (root / "backtest_portfolio_daily_returns.csv.gz").read_bytes()
            write_unavailable_portfolio_outputs(root, "Full run required.")
            self.assertEqual(first, (root / "backtest_portfolio_daily_returns.csv.gz").read_bytes())
            self.assertEqual(PORTFOLIO_MODEL_NAME, "canonical_equal_weight_active_v1")


if __name__ == "__main__":
    unittest.main()
