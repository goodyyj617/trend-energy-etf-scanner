import numpy as np
import pandas as pd
import pytest

from lab.data import panel_from_long
from lab.engine import StrategyConfig, run_backtest
from lab.grid import Gates, _neighbors_and_regions, check_gates, loyo
from lab.metrics import curve_metrics, trade_metrics


def make_panel(prices: dict[str, list[float]], start="2020-01-01", opens: dict[str, list[float]] | None = None):
    dates = pd.bdate_range(start, periods=len(next(iter(prices.values()))))
    rows = []
    for sym, closes in prices.items():
        op = (opens or {}).get(sym, closes)
        for d, c, o in zip(dates, closes, op):
            rows.append({"date": d, "symbol": sym, "open": o, "high": max(o, c), "low": min(o, c),
                         "close": c, "volume": 1_000_000})
    return panel_from_long(pd.DataFrame(rows)), dates


def cfg(**kw):
    base = dict(entries={"breakout": 5}, exits={"low_break": 5}, start="2020-01-01", end="2030-01-01",
                max_positions=1, cost_bps=0.0, min_price=0.0, min_dollar_volume=0.0)
    base.update(kw)
    return StrategyConfig(**base)


def test_signal_at_close_executes_next_open():
    flat = [10.0] * 30
    closes = flat[:25] + [12.0, 12.5, 13.0, 13.5, 14.0]  # breakout on day index 25
    opens = flat[:26] + [12.2, 12.6, 13.1, 13.6]
    panel, dates = make_panel({"AAA": closes, "SPY": flat}, opens={"AAA": opens})
    res = run_backtest(panel, cfg())
    assert len(res.open_positions) == 1
    pos = res.open_positions.iloc[0]
    assert pos["entry_date"] == dates[26]  # day after the signal
    assert pos["entry_price"] == pytest.approx(12.2)  # next day's open, not the signal close


def test_round_trip_cost_and_exit_reason():
    closes = [10.0] * 25 + [12.0, 13.0, 14.0, 9.0, 9.0, 9.0]
    panel, dates = make_panel({"AAA": closes, "SPY": [10.0] * len(closes)})
    res = run_backtest(panel, cfg(cost_bps=10.0))
    t = res.trades.iloc[0]
    assert t["entry_date"] == dates[26] and t["exit_date"] == dates[29]
    c = 0.001
    assert t["return"] == pytest.approx(9.0 * (1 - c) / (13.0 * (1 + c)) - 1)


def test_trailing_stop_uses_peak_close_since_entry():
    closes = [10.0] * 25 + [12.0, 20.0, 18.5, 17.0, 17.0, 17.0]
    panel, dates = make_panel({"AAA": closes, "SPY": [10.0] * len(closes)})
    res = run_backtest(panel, cfg(exits={"trailing_pct": 10}))
    t = res.trades.iloc[0]
    # peak close 20 -> stop below 18; 18.5 holds, 17 triggers on dates[28] -> exit at dates[29] open
    assert t["exit_date"] == dates[29]


def test_benchmark_is_data_only_after_subset():
    closes = [10.0 + i * 0.1 for i in range(60)]
    panel, _ = make_panel({"AAA": [10.0] * 60, "SPY": closes})
    sub = panel.subset(["AAA"])
    assert "SPY" in sub.symbols and sub.tradable_symbols == ["AAA"]
    res = run_backtest(sub, cfg())
    assert "SPY" not in set(res.trades["symbol"]) | set(res.open_positions["symbol"])


def test_hold_forever_matches_benchmark():
    rng = np.random.default_rng(0)
    spy = list(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)))
    panel, _ = make_panel({"SPY": spy})
    res = run_backtest(panel, cfg(entries={"momentum": 21}, exits={"stop_loss_pct": 99}, start="2020-03-02"))
    # bought at the open after the first signal and never sold
    assert len(res.trades) == 0 and len(res.open_positions) == 1


def test_curve_and_trade_metrics():
    eq = pd.Series([1.0, 1.2, 0.9, 1.5], index=pd.bdate_range("2020-01-01", periods=4))
    m = curve_metrics(eq)
    assert m["mdd"] == pytest.approx(0.9 / 1.2 - 1)
    tm = trade_metrics(pd.DataFrame({"return": [0.1, -0.05, 0.2], "holding_days": [5, 5, 5]}), 252)
    assert tm["profit_factor"] == pytest.approx(0.3 / 0.05)
    assert tm["win_rate"] == pytest.approx(2 / 3)


def test_neighbors_and_regions():
    # 3x3 grid, passing cells: an L-shaped region of 3 plus an isolated corner
    passing = {(0, 0), (1, 0), (0, 1), (2, 2)}
    cells = pd.DataFrame([{"xi": x, "yi": y, "pass": (x, y) in passing} for y in range(3) for x in range(3)])
    _neighbors_and_regions(cells)
    get = lambda x, y: cells[(cells.xi == x) & (cells.yi == y)].iloc[0]  # noqa: E731
    assert get(0, 0)["region_size"] == 3
    assert get(2, 2)["region_size"] == 1
    assert get(1, 1)["region_size"] == 0
    assert get(0, 0)["neighbor_survival"] == pytest.approx(1.0)  # both neighbours pass
    assert get(2, 2)["neighbor_survival"] == pytest.approx(0.0)


def test_gates_and_loyo():
    m = {"cagr": 0.10, "spy_cagr": 0.12, "mdd": -0.20, "spy_mdd": -0.30, "n_trades": 40}
    assert check_gates(m, Gates()) == []
    assert check_gates({**m, "n_trades": 10}, Gates()) == ["최소 거래 수"]
    idx = pd.bdate_range("2018-01-01", "2020-12-31")
    eq = pd.Series(np.linspace(1, 2, len(idx)), index=idx)
    ratio, failed = loyo(eq, eq.copy(), Gates())
    assert ratio == 1.0 and failed == []
