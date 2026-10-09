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


# --------------------------------------------------------------------------- phase 1: test bench

from lab import research_log  # noqa: E402
from lab.blocks import rsi  # noqa: E402
from lab.data import cash_from_closes  # noqa: E402
from lab.engine import baselines  # noqa: E402
from lab.grid import strategy_checklist  # noqa: E402
from lab.metrics import deflated_sharpe  # noqa: E402
from lab.universe import UniverseFilter, apply_filter, asset_class, near_duplicates  # noqa: E402


def test_rsi_extremes_and_balance():
    up = pd.DataFrame({"A": np.arange(1.0, 40.0)})
    assert rsi(up).iloc[-1, 0] == pytest.approx(100.0)
    zigzag = pd.DataFrame({"A": [10.0 + (i % 2) for i in range(200)]})
    assert rsi(zigzag).iloc[-1, 0] == pytest.approx(50.0, abs=3)


def test_fixed_rule_blocks_have_no_tunable_parameter():
    flat = [10.0] * 260
    panel, _ = make_panel({"AAA": flat, "SPY": flat})
    c = cfg(entries={"ma_stack": None, "up_candle": None}, exits={"low_break": 10})
    assert c.tunable() == ["low_break"]
    assert "None" not in c.describe()
    run_backtest(panel, c)  # runs without a parameter value


def test_atr_trailing_exit():
    closes = [10.0] * 25 + [12.0, 14.0, 16.0, 16.0, 13.0, 13.0, 13.0]
    panel, dates = make_panel({"AAA": closes, "SPY": [10.0] * len(closes)})
    res = run_backtest(panel, cfg(exits={"atr_trail": 1.0}))
    # ATR(14) is ~1 here; close 13 < peak 16 - 1*ATR -> signal on dates[29], exit at the next open
    assert res.trades.iloc[0]["exit_date"] == dates[30]


def test_idle_cash_earns_cash_return():
    flat = [10.0] * 60
    panel, _ = make_panel({"AAA": flat, "SPY": flat})
    panel.cash = pd.Series(0.0001, index=panel.close.index)
    res = run_backtest(panel, cfg())  # flat prices never break out -> always in cash
    assert res.equity.iloc[-1] == pytest.approx(1.0001 ** len(res.equity))
    off = run_backtest(panel, cfg(cash_yield=False))
    assert off.equity.iloc[-1] == pytest.approx(1.0)


def test_cash_series_splices_tbill_rate_before_bil():
    idx = pd.bdate_range("2007-01-01", periods=6)
    irx = pd.Series(5.04, index=idx)
    bil = pd.Series([np.nan, np.nan, np.nan, 100.0, 100.1, 100.2], index=idx)
    out = cash_from_closes(idx, bil, irx)
    assert out.iloc[1] == pytest.approx(0.0504 / 252)
    assert out.iloc[4] == pytest.approx(0.001)


def test_asset_class_mapping():
    assert asset_class("Ultrashort Bond") == "현금성"
    assert asset_class("Long Government") == "채권"
    assert asset_class("Target Maturity") == "채권"
    assert asset_class("Focused Region") == "해외 주식"
    assert asset_class("Equity Precious Metals") == "섹터·테마 주식"
    assert asset_class("Commodities Focused") == "원자재"
    assert asset_class("Large Blend") == "미국 주식"
    assert asset_class("Digital Assets") == "기타"
    assert asset_class("", "Bond") == "채권"


def test_dedup_keeps_first_of_near_identical_pair():
    rng = np.random.default_rng(1)
    a = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
    close = pd.DataFrame({"BIG": a, "CLONE": a * 2.0, "OTHER": 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))},
                         index=pd.bdate_range("2020-01-01", periods=400))
    assert near_duplicates(close, ["BIG", "CLONE", "OTHER"]) == {"CLONE": "BIG"}


def test_filter_funnel_excludes_cash_like_and_expensive():
    table = pd.DataFrame([
        {"symbol": "SPY", "name": "", "category": "Large Blend", "asset_class": "미국 주식", "aum": 5e11, "expense_ratio": 0.09},
        {"symbol": "BIL", "name": "", "category": "Ultrashort Bond", "asset_class": "현금성", "aum": 5e10, "expense_ratio": 0.13},
        {"symbol": "PRICY", "name": "", "category": "Technology", "asset_class": "섹터·테마 주식", "aum": 5e9, "expense_ratio": 1.2},
        {"symbol": "TINY", "name": "", "category": "Technology", "asset_class": "섹터·테마 주식", "aum": 1e8, "expense_ratio": 0.3},
    ])
    out, funnel = apply_filter(table, UniverseFilter(dedup=False))
    assert list(out.loc[out["status"] == "선택", "symbol"]) == ["SPY"]
    assert [n for _, n in funnel] == [4, 3, 3, 2, 1]


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(2)
    eq = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, 1000)), index=pd.bdate_range("2015-01-01", periods=1000))
    vals = [deflated_sharpe(eq, n, 0.0004) for n in (1, 10, 100)]
    assert vals[0] > vals[1] > vals[2]


def test_trial_log_dedupes_and_holdout_records(tmp_path, monkeypatch):
    monkeypatch.setattr(research_log, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(research_log, "TRIALS_PATH", tmp_path / "trials.csv")
    monkeypatch.setattr(research_log, "HOLDOUT_PATH", tmp_path / "holdout.json")
    eq = pd.Series(np.linspace(1, 1.5, 300), index=pd.bdate_range("2020-01-01", periods=300))
    research_log.log_trials([("backtest", cfg(), ["AAA"], eq), ("backtest", cfg(), ["AAA"], eq),
                             ("grid", cfg(max_positions=2), ["AAA"], eq)])
    assert research_log.trial_stats()[0] == 2
    state = research_log.set_holdout(True, "2023-01-01")
    assert state["start"] == "2023-01-01" and len(state["changes"]) == 1
    state = research_log.record_holdout_evaluation("x", cfg(), {"cagr": 0.1, "spy_cagr": 0.12, "spy_sharpe": 1.0}, ("a", "b"))
    assert len(state["evaluations"]) == 1 and "spy_sharpe" not in state["evaluations"][0]["metrics"]


def test_checklist_and_baselines_run():
    rng = np.random.default_rng(3)
    n = 700
    prices = {s: list(50 * np.cumprod(1 + rng.normal(0.0006, 0.012, n))) for s in ("AAA", "BBB", "SPY")}
    panel, _ = make_panel(prices)
    c = cfg(entries={"breakout": 20}, exits={"low_break": 10}, max_positions=2)
    res = run_backtest(panel, c)
    base = baselines(panel, c, res.equity.index)
    assert set(base) == {"spy", "spy_ma200", "equal_weight"}
    items = strategy_checklist(panel, c, Gates(), res, base["spy_ma200"], 0.5)
    assert [i["key"] for i in items] == ["gate", "neighbors", "cost", "loyo", "halves", "simple", "dsr"]


# --------------------------------------------------------------------------- strategy-family comparison

from lab import families, store  # noqa: E402
from lab.blocks import ENTRY_BLOCKS, EXIT_BLOCKS  # noqa: E402
from lab.engine import SignalCache  # noqa: E402
from lab.grid import run_grid  # noqa: E402


def _random_panel(n=600, seed=4):
    rng = np.random.default_rng(seed)
    prices = {s: list(50 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))) for s in ("AAA", "BBB", "CCC", "SPY")}
    return make_panel(prices)[0]


def test_menu_values_are_inside_block_bounds():
    for menu, blocks in ((families.ENTRY_MENU, ENTRY_BLOCKS), (families.EXIT_MENU, EXIT_BLOCKS)):
        for key, vals in menu.items():
            p = blocks[key].param
            assert len(vals) == 6 and vals == sorted(vals)
            assert all(p.minimum <= v <= p.maximum for v in vals)
    assert families.cell_count() == 9 * 36


def test_shared_cache_gives_identical_grid():
    panel = _random_panel()
    c = cfg(entries={"breakout": 20}, exits={"low_break": 10}, max_positions=2)
    fresh = run_grid(panel, c, "breakout", [10, 20], "low_break", [5, 10], Gates())
    shared = SignalCache(panel)
    run_grid(panel, c, "breakout", [20, 30], "low_break", [10, 15], Gates(), cache=shared)  # warm a different window use
    again = run_grid(panel, c, "breakout", [10, 20], "low_break", [5, 10], Gates(), cache=shared)
    pd.testing.assert_series_equal(fresh.cells["cagr"], again.cells["cagr"])
    pd.testing.assert_series_equal(fresh.cells["n_trades"], again.cells["n_trades"])


def test_family_comparison_summary_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr(families, "ENTRY_MENU", {"breakout": [10, 20], "momentum": [21, 42]})
    monkeypatch.setattr(families, "EXIT_MENU", {"low_break": [5, 10], "atr_trail": [2.0, 3.0]})
    panel = _random_panel()
    seen = []
    fc = families.run_family_comparison(panel, cfg(max_positions=2), Gates(min_cagr_ratio=0.0, max_mdd_ratio=2.0),
                                        on_result=lambda c, r: seen.append(c))
    assert len(seen) == 16 and len(fc.all_cells()) == 16
    s = fc.summary()
    assert list(s["순위"]) == [1, 2, 3, 4]
    keys = list(zip(s["pass_share"], s["largest_region"], s["beat_ma200"], s["median_calmar"]))
    assert keys == sorted(keys, reverse=True)  # lexicographic, no weights
    monkeypatch.setattr(store, "RESULTS_DIR", tmp_path)
    path = store.save_family(fc, "t", "test", ["AAA"])
    assert len(store.load_family_summary(path)) == 4 and len(store.load_cells(path)) == 16


# --------------------------------------------------------------------------- confirmation-condition test

from lab import refine  # noqa: E402


def test_verdict_is_pareto_on_ordering_metrics():
    base = {"pass_share": 0.2, "largest_region": 5, "beat_ma200": 0.1, "median_calmar": 0.15}
    assert refine.verdict(base, {**base, "median_calmar": 0.2}) == "improve"
    assert refine.verdict(base, {**base, "pass_share": 0.1}) == "worse"
    assert refine.verdict(base, {**base, "pass_share": 0.3, "median_calmar": 0.1}) == "mixed"
    assert refine.verdict(base, dict(base)) == "same"


def test_market_trend_block_follows_spy_only():
    n = 260
    spy = list(np.linspace(100, 200, n))  # steadily rising -> SPY above its MA once the MA exists
    panel, _ = make_panel({"AAA": list(np.linspace(200, 100, n)), "SPY": spy})
    f = ENTRY_BLOCKS["market_trend"].compute(panel, 200).fillna(False)
    assert not f.iloc[150].any() and f.iloc[-1].all()  # same value for every symbol, set by SPY alone


def test_refinement_runs_every_addon_on_the_family_grid(monkeypatch):
    monkeypatch.setattr(refine, "ENTRY_MENU", {"breakout": [10, 20]})
    monkeypatch.setattr(refine, "EXIT_MENU", {"low_break": [5, 10]})
    monkeypatch.setattr(refine, "addon_candidates", lambda e: ["up_candle", "rsi_min"])
    panel = _random_panel()
    seen = []
    ref = refine.run_refinement(panel, cfg(max_positions=2), "breakout", "low_break", Gates(),
                                on_result=lambda c, r: seen.append(c))
    assert len(seen) == 12
    assert {c.entries.get("rsi_min") for c in seen if "rsi_min" in c.entries} == {55}  # default value only
    t = ref.table()
    assert t.iloc[0]["verdict"] == "base" and set(t["addon"]) == {"", "up_candle", "rsi_min"}


def test_config_converts_numpy_numbers():
    c = cfg(entries={"breakout": np.int64(20), "momentum": np.float64(126.0)}, exits={"low_break": np.int64(10)})
    assert type(c.entries["breakout"]) is int and type(c.exits["low_break"]) is int
    assert c.fingerprint() == cfg(entries={"breakout": 20, "momentum": 126.0}, exits={"low_break": 10}).fingerprint()
