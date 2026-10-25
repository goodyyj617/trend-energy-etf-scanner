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
from lab.blocks import ALL_BLOCKS, ENTRY_BLOCKS, EXIT_BLOCKS  # noqa: E402
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
    assert families.cell_count() == 77 * 36
    # a family whose own block is the common add-on is skipped
    assert families.cell_count({"breakout": 20}) == 70 * 36
    assert families.cell_count(extra_exits={"stop_loss_pct": 10}) == 77 * 36


def test_ma_cross_uses_quarter_length_short_average():
    n = 300
    panel, _ = make_panel({"AAA": list(np.linspace(100, 200, n)), "SPY": list(np.linspace(200, 100, n))})
    f = ENTRY_BLOCKS["ma_cross"].compute(panel, 200)
    assert not f["AAA"].iloc[198]  # SMA200 not yet defined
    assert f["AAA"].iloc[-1] and not f["SPY"].iloc[-1]  # rising: SMA50 > SMA200; falling: below


def test_addon_test_adds_one_block_to_every_family(tmp_path, monkeypatch):
    monkeypatch.setattr(families, "ENTRY_MENU", {"breakout": [10, 20], "momentum": [21, 42]})
    monkeypatch.setattr(families, "EXIT_MENU", {"low_break": [5, 10], "atr_trail": [2.0, 3.0]})
    panel = _random_panel()
    seen = []
    at = families.run_addon_test(panel, cfg(max_positions=2), Gates(), "exit", "stop_loss_pct",
                                 on_result=lambda c, r: seen.append(c))
    assert len(seen) == 32  # 16 plain + 16 with the stop
    assert {c.exits.get("stop_loss_pct") for c in seen} == {None, 10}  # default value only
    t = at.table()
    assert len(t) == 4 and set(t["verdict"]) <= {"improve", "mixed", "same", "worse"}
    assert sum(at.counts().values()) == 4
    # an entry add-on skips the family built on that same block
    at2 = families.run_addon_test(panel, cfg(max_positions=2), Gates(), "entry", "momentum", plain=at.plain)
    assert set(at2.table()["entry"]) == {"breakout"}
    monkeypatch.setattr(store, "RESULTS_DIR", tmp_path)
    path = store.save_addon(at, "t", "test", ["AAA"])
    assert len(store.load_addon_table(path)) == 4 and len(store.load_cells(path)) == 32


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


# --------------------------------------------------------------------------- effective trials, sizing

def test_effective_trials_shrinks_for_correlated_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(research_log, "RESULTS_DIR", tmp_path)
    research_log._EFF_CACHE.clear()
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    common = rng.normal(0.0004, 0.01, len(idx))
    rows = []
    for k in range(4):  # four near-copies of one strategy
        eq = pd.Series(np.cumprod(1 + common + rng.normal(0, 0.001, len(idx))), index=idx)
        rows.append(("grid", cfg(max_positions=k + 1), ["AAA"], eq))
    research_log.log_trials(rows)
    n_eff, rho, covered = research_log.effective_trials()
    assert covered == 4 and rho > 0.9 and 1.0 <= n_eff < 1.5


def test_inverse_vol_sizing_allocates_by_volatility():
    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for sym, a, vol in (("CALM", 0.002, 3e6), ("MID", 0.01, 2e6), ("WILD", 0.03, 1e6), ("SPY", 0.01, 1e6)):
        r = np.array([a if i % 2 else -a for i in range(n)]) + 0.002  # std of daily returns ~= a, steady uptrend
        close = 100 * np.cumprod(1 + r)
        for d, c in zip(dates, close):
            rows.append({"date": d, "symbol": sym, "open": c, "high": c, "low": c, "close": c, "volume": vol})
    panel = panel_from_long(pd.DataFrame(rows)).subset(["CALM", "MID", "WILD"])
    c = cfg(entries={"momentum": 21}, exits={"stop_loss_pct": 99}, max_positions=3, start=str(dates[70].date()),
            sizing="inverse_vol")
    res = run_backtest(panel, c)
    pos = res.open_positions.set_index("symbol")
    assert (pos["entry_date"] == res.equity.index[1]).all()  # all signalled on the first day of the window
    equal = run_backtest(panel, replace_cfg(c, sizing="equal")).open_positions
    assert set(equal["symbol"]) == {"CALM", "MID", "WILD"}  # equal sizing: three 1/3 slots
    # inverse vol: CALM is capped at 2 slots (2/3), MID is the median volatility (1/3) -> no cash left for WILD
    assert set(pos.index) == {"CALM", "MID"}
    assert "sizing" not in cfg().to_dict()  # default omitted -> earlier trial keys unchanged


def replace_cfg(c, **kw):
    from dataclasses import replace
    return replace(c, **kw)


def test_strategy_fingerprint_ignores_dates():
    a = cfg(start="2000-01-01", end="2023-12-31")
    b = cfg(start="2024-01-01", end="2026-10-08")
    assert a.fingerprint() != b.fingerprint() and a.strategy_fingerprint() == b.strategy_fingerprint()
    assert a.strategy_fingerprint() != cfg(max_positions=3).strategy_fingerprint()


# --------------------------------------------------------------------------- macro regime blocks

def _fake_macro(dates, **cols):
    from lab import macro

    macro.use(pd.DataFrame(cols, index=dates))
    return macro


def test_macro_value_uses_previous_day_only():
    from lab import macro

    days = pd.bdate_range("2020-01-01", periods=6)
    raw = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=days)
    got = macro.on_dates(raw, days)
    assert np.isnan(got.iloc[0]) and got.iloc[1:].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    # a FRED holiday gap is filled with the last value; a long gap becomes missing
    sparse = raw.iloc[[0]]
    late = macro.on_dates(sparse, pd.bdate_range("2020-01-01", periods=15))
    assert late.iloc[1] == 1.0 and np.isnan(late.iloc[-1])


def test_macro_entry_and_exit_blocks_are_complements():
    from lab.blocks import REGIME_PAIRS

    panel = _random_panel(n=300)
    idx = panel.close.index
    rng = np.random.default_rng(1)
    _fake_macro(idx, T10Y3M=rng.normal(0.2, 1.0, len(idx)), BAA10Y=2 + np.cumsum(rng.normal(0, 0.05, len(idx))),
                VIXCLS=rng.uniform(10, 40, len(idx)))
    try:
        for ent, ext in REGIME_PAIRS.items():
            v = ENTRY_BLOCKS[ent].param.default
            if ent == "credit_calm":
                v = 50  # enough history in a short test panel
            a = ENTRY_BLOCKS[ent].compute(panel, v)
            b = EXIT_BLOCKS[ext].compute(panel, v)
            known = a | b
            assert known.iloc[-1].all()  # every day with data is either on or off
            assert not (a & b).any().any()
            assert (a.nunique(axis=1) == 1).all()  # market-wide: same for every symbol
        f = ENTRY_BLOCKS["vix_calm"].compute(panel, 25)
        from lab import macro
        prev = macro.data()["VIXCLS"].shift(1)
        assert (f["AAA"].iloc[1:] == (prev.iloc[1:] < 25)).all()
    finally:
        from lab import macro
        macro.use(None)


def test_regime_switch_addon_adds_filter_and_complement_exit(monkeypatch):
    monkeypatch.setattr(families, "ENTRY_MENU", {"breakout": [10, 20]})
    monkeypatch.setattr(families, "EXIT_MENU", {"low_break": [5, 10]})
    panel = _random_panel()
    seen = []
    at = families.run_addon_test(panel, cfg(max_positions=2), Gates(), "switch", "market_trend",
                                 on_result=lambda c, r: seen.append(c))
    added = [c for c in seen if "market_trend" in c.entries]
    assert len(added) == 4 and all(c.exits.get("market_off") == 200 for c in added)
    assert families.addon_extras("switch", "vix_calm") == ({"vix_calm": 25.0}, {"vix_high": 25.0})
    assert "국면 전환" in at.label and len(at.table()) == 1


def test_daily_alert_explains_regimes_and_flags_flips():
    from lab import daily_alert, macro

    panel = _random_panel(n=400)
    idx = panel.close.index
    vix = np.full(len(idx), 15.0)
    vix[-2] = 30.0  # VIX known for the last day is 30 (one-day lag) -> regime off; the day before it was on
    _fake_macro(idx, T10Y3M=np.full(len(idx), 1.0), BAA10Y=np.full(len(idx), 2.0), VIXCLS=vix)
    try:
        regimes = daily_alert.regime_rows(panel)
        rows = {r.key: r for r in regimes}
        assert [r.key for r in regimes] == ["market_trend", "curve_ok", "credit_calm", "vix_calm"]
        assert rows["vix_calm"].on_today is False and rows["vix_calm"].flipped
        assert rows["curve_ok"].on_today is True and not rows["curve_ok"].flipped
        assert rows["vix_calm"].since.endswith("1거래일") and rows["curve_ok"].since.endswith("거래일")
        assert all(r.context for r in regimes)  # every regime gets a plain-language reading
        as_of = str(idx[-1].date())
        text = daily_alert.render(regimes, as_of, "someone", dropped_rows=1)
        assert as_of in text and "VIX 낮음" in text and "⚠️ 오늘 바뀜" in text
        assert "지금 상황 읽기" in text and "지표 해설" in text and "마지막 1행" in text
        assert text.endswith("cc @someone")
        assert daily_alert.has_news(regimes)
        plain = daily_alert.plain_text(text)
        assert "**" not in plain and "<details>" not in plain and "|---" not in plain
    finally:
        macro.use(None)


def test_normalize_dates_maps_intraday_and_tz_aware_stamps_to_the_trading_day():
    from lab.data import normalize_dates

    cases = [
        pd.Series(["2026-10-22", "2026-10-23 16:00:02"]),  # yfinance's half-built current-day bar
        pd.Series(["2026-10-22 00:00:00-04:00", "2026-10-23 15:59:00-04:00"]),  # tz-aware
        pd.Series(["2026-10-22 04:00:00+00:00", "2026-10-23 20:00:00+00:00"]),  # UTC stamps
    ]
    for raw in cases:
        out = normalize_dates(raw)
        assert out.dt.tz is None
        assert list(out) == [pd.Timestamp("2026-10-22"), pd.Timestamp("2026-10-23")], raw.tolist()


def test_trim_incomplete_tail_drops_rows_without_a_benchmark_close():
    from lab import daily_alert

    up = list(np.linspace(10, 20, 30))
    panel, _ = make_panel({"AAA": up, "SPY": up})
    same, n = daily_alert.trim_incomplete_tail(panel)
    assert n == 0 and same.close.index.equals(panel.close.index)
    broken = up[:-2] + [np.nan, np.nan]
    panel, _ = make_panel({"AAA": up, "SPY": broken})
    cut, n = daily_alert.trim_incomplete_tail(panel)
    assert n == 2 and len(cut.close) == 28 and cut.close["SPY"].notna().all()
    assert cut.tradable == panel.tradable and cut.cash is None  # cash series is cut the same way when present


def test_pending_orders_are_the_last_close_decisions():
    up = list(np.linspace(10, 20, 30))
    panel, _ = make_panel({"AAA": up, "SPY": up})
    r = run_backtest(panel, cfg(entries={"breakout": 5}, exits={"low_break": 5}))
    assert r.pending_orders == {"buy": [], "sell": {}}  # holding AAA, no exit
    down = up[:-1] + [5.0]
    panel, _ = make_panel({"AAA": down, "SPY": up})
    r = run_backtest(panel, cfg(entries={"breakout": 5}, exits={"low_break": 5}))
    assert r.pending_orders["sell"] == {"AAA": "low_break"}


# --------------------------------------------------------------------------- menu v3 blocks and the exploration gate

from lab import blocks as lab_blocks  # noqa: E402
from lab.grid import EXPLORE_GATES, gate_caption  # noqa: E402


def test_exploration_gate_uses_calmar_instead_of_return():
    # half of SPY's return with a third of its drawdown: fails the final gate, passes the exploration gate
    m = {"cagr": 0.05, "mdd": -0.10, "spy_cagr": 0.10, "spy_mdd": -0.30, "n_trades": 40}
    assert check_gates(m, Gates()) == ["수익 유지"]
    assert check_gates(m, EXPLORE_GATES) == []
    worse = {**m, "cagr": 0.03, "mdd": -0.14}  # Calmar 0.21 < SPY 0.33
    assert check_gates(worse, EXPLORE_GATES) == ["위험 대비 수익 (탐색용)"]
    assert "min_calmar_ratio" not in Gates().to_dict()  # saved form of the final gate is unchanged
    assert "칼마 ≥ 1.00×SPY 칼마" in gate_caption(EXPLORE_GATES.to_dict())
    assert "CAGR" not in gate_caption(EXPLORE_GATES.to_dict())
    deep = {**m, "cagr": 0.09, "mdd": -0.18, "spy_mdd": -0.50}  # passes everything but the 15% limit
    assert check_gates(deep, EXPLORE_GATES) == ["낙폭 한도"]
    assert check_gates(deep, Gates()) == []
    assert "|MDD| ≤ 15%" in gate_caption(EXPLORE_GATES.to_dict())


def test_cmf_and_adx_directions():
    n = 80
    up = list(np.linspace(100, 160, n))
    down = list(np.linspace(160, 100, n))
    # closes at the top of the day's range on the way up, at the bottom on the way down
    panel, _ = make_panel({"UP": up, "DN": down, "SPY": up}, opens={"UP": [c - 1 for c in up], "DN": [c + 1 for c in down],
                                                                     "SPY": [c - 1 for c in up]})
    c = lab_blocks.cmf(panel, 20).iloc[-1]
    assert c["UP"] == pytest.approx(1.0) and c["DN"] == pytest.approx(-1.0)
    adx, plus_di, minus_di = lab_blocks.adx_di(panel, 14)
    assert adx.iloc[-1]["UP"] > 25 and plus_di.iloc[-1]["UP"] > minus_di.iloc[-1]["UP"]
    assert ENTRY_BLOCKS["adx_trend"].compute(panel, 25).iloc[-1].to_dict() == {"UP": True, "DN": False, "SPY": True}
    assert EXIT_BLOCKS["momentum_neg"].compute(panel, 21).iloc[-1].to_dict() == {"UP": False, "DN": True, "SPY": False}
    assert ENTRY_BLOCKS["near_high"].compute(panel, 5).iloc[-1].isna().sum() == 0  # < 252 days -> False, not NaN
# --------------------------------------------------------------------------- performance paths (results unchanged)

def test_exit_found_at_entry_matches_day_by_day_rule():
    """Every closed trade exits on the first close after entry where a selected exit fires
    (checked here day by day, in cfg.exits order), with gaps in the price data."""
    rng = np.random.default_rng(11)
    n = 700
    prices = {s: list(50 * np.cumprod(1 + rng.normal(0.0006, 0.018, n))) for s in ("AAA", "BBB", "CCC", "DDD", "SPY")}
    prices["CCC"][300:310] = [np.nan] * 10  # days without a bar
    panel = make_panel(prices)[0]
    c = cfg(entries={"breakout": 10}, exits={"trailing_pct": 8.0, "low_break": 15, "stop_loss_pct": 5.0, "atr_trail": 2.5},
            max_positions=3)
    res = run_backtest(panel, c)
    assert len(res.trades) > 20
    close, low_hit = panel.close, ALL_BLOCKS["low_break"].compute(panel, 15).fillna(False)
    atr14 = SignalCache(panel).atr14
    dates = list(close.index)
    for t in res.trades.itertuples():
        e, x = dates.index(t.entry_date), dates.index(t.exit_date)
        peak, signal_day, reason = t.entry_price, None, None
        for d in range(e, x):
            cj = close[t.symbol].iloc[d]
            if cj != cj:
                continue
            peak = max(peak, cj)
            a = atr14[t.symbol].iloc[d]
            hits = {"trailing_pct": cj < peak * 0.92, "low_break": bool(low_hit[t.symbol].iloc[d]),
                    "stop_loss_pct": cj < t.entry_price * 0.95, "atr_trail": a == a and cj < peak - 2.5 * a}
            fired = [k for k in c.exits if hits[k]]
            if fired:
                signal_day, reason = d, fired[0]
                break
        assert signal_day is not None
        assert all(close[t.symbol].iloc[signal_day + 1:x].isna())  # sold at the next available open
        assert t.exit_reason == EXIT_BLOCKS[reason].label


def test_parallel_family_comparison_is_identical(monkeypatch):
    monkeypatch.setattr(families, "ENTRY_MENU", {"breakout": [10, 20], "momentum": [21, 42]})
    monkeypatch.setattr(families, "EXIT_MENU", {"low_break": [5, 10], "trailing_pct": [5, 10]})
    panel = _random_panel()
    runs = {}
    for w in (1, 2):
        seen = []
        fc = families.run_family_comparison(panel, cfg(max_positions=2), Gates(), workers=w,
                                            on_result=lambda c, r: seen.append((c, r.equity)))
        runs[w] = (fc, seen)
    (one, seen1), (two, seen2) = runs[1], runs[2]
    pd.testing.assert_frame_equal(one.all_cells(), two.all_cells())
    pd.testing.assert_frame_equal(one.summary(), two.summary())
    assert [c for c, _ in seen1] == [c for c, _ in seen2]  # same trial order for the log
    for (_, a), (_, b) in zip(seen1, seen2):
        pd.testing.assert_series_equal(a, b)


def test_mean_pairwise_corr_matches_pandas():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.03, (120, 9)) + rng.normal(0, 0.03, (120, 1))
    x[:40, 5:] = np.nan  # a second date range
    x[:, 2] = 0.0  # constant (never traded): no correlation defined
    corr = pd.DataFrame(x).corr(min_periods=research_log.MIN_OVERLAP_MONTHS).to_numpy()
    off = corr[~np.eye(len(corr), dtype=bool)]
    assert research_log.mean_pairwise_corr(x) == pytest.approx(np.nanmean(off), rel=1e-12)


# --------------------------------------------------------------------------- drawdown-budget target (DECISIONS 64-65)

from lab import target  # noqa: E402
from lab.metrics import DEFINITIONS  # noqa: E402


def test_fit_weight_scales_down_only():
    # one 40% crash: w x 40% must stay within 15% -> w = 0.37 (0.37 x 0.4 = 14.8%)
    rets = np.array([0.01] * 50 + [-0.40] + [0.01] * 50)
    cash = np.zeros_like(rets)
    w = target.fit_weight(rets, cash)
    assert w == pytest.approx(0.37)
    assert target._mdd(target.blend(rets, cash, w)) >= -0.15
    rng = np.random.default_rng(8)
    for _ in range(30):  # bisection agrees with checking every weight
        r = rng.normal(0.0005, rng.uniform(0.005, 0.03), 800)
        c = np.full(800, 0.0001)
        assert target.fit_weight(r, c) == target.fit_weight_grid(r, c)
    calm = np.full(100, 0.001)  # never falls: stays at 100%, no leverage
    assert target.fit_weight(calm, np.zeros(100)) == 1.0


def test_target_report_and_windows():
    idx = pd.bdate_range("2010-01-01", "2016-12-31")
    rng = np.random.default_rng(5)
    spy = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.012, len(idx))), idx)
    strat = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.004, len(idx))), idx)
    cash = pd.Series(0.0, idx)
    assert target.windows(idx) == [(2010, 2012), (2011, 2013), (2012, 2014), (2013, 2015), (2014, 2016)]
    rep = target.target_report(strat, spy, cash, n_trades=40)
    assert rep["windows"] == 5 and 0 <= rep["window_share"] <= 1
    assert rep["dd_mdd"] >= -0.15 - 1e-9 and 0 < rep["dd_weight"] <= 1
    assert rep["dd_pass"] == (rep["dd_cagr"] >= rep["mix_cagr"])
    assert {"dd_weight", "dd_cagr", "mix_cagr", "dd_excess", "window_share"} <= set(DEFINITIONS)


# --------------------------------------------------------------------------- Sobol search and PBO (DECISIONS 66)

from lab import explore  # noqa: E402


def test_sobol_known_points_and_balance():
    pts = explore.sobol(8, 2)
    assert pts[:, 0].tolist() == [0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125]
    assert pts[:, 1].tolist() == [0, 0.5, 0.25, 0.75, 0.375, 0.875, 0.125, 0.625]
    big = explore.sobol(1024, 10)
    for d in range(10):  # each coordinate hits each of the 1024 equal cells exactly once
        assert len(set((big[:, d] * 1024).astype(int))) == 1024
    pairs = (big[:256, :2] * 16).astype(int)  # 2-D stratification of the first 256 points
    assert len({(a, b) for a, b in pairs}) == 256


def test_point_config_stays_in_menu_ranges():
    base = cfg()
    for u in explore.sobol(512, 8):
        c = explore.point_config(u, base)
        e = next(k for k in c.entries if k in explore.ENTRIES)
        x = next(k for k in c.exits if k in explore.EXITS)
        assert min(families.ENTRY_MENU[e]) <= c.entries[e] <= max(families.ENTRY_MENU[e])
        assert min(families.EXIT_MENU[x]) <= c.exits[x] <= max(families.EXIT_MENU[x])
        assert 5 <= c.max_positions <= 20 and c.sizing in ("equal", "inverse_vol")


def test_pbo_detects_noise_and_skill():
    rng = np.random.default_rng(9)
    noise = pd.DataFrame(rng.normal(0.005, 0.04, (240, 60)))
    assert explore.pbo(noise, blocks=10)["pbo"] > 0.4  # pure luck: at least a coin flip
    skill = noise.copy()
    skill[0] = rng.normal(0.03, 0.02, 240)  # one column truly better
    assert explore.pbo(skill, blocks=10)["pbo"] < 0.05


# --------------------------------------------------------------------------- strategy-only test (DECISIONS 67)

from dataclasses import replace  # noqa: E402

from lab import single  # noqa: E402


def test_single_mode_matches_trading_each_symbol_alone():
    rng = np.random.default_rng(21)
    n = 900
    prices = {s: list(40 * np.cumprod(1 + rng.normal(0.0007, 0.02, n))) for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "SPY")}
    panel = make_panel(prices)[0]
    base = cfg(entries={"breakout": 20}, exits={"trailing_pct": 8.0, "low_break": 30}, cost_bps=10.0)
    scfg = single.single_config(base, panel)
    res = run_backtest(panel, scfg)
    rets = single.symbol_returns(panel, scfg, res)
    for s in panel.tradable_symbols:
        alone = run_backtest(panel.subset([s]), replace(base, max_positions=1))
        mine = res.trades[res.trades.symbol == s].reset_index(drop=True)
        pd.testing.assert_frame_equal(mine[["entry_date", "exit_date", "exit_reason"]],
                                      alone.trades[["entry_date", "exit_date", "exit_reason"]].reset_index(drop=True))
        # rebuilt stand-alone account vs the engine's one-symbol account (cash here earns 0)
        eq = (1 + rets[s].fillna(0.0)).cumprod()
        assert eq.iloc[-1] == pytest.approx(alone.equity.iloc[-1] / alone.equity.iloc[0], rel=1e-9)
    out = single.evaluate(panel, base)
    assert out["symbols"] == len(panel.tradable_symbols)  # 900 days >= 3 years each
    assert out["n_trades"] == len(res.trades)


def test_save_study_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RESULTS_DIR", tmp_path)
    df = pd.DataFrame({"계열": ["a", "b"], "통과 비율": [0.5, 0.25]})
    idx = pd.bdate_range("2020-01-01", periods=5)
    raw = pd.DataFrame({"전략 A": np.linspace(1, 1.2, 5), "SPY": np.linspace(1, 1.1, 5)}, index=idx)
    store.save_study("연구", "설명", [{"title": "표", "frame": df, "pct": ["통과 비율"]}], ["메모"], "64", "docs/x.md",
                     curves=[{"group": "표본", "raw": raw, "baselines": ["SPY"]}])
    (meta,) = store.list_results()
    assert meta["kind"] == "study" and meta["tables"][0]["pct"] == ["통과 비율"]
    pd.testing.assert_frame_equal(store.load_study_table(meta["path"], meta["tables"][0]["file"]), df)
    (g,) = meta["curves"]
    assert "scaled" not in g
    pd.testing.assert_frame_equal(store.load_study_curves(meta["path"], g["raw"]), raw, check_freq=False)


# --------------------------------------------------------------------------- CRSP panel and point-in-time universe (DECISIONS 70)

from lab import crsp  # noqa: E402
from lab.engine import DELIST_LABEL  # noqa: E402


def _crsp_rows(permno, dates, p0, r, shrout, cfacpr_before_split=None, split_day=None, delist_code=None, dlret=np.nan,
               shrcd="11", exchcd="1", ticker="X"):
    rows = []
    for t, d in enumerate(dates):
        adj = p0 * (1 + r) ** t
        if split_day is not None and t < split_day:
            prc, cf = adj * cfacpr_before_split, cfacpr_before_split
        else:
            prc, cf = adj, 1.0
        last = t == len(dates) - 1
        rows.append({"PERMNO": permno, "date": d, "SHRCD": shrcd, "EXCHCD": exchcd, "TICKER": ticker, "PRC": prc,
                     "VOL": 1000.0, "RET": r, "RETX": r - (0.001 if (t % 63 == 62) else 0.0),  # a quarterly dividend
                     "SHROUT": shrout, "CFACPR": cf, "OPENPRC": prc * 0.99, "BIDLO": prc * 0.98,
                     "ASKHI": prc * 1.02, "DLSTCD": delist_code if (last and delist_code) else np.nan,
                     "DLRET": dlret if (last and delist_code) else np.nan})
    return rows


def test_crsp_adjust_removes_splits_and_adds_delisting_bar():
    dates = pd.bdate_range("2001-01-02", periods=40)
    rows = _crsp_rows("1", dates, 50.0, 0.01, 1000.0, cfacpr_before_split=2.0, split_day=20)
    rows += _crsp_rows("2", dates[:30], 20.0, 0.0, 500.0, delist_code=500, dlret=np.nan)
    adj = crsp.adjust(pd.DataFrame(rows))
    a = adj[adj.symbol == "1"].sort_values("date")
    assert np.allclose(a["close"].pct_change().dropna(), 0.01)  # no jump on the split day
    assert a["close"].iloc[-1] == pytest.approx(50.0 * 1.01 ** 39)  # anchored at the last split-adjusted price
    assert np.allclose(a["open"] / a["close"], 0.99)
    b = adj[adj.symbol == "2"]
    bar = b[b["date"].isna()]
    assert len(bar) == 1 and bar["close"].iloc[0] == pytest.approx(20.0 * 0.7)  # missing DLRET on code 500 -> -30%


def test_crsp_build_panel_point_in_time(tmp_path):
    y1 = pd.bdate_range("2001-01-02", "2001-12-31")
    y2 = pd.bdate_range("2002-01-02", "2002-12-31")
    dates = y1.append(y2)
    rows = _crsp_rows("1", dates, 100.0, 0.0005, 10_000.0, ticker="AAA")  # largest both years
    rows += _crsp_rows("2", dates[:len(y1) + 100], 50.0, 0.0, 5_000.0, delist_code=584, dlret=-0.5, ticker="BBB")
    rows += _crsp_rows("3", dates, 10.0, 0.004, 1_000.0, ticker="CCC")  # small in 2001, grows
    rows += _crsp_rows(crsp.SPY_PERMNO, dates, 100.0, 0.0003, 1.0, shrcd="73", exchcd="1", ticker="SPY")
    rows += _crsp_rows("9", dates, 5.0, 0.0, 90_000.0, shrcd="31", ticker="ADR")  # not a common stock: excluded
    pd.DataFrame(rows).to_parquet(tmp_path / "CRSP_20010101_20051231.parquet")
    panel, info, meta = crsp.build_panel(tmp_path, top_n=2, years=(2001, 2002), log=lambda m: None)
    assert set(panel.symbols) == {"1", "2", "3", "SPY"} and panel.tradable_symbols == ["1", "2", "3"]
    el = panel.eligible
    assert not el.loc["2001"].any().any()  # no year-end before 2001 in the data
    assert el.loc["2002", "1"].all() and not el.loc["2002", "3"].any() and not el["SPY"].any()
    # BBB: delisted 100 trading days into 2002 at -50%, one bar later, never bought on its last two bars
    b = panel.close["2"].dropna()
    assert len(b) == len(y1) + 101 and b.iloc[-1] == pytest.approx(b.iloc[-2] * 0.5, rel=1e-6)
    assert b.index[-1] == y2[100] and panel.open["2"].loc[y2[100]] == pytest.approx(b.iloc[-1], rel=1e-6)
    assert el.loc[b.index[-3], "2"] and not el.loc[b.index[-2:], "2"].any()
    assert panel.delist_exit and info.set_index("symbol").loc["1", "name"] == "AAA"
    assert panel.extra["cap"].loc[y1[-1], "1"] == pytest.approx(100.0 * 1.0005 ** (len(y1) - 1) * 10_000 / 1000)  # $M
    assert panel.extra["dividend"]["1"].gt(0).sum() == len(dates) // 63 and panel.extra["dividend"]["1"].max() == pytest.approx(0.001)
    crsp.save_panel(panel, info, meta, tmp_path / "out")
    back = crsp.load_panel(tmp_path / "out")
    pd.testing.assert_frame_equal(back.close, panel.close)
    pd.testing.assert_frame_equal(back.extra["cap"], panel.extra["cap"])
    pd.testing.assert_frame_equal(back.eligible, panel.eligible)
    assert back.subset(["1"]).eligible.columns.tolist() == ["1", "SPY"]


def test_engine_delist_exit_and_eligible_mask():
    n = 120
    up = list(np.linspace(100, 200, n))
    aaa = up[:80] + [np.nan] * (n - 80)  # series ends on day 79
    panel, dates = make_panel({"AAA": aaa, "BBB": up, "SPY": [100.0] * n})
    panel.delist_exit = True
    res = run_backtest(panel, cfg(entries={"above_ma": 5}, exits={"low_break": 5}, max_positions=2))
    t = res.trades[res.trades.symbol == "AAA"]
    assert len(t) == 1 and t.exit_reason.iloc[0] == DELIST_LABEL
    assert t.exit_date.iloc[0] == dates[79] and t.exit_price.iloc[0] == pytest.approx(panel.open["AAA"].iloc[79])
    assert (res.trades[res.trades.symbol == "BBB"].exit_reason != DELIST_LABEL).all()
    # the same rules on an unchanged panel: no forced exit
    plain, _ = make_panel({"AAA": aaa, "BBB": up, "SPY": [100.0] * n})
    assert (run_backtest(plain, cfg(entries={"above_ma": 5}, exits={"low_break": 5}, max_positions=2)).trades.symbol != "AAA").all()
    # eligibility: BBB may not be bought before day 60
    el = pd.DataFrame(True, index=dates, columns=["AAA", "BBB", "SPY"])
    el.iloc[:60, 1] = False
    plain.eligible = el
    res2 = run_backtest(plain, cfg(entries={"above_ma": 5}, exits={"low_break": 5}, max_positions=2))
    bbb = [t.entry_date for f in (res2.trades, res2.open_positions) for t in f.itertuples() if t.symbol == "BBB"]
    assert bbb and min(bbb) > dates[60]


# --------------------------------------------------------------------------- per-position risk management (DECISIONS 72)

from lab.engine import BREAKEVEN_LABEL  # noqa: E402


def _risk_panel():
    rng = np.random.default_rng(31)
    n = 400
    prices = {s: list(60 * np.cumprod(1 + rng.normal(0.0008, 0.015, n))) for s in ("AAA", "BBB", "CCC", "DDD", "SPY")}
    return make_panel(prices)


def test_atr_stop_exits_at_entry_minus_k_atr_of_signal_day():
    panel, dates = _risk_panel()
    c = cfg(entries={"breakout": 10}, exits={"atr_stop": 2.0}, max_positions=4, cost_bps=0.0)
    res = run_backtest(panel, c)
    from lab.blocks import atr as atr_fn
    a14 = atr_fn(panel, 14)
    stops = res.trades[res.trades.exit_reason == EXIT_BLOCKS["atr_stop"].label]
    assert len(stops) > 3
    idx = {d: i for i, d in enumerate(dates)}
    for t in stops.itertuples():
        e = idx[t.entry_date]
        level = t.entry_price - 2.0 * a14[t.symbol].iloc[e - 1]  # ATR of the signal day
        closes = panel.close[t.symbol].iloc[e:idx[t.exit_date]]  # closes from the entry day to the signal day
        assert closes.iloc[-1] < level and (closes.iloc[:-1] >= level).all()


def test_risk_sizing_heat_and_breakeven():
    panel, dates = _risk_panel()
    base = cfg(entries={"breakout": 10}, exits={"stop_loss_pct": 8.0}, max_positions=4, cost_bps=0.0)
    risk = replace(base, sizing="risk", risk_pct=2.0)
    res = run_backtest(panel, risk)
    # first position: shares x stop distance = 2% of equity (equity = 1 before the first buy)
    first = res.trades.sort_values("entry_date").iloc[0]
    shares = 0.02 * 1.0 / (first.entry_price * 0.08)
    assert "위험 2%/거래" in risk.describe()
    assert first.entry_price * shares == pytest.approx(0.25, rel=1e-9)  # 2% / 8% = 25% of equity
    capped = run_backtest(panel, replace(risk, max_weight=0.10))
    f2 = capped.trades.sort_values("entry_date").iloc[0]
    assert f2.entry_date == first.entry_date and f2.symbol == first.symbol
    day = f2.entry_date  # every buy on the first day is capped at 10% of equity (equity = 1, no cost)
    same_day = capped.trades[capped.trades.entry_date == day]
    invested = capped.exposure.loc[day] * capped.equity.loc[day]
    expected = sum(0.10 * panel.close[t.symbol].loc[day] / t.entry_price for t in same_day.itertuples())
    assert invested == pytest.approx(expected, rel=1e-9)
    # heat: with 2% risk per position, a 3% heat limit leaves room for one position at a time
    hot = run_backtest(panel, replace(risk, heat=0.03))
    held = pd.Series(0, index=dates)
    for t in hot.trades.itertuples():
        held.loc[t.entry_date:t.exit_date] += 1
    assert held.max() <= 2 and len(hot.trades) < len(res.trades)  # 2 only on the day one exits and another enters
    # breakeven: after +1R the stop sits at the entry price; such exits are labelled and lose nothing
    be = run_backtest(panel, replace(base, breakeven_r=1.0))
    b = be.trades[be.trades.exit_reason == BREAKEVEN_LABEL]
    assert len(b) > 0 and (b.exit_price > b.entry_price * 0.92).all()
    # defaults off: the config keys and results of earlier runs are unchanged
    assert "risk_pct" not in base.to_dict() and "heat" not in base.to_dict() and "breakeven_r" not in base.to_dict()
    pd.testing.assert_frame_equal(run_backtest(panel, base).trades, run_backtest(panel, replace(base, heat=None)).trades)
    # risk sizing needs an initial stop: without one nothing is bought
    assert run_backtest(panel, replace(risk, exits={"low_break": 10})).trades.empty


def test_fit_weights_matches_one_column_at_a_time():
    rng = np.random.default_rng(12)
    T, m = 900, 40
    R = rng.normal(0.0004, rng.uniform(0.003, 0.04, m), (T, m))
    C = np.full((T, m), 0.0001)
    starts = rng.integers(0, 400, m)
    for k, f in enumerate(starts):  # later starts: zero return and zero cash before the first day
        R[:f, k], C[:f, k] = 0.0, 0.0
    R[:, 0] = 0.0005  # never falls: 100%
    got = target.fit_weights(R, C)
    want = [target.fit_weight(R[f:, k], C[f:, k]) for k, f in enumerate(starts)]
    assert got.tolist() == want and got[0] == 1.0


# --------------------------------------------------------------------------- time stop and partial take-profit (DECISIONS 73)

from lab.engine import PARTIAL_SUFFIX  # noqa: E402


def test_time_stop_only_cuts_positions_not_in_profit():
    n = 80
    flat = [10.0] * 30
    up = flat + [11.0] + [11.0] * 5 + list(np.linspace(11.0, 14.0, n - 36))  # breakout, then a slow rise
    dull = flat + [11.0] + [10.8] * (n - 31)  # breakout, then stuck below the entry
    panel, dates = make_panel({"AAA": up, "BBB": dull, "SPY": [100.0] * n})
    res = run_backtest(panel, cfg(entries={"breakout": 5}, exits={"time_stop": 10}, max_positions=2))
    t = res.trades
    assert list(t.symbol) == ["BBB"] and t.exit_reason.iloc[0] == EXIT_BLOCKS["time_stop"].label
    entry = dates.get_loc(t.entry_date.iloc[0])
    assert dates.get_loc(t.exit_date.iloc[0]) == entry + 10 + 1  # signal on day entry+10, sold next open
    assert list(res.open_positions.symbol) == ["AAA"]


def test_partial_take_profit_one_row_and_stand_alone_account():
    n = 120
    rng = np.random.default_rng(3)
    base_path = [10.0] * 30 + list(10.0 * np.cumprod(1 + rng.normal(0.004, 0.012, n - 30)))
    panel, dates = make_panel({"AAA": base_path, "SPY": [100.0] * n})
    c = cfg(entries={"breakout": 5}, exits={"trailing_pct": 12.0, "take_profit_pct": 15.0}, cost_bps=10.0)
    res = run_backtest(panel, c)
    assert len(res.partial_exits) >= 1
    pe = res.partial_exits.iloc[0]
    trade = res.trades[res.trades.entry_date == pe.entry_date]
    if len(trade):  # closed: one row, both sales in its return
        t = trade.iloc[0]
        cc = 10 / 10_000
        want = (0.5 * pe.price + 0.5 * t.exit_price) * (1 - cc) / (t.entry_price * (1 + cc)) - 1
        assert t["return"] == pytest.approx(want) and t.exit_reason.endswith(PARTIAL_SUFFIX)
    # the stand-alone account rebuilt from the records equals the engine's one-slot account
    scfg = single.single_config(c, panel)
    sres = run_backtest(panel, scfg)
    pd.testing.assert_frame_equal(sres.partial_exits, res.partial_exits)
    rets = single.symbol_returns(panel, scfg, sres)
    eq = (1 + rets["AAA"].fillna(0.0)).cumprod()
    assert eq.iloc[-1] == pytest.approx(res.equity.iloc[-1] / res.equity.iloc[0], rel=1e-9)


# --------------------------------------------------------------------------- stock screens (DECISIONS 77)

from lab import screens  # noqa: E402


def _screen_panel():
    rng = np.random.default_rng(17)
    n = 700
    dates = pd.bdate_range("2018-01-01", periods=n)
    drift = {"UP1": 0.0012, "UP2": 0.0009, "FLAT": 0.0, "DOWN": -0.0008, "WILD": 0.0005, "SPY": 0.0004}
    vol = {"UP1": 0.012, "UP2": 0.012, "FLAT": 0.012, "DOWN": 0.012, "WILD": 0.05, "SPY": 0.01}
    prices = {s: list(50 * np.cumprod(1 + rng.normal(drift[s], vol[s], n))) for s in drift}
    panel, dates = make_panel(prices, start="2018-01-01")
    cap = pd.DataFrame({s: 1000.0 for s in panel.symbols}, index=dates)
    cap["FLAT"] = 100.0
    panel.extra = {"cap": cap, "dividend": pd.DataFrame(0.0, index=dates, columns=panel.symbols)}
    panel.eligible = pd.DataFrame(True, index=dates, columns=panel.symbols)
    return panel, dates


def test_monthly_mask_is_point_in_time_and_keeps_top_k():
    panel, dates = _screen_panel()
    cfg = screens.ScreenConfig(exclude={"x_volatility": 10, "x_small": 300}, select={"s_momentum": 12}, top_k=2)
    eligible, rank = screens.monthly_mask(panel, cfg)
    me = screens.month_ends(dates)
    assert not eligible["SPY"].any() and not eligible["FLAT"].any()  # benchmark never; FLAT fails the cap floor
    assert not eligible["WILD"].any()  # the most volatile 10% of six symbols = WILD only
    last = eligible.loc[me[-1]]
    assert last.sum() == 2 and last["UP1"] and last["UP2"]  # top 2 by 12-1 momentum
    mom = screens.SELECT_SCORES["s_momentum"].compute(panel, 12).loc[me[-1]]
    better, worse = ("UP1", "UP2") if mom["UP1"] > mom["UP2"] else ("UP2", "UP1")
    assert rank.loc[me[-1], better] == 1.0 and rank.loc[me[-1], worse] == 2.0
    # held through the following month: every day after a month end repeats that month end's decision
    for a, b in zip(me[-3:-1], me[-2:]):
        span = eligible.loc[a:b].iloc[:-1]
        assert (span.eq(eligible.loc[a]).all(axis=1)).all()
    assert not eligible.loc[: me[10]].any().any()  # 12-1 momentum needs 252 days: nothing before December
    assert "제외: 변동성 상위(10) + 시가총액 하한(300) | 선별: 모멘텀 (M-1개월 수익률)(12) 상위 2" == cfg.describe()


def test_engine_buys_best_rank_first_and_defaults_unchanged():
    panel, dates = _screen_panel()
    rng = np.random.default_rng(5)
    c = cfg(entries={"above_ma": 5}, exits={"low_break": 5}, max_positions=1, start="2019-06-01")
    plain = run_backtest(panel, c)
    screened = screens.apply_screens(panel, screens.ScreenConfig(exclude={}, select={"s_momentum": 6}, top_k=3))
    res = run_backtest(screened, c)
    assert res.trades.symbol.iloc[0] != "SPY" and len(res.trades) > 0
    # on its first entry day the engine took the symbol with the best rank among the eligible candidates
    first = pd.concat([res.trades, res.open_positions]).sort_values("entry_date").iloc[0]
    signal_day = dates[dates.get_loc(first.entry_date) - 1]
    ranks = screened.rank.loc[signal_day][screened.eligible.loc[signal_day]]
    from lab.blocks import ENTRY_BLOCKS
    sig = ENTRY_BLOCKS["above_ma"].compute(panel, 5).loc[signal_day]
    cands = ranks[sig.reindex(ranks.index).fillna(False).astype(bool)]
    assert first.symbol == cands.idxmin()
    # a panel without rank/screens behaves as before (dollar-volume order): same trades as the plain run
    pd.testing.assert_frame_equal(run_backtest(panel, c).trades, plain.trades)


def test_big_panel_path_matches_list_path(monkeypatch):
    from lab import engine as eng
    panel = _random_panel(n=500, seed=8)
    c = cfg(entries={"breakout": 10}, exits={"trailing_pct": 8.0, "low_break": 15}, max_positions=2, cost_bps=10.0,
            sizing="inverse_vol")
    a = run_backtest(panel, c)
    monkeypatch.setattr(eng, "LIST_MAX", 0)  # force the numpy-row path
    b = run_backtest(panel, c)
    pd.testing.assert_series_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_series_equal(a.exposure, b.exposure)


def test_signal_cache_share_from_reuses_frames():
    panel = _random_panel(n=300, seed=9)
    base = SignalCache(panel)
    base.frame("breakout", 10)
    other = SignalCache(panel).share_from(base)
    assert other.frame("breakout", 10) is base.frame("breakout", 10)
    assert len(base._frames) == 1


# --------------------------------------------------------------------------- practitioner screens (2026-10-25 redesign)

def _v2_panel():
    rng = np.random.default_rng(23)
    n = 600
    dates = pd.bdate_range("2017-01-02", periods=n)
    # LEAD: steady uptrend; CHOP: flat and noisy; FALL: downtrend; SPY: mild uptrend
    paths = {
        "LEAD": 50 * np.cumprod(1 + rng.normal(0.0012, 0.010, n)),
        "LEAD2": 40 * np.cumprod(1 + rng.normal(0.0010, 0.011, n)),
        "CHOP": 30 + 3 * np.sin(np.arange(n) / 9.0) + rng.normal(0, 0.3, n),
        "FALL": 80 * np.cumprod(1 + rng.normal(-0.0010, 0.012, n)),
        "SPY": 200 * np.cumprod(1 + rng.normal(0.0004, 0.008, n)),
    }
    panel, dates = make_panel({k: list(v) for k, v in paths.items()}, start="2017-01-02",
                              opens={k: list(v * (1 + rng.normal(0, 0.004, n))) for k, v in paths.items()})
    panel.info = pd.DataFrame({"symbol": panel.symbols, "name": panel.symbols, "asset_group": "",
                               "sic": [3570, 3571, 5812, 5813, np.nan]})  # LEAD/LEAD2 one industry, CHOP/FALL another
    panel.extra = {"cap": pd.DataFrame(2000.0, index=dates, columns=panel.symbols),
                   "dividend": pd.DataFrame(0.0, index=dates, columns=panel.symbols)}
    panel.eligible = pd.DataFrame(True, index=dates, columns=panel.symbols)
    return panel, dates


def test_trend_template_and_rs_rating():
    panel, dates = _v2_panel()
    last = dates[-1]
    fail = screens.EXCLUDE_SCREENS["tt_fail"].compute(panel, 25)
    assert not fail.loc[last, "LEAD"] and fail.loc[last, "FALL"]  # uptrend passes, downtrend fails
    assert fail.iloc[100].all()  # not enough history: nothing passes (flag = excluded)
    rs = screens.SELECT_SCORES["s_rs"].compute(panel, None).loc[last]
    assert rs["LEAD"] > rs["FALL"] and 0 <= rs.min() and rs.max() <= 100
    # industry: LEAD and LEAD2 share SIC 35, so their industry score is the same and above CHOP/FALL's
    ind = screens.SELECT_SCORES["s_industry_rs"].compute(panel, 6).loc[last]
    assert ind["LEAD"] == ind["LEAD2"] and ind["LEAD"] > ind["FALL"]
    weak = screens.EXCLUDE_SCREENS["x_weak_industry"].compute(panel, 50).loc[last]
    assert weak["FALL"] and not weak["LEAD"]
    acc = screens.SELECT_SCORES["s_accum"].compute(panel, 3).loc[last]
    assert acc["LEAD"] > 0 and np.isfinite(acc["LEAD"])


def test_profiles_run_through_monthly_mask():
    panel, dates = _v2_panel()
    for key, (label, cfg) in screens.PROFILES.items():
        eligible, rank = screens.monthly_mask(panel, cfg)
        assert eligible.shape == panel.close.shape and not eligible["SPY"].any(), key
        assert label and "제외:" in cfg.describe()
    eligible, rank = screens.monthly_mask(panel, screens.PROFILES["minervini"][1])
    last = screens.month_ends(dates)[-1]
    assert eligible.loc[last, "LEAD"] and not eligible.loc[last, "FALL"]
    assert rank.loc[last, "LEAD"] >= 1
    assert set(screens.STYLE_TILTS) <= set(screens.SELECT_SCORES)
    assert set(screens.HYGIENE_SCREENS) <= set(screens.ALL_SCREENS)


def test_every_selection_score_computes_and_points_the_right_way():
    panel, dates = _v2_panel()
    last = dates[-1]
    tiers = screens.scores_by_tier()
    assert len(screens.SELECT_SCORES) >= 35 and all(tiers[t] for t in (1, 2, 3))
    for key, sc in screens.SELECT_SCORES.items():
        val = sc.param.default if sc.param else None
        if key == "s_reversal":
            val = 24  # the default 36 months is longer than this synthetic history
        out = sc.compute(panel, val)
        assert out.shape == panel.close.shape, key
        assert np.isfinite(out.loc[last, "LEAD"]), key  # a long clean series has a score at the end
    # direction checks on the synthetic paths: LEAD (steady uptrend) beats FALL (downtrend)
    for key in ("s_rs_spy", "s_new_highs", "s_efficiency", "s_slope_r2", "s_ma_slope", "s_adx", "s_low52_dist",
                "s_consistency", "s_dd_shallow", "s_atr_momentum", "s_rs_line_high", "s_fresh_high"):
        sc = screens.SELECT_SCORES[key]
        v = sc.compute(panel, sc.param.default if sc.param else None).loc[last]
        assert v["LEAD"] > v["FALL"], key
    # squeeze scores: CHOP (flat, noisy) is not "contracting" more than a trend by construction; just finite
    for key in ("s_atr_squeeze", "s_bb_squeeze", "s_range_contraction"):
        sc = screens.SELECT_SCORES[key]
        assert np.isfinite(sc.compute(panel, sc.param.default if sc.param else None).loc[last, "CHOP"]), key
    # style tilts stay available as selection scores (small caps must remain testable)
    assert all(k in screens.SELECT_SCORES for k in screens.STYLE_TILTS)
    assert "h_cap" not in screens.HYGIENE_DEFAULT


from lab.blocks import ALL_BLOCKS  # noqa: E402


def test_left_list_exit_sells_the_day_after_a_stock_leaves_the_list():
    """Replacement layer (DECISIONS 79): out of the monthly list at day t's close -> sold at t+1 open;
    the frame is per cache, so a screened panel sharing another cache's block outputs keeps its own."""
    n = 40
    panel, dates = make_panel({"AAA": [10.0 + 0.1 * i for i in range(n)], "SPY": [10.0] * n})
    panel.eligible = pd.DataFrame(True, index=dates, columns=panel.symbols)
    panel.eligible["SPY"] = False
    panel.eligible.loc[dates[30]:, "AAA"] = False
    c = cfg(entries={"always": None}, exits={"left_list": None})
    base = SignalCache(replace(panel, eligible=None))
    base.frame("left_list", None)  # the unscreened frame must not leak into the screened run
    res = run_backtest(panel, c, SignalCache(panel).share_from(base))
    t = res.trades.iloc[0]
    assert t["entry_date"] < dates[30] and t["exit_date"] == dates[31] and t["exit_reason"] == ALL_BLOCKS["left_list"].label
    assert len(res.trades) == 1 and res.open_positions.empty  # not bought again while off the list
