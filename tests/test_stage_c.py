"""Stage C (DECISIONS 81): event entries, new exits, daily universe and the study runner."""
import numpy as np
import pandas as pd
import pytest

from lab import blocks, crsp, research_log as rlog
from lab.data import panel_from_long
from lab.engine import StrategyConfig, run_backtest


def panel_of(close: dict, high=None, low=None, open_=None, volume=None, start="2010-01-04"):
    n = len(next(iter(close.values())))
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for s, c in close.items():
        h = (high or {}).get(s, c)
        lo = (low or {}).get(s, c)
        o = (open_ or {}).get(s, c)
        v = (volume or {}).get(s, [1_000_000.0] * n)
        for d, ci, hi, li, oi, vi in zip(dates, c, h, lo, o, v):
            rows.append({"date": d, "symbol": s, "open": oi, "high": hi, "low": li, "close": ci, "volume": vi})
    return panel_from_long(pd.DataFrame(rows)), dates


def days(frame, sym="AAA"):
    return list(np.flatnonzero(frame[sym].to_numpy()))


# --------------------------------------------------------------------------- entries

def test_first_day_fires_once_per_run():
    cond = pd.DataFrame({"A": [False, True, True, False, True, np.nan, True]})
    assert days(blocks.first_day(cond), "A") == [1, 4, 6]


def test_c_donchian_first_close_above_prior_high():
    c = [10, 10, 10, 11, 12, 11, 13]
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * 7})
    # prior 3-day high: day3 10 -> 11 > 10 (fires), day4 12 > 11 (still true, no new event),
    # day5 11 < 12, day6 13 > 12 (fires again)
    assert days(blocks._c_donchian(p, 3)) == [3, 6]


def test_c_golden_cross_of_fast_over_200():
    c = [100.0] * 200 + [90.0] * 5 + [120.0] * 10
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * len(c)})
    fast = p.close["AAA"].rolling(2).mean()
    slow = p.close["AAA"].rolling(200).mean()
    above = (fast > slow).to_numpy()
    want = [i for i in range(1, len(c)) if above[i] and not above[i - 1]]
    assert days(blocks._c_golden(p, 2)) == want == [205]


def test_c_high_volume_needs_new_high_and_volume():
    n = 260
    c = [10.0] * 255 + [11.0, 11.5, 12.0, 12.5, 13.0]
    v = [100.0] * 255 + [140.0, 200.0, 100.0, 300.0, 300.0]
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * n}, volume={"AAA": v})
    # day 255: new high, volume 1.4x <1.5 -> no; 256: 2x -> fires; 257: low volume -> off;
    # 258: high + 3x avg -> fires (new run); 259: still true -> no
    assert days(blocks._c_high_volume(p, 1.5)) == [256, 258]


def test_c_squeeze_breakout_after_narrow_bands():
    rng = np.random.default_rng(1)
    c = list(100 + rng.normal(0, 3, 140)) + [100.0 + 0.01 * i for i in range(30)] + [110.0]
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * len(c)})
    got = days(blocks._c_squeeze(p, 20))
    assert got == [len(c) - 1]  # the jump after the quiet stretch is the only event


def test_c_rsi2_pullback_above_sma200():
    c = [100.0 + 0.1 * i for i in range(210)] + [115.0, 113.0, 111.0, 125.0]
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * len(c)})
    r2 = blocks.rsi(p.close, 2)["AAA"].to_numpy()
    sma = p.close["AAA"].rolling(200).mean().to_numpy()
    cond = (c > sma) & (r2 < 10)
    want = [i for i in range(len(c)) if cond[i] and not (i and cond[i - 1])]
    assert days(blocks._c_rsi2_pullback(p, 10)) == want and want


def test_c_macd_cross_above_zero_line():
    c = list(np.r_[np.linspace(100, 90, 60), np.linspace(90, 130, 60), np.linspace(130, 120, 15), np.linspace(120, 140, 20)])
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * len(c)})
    s = pd.Series(c)
    line = s.ewm(span=12, adjust=False, min_periods=12).mean() - s.ewm(span=26, adjust=False, min_periods=26).mean()
    sig = line.ewm(span=9, adjust=False, min_periods=9).mean()
    above = (line > sig).to_numpy()
    want = [i for i in range(1, len(c)) if above[i] and not above[i - 1] and line[i] > 0]
    assert days(blocks._c_macd(p, 12)) == want and len(want) >= 1


def test_c_di_cross_with_adx_floor():
    rng = np.random.default_rng(3)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.02, 200))
    p, _ = panel_of({"AAA": list(c), "SPY": [1.0] * 200}, high={"AAA": list(c * 1.01)}, low={"AAA": list(c * 0.99)})
    adx, plus, minus = blocks.adx_di(p, 14)
    above = (plus["AAA"] > minus["AAA"]).to_numpy()
    a = adx["AAA"].to_numpy()
    want = [i for i in range(1, 200) if above[i] and not above[i - 1] and a[i] >= 20]
    assert days(blocks._c_di_cross(p, 20)) == want


def test_c_tsmom_beats_bills_over_l_months():
    n = 80
    c = [100.0] * 63 + [100.5] * 5 + [101.0] * 12
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * n})
    p.cash = pd.Series(0.0001, index=p.close.index)  # 63-day bills ~ 0.63%
    # 3 months = 63 sessions: day 63 ret 0.5% < 0.63%; day 68 ret 1% > 0.63% -> first true day
    assert days(blocks._c_tsmom(p, 3)) == [68]


def test_c_gap_volume_gap_volume_and_green_day():
    n = 60
    c = [10.0] * 55 + [11.0, 11.0, 12.0, 12.0, 13.0]
    o = [10.0] * 55 + [10.5, 11.0, 11.6, 12.0, 13.5]
    h = [10.0] * 55 + [11.0, 11.0, 12.0, 12.0, 13.5]
    v = [100.0] * 55 + [400.0, 100.0, 400.0, 100.0, 400.0]
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * n}, high={"AAA": h}, open_={"AAA": o}, volume={"AAA": v})
    # day 55: gap 5% over 10, volume 4x, green -> fires; day 57: open 11.6 >= 11 x 1.04, 4x, green -> fires;
    # day 59: gap but close == open (not green) -> no
    assert days(blocks._c_gap_volume(p, 4)) == [55, 57]


# --------------------------------------------------------------------------- exits

def psar_reference(h, lo, step, mx=0.2):
    """Textbook Wilder SAR for one series, starting long on the second bar."""
    up, sar, ep, af = True, lo[0], max(h[0], h[1]), step
    out = [False, False]
    for i in range(2, len(h)):
        nxt = sar + af * (ep - sar)
        if up:
            nxt = min(nxt, lo[i - 1], lo[i - 2])
            if lo[i] < nxt:
                up, sar, ep, af = False, ep, lo[i], step
            else:
                sar = nxt
                if h[i] > ep:
                    ep, af = h[i], min(af + step, mx)
        else:
            nxt = max(nxt, h[i - 1], h[i - 2])
            if h[i] > nxt:
                up, sar, ep, af = True, ep, h[i], step
            else:
                sar = nxt
                if lo[i] < ep:
                    ep, af = lo[i], min(af + step, mx)
        out.append(not up)
    return out


def test_psar_matches_textbook_loop():
    rng = np.random.default_rng(7)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.02, 150))
    h, lo = c * 1.01, c * 0.99
    p, _ = panel_of({"AAA": list(c), "SPY": [1.0] * 150}, high={"AAA": list(h)}, low={"AAA": list(lo)})
    got = blocks.psar_down(p, 0.02)["AAA"].tolist()
    assert got[1:] == psar_reference(h, lo, 0.02)[1:]
    flips = days(blocks._x_psar(p, 0.02))
    ref = psar_reference(h, lo, 0.02)
    assert flips == [i for i in range(1, 150) if ref[i] and not ref[i - 1]]


def test_rsi2_exit_close_above_sma5_or_rsi2():
    c = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.5, 4.0]
    p, _ = panel_of({"AAA": c, "SPY": [1.0] * len(c)})
    got = blocks._x_rsi2(p, 70)["AAA"].tolist()
    sma5 = pd.Series(c).rolling(5).mean()
    r2 = blocks.rsi(p.close, 2)["AAA"]
    want = [(c[i] > sma5[i]) or (r2.iloc[i] > 70) for i in range(len(c))]
    assert got == want and got[6]  # the bounce day exits


def test_chandelier_uses_atr22_from_the_peak():
    n = 80
    c = list(np.linspace(100, 140, 60)) + [139.0, 137.0, 133.0, 125.0, 120.0] + [120.0] * 15
    p, dates = panel_of({"AAA": c, "SPY": [100.0] * n}, high={"AAA": [x * 1.002 for x in c]}, low={"AAA": [x * 0.98 for x in c]})
    cfg = StrategyConfig(entries={"breakout": 20}, exits={"chandelier": 3.0}, start=str(dates[0].date()),
                         end=str(dates[-1].date()), max_positions=1, cost_bps=0.0, min_price=0.0, min_dollar_volume=0.0)
    res = run_backtest(p, cfg)
    t = res.trades.iloc[0]
    a22 = blocks.atr(p, 22)["AAA"]
    close = p.close["AAA"]
    e = dates.get_loc(t["entry_date"])
    peak = np.maximum.accumulate(np.r_[p.open["AAA"].iloc[e], close.iloc[e:].to_numpy()])[1:]
    hit = np.flatnonzero(close.iloc[e:].to_numpy() < peak - 3.0 * a22.iloc[e:].to_numpy())[0] + e
    assert t["exit_date"] == dates[hit + 1] and t["exit_reason"] == blocks.EXIT_BLOCKS["chandelier"].label


# --------------------------------------------------------------------------- daily universe

def test_daily_membership_ranks_21_day_average_without_lookahead():
    dates = pd.bdate_range("2001-01-02", periods=40)
    caps = pd.DataFrame({"A": 100.0, "B": 50.0, "C": 10.0}, index=dates)
    caps.loc[dates[25]:, "C"] = 1000.0  # C jumps on day 25
    member, rank = crsp.daily_membership(caps, top_n=2, window=21)
    assert not member.iloc[:20].any().any()  # no 21-day average yet
    assert member.loc[dates[20], ["A", "B"]].all() and not member.loc[dates[20], "C"]
    # C's 21-day average passes B (50) once 1000 x k + 10 x (21 - k) > 50 x 21 -> k = 1: day 25 itself
    first = member["C"].idxmax()
    avg = caps["C"].rolling(21).mean()
    assert first == avg[avg > 50].index[0] == dates[25]
    assert rank.loc[dates[30], "C"] == 1.0


def test_daily_panel_admits_a_mid_year_riser(tmp_path):
    from tests.test_lab import _crsp_rows
    dates = pd.bdate_range("2001-01-02", "2002-12-31")
    rows = _crsp_rows("1", dates, 100.0, 0.0, 10_000.0, ticker="AAA")
    rows += _crsp_rows("2", dates, 50.0, 0.0, 5_000.0, ticker="BBB")
    rows += _crsp_rows("3", dates, 10.0, 0.004, 5_500.0, ticker="CCC")  # passes BBB in mid-2002
    rows += _crsp_rows(crsp.SPY_PERMNO, dates, 100.0, 0.0003, 1.0, shrcd="73", exchcd="1", ticker="SPY")
    pd.DataFrame(rows).to_parquet(tmp_path / "CRSP_20010101_20051231.parquet")
    panel, info, meta = crsp.build_panel(tmp_path, top_n=2, years=(2001, 2002), log=lambda m: None, daily=True)
    el = panel.eligible
    cap = panel.extra["cap"]
    avg = cap.rolling(21, min_periods=21).mean()
    switch = (avg["3"] > avg["2"]).idxmax()
    assert switch.year == 2002 and switch.month < 12  # a mid-year switch the year-end rule would miss
    assert el.loc[switch, "3"] and not el.loc[switch, "2"] and not el["SPY"].any()
    assert panel.extra["cap_rank"].loc[switch, "3"] == 2.0
    crsp.save_panel(panel, info, meta, tmp_path / "out")
    back = crsp.load_panel(tmp_path / "out")
    pd.testing.assert_frame_equal(back.extra["cap_rank"], panel.extra["cap_rank"])
    pd.testing.assert_frame_equal(back.eligible, panel.eligible)


# --------------------------------------------------------------------------- study

def test_menu_is_the_registered_one():
    from lab import stage_c_study as sc
    assert len(sc.ENTRIES) == 10 and len(sc.EXITS) == 9
    assert sum(len(sc.cells(e, x)) for e, x in sc.pairs()) == 756
    for e, (k, v) in sc.ENTRIES.items():
        assert k in blocks.ENTRY_BLOCKS and (len(v) == 3 or e == "E0")
    for x, (k, v) in sc.EXITS.items():
        assert k in blocks.EXIT_BLOCKS and len(v) == 3
    entries, exits, be = sc.rules("E1", "X9", 1, 1)
    assert entries == {"c_donchian": 55} and exits == {"atr_stop": 2.0, "time_stop": 60} and be == 2.0
    assert sc.canonical("E8", "X8") == (2, 2)
    assert sc.ENTRIES["E8"][1][2] == 12 and sc.EXITS["X8"][1][2] == 200


def _study_panel():
    rng = np.random.default_rng(11)
    n, syms = 700, ["A", "B", "C", "D", "E", "F"]
    close = {s: list(50 * np.cumprod(1 + rng.normal(0.0006, 0.02, n))) for s in syms}
    close["SPY"] = list(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)))
    p, dates = panel_of(close, high={s: [x * 1.01 for x in v] for s, v in close.items()},
                        low={s: [x * 0.99 for x in v] for s, v in close.items()}, start="2017-01-03")
    p.cash = pd.Series(0.0001, index=dates)
    cap = pd.DataFrame({s: 1000.0 + 100 * i for i, s in enumerate(p.symbols)}, index=dates)
    p.extra = {"cap": cap, "dividend": pd.DataFrame(0.0, index=dates, columns=p.symbols),
               "cap_rank": cap.rank(axis=1, ascending=False).astype("float32")}
    p.eligible = pd.DataFrame(True, index=dates, columns=p.symbols)
    p.delist_exit = True
    return p, dates


def test_study_pair_runs_writes_and_resumes(tmp_path, monkeypatch):
    from lab import stage_c_study as sc
    p, dates = _study_panel()
    monkeypatch.setattr(sc, "SAMPLES", {"a": (str(dates[260].date()), str(dates[-1].date()), "A")})
    monkeypatch.setattr(sc, "ENTRIES", {"E0": sc.ENTRIES["E0"], "E1": sc.ENTRIES["E1"]})
    monkeypatch.setattr(sc, "EXITS", {"X1": sc.EXITS["X1"], "X9": sc.EXITS["X9"]})
    monkeypatch.setattr(sc, "HYGIENE", {"h_price": 5})
    monkeypatch.setattr(rlog, "RESULTS_DIR", tmp_path / "res")
    elig = sc.daily_eligible(p)
    assert not elig["SPY"].any()
    sc._setup(p, elig)
    for job in sc.todo_jobs(tmp_path):
        sc.write_pair(tmp_path, sc._run_pair(job), p.tradable_symbols)
    assert sc.todo_jobs(tmp_path) == []
    c1 = pd.read_csv(tmp_path / "c1" / "a" / "E1__X9.csv")
    c2 = pd.read_csv(tmp_path / "c2" / "a" / "E1__X9.csv")
    assert len(c1) == 9 and len(c2) == 9 + 2  # canonical cell also with the two other priorities
    assert set(c2["priority"]) == set(sc.PRIORITIES)
    assert {"expectancy", "win_rate", "payoff", "avg_hold_days", "signals_per_year", "sleeve_dd_excess"} <= set(c1.columns)
    dv = c2[c2["priority"] == "dv"].set_index(["xi", "yi"])
    sig_n = c1.set_index(["xi", "yi"])["signals"]
    assert (dv["taken"] + dv["missed"] == sig_n.reindex(dv.index)).all()
    assert (dv["c2_unmatched"] == 0).all() and (dv["taken"] == dv["n_trades"]).all()  # every account buy is a signal
    assert {"sig_expectancy", "sig_win_rate", "sig_payoff", "sig_avg_hold_days", "sig_big_share"} <= set(c1.columns)
    sig = pd.read_parquet(tmp_path / "c1" / "a" / "E1__X9_signals.parquet")
    assert len(sig) == sig_n.sum() and (sig["signal_date"] < sig["entry_date"]).all()
    assert len(sig) >= c1["n_trades"].sum()  # overlapping signals of one symbol are all kept
    assert (pd.read_csv(tmp_path / "c1" / "a" / "E0__X1.csv")["xi"] == 0).all()
    trades = pd.read_parquet(tmp_path / "c1" / "a" / "E1__X9_trades.parquet")
    assert {"symbol", "entry_date", "return", "xi", "yi"} <= set(trades.columns)
    assert (c2["n_trades"] >= 0).all() and c2["xi"].max() <= 2
    # E1 never buys again while held, and X9's breakeven/time stop are used
    assert set(trades["exit_reason"]) <= {blocks.EXIT_BLOCKS[k].label for k in ("atr_stop", "time_stop")} | {"손익분기 손절", "상장폐지"} | set(trades["exit_reason"])
    # resume: remove one pair -> only that pair is to do
    (tmp_path / "c2" / "a" / "E0__X1.csv").unlink()
    assert sc.todo_jobs(tmp_path) == [("a", "E0", "X1")]
    sc.write_pair(tmp_path, sc._run_pair(("a", "E0", "X1")), p.tradable_symbols)
    summ = sc.summarize(tmp_path)
    assert summ["cells_c1"] == 3 * 2 + 9 * 2 and (tmp_path / "entry_ranks_c2.csv").exists()


def test_missed_vs_taken_counts():
    from lab import stage_c_study as sc
    d = pd.bdate_range("2020-01-01", periods=10)
    sig = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "signal_date": d[[1, 2, 3, 4]], "entry_date": d[[2, 3, 4, 5]],
                        "return": [0.6, -0.1, 0.0, 0.2]})
    c2 = pd.DataFrame({"symbol": ["A", "D", "E"], "entry_date": d[[2, 5, 6]]})
    rank = pd.DataFrame({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}, index=d)
    o = sc.missed_vs_taken(sig, c2, rank)
    assert (o["taken"], o["missed"], o["c2_unmatched"]) == (2, 2, 1)
    assert o["taken_expectancy"] == pytest.approx(0.4) and o["missed_expectancy"] == pytest.approx(-0.05)
    assert o["big_winners"] == 1 and o["big_winners_taken_share"] == 1.0  # +60% >= +50%, bought
    assert o["taken_cap_rank_p50"] == 2.5 and o["missed_cap_rank_p50"] == 2.5  # ranks on the signal day


def test_signal_outcomes_keep_overlapping_signals():
    # two breakouts of the same stock while the first position is still open: the account holds one,
    # the per-signal table trades both, each sold at the open after its own exit signal
    c = [10.0] * 30 + [11.0, 11.0, 12.0, 12.0, 12.0, 9.0, 9.0, 9.0]  # 30 days: the 20-day dollar volume exists
    p, dates = panel_of({"AAA": c, "SPY": [1.0] * len(c)})
    cfg = StrategyConfig(entries={"c_donchian": 3}, exits={"trailing_pct": 20}, start=str(dates[0].date()),
                         end=str(dates[-1].date()), max_positions=1, cost_bps=10.0, min_price=0.0, min_dollar_volume=0.0)
    res = run_backtest(p, cfg)
    sig = run_backtest(p, cfg, outcomes=True)
    assert list(sig["signal_date"]) == [dates[30], dates[32]] and len(res.trades) == 1
    # both exit on day 35 (9 < 12 x 0.8) and sell at the day-36 open
    assert (sig["exit_date"] == dates[36]).all() and sig["closed"].all()
    first = sig.iloc[0]
    assert first["return"] == pytest.approx(9.0 * 0.999 / (11.0 * 1.001) - 1)
    assert first["return"] == pytest.approx(res.trades.iloc[0]["return"]) and first["entry_date"] == res.trades.iloc[0]["entry_date"]


def test_signals_only_tables_replay_like_the_account(tmp_path, monkeypatch):
    from dataclasses import replace
    from lab import signal_replay as sr, stage_c_study as sc
    p, dates = _study_panel()
    monkeypatch.setattr(sc, "SAMPLES", {"a": (str(dates[260].date()), str(dates[-1].date()), "A")})
    monkeypatch.setattr(sc, "ENTRIES", {"E0": sc.ENTRIES["E0"], "E1": sc.ENTRIES["E1"]})
    monkeypatch.setattr(sc, "EXITS", {"X1": sc.EXITS["X1"]})
    monkeypatch.setattr(sc, "HYGIENE", {"h_price": 5})
    elig = sc.daily_eligible(p)
    sc._setup(p, elig)

    class Serial:
        def map(self, fn, jobs, chunksize=1):
            return [fn(j) for j in jobs]

    sc.run_signal_tables(Serial(), tmp_path, lambda m: None)
    files = sorted((tmp_path / "signals").rglob("*.parquet"))
    assert [f.name for f in files] == ["E1__X1_signals.parquet"]  # E0 has no table
    sig = pd.read_parquet(files[0])
    assert set(sig[["xi", "yi"]].itertuples(index=False, name=None)) == set(sc.cells("E1", "X1"))
    assert {"exit_signal_date", "exit_reason", "closed"} <= set(sig.columns)
    sc.run_signal_tables(Serial(), tmp_path, lambda m: None)  # resumes: nothing left to do
    pe = replace(p, eligible=elig)
    cfg = replace(sc.config("a", "E1", "X1", 1, 1), max_positions=2)
    res = run_backtest(pe, cfg)
    cell = sig[(sig["xi"] == 1) & (sig["yi"] == 1)]
    eq, _ = sr.replay(pe, cfg, cell)
    np.testing.assert_allclose(eq.to_numpy(), res.equity.to_numpy(), rtol=0, atol=1e-9)
