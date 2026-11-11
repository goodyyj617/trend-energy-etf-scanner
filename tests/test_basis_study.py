"""Stage R (lab/basis_study.py, DECISIONS 98): raw-price basis for the price and dollar-volume filters."""
import os

import numpy as np
import pandas as pd
import pytest

from lab import basis_study as bs, crsp, pick_study as ph, screens, stage_c_study as sc
from lab.data import Panel, has_raw, level_close, level_dollar_volume, with_basis
from lab.engine import SignalCache, StrategyConfig, run_backtest


def test_registered():
    assert bs.START == "1993-02-01" and bs.END == "2023-12-31"
    assert bs.GENS == {"E7": ["E7"], "E0": ["R0s1", "R0s2", "R0s3", "R0s4", "R0s5"]}
    assert bs.EXITS == ["X1", "H1"] and bs.FILTERS == ["none", "R3A1"] and bs.COSTS == (20.0, 40.0)
    assert bs.SETTINGS == [(10, "W2", "P0"), (10, "W2", "P1"), (20, "W2", "P1"), (50, "W2", "P1")]
    assert list(bs.BANDS) == ["U3000", "U1000", "U500"]
    assert (bs.SMALL_PP, bs.LARGE_PP) == (0.002, 0.005) and bs.SIZE_PERIODS == ("1993-2023", "1993-2007", "2008-2023")
    assert bs.CANDIDATE == ("E7", "R3A1") and len(bs.CANDIDATE_CELLS) == 4 and bs.CANDIDATES_TO_PROCEED == 3
    assert bs.E7_VALUE == 20 and bs.FWD == 252 and bs.LEVEL_SCREENS == ("h_price", "h_liquidity", "h_illiquid")
    assert sc.HYGIENE == {"h_history": 12, "h_price": 5, "h_liquidity": 1, "h_illiquid": 20}
    assert crsp.DEFAULT_BASIS == os.environ.get("LAB_PRICE_BASIS", "raw")  # raw since Stage R


def _panel(n=320, seed=1, raw=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-01", periods=n)
    syms = ["A", "B", "C", "D", "SPY"]
    close = pd.DataFrame(np.cumprod(1 + rng.normal(0.0008, 0.02, (n, len(syms))), axis=0), index=idx, columns=syms)
    close *= np.array([3.0, 8.0, 40.0, 4.0, 100.0])
    close.iloc[:30, 2] = np.nan  # C lists later
    opn = close * (1 + rng.normal(0, 0.003, close.shape))
    vol = pd.DataFrame(rng.uniform(2e5, 2e6, close.shape), index=idx, columns=syms).where(close.notna())
    p = Panel(opn, np.maximum(opn, close) * 1.01, np.minimum(opn, close) * 0.99, close, vol, tradable=syms[:-1],
              cash=pd.Series(0.0001, index=idx), kind="crsp", eligible=close.notna() & (close.columns != "SPY"),
              extra={"cap": close * 10, "cap_rank": close.rank(axis=1, ascending=False).astype("float32")})
    if raw is not None:
        p.extra.update(raw(close, vol))
    return p


def _old_screens(p):
    """The hygiene screens as written before Stage R."""
    dv = (p.close * p.volume)
    r = p.close.pct_change(fill_method=None).abs()
    amihud = (r / dv.replace(0, np.nan)).rolling(63, min_periods=31).mean()
    return {"h_price": p.close < 5, "h_liquidity": dv.rolling(21, min_periods=21).mean() < 1e6,
            "h_illiquid": amihud.rank(axis=1, pct=True) > 0.8}


def test_without_raw_series_nothing_changes():
    p = _panel()
    assert not has_raw(p) and level_close(p) is p.close
    pd.testing.assert_frame_equal(level_dollar_volume(p), p.close * p.volume)
    old = _old_screens(p)
    for k, v in (("h_price", 5), ("h_liquidity", 1), ("h_illiquid", 20)):
        pd.testing.assert_frame_equal(screens.ALL_SCREENS[k].compute(p, v), old[k])
    assert with_basis(p, "raw") is p and with_basis(p, "adjusted") is p
    with pytest.raises(ValueError):
        with_basis(p, "split")


def _signals(p, **kw):
    cfg = StrategyConfig(entries={"above_ma": 10}, exits={"trailing_pct": 25}, start="2001-01-01", end="2002-12-31",
                         cost_bps=20.0, min_price=5.0, min_dollar_volume=1e6, **kw)
    return run_backtest(p, cfg, SignalCache(p), outcomes=True)


def test_adjusted_basis_is_bit_identical():
    """Raw series equal to the adjusted ones give the same filters and signals; basis 'adjusted' drops
    any raw series, so the engine and the screens see the panel as before."""
    p = _panel()
    same = _panel(raw=lambda c, v: {"raw_close": c.copy(), "raw_volume": v.copy()})
    assert has_raw(same)
    pd.testing.assert_frame_equal(level_close(same), p.close)
    pd.testing.assert_frame_equal(level_dollar_volume(same), (p.close * p.volume).fillna(0.0).where(p.close.notna()))
    pd.testing.assert_frame_equal(_signals(same), _signals(p))
    pd.testing.assert_frame_equal(sc.daily_eligible(same), sc.daily_eligible(p))
    different = _panel(raw=lambda c, v: {"raw_close": c * 7.0, "raw_volume": v * 0.01})
    adj = with_basis(different, "adjusted")
    assert not has_raw(adj) and has_raw(different) and "cap" in adj.extra
    pd.testing.assert_frame_equal(_signals(adj), _signals(p))
    pd.testing.assert_frame_equal(sc.daily_eligible(adj), sc.daily_eligible(p))


def test_raw_basis_changes_only_the_level_filters():
    """A (adjusted 3-ish) and D (4-ish) fail the $5 floor on adjusted prices; with raw prices x 3 they
    pass. The new signals are the same stock-days with the same outcomes as with no price floor."""
    p = _panel()
    r = _panel(raw=lambda c, v: {"raw_close": c * 3.0, "raw_volume": v / 3.0})  # dollar volume unchanged
    a, b = _signals(p), _signals(r)
    assert not a["symbol"].isin(["A", "D"]).any() and b["symbol"].isin(["A", "D"]).any()
    free = run_backtest(p, StrategyConfig(entries={"above_ma": 10}, exits={"trailing_pct": 25}, start="2001-01-01", end="2002-12-31",
                                          cost_bps=20.0, min_price=0.0, min_dollar_volume=1e6), SignalCache(p), outcomes=True)
    pd.testing.assert_frame_equal(b.reset_index(drop=True), free[free["symbol"].isin(b["symbol"].unique())].reset_index(drop=True))
    common = a.merge(b, on=["symbol", "signal_date"], suffixes=("", "_r"))
    assert len(common) == len(a) and np.allclose(common["return"], common["return_r"], equal_nan=True)
    # the dollar volume filter reads raw close x raw volume
    thin = _panel(raw=lambda c, v: {"raw_close": c * 3.0, "raw_volume": v * 0.001})
    assert _signals(thin).empty
    ep, er = sc.daily_eligible(p), sc.daily_eligible(r)
    assert not ep[["A", "D"]].any().any() and er[["A", "D"]].any().all()
    assert ep["B"].equals(er["B"]) and ep["C"].equals(er["C"])


def test_level_frames_on_days_without_a_raw_price():
    def raw(c, v):
        rc = c.copy()
        rc.iloc[100:103, 1] = np.nan  # no CRSP price on 3 listed days
        return {"raw_close": rc, "raw_volume": v.copy()}
    p = _panel(raw=raw)
    lc, dv = level_close(p), level_dollar_volume(p)
    assert (lc.iloc[100:103, 1] == p.close.iloc[99, 1]).all()  # carried forward while listed
    assert (dv.iloc[100:103, 1] == 0.0).all()  # no dollar volume (the adjusted volume is 0 there too)
    assert lc["C"].iloc[:30].isna().all() and dv["C"].iloc[:30].isna().all()  # not listed: none


def test_eligible_parts_is_daily_eligible():
    for p in (_panel(), _panel(raw=lambda c, v: {"raw_close": c * 2.0, "raw_volume": v})):
        ok, parts = bs.eligible_parts(p)
        pd.testing.assert_frame_equal(ok, sc.daily_eligible(p))
        assert set(parts) == set(sc.HYGIENE)


def _crsp_rows(permno, dates, p0, r, shrout, split_cf=1.0, split_day=None, shrcd="11", exchcd="1", ticker="X"):
    rows = []
    for t, d in enumerate(dates):
        adj = p0 * (1 + r) ** t
        cf = split_cf if (split_day is not None and t < split_day) else 1.0
        prc = adj * cf
        rows.append({"PERMNO": permno, "date": d, "SHRCD": shrcd, "EXCHCD": exchcd, "TICKER": ticker, "SICCD": 3570, "PRC": prc,
                     "VOL": 50_000.0 / cf * (1 + t % 3), "RET": r, "RETX": r, "SHROUT": shrout / cf, "CFACPR": cf, "OPENPRC": prc * 0.99,
                     "BIDLO": prc * 0.98, "ASKHI": prc * 1.02, "DLSTCD": np.nan, "DLRET": np.nan})
    return rows


def test_crsp_raw_series_and_basis_loading(tmp_path, monkeypatch):
    """A 4:1 split late in the sample: before it the adjusted close is a quarter of the traded price, so
    the $5 floor excludes the stock on adjusted prices but not on raw prices."""
    dates = pd.bdate_range("2001-01-02", "2002-12-31")
    rows = _crsp_rows("1", dates, 3.0, 0.0005, 100_000.0, split_cf=4.0, split_day=400, ticker="SPL")  # traded ~12, adjusted ~3
    rows += _crsp_rows("2", dates, 20.0, 0.0003, 50_000.0, ticker="BIG")
    rows += _crsp_rows(crsp.SPY_PERMNO, dates, 100.0, 0.0003, 1.0, shrcd="73", ticker="SPY")
    src = tmp_path / "src"
    src.mkdir()
    pd.DataFrame(rows).to_parquet(src / "CRSP_20010101_20051231.parquet")
    panel, info, meta = crsp.build_panel(src, top_n=5, daily=True, log=lambda m: None)
    crsp.save_panel(panel, info, meta, tmp_path / "panel")
    chk = crsp.build_raw(src, tmp_path / "panel", log=lambda m: None)
    assert chk["cap_mismatch"] == 0 and chk["cap_only_panel"] == 0 and chk["raw_close_off_listed"] == 0
    adj = crsp.load_panel(tmp_path / "panel", "adjusted")
    raw = crsp.load_panel(tmp_path / "panel", "raw")
    assert not has_raw(adj) and has_raw(raw)
    monkeypatch.setattr(crsp, "DEFAULT_BASIS", "raw")
    assert has_raw(crsp.load_panel(tmp_path / "panel"))  # the default loads the raw series when they exist
    assert not has_raw(crsp.load_panel(tmp_path / "panel", "adjusted"))
    d0 = dates[100]
    assert raw.extra["raw_close"].loc[d0, "1"] == pytest.approx(4 * adj.close.loc[d0, "1"], rel=1e-5)  # pre-split
    assert raw.extra["raw_close"].loc[dates[-1], "1"] == pytest.approx(adj.close.loc[dates[-1], "1"], rel=1e-5)
    assert raw.extra["raw_volume"].loc[d0, "1"] == pytest.approx(adj.volume.loc[d0, "1"] / 4, rel=1e-5)
    assert raw.extra["raw_close"].loc[d0, "SPY"] == pytest.approx(adj.close.loc[d0, "SPY"], rel=1e-5)
    pa, pr = screens.ALL_SCREENS["h_price"].compute(adj, 5), screens.ALL_SCREENS["h_price"].compute(raw, 5)
    assert pa.loc[d0, "1"] and not pr.loc[d0, "1"] and not pa.loc[d0, "2"] and not pr.loc[d0, "2"]
    # dollar volume is split-invariant: the liquidity screens differ only by the dividend factor (none here)
    la = level_dollar_volume(adj)
    lr = level_dollar_volume(raw)
    assert np.allclose(la["1"].dropna(), lr["1"].reindex(la["1"].dropna().index), rtol=1e-5)
    # one file missing: neither raw series is loaded
    (tmp_path / "panel" / "extra_raw_volume.parquet").unlink()
    assert not has_raw(crsp.load_panel(tmp_path / "panel", "raw"))
    monkeypatch.setattr(crsp, "DEFAULT_BASIS", "adjusted")
    assert crsp.resolve_basis(None) == "adjusted" and crsp.resolve_basis("raw") == "raw"
    with pytest.raises(ValueError):
        crsp.resolve_basis("other")


def test_table_menus_equal_the_stored_tables_menus(monkeypatch):
    for name in ("ENTRIES", "ENTRY_CANON", "EXITS", "EXIT_CANON", "EXIT_EXTRA", "EXIT_BREAKEVEN", "SAMPLES"):
        monkeypatch.setattr(sc, name, dict(getattr(sc, name)))
    full = sc.config("full", "E7", "X1", sc.ENTRY_CANON["E7"], 1)  # the Stage C v2 table's cell
    bs.table_menu("E7__X1", None)
    assert sc.config("full", "E7", "X1", 0, 0) == full
    ph.use_h1_menu()
    h1 = sc.config("full", "E7", "H1", 0, 2)  # Stage H's H1 table at k = 3
    bs.table_menu("E7__H1", None)
    assert sc.config("full", "E7", "H1", 0, 0) == h1
    sc.ENTRIES.clear()
    sc.ENTRIES["R0"] = ("g_random", [1, 2, 3, 4, 5])
    sc.EXITS.clear()
    sc.EXITS["X1"] = ("trailing_pct", [25])
    r0x1 = sc.config("full", "R0", "X1", 2, 0)
    sc.EXITS["H1"] = ("atr_stop", list(ph.K_VALUES))
    sc.EXIT_EXTRA["H1"] = {"trailing_pct": ph.H1_TRAIL}
    r0h1 = sc.config("full", "R0", "H1", 2, 2)
    bs.table_menu("R0__X1", 3)
    assert sc.config("full", "R0", "X1", 0, 0) == r0x1
    bs.table_menu("R0__H1", 3)
    assert sc.config("full", "R0", "H1", 0, 0) == r0h1
    assert list(sc.config("full", "R0", "H1", 0, 0).exits) == ["atr_stop", "trailing_pct"]
    assert [bs.TABLES[n][2](s) for n, s in (("E7__X1", None), ("E7__H1", None), ("R0__X1", 1), ("R0__H1", 5))] == [1, 0, 0, 4]
    assert [bs.TABLES[n][3] for n in bs.TABLES] == [1, 2, 0, 2]


def test_size_verdict():
    v = bs.size_verdict({"a": {"x": 0.0019, "y": -0.001}, "b": {"x": 0.0}, "c": {"x": -0.0015}})
    assert v["label"] == "작음" and v["largest_abs"] == pytest.approx(0.0019)
    assert bs.size_verdict({"a": {"x": 0.002}, "b": {"x": 0.0}, "c": {"x": 0.0}})["label"] == "중간"
    assert bs.size_verdict({"a": {"x": 0.001}, "b": {"x": -0.005}, "c": {"x": 0.0}})["label"] == "큼"


def test_split_signals():
    k = dict(symbol=["a", "b", "c", "a"], signal_date=pd.to_datetime(["2001-01-02", "2001-01-02", "2001-01-03", "2001-01-04"]))
    adj = pd.DataFrame({**k, "seed": [1, 1, 1, 2], "return": [0.1, 0.2, 0.3, 0.4]})
    raw = pd.DataFrame({"symbol": ["a", "c", "d", "a"], "signal_date": pd.to_datetime(["2001-01-02", "2001-01-03", "2001-01-03", "2001-01-04"]),
                        "seed": [1, 1, 1, 1], "return": [0.1, 0.3, 0.5, 0.6]})
    s = bs.split_signals({"adjusted": adj, "raw": raw})
    assert sorted(s["new"]) == [0.5, 0.6] and sorted(s["common"]) == [0.1, 0.3] and sorted(s["adj_only"]) == [0.2, 0.4]
    assert s["common_max_diff"] == 0.0


def test_compare_frames_and_cut():
    idx = pd.bdate_range("2023-12-27", "2024-01-03")
    a = pd.DataFrame({"x": [True, False, True, True, False, True]}, index=idx)
    c = bs.cut(a)
    assert c.index.max() <= pd.Timestamp("2023-12-31") and len(c) == 3
    b = c.copy()
    b.iloc[0, 0] = False
    assert bs.compare_frames(c, c)["differ"] == 0 and bs.compare_frames(c, b)["differ"] == 1


def test_relative_counts_same_sign_in_both_halves():
    keys = []
    for st in bs.SETTINGS:
        for x in bs.EXITS:
            for f in bs.FILTERS:
                for band in bs.BANDS:
                    for gen in ("E7", "E0"):
                        keys.append({"setting": f"N{st[0]}_{st[1]}_{st[2]}", "gen": gen, "exit": x, "filter": f, "band": band,
                                     "cost": 20.0, "mode": "grid"})
    g = pd.DataFrame(keys)
    for p in ("1993-2023", "1993-2007", "2008-2023"):
        g[f"dd_cagr_{p}"] = np.where(g["gen"] == "E7", 0.06, 0.05)
    raw = g.copy()
    flip = (raw["gen"] == "E7") & (raw["setting"] == "N10_W2_P0") & (raw["exit"] == "X1")
    raw.loc[flip, "dd_cagr_2008-2023"] = 0.04  # E7 below E0 in one half on raw prices
    rel = bs.relative({"adjusted": g, "raw": raw})
    assert len(rel) == 48 and int(rel["same_sign_halves"].sum()) == 48 - 6
    main = rel[(rel["filter"] == "none") & (rel["band"] == "U3000")]
    assert len(main) == 8 and int(main["same_sign_halves"].sum()) == 7
