"""Asset-allocation lab, Korean-ETF layer: KRW product returns, data files, presets, rebalancing calculator."""
import json

import numpy as np
import pandas as pd
import pytest

from lab import aa_notes
from lab.aa_assets import BENCHMARK_CODE, CASH, CLASS_ORDER, DEFAULT_PRODUCTS, FX_LABELS, KRW_CASH, PRESETS, PRODUCTS, USD_CASH, Product
from lab.aa_engine import AllocConfig, Overlay, current_targets, run_allocation
from lab.aa_korea import (
    KrwModel, _index_from_returns, krw_cash_returns, krw_panel, load_bok_rate, load_usdkrw, product_returns, splice,
)
from lab.aa_rebalance import load_holdings, rebalance_orders, save_holdings
from lab.data import panel_from_long

TD = 252


def usd_panel(closes: dict[str, np.ndarray], start="2010-01-04", cash_rate=0.0):
    dates = pd.bdate_range(start, periods=len(next(iter(closes.values()))))
    rows = [{"date": d, "symbol": s, "open": v, "high": v, "low": v, "close": v, "volume": 1}
            for s, c in closes.items() for d, v in zip(dates, c)]
    p = panel_from_long(pd.DataFrame(rows))
    p.cash = pd.Series(cash_rate / TD, index=p.close.index)
    return p, dates


def flat_bok(rate=3.0):
    return pd.Series([rate], index=[pd.Timestamp("2000-01-01")])


def prod(fx, leverage=1.0, ter=0.0, proxy=(("SPY", 1),), drag=0.0):
    return Product("X", "x", "주식", tuple(proxy), fx, ter, "2020-01-01", leverage=leverage, extra_drag=drag)


# --------------------------------------------------------------------------- return formulas

def frames(r=0.01, fx=0.002, c_usd=0.0001, c_krw=0.00012, n=5):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return (pd.DataFrame({"SPY": r}, index=idx), pd.Series(fx, index=idx), pd.Series(c_usd, index=idx),
            pd.Series(c_krw, index=idx))


def test_unhedged_hedged_and_native_formulas():
    rets, fx, cu, ck = frames()
    m = KrwModel(hedge_extra_bps=0.0)
    assert product_returns(prod("unhedged"), rets, fx, cu, ck, m).iloc[0] == pytest.approx((1.01) * (1.002) - 1)
    assert product_returns(prod("hedged"), rets, fx, cu, ck, m).iloc[0] == pytest.approx(0.01 - 0.0001 + 0.00012)
    # 2x KRW-native: leverage on the KRW return, financed at the KRW rate
    r_krw = 1.01 * 1.002 - 1
    assert product_returns(prod("krw_native", 2.0), rets, fx, cu, ck, m).iloc[0] == pytest.approx(2 * (r_krw - 0.00012) + 0.00012)
    # 2x unhedged: leverage in USD, then FX
    assert product_returns(prod("unhedged", 2.0), rets, fx, cu, ck, m).iloc[0] == pytest.approx((1 + 2 * (0.01 - 0.0001) + 0.0001) * 1.002 - 1)


def test_cash_like_products_fees_and_hedge_cost():
    rets, fx, cu, ck = frames()
    krw = product_returns(prod("krw", proxy=((KRW_CASH, 1),), ter=0.252), rets, fx, cu, ck)
    assert krw.iloc[0] == pytest.approx(0.00012 - 0.00252 / TD)
    usd = product_returns(prod("unhedged", proxy=((USD_CASH, 1),)), rets, fx, cu, ck)
    assert usd.iloc[0] == pytest.approx(1.0001 * 1.002 - 1)
    h0 = product_returns(prod("hedged"), rets, fx, cu, ck, KrwModel(hedge_extra_bps=0))
    h1 = product_returns(prod("hedged"), rets, fx, cu, ck, KrwModel(hedge_extra_bps=25.2))
    assert (h0 - h1).iloc[0] == pytest.approx(0.00252 / TD)
    drag = product_returns(prod("hedged", drag=2.52), rets, fx, cu, ck, KrwModel(hedge_extra_bps=0))
    assert (h0 - drag).iloc[0] == pytest.approx(0.0252 / TD)
    nofee = product_returns(prod("hedged", ter=1.0), rets, fx, cu, ck, KrwModel(hedge_extra_bps=0, fees=False))
    assert nofee.iloc[0] == pytest.approx(h0.iloc[0])


def test_blended_proxy_and_index_builder():
    idx = pd.bdate_range("2020-01-01", periods=4)
    rets = pd.DataFrame({"SPY": [np.nan, 0.01, 0.01, 0.01], "EFA": [np.nan, np.nan, 0.02, 0.0]}, index=idx)
    zero = pd.Series(0.0, index=idx)
    r = product_returns(prod("hedged", proxy=(("SPY", 0.7), ("EFA", 0.3))), rets, zero, zero, zero, KrwModel(hedge_extra_bps=0))
    assert np.isnan(r.iloc[1]) and r.iloc[2] == pytest.approx(0.7 * 0.01 + 0.3 * 0.02)
    px = _index_from_returns(r)
    assert np.isnan(px.iloc[0]) and px.iloc[1] == 100.0 and px.iloc[2] == pytest.approx(100 * (1 + 0.013))


# --------------------------------------------------------------------------- KRW panel

def test_krw_panel_builds_products_cash_and_benchmark():
    n = 300
    rng = np.random.default_rng(3)
    closes = {s: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)) for s in ("SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "QQQ", "IWM", "IYR")}
    usd, dates = usd_panel(closes, cash_rate=0.0252)
    fx = pd.Series(1000 * np.cumprod(1 + rng.normal(0, 0.004, n)), index=dates)
    panel, unavailable = krw_panel(usd, fx, flat_bok(3.15))
    assert panel.benchmark == BENCHMARK_CODE and panel.kind == "krw"
    assert set(unavailable) == {"069500", "122630"}  # EWY not in the panel
    assert "379800" in panel.symbols and "459580" in panel.symbols
    assert panel.cash.iloc[5] == pytest.approx(0.0315 / TD)
    # unhedged S&P500 in KRW = SPY x FX (no fee to speak of)
    krw = panel.close["379800"] / panel.close["379800"].iloc[0]
    expect = (closes["SPY"] / closes["SPY"][0]) * (fx.to_numpy() / fx.iloc[0])
    assert krw.iloc[-1] == pytest.approx(expect[-1], rel=1e-3)
    # the engine uses the panel's benchmark
    res = run_allocation(panel, AllocConfig("t", {"379800": 100}, str(dates[0].date()), str(dates[-1].date()), rebalance="none", cost_bps=0))
    assert res.equity.iloc[-1] == pytest.approx(res.benchmark.iloc[-1])


def test_krw_cash_before_first_rate_uses_first_level():
    idx = pd.bdate_range("1999-12-28", periods=6)
    bok = pd.Series([4.75, 5.0], index=pd.to_datetime(["2000-01-01", "2000-01-04"]))
    c = krw_cash_returns(idx, bok)
    assert c.iloc[0] == pytest.approx(0.0475 / TD) and c.loc["2000-01-04"] == pytest.approx(0.05 / TD)


def test_splice_scales_the_newer_days():
    base = pd.Series([1000.0, 1010.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    tail = pd.Series([1005.0, 1015.0, 1020.0], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))
    out = splice(base, tail)
    assert len(out) == 4 and out.iloc[-1] == pytest.approx(1020.0 * 1010.0 / 1005.0)
    assert splice(base, None) is base


# --------------------------------------------------------------------------- data files and catalog

def test_repo_fx_and_bok_files():
    fx = load_usdkrw()
    assert fx.index.is_monotonic_increasing and fx.index[0] <= pd.Timestamp("1993-01-31")
    assert fx.between(700, 2100).all()
    gaps = fx.index[fx.index >= "1995-01-01"].to_series().diff().dt.days
    assert gaps.max() <= 10
    bok = load_bok_rate()
    assert bok.index.is_monotonic_increasing and bok.index[0] <= pd.Timestamp("2000-01-01")
    assert bok.between(0.25, 6.0).all()
    assert bok.loc["2023-01-13"] == 3.5


def test_catalog_and_presets():
    for code, p in PRODUCTS.items():
        assert p.code == code and p.fx in FX_LABELS and p.asset_class in CLASS_ORDER
        assert p.leverage in (1.0, 2.0) and 0 <= p.ter < 1.0
        pd.Timestamp(p.listed)
        assert abs(sum(w for _, w in p.proxy) - 1.0) < 1e-9
    assert all(c in PRODUCTS for c in DEFAULT_PRODUCTS)
    for p in PRESETS.values():
        assert sum(p.weights.values()) == pytest.approx(100.0), p.key
        assert all(c in PRODUCTS or c == CASH for c in p.weights), p.key
    assert PRESETS["aaa_2026"].weights == {"379800": 30, "251350": 15, "291890": 10, "484790": 10, "0091C0": 25, "411060": 10}


# --------------------------------------------------------------------------- rebalancing calculator

def test_full_rebalance_hits_targets():
    o = rebalance_orders({"A": 700, "B": 300}, {"A": 50, "B": 50}, contribution=0, mode="full")
    assert o.set_index("code")["매매 금액"].to_dict() == pytest.approx({"A": -200, "B": 200})
    assert o["매매 후 비중"].to_numpy() == pytest.approx([0.5, 0.5])


def test_buy_only_never_sells_and_places_the_contribution():
    o = rebalance_orders({"A": 700, "B": 300}, {"A": 50, "B": 50}, contribution=100, mode="buy_only").set_index("code")
    assert (o["매매 금액"] >= 0).all() and o["매매 금액"].sum() == pytest.approx(100)
    assert o.loc["B", "매매 금액"] == pytest.approx(100)  # all of it to the underweight product
    # more money than the shortfall: fill the gap, the rest by target weight
    o2 = rebalance_orders({"A": 700, "B": 300}, {"A": 50, "B": 50}, contribution=1000, mode="buy_only").set_index("code")
    assert o2.loc["A", "매매 금액"] == pytest.approx(300) and o2.loc["B", "매매 금액"] == pytest.approx(700)


def test_band_and_shares_and_untargeted_holdings():
    inside = rebalance_orders({"A": 520, "B": 480}, {"A": 50, "B": 50}, contribution=0, mode="full", band=5)
    assert (inside["매매 금액"] == 0).all() and inside["계산 방식"].iloc[0] == "입금액만 배분"
    o = rebalance_orders({"A": 0, "C": 100}, {"A": 100}, contribution=1000, mode="full", prices={"A": 300}).set_index("code")
    assert o.loc["C", "구분"] == "매도" and o.loc["A", "주 수"] == 3 and o.loc["A", "매매 금액"] == 900
    with pytest.raises(ValueError):
        rebalance_orders({}, {"A": 100}, contribution=0)


def test_holdings_roundtrip(tmp_path):
    path = tmp_path / "h.json"
    assert load_holdings(path) == {}
    save_holdings({"379800": {"amount": 1000.0, "price": 20000.0}}, path)
    assert load_holdings(path)["holdings"]["379800"]["amount"] == 1000.0


# --------------------------------------------------------------------------- run log with configs, current targets

def test_run_log_keeps_config_and_migrates_old_file(tmp_path):
    path = tmp_path / "runs.csv"
    old_cols = [c for c in aa_notes.RUN_COLUMNS if c != "config"]
    pd.DataFrame([dict.fromkeys(old_cols, "x")]).to_csv(path, index=False)
    cfg = AllocConfig("t", {"379800": 60, "0091C0": 40}, "2010-01-01", "2020-01-01")
    aa_notes.log_run("t", cfg.describe(), ("2010-01-01", "2020-01-01"), {"cagr": 0.1}, path=path, config=cfg.to_dict())
    runs = aa_notes.load_runs(path)
    assert list(runs.columns) == aa_notes.RUN_COLUMNS and len(runs) == 2
    assert AllocConfig.from_dict(json.loads(runs["config"].iloc[-1])) == cfg


def test_current_targets_static_and_trend():
    n = 400
    up = 100 * np.cumprod(np.full(n, 1.001))
    down = 100 * np.cumprod(np.full(n, 0.999))
    usd, dates = usd_panel({"SPY": up, "TLT": down})
    panel, _ = krw_panel(usd, pd.Series(1000.0, index=dates), flat_bok(), products={c: PRODUCTS[c] for c in ("379800", "484790", "459580")})
    static = AllocConfig("s", {"379800": 60, "484790": 40}, str(dates[0].date()), str(dates[-1].date()))
    w, _ = current_targets(panel, static)
    assert w == pytest.approx({"379800": 0.6, "484790": 0.4})
    trend = AllocConfig("t", {"379800": 60, "484790": 40}, str(dates[0].date()), str(dates[-1].date()), overlay=Overlay("sma", 10, "459580"))
    w, why = current_targets(panel, trend)
    assert w.get("379800") == pytest.approx(0.6) and w.get("459580") == pytest.approx(0.4) and "484790" not in w
