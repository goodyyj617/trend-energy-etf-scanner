"""A42 stage 0: price sets, Korean rates and bonds, quirks, calibration, sealed window (offline, synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from lab import aa_pricesets as PS
from lab import aa_real, aa_sealed
from lab.aa_assets import ADDED_A42, KTB3, KTB10, KTB30, PRODUCTS, _p
from lab.aa_korea import KTB_DURATION, KrwModel, krw_cash_returns, krw_panel, ktb_returns, product_returns
from lab.data import panel_from_long


def usd_panel(n=300, start="2015-01-02", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    syms = {"SPY": 3e-4, "QQQ": 4e-4, "TLT": 1e-4, "IEF": 1e-4, "GLD": 2e-4}
    rows = [{"date": d, "symbol": s, "open": v, "high": v, "low": v, "close": v, "volume": 1}
            for s, m in syms.items() for d, v in zip(dates, 100 * np.cumprod(1 + rng.normal(m, 0.01, n)))]
    p = panel_from_long(pd.DataFrame(rows))
    p.cash = pd.Series(1e-4, index=p.close.index)
    return p


def rates_frame(months):
    idx = pd.date_range(months[0], months[1], freq="MS")
    k = np.arange(len(idx))
    return pd.DataFrame({"cd91": 3 + 0.01 * k, "ktb3": 3 + 0.02 * k, "ktb5": 3.1 + 0.02 * k, "ktb10": 3.2 + 0.02 * k,
                         "ktb20": np.nan, "ktb30": 3.3 + 0.02 * k}, index=idx)


def test_ktb_returns_are_carry_minus_duration_times_the_yield_step():
    idx = pd.bdate_range("2015-01-02", "2015-04-30")
    r = ktb_returns(idx, rates_frame(("2015-01-01", "2015-04-01")))
    feb = idx[idx.searchsorted(pd.Timestamp("2015-02-01"))]
    day = idx[idx.get_loc(feb) + 1]
    assert np.isclose(r.loc[day, KTB3], 0.0302 / 252)  # inside a month: carry only
    assert np.isclose(r.loc[feb, KTB3], 0.03 / 252 - KTB_DURATION[KTB3] * 0.0002)  # first day: the yield steps
    assert np.isclose(r.loc[feb, KTB30], 0.033 / 252 - KTB_DURATION[KTB30] * 0.0002)
    assert np.isnan(r[KTB10].iloc[0])  # no previous yield yet


def test_krw_panel_with_rates_keeps_old_columns_and_adds_bond_products():
    usd = usd_panel()
    fx = pd.Series(1200.0, index=usd.close.index)
    bok = pd.Series([1.5], index=[pd.Timestamp("2014-01-01")])
    prods = {c: PRODUCTS[c] for c in ("379800", "379810", "484790", "114260", "148070", "439870")}
    old, una_old = krw_panel(usd, fx, bok, products=prods, rates=None)
    new, una_new = krw_panel(usd, fx, bok, products=prods, rates=rates_frame(("2014-12-01", "2016-03-01")))
    assert set(una_old) == {"114260", "148070", "439870"} and not una_new
    for c in ("379800", "379810", "484790"):
        assert old.close[c].equals(new.close[c])
    assert new.close["114260"].notna().sum() > 200


def test_hedge_bp_and_cash_choice_reach_the_product_returns():
    usd = usd_panel()
    idx = usd.close.index
    rets = usd.close.pct_change(fill_method=None)
    zero = pd.Series(0.0, index=idx)
    kc = krw_cash_returns(idx, pd.Series([1.5], index=[pd.Timestamp("2014-01-01")]))
    p = PRODUCTS["484790"]
    a = product_returns(p, rets, zero, usd.cash, kc, KrwModel())
    b = product_returns(p, rets, zero, usd.cash, kc, KrwModel(hedge_extra_bps=20.0))
    c = product_returns(p, rets, zero, usd.cash, kc, KrwModel(hedge_extra_bps=60.0))
    assert a.equals(b) and np.allclose((b - c).dropna(), 40 / 1e4 / 252)
    cd = krw_cash_returns(idx, pd.Series([3.0], index=[pd.Timestamp("2014-01-01")]))
    diff = (product_returns(p, rets, zero, usd.cash, cd) - a).dropna()
    assert np.allclose(diff, (cd - kc).loc[diff.index])


def test_price_sets_are_24_and_the_default_is_the_old_setting():
    sets = PS.all_sets()
    assert len(sets) == 24 and len({s.label for s in sets}) == 24
    assert PS.PriceSet().label == "real·h20·base"
    assert PS.nearest_grid(69.9) == 60.0 and PS.nearest_grid(10.0) == 20.0 and PS.nearest_grid(-5) == 0.0
    with pytest.raises(ValueError):
        PS.build("other")


def test_quirks_are_opt_in_and_interpolate_the_bad_close():
    dates = pd.bdate_range("2019-03-11", periods=9).strftime("%Y-%m-%d")
    close = [100.0, 100.0, 122.0, 100.5, 100.5, 100.5, 100.6, 100.7, 100.8]
    nav = pd.DataFrame({"code": "261240", "date": dates, "close": close})
    nav = pd.concat([nav, pd.DataFrame({"code": "X", "date": dates, "close": close})])
    yah = pd.DataFrame(columns=["code", "date", "close", "dividends"])
    q = ({"code": "261240", "dates": (dates[2],), "kind": "interp", "reason": "t"},)
    plain, fixed = aa_real.real_series(nav, yah), aa_real.real_series(nav, yah, quirks=q)
    assert plain["X"].equals(fixed["X"]) and plain["261240"].iloc[2] > 1.2
    assert np.isclose(fixed["261240"].iloc[2], np.sqrt(fixed["261240"].iloc[1] * fixed["261240"].iloc[3]))
    assert {x["code"] for x in aa_real.QUIRKS} == {"261240", "069500", "*"}


def test_calibration_and_adjusted_model_change_only_the_model_part():
    dates = pd.bdate_range("2018-01-02", periods=800)
    m = 100 * 1.0003 ** np.arange(800)
    rows = [{"date": d, "symbol": "A", "open": v, "high": v, "low": v, "close": v, "volume": 0} for d, v in zip(dates, m)]
    model = panel_from_long(pd.DataFrame(rows))
    real = pd.Series(50 * 1.0004 ** np.arange(500), index=dates[300:] + pd.Timedelta(days=1))
    cal = PS.calibration(model, {"A": real}, chains={}, end="2030-12-31")
    row = cal.iloc[0]
    assert row["보정에 씀"] and np.isclose(row["연 차이 (실제−모형)"], 1e-4 * 252, rtol=0.05)
    _, info = aa_real.hybrid_panel(model, {"A": real}, chains={})
    adj = PS.adjusted_model(model, cal, info)
    t0 = info["A"]["splice"]
    r_adj, r_m = adj.close["A"].pct_change(), model.close["A"].pct_change()
    before = r_adj.index < t0
    assert np.allclose((r_adj - r_m)[before].dropna(), row["연 차이 (실제−모형)"] / 252)
    short = PS.calibration(model, {"A": real}, chains={}, end=str(dates[350].date()))
    assert not short.iloc[0]["보정에 씀"]  # the overlap stops at `end`: fewer than 252 days


def test_sealed_window_needs_the_unseal_flag(tmp_path, monkeypatch):
    with pytest.raises(aa_sealed.SealedError):
        aa_sealed.guard("2000-08-31")
    aa_sealed.guard("2000-09-01")
    aa_sealed.guard("1995-01-02", unseal=True)
    monkeypatch.setattr(aa_sealed, "SEALED_DIR", tmp_path)
    (tmp_path / "f.csv").write_text("date,x\n1995-01-02,1\n", encoding="utf-8")
    import hashlib
    h = tmp_path / "h.csv"
    pd.DataFrame([{"file": "f.csv", "sha256": hashlib.sha256((tmp_path / "f.csv").read_bytes()).hexdigest()}]).to_csv(h, index=False)
    monkeypatch.setattr(aa_sealed, "HASHES", h)
    with pytest.raises(aa_sealed.SealedError):
        aa_sealed.read("f.csv")
    assert aa_sealed.read("f.csv", unseal=True)["x"].tolist() == [1]
    (tmp_path / "f.csv").write_text("date,x\n1995-01-02,2\n", encoding="utf-8")
    with pytest.raises(aa_sealed.SealedError):
        aa_sealed.read("f.csv", unseal=True)  # changed after the hash was recorded
    with pytest.raises(aa_sealed.SealedError):
        PS.usd_extended("1999-01-04")


def test_a42_products_are_registered_with_checked_facts():
    expect = {"114260": (KTB3, 0.15, "2009-07-29"), "148070": (KTB10, 0.05, "2011-10-20"),
              "439870": (KTB30, 0.05, "2022-08-23"), "446720": ("SCHD", 0.01, "2022-11-15")}
    assert set(expect) == set(ADDED_A42)
    for code, (proxy, ter, listed) in expect.items():
        p = PRODUCTS[code]
        assert p.proxy[0][0] == proxy and p.ter == ter and p.listed == listed
    assert PRODUCTS["446720"].fx == "unhedged" and all(PRODUCTS[c].fx == "krw" for c in ("114260", "148070", "439870"))
    assert "SCHD" in PRODUCTS["446720"].proxy_symbols and not PRODUCTS["114260"].proxy_symbols
    assert set(PS.HEDGE_PROBES) == {"267440", "304660"} and {"267440", "304660"} <= set(aa_real.REAL_CODES)
    assert isinstance(_p, type(lambda: 0))
