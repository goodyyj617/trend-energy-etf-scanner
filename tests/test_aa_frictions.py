"""Asset-allocation lab, A41 stage 0: frictions switch (off = bit-identical), new products, real-price chains, KRX gold."""
import numpy as np
import pandas as pd
import pytest

from lab import aa_real
from lab import aa_search as srch
from lab.aa_assets import CASH, KRX_GOLD, PRODUCTS
from lab.aa_engine import AllocConfig, run_allocation
from lab.aa_frictions import DepositRule, Frictions, TaxRule, resolve
from lab.data import panel_from_long


def panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-02", periods=n)
    cols = {s: 100 * np.cumprod(1 + rng.normal(m, v, n)) for s, m, v in (("A", 3e-4, 0.012), ("B", 1e-4, 0.006), ("C", 2e-4, 0.02))}
    rows = [{"date": d, "symbol": s, "open": x, "high": x, "low": x, "close": x, "volume": 1} for s, c in cols.items() for d, x in zip(dates, c)]
    p = panel_from_long(pd.DataFrame(rows))
    p.cash = pd.Series(1e-4, index=p.close.index)
    p.benchmark = "A"
    return p


def cfg(**kw):
    base = dict(start="2004-01-01", end="2030-01-01", rebalance="M", cost_bps=10.0)
    base.update(kw)
    return AllocConfig("t", {"A": 50, "B": 30, "C": 15, CASH: 5}, **base)


def test_frictions_off_is_bit_identical_in_the_engine_and_the_static_simulator():
    p = panel()
    for c in (cfg(), cfg(band=3.0), cfg(rebalance="Q", rebalance_offset=4)):
        base = run_allocation(p, c)
        for fr in (None, Frictions(), Frictions(enabled=False, fee_bps={"A": 50})):
            r = run_allocation(p, c, frictions=fr)
            assert r.equity.equals(base.equity) and r.weights.equals(base.weights) and r.turnover == base.turnover
    W = np.array([[0.5, 0.3, 0.15, 0.05], [0.2, 0.2, 0.2, 0.4]])
    a = srch.StaticSim(p, ["A", "B", "C"], "2004-01-02", "2005-06-30").run(W)
    b = srch.StaticSim(p, ["A", "B", "C"], "2004-01-02", "2005-06-30", frictions=Frictions()).run(W)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_frictions_on_add_per_product_costs_and_uniform_costs_match_cost_bps():
    p = panel()
    zero = run_allocation(p, cfg(), frictions=Frictions(enabled=True))
    assert np.allclose(zero.equity, run_allocation(p, cfg()).equity, rtol=0, atol=1e-12)
    uni = Frictions(enabled=True, fee_bps={s: 15.0 for s in ("A", "B", "C", CASH)})
    assert np.allclose(run_allocation(p, cfg(), frictions=uni).equity, run_allocation(p, cfg(cost_bps=25.0)).equity, rtol=0, atol=1e-12)
    one = run_allocation(p, cfg(), frictions=Frictions(enabled=True, spread_bps={"C": 40.0}))
    assert one.equity.iloc[-1] < zero.equity.iloc[-1]  # only C's trades cost more
    W = np.array([[0.5, 0.3, 0.15, 0.05]])
    s_uni = srch.StaticSim(p, ["A", "B", "C"], "2004-01-02", "2005-06-30", frictions=uni).run(W)[0]
    s_25 = srch.StaticSim(p, ["A", "B", "C"], "2004-01-02", "2005-06-30", cost_bps=25.0).run(W)[0]
    assert np.allclose(s_uni, s_25, rtol=0, atol=1e-12)


def test_taxes_and_deposit_only_are_settings_that_refuse_to_run():
    assert resolve(Frictions(taxes=(TaxRule("isa", 2e6, 0.099),))) is None  # off: ignored
    with pytest.raises(NotImplementedError):
        resolve(Frictions(enabled=True, taxes=(TaxRule("isa", 2e6, 0.099),)))
    with pytest.raises(NotImplementedError):
        resolve(Frictions(enabled=True, deposits=DepositRule("M", 1.0, (KRX_GOLD,))))
    assert Frictions().account_of(KRX_GOLD) == "gold" and Frictions().account_of("379810") == "isa"
    assert Frictions(account={"379810": "general"}).account_of("379810") == "general"


def test_a41_products_are_registered_with_checked_facts():
    expect = {"0046A0": ("USD_CASH", 0.09, "2025-04-29"), "329750": ("SHY", 0.30, "2019-07-24"),
              "305080": ("IEF", 0.29, "2018-08-30"), "476760": ("TLT", 0.05, "2024-03-12"), KRX_GOLD: ("GLD", 0.0, "2014-03-24")}
    for code, (proxy, ter, listed) in expect.items():
        p = PRODUCTS[code]
        assert p.proxy[0][0] == proxy and p.ter == ter and p.listed == listed and p.fx == "unhedged" and p.leverage == 1.0
    assert PRODUCTS[KRX_GOLD].account == "gold" and all(p.account == "isa" for c, p in PRODUCTS.items() if c != KRX_GOLD)
    assert aa_real.SYN_REAL["SYN_TLT_U"] == "476760" and "(H)" in aa_real.EXTRA_CODES["453850"]
    assert {"0046A0", "329750", "305080", "476760", KRX_GOLD, "261240", "453850"} <= set(aa_real.REAL_CODES)


def model_panel(cols, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(next(iter(cols.values()))))
    rows = [{"date": d, "symbol": s, "open": v, "high": v, "low": v, "close": v, "volume": 0} for s, c in cols.items() for d, v in zip(dates, c)]
    p = panel_from_long(pd.DataFrame(rows))
    p.benchmark = next(iter(cols))
    return p, dates


def test_chains_are_opt_in_and_hand_over_in_order():
    n = 120
    model = 100 * 1.0005 ** np.arange(n)
    p, dates = model_panel({"379800": model, "0046A0": model.copy()})
    stand = pd.Series(1.0 + 0.002 * np.arange(n - 20), index=dates[20:] + pd.Timedelta(days=1))  # 261240, listed earlier
    own = pd.Series(5.0 + 0.01 * np.arange(n - 70), index=dates[70:] + pd.Timedelta(days=1))  # 0046A0 itself
    real = {"261240": stand, "0046A0": own}
    plain, info0 = aa_real.hybrid_panel(p, real)
    chained, info = aa_real.hybrid_panel(p, real, chains=aa_real.A41_CHAINS)
    assert info0["0046A0"]["chain"] == [("0046A0", info0["0046A0"]["splice"])]  # without chains: own prices only
    (c1, t1), (c2, t2) = info["0046A0"]["chain"]
    assert (c1, c2) == ("261240", "0046A0") and t1 < t2 and info["0046A0"]["splice"] == t2
    s = chained.close["0046A0"]
    assert np.allclose(s[s.index < t1], model[:len(s[s.index < t1])])  # model before the stand-in
    mid = s[(s.index >= t1) & (s.index < t2)]
    v1, _ = aa_real.to_us_grid(stand, p.close.index, same_day=False)
    assert np.allclose(mid / mid.iloc[0], v1[mid.index] / v1[t1])  # stand-in returns in between
    v2, _ = aa_real.to_us_grid(own, p.close.index, same_day=False)
    tail = s[s.index >= t2]
    assert np.allclose(tail / tail.iloc[0], v2[tail.index] / v2[t2]) and np.isclose(tail.iloc[0], s.loc[t2])
    assert plain.close["379800"].equals(chained.close["379800"])  # other columns untouched


def test_krx_gold_pages_parse_to_a_price_file_row_set():
    pages = [[{"localTradedAt": "2014-03-25T00:00:00+09:00", "closePrice": "47,100"},
              {"localTradedAt": "2014-03-24T00:00:00+09:00", "closePrice": "46,950"}],
             [{"localTradedAt": "2014-03-24T00:00:00+09:00", "closePrice": "46,950"}]]
    df = aa_real.parse_gold(pages)
    assert list(df["date"]) == ["2014-03-24", "2014-03-25"] and list(df["close"]) == [46950.0, 47100.0]
    assert set(df["code"]) == {KRX_GOLD} and list(df.columns) == ["code", "date", "open", "high", "low", "close", "volume"]
    tr = aa_real.real_series(pd.concat([df] * 1, ignore_index=True).assign(date=lambda d: pd.to_datetime(d["date"])),
                             pd.DataFrame(columns=["code", "date", "close", "dividends"]))
    assert tr == {}  # two closes are fewer than SPLICE_LAG + 1: too short to splice
