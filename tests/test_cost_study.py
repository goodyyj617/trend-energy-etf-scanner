"""Stage T (DECISIONS 100): the EDGE transcription, the per-fill cost lookup and the cost table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab import cost_study as ct
from lab import goal_study as gs
from lab import large_study as ls
from lab import pick_study as ph
from lab import regime_study as rs
from lab.edge_spread import edge, edge_rows


# --------------------------------------------------------------------------- EDGE

def _ohlc(rng, G, T, missing=0.0):
    p = np.exp(np.cumsum(rng.normal(0, 0.02, (G, T * 4)), axis=1)).reshape(G, T, 4) * 20
    o, h, l_, c = p[:, :, 0].copy(), p.max(2), p.min(2), p[:, :, 3].copy()
    h = h * np.exp(np.abs(rng.normal(0, 0.003, (G, T))))
    l_ = l_ * np.exp(-np.abs(rng.normal(0, 0.003, (G, T))))
    if missing:
        miss = rng.random((G, T)) < missing
        for x in (o, h, l_, c):
            x[miss] = np.nan
        o[rng.random((G, T)) < missing / 2] = np.nan  # open alone missing on some days
    return o, h, l_, c


def test_edge_rows_equals_reference():
    rng = np.random.default_rng(0)
    for missing in (0.0, 0.15):
        o, h, l_, c = _ohlc(rng, 120, 63, missing)
        o[0, :] = np.nan  # a group with no opens -> NaN in both
        h[1, :61] = l_[1, :61] = c[1, :61] = np.nan  # too few days
        for sign in (False, True):
            v = edge_rows(o, h, l_, c, sign)
            r = np.array([edge(o[i], h[i], l_[i], c[i], sign) for i in range(len(o))])
            assert np.array_equal(v, r, equal_nan=True)
    assert np.isnan(edge_rows(o[:, :2], h[:, :2], l_[:, :2], c[:, :2])).all()


def test_edge_recovers_a_known_spread():
    """Efficient log price a random walk observed 100 times a day; every trade at the ask or the bid
    with probability 1/2; OHLC from the trades. EDGE over 4,000 days recovers the 1% spread."""
    rng = np.random.default_rng(7)
    days, per, spread = 4000, 100, 0.01
    lp = np.cumsum(rng.normal(0, 0.02 / np.sqrt(per), days * per)).reshape(days, per)
    side = np.where(rng.random((days, per)) < 0.5, 1.0, -1.0)
    tp = np.exp(lp + side * spread / 2)
    est = edge(tp[:, 0], tp.max(1), tp.min(1), tp[:, -1])
    assert abs(est - spread) < 0.15 * spread
    signed = edge_rows(tp[None, :, 0], tp.max(1)[None], tp.min(1)[None], tp[None, :, -1], sign=True)[0]
    assert abs(signed - est) < 1e-12


# --------------------------------------------------------------------------- buckets and codes

def test_buckets():
    assert ct.cap_bucket(np.array([1, 500, 501, 1000, 1001, 2000, 2001, 3000, 3001, np.nan])).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert ct.price_bucket(np.array([0.5, 4.99, 5.0, 9.99, 10.0, 20.0, 49.9, 50.0, 900.0, np.nan, 0.0])).tolist() == \
        [0, 0, 1, 1, 2, 3, 3, 4, 4, 255, 255]


def test_fill_codes_use_the_previous_day():
    cap = np.array([[10.0, 600, np.nan], [2500.0, 4000, np.nan], [np.nan, 100, np.nan]])
    raw = np.array([[3.0, np.nan, 300.0], [60.0, 12.0, np.nan], [np.nan, np.nan, 310.0]])
    code = ct.fill_codes(cap, raw, spy_col=2)
    assert code[0].tolist() == [ct.CODE_POOLED + 4] * 3
    # day 1 uses day 0: rank 10 price 3 -> (0, 0); rank 600 no price -> pooled cap 1; SPY cap 0 price 300 -> (0, 4)
    assert code[1].tolist() == [0, ct.CODE_POOLED + 1, 4]
    # day 2 uses day 1: rank 2500 price 60 -> (3, 4); rank 4000 price 12 -> (4, 2); SPY price carried 300 -> (0, 4)
    assert code[2].tolist() == [3 * 5 + 4, 4 * 5 + 2, 4]


def test_cell_cost_lookup():
    code = np.array([[0, 1], [2, 25], [3, 4]], dtype=np.uint8)
    yi = np.array([0, 0, 1], dtype=np.int16)
    tab = np.arange(2 * ct.N_CODES, dtype=float).reshape(2, ct.N_CODES) / 1e4
    cc = ct.CellCost(code, yi, tab, w0=1)
    assert cc.at(0, 1) == tab[0, 25] and cc.at(1, 0) == tab[1, 3]
    assert np.array_equal(cc.take(np.array([0, 1]), np.array([0, 1])), np.array([tab[0, 2], tab[1, 4]]))
    assert np.array_equal(cc.dense(), tab[yi[:, None], code])
    assert np.array_equal(cc.column(1), np.array([tab[0, 1], tab[0, 25], tab[1, 4]]))


def _const(n, m, cost, w0=0):
    return ct.CellCost(np.zeros((n + w0, m), dtype=np.uint8), np.zeros(n + w0, dtype=np.int16), np.full((1, ct.N_CODES), cost), w0)


# --------------------------------------------------------------------------- accounts: number vs table

def _signals(rng, n, m, k):
    ei = rng.integers(1, n - 5, k)
    sale = np.minimum(ei + rng.integers(1, 40, k), n + 3)
    closed = sale < n
    return ei - 1, ei, rng.integers(0, m, k), sale, closed, rng.uniform(0.02, 0.2, k), rng.random(k)


def test_replay_risk_number_equals_constant_table():
    rng = np.random.default_rng(3)
    n, m = 300, 40
    O = np.exp(np.cumsum(rng.normal(0, 0.02, (n, m)), axis=0)) * 30
    O[rng.random((n, m)) < 0.03] = np.nan
    LC = pd.DataFrame(O * np.exp(rng.normal(0, 0.01, (n, m)))).ffill().to_numpy()
    cash = np.full(n, 0.0001)
    si, ei, sj, sale, closed, w, key = _signals(rng, n, m, 900)
    psale = np.where(rng.random(900) < 0.3, np.minimum(ei + 3, n - 1), -1)
    force = [100, 200]
    for kw in ({}, {"psale": psale}, {"force_exit": force}, {"slots": 5, "ban": 0}):
        e1, t1 = ph.replay_risk(si, ei, sj, sale, closed, w, key, O, LC, cash, 0.002, **kw)
        fills: list = []
        e2, t2 = ph.replay_risk(si, ei, sj, sale, closed, w, key, O, LC, cash, _const(n, m, 0.002, w0=4), fills=fills, **kw)
        assert np.array_equal(e1, e2)
        if "psale" in kw:  # a partial sale's own cost enters the trade's return in another order of operations
            assert t1.drop(columns="ret").equals(t2.drop(columns="ret")) and np.allclose(t1["ret"], t2["ret"], rtol=0, atol=1e-14)
        else:
            assert t1.equals(t2)
        assert fills and all(c == 0.002 for *_, c in fills)
        e3, _ = ph.replay_risk(si, ei, sj, sale, closed, w, key, O, LC, cash, 0.002, fills=[], **kw)
        assert np.array_equal(e1, e3)


def test_replay_risk_charges_each_fill_its_own_cost():
    n, m = 6, 1
    O = np.array([[10.0], [10.0], [10.0], [11.0], [11.0], [11.0]])
    LC = O.copy()
    code = np.zeros((n, m), dtype=np.uint8)
    yi = np.array([0, 0, 0, 1, 1, 1], dtype=np.int16)
    tab = np.zeros((2, ct.N_CODES))
    tab[0, 0], tab[1, 0] = 0.01, 0.02
    cc = ct.CellCost(code, yi, tab)
    fills: list = []
    eq, tr = ph.replay_risk(np.array([0]), np.array([1]), np.array([0]), np.array([3]), np.array([True]), np.array([1.0]),
                            np.array([1.0]), O, LC, np.zeros(n), cc, fills=fills)
    assert abs(tr["ret"].iloc[0] - (11 * 0.98 / (10 * 1.01) - 1)) < 1e-15
    assert abs(eq[-1] - 11 * 0.98 / (10 * 1.01)) < 1e-12
    assert [round(f[3], 6) for f in fills] == [0.01, 0.02]


def test_calendar_weighted_number_equals_constant_table():
    rng = np.random.default_rng(5)
    n, m = 250, 30
    O = np.exp(np.cumsum(rng.normal(0, 0.02, (n, m)), axis=0)) * 30
    C = O * np.exp(rng.normal(0, 0.01, (n, m)))
    cash = np.full(n, 0.0001)
    _, ei, sj, sale, closed, w, _ = _signals(rng, n, m, 400)
    e1, g1 = ph.calendar_weighted(ei, sale, closed, sj, w, O, C, cash, 0.002)
    e2, g2 = ph.calendar_weighted(ei, sale, closed, sj, w, O, C, cash, _const(n, m, 0.002, w0=7))
    assert np.array_equal(e1, e2) and np.array_equal(g1, g2)
    e3, _ = ph.calendar_weighted(ei, sale, closed, sj, w, O, C, cash, _const(n, m, 0.004))
    assert e3[-1] < e1[-1]


# --------------------------------------------------------------------------- benchmarks

def test_ew_regime_and_top_n_ew_tables():
    rng = np.random.default_rng(9)
    n, m = 200, 25
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.02, (n, m)), axis=0)) * 20)
    member = rng.random((n, m)) < 0.6
    a = ls.top_n_ew(close, member, 0.002)
    b = ls.top_n_ew(close, member, np.full((n, m), 0.002))
    assert np.allclose(a, b, rtol=1e-12, atol=0)
    assert np.array_equal(ls.top_n_ew(close, member, 0.0), ls.top_n_ew(close, member, np.zeros((n, m))))
    ew = a / a[0]
    cash = np.full(n, 0.0001)
    state = rng.random(n) < 0.7
    assert np.array_equal(rs.ew_regime(ew, cash, state, 0.002), rs.ew_regime(ew, cash, state, np.full(n, 0.002)))
    hm = ct.held_mean(np.full((n, m), 0.003), member)
    assert np.allclose(hm[1:][member[:-1].any(axis=1)], 0.003)


def test_spy200_switch_cost():
    class Px:
        pass
    rng = np.random.default_rng(11)
    N = 600
    px = Px()
    px.dates = pd.bdate_range("2000-01-03", periods=N)
    px.spy_c = np.exp(np.cumsum(rng.normal(0.0003, 0.012, N))) * 100
    px.ew = np.cumprod(1 + rng.normal(0.0003, 0.01, N))
    w0, n = 250, 300
    wdates = px.dates[w0:w0 + n]
    cash = np.full(n, 0.0001)
    base = gs.bench_curves(px, wdates, w0, cash)
    assert np.array_equal(base["ma200"], gs.bench_curves(px, wdates, w0, cash, 0.0)["ma200"])
    c = gs.bench_curves(px, wdates, w0, cash, np.full(n, 0.001))["ma200"]
    full = pd.Series(px.spy_c)
    on = (full > full.rolling(200).mean()).shift(1, fill_value=False).to_numpy()[w0:w0 + n]
    sw = np.flatnonzero(on[1:] != on[:-1]) + 1
    assert len(sw) > 0
    first = int(sw[0])
    b = base["ma200"]
    assert np.array_equal(c[:first], b[:first])
    d_b, d_c = b[1:] / b[:-1] - 1, c[1:] / c[:-1] - 1
    paid = d_b - d_c
    assert np.allclose(paid[sw - 1], 0.001, atol=1e-12) and np.allclose(np.delete(paid, sw - 1), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- cost table and checks

def _sq(rows):
    return pd.DataFrame(rows, columns=["year", "cap_b", "price_b", "q"])


def test_cell_table_fallbacks_floor_and_cap():
    rows = []
    for y in ct.YEARS:
        for c in range(ct.N_CAP):
            rows += [(y, c, 2, (0.002 * (c + 1)) ** 2)] * 120  # own cell
            rows += [(y, c, 0, 0.9 ** 2)] * 30  # too few -> year x cap
    sq = _sq(rows)
    cells, yc, ec = ct.cell_table(sq)
    own = cells[(cells["price_b"] == 2)]
    assert (own["source"] == "칸").all() and np.allclose(own["half_bp_raw"], (own["cap_b"] + 1) * 10)
    fb = cells[(cells["price_b"] == 0)]
    assert (fb["source"] == "연도 × 시총").all()
    assert (cells["cost_bp"] <= ct.CAP_BP).all() and (cells["cost_bp"] >= ct.FLOOR_BP).all()
    neg = _sq([(y, c, 1, -1e-4) for y in ct.YEARS for c in range(ct.N_CAP) for _ in range(150)])
    cells2, _, _ = ct.cell_table(neg)
    assert (cells2.loc[cells2["price_b"] == 1, "cost_bp"] == ct.FLOOR_BP).all()


def test_checks():
    rows = []
    for y in ct.YEARS:
        for c in range(4):
            rows.append({"year": y, "cap_b": c, "half_bp": (c + 1) * (30 if y <= 2000 else 5)})
    ok = ct.checks(pd.DataFrame(rows))
    assert ok["a_pass"] and ok["b_pass"] and ok["pass"]
    bad = pd.DataFrame(rows)
    bad.loc[(bad["year"] == 2010) & (bad["cap_b"] == 0), "half_bp"] = 100
    r = ct.checks(bad)
    assert not r["a_pass"] and r["a_years_fail"] == [2010]


def test_combine_labels():
    lab = {"TZ": {"K1": "유지", "K2": "바뀜", "K3": "유지"}, "T1": {"K1": "유지", "K2": "유지", "K3": "바뀜"},
           "T2": {"K1": "유지", "K2": "유지", "K3": "유지"}}
    assert ct.combine(lab) == {"K1": "비용에 견고", "K2": "유지, 비용 수준에 민감", "K3": "바뀜"}
