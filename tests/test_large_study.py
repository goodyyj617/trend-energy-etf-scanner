"""Stage Q (lab/large_study.py, DECISIONS 96): bands, pseudo restrictions, top-N equal weight, verdicts."""
import types

import numpy as np
import pandas as pd

from lab import goal_study as gs, large_study as ls, regime_study as rs, tail_study as ts


def test_registered():
    assert ls.SETTINGS == rs.SETTINGS and ls.EXITS == ["X1", "H1"]
    assert ls.FILTERS == ["none", "R1A1", "R3A1", "R4A1", "R5A1"]
    assert ls.BANDS == {"U3000": np.inf, "U1000": 1000.0, "U500": 500.0} and ls.NARROW == ("U1000", "U500")
    assert ls.N_PSEUDO == 10 and ls.PSEUDO_ORDERS == 20 and ls.PSEUDO_REF == (20, "W2", "P1")
    assert ls.PSEUDO_FILTERS[(10, "W2", "P0")] == ["none", "R5A1"] and ls.PSEUDO_FILTERS[(20, "W2", "P1")] == ls.FILTERS
    assert (ls.HELP_PP, ls.HALF_TOL) == (0.005, 0.010)
    assert ls.N_NEW == 1_200 == 2 * 5 * 4 * 15 * 2 and ls.N_TRIALS == 74_174


def test_in_band():
    cap = np.array([1.0, 499.0, 500.0, 501.0, 1000.0, 2999.0])
    assert ls.in_band(cap, np.inf).all()
    assert ls.in_band(cap, 500.0).tolist() == [True, True, True, False, False, False]
    assert ls.in_band(cap, 1000.0).sum() == 5 and ls.in_band(cap, 9_999.0).all()


def test_pseudo_band_matches_the_day_count_among_m1():
    rng = np.random.default_rng(3)
    si = np.sort(rng.integers(0, 40, 2000))
    keep = rng.random(2000) < 0.7
    band = keep & (rng.random(2000) < 0.4)
    p = ls.pseudo_band(si, keep, band, ls.pseudo_rng("U500", "E1", 0))
    assert not (p & ~keep).any()  # only M1-passing signals
    assert np.array_equal(np.bincount(si[p], minlength=40), np.bincount(si[band], minlength=40))  # same day, same count
    assert np.array_equal(p, ls.pseudo_band(si, keep, band, ls.pseudo_rng("U500", "E1", 0)))  # seeded
    assert not np.array_equal(p, ls.pseudo_band(si, keep, band, ls.pseudo_rng("U500", "E1", 1)))
    assert np.array_equal(ls.pseudo_band(si, keep, keep, ls.pseudo_rng("U500", "E1", 0)), keep)  # a band keeping all M1 = M1


def test_top_n_ew():
    idx = pd.bdate_range("2001-01-01", periods=60)
    rng = np.random.default_rng(5)
    close = pd.DataFrame(50 * np.cumprod(1 + rng.normal(0, 0.02, (60, 6)), axis=0), index=idx, columns=list("abcdef"))
    close.iloc[10:14, 2] = np.nan
    elig = pd.DataFrame(rng.random((60, 6)) < 0.8, index=idx, columns=close.columns)
    panel = types.SimpleNamespace(close=close)
    assert np.allclose(ls.top_n_ew(close, elig.to_numpy(), 0.0), ts.equal_weight_index(panel, elig), rtol=0, atol=1e-13)
    const = np.ones((60, 6), dtype=bool)
    a, b = ls.top_n_ew(close, const, 0.002), ls.top_n_ew(close, const, 0.0)
    assert abs(a[1] - (b[1] - 0.002)) < 1e-12 and np.allclose(a[1:] / a[1], b[1:] / b[1])  # the first build only
    m = const.copy()
    m[30:, 0] = False  # 'a' leaves at the close of day 30: sold out of 6 names on day 31
    a, b = ls.top_n_ew(close, m, 0.002), ls.top_n_ew(close, m, 0.0)
    assert abs((a[31] / a[30]) - (b[31] / b[30] - 0.002 / 6)) < 1e-12 and np.allclose(a[1:31] / a[1], b[1:31] / b[1])


def _account(seed=11, n_days=300, n_names=20, n_sig=700):
    rng = np.random.default_rng(seed)
    O = 40 * np.cumprod(1 + rng.normal(0.0003, 0.02, (n_days, n_names)), axis=0)
    LC = O * (1 + rng.normal(0, 0.002, O.shape))
    ei = rng.integers(1, n_days - 3, n_sig)
    r = types.SimpleNamespace(ei=ei, sj=rng.integers(0, n_names, n_sig), sale=np.minimum(ei + rng.integers(2, 60, n_sig), n_days - 1),
                              closed=rng.random(n_sig) < 0.9, cap=rng.integers(1, 3001, n_sig).astype(float))
    rs._W.update(O=O, LC=LC, cash=np.full(n_days, 0.0001), w0=0, n=n_days)
    return r, ei - 1, np.full(n_sig, 0.1), rng.random(n_sig), rng.random(n_sig) < 0.7


def test_unbounded_band_is_bit_identical_and_bands_only_limit_entries():
    r, si, w, key, keep = _account()
    state = np.ones(len(rs._W["O"]), dtype=bool)
    state[120:170] = False
    for st in (None, state):
        eq0, tr0 = rs._replay(r, si, keep, w, key, 10, 0.002, st, False)
        for limit in (np.inf, 3000.0, 9_999.0):
            eq1, tr1 = rs._replay(r, si, keep & ls.in_band(r.cap, limit), w, key, 10, 0.002, st, False)
            assert np.array_equal(eq0, eq1) and tr0.equals(tr1)
    m = keep & ls.in_band(r.cap, 500.0)
    eq2, tr2 = rs._replay(r, si, m, w, key, 10, 0.002)
    bought = np.flatnonzero(m & np.isfinite(w))[tr2["row"].to_numpy()]
    assert len(tr2) and (r.cap[bought] <= 500).all() and not np.array_equal(eq0, eq2)


def _sys(rows):
    return pd.DataFrame([{"setting": st, "exit": x, "filter": f, "band": b,
                          **{f"dd_cagr_{p}": v + (0.0 if p != "1993-2007" else d93) for p in gs.GOAL_PERIODS}}
                         for st, x, f, b, v, d93 in rows])


def test_q1_needs_u3000_margin_and_pseudo_max():
    sysm = _sys([("N20_W2_P1", "X1", "none", "U3000", 0.050, 0.0), ("N20_W2_P1", "X1", "none", "U500", 0.058, 0.0),
                 ("N20_W2_P1", "X1", "none", "U1000", 0.053, 0.0),  # +0.3 pp: not enough
                 ("N50_W2_P1", "H1", "none", "U3000", 0.050, 0.0), ("N50_W2_P1", "H1", "none", "U500", 0.060, -0.03),  # 1993-2007 too low
                 ("N10_W2_P1", "H1", "none", "U3000", 0.050, 0.0), ("N10_W2_P1", "H1", "none", "U500", 0.057, 0.0)])
    ps = pd.DataFrame([{"setting": "N20_W2_P1", "filter": "none", "band": b, "k": k,
                        **{f"dd_cagr_{p}": 0.050 + 0.0006 * k for p in gs.GOAL_PERIODS}} for b in ("U500", "U1000") for k in range(10)])
    h, c = ls.q1(sysm, ps)
    h = h.set_index(["setting", "exit", "band"])
    assert h.loc[("N20_W2_P1", "X1", "U500"), "helps"] and not h.loc[("N20_W2_P1", "X1", "U1000"), "helps"]
    assert not h.loc[("N50_W2_P1", "H1", "U500"), "helps"] and h.loc[("N50_W2_P1", "H1", "U500"), "above_pseudo"]
    assert h.loc[("N10_W2_P1", "H1", "U500"), "pseudo_from"] == "N20_W2_P1" and h.loc[("N10_W2_P1", "H1", "U500"), "helps"]
    c = c.set_index(["band", "filter"])
    assert c.loc[("U500", "none"), "helps"] == 2 and c.loc[("U500", "none"), "systems"] == 3
    assert c.loc[("U500", "none"), "pseudo_tests"] == 10 and 0 < c.loc[("U500", "none"), "p0"] <= 0.1  # only k=9 beats the others by 0.5 pp


def test_candidates_need_adjacent_n_and_the_other_band():
    bench = {b: {f"dd_cagr_{p}": 0.05 for p in gs.GOAL_PERIODS} for b in gs.GATED}
    good = {**{f"dd_cagr_{p}": 0.08 for p in gs.GOAL_PERIODS}, "ws_mix": 0.7, "ws_ma200": 0.6, "trades": 200}
    bad = {**good, "dd_cagr_2008-2023": 0.04}
    rows = []
    for st in ("N10_W2_P1", "N20_W2_P1", "N50_W2_P1"):
        for b in ls.NARROW:
            for x in ls.EXITS:
                ok = not (st == "N50_W2_P1") and not (st == "N10_W2_P1" and b == "U1000")
                rows.append({"setting": st, "gen": "E7", "exit": x, "filter": "R5A1", "band": b, "cost": 20.0, "mode": "grid",
                             **(good if ok else bad)})
                rows.append({"setting": st, "gen": "E0", "exit": x, "filter": "R5A1", "band": b, "cost": 20.0, "mode": "grid",
                             **{**good, **{f"dd_cagr_{p}": 0.03 for p in gs.GOAL_PERIODS}}})
    g = pd.DataFrame(rows)
    pass40 = {(r["setting"], r["gen"], r["exit"], r["filter"], r["band"]): True for r in rows}
    cells, cand = ls.candidates(g, bench, pass40)
    c = cand.set_index(["setting", "band"])
    assert c.loc[("N20_W2_P1", "U500"), "ok"] and c.loc[("N20_W2_P1", "U1000"), "ok"] and c.loc[("N10_W2_P1", "U500"), "ok"]
    assert c.loc[("N20_W2_P1", "U500"), "candidate"] and c.loc[("N20_W2_P1", "U500"), "entry_specific"]
    assert not c.loc[("N10_W2_P1", "U500"), "candidate"]  # adjacent N fine, but N10 x U1000 fails
    assert not c.loc[("N20_W2_P1", "U1000"), "candidate"]  # other band fine, but no adjacent N at U1000
    _, cand2 = ls.candidates(g, bench, {})
    assert not cand2["candidate"].any()  # 40 bp is required
