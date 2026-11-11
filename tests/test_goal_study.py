"""Stage O (lab/goal_study.py, DECISIONS 94): goal metrics as lab/target.py, benchmarks, verdicts."""
import types

import numpy as np
import pandas as pd

from lab import account_study as ns, entry_study as es, exit_study as xs, goal_study as gs, target as tg


def test_registered():
    assert gs.EXITS == ["X1", "H1", "H9", "ST"] and gs.EXIT_AXIS == list(xs.EXITS_L) and len(gs.EXIT_AXIS) == 11
    assert gs.AXIS_SETTING == (10, "W2", "P0") and gs.BUDGET == 0.15 and gs.GATED == ("mix", "ma200")
    assert gs.MIN_WINDOW_SHARE == 0.5 and gs.MIN_EXITS == 2 and gs.N_TRIALS == 71_774
    assert gs.JUDGED == ("1993-2023", "1993-2007", "2008-2023") and "2017-2023" in gs.GOAL_PERIODS


def _days(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-02", periods=n)
    R = rng.normal(0.0004, 0.015, (n, 3))
    R[:, 2] *= 0.05  # one curve already inside the budget
    c = np.full(n, 0.0001)
    return idx, R, c


def test_goal_metrics_match_target():
    idx, R, c = _days()
    C = np.repeat(c[:, None], 3, axis=1)
    years = idx.year.to_numpy()
    wins = tg.windows(idx)
    saved = dict(es.PERIODS)
    es.PERIODS.clear()
    es.PERIODS.update({"1993-2023": (2001, 2012), "1993-2007": (2001, 2006), "2008-2023": (2007, 2012), "2017-2023": (2010, 2012),
                       "2000-2023": (2001, 2012)})
    try:
        gm = gs.goal_metrics(R, C, years, wins)
        for i in range(3):
            w = tg.fit_weight(R[:, i], c)
            assert gm["w"][i] == w
            assert abs(gm["dd_cagr_1993-2023"][i] - tg._cagr(tg.blend(R[:, i], c, w))) < 1e-9
            assert gm["dd_mdd"][i] >= -0.15 - 1e-9
            assert np.allclose(gm["win_cagr"][:, i], tg.window_cagrs(idx, R[:, i], c, w, wins), atol=1e-9)
            h = (years >= 2001) & (years <= 2006)
            assert gm["w_1993-2007"][i] == tg.fit_weight(R[h, i], c[h])
            assert abs(gm["dd_cagr_1993-2007"][i] - tg._cagr(tg.blend(R[h, i], c[h], w))) < 1e-9
        assert gm["w"][2] == 1.0  # no leverage
    finally:
        es.PERIODS.clear()
        es.PERIODS.update(saved)


def test_bench_curves_follow_entry_study_baselines():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("1992-01-02", "2024-03-29")
    spy = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates)))
    px = types.SimpleNamespace(spy_c=spy, dates=dates, ew=np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates))))
    w0 = 300
    wdates = dates[w0:]
    cash = np.full(len(wdates), 0.0001)
    b = gs.bench_curves(px, wdates, w0, cash)
    ref = es.baselines(px, wdates, w0, cash).set_index("baseline")
    for k, name in (("mix", "SPY"), ("ma200", "SPY 200일선"), ("ew", "동일비중 유니버스")):
        st = es.period_stats(pd.Series(b[k], index=wdates))
        assert all(abs(st[c] - ref.loc[name, c]) < 1e-12 for c in st if np.isfinite(ref.loc[name, c]))


def test_spans_and_cagr():
    years = np.array([2000] * 3 + [2001] * 4 + [2002] * 2)
    assert gs.spans(years, [(2001, 2001), (2000, 2002), (2005, 2006)]) == [(3, 6), (0, 8), (-1, -1)]
    B = np.full((9, 1), 0.01)
    L = np.vstack([[0.0], np.cumsum(np.log1p(B), axis=0)])
    assert abs(gs.span_cagr(L, 3, 6)[0] - (1.01 ** 252 - 1)) < 1e-9


def _bench(mix=0.06, ma=0.07):
    return {"mix": {f"dd_cagr_{p}": mix for p in gs.GOAL_PERIODS}, "ma200": {f"dd_cagr_{p}": ma for p in gs.GOAL_PERIODS}}


def _row(v, ws=0.6, trades=50):
    return {**{f"dd_cagr_{p}": v for p in gs.GOAL_PERIODS}, "ws_mix": ws, "ws_ma200": ws, "trades": trades}


def test_cell_pass():
    b = _bench()
    assert gs.cell_pass(_row(0.08), b) and not gs.cell_pass(_row(0.065), b)  # must beat both benchmarks
    assert not gs.cell_pass(_row(0.08, ws=0.4), b) and not gs.cell_pass(_row(0.08, trades=20), b)
    r = _row(0.08)
    r["dd_cagr_2017-2023"] = 0.0  # 2017-2023 is not gated
    assert gs.cell_pass(r, b)
    r["dd_cagr_1993-2007"] = 0.05
    assert not gs.cell_pass(r, b)


def test_candidates_need_two_exits_40bp_and_a_neighbour():
    b = _bench()
    rows = []
    for st in ns.settings():
        name = ns.setting_name(st)
        for gen in ("E1", "E2", "E0"):
            for x in gs.EXITS:
                good = gen == "E1" and st[1] in ("W2", "W3") and st[0] in (10, 20) and x in ("H1", "H9")
                v = 0.09 if good else (0.065 if gen == "E0" else 0.05)
                rows.append({"setting": name, "gen": gen, "exit": x, "kind": "rule", "cost": 20.0, "mode": "grid",
                             "N": st[0], "W": st[1], "P": st[2], **_row(v)})
    g = pd.DataFrame(rows)
    pass40 = {(r["setting"], r["gen"], r["exit"]): True for r in rows if r["gen"] == "E1" and r["dd_cagr_1993-2023"] > 0.08}
    pass40[("N20_W3_P0", "E1", "H9")] = False
    cells, cand = gs.candidates(g, b, pass40)
    c = cand.set_index(["setting", "gen"])
    assert c.loc[("N10_W2_P0", "E1"), "candidate"] and c.loc[("N20_W2_P0", "E1"), "candidate"]
    assert c.loc[("N10_W2_P0", "E1"), "entry_specific"]  # beats E0 (6.5%) in both halves
    assert not c.loc[("N20_W3_P0", "E1"), "ok"] and not c.loc[("N10_W3_P0", "E1"), "candidate"]  # 40 bp lost -> no neighbour
    assert c.loc[("N10_W2_P1", "E1"), "candidate"] and ("N10_W1_P0", "E1") not in c.index  # W1 is comparison only
    acc = pd.DataFrame([{**r, "src": s} for r in rows if r["gen"] == "E0" for s in ("R0s1", "R0s2")])
    ch = gs.chance(acc, cells, b).set_index(["setting", "exit"])
    assert ch.loc[("N10_W2_P0", "H1"), "passing"] == 1 and ch.loc[("N10_W2_P0", "H1"), "p0"] == 0.0
    assert ch.loc[("N10_W2_P0", "H1"), "beyond_chance"]


def test_systems_need_both_benchmarks():
    b = _bench()
    rows = [{"setting": "N20_W2_P1", "gen": g_, "exit": "X1", "kind": "rule", "cost": 20.0, "mode": "grid", **_row(v),
             "w": 0.4, "w_1993-2007": 0.4, "w_2008-2023": 0.4, "dd_mdd": -0.15, "worst_mix": 0.0, "worst_ma200": 0.0, "ws_ew": 0.5,
             "worst_ew": 0.0, "cagr_1993-2023": 0.1, "mdd_1993-2023": -0.4, "cagr_2017-2023": 0.08}
            for g_, v in (("E1", 0.08), ("E2", 0.071), ("E3", 0.069), ("E0", 0.07))]
    s = gs.systems(pd.DataFrame(rows), b)
    assert len(s) == 1 and s.iloc[0]["entries"] == 3 and abs(s.iloc[0]["dd_cagr_1993-2023"] - 0.071) < 1e-12
    assert bool(s.iloc[0]["system_pass"]) and s.iloc[0]["e0_dd_cagr_1993-2023"] == 0.07
