"""Stage P (lab/regime_study.py, DECISIONS 95): forced exits, regimes, placebos, verdicts."""
import types

import numpy as np
import pandas as pd

from lab import blocks, goal_study as gs, pick_study as ph, regime_study as rs


def test_registered():
    assert rs.SETTINGS == [(10, "W2", "P0"), (10, "W2", "P1"), (20, "W2", "P1"), (50, "W2", "P1")] and rs.EXITS == ["X1", "H1"]
    assert list(rs.REGIMES) == ["R1", "R2", "R3", "R4", "R5"] and list(rs.ACTIONS) == ["A1", "A2"] and len(rs.FILTERS) == 11
    assert rs.REGIME_BLOCKS == ("market_trend", "curve_ok", "credit_calm", "vix_calm")
    defaults = {k: blocks.ALL_BLOCKS[k].param.default for k in rs.REGIME_BLOCKS}
    assert defaults == {"market_trend": 200, "curve_ok": 0.0, "credit_calm": 200, "vix_calm": 25.0}
    assert rs.N_PLACEBO == 20 and rs.SHIFT == (252, 5000) and rs.N_NEW == 1_200 and rs.N_TRIALS == 72_974
    s = rs.shifts("R1")
    assert len(s) == 20 and s.min() >= 252 and s.max() <= 5000 and np.array_equal(s, rs.shifts("R1"))


def _account(seed=4, n_days=260, n_names=15, n_sig=500):
    rng = np.random.default_rng(seed)
    O = 40 * np.cumprod(1 + rng.normal(0.0003, 0.02, (n_days, n_names)), axis=0)
    LC = O * (1 + rng.normal(0, 0.002, O.shape))
    ei = rng.integers(1, n_days - 3, n_sig)
    args = dict(si=ei - 1, ei=ei, sj=rng.integers(0, n_names, n_sig), sale=np.minimum(ei + rng.integers(2, 60, n_sig), n_days - 1),
                closed=rng.random(n_sig) < 0.9, w=np.full(n_sig, 0.1), key=rng.random(n_sig), O=O, LC=LC,
                cash_r=np.full(n_days, 0.0001), cost=0.002)
    return args


def test_force_exit_default_is_bit_identical_and_sells_everything():
    a = _account()
    eq0, tr0 = ph.replay_risk(**a)
    for f in (None, np.array([], dtype=int)):
        eq1, tr1 = ph.replay_risk(**a, force_exit=f)
        assert np.array_equal(eq0, eq1) and tr0.equals(tr1)
    days = np.array([60, 150])
    log = {}
    eq2, tr2 = ph.replay_risk(**a, force_exit=days, log=log)
    assert not np.array_equal(eq0, eq2)
    held_before = log["positions"][days - 1]
    assert (held_before > 0).all() and (tr2["exit_i"].isin(days)).sum() >= held_before.sum()
    for d in days:  # nothing held over the forced open, and the ban keeps those names out for 20 sessions
        sold = tr2[tr2["exit_i"] == d]["j"].to_numpy()
        later = tr2[(tr2["entry_i"] > d) & (tr2["entry_i"] < d + ph.BAN) & tr2["j"].isin(sold)]
        assert later.empty
        assert set(log["held"][d]).isdisjoint(set(sold))


def test_force_days_and_placebo_state():
    state = np.array([True, True, False, False, True, False, True, True])
    assert rs.force_days(state, 0, 8).tolist() == [3, 6]  # the open after each first off close
    assert rs.force_days(state, 2, 4).tolist() == [1]
    long = np.r_[np.ones(400, bool), np.zeros(100, bool), np.ones(500, bool)]
    p = rs.placebo_state(long, 10, 900, 300)
    assert p[:9].all() and p[910:].tolist() == long[910:].tolist() and p[9:910].sum() == long[9:910].sum()
    assert not np.array_equal(p, long)


def test_ew_regime():
    ew = np.cumprod(np.r_[1.0, 1 + np.full(9, 0.01)])
    cash = np.full(10, 0.001)
    assert np.allclose(rs.ew_regime(ew, cash, np.ones(10, bool), 0.002), ew)
    assert np.allclose(rs.ew_regime(ew, cash, np.zeros(10, bool), 0.002), np.cumprod(np.r_[1.0, np.full(9, 1.001)]))
    st = np.array([True] * 5 + [False] * 5)
    e = rs.ew_regime(ew, cash, st, 0.002)
    assert abs(e[5] / e[4] - (1.01 - 0.0)) < 1e-12 and abs(e[6] / e[5] - (1.001 - 0.002)) < 1e-12  # off from day 6, cost on the switch


def test_regime_states_follow_the_blocks():
    idx = pd.bdate_range("1999-01-04", "2006-12-29")
    rng = np.random.default_rng(1)
    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(idx))), index=idx)
    st = rs.regime_states(spy)
    mini = types.SimpleNamespace(close=spy.to_frame("SPY"))
    assert st["R1"].equals(blocks.ALL_BLOCKS["market_trend"].compute(mini, 200)["SPY"].astype(bool))
    assert st["R4"].equals(blocks.ALL_BLOCKS["vix_calm"].compute(mini, 25.0)["SPY"].astype(bool))
    assert st["R5"].equals(st[["R1", "R2", "R3", "R4"]].all(axis=1)) and not st["R1"].iloc[:199].any()
    assert 0.0 < st["R4"].mean() < 1.0  # 1999-2006 had VIX above and below 25


def test_always_on_gate_equals_no_gate():
    a = _account(seed=7)
    n = a["O"].shape[0]
    rs._W.update(O=a["O"], LC=a["LC"], cash=a["cash_r"], w0=0, n=n)
    r = types.SimpleNamespace(ei=a["ei"], sj=a["sj"], sale=a["sale"], closed=a["closed"])
    si_panel = a["ei"] - 1
    mask = np.ones(len(a["ei"]), bool)
    eq0, tr0 = rs._replay(r, si_panel, mask, a["w"], a["key"], 10, a["cost"])
    on = np.ones(n, bool)
    for a2 in (False, True):
        eq1, tr1 = rs._replay(r, si_panel, mask, a["w"], a["key"], 10, a["cost"], on, a2)
        assert np.array_equal(eq0, eq1) and tr0.equals(tr1)
    off = on.copy()
    off[100:140] = False
    eq2, tr2 = rs._replay(r, si_panel, mask, a["w"], a["key"], 10, a["cost"], off, True)
    assert not tr2[(tr2["entry_i"] > 101) & (tr2["entry_i"] <= 140)].shape[0]  # no entry from signals of off days


def _sys(rows):
    return pd.DataFrame([{"setting": st, "exit": x, "filter": f, **{f"dd_cagr_{p}": v for p in gs.GOAL_PERIODS}} for st, x, f, v in rows])


def test_q1_helps_needs_none_and_placebo():
    sysm = _sys([("N20_W2_P1", "X1", "none", 0.05), ("N20_W2_P1", "X1", "R1A2", 0.07), ("N20_W2_P1", "X1", "R2A1", 0.055),
                 ("N10_W2_P1", "H1", "none", 0.05), ("N10_W2_P1", "H1", "R1A2", 0.065)])
    ps = pd.DataFrame([{"setting": "N20_W2_P1", "base": f, "k": k, **{f"dd_cagr_{p}": 0.05 + 0.0005 * k for p in gs.GOAL_PERIODS}}
                       for f in ("R1A2", "R2A1") for k in range(20)])
    h, c = rs.q1(sysm, ps)
    h = h.set_index(["setting", "exit", "filter"])
    assert h.loc[("N20_W2_P1", "X1", "R1A2"), "helps"] and not h.loc[("N20_W2_P1", "X1", "R2A1"), "helps"]  # 5.5% < placebo q95
    assert h.loc[("N10_W2_P1", "H1", "R1A2"), "placebo_from"] == "N20_W2_P1" and h.loc[("N10_W2_P1", "H1", "R1A2"), "helps"]
    c = c.set_index("filter")
    assert c.loc["R1A2", "helps"] == 2 and c.loc["R1A2", "systems"] == 2 and 0 < c.loc["R1A2", "p0"] <= 0.1
