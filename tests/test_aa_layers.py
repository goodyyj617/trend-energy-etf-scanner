"""A42 stage 4: dynamic layers (lab/aa_layers.py). Synthetic panel, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab import aa_canary as K
from lab import aa_layers as LY
from lab.aa_engine import AllocConfig, run_allocation
from lab.data import Panel

CODES = ["379810", "379800", "453630", "453640", "KRXGOLD", "459580"]
BASE = {"379810": 30.0, "379800": 20.0, "KRXGOLD": 30.0, "459580": 20.0}


def _panel(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-01", periods=n)
    r = rng.normal(0.0003, 0.01, size=(n, len(CODES)))
    r[:, -1] = 0.0001
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=CODES)
    cash = pd.Series(0.02 / 252, index=dates)
    p = Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6, tradable=CODES, cash=cash)
    p.benchmark = "379800"
    return p


def test_settings_count():
    s = LY.all_settings()
    assert len(s) == 873 and len(set(s)) == 873
    assert sum(x.signal == "none" for x in s) == 9


def test_confirm_and_delay():
    st = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0])
    assert LY.confirm(st, 0).tolist() == st.tolist()
    assert LY.confirm(st, 3).tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert np.array_equal(LY.confirm_many(np.vstack([st, st]), 3)[1], LY.confirm(st, 3))
    assert LY.delay(st, 2).tolist() == [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0]


def test_static_setting_equals_the_engine():
    p = _panel()
    cols = sorted(set(BASE) | set(K.DEFENSIVE) | {LY.CD})
    sim = K.DynamicSim(p, cols, str(p.close.index[0].date()), str(p.close.index[-1].date()))
    tab = LY.action_tables(BASE, "none", "none", sim.columns)
    P = LY.paths(sim, tab, np.zeros((1, sim.n_days), dtype=int), LY.Setting(), K.quarter_rows(sim.dates), None)
    eng = run_allocation(p, AllocConfig("x", BASE, str(p.close.index[0].date()), str(p.close.index[-1].date()), rebalance="Q",
                                        cost_bps=10.0)).equity.to_numpy()
    assert np.allclose(P[0], eng / eng[0], rtol=0, atol=1e-10)


def test_riskcd_moves_equity_to_cd_and_vol_scaling():
    cols = sorted(set(BASE) | set(K.DEFENSIVE) | {LY.CD}) + ["CASH"]
    tab = LY.action_tables(BASE, "k4", "riskcd", cols)
    assert tab.shape == (2, len(cols)) and np.allclose(tab.sum(axis=1), 1)
    assert tab[1][cols.index("379810")] == 0 and np.isclose(tab[1][cols.index(LY.CD)], 0.7)
    assert np.isclose(tab[1][cols.index("KRXGOLD")], 0.3)  # gold is not an equity product
    sc = LY.scaled_tables(tab, cols, True)
    assert sc.shape == (2 * len(LY.K_LEVELS), len(cols)) and np.allclose(sc.sum(axis=1), 1)


def test_vol_levels_and_tranches_run():
    p = _panel()
    cols = sorted(set(BASE) | set(K.DEFENSIVE) | {LY.CD})
    sim = K.DynamicSim(p, cols, str(p.close.index[0].date()), str(p.close.index[-1].date()))
    base_tab = LY.action_tables(BASE, "none", "none", sim.columns)
    bp, _, _ = sim.run(*K.static_rows(base_tab, sim))
    r = np.concatenate([[0.0], bp[0, 1:] / bp[0, :-1] - 1])
    lev = LY.vol_levels(r, sim.dates, 0.04)  # a low target: k below 1 most of the time
    assert lev[0] == len(LY.K_LEVELS) - 1 and lev.min() < len(LY.K_LEVELS) - 1
    rng = np.random.default_rng(1)
    S = (rng.random((3, sim.n_days)) < 0.02).cumsum(axis=1) % 2
    tab = LY.action_tables(BASE, "k4", "safe", sim.columns)
    for n in (1, 2, 4):
        P = LY.paths(sim, tab, S, LY.Setting("k4", "safe", 10, n, 0.04, 21), K.quarter_rows(sim.dates), lev)
        assert P.shape == (3, sim.n_days) and np.isfinite(P).all() and np.allclose(P[:, 0], 1.0)
    vt = LY.paths(sim, base_tab, np.zeros((1, sim.n_days), dtype=int), LY.Setting(vol=0.04), K.quarter_rows(sim.dates), lev)
    assert LY.mdd(vt)[0] > LY.mdd(bp)[0]  # less exposure, shallower drawdown


def test_mix_with_a_satellite():
    core = np.cumprod(np.full((1, 10), 1.01), axis=1)
    sat = np.concatenate([[np.nan] * 4, np.cumprod(np.full(6, 1.03))])
    m = LY.mix(core, sat, 0.5)
    assert np.allclose(m[0, :5] / m[0, 0], core[0, :5] / core[0, 0])  # before the satellite starts: the core
    assert m[0, -1] > core[0, -1] / core[0, 0]


def test_stage4_result_files_for_the_app():
    from lab import aa_experiments as ex
    t = ex.a42_4_checks()
    assert len(t) == 6 and t["통과 수 (위성 0·25·50% 합)"].sum() > 0
    v = ex.a42_4_verdicts()
    assert set(v["판정"]) <= {"통과", "탈락"} and v["① 가짜 95%보다 개선"].dtype == bool
