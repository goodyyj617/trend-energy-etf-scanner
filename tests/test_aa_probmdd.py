"""A46: probabilistic MDD verdict (lab/aa_probmdd.py). Synthetic panels, no network."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab import aa_accounts as AC
from lab import aa_aftertax as AT
from lab import aa_canary as K
from lab import aa_layers as LY
from lab import aa_probmdd as PM
from lab.aa_engine import decision_rows
from lab.data import Panel

ROOT = Path(__file__).resolve().parents[1]
CODES = ["379800", "KRXGOLD", "114260", "459580"]
W = {"379800": 50.0, "KRXGOLD": 20.0, "114260": 20.0, "459580": 10.0}


def _panel(n=900, seed=5, extra=K.DEFENSIVE):
    rng = np.random.default_rng(seed)
    cols = CODES + list(extra)
    dates = pd.bdate_range("2004-01-01", periods=n)
    r = rng.normal(0.0003, 0.012, size=(n, len(cols)))
    r[:, 3] = 0.03 / 252
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=cols)
    cash = pd.Series(0.03 / 252, index=dates)
    p = Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6, tradable=cols, cash=cash)
    p.benchmark = "379800"
    return p


def _sim(p, cost=10.0):
    cols = sorted(set(W) | set(K.DEFENSIVE) | {LY.CD})
    return K.DynamicSim(p, cols, str(p.close.index[0].date()), str(p.close.index[-1].date()), cost_bps=cost)


def _qrows(dates, phase=0, offset=0):
    rows = decision_rows(dates, "Q", phase) + offset
    return rows[(rows >= 0) & (rows < len(dates) - 1)]


def test_stationary_indices_shape_blocks_and_seed():
    idx = PM.stationary_indices(500, 200, 50, 1)
    assert idx.shape == (200, 500) and idx.min() >= 0 and idx.max() < 500
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % 500).mean()
    assert 0.96 < cont < 0.995  # 1 − 1/50 = 0.98 (a new start can also continue by chance)
    assert np.array_equal(idx, PM.stationary_indices(500, 200, 50, 1))
    assert not np.array_equal(idx, PM.stationary_indices(500, 200, 50, 2))
    counts = np.bincount(idx.ravel(), minlength=500)
    assert counts.min() > 0.6 * counts.mean()  # circular: the ends are not under-sampled
    with pytest.raises(ValueError):
        PM.stationary_indices(1, 5, 10, 0)


@pytest.mark.parametrize("cost", [10.0, 20.0])
def test_boot_run_with_the_historical_order_is_dynamicsim(cost):
    p = _panel()
    sim = _sim(p, cost)
    tab = LY.action_tables(W, "none", "none", sim.columns)
    q = _qrows(sim.dates, 1, 5)
    dec, ch, fi = PM.static_inputs(2, sim.n_days, q)
    tabs = np.broadcast_to(tab, (2,) + tab.shape)
    ref, _, _ = sim.run(tabs, dec, ch, fi)
    a = PM.boot_run(sim, None, tabs, dec, ch, fi)
    b = PM.boot_run(sim, PM.identity_indices(sim.n_days, 2), tabs, dec, ch, fi)
    assert np.array_equal(a, ref) and np.array_equal(b, ref)
    P = LY.paths(sim, tab, np.zeros((1, sim.n_days), dtype=int), LY.Setting(), q, None)
    assert np.allclose(a[0] / a[0, 0], P[0], rtol=0, atol=1e-13)


def test_boot_run_reads_each_paths_own_rows():
    p = _panel()
    sim = _sim(p)
    tab = LY.action_tables(W, "none", "none", sim.columns)
    T = sim.n_days
    idx = PM.stationary_indices(T, 3, 40, 9)
    dec, ch, fi = PM.static_inputs(3, T, _qrows(sim.dates))
    tabs = np.broadcast_to(tab, (3,) + tab.shape)
    out = PM.boot_run(sim, idx, tabs, dec, ch, fi)
    for k in range(3):  # one path at a time on a sim whose growth rows are reordered
        s2 = _sim(p)
        s2.growth = sim.growth[idx[k]]
        ref, _, _ = s2.run(tab[None], dec[:1], ch[:1], fi[:1])
        assert np.allclose(out[k], ref[0], rtol=1e-14, atol=0)
    assert not np.allclose(out[0], out[1])


@pytest.mark.parametrize("target", [0.08, 0.10, 0.12])
def test_vol_levels_many_equals_vol_levels(target):
    p = _panel(1200)
    sim = _sim(p)
    dates = sim.dates
    rng = np.random.default_rng(3)
    R = rng.normal(0, 0.008, size=(4, len(dates))) * np.linspace(0.5, 2.0, len(dates))[None, :]
    R[:, 0] = 0.0
    lev = PM.vol_levels_many(R, dates, target)
    for k in range(4):
        assert np.array_equal(lev[k], LY.vol_levels(R[k], dates, target))
    assert lev.min() < len(LY.K_LEVELS) - 1


def test_vol_levels_offset_moves_the_decisions():
    p = _panel(1200)
    dates = _sim(p).dates
    rng = np.random.default_rng(4)
    R = rng.normal(0, 0.012, size=(1, len(dates)))
    l0 = PM.vol_levels_many(R, dates, 0.08, 0)[0]
    l5 = PM.vol_levels_many(R, dates, 0.08, 5)[0]
    ch0 = np.flatnonzero(l0[1:] != l0[:-1]) + 1
    ch5 = np.flatnonzero(l5[1:] != l5[:-1]) + 1
    ends = set((decision_rows(dates, "M") + 1).tolist())
    assert set(ch0) <= ends and len(ch5) > 0 and set(ch5) <= {e + 5 for e in ends}


def test_vol_inputs_equal_compose_and_the_stage5_path():
    p = _panel(1200)
    sim = _sim(p)
    tab = LY.action_tables(W, "none", "none", sim.columns)
    q = _qrows(sim.dates)
    dec0, ch0, fi0 = PM.static_inputs(1, sim.n_days, q)
    base = PM.boot_run(sim, None, tab[None], dec0, ch0, fi0)
    lev = PM.vol_levels_many(PM.base_returns(base), sim.dates, 0.08)
    d, c, f = PM.vol_inputs(lev, q)
    rd, rc, rf = LY.compose(np.zeros((1, sim.n_days), dtype=int), q, 0, lev[0])
    assert np.array_equal(d, rd) and np.array_equal(c, rc) and np.array_equal(f, rf)
    tabs = LY.scaled_tables(tab, sim.columns, True)
    P = PM.boot_run(sim, None, tabs[None], d, c, f)
    ref = AT.stage5_vol_path(p, W, str(sim.dates[0].date()), str(sim.dates[-1].date()), 0.08)
    assert np.allclose(P[0] / P[0, 0], ref, rtol=0, atol=1e-13)


def test_mean_tabs_and_weights():
    cols = ["379800", "459580", "CASH"]
    row = np.array([0.8, 0.2, 0.0])
    t = PM.mean_tabs(row, cols, [1.0, 0.5])
    assert t.shape == (2, 1, 3)
    assert np.allclose(t[0, 0], row) and np.allclose(t[1, 0], [0.4, 0.6, 0.0])
    lev = np.array([[19, 19, 9, 9]])
    assert PM.mean_exposure(lev)[0] == pytest.approx(0.75)
    mw = PM.mean_weights(W, CODES, 0.75)
    ref = AT.mean_exposure_weights(W, 0.75)
    assert np.allclose(mw * 100, [ref[c] for c in CODES])


def test_mean_weights_reproduce_the_a45_static_in_the_account_simulator():
    p = _panel(900, extra=())
    rets = p.close[CODES].pct_change(fill_method=None).to_numpy()
    cash, dates = p.cash.to_numpy(), p.close.index
    mw = AT.mean_exposure_weights(W, 0.9)
    a = AC.run_policy(rets, cash, dates, CODES, np.array([mw[c] for c in CODES]), AC.DECLARED)
    b = AC.run_policy(rets, cash, dates, CODES, PM.mean_weights(W, CODES, 0.9), AC.DECLARED)
    assert abs(a["twr_mdd"] - b["twr_mdd"]) < 1e-12 and abs(a["twr_cagr"] - b["twr_cagr"]) < 1e-12


def test_zone_rule():
    assert PM.zone_of([-0.10] * 9 + [-0.19])["zone"] == "안"
    assert PM.zone_of([-0.14] * 8 + [-0.25] * 2)["zone"] == "경계"  # median inside, the bad tenth below −20 %
    assert PM.zone_of([-0.16] * 10)["zone"] == "경계"
    assert PM.zone_of([-0.18] * 10)["zone"] == "밖"
    z = PM.zone_of([-0.10, -0.16, np.nan])
    assert z["n"] == 2 and z["p15"] == 0.5
    assert PM.worst_zone("안", None, "경계") == "경계" and PM.worst_zone("안", "밖") == "밖"
    assert PM.percentile_of(-0.2, [-0.3, -0.2, -0.1, -0.05]) == 0.5


def test_layer_verdict():
    good = PM.layer_stats(np.full(100, 0.03))
    weak = PM.layer_stats(np.r_[np.full(60, 0.03), np.full(40, -0.01)])
    none = PM.layer_stats(np.r_[np.full(40, 0.03), np.full(60, -0.01)])
    assert good["holds"] and not weak["holds"]
    assert PM.layer_of(good, good) == "층 확인"
    assert PM.layer_of(good, weak) == "층 약함"
    assert PM.layer_of(good, none) == "층 없음"
    assert PM.layer_of(good, None) == "층 확인"


def _script():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("aa_a46_probmdd", ROOT / "scripts" / "aa_a46_probmdd.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aa_a46_probmdd"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_family_runs_on_a_synthetic_panel():
    S = _script()
    p = _panel(1300)
    fam = S.r6_family(p, W, str(p.close.index[0].date()), str(p.close.index[-1].date()))
    idx = PM.stationary_indices(fam["sim"].n_days, 20, 60, 1)
    out = S.boot_family(fam, idx, PM.VOL_OFFSETS[:1][0])
    assert set(out) == {"static", "vol", "mean", "kbar", "lev"}
    assert out["vol"].shape == (20, fam["sim"].n_days) and np.all((out["kbar"] > 0) & (out["kbar"] <= 1))
    h = S.boot_family(fam, PM.identity_indices(fam["sim"].n_days), 0)
    ref = AT.stage5_vol_path(p, W, str(fam["sim"].dates[0].date()), str(fam["sim"].dates[-1].date()), 0.10)
    assert np.allclose(h["vol"][0] / h["vol"][0, 0], ref, rtol=0, atol=1e-13)
