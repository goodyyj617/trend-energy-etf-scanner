"""A45: after-tax table and moving targets in the account simulator (lab/aa_aftertax.py, aa_accounts targets=).
Synthetic panels, no network."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab import aa_accounts as AC
from lab import aa_aftertax as AT
from lab import aa_canary as K
from lab import aa_layers as LY
from lab.aa_engine import decision_rows
from lab.data import Panel

ROOT = Path(__file__).resolve().parents[1]
CODES = ["379800", "KRXGOLD", "114260", "459580"]
W = {"379800": 50.0, "KRXGOLD": 20.0, "114260": 20.0, "459580": 10.0}


def _panel(n=1400, seed=5, extra=()):
    rng = np.random.default_rng(seed)
    cols = CODES + list(extra)
    dates = pd.bdate_range("2004-01-01", periods=n)
    r = rng.normal(0.0003, 0.01, size=(n, len(cols)))
    r[:, 3] = 0.03 / 252  # the CD-rate ETF
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=cols)
    cash = pd.Series(0.03 / 252, index=dates)
    p = Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6, tradable=cols, cash=cash)
    p.benchmark = "379800"
    return p


def _arrays(p):
    rets = p.close[CODES].pct_change(fill_method=None).to_numpy()
    return rets, p.cash.to_numpy(), p.close.index


def _levels(dates, seed=7):
    """Random exposure levels that change only the row after a month end (as vol_levels does)."""
    rng = np.random.default_rng(seed)
    T = len(dates)
    lev = np.full(T, len(LY.K_LEVELS) - 1)
    cur = lev[0]
    ends = set(decision_rows(dates, "M").tolist())
    for t in range(T):
        lev[t] = cur
        if t in ends and t > 60:
            cur = int(rng.integers(5, len(LY.K_LEVELS)))
    return lev


POLICIES = [AC.BASELINE, AC.DECLARED, AC.DECLARED_3, AC.Policy("single", "lump_monthly", "band", "double", "low", "isa"),
            AC.Policy("split", "monthly", "deposit", "flat", "general", "none", "invest")]


@pytest.mark.parametrize("pol", POLICIES, ids=[p.label() for p in POLICIES])
def test_constant_targets_are_bit_identical(pol):
    p = _panel()
    rets, cash, dates = _arrays(p)
    wv = np.array([W[c] for c in CODES])
    a = AC.run_policy(rets, cash, dates, CODES, wv, pol)
    b = AC.run_policy(rets, cash, dates, CODES, None, pol, targets=np.tile(wv, (len(dates), 1)))
    for k in ("value", "twr"):
        assert np.array_equal(a[k], b[k])
    for k in ("tax", "cost", "final", "irr"):
        assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k]))


@pytest.mark.parametrize("cost", [0.0, 10.0])
def test_moving_targets_match_dynamicsim(cost):
    p = _panel()
    rets, cash, dates = _arrays(p)
    lev = _levels(dates)
    sim = K.DynamicSim(p, CODES, str(dates[0].date()), str(dates[-1].date()), cost_bps=cost)
    tabs = LY.scaled_tables(LY.action_tables(W, "none", "none", sim.columns), sim.columns, True)
    qrows = decision_rows(sim.dates, "Q")
    qrows = qrows[qrows < len(sim.dates) - 1]
    decide, choice, first = LY.compose(np.zeros((1, len(dates)), dtype=int), qrows, 0, lev)
    path, _, _ = sim.run(tabs[None], decide, choice, first)
    pol = AC.Policy("single", "lump", "full", "zero" if cost == 0 else "flat", "none", "none")
    res = AC.run_policy(rets, cash, dates, CODES, None, pol, charge_first=False, targets=AT.vol_targets(W, CODES, lev))
    assert np.allclose(res["twr"], path[0] / path[0, 0], rtol=0, atol=1e-10)


def test_moving_targets_under_the_declared_policy_trade_more_and_keep_the_gold_account():
    p = _panel(2400)
    rets, cash, dates = _arrays(p)
    lev = _levels(dates)
    wv = np.array([W[c] for c in CODES])
    tg = AT.vol_targets(W, CODES, lev)
    stat = AC.run_policy(rets, cash, dates, CODES, wv, AC.DECLARED)
    dyn = AC.run_policy(rets, cash, dates, CODES, None, AC.DECLARED, targets=tg)
    assert dyn["cost"] > stat["cost"] and dyn["tax"] > 0
    assert np.isfinite(dyn["twr_mdd"]) and np.isfinite(dyn["irr"])
    with pytest.raises(ValueError):
        AC.run_policy(rets, cash, dates, CODES, None, AC.DECLARED, targets=tg[:-1])


def test_vol_targets_rows_and_guards():
    dates = pd.bdate_range("2010-01-01", periods=300)
    lev = _levels(dates)
    tg = AT.vol_targets(W, CODES, lev)
    assert tg.shape == (300, 4) and np.allclose(tg.sum(axis=1), 1.0)
    k = LY.K_LEVELS[lev[1:]]
    assert np.allclose(tg[:-1, 0], 0.5 * k)
    assert np.allclose(tg[:-1, 3], 0.1 * k + (1 - k))
    with pytest.raises(ValueError):
        AT.vol_targets({"379800": 100.0}, ["379800"], lev)
    bad = lev.copy()
    bad[0] = 0
    with pytest.raises(ValueError):
        AT.vol_targets(W, CODES, bad)


def test_vol_levels_of_runs_like_stage5():
    p = _panel(1400, extra=K.DEFENSIVE)
    dates, lev = AT.vol_levels_of(p, W, str(p.close.index[0].date()), str(p.close.index[-1].date()), 0.08)
    assert len(dates) == len(lev) == 1400
    assert lev[0] == lev[1] == len(LY.K_LEVELS) - 1
    assert lev.min() < len(LY.K_LEVELS) - 1  # synthetic vol ≈ 10 % > 8 %: exposure goes down at some point


def test_mean_exposure_and_weights():
    lev = np.array([19, 19, 9, 9])  # k = 1.0, 1.0, 0.5, 0.5
    assert AT.mean_exposure(lev) == pytest.approx(0.75)
    mw = AT.mean_exposure_weights(W, 0.75)
    assert sum(mw.values()) == pytest.approx(100.0)
    assert mw["379800"] == pytest.approx(37.5) and mw["459580"] == pytest.approx(7.5 + 25.0)


def test_zone_and_layer_verdict():
    assert AT.zone(-0.149) == "안" and AT.zone(-0.16) == "경계" and AT.zone(-0.171) == "밖"
    assert AT.layer_verdict(0.070, -0.07, 0.072, -0.10) == "층 유지"
    assert AT.layer_verdict(0.060, -0.07, 0.072, -0.10) == "층 비쌈"
    assert AT.layer_verdict(0.072, -0.09, 0.072, -0.10) == "층 없음"


def _script():
    spec = importlib.util.spec_from_file_location("aa_a45_aftertax", ROOT / "scripts" / "aa_a45_aftertax.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tables_from_policy_rows():
    S = _script()
    p = _panel(900)
    rets, cash, dates = _arrays(p)
    lev = _levels(dates)
    wv = np.array([W[c] for c in CODES])
    rows = []
    for name, kw in ((AT.VOL_NAME, {"targets": AT.vol_targets(W, CODES, lev)}),
                     ("R6 정적", {}), (AT.MEAN_NAME, {})):
        for pol in (AC.BASELINE, AC.DECLARED):
            r = AC.run_policy(rets, cash, dates, CODES, None if kw else wv, pol, **kw)
            rows.append(S.result_row(name, pol, r, dates, AC.DECLARED))
    R = pd.DataFrame(rows)
    T = AT.table_rows(R)
    assert list(T["대상"]) == [AT.VOL_NAME, "R6 정적", AT.MEAN_NAME]
    assert set(T["X1 세후 구역"]) <= {"안", "경계", "밖"}
    L = AT.layer_rows(R, S.POLICY_COLS)
    assert len(L) == 2 and L["선언 정책"].sum() == 1
    assert set(L["X3"]) <= {"층 유지", "층 비쌈", "층 없음"}
