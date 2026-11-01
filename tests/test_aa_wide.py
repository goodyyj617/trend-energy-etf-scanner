"""A42 stage 2 tools (lab/aa_wide.py): samplers, metrics, yearly windows, picking rules. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab import aa_wide as A


def _check_rows(W, D):
    assert W.shape[1] == D
    assert np.issubdtype(W.dtype, np.integer)
    assert (W.sum(axis=1) == 100).all()
    held = W[W > 0]
    assert held.min() >= A.MIN_PCT


def test_samplers_rows_are_whole_percent_and_reproducible():
    for fn in (A.sample_t1, A.sample_t2, A.sample_t3):
        W = fn(512, 7)
        _check_rows(W, len(A.U17))
        assert np.array_equal(W, fn(512, 7))
        assert not np.array_equal(W, fn(512, 8))
    L = A.sample_tl(512, 7)
    _check_rows(L, len(A.U19))
    lev = L[:, [A.U19.index(c) for c in A.LEVERAGE]].sum(axis=1)
    assert lev.max() <= A.LEV_CAP and (lev > 0).any()


def test_t3_has_each_k_equally():
    W = A.sample_t3(7 * 64, 1)
    k = (W > 0).sum(axis=1)
    # products under 2 % are dropped after the draw, so k can only shrink
    assert k.max() <= 8 and k.min() >= 1
    assert (k >= 2).mean() > 0.9


def test_neighbours_move_one_product_and_keep_100():
    w = np.zeros(len(A.U17), np.int64)
    w[[0, 1, 5]] = [50, 30, 20]
    V = A.neighbours(w, 5)
    assert (V.sum(axis=1) == 100).all()
    assert not (V == w).all(axis=1).any()
    assert len(V) == len(np.unique(V, axis=0))
    # 3 held products × up/down + 14 missing products added at 5 %
    assert len(V) == 3 * 2 + (len(A.U17) - 3)


def test_point_start_is_latest_first_date_of_held_products():
    fv = pd.Series({c: pd.Timestamp("2000-09-01") for c in A.U17})
    fv["446720"] = pd.Timestamp("2003-11-07")
    W = np.zeros((2, len(A.U17)), np.int64)
    W[0, 0] = 100
    W[1, [0, A.U17.index("446720")]] = 50
    st = pd.to_datetime(A.point_start(W, A.U17, fv))
    assert st[0] == pd.Timestamp("2000-09-01") and st[1] == pd.Timestamp("2003-11-07")


def _dates(start="2000-09-01", end="2024-12-31"):
    return pd.bdate_range(start, end)


def test_yearly_windows_constant_growth():
    d = _dates()
    g = 0.0003
    P = np.exp(g * np.arange(len(d)))[None, :].repeat(2, axis=0)
    P[1] *= 1 + 0.2 * (np.arange(len(d)) % 500 == 0)  # small bumps: some drawdown
    Y = A.yearly_windows(P, d)
    assert Y.shape == (2, len(A.CUT_YEARS), 6)
    exp_cagr = np.exp(g * 252) - 1
    assert np.allclose(Y[0, :, 0], exp_cagr, rtol=1e-4)
    assert np.allclose(Y[0, :, 1], 0.0, atol=1e-6)
    assert np.isnan(Y[0, 0, 2])  # 2005: fewer than ten years of history
    assert np.isfinite(Y[0, -1, 2])
    assert (Y[1, :, 1] <= 0).all()


def test_extra_metrics_flat_then_drop():
    d = _dates("2010-01-01", "2014-12-31")
    v = np.ones(len(d))
    v[300:] = 0.8  # one drop, never recovered
    m = A.extra_metrics(v[None, :], d)
    assert np.isnan(m["recovery_m"].iloc[0])
    assert m["underwater_m"].iloc[0] > 40
    assert m["cvar5_m"].iloc[0] < 0
    u = np.linspace(1, 2, len(d))
    m2 = A.extra_metrics(u[None, :], d)
    assert m2["ulcer"].iloc[0] == 0 and m2["underwater_m"].iloc[0] == 0


def test_pick_rules():
    c = np.array([0.10, 0.12, 0.08, np.nan])
    m = np.array([-0.10, -0.20, -0.05, -0.01])
    assert A.pick_r1(c, m) == 0
    assert A.pick_r3(c, m) == 2  # 0.08 / 0.05 is the best Calmar
    assert A.pick_r1(c, np.full(4, -0.5)) is None
    W = np.zeros((4, len(A.U17)), np.int64)
    W[:, 0] = [60, 40, 50, 100]
    W[:, 1] = [40, 60, 50, 0]
    w2 = A.pick_r2(W, c, m)
    assert w2.sum() == 100


def test_min_variance_and_max_diversification():
    S = np.diag([0.04, 0.01, 0.09])
    w = A.min_variance(S)
    inv = 1 / np.diag(S)
    assert np.allclose(w, inv / inv.sum(), atol=1e-4)
    md = A.max_diversification(S)  # uncorrelated: weights ∝ 1/σ
    sd = np.sqrt(np.diag(S))
    assert np.allclose(md, (1 / sd) / (1 / sd).sum(), atol=1e-3)
    assert np.isclose(w.sum(), 1) and (w >= 0).all()


def test_stage2_result_files_for_the_app():
    from lab import aa_experiments as ex
    w = ex.a42_2_walk()
    assert {"R1", "R4", "F5 S&P500"} <= set(w["규칙"])
    assert w["MDD"].le(0).all() and (w["24세트 최악 MDD"] <= w["MDD"] + 1e-9).all()
    r = ex.a42_2_rebalance()
    assert list(r.index) == ["월", "분기", "반기", "연"] and r.shape == (4, 4)
    assert ((r >= 0) & (r <= 100)).all().all()
