"""Stage H (lab/pick_study.py, DECISIONS 86) on small synthetic data."""
import numpy as np
import pandas as pd

from lab import pick_study as ph, tail_study as ts


def test_registered():
    assert (ph.RISK, ph.CAP, ph.SLOTS, ph.BAN, ph.BIG_CONTRIB, ph.TOP_SHARE) == (0.01, 0.10, 10, 20, 0.05, 0.30)
    assert ph.K_VALUES == [1.5, 2.0, 3.0] and ph.MAIN == ("logit", True, "main") and len(ph.FEATURES) == 55
    assert (ph.FIRST_TEST, ph.AUC_MIN, ph.N_PLACEBO, ph.N_RANDOM) == (2000, 0.55, 100, 100)


def test_logit_and_auc():
    rng = np.random.default_rng(0)
    X = rng.random((20000, 2)) - 0.5
    y = (rng.random(20000) < 1 / (1 + np.exp(-(-1 + 3 * X[:, 0])))).astype(float)
    b = ph.fit_logit(X, y, lam=1.0)
    assert abs(b[1] - 3) < 0.3 and abs(b[2]) < 0.3 and abs(b[0] + 1) < 0.1
    x = rng.random(5000)
    yy = (rng.random(5000) < x).astype(float)
    assert abs(ph.hist_auc(x, yy) - ts.auc(x, yy)) < 0.002


def test_to_pct_and_weights():
    assert list(ph.to_pct(np.array([1.0, 2, 3, 4]), np.array([0.5, 2, 4, np.nan]))) == [0.0, 0.5, 1.0, 0.5]


def test_replay_risk_by_hand():
    # 4 days, 2 symbols, open = close = 10 flat except symbol 0 doubles on day 3
    O = np.array([[10, 10], [10, 10], [10, 10], [20, 10]], dtype=float)
    LC = O.copy()
    cash = np.zeros(4)
    # signal rows: both bought at day 1; symbol 0 sold at day 3 open, symbol 1 never sold
    ei, sj = np.array([1, 1]), np.array([0, 1])
    sale, closed = np.array([3, -1]), np.array([True, False])
    w = np.array([0.10, 0.05])
    eq, tr = ph.replay_risk(ei - 1, ei, sj, sale, closed, w, np.array([2.0, 1.0]), O, LC, cash, 0.0)
    assert np.allclose(eq, [1.0, 1.0, 1.0, 1.10])  # +100% on 10% of equity
    assert len(tr) == 2 and abs(tr.set_index("j").loc[0, "ret"] - 1.0) < 1e-12
    assert abs(tr.set_index("j").loc[0, "weight"] - 0.10) < 1e-12
    # one slot: only the higher key is bought
    eq1, tr1 = ph.replay_risk(ei - 1, ei, sj, sale, closed, w, np.array([1.0, 2.0]), O, LC, cash, 0.0, slots=1)
    assert list(tr1["j"]) == [1]
    # re-entry ban: symbol 0 signalled again on day 3 (bought day 3 open = exit day) is skipped
    eq2, tr2 = ph.replay_risk(np.array([0, 2]), np.array([1, 3]), np.array([0, 0]), np.array([3, -1]),
                              np.array([True, False]), np.array([0.1, 0.1]), np.array([1.0, 1.0]), O, LC, cash, 0.0)
    assert len(tr2) == 1


def test_calendar_weighted_scales_to_full():
    O = np.ones((3, 2)) * 10
    C = np.array([[10, 10], [11, 11], [11, 11]], dtype=float)
    cash = np.full(3, 0.01)
    # two positions of weight 0.8 bought at day 0, open to the end: gross 1.6 -> scaled to 1
    eq, gross = ph.calendar_weighted(np.array([0, 0]), np.array([3, 3]), np.array([False, False]), np.array([0, 1]),
                                     np.array([0.8, 0.8]), O, C, cash, 0.0)
    assert np.allclose(gross, [1.6, 1.6, 1.6])
    assert np.allclose(eq, np.cumprod([1.0, 1.1, 1.0]))
    # one position of weight 0.5: half in T-bills
    eq, _ = ph.calendar_weighted(np.array([0]), np.array([3]), np.array([False]), np.array([0]), np.array([0.5]), O, C, cash, 0.0)
    assert np.allclose(eq, np.cumprod([1 + 0.5 * 0.01, 1 + 0.5 * 0.1 + 0.5 * 0.01, 1 + 0.5 * 0.01]))
