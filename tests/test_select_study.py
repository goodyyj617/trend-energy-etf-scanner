"""Stage K (lab/select_study.py, DECISIONS 91): lines, day-matched pseudo rules, verdicts, new scores."""
import numpy as np
import pandas as pd

from lab import score_study, screen_study, screens, select_study as ks
from lab.metrics import deflated_sharpe


def test_registered():
    ls = ks.lines()
    assert len(ls) == 29 and len(ks.FEATURES) == 13 and set(ks.CAP_GROUPS) == {"K4m", "K4l", "K4s"}
    assert ks.CUT == 0.30 and ks.N_NEW == 261 and ks.N_PSEUDO == 40 and ks.PSEUDO_ORDERS == 20 and ks.N_ORDERS == 100
    assert list(ks.GENERATORS) == ["E0", "V3", "E2"] and len(ks.GENERATORS["E0"]) == 5 and ks.EXITS == ["H1k2", "H1k3", "X1"]
    assert ks.HYPOTHESIS == ("K1", "K2", "K3", "K5", "K6", "K7", "K8")
    assert all(screens.SELECT_SCORES.get(s) is not None for _, s, _, _ in ks.FEATURES.values())
    # favoured sides as registered: low price, low ivol, high turnover, low 12-1, high industry RS, low 36m, ...
    assert ks.FEATURES["K5"][3] == "low" and ks.FEATURES["K12"][3] == "low" and ks.FEATURES["K7"][1:3] == ("s_reversal", 36)


def test_line_mask():
    b = pd.DataFrame({"K2": [0.1, 0.3, 0.5, 0.71, np.nan], "K5": [0.1, 0.3, 0.5, 0.71, 0.9],
                      "cap_rank": [5, 1000, 1001, 2000, 2500]})
    assert ks.line_mask(b, "K2f").tolist() == [False, False, False, True, False]  # favoured = high score = low ivol
    assert ks.line_mask(b, "K2o").tolist() == [True, True, False, False, False]
    assert ks.line_mask(b, "K5f").tolist() == [True, True, False, False, False]  # favoured = low 12-1 return
    assert ks.line_mask(b, "K5o").tolist() == [False, False, False, True, True]
    assert ks.line_mask(b, "K4m").tolist() == [False, False, True, True, False]
    assert ks.line_mask(b, "K4l").tolist() == [True, True, False, False, False]
    assert ks.line_mask(b, "K4s").tolist() == [False, False, False, False, True]


def test_pseudo_mask_is_day_matched():
    rng = np.random.default_rng(0)
    si = np.sort(rng.integers(0, 50, 2000))
    keep = rng.random(2000) < 0.3
    m1 = ks.pseudo_mask(si, keep, np.random.default_rng(1))
    m2 = ks.pseudo_mask(si, keep, np.random.default_rng(1))
    m3 = ks.pseudo_mask(si, keep, np.random.default_rng(2))
    assert np.array_equal(m1, m2) and not np.array_equal(m1, m3)
    assert np.array_equal(np.bincount(si[m1], minlength=50), np.bincount(si[keep], minlength=50))
    shuffled = rng.permutation(len(si))  # works on unsorted signal days too
    m4 = ks.pseudo_mask(si[shuffled], keep[shuffled], np.random.default_rng(3))
    assert np.array_equal(np.bincount(si[shuffled][m4], minlength=50), np.bincount(si[keep], minlength=50))


def test_verdicts_and_chance():
    rows = []
    for k in range(-1, 4):  # rule (k = -1) above every pseudo in both halves
        for o in range(3):
            for x in ("H1k2", "H1k3", "X1"):
                base = 0.10 if k == -1 else 0.05 + 0.005 * k
                rows.append({"line": "K1f", "gen": "E0", "exit": x, "kind": "rule" if k == -1 else "pseudo", "k": k, "trades": 50,
                             "sharpe": 0.5, "cap_rank_trades": 1500, **{f"cagr_{p}": base for p in ks.es.PERIODS},
                             **{f"mdd_{p}": -0.3 for p in ks.es.PERIODS}})
    acc = pd.DataFrame(rows)
    un = pd.DataFrame([{"line": "K1f", "gen": "E0", "exit": x, "kind": "rule" if k == -1 else "pseudo", "k": k, "signals": 100,
                        "kept_share": 0.3, "per_day": 1.0, "hold_median": 50, "stop_share": 0.5, "big_share": 0.1,
                        **{f"mean_contrib_{p}": (0.01 if k == -1 else 0.005) for p in ks.es.PERIODS}}
                       for k in range(-1, 4) for x in ("H1k2", "H1k3", "X1")])
    v = ks.summarize(acc, un)
    assert v["b_halves"].all() and v["a_halves"].all() and (v["pct_in_pseudo_1993-2023"] == 1.0).all()
    lv = ks.per_line(v)
    assert bool(lv.iloc[0]["meaningful_b"]) and bool(lv.iloc[0]["meaningful_a"])
    pv = ks.pseudo_vs_pseudo(acc, un)
    assert len(pv) == 12 and pv[pv["k"] == 3]["b_halves"].all() and pv[pv["k"] == 0]["b_below"].all()
    assert ks.q95(9, 0.2) == 4 and ks.q95(29, 0.0) == 0


def test_dsr_matches_metrics():
    rng = np.random.default_rng(5)
    eq = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.01, 2000)), index=pd.bdate_range("2000-01-03", periods=2000))
    m = ks._moments(eq.to_numpy())
    assert abs(ks.dsr(m["m_sr"], m["m_g3"], m["m_g4"], m["m_T"], 500, 1e-5) - deflated_sharpe(eq, 500, 1e-5)) < 1e-9
    assert abs(ks.psr(m["m_sr"], m["m_g3"], m["m_g4"], m["m_T"], 0.0) - deflated_sharpe(eq, 1, 0.0)) < 1e-9


def test_new_scores_leave_registered_menus_and_x_ivol_unchanged():
    assert set(screens.STAGE_K_SCORES) == {"s_low_price", "s_low_ivol"}
    assert not set(screens.STAGE_K_SCORES) & set(score_study.SCORE_KEYS)
    assert not set(screens.STAGE_K_SCORES) & set(screen_study.SELECT_KEYS)
    assert len(score_study.SCORE_KEYS) == len(screens.SELECT_SCORES) - 2
    idx = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(1)
    close = pd.DataFrame({"SPY": np.cumprod(1 + rng.normal(0, 0.01, 120)), "A": np.cumprod(1 + rng.normal(0, 0.02, 120)),
                          "B": np.cumprod(1 + rng.normal(0, 0.03, 120))}, index=idx) * 50
    panel = type("P", (), {"close": close})()
    lp = screens.SELECT_SCORES["s_low_price"].compute(panel, None)
    assert np.allclose(lp, -np.log(close))
    iv = screens.SELECT_SCORES["s_low_ivol"].compute(panel, None)
    # the old exclusion screen, recomputed by its original formula
    r = close.pct_change(fill_method=None)
    m = r["SPY"]
    beta = r.rolling(63, min_periods=40).cov(m).div(m.rolling(63, min_periods=40).var(), axis=0)
    old = (r - beta * m.to_numpy()[:, None]).rolling(63, min_periods=40).std()
    assert np.allclose(-iv.to_numpy(), old.to_numpy(), equal_nan=True)
    assert screens.EXCLUDE_SCREENS["x_ivol"].compute(panel, 50).equals(old.rank(axis=1, pct=True) > 0.5)
