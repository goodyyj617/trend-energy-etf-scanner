"""Stage G-1 (lab/tail_study.py, DECISIONS 85) on small synthetic data."""
import numpy as np
import pandas as pd

from lab import tail_study as ts


def test_registered_menu():
    assert len(ts.FEATURES) == 40 and len(ts.MARKET) == 6 and set(ts.MARKET) <= set(ts.FEATURES)
    assert (ts.AUC_HI, ts.AUC_LO, ts.BIG_R, ts.R_ATR, ts.BIG_RET, ts.SIGMA) == (0.55, 0.45, 5.0, 2.0, 0.5, 2.0)
    assert ts.FEATURE_EXITS == {"X1": 1, "X9": 1} and ts.MAIN_LABEL == "b5r"


def test_auc_and_separates():
    x = np.arange(10, dtype=float)
    assert ts.auc(x, (x >= 5).astype(float)) == 1.0
    assert ts.auc(x, (x < 5).astype(float)) == 0.0
    assert ts.auc(np.ones(10), (x >= 5).astype(float)) == 0.5
    row = {"auc_h1": 0.56, "auc_h2": 0.57, "auc_p1993": 0.51, "auc_p2000": 0.6, "auc_p2009": 0.52, "auc_p2017": 0.55}
    assert ts.separates(row)
    assert not ts.separates({**row, "auc_p2009": 0.49})  # one period the other way
    assert not ts.separates({**row, "auc_h2": 0.54})  # one half below 0.55
    assert ts.separates({k: 1 - v for k, v in row.items()})  # the low side


def test_tail_stats_by_hand():
    r = np.array([1.0, 0.6, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1] * 10)
    m = pd.DataFrame({"ret": r, "r_mult": r / 0.1, "b2s": np.nan, "x_spy": r - 0.01, "x_ew": r})
    t = ts.tail_stats(m)
    assert t["signals"] == 100 and abs(t["net"] - (160 - 80) / 10) < 1e-9
    srt = np.sort(r)[::-1]
    assert abs(t["top5_share"] - srt[:5].sum() / r.sum()) < 1e-12
    assert t["b50"] == 0.2 and t["b5r"] == 0.2 and np.isnan(t["b2s"])
    assert abs(t["beat_spy"] - 0.2) < 1e-12


def test_b2s_uses_only_signals_sold_before():
    dates = pd.bdate_range("2020-01-01", periods=400)
    n = 60
    # 60 signals sold on day 10 with returns 0..0.59, then one signal on day 20 and one on day 5
    m = pd.DataFrame({"si": [0] * n + [20, 5], "sale": [10] * n + [30, 8], "closed": True,
                      "ret": list(np.arange(n) / 100) + [5.0, 5.0]})
    f = ts.b2s_flags(m, dates)
    assert f[n] == 1.0  # day 20 sees the 60 sold signals and 5.0 is far above
    assert np.isnan(f[n + 1])  # day 5: nothing sold yet (no lookahead)
    r = np.arange(n) / 100
    thr = r.mean() + 2 * r.std(ddof=1)
    m2 = m.copy()
    m2.loc[n, "ret"] = thr - 1e-9
    assert ts.b2s_flags(m2, dates)[n] == 0.0


def test_days_since_high_and_equal_weight():
    idx = pd.bdate_range("2020-01-01", periods=300)
    c = pd.DataFrame({"a": np.r_[np.arange(260, dtype=float), np.full(40, 100.0)],
                      "b": np.arange(300, dtype=float)}, index=idx)
    hi = c.rolling(252, min_periods=200).max()
    d = ts.days_since_high(c, hi, chunk=1)
    assert d["b"].iloc[-1] == 0 and d["a"].iloc[-1] == 40 and d["a"].iloc[259] == 0
    assert d.iloc[:199].isna().all().all()


def test_overlap_share():
    sig = pd.DataFrame({"symbol": ["A", "A", "A", "B"],
                        "signal_date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-02-01", "2020-01-02"]),
                        "exit_signal_date": pd.to_datetime(["2020-01-10", "2020-01-07", pd.NaT, "2020-01-03"])})
    assert ts.overlap_share(sig) == 0.25  # only A's second signal arrives while the first is held


def test_placebo_counts_shape():
    rng = np.random.default_rng(0)
    n = 4000
    dates = pd.to_datetime(rng.choice(pd.bdate_range("1993-02-01", "2023-12-29"), n))
    s = pd.DataFrame({"signal_date": dates, "b5r": (rng.random(n) < 0.1).astype(float)})
    for f in ts.FEATURES:
        s[f] = rng.random(n)
    s["r1"] = s["b5r"] + rng.random(n) * 0.5  # one real separator
    ft = ts.feature_table(s, "b5r")
    assert ft.set_index("feature").loc["r1", "separates"] and ft["separates"].sum() <= 3
    pl = ts.placebo(s, "b5r", 3)
    assert len(pl) == 6 and set(pl["kind"]) == {"signal", "month_block"}
    assert (pl["separating"] <= 3).all()  # shuffled labels break the real separator
