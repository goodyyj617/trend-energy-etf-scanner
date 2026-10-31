"""Stage J / J3 (lab/pair_study.py, DECISIONS 88, 89): combination signals, bins, chance count, tests."""
import numpy as np
import pandas as pd

from lab import pair_study as pj


def test_registered():
    assert len(pj.RULES) == 15 and len(pj.pairs()) == 105 and pj.PAIR_WINDOW == 10
    assert list(pj.EXITS_J) == ["H1k2", "H1k3", "X1"] and pj.BINS == [0.0002, 0.0005, 0.001, 0.002, 0.004]
    assert pj.nearest_bin(0.0003) == 0.0002 and pj.nearest_bin(0.0004) == 0.0005 and pj.nearest_bin(0.02) == 0.004
    # Stage J3
    assert len(pj.combos(3)) == 455 and pj.N_NEW_J3 == 1_365 and pj.N_PLACEBO_J3 == 100
    assert pj.BINS_J3 == [0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004]
    assert pj.RET_EDGES[0] == -np.inf and -0.2 in pj.RET_EDGES and 0.5 in pj.RET_EDGES and 0.0 in pj.RET_EDGES


def test_triple_rows_span_and_latest_row(monkeypatch):
    def table(days):
        return pd.DataFrame({"symbol": [d[0] for d in days], "si": [d[1] for d in days], "tag": [str(d) for d in days]})
    tabs = {("A", "x"): table([("S", 100), ("T", 100), ("U", 50)]),
            ("B", "x"): table([("S", 95), ("T", 95), ("U", 50)]),
            ("C", "x"): table([("S", 104), ("T", 111), ("U", 50)])}
    monkeypatch.setattr(pj, "rule_table", lambda e, x: tabs[(e, x)])
    r = pj.combo_rows(("A", "B", "C"), "x").sort_values(["symbol", "si"])
    # S: 95, 100, 104 (span 9) -> bought after C's day 104, row from C's table; T: span 16 -> no;
    # U: all on day 50 -> one row
    assert list(zip(r["symbol"], r["si"], r["tag"])) == [("S", 104, "('S', 104)"), ("U", 50, "('U', 50)")]
    # a pair through the same function is the Stage J pair
    p = pj.pair_rows("A", "C", "x")
    assert set(zip(p["symbol"], p["si"])) == {("S", 104), ("U", 50)}


def test_mann_whitney_and_ks():
    auc, p = pj.mann_whitney(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert auc == 0.0 and abs(p - 0.0495) < 1e-3  # z = -4.5 / sqrt(5.25)
    auc, p = pj.mann_whitney(np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0, 3.0]))
    assert auc == 0.5 and abs(p - 1.0) < 1e-12
    assert pj.ks_stat(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])) == 1.0
    assert pj.ks_stat(np.arange(10.0), np.arange(10.0)) == 0.0
    rng = np.random.default_rng(0)
    st = pj.ret_stats(rng.normal(0.02, 0.1, 400), rng.normal(0.0, 0.1, 400), pj.boot_means(rng.normal(0.0, 0.1, 400), rng), rng)
    assert st["st_ci_lo"] < st["st_diff"] < st["st_ci_hi"] and st["st_mw_p"] < 0.05 and st["st_mw_auc"] > 0.5
    assert pj.ret_stats(np.zeros(5), np.zeros(50), np.zeros(10), rng) == {}


def test_pair_rows_window_and_order(monkeypatch):
    def table(days):
        return pd.DataFrame({"symbol": [d[0] for d in days], "si": [d[1] for d in days], "tag": [str(d) for d in days]})
    tabs = {("A", "x"): table([("S", 100), ("S", 200), ("T", 50)]),
            ("B", "x"): table([("S", 95), ("S", 215), ("T", 50), ("S", 300)])}
    monkeypatch.setattr(pj, "rule_table", lambda e, x: tabs[(e, x)])
    r = pj.pair_rows("A", "B", "x").sort_values(["symbol", "si"])
    # S100 (B at 95, within 10) from A; S215 is 15 after A200 -> no; T50 same day -> one row
    assert list(zip(r["symbol"], r["si"])) == [("S", 100), ("T", 50)]
    # the window is a gap of 0-10 trading days: 10 counts, 11 does not
    tabs = {("A", "x"): table([("S", 110), ("U", 111)]), ("B", "x"): table([("S", 100), ("U", 100)])}
    r = pj.pair_rows("A", "B", "x")
    assert list(zip(r["symbol"], r["si"])) == [("S", 110)]


def test_chance_count():
    vp = pd.DataFrame({"set": [f"p{i}" for i in range(100)], "meaningful_a": [i < 10 for i in range(100)],
                       "meaningful_b": [False] * 100})
    c = pj.chance(vp, 50, {"meaningful_a": 9, "meaningful_b": 1})
    assert abs(c["meaningful_a"]["p0"] - 0.1) < 1e-12 and c["meaningful_a"]["q95"] in (8, 9) and not c["meaningful_a"]["information"]
    assert c["meaningful_b"]["p0"] == 0 and c["meaningful_b"]["information"]
