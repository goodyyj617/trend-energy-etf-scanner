"""Stage J (lab/pair_study.py, DECISIONS 88): pair definition, bins and chance count."""
import numpy as np
import pandas as pd

from lab import pair_study as pj


def test_registered():
    assert len(pj.RULES) == 15 and len(pj.pairs()) == 105 and pj.PAIR_WINDOW == 10
    assert list(pj.EXITS_J) == ["H1k2", "H1k3", "X1"] and pj.BINS == [0.0002, 0.0005, 0.001, 0.002, 0.004]
    assert pj.nearest_bin(0.0003) == 0.0002 and pj.nearest_bin(0.0004) == 0.0005 and pj.nearest_bin(0.02) == 0.004


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
