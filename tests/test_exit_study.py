"""Stage L (lab/exit_study.py, DECISIONS 90): exits, engine menu, partial sales, replay, tables, chance."""
from pathlib import Path

import numpy as np
import pandas as pd

from lab import exit_study as xs, pair_study as pj, pick_study as ph, stage_c_study as sc


def test_registered():
    assert list(xs.EXITS_L) == ["X1", "H1", "CH", "PS", "LB", "MA", "H9", "TP", "ST", "RS", "MN"]
    assert xs.new_exits() == ["PS", "LB", "MA", "ST", "RS", "MN"]
    assert xs.N_NEW == 4_600 and xs.MIN_EXITS == 3 and xs.EXITS_L["TP"]["partial"] == (50.0, 0.5)
    assert xs.EXITS_L["X1"]["stop"] is None and xs.EXITS_L["CH"]["stop"] == ("atr22", 3.0) and xs.EXITS_L["ST"]["stop"] == ("atr14", 5.0)


def test_table_menu(monkeypatch):
    for k in ("ENTRIES", "EXITS", "EXIT_EXTRA", "ENTRY_CANON", "EXIT_CANON"):
        monkeypatch.setattr(sc, k, dict(getattr(sc, k)))
    monkeypatch.setattr(sc, "SAMPLES", dict(sc.SAMPLES))
    xs.use_table_menu(["E2", "V3", "R0"], ["PS", "ST"], [1, 2])
    assert sc.ENTRIES["E2"] == ("c_golden", [50]) and sc.ENTRIES["V3"][1] == [4] and sc.ENTRIES["R0"] == ("g_random", [1, 2])
    assert sc.rules("E2", "PS", 0, 0)[1] == {"psar_flip": 0.02, "atr_stop": 3.0}
    assert sc.rules("R0", "ST", 1, 0) == ({"g_random": 2}, {"atr_stop": 5.0}, None)
    assert sc.cells("R0", "PS") == [(0, 0), (1, 0)] and list(sc.SAMPLES) == ["full"]


def test_partial_sales():
    n, k = 12, 3
    O = np.full((n, k), 10.0)
    C = np.full((n, k), 11.0)
    C[4, 0] = 15.0  # symbol 0: +50% at the close of day 4 -> sold at the open of day 5
    C[4, 1] = 15.0
    O[5, 1] = np.nan  # symbol 1: no open on day 5 -> day 6
    C[7, 2] = 16.0  # symbol 2: the threshold only on the full exit's signal day 7 -> no partial
    ei, sj = np.array([2, 2, 2, 2]), np.array([0, 1, 2, 0])
    f_excl = np.array([7, 7, 7, 5])  # the last row: exit signal on day 5, sale at the open of day 5 still allowed
    assert xs.partial_sales(C, O, ei, sj, f_excl, n - 1, 0.5).tolist() == [5, 6, -1, 5]
    # the sale must come no later than the full exit's signal day (or the window end for open positions)
    assert xs.partial_sales(C, O, np.array([2]), np.array([1]), np.array([6]), n - 1, 0.5).tolist() == [6]
    assert xs.partial_sales(C, O, np.array([2]), np.array([1]), np.array([5]), n - 1, 0.5).tolist() == [-1]
    assert xs.partial_sales(C, O, np.array([2]), np.array([0]), np.array([12]), 4, 0.5).tolist() == [-1]


def _account(psale=None):
    n, k = 10, 2
    O = np.tile(np.array([[10.0, 20.0]]), (n, 1))
    O[4, 0] = 15.0  # symbol 0 open at 15 on day 4 (partial sale), 12 on day 7 (full sale)
    O[7, 0] = 12.0
    LC = O.copy()
    ei, sj = np.array([1, 1]), np.array([0, 1])
    sale, closed = np.array([7, 9]), np.array([True, False])
    w = np.array([0.1, 0.1])
    return ph.replay_risk(ei - 1, ei, sj, sale, closed, w, np.array([2.0, 1.0]), O, LC, np.zeros(n), 0.0, psale=psale)


def test_replay_partial():
    eq0, tr0 = _account()
    eq1, tr1 = _account(np.array([-1, -1]))
    assert np.array_equal(eq0, eq1) and tr0.equals(tr1)  # no partial sale: the account is unchanged
    eq, tr = _account(np.array([4, -1]))
    r0 = tr.set_index("j").loc[0, "ret"]
    assert abs(r0 - (0.5 * 15 + 0.5 * 12) / 10 + 1) < 1e-12  # both sales in one trade return
    # cash after the partial sale on day 4: 0.8 + 0.05 x 1.5; equity then marks the half left at 15
    assert abs(eq[4] - (0.8 + 0.075 + 0.05 * 1.5 + 0.1)) < 1e-12
    assert abs(eq[7] - (0.8 + 0.075 + 0.05 * 1.2 + 0.1)) < 1e-12 and abs(eq0[7] - (0.8 + 0.1 * 1.2 + 0.1)) < 1e-12


def test_table_of(monkeypatch):
    d = {k: Path(k) for k in ("stage_h", "stage_i", "v2", "events", "r0x1", "tables")}
    monkeypatch.setitem(xs._W, "dirs", d)
    monkeypatch.setattr(pj, "E0_SEEDS", [1, 2, 3, 4, 5])
    f = lambda e, x: (xs.table_of(e, x)[0].as_posix(), *xs.table_of(e, x)[1:])  # noqa: E731
    assert f("E8", "H9") == ("v2/signals/full/E8__X9_signals.parquet", 2, 2)
    assert f("V3", "CH") == ("events/signals/full/V3__X2_signals.parquet", 1, 1)
    assert f("E1", "TP") == ("stage_h/h1/signals/full/E1__H1_signals.parquet", 0, 2)
    assert f("E1", "PS") == ("tables/signals/full/E1__PS_signals.parquet", 0, 0)
    assert f("R0s3", "H9") == ("stage_i/r0/signals/full/R0__X9_signals.parquet", 2, 2)
    assert f("R0s3", "X1") == ("r0x1/signals/full/R0__X1_signals.parquet", 2, 0)
    assert f("R0s3", "CH") == ("tables/signals/full/R0__CH_signals.parquet", 2, 0)
    assert f("R0s2", "H1k2") == ("stage_i/r0/signals/full/R0__H1_signals.parquet", 1, 1)


def test_chance_by_exit():
    rows = [{"set": f"t{i}", "kind": "triple", "exit": "X1", "trades": 40 if i < 9 else 10, "a_halves": i < 1, "b_halves": i < 5}
            for i in range(10)]
    rows += [{"set": f"p{i}", "kind": "placebo", "exit": "X1", "trades": 50, "a_halves": i < 2, "b_halves": i < 2} for i in range(10)]
    c = xs.chance_by_exit(pd.DataFrame(rows), kinds=("triple",))["X1"]["triple"]
    assert c["n_judged"] == 9 and c["excluded_few_trades"] == 1
    assert c["b"]["observed"] == 5 and abs(c["b"]["p0"] - 0.2) < 1e-12 and c["b"]["q95"] == 4 and c["b"]["information"]
    assert c["a"]["observed"] == 1 and not c["a"]["information"]
