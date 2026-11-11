"""Stage U (lab/null_study.py, DECISIONS 101): the chance rate of the candidate procedure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab import basis_study as bs, exit_study as xs, goal_study as gs, large_study as ls, null_study as nu, pair_study as pj
from lab import regime_study as rs


def test_registered():
    assert nu.SEEDS == list(range(1001, 2001)) and nu.REPRO_SEEDS == [1, 2, 3, 4, 5]
    assert nu.EXITS == ["X1", "H1"] and nu.P1_NAMES == ["N10_W2_P1", "N20_W2_P1", "N50_W2_P1"]
    assert nu.P_FILTERS == ["R1A1", "R1A2", "R2A1", "R2A2", "R3A1", "R3A2", "R4A1", "R4A2", "R5A1", "R5A2"]
    assert nu.Q_FILTERS == ["none", "R1A1", "R3A1", "R4A1", "R5A1"] and nu.Q_BANDS == ["U1000", "U500"]
    assert (nu.CELLS_PER_SEED, nu.WORLD, nu.N_WORLDS, nu.BOUNDARY, nu.ALPHA) == (120, 15, 10_000, 0.10, 0.05)
    assert nu.FEATURE_KEYS == ["K2", "K8", "K13", "K11", "cap_rank", "dv20"]
    assert nu.TRIALS == 74_174 and nu.TOL == 1e-9


def test_job_lists_cover_the_120_cells():
    p, q = nu.p_jobs20("U1001", ["R0s1001"]), nu.q_jobs20("U1001", ["R0s1001"])
    assert len(p) == len(q) == 6
    cells = {(j[1], j[2], f) for j in p for f in j[3]} | {(j[1], j[2], f, b) for j in q for f in j[3] for b, _ in j[4]}
    assert len(cells) == 120
    assert all(j[4] == pj.COST_BPS and j[6] == "grid" and j[5] == ["R0s1001"] for j in p)
    assert all(j[5] == pj.COST_BPS and j[7] == "grid" and [b for b, _ in j[4]] == ["U1000", "U500"] for j in q)
    assert {j[2] for j in p + q} == set(rs.P1_SETTINGS)


def test_seed_tables_are_read_at_cell_zero(monkeypatch):
    """A seed's own tables hold it at entry cell 0; with E0_SEEDS = [seed] the stages read that cell
    from the paths basis_study writes."""
    root = Path("s1001")
    dirs = bs.table_dirs(root)
    monkeypatch.setitem(xs._W, "dirs", dirs)
    monkeypatch.setattr(pj, "E0_SEEDS", [1001])
    assert xs.table_of("R0s1001", "X1") == (bs.table_path(root, "R0__X1"), 0, nu.TABLE_YI["X1"])
    assert xs.table_of("R0s1001", "H1") == (bs.table_path(root, "R0__H1"), 0, nu.TABLE_YI["H1"])
    assert nu.TABLE_YI == {"X1": 0, "H1": xs.H_KI}


def test_clopper_pearson_and_the_boundary():
    n = 1000
    assert nu.cp_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-12)
    assert nu.cp_lower(0, n) == 0.0 and nu.cp_upper(n, n) == 1.0
    for k in (1, 7, 40):  # symmetry of the exact limits
        assert nu.cp_lower(k, n) == pytest.approx(1 - nu.cp_upper(n - k, n), abs=1e-10)
        assert nu.binom_cdf(k, n, nu.cp_upper(k, n)) == pytest.approx(0.05, abs=1e-9)
    labels = {k: nu.verdict(k, n)["label"] for k in range(0, 21)}
    assert all(labels[k] == "우연으로 보기 어려움" for k in (0, 1, 2))  # the design: <= 2 candidate seeds
    assert all(labels[k] == "가르지 못함" for k in range(3, 13))
    assert all(labels[k] == "우연으로 설명됨" for k in range(13, 21))  # >= 13
    v = nu.verdict(10, n)
    assert v["p_hat"] == 0.01 and v["P_hat"] == pytest.approx(1 - 0.99 ** 15)
    assert v["P_lower"] < v["P_hat"] < v["P_upper"]


def test_worlds_maxt_and_percentiles():
    W = nu.draw_worlds(1000)
    assert W.shape == (10_000, 15) and all(len(set(r)) == 15 for r in W[:50])
    assert np.array_equal(W, nu.draw_worlds(1000))  # fixed generator
    wm = np.array([0.0, 1.0, 2.0, 3.0])
    assert nu.maxt_p(wm, 2.0) == pytest.approx(3 / 5) and nu.maxt_p(wm, 9.0) == pytest.approx(1 / 5)
    assert nu.pct_rank(np.array([1.0, 2.0, 3.0, 4.0]), 3.0) == pytest.approx(62.5)
    assert nu.pct_rank(np.array([1.0, np.nan]), 5.0) == 100.0


def _acc(gen, src, path, cost, cells_good, bands=False):
    """Account rows (one per cell) with dd_cagr 0.08 where (setting, filter[, band]) is good, else 0.04."""
    rows = []
    filters = nu.Q_FILTERS if bands else nu.P_FILTERS
    for st in nu.P1_NAMES:
        for x in nu.EXITS:
            for f in filters:
                for b in (nu.Q_BANDS if bands else [None]):
                    key = (st, f, b) if bands else (st, f)
                    v = 0.08 if key in cells_good else 0.04
                    r = {"setting": st, "filter": f, "src": src, "order": 0, "trades": 200, "ws_mix": 0.7, "ws_ma200": 0.6,
                         **{f"dd_cagr_{p}": v for p in gs.GOAL_PERIODS}, "gen": gen, "exit": x, "cost": cost, "mode": "grid", "path": path}
                    if bands:
                        r["band"] = b
                    rows.append(r)
    return pd.DataFrame(rows)


def _e0(bands=False):
    rows = []
    for st in nu.P1_NAMES:
        for x in nu.EXITS:
            for f in (nu.Q_FILTERS if bands else nu.P_FILTERS):
                for b in (nu.Q_BANDS if bands else [None]):
                    r = {"setting": st, "gen": "E0", "exit": x, "filter": f, "cost": 20.0, "mode": "grid", "trades": 200, "ws_mix": 0.7,
                         "ws_ma200": 0.6, **{f"dd_cagr_{p}": 0.03 for p in gs.GOAL_PERIODS}, "orders": 20}
                    if bands:
                        r["band"] = b
                    rows.append(r)
    return pd.DataFrame(rows)


def test_candidate_path_routes_and_margins():
    bench = {b: {f"dd_cagr_{p}": 0.05 for p in gs.GOAL_PERIODS} for b in ("mix", "ma200", "ew")}
    # U1: route 95 at R3A1 for N10 and N20 (adjacent); U2: route 96 at N20 / N50 x R5A1 x both bands; U3: nothing
    g95 = {("N10_W2_P1", "R3A1"), ("N20_W2_P1", "R3A1")}
    g96 = {(st, "R5A1", b) for st in ("N20_W2_P1", "N50_W2_P1") for b in nu.Q_BANDS}
    parts = [_acc("U1", "R0s1", "P", 20.0, g95), _acc("U1", "R0s1", "Q", 20.0, set(), True),
             _acc("U2", "R0s2", "P", 20.0, set()), _acc("U2", "R0s2", "Q", 20.0, g96, True),
             _acc("U3", "R0s3", "P", 20.0, {("N50_W2_P1", "R1A1")}), _acc("U3", "R0s3", "Q", 20.0, set(), True),
             _acc("U1", "R0s1", "P", 40.0, g95), _acc("U2", "R0s2", "Q", 40.0, g96, True)]
    acc = nu.split_acc(pd.concat(parts, ignore_index=True))
    assert "band" not in acc["P20"] and len(acc["P20"]) == 3 * 60 and len(acc["Q20"]) == 3 * 60
    S = {"e0P": _e0(), "e0Q": _e0(True), "benchP": bench, "benchQ": bench}
    J = nu.judge_units(acc, S)
    t = nu.unit_table(J, bench, bench).set_index("gen")
    assert (t["cells"] == 120).all()
    assert t.loc["U1", "cand_95"] == 2 and t.loc["U1", "cand_96"] == 0 and t.loc["U1", "entry_specific"]
    assert t.loc["U2", "cand_95"] == 0 and t.loc["U2", "cand_96"] == 4 and t.loc["U2", "cand_cells"] == 4
    assert not t.loc["U3", "candidate"] and t.loc["U3", "pass_cells"] == 2
    assert t.loc["U1", "pass_cells"] == 4 and t.loc["U2", "pass_cells"] == 8
    assert t.loc["U1", "margin"] == pytest.approx(0.03) and t.loc["U2", "margin"] == pytest.approx(0.03)
    assert t.loc["U3", "margin"] == pytest.approx(-0.01)  # a lone passing setting: its neighbour binds
    cr = nu.candidate_rows(J)
    assert set(cr["route"]) == {"95", "96"} and len(cr) == 6
    # without 40 bp cells nothing is a candidate
    acc2 = dict(acc, P40=acc["P40"].iloc[:0], Q40=acc["Q40"].iloc[:0])
    t2 = nu.unit_table(nu.judge_units(acc2, S), bench, bench)
    assert not t2["candidate"].any()


def test_cell_margin():
    bench = {"mix": {f"dd_cagr_{p}": 0.04 for p in gs.GOAL_PERIODS}, "ma200": {f"dd_cagr_{p}": 0.06 for p in gs.GOAL_PERIODS}}
    c = pd.DataFrame({"dd_cagr_1993-2023": [0.07, 0.07], "dd_cagr_1993-2007": [0.08, 0.05], "dd_cagr_2008-2023": [0.065, 0.09],
                      "dd_cagr_2017-2023": [0.0, 0.0]})
    assert nu.cell_margin(c, bench) == pytest.approx([0.005, -0.01])  # 2017-2023 is not judged


def test_parse_seeds():
    assert nu.parse_seeds("1001-1003,1010") == [1001, 1002, 1003, 1010]
