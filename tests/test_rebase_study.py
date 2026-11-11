"""Stage S (lab/rebase_study.py, DECISIONS 99): the headline conclusions re-measured on the raw-price universe."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab import crsp, event_study as ev, exit_study as xs, pair_study as pj, pick_study as ph, rebase_study as rb, stage_c_study as sc


def test_registered():
    assert rb.START == "1993-02-01" and rb.END == "2023-12-31"
    assert rb.ENTRIES == [f"E{i}" for i in range(1, 10)] + [f"V{i}" for i in range(1, 7)]
    assert list(rb.GENS) == rb.ENTRIES + ["E0"] and rb.SEEDS == ["R0s1", "R0s2", "R0s3", "R0s4", "R0s5"]
    assert rb.EXITS == ["X1", "H1"] and rb.SETTINGS == [(10, "W2", "P0"), (10, "W2", "P1"), (20, "W2", "P1"), (50, "W2", "P1")]
    assert rb.A1 == ["R1A1", "R3A1", "R4A1", "R5A1"] and rb.A2 == ["R1A2", "R2A2", "R3A2", "R4A2", "R5A2"]
    assert rb.REUSED == ("E7", "E0") and rb.REPRO_TABLES == ("E2", "V3") and rb.TOL == 1e-9
    assert (rb.K1_SYSTEMS, rb.K1_CHANGED, rb.K5_PP, rb.NEAR_PP) == (6, 2, 0.005, 0.005)
    assert rb.TRIALS == 74_174


def _fresh_menus(monkeypatch):
    for name in ("ENTRIES", "ENTRY_CANON", "EXITS", "EXIT_CANON", "EXIT_EXTRA", "EXIT_BREAKEVEN", "SAMPLES"):
        monkeypatch.setattr(sc, name, dict(getattr(sc, name)))


def test_table_menus_equal_the_stored_tables_menus(monkeypatch):
    """Each rebuilt table runs the stored table's cell: Stage C v2 (E, X1 at the canonical entry value
    and 25% trailing), Stage G (V, X1) and Stage H (H1 at k = 3)."""
    _fresh_menus(monkeypatch)
    stored_x1 = {e: sc.config("full", e, "X1", sc.ENTRY_CANON[e], 1) for e in rb.ENTRIES if e.startswith("E")}
    ev.use_menu()
    stored_x1.update({e: sc.config("full", e, "X1", 1, 1) for e in ev.EVENTS})
    ph.use_h1_menu()
    stored_h1 = {e: sc.config("full", e, "H1", 0, 2) for e in rb.ENTRIES}
    for e in rb.ENTRIES:
        _fresh_menus(monkeypatch)
        rb.table_menu(e, "X1")
        assert sc.config("full", e, "X1", 0, 0) == stored_x1[e], e
        rb.table_menu(e, "H1")
        assert sc.config("full", e, "H1", 0, 0) == stored_h1[e], e
    assert stored_x1["E8"].entries == {"c_tsmom": 12}  # E8's canonical value is its third
    assert list(stored_h1["V3"].exits) == ["atr_stop", "trailing_pct"]


def test_table_spec_is_where_exit_study_reads(monkeypatch):
    dirs = rb.table_dirs(Path("root"))
    monkeypatch.setitem(xs._W, "dirs", dirs)
    for e in rb.ENTRIES:
        for x in rb.EXITS:
            f, xi, yi = xs.table_of(e, x)
            assert (f, xi, yi) == (rb.table_in(dirs, e, x), *rb.table_spec(e, x)[2:]), (e, x)
    assert rb.table_spec("E8", "X1")[2] == 2 and rb.table_spec("V3", "X1")[2:] == (1, 1) and rb.table_spec("E2", "H1")[2:] == (0, xs.H_KI)
    assert xs.table_of("R0s3", "X1") == (dirs["r0x1"] / "signals" / "full" / "R0__X1_signals.parquet", 2, 0)


def test_job_lists_are_the_stages_grids():
    p, q, o, u = rb.p_jobs(rb.GENS), rb.q_jobs(rb.GENS), rb.o_jobs(rb.GENS), rb.u_jobs(rb.GENS)
    grid_p = [j for _, j in p if j[6] == "grid"]
    assert len(grid_p) == 16 * 2 * (6 + 3) == 288 and len(p) - len(grid_p) == 15 * 2 * 5  # Stage P: 438 jobs
    assert len(q) == 301 and sum(j[7] == "pseudo" for _, j in q) == 45  # Stage Q's log: 301 (grid 256, pseudo 45)
    assert len(o) == 16 * 2 * 3 and len(u) == 20 * 2
    names = [rb.job_file(j) for j in p + q + o + u]
    assert len(set(names)) == len(names)
    assert ("p", "E1__X1__N20_W2_P1__grid__all") in names and ("q", "E0__H1__N10_W2_P0__grid__R3A1") in names
    assert ("p", "V2__X1__N10_W2_P0__placebo__R4") in names and ("unl", "unl__R0s2__H1") in names
    # every filter of both stages is covered once per gen x exit x setting
    cover = {}
    for _, j in p:
        if j[6] == "grid":
            for f in j[3]:
                cover[(j[0], j[1], j[2], f)] = cover.get((j[0], j[1], j[2], f), 0) + 1
    assert set(cover.values()) == {1} and len(cover) == 16 * 2 * 4 * 11


def test_compare():
    k = ["a", "b"]
    s = pd.DataFrame({"a": ["x", "x", "y"], "b": [1.0, 2.0, 1.0], "v": [0.1, np.nan, 0.3], "w": [1, 2, 3], "orders": [5, 5, 5]})
    m = s.copy()
    assert rb.summary_of(rb.compare(m, s, k))["pass"]
    m.loc[0, "v"] += 1e-8
    c = rb.summary_of(rb.compare(m, s, k))
    assert not c["pass"] and c["over_tol"] == 1 and c["max_diff"] == pytest.approx(1e-8)
    m = s.iloc[:2].copy()
    c = rb.summary_of(rb.compare(m, s, k))
    assert c["missing"] == 1 and not c["pass"]
    m = pd.concat([s, pd.DataFrame({"a": ["z"], "b": [0.0], "v": [0.0], "w": [0], "orders": [1]})], ignore_index=True)
    c = rb.summary_of(rb.compare(m, s, k))
    assert c["extra"] == 1 and not c["pass"]
    m = s.copy()
    m.loc[1, "v"] = 0.2  # NaN in the stored table, a value here
    assert not rb.summary_of(rb.compare(m, s, k))["pass"]


def _k1_frames(lift: dict, seed_lift: dict | None = None):
    rows, acc = [], []
    for st in rb.SETTING_NAMES:
        for x in rb.EXITS:
            for e in rb.ENTRIES + ["E0"]:
                v = 0.05 + lift.get(e, 0.0)
                rows.append({"setting": st, "gen": e, "exit": x, "filter": "none", "cost": 20.0, "mode": "grid",
                             **{f"dd_cagr_{p}": v for p in ("1993-2023", "1993-2007", "2008-2023")}})
            for s in rb.SEEDS:
                for o in range(4 if st.endswith("P0") else 1):
                    v = 0.05 + (seed_lift or {}).get(s, 0.0) + o * 1e-4
                    acc.append({"setting": st, "gen": "E0", "exit": x, "filter": "none", "cost": 20.0, "mode": "grid", "src": s, "order": o,
                                **{f"dd_cagr_{p}": v for p in ("1993-2023", "1993-2007", "2008-2023")}})
    unl = []
    for s in rb.ENTRIES + rb.SEEDS:
        for x in rb.EXITS:
            base = 0.10 + (0.03 if s == "E5" else 0.0) + (0.005 * int(s[3:]) if s.startswith("R0s") else 0.0)
            unl.append({"src": s, "exit": x, **{f"cagr_{p}": base for p in ("1993-2023", "1993-2007", "2008-2023")}})
    return pd.DataFrame(rows), pd.DataFrame(unl), pd.DataFrame(acc)


def test_k1_table():
    g, unl, acc = _k1_frames({"E5": 0.01, "E6": 0.01})  # E5: both parts; E6: slot accounts only
    k1, seeds = rb.k1_table(g, unl, acc)
    k = k1.set_index("entry")
    assert k.loc["E5", "a_pass"] and k.loc["E5", "b_systems"] == 8 and k.loc["E5", "above_random"]
    assert not k.loc["E6", "a_pass"] and k.loc["E6", "b_pass"] and not k.loc["E6", "above_random"]
    assert int(k1["above_random"].sum()) == 1
    s = seeds.set_index("seed")
    assert s.loc["R0s5", "a_pass"] and not s.loc["R0s4", "a_pass"]  # the largest seed beats the other four
    assert not s["b_pass"].any()  # equal seeds (P0 orders spread the same way): nothing beats the others
    g, unl, acc = _k1_frames({}, {"R0s2": 0.01})
    _, seeds = rb.k1_table(g, unl, acc)
    assert seeds.set_index("seed").loc["R0s2", "b_systems"] == 8


def test_k5_table_and_shortfall():
    rows = []
    for st in rb.SETTING_NAMES:
        for x in rb.EXITS:
            for f in ["none"] + rb.A1:
                for b, d in (("U3000", 0.0), ("U1000", 0.006), ("U500", -0.001)):
                    rows.append({"setting": st, "exit": x, "filter": f, "band": b, "dd_cagr_2008-2023": 0.05 + d,
                                 "dd_cagr_1993-2007": 0.06, "dd_cagr_1993-2023": 0.055})
    k5 = rb.k5_table(pd.DataFrame(rows))
    assert len(k5) == 80 and (k5.groupby("band").size() == 40).all()
    assert k5.loc[k5["band"] == "U1000", "d_2008-2023"].median() == pytest.approx(0.006)
    bench = {b: {f"dd_cagr_{p}": 0.06 for p in ("1993-2023", "1993-2007", "2008-2023")} for b in ("mix", "ma200")}
    bench["ma200"]["dd_cagr_2008-2023"] = 0.065
    row = {"dd_cagr_1993-2023": 0.07, "dd_cagr_1993-2007": 0.07, "dd_cagr_2008-2023": 0.062, "ws_mix": 0.6, "ws_ma200": 0.5, "trades": 40}
    s = rb.shortfall(row, bench)
    assert s["shortfall_pp"] == pytest.approx(0.3) and s["worst"] == "ma200_2008-2023" and s["ws_ok"] and s["trades_ok"]
    row.update({"dd_cagr_2008-2023": 0.07, "ws_ma200": 0.4})
    s = rb.shortfall(row, bench)
    assert s["shortfall_pp"] == 0.0 and not s["ws_ok"]


def test_use_basis_restores():
    before_env, before = os.environ.get("LAB_PRICE_BASIS"), crsp.DEFAULT_BASIS
    with rb.use_basis("adjusted"):
        assert os.environ["LAB_PRICE_BASIS"] == "adjusted" and crsp.resolve_basis(None) == "adjusted"
    assert os.environ.get("LAB_PRICE_BASIS") == before_env and crsp.DEFAULT_BASIS == before


def test_k_labels():
    assert rb.k_labels({"K1": {"changed": False}, "K2": {"changed": True}}) == {"K1": "유지", "K2": "바뀜"}
