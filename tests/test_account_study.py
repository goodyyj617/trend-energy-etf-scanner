"""Stage N (lab/account_study.py, DECISIONS 93): settings, sizing, Stage M's random draws, verdicts."""
import types
import zlib

import numpy as np
import pandas as pd

from lab import account_study as ns, exclude_study as ms, pair_study as pj, pick_study as ph


def test_registered():
    st = ns.settings()
    assert len(st) == len(set(st)) == 20 and ns.SLOTS_N == (10, 20, 30, 50) and ns.RISK_TOTAL == 0.10
    assert sum("A" in ns.grids(s) for s in st) == 12 and sum("B" in ns.grids(s) for s in st) == 12
    assert ns.EXITS == ["X1", "H1", "H9", "ST"] and ns.SIZE_STOP == {"X1": 3.0, "H1": 3.0, "H9": 3.0, "ST": 5.0}
    assert ns.N_NEW == len(st) * len(pj.RULES) * len(ns.EXITS) == 1_200 and ns.N_TRIALS_BEFORE == 70_574
    assert ns.N_PSEUDO == 10 and ns.PSEUDO_ORDERS == 20 and ns.N_ORDERS == 100 and ns.CANDIDATE_W == ("W2", "W3")
    assert ns.MIN_EXITS_Q2 == 3 and ns.MIN_EXITS_Q3 == 2 and ns.Q1_PP == 0.005 and ns.RULE == "M1"
    assert ns.base_setting("X1") == (10, "W1", "P0") and ns.base_setting("H1") == (10, "W2", "P0")
    assert ns.neighbours((10, "W3", "P0")) == [(20, "W3", "P0")] and ns.neighbours((30, "W2", "P2")) == [(20, "W2", "P2"), (50, "W2", "P2")]
    assert ns.parse_setting(ns.setting_name((50, "W3", "P0"))) == (50, "W3", "P0")


def test_sizing():
    stop = np.array([0.05, 0.08, 0.2, 0.0, np.nan])
    w, capped = ns.sizing(stop, 10, "W2")
    raw = np.where(stop > 0, ph.RISK / np.where(stop > 0, stop, 1.0), np.nan)
    assert np.array_equal(w, np.minimum(raw, ph.CAP), equal_nan=True)  # N = 10, W2 = the 1% rule with a 10% cap
    assert ns.RISK_TOTAL / 10 == ph.RISK and 1.0 / 10 == ph.CAP
    assert np.allclose(w[:3], [0.1, 0.1, 0.05]) and capped[:3].tolist() == [1.0, 1.0, 0.0] and np.isnan(w[3:]).all()
    w3, _ = ns.sizing(stop, 10, "W3")
    assert np.allclose(w3[:3], [0.2, 0.125, 0.05])
    w50, _ = ns.sizing(stop, 50, "W2")
    assert np.allclose(w50[:3], [0.02, 0.02, 0.002 / 0.2])  # r = 0.2%, cap 2%
    w1, c1 = ns.sizing(stop, 20, "W1")
    assert np.allclose(w1, 0.05) and np.isnan(c1).all()


def test_stage_m_draws_follow_exclude_study():
    rng = np.random.default_rng(0)
    for srcs, gen in ((["E2"], "E2"), (["R0s1", "R0s2"], "E0")):
        n_total = {s: 50 + 7 * i for i, s in enumerate(srcs)}
        keeps = {s: rng.random(n) < 0.7 for s, n in n_total.items()}
        pseudos = {s: [rng.random(n) < 0.7 for _ in range(ms.N_PSEUDO)] for s, n in n_total.items()}
        got = list(ns.stage_m_draws(gen, "H1", srcs, keeps, pseudos, n_total))
        # exclude_study.run_job's sequence for M1
        ref = np.random.default_rng([ms.SEED, zlib.crc32(f"M1|{gen}|H1|{pj.COST_BPS}".encode())])
        want = []
        for s in srcs:
            for o in range(ms.N_ORDERS // len(srcs)):
                want.append(("rule", -1, s, o, ref.random(int(keeps[s].sum()))))
            for k in range(ms.N_PSEUDO):
                for o in range(ms.PSEUDO_ORDERS // len(srcs)):
                    want.append(("pseudo", k, s, o, ref.random(int(pseudos[s][k].sum()))))
        want = [w for w in want if w[0] == "rule" or w[1] < ns.N_PSEUDO]
        assert len(got) == len(want) == len(srcs) * (ms.N_ORDERS // len(srcs) + ns.N_PSEUDO * (ms.PSEUDO_ORDERS // len(srcs)))
        for g, w in zip(got, want):
            mask = keeps[g[2]] if g[0] == "rule" else pseudos[g[2]][g[1]]
            assert g[:4] == w[:4] and np.array_equal(g[4][mask], w[4]) and np.isnan(g[4][~mask]).all()


def test_masked_replay_equals_subset_replay():
    rng = np.random.default_rng(3)
    n_days, n_names, n_sig = 120, 12, 300
    O = 50 * np.cumprod(1 + rng.normal(0, 0.02, (n_days, n_names)), axis=0)
    ns._W.update(O=O, LC=O * 1.001, cash=np.full(n_days, 0.0001))
    ei = rng.integers(1, n_days - 5, n_sig)
    r = types.SimpleNamespace(ei=ei, sj=rng.integers(0, n_names, n_sig), sale=np.minimum(ei + rng.integers(1, 30, n_sig), n_days - 1),
                              closed=np.ones(n_sig, dtype=bool))
    mask = rng.random(n_sig) < 0.6
    w = np.where(rng.random(n_sig) < 0.05, np.nan, 0.08)
    key = ns._spread(rng.random(int(mask.sum())), mask, n_sig)
    eq, tr, idx = ns._replay(r, mask, w, key, 10, 0.002)
    sub = mask & np.isfinite(w)
    eq2, tr2 = ph.replay_risk(r.ei[sub] - 1, r.ei[sub], r.sj[sub], r.sale[sub], r.closed[sub], w[sub], key[sub], O, O * 1.001,
                              np.full(n_days, 0.0001), 0.002, slots=10)
    assert np.array_equal(eq, eq2) and len(tr) == len(tr2) and np.array_equal(idx, np.flatnonzero(sub)[tr2["row"].to_numpy()])


def _acc_un():
    periods = list(ns.es.PERIODS)
    acc, un = [], []
    for st in ns.settings():
        s = ns.setting_name(st)
        lift = 0.01 * (st[0] >= 20)
        for gen in ["E1", "E2", "E0"]:
            for x in ns.EXITS:
                for kind, k, base in [("rule", -1, 0.10 + lift), ("none", -1, 0.08)] + [("pseudo", k, 0.08 + 0.001 * k) for k in range(10)]:
                    for o in range(2):
                        c = base + (0.02 if gen != "E0" and kind == "rule" else 0.0)
                        acc.append({"setting": s, "gen": gen, "exit": x, "kind": kind, "k": k, "src": gen, "order": o, "trades": 60,
                                    "sharpe": 0.5, "cap_rank_trades": 900.0, **{f"cagr_{p}": c for p in periods},
                                    **{f"mdd_{p}": -0.3 for p in periods}})
                    un.append({"setting": s, "gen": gen, "exit": x, "kind": kind, "k": k, "signals": 100,
                               **{f"mean_contrib_{p}": 0.002 if kind == "rule" else 0.001 for p in periods}})
    return pd.DataFrame(acc), pd.DataFrame(un)


def test_verdicts():
    acc, un = _acc_un()
    v = ns.summarize(acc, un)
    assert len(v) == 20 * 3 * 4 and v["beyond_q95"].all()
    e1 = v[v["gen"] == "E1"]
    assert np.allclose(v["pseudo_cagr_1993-2023"], 0.0845) and np.allclose(e1["e0_cagr_1993-2023"], e1["cagr_1993-2023"] - 0.02)
    pv = ns.pseudo_vs_pseudo(acc)
    assert len(pv) == 20 * 3 * 4 * 10 and pv.loc[pv["k"] == 9, "beyond_q95"].all() and not pv.loc[pv["k"] < 9, "beyond_q95"].any()
    t = ns.q1(v, pv)
    b = t.set_index("setting").loc["N10_W2_P0"]
    assert b["cells"] == 8 and abs(b["transfer_1993-2023"] - 3.55) < 1e-9 and abs(b["contrib_excess_pp"] - 0.1) < 1e-12
    assert t.loc[t["N"] >= 20, "grew"].all() and not t.loc[t["N"] == 10, "grew"].any()
    t2, p0 = ns.q2(v, acc)
    assert p0 == 0.0 and (t2["passing"] == 2).all() and (t2["q95"] == 0).all()
    base = pd.DataFrame([{"baseline": "SPY", **{f"cagr_{p}": 0.099 for p in ns.PERIODS}, **{f"mdd_{p}": -0.55 for p in ns.PERIODS}}])
    v["market_pass"] = ns.market_pass(v, base)
    assert v.loc[v["gen"] != "E0", "market_pass"].all() and not v.loc[v["gen"] == "E0", "market_pass"].any()
    pass40 = v["market_pass"] & (v["setting"] != "N50_W3_P0")
    cand = ns.candidates(v, pass40).set_index(["setting", "gen"])
    assert not cand.loc[("N10_W1_P0", "E1"), "candidate"]  # W1 is comparison only
    assert cand.loc[("N30_W3_P0", "E1"), "candidate"] and cand.loc[("N10_W3_P0", "E1"), "candidate"]
    assert not cand.loc[("N50_W3_P0", "E1"), "candidate"]  # fails at 40 bp
    assert cand.loc[("N20_W2_P1", "E2"), "candidate"] and cand.loc[("N20_W2_P1", "E2"), "neighbour"]
