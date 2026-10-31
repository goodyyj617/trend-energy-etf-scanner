"""Stage M (lab/exclude_study.py, DECISIONS 92): exclusion rules, day-matched pseudo exclusions, verdicts."""
import numpy as np
import pandas as pd

from lab import exclude_study as ms, exit_study as xs, pair_study as pj, select_study as ks


def test_registered():
    assert list(ms.EXCLUSIONS) == ["M1", "M2", "M3"] and ms.CUT == 0.30
    assert ms.EXCLUSIONS["M1"][1] == ("K2",) and ms.EXCLUSIONS["M2"][1] == ("K8",) and ms.EXCLUSIONS["M3"][1] == ("K2", "K13", "K11")
    # the excluded side is the low percentile of scores whose favoured side is high (low ivol, low vol, shallow dd, near the high)
    assert all(ks.FEATURES[f][3] == "high" for f in ms.FEATURES)
    assert [ks.FEATURES[f][1:3] for f in ms.FEATURES] == [("s_low_ivol", None), ("s_lowvol", None), ("s_dd_shallow", 12), ("s_high52", None)]
    assert ms.EXITS == list(xs.EXITS_L) and len(ms.EXITS) == 11
    g = ms.generators()
    assert list(g)[:15] == pj.RULES and g["E0"] == [f"R0s{s}" for s in (1, 2, 3, 4, 5)]
    assert ms.N_NEW == len(ms.EXCLUSIONS) * len(pj.RULES) * len(ms.EXITS) == 495
    assert ms.N_TRIALS_BEFORE == 70_079 and ms.N_ORDERS == 100 and ms.N_PSEUDO == 20 and ms.PSEUDO_ORDERS == 20
    assert ms.MIN_INFO_EXITS == 6 and ms.MIN_EXITS == 3 and ms.GAP_PP == 0.01 and ms.SIZE_RHO == 0.5
    # the cells without exclusion use exit_study.run_set's random generator names
    assert pj.set_name(("e0full", 1)) == "e0full__1" and pj.set_name(("single", "E2")) == "single__E2"


def test_excluded():
    b = pd.DataFrame({"K2": [0.1, 0.3, 0.31, 0.9, np.nan, 0.8], "K8": [0.9, 0.9, 0.2, 0.9, 0.1, 0.8],
                      "K13": [0.9, 0.9, 0.9, 0.25, 0.9, np.nan], "K11": [0.9, 0.9, 0.9, 0.9, 0.9, 0.3]})
    assert ms.excluded(b, "M1").tolist() == [True, True, False, False, False, False]  # NaN excludes nothing
    assert ms.excluded(b, "M2").tolist() == [False, False, True, False, True, False]
    assert ms.excluded(b, "M3").tolist() == [True, True, False, True, False, True]  # union of K2, K13, K11


def test_pseudo_exclusion_is_day_matched():
    rng = np.random.default_rng(0)
    si = rng.integers(0, 40, 3000)
    b = pd.DataFrame({"si": si, "K2": rng.random(3000), "K8": rng.random(3000), "K13": rng.random(3000), "K11": rng.random(3000)})
    for m in ms.EXCLUSIONS:
        keep = ~ms.excluded(b, m)
        pm = ks.pseudo_mask(si, keep, np.random.default_rng(1))
        assert np.array_equal(np.bincount(si[~pm], minlength=40), np.bincount(si[~keep], minlength=40))
    assert ms.excluded(b, "M3").mean() > ms.excluded(b, "M1").mean()


def test_gap_verdict():
    assert ms.gap_verdict(0.02) == "줄었다" and ms.gap_verdict(-0.02) == "커졌다"
    assert ms.gap_verdict(0.005) == "그대로다" and ms.gap_verdict(-0.01) == "그대로다" and ms.gap_verdict(np.nan) == "판정 불가"


def _synthetic():
    periods = list(ms.es.PERIODS)
    acc, un = [], []
    gens, exits = ["E1", "E2", "E0"], ["X1", "H1"]
    for gi, gen in enumerate(gens):
        for xi, x in enumerate(exits):
            cell = gi * len(exits) + xi
            for o in range(3):  # no exclusion
                acc.append({"rule": "none", "gen": gen, "exit": x, "kind": "none", "k": -1, "src": gen, "trades": 60, "sharpe": 0.4,
                            "cap_rank_trades": 1800.0, **{f"cagr_{p}": 0.07 for p in periods}, **{f"mdd_{p}": -0.3 for p in periods}})
            for k in range(-1, 4):
                for o in range(3):
                    c = 0.10 + 0.002 * cell if k == -1 else 0.05 + 0.005 * k
                    acc.append({"rule": "M1", "gen": gen, "exit": x, "kind": "rule" if k == -1 else "pseudo", "k": k, "src": gen,
                                "trades": 50, "sharpe": 0.5, "cap_rank_trades": 1700.0 - 10 * cell if k == -1 else 1800.0,
                                **{f"cagr_{p}": c for p in periods}, **{f"mdd_{p}": -0.3 for p in periods}})
                un.append({"rule": "M1", "gen": gen, "exit": x, "kind": "rule" if k == -1 else "pseudo", "k": k, "src": "*", "signals": 100,
                           "kept_share": 0.7, "per_day": 1.0, "hold_median": 50, "stop_share": 0.5, "big_share": 0.1, "cap_rank_signals": 1500,
                           **{f"n_{p}": 10 for p in periods}, **{f"n_ret_{p}": 10 for p in periods},
                           **{f"mean_contrib_{p}": (0.01 if k == -1 else 0.005) for p in periods},
                           **{f"mean_ret_{p}": 0.02 for p in periods}})
            un.append({"rule": "none", "gen": gen, "exit": x, "kind": "none", "k": -1, "src": "*", "signals": 140,
                       **{f"n_{p}": 14 for p in periods}, **{f"mean_contrib_{p}": 0.004 for p in periods}})
    base = pd.DataFrame([{"baseline": "SPY", **{f"cagr_{p}": 0.132 for p in periods}, **{f"mdd_{p}": -0.5 for p in periods}},
                         {"baseline": "SPY 200일선", **{f"cagr_{p}": 0.09 for p in periods}, **{f"mdd_{p}": -0.2 for p in periods}},
                         {"baseline": "동일비중 유니버스", **{f"cagr_{p}": 0.085 for p in periods}, **{f"mdd_{p}": -0.55 for p in periods}}])
    return pd.DataFrame(acc), pd.DataFrame(un), base


def test_summary_chance_gap_size():
    acc, un, base = _synthetic()
    v = ms.summarize(acc, un)
    assert len(v) == 6 and v["b_halves"].all() and v["a_halves"].all()
    assert np.allclose(v["none_cagr_2017-2023"], 0.07) and np.allclose(v["none_contrib_1993-2007"], 0.004)
    assert np.allclose(v["pseudo_cagr_1993-2023"], 0.0575)  # median of the four pseudo medians
    pv = ms.pseudo_vs_pseudo(acc, un)
    assert len(pv) == 24 and pv["b_halves"].mean() == 0.5
    ch = ms.chance(v, pv)
    x1 = ch["M1"]["by_exit"]["X1"]
    assert x1["n_judged"] == 2 and x1["b"]["observed"] == 2 and x1["b"]["p0"] == 0.5 and x1["b"]["q95"] == 2
    assert not x1["b"]["information"] and not ch["M1"]["information"] and x1["e0"]["b"]
    gp = ms.gap(v, base)["M1"]
    assert gp["cells"] == 4 and gp["verdict"] == "줄었다" and gp["ge_spy_before"] == 0 and gp["ge_spy_after"] == 0
    assert gp["ge_ew_before"] == 0 and gp["ge_ew_after"] == 4
    sz = ms.size_diagnostic(v)["M1"]  # bigger companies and higher CAGR together: rank correlation -1
    assert abs(sz["rho_vs_none"] + 1) < 1e-12 and sz["not_separated"] and sz["median_shift"] < 0
    v["market_pass"] = ms.market_pass(v, base)
    assert not v["market_pass"].any()  # below SPY's CAGR


def test_repro(tmp_path):
    acc, un, _ = _synthetic()
    periods = list(ms.es.PERIODS)
    un_src = un[un["kind"] == "none"].assign(src=lambda d: d["gen"].where(d["gen"] != "E0", "R0s1"))
    acc = acc.assign(src=acc["gen"].where(acc["gen"] != "E0", "R0s1"))
    sets = {"E1": "single__E1", "E2": "single__E2", "E0": "e0full__1"}
    lv = pd.DataFrame([{"set": sets[g], "exit": x, "trades": 60.0, **{f"cagr_{p}": 0.07 for p in periods}, **{f"mdd_{p}": -0.3 for p in periods}}
                       for g in sets for x in ("X1", "H1")])
    lu = pd.DataFrame([{"set": sets[g], "exit": x, "signals": 140, **{f"mean_contrib_{p}": 0.004 for p in periods}} for g in sets for x in ("X1", "H1")])
    lv.to_csv(tmp_path / "verdicts.csv.gz", index=False)
    lu.to_csv(tmp_path / "unlimited.csv.gz", index=False)
    rp = ms.repro(acc, pd.concat([un, un_src], ignore_index=True), tmp_path)
    assert len(rp) == 6 * (1 + 2 * len(periods) + len(periods) + 1) and rp["diff"].max() < 1e-12
    lv.loc[0, "cagr_1993-2023"] = 0.08
    lv.to_csv(tmp_path / "verdicts.csv.gz", index=False)
    rp = ms.repro(acc, pd.concat([un, un_src], ignore_index=True), tmp_path)
    assert (rp["diff"] > 1e-9).sum() == 1
