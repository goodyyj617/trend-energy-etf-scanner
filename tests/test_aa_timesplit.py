"""A43: time-split test of the dynamic-setting selection rule (lab/aa_timesplit.py). Synthetic data, no network."""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from lab import aa_canary as K
from lab import aa_layers as LY
from lab import aa_timesplit as TS
from lab.data import Panel

ROOT = Path(__file__).resolve().parents[1]
CODES = ["379810", "379800", "453630", "453640", "KRXGOLD", "459580"]
BASE = {"379810": 30.0, "379800": 20.0, "KRXGOLD": 30.0, "459580": 20.0}


def _panel(n=6300, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-09-01", periods=n)
    r = rng.normal(0.0003, 0.01, size=(n, len(CODES)))
    r[:, -1] = 0.0001
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=CODES)
    cash = pd.Series(0.02 / 252, index=dates)
    p = Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6, tradable=CODES, cash=cash)
    p.benchmark = "379800"
    return p


def test_windows_are_fixed_and_cover_the_research_window():
    assert list(TS.WINDOWS) == ["전체", "전반", "후반", "1구간", "2구간", "3구간", "1+2구간", "2+3구간"]
    assert TS.WINDOWS["전체"] == ("2000-09-01", "2024-12-31")
    assert TS.WINDOWS["전반"][1] < TS.WINDOWS["후반"][0]
    assert [d[1:] for d in TS.DIRECTIONS] == [("전반", "후반"), ("후반", "전반"), ("1+2구간", "3구간"), ("2+3구간", "1구간")]


def test_window_metrics_match_a_fresh_rebase_and_the_whole_window_matches_stage4_formulas():
    p = _panel()
    cols = sorted(set(BASE) | set(K.DEFENSIVE) | {LY.CD})
    sim = K.DynamicSim(p, cols, "2000-09-01", "2024-12-31")
    tab = LY.action_tables(BASE, "none", "none", sim.columns)
    P = LY.paths(sim, tab, np.zeros((1, sim.n_days), dtype=int), LY.Setting(), K.quarter_rows(sim.dates), None)
    rows = TS.window_rows(sim.dates, "전체")
    cg, md = TS.window_metrics(P, rows)
    assert np.isclose(cg[0], LY.cagr(P)[0]) and np.isclose(md[0], LY.mdd(P)[0])
    rows = TS.window_rows(sim.dates, "후반")
    assert sim.dates[rows[0]] >= pd.Timestamp("2013-01-01") and sim.dates[rows[-1]] <= pd.Timestamp("2024-12-31")
    cg, md = TS.window_metrics(P, rows)
    sub = P[0, rows] / P[0, rows[0]]
    assert np.isclose(cg[0], sub[-1] ** (252 / (len(sub) - 1)) - 1)
    assert np.isclose(md[0], (sub / np.maximum.accumulate(sub) - 1).min())
    rv = TS.rank_value(P, sim.dates, rows)
    assert rv.shape == (1,) and np.isfinite(rv[0])
    # a window with no rows gives NaN, not an error
    empty = np.array([], dtype=int)
    assert np.isnan(TS.window_metrics(P, empty)[0][0]) and np.isnan(TS.rank_value(P, sim.dates, empty)[0])


def test_equal_cost_interpolates():
    gc = np.array([0.08, 0.07, 0.06, 0.05])
    gm = np.array([-0.20, -0.16, -0.12, -0.08])
    assert TS.equal_cost(gc, gm, 0.09) == (0.0, -0.20)
    x, m = TS.equal_cost(gc, gm, 0.065)
    assert np.isclose(x, 0.5) and np.isclose(m, -0.14)
    assert np.isnan(TS.equal_cost(gc, gm, float("nan"))[0])


def test_hypergeom_matches_brute_force():
    N, Kp, n = 12, 5, 4
    rng = np.random.default_rng(0)
    draws = np.array([(rng.permutation(N)[:n] < Kp).sum() for _ in range(40000)])
    for k in range(0, 5):
        assert abs(TS.hypergeom_sf(k, N, Kp, n) - (draws >= k).mean()) < 0.02
    assert TS.hypergeom_sf(0, N, Kp, n) == 1.0 and TS.hypergeom_sf(5, N, Kp, n) == 0.0


def _toy_frames():
    """Two targets, 4 settings, 1 satellite share, 3 sets, windows 전반 · 후반 · 1+2구간 · 3구간 · 2+3구간 · 1구간 · 전체.
    Setting s1 passes everywhere, s2 passes only in 전반, s3 never, 'none' has no placebo."""
    wins = list(TS.WINDOWS)
    settings = [LY.Setting("k2", "safe", 0, 1, 0.08, 21), LY.Setting("k4", "riskcd", 0, 1, 0.0, 0),
                LY.Setting("k4", "safe", 10, 2, 0.0, 0), LY.Setting("none", "none", 0, 1, 0.08, 0)]
    rows, pl = [], []
    for sname in ("대표", "극단 1", "극단 2"):
        for t in ("F4 (가)", "F3 현재 보유"):
            for wn in wins:
                for sat in (0.0, 0.5):  # the verdict compares with the plain static (satellite 0), as stage 4 did
                    rows.append({"세트": sname, "대상": t, "종류": "정적", "위성": sat, "구간": wn, "cagr": 0.08, "mdd": -0.14, "순위값": 0.07})
                for i, st in enumerate(settings):
                    good = i == 0 or (i == 1 and wn in ("전반", "1+2구간", "1구간"))
                    mdd = -0.08 if good else -0.135
                    rows.append({"세트": sname, "대상": t, "종류": "설정", "위성": 0.5, "구간": wn, **st.__dict__, "설정 이름": st.label(),
                                 "cagr": 0.079 - 0.001 * i, "mdd": mdd, "같은 비용 x": 0.3, "같은 비용 mdd": -0.12,
                                 "순위값": 0.075 - 0.002 * i})
                    if st.signal != "none" and sname == "대표":
                        pl.append({"대상": t, **st.__dict__, "위성": 0.5, "구간": wn, "가짜 mdd q95": -0.10, "가짜 mdd 중앙": -0.12, "가짜 수": 8})
    return pd.DataFrame(rows), pd.DataFrame(pl)


def test_verdicts_transfer_best_pick_and_judgements_on_a_toy():
    R, PL = _toy_frames()
    V = TS.verdicts(R, PL)
    assert set(V["세트"]) == {"대표"} and len(V) == 2 * 4 * len(TS.WINDOWS)
    v = V.set_index(["대상", "signal", "action", "구간"])
    assert v.loc[("F4 (가)", "k2", "safe", "후반"), "판정"] == "통과"
    assert v.loc[("F4 (가)", "k4", "riskcd", "전반"), "판정"] == "통과"
    assert v.loc[("F4 (가)", "k4", "riskcd", "후반"), "판정"] == "탈락"
    assert v.loc[("F4 (가)", "none", "none", "전반"), "판정"] == "탈락"  # no placebo → ① is False
    tr = TS.transfer(V, "전반", "후반")
    whole = tr[tr["대상"] == "전체"].iloc[0]
    assert whole["설정 수"] == 8 and whole["고르는 구간 통과 n"] == 4 and whole["시험 구간 유지 k"] == 2
    assert np.isclose(whole["유지율"], 0.5) and np.isclose(whole["바탕 통과율"], 0.25) and np.isclose(whole["비"], 2.0)
    assert 0 < whole["p값 (초기하)"] <= 1
    bp = TS.best_pick(V, "전반", "후반")
    ga = bp[bp["대상"] == "F4 (가)"].iloc[0]
    assert ga["signal"] == "k2" and ga["시험 엄격 통과"] and ga["시험 완화 ①③"]
    r3 = TS.rep3_rows(V, R)
    assert (r3["이름"].str.startswith("③")).sum() == len(TS.WINDOWS)
    assert set(r3["이름"]) >= {"(가) 정적", "(가) 50 + 위성 50 (신호 없음)", "(가) 50 + 위성 50 + 변동성 목표 8%"}
    TR = pd.concat([TS.transfer(V, s, t) for _, s, t in TS.DIRECTIONS], ignore_index=True)
    BP = pd.concat([TS.best_pick(V, s, t) for _, s, t in TS.DIRECTIONS], ignore_index=True)
    J = TS.judgements(TR, BP, r3)
    assert J["J2 최선 하나 (가)"] == "버팀" and J["J3 ③ 자체"] == "양 구간 통과"
    assert J["J1 선택 규칙 A"] in ("있음", "약함", "없음") and len(J["J1 방향별"]) == 4


def test_best_pick_without_passers_says_so():
    R, PL = _toy_frames()
    PL["가짜 mdd q95"] = -0.01  # nothing beats the placebo
    V = TS.verdicts(R, PL)
    bp = TS.best_pick(V, "전반", "후반")
    assert (bp["선택"] == "없음").all()
    J = TS.judgements(TS.transfer(V, "전반", "후반"), bp, TS.rep3_rows(V, R))
    assert J["J2 최선 하나 (가)"] == "선택 없음" and J["J3 ③ 자체"].startswith("탈락")


def test_design_memo_and_decision_exist():
    memo = ROOT / "docs" / "research" / "allocation" / "2026-11-06_a43_design.md"
    assert memo.exists()
    text = memo.read_text(encoding="utf-8")
    for w in ("전반 2000-09-01 – 2012-12-31", "J1", "J2", "J3", "초기하"):
        assert w in text
    dec = (ROOT / "docs" / "research" / "allocation" / "DECISIONS.md").read_text(encoding="utf-8")
    assert "A43." in dec
    assert (ROOT / "scripts" / "aa_a43_timesplit.py").exists()


def test_script_jobs_run_on_a_synthetic_context(tmp_path):
    """job_real / job_placebo of scripts/aa_a43_timesplit.py on a fake context, then the verdict and tables."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("aa_a43_timesplit", ROOT / "scripts" / "aa_a43_timesplit.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    p = _panel()
    cols = sorted(set(BASE) | set(K.DEFENSIVE) | {LY.CD})
    rng = np.random.default_rng(3)
    ctx = {"targets": {"F4 (가)": BASE}, "sets": {}}
    for sname in M.SETS:
        sim = K.DynamicSim(p, cols, "2000-09-01", "2024-12-31")
        base_tab = LY.action_tables(BASE, "none", "none", sim.columns)
        base_p, _, _ = sim.run(*K.static_rows(base_tab, sim))
        base_r = np.concatenate([[0.0], base_p[0, 1:] / base_p[0, :-1] - 1.0])
        st = (rng.random(sim.n_days) < 0.2).astype(int)
        sat = np.cumprod(1 + rng.normal(0.0003, 0.008, sim.n_days))
        sat[:500] = np.nan  # the satellite starts later, as HAA50+BAA50 does
        ctx["sets"][(sname, "F4 (가)")] = {"sim": sim, "qrows": K.quarter_rows(sim.dates), "states": {"k2": st, "k4": st[::-1].copy()},
                                          "base_path": base_p[0], "base_ret": base_r, "sat": sat, "start": "2000-09-01"}
    M._init(ctx)
    settings = [LY.Setting("k2", "safe", 0, 1, 0.08, 21), LY.Setting("k4", "riskcd", 10, 2, 0.0, 0), LY.Setting("none", "none", 0, 1, 0.08, 0)]
    R = pd.concat([M.job_real((s, "F4 (가)", settings)) for s in M.SETS], ignore_index=True)
    PL = pd.concat([M.job_placebo(("F4 (가)", sg, [s for s in settings if s.signal == sg], 6)) for sg in ("k2", "k4")], ignore_index=True)
    assert set(R["구간"]) == set(TS.WINDOWS) and set(R["위성"]) == set(M.SAT)
    assert R[(R["종류"] == "정적") & (R["세트"] == "대표")]["순위값"].notna().all()
    assert R[R["세트"] != "대표"]["순위값"].isna().all()
    # the whole window equals the stage-4 formulas on the same path
    whole = R[(R["세트"] == "대표") & (R["구간"] == "전체") & (R["종류"] == "정적") & (R["위성"] == 0.0)].iloc[0]
    bp = ctx["sets"][("대표", "F4 (가)")]["base_path"][None, :]
    assert np.isclose(whole["cagr"], LY.cagr(bp)[0]) and np.isclose(whole["mdd"], LY.mdd(bp)[0])
    summ = M.analyse(tmp_path, R, PL, False, {"quick": True})
    assert set(summ["judgements"]) >= {"J1 선택 규칙 A", "J2 최선 하나 (가)", "J3 ③ 자체"}
    for f in ("verdicts_by_window.csv.gz", "transfer.csv", "best_pick.csv", "rep3_windows.csv", "summary.json"):
        assert (tmp_path / f).exists()
    V = pd.read_csv(tmp_path / "verdicts_by_window.csv.gz")
    assert len(V) == len(settings) * len(M.SAT) * len(TS.WINDOWS)
