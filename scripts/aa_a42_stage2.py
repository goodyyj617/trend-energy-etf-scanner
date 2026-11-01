"""A42 stage 2: wide static search v2, second grid, walk-forward of the picking rules, PBO (allocation DECISIONS A42 §2).

    .venv/Scripts/python.exe scripts/aa_a42_stage2.py --quick --out %TEMP%\\a42s2_quick            # pipeline check
    .venv/Scripts/python.exe scripts/aa_a42_stage2.py --stage map --out docs/research/allocation/2026-11-01_a42_stage2
    ... --stage grid | walk | pbo | report   (or --stage all)

Every stage saves its work in chunks under <out>/_work and skips chunks that already exist, so a stopped run
continues where it stopped. Use --workers to set the processes; scripts/cpu_daytime.ps1 -Match aa_a42_stage2 lowers
the priority and the cores in the daytime (10–21 h).

  map     T1 / T2 / T3 / TL, 2^20 each (seed 2042), representative price set (real · h60 · CD91, quarterly, 10 bp):
          A41 metrics, CVaR / Ulcer / under water / recovery / real CAGR, exposures (stage-1 betas), yearly windows
  grid    50,000 inside + 10,000 border + 20,000 outside points × 24 price sets × 48 rebalancing settings
          (monthly / quarterly / half-yearly / yearly × band 0 / 3 / 5 / 10 %p × decision offset 0 / 5 / 10 days)
  walk    R1–R8 yearly re-pick (expanding and 10-year windows) 2005–2023, hold the next year; F1–F5 fixed;
          selection on the representative set, the held weights evaluated on all 24 sets
  pbo     CSCV (10 blocks) on 20,000 random points, rules R1 and R2
  report  tables, figures, summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import aa_landscape as LS  # noqa: E402
from lab import aa_pricesets as PS  # noqa: E402
from lab import aa_search as srch  # noqa: E402
from lab import aa_wide as A  # noqa: E402
from lab.aa_assets import PRESETS  # noqa: E402
from lab.aa_composition import whole  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation  # noqa: E402
from lab.aa_score import CRISIS_WINDOWS  # noqa: E402

REP = PS.PriceSet("real", 60.0, "cd91")
START, END = PS.START, "2024-12-31"
WINDOWS = {"dotcom": LS.DOTCOM, "2008": CRISIS_WINDOWS["금융위기 2007–09"], "2020": CRISIS_WINDOWS["코로나 2020"],
           "2022": CRISIS_WINDOWS["금리 급등 2022"]}
FULL = dict(n=2 ** 20, chunk=2048, grid=(50_000, 10_000, 20_000), pbo=20_000)
QUICK = dict(n=2 ** 11, chunk=512, grid=(300, 100, 100), pbo=400)
SAMPLES = {"T1": (A.sample_t1, A.U17), "T2": (A.sample_t2, A.U17), "T3": (A.sample_t3, A.U17), "TL": (A.sample_tl, A.U19)}
REBAL = [(f, b, o) for f in ("M", "Q", "S", "A") for b in (0.0, 3.0, 5.0, 10.0) for o in (0, 5, 10)]
FIXED = {"F1 60/40": {"379800": 60, "0091C0": 40}, "F2 안정형 40": "stable40", "F3 현재 보유": "aaa_2026",
         "F4 (가)": "a41_ga", "F5 S&P500": "sp500"}
BETAS = ROOT / "docs" / "research" / "allocation" / "2026-10-31_a42_stage1" / "betas.csv"


# --------------------------------------------------------------------------- panels and simulators

_PANELS: dict = {}


def panel(ps: PS.PriceSet):
    if ps.label not in _PANELS:
        _PANELS[ps.label] = PS.build(ps.prices, ps.hedge_bp, ps.cash, quirks=True)[0]
    return _PANELS[ps.label]


def first_valid(p, universe) -> pd.Series:
    fv = p.close[universe].apply(lambda s: s.first_valid_index())
    return fv.clip(lower=pd.Timestamp(START))


def start_groups(W: np.ndarray, universe, fv: pd.Series) -> dict[pd.Timestamp, np.ndarray]:
    st = pd.to_datetime(A.point_start(W, universe, fv))
    return {pd.Timestamp(d): np.flatnonzero(st == d) for d in sorted(set(st))}


def live_cols(p, universe, start) -> list[int]:
    """Positions in `universe` of the products with data on `start` (a start group's points hold only these)."""
    fv = first_valid(p, universe)
    return [i for i, c in enumerate(universe) if fv[c] <= pd.Timestamp(start)]


def sim_for(p, universe, start, freq="Q", band=0.0, offset=0) -> srch.StaticSim:
    universe = [universe[i] for i in live_cols(p, universe, start)]
    return srch.StaticSim(p, universe, str(pd.Timestamp(start).date()), END, rebalance=freq, cost_bps=10.0,
                          rebalance_offset=offset, band=band)


def frac(W):
    F = np.zeros((W.shape[0], W.shape[1] + 1))
    F[:, :-1] = W / 100.0
    return F


# --------------------------------------------------------------------------- map

_CTX: dict = {}


def _init(ctx):
    _CTX.clear()
    _CTX.update(ctx)


def _map_chunk(args):
    key, W = args
    sim, lay, cols = _CTX["sims"][key]
    P, turn = sim.run(frac(W[:, cols]))
    m = LS.path_metrics(P, turn, lay)
    for k in WINDOWS:
        if f"dd_{k}" not in m:
            m[f"dd_{k}"] = np.nan
    x = A.extra_metrics(P, sim.dates, _CTX["cpi"])
    yw = A.yearly_windows(P, sim.dates)
    return pd.concat([m, x], axis=1), yw


def stage_map(P_, out: Path, work: Path, workers: int) -> dict:
    p = panel(REP)
    cpi = A.load_cpi()
    beta = pd.read_csv(BETAS, dtype={"상품": str}).set_index("상품")
    res = {}
    for name, (fn, universe) in SAMPLES.items():
        d = work / "map" / name
        d.mkdir(parents=True, exist_ok=True)
        wpath = d / "W.npy"
        if wpath.exists():
            W = np.load(wpath)
        else:
            W = fn(P_["n"], A.SEED + list(SAMPLES).index(name))
            np.save(wpath, W)
        fv = first_valid(p, universe)
        groups = start_groups(W, universe, fv)
        sims = {}
        for s0 in groups:
            sim = sim_for(p, universe, s0)
            # a crisis window counts only if the path starts within a month of it (else NaN for that column)
            wins = {k: v for k, v in WINDOWS.items() if sim.dates[0] <= pd.Timestamp(v[0]) + pd.Timedelta(days=31)}
            sims[str(s0.date())] = (sim, LS.layout(sim.dates, wins), live_cols(p, universe, s0))
        jobs = []
        for s0, idx in groups.items():
            for a in range(0, len(idx), P_["chunk"]):
                tag = f"{s0.date()}_{a:07d}"
                if not (d / f"m_{tag}.pkl").exists():
                    jobs.append((tag, str(s0.date()), idx[a:a + P_["chunk"]]))
        t0 = time.time()
        print(f"  {name}: 점 {len(W):,}, 시작일 묶음 {len(groups)}, 남은 조각 {len(jobs)}", flush=True)
        if jobs:
            with ProcessPoolExecutor(workers, initializer=_init, initargs=({"sims": sims, "cpi": cpi},)) as ex:
                futs = {ex.submit(_map_chunk, (key, W[idx])): (tag, idx) for tag, key, idx in jobs}
                for k, fut in enumerate(futs):
                    tag, idx = futs[fut]
                    m, yw = fut.result()
                    m.insert(0, "row", idx)
                    m.to_pickle(d / f"m_{tag}.pkl")  # own work file, read back only by this script
                    np.save(d / f"y_{tag}.npy", np.concatenate([idx[:, None, None].repeat(len(A.CUT_YEARS), 1).astype(np.float32), yw], axis=2))
                    if k % 25 == 0:
                        print(f"    {name} {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        # the m_*.pkl files are this script's own work files (written above), never outside input
        M = pd.concat([pd.read_pickle(f) for f in sorted(d.glob("m_*.pkl"))]).sort_values("row").reset_index(drop=True)
        st = pd.to_datetime(A.point_start(W, universe, fv))
        M["start"] = st[M["row"].to_numpy()]
        M["zone"] = LS.zone(M["mdd"])
        cols = [c for c in universe]
        Bm = beta.reindex(cols)[["EQ", "UST", "FX", "GOLD"]].to_numpy(float)
        M[["expo_eq", "expo_ust", "expo_fx", "expo_gold"]] = (W[M["row"].to_numpy()] / 100.0) @ Bm
        M.to_pickle(d / "metrics.pkl")
        res[name] = {"n": int(len(M)), "zones": M["zone"].value_counts().to_dict(),
                     "starts": {str(k.date()): int(len(v)) for k, v in groups.items()}}
    return res


def load_map(work: Path, name: str):
    d = work / "map" / name
    W = np.load(d / "W.npy")
    M = pd.read_pickle(d / "metrics.pkl")
    Y = np.concatenate([np.load(f) for f in sorted(d.glob("y_*.npy"))])
    order = np.argsort(Y[:, 0, 0])
    return W, M, Y[order, :, 1:]


# --------------------------------------------------------------------------- grid

def _grid_chunk(args):
    key, W = args
    sims = _CTX.setdefault("sims", {})
    if key not in sims:  # built once per worker from the panel sent at start-up (sending ~700 simulators would not fit)
        name, s0, f, b, o = key.split("|")
        sims.clear() if len(sims) > 64 else None
        universe = SAMPLES[name][1]
        sims[key] = (sim_for(_CTX["panel"], universe, s0, f, float(b), int(o)), live_cols(_CTX["panel"], universe, s0))
    sim, cols = sims[key]
    P, _ = sim.run(frac(W[:, cols]))
    v = P[:, -1] / P[:, 0]
    cagr = v ** (252 / max(P.shape[1] - 1, 1)) - 1
    mdd = (P / np.maximum.accumulate(P, axis=1) - 1).min(axis=1)
    return np.c_[cagr, mdd].astype(np.float32)


def grid_points(work: Path, P_, seed: int = A.SEED) -> pd.DataFrame:
    path = work / "grid" / "points.pkl"
    if path.exists():
        return pd.read_pickle(path)
    rng = np.random.default_rng(seed)
    parts = []
    for name in SAMPLES:
        _, M, _ = load_map(work, name)
        parts.append(M[["row", "zone", "mdd", "cagr", "start"]].assign(sample=name))
    allm = pd.concat(parts, ignore_index=True)
    pick = []
    for zone, n in zip(("안", "경계", "밖"), P_["grid"]):
        pool = np.flatnonzero(allm["zone"].to_numpy() == zone)
        pick.append(rng.choice(pool, size=min(n, len(pool)), replace=False))
    G = allm.iloc[np.sort(np.concatenate(pick))].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    G.to_pickle(path)
    return G


def stage_grid(P_, out: Path, work: Path, workers: int) -> dict:
    G = grid_points(work, P_)
    Ws = {name: np.load(work / "map" / name / "W.npy") for name in SAMPLES}
    d = work / "grid"
    t0 = time.time()
    for si, ps in enumerate(PS.all_sets()):
        todo = [(f, b, o) for f, b, o in REBAL if not (d / f"r_{ps.label}_{f}{b:g}_{o}.npy").exists()]
        if not todo:
            continue
        p = panel(ps)
        jobs = []
        for f, b, o in todo:
            for name in SAMPLES:
                g = G[G["sample"] == name]
                for s0, gg in g.groupby("start"):
                    key = f"{name}|{pd.Timestamp(s0).date()}|{f}|{b:g}|{o}"
                    rows = gg.index.to_numpy()
                    for a in range(0, len(rows), P_["chunk"]):
                        jobs.append((key, (f, b, o), rows[a:a + P_["chunk"]], Ws[name][gg["row"].to_numpy()[a:a + P_["chunk"]]]))
        res = {setting: np.full((len(G), 2), np.nan, dtype=np.float32) for setting in todo}
        with ProcessPoolExecutor(workers, initializer=_init, initargs=({"panel": p},)) as ex:
            futs = [(ex.submit(_grid_chunk, (key, W)), setting, rows) for key, setting, rows, W in jobs]
            for fut, setting, rows in futs:
                res[setting][rows] = fut.result()
        for (f, b, o), arr in res.items():
            np.save(d / f"r_{ps.label}_{f}{b:g}_{o}.npy", arr)
        print(f"  격자 {si + 1}/24 {ps.label} ({time.time() - t0:.0f}s)", flush=True)
        _PANELS.pop(ps.label, None)
    return {"points": int(len(G)), "settings": len(REBAL) * 24}


def load_grid(work: Path) -> tuple[pd.DataFrame, dict]:
    G = pd.read_pickle(work / "grid" / "points.pkl")
    R = {}
    for ps in PS.all_sets():
        for f, b, o in REBAL:
            R[(ps.label, f, b, o)] = np.load(work / "grid" / f"r_{ps.label}_{f}{b:g}_{o}.npy")
    return G, R


# --------------------------------------------------------------------------- walk-forward

def _cut_dates(dates: pd.DatetimeIndex) -> dict[int, pd.Timestamp]:
    return {y: dates[r] for y, r in A.cut_rows(dates).items()}


def walk_picks(work: Path) -> pd.DataFrame:
    """Per rule, window and selection year: the weights held over the next year (representative set)."""
    p = panel(REP)
    rets = p.close[A.U17].ffill().pct_change(fill_method=None)
    cuts = _cut_dates(p.close.index[(p.close.index >= START) & (p.close.index <= END)])
    pool = []
    for name in ("T1", "T2", "T3"):  # R1–R3 pick from the 17-product samples (the leverage arm is not a candidate)
        W, M, Y = load_map(work, name)
        pool.append((W, Y))
    W = np.vstack([w for w, _ in pool])
    Y = np.concatenate([y for _, y in pool])
    rows = []
    for j, y in enumerate(A.CUT_YEARS):
        if y not in cuts:
            continue
        for window, (ci, mi) in (("확장", (0, 1)), ("10년 이동", (2, 3))):
            tc, tm = Y[:, j, ci].astype(float), Y[:, j, mi].astype(float)
            if ci == 2:  # a path with less than ten years of history uses all it has (the test window stays 2006–2024)
                short = ~np.isfinite(tc)
                tc[short], tm[short] = Y[short, j, 0], Y[short, j, 1]
            for rule, fn in (("R1", A.pick_r1), ("R3", A.pick_r3)):
                k = fn(tc, tm)
                if k is not None:
                    rows.append({"규칙": rule, "창": window, "연도": y, "비중": W[k]})
            w2 = A.pick_r2(W, tc, tm)
            if w2 is not None:
                rows.append({"규칙": "R2", "창": window, "연도": y, "비중": w2})
        row = int(rets.index.get_loc(cuts[y]))
        for rule in ("R4", "R5", "R6", "R7", "R8"):  # rule-based weights read only the last 3 years: one window
            wd = A.rule_weights(rule, rets, A.U17, row)
            vec = np.array([wd.get(c, 0.0) for c in A.U17]) * 100
            rows.append({"규칙": rule, "창": "직전 3년" if rule != "R4" else "훈련 없음", "연도": y, "비중": whole(vec)})
    return pd.DataFrame(rows)


def walk_curve(p, picks: pd.DataFrame, rule: str, window: str) -> pd.Series:
    """Chain the yearly holdings (quarterly rebalancing inside each year) into one equity curve 2006 – 2024."""
    sub = picks[(picks["규칙"] == rule) & (picks["창"] == window)].sort_values("연도")
    dates = p.close.index[(p.close.index >= START) & (p.close.index <= END)]
    cuts = _cut_dates(dates)
    years = sub["연도"].tolist()
    eq = []
    level = 1.0
    for k, (_, r) in enumerate(sub.iterrows()):
        a = cuts[r["연도"]]
        b = cuts[years[k + 1]] if k + 1 < len(years) else dates[-1]
        w = {A.U17[i]: float(x) for i, x in enumerate(r["비중"]) if x > 0}
        res = run_allocation(p, AllocConfig(rule, w, str(a.date()), str(b.date()), rebalance="Q", cost_bps=10.0))
        e = res.equity / res.equity.iloc[0] * level
        if eq:
            e = e.iloc[1:]
        eq.append(e)
        level = float(e.iloc[-1])
    return pd.concat(eq)


def curve_stats(e: pd.Series) -> dict:
    v = e.to_numpy()
    peak = np.maximum.accumulate(v)
    dd = v / peak - 1
    under = dd < -1e-12
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return {"cagr": float((v[-1] / v[0]) ** (252 / max(len(v) - 1, 1)) - 1), "mdd": float(dd.min()), "최장 회복 (월)": longest / 21.0}


def stage_walk(P_, out: Path, work: Path, workers: int) -> dict:
    pk_path = work / "walk" / "picks.pkl"
    pk_path.parent.mkdir(parents=True, exist_ok=True)
    picks = pd.read_pickle(pk_path) if pk_path.exists() else walk_picks(work)
    picks.to_pickle(pk_path)
    rows = []
    for ps in PS.all_sets():
        f = work / "walk" / f"c_{ps.label}.pkl"
        if f.exists():
            rows += pd.read_pickle(f).to_dict("records")
            continue
        p = panel(ps)
        part = []
        for (rule, window), _ in picks.groupby(["규칙", "창"]):
            e = walk_curve(p, picks, rule, window)
            part.append({"세트": ps.label, "대표": ps == REP, "규칙": rule, "창": window, "시간 밖": True, **curve_stats(e)})
        first = pd.Timestamp(f"{A.CUT_YEARS[0]}-12-01")
        for name, w in FIXED.items():
            wt = w if isinstance(w, dict) else PRESETS[w].weights
            a = p.close.index[(p.close.index >= first) & (p.close.index.month == 12) & (p.close.index.year == A.CUT_YEARS[0])][-1]
            res = run_allocation(p, AllocConfig(name, dict(wt), str(a.date()), END, rebalance="Q", cost_bps=10.0))
            part.append({"세트": ps.label, "대표": ps == REP, "규칙": name, "창": "고정", "시간 밖": name.startswith("F5"),
                         **curve_stats(res.equity)})
        pd.DataFrame(part).to_pickle(f)
        rows += part
        _PANELS.pop(ps.label, None)
        print(f"  walk {ps.label}", flush=True)
    WF = pd.DataFrame(rows)
    WF.to_csv(out / "walkforward.csv", index=False)
    return {"rules": int(WF[["규칙", "창"]].drop_duplicates().shape[0])}


# --------------------------------------------------------------------------- PBO

def stage_pbo(P_, out: Path, work: Path, workers: int) -> dict:
    rng = np.random.default_rng(A.SEED)
    p = panel(REP)
    W17 = []
    for name in ("T1", "T2", "T3"):
        W, M, _ = load_map(work, name)
        W17.append((W, M))
    allW = np.vstack([w for w, _ in W17])
    allM = pd.concat([m for _, m in W17], ignore_index=True)
    ok = np.flatnonzero(pd.to_datetime(allM["start"]).to_numpy() == np.datetime64(pd.Timestamp(START)))
    pick = rng.choice(ok, size=min(P_["pbo"], len(ok)), replace=False)
    sim = sim_for(p, A.U17, START)
    paths, _ = sim.run(frac(allW[pick][:, live_cols(p, A.U17, START)]))  # allM row i is allW row i (each sample's metrics are sorted by row)
    R = srch.monthly_returns(paths, sim.dates)
    r1 = {k: v for k, v in srch.pbo_selection(R, mdd_line=-0.15, blocks=10).items() if not isinstance(v, (list, np.ndarray))}
    sys.path.insert(0, str(ROOT / "scripts"))
    import aa_redesign_study as A41
    r2 = A41._pbo_r2(R, allW[pick])
    return {"R1": r1, "R2": r2, "points": int(len(pick))}


# --------------------------------------------------------------------------- flat-region centre on the whole window

NB_STEP = 5
POLICY = [(ph, off) for ph in range(3) for off in (0, 5, 10, 15, 20)]  # declared policy: quarterly, no band (design §3)
FIG_DIR = ROOT / "docs" / "research" / "allocation" / "figures" / "a42_stage2"


def wstr(w, universe=A.U17) -> str:
    return ";".join(f"{universe[i]}:{int(x)}" for i, x in enumerate(w) if x > 0)


def wparse(s: str, universe=A.U17) -> np.ndarray:
    v = np.zeros(len(universe), np.int64)
    for part in str(s).split(";"):
        if part:
            c, x = part.split(":")
            v[universe.index(c)] = int(x)
    return v


def _run(p, w: np.ndarray, **kw):
    wt = {A.U17[i]: float(x) for i, x in enumerate(w) if x > 0}
    base = dict(start=START, end=END, rebalance="Q", cost_bps=10.0)
    base.update(kw)
    r = run_allocation(p, AllocConfig("c", wt, **base))
    return {**curve_stats(r.equity), "시작": str(r.equity.index[0].date())}


def verdicts(name: str, w: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """A41's four verdicts on the representative set; verdict 3 counts only the declared policy's variants."""
    p = panel(REP)
    rows = [{"시험": "기준", "변형": "분기·10bp", **_run(p, w)}]
    for v in A.neighbours(w, NB_STEP):
        rows.append({"시험": f"이웃 ±{NB_STEP}", "변형": wstr(v), **_run(p, v)})
    for ph, off in POLICY:
        rows.append({"시험": "선언 정책", "변형": f"Q{ph}·+{off}일·10bp", **_run(p, w, rebalance_phase=ph, rebalance_offset=off)})
    rows.append({"시험": "선언 정책", "변형": "Q0·+0일·20bp", **_run(p, w, cost_bps=20.0)})
    first = pd.Timestamp(rows[0]["시작"])
    for m in pd.date_range("2000-09-01", "2014-12-01", freq="MS"):
        if m < first - pd.Timedelta(days=31):
            continue
        end = (m + pd.DateOffset(years=10) - pd.Timedelta(days=1)).date()
        rows.append({"시험": "10년 창", "변형": str(m.date()), **_run(p, w, start=str(max(m, first).date()), end=str(end))})
    sets = []
    for ps in PS.all_sets():
        sets.append({"시험": "가격 세트", "변형": ps.label, **_run(panel(ps), w)})
    R = pd.DataFrame(rows + sets).assign(대상=name)

    def g(t):
        return R[R["시험"] == t]
    nb, pol, win, st = g(f"이웃 ±{NB_STEP}"), g("선언 정책"), g("10년 창"), g("가격 세트")
    v = {"대상": name, "비중": wstr(w), "cagr": float(g("기준")["cagr"].iloc[0]), "mdd": float(g("기준")["mdd"].iloc[0]),
         "1 기준선": bool(g("기준")["mdd"].iloc[0] >= LS.LINE),
         "2 이웃 안 비율": float((nb["mdd"] >= LS.LINE).mean()) if len(nb) else np.nan,
         "3 선언 정책 안 비율": float((pol["mdd"] >= LS.LINE).mean()),
         "4 10년 창 모두 안": bool((win["mdd"] >= LS.LINE).all()), "10년 창 최악 MDD": float(win["mdd"].min()),
         "24세트 최악 MDD": float(st["mdd"].min()), "24세트 CAGR 최소": float(st["cagr"].min()),
         "24세트 CAGR 최대": float(st["cagr"].max()), "점 조건 (24세트 ≥ −17%)": bool(st["mdd"].min() >= LS.NEAR)}
    v["2 통과"] = bool(v["2 이웃 안 비율"] >= LS.NB_SHARE)
    v["3 통과"] = bool(v["3 선언 정책 안 비율"] >= 0.90)
    v["모두 통과"] = bool(v["1 기준선"] and v["2 통과"] and v["3 통과"] and v["4 10년 창 모두 안"])
    return v, R


def stage_centre(P_, out: Path, work: Path, workers: int) -> dict:
    """R2 on the whole research window over T1–T3 (points that start by 2000-10, i.e. on the full window): points with
    MDD and worst 10-year window inside −15 %, rank value within 0.3 %p of the best, the largest major-product group's
    mean. The plain mean of the region is measured too (A41 4b style, reference)."""
    Ws, Ms = [], []
    for name in ("T1", "T2", "T3"):
        W, M, _ = load_map(work, name)
        Ws.append(W[M["row"].to_numpy()])
        Ms.append(M.assign(sample=name))
    W = np.vstack(Ws)
    M = pd.concat(Ms, ignore_index=True)
    full = (pd.to_datetime(M["start"]) <= pd.Timestamp("2000-10-31")).to_numpy()
    rank = np.where(full & (M["win_worst"].to_numpy() >= LS.LINE), M["rank"].to_numpy(), np.nan)
    ok = np.flatnonzero(np.isfinite(rank) & (M["mdd"].to_numpy() >= LS.LINE))
    best = np.nanmax(rank[ok]) if len(ok) else np.nan
    region = ok[rank[ok] >= best - LS.FLAT_PP] if len(ok) else ok
    targets = {}
    c = A.pick_r2(W, rank, M["mdd"].to_numpy())
    if c is not None:
        targets["R2 평탄 중심 (가장 큰 묶음)"] = c
    if len(region):
        mean = W[region].mean(axis=0)
        mean[mean < A.MIN_PCT] = 0.0
        m = whole(mean)
        if c is None or not np.array_equal(m, c):  # one group only: the mean is the centre, not measured twice
            targets["평탄 영역 평균 (참고)"] = m
    V, R = [], []
    for name, w in targets.items():
        v, r = verdicts(name, w)
        V.append(v)
        R.append(r)
    pd.DataFrame(V).to_csv(out / "centre_verdicts.csv", index=False)
    if R:
        pd.concat(R).to_csv(out / "centre_runs.csv.gz", index=False)
    reg = M.iloc[region][["sample", "row", "cagr", "mdd", "rank", "win_worst"]].copy()
    reg["비중"] = [wstr(x) for x in W[region]]
    reg.to_csv(out / "flat_region.csv", index=False)
    return {"region": int(len(region)), "best_rank": float(best) if np.isfinite(best) else None,
            "targets": {k: wstr(w) for k, w in targets.items()}, "verdicts": V}


# --------------------------------------------------------------------------- report

def pareto(df: pd.DataFrame) -> np.ndarray:
    """Non-dominated rows on (CAGR ↑, MDD ↑, longest recovery ↓)."""
    x = df[["cagr", "mdd"]].to_numpy()
    r = df["최장 회복 (월)"].to_numpy()
    keep = np.ones(len(df), bool)
    for i in range(len(df)):
        dom = (x[:, 0] >= x[i, 0]) & (x[:, 1] >= x[i, 1]) & (r <= r[i]) & ((x[:, 0] > x[i, 0]) | (x[:, 1] > x[i, 1]) | (r < r[i]))
        keep[i] = not dom.any()
    return keep


def stage_report(P_, out: Path, work: Path, workers: int) -> dict:
    res = {}
    # map: zones per sample, a 20,000-point sample of each for the figures
    zs, samples = [], []
    rng = np.random.default_rng(A.SEED)
    for name, (_, universe) in SAMPLES.items():
        W, M, _ = load_map(work, name)
        z = M["zone"].value_counts()
        zs.append({"표본": name, "점": len(M), **{k: int(z.get(k, 0)) for k in ("안", "경계", "밖")},
                   "전체 기간 점": int((pd.to_datetime(M["start"]) <= pd.Timestamp("2000-10-31")).sum()),
                   "안 CAGR 최대": float(M.loc[M["zone"] == "안", "cagr"].max())})
        k = rng.choice(len(M), size=min(20_000, len(M)), replace=False)
        s = M.iloc[k].copy()
        s["비중"] = [wstr(x, universe) for x in W[s["row"].to_numpy()]]
        samples.append(s.assign(표본=name))
    pd.DataFrame(zs).to_csv(out / "map_zones.csv", index=False)
    pd.concat(samples).to_csv(out / "map_sample.csv.gz", index=False)
    res["zones"] = zs
    # grid
    G, R = load_grid(work)
    labels = [ps.label for ps in PS.all_sets()]
    base = np.stack([R[(lb, "Q", 0.0, 0)][:, 1] for lb in labels], axis=1)  # MDD per set, quarterly / no band
    cag = np.stack([R[(lb, "Q", 0.0, 0)][:, 0] for lb in labels], axis=1)
    every = np.stack([R[k][:, 1] for k in R], axis=1)
    G = G.copy()
    G["24세트 최악 MDD"] = base.min(axis=1)
    G["24세트 CAGR 최소"], G["24세트 CAGR 최대"] = cag.min(axis=1), cag.max(axis=1)
    G["대표 CAGR"], G["대표 MDD"] = R[(REP.label, "Q", 0.0, 0)][:, 0], R[(REP.label, "Q", 0.0, 0)][:, 1]
    G["1,152 최악 MDD"] = every.min(axis=1)
    G["1,152 안 비율"] = (every >= LS.LINE).mean(axis=1)
    G["통과"] = G["24세트 최악 MDD"] >= LS.NEAR
    Ws = {name: np.load(work / "map" / name / "W.npy") for name in SAMPLES}
    G["비중"] = [wstr(Ws[s][r], SAMPLES[s][1]) for s, r in zip(G["sample"], G["row"])]
    G.to_csv(out / "grid_points.csv.gz", index=False)
    rows = []
    for (lb, f, b, o), arr in R.items():
        rows.append({"세트": lb, "주기": f, "밴드": b, "위상": o, "안 비율": float((arr[:, 1] >= LS.LINE).mean()),
                     "CAGR 중앙": float(np.nanmedian(arr[:, 0])), "MDD 중앙": float(np.nanmedian(arr[:, 1]))})
    GS = pd.DataFrame(rows)
    GS.to_csv(out / "grid_settings.csv", index=False)
    res["grid"] = {"points": int(len(G)), "pass": int(G["통과"].sum()),
                   "pass_by_zone": G.groupby("zone")["통과"].mean().round(4).to_dict(),
                   "pass_by_sample": G.groupby("sample")["통과"].mean().round(4).to_dict()}
    # walk-forward
    WF = pd.read_csv(out / "walkforward.csv")
    key = ["규칙", "창"]
    rep = WF[WF["대표"]].set_index(key)
    agg = WF.groupby(key).agg(**{"24세트 최악 MDD": ("mdd", "min"), "24세트 CAGR 최소": ("cagr", "min"),
                                  "24세트 CAGR 최대": ("cagr", "max"), "24세트 최장 회복 최대": ("최장 회복 (월)", "max")})
    T = rep[["cagr", "mdd", "최장 회복 (월)", "시간 밖"]].join(agg).reset_index()
    T["규칙 통과 (24세트 ≥ −17%)"] = T["24세트 최악 MDD"] >= LS.NEAR
    T["파레토 (대표)"] = pareto(T)
    r4 = T[T["규칙"] == "R4"]
    fail = []
    for _, r in T.iterrows():
        f = False
        if r["규칙"] in ("R1", "R2", "R3", "R5", "R6", "R7", "R8") and len(r4):
            q = r4.iloc[0]
            f = bool(q["cagr"] >= r["cagr"] and q["mdd"] >= r["mdd"] and q["최장 회복 (월)"] <= r["최장 회복 (월)"]
                     and (q["cagr"] > r["cagr"] or q["mdd"] > r["mdd"] or q["최장 회복 (월)"] < r["최장 회복 (월)"]))
        fail.append(f)
    T["고르는 데 실패 (R4에 열위)"] = fail
    T.to_csv(out / "walkforward_table.csv", index=False)
    pk = pd.read_pickle(work / "walk" / "picks.pkl")  # this script's own work file
    pk = pk.assign(비중=[wstr(x) for x in pk["비중"]])
    pk.to_csv(out / "walkforward_picks.csv", index=False)
    res["walk"] = T.to_dict("records")
    figures(out)
    return res


def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    dest.mkdir(parents=True, exist_ok=True)
    made = []
    S = pd.read_csv(out / "map_sample.csv.gz")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True)
    for ax, (name, g) in zip(axes, S.groupby("표본")):
        for st, c in (("2003-11", "#BBBBBB"), ("2000-10", "#F58518"), ("2000-09", "#4C78A8")):
            h = g[g["start"].astype(str).str[:7] == st]
            ax.scatter(h["mdd"] * 100, h["cagr"] * 100, s=2, alpha=0.25, color=c, rasterized=True,
                       label={"2000-09": "2000-09 시작", "2000-10": "2000-10 시작 (국고채 10·30년)",
                              "2003-11": "2003-11 시작 (미국 배당, 닷컴 없음)"}[st])
        ax.axvline(-15, color="gray", lw=0.8, ls="--")
        ax.axvline(-17, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{name} (무작위 2만 점)")
        ax.set_xlabel("MDD (%)")
    axes[0].set_ylabel("CAGR (%)")
    axes[0].set_xlim(-60, 0)
    axes[0].legend(markerscale=6, fontsize=7, loc="lower left")
    fig.suptitle("① 1차 표본: CAGR–MDD (대표 가격 세트, 분기, 10bp, 연구 구간)")
    made.append(str(figs._save(fig, dest / "1_map.png")))
    G = pd.read_csv(out / "grid_points.csv.gz")
    fig, ax = plt.subplots(figsize=(8, 6))
    for z, c in (("안", "#4C78A8"), ("경계", "#F58518"), ("밖", "#BBBBBB")):
        g = G[G["zone"] == z]
        ax.scatter(g["대표 MDD"] * 100, g["24세트 최악 MDD"] * 100, s=2, alpha=0.3, color=c, label=f"{z} ({len(g):,})", rasterized=True)
    ax.axhline(-17, color="black", lw=0.8, ls="--")
    ax.plot([-40, 0], [-40, 0], color="gray", lw=0.6)
    ax.set_xlim(-40, 0)
    ax.set_ylim(-45, 0)
    ax.set_xlabel("대표 세트 MDD (%)")
    ax.set_ylabel("24세트 중 최악 MDD (%)")
    ax.set_title(f"② 2차 격자: 대표 세트 vs 가격 세트 24개 최악 (점선 −17% = 점 조건)\n통과 {int(G['통과'].sum()):,} / {len(G):,}")
    ax.legend(markerscale=6, loc="lower right")
    made.append(str(figs._save(fig, dest / "2_grid_sets.png")))
    GS = pd.read_csv(out / "grid_settings.csv")
    piv = GS[GS["세트"] == REP.label].groupby(["주기", "밴드"])["안 비율"].mean().unstack().reindex(["M", "Q", "S", "A"]) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(piv.to_numpy(), cmap="Blues", aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iat[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(piv.shape[1]), [f"밴드 {b:g}%p" for b in piv.columns])
    ax.set_yticks(range(piv.shape[0]), ["월", "분기", "반기", "연"])
    ax.set_title("③ 리밸런싱 설정별 '안'(MDD ≥ −15%) 비율 (%)\n대표 세트, 격자 점 전체, 위상 3개 평균")
    fig.colorbar(im, ax=ax)
    made.append(str(figs._save(fig, dest / "3_rebalance.png")))
    T = pd.read_csv(out / "walkforward_table.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, r in T.iterrows():
        col = {"고정": "#E45756", "확장": "#4C78A8", "10년 이동": "#72B7B2"}.get(r["창"], "#9D755D")
        ax.plot([r["24세트 최악 MDD"] * 100, r["mdd"] * 100], [r["cagr"] * 100] * 2, color=col, lw=0.8, alpha=0.6)
        ax.scatter(r["mdd"] * 100, r["cagr"] * 100, color=col, s=28, zorder=3)
        ax.annotate(f"{r['규칙']}" + (f" {r['창'][:2]}" if r["창"] in ("확장", "10년 이동") else ""), (r["mdd"] * 100, r["cagr"] * 100), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(-17, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("walk-forward MDD (%) — 점 = 대표 세트, 선 = 24세트 최악까지")
    ax.set_ylabel("walk-forward CAGR (%), 대표 세트")
    ax.set_title("④ 고르는 규칙의 시간 밖 성적 (2006–2024, 매년 다시 고름)\n파랑 = 확장 창, 청록 = 10년 이동 창, 갈색 = 규칙형(R4–R8), 빨강 = 고정 배분")
    made.append(str(figs._save(fig, dest / "4_walkforward.png")))
    return made


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=("map", "grid", "walk", "pbo", "centre", "report", "figures", "all"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args(argv)
    P_ = QUICK if args.quick else FULL
    out = Path(args.out)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    sp = out / "summary.json"
    summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    summ["params"] = {"quick": args.quick, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in P_.items()},
                      "price_set": REP.label, "seed": A.SEED}
    if args.stage == "figures":
        print(figures(out))
        return
    stages = ["map", "grid", "walk", "pbo", "centre", "report"] if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        print(f"=== {s}", flush=True)
        res = {"map": stage_map, "grid": stage_grid, "walk": stage_walk, "pbo": stage_pbo, "centre": stage_centre,
               "report": stage_report}[s](P_, out, work, args.workers)
        res["seconds"] = round(time.time() - t0, 1)
        summ[s] = res
        sp.write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, default=str)[:1500], flush=True)
    print(f"끝 → {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
