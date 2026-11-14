"""A46: probabilistic MDD verdict (allocation DECISIONS A46, design docs/research/allocation/2026-11-11_a46_design.md).

    .venv/Scripts/python.exe scripts/aa_a46_probmdd.py --quick --out %TEMP%\\a46_quick
    .venv/Scripts/python.exe scripts/aa_a46_probmdd.py --out docs/research/allocation/2026-11-11_a46_probmdd
    ... --figures-only

Eight targets ((가), 안정형 40, R6 + vol target 10 %, R6 static, R6 at the same average exposure, 현재 보유, 60/40,
S&P500), each over its own window to 2024-12:
  M1  24 price sets × the 16 stage-5 date/cost variants (× 4 vol decision offsets for the vol target and its static)
  M2  representative set, stationary block bootstrap (mean block 126; 63 and 252 reported), 1,000 paths, pre-tax
  M3  the same 1,000 paths (block 126) under the declared account policy (A45), five targets, after tax
Reproduction checks: (b) bootstrap code with the historical order = DynamicSim, vectorised vol levels = aa_layers,
(c) M3 with the historical order = A45's declared row, (a) M1's base cells = stage 5's 24-set file. Any failure stops
the run. Pieces are saved under <out>/_work, so a stopped run continues.
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

from lab import aa_accounts as AC  # noqa: E402
from lab import aa_aftertax as AT  # noqa: E402
from lab import aa_canary as K  # noqa: E402
from lab import aa_layers as LY  # noqa: E402
from lab import aa_pricesets as PS  # noqa: E402
from lab import aa_probmdd as PM  # noqa: E402
from lab.aa_engine import decision_rows  # noqa: E402

ALLOC = ROOT / "docs" / "research" / "allocation"
STAGE3_TARGETS = ALLOC / "2026-11-01_a42_stage3" / "targets.csv"
STAGE5 = ALLOC / "2026-11-02_a42_stage5" / "runs.csv.gz"
A45_TABLE = ALLOC / "2026-11-11_a45_aftertax" / "table.csv"
FIG_DIR = ALLOC / "figures" / "a46_probmdd"
REP = PS.PriceSet("real", 60.0, "cd91")
START, END = PS.START, "2024-12-31"
VOL = AT.VOL
VOL_NAME, MEAN_NAME, R6 = AT.VOL_NAME, AT.MEAN_NAME, "R6 정적"
STAGE3_OF = {"(가)": "F4 (가)", "안정형 40": "F2 안정형 40", VOL_NAME: "R6 주식 위험 몫 상한", R6: "R6 주식 위험 몫 상한",
             MEAN_NAME: "R6 주식 위험 몫 상한", "현재 보유": "F3 현재 보유", "60/40": "F1 60/40", "S&P500": "F5 S&P500"}
ORDER = tuple(STAGE3_OF)
STATICS = ("(가)", "안정형 40", R6, "현재 보유", "60/40", "S&P500")
FAMILY = (VOL_NAME, R6, MEAN_NAME)
M3_TARGETS = ("(가)", "안정형 40", VOL_NAME, R6, MEAN_NAME)
RECOMMENDED = AT.RECOMMENDED
STAGE5_OF = {"(가)": "F4 (가)", "안정형 40": "F2 안정형 40", R6: "R6 주식 위험 몫 상한", "현재 보유": "F3 현재 보유",
             "60/40": "F1 60/40", "S&P500": "F5 S&P500", VOL_NAME: AT.STAGE5_REF}
POLICY = [(ph, off, 10.0) for ph in range(3) for off in (0, 5, 10, 15, 20)] + [(0, 0, 20.0)]  # stage 5's variants
TOL_FILE, TOL_SAME = 1e-6, 1e-12
CHUNK = 50


def load_targets(path: Path = STAGE3_TARGETS) -> dict[str, dict[str, float]]:
    t = pd.read_csv(path, dtype=str)
    return {n: {c: float(x) for c, x in (p.split(":") for p in s.split(";"))} for n, s in zip(t["대상"], t["비중"])}


def weights_of(name: str) -> dict[str, float]:
    return load_targets()[STAGE3_OF[name]]


def own_start(p, w: dict[str, float]) -> str:
    first = p.close[list(w)].apply(lambda s: s.first_valid_index()).max()
    return str(max(pd.Timestamp(START), first).date())


def qrows_of(dates, phase=0, offset=0):
    rows = decision_rows(dates, "Q", phase) + int(offset)
    return rows[(rows >= 0) & (rows < len(dates) - 1)]


def sim_of(p, w: dict[str, float], start: str, cost: float = 10.0, end: str = END) -> K.DynamicSim:
    cols = sorted(set(w) | set(K.DEFENSIVE) | {LY.CD})
    return K.DynamicSim(p, cols, start, end, cost_bps=cost)


def _metrics(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return LY.cagr(P), LY.mdd(P)


# --------------------------------------------------------------------------- the R6 family (static, vol, mean)

def r6_family(p, w: dict[str, float], start: str, end: str = END) -> dict:
    sims = {c: sim_of(p, w, start, c, end) for c in (10.0, 20.0)}
    sim = sims[10.0]
    tab = LY.action_tables(w, "none", "none", sim.columns)
    return {"sims": sims, "sim": sim, "w": w, "tab": tab, "scaled": LY.scaled_tables(tab, sim.columns, True),
            "dates": sim.dates}


def boot_family(fam: dict, idx: np.ndarray, vol_offset: int = 0) -> dict:
    """R6 static, R6 + vol 10 % and the same-average-exposure static on the paths ``idx`` (n × T), 10 bp, quarter ends;
    the exposure levels come from each path's own static return."""
    sim, tab, q = fam["sim"], fam["tab"], qrows_of(fam["dates"])
    n, T = idx.shape
    d, c, f = PM.static_inputs(n, T, q)
    static = PM.boot_run(sim, idx, np.broadcast_to(tab, (n,) + tab.shape), d, c, f)
    lev = PM.vol_levels_many(PM.base_returns(static), fam["dates"], VOL, vol_offset)
    vd, vc, vf = PM.vol_inputs(lev, q)
    vol = PM.boot_run(sim, idx, np.broadcast_to(fam["scaled"], (n,) + fam["scaled"].shape), vd, vc, vf)
    kbar = PM.mean_exposure(lev)
    mean = PM.boot_run(sim, idx, PM.mean_tabs(tab[0], sim.columns, kbar), d, c, f)
    return {"static": static, "vol": vol, "mean": mean, "kbar": kbar, "lev": lev}


# --------------------------------------------------------------------------- M1: one price set

def _policy_inputs(dates, T, variants):
    dec = np.zeros((len(variants), T), dtype=bool)
    for i, (ph, off, _) in enumerate(variants):
        dec[i, qrows_of(dates, ph, off)] = True
    return dec


def m1_set(label: str, quick: bool = False) -> pd.DataFrame:
    ps = next(x for x in PS.all_sets() if x.label == label)
    p = PS.build(ps.prices, ps.hedge_bp, ps.cash, quirks=True)[0]
    rows = []
    by_cost = {c: [v for v in POLICY if v[2] == c] for c in (10.0, 20.0)}
    for name in STATICS:
        w = weights_of(name)
        start = own_start(p, w)
        for cost, variants in by_cost.items():
            sim = sim_of(p, w, start, cost)
            tab = LY.action_tables(w, "none", "none", sim.columns)
            n, T = len(variants), sim.n_days
            dec = _policy_inputs(sim.dates, T, variants)
            P = PM.boot_run(sim, None, np.broadcast_to(tab, (n,) + tab.shape), dec, np.zeros((n, T), int), np.zeros(n, int))
            cg, md = _metrics(P)
            for i, (ph, off, c) in enumerate(variants):
                rows.append({"세트": label, "대상": name, "위상": ph, "이동": off, "비용": c, "변동성 이동": 0,
                             "시작": str(sim.dates[0].date()), "cagr": cg[i], "mdd": md[i], "kbar": np.nan})
    w = weights_of(R6)
    fam = r6_family(p, w, own_start(p, w))
    sim10, T, dates = fam["sim"], fam["sim"].n_days, fam["dates"]
    d0, c0, f0 = PM.static_inputs(1, T, qrows_of(dates))
    base_r = PM.base_returns(PM.boot_run(sim10, None, fam["tab"][None], d0, c0, f0))
    for voff in (PM.VOL_OFFSETS[:2] if quick else PM.VOL_OFFSETS):
        lev = PM.vol_levels_many(base_r, dates, VOL, voff)
        kbar = float(PM.mean_exposure(lev)[0])
        for cost, variants in by_cost.items():
            sim = fam["sims"][cost]
            n = len(variants)
            q_dec = _policy_inputs(dates, T, variants)
            vdec = q_dec.copy()
            vdec[:, :-1] |= lev[0, 1:] != lev[0, :-1]
            vdec[:, T - 1] = False
            vch = np.tile(np.concatenate([lev[0, 1:], lev[0, -1:]]), (n, 1))
            Pv = PM.boot_run(sim, None, np.broadcast_to(fam["scaled"], (n,) + fam["scaled"].shape), vdec, vch,
                             np.full(n, lev[0, 0]))
            Pm = PM.boot_run(sim, None, np.broadcast_to(PM.mean_tabs(fam["tab"][0], sim.columns, [kbar])[0], (n, 1, len(sim.columns))),
                             q_dec, np.zeros((n, T), int), np.zeros(n, int))
            for nm, P in ((VOL_NAME, Pv), (MEAN_NAME, Pm)):
                cg, md = _metrics(P)
                for i, (ph, off, c) in enumerate(variants):
                    rows.append({"세트": label, "대상": nm, "위상": ph, "이동": off, "비용": c, "변동성 이동": voff,
                                 "시작": str(dates[0].date()), "cagr": cg[i], "mdd": md[i], "kbar": kbar})
    return pd.DataFrame(rows)


def _m1_job(args):
    return m1_set(*args)


def run_m1(work: Path, quick: bool, workers: int) -> pd.DataFrame:
    sets = [REP.label] + [x.label for x in PS.all_sets() if x != REP]
    if quick:
        sets = sets[:3]
    todo = [lb for lb in sets if not (work / f"m1_{lb}.csv").exists()]
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max(1, min(workers, len(todo)))) as ex:
            for k, (lb, res) in enumerate(zip(todo, ex.map(_m1_job, [(lb, quick) for lb in todo]))):
                res.to_csv(work / f"m1_{lb}.csv", index=False)
                print(f"  M1 세트 {k + 1}/{len(todo)} {lb} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat([pd.read_csv(work / f"m1_{lb}.csv") for lb in sets], ignore_index=True)


# --------------------------------------------------------------------------- M2: block bootstrap, pre-tax

def windows(p) -> dict[str, list[str]]:
    """start date → targets with that window."""
    out: dict[str, list[str]] = {}
    for name in ORDER:
        out.setdefault(own_start(p, weights_of(name)), []).append(name)
    return out


def indices_for(T: int, n: int, block: int) -> np.ndarray:
    return PM.stationary_indices(T, n, block, PM.SEED + int(block))


def m2_block(p, block: int, n: int) -> tuple[pd.DataFrame, dict]:
    rows, fams = [], {}
    for start, names in windows(p).items():
        T = None
        for name in names:
            if name in FAMILY:
                continue
            w = weights_of(name)
            sim = sim_of(p, w, start)
            T = sim.n_days
            idx = indices_for(T, n, block)
            tab = LY.action_tables(w, "none", "none", sim.columns)
            d, c, f = PM.static_inputs(n, T, qrows_of(sim.dates))
            cg, md = _metrics(PM.boot_run(sim, idx, np.broadcast_to(tab, (n,) + tab.shape), d, c, f))
            rows += [{"대상": name, "블록": block, "경로": i, "cagr": cg[i], "mdd": md[i], "kbar": np.nan} for i in range(n)]
        if any(nm in FAMILY for nm in names):
            fam = r6_family(p, weights_of(R6), start)
            idx = indices_for(fam["sim"].n_days, n, block)
            out = boot_family(fam, idx, 0)
            for nm, key in ((R6, "static"), (VOL_NAME, "vol"), (MEAN_NAME, "mean")):
                cg, md = _metrics(out[key])
                rows += [{"대상": nm, "블록": block, "경로": i, "cagr": cg[i], "mdd": md[i],
                          "kbar": out["kbar"][i] if nm != R6 else np.nan} for i in range(n)]
            fams[start] = {"lev": out["lev"].astype(np.int8), "kbar": out["kbar"]}
    return pd.DataFrame(rows), fams


def run_m2(p, work: Path, blocks, n: int) -> tuple[pd.DataFrame, dict]:
    parts, lev126 = [], None
    for b in blocks:
        f = work / f"m2_{b}.csv"
        if f.exists() and b != PM.BLOCK:
            parts.append(pd.read_csv(f))
            continue
        t0 = time.time()
        R, fams = m2_block(p, b, n)
        R.to_csv(f, index=False)
        parts.append(R)
        if b == PM.BLOCK:
            lev126 = fams
        print(f"  M2 블록 {b}: 경로 {n} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True), lev126


# --------------------------------------------------------------------------- M3: the same paths after tax

def arrays(p, w: dict[str, float], start: str):
    """codes, dates, daily returns, cash, weights over [start, END] (as A45)."""
    codes = list(w)
    idx = p.close.index
    dates = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(END))]
    rets = p.close[codes].ffill().pct_change(fill_method=None).loc[dates].to_numpy()
    cash = p.cash.reindex(idx).fillna(0.0).loc[dates].to_numpy()
    return codes, dates, rets, cash, np.array([w[c] for c in codes], float)


_CTX: dict = {}


def _init(ctx):
    _CTX.clear()
    _CTX.update(ctx)


def m3_one(name: str, i: int | None) -> dict:
    """One path (i None = the historical order) of one target under the declared policy."""
    x = _CTX[name]
    codes, dates, rets, cash, wv = x["codes"], x["dates"], x["rets"], x["cash"], x["wv"]
    if i is None:
        r, c, lev, kbar = rets, cash, x.get("lev0"), x.get("kbar0")
    else:
        rr = x["idx"][i]
        r, c = rets[rr], cash[rr]
        lev = x["lev"][i].astype(int) if "lev" in x else None
        kbar = float(x["kbar"][i]) if "kbar" in x else None
    kw = {}
    weights = wv
    if name == VOL_NAME:
        kw["targets"] = AT.vol_targets(x["w"], codes, lev)
        weights = None
    elif name == MEAN_NAME:
        weights = PM.mean_weights(x["w"], codes, kbar)
    res = AC.run_policy(r, c, dates, codes, weights, AC.DECLARED, **kw)
    return {"대상": name, "경로": -1 if i is None else i, "cagr": res["twr_cagr"], "mdd": res["twr_mdd"], "irr": res["irr"],
            "kbar": kbar if name in (VOL_NAME, MEAN_NAME) else np.nan}


def _m3_job(args):
    name, lo, hi = args
    return [m3_one(name, i) for i in range(lo, hi)]


def m3_context(p, lev126: dict, n: int) -> dict:
    ctx = {}
    for name in M3_TARGETS:
        w = weights_of(name)
        start = own_start(p, w)
        codes, dates, rets, cash, wv = arrays(p, w, start)
        x = {"w": w, "codes": codes, "dates": dates, "rets": rets, "cash": cash, "wv": wv,
             "idx": indices_for(len(dates), n, PM.BLOCK).astype(np.int32)}
        if name in (VOL_NAME, MEAN_NAME):
            fam = r6_family(p, w, start)
            if not fam["dates"].equals(dates):
                raise RuntimeError("계좌 시뮬레이터와 DynamicSim의 날짜가 다릅니다")
            h = boot_family(fam, PM.identity_indices(len(dates)), 0)
            x["lev0"], x["kbar0"] = h["lev"][0], float(h["kbar"][0])
            x["lev"], x["kbar"] = lev126[start]["lev"], lev126[start]["kbar"]
        else:
            sim = sim_of(p, w, start)
            if not sim.dates.equals(dates):
                raise RuntimeError("계좌 시뮬레이터와 DynamicSim의 날짜가 다릅니다")
        ctx[name] = x
    return ctx


def run_m3(ctx: dict, work: Path, n: int, workers: int) -> pd.DataFrame:
    parts = []
    t0 = time.time()
    for k, name in enumerate(M3_TARGETS):
        f = work / f"m3_{k}.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
            continue
        jobs = [(name, lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]
        with ProcessPoolExecutor(max(1, workers), initializer=_init, initargs=({name: ctx[name]},)) as ex:
            R = pd.DataFrame([r for part in ex.map(_m3_job, jobs) for r in part])
        R.to_csv(f, index=False)
        parts.append(R)
        print(f"  M3 {name}: 경로 {n} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- reproduction

def repro_b(p) -> dict:
    """Bootstrap code with the historical order = DynamicSim (every target, REP), vectorised vol levels = aa_layers."""
    out = {"targets": {}}
    for name in STATICS:
        w = weights_of(name)
        sim = sim_of(p, w, own_start(p, w))
        tab = LY.action_tables(w, "none", "none", sim.columns)
        d, c, f = PM.static_inputs(1, sim.n_days, qrows_of(sim.dates))
        ref, _, _ = sim.run(tab[None], d, c, f)
        got = PM.boot_run(sim, PM.identity_indices(sim.n_days), tab[None], d, c, f)
        out["targets"][name] = float(np.max(np.abs(got / got[:, :1] - ref / ref[:, :1])))
    w = weights_of(R6)
    start = own_start(p, w)
    fam = r6_family(p, w, start)
    h = boot_family(fam, PM.identity_indices(fam["sim"].n_days), 0)
    _, lev_ref = AT.vol_levels_of(p, w, start, END, VOL)
    out["vol_levels_equal"] = bool(np.array_equal(h["lev"][0], lev_ref))
    P5 = AT.stage5_vol_path(p, w, start, END, VOL)
    out["targets"][VOL_NAME] = float(np.max(np.abs(h["vol"][0] / h["vol"][0, 0] - P5)))
    out["kbar_hist"] = float(h["kbar"][0])
    out["ok"] = bool(out["vol_levels_equal"] and max(out["targets"].values()) < TOL_SAME)
    return out


def repro_c(ctx: dict) -> dict:
    """M3 with the historical order = A45's declared row (TWR CAGR, MDD)."""
    t = pd.read_csv(A45_TABLE).set_index("대상")
    rows = []
    for name in M3_TARGETS:
        _init({name: ctx[name]})
        r = m3_one(name, None)
        ref = t.loc[name]
        d = max(abs(r["cagr"] - ref["세후 CAGR (TWR)"]), abs(r["mdd"] - ref["세후 MDD (TWR)"]))
        rows.append({"대상": name, "cagr": r["cagr"], "mdd": r["mdd"], "a45_cagr": float(ref["세후 CAGR (TWR)"]),
                     "a45_mdd": float(ref["세후 MDD (TWR)"]), "diff": float(d), "ok": bool(d < TOL_FILE)})
    return {"rows": rows, "ok": all(r["ok"] for r in rows)}


def repro_a(M1: pd.DataFrame) -> dict:
    """M1 base cells = stage 5's file on every set present; REP's 16-variant worst MDD = stage 5's."""
    s5 = pd.read_csv(STAGE5)
    base = M1[(M1["위상"] == 0) & (M1["이동"] == 0) & (M1["비용"] == 10.0) & (M1["변동성 이동"] == 0)]
    rows, worst = [], []
    for name, s5name in STAGE5_OF.items():
        for lb in sorted(set(M1["세트"])):
            ref = s5[(s5["세트"] == lb) & (s5["이름"] == s5name)]
            got = base[(base["세트"] == lb) & (base["대상"] == name)]
            if ref.empty or got.empty:
                rows.append({"대상": name, "세트": lb, "diff": np.inf, "ok": False})
                continue
            d = max(abs(got["cagr"].iloc[0] - ref["cagr"].iloc[0]), abs(got["mdd"].iloc[0] - ref["mdd"].iloc[0]))
            rows.append({"대상": name, "세트": lb, "diff": float(d), "ok": bool(d < TOL_FILE)})
        ref = s5[(s5["세트"] == REP.label) & (s5["이름"] == s5name)]["정책 변형 최악 MDD"].iloc[0]
        cells = M1[(M1["세트"] == REP.label) & (M1["대상"] == name) & (M1["변동성 이동"] == 0)]
        d = abs(float(cells["mdd"].min()) - float(ref))
        worst.append({"대상": name, "m1_worst": float(cells["mdd"].min()), "stage5_worst": float(ref), "diff": float(d),
                      "ok": bool(d < TOL_FILE)})
    return {"max_diff": float(max(r["diff"] for r in rows)), "cells": len(rows), "failed": [r for r in rows if not r["ok"]],
            "worst": worst, "ok": all(r["ok"] for r in rows) and all(r["ok"] for r in worst)}


# --------------------------------------------------------------------------- verdicts

def analyse(M1: pd.DataFrame, M2: pd.DataFrame, M3: pd.DataFrame, hist: dict, hist3: dict) -> dict:
    zrows = []
    for name in ORDER:
        m1 = M1[M1["대상"] == name]["mdd"]
        z1 = PM.zone_of(m1)
        m2 = M2[(M2["대상"] == name) & (M2["블록"] == PM.BLOCK)]["mdd"]
        z2 = PM.zone_of(m2)
        row = {"대상": name, "역사 MDD": hist[name]["mdd"], "역사 CAGR": hist[name]["cagr"],
               "Z1 구역": z1["zone"], "M1 중앙값": z1["median"], "M1 하위 10%": z1["q10"], "M1 최악": z1["worst"],
               "M1 −15% 안 비율": z1["p15"], "M1 칸 수": z1["n"],
               "Z2 구역": z2["zone"], "M2 중앙값": z2["median"], "M2 하위 10%": z2["q10"], "M2 최악": z2["worst"],
               "M2 −15% 안 확률": z2["p15"], "M2 −20% 안 확률": z2["p20"],
               "M2 CAGR 중앙값": float(M2[(M2["대상"] == name) & (M2["블록"] == PM.BLOCK)]["cagr"].median()),
               "역사 MDD의 M2 분위": PM.percentile_of(hist[name]["mdd"], m2)}
        for b in PM.BLOCKS:
            if b == PM.BLOCK:
                continue
            mb = M2[(M2["대상"] == name) & (M2["블록"] == b)]["mdd"]
            if len(mb):
                zb = PM.zone_of(mb)
                row.update({f"블록 {b} 구역": zb["zone"], f"블록 {b} 중앙값": zb["median"], f"블록 {b} 하위 10%": zb["q10"]})
        z3 = None
        if name in M3_TARGETS and len(M3):
            m3 = M3[(M3["대상"] == name) & (M3["경로"] >= 0)]["mdd"]
            z3 = PM.zone_of(m3)
            row.update({"Z3 구역": z3["zone"], "M3 중앙값": z3["median"], "M3 하위 10%": z3["q10"], "M3 최악": z3["worst"],
                        "M3 −15% 안 확률": z3["p15"], "역사 세후 MDD": hist3.get(name, {}).get("mdd", np.nan),
                        "M3 CAGR 중앙값": float(M3[(M3["대상"] == name) & (M3["경로"] >= 0)]["cagr"].median())})
        row["확률 구역"] = PM.worst_zone(z1["zone"], z2["zone"], z3["zone"] if z3 else None)
        row["역사 구역"] = AT.zone(hist[name]["mdd"])
        zrows.append(row)
    Z = pd.DataFrame(zrows)

    def paired(D, by, extra=None):
        v = D[D["대상"] == VOL_NAME].set_index(by)
        m = D[D["대상"] == MEAN_NAME].set_index(by)
        j = v.join(m, lsuffix="_v", rsuffix="_m", how="inner")
        return j["mdd_v"] - j["mdd_m"], j["cagr_m"] - j["cagr_v"]

    d2, c2 = paired(M2[M2["블록"] == PM.BLOCK], ["경로"])
    L2 = PM.layer_stats(d2, -c2)
    L3 = None
    if len(M3):
        d3, c3 = paired(M3[M3["경로"] >= 0], ["경로"])
        L3 = PM.layer_stats(d3, -c3)
    d1, _ = paired(M1, ["세트", "위상", "이동", "비용", "변동성 이동"])
    by_off = d1.groupby(level="변동성 이동").median()
    blocks = {}
    for b in PM.BLOCKS:
        db, _ = paired(M2[M2["블록"] == b], ["경로"])
        if len(db):
            blocks[str(b)] = PM.layer_stats(db)
    k2 = M2[(M2["대상"] == VOL_NAME) & (M2["블록"] == PM.BLOCK)]["kbar"]
    layer = {"verdict": PM.layer_of(L2, L3), "M2": L2, "M3": L3,
             "M1_share_ge_2pp": float(np.mean(d1 >= PM.LAYER_PP)), "M1_median": float(d1.median()),
             "M1_median_by_vol_offset": {str(k): float(v) for k, v in by_off.items()},
             "blocks": blocks, "kbar_M2_median": float(k2.median()), "kbar_M2_q10_q90": [float(k2.quantile(0.1)), float(k2.quantile(0.9))]}
    pairs = pd.DataFrame({"ΔMDD (M2)": d2.to_numpy(), "ΔCAGR 손실 (M2)": c2.to_numpy()})
    if L3 is not None:
        pairs["ΔMDD (M3)"] = d3.reindex(d2.index).to_numpy()
    t = Z.set_index("대상")
    summary = {
        "zones": {n: {"확률 구역": t.loc[n, "확률 구역"], "Z1": t.loc[n, "Z1 구역"], "Z2": t.loc[n, "Z2 구역"],
                      "Z3": t.loc[n, "Z3 구역"] if "Z3 구역" in t and isinstance(t.loc[n, "Z3 구역"], str) else None,
                      "역사 MDD": float(t.loc[n, "역사 MDD"]), "M2 중앙값": float(t.loc[n, "M2 중앙값"]),
                      "M2 하위 10%": float(t.loc[n, "M2 하위 10%"]), "M2 −15% 안 확률": float(t.loc[n, "M2 −15% 안 확률"]),
                      "M3 −15% 안 확률": float(t.loc[n, "M3 −15% 안 확률"]) if "M3 −15% 안 확률" in t and np.isfinite(t.loc[n, "M3 −15% 안 확률"]) else None,
                      "역사 MDD의 M2 분위": float(t.loc[n, "역사 MDD의 M2 분위"])}
                  for n in ORDER},
        "layer": layer,
    }
    return {"zones": Z, "pairs": pairs, "summary": summary}


def history(M1: pd.DataFrame) -> dict:
    b = M1[(M1["세트"] == REP.label) & (M1["위상"] == 0) & (M1["이동"] == 0) & (M1["비용"] == 10.0) & (M1["변동성 이동"] == 0)]
    return {r["대상"]: {"cagr": float(r["cagr"]), "mdd": float(r["mdd"])} for _, r in b.iterrows()}


def run(out: Path, quick: bool, workers: int, n: int) -> dict:
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = PS.build(REP.prices, REP.hedge_bp, REP.cash, quirks=True)[0]
    rep = {"b_bootstrap_code": repro_b(p)}
    m3n = min(n, 60) if quick else n
    blocks = (PM.BLOCK,) if quick else PM.BLOCKS
    _, lev_quick = m2_block(p, PM.BLOCK, 2)  # contexts only (the M3 checks use the historical order)
    ctx = m3_context(p, lev_quick, 2)
    rep["c_m3_vs_a45"] = repro_c(ctx)
    rep["ok"] = rep["b_bootstrap_code"]["ok"] and rep["c_m3_vs_a45"]["ok"]
    _save_json(out / "repro.json", rep)
    if not rep["ok"]:
        print("재현 점검: 실패", flush=True)
        raise SystemExit("재현 점검 실패 (b 또는 c): repro.json을 보세요. 더 돌리지 않습니다.")
    print(f"재현 점검 (b · c): 통과 ({time.time() - t0:.0f}s)", flush=True)
    M1 = run_m1(work, quick, workers)
    rep["a_m1_vs_stage5"] = repro_a(M1)
    rep["ok"] = rep["ok"] and rep["a_m1_vs_stage5"]["ok"]
    _save_json(out / "repro.json", rep)
    print("재현 점검:", "통과" if rep["ok"] else "실패", flush=True)
    if not rep["ok"]:
        raise SystemExit("재현 점검 실패 (a): repro.json을 보세요. 더 돌리지 않습니다.")
    M1.to_csv(out / "m1_cells.csv.gz", index=False)
    M2, lev126 = run_m2(p, work, blocks, n)
    M2.to_csv(out / "m2_paths.csv.gz", index=False)
    ctx = m3_context(p, lev126, n)
    M3 = run_m3(ctx, work, m3n, workers)
    hist3 = {}
    for name in M3_TARGETS:
        _init({name: ctx[name]})
        hist3[name] = m3_one(name, None)
    M3 = pd.concat([M3, pd.DataFrame(hist3.values())], ignore_index=True)
    M3.to_csv(out / "m3_paths.csv.gz", index=False)
    A = analyse(M1, M2, M3, history(M1), hist3)
    A["zones"].to_csv(out / "zones.csv", index=False)
    A["pairs"].to_csv(out / "layer_pairs.csv.gz", index=False)
    return {"repro_ok": rep["ok"], **A["summary"], "paths": n, "m3_paths": m3n, "blocks": list(blocks),
            "m1_cells": int(len(M1)), "quick": quick}


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x)),
                    encoding="utf-8")


# --------------------------------------------------------------------------- figures

ZCOL = {"안": "#54A24B", "경계": "#F2A93B", "밖": "#E45756"}


def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    dest.mkdir(parents=True, exist_ok=True)
    made = []
    Z = pd.read_csv(out / "zones.csv")
    M1 = pd.read_csv(out / "m1_cells.csv.gz")
    M2 = pd.read_csv(out / "m2_paths.csv.gz")
    M3 = pd.read_csv(out / "m3_paths.csv.gz")
    names = list(Z["대상"])

    fig, ax = plt.subplots(figsize=(14, 6))
    pos = np.arange(len(names))
    groups = [(-0.27, "#9ECAE9", "M1 가정 격자"), (0.0, "#4C78A8", f"M2 표본 경로 (세전, 블록 {PM.BLOCK})"),
              (0.27, "#F58518", "M3 표본 경로 (세후)")]
    for k, (dx, col, lab) in enumerate(groups):
        vals = []
        for n in names:
            if k == 0:
                vals.append(M1[M1["대상"] == n]["mdd"].to_numpy() * 100)
            elif k == 1:
                vals.append(M2[(M2["대상"] == n) & (M2["블록"] == PM.BLOCK)]["mdd"].to_numpy() * 100)
            else:
                v = M3[(M3["대상"] == n) & (M3["경로"] >= 0)]["mdd"].to_numpy() * 100
                vals.append(v if len(v) else np.array([np.nan]))
        ok = [i for i, v in enumerate(vals) if np.isfinite(v).any()]
        bp = ax.boxplot([vals[i] for i in ok], positions=pos[ok] + dx, widths=0.22, whis=(10, 90), showfliers=False,
                        patch_artist=True)
        for b in bp["boxes"]:
            b.set_facecolor(col)
            b.set_alpha(0.8)
        ax.plot([], [], color=col, lw=6, label=lab)
    ax.scatter(pos, Z["역사 MDD"] * 100, marker="D", color="black", zorder=4, s=25, label="역사 경로 (대표 세트, 세전)")
    for y, c, t in ((-15, "#E45756", "−15%"), (-17, "#B279A2", "−17%"), (-20, "#555555", "−20%")):
        ax.axhline(y, color=c, lw=0.8, ls="--")
    for i, r in Z.iterrows():
        ax.text(i, 1.0, r["확률 구역"], ha="center", fontsize=9, color=ZCOL[r["확률 구역"]], fontweight="bold")
    ax.set_xticks(pos, names, rotation=15, fontsize=9)
    ax.set_ylabel("MDD (%) — 상자 = 25–75%, 수염 = 10–90%")
    ax.set_ylim(bottom=max(-60, float(np.nanmin(Z[["M2 최악"]].to_numpy()) * 100) - 3), top=3)
    ax.set_title("① MDD 분포: 가정 격자 · 표본 경로 세전 · 세후, 위 글자 = 확률 구역 (중앙값 −15% 안 & 하위 10% −20% 안 → 안)")
    ax.legend(fontsize=8, loc="lower center", ncol=4)
    made.append(str(figs._save(fig, dest / "1_distributions.png")))

    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.27
    cols = [("M1 −15% 안 비율", "M1 가정 격자", "#9ECAE9"), ("M2 −15% 안 확률", "M2 세전", "#4C78A8"),
            ("M3 −15% 안 확률", "M3 세후", "#F58518")]
    for k, (c, lab, col) in enumerate(cols):
        if c in Z:
            ax.bar(pos + (k - 1) * w, Z[c].fillna(0) * 100, w, label=lab, color=col)
    ax.axhline(50, color="#E45756", ls="--", lw=0.8)
    ax.set_xticks(pos, names, rotation=15, fontsize=9)
    ax.set_ylabel("MDD가 −15% 안에 머문 비율 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("② −15% 안에 머물 확률 (50% 선 = 중앙값이 −15%)")
    ax.legend(fontsize=8)
    made.append(str(figs._save(fig, dest / "2_inside_prob.png")))

    P = pd.read_csv(out / "layer_pairs.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    a = axes[0]
    bins = np.linspace(-6, 12, 46)
    a.hist(P["ΔMDD (M2)"] * 100, bins=bins, alpha=0.7, color="#4C78A8", label="M2 세전")
    if "ΔMDD (M3)" in P:
        a.hist(P["ΔMDD (M3)"].dropna() * 100, bins=bins, alpha=0.5, color="#F58518", label="M3 세후")
    a.axvline(0, color="black", lw=0.8)
    a.axvline(PM.LAYER_PP * 100, color="#E45756", ls="--", lw=0.8, label="2%p")
    a.set_xlabel("ΔMDD (%p) = 변동성 목표 MDD − 같은 평균 노출 정적 MDD (+면 변동성 목표가 얕음)")
    a.set_ylabel("경로 수")
    a.set_title("③ 같은 경로끼리 짝지은 변동성 층의 MDD 이점")
    a.legend(fontsize=8)
    b = axes[1]
    v = M1[M1["대상"] == VOL_NAME].set_index(["세트", "위상", "이동", "비용", "변동성 이동"])["mdd"]
    m = M1[M1["대상"] == MEAN_NAME].set_index(["세트", "위상", "이동", "비용", "변동성 이동"])["mdd"]
    d = (v - m).dropna()
    offs = sorted(d.index.get_level_values("변동성 이동").unique())
    b.boxplot([d.xs(o, level="변동성 이동").to_numpy() * 100 for o in offs], whis=(10, 90), showfliers=False)
    b.set_xticks(range(1, len(offs) + 1), [f"월말 + {o}일" for o in offs])
    b.axhline(PM.LAYER_PP * 100, color="#E45756", ls="--", lw=0.8)
    b.axhline(0, color="black", lw=0.8)
    b.set_ylabel("ΔMDD (%p)")
    b.set_title("M1 가정 격자: 변동성 결정일을 옮기면")
    made.append(str(figs._save(fig, dest / "3_layer.png")))

    fig, ax = plt.subplots(figsize=(13, 4.8))
    bl = [b for b in PM.BLOCKS if (M2["블록"] == b).any()]
    for k, b in enumerate(bl):
        med = [M2[(M2["대상"] == n) & (M2["블록"] == b)]["mdd"].median() * 100 for n in names]
        q10 = [M2[(M2["대상"] == n) & (M2["블록"] == b)]["mdd"].quantile(0.1) * 100 for n in names]
        x = pos + (k - (len(bl) - 1) / 2) * 0.25
        ax.scatter(x, med, marker="o", label=f"블록 {b}일 중앙값", zorder=3)
        ax.vlines(x, q10, med, lw=2, alpha=0.6)
    for y, c in ((-15, "#E45756"), (-17, "#B279A2"), (-20, "#555555")):
        ax.axhline(y, color=c, lw=0.8, ls="--")
    ax.set_xticks(pos, names, rotation=15, fontsize=9)
    ax.set_ylabel("MDD (%) — 점 = 중앙값, 선 끝 = 하위 10%")
    ax.set_title("④ 평균 블록 길이에 따른 M2 (판정은 126일)")
    ax.legend(fontsize=8)
    made.append(str(figs._save(fig, dest / "4_blocks.png")))
    return made


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--paths", type=int, default=PM.N_PATHS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = min(args.paths, 100) if args.quick else args.paths
    if not args.figures_only:
        t0 = time.time()
        summ = run(out, args.quick, args.workers, n)
        summ["seconds"] = round(time.time() - t0, 1)
        _save_json(out / "summary.json", summ)
        print(json.dumps({k: summ[k] for k in ("zones", "layer")}, ensure_ascii=False, indent=1, default=str))
    print(figures(out, FIG_DIR if not args.quick else out / "figures"))


if __name__ == "__main__":
    main(sys.argv[1:])
