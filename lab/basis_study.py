"""Stage R: price and dollar-volume filters on raw prices (DECISIONS 98).

    python -m lab.basis_study --stage raw-extra --panel DIR --crsp SRC --out DIR
    python -m lab.basis_study --stage repro|raw|report --panel DIR --stored-eligible FILE --stage-h DIR --stage-i DIR
                              --v2 DIR --r0x1 DIR --regimes FILE --stage-q DIR [--stage-q-jobs DIR] --out DIR
                              [--workers N] [--table-workers N] [--feature-workers N]

The hygiene screens h_price / h_liquidity / h_illiquid and the engine's min_price / min_dollar_volume
read the total-return adjusted close of lab/crsp.adjust, anchored at the panel's last date, so on the
"adjusted" basis (Stages C-Q) a price floor or a dollar volume on day t depends on splits and
dividends after t. The "raw" basis reads the day's CRSP |PRC| and VOL instead (lab/data.level_close,
level_dollar_volume; stored beside the panel as extra_raw_close / extra_raw_volume by --stage
raw-extra, existing panel files untouched). Indicators, ATR, signals and returns stay adjusted.

Stages (each skips finished files):
  raw-extra  the raw series beside the panel, with the alignment check |PRC| x SHROUT = extra_cap.
  repro      basis "adjusted" with the new code: the daily eligible mask against the stored one, the
             E7 (and E0) signal tables against the stored Stage L tables (rows and values), the slot
             accounts against Stage Q's goal table (1e-9; 40 bp against Stage Q's 40 bp jobs when
             given). Writes repro.json; the raw stage refuses to run unless it passed.
  raw        basis "raw": eligible mask, Q1 (sizes) and Q2 (forward returns), signal tables, accounts.
  report     both bases: Q3 (equal weight, unlimited, slot accounts, per trade), Q4 (Stage Q's four
             candidate cells), the registered verdicts, figures.
Only dates 1993-02-01 - 2023-12-31 enter a metric; the Q1 / Q2 frames are cut at 2023-12-31 before any
computation and forward windows past it are dropped. Output (aggregates only, the top-30 names table
stays in --out): eligible / tables / base / jobs per basis, q1_*.csv, q2_*.csv, q3_*.csv, q4_*.csv,
trade_stats.csv, verdict.json, figures/.

Details fixed before the raw run (the design leaves them open):
  - raw price on a listed day without a CRSP price: carried forward (the adjusted close does the same
    there, RET being missing); raw dollar volume on such a day 0 (as the adjusted volume).
  - Q1 / Q2 use the hygiene universe (daily top 3,000 minus the four hygiene screens, sc.daily_eligible);
    the engine's own min_price / min_dollar_volume (20-day) apply to the signal tables only.
  - Q1 shares are of the raw-basis passing stock-days; 5-year blocks 1993-97, 98-02, 03-07, 08-12,
    13-17, 18-23 (the last six years); cap bands by that day's cap_rank 1-500, 501-1,000, 1,001-3,000.
    The screen split counts each failing adjusted (raw) screen of a wrongly excluded (included)
    stock-day, and the stock-days failing only that screen.
  - Q2 forward return = adjusted close ffilled 252 sessions later / close - 1 (a delisted name keeps its
    delisting value); a day enters when t + 252 <= the last session of 2023, so the years are 1993-2022
    (30, not 31); yearly mean of the daily (group mean - both-pass mean) over days with the group; t =
    mean / (sd / sqrt(years)); median and >= +100% share pooled over stock-days.
  - Unlimited accounts: pick_study.calendar_weighted on the M1 rows with the 1% rule at ATR(14) x 3
    (account_study.sizing(stop, 10, "W2"): min(1% / stop, 10%)) for both exits, no gate, 20 bp.
  - (b) per E0 seed x exit the CAGR difference raw - adjusted, median of the 10; (c) per E0 seed x exit x
    setting the seed's curve value (P0: median over its 20 orders; P1: its one curve), difference raw -
    adjusted, median of the 40; noise = per setting x exit the sd of the 5 seed values, median of 8.
  - Relative verdict on the 8 setting x exit cells (no gate, U3000, 20 bp; E0 = the goal row, median
    over all its curves); the same count over all 48 cells (2 gates x 3 bands) reported.
  - Per trade: the signal-table rows (each signal traded alone, 20 bp, open ones at the last 2023
    close), no M1, cancelled orders out; keys (symbol, signal day); E0 = the 5 seeds pooled.
  - Regime states: Stage Q's regimes.parquet (independent of the basis).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import account_study as ns, crsp, entry_study as es, exclude_study as ms, exit_study as xs, figures, goal_study as gs
from . import large_study as ls, pair_study as pj, pick_study as ph, screens, stage_c_study as sc, tail_study as ts
from .crsp_study import SAMPLES, load_panel_with_cash
from .data import BASES, BENCHMARK, with_basis
from .engine import CANCELLED

# ---- registered (DECISIONS 98): fixed before any result
START, END = es.START, es.END  # 1993-02-01, 2023-12-31
GENS = {"E7": ["E7"], "E0": [f"R0s{s}" for s in pj.E0_SEEDS]}
EXITS = ["X1", "H1"]
SETTINGS = list(ls.SETTINGS)
FILTERS = ["none", "R3A1"]
BANDS = dict(ls.BANDS)
COSTS = (pj.COST_BPS, pj.CHECK_COST_BPS)  # 20 and 40 bp
SMALL_PP, LARGE_PP = 0.002, 0.005  # size verdict: all < 0.2 pp small, any >= 0.5 pp large
SIZE_PERIODS = ("1993-2023", "1993-2007", "2008-2023")
CANDIDATE_CELLS = [("N10_W2_P1", "U1000"), ("N10_W2_P1", "U500"), ("N20_W2_P1", "U1000"), ("N20_W2_P1", "U500")]
CANDIDATE = ("E7", "R3A1")
CANDIDATES_TO_PROCEED = 3  # >= 3 of 4: decision 97 runs on raw definitions; <= 2: cancelled unopened
GAP_SETTINGS = ("N20_W2_P1", "N50_W2_P1")  # (6) 2008-2023 gap, reported only
FWD = 252
CAP_BANDS = {"1–500": (1, 500), "501–1,000": (501, 1000), "1,001–3,000": (1001, 3000)}
BLOCKS = [(1993, 1997), (1998, 2002), (2003, 2007), (2008, 2012), (2013, 2017), (2018, 2023)]
LEVEL_SCREENS = ("h_price", "h_liquidity", "h_illiquid")  # the screens whose definition changes
TOP_NAMES = 30
SEED = 98

# signal tables in the stored layout (exit_study.table_of reads them): (dir, file, entry cell, exit cell)
E7_VALUE = ph._C_ENTRIES["E7"][1][ph._C_CANON["E7"]]  # +DI(14) crosses -DI(14) with ADX >= 20
TABLES = {
    "E7__X1": ("v2/signals/full", "E7__X1_signals.parquet", lambda s: pj._C_CANON["E7"], 1),
    "E7__H1": ("stage_h/h1/signals/full", "E7__H1_signals.parquet", lambda s: 0, xs.H_KI),
    "R0__X1": ("r0x1/signals/full", "R0__X1_signals.parquet", lambda s: pj.E0_SEEDS.index(s), 0),
    "R0__H1": ("stage_i/r0/signals/full", "R0__H1_signals.parquet", lambda s: pj.E0_SEEDS.index(s), xs.H_KI),
}
EXIT_DEF = {"X1": (("trailing_pct", ph.H1_TRAIL), {}),
            "H1": (("atr_stop", ph.K_VALUES[xs.H_KI]), {"trailing_pct": ph.H1_TRAIL})}
_W: dict = {}


def log(m: str) -> None:
    ts.log(m)


def basis_dir(out: Path, basis: str) -> Path:
    return out / basis


def table_dirs(root: Path) -> dict:
    """exit_study.table_of's folders; events / tables are not read for E7 and E0 under X1 / H1."""
    return {"stage_h": root / "stage_h", "stage_i": root / "stage_i", "v2": root / "v2", "events": root / "events",
            "r0x1": root / "r0x1", "tables": root / "tables"}


def cfg_of(bdir: Path, regimes: str) -> dict:
    return {"bins": pj.BINS_J3, "seeds": pj.E0_SEEDS, "base": str(bdir / "base.parquet"), "regimes": regimes}


# --------------------------------------------------------------------------- universe

def eligible_parts(panel) -> tuple[pd.DataFrame, dict]:
    """(daily eligible mask, the four hygiene screens' True = excluded masks) on the panel's price basis:
    the same mask as stage_c_study.daily_eligible without screening."""
    member = panel.eligible.reindex(index=panel.close.index, columns=panel.close.columns, fill_value=False).astype(bool)
    parts = {k: screens.ALL_SCREENS[k].compute(panel, v).astype("float64").fillna(0.0) > 0.5 for k, v in sc.HYGIENE.items()}
    ok = member.copy()
    for k in sc.HYGIENE:
        ok &= ~parts[k]
    ok[BENCHMARK] = False
    return ok, parts


def cut(frame: pd.DataFrame, end: str = END) -> pd.DataFrame:
    return frame.loc[:pd.Timestamp(end)]


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Cell-by-cell equality of two masks on a's rows and columns (b reindexed, missing = False)."""
    b = b.reindex(index=a.index, columns=a.columns, fill_value=False).astype(bool)
    d = a.to_numpy(dtype=bool) != b.to_numpy(dtype=bool)
    return {"rows": int(a.shape[0]), "cells": int(a.size), "true_a": int(a.to_numpy(dtype=bool).sum()),
            "true_b": int(b.to_numpy(dtype=bool).sum()), "differ": int(d.sum())}


# --------------------------------------------------------------------------- signal tables

def _t_init(panel_dir: str, elig_path: str, basis: str):
    panel = load_panel_with_cash(Path(panel_dir), basis)
    elig = pd.read_parquet(elig_path).reindex(index=panel.close.index, columns=panel.close.columns, fill_value=False)
    sc.SAMPLES = {"full": SAMPLES["full"]}
    sc._setup(panel, elig.astype(bool))
    _W["basis"] = basis


def table_menu(name: str, seed: int | None) -> tuple[str, str]:
    """Stage C's signals-only runner on one rule (E7 at its middle value or R0 = random buys of one
    seed) and one exit at the stored table's value (X1 25% trailing; H1 ATR x 3 stop + 25% trailing)."""
    e, x = name.split("__")
    key, value = ("g_random", seed) if e == "R0" else ("c_di_cross", E7_VALUE)
    sc.ENTRIES.clear()
    sc.ENTRIES[e] = (key, [value])
    sc.ENTRY_CANON[e] = 0
    (blk, val), extra = EXIT_DEF[x]
    sc.EXITS.clear()
    sc.EXITS[x] = (blk, [val])
    sc.EXIT_CANON[x] = 0
    sc.EXIT_EXTRA.pop(x, None)
    if extra:
        sc.EXIT_EXTRA[x] = dict(extra)
    sc.EXIT_BREAKEVEN.pop(x, None)
    return e, x


def _t_job(job):
    name, seed = job
    t0 = time.time()
    e, x = table_menu(name, seed)
    _, sig = sc._run_signals(("full", e, x))
    _, _, xi, yi = TABLES[name]
    sig = sig.assign(xi=xi(seed), yi=yi).astype({"return": "float32", "holding_days": "int32", "xi": "int64", "yi": "int64"})
    return job, sig, time.time() - t0


def table_path(root: Path, name: str) -> Path:
    d, f, _, _ = TABLES[name]
    return root / d / f


def build_tables(a, basis: str) -> None:
    root = basis_dir(Path(a.out), basis) / "tables"
    todo = [n for n in TABLES if not table_path(root, n).exists()]
    if not todo:
        return
    jobs = [(n, s) for n in todo for s in (pj.E0_SEEDS if n.startswith("R0") else [None])]
    log(f"[{basis}] 신호표 {len(todo)}개 ({len(jobs)}개 작업)")
    parts: dict = {n: {} for n in todo}
    elig = str(basis_dir(Path(a.out), basis) / "eligible_daily.parquet")
    with ProcessPoolExecutor(max_workers=min(a.table_workers, len(jobs)), initializer=_t_init, initargs=(a.panel, elig, basis)) as ex:
        for (name, seed), sig, sec in ex.map(_t_job, jobs, chunksize=1):
            parts[name][seed] = sig
            log(f"  {name} 씨앗 {seed}: 신호 {len(sig):,} · {sec:.0f}s")
    for name, by in parts.items():
        t = pd.concat([by[s] for s in sorted(by, key=lambda v: -1 if v is None else v)], ignore_index=True)
        cols = ["xi", "yi"] + sc.SIGNAL_KEEP
        f = table_path(root, name)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        t[cols].to_parquet(tmp, compression="zstd")
        tmp.replace(f)


TABLE_COLS = ["symbol", "signal_date", "entry_date", "exit_signal_date", "return", "holding_days", "closed", "exit_reason"]


def read_cell(f: Path, xi: int, yi: int) -> pd.DataFrame:
    s = pd.read_parquet(f, columns=["xi", "yi"] + TABLE_COLS)
    return s[(s["xi"] == xi) & (s["yi"] == yi)].drop(columns=["xi", "yi"]).reset_index(drop=True)


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Row-by-row equality (order kept) of two signal tables; NaN / NaT equal to NaN / NaT."""
    out = {"rows_new": len(a), "rows_stored": len(b), "equal": False, "max_return_diff": np.nan}
    if len(a) != len(b):
        return out
    ok = True
    for c in TABLE_COLS:
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if c == "return":
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            both = np.isfinite(xv) & np.isfinite(yv)
            out["max_return_diff"] = float(np.abs(xv[both] - yv[both]).max()) if both.any() else 0.0
            ok &= bool(np.array_equal(np.isfinite(xv), np.isfinite(yv)) and out["max_return_diff"] == 0.0)
        else:
            ok &= bool(((x == y) | (x.isna() & y.isna())).all())
    out["equal"] = ok
    return out


def stored_table(a, name: str) -> Path:
    stored = {"E7__X1": Path(a.v2) / "signals" / "full" / "E7__X1_signals.parquet",
              "E7__H1": Path(a.stage_h) / "h1" / "signals" / "full" / "E7__H1_signals.parquet",
              "R0__X1": Path(a.r0x1) / "signals" / "full" / "R0__X1_signals.parquet",
              "R0__H1": Path(a.stage_i) / "r0" / "signals" / "full" / "R0__H1_signals.parquet"}
    return stored[name]


def table_repro(a) -> pd.DataFrame:
    root = basis_dir(Path(a.out), "adjusted") / "tables"
    rows = []
    for name, (_, _, xi, yi) in TABLES.items():
        for s in (pj.E0_SEEDS if name.startswith("R0") else [None]):
            mine = read_cell(table_path(root, name), xi(s), yi)
            theirs = read_cell(stored_table(a, name), xi(s), yi)
            rows.append({"table": name, "seed": s, **same_rows(mine, theirs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- accounts

def _a_init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    ls._init(panel_dir, eligible, dirs, cfg)


def unlimited(src: str, x: str, cost_bps: float = pj.COST_BPS) -> tuple[dict, np.ndarray]:
    """The M1 rows of a source in pick_study.calendar_weighted (no stock limit) with the 1% rule at
    ATR(14) x 3 capped at 10%, measured as the slot accounts (goal_study._measure)."""
    d = ms.prepared(src, x)
    b = ms.source_base(src)
    r = gs.GRows(d, b, x)
    keep = ~ms.excluded(b, ns.RULE)[r.brow]
    w = ns.sizing(r.stop, 10, "W2")[0]
    m = keep & np.isfinite(w)
    eq, gross = ph.calendar_weighted(r.ei[m], r.sale[m], r.closed[m], r.sj[m], w[m], ls._W["O"], ls._W["LC"], ls._W["cash"],
                                     cost_bps / 10_000)
    rows, _, _ = gs._measure([(src, 0, int(m.sum()), eq)], cost_bps)
    row = {**rows[0], "gross_mean": float(gross.mean()), "gross_median_held": float(np.median(gross[gross > 0])) if (gross > 0).any() else 0.0,
           "beta": ls.beta(eq, ls._W["spy_r"])}
    return row, eq


def _a_job(job):
    if job[0] == "unlimited":
        _, src, x = job
        t0 = time.time()
        row, eq = unlimited(src, x)
        ms._W["prep"].clear()
        pj._W["tables"].clear()
        gen = "E0" if src.startswith("R0") else src
        return job, pd.DataFrame([{**row, "gen": gen, "exit": x}]), eq, time.time() - t0
    return (job,) + tuple(ls.run_job(job[1:] + ("grid",))[1:])


def a_job_name(job) -> str:
    if job[0] == "unlimited":
        return f"unl__{job[1]}__{job[2]}"
    _, gen, x, st, filters, bands, cost, srcs = job
    return ls.job_name(gen, x, st, filters, cost, "grid")


def account_jobs() -> list:
    bands = list(BANDS.items())
    jobs = []
    for gen, srcs in GENS.items():
        for x in EXITS:
            for st in SETTINGS:
                for cost in COSTS:
                    if st[2] == "P0":
                        jobs += [("grid", gen, x, st, [f], bands, cost, srcs) for f in FILTERS]
                    else:
                        jobs.append(("grid", gen, x, st, list(FILTERS), bands, cost, srcs))
            jobs += [("unlimited", s, x) for s in srcs]
    heavy = lambda j: 0 if j[0] == "unlimited" else (100 if j[3][2] == "P0" else 1) * (5 if j[1] == "E0" else 1)  # noqa: E731
    return sorted(jobs, key=lambda j: -heavy(j))


def run_accounts(a, basis: str) -> None:
    out = Path(a.out)
    bdir = basis_dir(out, basis)
    elig = str(bdir / "eligible_daily.parquet")
    dirs = table_dirs(bdir / "tables")
    cfg = cfg_of(bdir, a.regimes)
    A = types.SimpleNamespace(out=str(bdir), panel=a.panel, eligible=elig, dirs=dirs, cfg=cfg, feature_workers=a.feature_workers)
    saved = list(ns.EXITS)
    ns.EXITS[:] = list(EXITS)
    base = ns.build_base(A, GENS)
    ns.EXITS[:] = saved
    log(f"[{basis}] 기준표: 신호 {len(base):,}개, M1 통과 {(~ms.excluded(base, ns.RULE)).mean():.1%}")
    if not (bdir / "benchmarks.csv").exists():
        xs._init(a.panel, elig, dirs, cfg)
        px, wdates, w0, cash = pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"]
        btab, _ = gs.benchmarks_table(px, wdates, w0, cash)
        btab.to_csv(bdir / "benchmarks.csv", index=False)
        curves = gs.bench_curves(px, wdates, w0, cash)
        pd.DataFrame({k: curves[k] for k in ("mix", "ma200", "ew")}, index=wdates).to_parquet(bdir / "bench_curves.parquet")
    jdir = bdir / "jobs"
    jdir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in account_jobs() if not (jdir / f"{a_job_name(j)}_acc.csv").exists()]
    log(f"[{basis}] 계좌 작업 {len(todo)}개 남음")
    if todo:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_a_init, initargs=(a.panel, elig, dirs, cfg)) as ex:
            for i, res in enumerate(ex.map(_a_job, todo, chunksize=1)):
                job = res[0]
                nm = a_job_name(job)
                if job[0] == "unlimited":
                    _, acc, eq, sec = res
                    pd.DataFrame({"eq": eq}).to_parquet(jdir / f"{nm}_curve.parquet")
                else:
                    _, acc, win, mom, tr, sec = res
                    win.to_csv(jdir / f"{nm}_win.csv", index=False)
                    mom.to_csv(jdir / f"{nm}_mom.csv", index=False)
                acc.to_csv(jdir / f"{nm}_acc.csv", index=False)  # last: marks the job done
                log(f"  [{i + 1}/{len(todo)}] {nm} {sec:.0f}s · 남은 예상 {(len(todo) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")
    acc = pd.concat([pd.read_csv(f) for f in sorted(jdir.glob("*_acc.csv")) if not f.name.startswith("unl__")], ignore_index=True)
    acc.to_parquet(bdir / "acc.parquet")
    ls.summarize(acc).to_csv(bdir / "goal.csv", index=False)
    pd.concat([pd.read_csv(f) for f in sorted(jdir.glob("unl__*_acc.csv"))], ignore_index=True).to_csv(bdir / "unlimited.csv", index=False)


def goal_repro(mine: pd.DataFrame, stage_q: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Every numeric value of the adjusted-basis cells against Stage Q's (same keys)."""
    k = ["setting", "gen", "exit", "filter", "band"]
    q = stage_q[(stage_q["mode"] == "grid") & (stage_q["cost"] == cost) & stage_q["gen"].isin(GENS) & stage_q["filter"].isin(FILTERS)].set_index(k)
    m = mine[(mine["mode"] == "grid") & (mine["cost"] == cost)].set_index(k)
    cols = [c for c in q.columns if c in m.columns and c not in ("cost", "mode") and pd.api.types.is_numeric_dtype(q[c])]
    rows = []
    for key in q.index:
        if key not in m.index:
            rows.append({**dict(zip(k, key)), "value": "(missing)", "mine": np.nan, "stage_q": np.nan, "diff": np.inf})
            continue
        for c in cols:
            x, y = float(m.loc[key, c]), float(q.loc[key, c])
            d = 0.0 if (np.isnan(x) and np.isnan(y)) else abs(x - y)
            rows.append({**dict(zip(k, key)), "value": c, "mine": x, "stage_q": y, "diff": d if np.isfinite(d) else np.inf})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- stages

def stage_raw_extra(a) -> None:
    f = Path(a.out) / "raw_check.json"
    if all((Path(a.panel) / f"extra_{k}.parquet").exists() for k in crsp.RAW_EXTRA) and f.exists():
        log("원 가격 계열: 이미 있음")
        return
    chk = crsp.build_raw(Path(a.crsp), Path(a.panel), log)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(chk, indent=1), encoding="utf-8")
    log("원 가격 계열 저장 · 시총 맞춤 점검: " + json.dumps(chk, ensure_ascii=False))


def write_eligible(a, basis: str) -> pd.DataFrame:
    bdir = basis_dir(Path(a.out), basis)
    f = bdir / "eligible_daily.parquet"
    if f.exists():
        return pd.read_parquet(f)
    bdir.mkdir(parents=True, exist_ok=True)
    panel = load_panel_with_cash(Path(a.panel), basis)
    ok, _ = eligible_parts(panel)
    ok = cut(ok)
    ok.to_parquet(f)
    return ok


def stage_repro(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    log("재현 1: 수정주가 기준 매수 가능 목록")
    mine = write_eligible(a, "adjusted")
    stored = cut(pd.read_parquet(a.stored_eligible))
    res["eligible"] = compare_frames(mine, stored)
    log(f"  {res['eligible']}")
    log("재현 2: 수정주가 기준 신호표 (E7 · E0) vs 저장된 Stage L 표")
    build_tables(a, "adjusted")
    tr = table_repro(a)
    tr.to_csv(out / "repro_tables.csv", index=False)
    res["tables"] = tr.to_dict("records")
    log("\n" + tr.to_string(index=False))
    log("재현 3: 수정주가 기준 자리 제한 계좌 vs Stage Q 결과표")
    run_accounts(a, "adjusted")
    g = pd.read_csv(basis_dir(out, "adjusted") / "goal.csv")
    rp = goal_repro(g, pd.read_csv(Path(a.stage_q) / "goal.csv.gz"), pj.COST_BPS)
    rp.to_csv(out / "repro_goal.csv", index=False)
    ncell = int(rp[["setting", "gen", "exit", "filter", "band"]].drop_duplicates().shape[0])
    res["goal"] = {"cells": ncell, "values": len(rp), "max_diff": float(rp["diff"].max()), "over_1e-9": int((rp["diff"] > 1e-9).sum())}
    log(f"  20bp: {res['goal']}")
    if a.stage_q_jobs:
        files = sorted(Path(a.stage_q_jobs).glob("E7__*__40bp_acc.csv"))
        if files:
            q40 = ls.summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
            rp40 = goal_repro(g, q40, pj.CHECK_COST_BPS)
            rp40.to_csv(out / "repro_goal_40bp.csv", index=False)
            res["goal_40bp"] = {"cells": int(rp40[["setting", "gen", "exit", "filter", "band"]].drop_duplicates().shape[0]),
                                "values": len(rp40), "max_diff": float(rp40["diff"].max()), "over_1e-9": int((rp40["diff"] > 1e-9).sum())}
            log(f"  40bp: {res['goal_40bp']}")
    e7 = tr[tr["table"].str.startswith("E7")]
    res["pass"] = bool(res["eligible"]["differ"] == 0 and e7["equal"].all() and res["goal"]["over_1e-9"] == 0 and res["goal"]["cells"] > 0)
    (out / "repro.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"재현 판정: {'통과' if res['pass'] else '실패 — 실제 가격 기준은 돌리지 않습니다'}")
    return res


# ---- Q1 / Q2 (descriptive)

def q1_q2(a) -> None:
    out = Path(a.out)
    if (out / "q2_years.csv").exists():
        return
    log("Q1 · Q2: 두 기준의 위생 필터")
    panel = load_panel_with_cash(Path(a.panel), "raw")
    if "raw_close" not in panel.extra:
        raise SystemExit("패널에 원 가격 계열이 없습니다 (--stage raw-extra).")
    ok_r, parts_r = eligible_parts(panel)
    ok_a, parts_a = eligible_parts(with_basis(panel, "adjusted"))
    stored_r = pd.read_parquet(basis_dir(out, "raw") / "eligible_daily.parquet")
    assert compare_frames(cut(ok_r), stored_r)["differ"] == 0
    dates = panel.close.index
    win = (dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))
    cols = [c for c in panel.close.columns if c != BENCHMARK]
    take = lambda f: f.loc[win, cols].to_numpy(dtype=bool)  # noqa: E731
    R, A = take(ok_r), take(ok_a)
    WE, WI, BP = R & ~A, A & ~R, R & A
    cap = panel.extra["cap_rank"].reindex(index=dates, columns=cols).loc[win].to_numpy(dtype=float)
    yr = dates[win].year.to_numpy()
    # Q1 counts
    rows = []
    for y in np.unique(yr):
        m = yr == y
        row = {"year": int(y), "days": int(m.sum()), "raw_pass": int(R[m].sum()), "adj_pass": int(A[m].sum()),
               "wrong_out": int(WE[m].sum()), "wrong_in": int(WI[m].sum())}
        for b, (lo, hi) in CAP_BANDS.items():
            cb = (cap[m] >= lo) & (cap[m] <= hi)
            row[f"raw_pass_{b}"] = int((R[m] & cb).sum())
            row[f"wrong_out_{b}"] = int((WE[m] & cb).sum())
            row[f"wrong_in_{b}"] = int((WI[m] & cb).sum())
        rows.append(row)
    yt = pd.DataFrame(rows)
    yt["wrong_out_share"] = yt["wrong_out"] / yt["raw_pass"]
    yt["wrong_in_share"] = yt["wrong_in"] / yt["raw_pass"]
    for b in CAP_BANDS:
        yt[f"wrong_out_share_{b}"] = yt[f"wrong_out_{b}"] / yt[f"raw_pass_{b}"]
        yt[f"wrong_in_share_{b}"] = yt[f"wrong_in_{b}"] / yt[f"raw_pass_{b}"]
    yt.to_csv(out / "q1_years.csv", index=False)
    blk = []
    for lo, hi in BLOCKS + [(1993, 2023)]:
        s = yt[(yt["year"] >= lo) & (yt["year"] <= hi)].sum(numeric_only=True)
        row = {"block": f"{lo}–{hi}", "days": int(s["days"]), "raw_pass": int(s["raw_pass"]), "adj_pass": int(s["adj_pass"]),
               "wrong_out": int(s["wrong_out"]), "wrong_in": int(s["wrong_in"]),
               "wrong_out_share": s["wrong_out"] / s["raw_pass"], "wrong_in_share": s["wrong_in"] / s["raw_pass"],
               "raw_pass_per_day": s["raw_pass"] / s["days"], "wrong_out_per_day": s["wrong_out"] / s["days"], "wrong_in_per_day": s["wrong_in"] / s["days"]}
        for b in CAP_BANDS:
            row[f"wrong_out_share_{b}"] = s[f"wrong_out_{b}"] / s[f"raw_pass_{b}"]
            row[f"wrong_in_share_{b}"] = s[f"wrong_in_{b}"] / s[f"raw_pass_{b}"]
            row[f"wrong_out_{b}"] = int(s[f"wrong_out_{b}"])
            row[f"wrong_in_{b}"] = int(s[f"wrong_in_{b}"])
        blk.append(row)
    pd.DataFrame(blk).to_csv(out / "q1_blocks.csv", index=False)
    # which screen
    sr = []
    for grp, M, parts, lab in (("wrong_out", WE, parts_a, "수정주가 기준에서 걸린 필터"), ("wrong_in", WI, parts_r, "실제 가격 기준에서 걸린 필터")):
        F = {k: take(parts[k]) & M for k in LEVEL_SCREENS}
        tot = int(M.sum())
        for k in LEVEL_SCREENS:
            only = F[k] & ~np.logical_or.reduce([F[o] for o in LEVEL_SCREENS if o != k])
            sr.append({"group": grp, "screen": k, "label": lab, "stock_days": int(F[k].sum()), "share": F[k].sum() / tot if tot else np.nan,
                       "only": int(only.sum()), "only_share": only.sum() / tot if tot else np.nan, "group_total": tot})
    pd.DataFrame(sr).to_csv(out / "q1_screens.csv", index=False)
    # names (results folder only, never the repository)
    cnt = WE.sum(axis=0)
    top = np.argsort(-cnt)[:TOP_NAMES]
    names = panel.info.set_index("symbol")["name"] if "name" in panel.info else pd.Series(dtype=str)
    first = [int(yr[np.flatnonzero(WE[:, j])[0]]) if cnt[j] else np.nan for j in top]
    last = [int(yr[np.flatnonzero(WE[:, j])[-1]]) if cnt[j] else np.nan for j in top]
    pd.DataFrame({"symbol": [cols[j] for j in top], "ticker": [names.get(cols[j], "") for j in top], "days_wrong_out": cnt[top],
                  "first_year": first, "last_year": last}).to_csv(out / "names_top30.csv", index=False)
    # Q2: forward 252-session total return, frames cut at the end of 2023 before computing
    close = cut(panel.close)
    lc = close.ffill()
    fwd = (lc.shift(-FWD) / close - 1.0).loc[pd.Timestamp(START):, cols].to_numpy(dtype="float32")
    assert fwd.shape == R.shape
    ok_t = np.zeros(len(fwd), dtype=bool)
    ok_t[:max(0, len(fwd) - FWD)] = True  # t + 252 inside the window
    fin = np.isfinite(fwd) & ok_t[:, None]
    del panel, ok_r, ok_a, parts_r, parts_a, lc
    out_rows, pooled = [], {}
    bp_mean = np.where((BP & fin).sum(1) > 0, np.where(BP & fin, fwd, 0).sum(1) / np.maximum((BP & fin).sum(1), 1), np.nan)
    for grp, M in (("wrong_out", WE), ("wrong_in", WI)):
        g = M & fin
        n = g.sum(1)
        gm = np.where(n > 0, np.where(g, fwd, 0).sum(1) / np.maximum(n, 1), np.nan)
        day = gm - bp_mean
        for y in np.unique(yr[ok_t]):
            m = (yr == y) & ok_t & np.isfinite(day)
            out_rows.append({"group": grp, "year": int(y), "days": int(m.sum()), "stock_days": int(n[(yr == y) & ok_t].sum()),
                             "diff_mean": float(np.mean(day[m])) if m.any() else np.nan,
                             "group_mean": float(np.nanmean(gm[m])) if m.any() else np.nan,
                             "both_mean": float(np.nanmean(bp_mean[m])) if m.any() else np.nan})
        v = fwd[g]
        pooled[grp] = {"stock_days": int(g.sum()), "mean": float(v.mean()), "median": float(np.median(v)), "share_100": float((v >= 1.0).mean())}
    v = fwd[BP & fin]
    pooled["both"] = {"stock_days": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "share_100": float((v >= 1.0).mean())}
    q2y = pd.DataFrame(out_rows)
    summ = {}
    for grp in ("wrong_out", "wrong_in"):
        d = q2y.loc[(q2y["group"] == grp), "diff_mean"].dropna().to_numpy()
        summ[grp] = {"years": len(d), "first_year": int(q2y.loc[q2y["group"] == grp, "year"].min()),
                     "last_year": int(q2y.loc[q2y["group"] == grp, "year"].max()), "mean": float(d.mean()),
                     "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else np.nan,
                     "positive_years": int((d > 0).sum())}
    q2y.to_csv(out / "q2_years.csv", index=False)
    (out / "q2_summary.json").write_text(json.dumps({"yearly": summ, "pooled": pooled}, ensure_ascii=False, indent=1), encoding="utf-8")


def stage_raw(a) -> None:
    out = Path(a.out)
    rp = json.loads((out / "repro.json").read_text(encoding="utf-8")) if (out / "repro.json").exists() else {}
    if not rp.get("pass"):
        raise SystemExit("재현 점검을 통과하지 않아 실제 가격 기준은 돌리지 않습니다 (repro.json).")
    log("실제 가격 기준: 매수 가능 목록")
    write_eligible(a, "raw")
    q1_q2(a)
    build_tables(a, "raw")
    run_accounts(a, "raw")


# --------------------------------------------------------------------------- report

def tables_both(a, name: str) -> dict:
    """{basis: rows of the cell(s) within the window} for a table; E0 = 5 seeds with a 'seed' column."""
    out = {}
    for b in BASES:
        f = table_path(basis_dir(Path(a.out), b) / "tables", name)
        _, _, xi, yi = TABLES[name]
        parts = []
        for s in (pj.E0_SEEDS if name.startswith("R0") else [None]):
            t = read_cell(f, xi(s), yi)
            t = t[t["exit_reason"] != CANCELLED].drop_duplicates(["symbol", "signal_date"])
            parts.append(t.assign(seed=-1 if s is None else s))
        out[b] = pd.concat(parts, ignore_index=True)
    return out


def split_signals(t: dict) -> dict:
    """new (raw only), common and adjusted-only rows; the common rows' returns must agree."""
    k = ["seed", "symbol", "signal_date"]
    m = t["raw"].merge(t["adjusted"][k + ["return"]], on=k, how="outer", suffixes=("", "_adj"), indicator=True)
    common = m[m["_merge"] == "both"]
    diff = np.abs(common["return"].to_numpy(dtype=float) - common["return_adj"].to_numpy(dtype=float))
    both = np.isfinite(diff)
    return {"new": m.loc[m["_merge"] == "left_only", "return"].to_numpy(dtype=float),
            "common": common["return"].to_numpy(dtype=float),
            "adj_only": m.loc[m["_merge"] == "right_only", "return_adj"].to_numpy(dtype=float),
            "new_years": pd.DatetimeIndex(m.loc[m["_merge"] == "left_only", "signal_date"]).year.to_numpy(),
            "common_max_diff": float(diff[both].max()) if both.any() else 0.0}


def trade_stats(a) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng([SEED, 5])
    rows, hist = [], {}
    for gen, name in (("E7", "E7__X1"), ("E0", "R0__X1")):
        sp = split_signals(tables_both(a, name))
        hist[gen] = sp
        ref = sp["common"][np.isfinite(sp["common"])]
        rb = pj.boot_means(ref, rng)
        for grp in ("new", "adj_only"):
            r = sp[grp][np.isfinite(sp[grp])]
            row = {"gen": gen, "group": grp, "n": len(r), "ref_n": len(ref), "common_max_diff": sp["common_max_diff"],
                   "mean": float(r.mean()) if len(r) else np.nan, "median": float(np.median(r)) if len(r) else np.nan,
                   "big50": float((r >= 0.5).mean()) if len(r) else np.nan, "loss20": float((r <= -0.2).mean()) if len(r) else np.nan,
                   "ref_mean": float(ref.mean()), "ref_median": float(np.median(ref)), "ref_big50": float((ref >= 0.5).mean()),
                   "ref_loss20": float((ref <= -0.2).mean())}
            row.update(pj.ret_stats(r, ref, rb, rng))
            rows.append(row)
    return pd.DataFrame(rows), hist


def seed_values(acc: pd.DataFrame, cost: float, f: str = "none", band: str = "U3000") -> pd.DataFrame:
    """E0 per setting x exit x seed: the seed's curve value (P0: median over its orders)."""
    e = acc[(acc["gen"] == "E0") & (acc["cost"] == cost) & (acc["filter"] == f) & (acc["band"] == band) & (acc["mode"] == "grid")]
    return e.groupby(["setting", "exit", "src"]).median(numeric_only=True).reset_index()


METRICS = ["cagr_1993-2023", "cagr_1993-2007", "cagr_2008-2023", "cagr_2017-2023", "mdd_1993-2023", "w", "dd_cagr_1993-2023",
           "dd_cagr_1993-2007", "dd_cagr_2008-2023", "dd_cagr_2017-2023", "beta", "trades"]


def size_verdict(values: dict) -> dict:
    """values = {measure: {period: raw - adjusted}}: the largest absolute value decides."""
    biggest = max(abs(v) for d in values.values() for v in d.values() if np.isfinite(v))
    label = "작음" if all(abs(v) < SMALL_PP for d in values.values() for v in d.values() if np.isfinite(v)) else \
        "큼" if biggest >= LARGE_PP else "중간"
    return {"label": label, "largest_abs": biggest, "values": values}


def relative(goal: dict, cells=None) -> pd.DataFrame:
    """E7 minus E0 (goal rows, 20 bp) per setting x exit [x gate x band] on both bases."""
    rows = []
    g = {b: goal[b][(goal[b]["cost"] == pj.COST_BPS) & (goal[b]["mode"] == "grid")].set_index(["setting", "gen", "exit", "filter", "band"]) for b in BASES}
    for f in FILTERS:
        for band in BANDS:
            for st in SETTINGS:
                for x in EXITS:
                    s = ns.setting_name(st)
                    row = {"setting": s, "exit": x, "filter": f, "band": band}
                    for b in BASES:
                        for p in ("1993-2023",) + es.HALVES:
                            row[f"{b}_{p}"] = g[b].loc[(s, "E7", x, f, band), f"dd_cagr_{p}"] - g[b].loc[(s, "E0", x, f, band), f"dd_cagr_{p}"]
                    row["same_sign_halves"] = all(np.sign(row[f"adjusted_{p}"]) == np.sign(row[f"raw_{p}"]) for p in es.HALVES)
                    rows.append(row)
    return pd.DataFrame(rows)


def stage_report(a) -> dict:
    out = Path(a.out)
    agg = out / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    goal = {b: pd.read_csv(basis_dir(out, b) / "goal.csv") for b in BASES}
    acc = {b: pd.read_parquet(basis_dir(out, b) / "acc.parquet") for b in BASES}
    unl = {b: pd.read_csv(basis_dir(out, b) / "unlimited.csv") for b in BASES}
    btab = {b: pd.read_csv(basis_dir(out, b) / "benchmarks.csv").set_index("benchmark") for b in BASES}
    verdict: dict = {"repro": json.loads((out / "repro.json").read_text(encoding="utf-8"))}
    # (a) equal weight
    ew = pd.DataFrame([{"basis": b, **btab[b].loc["ew"].drop("label").to_dict()} for b in BASES])
    ew.to_csv(agg / "q3_ew.csv", index=False)
    va = {p: float(btab["raw"].loc["ew", f"cagr_{p}"] - btab["adjusted"].loc["ew", f"cagr_{p}"]) for p in SIZE_PERIODS}
    # (b) unlimited
    keys = ["gen", "exit", "src"]
    u = unl["raw"].set_index(keys)[METRICS].sub(unl["adjusted"].set_index(keys)[METRICS])
    ut = pd.concat({"adjusted": unl["adjusted"].set_index(keys)[METRICS + ["gross_mean"]], "raw": unl["raw"].set_index(keys)[METRICS + ["gross_mean"]],
                    "diff": u}, axis=1)
    ut.columns = [f"{x}_{y}" for x, y in ut.columns]
    ut.reset_index().to_csv(agg / "q3_unlimited.csv", index=False)
    ue = u.loc["E0"]
    vb = {p: float(ue[f"cagr_{p}"].median()) for p in SIZE_PERIODS}
    # (c) slot-account E0 curves
    sv = {b: seed_values(acc[b], pj.COST_BPS).set_index(["setting", "exit", "src"]) for b in BASES}
    d = sv["raw"][METRICS].sub(sv["adjusted"][METRICS])
    vc = {p: float(d[f"dd_cagr_{p}"].median()) for p in SIZE_PERIODS}
    noise = {b: {p: float(sv[b].groupby(["setting", "exit"])[f"dd_cagr_{p}"].std().median()) for p in SIZE_PERIODS} for b in BASES}
    slot = []
    for f in FILTERS:
        for band in BANDS:
            for cost in COSTS:
                s_ = {b: seed_values(acc[b], cost, f, band).set_index(["setting", "exit", "src"]) for b in BASES}
                dd = s_["raw"][METRICS].sub(s_["adjusted"][METRICS])
                slot.append({"filter": f, "band": band, "cost": cost, "curves": len(dd),
                             **{f"adjusted_{m}": float(s_["adjusted"][m].median()) for m in METRICS},
                             **{f"raw_{m}": float(s_["raw"][m].median()) for m in METRICS},
                             **{f"diff_{m}": float(dd[m].median()) for m in METRICS},
                             **{f"seed_sd_{b}_{p}": float(s_[b].groupby(["setting", "exit"])[f"dd_cagr_{p}"].std().median()) for b in BASES for p in SIZE_PERIODS}})
    pd.DataFrame(slot).to_csv(agg / "q3_slot_e0.csv", index=False)
    verdict["size"] = size_verdict({"a_ew_cagr": va, "b_unlimited_e0_cagr": vb, "c_slot_e0_dd_cagr": vc})
    verdict["size"]["seed_noise_sd"] = noise
    # cells side by side (E7 and E0 goal rows)
    k = ["setting", "gen", "exit", "filter", "band", "cost"]
    cols = METRICS + ["ws_mix", "ws_ma200", "cap_fill"]
    gg = {b: goal[b][goal[b]["mode"] == "grid"].set_index(k)[cols] for b in BASES}
    cells = pd.concat({"adjusted": gg["adjusted"], "raw": gg["raw"], "diff": gg["raw"] - gg["adjusted"]}, axis=1)
    cells.columns = [f"{x}_{y}" for x, y in cells.columns]
    cells.reset_index().to_csv(agg / "q3_cells.csv", index=False)
    # (3) direction, Q2
    q2 = json.loads((out / "q2_summary.json").read_text(encoding="utf-8"))
    verdict["direction"] = {"signs": {m: {p: int(np.sign(v)) for p, v in vals.items()} for m, vals in verdict["size"]["values"].items()},
                            "q2_wrong_out_t": q2["yearly"]["wrong_out"]["t"], "q2_wrong_out_mean": q2["yearly"]["wrong_out"]["mean"],
                            "q2_wrong_in_t": q2["yearly"]["wrong_in"]["t"], "q2_wrong_in_mean": q2["yearly"]["wrong_in"]["mean"]}
    # (4) relative
    rel = relative(goal)
    rel.to_csv(agg / "q3_relative.csv", index=False)
    main = rel[(rel["filter"] == "none") & (rel["band"] == "U3000")]
    verdict["relative"] = {"cells": len(main), "same_sign_both_halves": int(main["same_sign_halves"].sum()),
                           "all_cells": len(rel), "all_same_sign_both_halves": int(rel["same_sign_halves"].sum())}
    # (5) candidates on both bases
    cand_out, q4 = {}, []
    for b in BASES:
        bench = {k_: {**btab[b].loc[k_].to_dict()} for k_ in ("mix", "ma200", "ew")}
        g20 = goal[b][(goal[b]["cost"] == pj.COST_BPS) & (goal[b]["mode"] == "grid")]
        g40 = goal[b][(goal[b]["cost"] == pj.CHECK_COST_BPS) & (goal[b]["mode"] == "grid")]
        pass40 = {(r["setting"], r["gen"], r["exit"], r["filter"], r["band"]): gs.cell_pass(r, bench) for _, r in g40.iterrows()}
        c, s = ls.candidates(g20, bench, pass40)
        c.to_csv(agg / f"q4_cells_{b}.csv", index=False)
        s.to_csv(agg / f"q4_candidates_{b}.csv", index=False)
        for st, band in CANDIDATE_CELLS:
            r = s[(s["setting"] == st) & (s["gen"] == CANDIDATE[0]) & (s["filter"] == CANDIDATE[1]) & (s["band"] == band)]
            r = r.iloc[0].to_dict() if len(r) else {}
            q4.append({"basis": b, "setting": st, "band": band, **{k_: r.get(k_) for k_ in ("n20", "n40", "n_e0", "ok", "neighbour_n", "neighbour_band",
                                                                                           "candidate", "entry_specific")}})
        cand_out[b] = int(sum(1 for r in q4 if r["basis"] == b and r["candidate"]))
    q4 = pd.DataFrame(q4)
    q4.to_csv(agg / "q4_four_cells.csv", index=False)
    verdict["candidates"] = {"passing": cand_out, "proceed_97": cand_out["raw"] >= CANDIDATES_TO_PROCEED}
    # (6) 2008-2023 gap, reported only
    gap = []
    for b in BASES:
        g20 = goal[b][(goal[b]["cost"] == pj.COST_BPS) & (goal[b]["mode"] == "grid")]
        for st in GAP_SETTINGS:
            for x in EXITS:
                for band in BANDS:
                    for gen in GENS:
                        r = g20[(g20["setting"] == st) & (g20["exit"] == x) & (g20["filter"] == "R3A1") & (g20["band"] == band) & (g20["gen"] == gen)]
                        if len(r):
                            r = r.iloc[0]
                            gap.append({"basis": b, "setting": st, "exit": x, "band": band, "gen": gen, "dd_cagr_2008-2023": r["dd_cagr_2008-2023"],
                                        "gap_vs_ma200": r["dd_cagr_2008-2023"] - btab[b].loc["ma200", "dd_cagr_2008-2023"], "beta": r["beta"],
                                        "cagr_2017-2023": r["cagr_2017-2023"]})
    pd.DataFrame(gap).to_csv(agg / "q6_gap.csv", index=False)
    # per trade
    tstats, hist = trade_stats(a)
    tstats.to_csv(agg / "trade_stats.csv", index=False)
    verdict["trades"] = tstats.to_dict("records")
    for f in ("q1_years.csv", "q1_blocks.csv", "q1_screens.csv", "q2_years.csv", "q2_summary.json", "repro_tables.csv", "repro_goal.csv",
              "repro_goal_40bp.csv", "raw_check.json"):
        if (out / f).exists():
            shutil.copy2(out / f, agg / f)
    for b in BASES:
        btab[b].reset_index().to_csv(agg / f"benchmarks_{b}.csv", index=False)
    (agg / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if figures.available():
        write_figures(a, agg, goal, hist, btab)
    log("판정: " + json.dumps({k_: verdict[k_] for k_ in ("size", "direction", "relative", "candidates")}, ensure_ascii=False, default=float))
    return verdict


# --------------------------------------------------------------------------- figures

BASIS_LABEL = {"adjusted": "수정주가 기준 (Stage C–Q)", "raw": "실제 가격 기준 (고친 것)"}
BASIS_COLOR = {"adjusted": "#777777", "raw": figures.COLORS[0]}


def fig_q1(agg: Path) -> Path:
    plt = figures._plt()
    yt = pd.read_csv(agg / "q1_years.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, grp, title in zip(axes, ("wrong_out", "wrong_in"), ("잘못 빠짐: 실제 가격으로는 통과, 수정주가로는 제외", "잘못 들어옴: 수정주가로는 통과, 실제 가격으로는 제외")):
        ax.plot(yt["year"], yt[f"{grp}_share"] * 100, color="black", lw=2, marker="o", ms=3, label="전체")
        for k, b in enumerate(CAP_BANDS):
            ax.plot(yt["year"], yt[f"{grp}_share_{b}"] * 100, color=figures.COLORS[k], lw=1.2, label=f"시총 {b}위")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("연도")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("실제 가격 기준 통과 종목 · 일 대비 비율 (%)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Q1 위생 필터(주가 5달러 · 거래대금 100만 달러 · 비유동성 상위 20%)가 기준에 따라 갈린 종목 · 일 (상위 3,000, 1993–2023)", fontsize=10.5)
    return figures._save(fig, agg / "figures" / "q1_shares.png")


def fig_q2(agg: Path) -> Path:
    plt = figures._plt()
    q = pd.read_csv(agg / "q2_years.csv")
    s = json.loads((agg / "q2_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    w = 0.4
    for k, (grp, lab) in enumerate((("wrong_out", "잘못 빠진 종목 · 일"), ("wrong_in", "잘못 들어온 종목 · 일"))):
        g = q[q["group"] == grp]
        y = s["yearly"][grp]
        ax.bar(g["year"] + (k - 0.5) * w, g["diff_mean"] * 100, width=w, color=figures.COLORS[k],
               label=f"{lab}: 연도 평균 {y['mean'] * 100:+.1f}%p, t = {y['t']:.2f} ({y['years']}년)")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("이후 252거래일 총수익 − 같은 날 두 기준 모두 통과 종목 평균 (%p)")
    ax.set_xlabel("신호 날짜의 연도 (2023년은 이후 252일이 2023년을 넘어 제외)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8.5, frameon=False)
    p = s["pooled"]
    ax.set_title(f"Q2 갈린 종목 · 일의 이후 1년 수익 (중앙값: 잘못 빠짐 {p['wrong_out']['median'] * 100:+.1f}%, 잘못 들어옴 {p['wrong_in']['median'] * 100:+.1f}%, "
                 f"둘 다 통과 {p['both']['median'] * 100:+.1f}% · +100% 이상 {p['wrong_out']['share_100']:.1%} / {p['wrong_in']['share_100']:.1%} / {p['both']['share_100']:.1%})",
                 fontsize=9.5)
    return figures._save(fig, agg / "figures" / "q2_forward.png")


def fig_curves(a, agg: Path) -> Path:
    plt = figures._plt()
    out = Path(a.out)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for b in BASES:
        bdir = basis_dir(out, b)
        bc = pd.read_parquet(bdir / "bench_curves.parquet")
        axes[0].plot(bc.index, bc["ew"] / bc["ew"].iloc[0], color=BASIS_COLOR[b], lw=1.5, label=f"동일비중 · {BASIS_LABEL[b]}")
        for k, s in enumerate(GENS["E0"]):
            for x, ls_ in (("X1", "-"), ("H1", ":")):
                eq = pd.read_parquet(bdir / "jobs" / f"unl__{s}__{x}_curve.parquet")["eq"].to_numpy()
                axes[1].plot(bc.index, eq / eq[0], color=BASIS_COLOR[b], lw=0.8, ls=ls_, alpha=0.7,
                             label=f"무작위 매수 E0 · {BASIS_LABEL[b]}" if (k == 0 and x == "X1") else None)
    for ax, t in zip(axes, ("유니버스 동일비중 (매일 재구성, 비용 없음)", "종목 수 제한 없는 계좌: 무작위 매수 E0 씨앗 5개 × 매도 X1(실선) · H1(점선), M1, 1% 룰")):
        ax.set_yscale("log")
        ax.set_title(t, fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("두 기준의 기준선 곡선 (1993-02 – 2023-12, 로그)", fontsize=11)
    return figures._save(fig, agg / "figures" / "curves.png")


def fig_scatter(agg: Path, goal: dict, btab: dict) -> Path:
    plt = figures._plt()
    k = ["setting", "gen", "exit", "filter", "band", "cost"]
    g = {b: goal[b][(goal[b]["mode"] == "grid") & (goal[b]["cost"] == pj.COST_BPS)].set_index(k) for b in BASES}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, p in zip(axes, SIZE_PERIODS):
        for gen, col, mk in (("E0", "#999999", "o"), ("E7", figures.COLORS[1], "^")):
            idx = [i for i in g["adjusted"].index if i[1] == gen]
            x = g["adjusted"].loc[idx, f"dd_cagr_{p}"] * 100
            y = g["raw"].loc[idx, f"dd_cagr_{p}"].reindex(x.index) * 100
            ax.scatter(x, y, s=18, color=col, marker=mk, alpha=0.8, label=f"{gen} ({len(x)}칸)")
        lo, hi = ax.get_xlim()
        lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#333", lw=0.8)
        ax.axhline(btab["raw"].loc["ma200", f"dd_cagr_{p}"] * 100, color="#8e44ad", ls="--", lw=1, label="SPY 200일선 + 국채")
        ax.set_xlabel("수정주가 기준 (%)")
        ax.set_ylabel("실제 가격 기준 (%)")
        ax.set_title(f"{p} 낙폭 15% 맞춤 수익", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("자리 제한 계좌: 칸마다 한 점 (설정 4 × 매도 2 × 국면 2 × 범위 3, 편도 20bp; E0 = 곡선 전체의 중앙값)", fontsize=10.5)
    return figures._save(fig, agg / "figures" / "scatter_dd_cagr.png")


def fig_trades(agg: Path, hist: dict) -> Path:
    plt = figures._plt()
    st = pd.read_csv(agg / "trade_stats.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, gen, title in zip(axes, ("E7", "E0"), ("+DI 교차 E7 × X1", "무작위 매수 E0 × X1 (씨앗 5개 합)")):
        sp = hist[gen]
        for grp, col, lab, lsty in (("common", "#777", "두 기준 공통 신호", "-"), ("new", figures.COLORS[0], "실제 가격 기준에서 새로 생긴 신호", "-"),
                                    ("adj_only", figures.COLORS[1], "수정주가 기준에만 있던 신호", "--")):
            r = sp[grp][np.isfinite(sp[grp])]
            if len(r):
                pj._bar(ax, pj._density(np.histogram(r, pj.RET_EDGES)[0]), color=col, lw=1.5, ls=lsty, label=f"{lab} ({len(r):,}개, 평균 {r.mean() * 100:+.1f}%)")
        lines = []
        for grp, lab in (("new", "새로 생긴 vs 공통"), ("adj_only", "수정주가에만 vs 공통")):
            s = st[(st["gen"] == gen) & (st["group"] == grp)]
            if len(s) and pd.notna(s.iloc[0].get("st_diff")):
                s = s.iloc[0]
                lines.append(f"{lab}: 평균 차 {s['st_diff'] * 100:+.1f}%p (95% CI {s['st_ci_lo'] * 100:+.1f} – {s['st_ci_hi'] * 100:+.1f}), "
                             f"p = {s['st_mw_p']:.3f}, +50% 이상 {s['big50']:.1%} (공통 {s['ref_big50']:.1%}), −20% 이하 {s['loss20']:.1%} (공통 {s['ref_loss20']:.1%})")
        ax.text(0.99, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.axvline(0, color="#bbb", lw=0.8)
        ax.set_xlabel("신호 하나의 거래 수익률 (%, 편도 20bp, 양 끝 칸은 −60% 미만 / +200% 초과)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, frameon=False, loc="center right")
    fig.suptitle("거래 단위: 기준을 고치며 새로 생긴 신호와 사라진 신호의 수익률 분포 (밀도)", fontsize=10.5)
    return figures._save(fig, agg / "figures" / "trades_hist.png")


def write_figures(a, agg: Path, goal: dict, hist: dict, btab: dict) -> list[Path]:
    return [fig_q1(agg), fig_q2(agg), fig_curves(a, agg), fig_scatter(agg, goal, btab), fig_trades(agg, hist)]


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["raw-extra", "repro", "raw", "report"])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crsp", default=None, help="CRSP_*.parquet 폴더 (raw-extra)")
    for k in ("stored_eligible", "stage_h", "stage_i", "v2", "r0x1", "regimes", "stage_q", "stage_q_jobs"):
        ap.add_argument(f"--{k.replace('_', '-')}", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--table-workers", type=int, default=4)
    ap.add_argument("--feature-workers", type=int, default=6)
    a = ap.parse_args(argv)
    {"raw-extra": stage_raw_extra, "repro": stage_repro, "raw": stage_raw, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main(sys.argv[1:])
