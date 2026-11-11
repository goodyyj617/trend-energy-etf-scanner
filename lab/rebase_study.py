"""Stage S: the headline conclusions re-measured on the raw-price universe (DECISIONS 99).

    python -m lab.rebase_study --stage repro|tables|accounts|report --panel DIR --out DIR
        --stored-eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR --r0x1 DIR --tables DIR
        --stage-p DIR --stage-q DIR --stage-o DIR --stage-r DIR [--workers N] [--table-workers N] [--feature-workers N]

Stage R (lab/basis_study.py) re-measured E7 and the random buys E0 on the raw-price basis (the day's
CRSP |PRC| / VOL for the price and dollar-volume hygiene filters; indicators, signals and returns stay
adjusted). This runner rebuilds the signal tables of the other 13 entries (E1-E9 without E7, V1-V6)
x exits X1 / H1 on that basis (E7 and E0 reuse Stage R's raw tables), builds the base table (M1 / P1
percentiles in the corrected universe) and replays, unchanged, Stage P's whole grid
(lab/regime_study.run_job: 4 settings x 2 exits x 15 entries + E0 x 11 gates, 20 placebo regimes),
Stage Q's whole grid (lab/large_study.run_job: bands U3000 / U1000 / U500 x 5 gates, 10 pseudo
restrictions), Stage O's core with and without M1 (lab/goal_study.run_job) and the unlimited-slot
accounts (lab/basis_study.unlimited), then judges the five headline conclusions K1-K5 with the
stages' own verdict functions on both bases (adjusted = the stored results).

Stages (each skips finished files; a later stage refuses to run unless repro.json passed):
  repro     (a) the stored Stage L tables and eligible mask (adjusted) through this runner's job path
            = Stage P's and Stage Q's goal tables (every row: grid, placebo, pseudo; 20 bp) and their
            40 bp cells, and their benchmark tables, to 1e-9; (b) E2 and V3 x X1 / H1 tables rebuilt on
            the adjusted basis = the stored tables row for row; (c) the raw eligible mask rebuilt =
            Stage R's, and the E7 / E0 accounts on Stage R's raw tables (basis_study.run_accounts) =
            Stage R's raw goal table (96 cells x 20 / 40 bp) and unlimited table, to 1e-9.
  tables    raw signal tables E1-E9, V1-V6 x X1 / H1 (E7 / E0 copied from Stage R).
  accounts  raw: base table, bundles A (P grid), B (Q grid), C (O core, +/- M1), D (unlimited),
            40 bp cells, E (benchmarks); adjusted: the unlimited accounts on the stored tables.
            The raw E7 / E0 cells of the full run are checked again against Stage R (accounts.json).
  report    K1-K5 on both bases, candidates, reported-only tables, F (per trade), figures, agg/.
Only dates 1993-02-01 - 2023-12-31 enter a metric; nothing on a later date is computed into a metric
or printed (benchmark closes are cut at 2023-12-31; the eligible masks end there).

Details fixed before the raw run (the design leaves them open):
  - Regime states: Stage Q's regimes.parquet (checked equal to Stage P's). Seeds, orders, placebo shifts
    and pseudo draws are the stage functions' own (unchanged).
  - (a) compares every stored key and every numeric column present in both tables; a stored key
    missing in this run, or an extra key, fails. The 40 bp job set is derived from this run's 20 bp
    cells by the stages' own rule and must equal the stored set of 40 bp cells.
  - (c) and the full raw run: the E7 / E0 cells with gates none / R3 A1 must equal Stage R's raw goal
    table to 1e-9 (20 and 40 bp; Stage R's 40 bp cells were run for all of them).
  - K1 (a) per exit x half: the entry's unlimited CAGR strictly above the largest of the 5 E0 seeds'
    (same exit); (b) per no-gate U3000 system (setting x exit, 20 bp), the entry's dd_cagr strictly
    above E0's result-table row in 1993-2023 and both halves; "above random" = (a) for both exits and
    both halves and (b) in >= 6 of 8. Chance: each E0 seed in place of an entry against the other four
    (a: the largest of the other four; b: the seed's value = median over its own curves, E0 = median
    over the other four seeds' curves), counted out of 5.
  - K2: Stage O's goal-passing test (goal_study.beats) on the 8 no-gate systems (regime_study.systems).
  - K3: regime_study.q1's counts; an A1 gate (R1 / R3 / R4 / R5) "falls to chance" if helps <= q95, an
    A2 gate "exceeds chance" if helps > q95. R2 A1 reported.
  - K4: large_study.systems's goal-passing flags over its 120 systems (none + 4 A1 gates x 3 bands).
  - K5: per band the median over the 40 systems (setting x exit x gate) of dd_cagr_2008-2023 minus the
    same system's U3000 value; "changed" if either median >= +0.5 pp; large_study.q1's counts reported.
  - Candidates (verdict 4, descriptive): regime_study.candidates (U3000, gated cells) and large_study.candidates
    (bands) with 40 bp cells; every passing cell, every setting x entry x gate (x band) with at least one
    passing cell (with its failing conditions), and every other cell that fails only the matched-
    return comparisons by at most 0.5 pp (near miss). Shortfall = max over the three judged periods
    and the two gated benchmarks of (benchmark - cell), in pp. E0 seeds' pass rate from the stages'
    chance tables. No DSR / PSR, holdout untouched.
  - Bundle C: goal_study.run_job (mode grid) at N = 10 / 20 / 50, the 4 registered settings kept (it
    also computes the other settings of those slot counts; dropped). M1 effect = goal_study.systems
    'rule' minus 'none' per setting x exit. Its 'rule' rows are checked against the P-path no-gate
    cells. The adjusted side is Stage O's stored goal table.
  - Bundle D: basis_study.unlimited (M1 rows, 1% rule at ATR(14) x 3, 10% cap, calendar_weighted,
    20 bp) for the 15 entries and 5 E0 seeds x X1 / H1, on both bases (adjusted on the stored tables).
  - Bundle E: goal_study.benchmarks_table (SPY + T-bill, SPY200 + T-bill, EW), EW x regime as
    regime_study (cash when off) and large_study.band_benchmarks on the raw eligible mask.
  - P1 effect: per exit x gate x band, N10 / W2 / P1 minus N10 / W2 / P0 (system medians).
  - 2008-2023 gap: the A1 systems' dd_cagr_2008-2023 minus SPY200 + T-bill's, with SPY beta and
    plain CAGR 2017-2023.
  - Bundle F: the X1 signal-table rows of each entry (cancelled orders out, one row per symbol x
    signal day), split into new on raw / common / adjusted-only; per entry the new (and adjusted-only)
    rows against the common rows: mean difference with bootstrap 95% CI, Mann-Whitney p, +50% / -20%
    shares (pair_study.ret_stats); E0's 5 seeds get the same rows as the random-buy reference (signals
    of the same days are correlated, so a p value is read against the seeds, never alone); pooled
    return-% histogram. Descriptive.
Output: per basis base / tables / jobs, repro.json, accounts.json, agg/ (aggregates only), figures.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import account_study as ns, basis_study as bs, crsp, entry_study as es, event_study as ev, exclude_study as ms
from . import exit_study as xs, figures, goal_study as gs, large_study as ls, pair_study as pj, pick_study as ph
from . import regime_study as rs, stage_c_study as sc, tail_study as ts
from .crsp_study import load_panel_with_cash
from .engine import CANCELLED

# ---- registered (DECISIONS 99): fixed before any result
START, END = es.START, es.END  # 1993-02-01, 2023-12-31
ENTRIES = list(pj.RULES)  # E1-E9, V1-V6
GENS = ms.generators()  # the 15 entries + E0 (seeds 1-5)
SEEDS = GENS["E0"]
EXITS = ["X1", "H1"]
SETTINGS = list(rs.SETTINGS)
SETTING_NAMES = [ns.setting_name(s) for s in SETTINGS]
A1 = ["R1A1", "R3A1", "R4A1", "R5A1"]  # the A1 gates that beat placebos in 95
A2 = [f"{r}A2" for r in rs.REGIMES]
REUSED = ("E7", "E0")  # Stage R's raw tables
REPRO_TABLES = ("E2", "V3")  # (b)
TOL = 1e-9
K1_SYSTEMS, K1_CHANGED = 6, 2  # >= 6 of 8 systems; changed if >= 2 of 15 entries
K5_PP = 0.005  # changed if a band's median 2008-2023 change >= +0.5 pp
NEAR_PP = 0.005  # near-miss listing: fails only matched-return comparisons, by at most 0.5 pp
TRIALS = 74_174  # cumulative, unchanged (no trials added)
SEED = 99
_W: dict = {}


def log(m: str) -> None:
    ts.log(m)


@contextlib.contextmanager
def use_basis(basis: str):
    """The price basis for panels loaded here and in workers started inside (LAB_PRICE_BASIS)."""
    old_env, old = os.environ.get("LAB_PRICE_BASIS"), crsp.DEFAULT_BASIS
    os.environ["LAB_PRICE_BASIS"] = basis
    crsp.DEFAULT_BASIS = basis
    try:
        yield
    finally:
        crsp.DEFAULT_BASIS = old
        if old_env is None:
            os.environ.pop("LAB_PRICE_BASIS", None)
        else:
            os.environ["LAB_PRICE_BASIS"] = old_env


def cfg_of(bdir: Path, regimes: Path) -> dict:
    return {"bins": pj.BINS_J3, "seeds": pj.E0_SEEDS, "base": str(bdir / "base.parquet"), "regimes": str(regimes)}


def stored_dirs(a) -> dict:
    return {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events),
            "r0x1": Path(a.r0x1), "tables": Path(a.tables)}


def table_dirs(root: Path) -> dict:
    """exit_study.table_of's folders under one tables root (the stored layout)."""
    return {k: root / k for k in ("stage_h", "stage_i", "v2", "events", "r0x1", "tables")}


# --------------------------------------------------------------------------- signal tables

def table_spec(e: str, x: str) -> tuple[str, str, int, int]:
    """(folder under a tables root, file, entry cell, exit cell) of entry e under X1 / H1, as
    exit_study.table_of reads them (Stage C v2 / G tables for X1, Stage H's for H1)."""
    if x == "X1":
        root = "v2" if e.startswith("E") else "events"
        return f"{root}/signals/full", f"{e}__X1_signals.parquet", pj._C_CANON.get(e, 1), 1
    if x == "H1":
        return "stage_h/h1/signals/full", f"{e}__H1_signals.parquet", 0, xs.H_KI
    raise ValueError(x)


def table_in(dirs: dict, e: str, x: str) -> Path:
    """The file of entry e under exit x in a set of exit_study.table_of folders."""
    d, f, _, _ = table_spec(e, x)
    head, rest = d.split("/", 1)
    return dirs[head] / rest / f


def table_file(root: Path, e: str, x: str) -> Path:
    return table_in(table_dirs(root), e, x)


def table_menu(e: str, x: str) -> None:
    """Stage C's signals-only runner on entry e at its stored value and exit x at the stored cell's
    value (as basis_study.table_menu for E7)."""
    key, vals = ph._C_ENTRIES[e] if e in ph._C_ENTRIES else ev.EVENTS[e]
    sc.ENTRIES.clear()
    sc.ENTRIES[e] = (key, [vals[pj._C_CANON.get(e, 1)]])
    sc.ENTRY_CANON[e] = 0
    (blk, val), extra = bs.EXIT_DEF[x]
    sc.EXITS.clear()
    sc.EXITS[x] = (blk, [val])
    sc.EXIT_CANON[x] = 0
    sc.EXIT_EXTRA.pop(x, None)
    if extra:
        sc.EXIT_EXTRA[x] = dict(extra)
    sc.EXIT_BREAKEVEN.pop(x, None)


def _t_job(job):
    e, x = job
    t0 = time.time()
    table_menu(e, x)
    _, sig = sc._run_signals(("full", e, x))
    _, _, xi, yi = table_spec(e, x)
    sig = sig.assign(xi=xi, yi=yi).astype({"return": "float32", "holding_days": "int32", "xi": "int64", "yi": "int64"})
    return job, sig, time.time() - t0


def build_tables(root: Path, panel: str, eligible: str, basis: str, pairs: list, workers: int) -> None:
    todo = [p for p in pairs if not table_file(root, *p).exists()]
    if not todo:
        return
    log(f"[{basis}] 신호표 {len(todo)}개")
    with ProcessPoolExecutor(max_workers=min(workers, len(todo)), initializer=bs._t_init, initargs=(panel, eligible, basis)) as ex:
        for job, sig, sec in ex.map(_t_job, todo, chunksize=1):
            f = table_file(root, *job)
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            sig[["xi", "yi"] + sc.SIGNAL_KEEP].to_parquet(tmp, compression="zstd")
            tmp.replace(f)
            log(f"  {job[0]}×{job[1]}: 신호 {len(sig):,} · {sec:.0f}s")


def read_x1(dirs: dict, e: str) -> pd.DataFrame:
    """The X1 cell of entry e (or E0 seed 'R0s<k>') in a set of table folders, cancelled orders out,
    one row per symbol x signal day."""
    if e.startswith("R0s"):
        t = bs.read_cell(dirs["r0x1"] / "signals" / "full" / "R0__X1_signals.parquet", pj.E0_SEEDS.index(int(e[3:])), 0)
    else:
        _, _, xi, yi = table_spec(e, "X1")
        t = bs.read_cell(table_in(dirs, e, "X1"), xi, yi)
    return t[t["exit_reason"] != CANCELLED].drop_duplicates(["symbol", "signal_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- job lists (as the stages' main)

def p_jobs(gens: dict) -> list:
    """regime_study.main's job list (not dry)."""
    flt = list(rs.FILTERS)
    jobs = []
    for gen, srcs in gens.items():
        for x in rs.EXITS:
            for st in rs.SETTINGS:
                if st[2] == "P0":
                    jobs.append((gen, x, st, ["none"], pj.COST_BPS, srcs, "grid"))
                    jobs += [(gen, x, st, [f"{r}A1", f"{r}A2"], pj.COST_BPS, srcs, "grid") for r in rs.REGIMES]
                else:
                    jobs.append((gen, x, st, flt, pj.COST_BPS, srcs, "grid"))
        if gen != "E0":
            for st in rs.PLACEBO_SETTINGS:
                jobs += [(gen, rs.PLACEBO_EXIT, st, [r], pj.COST_BPS, srcs, "placebo") for r in rs.REGIMES]
    return [("P", j) for j in jobs]


def q_jobs(gens: dict) -> list:
    """large_study.main's job list (not dry)."""
    bands = list(ls.BANDS.items())
    narrow = [(b, ls.BANDS[b]) for b in ls.NARROW]
    jobs = []
    for gen, srcs in gens.items():
        for x in ls.EXITS:
            for st in ls.SETTINGS:
                if st[2] == "P0":
                    jobs += [(gen, x, st, [f], bands, pj.COST_BPS, srcs, "grid") for f in ls.FILTERS]
                else:
                    jobs.append((gen, x, st, list(ls.FILTERS), bands, pj.COST_BPS, srcs, "grid"))
        if gen != "E0":
            for st in ls.PSEUDO_SETTINGS:
                fs = list(ls.PSEUDO_FILTERS[st])
                if st[2] == "P0":
                    jobs += [(gen, ls.PSEUDO_EXIT, st, [f], narrow, pj.COST_BPS, srcs, "pseudo") for f in fs]
                else:
                    jobs.append((gen, ls.PSEUDO_EXIT, st, fs, narrow, pj.COST_BPS, srcs, "pseudo"))
    return [("Q", j) for j in jobs]


def o_jobs(gens: dict) -> list:
    return [("O", (gen, x, n, pj.COST_BPS, srcs, "grid")) for gen, srcs in gens.items() for x in EXITS for n in sorted({s[0] for s in SETTINGS})]


def u_jobs(gens: dict) -> list:
    return [("U", (s, x)) for srcs in gens.values() for s in srcs for x in EXITS]


def p_jobs40(g: pd.DataFrame, bench: dict, gens: dict) -> list:
    """regime_study.main's 40 bp jobs: the P1 setting x entry x gate groups with both exits passing."""
    cells, _ = rs.candidates(g, bench, {})
    p1 = {ns.setting_name(s) for s in rs.P1_SETTINGS}
    need = cells[(cells["gen"] != "E0") & cells["cell_pass"] & cells["setting"].isin(p1)]
    per = need.groupby(["setting", "gen", "filter"])["cell_pass"].sum()
    if not (per >= len(rs.EXITS)).any():
        return []
    by: dict = {}
    for gen, x, st, f in sorted({(gen, x, st, f) for (st, gen, f) in per[per >= len(rs.EXITS)].index for x in rs.EXITS}):
        by.setdefault((gen, x, st), []).append(f)
    return [("P", (gen, x, ns.parse_setting(st), fs, pj.CHECK_COST_BPS, gens[gen], "grid")) for (gen, x, st), fs in by.items()]


def q_jobs40(g: pd.DataFrame, bench: dict, gens: dict) -> list:
    """large_study.main's 40 bp jobs."""
    cells, _ = ls.candidates(g, bench, {})
    p1 = {ns.setting_name(s) for s in ls.P1_SETTINGS}
    need = cells[(cells["gen"] != "E0") & cells["cell_pass"] & cells["setting"].isin(p1)]
    per = need.groupby(["setting", "gen", "filter", "band"])["cell_pass"].sum()
    if not (per >= len(ls.EXITS)).any():
        return []
    by_gs: dict = {}
    for (st, gen, f, b) in per[per >= len(ls.EXITS)].index:
        for x in ls.EXITS:
            by_gs.setdefault((gen, x, st), {}).setdefault(f, set()).add(b)
    out = []
    for (gen, x, st), fb in by_gs.items():
        bset = sorted({b for v in fb.values() for b in v})
        out.append(("Q", (gen, x, ns.parse_setting(st), sorted(fb), [(b, ls.BANDS[b]) for b in bset], pj.CHECK_COST_BPS, gens[gen], "grid")))
    return out


def job_file(job) -> tuple[str, str]:
    kind, j = job
    if kind == "P":
        return "p", rs.job_name(j[0], j[1], j[2], j[3], j[4], j[6])
    if kind == "Q":
        return "q", ls._name(j)
    if kind == "O":
        return "o", gs.job_name(j[0], j[1], j[2], j[3], j[5])
    return "unl", f"unl__{j[0]}__{j[1]}"


def job_cost(job, size: pd.Series) -> float:
    """A rough work estimate (replays x signals x slots) to start the heavy jobs first."""
    kind, j = job
    if kind == "U":
        return 1.0
    if kind == "O":
        gen, x, n, _, srcs, _ = j
        return float(size.reindex(srcs).sum()) / len(srcs) * (600 + 4 * len(srcs)) * (1 + n / 50)
    if kind == "P":
        gen, x, st, filters, _, srcs, mode = j
        per = (rs.PLACEBO_ORDERS if mode == "placebo" else 100) if st[2] == "P0" else len(srcs)
        reps = per * (2 * rs.N_PLACEBO if mode == "placebo" else len(filters))
    else:
        gen, x, st, filters, bands, _, srcs, mode = j
        per = (ls.PSEUDO_ORDERS if mode == "pseudo" else 100) if st[2] == "P0" else len(srcs)
        reps = per * len(filters) * (len(bands) * (ls.N_PSEUDO if mode == "pseudo" else 1))
    return float(size.reindex(srcs).sum()) / len(srcs) * reps * (1 + st[0] / 50)


# --------------------------------------------------------------------------- workers

def _w_init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    ls._init(panel_dir, eligible, dirs, cfg)  # sets up regime_study, goal_study, account_study, exclude_study too


def _w_job(job):
    kind, j = job
    t0 = time.time()
    if kind == "P":
        _, acc, win, mom, tr, _sig, sec = rs.run_job(j)
        return job, {"win": win, "mom": mom, "trades": tr, "acc": acc}, sec
    if kind == "Q":
        _, acc, win, mom, tr, sec = ls.run_job(j)
        return job, {"win": win, "mom": mom, "trades": tr, "acc": acc}, sec
    if kind == "O":
        _, acc, win, mom, sec = gs.run_job(j)
        return job, {"win": win, "mom": mom, "acc": acc}, sec
    src, x = j
    row, eq = bs.unlimited(src, x)
    ms._W["prep"].clear()
    pj._W["tables"].clear()
    acc = pd.DataFrame([{**row, "gen": "E0" if src.startswith("R0") else src, "exit": x}])
    return job, {"curve": pd.DataFrame({"eq": eq}), "acc": acc}, time.time() - t0


def done(bdir: Path, job) -> bool:
    sub, name = job_file(job)
    return (bdir / sub / "jobs" / f"{name}_acc.csv").exists()


def run_pool(jobs: list, bdir: Path, a, eligible: str, dirs: dict, cfg: dict, basis: str) -> None:
    todo = [j for j in jobs if not done(bdir, j)]
    log(f"[{basis}] 계좌 작업 {len(todo)}개 남음 (전체 {len(jobs)})")
    if not todo:
        return
    size = pd.read_parquet(cfg["base"], columns=["src"]).groupby("src").size()
    todo.sort(key=lambda j: -job_cost(j, size))
    t0 = time.time()
    with use_basis(basis), ProcessPoolExecutor(max_workers=a.workers, initializer=_w_init, initargs=(a.panel, eligible, dirs, cfg)) as ex:
        futs = [ex.submit(_w_job, j) for j in todo]
        for i, f in enumerate(as_completed(futs)):
            job, parts, sec = f.result()
            sub, name = job_file(job)
            d = bdir / sub / "jobs"
            d.mkdir(parents=True, exist_ok=True)
            for k, frame in parts.items():
                if k in ("trades", "curve"):
                    if len(frame):
                        frame.to_parquet(d / f"{name}_{k}.parquet")
                elif k != "acc":
                    frame.to_csv(d / f"{name}_{k}.csv", index=False)
            parts["acc"].to_csv(d / f"{name}_acc.csv", index=False)  # last: marks the job done
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                log(f"  [{i + 1}/{len(todo)}] {name} {sec:.0f}s · 경과 {el / 60:.0f}분 · 남은 예상 {(len(todo) - i - 1) * el / (i + 1) / 60:.0f}분")


def read_jobs(d: Path, bp: bool = False, prefix: str = "") -> pd.DataFrame:
    files = [f for f in sorted(d.glob(f"{prefix}*_acc.csv")) if ("bp_" in f.name) == bp and f.stat().st_size > 2]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


# --------------------------------------------------------------------------- benchmarks (bundle E)

def benchmarks(a, eligible: str, dirs: dict, cfg: dict, states: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage P's benchmark table (SPY mix, SPY200, EW, EW x each regime) and Stage Q's (the first three
    plus the top-N EW of each band and that under each A1 regime), built as the stages' main."""
    with use_basis(basis):
        pj._init(a.panel, eligible, dirs, cfg)
        px, wdates, w0, cash = pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"]
        btab, _ = gs.benchmarks_table(px, wdates, w0, cash)
        sp = gs._setup(cash, wdates)
        ew = gs.bench_curves(px, wdates, w0, cash)["ew"]
        st_w = {r: states[r].reindex(wdates).fillna(False).to_numpy(dtype=bool) for r in rs.REGIMES}
        EQ = np.column_stack([rs.ew_regime(ew, cash, st_w[r], pj.COST_BPS / 10_000) for r in rs.REGIMES])
        R = EQ[1:] / EQ[:-1] - 1.0
        gm = gs.goal_metrics(R, np.repeat(sp["c"][:, None], R.shape[1], axis=1), sp["years"], sp["wins"])
        ewr = []
        for i, r in enumerate(rs.REGIMES):
            ewr.append({"benchmark": f"ew_{r}", "label": f"동일비중 × {r} {rs.REGIMES[r]} (A2) + 국채",
                        **{k: float(v[i]) for k, v in gm.items() if k != "win_cagr"}, **es.period_stats(pd.Series(EQ[:, i], index=wdates))})
        btab_p = pd.concat([btab, pd.DataFrame(ewr)], ignore_index=True)
        panel = load_panel_with_cash(Path(a.panel), basis)
        pc = types.SimpleNamespace(close=bs.cut(panel.close), extra={"cap_rank": bs.cut(panel.extra["cap_rank"])})
        del panel
        btab_b, _ = ls.band_benchmarks(pc, pd.read_parquet(eligible), wdates, w0, cash, states)
        btab_q = pd.concat([btab, btab_b], ignore_index=True)
    return btab_p, btab_q


def bench_from(btab: pd.DataFrame) -> tuple[dict, dict, dict]:
    """(SPY mix / SPY200 / EW, EW x regime by regime, top-N EW by (band, regime)) from a benchmark table."""
    t = btab.set_index("benchmark")
    bench = {k: t.loc[k].to_dict() for k in ("mix", "ma200", "ew")}
    ewr = {r: t.loc[f"ew_{r}"].to_dict() for r in rs.REGIMES if f"ew_{r}" in t.index}
    ewb = {}
    if "band" in btab:
        for _, r in btab[btab["benchmark"].astype(str).str.startswith("ew_U")].iterrows():
            ewb[(r["band"], r["regime"])] = r.to_dict()
    return bench, ewr, ewb


# --------------------------------------------------------------------------- comparisons

def compare(mine: pd.DataFrame, stored: pd.DataFrame, keys: list[str], skip=("orders",)) -> pd.DataFrame:
    """Per stored key and numeric column present in both: |mine - stored| (NaN = NaN is 0); a stored key
    missing here or a key only here gets diff inf."""
    m = mine.set_index(keys)
    s = stored.set_index(keys)
    if m.index.duplicated().any() or s.index.duplicated().any():
        raise ValueError("중복 키")
    cols = [c for c in s.columns if c in m.columns and c not in skip and pd.api.types.is_numeric_dtype(s[c])]
    common = s.index.intersection(m.index)
    a_ = m.loc[common, cols].to_numpy(dtype=float)
    b_ = s.loc[common, cols].to_numpy(dtype=float)
    d = np.where(np.isnan(a_) & np.isnan(b_), 0.0, np.abs(a_ - b_))
    d = np.where(np.isnan(d), np.inf, d)
    rows = [pd.DataFrame({"key": [" | ".join(map(str, k)) for k in common], "max_diff": d.max(axis=1) if len(cols) else 0.0,
                          "values": len(cols)})]
    miss = s.index.difference(m.index)
    extra = m.index.difference(s.index)
    for kind, ks_ in (("missing", miss), ("extra", extra)):
        if len(ks_):
            rows.append(pd.DataFrame({"key": [f"{kind}: " + " | ".join(map(str, k)) for k in ks_], "max_diff": np.inf, "values": 0}))
    return pd.concat(rows, ignore_index=True)


def summary_of(c: pd.DataFrame) -> dict:
    bad = c["max_diff"] > TOL
    return {"cells": int((~c["key"].str.startswith(("missing", "extra"))).sum()), "values": int(c["values"].sum()),
            "max_diff": float(c.loc[np.isfinite(c["max_diff"]), "max_diff"].max()) if np.isfinite(c["max_diff"]).any() else np.nan,
            "missing": int(c["key"].str.startswith("missing").sum()), "extra": int(c["key"].str.startswith("extra").sum()),
            "over_tol": int(bad.sum()), "pass": bool(len(c) and not bad.any())}


P_KEYS = ["setting", "gen", "exit", "filter", "cost", "mode"]
Q_KEYS = ["setting", "gen", "exit", "filter", "band", "cost", "mode"]


# --------------------------------------------------------------------------- stages

def regimes_file(a) -> Path:
    return Path(a.out) / "regimes.parquet"


def stage_repro(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    sp_, sq_ = Path(a.stage_p), Path(a.stage_q)
    rp, rq = pd.read_parquet(sp_ / "regimes.parquet"), pd.read_parquet(sq_ / "regimes.parquet")
    res["regimes_equal"] = bool(rp.equals(rq))
    if not regimes_file(a).exists():
        shutil.copy2(sq_ / "regimes.parquet", regimes_file(a))
    # (b) tables first (cheap)
    log("재현 (b): 수정주가 기준 E2 · V3 × X1 · H1 신호표 vs 저장본")
    root = out / "adjusted" / "tables_repro"
    build_tables(root, a.panel, a.stored_eligible, "adjusted", [(e, x) for e in REPRO_TABLES for x in EXITS], a.table_workers)
    tb = []
    for e in REPRO_TABLES:
        for x in EXITS:
            _, _, xi, yi = table_spec(e, x)
            mine = bs.read_cell(table_file(root, e, x), xi, yi)
            theirs = bs.read_cell(table_in(stored_dirs(a), e, x), xi, yi)
            tb.append({"table": f"{e}__{x}", **bs.same_rows(mine, theirs)})
    tb = pd.DataFrame(tb)
    tb.to_csv(out / "repro_tables.csv", index=False)
    res["b_tables"] = tb.to_dict("records")
    log("\n" + tb.to_string(index=False))
    # (c) raw eligible and Stage R's E7 / E0 accounts
    log("재현 (c): 실제 가격 기준 매수 가능 목록과 E7 · E0 계좌 vs Stage R")
    rdir = out / "raw"
    rdir.mkdir(parents=True, exist_ok=True)
    elig_raw = rdir / "eligible_daily.parquet"
    if not elig_raw.exists():
        with use_basis("raw"):
            panel = load_panel_with_cash(Path(a.panel), "raw")
            if "raw_close" not in panel.extra:
                raise SystemExit("패널에 실제 가격 계열이 없습니다.")
            ok, _ = bs.eligible_parts(panel)
            del panel
        bs.cut(ok).to_parquet(elig_raw)
    stage_r = Path(a.stage_r)
    res["c_eligible"] = bs.compare_frames(pd.read_parquet(elig_raw), pd.read_parquet(stage_r / "raw" / "eligible_daily.parquet"))
    log(f"  매수 가능 목록: {res['c_eligible']}")
    cdir = out / "repro_c"
    cb = cdir / "raw"
    cb.mkdir(parents=True, exist_ok=True)
    if not (cb / "eligible_daily.parquet").exists():
        shutil.copy2(elig_raw, cb / "eligible_daily.parquet")
    if not (cb / "tables").exists():
        shutil.copytree(stage_r / "raw" / "tables", cb / "tables")
    A = types.SimpleNamespace(out=str(cdir), panel=a.panel, regimes=str(regimes_file(a)), workers=a.workers, feature_workers=a.feature_workers)
    with use_basis("raw"):
        bs.run_accounts(A, "raw")
    g_r = pd.read_csv(stage_r / "raw" / "goal.csv")
    g_c = pd.read_csv(cb / "goal.csv")
    c_goal = compare(g_c, g_r, Q_KEYS)
    c_goal.to_csv(out / "repro_c_goal.csv", index=False)
    res["c_goal"] = summary_of(c_goal)
    res["c_goal_20bp_cells"] = int(len(g_r[g_r["cost"] == pj.COST_BPS]))
    u_c = compare(pd.read_csv(cb / "unlimited.csv"), pd.read_csv(stage_r / "raw" / "unlimited.csv"), ["src", "exit"])
    res["c_unlimited"] = summary_of(u_c)
    log(f"  계좌: {res['c_goal']} · 제한 없음: {res['c_unlimited']}")
    # (a) Stage P and Q on the stored tables through this runner
    log("재현 (a): 저장된 Stage L 표(수정주가) → Stage P · Q 격자 전체 vs 결과표")
    adir = out / "adjusted"
    dirs = stored_dirs(a)
    cfg = cfg_of(adir, regimes_file(a))
    base_a = build_base(a, adir, a.stored_eligible, dirs, cfg, "adjusted")
    sb = pd.read_parquet(sq_ / "base.parquet")
    res["a_base"] = summary_of(compare(base_a, sb, ["src", "si", "sj"]))
    log(f"  기준표 vs Stage Q 기준표: {res['a_base']}")
    run_pool(p_jobs(GENS) + q_jobs(GENS), adir, a, a.stored_eligible, dirs, cfg, "adjusted")
    btab_p, btab_q = benchmarks(a, a.stored_eligible, dirs, cfg, pd.read_parquet(regimes_file(a)), "adjusted")
    btab_p.to_csv(adir / "benchmarks_p.csv", index=False)
    btab_q.to_csv(adir / "benchmarks_q.csv", index=False)
    bench, _, _ = bench_from(btab_p)
    gP = rs.summarize(read_jobs(adir / "p" / "jobs"))
    gQ = ls.summarize(read_jobs(adir / "q" / "jobs"))
    j40 = p_jobs40(gP, bench, GENS) + q_jobs40(gQ, bench, GENS)
    run_pool(j40, adir, a, a.stored_eligible, dirs, cfg, "adjusted")
    gP.to_csv(adir / "p" / "goal.csv", index=False)
    gQ.to_csv(adir / "q" / "goal.csv", index=False)
    gP40 = rs.summarize(read_jobs(adir / "p" / "jobs", bp=True))
    gQ40 = ls.summarize(read_jobs(adir / "q" / "jobs", bp=True))
    sP40 = rs.summarize(read_jobs(sp_ / "jobs", bp=True))
    sQ40 = ls.summarize(read_jobs(sq_ / "jobs", bp=True))
    checks = {"a_p_goal": compare(gP, pd.read_csv(sp_ / "goal.csv"), P_KEYS), "a_q_goal": compare(gQ, pd.read_csv(sq_ / "goal.csv"), Q_KEYS),
              "a_p_40bp": compare(gP40, sP40, P_KEYS), "a_q_40bp": compare(gQ40, sQ40, Q_KEYS),
              "a_p_benchmarks": compare(btab_p, pd.read_csv(sp_ / "benchmarks.csv"), ["benchmark"]),
              "a_q_benchmarks": compare(btab_q, pd.read_csv(sq_ / "benchmarks.csv"), ["benchmark"])}
    for k, c in checks.items():
        c.to_csv(out / f"repro_{k}.csv.gz", index=False)
        res[k] = summary_of(c)
        log(f"  {k}: {res[k]}")
    res["pass"] = bool(res["regimes_equal"] and tb["equal"].all() and res["c_eligible"]["differ"] == 0 and res["c_goal"]["pass"]
                       and res["c_unlimited"]["pass"] and all(res[k]["pass"] for k in checks))
    (out / "repro.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"재현 판정: {'통과' if res['pass'] else '실패 — 뒤 단계는 돌지 않습니다'}")
    return res


def build_base(a, bdir: Path, eligible: str, dirs: dict, cfg: dict, basis: str) -> pd.DataFrame:
    A = types.SimpleNamespace(out=str(bdir), panel=a.panel, eligible=eligible, dirs=dirs, cfg=cfg, feature_workers=a.feature_workers)
    bdir.mkdir(parents=True, exist_ok=True)
    saved = list(ns.EXITS)
    ns.EXITS[:] = list(EXITS)
    try:
        with use_basis(basis):
            base = ns.build_base(A, GENS)
    finally:
        ns.EXITS[:] = saved
    log(f"[{basis}] 기준표: 신호 {len(base):,}개, M1 통과 {(~ms.excluded(base, ns.RULE)).mean():.1%}")
    return base


def need_repro(a) -> dict:
    f = Path(a.out) / "repro.json"
    rp = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if not rp.get("pass"):
        raise SystemExit("재현 점검을 통과하지 않아 이 단계는 돌지 않습니다 (repro.json).")
    return rp


def stage_tables(a) -> None:
    need_repro(a)
    out = Path(a.out)
    root = out / "raw" / "tables"
    stage_r = Path(a.stage_r) / "raw" / "tables"
    for rel in ("v2/signals/full/E7__X1_signals.parquet", "stage_h/h1/signals/full/E7__H1_signals.parquet",
                "r0x1/signals/full/R0__X1_signals.parquet", "stage_i/r0/signals/full/R0__H1_signals.parquet"):
        f = root / rel
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage_r / rel, f)
    pairs = [(e, x) for e in ENTRIES if e not in REUSED for x in EXITS]
    build_tables(root, a.panel, str(out / "raw" / "eligible_daily.parquet"), "raw", pairs, a.table_workers)
    missing = [p for p in [(e, x) for e in ENTRIES for x in EXITS] if not table_file(root, *p).exists()]
    if missing:
        raise SystemExit(f"신호표가 없습니다: {missing}")


def stage_accounts(a) -> dict:
    need_repro(a)
    out = Path(a.out)
    rdir = out / "raw"
    elig = str(rdir / "eligible_daily.parquet")
    dirs = table_dirs(rdir / "tables")
    cfg = cfg_of(rdir, regimes_file(a))
    build_base(a, rdir, elig, dirs, cfg, "raw")
    run_pool(p_jobs(GENS) + q_jobs(GENS) + o_jobs(GENS) + u_jobs(GENS), rdir, a, elig, dirs, cfg, "raw")
    states = pd.read_parquet(regimes_file(a))
    if not (rdir / "benchmarks_q.csv").exists():
        btab_p, btab_q = benchmarks(a, elig, dirs, cfg, states, "raw")
        btab_p.to_csv(rdir / "benchmarks_p.csv", index=False)
        btab_q.to_csv(rdir / "benchmarks_q.csv", index=False)
    bench, _, _ = bench_from(pd.read_csv(rdir / "benchmarks_p.csv"))
    gP = rs.summarize(read_jobs(rdir / "p" / "jobs"))
    gQ = ls.summarize(read_jobs(rdir / "q" / "jobs"))
    run_pool(p_jobs40(gP, bench, GENS) + q_jobs40(gQ, bench, GENS), rdir, a, elig, dirs, cfg, "raw")
    gP.to_csv(rdir / "p" / "goal.csv", index=False)
    gQ.to_csv(rdir / "q" / "goal.csv", index=False)
    rs.summarize(read_jobs(rdir / "p" / "jobs", bp=True)).to_csv(rdir / "p" / "goal40.csv", index=False)
    ls.summarize(read_jobs(rdir / "q" / "jobs", bp=True)).to_csv(rdir / "q" / "goal40.csv", index=False)
    gs.summarize(read_jobs(rdir / "o" / "jobs")).to_csv(rdir / "o" / "goal.csv", index=False)
    read_jobs(rdir / "unl" / "jobs").to_csv(rdir / "unlimited.csv", index=False)
    # adjusted unlimited accounts on the stored tables (bundle D's adjusted side)
    adir = out / "adjusted"
    run_pool(u_jobs(GENS), adir, a, a.stored_eligible, stored_dirs(a), cfg_of(adir, regimes_file(a)), "adjusted")
    read_jobs(adir / "unl" / "jobs").to_csv(adir / "unlimited.csv", index=False)
    # the full raw run's E7 / E0 cells against Stage R
    g_r = pd.read_csv(Path(a.stage_r) / "raw" / "goal.csv")
    mine = pd.concat([gQ, pd.read_csv(rdir / "q" / "goal40.csv")], ignore_index=True)
    mine = mine[mine["gen"].isin(REUSED) & mine["filter"].isin(bs.FILTERS) & (mine["mode"] == "grid")]
    c = compare(mine, g_r, Q_KEYS)
    # Stage R ran 40 bp for every E7 / E0 cell; here 40 bp runs only where the stage rule asks: 20 bp
    # must match all, 40 bp the cells that exist
    c20 = c[~c["key"].str.contains(r"\| 40\.0 \|", regex=True)]
    c40 = c[c["key"].str.contains(r"\| 40\.0 \|", regex=True) & ~c["key"].str.startswith("missing")]
    res = {"stage_r_20bp": summary_of(c20), "stage_r_40bp_present": summary_of(c40) if len(c40) else {"cells": 0}}
    # bundle C's 'rule' rows against the P path's no-gate cells
    gO = pd.read_csv(rdir / "o" / "goal.csv")
    o_rule = gO[(gO["kind"] == "rule") & gO["setting"].isin(SETTING_NAMES)].assign(filter="none").drop(columns=["kind"])
    p_none = gP[(gP["filter"] == "none") & (gP["mode"] == "grid")]
    cols = ["setting", "gen", "exit", "filter", "cost"] + [c_ for c_ in gs.GOAL_COLS + ["cagr_1993-2023", "mdd_1993-2023", "trades"]]
    res["o_rule_vs_p_none"] = summary_of(compare(o_rule[cols], p_none[cols], ["setting", "gen", "exit", "filter", "cost"]))
    (out / "accounts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"실제 가격 기준 E7 · E0 vs Stage R: {res['stage_r_20bp']} · 40bp {res['stage_r_40bp_present']} · "
        f"묶음 C 'rule' vs P 필터 없음: {res['o_rule_vs_p_none']}")
    return res


# --------------------------------------------------------------------------- verdicts

def load_basis(a, basis: str) -> dict:
    """The tables the verdicts read: adjusted = the stored Stage O / P / Q results (and this run's
    unlimited accounts on the stored tables), raw = this run."""
    out = Path(a.out)
    if basis == "adjusted":
        sp_, sq_ = Path(a.stage_p), Path(a.stage_q)
        go = pd.read_csv(Path(a.stage_o) / "goal.csv.gz")
        return {"gP": pd.read_csv(sp_ / "goal.csv"), "gP40": rs.summarize(read_jobs(sp_ / "jobs", bp=True)),
                "accP": read_jobs(sp_ / "jobs", prefix="E0__"), "btabP": pd.read_csv(sp_ / "benchmarks.csv"),
                "gQ": pd.read_csv(sq_ / "goal.csv"), "gQ40": ls.summarize(read_jobs(sq_ / "jobs", bp=True)),
                "accQ": read_jobs(sq_ / "jobs", prefix="E0__"), "btabQ": pd.read_csv(sq_ / "benchmarks.csv"),
                "gO": go[go["mode"] == "grid"], "unl": pd.read_csv(out / "adjusted" / "unlimited.csv")}
    r = out / "raw"
    return {"gP": pd.read_csv(r / "p" / "goal.csv"), "gP40": pd.read_csv(r / "p" / "goal40.csv"),
            "accP": read_jobs(r / "p" / "jobs", prefix="E0__"), "btabP": pd.read_csv(r / "benchmarks_p.csv"),
            "gQ": pd.read_csv(r / "q" / "goal.csv"), "gQ40": pd.read_csv(r / "q" / "goal40.csv"),
            "accQ": read_jobs(r / "q" / "jobs", prefix="E0__"), "btabQ": pd.read_csv(r / "benchmarks_q.csv"),
            "gO": pd.read_csv(r / "o" / "goal.csv"), "unl": pd.read_csv(r / "unlimited.csv")}


def k1_table(gP: pd.DataFrame, unl: pd.DataFrame, accP: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K1 per entry (and per E0 seed against the other four): (a) unlimited CAGR above every E0 seed for
    both exits in both halves, (b) dd_cagr above E0 in the three judged periods in >= 6 of the 8
    no-gate U3000 systems."""
    none = gP[(gP["mode"] == "grid") & (gP["cost"] == pj.COST_BPS) & (gP["filter"] == "none")].set_index(["setting", "gen", "exit"])
    u = unl.set_index(["src", "exit"])
    systems = [(st, x) for st in SETTING_NAMES for x in EXITS]

    def part_a(name: str, refs: list[str]) -> tuple[bool, float]:
        gaps = [u.loc[(name, x), f"cagr_{h}"] - max(u.loc[(s, x), f"cagr_{h}"] for s in refs) for x in EXITS for h in es.HALVES]
        return bool(min(gaps) > 0), float(min(gaps))

    rows = []
    for e in ENTRIES:
        ok_a, gap_a = part_a(e, SEEDS)
        wins = [all(none.loc[(st, e, x), f"dd_cagr_{p}"] > none.loc[(st, "E0", x), f"dd_cagr_{p}"] for p in gs.JUDGED) for st, x in systems]
        d93 = [none.loc[(st, e, x), "dd_cagr_1993-2023"] - none.loc[(st, "E0", x), "dd_cagr_1993-2023"] for st, x in systems]
        rows.append({"entry": e, "a_pass": ok_a, "a_min_gap": gap_a, "b_systems": int(sum(wins)), "b_pass": sum(wins) >= K1_SYSTEMS,
                     "b_median_diff_1993-2023": float(np.median(d93)),
                     **{f"unl_cagr_{x}_{h}": float(u.loc[(e, x), f"cagr_{h}"]) for x in EXITS for h in ("1993-2023",) + es.HALVES},
                     "above_random": bool(ok_a and sum(wins) >= K1_SYSTEMS)})
    e0 = accP[(accP["gen"] == "E0") & (accP["mode"] == "grid") & (accP["cost"] == pj.COST_BPS) & (accP["filter"] == "none")]
    per = e0.groupby(["setting", "exit", "src"])[[f"dd_cagr_{p}" for p in gs.JUDGED]].median()
    seeds = []
    for s in SEEDS:
        others = [o for o in SEEDS if o != s]
        ok_a, gap_a = part_a(s, others)
        wins, d93 = [], []
        for st, x in systems:
            mine = per.loc[(st, x, s)]
            ref = e0[(e0["setting"] == st) & (e0["exit"] == x) & e0["src"].isin(others)][[f"dd_cagr_{p}" for p in gs.JUDGED]].median()
            wins.append(all(mine[f"dd_cagr_{p}"] > ref[f"dd_cagr_{p}"] for p in gs.JUDGED))
            d93.append(mine["dd_cagr_1993-2023"] - ref["dd_cagr_1993-2023"])
        seeds.append({"seed": s, "a_pass": ok_a, "a_min_gap": gap_a, "b_systems": int(sum(wins)), "b_pass": sum(wins) >= K1_SYSTEMS,
                      "b_median_diff_1993-2023": float(np.median(d93)), "above_random": bool(ok_a and sum(wins) >= K1_SYSTEMS)})
    return pd.DataFrame(rows), pd.DataFrame(seeds)


def k5_table(sysQ: pd.DataFrame) -> pd.DataFrame:
    """Per narrowed band x system: dd_cagr change from U3000 (2008-2023, 1993-2007, 1993-2023)."""
    k = ["setting", "exit", "filter"]
    u = sysQ[sysQ["band"] == "U3000"].set_index(k)
    rows = []
    for b in ls.NARROW:
        s = sysQ[sysQ["band"] == b].set_index(k)
        for key in s.index:
            rows.append({"band": b, **dict(zip(k, key)), **{f"d_{p}": float(s.loc[key, f"dd_cagr_{p}"] - u.loc[key, f"dd_cagr_{p}"])
                                                           for p in ("2008-2023", "1993-2007", "1993-2023")}})
    return pd.DataFrame(rows)


def shortfall(row, bench: dict) -> dict:
    """How far a cell is from the per-cell test: the largest (benchmark - cell) over the judged
    periods and gated benchmarks (>= 0 fails), and whether the window-share and trade conditions hold."""
    gaps = {f"{b}_{p}": bench[b][f"dd_cagr_{p}"] - row[f"dd_cagr_{p}"] for p in gs.JUDGED for b in gs.GATED}
    k = max(gaps, key=gaps.get)
    return {"shortfall_pp": max(gaps[k], 0.0) * 100, "worst": k, "ws_ok": bool(all(row[f"ws_{b}"] >= gs.MIN_WINDOW_SHARE for b in gs.GATED)),
            "trades_ok": bool(row["trades"] >= pj.MIN_TRADES)}


def candidate_lists(cells: pd.DataFrame, cand: pd.DataFrame, chance: pd.DataFrame, bench: dict, keys: list[str], scope: str) -> dict:
    """Passing cells, groups with at least one passing cell (and what failed) and near misses."""
    c = cells[(cells["gen"] != "E0") & (cells["cost"] == pj.COST_BPS)].copy()
    c["p1_setting"] = c["setting"].isin({ns.setting_name(s) for s in rs.P1_SETTINGS})  # only P1 cells can make a candidate
    sf = pd.DataFrame([shortfall(r, bench) for _, r in c.iterrows()], index=c.index)
    c = pd.concat([c, sf], axis=1)
    ck = ["setting", "exit", "filter"] + (["band"] if "band" in keys else [])
    seeds = chance.set_index(ck)[["seeds_passing", "p0"]] if len(chance) else None
    if seeds is not None:
        c = c.join(seeds, on=ck)
    show = ["setting", "gen", "exit", "filter"] + (["band"] if "band" in keys else []) + \
        ["p1_setting", "cell_pass", "pass_40bp", "beats_e0", "shortfall_pp", "worst", "ws_ok", "trades_ok", "trades", "w", "ws_mix", "ws_ma200",
         "seeds_passing", "p0"] + [f"dd_cagr_{p}" for p in gs.GOAL_PERIODS]
    passing = c[c["cell_pass"]][show].assign(scope=scope)
    gk = ["setting", "gen", "filter"] + (["band"] if "band" in keys else [])
    groups = cand[cand["n20"] >= 1].copy() if len(cand) else cand
    if len(groups):
        info = []
        for _, g_ in groups.iterrows():
            sel = np.ones(len(c), dtype=bool)
            for k in gk:
                sel &= (c[k] == g_[k]).to_numpy()
            cc = c[sel]
            fails = cc[~cc["cell_pass"]]
            info.append({"failing_exits": " ".join(fails["exit"]), "failing_shortfall_pp": float(fails["shortfall_pp"].max()) if len(fails) else 0.0,
                         "failing_worst": " ".join(fails["worst"]) if len(fails) else "",
                         "failing_ws_or_trades": bool(len(fails) and not (fails["ws_ok"] & fails["trades_ok"]).all())})
        groups = pd.concat([groups.reset_index(drop=True), pd.DataFrame(info)], axis=1).assign(scope=scope)
    near = c[~c["cell_pass"] & c["ws_ok"] & c["trades_ok"] & (c["shortfall_pp"] <= NEAR_PP * 100)][show].assign(scope=scope)
    return {"passing": passing, "groups": groups, "near": near}


def judge(d: dict) -> dict:
    """Every registered verdict and reported table on one basis's tables."""
    bench, ewr, _ = bench_from(d["btabP"])
    benchQ, _, ewb = bench_from(d["btabQ"])
    gP, gQ = d["gP"], d["gQ"]
    out: dict = {}
    sysP = rs.systems(gP, bench, ewr)
    psP = rs.placebo_systems(gP)
    helpsP, cntP = rs.q1(sysP, psP)
    pass40P = {(r["setting"], r["gen"], r["exit"], r["filter"]): gs.cell_pass(r, bench) for _, r in d["gP40"].iterrows()}
    cellsP, candP = rs.candidates(gP, bench, pass40P)
    chP = rs.chance(d["accP"], cellsP, bench)
    sysQ = ls.systems(gQ, benchQ, ewb)
    psQ = ls.pseudo_systems(gQ)
    helpsQ, cntQ = ls.q1(sysQ, psQ)
    pass40Q = {(r["setting"], r["gen"], r["exit"], r["filter"], r["band"]): gs.cell_pass(r, benchQ) for _, r in d["gQ40"].iterrows()}
    cellsQ, candQ = ls.candidates(gQ, benchQ, pass40Q)
    chQ = ls.chance(d["accQ"], cellsQ, benchQ)
    out.update(sysP=sysP, psP=psP, helpsP=helpsP, cntP=cntP, cellsP=cellsP, candP=candP, chP=chP, sysQ=sysQ, psQ=psQ, helpsQ=helpsQ,
               cntQ=cntQ, cellsQ=cellsQ, candQ=candQ, chQ=chQ, bench=bench, ewr=ewr, ewb=ewb, gP=gP, gQ=gQ)
    # K1
    k1, k1s = k1_table(gP, d["unl"], d["accP"])
    out.update(k1=k1, k1_seeds=k1s)
    # K2 (+ bundle C: the same 8 without M1)
    nP = sysP[sysP["filter"] == "none"]
    sysO = gs.systems(d["gO"][d["gO"]["setting"].isin(SETTING_NAMES)], bench)
    m1 = []
    for st in SETTING_NAMES:
        for x in EXITS:
            r_ = sysO[(sysO["setting"] == st) & (sysO["exit"] == x) & (sysO["kind"] == "rule")]
            n_ = sysO[(sysO["setting"] == st) & (sysO["exit"] == x) & (sysO["kind"] == "none")]
            if len(r_) and len(n_):
                r_, n_ = r_.iloc[0], n_.iloc[0]
                m1.append({"setting": st, "exit": x, **{f"m1_{p}": r_[f"dd_cagr_{p}"] - n_[f"dd_cagr_{p}"] for p in gs.GOAL_PERIODS},
                           **{f"none_dd_cagr_{p}": n_[f"dd_cagr_{p}"] for p in gs.GOAL_PERIODS}, "none_w": n_["w"],
                           "none_system_pass": bool(n_["system_pass"]), "rule_system_pass": bool(r_["system_pass"])})
    out["m1"] = pd.DataFrame(m1)
    out["sysO"] = sysO
    # K3
    k3 = cntP.set_index("filter")
    # K4
    # K5
    k5 = k5_table(sysQ)
    out["k5_rows"] = k5
    k5m = {b: float(k5.loc[k5["band"] == b, "d_2008-2023"].median()) for b in ls.NARROW}
    out["K"] = {
        "K1": {"above_random": int(k1["above_random"].sum()), "entries": k1.loc[k1["above_random"], "entry"].tolist(),
               "a_pass": int(k1["a_pass"].sum()), "b_pass": int(k1["b_pass"].sum()),
               "seeds_above_random": int(k1s["above_random"].sum()), "seeds_a": int(k1s["a_pass"].sum()), "seeds_b": int(k1s["b_pass"].sum()),
               "changed": bool(k1["above_random"].sum() >= K1_CHANGED)},
        "K2": {"systems": len(nP), "passing": int(nP["system_pass"].sum()),
               "best": float(nP["dd_cagr_1993-2023"].max()), "median": float(nP["dd_cagr_1993-2023"].median()),
               "median_2008-2023": float(nP["dd_cagr_2008-2023"].median()),
               "no_m1_passing": int(out["m1"]["none_system_pass"].sum()) if len(out["m1"]) else None,
               "changed": bool(nP["system_pass"].sum() >= 1)},
        "K3": {"counts": {f: {"helps": int(k3.loc[f, "helps"]), "q95": int(k3.loc[f, "q95"]), "p0": float(k3.loc[f, "p0"]),
                              "beyond_chance": bool(k3.loc[f, "beyond_chance"])} for f in k3.index},
               "a1_fell_to_chance": [f for f in A1 if not bool(k3.loc[f, "beyond_chance"])],
               "a2_beyond_chance": [f for f in A2 if bool(k3.loc[f, "beyond_chance"])]},
        "K4": {"systems": len(sysQ), "passing": int(sysQ["system_pass"].sum()),
               "passing_list": sysQ.loc[sysQ["system_pass"], ["setting", "exit", "filter", "band"]].to_dict("records"),
               "changed": bool(sysQ["system_pass"].sum() >= 1)},
        "K5": {"median_d2008": k5m, "systems_per_band": {b: int((k5["band"] == b).sum()) for b in ls.NARROW},
               "helps": {f"{r['band']} {r['filter']}": {"helps": int(r["helps"]), "q95": int(r["q95"]), "beyond_chance": bool(r["beyond_chance"])}
                         for _, r in cntQ.iterrows()},
               "helps_total": int(cntQ["helps"].sum()), "changed": bool(any(v >= K5_PP for v in k5m.values()))},
    }
    out["K"]["K3"]["changed"] = bool(out["K"]["K3"]["a1_fell_to_chance"] or out["K"]["K3"]["a2_beyond_chance"])
    # reported
    ga = sysP[sysP["filter"].isin(A1)]
    out["reported"] = {
        "beats_ew_band": int(sysQ["beats_ew_band"].sum()), "systems_q": len(sysQ),
        "beats_ew_regime_a1": int(ga["beats_ew_regime"].sum()), "systems_a1": len(ga),
        "p_gated_passing": int(sysP.loc[sysP["filter"] != "none", "system_pass"].sum()),
        "p_cells_passing": int(cellsP.loc[cellsP["gen"] != "E0", "cell_pass"].sum()),
        "q_cells_passing": {b: int(cellsQ.loc[(cellsQ["gen"] != "E0") & (cellsQ["band"] == b), "cell_pass"].sum()) for b in ls.NARROW},
        "p_candidates": candP.loc[candP["candidate"], ["setting", "gen", "filter", "entry_specific"]].to_dict("records"),
        "q_candidates": candQ.loc[candQ["candidate"], ["setting", "gen", "filter", "band", "entry_specific"]].to_dict("records") if len(candQ) else [],
        "e0_seed_pass_p": {"passing": int(chP["seeds_passing"].sum()), "of": int(5 * len(chP))},
        "e0_seed_pass_q": {"passing": int(chQ["seeds_passing"].sum()), "of": int(5 * len(chQ))},
        "benchmarks": {k: {c: bench[k][c] for c in ("w", "dd_cagr_1993-2023", "dd_cagr_1993-2007", "dd_cagr_2008-2023", "dd_cagr_2017-2023",
                                                     "cagr_1993-2023", "cagr_1993-2007", "cagr_2008-2023", "mdd_1993-2023")} for k in bench},
    }
    # P1 effect, 2008-2023 gap
    p1 = []
    for x in EXITS:
        for f in ls.FILTERS:
            for b in ls.BANDS:
                s = sysQ[(sysQ["exit"] == x) & (sysQ["filter"] == f) & (sysQ["band"] == b)].set_index("setting")
                if {"N10_W2_P1", "N10_W2_P0"} <= set(s.index):
                    p1.append({"exit": x, "filter": f, "band": b, **{f"p1_{p}": s.loc["N10_W2_P1", f"dd_cagr_{p}"] - s.loc["N10_W2_P0", f"dd_cagr_{p}"]
                                                                       for p in gs.GOAL_PERIODS}})
    out["p1"] = pd.DataFrame(p1)
    gap = sysQ[sysQ["filter"].isin(A1)][["setting", "exit", "filter", "band", "dd_cagr_2008-2023", "beta", "cagr_2017-2023", "w"]].copy()
    gap["gap_vs_ma200"] = gap["dd_cagr_2008-2023"] - bench["ma200"]["dd_cagr_2008-2023"]
    out["gap"] = gap
    out["candidates"] = {"P": candidate_lists(cellsP[cellsP["filter"] != "none"], candP, chP, bench, P_KEYS, "U3000 (결정 95)"),
                         "Q": candidate_lists(cellsQ, candQ, chQ, benchQ, Q_KEYS, "U1000 · U500 (결정 96)")}
    return out


def k_labels(K: dict) -> dict:
    return {k: ("바뀜" if v["changed"] else "유지") for k, v in K.items()}


# --------------------------------------------------------------------------- bundle F (per trade)

def trade_stats(a) -> tuple[pd.DataFrame, dict]:
    """X1 rows of each entry and each E0 seed: new on raw / common / adjusted-only."""
    adj_dirs, raw_dirs = stored_dirs(a), table_dirs(Path(a.out) / "raw" / "tables")
    rows, pooled = [], {"new": [], "common": [], "adj_only": []}
    for i, name in enumerate(ENTRIES + SEEDS):
        ta, tr = read_x1(adj_dirs, name), read_x1(raw_dirs, name)
        sp = bs.split_signals({"raw": tr.assign(seed=-1), "adjusted": ta.assign(seed=-1)})
        rng = np.random.default_rng([SEED, i])
        ref = sp["common"][np.isfinite(sp["common"])]
        rb = pj.boot_means(ref, rng)
        if not name.startswith("R0s"):
            for k in pooled:
                pooled[k].append(sp[k][np.isfinite(sp[k])])
        for grp in ("new", "adj_only"):
            r = sp[grp][np.isfinite(sp[grp])]
            row = {"entry": "E0" if name.startswith("R0s") else name, "seed": name if name.startswith("R0s") else "", "group": grp,
                   "rows_raw": len(tr), "rows_adjusted": len(ta), "n": len(r), "ref_n": len(ref), "common_max_diff": sp["common_max_diff"],
                   "mean": float(r.mean()) if len(r) else np.nan, "median": float(np.median(r)) if len(r) else np.nan,
                   "big50": float((r >= 0.5).mean()) if len(r) else np.nan, "loss20": float((r <= -0.2).mean()) if len(r) else np.nan,
                   "ref_mean": float(ref.mean()), "ref_median": float(np.median(ref)), "ref_big50": float((ref >= 0.5).mean()),
                   "ref_loss20": float((ref <= -0.2).mean())}
            row.update(pj.ret_stats(r, ref, rb, rng))
            rows.append(row)
    pool = {k: np.concatenate(v) for k, v in pooled.items()}
    rng = np.random.default_rng([SEED, 999])
    rb = pj.boot_means(pool["common"], rng)
    for grp in ("new", "adj_only"):
        r = pool[grp]
        row = {"entry": "15개 합", "seed": "", "group": grp, "n": len(r), "ref_n": len(pool["common"]), "mean": float(r.mean()),
               "median": float(np.median(r)), "big50": float((r >= 0.5).mean()), "loss20": float((r <= -0.2).mean()),
               "ref_mean": float(pool["common"].mean()), "ref_median": float(np.median(pool["common"])),
               "ref_big50": float((pool["common"] >= 0.5).mean()), "ref_loss20": float((pool["common"] <= -0.2).mean())}
        row.update(pj.ret_stats(r, pool["common"], rb, rng))
        rows.append(row)
    return pd.DataFrame(rows), pool


# --------------------------------------------------------------------------- report

def stage_report(a) -> dict:
    rp = need_repro(a)
    out = Path(a.out)
    acc_chk = json.loads((out / "accounts.json").read_text(encoding="utf-8"))
    if not acc_chk["stage_r_20bp"]["pass"]:
        raise SystemExit("실제 가격 기준 E7 · E0 칸이 Stage R과 다릅니다 (accounts.json).")
    agg = out / "agg"
    (agg / "figures").mkdir(parents=True, exist_ok=True)
    J = {b: judge(load_basis(a, b)) for b in ("adjusted", "raw")}
    verdict = {"repro": rp, "accounts_checks": acc_chk, "K": {b: J[b]["K"] for b in J}, "labels": k_labels(J["raw"]["K"]), "labels_adjusted": k_labels(J["adjusted"]["K"]),
               "reported": {b: J[b]["reported"] for b in J}, "trials": {"cumulative": TRIALS, "new": 0}}
    # tables (aggregates)
    for name in ("k1", "k1_seeds", "m1", "p1", "gap", "k5_rows"):
        pd.concat([J[b][name].assign(basis=b) for b in J], ignore_index=True).to_csv(agg / f"{name}.csv", index=False)
    for name, f in (("sysP", "systems_p"), ("sysQ", "systems_q"), ("cntP", "helps_p"), ("cntQ", "helps_q"), ("chP", "chance_p"),
                    ("chQ", "chance_q"), ("candP", "candidates_p"), ("candQ", "candidates_q"), ("psP", "placebo_p"), ("psQ", "pseudo_q"),
                    ("sysO", "systems_o")):
        pd.concat([J[b][name].assign(basis=b) for b in J], ignore_index=True).to_csv(agg / f"{f}.csv", index=False)
    for b in J:
        for scope in ("P", "Q"):
            for kind, frame in J[b]["candidates"][scope].items():
                if len(frame):
                    frame.to_csv(agg / f"cand_{kind}_{scope.lower()}_{b}.csv", index=False)
    cells = []
    for b in J:
        for scope, key in (("P", "cellsP"), ("Q", "cellsQ")):
            c = J[b][key]
            cells.append(c[c["cost"] == pj.COST_BPS].assign(basis=b, scope=scope))
    pd.concat(cells, ignore_index=True).to_csv(agg / "cells.csv.gz", index=False)
    r = out / "raw"
    for f, dst in ((r / "p" / "goal.csv", "goal_p_raw.csv.gz"), (r / "q" / "goal.csv", "goal_q_raw.csv.gz"), (r / "o" / "goal.csv", "goal_o_raw.csv.gz"),
                   (r / "p" / "goal40.csv", "goal40_p_raw.csv"), (r / "q" / "goal40.csv", "goal40_q_raw.csv"),
                   (r / "benchmarks_p.csv", "benchmarks_p_raw.csv"), (r / "benchmarks_q.csv", "benchmarks_q_raw.csv")):
        (pd.read_csv(f).to_csv(agg / dst, index=False))
    unl = pd.concat([pd.read_csv(out / b / "unlimited.csv").assign(basis=b) for b in ("adjusted", "raw")], ignore_index=True)
    unl.to_csv(agg / "unlimited.csv", index=False)
    for f in ("repro.json", "accounts.json", "repro_tables.csv", "repro_c_goal.csv"):
        if (out / f).exists():
            shutil.copy2(out / f, agg / f)
    for f in out.glob("repro_a_*.csv.gz"):
        shutil.copy2(f, agg / f.name)
    # F
    tstats, pool = trade_stats(a)
    tstats.to_csv(agg / "trade_stats.csv", index=False)
    hist = pd.DataFrame({k: np.histogram(v, pj.RET_EDGES)[0] for k, v in pool.items()})
    hist.to_csv(agg / "trade_hist.csv", index=False)
    pooled = tstats[tstats["entry"] == "15개 합"]
    verdict["trades"] = {"pooled": pooled.to_dict("records"),
                         "entries_p_lt_05_new": int((tstats[(tstats["seed"] == "") & (tstats["entry"] != "15개 합") & (tstats["group"] == "new")]["st_mw_p"] < 0.05).sum()),
                         "seeds_p_lt_05_new": int((tstats[(tstats["seed"] != "") & (tstats["group"] == "new")]["st_mw_p"] < 0.05).sum())}
    (agg / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if figures.available():
        write_figures(a, agg, J, tstats, pool)
    log("판정: " + json.dumps(verdict["labels"], ensure_ascii=False) + " · " + json.dumps(
        {k: {b: {kk: vv for kk, vv in verdict["K"][b][k].items() if kk not in ("passing_list", "helps", "counts")} for b in J} for k in verdict["K"]["raw"]},
        ensure_ascii=False, default=float))
    return verdict


# --------------------------------------------------------------------------- figures

BASIS_LABEL = {"adjusted": "수정주가 기준", "raw": "실제 가격 기준"}
BASIS_COLOR = {"adjusted": "#888888", "raw": figures.COLORS[0]}


def fig_scatter(agg: Path, J: dict) -> Path:
    """(1) Every cell (15 entries; Stage P's 11 gates at U3000 and Stage Q's narrowed bands, 20 bp):
    adjusted vs raw drawdown-matched return, three periods."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    kp = ["setting", "gen", "exit", "filter"]
    groups = []
    for scope, key, keys in (("P", "gP", kp), ("Q", "gQ", kp + ["band"])):
        def sel(g: pd.DataFrame) -> pd.DataFrame:
            g = g[(g["gen"] != "E0") & (g["cost"] == pj.COST_BPS) & (g["mode"] == "grid")]
            return (g[g["band"].isin(ls.NARROW)] if scope == "Q" else g).set_index(keys)
        a_, r_ = sel(J["adjusted"][key]), sel(J["raw"][key])
        idx = a_.index.intersection(r_.index)
        groups.append((scope, a_.loc[idx], r_.loc[idx]))
    bench = J["raw"]["bench"]
    for ax, p in zip(axes, gs.JUDGED):
        for scope, a_, r_ in groups:
            if scope == "P":
                f = a_.index.get_level_values("filter")
                for lab, sel, col in (("U3000 · 필터 없음", f == "none", "#555555"), ("U3000 · A1", f.str.endswith("A1"), figures.COLORS[0]),
                                      ("U3000 · A2", f.str.endswith("A2"), figures.COLORS[1])):
                    ax.scatter(a_.loc[sel, f"dd_cagr_{p}"] * 100, r_.loc[sel, f"dd_cagr_{p}"] * 100, s=6, alpha=0.5, color=col, label=f"{lab} ({int(sel.sum())}칸)")
            else:
                ax.scatter(a_[f"dd_cagr_{p}"] * 100, r_[f"dd_cagr_{p}"] * 100, s=6, alpha=0.5, color=figures.COLORS[2],
                           label=f"U1000 · U500 (필터 5개, {len(a_)}칸)")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#222", lw=0.8)
        ax.axhline(bench["ma200"][f"dd_cagr_{p}"] * 100, color="#8e44ad", ls="--", lw=1, label="SPY 200일선 + 국채")
        allx = np.concatenate([g[1][f"dd_cagr_{p}"].to_numpy() for g in groups])
        ally = np.concatenate([g[2][f"dd_cagr_{p}"].to_numpy() for g in groups])
        ax.set_title(f"{p} 낙폭 15% 맞춤 수익 · 차이 중앙값 {np.median(ally - allx) * 100:+.2f}%p", fontsize=10)
        ax.set_xlabel("수정주가 기준 (%)")
        ax.set_ylabel("실제 가격 기준 (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("칸마다 수정주가 기준 vs 실제 가격 기준 (매수 15개 × 설정 4 × 매도 2, 편도 20bp)", fontsize=11)
    return figures._save(fig, agg / "figures" / "scatter_cells.png")


def fig_gates(agg: Path, J: dict) -> Path:
    """(2) Per gate the median over the 8 systems, both bases, with the benchmarks."""
    plt = figures._plt()
    fs = list(rs.FILTERS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)
    xx = np.arange(len(fs))
    for ax, p in zip(axes, gs.JUDGED):
        for k, b in enumerate(("adjusted", "raw")):
            s = J[b]["sysP"]
            v = [float(s.loc[s["filter"] == f, f"dd_cagr_{p}"].median()) * 100 for f in fs]
            ax.bar(xx + (k - 0.5) * 0.38, v, width=0.38, color=BASIS_COLOR[b], label=BASIS_LABEL[b])
            ax.axhline(J[b]["bench"]["ew"][f"dd_cagr_{p}"] * 100, color=BASIS_COLOR[b], ls=":", lw=1.2, label=f"동일비중 + 국채 ({BASIS_LABEL[b]})")
        bench = J["raw"]["bench"]
        ax.axhline(bench["ma200"][f"dd_cagr_{p}"] * 100, color="#8e44ad", ls="--", lw=1.2, label="SPY 200일선 + 국채")
        ax.axhline(bench["mix"][f"dd_cagr_{p}"] * 100, color="#c0392b", ls="--", lw=1.0, label="SPY + 국채")
        ax.set_xticks(xx, ["없음" if f == "none" else f for f in fs], rotation=60, fontsize=8)
        ax.set_title(f"{p} 낙폭 15% 맞춤 수익 (%)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle("국면 필터별 시스템 중앙값 (시스템 8개 = 설정 4 × 매도 2, 매수 15개 중앙값의 중앙값; U3000)", fontsize=11)
    return figures._save(fig, agg / "figures" / "gates_bases.png")


def fig_bands(agg: Path, J: dict) -> Path:
    """(4) U3000 -> U1000 -> U500: per gate the median over the 8 systems of the 2008-2023 matched
    return, both bases."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    bl = list(ls.BANDS)
    for ax, p in zip(axes, ("2008-2023", "1993-2007")):
        for k, f in enumerate(ls.FILTERS):
            col = figures.COLORS[k % 4] if k < 4 else "#555555"
            for b, lsty in (("adjusted", "--"), ("raw", "-")):
                s = J[b]["sysQ"]
                ys = [float(s.loc[(s["band"] == bb) & (s["filter"] == f), f"dd_cagr_{p}"].median()) * 100 for bb in bl]
                ax.plot(range(3), ys, marker="o", ls=lsty, color=col, lw=1.4 if b == "raw" else 1.0,
                        label=f"{'필터 없음' if f == 'none' else f} · {BASIS_LABEL[b]}")
        ax.axhline(J["raw"]["bench"]["ma200"][f"dd_cagr_{p}"] * 100, color="#8e44ad", ls=":", lw=1.2, label="SPY 200일선 + 국채")
        ax.set_xticks(range(3), bl)
        ax.set_title(f"{p} 낙폭 15% 맞춤 수익 (%)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    km = {b: J[b]["K"]["K5"]["median_d2008"] for b in J}
    fig.suptitle("사는 범위를 좁힐 때 (시스템 8개 중앙값; 실선 실제 가격, 점선 수정주가)\nU3000 대비 2008–2023 변화 중앙값(시스템 40개): "
                 + ", ".join(f"{bb} {BASIS_LABEL[b]} {km[b][bb] * 100:+.2f}%p" for bb in ls.NARROW for b in J), fontsize=10)
    return figures._save(fig, agg / "figures" / "bands_2008.png")


def fig_entries(agg: Path, J: dict) -> Path:
    """(5) Per entry the margin over random buys: unlimited CAGR minus the best E0 seed (X1 / H1, 1993-
    2023) and the median over the 8 no-gate systems of dd_cagr minus E0, both bases; E0 seeds against
    the other four as the yardstick."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    names = ENTRIES + SEEDS
    y = np.arange(len(names))
    for k, b in enumerate(("adjusted", "raw")):
        unl = pd.read_csv(Path(agg) / "unlimited.csv")
        u = unl[unl["basis"] == b].set_index(["src", "exit"])
        for j, x in enumerate(EXITS):
            v = []
            for n_ in names:
                refs = [s for s in SEEDS if s != n_]
                v.append((u.loc[(n_, x), "cagr_1993-2023"] - max(u.loc[(s, x), "cagr_1993-2023"] for s in refs)) * 100)
            axes[0].scatter(v, y + (k - 0.5) * 0.3, marker="o" if x == "X1" else "^", s=24, color=BASIS_COLOR[b],
                            label=f"{BASIS_LABEL[b]} · {x}")
        k1, k1s = J[b]["k1"].set_index("entry"), J[b]["k1_seeds"].set_index("seed")
        v = [k1.loc[n_, "b_median_diff_1993-2023"] * 100 if n_ in k1.index else k1s.loc[n_, "b_median_diff_1993-2023"] * 100 for n_ in names]
        axes[1].barh(y + (k - 0.5) * 0.38, v, height=0.38, color=BASIS_COLOR[b], label=BASIS_LABEL[b])
    for ax in axes:
        ax.axvline(0, color="#333", lw=0.8)
        ax.axhspan(len(ENTRIES) - 0.5, len(names) - 0.5, color="#eeeeee", zorder=0)
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(y, [f"{n_} {ev.NAMES.get(n_, '')}" if not n_.startswith("R0s") else f"E0 씨앗 {n_[3:]} (나머지 넷과)" for n_ in names], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("제한 없는 계좌 CAGR 1993–2023 − E0 씨앗 최댓값 (%p)")
    axes[1].set_xlabel("자리 제한 8개 시스템(필터 없음, U3000): 맞춘 수익 1993–2023 − E0, 중앙값 (%p)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    axes[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    lab = {b: J[b]["K"]["K1"] for b in J}
    fig.suptitle("매수 규칙 vs 무작위 매수 E0 (회색 띠 = E0 씨앗 하나를 나머지 넷과 견준 우연 기준) — '무작위보다 나음' "
                 + ", ".join(f"{BASIS_LABEL[b]} {lab[b]['above_random']}/15 (씨앗 {lab[b]['seeds_above_random']}/5)" for b in J), fontsize=10.5)
    return figures._save(fig, agg / "figures" / "entries_vs_e0.png")


def fig_trades(agg: Path, tstats: pd.DataFrame, pool: dict) -> Path:
    """(6) Pooled per-trade return histogram (15 entries x X1) and the per-entry mean difference
    (new on raw minus common) with its CI, next to E0's seeds (the random-buy yardstick)."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for grp, col, lab, lsty in (("common", "#777777", "두 기준 공통 신호", "-"), ("new", figures.COLORS[0], "실제 가격 기준에서 새로 생긴 신호", "-"),
                                ("adj_only", figures.COLORS[1], "수정주가 기준에만 있던 신호", "--")):
        r = pool[grp]
        pj._bar(ax, pj._density(np.histogram(r, pj.RET_EDGES)[0]), color=col, lw=1.5, ls=lsty, label=f"{lab} ({len(r):,}개, 평균 {r.mean() * 100:+.1f}%)")
    pooled = tstats[tstats["entry"] == "15개 합"].set_index("group")
    lines = []
    for grp, lab in (("new", "새로 생김 vs 공통"), ("adj_only", "수정주가에만 vs 공통")):
        if grp in pooled.index and pd.notna(pooled.loc[grp].get("st_diff")):
            s = pooled.loc[grp]
            lines.append(f"{lab}: 평균 차 {s['st_diff'] * 100:+.1f}%p (95% CI {s['st_ci_lo'] * 100:+.1f} – {s['st_ci_hi'] * 100:+.1f}), "
                         f"+50% 이상 {s['big50']:.1%} (공통 {s['ref_big50']:.1%}), −20% 이하 {s['loss20']:.1%} (공통 {s['ref_loss20']:.1%})")
    ent = tstats[(tstats["seed"] == "") & (tstats["entry"] != "15개 합") & (tstats["group"] == "new")]
    sd = tstats[(tstats["seed"] != "") & (tstats["group"] == "new")]
    lines.append(f"순위 검정 p < 0.05: 매수 15개 중 {int((ent['st_mw_p'] < 0.05).sum())}개, 무작위 매수 씨앗 5개 중 {int((sd['st_mw_p'] < 0.05).sum())}개")
    ax.text(0.99, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=7.3,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("신호 하나의 거래 수익률 (%, 편도 20bp, 양 끝 칸은 −60% 미만 / +200% 초과)")
    ax.legend(fontsize=7.5, frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_title("매수 15개 × X1 합산 (밀도)", fontsize=10)
    ax = axes[1]
    rows = pd.concat([ent, sd], ignore_index=True)
    lab = [r["entry"] if r["seed"] == "" else f"E0 씨앗 {r['seed'][3:]}" for _, r in rows.iterrows()]
    yy = np.arange(len(rows))
    ax.errorbar(rows["st_diff"] * 100, yy, xerr=[(rows["st_diff"] - rows["st_ci_lo"]) * 100, (rows["st_ci_hi"] - rows["st_diff"]) * 100], fmt="o",
                color=figures.COLORS[0], ms=4, capsize=2)
    ax.axhspan(len(ent) - 0.5, len(rows) - 0.5, color="#eeeeee", zorder=0)
    ax.set_yticks(yy, lab, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("새로 생긴 신호 − 공통 신호, 평균 거래 수익률 차 (%p, 95% CI)")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("매수별 (회색 띠 = 무작위 매수 씨앗, 같은 비교)", fontsize=10)
    fig.suptitle("거래 단위: 실제 가격 기준으로 고치며 새로 생긴 신호 · 사라진 신호의 수익률", fontsize=11)
    return figures._save(fig, agg / "figures" / "trades_hist.png")


def write_figures(a, agg: Path, J: dict, tstats: pd.DataFrame, pool: dict) -> list[Path]:
    paths = [fig_scatter(agg, J), fig_gates(agg, J)]
    paths.append(rs.fig_placebo(agg, J["raw"]["psP"], J["raw"]["sysP"]))  # (3) raw basis
    paths += [fig_bands(agg, J), fig_entries(agg, J), fig_trades(agg, tstats, pool)]
    return paths


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["repro", "tables", "accounts", "report"])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    for k in ("stored_eligible", "stage_h", "stage_i", "v2", "events", "r0x1", "tables", "stage_p", "stage_q", "stage_o", "stage_r"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--table-workers", type=int, default=4)
    ap.add_argument("--feature-workers", type=int, default=6)
    a = ap.parse_args(argv)
    {"repro": stage_repro, "tables": stage_tables, "accounts": stage_accounts, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main(sys.argv[1:])
