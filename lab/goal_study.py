"""Stage O: the Stage N grid judged by drawdown-matched return (DECISIONS 94).

    python -m lab.goal_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                             --r0x1 DIR --tables DIR --out DIR [--stage-n DIR] [--workers N] [--feature-workers N] [--dry]

The accounts of Stage N (lab/account_study.py: the 20 settings, the M1 exclusion and no exclusion,
the same 100 random orders from Stage M / L's draws; no pseudo exclusions) are replayed again from the
stored Stage L tables, and every equity curve is re-measured with the user's goal metrics of
DECISIONS 64-65 (lab/target.py): w* = the largest weight (0.01 grid, no leverage) whose daily
w x account + (1 - w) x T-bill blend keeps the 1993-2023 drawdown within 15%; the blend's CAGR in
1993-2023, both halves and 2017-2023 at that w* (half-refit w* reported); 3-year calendar windows
won against the SPY + T-bill mix, SPY200 (+ T-bill) and the equal-weight universe, each at its own
w*. Curves are not saved. The exit axis (Q2) adds Stage L's other 7 exits at N=10 / W2 / P0 with
Stage L's own weights. --stage-n checks the plain CAGR / MDD against Stage N's accounts table first.
Output (aggregates only): base.parquet, jobs/*, goal.csv, windows.csv, benchmarks.csv, systems.csv,
cells.csv, candidates.csv, chance.csv, exits.csv, repro.csv, chance.json, figures/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import account_study as ns, entry_study as es, exclude_study as ms, exit_study as xs, figures, pair_study as pj
from . import pick_study as ph, select_study as ks, target as tg, tail_study as ts
from .metrics import TRADING_DAYS

# ---- registered (DECISIONS 94): fixed before any result
EXITS = list(ns.EXITS)  # X1, H1, H9, ST: the grid of 93
EXIT_AXIS = list(xs.EXITS_L)  # Q2: Stage L's 11 exits at N=10 / W2 / P0
AXIS_SETTING = (10, "W2", "P0")
KINDS = ("rule", "none")  # the M1 account and no exclusion
BUDGET = tg.DD_BUDGET  # 15%
GOAL_PERIODS = ("1993-2023", "1993-2007", "2008-2023", "2017-2023")
JUDGED = ("1993-2023",) + es.HALVES  # 2017-2023 reported, not gated
BENCHMARKS = {"mix": "SPY + 국채", "ma200": "SPY 200일선 + 국채", "ew": "유니버스 동일비중 + 국채"}
GATED = ("mix", "ma200")  # a system / candidate must beat both
MIN_WINDOW_SHARE = 0.5
MIN_EXITS = 2
N_TRIALS = ns.N_TRIALS_BEFORE + ns.N_NEW  # 71,774; no new trials (the same strategies re-measured)
SEED = 94
_W: dict = {}


def spans(years: np.ndarray, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """(first, last) row of each inclusive calendar-year range in a sorted year array."""
    out = []
    for a, b in ranges:
        rows = np.flatnonzero((years >= a) & (years <= b))
        out.append((int(rows[0]), int(rows[-1])) if len(rows) else (-1, -1))
    return out


def span_cagr(L: np.ndarray, s: int, e: int) -> np.ndarray:
    """CAGR (TRADING_DAYS convention of lab/target.py) of rows s..e from L = cumulative log growth with
    a leading zero row."""
    return np.exp((L[e + 1] - L[s]) * TRADING_DAYS / (e - s + 1)) - 1.0


def goal_metrics(R: np.ndarray, C: np.ndarray, years: np.ndarray, wins: list[tuple[int, int]]) -> dict:
    """Per column of daily returns R (days x curves) with T-bill returns C (same shape): w*, the blend's
    CAGR per period at w*, the half-refit w*, the blend's MDD and window CAGRs (windows x curves)."""
    w = tg.fit_weights(R, C, BUDGET)
    B = w * R + (1.0 - w) * C
    L = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.log1p(B), axis=0)])
    out = {"w": w}
    for p, (s, e) in zip(GOAL_PERIODS, spans(years, [es.PERIODS[p] for p in GOAL_PERIODS])):
        out[f"dd_cagr_{p}"] = span_cagr(L, s, e)
    for h, (s, e) in zip(es.HALVES, spans(years, [es.PERIODS[h] for h in es.HALVES])):
        out[f"w_{h}"] = tg.fit_weights(R[s:e + 1], C[s:e + 1], BUDGET)
    eq = np.exp(L)
    out["dd_mdd"] = (eq / np.maximum.accumulate(eq, axis=0) - 1.0).min(axis=0)
    out["win_cagr"] = np.array([span_cagr(L, s, e) for s, e in spans(years, wins)])
    return out


def bench_curves(px, wdates, w0, cash) -> dict:
    """Equity of SPY, SPY200 (cash when off) and the equal-weight universe on the window, built as
    lab/entry_study.baselines."""
    n = len(wdates)
    spy = pd.Series(px.spy_c[w0:w0 + n], index=wdates)
    r = spy.pct_change(fill_method=None).fillna(0.0)
    full_spy = pd.Series(px.spy_c, index=px.dates)
    on = (full_spy > full_spy.rolling(200).mean()).shift(1, fill_value=False).to_numpy()[w0:w0 + n]
    ma = np.cumprod(1 + np.where(on, r.to_numpy(), cash))
    return {"mix": (spy / spy.iloc[0]).to_numpy(), "ma200": ma, "ew": px.ew[w0:w0 + n] / px.ew[w0]}


# --------------------------------------------------------------------------- workers

def _setup(cash: np.ndarray, wdates: pd.DatetimeIndex) -> dict:
    """Return days, their years, the 3-year windows and the benchmarks' metrics (shared by all jobs)."""
    idx = wdates[1:]
    years = idx.year.to_numpy()
    wins = tg.windows(idx)
    c = cash[1:]
    return {"years": years, "wins": wins, "c": c}


def _init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    ns._init(panel_dir, eligible, dirs, cfg)
    _W.update(ns._W)
    _W.update(_setup(_W["cash"], _W["wdates"]))
    _W["bench"] = benchmarks_table(_W["px"], _W["wdates"], _W["w0"], _W["cash"])[1]


def benchmarks_table(px, wdates, w0, cash) -> tuple[pd.DataFrame, dict]:
    """The three benchmarks at their own w*: a table and the metrics by name (window CAGRs kept)."""
    sp = _setup(cash, wdates)
    curves = bench_curves(px, wdates, w0, cash)
    names = list(BENCHMARKS)
    EQ = np.column_stack([curves[k] for k in names])
    R = EQ[1:] / EQ[:-1] - 1.0
    C = np.repeat(sp["c"][:, None], len(names), axis=1)
    gm = goal_metrics(R, C, sp["years"], sp["wins"])
    rows, by = [], {}
    for i, k in enumerate(names):
        row = {"benchmark": k, "label": BENCHMARKS[k], **{c: float(v[i]) for c, v in gm.items() if c != "win_cagr"},
               **es.period_stats(pd.Series(EQ[:, i], index=wdates))}
        rows.append(row)
        by[k] = {**row, "win_cagr": gm["win_cagr"][:, i]}
    return pd.DataFrame(rows), by


class GRows(ns.Rows):
    """account_study.Rows for any Stage L exit (no sizing stop outside the grid's 4 exits), with the
    partial-sale day of TP."""

    def __init__(self, d: pd.DataFrame, b: pd.DataFrame, x: str):
        if x in ns.SIZE_STOP:
            super().__init__(d, b, x)
        else:
            w0, nd = _W["w0"], _W["n"]
            self.ei = d["ei"].to_numpy() - w0
            self.sale = d["sale"].to_numpy() - w0
            self.closed = d["closed"].to_numpy(dtype=bool) & (self.sale < nd)
            self.sj = d["sj"].to_numpy()
            self.ret = d["ret"].to_numpy(dtype=float)
            self.year = pd.DatetimeIndex(d["signal_date"]).year.to_numpy()
            self.brow = d["brow"].to_numpy()
            self.cap = b["cap_rank"].to_numpy(dtype=float)[self.brow]
            self.k2 = b["K2"].to_numpy(dtype=float)[self.brow]
            self.dv = b["dv20"].to_numpy(dtype=float)[self.brow]
            self.stop = np.full(len(self.ei), np.nan)
            self.w_stage = d["w"].to_numpy(dtype=float)
        part = xs.EXITS_L[x].get("partial")
        self.psale = d["psale"].to_numpy() if part else None
        self.pfrac = part[1] if part else 0.5


def _replay(r: GRows, mask: np.ndarray, w: np.ndarray, key: np.ndarray, n: int, cost: float) -> tuple[np.ndarray, int]:
    """account_study._replay with TP's partial sales (as exit_study.run_set)."""
    idx = np.flatnonzero(mask & np.isfinite(w))
    eq, tr = ph.replay_risk(r.ei[idx] - 1, r.ei[idx], r.sj[idx], r.sale[idx], r.closed[idx], w[idx], key[idx],
                            _W["O"], _W["LC"], _W["cash"], cost, slots=n,
                            psale=r.psale[idx] if r.psale is not None else None, pfrac=r.pfrac)
    return eq, len(tr)


def _measure(curves: list, cost_bps: float) -> tuple[list[dict], list[dict], dict | None]:
    """Per curve: plain period stats, trades and the goal metrics against each benchmark; the window
    CAGR medians over the curves; the daily moments of the median curve's blend (20 bp)."""
    years, wins, c = _W["years"], _W["wins"], _W["c"]
    EQ = np.column_stack([eq for *_, eq in curves])
    R = EQ[1:] / EQ[:-1] - 1.0
    C = np.repeat(c[:, None], R.shape[1], axis=1)
    gm = goal_metrics(R, C, years, wins)
    rows = []
    for i, (src, o, trades, eq) in enumerate(curves):
        row = {"src": src, "order": o, "trades": trades, **es.period_stats(pd.Series(eq, index=_W["wdates"])),
               **{k: float(v[i]) for k, v in gm.items() if k != "win_cagr"}}
        for b in BENCHMARKS:
            bw = _W["bench"][b]["win_cagr"]
            d = gm["win_cagr"][:, i] - bw
            row[f"ws_{b}"] = float((d >= 0).mean()) if len(d) else np.nan
            row[f"worst_{b}"] = float(d.min()) if len(d) else np.nan
        rows.append(row)
    win_rows = [{"w_idx": j, "a": a, "b": b, "win_cagr": float(np.median(gm["win_cagr"][j]))} for j, (a, b) in enumerate(wins)]
    mom = None
    if cost_bps == pj.COST_BPS:
        i = int(np.argsort(gm["dd_cagr_1993-2023"])[len(curves) // 2])
        blended = gm["w"][i] * R[:, i] + (1.0 - gm["w"][i]) * C[:, i]
        mom = ks._moments(np.concatenate([[1.0], np.cumprod(1.0 + blended)]))
    return rows, win_rows, mom


def run_job(job):
    """One buy rule x exit x slot count (mode 'grid': the 5 settings of that slot count; 'axis': N=10 /
    W2 / P0 with Stage L's weights; 'rule40': the M1 account only, at the job's cost)."""
    gen, x, n_slots, cost_bps, srcs, mode = job
    t0 = time.time()
    cost = cost_bps / 10_000
    sets = [AXIS_SETTING] if mode == "axis" else [s for s in ns.settings() if s[0] == n_slots]
    rows = {s: GRows(ms.prepared(s, x), ms.source_base(s), x) for s in srcs}
    keeps, pseudos = {}, {}
    for s in srcs:
        b = ms.source_base(s)
        keep_b = ~ms.excluded(b, ns.RULE)
        keeps[s] = keep_b[rows[s].brow]
        pseudos[s] = [ks.pseudo_mask(b["si"].to_numpy(), keep_b, np.random.default_rng([ms.SEED, zlib.crc32(f"{ns.RULE}|{s}".encode()), k]))[rows[s].brow]
                      for k in range(ms.N_PSEUDO)]  # only to stay in step with Stage M's draws
    if mode == "axis":
        wts = {s: {"W2": rows[s].w_stage} for s in srcs}
    else:
        wts = {s: {w: ns.sizing(rows[s].stop, n_slots, w)[0] for w in ns.SIZING} for s in srcs}
    curves = {(st, k): [] for st in sets for k in KINDS}
    p0 = [st for st in sets if st[2] == "P0"]
    n_total = {s: len(rows[s]) for s in srcs}
    for kind, k, s, o, key in ns.stage_m_draws(gen, x, srcs, keeps, pseudos, n_total):
        if kind != "rule":
            continue
        for st in p0:
            eq, nt = _replay(rows[s], keeps[s], wts[s][st[1]], key, n_slots, cost)
            curves[(st, "rule")].append((s, o, nt, eq))
    if mode != "rule40":
        for s in srcs:
            name = pj.set_name(("e0full", int(s[3:])) if s.startswith("R0s") else ("single", s))
            rng = np.random.default_rng([ms.xs.SEED, zlib.crc32(f"{name}|{x}|{pj.COST_BPS}".encode())])
            allm = np.ones(len(rows[s]), dtype=bool)
            for o in range(ns.N_NONE // len(srcs)):
                key = rng.random(len(rows[s]))
                for st in p0:
                    eq, nt = _replay(rows[s], allm, wts[s][st[1]], key, n_slots, cost)
                    curves[(st, "none")].append((s, o, nt, eq))
    for st in [st for st in sets if st[2] != "P0"]:
        for s in srcs:
            r = rows[s]
            tie = np.random.default_rng([ns.SEED, zlib.crc32(f"tie|{s}|{x}".encode())]).random(len(r))
            key = r.k2 + tie * 1e-9 if st[2] == "P1" else -r.cap + tie * 1e-6
            eq, nt = _replay(r, keeps[s], wts[s][st[1]], key, n_slots, cost)
            curves[(st, "rule")].append((s, 0, nt, eq))
            if mode != "rule40":
                eq, nt = _replay(r, np.ones(len(r), dtype=bool), wts[s][st[1]], key, n_slots, cost)
                curves[(st, "none")].append((s, 0, nt, eq))
    accs, wins, moms = [], [], []
    for (st, kind), cv in curves.items():
        if not cv:
            continue
        rows_, win_rows, mom = _measure(cv, cost_bps)
        head = {"setting": ns.setting_name(st), "kind": kind}
        accs += [{**head, **r} for r in rows_]
        wins += [{**head, **r} for r in win_rows]
        if mom is not None and kind == "rule":
            moms.append({**head, **mom})
    ms._W["prep"].clear()
    pj._W["tables"].clear()
    tail = {"gen": gen, "exit": x, "cost": cost_bps, "mode": mode}
    frame = lambda rs: pd.DataFrame([{**r, **tail} for r in rs])  # noqa: E731
    return job, frame(accs), frame(wins), frame(moms), time.time() - t0


def job_name(gen: str, x: str, n: int, cost: float, mode: str) -> str:
    return f"{gen}__{x}__N{n}" + ("__axis" if mode == "axis" else "") + ("" if cost == pj.COST_BPS else f"__{cost:g}bp")


def run_jobs(jobs: list, a) -> None:
    d = Path(a.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs, a.cfg)) as ex:
        for i, (job, acc, win, mom, sec) in enumerate(ex.map(run_job, jobs, chunksize=1)):
            nm = job_name(job[0], job[1], job[2], job[3], job[5])
            for suffix, f in (("win", win), ("mom", mom)):
                f.to_csv(d / f"{nm}_{suffix}.csv", index=False)
            acc.to_csv(d / f"{nm}_acc.csv", index=False)  # last: its presence marks the job done
            if (i + 1) % 5 == 0 or i + 1 == len(jobs):
                ts.log(f"  [{i + 1}/{len(jobs)}] {nm} {sec:.0f}s · 남은 예상 {(len(jobs) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


# --------------------------------------------------------------------------- verdicts

GOAL_COLS = ["w", "w_1993-2007", "w_2008-2023", "dd_mdd"] + [f"dd_cagr_{p}" for p in GOAL_PERIODS] \
    + [f"{m}_{b}" for b in BENCHMARKS for m in ("ws", "worst")]


def summarize(acc: pd.DataFrame) -> pd.DataFrame:
    """Per setting x buy rule x exit x kind x cost: medians over the orders (and the 25 / 75% of w*
    and the full-period dd_cagr)."""
    keys = ["setting", "gen", "exit", "kind", "cost", "mode"]
    g = acc.groupby(keys, sort=False)
    med = g.median(numeric_only=True).drop(columns=["order"], errors="ignore")
    for c in ("dd_cagr_1993-2023", "w"):
        med[f"{c}_q25"] = g[c].quantile(0.25)
        med[f"{c}_q75"] = g[c].quantile(0.75)
    med["orders"] = g.size()
    out = med.reset_index()
    nwp = out["setting"].map(ns.parse_setting)
    out["N"], out["W"], out["P"] = nwp.str[0], nwp.str[1], nwp.str[2]
    return out


def beats(row, bench: dict, periods=JUDGED) -> bool:
    """dd_cagr above both gated benchmarks in every judged period."""
    return all(row[f"dd_cagr_{p}"] > bench[b][f"dd_cagr_{p}"] for p in periods for b in GATED)


def cell_pass(row, bench: dict) -> bool:
    """The per-cell part of the candidate test: dd_cagr above both benchmarks in the full period and
    both halves, window share >= 0.5 against both, >= 30 trades."""
    return bool(beats(row, bench) and all(row[f"ws_{b}"] >= MIN_WINDOW_SHARE for b in GATED) and row["trades"] >= pj.MIN_TRADES)


def systems(g: pd.DataFrame, bench: dict) -> pd.DataFrame:
    """Per setting x exit x kind: median over the 15 entries of the goal metrics; 'goal-passing system'
    if the medians beat both benchmarks in the full period and both halves; the same setting's E0."""
    cells = g[(g["gen"] != "E0") & (g["mode"] != "axis") & (g["cost"] == pj.COST_BPS)]
    e0 = g[(g["gen"] == "E0") & (g["mode"] != "axis") & (g["cost"] == pj.COST_BPS)].set_index(["setting", "exit", "kind"])
    rows = []
    for (st, x, kind), c in cells.groupby(["setting", "exit", "kind"], sort=False):
        row = {"setting": st, "exit": x, "kind": kind, "entries": len(c),
               **{f"{col}": float(c[col].median()) for col in GOAL_COLS + ["cagr_1993-2023", "mdd_1993-2023", "cagr_2017-2023"]}}
        row["system_pass"] = beats(row, bench)
        for col in [f"dd_cagr_{p}" for p in GOAL_PERIODS] + ["w"]:
            row[f"e0_{col}"] = float(e0.loc[(st, x, kind), col]) if (st, x, kind) in e0.index else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    nwp = out["setting"].map(ns.parse_setting)
    out["N"], out["W"], out["P"] = nwp.str[0], nwp.str[1], nwp.str[2]
    return out


def candidates(g: pd.DataFrame, bench: dict, pass40: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cells (M1 account, 20 bp, W2 / W3) with the per-cell test; per setting x entry: >= 2 exits at
    20 and 40 bp and the same entry passing at an adjacent N with the same sizing and priority;
    'entry-specific' if its passing cells also beat the same cell's E0 in both halves."""
    c = g[(g["kind"] == "rule") & (g["mode"] != "axis") & (g["cost"] == pj.COST_BPS)].copy()
    c["cell_pass"] = [cell_pass(r, bench) for _, r in c.iterrows()]
    e0 = c[c["gen"] == "E0"].set_index(["setting", "exit"])
    c["beats_e0"] = [all(r[f"dd_cagr_{h}"] > e0.loc[(r["setting"], r["exit"]), f"dd_cagr_{h}"] for h in es.HALVES)
                     if (r["setting"], r["exit"]) in e0.index else False for _, r in c.iterrows()]
    c["pass_40bp"] = [bool(pass40.get((r["setting"], r["gen"], r["exit"]), False)) for _, r in c.iterrows()]
    cc = c[(c["gen"] != "E0") & c["W"].isin(ns.CANDIDATE_W)]
    s = cc.groupby(["setting", "gen"]).agg(n20=("cell_pass", "sum"), n40=("pass_40bp", "sum"),
                                         n_e0=("beats_e0", lambda b: int((b & cc.loc[b.index, "cell_pass"]).sum()))).reset_index()
    s["ok"] = (s["n20"] >= MIN_EXITS) & (s["n40"] >= MIN_EXITS)
    ok = set(zip(s.loc[s["ok"], "setting"], s.loc[s["ok"], "gen"]))
    s["neighbour"] = [any((ns.setting_name(nb), gen) in ok for nb in ns.neighbours(ns.parse_setting(st))) for st, gen in zip(s["setting"], s["gen"])]
    s["candidate"] = s["ok"] & s["neighbour"]
    s["entry_specific"] = s["candidate"] & (s["n_e0"] >= MIN_EXITS)
    return c, s


def chance(acc: pd.DataFrame, cells: pd.DataFrame, bench: dict) -> pd.DataFrame:
    """Per setting x exit (M1, 20 bp): entries passing the cell test vs the 5 E0 seeds' pass rate p0
    (each seed's own median over its orders) and Binomial(15, p0) q95."""
    e0 = acc[(acc["gen"] == "E0") & (acc["kind"] == "rule") & (acc["cost"] == pj.COST_BPS) & (acc["mode"] != "axis")]
    seeds = e0.groupby(["setting", "exit", "src"]).median(numeric_only=True).reset_index()
    seeds["cell_pass"] = [cell_pass(r, bench) for _, r in seeds.iterrows()]
    rows = []
    c = cells[cells["gen"] != "E0"]
    for (st, x), g in c.groupby(["setting", "exit"], sort=False):
        sd = seeds[(seeds["setting"] == st) & (seeds["exit"] == x)]
        p0 = float(sd["cell_pass"].mean()) if len(sd) else np.nan
        rows.append({"setting": st, "exit": x, "entries": len(g), "passing": int(g["cell_pass"].sum()),
                     "passing_list": " ".join(sorted(g.loc[g["cell_pass"], "gen"])), "seeds_passing": int(sd["cell_pass"].sum()),
                     "p0": p0, "q95": ks.q95(len(g), p0) if np.isfinite(p0) else len(g)})
    out = pd.DataFrame(rows)
    out["beyond_chance"] = out["passing"] > out["q95"]
    return out


def repro(g: pd.DataFrame, stage_n: Path) -> pd.DataFrame:
    """Plain CAGR / MDD / trades of the grid's M1 ('rule') and no-exclusion accounts against Stage N's
    accounts table (medians over the same orders)."""
    sa = pd.read_csv(stage_n / "accounts.csv.gz")
    sa = sa[(sa["k"] == -1) & sa["kind"].isin(list(KINDS))].set_index(["setting", "gen", "exit", "kind"])
    mine = g[(g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)].set_index(["setting", "gen", "exit", "kind"])
    cols = ["trades"] + [f"{s}_{p}" for p in es.PERIODS for s in ("cagr", "mdd")]
    out = []
    for key in mine.index.intersection(sa.index):
        for col in cols:
            out.append({"setting": key[0], "gen": key[1], "exit": key[2], "kind": key[3], "value": col,
                        "mine": float(mine.loc[key, col]), "stage_n": float(sa.loc[key, col])})
    r = pd.DataFrame(out)
    if len(r):
        r["diff"] = (r["mine"] - r["stage_n"]).abs()
    return r


def exit_axis(g: pd.DataFrame) -> pd.DataFrame:
    """Q2 at N=10 / W2 / P0: per exit, the 15 entries' medians (M1, no exclusion) and E0, plain and
    drawdown-matched."""
    st = ns.setting_name(AXIS_SETTING)
    a = g[(g["setting"] == st) & (g["cost"] == pj.COST_BPS)]
    rows = []
    for x in EXIT_AXIS:
        ax = a[a["exit"] == x]
        if ax.empty:
            continue
        row = {"exit": x, "label": xs.EXITS_L[x]["label"]}
        for kind in KINDS:
            c = ax[(ax["kind"] == kind) & (ax["gen"] != "E0")]
            for col in ["cagr_1993-2023", "mdd_1993-2023", "w"] + [f"dd_cagr_{p}" for p in GOAL_PERIODS]:
                row[f"{kind}_{col}"] = float(c[col].median()) if len(c) else np.nan
        e0 = ax[(ax["kind"] == "rule") & (ax["gen"] == "E0")]
        for col in ["cagr_1993-2023", "mdd_1993-2023", "w", "dd_cagr_1993-2023"]:
            row[f"e0_{col}"] = float(e0[col].iloc[0]) if len(e0) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["rank_plain"] = out["rule_cagr_1993-2023"].rank(ascending=False).astype(int)
        out["rank_goal"] = out["rule_dd_cagr_1993-2023"].rank(ascending=False).astype(int)
    return out


def psr_dsr(cells: pd.DataFrame, cand: pd.DataFrame, mom: pd.DataFrame, bench_sr: float) -> pd.DataFrame:
    """DSR (N = 71,774) and PSR vs the SPY mix's daily Sharpe for the candidates' passing cells."""
    m = mom.set_index(["setting", "gen", "exit"])
    sr_var = float(np.var(m.loc[[i for i in m.index if i[1] != "E0"], "m_sr"].dropna())) if len(m) else np.nan
    ok = set(zip(cand.loc[cand["candidate"], "setting"], cand.loc[cand["candidate"], "gen"]))
    rows = []
    for _, r in cells[cells["cell_pass"]].iterrows():
        key = (r["setting"], r["gen"], r["exit"])
        if (r["setting"], r["gen"]) not in ok or key not in m.index:
            continue
        mm = m.loc[key]
        rows.append({"setting": key[0], "gen": key[1], "exit": key[2],
                     "dsr": ks.dsr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), N_TRIALS, sr_var),
                     "psr_mix": ks.psr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), bench_sr)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures

def fig_grids(out: Path, cells: pd.DataFrame, bench: dict) -> list[Path]:
    """① Grid A (N x W) and B (N x P) of the median dd_cagr over the 60 cells (M1, 15 entries x 4 exits)."""
    plt = figures._plt()
    made = []
    for grid, cols in (("A", list(ns.SIZING)), ("B", list(ns.PRIORITY))):
        fig, axes = plt.subplots(1, len(GOAL_PERIODS), figsize=(5.0 * len(GOAL_PERIODS), 4.4))
        for ax, per in zip(axes, GOAL_PERIODS):
            mat = np.full((len(ns.SLOTS_N), len(cols)), np.nan)
            for i, n in enumerate(ns.SLOTS_N):
                for j, c in enumerate(cols):
                    st = ns.setting_name((n, c, "P0") if grid == "A" else (n, "W2", c))
                    g = cells[(cells["setting"] == st) & (cells["gen"] != "E0")]
                    if len(g):
                        mat[i, j] = float(g[f"dd_cagr_{per}"].median()) * 100
            im = ax.imshow(mat, cmap="viridis", aspect="auto")
            mix, ma = bench["mix"][f"dd_cagr_{per}"] * 100, bench["ma200"][f"dd_cagr_{per}"] * 100
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.1f}%\n({mat[i, j] - mix:+.1f} / {mat[i, j] - ma:+.1f})", ha="center", va="center",
                                fontsize=7.5, color="white" if mat[i, j] < np.nanmean(mat) else "black")
            ax.set_xticks(range(len(cols)), cols, fontsize=8)
            ax.set_yticks(range(len(ns.SLOTS_N)), [f"N={n}" for n in ns.SLOTS_N], fontsize=8)
            ax.set_title(f"{per.replace('-', '–')}: SPY+국채 {mix:.1f}% · 200일선 {ma:.1f}%", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
        what = "자리 수 × 비중 (무작위 순)" if grid == "A" else "자리 수 × 우선순위 (W2)"
        fig.suptitle(f"격자 {grid}: {what} — 낙폭 15% 맞춤 수익 (M1 계좌, 매수 15 × 매도 4 = 60칸 중앙값; 괄호 = SPY+국채 / 200일선 대비 %p)",
                     fontsize=10.5)
        made.append(figures._save(fig, out / "figures" / f"grid_{grid}_goal.png"))
    return made


def fig_scatter(out: Path, cells: pd.DataFrame, btab: pd.DataFrame) -> Path:
    """② Plain MDD vs dd_cagr (W2 / W3 cells, M1) and plain MDD vs w*, with the benchmarks."""
    plt = figures._plt()
    g = cells[(cells["gen"] != "E0") & cells["W"].isin(ns.CANDIDATE_W)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    sc_ = axes[0].scatter(g["mdd_1993-2023"] * 100, g["dd_cagr_1993-2023"] * 100, c=g["w"], cmap="viridis", s=9, alpha=0.7)
    fig.colorbar(sc_, ax=axes[0], label="w* (쓸 수 있는 자본 비율)")
    marks = {"mix": ("*", "#c0392b"), "ma200": ("D", "#8e44ad"), "ew": ("s", "#555")}
    for _, b in btab.iterrows():
        m, col = marks[b["benchmark"]]
        axes[0].scatter([b["mdd_1993-2023"] * 100], [b["dd_cagr_1993-2023"] * 100], marker=m, s=160, color=col, label=b["label"])
        if b["benchmark"] in GATED:
            axes[0].axhline(b["dd_cagr_1993-2023"] * 100, color=col, ls="--", lw=0.9)
    axes[0].set_xlabel("보통 MDD 1993–2023 (%)")
    axes[0].set_ylabel("낙폭 15% 맞춤 수익 1993–2023 (%)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(g["mdd_1993-2023"] * 100, g["w"], s=9, alpha=0.6, color=figures.COLORS[0], label="설정 × 매수 × 매도")
    xx = np.linspace(-80, -10, 50)
    axes[1].plot(xx, np.minimum(1.0, 15 / np.abs(xx)), color="#999", lw=1, ls=":", label="15 ÷ |MDD| (단순 비례)")
    for _, b in btab.iterrows():
        m, col = marks[b["benchmark"]]
        axes[1].scatter([b["mdd_1993-2023"] * 100], [b["w"]], marker=m, s=160, color=col, label=b["label"])
    axes[1].set_xlabel("보통 MDD 1993–2023 (%)")
    axes[1].set_ylabel("w*")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    axes[1].grid(alpha=0.3)
    fig.suptitle("낙폭 15%에 맞추면: 보통 낙폭이 깊을수록 쓸 수 있는 자본(w*)이 줄고, 맞춘 수익이 기준선과 어디에 있는가 (W2 · W3, M1 계좌)", fontsize=10.5)
    return figures._save(fig, out / "figures" / "mdd_vs_goal.png")


def fig_exits(out: Path, ex: pd.DataFrame, bench: dict) -> Path:
    """③ Q2: the 11 exits at N=10 / W2 / P0, plain CAGR vs drawdown-matched return."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    y = np.arange(len(ex))
    for ax, (col, title) in zip(axes, (("cagr_1993-2023", "보통 CAGR 1993–2023"), ("dd_cagr_1993-2023", "낙폭 15% 맞춤 수익 1993–2023"))):
        ax.barh(y - 0.27, ex[f"rule_{col}"] * 100, height=0.27, color=figures.COLORS[0], label="M1 계좌 (매수 15개 중앙값)")
        ax.barh(y, ex[f"none_{col}"] * 100, height=0.27, color=figures.COLORS[1], label="제외 없음")
        ax.barh(y + 0.27, ex[f"e0_{col}"] * 100, height=0.27, color="#999", label="무작위 매수 E0 (M1)")
        if col.startswith("dd"):
            for b in GATED:
                ax.axvline(bench[b]["dd_cagr_1993-2023"] * 100, color="#c0392b" if b == "mix" else "#8e44ad", ls="--", lw=1,
                           label=BENCHMARKS[b])
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [f"{x} {lab}" for x, lab in zip(ex["exit"], ex["label"])], fontsize=7.5)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Q2 매도 축 (N=10 · W2 · P0): 낙폭을 맞추면 매도의 순서가 바뀌는가", fontsize=11)
    return figures._save(fig, out / "figures" / "exits_goal.png")


def fig_windows(out: Path, win: pd.DataFrame, sysm: pd.DataFrame, bench: dict, wins: list) -> Path | None:
    """④ 3-year windows: the best system (setting x exit, M1) vs the two gated benchmarks."""
    plt = figures._plt()
    s = sysm[sysm["kind"] == "rule"].sort_values("dd_cagr_1993-2023", ascending=False)
    if s.empty:
        return None
    best = s.iloc[0]
    w = win[(win["setting"] == best["setting"]) & (win["exit"] == best["exit"]) & (win["kind"] == "rule") & (win["gen"] != "E0")
            & (win["cost"] == pj.COST_BPS)]
    sysw = w.groupby("w_idx")["win_cagr"].median().reindex(range(len(wins)))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(wins))
    ax.plot(x, sysw.to_numpy() * 100, marker="o", color=figures.COLORS[0], label=f"시스템 {best['setting']} × {best['exit']} (매수 15개 중앙값)")
    for b, col in (("mix", "#c0392b"), ("ma200", "#8e44ad")):
        bw = bench[b]["win_cagr"] * 100
        share = float((sysw.to_numpy() * 100 >= bw).mean())
        ax.plot(x, bw, marker="s", ms=4, color=col, label=f"{BENCHMARKS[b]} (시스템이 앞선 창 {share:.0%})")
    ax.set_xticks(x, [f"{a}–{str(b)[2:]}" for a, b in wins], rotation=90, fontsize=7.5)
    ax.set_ylabel("3년 창 CAGR, 낙폭 15% 맞춤 (%)")
    ax.axhline(0, color="#999", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("3년 롤링 창: 낙폭 맞춤 수익이 가장 높은 시스템 vs 같은 낙폭의 기준선 (각자 전체 기간 w*)", fontsize=10.5)
    return figures._save(fig, out / "figures" / "windows_goal.png")


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "r0x1", "tables", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--stage-n", default=None, help="Stage N 결과표 폴더 (재현 점검)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="E2·V3·E0(씨앗 2), 자리 수 10·20, 매도 축은 E2·V3")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.dirs = {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events),
              "r0x1": Path(a.r0x1), "tables": Path(a.tables)}
    gens = ms.generators(["E2", "V3"], pj.E0_SEEDS[:2]) if a.dry else ms.generators()
    slots = ns.SLOTS_N[:2] if a.dry else ns.SLOTS_N
    a.cfg = {"bins": pj.BINS_J3, "seeds": pj.E0_SEEDS, "base": str(out / "base.parquet")}
    ts.log(f"1단계: 신호 기준표 (매수 {len(gens)} × 매도 {len(EXIT_AXIS)})")
    saved = list(ns.EXITS)
    ns.EXITS[:] = [x for x in EXIT_AXIS if not xs.EXITS_L[x].get("partial")]  # every exit of the axis (TP = the H1 table)
    base = ns.build_base(a, gens)
    ns.EXITS[:] = saved
    ts.log(f"  신호 {len(base):,}개")
    size = base.groupby("src").size()
    jobs = [(gen, x, n, pj.COST_BPS, srcs, "grid") for gen, srcs in gens.items() for x in EXITS for n in slots]
    jobs += [(gen, x, 10, pj.COST_BPS, srcs, "axis") for gen, srcs in gens.items() for x in EXIT_AXIS if x not in EXITS]
    jobs.sort(key=lambda j: -int(size.reindex(j[4]).sum()) * (1 + j[2] / 50) * (3 if j[5] == "grid" else 1))
    jdir = out / "jobs"
    todo = [j for j in jobs if not (jdir / f"{job_name(j[0], j[1], j[2], j[3], j[5])}_acc.csv").exists()]
    ts.log(f"2단계: 작업 {len(todo)}개 남음 (격자 {sum(j[5] == 'grid' for j in jobs)}, 매도 축 {sum(j[5] == 'axis' for j in jobs)})")
    run_jobs(todo, a)
    read = lambda s: pd.concat([pd.read_csv(f) for f in sorted(jdir.glob(f"*_{s}.csv"))  # noqa: E731
                                if "bp_" not in f.name and f.stat().st_size > 2], ignore_index=True)
    acc, win, mom = read("acc"), read("win"), read("mom")
    pj._init(a.panel, a.eligible, a.dirs, a.cfg)
    btab, bench = benchmarks_table(pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"])
    btab.to_csv(out / "benchmarks.csv", index=False)
    g = summarize(acc)
    if a.stage_n:
        rp = repro(g, Path(a.stage_n))
        rp.to_csv(out / "repro.csv", index=False)
        bad = int((rp["diff"] > 1e-9).sum()) if len(rp) else -1
        ts.log(f"재현 점검: Stage N 대비 {rp[['setting', 'gen', 'exit', 'kind']].drop_duplicates().shape[0]}칸 {len(rp)}값, "
               f"최대 차 {rp['diff'].max():.2e}, 1e-9 넘는 값 {bad}개")
    sysm = systems(g, bench)
    cells, _ = candidates(g, bench, {})
    need = cells[(cells["gen"] != "E0") & cells["W"].isin(ns.CANDIDATE_W) & cells["cell_pass"]]
    per = need.groupby(["setting", "gen"])["cell_pass"].sum()
    rows40 = need.set_index(["setting", "gen"]).loc[per[per >= MIN_EXITS].index].reset_index() if (per >= MIN_EXITS).any() else need.iloc[:0]
    pass40 = {}
    if len(rows40):  # the cells of setting x entry pairs with >= 2 passing exits, again at 40 bp (M1 only)
        want = sorted({(r.gen, r.exit, int(r.N)) for r in rows40.itertuples()})
        run_jobs([(gen, x, n, pj.CHECK_COST_BPS, gens[gen], "rule40") for gen, x, n in want
                  if not (jdir / f"{job_name(gen, x, n, pj.CHECK_COST_BPS, 'rule40')}_acc.csv").exists()], a)
        a40 = pd.concat([pd.read_csv(jdir / f"{job_name(gen, x, n, pj.CHECK_COST_BPS, 'rule40')}_acc.csv") for gen, x, n in want], ignore_index=True)
        g40 = summarize(a40)
        for _, r in g40.iterrows():
            pass40[(r["setting"], r["gen"], r["exit"])] = cell_pass(r, bench)
    cells, cand = candidates(g, bench, pass40)
    ch = chance(acc, cells, bench)
    ex = exit_axis(g)
    curves = bench_curves(pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"])
    rm = curves["mix"][1:] / curves["mix"][:-1] - 1.0
    bl = bench["mix"]["w"] * rm + (1 - bench["mix"]["w"]) * pj._W["cash"][1:]  # the SPY mix's daily blend, for PSR
    dsr = psr_dsr(cells, cand, mom, float(bl.mean() / bl.std())) if cand["candidate"].any() else pd.DataFrame()
    g.to_csv(out / "goal.csv", index=False)
    win.groupby(["setting", "gen", "exit", "kind", "cost", "w_idx", "a", "b"])["win_cagr"].median().reset_index().to_csv(out / "windows.csv", index=False)
    sysm.to_csv(out / "systems.csv", index=False)
    cells.to_csv(out / "cells.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    ch.to_csv(out / "chance.csv", index=False)
    ex.to_csv(out / "exits.csv", index=False)
    s_rule = sysm[sysm["kind"] == "rule"]
    summary = {
        "benchmarks": {k: {c: bench[k][c] for c in ["w", "dd_cagr_1993-2023", "dd_cagr_1993-2007", "dd_cagr_2008-2023", "dd_cagr_2017-2023",
                                                     "cagr_1993-2023", "mdd_1993-2023"]} for k in BENCHMARKS},
        "windows": len(_setup(pj._W["cash"], pj._W["wdates"])["wins"]),
        "systems_passing": s_rule.loc[s_rule["system_pass"], ["setting", "exit"]].to_dict("records"),
        "systems_passing_none": sysm.loc[(sysm["kind"] == "none") & sysm["system_pass"], ["setting", "exit"]].to_dict("records"),
        "best_system": s_rule.sort_values("dd_cagr_1993-2023", ascending=False).head(5)[["setting", "exit", "dd_cagr_1993-2023", "w"]].to_dict("records"),
        "cells_passing": int(cells.loc[cells["gen"] != "E0", "cell_pass"].sum()),
        "candidates": cand.loc[cand["candidate"], ["setting", "gen", "entry_specific"]].to_dict("records"),
        "chance_beyond": ch.loc[ch["beyond_chance"], ["setting", "exit", "passing", "q95"]].to_dict("records"),
        "dsr": dsr.to_dict("records"), "trials": {"total": N_TRIALS, "new": 0},
    }
    (out / "chance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if figures.available():
        fig_grids(out, cells, bench)
        fig_scatter(out, cells, btab)
        if len(ex):
            fig_exits(out, ex, bench)
        fig_windows(out, win, sysm, bench, _setup(pj._W["cash"], pj._W["wdates"])["wins"])
    ts.log(f"완료 · 기준 w* SPY+국채 {bench['mix']['w']:.2f} ({bench['mix']['dd_cagr_1993-2023']:.2%}) · 200일선 {bench['ma200']['w']:.2f} "
           f"({bench['ma200']['dd_cagr_1993-2023']:.2%}) · 목표 통과 시스템 {len(summary['systems_passing'])} · 통과 칸 {summary['cells_passing']} · "
           f"후보 {len(summary['candidates'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
