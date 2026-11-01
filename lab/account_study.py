"""Stage N: account and sizing rules (DECISIONS 93).

    python -m lab.account_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                                --r0x1 DIR --tables DIR --out DIR [--stage-m DIR] [--workers N] [--feature-workers N] [--dry]

The same buy signals (E1-E9, V1-V6 and the random buys E0, seeds 1-5, from the stored Stage L tables;
--tables = Stage L's tables/ folder), the M1 exclusion of Stage M (lab/exclude_study.py) and the exits
X1, H1, H9, ST, replayed (pick_study.replay_risk, no engine run) under 20 account settings: slots N in
{10, 20, 30, 50} x sizing W in {W1 equal 1/N, W2 min(r / stop, 1/N), W3 min(r / stop, 2/N)} with
r = 10% / N at random order (grid A), and N x fill priority P in {P0 random, P1 lowest idiosyncratic
vol first, P2 largest cap first} at W2 (grid B). Stop distance for sizing = ATR(14) x 3 (ST: x 5;
X1: a sizing distance only). Random orders reuse Stage M's random draws (its M1 rule and pseudo
exclusions 0-9) and Stage L's (no exclusion), so N=10 / W2 / P0 (X1: W1) gives Stage M's numbers;
--stage-m checks them (repro.csv). Per setting: the M1 account, 10 day-matched pseudo exclusions, the
no-exclusion account and E0 under the same setting; Q1 transfer, Q2 entries vs E0, Q3 candidates
(neighbour N, 40 bp), per-signal statistics of the bought signals, account statistics, figures.
Details not fixed by the design and set before the run: the Q1 count uses 1993-2023 CAGR above the
95th percentile of the cell's 10 pseudo medians; Q2's p0 pools the seed leave-one-out tests of all
20 settings; per-signal statistics compare the first random order (P1 / P2: the single order) with
N=10 / W2 / P0's first order, with order 1 vs order 0 at N=10 / W2 / P0 as the null.
Output (aggregates only): base.parquet, jobs/*, verdicts.csv, settings.csv, shown.csv, stats.csv,
hist.csv, accounts.csv, repro.csv, chance.json, baselines.csv, figures/.
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

from . import entry_study as es, exclude_study as ms, figures, pair_study as pj, pick_study as ph, select_study as ks, tail_study as ts
from .engine import SignalCache

# ---- registered (DECISIONS 93): fixed before any result
SLOTS_N = (10, 20, 30, 50)
RISK_TOTAL = 0.10  # per-name risk r = RISK_TOTAL / N (N = 10: the 1% rule)
SIZING = {"W1": ("동일 비중 (비교용)", None), "W2": ("1% 룰 (상한 1/N)", 1.0), "W3": ("1% 룰, 상한 넓힘 (2/N)", 2.0)}
PRIORITY = {"P0": "무작위 순", "P1": "덜 흔들리는 종목 먼저", "P2": "큰 회사 먼저"}
EXITS = ["X1", "H1", "H9", "ST"]
SIZE_STOP = {"X1": 3.0, "H1": 3.0, "H9": 3.0, "ST": 5.0}  # ATR(14) multiple for sizing (X1: sizing only)
RULE = "M1"
BASE = (10, "W2", "P0")
N_ORDERS, N_PSEUDO, PSEUDO_ORDERS, N_NONE = 100, 10, 20, 100  # E0: split over its 5 seeds
MIN_EXITS_Q2, MIN_EXITS_Q3, Q1_PP = 3, 2, 0.005
CANDIDATE_W = ("W2", "W3")
ACCOUNTS = (100_000.0, 1_000_000.0)
CAPACITY_LIMIT = 0.05  # order above 5% of the 20-day average dollar volume (as Stage D)
N_TRIALS_BEFORE = ms.N_TRIALS_BEFORE + ms.N_NEW  # 70,574
N_NEW = 1_200  # 20 settings x 15 buy rules x 4 exits
FIG_EXIT = "H1"
SEED = 93
_W: dict = {}


def settings() -> list[tuple[int, str, str]]:
    """Grid A (N x W at P0) then grid B's new settings (N x P1 / P2 at W2): 20 distinct."""
    return [(n, w, "P0") for n in SLOTS_N for w in SIZING] + [(n, "W2", p) for n in SLOTS_N for p in ("P1", "P2")]


def setting_name(s: tuple) -> str:
    return f"N{s[0]}_{s[1]}_{s[2]}"


def parse_setting(name: str) -> tuple[int, str, str]:
    n, w, p = name.split("_")
    return int(n[1:]), w, p


def grids(s: tuple) -> list[str]:
    return (["A"] if s[2] == "P0" else []) + (["B"] if s[1] == "W2" else [])


def neighbours(s: tuple) -> list[tuple[int, str, str]]:
    """Settings with the same sizing and priority and the adjacent slot count."""
    i = SLOTS_N.index(s[0])
    return [(SLOTS_N[j], s[1], s[2]) for j in (i - 1, i + 1) if 0 <= j < len(SLOTS_N)]


def base_setting(x: str) -> tuple[int, str, str]:
    """The setting that equals Stage M's account under exit x (X1 there was a fixed 10% weight)."""
    return (10, "W1", "P0") if x == "X1" else BASE


def sizing(stop_pct: np.ndarray, n: int, w: str) -> tuple[np.ndarray, np.ndarray]:
    """(weight, bound by the cap) per signal: W1 1/N; W2 / W3 min(r / stop, cap) with r = 10% / N."""
    if SIZING[w][1] is None:
        return np.full(len(stop_pct), 1.0 / n), np.full(len(stop_pct), np.nan)
    r, cap = RISK_TOTAL / n, SIZING[w][1] / n
    raw = np.where(stop_pct > 0, r / np.where(stop_pct > 0, stop_pct, 1.0), np.nan)
    return np.minimum(raw, cap), np.where(np.isfinite(raw), raw >= cap, np.nan)


# --------------------------------------------------------------------------- stage 1: base table

def point_values(job):
    """A lab/select_study.py feature or cap rank at (si, sj), or the 20-day average dollar volume."""
    key, si, sj = job
    if key != "dv20":
        return ks.feature_values(job)
    t0 = time.time()
    panel = ks._F["panel"]
    dv = SignalCache(panel).dollar_volume(20)
    return key, dv.to_numpy(dtype="float32")[si, sj], time.time() - t0


def build_base(a, gens: dict) -> pd.DataFrame:
    """Stage M's base rows (src, si, sj, four features, cap rank) plus the 20-day dollar volume."""
    f = Path(a.out) / "base.parquet"
    if f.exists():
        return pd.read_parquet(f)
    ms.xs._init(a.panel, a.eligible, a.dirs, a.cfg)
    cols = pj._W["px"].cols
    parts = []
    for srcs in gens.values():
        for s in srcs:
            for x in EXITS:
                t = pj.rule_table(s, x)
                parts.append(pd.DataFrame({"src": s, "si": t["si"].to_numpy(), "sj": cols.get_indexer(t["symbol"].astype(str))}))
            pj._W["tables"].clear()
    base = pd.concat(parts, ignore_index=True).drop_duplicates(["src", "si", "sj"])
    base = base.sort_values(["src", "si", "sj"], kind="stable").reset_index(drop=True)
    pts = base[["si", "sj"]].drop_duplicates().reset_index(drop=True)
    si, sj = pts["si"].to_numpy(), pts["sj"].to_numpy()
    with ProcessPoolExecutor(max_workers=a.feature_workers, initializer=ks._f_init, initargs=(a.panel, a.eligible)) as ex:
        for key, vals, sec in ex.map(point_values, [(k, si, sj) for k in list(ms.FEATURES) + ["cap_rank", "dv20"]], chunksize=1):
            pts[key] = vals
            ts.log(f"  {key}: {sec:.0f}s, 값 있음 {np.isfinite(vals).mean():.1%}")
    base = base.merge(pts, on=["si", "sj"], how="left")
    base.to_parquet(f)
    return base


# --------------------------------------------------------------------------- stage 2: per buy rule x exit x slots

def _init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    ms._init(panel_dir, eligible, dirs, cfg)
    _W.update(ms._W)
    _W["years"] = (_W["wdates"][-1] - _W["wdates"][0]).days / 365.25


class Rows:
    """The arrays of one source's signals under one exit that a replay needs (window indices)."""

    def __init__(self, d: pd.DataFrame, b: pd.DataFrame, x: str):
        w0, nd = _W["w0"], _W["n"]
        px = _W["px"]
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
        atr = px.ATR[d["si"].to_numpy(), self.sj].astype(float)
        entry = px.O[d["ei"].to_numpy(), self.sj].astype(float)
        self.stop = SIZE_STOP[x] * atr / entry
        self.w_stage = d["w"].to_numpy(dtype=float)  # Stage L / M weight (a check of sizing())

    def __len__(self):
        return len(self.ei)


def _replay(r: Rows, mask: np.ndarray, w: np.ndarray, key: np.ndarray, n: int, cost: float, log: dict | None = None):
    """One account on the rows of `mask` (signals with a weight); returns (equity, trades, row index)."""
    m = mask & np.isfinite(w)
    idx = np.flatnonzero(m)
    eq, tr = ph.replay_risk(r.ei[idx] - 1, r.ei[idx], r.sj[idx], r.sale[idx], r.closed[idx], w[idx], key[idx],
                            _W["O"], _W["LC"], _W["cash"], cost, slots=n, log=log)
    return eq, tr, idx[tr["row"].to_numpy()] if len(tr) else np.array([], dtype=int)


def _stats(eq: np.ndarray, tr: pd.DataFrame, rows: np.ndarray, r: Rows) -> dict:
    d = np.diff(eq) / eq[:-1]
    return {"trades": len(tr), "sharpe": float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else np.nan,
            "cap_rank_trades": float(np.nanmedian(r.cap[rows])) if len(rows) else np.nan,
            **es.period_stats(pd.Series(eq, index=_W["wdates"]))}


def _shown(eq: np.ndarray, tr: pd.DataFrame, rows: np.ndarray, r: Rows, log: dict, capped: np.ndarray, n_rows: int) -> dict:
    """Account statistics of one replay: positions, cash, fill rate, weights, turnover, capacity."""
    out = {"positions_mean": float(np.mean(log["positions"])), "cash_share": float(1.0 - np.mean(log["invested"])),
           "fill_rate": len(tr) / max(n_rows, 1), "weight_mean": float(tr["weight"].mean()) if len(tr) else np.nan,
           "weight_p10": float(tr["weight"].quantile(0.1)) if len(tr) else np.nan,
           "weight_p90": float(tr["weight"].quantile(0.9)) if len(tr) else np.nan,
           "capped_share": float(np.nanmean(capped[rows])) if len(rows) and np.isfinite(capped[rows]).any() else np.nan,
           "turnover": float(tr["weight"].sum() / _W["years"]) if len(tr) else 0.0}
    out["rule_working_share"] = 1.0 - out["capped_share"] if np.isfinite(out["capped_share"]) else np.nan
    if len(tr):
        prev = np.maximum(tr["entry_i"].to_numpy() - 1, 0)
        for acct in ACCOUNTS:
            share = acct * eq[prev] * tr["weight"].to_numpy() / r.dv[rows]
            share = share[np.isfinite(share)]
            tag = f"{int(acct / 1000)}k"
            out[f"order_share_median_{tag}"] = float(np.median(share)) if len(share) else np.nan
            out[f"order_share_over5_{tag}"] = float((share > CAPACITY_LIMIT).mean()) if len(share) else np.nan
    return out


def _means(r: Rows, mask: np.ndarray, w: np.ndarray) -> dict:
    """No-limit contribution (weight x return) and return per period over the rows with a weight."""
    m = mask & np.isfinite(w)
    out = {"signals": int(m.sum())}
    for p, (a, b) in es.PERIODS.items():
        mm = m & (r.year >= a) & (r.year <= b)
        out[f"n_{p}"] = int(mm.sum())
        out[f"mean_contrib_{p}"] = float((w[mm] * r.ret[mm]).mean()) if mm.any() else np.nan
        out[f"mean_ret_{p}"] = float(np.nanmean(r.ret[mm])) if mm.any() else np.nan
    return out


def stage_m_draws(gen: str, x: str, srcs: list[str], keeps: dict, pseudos: dict, n_total: dict):
    """Stage M's random draws for M1 in its order (lab/exclude_study.run_job): per source its rule
    orders, then 20 pseudo exclusions x their orders; yields ('rule' | 'pseudo', k, src, order, key)
    for the rule and pseudo 0..N_PSEUDO-1 (the others are drawn and dropped to stay in step)."""
    rng = np.random.default_rng([ms.SEED, zlib.crc32(f"{RULE}|{gen}|{x}|{pj.COST_BPS}".encode())])
    for i, s in enumerate(srcs):
        nk = int(keeps[s].sum())
        for o in range(ms.N_ORDERS // len(srcs)):
            yield "rule", -1, s, o, _spread(rng.random(nk), keeps[s], n_total[s])
        for k in range(ms.N_PSEUDO):
            if k >= N_PSEUDO and i == len(srcs) - 1:
                return
            npk = int(pseudos[s][k].sum())
            for o in range(ms.PSEUDO_ORDERS // len(srcs)):
                key = rng.random(npk)
                if k < N_PSEUDO:
                    yield "pseudo", k, s, o, _spread(key, pseudos[s][k], n_total[s])


def _spread(key: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """A key drawn for the rows of `mask` placed on all rows (others NaN, never bought)."""
    out = np.full(n, np.nan)
    out[mask] = key
    return out


def run_job(job):
    """One buy rule x exit x slot count: the 3 random-order sizings and P1 / P2 at W2 (M1 rule, pseudo
    exclusions 0-9, no exclusion), contributions, account statistics and bought-signal returns."""
    gen, x, n_slots, cost_bps, srcs, only_rule = job
    t0 = time.time()
    cost = cost_bps / 10_000
    sets = [s for s in settings() if s[0] == n_slots]
    rows = {s: Rows(ms.prepared(s, x), ms.source_base(s), x) for s in srcs}
    keeps, pseudos = {}, {}
    for s in srcs:
        b = ms.source_base(s)
        keep_b = ~ms.excluded(b, RULE)
        keeps[s] = keep_b[rows[s].brow]
        pseudos[s] = [ks.pseudo_mask(b["si"].to_numpy(), keep_b, np.random.default_rng([ms.SEED, zlib.crc32(f"{RULE}|{s}".encode()), k]))[rows[s].brow]
                      for k in range(ms.N_PSEUDO)]
    wts = {s: {w: sizing(rows[s].stop, n_slots, w) for w in SIZING} for s in srcs}
    if n_slots == 10:  # N=10: W2 (X1: W1) is Stage L / M's weight, bit for bit
        for s in srcs:
            same = wts[s]["W1" if x == "X1" else "W2"][0]
            if not np.array_equal(same, rows[s].w_stage):
                raise ValueError(f"{s} {x}: N=10 비중이 Stage M과 다름")
    accs, uns, shown, bought, hists, mom = [], [], [], [], [], []
    curves: dict = {}

    def record(st, kind, k, s, o, eq, tr, idx, capped=None, log=None):
        r = rows[s]
        accs.append({"setting": setting_name(st), "kind": kind, "k": k, "src": s, "order": o, **_stats(eq, tr, idx, r)})
        if kind == "rule" and cost_bps == pj.COST_BPS:
            curves.setdefault(st, []).append(eq)
            hh = hist_acc.setdefault(st, {p: np.zeros(len(pj.RET_EDGES) - 1) for p in pj.HIST_PERIODS})
            for p in pj.HIST_PERIODS:
                a_, b_ = es.PERIODS[p]
                rr = r.ret[idx][(r.year[idx] >= a_) & (r.year[idx] <= b_)]
                hh[p] += np.histogram(rr[np.isfinite(rr)], pj.RET_EDGES)[0]
            n_rep[st] = n_rep.get(st, 0) + 1
            if gen != "E0" and (o == 0 or (o == 1 and st == BASE)):  # first order (and the null's second)
                bought.append(pd.DataFrame({"setting": setting_name(st), "order": o, "src": s, "ret": r.ret[idx].astype("float32"),
                                            "year": r.year[idx].astype("int16")}))
        if log is not None:
            shown.append({"setting": setting_name(st), "kind": kind, "src": s,
                          **_shown(eq, tr, idx, r, log, capped, int((keeps[s] if kind == "rule" else np.ones(len(r), bool)).sum()))})

    hist_acc: dict = {}
    n_rep: dict = {}
    p0 = [s for s in sets if s[2] == "P0"]
    # random order: Stage M's draws (M1 rule, pseudo 0-9) and Stage L's (no exclusion)
    n_total = {s: len(rows[s]) for s in srcs}
    for kind, k, s, o, key in stage_m_draws(gen, x, srcs, keeps, pseudos, n_total):
        if only_rule and kind != "rule":
            continue
        mask = keeps[s] if kind == "rule" else pseudos[s][k]
        for st in p0:
            w, capped = wts[s][st[1]]
            log = {} if kind == "rule" and o == 0 and cost_bps == pj.COST_BPS else None
            eq, tr, idx = _replay(rows[s], mask, w, key, n_slots, cost, log)
            record(st, kind, k, s, o, eq, tr, idx, capped, log)
    if not only_rule:
        for s in srcs:
            name = pj.set_name(("e0full", int(s[3:])) if s.startswith("R0s") else ("single", s))
            rng = np.random.default_rng([ms.xs.SEED, zlib.crc32(f"{name}|{x}|{pj.COST_BPS}".encode())])
            allm = np.ones(len(rows[s]), dtype=bool)
            for o in range(N_NONE // len(srcs)):
                key = rng.random(len(rows[s]))
                for st in p0:
                    w, capped = wts[s][st[1]]
                    log = {} if o == 0 else None
                    eq, tr, idx = _replay(rows[s], allm, w, key, n_slots, cost, log)
                    record(st, "none", -1, s, o, eq, tr, idx, capped, log)
    # fixed priorities: one replay each
    for st in [s for s in sets if s[2] != "P0"]:
        for s in srcs:
            r = rows[s]
            tie = np.random.default_rng([SEED, zlib.crc32(f"tie|{s}|{x}".encode())]).random(len(r))
            key = r.k2 + tie * 1e-9 if st[2] == "P1" else -r.cap + tie * 1e-6
            w, capped = wts[s][st[1]]
            log = {} if cost_bps == pj.COST_BPS else None
            eq, tr, idx = _replay(r, keeps[s], w, key, n_slots, cost, log)
            record(st, "rule", -1, s, 0, eq, tr, idx, capped, log)
            if only_rule:
                continue
            for k in range(N_PSEUDO):
                eq, tr, idx = _replay(r, pseudos[s][k], w, key, n_slots, cost)
                record(st, "pseudo", k, s, 0, eq, tr, idx)
            eq, tr, idx = _replay(r, np.ones(len(r), dtype=bool), w, key, n_slots, cost)
            record(st, "none", -1, s, 0, eq, tr, idx)
    if cost_bps == pj.COST_BPS and not only_rule:
        for st in sets:
            w = st[1]
            alls = {s: np.ones(len(rows[s]), dtype=bool) for s in srcs}
            for kind, masks in (("rule", {-1: keeps}), ("pseudo", {k: {s: pseudos[s][k] for s in srcs} for k in range(N_PSEUDO)}),
                                ("none", {-1: alls})):
                for k, mk in masks.items():
                    parts = [(rows[s], mk[s], wts[s][w][0]) for s in srcs]
                    uns.append({"setting": setting_name(st), "kind": kind, "k": k, **_pooled_means(parts)})
            for p, h in hist_acc.get(st, {}).items():
                hists.append({"setting": setting_name(st), "kind": "rule", "period": p,
                              **{f"h{i}": float(c) / max(n_rep.get(st, 1), 1) for i, c in enumerate(h)}})
            cv = curves.get(st, [])
            if cv:
                c = [es.period_stats(pd.Series(e, index=_W["wdates"]))["cagr_1993-2023"] for e in cv]
                mom.append({"setting": setting_name(st), **ks._moments(cv[int(np.argsort(c)[len(c) // 2])])})
    ms._W["prep"].clear()
    pj._W["tables"].clear()
    head = {"gen": gen, "exit": x, "cost": cost_bps}
    frame = lambda rs: pd.DataFrame([{**head, **r} for r in rs])  # noqa: E731
    b = pd.concat(bought, ignore_index=True).assign(gen=gen, exit=x) if bought else pd.DataFrame()
    return job, frame(accs), frame(uns), frame(shown), frame(hists), frame(mom), b, time.time() - t0


def _pooled_means(parts: list) -> dict:
    """_means over several sources (E0 seeds) pooled by signal counts."""
    ms_ = [_means(r, m, w) for r, m, w in parts]
    out = {"signals": sum(m["signals"] for m in ms_)}
    for p in es.PERIODS:
        n = sum(m[f"n_{p}"] for m in ms_)
        out[f"n_{p}"] = n
        for c in ("mean_contrib", "mean_ret"):
            v = [(m[f"{c}_{p}"], m[f"n_{p}"]) for m in ms_ if m[f"n_{p}"] and np.isfinite(m[f"{c}_{p}"])]
            out[f"{c}_{p}"] = float(sum(a * b for a, b in v) / sum(b for _, b in v)) if v else np.nan
    return out


def job_name(gen: str, x: str, n: int, cost: float) -> str:
    return f"{gen}__{x}__N{n}" + ("" if cost == pj.COST_BPS else f"__{cost:g}bp")


def run_jobs(jobs: list, a) -> None:
    d = Path(a.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs, a.cfg)) as ex:
        for i, (job, acc, un, sh, hist, mom, bought, sec) in enumerate(ex.map(run_job, jobs, chunksize=1)):
            nm = job_name(job[0], job[1], job[2], job[3])
            for suffix, f in (("un", un), ("shown", sh), ("hist", hist), ("mom", mom)):
                f.to_csv(d / f"{nm}_{suffix}.csv", index=False)
            if len(bought):
                bought.to_parquet(d / f"{nm}_bought.parquet")
            acc.to_csv(d / f"{nm}_acc.csv", index=False)  # last: its presence marks the job done
            if (i + 1) % 5 == 0 or i + 1 == len(jobs):
                ts.log(f"  [{i + 1}/{len(jobs)}] {nm} {sec:.0f}s · 남은 예상 {(len(jobs) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


# --------------------------------------------------------------------------- verdicts

PERIODS = list(es.PERIODS)


def summarize(acc: pd.DataFrame, un: pd.DataFrame) -> pd.DataFrame:
    """Per setting x buy rule x exit: the M1 account's medians, the pseudo exclusions' medians (each
    pseudo's own median, then the median and 95th percentile over the 10), no exclusion, contributions."""
    keys = ["setting", "gen", "exit"]
    med = acc.groupby(keys + ["kind", "k"]).median(numeric_only=True)
    rows = []
    for (st, gen, x), g in acc.groupby(keys, sort=False):
        mg = med.loc[(st, gen, x)]
        r = mg.loc[("rule", -1)]
        p = mg.loc["pseudo"] if "pseudo" in mg.index.get_level_values(0) else None
        nb = mg.loc[("none", -1)] if ("none", -1) in mg.index else None
        u = un[(un["setting"] == st) & (un["gen"] == gen) & (un["exit"] == x)]
        ur = u[u["kind"] == "rule"]
        up = u[u["kind"] == "pseudo"]
        un_ = u[u["kind"] == "none"]
        n, w, pr = parse_setting(st)
        row = {"setting": st, "N": n, "W": w, "P": pr, "gen": gen, "exit": x, "trades": r["trades"], "sharpe": r["sharpe"],
               "cap_rank_trades": r["cap_rank_trades"], "none_trades": nb["trades"] if nb is not None else np.nan,
               "pseudo_sharpe": float(p["sharpe"].median()) if p is not None else np.nan}
        for per in PERIODS:
            row[f"cagr_{per}"], row[f"mdd_{per}"] = r[f"cagr_{per}"], r[f"mdd_{per}"]
            pk = p[f"cagr_{per}"].to_numpy(dtype=float) if p is not None else np.array([])
            row[f"pseudo_cagr_{per}"] = float(np.median(pk)) if len(pk) else np.nan
            row[f"pseudo_q95_{per}"] = float(np.percentile(pk, 95)) if len(pk) else np.nan
            row[f"none_cagr_{per}"] = nb[f"cagr_{per}"] if nb is not None else np.nan
            row[f"none_mdd_{per}"] = nb[f"mdd_{per}"] if nb is not None else np.nan
            row[f"contrib_{per}"] = float(ur[f"mean_contrib_{per}"].iloc[0]) if len(ur) else np.nan
            row[f"pseudo_contrib_{per}"] = float(up[f"mean_contrib_{per}"].median()) if len(up) else np.nan
            row[f"none_contrib_{per}"] = float(un_[f"mean_contrib_{per}"].iloc[0]) if len(un_) else np.nan
        rows.append(row)
    v = pd.DataFrame(rows)
    e0 = v[v["gen"] == "E0"].set_index(["setting", "exit"])
    for per in PERIODS:
        v[f"e0_cagr_{per}"] = [e0[f"cagr_{per}"].get((s, x), np.nan) for s, x in zip(v["setting"], v["exit"])]
        v[f"x_{per}"] = (v[f"cagr_{per}"] - v[f"pseudo_cagr_{per}"]) * 100
    v["beyond_q95"] = v["cagr_1993-2023"] > v["pseudo_q95_1993-2023"]
    v["beats_e0"] = np.logical_and.reduce([v[f"cagr_{h}"] > v[f"e0_cagr_{h}"] for h in es.HALVES])
    return v


def pseudo_vs_pseudo(acc: pd.DataFrame) -> pd.DataFrame:
    """Each pseudo exclusion's 1993-2023 median against the 95th percentile of the other 9 of its cell."""
    med = acc[acc["kind"] == "pseudo"].groupby(["setting", "gen", "exit", "k"])["cagr_1993-2023"].median().reset_index()
    out = []
    for (st, gen, x), g in med.groupby(["setting", "gen", "exit"], sort=False):
        c = g.set_index("k")["cagr_1993-2023"]
        for k in c.index:
            o = c.drop(index=k).to_numpy()
            out.append({"setting": st, "gen": gen, "exit": x, "k": k, "beyond_q95": bool(c[k] > np.percentile(o, 95)),
                        "x_1993-2023": float((c[k] - np.median(o)) * 100)})
    return pd.DataFrame(out)


def q1(v: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """Per setting over the 60 cells: median transfer (M1 - pseudo median) per period, the change
    against N=10 / W2 / P0, cells beyond the pseudo 95th percentile vs Binomial(60, p0) q95, and the
    transfer ratio (account excess / no-limit contribution excess, both in pp)."""
    cells = v[v["gen"] != "E0"]
    pc = pv[pv["gen"] != "E0"]
    rows = []
    for st, g in cells.groupby("setting", sort=False):
        n, w, p = parse_setting(st)
        row = {"setting": st, "N": n, "W": w, "P": p, "cells": len(g)}
        for per in PERIODS:
            row[f"transfer_{per}"] = float(g[f"x_{per}"].median())
            row[f"transfer_q25_{per}"], row[f"transfer_q75_{per}"] = (float(g[f"x_{per}"].quantile(q)) for q in (0.25, 0.75))
        cx = (g["contrib_1993-2023"] - g["pseudo_contrib_1993-2023"]) * 100
        row["contrib_excess_pp"] = float(cx.median())
        row["transfer_ratio"] = row["transfer_1993-2023"] / row["contrib_excess_pp"] if row["contrib_excess_pp"] else np.nan
        p0 = float(pc.loc[pc["setting"] == st, "beyond_q95"].mean()) if (pc["setting"] == st).any() else np.nan
        row.update({"beyond_q95": int(g["beyond_q95"].sum()), "p0": p0,
                    "q95": ks.q95(len(g), p0) if np.isfinite(p0) else len(g)})
        row["beyond_chance"] = row["beyond_q95"] > row["q95"]
        rows.append(row)
    out = pd.DataFrame(rows)
    b = out.set_index("setting").loc[setting_name(BASE)]
    for h in es.HALVES:
        out[f"delta_{h}"] = out[f"transfer_{h}"] - b[f"transfer_{h}"]
    out["grew"] = (out["N"] >= 20) & np.logical_and.reduce([out[f"delta_{h}"] >= Q1_PP * 100 for h in es.HALVES]) & out["beyond_chance"]
    return out


def q2(v: pd.DataFrame, acc: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per setting: entries beating the same setting's E0 in both halves under >= 3 of 4 exits; chance
    from each E0 seed against the other four (pooled over the settings) and Binomial(15, p0) q95."""
    e0 = acc[(acc["gen"] == "E0") & (acc["kind"] == "rule")]
    seeds = []
    for (st, s), g in e0.groupby(["setting", "src"], sort=False):
        o = e0[(e0["setting"] == st) & (e0["src"] != s)]
        n_ok = 0
        for x in EXITS:
            gx, ox = g[g["exit"] == x], o[o["exit"] == x]
            n_ok += all(gx[f"cagr_{h}"].median() > ox[f"cagr_{h}"].median() for h in es.HALVES)
        seeds.append({"setting": st, "src": s, "pass": n_ok >= MIN_EXITS_Q2})
    sd = pd.DataFrame(seeds)
    p0 = float(sd["pass"].mean()) if len(sd) else np.nan
    cells = v[v["gen"] != "E0"]
    rows = []
    for st, g in cells.groupby("setting", sort=False):
        per_gen = g.groupby("gen")["beats_e0"].sum()
        passing = sorted(per_gen[per_gen >= MIN_EXITS_Q2].index.tolist())
        n = int(per_gen.size)
        rows.append({"setting": st, **dict(zip(("N", "W", "P"), parse_setting(st))), "entries": n, "passing": len(passing),
                     "passing_list": " ".join(passing), "p0_setting": float(sd.loc[sd["setting"] == st, "pass"].mean()),
                     "q95": ks.q95(n, p0), "beyond_chance": len(passing) > ks.q95(n, p0)})
    return pd.DataFrame(rows), p0


def market_pass(v: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    return ks.market_pass(v, base) & (v["gen"] != "E0")


def candidates(v: pd.DataFrame, pass40: pd.Series) -> pd.DataFrame:
    """Setting x entry: >= 2 exits at 20 bp and at 40 bp (W2 / W3 only), and the same entry passing at
    an adjacent slot count with the same sizing and priority."""
    g = v.assign(p40=pass40).groupby(["setting", "gen"]).agg(n20=("market_pass", "sum"), n40=("p40", "sum")).reset_index()
    g["W"] = g["setting"].map(lambda s: parse_setting(s)[1])
    g["ok"] = (g["n20"] >= MIN_EXITS_Q3) & (g["n40"] >= MIN_EXITS_Q3) & g["W"].isin(CANDIDATE_W)
    ok = set(zip(g.loc[g["ok"], "setting"], g.loc[g["ok"], "gen"]))
    g["neighbour"] = [any((setting_name(nb), gen) in ok for nb in neighbours(parse_setting(st))) for st, gen in zip(g["setting"], g["gen"])]
    g["candidate"] = g["ok"] & g["neighbour"]
    return g


def signal_stats(bought: pd.DataFrame) -> pd.DataFrame:
    """Bought-signal returns of each setting's first order vs N=10 / W2 / P0's first order (same entry
    and exit), and N=10 / W2 / P0 order 1 vs order 0 as the null."""
    bx = setting_name(BASE)
    rows = []
    for (gen, x), g in bought.groupby(["gen", "exit"], sort=False):
        ref = g[(g["setting"] == bx) & (g["order"] == 0)]["ret"].to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        if len(ref) < pj.MIN_STAT_SIGNALS:
            continue
        rng = np.random.default_rng([SEED, zlib.crc32(f"{gen}|{x}|stat".encode())])
        ref_boot = pj.boot_means(ref, rng)
        for (st, o), gs in g.groupby(["setting", "order"], sort=False):
            if (st == bx and o == 0) or (o == 1 and st != bx):
                continue
            r = gs["ret"].to_numpy(dtype=float)
            r = r[np.isfinite(r)]
            row = {"setting": st if o == 0 else "null", "gen": gen, "exit": x, "n": len(r), **pj.ret_stats(r, ref, ref_boot, rng)}
            for c, f in (("mean", np.mean), ("big50", lambda a: (a >= 0.5).mean()), ("loss20", lambda a: (a <= -0.2).mean())):
                row[c] = float(f(r)) if len(r) else np.nan
                row[f"ref_{c}"] = float(f(ref))
            rows.append(row)
    return pd.DataFrame(rows)


def repro(v: pd.DataFrame, acc: pd.DataFrame, stage_m: Path) -> pd.DataFrame:
    """N=10 / W2 / P0 (X1: W1) against Stage M's M1 cells: rule medians and contributions
    (verdicts.csv.gz), pseudo 0-9 medians (accounts.csv.gz) and no exclusion (single buy rules)."""
    mv = pd.read_csv(stage_m / "verdicts.csv.gz")
    mv = mv[mv["rule"] == RULE].set_index(["gen", "exit"])
    ma = pd.read_csv(stage_m / "accounts.csv.gz")
    ma = ma[(ma["rule"] == RULE) & (ma["kind"] == "pseudo")].set_index(["gen", "exit", "k"])
    out = []
    for x in EXITS:
        st = setting_name(base_setting(x))
        g = v[(v["setting"] == st) & (v["exit"] == x)].set_index("gen")
        pk = acc[(acc["setting"] == st) & (acc["exit"] == x) & (acc["kind"] == "pseudo")].groupby(["gen", "k"]).median(numeric_only=True)
        for gen in g.index:
            if (gen, x) not in mv.index:
                continue
            m = mv.loc[(gen, x)]
            cols = ["trades"] + [f"{s}_{p}" for p in PERIODS for s in ("cagr", "mdd")] + [f"contrib_{p}" for p in PERIODS]
            if gen != "E0":
                cols += [f"none_cagr_{p}" for p in PERIODS]
            for c in cols:
                out.append({"gen": gen, "exit": x, "value": c, "mine": float(g.loc[gen, c]), "stage_m": float(m[c])})
            for k in range(N_PSEUDO):
                if (gen, x, k) in ma.index and (gen, k) in pk.index:
                    out.append({"gen": gen, "exit": x, "value": f"pseudo{k}_cagr_1993-2023", "mine": float(pk.loc[(gen, k), "cagr_1993-2023"]),
                                "stage_m": float(ma.loc[(gen, x, k), "cagr_1993-2023"])})
    r = pd.DataFrame(out)
    if len(r):
        r["diff"] = (r["mine"] - r["stage_m"]).abs()
    return r


def psr_dsr(v: pd.DataFrame, mom: pd.DataFrame, base: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """DSR (N = 70,574 + 1,200) and PSR vs SPY and the same setting's E0 for the candidate rows."""
    m = mom.set_index(["setting", "gen", "exit"])
    sr_var = float(np.var(m.loc[[i for i in m.index if i[1] != "E0"], "m_sr"].dropna()))
    spy_r = pd.Series(pj._W["px"].spy_c[pj._W["w0"]:pj._W["w0"] + pj._W["n"]]).pct_change().dropna()
    spy_sr = float(spy_r.mean() / spy_r.std())
    rows = []
    ok = set(zip(cand.loc[cand["candidate"], "setting"], cand.loc[cand["candidate"], "gen"]))
    for _, r in v[v["market_pass"]].iterrows():
        if (r["setting"], r["gen"]) not in ok or (r["setting"], r["gen"], r["exit"]) not in m.index:
            continue
        mm = m.loc[(r["setting"], r["gen"], r["exit"])]
        e0 = m.loc[(r["setting"], "E0", r["exit"]), "m_sr"] if (r["setting"], "E0", r["exit"]) in m.index else np.nan
        rows.append({"setting": r["setting"], "gen": r["gen"], "exit": r["exit"],
                     "dsr": ks.dsr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), N_TRIALS_BEFORE + N_NEW, sr_var),
                     "psr_spy": ks.psr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), spy_sr),
                     "psr_e0": ks.psr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), e0) if np.isfinite(e0) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures

FAMILIES = [("W1", "P0"), ("W2", "P0"), ("W3", "P0"), ("W2", "P1"), ("W2", "P2")]
FAMILY_LABEL = {("W1", "P0"): "W1 동일 비중", ("W2", "P0"): "W2 1% 룰", ("W3", "P0"): "W3 1% 룰·상한 2/N",
                ("W2", "P1"): "W2 · P1 저변동성 먼저", ("W2", "P2"): "W2 · P2 대형 먼저"}


def fig_grid(out: Path, v: pd.DataFrame, base: pd.DataFrame, grid: str) -> Path:
    """① / ② Median over the 60 cells of the M1 account's CAGR, N x W (A) or N x P (B)."""
    plt = figures._plt()
    cols = list(SIZING) if grid == "A" else list(PRIORITY)
    pers = ("1993-2007", "2008-2023", "2017-2023")
    spy = base.set_index("baseline").loc["SPY"]
    fig, axes = plt.subplots(1, len(pers), figsize=(5.2 * len(pers), 4.4))
    cells = v[v["gen"] != "E0"]
    for ax, per in zip(axes, pers):
        mat = np.full((len(SLOTS_N), len(cols)), np.nan)
        for i, n in enumerate(SLOTS_N):
            for j, c in enumerate(cols):
                st = setting_name((n, c, "P0") if grid == "A" else (n, "W2", c))
                g = cells[cells["setting"] == st]
                if len(g):
                    mat[i, j] = g[f"cagr_{per}"].median() * 100
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}%\n({mat[i, j] - spy[f'cagr_{per}'] * 100:+.1f})", ha="center", va="center",
                            fontsize=8, color="white" if mat[i, j] < np.nanmean(mat) else "black")
        ax.set_xticks(range(len(cols)), [f"{c}\n{(SIZING[c][0] if grid == 'A' else PRIORITY[c])}" for c in cols], fontsize=7.5)
        ax.set_yticks(range(len(SLOTS_N)), [f"N={n}" for n in SLOTS_N], fontsize=8)
        ax.set_title(f"{per.replace('-', '–')} (SPY {spy[f'cagr_{per}'] * 100:.1f}%)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    what = "자리 수 × 비중 (무작위 순)" if grid == "A" else "자리 수 × 우선순위 (W2 1% 룰)"
    fig.suptitle(f"격자 {grid}: {what} — M1 제외 계좌 CAGR, 매수 15 × 매도 4 = 60칸의 중앙값 (괄호 = SPY 대비 %p)", fontsize=11)
    return figures._save(fig, out / "figures" / f"grid_{grid}.png")


def fig_transfer(out: Path, t: pd.DataFrame) -> Path:
    """③ Q1: median over the 60 cells of (M1 - pseudo median) against N, per family, two halves."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, h in zip(axes, es.HALVES):
        for i, fam in enumerate(FAMILIES):
            g = t[(t["W"] == fam[0]) & (t["P"] == fam[1])].sort_values("N")
            if g.empty:
                continue
            c = figures.COLORS[i % len(figures.COLORS)] if i < 4 else "#555"
            ax.plot(g["N"], g[f"transfer_{h}"], marker="o", color=c, lw=1.6, label=FAMILY_LABEL[fam])
            ax.fill_between(g["N"], g[f"transfer_q25_{h}"], g[f"transfer_q75_{h}"], color=c, alpha=0.08)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xticks(SLOTS_N)
        ax.set_xlabel("자리 수 N")
        ax.set_title(h.replace("-", "–"), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("M1 계좌 CAGR − 가짜 제외 중앙값 (%p, 60칸 중앙값, 띠 = 25–75%)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Q1: 신호 질의 개선(M1 제외)이 계좌로 옮겨지는 정도 — 자리 수 · 비중 · 우선순위별", fontsize=11)
    return figures._save(fig, out / "figures" / "q1_transfer.png")


def fig_q2(out: Path, t2: pd.DataFrame) -> Path:
    """④ Q2: entries beating the same setting's E0 against N, per family, with the chance line."""
    plt = figures._plt()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, fam in enumerate(FAMILIES):
        g = t2[(t2["W"] == fam[0]) & (t2["P"] == fam[1])].sort_values("N")
        if len(g):
            ax.plot(g["N"], g["passing"], marker="o", lw=1.6, color=figures.COLORS[i % len(figures.COLORS)] if i < 4 else "#555",
                    label=FAMILY_LABEL[fam])
    q = int(t2["q95"].iloc[0]) if len(t2) else 0
    ax.axhline(q, color="#c0392b", ls="--", lw=1, label=f"우연 95% 분위 ({q})")
    ax.set_xticks(SLOTS_N)
    ax.set_xlabel("자리 수 N")
    ax.set_ylabel("같은 계좌 E0를 이긴 매수 수 (15개 중, 두 반쪽 · 매도 3종 이상)")
    ax.set_ylim(-0.5, 15.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Q2: 매수 규칙이 같은 계좌의 무작위 매수를 이기는 수", fontsize=11)
    return figures._save(fig, out / "figures" / "q2_vs_e0.png")


def fig_bought(out: Path, hist: pd.DataFrame, picks: list[str], x: str = FIG_EXIT) -> Path:
    """⑤ Returns of the signals the account bought (15 entries pooled, per replay) for a few settings."""
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, h in zip(axes, es.HALVES):
        for i, st in enumerate(picks):
            sel = hist[(hist["setting"] == st) & (hist["exit"] == x) & (hist["period"] == h) & (hist["gen"] != "E0")]
            if sel.empty:
                continue
            cnt = sel[hc].to_numpy().sum(axis=0)
            dens = pj._density(cnt)
            mids = np.r_[-62.5, np.arange(-57.5, 200, 5), 207.5] / 100
            mean = float((dens * np.clip(mids, -0.6, 2.0)).sum()) * 100
            pj._bar(ax, dens, color=figures.COLORS[i % len(figures.COLORS)] if i < 4 else "#555", lw=1.5,
                    label=f"{st} (구간 중앙 평균 약 {mean:+.1f}%, −20% 이하 {dens[:9].sum() * 100:.1f}%)")
        ax.axvline(0, color="#bbb", lw=0.8)
        ax.set_title(h.replace("-", "–"), fontsize=10)
        ax.set_xlabel("계좌가 산 신호 하나의 수익률 (%, 편도 20bp, 양 끝 칸은 −60% 미만 / +200% 초과)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle(f"계좌가 실제로 산 신호의 수익률 분포 (매수 15개 합산, 매도 {x}, 무작위 순마다 평균)", fontsize=11)
    return figures._save(fig, out / "figures" / "bought_returns.png")


def fig_scatter(out: Path, v: pd.DataFrame, base: pd.DataFrame) -> Path:
    """⑥ 1993-2023 CAGR vs MDD of every W2 / W3 setting x entry x exit, coloured by 2017-2023 CAGR."""
    plt = figures._plt()
    g = v[(v["gen"] != "E0") & v["W"].isin(CANDIDATE_W)]
    spy = base.set_index("baseline").loc["SPY"]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sc_ = ax.scatter(g["mdd_1993-2023"] * 100, g["cagr_1993-2023"] * 100, c=g["cagr_2017-2023"] * 100, cmap="viridis", s=9, alpha=0.7)
    ax.scatter([spy["mdd_1993-2023"] * 100], [spy["cagr_1993-2023"] * 100], marker="*", s=220, color="#c0392b",
               label=f"SPY (2017–2023 {spy['cagr_2017-2023'] * 100:.1f}%)")
    ax.axhline(spy["cagr_1993-2023"] * 100, color="#c0392b", ls="--", lw=0.8)
    ax.axvline(spy["mdd_1993-2023"] * 100, color="#c0392b", ls="--", lw=0.8)
    fig.colorbar(sc_, ax=ax, label="2017–2023 CAGR (%)")
    ax.set_xlabel("MDD 1993–2023 (%)")
    ax.set_ylabel("CAGR 1993–2023 (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("W2 · W3 설정 × 매수 × 매도 (M1 제외, 무작위 순 중앙값) — 오른쪽 위 사분면이 SPY보다 높고 얕음", fontsize=10)
    return figures._save(fig, out / "figures" / "cagr_mdd_scatter.png")


def write_figures(out: Path, v: pd.DataFrame, t: pd.DataFrame, t2: pd.DataFrame, hist: pd.DataFrame, base: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    made = [fig_grid(out, v, base, "A"), fig_grid(out, v, base, "B"), fig_transfer(out, t), fig_q2(out, t2), fig_scatter(out, v, base)]
    big = t[t["N"] >= 20].sort_values("transfer_1993-2023", ascending=False)
    top = big.iloc[0] if len(big) else None
    picks = [setting_name(BASE)]
    if top is not None:
        picks.append(top["setting"])
        picks += [setting_name((int(top["N"]), "W2", p)) for p in ("P1", "P2")]
    made.append(fig_bought(out, hist, list(dict.fromkeys(picks))))
    return made


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "r0x1", "tables", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--stage-m", default=None, help="Stage M 결과표 폴더 (재현 점검)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="E2·V3·E0(씨앗 2), 매도 4, 자리 수 10·20")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.dirs = {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events),
              "r0x1": Path(a.r0x1), "tables": Path(a.tables)}
    gens = ms.generators(["E2", "V3"], pj.E0_SEEDS[:2]) if a.dry else ms.generators()
    slots = SLOTS_N[:2] if a.dry else SLOTS_N
    a.cfg = {"bins": pj.BINS_J3, "seeds": pj.E0_SEEDS, "base": str(out / "base.parquet")}
    ts.log(f"1단계: 신호 기준표 (매수 {len(gens)} × 매도 {len(EXITS)}) + 특징 · 시총 순위 · 거래대금")
    base = build_base(a, gens)
    ts.log(f"  신호 {len(base):,}개, M1 제외 비율 {ms.excluded(base, RULE).mean():.1%}")
    size = base.groupby("src").size()
    jobs = [(gen, x, n, pj.COST_BPS, srcs, False) for gen, srcs in gens.items() for x in EXITS for n in slots]
    jobs.sort(key=lambda j: -int(size.reindex(j[4]).sum()) * (1 + j[2] / 50))  # the long jobs first
    jdir = out / "jobs"
    todo = [j for j in jobs if not (jdir / f"{job_name(j[0], j[1], j[2], j[3])}_acc.csv").exists()]
    ts.log(f"2단계: 작업 {len(todo)}개 남음 (매수 {len(gens)} × 매도 {len(EXITS)} × 자리 수 {len(slots)}, 설정 {len(settings())})")
    run_jobs(todo, a)
    read = lambda s: pd.concat([pd.read_csv(f) for f in sorted(jdir.glob(f"*_{s}.csv"))  # noqa: E731
                                if "bp_" not in f.name and f.stat().st_size > 2], ignore_index=True)
    acc, un, sh, hist, mom = read("acc"), read("un"), read("shown"), read("hist"), read("mom")
    bought = pd.concat([pd.read_parquet(f) for f in sorted(jdir.glob("*_bought.parquet")) if "bp_" not in f.name], ignore_index=True)
    pj._init(a.panel, a.eligible, a.dirs, a.cfg)
    bl = es.baselines(pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"])
    bl.to_csv(out / "baselines.csv", index=False)
    v = summarize(acc, un)
    if a.stage_m:
        rp = repro(v, acc, Path(a.stage_m))
        rp.to_csv(out / "repro.csv", index=False)
        ts.log(f"재현 점검: Stage M 대비 {rp[['gen', 'exit']].drop_duplicates().shape[0]}칸 {len(rp)}값, 최대 차 {rp['diff'].max():.2e}, "
               f"1e-9 넘는 값 {int((rp['diff'] > 1e-9).sum())}개")
    pv = pseudo_vs_pseudo(acc)
    t = q1(v, pv)
    t2, p0_q2 = q2(v, acc)
    v["market_pass"] = market_pass(v, bl)
    g20 = v[v["W"].isin(CANDIDATE_W)].groupby(["setting", "gen"])["market_pass"].transform("sum")
    v["candidate_20bp"] = False
    v.loc[g20.index, "candidate_20bp"] = (g20 >= MIN_EXITS_Q3) & v.loc[g20.index, "market_pass"]
    pass40 = pd.Series(False, index=v.index)
    c20 = v[v["candidate_20bp"]]
    if len(c20):  # rows again at 40 bp (the M1 account only)
        need = sorted({(r.gen, r.exit, r.N) for r in c20.itertuples()})
        run_jobs([(gen, x, n, pj.CHECK_COST_BPS, gens[gen], True) for gen, x, n in need
                  if not (jdir / f"{job_name(gen, x, n, pj.CHECK_COST_BPS)}_acc.csv").exists()], a)
        a40 = pd.concat([pd.read_csv(jdir / f"{job_name(gen, x, n, pj.CHECK_COST_BPS)}_acc.csv") for gen, x, n in need], ignore_index=True)
        m40 = a40[a40["kind"] == "rule"].groupby(["setting", "gen", "exit"]).median(numeric_only=True)
        spy = bl.set_index("baseline").loc["SPY"]
        for i, r in c20.iterrows():
            if (r["setting"], r["gen"], r["exit"]) in m40.index:
                mm = m40.loc[(r["setting"], r["gen"], r["exit"])]
                pass40[i] = all(mm[f"cagr_{p}"] > spy[f"cagr_{p}"] and abs(mm[f"mdd_{p}"]) < abs(spy[f"mdd_{p}"]) for p in ("1993-2023", "2017-2023"))
    cand = candidates(v, pass40)
    v["pass_40bp"] = pass40
    cs = cand[cand["candidate"]]
    stats = signal_stats(bought)
    dsr = psr_dsr(v, mom, bl, cand) if len(cs) else pd.DataFrame()
    v.to_csv(out / "verdicts.csv", index=False)
    t.merge(t2[["setting", "passing", "passing_list", "p0_setting", "q95", "beyond_chance"]].rename(
        columns={"passing": "q2_passing", "passing_list": "q2_list", "p0_setting": "q2_p0_setting", "q95": "q2_q95", "beyond_chance": "q2_beyond"}),
        on="setting").to_csv(out / "settings.csv", index=False)
    pv.to_csv(out / "pseudo_vs_pseudo.csv", index=False)
    sh.to_csv(out / "shown.csv", index=False)
    stats.to_csv(out / "stats.csv", index=False)
    hist.to_csv(out / "hist.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    acc.groupby(["setting", "gen", "exit", "kind", "k"]).median(numeric_only=True).to_csv(out / "accounts.csv")
    st_null = stats[stats["setting"] == "null"]
    st_set = stats[stats["setting"] != "null"]
    ch = {"q1": {"grew": bool(t["grew"].any()), "settings_grew": t.loc[t["grew"], "setting"].tolist(),
                 "base": t.set_index("setting").loc[setting_name(BASE)].to_dict()},
          "q2": {"p0_pooled": p0_q2, "q95": int(t2["q95"].iloc[0]) if len(t2) else None,
                 "base_passing": int(t2.set_index("setting").loc[setting_name(BASE), "passing"]) if len(t2) else None,
                 "settings_beyond_chance": t2.loc[t2["beyond_chance"], "setting"].tolist()},
          "q3": {"candidates": cs[["setting", "gen"]].to_dict("records"), "rows_20bp": int(v["candidate_20bp"].sum()),
                 "market_pass_cells": int(v["market_pass"].sum()), "dsr": dsr.to_dict("records")},
          "stats": {"null_p_lt_05": float((st_null["st_mw_p"] < 0.05).mean()) if len(st_null) else np.nan,
                    "by_setting_p_lt_05": st_set.groupby("setting")["st_mw_p"].apply(lambda p: float((p < 0.05).mean())).to_dict()},
          "trials": {"before": N_TRIALS_BEFORE, "new": N_NEW}}
    (out / "chance.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, v, t, t2, hist, bl)
    ts.log("완료 · Q1 옮겨짐 커짐 " + str(ch["q1"]["settings_grew"]) + " · Q2 우연 넘은 설정 " + str(ch["q2"]["settings_beyond_chance"])
           + f" (기준 설정 {ch['q2']['base_passing']} / q95 {ch['q2']['q95']}) · 후보 {len(cs)} (20bp 행 {ch['q3']['rows_20bp']})")


if __name__ == "__main__":
    main(sys.argv[1:])
