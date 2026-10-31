"""Stage M: fixed "avoid" exclusions across all buy rules and exits, against day-matched pseudo exclusions (DECISIONS 92).

    python -m lab.exclude_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                                --r0x1 DIR --tables DIR --out DIR [--stage-l DIR] [--workers N] [--feature-workers N] [--dry]

On the signal day, an exclusion rule leaves out the stocks in the top 30% of that day's hygiene
universe by idiosyncratic volatility (M1), by 60-day volatility (M2), or by any of idiosyncratic
volatility, 12-month drawdown depth and distance from the 52-week high (M3, union); a missing value
excludes nothing. The 15 buy rules (E1-E9, V1-V6, middle value) and the random buys E0 (seeds 1-5)
are replayed from the stored Stage L tables (lab/exit_study.py's table_of, all 11 exits; --tables =
Stage L's tables/ folder; no engine run) without the excluded signals; held stocks are sold by
their exit as before. Each cell's null is 20 pseudo exclusions that leave out, on every day the rule
left out n of a buy rule's signals, n of that day's signals at random (lab/select_study.py's
pseudo_mask). The cells without exclusion are replayed with Stage L's own random orders and, with
--stage-l, checked against its result tables (repro.csv). 10-stock 1%-rule accounts (100 random
orders, E0: 20 per seed; pseudo 20 each), no-limit contributions, per-signal return histograms and
tests; verdicts per cell, chance counts per exit, the 2017-2023 gap, candidates, size diagnostic.
Output (aggregates only): base.parquet, jobs/*.csv, verdicts.csv, pseudo_vs_pseudo.csv, repro.csv,
stats.csv, hist.csv, accounts.csv, unlimited.csv, chance.json, baselines.csv, figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import entry_study as es, exit_study as xs, figures, pair_study as pj, pick_study as ph, select_study as ks, tail_study as ts

# ---- registered (DECISIONS 92): fixed before any result
# rule -> label, the lab/select_study.py features whose low percentile (the unfavoured 30%) is left out
EXCLUSIONS = {
    "M1": ("고유 변동성 상위 30% 제외", ("K2",)),
    "M2": ("60일 변동성 상위 30% 제외", ("K8",)),
    "M3": ("고유 변동성 상위 · 12개월 낙폭 깊은 · 52주 고가에서 먼 30% 중 하나라도 제외", ("K2", "K13", "K11")),
}
FEATURES = ("K2", "K8", "K13", "K11")
CUT = 0.30
EXITS = list(xs.EXITS_L)
N_ORDERS, N_PSEUDO, PSEUDO_ORDERS = 100, 20, 20  # E0: split over its 5 seeds (20 and 4 per seed)
MIN_INFO_EXITS = 6  # a rule has information if (b) beats chance under at least this many of the 11 exits
MIN_EXITS = 3  # a candidate passes the market test under at least this many exits (as 90)
GAP_PP = 0.01  # the 2017-2023 gap shrank / grew if the median change is beyond +-1 pp
SIZE_RHO = 0.5  # |rank correlation| of cap-rank shift and CAGR change flagged "not separated from size"
MAX_REF = ks.MAX_REF
FIG_EXIT = "H1"
N_TRIALS_BEFORE = ks.N_TRIALS_BEFORE + ks.N_NEW  # 70,079 (Stages B-K, approx.)
N_NEW = 495  # 3 rules x 15 buy rules x 11 exits (E0 and pseudo exclusions not counted)
SEED = 92
_W: dict = {}


def generators(rules: list[str] | None = None, seeds: list[int] | None = None) -> dict[str, list[str]]:
    """Buy rule -> its signal sources (E0: one per seed)."""
    out = {r: [r] for r in (rules if rules is not None else pj.RULES)}
    out["E0"] = [f"R0s{s}" for s in (seeds if seeds is not None else pj.E0_SEEDS)]
    return out


def excluded(base: pd.DataFrame, rule: str) -> np.ndarray:
    """True for the signals the rule leaves out: a percentile of one of its features at or below the
    cut (the unfavoured side; NaN excludes nothing)."""
    out = np.zeros(len(base), dtype=bool)
    for f in EXCLUSIONS[rule][1]:
        out |= base[f].to_numpy(dtype=float) <= CUT
    return out


def gap_verdict(change: float) -> str:
    """The 2017-2023 gap to SPY from the median CAGR change of a rule's cells."""
    if not np.isfinite(change):
        return "판정 불가"
    return "줄었다" if change > GAP_PP else "커졌다" if change < -GAP_PP else "그대로다"


# --------------------------------------------------------------------------- stage 1: base table

def build_base(a, gens: dict) -> pd.DataFrame:
    """Every source's signals (src, si, sj) over the exits' tables with the four feature percentiles
    and the cap rank on the signal day (lab/select_study.py's feature workers)."""
    f = Path(a.out) / "base.parquet"
    if f.exists():
        return pd.read_parquet(f)
    xs._init(a.panel, a.eligible, a.dirs, a.cfg)
    cols = pj._W["px"].cols
    parts = []
    for srcs in gens.values():
        for s in srcs:
            for x in EXITS:
                if xs.EXITS_L[x].get("partial"):  # TP = the H1 table
                    continue
                t = pj.rule_table(s, x)
                parts.append(pd.DataFrame({"src": s, "si": t["si"].to_numpy(), "sj": cols.get_indexer(t["symbol"].astype(str))}))
            pj._W["tables"].clear()
    base = pd.concat(parts, ignore_index=True).drop_duplicates(["src", "si", "sj"])
    base = base.sort_values(["src", "si", "sj"], kind="stable").reset_index(drop=True)
    pts = base[["si", "sj"]].drop_duplicates().reset_index(drop=True)
    si, sj = pts["si"].to_numpy(), pts["sj"].to_numpy()
    with ProcessPoolExecutor(max_workers=a.feature_workers, initializer=ks._f_init, initargs=(a.panel, a.eligible)) as ex:
        for key, vals, sec in ex.map(ks.feature_values, [(k, si, sj) for k in list(FEATURES) + ["cap_rank"]], chunksize=1):
            pts[key] = vals
            ts.log(f"  특징 {key}: {sec:.0f}s, 값 있음 {np.isfinite(vals).mean():.1%}")
    base = base.merge(pts, on=["si", "sj"], how="left")
    base.to_parquet(f)
    return base


# --------------------------------------------------------------------------- stage 2: per buy rule x exit

def _init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    xs._init(panel_dir, eligible, dirs, cfg)
    _W.update({k: pj._W[k] for k in ("px", "w0", "n", "wdates", "cash", "O", "LC")})
    _W["base"] = pd.read_parquet(cfg["base"])
    _W["prep"] = {}


def source_base(src: str) -> pd.DataFrame:
    """The base rows of one source in (si, sj) order (cached per job)."""
    key = ("base", src)
    if key not in _W["prep"]:
        b = _W["base"]
        _W["prep"][key] = b[b["src"] == src].reset_index(drop=True)
    return _W["prep"][key]


def prepared(src: str, x: str) -> pd.DataFrame:
    """Stage L's prepared rows of a source under exit x (same rows and order as its run) with the
    position of each row in the source's base rows ("brow") and the cap rank."""
    key = (src, x)
    if key not in _W["prep"]:
        spec = ("e0full", int(src[3:])) if src.startswith("R0s") else ("single", src)
        d = xs.prepared_l(spec, x)
        b = source_base(src)
        pos = pd.Series(np.arange(len(b)), index=pd.MultiIndex.from_arrays([b["si"].to_numpy(), b["sj"].to_numpy()]))
        brow = pos.reindex(pd.MultiIndex.from_arrays([d["si"].to_numpy(), d["sj"].to_numpy()])).to_numpy()
        if np.isnan(brow).any():
            raise ValueError(f"{src} {x}: 기준표에 없는 신호 {int(np.isnan(brow).sum())}개")
        d["brow"] = brow.astype(np.int64)
        d["cap_rank"] = b["cap_rank"].to_numpy(dtype=float)[d["brow"].to_numpy()]
        _W["prep"][key] = d
    return _W["prep"][key]


def _replays(d: pd.DataFrame, n: int, rng: np.random.Generator, cost: float, x: str, curves: list | None = None) -> list[dict]:
    """n random orders of a 10-stock account on rows d (exit_study.run_set's call, so a cell without
    exclusion and Stage L's random generator give Stage L's accounts)."""
    w0, nd, wdates = _W["w0"], _W["n"], _W["wdates"]
    ei, sale = d["ei"].to_numpy() - w0, d["sale"].to_numpy() - w0
    closed = d["closed"].to_numpy(dtype=bool) & (sale < nd)
    psale = d["psale"].to_numpy() if xs.EXITS_L[x].get("partial") else None
    frac = xs.EXITS_L[x]["partial"][1] if psale is not None else 0.5
    cap = d["cap_rank"].to_numpy(dtype=float)
    out = []
    for _ in range(n):
        eq, tr = ph.replay_risk(ei - 1, ei, d["sj"].to_numpy(), sale, closed, d["w"].to_numpy(), rng.random(len(d)),
                                _W["O"], _W["LC"], _W["cash"], cost, psale=psale, pfrac=frac)
        r = np.diff(eq) / eq[:-1]
        out.append({"trades": len(tr), "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
                    "cap_rank_trades": float(np.nanmedian(cap[tr["row"].to_numpy()])) if len(tr) else np.nan,
                    **es.period_stats(pd.Series(eq, index=wdates))})
        if curves is not None:
            curves.append(eq)
    return out


def _period_means(d: pd.DataFrame) -> dict:
    """Per period: signals, mean contribution, mean return, +50% and -20% shares (signal year)."""
    yr = pd.DatetimeIndex(d["signal_date"]).year.to_numpy()
    ret = d["ret"].to_numpy(dtype=float)
    out = {}
    for p, (a, b) in es.PERIODS.items():
        mm = (yr >= a) & (yr <= b)
        r = ret[mm & np.isfinite(ret)]
        out[f"n_{p}"] = int(mm.sum())
        out[f"mean_contrib_{p}"] = float(d["contrib"].to_numpy()[mm].mean()) if mm.any() else np.nan
        out[f"n_ret_{p}"] = len(r)
        out[f"mean_ret_{p}"] = float(r.mean()) if len(r) else np.nan
        out[f"big50_{p}"] = float((r >= 0.5).mean()) if len(r) else np.nan
        out[f"loss20_{p}"] = float((r <= -0.2).mean()) if len(r) else np.nan
    return out


def _hist(kind: str, frame: pd.DataFrame, scale: float = 1.0) -> list[dict]:
    yr = pd.DatetimeIndex(frame["signal_date"]).year.to_numpy()
    out = []
    for per in pj.HIST_PERIODS:
        a_, b_ = es.PERIODS[per]
        r = frame["ret"].to_numpy(dtype=float)[(yr >= a_) & (yr <= b_)]
        r = r[np.isfinite(r)]
        out.append({"kind": kind, "period": per, **{f"h{i}": float(c) * scale for i, c in enumerate(np.histogram(r, pj.RET_EDGES)[0])}})
    return out


def _signal_stats(rd: pd.DataFrame, pds: list[pd.DataFrame], tag: str) -> list[dict]:
    """The rule's kept signals vs the pseudo exclusions' kept signals (pooled, sampled to MAX_REF), as
    Stage K; each pseudo against the others by the rank test (calibration)."""
    r_rule = rd["ret"].to_numpy(dtype=float)
    r_rule = r_rule[np.isfinite(r_rule)]
    labels = np.concatenate([np.full(len(p), k) for k, p in enumerate(pds)])
    pooled = np.concatenate([p["ret"].to_numpy(dtype=float) for p in pds])
    ok = np.isfinite(pooled)
    labels, pooled = labels[ok], pooled[ok]
    srng = np.random.default_rng([SEED, zlib.crc32((tag + "|ref").encode())])
    take = srng.choice(len(pooled), size=min(MAX_REF, len(pooled)), replace=False) if len(pooled) else np.array([], int)
    ref, ref_lab = pooled[take], labels[take]
    st = {"kind": "rule", "k": -1, **pj.ret_stats(r_rule, ref, pj.boot_means(ref, srng), srng)}
    for c, f in (("mean_ret", np.mean), ("median_ret", np.median)):
        st[c] = float(f(r_rule)) if len(r_rule) else np.nan
        st[f"pseudo_{c}"] = float(f(pooled)) if len(pooled) else np.nan
    for c, f in (("big50", lambda v: (v >= 0.5).mean()), ("loss20", lambda v: (v <= -0.2).mean()), ("win", lambda v: (v > 0).mean())):
        st[c] = float(f(r_rule)) if len(r_rule) else np.nan
        st[f"pseudo_{c}"] = float(f(pooled)) if len(pooled) else np.nan
    out = [st]
    for k, p in enumerate(pds):
        rk = p["ret"].to_numpy(dtype=float)
        rk = rk[np.isfinite(rk)]
        other = ref[ref_lab != k]
        if len(rk) >= pj.MIN_STAT_SIGNALS and len(other) >= pj.MIN_STAT_SIGNALS:
            auc, pv = pj.mann_whitney(rk, other)
            out.append({"kind": "pseudo", "k": k, "st_mw_auc": auc, "st_mw_p": pv, "st_diff": float(rk.mean() - other.mean())})
    return out


def run_job(job):
    """One buy rule x exit: the cells without exclusion (Stage L's orders) and, per exclusion rule,
    the rule and its pseudo exclusions (accounts, contributions, per-signal statistics)."""
    gen, x, n_base, n_orders, n_pseudo, p_orders, cost_bps, srcs, rules = job
    t0 = time.time()
    cost = cost_bps / 10_000
    accs, uns, stats, hists = [], [], [], []
    ds = {s: prepared(s, x) for s in srcs}
    if n_base:  # no exclusion = Stage L: the same rows, names and random generator as exit_study.run_set
        for s in srcs:
            name = pj.set_name(("e0full", int(s[3:])) if s.startswith("R0s") else ("single", s))
            rng = np.random.default_rng([xs.SEED, zlib.crc32(f"{name}|{x}|{cost_bps}".encode())])
            for row in _replays(ds[s], n_base, rng, cost, x):
                accs.append({"rule": "none", "kind": "none", "k": -1, "src": s, **row})
            uns.append({"rule": "none", "kind": "none", "k": -1, "src": s, "signals": len(ds[s]), **_period_means(ds[s])})
        alld = pd.concat(ds.values(), ignore_index=True)
        uns.append({"rule": "none", "kind": "none", "k": -1, "src": "*", "signals": len(alld), **_period_means(alld)})
        hists += [{"rule": "none", **h} for h in _hist("none", alld)]
    n_all = sum(len(d) for d in ds.values())
    for m in rules:
        tag = f"{m}|{gen}|{x}|{cost_bps}"
        rng = np.random.default_rng([SEED, zlib.crc32(tag.encode())])
        curves: list = []
        rule_d, pseudo_d = [], [[] for _ in range(n_pseudo)]
        for s in srcs:
            b = source_base(s)
            keep = ~excluded(b, m)
            d = ds[s]
            brow = d["brow"].to_numpy()
            dk = d[keep[brow]].reset_index(drop=True)
            rule_d.append(dk)
            for row in _replays(dk, n_orders // len(srcs), rng, cost, x, curves):
                accs.append({"rule": m, "kind": "rule", "k": -1, "src": s, **row})
            for k in range(n_pseudo):  # the same pseudo exclusion under every exit (as the rule)
                pm = ks.pseudo_mask(b["si"].to_numpy(), keep, np.random.default_rng([SEED, zlib.crc32(f"{m}|{s}".encode()), k]))
                dp = d[pm[brow]].reset_index(drop=True)
                pseudo_d[k].append(dp)
                for row in _replays(dp, p_orders // len(srcs), rng, cost, x):
                    accs.append({"rule": m, "kind": "pseudo", "k": k, "src": s, **row})
        rd = pd.concat(rule_d, ignore_index=True)
        cl = rd["closed"].to_numpy(dtype=bool)
        un = {"rule": m, "kind": "rule", "k": -1, "src": "*", "signals": len(rd), "kept_share": len(rd) / max(n_all, 1),
              "per_day": len(rd) / _W["n"] / len(srcs),
              "hold_median": float(np.median(rd["holding_days"].to_numpy()[cl])) if cl.any() else np.nan,
              "stop_share": float((rd["exit_reason"].to_numpy()[cl] == xs.STOP_LABEL).mean()) if cl.any() else np.nan,
              "big_share": float((rd["contrib"] >= ph.BIG_CONTRIB).mean()) if len(rd) else np.nan,
              "cap_rank_signals": float(np.nanmedian(rd["cap_rank"])) if len(rd) else np.nan, **_period_means(rd)}
        if curves and cost_bps == pj.COST_BPS:  # the median-order curve's moments, for DSR / PSR of candidates
            c = [r["cagr_1993-2023"] for r in accs if r["rule"] == m and r["kind"] == "rule"]
            un.update(ks._moments(curves[int(np.argsort(c)[len(c) // 2])]))
        uns.append(un)
        pds = [pd.concat(p, ignore_index=True) for p in pseudo_d]
        for k, p in enumerate(pds):
            uns.append({"rule": m, "kind": "pseudo", "k": k, "src": "*", "signals": len(p),
                        "cap_rank_signals": float(np.nanmedian(p["cap_rank"])) if len(p) else np.nan, **_period_means(p)})
        if cost_bps == pj.COST_BPS and n_pseudo:
            stats += [{"rule": m, **r} for r in _signal_stats(rd, pds, tag)]
            hists += [{"rule": m, **h} for h in _hist("rule", rd)]
            hists += [{"rule": m, **h} for h in _hist("pseudo", pd.concat(pds, ignore_index=True), 1.0 / n_pseudo)]
    _W["prep"].clear()
    pj._W["tables"].clear()
    head = {"gen": gen, "exit": x, "cost": cost_bps}
    frame = lambda rows: pd.DataFrame([{**head, **r} for r in rows])  # noqa: E731
    return job, frame(accs), frame(uns), frame(stats), frame(hists), time.time() - t0


def job_name(gen: str, x: str, cost: float, rules: list[str] | None = None) -> str:
    return f"{gen}__{x}" + ("__" + ("_".join(rules) or "none") if rules is not None else "") + ("" if cost == pj.COST_BPS else f"__{cost:g}bp")


def run_jobs(jobs: list, a) -> None:
    d = Path(a.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs, a.cfg)) as ex:
        for i, (job, acc, un, st, hist, sec) in enumerate(ex.map(run_job, jobs, chunksize=1)):
            nm = job_name(job[0], job[1], job[6], job[8])
            for suffix, f in (("acc", acc), ("un", un), ("stats", st), ("hist", hist)):
                f.to_csv(d / f"{nm}_{suffix}.csv", index=False)
            if (i + 1) % 5 == 0 or i + 1 == len(jobs):
                ts.log(f"  [{i + 1}/{len(jobs)}] {nm} {sec:.0f}s · 남은 예상 {(len(jobs) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


# --------------------------------------------------------------------------- verdicts

def _pooled(u: pd.DataFrame, col: str, p: str) -> float:
    n = u[f"n_{p}"] if col == "mean_contrib" else u[f"n_ret_{p}"]
    return float((u[f"{col}_{p}"] * n).sum() / n.sum()) if n.sum() else np.nan


def summarize(acc: pd.DataFrame, un: pd.DataFrame) -> pd.DataFrame:
    """Per rule x buy rule x exit: the rule's medians (over all its orders), the pseudo exclusions'
    medians (each pseudo's own median), the cell without exclusion (median over its orders) and the
    (a) / (b) tests in both halves."""
    keys = ["rule", "gen", "exit"]
    med = acc[acc["kind"] != "none"].groupby(keys + ["kind", "k"]).median(numeric_only=True).reset_index()
    none = acc[acc["kind"] == "none"].groupby(["gen", "exit"]).median(numeric_only=True)
    un_none = un[(un["kind"] == "none") & (un["src"] == "*")].set_index(["gen", "exit"])
    rows = []
    for (m, gen, x), g in med.groupby(keys, sort=False):
        r = g[g["kind"] == "rule"].iloc[0]
        p = g[g["kind"] == "pseudo"].set_index("k")
        u = un[(un["rule"] == m) & (un["gen"] == gen) & (un["exit"] == x)]
        ur = u[u["kind"] == "rule"].iloc[0]
        up = u[u["kind"] == "pseudo"].set_index("k")
        nb = none.loc[(gen, x)] if (gen, x) in none.index else None
        ub = un_none.loc[(gen, x)] if (gen, x) in un_none.index else None
        row = {"rule": m, "gen": gen, "exit": x, "trades": r["trades"], "signals": ur["signals"], "kept_share": ur["kept_share"],
               "per_day": ur["per_day"], "hold_median": ur["hold_median"], "stop_share": ur["stop_share"], "big_share": ur["big_share"],
               "cap_rank_signals": ur["cap_rank_signals"], "cap_rank_trades": r["cap_rank_trades"],
               "pseudo_cap_rank_trades": float(p["cap_rank_trades"].median()) if len(p) else np.nan,
               "none_cap_rank_trades": nb["cap_rank_trades"] if nb is not None else np.nan,
               "none_trades": nb["trades"] if nb is not None else np.nan,
               "sharpe": r["sharpe"], "pseudo_sharpe": float(p["sharpe"].median()) if len(p) else np.nan}
        for c in ("m_sr", "m_g3", "m_g4", "m_T"):
            row[c] = ur.get(c, np.nan)
        for per in es.PERIODS:
            row[f"cagr_{per}"], row[f"mdd_{per}"] = r[f"cagr_{per}"], r[f"mdd_{per}"]
            row[f"pseudo_cagr_{per}"] = float(p[f"cagr_{per}"].median()) if len(p) else np.nan
            row[f"pseudo_q05_{per}"], row[f"pseudo_q95_{per}"] = ((float(p[f"cagr_{per}"].quantile(q)) if len(p) else np.nan) for q in (0.05, 0.95))
            row[f"pct_in_pseudo_{per}"] = float((p[f"cagr_{per}"] < r[f"cagr_{per}"]).mean()) if len(p) else np.nan
            row[f"none_cagr_{per}"] = nb[f"cagr_{per}"] if nb is not None else np.nan
            row[f"none_mdd_{per}"] = nb[f"mdd_{per}"] if nb is not None else np.nan
            row[f"contrib_{per}"] = ur[f"mean_contrib_{per}"]
            row[f"pseudo_contrib_{per}"] = float(up[f"mean_contrib_{per}"].median()) if len(up) else np.nan
            row[f"none_contrib_{per}"] = ub[f"mean_contrib_{per}"] if ub is not None else np.nan
            row[f"mean_ret_{per}"] = ur[f"mean_ret_{per}"]
            row[f"pseudo_mean_ret_{per}"] = float(up[f"mean_ret_{per}"].median()) if len(up) else np.nan
        row["a_halves"] = all(row[f"contrib_{h}"] > row[f"pseudo_contrib_{h}"] for h in es.HALVES)
        row["b_halves"] = all(row[f"cagr_{h}"] > row[f"pseudo_cagr_{h}"] for h in es.HALVES)
        row["b_below"] = all(row[f"cagr_{h}"] < row[f"pseudo_cagr_{h}"] for h in es.HALVES)
        rows.append(row)
    return pd.DataFrame(rows)


def pseudo_vs_pseudo(acc: pd.DataFrame, un: pd.DataFrame) -> pd.DataFrame:
    """Each pseudo exclusion against the median of the other pseudo exclusions of its cell."""
    keys = ["rule", "gen", "exit"]
    med = acc[acc["kind"] == "pseudo"].groupby(keys + ["k"]).median(numeric_only=True).reset_index()
    up = un[un["kind"] == "pseudo"]
    out = []
    for (m, gen, x), g in med.groupby(keys, sort=False):
        u = up[(up["rule"] == m) & (up["gen"] == gen) & (up["exit"] == x)].set_index("k")
        g = g.set_index("k")
        for k in g.index:
            o, uo = g.drop(index=k), u.drop(index=k)
            row = {"rule": m, "gen": gen, "exit": x, "k": k, "trades": g.loc[k, "trades"]}
            for h in es.PERIODS:
                row[f"x_{h}"] = (g.loc[k, f"cagr_{h}"] - o[f"cagr_{h}"].median()) * 100
            row["a_halves"] = all(u.loc[k, f"mean_contrib_{h}"] > uo[f"mean_contrib_{h}"].median() for h in es.HALVES)
            row["b_halves"] = all(g.loc[k, f"cagr_{h}"] > o[f"cagr_{h}"].median() for h in es.HALVES)
            row["b_below"] = all(g.loc[k, f"cagr_{h}"] < o[f"cagr_{h}"].median() for h in es.HALVES)
            out.append(row)
    return pd.DataFrame(out)


def chance(v: pd.DataFrame, pv: pd.DataFrame) -> dict:
    """Per rule and exit: pass counts of the buy rules (E0 apart, >= 30 trades) in (a) and (b) vs the
    pseudo-vs-pseudo pass rate p0 and the 95th percentile of Binomial(N, p0); a rule has information
    if (b) beats chance under >= MIN_INFO_EXITS exits. E0's own pass is listed per exit."""
    out = {}
    for m in EXCLUSIONS:
        vm, pm = v[(v["rule"] == m) & (v["gen"] != "E0")], pv[(pv["rule"] == m) & (pv["gen"] != "E0")]
        if vm.empty:
            continue
        by_exit = {}
        for x in EXITS:
            g = vm[vm["exit"] == x]
            if g.empty:
                continue
            ok = g[g["trades"] >= pj.MIN_TRADES]
            p = pm[pm["exit"] == x]
            row = {"n_judged": len(ok), "excluded_few_trades": int((g["trades"] < pj.MIN_TRADES).sum())}
            for t, col in (("a", "a_halves"), ("b", "b_halves")):
                p0 = float(p[col].mean()) if len(p) else np.nan
                obs = int(ok[col].sum())
                q = ks.q95(len(ok), p0) if np.isfinite(p0) else len(ok)
                row[t] = {"observed": obs, "p0": p0, "expected": p0 * len(ok), "q95": q, "information": obs > q,
                          "passing": sorted(ok.loc[ok[col], "gen"].tolist())}
            e0 = v[(v["rule"] == m) & (v["gen"] == "E0") & (v["exit"] == x)]
            if len(e0):
                row["e0"] = {"a": bool(e0.iloc[0]["a_halves"]), "b": bool(e0.iloc[0]["b_halves"]),
                             "excess_1993_2023_pp": float((e0.iloc[0]["cagr_1993-2023"] - e0.iloc[0]["pseudo_cagr_1993-2023"]) * 100)}
            by_exit[x] = row
        info = {t: sorted(x for x in by_exit if by_exit[x][t]["information"]) for t in ("a", "b")}
        out[m] = {"by_exit": by_exit, "exits_with_information": info,
                  "information": len(info["b"]) >= MIN_INFO_EXITS}
    return out


def gap(v: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Per rule: the 2017-2023 CAGR change of its cells (E0 apart) against no exclusion, and the cells
    at or above SPY (and equal weight) before and after."""
    bl = base.set_index("baseline")
    spy, ew = float(bl.loc["SPY", "cagr_2017-2023"]), float(bl.iloc[2]["cagr_2017-2023"])
    out = {}
    for m, g in v[v["gen"] != "E0"].groupby("rule", sort=False):
        ch = (g["cagr_2017-2023"] - g["none_cagr_2017-2023"]).to_numpy(dtype=float)
        chp = (g["pseudo_cagr_2017-2023"] - g["none_cagr_2017-2023"]).to_numpy(dtype=float)
        med = float(np.nanmedian(ch))
        out[m] = {"cells": len(g), "median_change_pp": med * 100, "pseudo_median_change_pp": float(np.nanmedian(chp)) * 100,
                  "share_up": float(np.mean(ch > 0)), "spy": spy, "ew": ew,
                  "ge_spy_before": int((g["none_cagr_2017-2023"] >= spy).sum()), "ge_spy_after": int((g["cagr_2017-2023"] >= spy).sum()),
                  "ge_ew_before": int((g["none_cagr_2017-2023"] >= ew).sum()), "ge_ew_after": int((g["cagr_2017-2023"] >= ew).sum()),
                  "verdict": gap_verdict(med)}
    return out


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation (average ranks, pairs with a missing value dropped) without scipy."""
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 2 else np.nan


def size_diagnostic(v: pd.DataFrame) -> dict:
    """Per rule (E0 apart): rank correlation of the cap-rank shift of the filled trades (after minus
    before; negative = larger companies) with the CAGR change, and the same against the pseudo median."""
    out = {}
    for m, g in v[v["gen"] != "E0"].groupby("rule", sort=False):
        shift = g["cap_rank_trades"] - g["none_cap_rank_trades"]
        dc = g["cagr_1993-2023"] - g["none_cagr_1993-2023"]
        shift_p = g["cap_rank_trades"] - g["pseudo_cap_rank_trades"]
        dc_p = g["cagr_1993-2023"] - g["pseudo_cagr_1993-2023"]
        rho = spearman(shift, dc)
        out[m] = {"median_shift": float(shift.median()), "median_shift_vs_pseudo": float(shift_p.median()),
                  "rho_vs_none": rho, "rho_vs_pseudo": spearman(shift_p, dc_p),
                  "not_separated": bool(abs(rho) >= SIZE_RHO)}
    return out


def x1_check(v: pd.DataFrame) -> dict:
    """The fixed 10% weight exit X1 apart (the 1%-rule weight interacts with volatility exclusions)."""
    out = {}
    for m, g in v[(v["gen"] != "E0")].groupby("rule", sort=False):
        ex = (g["cagr_1993-2023"] - g["pseudo_cagr_1993-2023"]) * 100
        x1 = g["exit"] == "X1"
        out[m] = {"x1_b_pass": int(g.loc[x1, "b_halves"].sum()), "x1_median_excess_pp": float(ex[x1].median()),
                  "risk_exits_median_excess_pp": float(ex[~x1].median()),
                  "x1_same_sign": bool(np.sign(ex[x1].median()) == np.sign(ex[~x1].median()))}
    return out


def stats_summary(st: pd.DataFrame) -> dict:
    """Per rule: share of cells whose kept signals differ from the pseudo-kept at p < 0.05 (and the
    pseudo exclusions against each other), CI above / below 0, median difference."""
    out = {}
    for m, g in st.groupby("rule", sort=False):
        r = g[(g["kind"] == "rule") & (g["gen"] != "E0") & g["st_mw_p"].notna()]
        p = g[(g["kind"] == "pseudo") & (g["gen"] != "E0")]
        e0 = g[(g["kind"] == "rule") & (g["gen"] == "E0")]
        out[m] = {"cells": len(r), "rule_p_lt_05": float((r["st_mw_p"] < 0.05).mean()),
                  "rule_p_lt_05_higher": float(((r["st_mw_p"] < 0.05) & (r["st_mw_auc"] > 0.5)).mean()),
                  "pseudo_p_lt_05": float((p["st_mw_p"] < 0.05).mean()), "rule_ci_above_0": float((r["st_ci_lo"] > 0).mean()),
                  "rule_ci_below_0": float((r["st_ci_hi"] < 0).mean()), "median_diff_pp": float(r["st_diff"].median() * 100),
                  "median_auc": float(r["st_mw_auc"].median()),
                  "e0_diff_pp": float(e0["st_diff"].median() * 100) if len(e0) else np.nan,
                  "e0_p_lt_05": float((e0["st_mw_p"] < 0.05).mean()) if len(e0) else np.nan}
    return out


def repro(acc: pd.DataFrame, un: pd.DataFrame, stage_l: Path) -> pd.DataFrame:
    """The cells without exclusion against Stage L's verdicts.csv / unlimited.csv: one row per (set,
    exit, value) with both numbers."""
    lv = pd.read_csv(stage_l / "verdicts.csv.gz" if (stage_l / "verdicts.csv.gz").exists() else stage_l / "verdicts.csv")
    lu = pd.read_csv(stage_l / "unlimited.csv.gz" if (stage_l / "unlimited.csv.gz").exists() else stage_l / "unlimited.csv")
    a = acc[acc["kind"] == "none"].copy()
    a["set"] = np.where(a["src"].str.startswith("R0s"), "e0full__" + a["src"].str[3:], "single__" + a["src"])
    med = a.groupby(["set", "exit"]).median(numeric_only=True)
    u = un[(un["kind"] == "none") & (un["src"] != "*")].copy()
    u["set"] = np.where(u["src"].str.startswith("R0s"), "e0full__" + u["src"].str[3:], "single__" + u["src"])
    u = u.set_index(["set", "exit"])
    lv, lu = lv.set_index(["set", "exit"]), lu.set_index(["set", "exit"])
    rows = []
    for key in med.index:
        if key not in lv.index:
            continue
        for c in ["trades"] + [f"{s}_{p}" for p in es.PERIODS for s in ("cagr", "mdd")]:
            rows.append({"set": key[0], "exit": key[1], "value": c, "mine": float(med.loc[key, c]), "stage_l": float(lv.loc[key, c])})
        for p in es.PERIODS:
            rows.append({"set": key[0], "exit": key[1], "value": f"mean_contrib_{p}", "mine": float(u.loc[key, f"mean_contrib_{p}"]),
                         "stage_l": float(lu.loc[key, f"mean_contrib_{p}"])})
        rows.append({"set": key[0], "exit": key[1], "value": "signals", "mine": float(u.loc[key, "signals"]),
                     "stage_l": float(lu.loc[key, "signals"])})
    r = pd.DataFrame(rows)
    if len(r):
        r["diff"] = (r["mine"] - r["stage_l"]).abs()
    return r


def market_pass(v: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    return ks.market_pass(v, base) & (v["gen"] != "E0")


# --------------------------------------------------------------------------- figures

def _cols(v: pd.DataFrame) -> list[str]:
    gens = [g for g in pj.RULES if g in set(v["gen"])]
    return gens + (["E0"] if "E0" in set(v["gen"]) else [])


def fig_heatmap(out: Path, v: pd.DataFrame, m: str) -> Path:
    """① Buy rule x exit: 10-stock CAGR minus the pseudo median (pp, 1993-2023), (b) passes boxed."""
    plt = figures._plt()
    g = v[v["rule"] == m].set_index(["exit", "gen"])
    cols = _cols(v)
    xs_ = [x for x in EXITS if x in set(v["exit"])]
    mat = np.full((len(xs_), len(cols)), np.nan)
    passed = np.zeros_like(mat, dtype=bool)
    for i, x in enumerate(xs_):
        for j, gen in enumerate(cols):
            if (x, gen) in g.index:
                r = g.loc[(x, gen)]
                mat[i, j] = (r["cagr_1993-2023"] - r["pseudo_cagr_1993-2023"]) * 100
                passed[i, j] = bool(r["b_halves"])
    plain = [i for i, x in enumerate(xs_) if x in xs.VALUES_ONLY]
    colored = np.delete(mat, plain, axis=0)
    lim = max(1.0, float(np.nanmax(np.abs(colored)))) if np.isfinite(colored).any() else 1.0
    shown = mat.copy()
    shown[plain] = np.nan
    fig, ax = plt.subplots(figsize=(13, 6.2))
    im = ax.imshow(np.ma.masked_invalid(shown), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in plain:
        ax.axhspan(i - 0.5, i + 0.5, color="#eeeeee")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if i not in plain and abs(mat[i, j]) > 0.6 * lim else "black")
            if passed[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#111", lw=1.4))
    if "E0" in cols:
        ax.axvline(len(cols) - 1.5, color="#333", lw=1.2)
    ax.set_xticks(range(len(cols)), [c if c != "E0" else "E0\n(참고)" for c in cols], fontsize=8)
    ax.set_yticks(range(len(xs_)), xs_, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{m}: {EXCLUSIONS[m][0]} — 10종목 CAGR − 같은 날 같은 수의 가짜 제외 중앙값 (%p, 1993–2023)\n"
                 f"파랑 = 가짜보다 높음, 테두리 = 두 반쪽 모두 가짜보다 높음 (b)"
                 + (f"; 회색 줄({', '.join(xs.VALUES_ONLY)})은 값만 표시" if plain else ""), fontsize=10)
    return figures._save(fig, out / "figures" / f"heatmap_{m}.png")


def fig_returns(out: Path, hist: pd.DataFrame, un: pd.DataFrame, st: pd.DataFrame, x: str = FIG_EXIT) -> Path:
    """② Per-signal returns kept by each rule vs kept by its pseudo exclusions (15 buy rules pooled)."""
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    rules = [m for m in EXCLUSIONS if m in set(hist["rule"])]
    fig, axes = plt.subplots(len(rules), 2, figsize=(13, 3.1 * len(rules)), squeeze=False, sharex=True)
    for r_, m in enumerate(rules):
        for c, h in enumerate(es.HALVES):
            ax = axes[r_, c]
            sel = (hist["rule"] == m) & (hist["exit"] == x) & (hist["period"] == h) & (hist["gen"] != "E0")
            pj._bar(ax, pj._density(hist[sel & (hist["kind"] == "pseudo")][hc].to_numpy().sum(axis=0)), color="#777", lw=1.3,
                    label="가짜 제외가 남긴 신호")
            pj._bar(ax, pj._density(hist[sel & (hist["kind"] == "rule")][hc].to_numpy().sum(axis=0)), color=figures.COLORS[0], lw=1.5,
                    label="제외 규칙이 남긴 신호")
            ax.axvline(0, color="#bbb", lw=0.8)
            u = un[(un["rule"] == m) & (un["exit"] == x) & (un["gen"] != "E0")]
            ur, up = u[u["kind"] == "rule"], u[u["kind"] == "pseudo"]
            s = st[(st["rule"] == m) & (st["exit"] == x) & (st["gen"] != "E0")]
            sr, sp = s[s["kind"] == "rule"], s[s["kind"] == "pseudo"]
            ax.text(0.98, 0.97, f"평균 {_pooled(ur, 'mean_ret', h) * 100:+.1f}% (가짜 {_pooled(up, 'mean_ret', h) * 100:+.1f}%)\n"
                                f"+50% 이상 {_pooled(ur, 'big50', h) * 100:.1f}% (가짜 {_pooled(up, 'big50', h) * 100:.1f}%)\n"
                                f"−20% 이하 {_pooled(ur, 'loss20', h) * 100:.1f}% (가짜 {_pooled(up, 'loss20', h) * 100:.1f}%)\n"
                                f"칸별 순위 검정 p<0.05: 규칙 {np.mean(sr['st_mw_p'] < 0.05) * 100:.0f}% · 가짜끼리 "
                                f"{np.mean(sp['st_mw_p'] < 0.05) * 100:.0f}% (1993–2023)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
            ax.set_title(f"{m}: {EXCLUSIONS[m][0]}, {h.replace('-', '–')}", fontsize=9)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("신호 하나의 수익률 (%, 편도 20bp 반영, 양 끝 칸은 −60% 미만 / +200% 초과)")
    fig.suptitle(f"신호 하나의 수익률: 제외 규칙이 남긴 신호 vs 가짜 제외가 남긴 신호 (매수 15개 합산, 매도 {x})", fontsize=11)
    return figures._save(fig, out / "figures" / "returns_kept_vs_pseudo.png")


def fig_gap(out: Path, v: pd.DataFrame, base: pd.DataFrame, gp: dict) -> Path:
    """③ 2017-2023 CAGR before (no exclusion) vs after, per rule, with SPY and equal weight."""
    plt = figures._plt()
    rules = [m for m in EXCLUSIONS if m in set(v["rule"])]
    fig, axes = plt.subplots(1, len(rules), figsize=(5.2 * len(rules), 5.2), squeeze=False, sharex=True, sharey=True)
    bl = base.set_index("baseline")
    spy, ew = float(bl.loc["SPY", "cagr_2017-2023"]) * 100, float(bl.iloc[2]["cagr_2017-2023"]) * 100
    g_all = v[v["gen"] != "E0"]
    lo = float(np.nanmin(g_all[["none_cagr_2017-2023", "cagr_2017-2023"]].to_numpy())) * 100 - 1
    hi = max(float(np.nanmax(g_all[["none_cagr_2017-2023", "cagr_2017-2023"]].to_numpy())) * 100, spy) + 1
    for ax, m in zip(axes[0], rules):
        g = g_all[g_all["rule"] == m]
        x1 = g["exit"] == "X1"
        ax.scatter(g.loc[~x1, "none_cagr_2017-2023"] * 100, g.loc[~x1, "cagr_2017-2023"] * 100, s=14, color=figures.COLORS[0],
                   alpha=0.7, label="1% 룰 매도 10종")
        ax.scatter(g.loc[x1, "none_cagr_2017-2023"] * 100, g.loc[x1, "cagr_2017-2023"] * 100, s=18, color=figures.COLORS[1],
                   marker="^", label="X1 (10% 고정)")
        ax.plot([lo, hi], [lo, hi], color="#999", lw=0.8)
        ax.axhline(spy, color="#c0392b", ls="--", lw=1, label=f"SPY {spy:.1f}%")
        ax.axvline(spy, color="#c0392b", ls="--", lw=1)
        ax.axhline(ew, color="#555", ls=":", lw=1, label=f"동일비중 {ew:.1f}%")
        d = gp.get(m, {})
        ax.text(0.02, 0.97, f"변화 중앙값 {d.get('median_change_pp', np.nan):+.1f}%p (가짜 {d.get('pseudo_median_change_pp', np.nan):+.1f}%p)"
                            f"\nSPY 이상 칸 {d.get('ge_spy_before', 0)} → {d.get('ge_spy_after', 0)} / {d.get('cells', 0)}"
                            f"\n격차: {d.get('verdict', '')}", transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{m}: {EXCLUSIONS[m][0]}", fontsize=9)
        ax.set_xlabel("제외 없음 2017–2023 CAGR (%)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("제외 후 2017–2023 CAGR (%)")
    axes[0, -1].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.suptitle("2017–2023 CAGR: 제외 전(가로) vs 후(세로), 매수 15 × 매도 11칸 — 대각선 위 = 제외 뒤 높아짐", fontsize=11)
    return figures._save(fig, out / "figures" / "gap_2017_2023.png")


def fig_excess(out: Path, v: pd.DataFrame, pv: pd.DataFrame) -> Path:
    """④ 10-stock CAGR minus the pseudo median (pp): each rule's cells vs pseudo-vs-pseudo, two halves."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    vv, pp = v[v["gen"] != "E0"], pv[pv["gen"] != "E0"]
    for ax, h in zip(axes, es.HALVES):
        b = pp[f"x_{h}"].to_numpy(dtype=float)
        parts = {m: ((g[f"cagr_{h}"] - g[f"pseudo_cagr_{h}"]) * 100).to_numpy(dtype=float) for m, g in vv.groupby("rule", sort=False)}
        allx = np.concatenate([b] + list(parts.values()))
        edges = np.linspace(np.floor(np.nanmin(allx)) - 0.5, np.ceil(np.nanmax(allx)) + 0.5, 41)
        ax.hist(b, edges, density=True, color="#bbbbbb", label="가짜 제외 vs 나머지 가짜")
        q = np.percentile(b, 95)
        lines = []
        for i, (m, a) in enumerate(parts.items()):
            ax.hist(a, edges, density=True, histtype="step", lw=1.7, color=figures.COLORS[i % len(figures.COLORS)], label=m)
            _, p = pj.mann_whitney(a, b)
            lines.append(f"{m}: 평균 {a.mean():+.2f}%p, 95% 선 넘음 {np.mean(a > q) * 100:.0f}%, p = {p:.3f}")
        ax.axvline(q, color="#555", ls="--", lw=1)
        ax.axvline(0, color="#999", lw=0.8)
        ax.text(0.02, 0.97, "\n".join(lines + [f"가짜: 평균 {b.mean():+.2f}%p (우연이면 95% 선 넘음 5%)"]), transform=ax.transAxes, va="top",
                fontsize=7.5, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.set_title(h.replace("-", "–"), fontsize=10)
        ax.set_xlabel("10종목 CAGR − 같은 칸 가짜 제외 중앙값 (%p)")
        ax.grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("제외 규칙별 매수 15 × 매도 11칸의 초과 수익 분포 vs 가짜 제외끼리 (밀도)", fontsize=11)
    return figures._save(fig, out / "figures" / "excess_vs_pseudo.png")


def fig_size(out: Path, v: pd.DataFrame, sz: dict) -> Path:
    """⑤ CAGR change against the cap-rank shift of the filled trades (size diagnostic)."""
    plt = figures._plt()
    rules = [m for m in EXCLUSIONS if m in set(v["rule"])]
    fig, axes = plt.subplots(1, len(rules), figsize=(5.2 * len(rules), 4.8), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], rules):
        g = v[(v["rule"] == m) & (v["gen"] != "E0")]
        ax.scatter(g["pseudo_cap_rank_trades"] - g["none_cap_rank_trades"], (g["pseudo_cagr_1993-2023"] - g["none_cagr_1993-2023"]) * 100,
                   s=10, color="#bbbbbb", label="가짜 제외 중앙값")
        ax.scatter(g["cap_rank_trades"] - g["none_cap_rank_trades"], (g["cagr_1993-2023"] - g["none_cagr_1993-2023"]) * 100, s=14,
                   color=figures.COLORS[0], alpha=0.8, label="제외 규칙")
        ax.axhline(0, color="#999", lw=0.8)
        ax.axvline(0, color="#999", lw=0.8)
        d = sz.get(m, {})
        ax.text(0.02, 0.97, f"순위상관 ρ = {d.get('rho_vs_none', np.nan):+.2f}"
                            + (" (크기 효과와 분리되지 않음)" if d.get("not_separated") else "")
                            + f"\n순위 이동 중앙값 {d.get('median_shift', np.nan):+.0f}", transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.set_title(f"{m}: {EXCLUSIONS[m][0]}", fontsize=9)
        ax.set_xlabel("채운 거래의 시총 순위 중앙값 변화 (음수 = 더 큰 회사)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("10종목 CAGR 변화 1993–2023 (%p, 제외 후 − 제외 없음)")
    axes[0, -1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("크기 진단: 제외가 보유 종목을 큰 회사 쪽으로 옮긴 만큼 CAGR이 바뀌었나", fontsize=11)
    return figures._save(fig, out / "figures" / "size_diagnostic.png")


def fig_candidates(out: Path, hist: pd.DataFrame, picks: list[tuple], title: str) -> Path:
    """⑥ Per-signal returns of the candidates (or the best market-test cells) vs their pseudo exclusions."""
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.8 * len(picks)), squeeze=False, sharex=True)
    for ax, (m, gen, x) in zip(axes[:, 0], picks):
        h = hist[(hist["rule"] == m) & (hist["gen"] == gen) & (hist["exit"] == x) & (hist["period"] == "1993-2023")]
        pj._bar(ax, pj._density(h[h["kind"] == "pseudo"][hc].to_numpy().sum(axis=0)), color="#777", lw=1.3, label="가짜 제외")
        pj._bar(ax, pj._density(h[h["kind"] == "rule"][hc].to_numpy().sum(axis=0)), color=figures.COLORS[0], lw=1.4, label="제외 규칙")
        ax.axvline(0, color="#bbb", lw=0.8)
        ax.set_title(f"{m} × {gen} × 매도 {x}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("신호 하나의 수익률 (%, 1993–2023, 편도 20bp 반영)")
    fig.suptitle(title, fontsize=11)
    return figures._save(fig, out / "figures" / "candidates_returns.png")


def write_figures(out: Path, v: pd.DataFrame, pv: pd.DataFrame, un: pd.DataFrame, st: pd.DataFrame, hist: pd.DataFrame,
                  base: pd.DataFrame, ch: dict) -> list[Path]:
    if not figures.available():
        return []
    made = [fig_heatmap(out, v, m) for m in EXCLUSIONS if m in set(v["rule"])]
    made += [fig_returns(out, hist, un, st), fig_gap(out, v, base, ch["gap"]), fig_excess(out, v, pv), fig_size(out, v, ch["size"])]
    if ch["candidates"]:
        picks = [(c["rule"], c["gen"], c["exit"]) for c in ch["candidates"]][:6]
        title = "후보의 신호 하나 수익률 vs 가짜 제외"
    else:  # no candidate: the cells passing the market test under the most exits
        mp = v[v["market_pass"]].groupby(["rule", "gen"]).agg(n=("exit", "size"), cagr=("cagr_1993-2023", "mean"))
        top = mp.sort_values(["n", "cagr"], ascending=False).head(5).index.tolist()
        picks = [(m, gen, v[(v["rule"] == m) & (v["gen"] == gen) & v["market_pass"]].sort_values("cagr_1993-2023").iloc[-1]["exit"])
                 for m, gen in top]
        title = "후보 없음: SPY 기준을 가장 많은 매도에서 통과한 칸의 신호 하나 수익률 vs 가짜 제외"
    if picks:
        made.append(fig_candidates(out, hist, picks, title))
    return made


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "r0x1", "tables", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--stage-l", default=None, help="Stage L 결과표 폴더 (재현 점검)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-workers", type=int, default=5, help="특징 계산 작업 수 (작업당 메모리 약 10GB)")
    ap.add_argument("--dry", action="store_true", help="E2·V3·E0(씨앗 2), 매도 11, 제외 없음은 Stage L 순서 100번, 규칙 순서 10, 가짜 4 × 2")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.dirs = {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events),
              "r0x1": Path(a.r0x1), "tables": Path(a.tables)}
    gens = generators(["E2", "V3"], pj.E0_SEEDS[:2]) if a.dry else generators()
    n_orders, n_pseudo, p_orders = (10, 4, 2) if a.dry else (N_ORDERS, N_PSEUDO, PSEUDO_ORDERS)
    a.cfg = {"bins": pj.BINS_J3, "seeds": pj.E0_SEEDS, "base": str(out / "base.parquet")}  # all 5 seeds: the tables' seed index
    ts.log(f"1단계: 신호 기준표 (매수 {len(gens)} × 매도 {len(EXITS)}) + 특징 {len(FEATURES)}개 · 시총 순위")
    base = build_base(a, gens)
    ts.log(f"  신호 {len(base):,}개 (" + ", ".join(f"{s} {int((base['src'] == s).sum()):,}" for ss in gens.values() for s in ss) + ")")
    for m in EXCLUSIONS:
        ts.log(f"  {m} 제외 비율 {excluded(base, m).mean():.1%}")
    rules = list(EXCLUSIONS)
    jobs = []
    for gen, srcs in gens.items():
        for x in EXITS:
            if gen == "E0":  # heavy: one job per part
                jobs.append(("E0", x, N_ORDERS, n_orders, n_pseudo, p_orders, pj.COST_BPS, srcs, []))
                jobs += [("E0", x, 0, n_orders, n_pseudo, p_orders, pj.COST_BPS, srcs, [m]) for m in rules]
            else:
                jobs.append((gen, x, N_ORDERS, n_orders, n_pseudo, p_orders, pj.COST_BPS, srcs, rules))
    jobs.sort(key=lambda j: j[0] != "E0")  # the long E0 jobs first
    jdir = out / "jobs"
    todo = [j for j in jobs if not (jdir / f"{job_name(j[0], j[1], j[6], j[8])}_acc.csv").exists()]
    ts.log(f"2단계: 작업 {len(todo)}개 남음 (매수 {len(gens)} × 매도 {len(EXITS)}, 규칙 {len(rules)}, 가짜 {n_pseudo}개씩)")
    run_jobs(todo, a)
    read = lambda s: pd.concat([pd.read_csv(f) for f in sorted(jdir.glob(f"*_{s}.csv"))  # noqa: E731
                                if "bp_" not in f.name and f.stat().st_size > 2], ignore_index=True)  # (empty: no rule in the job)
    acc, un, st, hist = read("acc"), read("un"), read("stats"), read("hist")
    pj._init(a.panel, a.eligible, a.dirs, a.cfg)
    bl = es.baselines(pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"])
    bl.to_csv(out / "baselines.csv", index=False)
    if a.stage_l:
        rp = repro(acc, un, Path(a.stage_l))
        rp.to_csv(out / "repro.csv", index=False)
        bad = rp[rp["diff"] > 1e-9]
        ts.log(f"재현 점검: Stage L 대비 {rp[['set', 'exit']].drop_duplicates().shape[0]}칸 {len(rp)}값, "
               f"최대 차 {rp['diff'].max():.2e}, 1e-9 넘는 값 {len(bad)}개")
    v = summarize(acc, un)
    pv = pseudo_vs_pseudo(acc, un)
    v["market_pass"] = market_pass(v, bl)
    v["candidate_20bp"] = v["market_pass"] & (v.groupby(["rule", "gen"])["market_pass"].transform("sum") >= MIN_EXITS)
    v["candidate_40bp"] = False
    cand = v[v["candidate_20bp"]]
    if len(cand):  # candidate cells again at 40 bp (the rule only)
        run_jobs([(r.gen, r.exit, 0, n_orders, 0, 0, pj.CHECK_COST_BPS, gens[r.gen], [r.rule]) for r in cand.itertuples()
                  if not (jdir / f"{job_name(r.gen, r.exit, pj.CHECK_COST_BPS, [r.rule])}_acc.csv").exists()], a)
        spy = bl.set_index("baseline").loc["SPY"]
        for i, r in cand.iterrows():
            g = pd.read_csv(jdir / f"{job_name(r['gen'], r['exit'], pj.CHECK_COST_BPS, [r['rule']])}_acc.csv")
            v.loc[i, "candidate_40bp"] = all(g[f"cagr_{p}"].median() > spy[f"cagr_{p}"] and abs(g[f"mdd_{p}"].median()) < abs(spy[f"mdd_{p}"])
                                             for p in ("1993-2023", "2017-2023"))
    v["candidate"] = v["candidate_40bp"] & (v.groupby(["rule", "gen"])["candidate_40bp"].transform("sum") >= MIN_EXITS)
    sr_daily = v["m_sr"].dropna().to_numpy()
    sr_var = float(np.var(sr_daily)) if len(sr_daily) else np.nan
    spy_r = pd.Series(pj._W["px"].spy_c[pj._W["w0"]:pj._W["w0"] + pj._W["n"]]).pct_change().dropna()
    spy_sr = float(spy_r.mean() / spy_r.std())
    for i, r in v[v["candidate_20bp"]].iterrows():
        if pd.notna(r["m_sr"]):
            v.loc[i, "dsr"] = ks.dsr(r["m_sr"], r["m_g3"], r["m_g4"], int(r["m_T"]), N_TRIALS_BEFORE + N_NEW, sr_var)
            v.loc[i, "psr_spy"] = ks.psr(r["m_sr"], r["m_g3"], r["m_g4"], int(r["m_T"]), spy_sr)
            v.loc[i, "psr_pseudo"] = ks.psr(r["m_sr"], r["m_g3"], r["m_g4"], int(r["m_T"]), r["pseudo_sharpe"] / np.sqrt(252))
    v["label"] = v["rule"].map(lambda m: EXCLUSIONS[m][0])
    v.to_csv(out / "verdicts.csv", index=False)
    pv.to_csv(out / "pseudo_vs_pseudo.csv", index=False)
    st.to_csv(out / "stats.csv", index=False)
    hist.to_csv(out / "hist.csv", index=False)
    un.to_csv(out / "unlimited.csv", index=False)
    acc.groupby(["rule", "gen", "exit", "kind", "k"]).median(numeric_only=True).to_csv(out / "accounts.csv")
    kept = {m: {"base_share": float(excluded(base, m).mean()),
                "kept_share_median": float(v.loc[v["rule"] == m, "kept_share"].median())} for m in EXCLUSIONS}
    ch = {"by_rule": chance(v, pv), "gap": gap(v, bl), "size": size_diagnostic(v), "x1": x1_check(v),
          "stats": stats_summary(st), "excluded": kept,
          "candidates_20bp": v.loc[v["candidate_20bp"], ["rule", "gen", "exit"]].to_dict("records"),
          "candidates": v.loc[v["candidate"], ["rule", "gen", "exit"]].to_dict("records"),
          "trials": {"before": N_TRIALS_BEFORE, "new": N_NEW}, "sr_var_daily": sr_var}
    (out / "chance.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, v, pv, un, st, hist, bl, ch)
    for m, c in ch["by_rule"].items():
        be = c["by_exit"]
        ts.log(f"완료 {m} · 매도별 (b) 통과 / 우연 95% 분위: " + ", ".join(f"{x} {be[x]['b']['observed']}/{be[x]['b']['q95']}" for x in be)
               + f" · 정보 있음 매도 수 (a) {len(c['exits_with_information']['a'])} · (b) {len(c['exits_with_information']['b'])}"
               + f" · 격차 {ch['gap'][m]['verdict']} ({ch['gap'][m]['median_change_pp']:+.1f}%p)")
    ts.log(f"후보 {len(ch['candidates'])} (20bp {len(ch['candidates_20bp'])}칸)")


if __name__ == "__main__":
    main(sys.argv[1:])
