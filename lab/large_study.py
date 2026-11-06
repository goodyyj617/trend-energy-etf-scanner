"""Stage Q: large-cap universe on the regime-gated account (DECISIONS 96).

    python -m lab.large_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                              --r0x1 DIR --tables DIR --out DIR [--stage-p DIR] [--workers N] [--feature-workers N] [--dry]

Stage P's M1 accounts (lab/regime_study.py: the stored Stage L tables, the same random orders, the
regime series and the replay) with entries limited by the signal-day 21-day average cap rank of the
base table: U3000 (all, = Stage P), U1000 (rank <= 1,000) and U500 (<= 500); holdings keep their
exits. Settings N10 / W2 / P0 and N10 / N20 / N50 x W2 x P1, exits X1 and H1, gates none and
R1 / R3 / R4 / R5 A1. Pseudo restrictions: per band, 10 seeded day-matched random keeps of the
band's daily count among the M1-passing signals (select_study.pseudo_mask), at N10 / W2 / P0
(20 orders, gates none and R5 A1) and N20 / W2 / P1 (all five gates: the distribution the other
cells borrow), exit X1, the 15 entries. Every curve is measured as Stage O / P plus the fills' cap
rank and SPY beta, and (first order) the positions and cash share; the benchmarks add per band the
top-N equal-weight universe (rebuilt daily, 20 bp one-way on its membership changes) and that
universe under each regime. --stage-p checks the U3000 cells against Stage P's goal table; --dry
adds a band whose limit is above every rank and must equal U3000. Output (aggregates only):
base.parquet, regimes.parquet, jobs/*, goal.csv, systems.csv, helps.csv, help_counts.csv, cells.csv,
pseudo.csv, benchmarks.csv, candidates.csv, chance.csv, trade_stats.csv, repro.csv, windows.csv,
chance.json, figures/.
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

from . import account_study as ns, exclude_study as ms, figures, goal_study as gs, pair_study as pj
from . import regime_study as rs, select_study as ks, tail_study as ts
from .crsp_study import load_panel_with_cash

# ---- registered (DECISIONS 96): fixed before any result
SETTINGS = list(rs.SETTINGS)
P1_SETTINGS = SETTINGS[1:]
EXITS = list(rs.EXITS)
REGIMES = ("R1", "R3", "R4", "R5")  # the four that beat their placebos in 95; R2 and A2 dropped
FILTERS = ["none"] + [f"{r}A1" for r in REGIMES]
BANDS = {"U3000": np.inf, "U1000": 1000.0, "U500": 500.0}  # signal-day cap rank limit (Stage D's band edges)
NARROW = ("U1000", "U500")
BAND_LABELS = {"U3000": "상위 3,000 (전부)", "U1000": "시총 1,000위 이내", "U500": "시총 500위 이내"}
N_PSEUDO = 10
PSEUDO_SETTINGS = [(10, "W2", "P0"), (20, "W2", "P1")]
PSEUDO_FILTERS = {(10, "W2", "P0"): ["none", "R5A1"], (20, "W2", "P1"): list(FILTERS)}
PSEUDO_REF = (20, "W2", "P1")  # the setting whose pseudo distribution the other cells borrow (same gate)
PSEUDO_EXIT, PSEUDO_ORDERS = "X1", 20
TRADE_SETTING, TRADE_EXIT, TRADE_FILTER = (20, "W2", "P1"), "X1", "R5A1"
HELP_PP, HALF_TOL = 0.005, 0.010  # Q1: +0.5 pp in 2008-2023, not more than 1.0 pp lower in 1993-2007
N_NEW = 1_200  # 2 bands x 5 gates x 4 settings x 15 entries x 2 exits
N_TRIALS = rs.N_TRIALS + N_NEW  # 74,174 for the candidates' DSR
DRY_BAND = ("U9999", 9_999.0)  # dry: a limit above every rank (the panel ranks to 3,000)
SEED = 96
_W: dict = {}


def in_band(cap: np.ndarray, limit: float) -> np.ndarray:
    """Signals a band may buy: signal-day cap rank at or below the limit (an unbounded band keeps all)."""
    if not np.isfinite(limit):
        return np.ones(len(cap), dtype=bool)
    return np.asarray(cap, dtype=float) <= limit


def pseudo_band(si: np.ndarray, keep: np.ndarray, band: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A day-matched random restriction: on every day, as many of the M1-passing signals (`keep`) as
    the band kept that day (`band`, a subset of keep), chosen at random among them."""
    idx = np.flatnonzero(keep)
    out = np.zeros(len(si), dtype=bool)
    if len(idx):
        out[idx] = ks.pseudo_mask(si[idx], band[idx], rng)
    return out


def pseudo_rng(band: str, src: str, k: int) -> np.random.Generator:
    """The same restriction for every exit, setting and gate of a source."""
    return np.random.default_rng([SEED, zlib.crc32(f"{band}|{src}".encode()), k])


def top_n_ew(close: pd.DataFrame, member, cost: float) -> np.ndarray:
    """Close-based level of an equal-weight universe rebuilt daily: names that are members at the close
    of t-1 earn their close-to-close return on t; one-way `cost` on the weight sold and bought when the
    membership changes (members = the hygiene universe and no cost give tail_study.equal_weight_index)."""
    r = (close / close.ffill().shift(1) - 1.0).to_numpy(dtype=float)
    m = np.asarray(member, dtype=bool)
    held = np.vstack([np.zeros((1, m.shape[1]), dtype=bool), m[:-1]])
    h = held & np.isfinite(r)
    n = h.sum(axis=1)
    day = np.where(n > 0, np.where(h, r, 0.0).sum(axis=1) / np.maximum(n, 1), 0.0)
    if cost:
        prev = np.vstack([np.zeros((1, m.shape[1]), dtype=bool), held[:-1]])
        nh, npv = held.sum(axis=1), prev.sum(axis=1)
        turn = np.where(nh > 0, (held & ~prev).sum(axis=1) / np.maximum(nh, 1), 0.0) \
            + np.where(npv > 0, (prev & ~held).sum(axis=1) / np.maximum(npv, 1), 0.0)
        day = day - cost * turn
    return np.cumprod(1.0 + day)


def beta(eq: np.ndarray, spy_r: np.ndarray) -> float:
    d = eq[1:] / eq[:-1] - 1.0
    v = float(np.var(spy_r))
    return float(np.mean((d - d.mean()) * (spy_r - spy_r.mean())) / v) if v > 0 else np.nan


# --------------------------------------------------------------------------- workers

def _init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    rs._init(panel_dir, eligible, dirs, cfg)
    _W.update(rs._W)
    spy = _W["px"].spy_c[_W["w0"]:_W["w0"] + _W["n"]]
    _W["spy_r"] = spy[1:] / spy[:-1] - 1.0


def _sources(gen: str, x: str, srcs: list[str]):
    """Per source: rows, signal indices, the M1 keep, Stage M's pseudo exclusions (for the random
    draws), the base rows' M1 keep, signal days and cap ranks."""
    out = {}
    for s in srcs:
        d = ms.prepared(s, x)
        b = ms.source_base(s)
        r = gs.GRows(d, b, x)
        keep_b = ~ms.excluded(b, ns.RULE)
        si_b = b["si"].to_numpy()
        pseudos = [ks.pseudo_mask(si_b, keep_b, np.random.default_rng([ms.SEED, zlib.crc32(f"{ns.RULE}|{s}".encode()), k]))[r.brow]
                   for k in range(ms.N_PSEUDO)]
        out[s] = {"rows": r, "si": d["si"].to_numpy(), "keep": keep_b[r.brow], "pseudos": pseudos, "keep_b": keep_b,
                  "si_b": si_b, "cap_b": b["cap_rank"].to_numpy(dtype=float)}
    return out


def _keys(gen: str, x: str, st: tuple, srcs: list[str], src: dict, limit: int):
    """Stage P's priorities: P0 = Stage M / N / O / P's random orders (the first `limit` per source),
    P1 = the M1 feature with Stage N's tie-break, P2 = cap."""
    if st[2] == "P0":
        keeps = {s: src[s]["keep"] for s in srcs}
        pseudos = {s: src[s]["pseudos"] for s in srcs}
        for kind, k, s, o, key in ns.stage_m_draws(gen, x, srcs, keeps, pseudos, {s: len(src[s]["rows"]) for s in srcs}):
            if kind == "rule" and o < limit:
                yield s, o, key
    else:
        for s in srcs:
            r = src[s]["rows"]
            tie = np.random.default_rng([ns.SEED, zlib.crc32(f"tie|{s}|{x}".encode())]).random(len(r))
            yield s, 0, (r.k2 + tie * 1e-9) if st[2] == "P1" else (-r.cap + tie * 1e-6)


def band_masks(src: dict, s: str, bands: list, mode: str) -> dict:
    """Per band name (pseudo: 'U500#k'), the rows the account may buy (the M1 keep within the band)."""
    v = src[s]
    r = v["rows"]
    out = {}
    for name, limit in bands:
        band_b = v["keep_b"] & in_band(v["cap_b"], limit)
        if mode == "grid":
            out[name] = band_b[r.brow]
        else:
            for k in range(N_PSEUDO):
                out[f"{name}#{k}"] = pseudo_band(v["si_b"], v["keep_b"], band_b, pseudo_rng(name, s, k))[r.brow]
    return out


def run_job(job):
    """mode 'grid': one buy rule x exit x setting under the given gates, each in every band (one key
    draw replayed in all bands); 'pseudo': the same with each band's 10 day-matched restrictions."""
    gen, x, st, filters, bands, cost_bps, srcs, mode = job
    t0 = time.time()
    cost = cost_bps / 10_000
    n_slots = st[0]
    src = _sources(gen, x, srcs)
    wts = {s: ns.sizing(src[s]["rows"].stop, n_slots, "W2")[0] for s in srcs}
    masks = {s: band_masks(src, s, bands, mode) for s in srcs}
    names = list(masks[srcs[0]])
    limit = PSEUDO_ORDERS if mode == "pseudo" else ms.N_ORDERS // len(srcs)
    keep_trades = st == TRADE_SETTING and x == TRADE_EXIT and gen != "E0" and cost_bps == pj.COST_BPS
    accs, wins, moms, trades = [], [], [], []
    for f in filters:
        state = None if f == "none" else _W["states"][f[:2]]
        curves = {nm: [] for nm in names}
        extra = {nm: [] for nm in names}
        for s, o, key in _keys(gen, x, st, srcs, src, limit):
            r = src[s]["rows"]
            for nm in names:
                m = masks[s][nm]
                log = {} if o == 0 and mode == "grid" else None
                eq, tr = rs._replay(r, src[s]["si"], m, wts[s], key, n_slots, cost, state, False, log)
                sel = m & np.isfinite(wts[s])
                if state is not None:
                    sel &= state[src[s]["si"]]
                bought = np.flatnonzero(sel)[tr["row"].to_numpy()] if len(tr) else np.array([], dtype=int)
                curves[nm].append((s, o, len(tr), eq))
                ex = {**rs.crisis_mdd(eq, _W["wdates"]), "beta": beta(eq, _W["spy_r"]),
                      "cap_fill": float(np.median(r.cap[bought])) if len(bought) else np.nan}
                if log is not None:
                    ex.update({"time_in_market": float((log["positions"] > 0).mean()), "invested": float(np.mean(log["invested"])),
                               "positions_mean": float(np.mean(log["positions"])), "cash_share": float(1.0 - np.mean(log["invested"])),
                               "turnover": float(tr["weight"].sum() / ((_W["wdates"][-1] - _W["wdates"][0]).days / 365.25)) if len(tr) else 0.0})
                extra[nm].append(ex)
                if keep_trades and f in (TRADE_FILTER, "none"):
                    trades.append(pd.DataFrame({"filter": f, "band": nm, "src": s, "ret": tr["ret"].to_numpy(dtype="float32"),
                                                "cap": r.cap[bought].astype("float32")}))
        for nm in names:
            r_, w_, mom = gs._measure(curves[nm], cost_bps)
            head = {"setting": ns.setting_name(st), "filter": f, "band": nm}
            accs += [{**head, **a_, **e_} for a_, e_ in zip(r_, extra[nm])]
            if mode == "grid":
                wins += [{**head, **w} for w in w_]
                if mom is not None:
                    moms.append({**head, **mom})
    ms._W["prep"].clear()
    pj._W["tables"].clear()
    tail = {"gen": gen, "exit": x, "cost": cost_bps, "mode": mode}
    frame = lambda rows: pd.DataFrame([{**r, **tail} for r in rows])  # noqa: E731
    tr_all = pd.concat(trades, ignore_index=True).assign(gen=gen) if trades else pd.DataFrame()
    return job, frame(accs), frame(wins), frame(moms), tr_all, time.time() - t0


def job_name(gen: str, x: str, st: tuple, filters: list, cost: float, mode: str) -> str:
    tag = "_".join(filters) if len(filters) <= 2 else "all"
    return f"{gen}__{x}__{ns.setting_name(st)}__{mode}__{tag}" + ("" if cost == pj.COST_BPS else f"__{cost:g}bp")


def _name(j) -> str:
    return job_name(j[0], j[1], j[2], j[3], j[5], j[7])


def run_jobs(jobs: list, a) -> None:
    d = Path(a.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs, a.cfg)) as ex:
        for i, (job, acc, win, mom, tr, sec) in enumerate(ex.map(run_job, jobs, chunksize=1)):
            nm = _name(job)
            for suffix, f in (("win", win), ("mom", mom)):
                f.to_csv(d / f"{nm}_{suffix}.csv", index=False)
            if len(tr):
                tr.to_parquet(d / f"{nm}_trades.parquet")
            acc.to_csv(d / f"{nm}_acc.csv", index=False)  # last: its presence marks the job done
            if (i + 1) % 5 == 0 or i + 1 == len(jobs):
                ts.log(f"  [{i + 1}/{len(jobs)}] {nm} {sec:.0f}s · 남은 예상 {(len(jobs) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


# --------------------------------------------------------------------------- benchmarks

def band_benchmarks(panel, elig: pd.DataFrame, wdates: pd.DatetimeIndex, w0: int, cash: np.ndarray, states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per band: the top-N equal-weight universe (hygiene universe within the band, rebuilt daily,
    20 bp one-way on membership changes) + T-bill, and that universe under each regime (cash when
    the regime was off at the previous close, 20 bp per switch), each at its own w*."""
    cost = pj.COST_BPS / 10_000
    close = panel.close
    e = elig.reindex(index=close.index, columns=close.columns, fill_value=False).to_numpy(dtype=bool)
    cap = panel.extra["cap_rank"].reindex(index=close.index, columns=close.columns).to_numpy(dtype=float)
    n = len(wdates)
    st_w = {r: states[r].reindex(wdates).fillna(False).to_numpy(dtype=bool) for r in REGIMES}
    names, curves = [], []
    for b, limit in BANDS.items():
        member = e if not np.isfinite(limit) else e & (cap <= limit)
        full = top_n_ew(close, member, cost)
        ew = full[w0:w0 + n] / full[w0]
        names.append((b, "none"))
        curves.append(ew)
        for r in REGIMES:
            names.append((b, r))
            curves.append(rs.ew_regime(ew, cash, st_w[r], cost))
    sp = gs._setup(cash, wdates)
    EQ = np.column_stack(curves)
    R = EQ[1:] / EQ[:-1] - 1.0
    gm = gs.goal_metrics(R, np.repeat(sp["c"][:, None], R.shape[1], axis=1), sp["years"], sp["wins"])
    rows, by = [], {}
    for i, (b, r) in enumerate(names):
        lab = f"{BAND_LABELS[b]} 동일비중" + ("" if r == "none" else f" × {r} {rs.REGIMES[r]}") + " + 국채"
        row = {"benchmark": f"ew_{b}" + ("" if r == "none" else f"_{r}"), "label": lab, "band": b, "regime": r,
               **{k: float(v[i]) for k, v in gm.items() if k != "win_cagr"}, **ts_period_stats(EQ[:, i], wdates)}
        rows.append(row)
        by[(b, r)] = {**row, "win_cagr": gm["win_cagr"][:, i], "eq": EQ[:, i]}
    return pd.DataFrame(rows), by


def ts_period_stats(eq: np.ndarray, wdates: pd.DatetimeIndex) -> dict:
    from .entry_study import period_stats
    return period_stats(pd.Series(eq, index=wdates))


# --------------------------------------------------------------------------- verdicts

KEYS = ["setting", "gen", "exit", "filter", "band", "cost", "mode"]


def summarize(acc: pd.DataFrame) -> pd.DataFrame:
    g = acc.groupby(KEYS, sort=False)
    med = g.median(numeric_only=True).drop(columns=["order"], errors="ignore")
    med["orders"] = g.size()
    return med.reset_index()


SYS_COLS = ["w", "w_1993-2007", "w_2008-2023", "dd_mdd", "cagr_1993-2023", "mdd_1993-2023", "cagr_2017-2023", "trades",
            "mdd_2000-2002", "mdd_2007-2009", "time_in_market", "turnover", "cap_fill", "beta", "positions_mean", "cash_share"] \
    + [f"dd_cagr_{p}" for p in gs.GOAL_PERIODS] + [f"{m}_{b}" for b in gs.BENCHMARKS for m in ("ws", "worst")]


def ew_ref(f: str) -> str:
    return "none" if f == "none" else f[:2]


def systems(g: pd.DataFrame, bench: dict, ewb: dict) -> pd.DataFrame:
    """Per setting x exit x gate x band: medians over the 15 entries; goal-passing as Stage O / P;
    also beating the band's equal-weight universe under the same gate in the three judged periods."""
    sel = (g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)
    cells = g[sel & (g["gen"] != "E0")]
    e0 = g[sel & (g["gen"] == "E0")].set_index(["setting", "exit", "filter", "band"])
    rows = []
    for (st, x, f, b), c in cells.groupby(["setting", "exit", "filter", "band"], sort=False):
        row = {"setting": st, "exit": x, "filter": f, "band": b, "entries": len(c), **{k: float(c[k].median()) for k in SYS_COLS if k in c}}
        row["system_pass"] = gs.beats(row, bench)
        ref = ewb.get((b, ew_ref(f)))
        row["beats_ew_band"] = bool(ref is not None and all(row[f"dd_cagr_{p}"] > ref[f"dd_cagr_{p}"] for p in gs.JUDGED))
        for k in ["w"] + [f"dd_cagr_{p}" for p in gs.GOAL_PERIODS]:
            row[f"e0_{k}"] = float(e0.loc[(st, x, f, b), k]) if (st, x, f, b) in e0.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pseudo_systems(g: pd.DataFrame) -> pd.DataFrame:
    """Per pseudo setting x gate x band x restriction k: median over the 15 entries (exit X1)."""
    p = g[(g["mode"] == "pseudo") & (g["gen"] != "E0") & (g["cost"] == pj.COST_BPS)].copy()
    if not len(p):
        return pd.DataFrame(columns=["setting", "filter", "band", "k"])
    p["k"] = p["band"].str.split("#").str[1].astype(int)
    p["band"] = p["band"].str.split("#").str[0]
    cols = ["w", "cagr_1993-2023", "mdd_1993-2023", "trades", "cap_fill"] + [f"dd_cagr_{q}" for q in gs.GOAL_PERIODS]
    return p.groupby(["setting", "filter", "band", "k"])[cols].median().reset_index()


def pseudo_ref(setting: str, x: str, f: str) -> str:
    st = ns.parse_setting(setting)
    own = st in PSEUDO_FILTERS and f in PSEUDO_FILTERS[st] and x == PSEUDO_EXIT
    return setting if own else ns.setting_name(PSEUDO_REF)


def _helps(v: dict, u: dict) -> tuple[bool, float, float]:
    d08 = v["dd_cagr_2008-2023"] - u["dd_cagr_2008-2023"]
    d93 = v["dd_cagr_1993-2007"] - u["dd_cagr_1993-2007"]
    return bool(d08 >= HELP_PP and d93 >= -HALF_TOL), d08, d93


def q1(sysm: pd.DataFrame, ps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """'Large caps help' per setting x exit x gate x band: +0.5 pp over U3000 in 2008-2023, not more
    than 1.0 pp below it in 1993-2007, and above the maximum of the band's 10 pseudo restrictions
    (own setting when run, else N20 / W2 / P1 x X1 at the same gate); counted over the 8 systems per
    band x gate against Binomial(8, p0), p0 = a pseudo passing the same test against the other 9."""
    u = sysm[sysm["band"] == "U3000"].set_index(["setting", "exit", "filter"])
    rows = []
    for _, r in sysm[sysm["band"].isin(NARROW)].iterrows():
        if (r["setting"], r["exit"], r["filter"]) not in u.index:
            continue
        ub = u.loc[(r["setting"], r["exit"], r["filter"])]
        ref_st = pseudo_ref(r["setting"], r["exit"], r["filter"])
        pv = ps[(ps["setting"] == ref_st) & (ps["band"] == r["band"]) & (ps["filter"] == r["filter"])]["dd_cagr_2008-2023"].to_numpy()
        above_u, d08, d93 = _helps(r, ub)
        pmax = float(pv.max()) if len(pv) else np.nan
        rows.append({"setting": r["setting"], "exit": r["exit"], "filter": r["filter"], "band": r["band"], "pseudo_from": ref_st,
                     "delta_1993-2023": r["dd_cagr_1993-2023"] - ub["dd_cagr_1993-2023"], "delta_1993-2007": d93, "delta_2008-2023": d08,
                     "pseudo_max": pmax, "pseudo_median": float(np.median(pv)) if len(pv) else np.nan,
                     "above_u3000": above_u, "above_pseudo": bool(len(pv) and r["dd_cagr_2008-2023"] > pmax),
                     "helps": bool(above_u and len(pv) and r["dd_cagr_2008-2023"] > pmax)})
    h = pd.DataFrame(rows)
    pp = []  # each pseudo against U3000 and the other 9 of its band
    for (st, f, b), g_ in ps.groupby(["setting", "filter", "band"]):
        if (st, PSEUDO_EXIT, f) not in u.index:
            continue
        ub = u.loc[(st, PSEUDO_EXIT, f)]
        for _, r in g_.iterrows():
            o = g_[g_["k"] != r["k"]]["dd_cagr_2008-2023"].to_numpy()
            ok = _helps(r, ub)[0] and len(o) and r["dd_cagr_2008-2023"] > o.max()
            pp.append({"setting": st, "filter": f, "band": b, "k": r["k"], "pass": bool(ok)})
    pp = pd.DataFrame(pp, columns=["setting", "filter", "band", "k", "pass"])
    cnt = []
    for b in NARROW:
        for f in FILTERS:
            hf = h[(h["band"] == b) & (h["filter"] == f)] if len(h) else h
            if not len(hf):
                continue
            pf = pp[(pp["band"] == b) & (pp["filter"] == f)]
            p0 = float(pf["pass"].mean()) if len(pf) else np.nan
            cnt.append({"band": b, "filter": f, "systems": len(hf), "helps": int(hf["helps"].sum()),
                        "above_u3000": int(hf["above_u3000"].sum()), "above_pseudo": int(hf["above_pseudo"].sum()),
                        "pseudo_tests": len(pf), "p0": p0, "q95": ks.q95(len(hf), p0) if np.isfinite(p0) else len(hf)})
    c = pd.DataFrame(cnt)
    if len(c):
        c["beyond_chance"] = c["helps"] > c["q95"]
    return h, c


def candidates(g: pd.DataFrame, bench: dict, pass40: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Narrowed cells (20 bp) with Stage O's per-cell test; per P1 setting x entry x gate x band: both
    exits at 20 and 40 bp, the same entry x gate passing at an adjacent P1 setting (same band) and at
    the other narrowed band (same setting); entry-specific by the same cell's E0."""
    c = g[(g["mode"] == "grid") & (g["cost"] == pj.COST_BPS) & g["band"].isin(NARROW)].copy()
    c["cell_pass"] = [gs.cell_pass(r, bench) for _, r in c.iterrows()]
    e0 = c[c["gen"] == "E0"].set_index(["setting", "exit", "filter", "band"])
    c["beats_e0"] = [all(r[f"dd_cagr_{h}"] > e0.loc[(r["setting"], r["exit"], r["filter"], r["band"]), f"dd_cagr_{h}"] for h in gs.es.HALVES)
                     if (r["setting"], r["exit"], r["filter"], r["band"]) in e0.index else False for _, r in c.iterrows()]
    c["pass_40bp"] = [bool(pass40.get((r["setting"], r["gen"], r["exit"], r["filter"], r["band"]), False)) for _, r in c.iterrows()]
    p1 = [ns.setting_name(s) for s in P1_SETTINGS]
    cc = c[(c["gen"] != "E0") & c["setting"].isin(p1)]
    keys = ["setting", "gen", "filter", "band"]
    if not len(cc):
        return c, pd.DataFrame(columns=keys + ["n20", "n40", "n_e0", "ok", "neighbour_n", "neighbour_band", "candidate", "entry_specific"])
    s = cc.groupby(keys).agg(n20=("cell_pass", "sum"), n40=("pass_40bp", "sum"),
                             n_e0=("beats_e0", lambda b: int((b & cc.loc[b.index, "cell_pass"]).sum()))).reset_index()
    s["ok"] = (s["n20"] >= len(EXITS)) & (s["n40"] >= len(EXITS))
    ok = set(zip(s.loc[s["ok"], "setting"], s.loc[s["ok"], "gen"], s.loc[s["ok"], "filter"], s.loc[s["ok"], "band"]))
    other = {"U1000": "U500", "U500": "U1000"}
    s["neighbour_n"] = [any((p1[j], gen, f, b) in ok for j in (p1.index(st) - 1, p1.index(st) + 1) if 0 <= j < len(p1))
                        for st, gen, f, b in zip(s["setting"], s["gen"], s["filter"], s["band"])]
    s["neighbour_band"] = [(st, gen, f, other[b]) in ok for st, gen, f, b in zip(s["setting"], s["gen"], s["filter"], s["band"])]
    s["candidate"] = s["ok"] & s["neighbour_n"] & s["neighbour_band"]
    s["entry_specific"] = s["candidate"] & (s["n_e0"] >= len(EXITS))
    return c, s


def chance(acc: pd.DataFrame, cells: pd.DataFrame, bench: dict) -> pd.DataFrame:
    """Per narrowed setting x exit x gate x band: passing entries against Binomial(15, p0), p0 from
    the 5 E0 seeds under the same per-cell test."""
    e0 = acc[(acc["gen"] == "E0") & (acc["mode"] == "grid") & (acc["cost"] == pj.COST_BPS) & acc["band"].isin(NARROW)]
    seeds = e0.groupby(["setting", "exit", "filter", "band", "src"]).median(numeric_only=True).reset_index()
    seeds["cell_pass"] = [gs.cell_pass(r, bench) for _, r in seeds.iterrows()]
    rows = []
    for (st, x, f, b), g_ in cells[cells["gen"] != "E0"].groupby(["setting", "exit", "filter", "band"], sort=False):
        sd = seeds[(seeds["setting"] == st) & (seeds["exit"] == x) & (seeds["filter"] == f) & (seeds["band"] == b)]
        p0 = float(sd["cell_pass"].mean()) if len(sd) else np.nan
        rows.append({"setting": st, "exit": x, "filter": f, "band": b, "entries": len(g_), "passing": int(g_["cell_pass"].sum()),
                     "passing_list": " ".join(sorted(g_.loc[g_["cell_pass"], "gen"])), "seeds_passing": int(sd["cell_pass"].sum()),
                     "p0": p0, "q95": ks.q95(len(g_), p0) if np.isfinite(p0) else len(g_)})
    out = pd.DataFrame(rows)
    if len(out):
        out["beyond_chance"] = out["passing"] > out["q95"]
    return out


def trade_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """N20 / W2 / P1 x X1 x R5 A1, pooled over the 15 entries: each band's and each pseudo
    restriction's trade returns against U3000's; each band against its pooled pseudos and each pseudo
    against its band's other 9 pooled (the share of p < 0.05 among pseudos is the yardstick)."""
    t = trades[trades["filter"] == TRADE_FILTER]
    rng = np.random.default_rng([SEED, 7])
    fin = lambda a: a[np.isfinite(a)]  # noqa: E731
    shares = lambda a: {"mean": float(a.mean()), "median": float(np.median(a)), "loss20": float((a <= -0.2).mean()),  # noqa: E731
                        "big50": float((a >= 0.5).mean())}
    by = {b: fin(g_["ret"].to_numpy(dtype=float)) for b, g_ in t.groupby("band")}
    caps = {b: g_["cap"].to_numpy(dtype=float) for b, g_ in t.groupby("band")}
    if "U3000" not in by:
        return pd.DataFrame()
    rows = []

    def add(kind, name, r, ref, ref_name):
        if len(r) < pj.MIN_STAT_SIGNALS or len(ref) < pj.MIN_STAT_SIGNALS:
            return
        rows.append({"kind": kind, "band": name, "ref": ref_name, "n": len(r), "cap_median": float(np.nanmedian(caps[name])) if name in caps else np.nan,
                     **pj.ret_stats(r, ref, pj.boot_means(ref, rng), rng), **shares(r), **{f"ref_{k}": v for k, v in shares(ref).items()}})

    ref = by["U3000"]
    for b in NARROW:
        if b not in by:
            continue
        add("band", b, by[b], ref, "U3000")
        ps_ = sorted(k for k in by if k.startswith(f"{b}#"))
        for k in ps_:
            add("pseudo", k, by[k], ref, "U3000")
        if ps_:
            pool = np.concatenate([by[k] for k in ps_])
            add("band_vs_pseudo", b, by[b], pool, f"{b} 가짜 {len(ps_)}개")
            for k in ps_:
                others = np.concatenate([by[o] for o in ps_ if o != k])
                add("pseudo_vs_pseudo", k, by[k], others, f"{b} 나머지 가짜")
    return pd.DataFrame(rows)


def repro(g: pd.DataFrame, stage_p: Path) -> pd.DataFrame:
    """U3000 cells against Stage P's goal table (same gates, 20 bp)."""
    sp = pd.read_csv(stage_p / "goal.csv.gz")
    sp = sp[(sp["mode"] == "grid") & (sp["cost"] == pj.COST_BPS) & sp["filter"].isin(FILTERS)].set_index(["setting", "gen", "exit", "filter"])
    mine = g[(g["band"] == "U3000") & (g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)].set_index(["setting", "gen", "exit", "filter"])
    cols = ["trades"] + [f"{s}_{p}" for p in gs.es.PERIODS for s in ("cagr", "mdd")] + gs.GOAL_COLS \
        + ["mdd_2000-2002", "mdd_2007-2009", "time_in_market", "invested", "turnover"]
    out = []
    for key in mine.index.intersection(sp.index):
        for c in cols:
            if c in sp.columns and c in mine.columns:
                out.append({"setting": key[0], "gen": key[1], "exit": key[2], "filter": key[3], "value": c,
                            "mine": float(mine.loc[key, c]), "stage_p": float(sp.loc[key, c])})
    r = pd.DataFrame(out)
    if len(r):
        r["diff"] = (r["mine"] - r["stage_p"]).abs()
    return r


def unbounded_check(acc: pd.DataFrame, name: str) -> pd.DataFrame:
    """Dry: a band above every rank against U3000, curve by curve."""
    a = acc[acc["mode"] == "grid"]
    idx = ["setting", "gen", "exit", "filter", "src", "order"]
    u = a[a["band"] == "U3000"].set_index(idx)
    x = a[a["band"] == name].set_index(idx)
    common = x.index.intersection(u.index)
    cols = ["trades", "cagr_1993-2023", "mdd_1993-2023", "w", "dd_cagr_1993-2023", "dd_cagr_2008-2023", "ws_ma200", "cap_fill", "beta"]
    d = (x.loc[common, cols] - u.loc[common, cols]).abs().to_numpy()
    return pd.DataFrame([{"band": name, "curves": len(common), "max_diff": float(np.nanmax(d)) if len(common) else np.nan}])


# --------------------------------------------------------------------------- figures

BAND_COLORS = {"U3000": "#777777", "U1000": None, "U500": None}


def _bc(b: str) -> str:
    return {"U3000": "#777777", "U1000": figures.COLORS[1], "U500": figures.COLORS[0]}.get(b, "#555")


def fig_heatmaps(out: Path, sysm: pd.DataFrame, bench: dict) -> list[Path]:
    """① Band x gate per setting, one figure per exit: the 2008-2023 value (colour) and 1993-2023, each
    with its change from U3000."""
    plt = figures._plt()
    paths = []
    for x in EXITS:
        fig, axes = plt.subplots(1, len(SETTINGS), figsize=(5.0 * len(SETTINGS), 3.9), squeeze=False)
        for j, st in enumerate(SETTINGS):
            ax = axes[0, j]
            g = sysm[(sysm["setting"] == ns.setting_name(st)) & (sysm["exit"] == x)].set_index(["band", "filter"])
            mat = np.full((len(BANDS), len(FILTERS)), np.nan)
            full = np.full_like(mat, np.nan)
            for a_, b in enumerate(BANDS):
                for b_, f in enumerate(FILTERS):
                    if (b, f) in g.index:
                        mat[a_, b_] = g.loc[(b, f), "dd_cagr_2008-2023"] * 100
                        full[a_, b_] = g.loc[(b, f), "dd_cagr_1993-2023"] * 100
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=3.0, vmax=8.0)
            for a_ in range(mat.shape[0]):
                for b_ in range(mat.shape[1]):
                    if np.isfinite(mat[a_, b_]):
                        d1, d2 = mat[a_, b_] - mat[0, b_], full[a_, b_] - full[0, b_]
                        txt = f"{mat[a_, b_]:.1f}" + ("" if a_ == 0 else f" ({d1:+.1f})") + f"\n전체 {full[a_, b_]:.1f}" + ("" if a_ == 0 else f" ({d2:+.1f})")
                        ax.text(b_, a_, txt, ha="center", va="center", fontsize=7, color="white" if mat[a_, b_] < 5.6 else "black")
            ax.set_xticks(range(len(FILTERS)), ["없음" if f == "none" else f for f in FILTERS], fontsize=8)
            ax.set_yticks(range(len(BANDS)), list(BANDS), fontsize=8)
            ax.set_title(f"{ns.setting_name(st)} × {x}", fontsize=9.5)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"유니버스 범위 × 국면 필터: 2008–2023 낙폭 15% 맞춤 수익(색, %) · 아래 줄 1993–2023 (괄호 = U3000 대비 %p, 매수 15개 중앙값) — "
                     f"SPY 200일선+국채 2008–23 {bench['ma200']['dd_cagr_2008-2023'] * 100:.1f}% · 전체 {bench['ma200']['dd_cagr_1993-2023'] * 100:.1f}%, "
                     f"SPY+국채 {bench['mix']['dd_cagr_2008-2023'] * 100:.1f}% · {bench['mix']['dd_cagr_1993-2023'] * 100:.1f}%", fontsize=10)
        paths.append(figures._save(fig, out / "figures" / f"band_heatmap_{x}.png"))
    return paths


def fig_ladder(out: Path, sysm: pd.DataFrame, bench: dict) -> Path:
    """② U3000 -> U1000 -> U500: medians over the 8 systems per gate of the plain MDD, w*, the two
    halves' matched returns, the fills' cap rank and the SPY beta."""
    plt = figures._plt()
    panels = [("mdd_1993-2023", "보통 MDD 1993–2023 (%)", 100), ("w", "쓸 수 있는 자본 w*", 1),
              ("dd_cagr_1993-2007", "맞춘 수익 1993–2007 (%)", 100), ("dd_cagr_2008-2023", "맞춘 수익 2008–2023 (%)", 100),
              ("cap_fill", "산 종목의 시총 순위 중앙값", 1), ("beta", "SPY 베타", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (col, lab, sc) in zip(axes.ravel(), panels):
        for k, f in enumerate(FILTERS):
            ys = [float(sysm.loc[(sysm["band"] == b) & (sysm["filter"] == f), col].median()) * sc for b in BANDS]
            ax.plot(range(len(BANDS)), ys, marker="o", color=figures.COLORS[k % 4] if k < 4 else "#555", label="필터 없음" if f == "none" else f"{f} {rs.REGIMES[f[:2]]}")
        if col.startswith("dd_cagr") or col in ("w", "mdd_1993-2023"):
            key = "mdd_1993-2023" if col == "mdd_1993-2023" else col
            ax.axhline(bench["ma200"][key] * sc, color="#8e44ad", ls="--", lw=1, label="SPY 200일선 (+국채)")
        ax.set_xticks(range(len(BANDS)), list(BANDS))
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("크기 사다리: 사는 범위를 시총 상위 3,000 → 1,000 → 500으로 좁힐 때 (시스템 8개 = 설정 4 × 매도 2의 중앙값)", fontsize=11)
    return figures._save(fig, out / "figures" / "size_ladder.png")


def fig_pseudo(out: Path, ps: pd.DataFrame, sysm: pd.DataFrame) -> Path:
    """③ Pseudo restrictions: the 10 day-matched random keeps' 2008-2023 values vs the band (N20 / W2 /
    P1 x X1), per band x gate."""
    plt = figures._plt()
    st = ns.setting_name(PSEUDO_REF)
    fig, axes = plt.subplots(len(NARROW), len(FILTERS), figsize=(3.4 * len(FILTERS), 3.1 * len(NARROW)), squeeze=False)
    for i, b in enumerate(NARROW):
        for j, f in enumerate(FILTERS):
            ax = axes[i, j]
            pv = ps[(ps["setting"] == st) & (ps["band"] == b) & (ps["filter"] == f)]["dd_cagr_2008-2023"].to_numpy() * 100
            s = sysm[(sysm["setting"] == st) & (sysm["exit"] == PSEUDO_EXIT) & (sysm["filter"] == f)].set_index("band")
            if len(pv):
                ax.hist(pv, bins=8, color="#bbbbbb", label="가짜 좁히기 10개")
                ax.axvline(pv.max(), color="#555", ls="--", lw=1, label="가짜 최댓값")
            if b in s.index:
                ax.axvline(s.loc[b, "dd_cagr_2008-2023"] * 100, color=_bc(b), lw=2, label=b)
            if "U3000" in s.index:
                ax.axvline(s.loc["U3000", "dd_cagr_2008-2023"] * 100, color="#c0392b", ls=":", lw=1.2, label="U3000")
            ax.set_title(f"{b} · {'필터 없음' if f == 'none' else f}", fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"같은 날 같은 수를 무작위로 남긴 가짜 좁히기 10개 vs 실제 범위: 2008–2023 낙폭 15% 맞춤 수익 ({st} × {PSEUDO_EXIT}, 매수 15개 중앙값, %)", fontsize=10.5)
    return figures._save(fig, out / "figures" / "pseudo.png")


def fig_trades(out: Path, trades: pd.DataFrame, tstats: pd.DataFrame) -> Path | None:
    """④ Trade returns: U3000 vs U500 vs U500's pooled pseudos (N20 / W2 / P1 x X1 x R5 A1)."""
    t = trades[trades["filter"] == TRADE_FILTER]
    if not len(t):
        return None
    plt = figures._plt()
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    parts = [("U3000", "#777", "U3000"), ("U1000", figures.COLORS[1], "U1000"), ("U500", figures.COLORS[0], "U500")]
    for b, col, lab in parts:
        r = t.loc[t["band"] == b, "ret"].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        if len(r):
            pj._bar(ax, pj._density(np.histogram(r, pj.RET_EDGES)[0]), color=col, lw=1.5, label=f"{lab} (거래 {len(r):,}개, 평균 {r.mean() * 100:+.1f}%)")
    pr = t.loc[t["band"].str.startswith("U500#"), "ret"].to_numpy(dtype=float)
    pr = pr[np.isfinite(pr)]
    if len(pr):
        pj._bar(ax, pj._density(np.histogram(pr, pj.RET_EDGES)[0]), color="#bbbbbb", lw=1.2, ls="--",
                label=f"U500의 가짜 좁히기 10개 합 (거래 {len(pr):,}개, 평균 {pr.mean() * 100:+.1f}%)")
    lines = []
    for kind, b, lab in (("band", "U500", "U500 vs U3000"), ("band_vs_pseudo", "U500", "U500 vs 가짜"), ("band", "U1000", "U1000 vs U3000")):
        s = tstats[(tstats["kind"] == kind) & (tstats["band"] == b)]
        if len(s) and pd.notna(s.iloc[0].get("st_diff")):
            s = s.iloc[0]
            lines.append(f"{lab}: 평균 차 {s['st_diff'] * 100:+.1f}%p (95% CI {s['st_ci_lo'] * 100:+.1f} – {s['st_ci_hi'] * 100:+.1f}), p = {s['st_mw_p']:.3f}, "
                         f"−20% 이하 {s['loss20'] * 100:.1f}% (기준 {s['ref_loss20'] * 100:.1f}%), +50% 이상 {s['big50'] * 100:.1f}% (기준 {s['ref_big50'] * 100:.1f}%)")
    pv = tstats[tstats["kind"] == "pseudo_vs_pseudo"]
    if len(pv):
        lines.append(f"가짜끼리 p < 0.05 비율 {np.mean(pv['st_mw_p'] < 0.05):.0%} ({len(pv)}개)")
    ax.text(0.99, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=7.3,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("거래 하나의 수익률 (%, 편도 20bp, 양 끝 칸은 −60% 미만 / +200% 초과)")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_title(f"계좌의 거래 수익률: 범위별 ({ns.setting_name(TRADE_SETTING)} × {TRADE_EXIT} × {TRADE_FILTER}, 매수 15개 합산)", fontsize=10.5)
    return figures._save(fig, out / "figures" / "trades_hist.png")


def fig_windows(out: Path, win: pd.DataFrame, best: tuple | None, bench: dict, ewb: dict, wins: list) -> Path | None:
    """⑤ 3-year windows: the best narrowed system, its U3000 counterpart, SPY200 + T-bill, the SPY mix
    and the band's equal-weight universe under the same gate."""
    if best is None:
        return None
    plt = figures._plt()
    st, x, f, b = best
    xs = np.arange(len(wins))

    def sys_w(band):
        w = win[(win["setting"] == st) & (win["exit"] == x) & (win["filter"] == f) & (win["band"] == band) & (win["gen"] != "E0")
                & (win["cost"] == pj.COST_BPS)]
        return w.groupby("w_idx")["win_cagr"].median().reindex(range(len(wins))).to_numpy() * 100

    sysw = sys_w(b)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, sysw, marker="o", color=_bc(b), label=f"시스템 {st} × {x} × {f} × {b} (매수 15개 중앙값)")
    ax.plot(xs, sys_w("U3000"), marker=".", color="#777", lw=1, label="같은 시스템의 U3000")
    for k, col, lab in (("mix", "#c0392b", "SPY + 국채"), ("ma200", "#8e44ad", "SPY 200일선 + 국채")):
        bw = bench[k]["win_cagr"] * 100
        ax.plot(xs, bw, marker="s", ms=4, color=col, label=f"{lab} (시스템이 앞선 창 {np.mean(sysw >= bw):.0%})")
    ref = ewb.get((b, ew_ref(f)))
    if ref is not None:
        ew = ref["win_cagr"] * 100
        ax.plot(xs, ew, marker="^", ms=4, color="#555", label=f"{ref['label']} (시스템이 앞선 창 {np.mean(sysw >= ew):.0%})")
    ax.set_xticks(xs, [f"{a}–{str(c)[2:]}" for a, c in wins], rotation=90, fontsize=7.5)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("3년 창 CAGR, 낙폭 15% 맞춤 (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("3년 롤링 창: 2008–2023이 가장 높은 좁힌 시스템 vs 기준선 (각자 전체 기간 w*)", fontsize=10.5)
    return figures._save(fig, out / "figures" / "windows_band.png")


def fig_curves(out: Path, curves: dict, wdates: pd.DatetimeIndex) -> Path:
    """⑥ w* curves: the best narrowed system's median entry, its U3000 counterpart and the benchmarks."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for k, (lab, s) in enumerate(curves.items()):
        col = [figures.COLORS[0], "#777", "#8e44ad", "#555", "#c0392b"][k % 5]
        for ax, (a, b) in zip(axes, (("1993-01-01", "2023-12-31"), ("2007-01-01", "2010-12-31"))):
            part = s.loc[a:b]
            ax.plot(part.index, part / part.iloc[0], color=col, lw=1.3, label=lab)
    for ax, t in zip(axes, ("1993–2023 (로그)", "2007–2010 확대")):
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("낙폭 15%에 맞춘 곡선 (각자 w*로 국채와 섞음)", fontsize=11)
    return figures._save(fig, out / "figures" / "curves.png")


def fig_candidates(out: Path, g: pd.DataFrame, cand: pd.DataFrame, bench: dict) -> list[Path]:
    """⑦ (when there are candidates) Each candidate entry x gate in every band, setting and exit:
    matched returns of the two halves against SPY200 + T-bill, with the same cell's E0."""
    plt = figures._plt()
    paths = []
    sel = (g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)
    for gen, f in sorted({(r["gen"], r["filter"]) for _, r in cand[cand["candidate"]].iterrows()}):
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)
        groups = [(ns.setting_name(st), x) for st in SETTINGS for x in EXITS]
        xs = np.arange(len(groups))
        for ax, p in zip(axes, gs.es.HALVES):
            for k, b in enumerate(BANDS):
                c = g[sel & (g["gen"] == gen) & (g["filter"] == f) & (g["band"] == b)].set_index(["setting", "exit"])
                e0 = g[sel & (g["gen"] == "E0") & (g["filter"] == f) & (g["band"] == b)].set_index(["setting", "exit"])
                v = [c.loc[key, f"dd_cagr_{p}"] * 100 if key in c.index else np.nan for key in groups]
                v0 = [e0.loc[key, f"dd_cagr_{p}"] * 100 if key in e0.index else np.nan for key in groups]
                ax.bar(xs + (k - 1) * 0.27, v, width=0.27, color=_bc(b), label=f"{gen} × {f} · {b}")
                ax.scatter(xs + (k - 1) * 0.27, v0, marker="_", s=120, color="black", zorder=3, label="같은 칸의 무작위 매수 E0" if k == 0 else None)
            ax.axhline(bench["ma200"][f"dd_cagr_{p}"] * 100, color="#8e44ad", ls="--", lw=1.2, label="SPY 200일선 + 국채")
            ax.axhline(bench["mix"][f"dd_cagr_{p}"] * 100, color="#c0392b", ls=":", lw=1.2, label="SPY + 국채")
            ax.set_xticks(xs, [f"{s}\n{x}" for s, x in groups], fontsize=7.5)
            ax.set_title(f"{p} 낙폭 15% 맞춤 수익 (%)", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
        axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
        names = ", ".join(f"{r['setting']} · {r['band']}" for _, r in cand[cand["candidate"] & (cand["gen"] == gen) & (cand["filter"] == f)].iterrows())
        fig.suptitle(f"후보 계열 {gen} × {f}: 범위 · 설정 · 매도별 (후보 = {names})", fontsize=10.5)
        paths.append(figures._save(fig, out / "figures" / f"candidate_{gen}_{f}.png"))
    return paths


def best_curves(a, sysm: pd.DataFrame, g: pd.DataFrame, best: tuple, bench: dict, ewb: dict, px, wdates, w0, cash, gens) -> dict:
    """The best narrowed system's median entry (its first order) in the band and in U3000, and the
    benchmarks, each blended at its own w*."""
    from . import target as tg
    st_name, x, f, b = best
    c = g[(g["setting"] == st_name) & (g["exit"] == x) & (g["filter"] == f) & (g["band"] == b) & (g["gen"] != "E0")
          & (g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)].sort_values("dd_cagr_2008-2023")
    gen = c["gen"].iloc[len(c) // 2]
    _init(a.panel, a.eligible, a.dirs, a.cfg)
    st = ns.parse_setting(st_name)
    srcs = gens[gen]
    src = _sources(gen, x, srcs[:1])
    s = srcs[0]
    w = ns.sizing(src[s]["rows"].stop, st[0], "W2")[0]
    key = next(k for s_, o, k in _keys(gen, x, st, srcs[:1], src, 1))
    state = None if f == "none" else _W["states"][f[:2]]
    masks = band_masks(src, s, [(b, BANDS[b]), ("U3000", BANDS["U3000"])], "grid")

    def blended(eq_, ww):
        rr = eq_[1:] / eq_[:-1] - 1.0
        return pd.Series(np.concatenate([[1.0], np.cumprod(1 + ww * rr + (1 - ww) * cash[1:])]), index=wdates)

    out = {}
    for nm in (b, "U3000"):
        eq, _ = rs._replay(src[s]["rows"], src[s]["si"], masks[nm], w, key, st[0], pj.COST_BPS / 10_000, state, False)
        rr = eq[1:] / eq[:-1] - 1.0
        ws = tg.fit_weight(rr, cash[1:])
        out[f"{st_name} × {gen} × {x} × {f} × {nm} (w* {ws:.2f})"] = blended(eq, ws)
    cv = gs.bench_curves(px, wdates, w0, cash)
    out[f"SPY 200일선 + 국채 (w* {bench['ma200']['w']:.2f})"] = blended(cv["ma200"], bench["ma200"]["w"])
    ref = ewb[(b, ew_ref(f))]
    out[f"{ref['label']} (w* {ref['w']:.2f})"] = blended(ref["eq"], ref["w"])
    out[f"SPY + 국채 (w* {bench['mix']['w']:.2f})"] = blended(cv["mix"], bench["mix"]["w"])
    return out


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "r0x1", "tables", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--stage-p", default=None, help="Stage P 결과표 폴더 (재현 점검)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="E2·V3·E0(씨앗 2), 필터 none·R5A1, 순위 9,999 범위 점검")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.dirs = {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events),
              "r0x1": Path(a.r0x1), "tables": Path(a.tables)}
    gens = ms.generators(["E2", "V3"], pj.E0_SEEDS[:2]) if a.dry else ms.generators()
    a.cfg = {"bins": pj.BINS_J3, "seeds": pj.E0_SEEDS, "base": str(out / "base.parquet"), "regimes": str(out / "regimes.parquet")}
    ts.log(f"1단계: 신호 기준표 (매수 {len(gens)} × 매도 {len(EXITS)}) · 국면")
    saved = list(ns.EXITS)
    ns.EXITS[:] = list(EXITS)
    base = ns.build_base(a, gens)
    ns.EXITS[:] = saved
    if not (out / "regimes.parquet").exists():
        panel = load_panel_with_cash(Path(a.panel))
        rs.regime_states(panel.close["SPY"]).to_parquet(out / "regimes.parquet")
        del panel
    states = pd.read_parquet(out / "regimes.parquet")
    k1 = ~ms.excluded(base, ns.RULE)
    ts.log(f"  신호 {len(base):,}개, M1 통과 {k1.mean():.1%} · 그중 시총 1,000위 이내 {(base.loc[k1, 'cap_rank'] <= 1000).mean():.1%}, "
           f"500위 이내 {(base.loc[k1, 'cap_rank'] <= 500).mean():.1%} · 순위 없는 신호 {int(base['cap_rank'].isna().sum())}개")
    filters = ["none", "R5A1"] if a.dry else list(FILTERS)
    bands = list(BANDS.items()) + ([DRY_BAND] if a.dry else [])
    narrow = [(b, BANDS[b]) for b in NARROW]
    size = base.groupby("src").size()
    jobs = []
    for gen, srcs in gens.items():
        for x in EXITS:
            for st in SETTINGS:
                if st[2] == "P0":  # heavy: one job per gate
                    jobs += [(gen, x, st, [f], bands, pj.COST_BPS, srcs, "grid") for f in filters]
                else:
                    jobs.append((gen, x, st, filters, bands, pj.COST_BPS, srcs, "grid"))
        if gen != "E0":
            for st in PSEUDO_SETTINGS:
                fs = [f for f in PSEUDO_FILTERS[st] if f in filters]
                if st[2] == "P0":
                    jobs += [(gen, PSEUDO_EXIT, st, [f], narrow, pj.COST_BPS, srcs, "pseudo") for f in fs]
                else:
                    jobs.append((gen, PSEUDO_EXIT, st, fs, narrow, pj.COST_BPS, srcs, "pseudo"))
    weight = lambda j: int(size.reindex(j[6]).sum()) * len(j[3]) * (100 if j[2][2] == "P0" and j[7] == "grid" else 20 if j[2][2] == "P0" else 1) \
        * (len(j[4]) if j[7] == "grid" else 2 * N_PSEUDO) * (1 + j[2][0] / 50)  # noqa: E731
    jobs.sort(key=lambda j: -weight(j))
    jdir = out / "jobs"
    todo = [j for j in jobs if not (jdir / f"{_name(j)}_acc.csv").exists()]
    ts.log(f"2단계: 작업 {len(todo)}개 남음 (격자 {sum(j[7] == 'grid' for j in jobs)}, 가짜 {sum(j[7] == 'pseudo' for j in jobs)})")
    run_jobs(todo, a)
    read = lambda s: pd.concat([pd.read_csv(f) for f in sorted(jdir.glob(f"*_{s}.csv"))  # noqa: E731
                                if "bp_" not in f.name and f.stat().st_size > 2], ignore_index=True)
    acc, win, mom = read("acc"), read("win"), read("mom")
    trades = pd.concat([pd.read_parquet(f) for f in sorted(jdir.glob("*_trades.parquet")) if "bp_" not in f.name], ignore_index=True)
    ts.log("3단계: 기준선 (상위 N 동일비중, × 국면)")
    pj._init(a.panel, a.eligible, a.dirs, a.cfg)
    px, wdates, w0, cash = pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"]
    btab, bench = gs.benchmarks_table(px, wdates, w0, cash)
    sp = gs._setup(cash, wdates)
    panel = load_panel_with_cash(Path(a.panel))
    btab_b, ewb = band_benchmarks(panel, pd.read_parquet(a.eligible), wdates, w0, cash, states)
    del panel
    btab = pd.concat([btab, btab_b], ignore_index=True)
    btab.to_csv(out / "benchmarks.csv", index=False)
    g = summarize(acc)
    summary: dict = {}
    if a.stage_p:
        rp = repro(g, Path(a.stage_p))
        rp.to_csv(out / "repro.csv", index=False)
        ncell = int(rp[["setting", "gen", "exit", "filter"]].drop_duplicates().shape[0]) if len(rp) else 0
        ts.log(f"재현 점검: Stage P 대비 {ncell}칸 {len(rp)}값, 최대 차 {rp['diff'].max():.2e}, 1e-9 넘는 값 {int((rp['diff'] > 1e-9).sum())}개")
        summary["repro"] = {"cells": ncell, "values": len(rp), "max_diff": float(rp["diff"].max()) if len(rp) else np.nan}
    if a.dry:
        uc = unbounded_check(acc, DRY_BAND[0])
        uc.to_csv(out / "unbounded.csv", index=False)
        ts.log("순위 9,999 범위 vs U3000: " + ", ".join(f"{r.curves}곡선 최대 차 {r.max_diff:.2e}" for r in uc.itertuples()))
        g = g[g["band"] != DRY_BAND[0]]
        acc = acc[acc["band"] != DRY_BAND[0]]
    sysm = systems(g, bench, ewb)
    ps = pseudo_systems(g)
    help_rows, help_cnt = q1(sysm, ps)
    cells, _ = candidates(g, bench, {})
    p1 = {ns.setting_name(s) for s in P1_SETTINGS}
    need = cells[(cells["gen"] != "E0") & cells["cell_pass"] & cells["setting"].isin(p1)]
    per = need.groupby(["setting", "gen", "filter", "band"])["cell_pass"].sum()
    pass40 = {}
    if (per >= len(EXITS)).any():
        by_gs: dict = {}
        for (st, gen, f, b) in per[per >= len(EXITS)].index:
            for x in EXITS:
                by_gs.setdefault((gen, x, st), {}).setdefault(f, set()).add(b)
        jobs40 = []
        for (gen, x, st), fb in by_gs.items():
            bset = sorted({b for v in fb.values() for b in v})
            jobs40.append((gen, x, ns.parse_setting(st), sorted(fb), [(b, BANDS[b]) for b in bset], pj.CHECK_COST_BPS, gens[gen], "grid"))
        ts.log(f"4단계: 편도 40bp 점검 {len(jobs40)}개 작업")
        run_jobs([j for j in jobs40 if not (jdir / f"{_name(j)}_acc.csv").exists()], a)
        a40 = pd.concat([pd.read_csv(jdir / f"{_name(j)}_acc.csv") for j in jobs40], ignore_index=True)
        for _, r in summarize(a40).iterrows():
            pass40[(r["setting"], r["gen"], r["exit"], r["filter"], r["band"])] = gs.cell_pass(r, bench)
    cells, cand = candidates(g, bench, pass40)
    ch = chance(acc, cells, bench)
    tstats = trade_stats(trades[trades["band"] != DRY_BAND[0]]) if len(trades) else pd.DataFrame()
    dsr = pd.DataFrame()
    if len(cand) and cand["candidate"].any():
        m = mom.set_index(["setting", "gen", "exit", "filter", "band"])
        sr_var = float(np.var(m["m_sr"].dropna()))
        curves = gs.bench_curves(px, wdates, w0, cash)
        rm = curves["mix"][1:] / curves["mix"][:-1] - 1.0
        bl = bench["mix"]["w"] * rm + (1 - bench["mix"]["w"]) * cash[1:]
        rows = []
        for _, r in cells[cells["cell_pass"]].iterrows():
            ok = cand[(cand["setting"] == r["setting"]) & (cand["gen"] == r["gen"]) & (cand["filter"] == r["filter"])
                      & (cand["band"] == r["band"]) & cand["candidate"]]
            key = (r["setting"], r["gen"], r["exit"], r["filter"], r["band"])
            if len(ok) and key in m.index:
                mm = m.loc[key]
                rows.append({"setting": key[0], "gen": key[1], "exit": key[2], "filter": key[3], "band": key[4],
                             "dsr": ks.dsr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), N_TRIALS, sr_var),
                             "psr_mix": ks.psr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), float(bl.mean() / bl.std()))})
        dsr = pd.DataFrame(rows)
        dsr.to_csv(out / "dsr.csv", index=False)
    g.to_csv(out / "goal.csv", index=False)
    sysm.to_csv(out / "systems.csv", index=False)
    cells.to_csv(out / "cells.csv", index=False)
    ps.to_csv(out / "pseudo.csv", index=False)
    help_rows.to_csv(out / "helps.csv", index=False)
    help_cnt.to_csv(out / "help_counts.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    ch.to_csv(out / "chance.csv", index=False)
    tstats.to_csv(out / "trade_stats.csv", index=False)
    win.groupby(["setting", "gen", "exit", "filter", "band", "cost", "w_idx", "a", "b"])["win_cagr"].median().reset_index().to_csv(out / "windows.csv", index=False)
    nar = sysm[sysm["band"].isin(NARROW)]
    best = nar.sort_values("dd_cagr_2008-2023", ascending=False).head(1)
    best_key = tuple(best[["setting", "exit", "filter", "band"]].iloc[0]) if len(best) else None
    cols_b = ["w", "dd_cagr_1993-2023", "dd_cagr_1993-2007", "dd_cagr_2008-2023", "dd_cagr_2017-2023", "mdd_1993-2023"]
    summary.update({
        "benchmarks": {r["benchmark"]: {c: r[c] for c in cols_b} for _, r in btab.iterrows()},
        "q1_counts": help_cnt.to_dict("records"),
        "band_medians": sysm.groupby(["band", "filter"])[["dd_cagr_1993-2023", "dd_cagr_1993-2007", "dd_cagr_2008-2023", "mdd_1993-2023",
                                                          "w", "cap_fill", "beta", "positions_mean", "cash_share"]].median().reset_index().to_dict("records"),
        "systems_passing": nar.loc[nar["system_pass"], ["setting", "exit", "filter", "band", "beats_ew_band"]].to_dict("records"),
        "best_narrow": nar.sort_values("dd_cagr_2008-2023", ascending=False).head(8)[["setting", "exit", "filter", "band", "w", "dd_cagr_1993-2023",
                                                                                     "dd_cagr_1993-2007", "dd_cagr_2008-2023"]].to_dict("records"),
        "cells_passing": int(cells.loc[cells["gen"] != "E0", "cell_pass"].sum()) if len(cells) else 0,
        "candidates": cand.loc[cand["candidate"], ["setting", "gen", "filter", "band", "entry_specific"]].to_dict("records") if len(cand) else [],
        "dsr": dsr.to_dict("records"),
        "trade_tests": {k: float((tstats.loc[tstats["kind"] == k, "st_mw_p"] < 0.05).mean()) if len(tstats) and (tstats["kind"] == k).any() else np.nan
                        for k in ("band", "pseudo", "band_vs_pseudo", "pseudo_vs_pseudo")},
        "trials": {"before": rs.N_TRIALS, "new": N_NEW}})
    (out / "chance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if figures.available():
        fig_heatmaps(out, sysm, bench)
        fig_ladder(out, sysm, bench)
        if len(ps):
            fig_pseudo(out, ps, sysm)
        if len(trades):
            fig_trades(out, trades, tstats)
        fig_windows(out, win, best_key, bench, ewb, sp["wins"])
        if len(cand) and cand["candidate"].any():
            fig_candidates(out, g, cand, bench)
        if best_key is not None:
            fig_curves(out, best_curves(a, sysm, g, best_key, bench, ewb, px, wdates, w0, cash, gens), wdates)
    ts.log("완료 · '대형주가 도움'(시스템 8개 중): " + ", ".join(f"{r.band} {r.filter} {r.helps}/{r.q95}" for r in help_cnt.itertuples())
           + f" · 목표 통과 시스템 {len(summary['systems_passing'])} · 통과 칸 {summary['cells_passing']} · 후보 {len(summary['candidates'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
