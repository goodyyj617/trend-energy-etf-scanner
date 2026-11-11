"""Stage P: market-regime gates on the stock account (DECISIONS 95).

    python -m lab.regime_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                               --r0x1 DIR --tables DIR --out DIR [--stage-o DIR] [--workers N] [--feature-workers N] [--dry]

The M1 accounts of Stage O (lab/goal_study.py; the stored Stage L tables, the same random orders)
for N10 / W2 / P0 and N10 / N20 / N50 x W2 x P1, exits X1 and H1, under 10 market-regime gates:
regimes R1 SPY > SMA200, R2 T10Y3M > 0, R3 BAA10Y <= its 200-observation SMA, R4 VIX < 25 (the
registered defaults of lab/blocks.py's market_trend / curve_ok / credit_calm / vix_calm; macro at
t-1) and R5 all four, each with A1 (no new entry on a signal day whose regime is off) or A2 (A1 plus
every holding sold at the open after the regime turns off; pick_study.replay_risk's force_exit).
Every curve is measured as Stage O (w* at the 15% budget, drawdown-matched CAGR, half refits,
3-year windows against the SPY mix, SPY200 and the equal-weight universe) plus the equal-weight
universe under the same regime (A2). Placebos: 20 circular shifts (252-5,000 sessions) of each
regime's on/off series, at N10 / W2 / P0 (20 orders) and N20 / W2 / P1, exit X1, the 15 entries.
--stage-o checks the no-gate cells against Stage O's goal table; --dry adds an always-on regime
that must equal no gate. Output (aggregates only): base.parquet, regimes.parquet, jobs/*,
regimes.csv, goal.csv, systems.csv, cells.csv, placebo.csv, benchmarks.csv, candidates.csv,
chance.csv, trades_stats.csv, crises.csv, repro.csv, chance.json, figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import account_study as ns, entry_study as es, exclude_study as ms, figures, goal_study as gs, pair_study as pj
from . import pick_study as ph, select_study as ks, tail_study as ts
from .blocks import ALL_BLOCKS
from .crsp_study import load_panel_with_cash

# ---- registered (DECISIONS 95): fixed before any result
SETTINGS = [(10, "W2", "P0"), (10, "W2", "P1"), (20, "W2", "P1"), (50, "W2", "P1")]
P1_SETTINGS = SETTINGS[1:]
EXITS = ["X1", "H1"]
REGIME_BLOCKS = ("market_trend", "curve_ok", "credit_calm", "vix_calm")  # their registered default values
REGIMES = {"R1": "SPY 200일선 위", "R2": "장단기 금리차 플러스", "R3": "신용 스프레드 안정", "R4": "VIX 25 미만", "R5": "넷 모두 좋음"}
ACTIONS = {"A1": "새 매수 중단", "A2": "새 매수 중단 + 전량 청산"}
FILTERS = ["none"] + [f"{r}{a}" for r in REGIMES for a in ACTIONS]
N_PLACEBO, SHIFT = 20, (252, 5000)
PLACEBO_SETTINGS = [(10, "W2", "P0"), (20, "W2", "P1")]
PLACEBO_REF = (20, "W2", "P1")  # the placebo distribution the other settings and exit H1 borrow
PLACEBO_EXIT, PLACEBO_ORDERS = "X1", 20
TRADE_SETTING, TRADE_EXIT = (20, "W2", "P1"), "X1"
CRISES = {"2000-2002": ("2000-01-01", "2002-12-31"), "2007-2009": ("2007-07-01", "2009-06-30")}
N_TRIALS = gs.N_TRIALS + 1_200  # 72,974 for the candidates' DSR
N_NEW = 1_200  # 10 gates x 4 settings x 15 entries x 2 exits
SEED = 95
_W: dict = {}


def regime_states(close_spy: pd.Series) -> pd.DataFrame:
    """R1-R5 on the panel's dates (True = on), from the blocks' compute at their default values."""
    mini = types.SimpleNamespace(close=close_spy.to_frame("SPY"))
    raw = {}
    for key in REGIME_BLOCKS:
        b = ALL_BLOCKS[key]
        raw[key] = b.compute(mini, b.param.default)["SPY"].astype(bool)
    df = pd.DataFrame(raw)
    out = pd.DataFrame({"R1": df["market_trend"], "R2": df["curve_ok"], "R3": df["credit_calm"], "R4": df["vix_calm"]})
    out["R5"] = out.all(axis=1)
    return out


def force_days(state: np.ndarray, w0: int, n: int) -> np.ndarray:
    """Window days whose open sells everything: the session after each on -> off close."""
    t = np.flatnonzero(state[:-1] & ~state[1:]) + 1  # first off close (panel index)
    d = t + 1 - w0
    return d[(d >= 0) & (d < n)]


def placebo_state(state: np.ndarray, w0: int, n: int, shift: int) -> np.ndarray:
    """The on/off series circularly shifted by `shift` sessions inside the window (and its day before)."""
    out = state.copy()
    seg = slice(max(w0 - 1, 0), w0 + n)
    out[seg] = np.roll(state[seg], shift)
    return out


def shifts(regime: str) -> np.ndarray:
    return np.random.default_rng([SEED, list(REGIMES).index(regime)]).integers(SHIFT[0], SHIFT[1] + 1, N_PLACEBO)


def regime_stats(states: pd.DataFrame, wdates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per regime on the window: off share, off spells and their mean length, on -> off switches."""
    rows = []
    for r in states.columns:
        s = states[r].reindex(wdates).to_numpy(dtype=bool)
        off = ~s
        starts = np.flatnonzero(off & ~np.r_[False, off[:-1]])
        ends = np.flatnonzero(off & ~np.r_[off[1:], False])
        rows.append({"regime": r, "label": REGIMES.get(r, r), "off_share": float(off.mean()), "off_spells": len(starts),
                     "off_mean_days": float((ends - starts + 1).mean()) if len(starts) else 0.0,
                     "switches_off": int((s[:-1] & ~s[1:]).sum())})
    return pd.DataFrame(rows)


def ew_regime(ew: np.ndarray, cash: np.ndarray, state_w: np.ndarray, cost) -> np.ndarray:
    """Equal-weight universe held while the regime was on at the previous close, T-bill otherwise;
    one-way cost on each switch day. cost: a number, or per window day (array of len(ew); the switch
    whose return lands on day t pays cost[t], DECISIONS 100). A number gives the result as before."""
    r = ew[1:] / ew[:-1] - 1.0
    on = state_w[:-1]
    d = np.where(on, r, cash[1:])
    sw = np.r_[False, on[1:] != on[:-1]]
    d = d - np.where(sw, cost if np.ndim(cost) == 0 else np.asarray(cost, dtype=float)[1:], 0.0)
    return np.concatenate([[1.0], np.cumprod(1.0 + d)])


def crisis_mdd(eq: np.ndarray, wdates: pd.DatetimeIndex) -> dict:
    out = {}
    for k, (a, b) in CRISES.items():
        sel = (wdates >= pd.Timestamp(a)) & (wdates <= pd.Timestamp(b))
        e = eq[sel]
        out[f"mdd_{k}"] = float((e / np.maximum.accumulate(e) - 1.0).min()) if len(e) else np.nan
    return out


# --------------------------------------------------------------------------- workers

def _init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    gs._init(panel_dir, eligible, dirs, cfg)
    _W.update(gs._W)
    st = pd.read_parquet(cfg["regimes"]).reindex(_W["px"].dates)
    _W["states"] = {r: st[r].fillna(False).to_numpy(dtype=bool) for r in st.columns}


def _replay(r, si_panel, mask, w, key, n, cost, state=None, a2=False, log=None):
    """One account; with a regime state (panel index): signals on off days are dropped (A1), and with
    a2 the holdings are sold at the open after each on -> off close."""
    m = mask & np.isfinite(w)
    if state is not None:
        m &= state[si_panel]
    idx = np.flatnonzero(m)
    force = force_days(state, _W["w0"], _W["n"]) if (state is not None and a2) else None
    eq, tr = ph.replay_risk(r.ei[idx] - 1, r.ei[idx], r.sj[idx], r.sale[idx], r.closed[idx], w[idx], key[idx],
                            _W["O"], _W["LC"], _W["cash"], cost, slots=n, log=log, force_exit=force)
    return eq, tr


def _filter_state(f: str) -> tuple[np.ndarray | None, bool]:
    if f == "none":
        return None, False
    return _W["states"][f[:2]], f.endswith("A2")


def run_job(job):
    """mode 'grid': one buy rule x exit x setting under the given gates; 'placebo': X1 under the 20
    shifted series of one regime (both actions); 'rule40': the grid at the job's cost."""
    gen, x, st, filters, cost_bps, srcs, mode = job
    t0 = time.time()
    cost = cost_bps / 10_000
    n_slots = st[0]
    rows, sis, keeps, pseudos = {}, {}, {}, {}
    for s in srcs:
        d = ms.prepared(s, x)
        b = ms.source_base(s)
        rows[s] = gs.GRows(d, b, x)
        sis[s] = d["si"].to_numpy()
        keep_b = ~ms.excluded(b, ns.RULE)
        keeps[s] = keep_b[rows[s].brow]
        pseudos[s] = [ks.pseudo_mask(b["si"].to_numpy(), keep_b, np.random.default_rng([ms.SEED, zlib.crc32(f"{ns.RULE}|{s}".encode()), k]))[rows[s].brow]
                      for k in range(ms.N_PSEUDO)]
    wts = {s: ns.sizing(rows[s].stop, n_slots, "W2")[0] for s in srcs}
    def keys():  # drawn again for every gate (not kept: 100 orders of a large rule would not fit many workers)
        if st[2] == "P0":  # Stage M / N / O's random orders (the first PLACEBO_ORDERS for placebos)
            limit = ms.N_ORDERS // len(srcs) if mode != "placebo" else PLACEBO_ORDERS
            for kind, k, s, o, key in ns.stage_m_draws(gen, x, srcs, keeps, pseudos, {s: len(rows[s]) for s in srcs}):
                if kind == "rule" and o < limit:
                    yield s, o, key
        else:
            for s in srcs:
                r = rows[s]
                tie = np.random.default_rng([ns.SEED, zlib.crc32(f"tie|{s}|{x}".encode())]).random(len(r))
                yield s, 0, (r.k2 + tie * 1e-9) if st[2] == "P1" else (-r.cap + tie * 1e-6)
    if mode == "placebo":
        reg = filters[0]
        base_state = _W["states"][reg]
        gates = [(f"{reg}{a}#{k}", placebo_state(base_state, _W["w0"], _W["n"], int(sh)), a == "A2")
                 for k, sh in enumerate(shifts(reg)) for a in ACTIONS]
    else:
        gates = []
        for f in filters:
            if f.startswith("R0"):  # the always-on check (dry)
                gates.append((f, np.ones(len(_W["px"].dates), dtype=bool), f.endswith("A2")))
            else:
                state, a2 = _filter_state(f)
                gates.append((f, state, a2))
    accs, wins, moms, trades = [], [], [], []
    for f, state, a2 in gates:
        curves, extra = [], []
        for s, o, key in keys():
            log = {} if o == 0 and mode == "grid" else None
            eq, tr = _replay(rows[s], sis[s], keeps[s], wts[s], key, n_slots, cost, state, a2, log)
            curves.append((s, o, len(tr), eq))
            ex = crisis_mdd(eq, _W["wdates"])
            if log is not None:
                ex.update({"time_in_market": float((log["positions"] > 0).mean()), "invested": float(np.mean(log["invested"])),
                           "turnover": float(tr["weight"].sum() / ((_W["wdates"][-1] - _W["wdates"][0]).days / 365.25)) if len(tr) else 0.0})
            extra.append(ex)
            if st == TRADE_SETTING and x == TRADE_EXIT and gen != "E0" and cost_bps == pj.COST_BPS:
                trades.append(pd.DataFrame({"filter": f, "src": s, "ret": tr["ret"].to_numpy(dtype="float32")}))
        r_, w_, mom = gs._measure(curves, cost_bps)
        head = {"setting": ns.setting_name(st), "filter": f}
        accs += [{**head, **a, **e} for a, e in zip(r_, extra)]
        if mode == "grid":
            wins += [{**head, **w} for w in w_]
            if mom is not None:
                moms.append({**head, **mom})
    ms._W["prep"].clear()
    pj._W["tables"].clear()
    sig = None
    if mode == "grid" and st == TRADE_SETTING and x == TRADE_EXIT and gen != "E0" and "none" in filters and cost_bps == pj.COST_BPS:
        parts = []
        for s in srcs:  # A1's bought vs blocked signals: the M1 rows' own returns and each regime's state on their signal day
            m = keeps[s] & np.isfinite(wts[s])
            parts.append(pd.DataFrame({"src": s, "ret": rows[s].ret[m].astype("float32"),
                                       **{r: _W["states"][r][sis[s][m]] for r in REGIMES}}))
        sig = pd.concat(parts, ignore_index=True)
    tail = {"gen": gen, "exit": x, "cost": cost_bps, "mode": mode}
    frame = lambda rs: pd.DataFrame([{**r, **tail} for r in rs])  # noqa: E731
    tr_all = pd.concat(trades, ignore_index=True).assign(gen=gen) if trades else pd.DataFrame()
    return job, frame(accs), frame(wins), frame(moms), tr_all, sig, time.time() - t0


def job_name(gen: str, x: str, st: tuple, filters: list, cost: float, mode: str) -> str:
    tag = "_".join(filters) if len(filters) <= 3 else "all"
    return f"{gen}__{x}__{ns.setting_name(st)}__{mode}__{tag}" + ("" if cost == pj.COST_BPS else f"__{cost:g}bp")


def run_jobs(jobs: list, a) -> None:
    d = Path(a.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs, a.cfg)) as ex:
        for i, (job, acc, win, mom, tr, sig, sec) in enumerate(ex.map(run_job, jobs, chunksize=1)):
            nm = job_name(job[0], job[1], job[2], job[3], job[4], job[6])
            for suffix, f in (("win", win), ("mom", mom)):
                f.to_csv(d / f"{nm}_{suffix}.csv", index=False)
            if len(tr):
                tr.to_parquet(d / f"{nm}_trades.parquet")
            if sig is not None:
                sig.to_parquet(d / f"{nm}_signals.parquet")
            acc.to_csv(d / f"{nm}_acc.csv", index=False)  # last: its presence marks the job done
            if (i + 1) % 5 == 0 or i + 1 == len(jobs):
                ts.log(f"  [{i + 1}/{len(jobs)}] {nm} {sec:.0f}s · 남은 예상 {(len(jobs) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


# --------------------------------------------------------------------------- verdicts

def summarize(acc: pd.DataFrame) -> pd.DataFrame:
    keys = ["setting", "gen", "exit", "filter", "cost", "mode"]
    g = acc.groupby(keys, sort=False)
    med = g.median(numeric_only=True).drop(columns=["order"], errors="ignore")
    med["orders"] = g.size()
    return med.reset_index()


def systems(g: pd.DataFrame, bench: dict, ewr: dict) -> pd.DataFrame:
    """Per setting x exit x gate: median over the 15 entries; goal-passing as Stage O; also beating
    the equal-weight universe under the same regime (A2) in the three judged periods."""
    cells = g[(g["gen"] != "E0") & (g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)]
    e0 = g[(g["gen"] == "E0") & (g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)].set_index(["setting", "exit", "filter"])
    cols = ["w", "w_1993-2007", "w_2008-2023", "dd_mdd", "cagr_1993-2023", "mdd_1993-2023", "cagr_2017-2023", "trades",
            "mdd_2000-2002", "mdd_2007-2009", "time_in_market", "turnover"] + [f"dd_cagr_{p}" for p in gs.GOAL_PERIODS] \
        + [f"{m}_{b}" for b in gs.BENCHMARKS for m in ("ws", "worst")]
    rows = []
    for (st, x, f), c in cells.groupby(["setting", "exit", "filter"], sort=False):
        row = {"setting": st, "exit": x, "filter": f, "entries": len(c), **{k: float(c[k].median()) for k in cols if k in c}}
        row["system_pass"] = gs.beats(row, bench)
        reg = f[:2] if f != "none" else None
        row["beats_ew_regime"] = bool(reg in ewr and all(row[f"dd_cagr_{p}"] > ewr[reg][f"dd_cagr_{p}"] for p in gs.JUDGED)) if reg else False
        for k in ["w"] + [f"dd_cagr_{p}" for p in gs.GOAL_PERIODS]:
            row[f"e0_{k}"] = float(e0.loc[(st, x, f), k]) if (st, x, f) in e0.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def placebo_systems(g: pd.DataFrame) -> pd.DataFrame:
    """Per placebo setting x regime x action x shift: median over the 15 entries (exit X1)."""
    p = g[(g["mode"] == "placebo") & (g["gen"] != "E0")].copy()
    p["base"] = p["filter"].str.split("#").str[0]
    p["k"] = p["filter"].str.split("#").str[1].astype(int)
    cols = ["w", "cagr_1993-2023", "mdd_1993-2023"] + [f"dd_cagr_{q}" for q in gs.GOAL_PERIODS]
    return p.groupby(["setting", "base", "k"])[cols].median().reset_index()


def q1(sysm: pd.DataFrame, ps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """'Gate helps' per setting x exit x gate: above no gate in the full period and both halves and
    above the 95th percentile of the placebos (own setting when run, else N20 / W2 / P1, exit X1);
    counts over the 8 systems against Binomial(8, p0) with p0 from placebo vs the other 19."""
    none = sysm[sysm["filter"] == "none"].set_index(["setting", "exit"])
    rows = []
    for _, r in sysm[sysm["filter"] != "none"].iterrows():
        nb = none.loc[(r["setting"], r["exit"])]
        ref_st = r["setting"] if r["setting"] in {ns.setting_name(s) for s in PLACEBO_SETTINGS} and r["exit"] == PLACEBO_EXIT \
            else ns.setting_name(PLACEBO_REF)
        pv = ps[(ps["setting"] == ref_st) & (ps["base"] == r["filter"])]["dd_cagr_1993-2023"].to_numpy()
        q = float(np.percentile(pv, 95)) if len(pv) else np.nan
        above_none = all(r[f"dd_cagr_{p}"] > nb[f"dd_cagr_{p}"] for p in gs.JUDGED)
        rows.append({"setting": r["setting"], "exit": r["exit"], "filter": r["filter"], "placebo_from": ref_st,
                     "delta_1993-2023": r["dd_cagr_1993-2023"] - nb["dd_cagr_1993-2023"],
                     "delta_1993-2007": r["dd_cagr_1993-2007"] - nb["dd_cagr_1993-2007"],
                     "delta_2008-2023": r["dd_cagr_2008-2023"] - nb["dd_cagr_2008-2023"],
                     "placebo_q95": q, "placebo_median": float(np.median(pv)) if len(pv) else np.nan,
                     "above_none": above_none, "above_placebo": bool(r["dd_cagr_1993-2023"] > q), "helps": bool(above_none and r["dd_cagr_1993-2023"] > q)})
    h = pd.DataFrame(rows)
    # placebo vs placebo: the same test for each placebo against the other 19 (and no gate)
    pp = []
    for (st, base), g_ in ps.groupby(["setting", "base"]):
        if (st, PLACEBO_EXIT) not in none.index:
            continue
        nb = none.loc[(st, PLACEBO_EXIT)]
        for _, r in g_.iterrows():
            o = g_[g_["k"] != r["k"]]["dd_cagr_1993-2023"].to_numpy()
            ok = all(r[f"dd_cagr_{p}"] > nb[f"dd_cagr_{p}"] for p in gs.JUDGED) and r["dd_cagr_1993-2023"] > np.percentile(o, 95)
            pp.append({"setting": st, "base": base, "k": r["k"], "pass": bool(ok)})
    pp = pd.DataFrame(pp)
    cnt = []
    for f in FILTERS[1:]:
        hf = h[h["filter"] == f]
        p0 = float(pp.loc[pp["base"] == f, "pass"].mean()) if len(pp) else np.nan
        cnt.append({"filter": f, "systems": len(hf), "helps": int(hf["helps"].sum()), "above_none": int(hf["above_none"].sum()),
                    "p0": p0, "q95": ks.q95(len(hf), p0) if np.isfinite(p0) else len(hf)})
    c = pd.DataFrame(cnt)
    c["beyond_chance"] = c["helps"] > c["q95"]
    return h, c


def candidates(g: pd.DataFrame, bench: dict, pass40: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cells (P1 settings, gates, 20 bp) with Stage O's per-cell test; per setting x entry x gate: both
    exits at 20 and 40 bp and the same entry x gate at an adjacent P1 setting; entry-specific by E0."""
    c = g[(g["mode"] == "grid") & (g["cost"] == pj.COST_BPS) & (g["filter"] != "none")].copy()
    c["cell_pass"] = [gs.cell_pass(r, bench) for _, r in c.iterrows()]
    e0 = c[c["gen"] == "E0"].set_index(["setting", "exit", "filter"])
    c["beats_e0"] = [all(r[f"dd_cagr_{h}"] > e0.loc[(r["setting"], r["exit"], r["filter"]), f"dd_cagr_{h}"] for h in es.HALVES)
                     if (r["setting"], r["exit"], r["filter"]) in e0.index else False for _, r in c.iterrows()]
    c["pass_40bp"] = [bool(pass40.get((r["setting"], r["gen"], r["exit"], r["filter"]), False)) for _, r in c.iterrows()]
    p1 = {ns.setting_name(s) for s in P1_SETTINGS}
    cc = c[(c["gen"] != "E0") & c["setting"].isin(p1)]
    s = cc.groupby(["setting", "gen", "filter"]).agg(n20=("cell_pass", "sum"), n40=("pass_40bp", "sum"),
                                                   n_e0=("beats_e0", lambda b: int((b & cc.loc[b.index, "cell_pass"]).sum()))).reset_index()
    s["ok"] = (s["n20"] >= len(EXITS)) & (s["n40"] >= len(EXITS))
    ok = set(zip(s.loc[s["ok"], "setting"], s.loc[s["ok"], "gen"], s.loc[s["ok"], "filter"]))
    order = [ns.setting_name(x) for x in P1_SETTINGS]
    s["neighbour"] = [any((order[j], gen, f) in ok for j in (order.index(st) - 1, order.index(st) + 1) if 0 <= j < len(order))
                      for st, gen, f in zip(s["setting"], s["gen"], s["filter"])]
    s["candidate"] = s["ok"] & s["neighbour"]
    s["entry_specific"] = s["candidate"] & (s["n_e0"] >= len(EXITS))
    return c, s


def chance(acc: pd.DataFrame, cells: pd.DataFrame, bench: dict) -> pd.DataFrame:
    e0 = acc[(acc["gen"] == "E0") & (acc["mode"] == "grid") & (acc["cost"] == pj.COST_BPS) & (acc["filter"] != "none")]
    seeds = e0.groupby(["setting", "exit", "filter", "src"]).median(numeric_only=True).reset_index()
    seeds["cell_pass"] = [gs.cell_pass(r, bench) for _, r in seeds.iterrows()]
    rows = []
    for (st, x, f), g_ in cells[cells["gen"] != "E0"].groupby(["setting", "exit", "filter"], sort=False):
        sd = seeds[(seeds["setting"] == st) & (seeds["exit"] == x) & (seeds["filter"] == f)]
        p0 = float(sd["cell_pass"].mean()) if len(sd) else np.nan
        rows.append({"setting": st, "exit": x, "filter": f, "entries": len(g_), "passing": int(g_["cell_pass"].sum()),
                     "passing_list": " ".join(sorted(g_.loc[g_["cell_pass"], "gen"])), "seeds_passing": int(sd["cell_pass"].sum()),
                     "p0": p0, "q95": ks.q95(len(g_), p0) if np.isfinite(p0) else len(g_)})
    out = pd.DataFrame(rows)
    out["beyond_chance"] = out["passing"] > out["q95"] if len(out) else []
    return out


def trade_stats(trades: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """N20 / W2 / P1 x X1: each gate's (and placebo's) trade returns vs no gate's, and A1's blocked vs
    bought signals (each regime), pooled over the 15 entries."""
    rows = []
    ref = trades[trades["filter"] == "none"]["ret"].to_numpy(dtype=float)
    ref = ref[np.isfinite(ref)]
    rng = np.random.default_rng([SEED, 7])
    ref_boot = pj.boot_means(ref, rng)
    shares = lambda a: {"mean": float(a.mean()), "loss20": float((a <= -0.2).mean()), "big50": float((a >= 0.5).mean())}  # noqa: E731
    for f, g_ in trades[trades["filter"] != "none"].groupby("filter", sort=False):
        r = g_["ret"].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        rows.append({"kind": "placebo" if "#" in f else "gate", "filter": f, "base": f.split("#")[0], "n": len(r),
                     **pj.ret_stats(r, ref, ref_boot, rng), **shares(r), **{f"ref_{k}": v for k, v in shares(ref).items()}})
    if sig is not None and len(sig):
        for reg in REGIMES:
            on, off = sig.loc[sig[reg], "ret"].to_numpy(dtype=float), sig.loc[~sig[reg], "ret"].to_numpy(dtype=float)
            on, off = on[np.isfinite(on)], off[np.isfinite(off)]
            if len(off) >= pj.MIN_STAT_SIGNALS:
                rows.append({"kind": "blocked_vs_bought", "filter": f"{reg}A1", "base": f"{reg}A1", "n": len(off),
                             **pj.ret_stats(off, on, pj.boot_means(on, rng), rng), **shares(off), **{f"ref_{k}": v for k, v in shares(on).items()}})
    return pd.DataFrame(rows)


def repro(g: pd.DataFrame, stage_o: Path) -> pd.DataFrame:
    """No-gate cells against Stage O's goal table (M1 account, 20 bp)."""
    so = pd.read_csv(stage_o / "goal.csv.gz")
    so = so[(so["kind"] == "rule") & (so["mode"] == "grid") & (so["cost"] == pj.COST_BPS)].set_index(["setting", "gen", "exit"])
    mine = g[(g["filter"] == "none") & (g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)].set_index(["setting", "gen", "exit"])
    cols = ["trades"] + [f"{s}_{p}" for p in es.PERIODS for s in ("cagr", "mdd")] + gs.GOAL_COLS
    out = []
    for key in mine.index.intersection(so.index):
        for c in cols:
            if c in so.columns and c in mine.columns:
                out.append({"setting": key[0], "gen": key[1], "exit": key[2], "value": c, "mine": float(mine.loc[key, c]),
                            "stage_o": float(so.loc[key, c])})
    r = pd.DataFrame(out)
    if len(r):
        r["diff"] = (r["mine"] - r["stage_o"]).abs()
    return r


def always_on_check(acc: pd.DataFrame) -> pd.DataFrame:
    """Dry: the always-on regime (R0A1 / R0A2) curve by curve against no gate."""
    a = acc[acc["mode"] == "grid"]
    none = a[a["filter"] == "none"].set_index(["setting", "gen", "exit", "src", "order"])
    cols = ["trades", "cagr_1993-2023", "mdd_1993-2023", "w", "dd_cagr_1993-2023", "ws_ma200"]
    out = []
    for f in ("R0A1", "R0A2"):
        x = a[a["filter"] == f].set_index(["setting", "gen", "exit", "src", "order"])
        common = x.index.intersection(none.index)
        if len(common):
            d = (x.loc[common, cols] - none.loc[common, cols]).abs().to_numpy()
            out.append({"filter": f, "curves": len(common), "max_diff": float(np.nanmax(d))})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- figures

def fig_heatmaps(out: Path, sysm: pd.DataFrame, bench: dict, ewr: dict) -> Path:
    """① Regime x action per setting and exit: the system value (1993-2023) and its change from no gate."""
    plt = figures._plt()
    fig, axes = plt.subplots(len(EXITS), len(SETTINGS), figsize=(4.3 * len(SETTINGS), 4.0 * len(EXITS)), squeeze=False)
    for i, x in enumerate(EXITS):
        for j, st in enumerate(SETTINGS):
            ax = axes[i, j]
            g = sysm[(sysm["setting"] == ns.setting_name(st)) & (sysm["exit"] == x)].set_index("filter")
            base = float(g.loc["none", "dd_cagr_1993-2023"]) * 100 if "none" in g.index else np.nan
            mat = np.full((len(REGIMES), len(ACTIONS)), np.nan)
            for a_, reg in enumerate(REGIMES):
                for b_, act in enumerate(ACTIONS):
                    f = f"{reg}{act}"
                    if f in g.index:
                        mat[a_, b_] = float(g.loc[f, "dd_cagr_1993-2023"]) * 100
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=3.5, vmax=8.0)
            for a_ in range(mat.shape[0]):
                for b_ in range(mat.shape[1]):
                    if np.isfinite(mat[a_, b_]):
                        ax.text(b_, a_, f"{mat[a_, b_]:.1f}%\n({mat[a_, b_] - base:+.1f})", ha="center", va="center", fontsize=8,
                                color="white" if mat[a_, b_] < 5.8 else "black")
            ax.set_xticks(range(len(ACTIONS)), [f"{k} {v}" for k, v in ACTIONS.items()], fontsize=7)
            ax.set_yticks(range(len(REGIMES)), [f"{k} {REGIMES[k]}\n(동일비중×국면 {ewr[k]['dd_cagr_1993-2023'] * 100:.1f}%)" for k in REGIMES], fontsize=7)
            ax.set_title(f"{ns.setting_name(st)} × {x}: 필터 없음 {base:.1f}%", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"국면 필터별 낙폭 15% 맞춤 수익 1993–2023 (매수 15개 중앙값; 괄호 = 필터 없음 대비 %p) — 기준선 SPY+국채 "
                 f"{bench['mix']['dd_cagr_1993-2023'] * 100:.1f}%, SPY 200일선+국채 {bench['ma200']['dd_cagr_1993-2023'] * 100:.1f}%, "
                 f"동일비중+국채 {bench['ew']['dd_cagr_1993-2023'] * 100:.1f}%", fontsize=10.5)
    return figures._save(fig, out / "figures" / "gates_heatmap.png")


def fig_before_after(out: Path, sysm: pd.DataFrame, bench: dict) -> Path:
    """② The 8 systems' plain MDD and w* from no gate to A1 to A2, per regime (medians over the 8)."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    steps = ["none", "A1", "A2"]
    for c_, (col, lab) in enumerate((("mdd_1993-2023", "보통 MDD 1993–2023 (%)"), ("w", "쓸 수 있는 자본 w*"), ("dd_cagr_1993-2023", "낙폭 15% 맞춤 수익 (%)"))):
        ax = axes[c_]
        for k, reg in enumerate(REGIMES):
            ys = []
            for s_ in steps:
                f = "none" if s_ == "none" else f"{reg}{s_}"
                ys.append(float(sysm.loc[sysm["filter"] == f, col].median()))
            scale = 1 if col == "w" else 100
            ax.plot(range(3), np.array(ys) * scale, marker="o", color=figures.COLORS[k % 4] if k < 4 else "#555", label=f"{reg} {REGIMES[reg]}")
        if col != "mdd_1993-2023":
            ref = bench["ma200"]["w" if col == "w" else "dd_cagr_1993-2023"] * (1 if col == "w" else 100)
            ax.axhline(ref, color="#8e44ad", ls="--", lw=1, label="SPY 200일선 + 국채")
        else:
            ax.axhline(bench["ma200"]["mdd_1993-2023"] * 100, color="#8e44ad", ls="--", lw=1, label="SPY 200일선")
        ax.set_xticks(range(3), ["필터 없음", "A1 새 매수 중단", "A2 + 전량 청산"], fontsize=8)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("국면 필터 전후: 시스템 8개(설정 4 × 매도 2)의 중앙값 — 낙폭 · 쓸 수 있는 자본 · 맞춘 수익", fontsize=11)
    return figures._save(fig, out / "figures" / "before_after.png")


def fig_placebo(out: Path, ps: pd.DataFrame, sysm: pd.DataFrame) -> Path:
    """④ Placebo regimes: the 20 shifted series' system values vs the real one (N20 / W2 / P1 x X1)."""
    plt = figures._plt()
    fig, axes = plt.subplots(len(ACTIONS), len(REGIMES), figsize=(3.4 * len(REGIMES), 3.0 * len(ACTIONS)), squeeze=False)
    st = ns.setting_name(PLACEBO_REF)
    none = float(sysm[(sysm["setting"] == st) & (sysm["exit"] == PLACEBO_EXIT) & (sysm["filter"] == "none")]["dd_cagr_1993-2023"].iloc[0]) * 100
    for i, act in enumerate(ACTIONS):
        for j, reg in enumerate(REGIMES):
            ax = axes[i, j]
            f = f"{reg}{act}"
            pv = ps[(ps["setting"] == st) & (ps["base"] == f)]["dd_cagr_1993-2023"].to_numpy() * 100
            real = sysm[(sysm["setting"] == st) & (sysm["exit"] == PLACEBO_EXIT) & (sysm["filter"] == f)]["dd_cagr_1993-2023"]
            ax.hist(pv, bins=12, color="#bbbbbb")
            if len(real):
                ax.axvline(float(real.iloc[0]) * 100, color=figures.COLORS[0], lw=2, label="실제 국면")
            if len(pv):
                ax.axvline(np.percentile(pv, 95), color="#555", ls="--", lw=1, label="가짜 95%")
            ax.axvline(none, color="#c0392b", ls=":", lw=1.2, label="필터 없음")
            ax.set_title(f"{reg} {REGIMES[reg]} · {act}", fontsize=8.5)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"가짜 국면(시점만 1년 이상 민 것) 20개의 낙폭 맞춤 수익 분포 vs 실제 국면 ({st} × {PLACEBO_EXIT}, 매수 15개 중앙값, %)", fontsize=10.5)
    return figures._save(fig, out / "figures" / "placebo.png")


def fig_trades(out: Path, trades: pd.DataFrame, tstats: pd.DataFrame, best: str | None) -> Path | None:
    """⑤ Trade returns: no gate vs the best A2 gate (N20 / W2 / P1 x X1)."""
    if best is None:
        return None
    plt = figures._plt()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for f, col, lab in (("none", "#777", "필터 없음"), (best, figures.COLORS[0], f"{best} {REGIMES[best[:2]]} · {ACTIONS[best[2:]]}")):
        r = trades.loc[trades["filter"] == f, "ret"].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        pj._bar(ax, pj._density(np.histogram(r, pj.RET_EDGES)[0]), color=col, lw=1.5, label=f"{lab} (거래 {len(r):,}개, 평균 {r.mean() * 100:+.1f}%)")
    s = tstats[(tstats["kind"] == "gate") & (tstats["filter"] == best)]
    if len(s) and pd.notna(s.iloc[0].get("st_diff")):
        s = s.iloc[0]
        ax.text(0.98, 0.6, f"평균 차 {s['st_diff'] * 100:+.1f}%p (95% CI {s['st_ci_lo'] * 100:+.1f} – {s['st_ci_hi'] * 100:+.1f})\n"
                           f"순위 검정 p = {s['st_mw_p']:.3f}\n−20% 이하 {s['loss20'] * 100:.1f}% (필터 없음 {s['ref_loss20'] * 100:.1f}%)\n"
                           f"+50% 이상 {s['big50'] * 100:.1f}% (필터 없음 {s['ref_big50'] * 100:.1f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("거래 하나의 수익률 (%, 편도 20bp, 양 끝 칸은 −60% 미만 / +200% 초과)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_title(f"계좌의 거래 수익률: 필터 없음 vs 가장 나은 A2 필터 ({ns.setting_name(TRADE_SETTING)} × {TRADE_EXIT}, 매수 15개 합산)", fontsize=10.5)
    return figures._save(fig, out / "figures" / "trades_hist.png")


def fig_windows(out: Path, win: pd.DataFrame, best: tuple | None, bench: dict, ewr: dict, wins: list) -> Path | None:
    """⑥ 3-year windows: the best gated system vs the SPY mix, SPY200 and EW x the same regime."""
    if best is None:
        return None
    plt = figures._plt()
    st, x, f = best
    w = win[(win["setting"] == st) & (win["exit"] == x) & (win["filter"] == f) & (win["gen"] != "E0") & (win["cost"] == pj.COST_BPS)]
    sysw = w.groupby("w_idx")["win_cagr"].median().reindex(range(len(wins))).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(wins))
    ax.plot(xs, sysw, marker="o", color=figures.COLORS[0], label=f"시스템 {st} × {x} × {f} (매수 15개 중앙값)")
    for b, col, lab in (("mix", "#c0392b", "SPY + 국채"), ("ma200", "#8e44ad", "SPY 200일선 + 국채")):
        bw = bench[b]["win_cagr"] * 100
        ax.plot(xs, bw, marker="s", ms=4, color=col, label=f"{lab} (시스템이 앞선 창 {np.mean(sysw >= bw):.0%})")
    reg = f[:2]
    if reg in ewr:
        ew = ewr[reg]["win_cagr"] * 100
        ax.plot(xs, ew, marker="^", ms=4, color="#555", label=f"동일비중 × {reg} (시스템이 앞선 창 {np.mean(sysw >= ew):.0%})")
    ax.set_xticks(xs, [f"{a}–{str(b)[2:]}" for a, b in wins], rotation=90, fontsize=7.5)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("3년 창 CAGR, 낙폭 15% 맞춤 (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("3년 롤링 창: 가장 나은 국면 필터 시스템 vs 기준선 (각자 전체 기간 w*)", fontsize=10.5)
    return figures._save(fig, out / "figures" / "windows_gate.png")


def fig_curves(out: Path, curves: dict) -> Path:
    """③ w* curves: the best gated cell vs SPY200 + T-bill vs EW x the same regime, 1993-2023 and 2007-2010."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for k, (lab, s) in enumerate(curves.items()):
        col = [figures.COLORS[0], "#8e44ad", "#555", "#c0392b"][k % 4]
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


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "r0x1", "tables", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--stage-o", default=None, help="Stage O 결과표 폴더 (재현 점검)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="E2·V3·E0(씨앗 2), 국면 R1·R5, 가짜 R1, '항상 좋음' 국면 점검")
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
        regime_states(panel.close["SPY"]).to_parquet(out / "regimes.parquet")
        del panel
    states = pd.read_parquet(out / "regimes.parquet")
    ts.log(f"  신호 {len(base):,}개 · 국면 꺼진 날 비율(전체 패널): " + ", ".join(f"{r} {1 - states[r].mean():.0%}" for r in states.columns))
    regs = ["R1", "R5"] if a.dry else list(REGIMES)
    flt = ["none"] + [f"{r}{x}" for r in regs for x in ACTIONS]
    size = base.groupby("src").size()
    jobs = []
    for gen, srcs in gens.items():
        for x in EXITS:
            for st in SETTINGS:
                if st[2] == "P0":  # heavy: one job per regime
                    jobs.append((gen, x, st, ["none"] + (["R0A1", "R0A2"] if a.dry else []), pj.COST_BPS, srcs, "grid"))
                    jobs += [(gen, x, st, [f"{r}A1", f"{r}A2"], pj.COST_BPS, srcs, "grid") for r in regs]
                else:
                    jobs.append((gen, x, st, flt + (["R0A1", "R0A2"] if a.dry else []), pj.COST_BPS, srcs, "grid"))
        if gen != "E0":
            for st in PLACEBO_SETTINGS:
                jobs += [(gen, PLACEBO_EXIT, st, [r], pj.COST_BPS, srcs, "placebo") for r in (["R1"] if a.dry else regs)]
    jobs.sort(key=lambda j: -int(size.reindex(j[5]).sum()) * (5 if j[2][2] == "P0" else 1) * (1 + j[2][0] / 50))
    jdir = out / "jobs"
    todo = [j for j in jobs if not (jdir / f"{job_name(j[0], j[1], j[2], j[3], j[4], j[6])}_acc.csv").exists()]
    ts.log(f"2단계: 작업 {len(todo)}개 남음 (격자 {sum(j[6] == 'grid' for j in jobs)}, 가짜 {sum(j[6] == 'placebo' for j in jobs)})")
    run_jobs(todo, a)
    read = lambda s: pd.concat([pd.read_csv(f) for f in sorted(jdir.glob(f"*_{s}.csv"))  # noqa: E731
                                if "bp_" not in f.name and f.stat().st_size > 2], ignore_index=True)
    acc, win, mom = read("acc"), read("win"), read("mom")
    trades = pd.concat([pd.read_parquet(f) for f in sorted(jdir.glob("*_trades.parquet")) if "bp_" not in f.name], ignore_index=True)
    sigs = [pd.read_parquet(f) for f in sorted(jdir.glob("*_signals.parquet"))]
    sig = pd.concat(sigs, ignore_index=True) if sigs else None
    pj._init(a.panel, a.eligible, a.dirs, a.cfg)
    px, wdates, w0, cash = pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"]
    btab, bench = gs.benchmarks_table(px, wdates, w0, cash)
    sp = gs._setup(cash, wdates)
    ew = gs.bench_curves(px, wdates, w0, cash)["ew"]
    st_w = {r: states[r].reindex(wdates).fillna(False).to_numpy(dtype=bool) for r in REGIMES}
    EQ = np.column_stack([ew_regime(ew, cash, st_w[r], pj.COST_BPS / 10_000) for r in REGIMES])
    R = EQ[1:] / EQ[:-1] - 1.0
    gm = gs.goal_metrics(R, np.repeat(sp["c"][:, None], R.shape[1], axis=1), sp["years"], sp["wins"])
    ewr = {}
    for i, r in enumerate(REGIMES):
        ewr[r] = {**{k: float(v[i]) for k, v in gm.items() if k != "win_cagr"}, "win_cagr": gm["win_cagr"][:, i],
                  **es.period_stats(pd.Series(EQ[:, i], index=wdates))}
    btab = pd.concat([btab, pd.DataFrame([{"benchmark": f"ew_{r}", "label": f"동일비중 × {r} {REGIMES[r]} (A2) + 국채",
                                           **{k: v for k, v in ewr[r].items() if k != "win_cagr"}} for r in REGIMES])], ignore_index=True)
    btab.to_csv(out / "benchmarks.csv", index=False)
    rst = regime_stats(states[list(REGIMES)], wdates)
    rst.to_csv(out / "regimes.csv", index=False)
    g = summarize(acc)
    summary: dict = {}
    if a.stage_o:
        rp = repro(g, Path(a.stage_o))
        rp.to_csv(out / "repro.csv", index=False)
        ts.log(f"재현 점검: Stage O 대비 {rp[['setting', 'gen', 'exit']].drop_duplicates().shape[0]}칸 {len(rp)}값, 최대 차 {rp['diff'].max():.2e}, "
               f"1e-9 넘는 값 {int((rp['diff'] > 1e-9).sum())}개")
        summary["repro"] = {"cells": int(rp[["setting", "gen", "exit"]].drop_duplicates().shape[0]), "values": len(rp), "max_diff": float(rp["diff"].max())}
    if a.dry:
        ac = always_on_check(acc)
        ac.to_csv(out / "always_on.csv", index=False)
        ts.log("'항상 좋음' 국면 vs 필터 없음: " + ", ".join(f"{r.filter} {r.curves}곡선 최대 차 {r.max_diff:.2e}" for r in ac.itertuples()))
    g = g[~g["filter"].str.startswith("R0")]
    sysm = systems(g, bench, ewr)
    ps = placebo_systems(g)
    help_rows, help_cnt = q1(sysm, ps)
    cells, _ = candidates(g, bench, {})
    need = cells[(cells["gen"] != "E0") & cells["cell_pass"] & cells["setting"].isin({ns.setting_name(s) for s in P1_SETTINGS})]
    per = need.groupby(["setting", "gen", "filter"])["cell_pass"].sum()
    pass40 = {}
    if (per >= len(EXITS)).any():
        want = sorted({(gen, x, st, f) for (st, gen, f) in per[per >= len(EXITS)].index for x in EXITS})
        by_gs = {}
        for gen, x, st, f in want:
            by_gs.setdefault((gen, x, st), []).append(f)
        jobs40 = [(gen, x, ns.parse_setting(st), fs, pj.CHECK_COST_BPS, gens[gen], "grid") for (gen, x, st), fs in by_gs.items()]
        run_jobs([j for j in jobs40 if not (jdir / f"{job_name(j[0], j[1], j[2], j[3], j[4], j[6])}_acc.csv").exists()], a)
        a40 = pd.concat([pd.read_csv(jdir / f"{job_name(j[0], j[1], j[2], j[3], j[4], j[6])}_acc.csv") for j in jobs40], ignore_index=True)
        for _, r in summarize(a40).iterrows():
            pass40[(r["setting"], r["gen"], r["exit"], r["filter"])] = gs.cell_pass(r, bench)
    cells, cand = candidates(g, bench, pass40)
    ch = chance(acc[~acc["filter"].str.startswith("R0")], cells, bench)
    tstats = trade_stats(trades[~trades["filter"].str.startswith("R0")], sig)
    crises = sysm[["setting", "exit", "filter", "mdd_2000-2002", "mdd_2007-2009", "mdd_1993-2023", "time_in_market"]]
    dsr = pd.DataFrame()
    if cand["candidate"].any():
        m = mom.set_index(["setting", "gen", "exit", "filter"])
        sr_var = float(np.var(m["m_sr"].dropna()))
        curves = gs.bench_curves(px, wdates, w0, cash)
        rm = curves["mix"][1:] / curves["mix"][:-1] - 1.0
        bl = bench["mix"]["w"] * rm + (1 - bench["mix"]["w"]) * cash[1:]
        rows = []
        for _, r in cells[cells["cell_pass"]].iterrows():
            ok = cand[(cand["setting"] == r["setting"]) & (cand["gen"] == r["gen"]) & (cand["filter"] == r["filter"]) & cand["candidate"]]
            key = (r["setting"], r["gen"], r["exit"], r["filter"])
            if len(ok) and key in m.index:
                mm = m.loc[key]
                rows.append({"setting": key[0], "gen": key[1], "exit": key[2], "filter": key[3],
                             "dsr": ks.dsr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), N_TRIALS, sr_var),
                             "psr_mix": ks.psr(mm["m_sr"], mm["m_g3"], mm["m_g4"], int(mm["m_T"]), float(bl.mean() / bl.std()))})
        dsr = pd.DataFrame(rows)
    g.to_csv(out / "goal.csv", index=False)
    sysm.to_csv(out / "systems.csv", index=False)
    cells.to_csv(out / "cells.csv", index=False)
    ps.to_csv(out / "placebo.csv", index=False)
    help_rows.to_csv(out / "helps.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    ch.to_csv(out / "chance.csv", index=False)
    tstats.to_csv(out / "trades_stats.csv", index=False)
    crises.to_csv(out / "crises.csv", index=False)
    win.groupby(["setting", "gen", "exit", "filter", "cost", "w_idx", "a", "b"])["win_cagr"].median().reset_index().to_csv(out / "windows.csv", index=False)
    s_g = sysm[sysm["filter"] != "none"]
    best = s_g.sort_values("dd_cagr_1993-2023", ascending=False).head(1)
    best_key = tuple(best[["setting", "exit", "filter"]].iloc[0]) if len(best) else None
    a2 = s_g[(s_g["setting"] == ns.setting_name(TRADE_SETTING)) & (s_g["exit"] == TRADE_EXIT) & s_g["filter"].str.endswith("A2")]
    best_a2 = a2.sort_values("dd_cagr_1993-2023", ascending=False)["filter"].iloc[0] if len(a2) else None
    gate_stats = tstats[tstats["kind"] == "gate"]
    pl_stats = tstats[tstats["kind"] == "placebo"]
    summary.update({
        "benchmarks": {r["benchmark"]: {c: r[c] for c in ["w", "dd_cagr_1993-2023", "dd_cagr_1993-2007", "dd_cagr_2008-2023", "dd_cagr_2017-2023"]}
                       for _, r in btab.iterrows()},
        "regimes": rst.to_dict("records"),
        "q1_counts": help_cnt.to_dict("records"),
        "systems_passing": s_g.loc[s_g["system_pass"], ["setting", "exit", "filter", "beats_ew_regime"]].to_dict("records"),
        "best_system": s_g.sort_values("dd_cagr_1993-2023", ascending=False).head(8)[["setting", "exit", "filter", "w", "dd_cagr_1993-2023",
                                                                                       "dd_cagr_1993-2007", "dd_cagr_2008-2023"]].to_dict("records"),
        "cells_passing": int(cells.loc[cells["gen"] != "E0", "cell_pass"].sum()),
        "candidates": cand.loc[cand["candidate"], ["setting", "gen", "filter", "entry_specific"]].to_dict("records"),
        "dsr": dsr.to_dict("records"),
        "trade_tests": {"gates_p_lt_05": float((gate_stats["st_mw_p"] < 0.05).mean()) if len(gate_stats) else np.nan,
                        "placebos_p_lt_05": float((pl_stats["st_mw_p"] < 0.05).mean()) if len(pl_stats) else np.nan},
        "trials": {"before": gs.N_TRIALS, "new": N_NEW}})
    (out / "chance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if figures.available():
        fig_heatmaps(out, sysm, bench, ewr)
        fig_before_after(out, sysm, bench)
        if len(ps):
            fig_placebo(out, ps, sysm)
        fig_trades(out, trades, tstats, best_a2)
        fig_windows(out, win, best_key, bench, ewr, sp["wins"])
        best_cell = cells[(cells["gen"] != "E0") & cells["setting"].isin({ns.setting_name(s) for s in P1_SETTINGS})
                          & (cells["cost"] == pj.COST_BPS)].sort_values("dd_cagr_1993-2023", ascending=False).head(1)
        if len(best_cell):
            r = best_cell.iloc[0]
            fig_curves(out, best_curves(a, r, bench, ewr, px, wdates, w0, cash, st_w, gens))
    ts.log(f"완료 · '도움' 필터(8개 시스템 중): " + ", ".join(f"{r.filter} {r.helps}/{r.q95}" for r in help_cnt.itertuples())
           + f" · 목표 통과 시스템 {len(summary['systems_passing'])} · 통과 칸 {summary['cells_passing']} · 후보 {len(summary['candidates'])}")


def best_curves(a, r, bench, ewr, px, wdates, w0, cash, st_w, gens) -> dict:
    """The best gated cell's curve (its median order) and the benchmarks, each blended at its own w*."""
    _init(a.panel, a.eligible, a.dirs, a.cfg)
    st = ns.parse_setting(r["setting"])
    srcs = gens[r["gen"]]
    s = srcs[0]
    d = ms.prepared(s, r["exit"])
    b = ms.source_base(s)
    rows = gs.GRows(d, b, r["exit"])
    keep = (~ms.excluded(b, ns.RULE))[rows.brow]
    w = ns.sizing(rows.stop, st[0], "W2")[0]
    tie = np.random.default_rng([ns.SEED, zlib.crc32(f"tie|{s}|{r['exit']}".encode())]).random(len(rows))
    key = rows.k2 + tie * 1e-9 if st[2] == "P1" else -rows.cap + tie * 1e-6
    state, a2 = _filter_state(r["filter"])
    eq, _ = _replay(rows, d["si"].to_numpy(), keep, w, key, st[0], pj.COST_BPS / 10_000, state, a2)
    out = {}

    def blended(eq_, ww):
        rr = eq_[1:] / eq_[:-1] - 1.0
        return pd.Series(np.concatenate([[1.0], np.cumprod(1 + ww * rr + (1 - ww) * cash[1:])]), index=wdates)

    rr = eq[1:] / eq[:-1] - 1.0
    from . import target as tg
    w_s = tg.fit_weight(rr, cash[1:])
    out[f"{r['setting']} × {r['gen']} × {r['exit']} × {r['filter']} (w* {w_s:.2f})"] = blended(eq, w_s)
    cv = gs.bench_curves(px, wdates, w0, cash)
    out[f"SPY 200일선 + 국채 (w* {bench['ma200']['w']:.2f})"] = blended(cv["ma200"], bench["ma200"]["w"])
    reg = r["filter"][:2]
    ew_c = ew_regime(cv["ew"], cash, st_w[reg], pj.COST_BPS / 10_000)
    out[f"동일비중 × {reg} (w* {ewr[reg]['w']:.2f})"] = blended(ew_c, ewr[reg]["w"])
    out[f"SPY + 국채 (w* {bench['mix']['w']:.2f})"] = blended(cv["mix"], bench["mix"]["w"])
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
