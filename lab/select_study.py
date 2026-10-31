"""Stage K: stock-selection rules one at a time, against day-matched pseudo rules (DECISIONS 91).

    python -m lab.select_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                               --r0x1 DIR --out DIR [--workers N] [--feature-workers N] [--dry]

Among the signals of a generator (E0 random 1% buys, 5 seeds; V3 volume burst; E2 golden cross:
the saved Stage I tables, no engine run), a line keeps only the stocks on one side of one feature:
the favoured 30% or the opposite 30% of that day's hygiene universe (percentile on the signal day,
data up to that close), or a cap-rank group. Its null is 40 pseudo rules that keep, on every day the
line kept n of N signals, n of the same N at random, so only "which stock" differs. Exits H1 k 2,
H1 k 3 and X1; 10-stock 1%-rule accounts (100 random orders, E0: 20 per seed) and the no-limit
contribution; verdicts against the pseudo median, chance counts from pseudo-vs-pseudo, candidates
as Stage I, per-signal return histograms and tests as Stage J3.
Output (aggregates only): base.parquet (stage 1), jobs/*.csv, lines.csv, verdicts.csv, stats.csv,
hist.csv, accounts.csv, chance.json, baselines.csv, figures/.
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
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import entry_study as es, figures, pair_study as pj, pick_study as ph, screens, tail_study as ts
from .crsp_study import load_panel_with_cash
from .engine import EXIT_BLOCKS

# ---- registered (DECISIONS 91): fixed before any result
# feature -> label, score (higher = the score's direction), parameter, favoured side of the score
FEATURES = {
    "K1": ("주가 낮음", "s_low_price", None, "high"),
    "K2": ("고유 변동성 낮음 (63일, SPY 대비 잔차)", "s_low_ivol", None, "high"),
    "K3": ("회전율 높음", "s_turnover", None, "high"),
    "K5": ("12-1개월 수익률 낮음", "s_momentum", 12, "low"),
    "K6": ("업종 상대강도 높음 (6-1)", "s_industry_rs", 6, "high"),
    "K7": ("36개월 수익률 낮음", "s_reversal", 36, "high"),  # s_reversal(36) = -(36-month return)
    "K8": ("60일 변동성 낮음", "s_lowvol", None, "high"),
    "K9": ("추세 효율 높음 (6개월)", "s_efficiency", 6, "high"),
    "K10": ("상대강도 높음 (IBD식)", "s_rs", None, "high"),
    "K11": ("52주 고가 근접", "s_high52", None, "high"),
    "K12": ("거래대금 낮음 (63일)", "s_dollar_volume", None, "low"),
    "K13": ("12개월 낙폭 얕음", "s_dd_shallow", 12, "high"),
    "K14": ("매집/분산 높음 (3개월)", "s_accum", 3, "high"),
}
CAP_GROUPS = {"K4m": ("시총 중형 (1,001–2,000위)", 1001, 2000), "K4l": ("시총 대형 (1–1,000위)", 1, 1000),
              "K4s": ("시총 소형 (2,001–3,000위)", 2001, 3000)}
HYPOTHESIS = ("K1", "K2", "K3", "K5", "K6", "K7", "K8")  # Stage H coefficients, Stage E screens
CUT = 0.30
GENERATORS = {"E0": [f"R0s{s}" for s in pj.E0_SEEDS], "V3": ["V3"], "E2": ["E2"]}
EXITS = list(pj.EXITS_J)  # H1k2, H1k3, X1
N_ORDERS, N_PSEUDO, PSEUDO_ORDERS = 100, 40, 20
MAX_REF = 300_000  # pooled pseudo signals sampled for the per-signal tests
MIN_EXITS = 2
FIG_EXIT = "H1k3"
N_TRIALS_BEFORE = 69_818  # Stages B-L (approx.)
N_NEW = 261  # 29 lines x 3 generators x 3 exits
SEED = 91
STOP_LABEL = EXIT_BLOCKS["atr_stop"].label
_W: dict = {}
_F: dict = {}


def lines() -> list[str]:
    out = []
    for f in FEATURES:
        out += [f"{f}f", f"{f}o"]
    return out + list(CAP_GROUPS)


def line_label(line: str) -> str:
    if line in CAP_GROUPS:
        return CAP_GROUPS[line][0]
    lab = FEATURES[line[:-1]][0]
    return lab if line.endswith("f") else f"반대쪽: {lab}"


def line_mask(base: pd.DataFrame, line: str) -> np.ndarray:
    """True for the signals the line keeps (NaN feature values are never kept)."""
    if line in CAP_GROUPS:
        _, lo, hi = CAP_GROUPS[line]
        c = base["cap_rank"].to_numpy(dtype=float)
        return (c >= lo) & (c <= hi)
    f, side = line[:-1], line[-1]
    high = (FEATURES[f][3] == "high") == (side == "f")
    p = base[f].to_numpy(dtype=float)
    return p > 1.0 - CUT if high else p <= CUT


def pseudo_mask(si: np.ndarray, keep: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A day-matched random rule: on every day, as many signals as `keep` kept that day, chosen at
    random among that day's signals."""
    n_keep = np.bincount(si[keep], minlength=int(si.max()) + 1 if len(si) else 0)
    order = np.lexsort((rng.random(len(si)), si))
    s_sorted = si[order]
    first = np.searchsorted(s_sorted, s_sorted, side="left")
    pos = np.arange(len(si)) - first
    out = np.zeros(len(si), dtype=bool)
    out[order] = pos < n_keep[s_sorted]
    return out


# --------------------------------------------------------------------------- stage 1: base table

def _f_init(panel_dir: str, eligible: str):
    panel = load_panel_with_cash(Path(panel_dir))
    _F["panel"] = panel
    _F["elig"] = pd.read_parquet(eligible).reindex(index=panel.close.index, columns=panel.close.columns,
                                                   fill_value=False).astype(bool)


def feature_values(job):
    """Percentile (0-1] of a feature in the day's hygiene universe at (si, sj); cap rank as is."""
    key, si, sj = job
    panel, elig = _F["panel"], _F["elig"]
    t0 = time.time()
    if key == "cap_rank":
        frame = panel.extra["cap_rank"].reindex(index=panel.close.index, columns=panel.close.columns)
        vals = frame.to_numpy(dtype="float32")[si, sj]
    else:
        _, score, param, _ = FEATURES[key]
        frame = screens.SELECT_SCORES[score].compute(panel, param).where(elig).rank(axis=1, pct=True)
        vals = frame.to_numpy(dtype="float32")[si, sj]
    return key, vals, time.time() - t0


def build_base(a, gens: dict, feats: list[str]) -> pd.DataFrame:
    """Every generator's signals (symbol, signal day) with the feature percentiles and cap rank."""
    f = Path(a.out) / "base.parquet"
    if f.exists():
        return pd.read_parquet(f)
    pj._init(a.panel, a.eligible, a.dirs, a.cfg)
    parts = []
    for g, srcs in gens.items():
        for s in srcs:
            t = pj.rule_table(s, "H1k2")
            sj = pj._W["px"].cols.get_indexer(t["symbol"].astype(str))
            parts.append(pd.DataFrame({"gen": g, "src": s, "symbol": t["symbol"].astype(str).to_numpy(), "si": t["si"].to_numpy(),
                                       "sj": sj}))
    base = pd.concat(parts, ignore_index=True)
    si, sj = base["si"].to_numpy(), base["sj"].to_numpy()
    with ProcessPoolExecutor(max_workers=a.feature_workers, initializer=_f_init, initargs=(a.panel, a.eligible)) as ex:
        for key, vals, sec in ex.map(feature_values, [(k, si, sj) for k in feats + ["cap_rank"]], chunksize=1):
            base[key] = vals
            ts.log(f"  특징 {key}: {sec:.0f}s, 값 있음 {np.isfinite(vals).mean():.1%}")
    base.to_parquet(f)
    return base


# --------------------------------------------------------------------------- stage 2: per line x generator x exit

def _init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    pj._init(panel_dir, eligible, dirs, cfg)
    _W.update(pj._W)
    _W["base"] = pd.read_parquet(cfg["base"])
    _W["prep"] = {}


def source_base(src: str) -> pd.DataFrame:
    """The base rows (signals with features) of one source, in a fixed order (cached per worker)."""
    key = ("base", src)
    if key not in _W["prep"]:
        b = _W["base"]
        _W["prep"][key] = b[b["src"] == src].reset_index(drop=True)
    return _W["prep"][key]


def prepared(src: str, exit_key: str) -> pd.DataFrame:
    """A source's filled signals in the window with weight, contribution and return, plus the row
    of the base table they belong to (cached per worker)."""
    key = (src, exit_key)
    if key not in _W["prep"]:
        x, ki = pj.EXITS_J[exit_key]
        px, w0, n = _W["px"], _W["w0"], _W["n"]
        s = pj.rule_table(src, exit_key)
        m = px.measures(s, es.END)
        if x == "H1":
            w = ph.weights(px, m, ph.K_VALUES[ki])
        else:  # no stop: fixed 10% weight
            w = pd.DataFrame({"w": np.full(len(m), ph.CAP), "capped": True})
            w["contrib"] = w["w"] * m["ret"].to_numpy()
        d = pd.concat([m[["si", "ei", "sj", "sale", "closed", "x_spy", "ret"]], w[["w", "contrib"]],
                       s[["symbol", "signal_date", "holding_days", "exit_reason"]]], axis=1)
        ok = (d["ei"].to_numpy() >= w0) & (d["ei"].to_numpy() < w0 + n) & np.isfinite(d["w"].to_numpy())
        d = d[ok].reset_index(drop=True)
        d["symbol"] = d["symbol"].astype(str)
        b = source_base(src)[["symbol", "si"]].assign(brow=lambda x: np.arange(len(x)))  # position in the source's base rows
        d = d.merge(b, on=["symbol", "si"], how="left")
        _W["prep"][key] = d
    return _W["prep"][key]


def _replays(d: pd.DataFrame, n: int, rng: np.random.Generator, cost: float, curves: list | None = None) -> list[dict]:
    w0, nd, wdates = _W["w0"], _W["n"], _W["wdates"]
    ei, sale = d["ei"].to_numpy() - w0, d["sale"].to_numpy() - w0
    closed = d["closed"].to_numpy(dtype=bool) & (sale < nd)
    out = []
    for _ in range(n):
        eq, tr = ph.replay_risk(ei - 1, ei, d["sj"].to_numpy(), sale, closed, d["w"].to_numpy(), rng.random(len(d)),
                                _W["O"], _W["LC"], _W["cash"], cost)
        r = np.diff(eq) / eq[:-1]
        row = {"trades": len(tr), "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
               "cap_rank_trades": float(np.nanmedian(d["cap_rank"].to_numpy()[tr["row"].to_numpy()])) if len(tr) else np.nan,
               **es.period_stats(pd.Series(eq, index=wdates))}
        if curves is not None:
            curves.append(eq)
        out.append(row)
    return out


def _period_means(d: pd.DataFrame) -> dict:
    yr = pd.DatetimeIndex(d["signal_date"]).year.to_numpy()
    out = {}
    for p, (a, b) in es.PERIODS.items():
        mm = (yr >= a) & (yr <= b)
        out[f"n_{p}"] = int(mm.sum())
        out[f"mean_contrib_{p}"] = float(d["contrib"].to_numpy()[mm].mean()) if mm.any() else np.nan
    return out


def _moments(eq: np.ndarray) -> dict:
    """Daily-return moments of a curve for the deflated Sharpe ratio (as metrics.deflated_sharpe)."""
    r = np.diff(eq) / eq[:-1]
    sd = r.std(ddof=1)
    sd0 = r.std(ddof=0)
    return {"m_sr": float(r.mean() / sd), "m_g3": float(((r - r.mean()) ** 3).mean() / sd0 ** 3),
            "m_g4": float(((r - r.mean()) ** 4).mean() / sd0 ** 4), "m_T": len(r)}


def run_job(job):
    """One line x generator x exit: the line and its pseudo rules (accounts, contributions, stats)."""
    line, gen, exit_key, n_orders, n_pseudo, p_orders, cost_bps, srcs = job
    t0 = time.time()
    li, gi = lines().index(line), list(GENERATORS).index(gen)
    tag = f"{line}|{gen}|{exit_key}|{cost_bps}"
    rng = np.random.default_rng([SEED, zlib.crc32(tag.encode())])
    accs, uns = [], []
    curves: list = []
    rule_d, pseudo_d = [], [[] for _ in range(n_pseudo)]
    for si_, src in enumerate(srcs):
        b = source_base(src)
        keep = line_mask(b, line)
        d = prepared(src, exit_key)
        d = d[d["brow"].notna()].reset_index(drop=True)
        brow = d["brow"].to_numpy(dtype=int)
        d = d.assign(cap_rank=b["cap_rank"].to_numpy()[brow])
        dk = d[keep[brow]].reset_index(drop=True)
        rule_d.append(dk)
        per_src = n_orders // len(srcs)
        for row in _replays(dk, per_src, rng, cost_bps / 10_000, curves):
            accs.append({"kind": "rule", "k": -1, "src": src, **row})
        for k in range(n_pseudo):
            pm = pseudo_mask(b["si"].to_numpy(), keep, np.random.default_rng([SEED, li, gi, si_, k]))
            dp = d[pm[brow]].reset_index(drop=True)
            pseudo_d[k].append(dp)
            for row in _replays(dp, p_orders // len(srcs), rng, cost_bps / 10_000):
                accs.append({"kind": "pseudo", "k": k, "src": src, **row})
    rd = pd.concat(rule_d, ignore_index=True)
    base_n = int((_W["base"]["src"].isin(srcs)).sum())
    un = {"kind": "rule", "k": -1, "signals": len(rd), "kept_share": len(rd) / max(sum(len(prepared(s, exit_key)) for s in srcs), 1),
          "base_signals": base_n, "per_day": len(rd) / _W["n"] / len(srcs),
          "hold_median": float(np.median(rd["holding_days"][rd["closed"]])) if rd["closed"].any() else np.nan,
          "stop_share": float((rd["exit_reason"][rd["closed"]] == STOP_LABEL).mean()) if rd["closed"].any() else np.nan,
          "big_share": float((rd["contrib"] >= ph.BIG_CONTRIB).mean()) if len(rd) else np.nan,
          "cap_rank_signals": float(np.nanmedian(rd["cap_rank"])) if len(rd) else np.nan, **_period_means(rd)}
    if curves and cost_bps == pj.COST_BPS:  # the median-order rule curve's moments, for DSR of candidates
        c = [r["cagr_1993-2023"] for r in accs if r["kind"] == "rule"]
        un.update(_moments(curves[int(np.argsort(c)[len(c) // 2])]))
    uns.append(un)
    pds = []
    for k in range(n_pseudo):
        pk = pd.concat(pseudo_d[k], ignore_index=True)
        pds.append(pk)
        uns.append({"kind": "pseudo", "k": k, "signals": len(pk), **_period_means(pk)})
    stats, hists = [], []
    if cost_bps == pj.COST_BPS and n_pseudo:
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
        for c, v in (("mean_ret", np.mean), ("median_ret", np.median)):
            st[c] = float(v(r_rule)) if len(r_rule) else np.nan
            st[f"pseudo_{c}"] = float(v(pooled)) if len(pooled) else np.nan
        for c, f in (("big50", lambda x: (x >= 0.5).mean()), ("loss20", lambda x: (x <= -0.2).mean()), ("win", lambda x: (x > 0).mean())):
            st[c] = float(f(r_rule)) if len(r_rule) else np.nan
            st[f"pseudo_{c}"] = float(f(pooled)) if len(pooled) else np.nan
        stats.append(st)
        for k, p in enumerate(pds):  # each pseudo rule against the others (rank test only)
            rk = p["ret"].to_numpy(dtype=float)
            rk = rk[np.isfinite(rk)]
            other = ref[ref_lab != k]
            if len(rk) >= pj.MIN_STAT_SIGNALS and len(other) >= pj.MIN_STAT_SIGNALS:
                auc, pv = pj.mann_whitney(rk, other)
                stats.append({"kind": "pseudo", "k": k, "st_mw_auc": auc, "st_mw_p": pv, "st_diff": float(rk.mean() - other.mean())})
        yr_r = pd.DatetimeIndex(rd["signal_date"]).year.to_numpy()
        pall = pd.concat(pds, ignore_index=True)
        yr_p = pd.DatetimeIndex(pall["signal_date"]).year.to_numpy()
        for per in pj.HIST_PERIODS:
            a_, b_ = es.PERIODS[per]
            for kind, frame, yr, scale in (("rule", rd, yr_r, 1.0), ("pseudo", pall, yr_p, 1.0 / max(n_pseudo, 1))):
                r = frame["ret"].to_numpy(dtype=float)[(yr >= a_) & (yr <= b_)]
                r = r[np.isfinite(r)]
                hists.append({"kind": kind, "period": per, **{f"h{i}": float(c) * scale for i, c in enumerate(np.histogram(r, pj.RET_EDGES)[0])}})
    head = {"line": line, "gen": gen, "exit": exit_key, "cost": cost_bps}
    frame = lambda rows: pd.DataFrame([{**head, **r} for r in rows])  # noqa: E731
    return job, frame(accs), frame(uns), frame(stats), frame(hists), time.time() - t0


def job_name(line: str, gen: str, exit_key: str, cost: float) -> str:
    return f"{line}__{gen}__{exit_key}" + ("" if cost == pj.COST_BPS else f"__{cost:g}bp")


def run_jobs(jobs: list, a) -> None:
    d = Path(a.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs, a.cfg)) as ex:
        for i, (job, acc, un, st, hist, sec) in enumerate(ex.map(run_job, jobs, chunksize=1)):
            nm = job_name(job[0], job[1], job[2], job[6])
            for suffix, f in (("acc", acc), ("un", un), ("stats", st), ("hist", hist)):
                f.to_csv(d / f"{nm}_{suffix}.csv", index=False)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                ts.log(f"  [{i + 1}/{len(jobs)}] {nm} {sec:.0f}s · 남은 예상 {(len(jobs) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


# --------------------------------------------------------------------------- verdicts

def summarize(acc: pd.DataFrame, un: pd.DataFrame) -> pd.DataFrame:
    """Per line x generator x exit: the line's medians, the pseudo rules' medians (each pseudo's own
    median over its orders) and the (a) / (b) tests in both halves; with each pseudo rule judged the
    same way against the other pseudo rules (for the chance counts)."""
    keys = ["line", "gen", "exit"]
    med = acc.groupby(keys + ["kind", "k"]).median(numeric_only=True).reset_index()
    rows = []
    for (line, gen, x), g in med.groupby(keys, sort=False):
        r = g[g["kind"] == "rule"].iloc[0]
        p = g[g["kind"] == "pseudo"].set_index("k")
        u = un[(un["line"] == line) & (un["gen"] == gen) & (un["exit"] == x)]
        ur = u[u["kind"] == "rule"].iloc[0]
        up = u[u["kind"] == "pseudo"].set_index("k")
        row = {"line": line, "gen": gen, "exit": x, "trades": r["trades"], "signals": ur["signals"], "kept_share": ur["kept_share"],
               "per_day": ur["per_day"], "hold_median": ur["hold_median"], "stop_share": ur["stop_share"], "big_share": ur["big_share"],
               "cap_rank_trades": r["cap_rank_trades"], "pseudo_cap_rank_trades": float(p["cap_rank_trades"].median()),
               "sharpe": r["sharpe"], "pseudo_sharpe": float(p["sharpe"].median())}
        for c in ("m_sr", "m_g3", "m_g4", "m_T"):
            row[c] = ur.get(c, np.nan)
        for per in es.PERIODS:
            row[f"cagr_{per}"], row[f"mdd_{per}"] = r[f"cagr_{per}"], r[f"mdd_{per}"]
            row[f"pseudo_cagr_{per}"] = float(p[f"cagr_{per}"].median())
            row[f"pseudo_q05_{per}"], row[f"pseudo_q95_{per}"] = (float(p[f"cagr_{per}"].quantile(q)) for q in (0.05, 0.95))
            row[f"pct_in_pseudo_{per}"] = float((p[f"cagr_{per}"] < r[f"cagr_{per}"]).mean())
            row[f"contrib_{per}"] = ur[f"mean_contrib_{per}"]
            row[f"pseudo_contrib_{per}"] = float(up[f"mean_contrib_{per}"].median())
        row["a_halves"] = all(row[f"contrib_{h}"] > row[f"pseudo_contrib_{h}"] for h in es.HALVES)
        row["b_halves"] = all(row[f"cagr_{h}"] > row[f"pseudo_cagr_{h}"] for h in es.HALVES)
        row["b_below"] = all(row[f"cagr_{h}"] < row[f"pseudo_cagr_{h}"] for h in es.HALVES)
        rows.append(row)
    return pd.DataFrame(rows)


def pseudo_vs_pseudo(acc: pd.DataFrame, un: pd.DataFrame) -> pd.DataFrame:
    """Each pseudo rule against the median of the other pseudo rules of its line: the same tests."""
    keys = ["line", "gen", "exit"]
    med = acc[acc["kind"] == "pseudo"].groupby(keys + ["k"]).median(numeric_only=True).reset_index()
    up = un[un["kind"] == "pseudo"]
    out = []
    for (line, gen, x), g in med.groupby(keys, sort=False):
        u = up[(up["line"] == line) & (up["gen"] == gen) & (up["exit"] == x)].set_index("k")
        g = g.set_index("k")
        for k in g.index:
            o, uo = g.drop(index=k), u.drop(index=k)
            row = {"line": line, "gen": gen, "exit": x, "k": k}
            for h in es.HALVES:
                row[f"x_{h}"] = (g.loc[k, f"cagr_{h}"] - o[f"cagr_{h}"].median()) * 100
            row["a_halves"] = all(u.loc[k, f"mean_contrib_{h}"] > uo[f"mean_contrib_{h}"].median() for h in es.HALVES)
            row["b_halves"] = all(g.loc[k, f"cagr_{h}"] > o[f"cagr_{h}"].median() for h in es.HALVES)
            row["b_below"] = all(g.loc[k, f"cagr_{h}"] < o[f"cagr_{h}"].median() for h in es.HALVES)
            out.append(row)
    return pd.DataFrame(out)


def per_line(v: pd.DataFrame) -> pd.DataFrame:
    """Line x generator verdicts over the exits (>= MIN_EXITS of 3)."""
    g = v.groupby(["line", "gen"], sort=False)
    out = g.agg(a_n=("a_halves", "sum"), b_n=("b_halves", "sum"), below_n=("b_below", "sum"), trades_min=("trades", "min")).reset_index()
    out["meaningful_a"] = out["a_n"] >= MIN_EXITS
    out["meaningful_b"] = out["b_n"] >= MIN_EXITS
    out["below_b"] = out["below_n"] >= MIN_EXITS
    return out


def q95(n: int, p0: float) -> int:
    cdf = 0.0
    for c in range(n + 1):
        cdf += math.comb(n, c) * p0 ** c * (1 - p0) ** (n - c)
        if cdf >= 0.95:
            return c
    return n


def feature_info(lv: pd.DataFrame, gen: str = "E0") -> pd.DataFrame:
    """Per feature: favoured side passes (b) and the opposite side sits below the pseudo median in both
    halves (>= MIN_EXITS exits). K4 is shown only."""
    g = lv[lv["gen"] == gen].set_index("line")
    rows = []
    for f in FEATURES:
        fav, opp = g.loc[f"{f}f"], g.loc[f"{f}o"]
        rows.append({"feature": f, "label": FEATURES[f][0], "gen": gen, "hypothesis": f in HYPOTHESIS, "fav_b": bool(fav["meaningful_b"]),
                     "fav_a": bool(fav["meaningful_a"]), "opp_below": bool(opp["below_b"]),
                     "information": bool(fav["meaningful_b"] and opp["below_b"])})
    return pd.DataFrame(rows)


def chance(v: pd.DataFrame, pv: pd.DataFrame) -> dict:
    """Pass counts of the 29 lines per generator vs the pseudo-vs-pseudo rate (Binomial(29, p0) 95th
    percentile); the feature verdict vs pseudo pairs (favoured pseudo k with opposite pseudo k)."""
    lv = per_line(v)
    pl = pv.groupby(["line", "gen", "k"]).agg(a_n=("a_halves", "sum"), b_n=("b_halves", "sum"), below_n=("b_below", "sum")).reset_index()
    out = {}
    for gen, g in lv.groupby("gen", sort=False):
        p = pl[pl["gen"] == gen]
        n = len(g)
        row = {"n_lines": n, "excluded_few_trades": int((g["trades_min"] < pj.MIN_TRADES).sum())}
        ok = g[g["trades_min"] >= pj.MIN_TRADES]
        for t, col in (("a", "a_n"), ("b", "b_n")):
            p0 = float((p[col] >= MIN_EXITS).mean())
            obs = int((ok[col] >= MIN_EXITS).sum())
            row[t] = {"observed": obs, "p0": p0, "expected": p0 * len(ok), "q95": q95(len(ok), p0), "information": obs > q95(len(ok), p0),
                      "passing": sorted(ok.loc[ok[col] >= MIN_EXITS, "line"].tolist())}
        fi = feature_info(lv, gen)
        pp = []
        for f in FEATURES:
            fav = p[p["line"] == f"{f}f"].set_index("k")
            opp = p[p["line"] == f"{f}o"].set_index("k")
            for k in fav.index.intersection(opp.index):
                pp.append(fav.loc[k, "b_n"] >= MIN_EXITS and opp.loc[k, "below_n"] >= MIN_EXITS)
        p0f = float(np.mean(pp)) if pp else np.nan
        obs_f = int(fi["information"].sum())
        row["feature"] = {"observed": obs_f, "p0": p0f, "expected": p0f * len(FEATURES), "q95": q95(len(FEATURES), p0f),
                          "information": obs_f > q95(len(FEATURES), p0f), "features": fi.loc[fi["information"], "feature"].tolist()}
        out[gen] = row
    return out


def market_pass(v: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    spy = base.set_index("baseline").loc["SPY"]
    ok = pd.Series(True, index=v.index)
    for p in ("1993-2023", "2017-2023"):
        ok &= (v[f"cagr_{p}"] > spy[f"cagr_{p}"]) & (v[f"mdd_{p}"].abs() < abs(spy[f"mdd_{p}"]))
    return ok & (v["trades"] >= pj.MIN_TRADES)


def psr(sr: float, g3: float, g4: float, T: int, sr0: float) -> float:
    """Probabilistic Sharpe ratio of daily moments against a daily benchmark Sharpe sr0."""
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    return float(NormalDist().cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(den))) if den > 0 else float("nan")


def dsr(sr: float, g3: float, g4: float, T: int, n_trials: int, sr_var: float) -> float:
    """As metrics.deflated_sharpe, from the stored moments."""
    nd = NormalDist()
    g = 0.5772156649
    sr0 = math.sqrt(sr_var) * ((1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    return psr(sr, g3, g4, T, sr0)


# --------------------------------------------------------------------------- figures

def _order(ls: list[str]) -> list[str]:
    return [x for x in lines() if x in ls]


def fig_points(out: Path, v: pd.DataFrame, gens: list[str], name: str, x: str = FIG_EXIT) -> Path:
    """Each line's 10-stock CAGR minus its pseudo median (pp), over the pseudo 5-95% band."""
    plt = figures._plt()
    fig, axes = plt.subplots(len(gens), 2, figsize=(14, 4.2 * len(gens)), squeeze=False, sharey="row")
    for r_, gen in enumerate(gens):
        g = v[(v["gen"] == gen) & (v["exit"] == x)].set_index("line")
        ls = _order(list(g.index))
        xs = np.arange(len(ls))
        for c, h in enumerate(es.HALVES):
            ax = axes[r_, c]
            mid = g.loc[ls, f"pseudo_cagr_{h}"].to_numpy()
            lo = (g.loc[ls, f"pseudo_q05_{h}"].to_numpy() - mid) * 100
            hi = (g.loc[ls, f"pseudo_q95_{h}"].to_numpy() - mid) * 100
            ax.vlines(xs, lo, hi, color="#bbbbbb", lw=7, label="가짜 규칙 40개 (5–95%)")
            y = (g.loc[ls, f"cagr_{h}"].to_numpy() - mid) * 100
            col = [figures.COLORS[0] if l.endswith("f") else figures.COLORS[2] if l.startswith("K4") else figures.COLORS[1] for l in ls]
            ax.scatter(xs, y, c=col, s=28, zorder=3)
            ax.axhline(0, color="#555", lw=1)
            ax.set_xticks(xs, [l.replace("f", "+").replace("o", "−") if not l.startswith("K4") else l for l in ls], fontsize=7, rotation=90)
            ax.set_title(f"{gen}, {h.replace('-', '–')} (매도 {x})", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        axes[r_, 0].set_ylabel("10종목 CAGR − 가짜 중앙값 (%p)")
    fig.suptitle("특징 하나로 고른 줄(파랑 = 유리한 쪽 +, 주황 = 반대쪽 −, 초록 = 시총 구간) vs 같은 날 같은 수를 무작위로 고른 가짜 규칙", fontsize=11)
    return figures._save(fig, out / "figures" / name)


def fig_returns(out: Path, hist: pd.DataFrame, st: pd.DataFrame, gen: str = "E0", x: str = FIG_EXIT) -> Path:
    """Per-signal returns of each favoured line vs its pseudo rules (pooled), one panel per feature."""
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    ls = [f"{f}f" for f in FEATURES] + ["K4m"]
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(len(ls) / ncol)), ncol, figsize=(4.4 * ncol, 3.0 * int(np.ceil(len(ls) / ncol))), squeeze=False)
    for ax in axes.ravel()[len(ls):]:
        ax.set_visible(False)
    for ax, l in zip(axes.ravel(), ls):
        h = hist[(hist["line"] == l) & (hist["gen"] == gen) & (hist["exit"] == x) & (hist["period"] == "1993-2023")]
        if h.empty:
            ax.set_visible(False)
            continue
        pj._bar(ax, pj._density(h[h["kind"] == "pseudo"][hc].to_numpy().sum(axis=0)), color="#777", lw=1.3)
        pj._bar(ax, pj._density(h[h["kind"] == "rule"][hc].to_numpy().sum(axis=0)), color=figures.COLORS[0], lw=1.4)
        ax.axvline(0, color="#bbb", lw=0.8)
        s = st[(st["line"] == l) & (st["gen"] == gen) & (st["exit"] == x) & (st["kind"] == "rule")]
        if len(s) and pd.notna(s.iloc[0].get("st_diff")):
            s = s.iloc[0]
            ax.text(0.98, 0.97, f"평균 {s['mean_ret'] * 100:+.1f}% (가짜 {s['pseudo_mean_ret'] * 100:+.1f}%)\n"
                                f"차 95% CI {s['st_ci_lo'] * 100:+.1f} – {s['st_ci_hi'] * 100:+.1f}%p\np = {s['st_mw_p']:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_title(f"{l[:-1] if l.endswith('f') else l}: {line_label(l)}", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"신호 하나의 수익률 (%): 유리한 쪽으로 고른 신호(파랑) vs 같은 날 같은 수의 가짜(회색), {gen}, 매도 {x}, 1993–2023", fontsize=11)
    return figures._save(fig, out / "figures" / f"returns_favoured_{gen}.png")


def fig_excess(out: Path, v: pd.DataFrame, pv: pd.DataFrame) -> Path:
    """10-stock CAGR minus the pseudo median (pp): all lines x generators x exits vs pseudo-vs-pseudo."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, h in zip(axes, es.HALVES):
        a = ((v[f"cagr_{h}"] - v[f"pseudo_cagr_{h}"]) * 100).to_numpy()
        b = pv[f"x_{h}"].to_numpy()
        edges = np.linspace(np.floor(min(a.min(), b.min())) - 0.5, np.ceil(max(a.max(), b.max())) + 0.5, 41)
        ax.hist(b, edges, density=True, color="#bbbbbb", label="가짜 규칙 vs 나머지 가짜")
        ax.hist(a, edges, density=True, histtype="step", lw=1.8, color=figures.COLORS[0], label="선택 규칙")
        q = np.percentile(b, 95)
        ax.axvline(q, color="#555", ls="--", lw=1)
        ax.axvline(0, color="#999", lw=0.8)
        _, p = pj.mann_whitney(a, b)
        ax.text(0.02, 0.97, f"규칙 {len(a)}칸: 평균 {a.mean():+.2f}%p (가짜 {b.mean():+.2f}%p)\n가짜 95% 선 넘음 {np.mean(a > q) * 100:.0f}% (우연이면 5%)"
                            f"\n순위 검정 p = {p:.3f}", transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_title(h.replace("-", "–"), fontsize=10)
        ax.set_xlabel("10종목 CAGR − 같은 줄 가짜 규칙 중앙값 (%p)")
        ax.grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("선택 규칙 29줄 × 생성기 3 × 매도 3의 초과 수익 분포 vs 가짜 규칙끼리 (밀도)", fontsize=11)
    return figures._save(fig, out / "figures" / "excess_vs_pseudo.png")


def fig_cap(out: Path, v: pd.DataFrame, x: str = FIG_EXIT) -> Path:
    """⑤ 10-stock CAGR against the median cap rank of the filled trades (size effect)."""
    plt = figures._plt()
    fig, ax = plt.subplots(figsize=(9, 6))
    g = v[v["exit"] == x]
    ax.scatter(g["pseudo_cap_rank_trades"], g["pseudo_cagr_1993-2023"] * 100, s=14, color="#bbbbbb", label="가짜 규칙 중앙값")
    for i, (gen, gg) in enumerate(g.groupby("gen", sort=False)):
        ax.scatter(gg["cap_rank_trades"], gg["cagr_1993-2023"] * 100, s=22, color=figures.COLORS[i % len(figures.COLORS)], label=f"선택 규칙 ({gen})")
    ax.set_xlabel("채운 거래의 시총 순위 중앙값 (작을수록 큰 회사)")
    ax.set_ylabel("10종목 CAGR 1993–2023 (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"CAGR 대 보유 종목 크기 (매도 {x})", fontsize=11)
    return figures._save(fig, out / "figures" / "cagr_vs_cap_rank.png")


def fig_candidates(out: Path, hist: pd.DataFrame, picks: list[tuple]) -> Path:
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.8 * len(picks)), squeeze=False, sharex=True)
    for ax, (l, gen, x) in zip(axes[:, 0], picks):
        h = hist[(hist["line"] == l) & (hist["gen"] == gen) & (hist["exit"] == x) & (hist["period"] == "1993-2023")]
        pj._bar(ax, pj._density(h[h["kind"] == "pseudo"][hc].to_numpy().sum(axis=0)), color="#777", lw=1.3, label="가짜 규칙")
        pj._bar(ax, pj._density(h[h["kind"] == "rule"][hc].to_numpy().sum(axis=0)), color=figures.COLORS[0], lw=1.4, label="후보")
        ax.set_title(f"{line_label(l)}, {gen}, 매도 {x}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("후보의 신호 하나 수익률 vs 가짜 규칙", fontsize=11)
    return figures._save(fig, out / "figures" / "candidates_returns.png")


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "r0x1", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-workers", type=int, default=4, help="특징 계산 작업 수 (작업당 메모리 약 10GB)")
    ap.add_argument("--dry", action="store_true", help="K2·K5·K12·시총, 생성기 E0(씨앗 2)·V3, 순서 10, 가짜 4 × 순서 2")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.dirs = {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events), "r0x1": Path(a.r0x1)}
    global FEATURES
    gens = {g: list(s) for g, s in GENERATORS.items()}
    n_orders, n_pseudo, p_orders = N_ORDERS, N_PSEUDO, PSEUDO_ORDERS
    if a.dry:
        FEATURES = {k: FEATURES[k] for k in ("K2", "K5", "K12")}
        gens = {"E0": gens["E0"][:2], "V3": ["V3"]}
        n_orders, n_pseudo, p_orders = 10, 4, 2
    seeds = [int(s[3:]) for s in gens["E0"]]
    a.cfg = {"bins": pj.BINS, "seeds": pj.E0_SEEDS, "base": str(out / "base.parquet")}
    ts.log(f"1단계: 생성기 신호에 특징 {len(FEATURES)}개 + 시총 순위")
    base = build_base(a, gens, list(FEATURES))
    ts.log(f"  신호 {len(base):,}개 (" + ", ".join(f"{g} {int((base['gen'] == g).sum()):,}" for g in gens) + ")")
    jobs = [(l, g, x, n_orders if g != "E0" else n_orders - n_orders % len(gens[g]), n_pseudo,
             p_orders if g != "E0" else max(len(gens[g]), p_orders - p_orders % len(gens[g])), pj.COST_BPS, gens[g])
            for l in lines() for g in gens for x in EXITS]
    if a.dry:
        jobs = [(l, g, x, 10, 4, 2, pj.COST_BPS, gens[g]) for l in lines() for g in gens for x in EXITS]
    jdir = out / "jobs"
    todo = [j for j in jobs if not (jdir / f"{job_name(j[0], j[1], j[2], j[6])}_acc.csv").exists()]
    ts.log(f"2단계: 작업 {len(todo)}개 남음 (줄 {len(lines())} × 생성기 {len(gens)} × 매도 {len(EXITS)}), 가짜 {n_pseudo}개씩")
    run_jobs(todo, a)
    read = lambda s: pd.concat([pd.read_csv(f) for f in sorted(jdir.glob(f"*_{s}.csv")) if "bp_" not in f.name], ignore_index=True)  # noqa: E731
    acc, un, st, hist = read("acc"), read("un"), read("stats"), read("hist")
    pj._init(a.panel, a.eligible, a.dirs, a.cfg)
    bl = es.baselines(pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"])
    bl.to_csv(out / "baselines.csv", index=False)
    v = summarize(acc, un)
    pv = pseudo_vs_pseudo(acc, un)
    v["market_pass"] = market_pass(v, bl)
    v["candidate_20bp"] = v["market_pass"] & (v.groupby(["line", "gen"])["market_pass"].transform("sum") >= MIN_EXITS)
    v["candidate_40bp"] = False
    cand = v[v["candidate_20bp"]]
    if len(cand):  # candidate rows again at 40 bp (the line only)
        run_jobs([(r.line, r.gen, r.exit, n_orders, 0, 0, pj.CHECK_COST_BPS, gens[r.gen]) for r in cand.itertuples()
                  if not (jdir / f"{job_name(r.line, r.gen, r.exit, pj.CHECK_COST_BPS)}_acc.csv").exists()], a)
        spy = bl.set_index("baseline").loc["SPY"]
        for i, r in cand.iterrows():
            g = pd.read_csv(jdir / f"{job_name(r['line'], r['gen'], r['exit'], pj.CHECK_COST_BPS)}_acc.csv")
            g = g[g["kind"] == "rule"]
            v.loc[i, "candidate_40bp"] = all(g[f"cagr_{p}"].median() > spy[f"cagr_{p}"] and abs(g[f"mdd_{p}"].median()) < abs(spy[f"mdd_{p}"])
                                             for p in ("1993-2023", "2017-2023"))
    v["candidate"] = v["candidate_40bp"] & (v.groupby(["line", "gen"])["candidate_40bp"].transform("sum") >= MIN_EXITS)
    sr_daily = v["m_sr"].dropna().to_numpy()
    sr_var = float(np.var(sr_daily)) if len(sr_daily) else np.nan
    spy_r = pd.Series(pj._W["px"].spy_c[pj._W["w0"]:pj._W["w0"] + pj._W["n"]]).pct_change().dropna()
    spy_sr = float(spy_r.mean() / spy_r.std())
    for i, r in v[v["candidate_20bp"]].iterrows():
        if pd.notna(r["m_sr"]):
            v.loc[i, "dsr"] = dsr(r["m_sr"], r["m_g3"], r["m_g4"], int(r["m_T"]), N_TRIALS_BEFORE + N_NEW, sr_var)
            v.loc[i, "psr_spy"] = psr(r["m_sr"], r["m_g3"], r["m_g4"], int(r["m_T"]), spy_sr)
            v.loc[i, "psr_pseudo"] = psr(r["m_sr"], r["m_g3"], r["m_g4"], int(r["m_T"]), r["pseudo_sharpe"] / np.sqrt(252))
    v["label"] = v["line"].map(line_label)
    v.to_csv(out / "verdicts.csv", index=False)
    lv = per_line(v)
    lv.to_csv(out / "lines.csv", index=False)
    pv.to_csv(out / "pseudo_vs_pseudo.csv", index=False)
    st.to_csv(out / "stats.csv", index=False)
    hist.to_csv(out / "hist.csv", index=False)
    acc.groupby(["line", "gen", "exit", "kind", "k"]).median(numeric_only=True).to_csv(out / "accounts.csv")
    ch = {"by_gen": chance(v, pv), "feature_info": {g: feature_info(lv, g).to_dict("records") for g in gens},
          "stats": {"rule_p_lt_05": float((st.loc[st["kind"] == "rule", "st_mw_p"] < 0.05).mean()),
                    "pseudo_p_lt_05": float((st.loc[st["kind"] == "pseudo", "st_mw_p"] < 0.05).mean()),
                    "rule_ci_above_0": float((st.loc[st["kind"] == "rule", "st_ci_lo"] > 0).mean()),
                    "rule_ci_below_0": float((st.loc[st["kind"] == "rule", "st_ci_hi"] < 0).mean())},
          "candidates_20bp": v.loc[v["candidate_20bp"], ["line", "gen", "exit"]].to_dict("records"),
          "candidates": v.loc[v["candidate"], ["line", "gen", "exit"]].to_dict("records"),
          "trials": {"before": N_TRIALS_BEFORE, "new": N_NEW}, "sr_var_daily": sr_var}
    (out / "chance.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if figures.available():
        fig_points(out, v, ["E0"], "points_E0.png")
        others = [g for g in gens if g != "E0"]
        if others:
            fig_points(out, v, others, "points_V3_E2.png")
        fig_returns(out, hist, st)
        fig_excess(out, v, pv)
        fig_cap(out, v)
        picks = [tuple(r) for r in v.loc[v["candidate"], ["line", "gen", "exit"]].itertuples(index=False)][:6]
        if picks:
            fig_candidates(out, hist, picks)
    e0 = ch["by_gen"].get("E0", {})
    ts.log(f"완료 · E0 줄 통과 (a) {e0.get('a', {}).get('observed')} / 95% 분위 {e0.get('a', {}).get('q95')} · (b) "
           f"{e0.get('b', {}).get('observed')} / {e0.get('b', {}).get('q95')} · 특징 정보 {e0.get('feature', {}).get('features')} · "
           f"후보 {len(ch['candidates'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
