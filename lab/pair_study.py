"""Stage J / J3: buy rules in pairs or threes, against random buys of the same frequency (DECISIONS 88, 89).

    python -m lab.pair_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                             --out DIR [--size 2|3] [--workers N] [--dry]
    python -m lab.pair_study --redraw DIR [--size 2|3]      # figures again from a finished run's tables

A combination of rules buys at the next open after the latest first-day signal when every rule of
it fired on the same stock within PAIR_WINDOW trading days (any order, same day included). The latest
signal's stored table row carries the exit, so combinations are built from the saved per-signal
tables. Exits: H1 k 2, H1 k 3 (25% trailing or ATR stop) and X1 (25% trailing only, fixed 10% weight).
Baseline: random buys E0 thinned to the nearest frequency bin; chance count from thinned random
pseudo-sets. Per-signal returns (20 bp, no stock limit) are compared with the same-frequency E0 by
histograms and tests (J3). Output (aggregates only): sets/*.csv, accounts_quantiles.csv,
unlimited.csv, verdicts.csv, stats.csv, hist.csv, chance.json, baselines.csv, figures/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import entry_study as es, event_study as ev, figures, pick_study as ph, stage_c_study as sc, tail_study as ts
from .crsp_study import SAMPLES, load_panel_with_cash
from .engine import CANCELLED

# ---- registered (DECISIONS 88): fixed before any result
RULES = [f"E{i}" for i in range(1, 10)] + list(ev.EVENTS)
PAIR_WINDOW = 10
EXITS_J = {"H1k2": ("H1", 1), "H1k3": ("H1", 2), "X1": ("X1", None)}
BINS = [0.0002, 0.0005, 0.001, 0.002, 0.004]  # share of eligible stock-days
E0_SEEDS = [1, 2, 3, 4, 5]
N_ORDERS = 100
N_PLACEBO_PER_BIN, PLACEBO_ORDERS = 40, 20
MIN_TRADES = 30
STOCK_DAYS = 15_680_749  # hygiene-eligible stock-days 1993-02-01 - 2023-12-31 (Stage I)
COST_BPS, CHECK_COST_BPS = 20.0, 40.0
N_TRIALS_BEFORE = es.N_TRIALS_BEFORE + es.N_NEW  # Stages B-I (approx.)
N_NEW = 315  # 105 pairs x 3 exits (singles, E0 and pseudo-pairs not counted)
SEED = 88
# ---- registered (DECISIONS 89, Stage J3): what differs from 88
BINS_J3 = [0.00002, 0.00005, 0.0001] + BINS  # down to about 0.04 signals a day
N_PLACEBO_J3 = 100
N_NEW_J3 = 1_365  # 455 triples x 3 exits
RET_EDGES = np.r_[-np.inf, np.arange(-60, 205, 5) / 100, np.inf]  # per-signal return histogram (fraction)
HIST_PERIODS = ("1993-2023",) + es.HALVES
N_BOOT = 1_000
MIN_STAT_SIGNALS = 10
TOP_EXIT = "H1k3"
SIZES = {2: "pair", 3: "triple"}
_C_CANON = dict(sc.ENTRY_CANON)
_W: dict = {}


def combos(k: int) -> list[tuple[str, ...]]:
    return list(itertools.combinations(RULES, k))


def pairs() -> list[tuple[str, str]]:
    return combos(2)


def set_name(s: tuple) -> str:
    return "__".join(map(str, s))


def nearest_bin(freq: float) -> float:
    return min(BINS, key=lambda b: abs(np.log(max(freq, 1e-9)) - np.log(b)))


# --------------------------------------------------------------------------- statistics

def mann_whitney(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(P(X > Y) + P(X = Y) / 2, two-sided p) of the Mann-Whitney U test, normal approximation with the
    tie correction and no continuity correction."""
    n1, n2 = len(x), len(y)
    a = np.concatenate([x, y])
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    start = np.r_[0, np.flatnonzero(np.diff(sa)) + 1]
    counts = np.diff(np.r_[start, len(sa)])
    ranks = np.empty(len(a))
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - float((counts.astype(float) ** 3 - counts).sum()) / (n * (n - 1)))
    z = (u - n1 * n2 / 2.0) / math.sqrt(var) if var > 0 else 0.0
    return u / (n1 * n2), math.erfc(abs(z) / math.sqrt(2.0))


def ks_stat(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two empirical CDFs."""
    xs, ys = np.sort(x), np.sort(y)
    grid = np.concatenate([xs, ys])
    return float(np.abs(np.searchsorted(xs, grid, side="right") / len(xs) - np.searchsorted(ys, grid, side="right") / len(ys)).max())


def boot_means(r: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> np.ndarray:
    if len(r) < MIN_STAT_SIGNALS:
        return np.full(n_boot, np.nan)
    return np.array([r[rng.integers(0, len(r), len(r))].mean() for _ in range(n_boot)])


def ret_stats(r: np.ndarray, ref: np.ndarray, ref_boot: np.ndarray, rng: np.random.Generator) -> dict:
    """Per-signal returns of a set vs its same-frequency E0: mean difference with a bootstrap 95% CI
    (both samples resampled), Mann-Whitney U and KS. Empty below MIN_STAT_SIGNALS signals."""
    if len(r) < MIN_STAT_SIGNALS or len(ref) < MIN_STAT_SIGNALS:
        return {}
    diff = boot_means(r, rng) - ref_boot
    auc, p = mann_whitney(r, ref)
    return {"st_n": len(r), "st_ref_n": len(ref), "st_mean": float(r.mean()), "st_ref_mean": float(ref.mean()),
            "st_diff": float(r.mean() - ref.mean()), "st_ci_lo": float(np.percentile(diff, 2.5)),
            "st_ci_hi": float(np.percentile(diff, 97.5)), "st_mw_auc": auc, "st_mw_p": p, "st_ks": ks_stat(r, ref)}


# --------------------------------------------------------------------------- E0 x X1 tables

def use_random_x1_menu(dry: bool = False) -> None:
    sc.ENTRIES.clear()
    sc.ENTRIES["R0"] = ("g_random", list(E0_SEEDS if not dry else E0_SEEDS[:2]))
    sc.EXITS.clear()
    sc.EXITS["X1"] = ("trailing_pct", [25])
    sc.SAMPLES = {"full": SAMPLES["full"]}


def _r_init(panel_dir: str, eligible: str, dry: bool):
    use_random_x1_menu(dry)
    sc._setup(load_panel_with_cash(Path(panel_dir)), pd.read_parquet(eligible))


# --------------------------------------------------------------------------- signal sets

def table_spec(e: str, exit_key: str) -> tuple[Path, int, int]:
    """(signal table, entry cell, exit cell) of rule e (or E0 seed 'R0s<k>') under a Stage J exit."""
    d = _W["dirs"]
    x, yi = EXITS_J[exit_key]
    if e.startswith("R0s"):
        seed_i = E0_SEEDS.index(int(e[3:]))
        f = d["stage_i"] / "r0" / "signals" / "full" / "R0__H1_signals.parquet" if x == "H1" else d["r0x1"] / "signals" / "full" / "R0__X1_signals.parquet"
        return f, seed_i, (yi if x == "H1" else 0)
    if x == "H1":
        return d["stage_h"] / "h1" / "signals" / "full" / f"{e}__H1_signals.parquet", 0, yi
    root = d["v2"] if e.startswith("E") else d["events"]
    return root / "signals" / "full" / f"{e}__X1_signals.parquet", _C_CANON.get(e, 1), 1


def rule_table(e: str, exit_key: str) -> pd.DataFrame:
    """Filled first-day signals of rule e (middle value) with their exit under exit_key. Another
    study's exits can be served by setting _W['table_of'] (same signature as table_spec)."""
    key = (e, exit_key)
    if key in _W["tables"]:
        return _W["tables"][key]
    f, xi, yi = _W.get("table_of", table_spec)(e, exit_key)
    s = pd.read_parquet(f, columns=["xi", "yi", "symbol", "signal_date", "entry_date", "exit_signal_date", "return",
                                    "holding_days", "closed", "exit_reason"])
    s = s[(s["xi"] == xi) & (s["yi"] == yi) & (s["exit_reason"] != CANCELLED)].drop(columns=["xi", "yi"])
    s = s.drop_duplicates(["symbol", "signal_date"]).reset_index(drop=True)
    s["si"] = _W["px"].dates.get_indexer(pd.DatetimeIndex(s["signal_date"]))
    if "table_post" in _W:  # e.g. a partial take-profit added to the stored exits (lab/exit_study.py)
        s = _W["table_post"](s, e, exit_key)
    _W["tables"][key] = s
    return s


def combo_rows(rules: tuple, exit_key: str) -> pd.DataFrame:
    """Rows of a combination signal: days d of one rule on which every other rule of the combination
    has a signal in [d - W, d] on the same stock (so the earliest and latest are at most W apart),
    each row taken from its own (latest) rule's table; the same stock-day gives one row."""
    tabs = [rule_table(r, exit_key) for r in rules]
    out = []
    for i, t in enumerate(tabs):
        m = t.sort_values("si")
        for j, o in enumerate(tabs):
            if j == i or m.empty:
                continue
            early = o[["symbol", "si"]].rename(columns={"si": "si_early"}).sort_values("si_early")
            m = pd.merge_asof(m, early, left_on="si", right_on="si_early", by="symbol", direction="backward",
                              tolerance=PAIR_WINDOW)
            m = m[m["si_early"].notna()].drop(columns="si_early")
        out.append(m)
    return pd.concat(out, ignore_index=True).drop_duplicates(["symbol", "si"]).reset_index(drop=True)


def pair_rows(a: str, b: str, exit_key: str) -> pd.DataFrame:
    """Rows of the pair signal: days of A with a B signal in [d - W, d] and days of B with an A signal
    in [d - W, d], each taken from its own rule's table (same stock-day -> one row)."""
    return combo_rows((a, b), exit_key)


def thinned(seed: int, bin_: float, draw: int, exit_key: str) -> pd.DataFrame:
    """Random buys of seed `seed` thinned to share `bin_` of stock-days (E0 is 1%); the same stock-days
    for every exit (draw 0..: independent thinnings)."""
    base = rule_table(f"R0s{seed}", "H1k2")[["symbol", "si"]].sort_values(["si", "symbol"]).reset_index(drop=True)
    u = np.random.default_rng([SEED, seed, int(bin_ * 1e6), draw]).random(len(base))
    keep = base[u < bin_ / 0.01]
    t = rule_table(f"R0s{seed}", exit_key)
    return t.merge(keep, on=["symbol", "si"], how="inner")


def signals_for(spec: tuple, exit_key: str) -> pd.DataFrame:
    kind = spec[0]
    if kind == "single":
        return rule_table(spec[1], exit_key)
    if kind in ("pair", "triple"):
        return combo_rows(spec[1:], exit_key)
    _, seed, bin_, draw = spec  # "e0" (draw 0) or "placebo" (draw >= 1)
    return thinned(seed, bin_, draw, exit_key)


# --------------------------------------------------------------------------- per set: accounts

def _init(panel_dir: str, eligible: str, dirs: dict, cfg: dict | None = None):
    global BINS, E0_SEEDS
    if cfg:  # the run's bins and seeds (a worker imports the module defaults)
        BINS, E0_SEEDS = list(cfg["bins"]), list(cfg["seeds"])
    es._init(panel_dir, eligible, "", dirs)
    _W.update({k: es._W[k] for k in ("px", "w0", "n", "wdates", "cash", "O", "LC")})
    es._W.pop("dv", None)  # dollar-volume order is not used here
    _W["dirs"] = dirs
    _W["tables"] = {}
    _W["refs"] = {}


def prepared(spec: tuple, exit_key: str) -> pd.DataFrame:
    """The set's filled signals inside the window with their weight, contribution and return."""
    x, ki = EXITS_J[exit_key]
    px, w0, n = _W["px"], _W["w0"], _W["n"]
    s = signals_for(spec, exit_key)
    m = px.measures(s, es.END)
    if x == "H1":
        w = ph.weights(px, m, ph.K_VALUES[ki])
    else:  # no stop: fixed 10% weight
        w = pd.DataFrame({"w": np.full(len(m), ph.CAP), "capped": True})
        w["contrib"] = w["w"] * m["ret"].to_numpy()
    d = pd.concat([m[["si", "ei", "sj", "sale", "closed", "x_spy", "ret"]], w[["w", "contrib"]], s[["signal_date"]]], axis=1)
    ok = (d["ei"].to_numpy() >= w0) & (d["ei"].to_numpy() < w0 + n) & np.isfinite(d["w"].to_numpy())
    return d[ok].reset_index(drop=True)


def e0_reference(bin_: float, exit_key: str, without: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-signal returns of E0 at bin_ pooled over the seeds and their bootstrap means. A pseudo-set
    made from seed s is compared without seed s (its own thinning base would overlap it)."""
    key = (bin_, exit_key, without)
    if key not in _W["refs"]:
        r = np.concatenate([prepared(("e0", s, bin_, 0), exit_key)["ret"].to_numpy() for s in E0_SEEDS if s != without])
        r = r[np.isfinite(r)]
        rng = np.random.default_rng([SEED, 3, int(round(bin_ * 1e7)), list(EXITS_J).index(exit_key), without or 0])
        _W["refs"][key] = (r, boot_means(r, rng))
    return _W["refs"][key]


def run_set(job):
    spec, n_orders, cost_bps = job
    w0, n, wdates = _W["w0"], _W["n"], _W["wdates"]
    accs, uns, hists = [], [], []
    for exit_key in EXITS_J:
        d = prepared(spec, exit_key)
        ei, sale = d["ei"].to_numpy() - w0, d["sale"].to_numpy() - w0
        closed = d["closed"].to_numpy(dtype=bool) & (sale < n)
        rng = np.random.default_rng([SEED, zlib.crc32(f"{set_name(spec)}|{exit_key}|{cost_bps}".encode())])
        for o in range(n_orders):
            eq, tr = ph.replay_risk(ei - 1, ei, d["sj"].to_numpy(), sale, closed, d["w"].to_numpy(), rng.random(len(d)),
                                    _W["O"], _W["LC"], _W["cash"], cost_bps / 10_000)
            accs.append({"set": set_name(spec), "kind": spec[0], "exit": exit_key, "order": o, "trades": len(tr),
                         **es.period_stats(pd.Series(eq, index=wdates))})
        yr = pd.DatetimeIndex(d["signal_date"]).year.to_numpy()
        ret = d["ret"].to_numpy(dtype=float)
        un = {"set": set_name(spec), "kind": spec[0], "exit": exit_key, "signals": len(d), "freq": len(d) / STOCK_DAYS,
              "days": n}
        for p, (a, b) in es.PERIODS.items():
            mm = (yr >= a) & (yr <= b)
            r = ret[mm & np.isfinite(ret)]
            un[f"n_{p}"] = int(mm.sum())
            un[f"mean_contrib_{p}"] = float(d["contrib"].to_numpy()[mm].mean()) if mm.any() else np.nan
            un[f"mean_x_spy_{p}"] = float(d["x_spy"].to_numpy()[mm].mean()) if mm.any() else np.nan
            un[f"mean_ret_{p}"] = float(r.mean()) if len(r) else np.nan
            un[f"median_ret_{p}"] = float(np.median(r)) if len(r) else np.nan
            un[f"win_{p}"] = float((r > 0).mean()) if len(r) else np.nan
            un[f"big50_{p}"] = float((r >= 0.5).mean()) if len(r) else np.nan
            un[f"loss20_{p}"] = float((r <= -0.2).mean()) if len(r) else np.nan
            if p in HIST_PERIODS:
                hists.append({"set": set_name(spec), "kind": spec[0], "exit": exit_key, "period": p,
                              **{f"h{i}": int(c) for i, c in enumerate(np.histogram(r, RET_EDGES)[0])}})
        if spec[0] != "e0" and cost_bps == COST_BPS:
            b = nearest_bin(len(d) / STOCK_DAYS)
            ref, ref_boot = e0_reference(b, exit_key, spec[1] if spec[0] == "placebo" else None)
            rng = np.random.default_rng([SEED, zlib.crc32(f"{set_name(spec)}|{exit_key}|boot".encode())])
            un.update({"st_bin": b, **ret_stats(ret[np.isfinite(ret)], ref, ref_boot, rng)})
        uns.append(un)
    return spec, pd.DataFrame(accs), pd.DataFrame(uns), pd.DataFrame(hists)


# --------------------------------------------------------------------------- verdicts

def judge(acc: pd.DataFrame, un: pd.DataFrame, ref_acc: pd.DataFrame, ref_un: pd.DataFrame) -> pd.DataFrame:
    """Per set and exit: medians vs the reference (same-frequency E0) and the (a) / (b) half tests."""
    rows = []
    med = acc.groupby(["set", "exit"]).median(numeric_only=True)
    for (st, x), r in med.iterrows():
        u = un[(un["set"] == st) & (un["exit"] == x)].iloc[0]
        b = nearest_bin(float(u["freq"]))
        ra = ref_acc[(ref_acc["bin"] == b) & (ref_acc["exit"] == x)]
        ru = ref_un[(ref_un["bin"] == b) & (ref_un["exit"] == x)]
        row = {"set": st, "exit": x, "freq": u["freq"], "signals": u["signals"], "days": u["days"],
               "per_day": u["signals"] / u["days"], "bin": b, "trades": r["trades"]}
        for p in es.PERIODS:
            row[f"cagr_{p}"], row[f"mdd_{p}"] = r[f"cagr_{p}"], r[f"mdd_{p}"]
            row[f"e0_cagr_{p}"] = float(ra[f"cagr_{p}"].median())
            row[f"pct_in_e0_{p}"] = float((ra[f"cagr_{p}"] < r[f"cagr_{p}"]).mean())
            row[f"contrib_{p}"] = float(u[f"mean_contrib_{p}"])
            # E0 mean contribution pooled over the seeds (weighted by the period's signal counts)
            row[f"e0_contrib_{p}"] = float((ru[f"mean_contrib_{p}"] * ru[f"n_{p}"]).sum() / ru[f"n_{p}"].sum())
            row[f"x_spy_{p}"] = float(u[f"mean_x_spy_{p}"])
        row["a_halves"] = all(row[f"contrib_{h}"] > row[f"e0_contrib_{h}"] for h in es.HALVES)
        row["b_halves"] = all(row[f"cagr_{h}"] > row[f"e0_cagr_{h}"] for h in es.HALVES)
        rows.append(row)
    v = pd.DataFrame(rows)
    g = v.groupby("set")
    v["enough_trades"] = g["trades"].transform("min") >= MIN_TRADES
    v["meaningful_a"] = g["a_halves"].transform("sum") >= 2
    v["meaningful_b"] = g["b_halves"].transform("sum") >= 2
    return v


def candidates(v: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    spy = base.set_index("baseline").loc["SPY"]
    ok = pd.Series(True, index=v.index)
    for p in ("1993-2023", "2017-2023"):
        ok &= (v[f"cagr_{p}"] > spy[f"cagr_{p}"]) & (v[f"mdd_{p}"].abs() < abs(spy[f"mdd_{p}"]))
    ok &= v["trades"] >= MIN_TRADES
    return ok & (ok.groupby(v["set"]).transform("sum") >= 2)


def chance(vp: pd.DataFrame, n_pairs: int, passed: dict) -> dict:
    """Pass rate of the random pseudo-sets and the 95th percentile of the pass count expected by chance."""
    out = {"n_pairs_judged": n_pairs, "observed": passed}
    one = vp.drop_duplicates("set")
    for k in ("meaningful_a", "meaningful_b"):
        p0 = float(one[k].mean())
        cdf, q95 = 0.0, n_pairs
        for x in range(n_pairs + 1):
            cdf += math.comb(n_pairs, x) * p0 ** x * (1 - p0) ** (n_pairs - x)
            if cdf >= 0.95:
                q95 = x
                break
        out[k] = {"p0": p0, "expected": p0 * n_pairs, "q95": q95, "information": passed[k] > q95}
        if "bin" in one:  # information only: the pass rate within each frequency bin
            out[k]["p0_by_bin"] = {f"{b:g}": float(g[k].mean()) for b, g in one.groupby("bin")}
    return out


def stacking(v: pd.DataFrame) -> pd.DataFrame:
    """Each pair / triple vs the best of its singles and (triples) of its pairs, same exit, median CAGR."""
    cagr = {(r["set"], r["exit"]): r for _, r in v[v["kind"].isin(["single", "pair"])].iterrows()}
    for p in ("1993-2023",) + es.HALVES:
        def best(r, part, p=p):
            if r["kind"] not in ("pair", "triple") or (part == "pair" and r["kind"] != "triple"):
                return np.nan
            rules = r["set"].split("__")[1:]
            k = 1 if part == "single" else 2
            vals = [cagr[(f"{part}__" + "__".join(c), r["exit"])][f"cagr_{p}"] for c in itertools.combinations(rules, k)
                    if (f"{part}__" + "__".join(c), r["exit"]) in cagr]
            return max(vals) if vals else np.nan
        v[f"best_single_cagr_{p}"] = v.apply(best, axis=1, part="single")
        v[f"best_pair_cagr_{p}"] = v.apply(best, axis=1, part="pair")
    for part in ("single", "pair"):
        v[f"best_{part}_cagr"] = v[f"best_{part}_cagr_1993-2023"]
        v[f"stacks_{part}"] = v["cagr_1993-2023"] > v[f"best_{part}_cagr"]
        v[f"stacks_{part}_halves"] = np.logical_and.reduce([v[f"cagr_{h}"] > v[f"best_{part}_cagr_{h}"] for h in es.HALVES])
    # Stage J names (pairs vs their better single)
    v["stacks"], v["stacks_halves"] = v["stacks_single"], v["stacks_single_halves"]
    return v


def stats_summary(st: pd.DataFrame) -> dict:
    """Share of sets whose per-signal returns differ from the same-frequency E0 at p < 0.05, by kind."""
    out = {}
    for kind, g in st[st["st_n"].notna()].groupby("kind"):
        rows = {"all": g, **{x: gx for x, gx in g.groupby("exit")}}
        out[kind] = {k: {"n": int(len(gx)), "p_lt_05": float((gx["st_mw_p"] < 0.05).mean()),
                         "p_lt_05_higher": float(((gx["st_mw_p"] < 0.05) & (gx["st_mw_auc"] > 0.5)).mean()),
                         "ci_above_0": float((gx["st_ci_lo"] > 0).mean()), "ci_below_0": float((gx["st_ci_hi"] < 0).mean()),
                         "median_diff": float(gx["st_diff"].median()), "median_auc": float(gx["st_mw_auc"].median()),
                         "median_ks": float(gx["st_ks"].median())} for k, gx in rows.items()}
    return out


# --------------------------------------------------------------------------- figures

def _excess(g: pd.DataFrame, h: str) -> np.ndarray:
    return ((g[f"cagr_{h}"] - g[f"e0_cagr_{h}"]) * 100).to_numpy(dtype=float)


def fig_excess(out: Path, v: pd.DataFrame, kind: str, label: str, name: str) -> Path:
    """10-stock CAGR in excess of the same-frequency E0 median (pp): the sets vs the pseudo-sets."""
    plt = figures._plt()
    sets = v[(v["kind"] == kind) & v["enough_trades"]]
    pl = v[v["kind"] == "placebo"]
    fig, axes = plt.subplots(len(EXITS_J), 2, figsize=(12, 3.2 * len(EXITS_J)), sharex=True)
    allx = np.concatenate([_excess(sets, h) for h in es.HALVES] + [_excess(pl, h) for h in es.HALVES])
    edges = np.linspace(np.floor(np.nanmin(allx)) - 0.5, np.ceil(np.nanmax(allx)) + 0.5, 41)
    for r_, x in enumerate(EXITS_J):
        for c, h in enumerate(es.HALVES):
            ax = axes[r_, c]
            a, b = _excess(sets[sets["exit"] == x], h), _excess(pl[pl["exit"] == x], h)
            ax.hist(b, edges, density=True, color="#bbbbbb", label="가짜 묶음 (우연 분포)")
            ax.hist(a, edges, density=True, histtype="step", lw=1.8, color=figures.COLORS[0], label=label)
            q = np.percentile(b, 95)
            ax.axvline(q, color="#555", ls="--", lw=1)
            ax.axvline(0, color="#999", lw=0.8)
            auc, p = mann_whitney(a, b) if len(a) and len(b) else (np.nan, np.nan)
            ax.text(0.02, 0.97, f"{label} {len(a)}개: 평균 {a.mean():+.1f}%p (가짜 {b.mean():+.1f}%p)\n"
                                f"가짜 95% 선(점선) 넘음 {np.mean(a > q) * 100:.0f}% (우연이면 5%)\n순위 검정 p = {p:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
            ax.set_title(f"매도 {x}, {h.replace('-', '–')}", fontsize=10)
            ax.grid(alpha=0.3)
            if r_ == len(EXITS_J) - 1:
                ax.set_xlabel("10종목 CAGR − 같은 빈도 무작위 매수 중앙값 (%p)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.99, 0.985), fontsize=8, frameon=False)
    fig.suptitle(f"{label}의 10종목 계좌 초과 수익 분포 vs 같은 빈도 무작위 가짜 묶음 (밀도)", fontsize=11)
    return figures._save(fig, out / "figures" / name)


def _density(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum()
    return counts / tot if tot else counts * 0.0


def _bar(ax, dens: np.ndarray, **kw) -> None:
    """A histogram over RET_EDGES drawn on -60%-+200% (the two open-ended bins at the ends)."""
    x = np.r_[-65, np.arange(-60, 205, 5), 205]
    ax.step(x, np.r_[dens, dens[-1]], where="post", **kw)


def fig_returns_pooled(out: Path, v: pd.DataFrame, un: pd.DataFrame, hist: pd.DataFrame, kind: str, label: str) -> list[Path]:
    """All signals of the sets pooled vs E0 mixed by the share of the sets' signals in each bin."""
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    made = []
    binof = v.set_index(["set", "exit"])["bin"]
    for x in EXITS_J:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for ax, h in zip(axes, es.HALVES):
            hs = hist[(hist["kind"] == kind) & (hist["exit"] == x) & (hist["period"] == h)]
            us = un[(un["kind"] == kind) & (un["exit"] == x)]
            cs = hs[hc].to_numpy().sum(axis=0)
            nb = us.assign(bin=[binof.get((s, x), np.nan) for s in us["set"]]).groupby("bin")[f"n_{h}"].sum()
            he = hist[(hist["kind"] == "e0") & (hist["exit"] == x) & (hist["period"] == h)]
            he = he.assign(bin=he["set"].str.split("__").str[2].astype(float))
            ue = un[(un["kind"] == "e0") & (un["exit"] == x)]
            ue = ue.assign(bin=ue["set"].str.split("__").str[2].astype(float))
            mix, mean_e0 = np.zeros(len(hc)), 0.0
            for b, nn in nb.items():
                wb = nn / nb.sum()
                mix += wb * _density(he[he["bin"] == b][hc].to_numpy().sum(axis=0))
                ue_b = ue[ue["bin"] == b]
                mean_e0 += wb * float((ue_b[f"mean_ret_{h}"] * ue_b[f"n_{h}"]).sum() / ue_b[f"n_{h}"].sum())
            mean_s = float((us[f"mean_ret_{h}"] * us[f"n_{h}"]).sum() / us[f"n_{h}"].sum())
            _bar(ax, mix, color="#777", lw=1.5, label="같은 빈도 무작위 매수 (구간 섞음)")
            _bar(ax, _density(cs), color=figures.COLORS[0], lw=1.5, label=f"{label} 신호 전부")
            ax.axvline(0, color="#bbb", lw=0.8)
            share = lambda dens, lo, hi: dens[(RET_EDGES[:-1] >= lo) & (RET_EDGES[1:] <= hi)].sum() * 100  # noqa: E731
            ax.text(0.98, 0.97, f"신호 {int(cs.sum()):,}개\n평균 {mean_s * 100:+.1f}% (무작위 {mean_e0 * 100:+.1f}%)\n"
                                f"+50% 이상 {share(_density(cs), 0.5, np.inf):.1f}% (무작위 {share(mix, 0.5, np.inf):.1f}%)\n"
                                f"−20% 이하 {share(_density(cs), -np.inf, -0.2):.1f}% (무작위 {share(mix, -np.inf, -0.2):.1f}%)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.set_title(h.replace("-", "–"), fontsize=10)
            ax.set_xlabel("신호 하나의 수익률 (%, 편도 20bp 반영, 양 끝 칸은 −60% 미만 / +200% 초과)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("신호 비율 (5%p 칸)")
        axes[0].legend(loc="center right", fontsize=8, frameon=False)
        fig.suptitle(f"신호 단위 수익률 분포, 매도 {x}: {label} vs 같은 빈도 무작위 매수 (종목 수 제한 없음)", fontsize=11)
        made.append(figures._save(fig, out / "figures" / f"{kind}_returns_{x}.png"))
    return made


def top_sets(v: pd.DataFrame, kind: str, x: str = TOP_EXIT, n: int = 5) -> list[str]:
    """The (b)-passing sets with the highest 1993-2023 10-stock CAGR under exit x (all sets if none pass)."""
    g = v[(v["kind"] == kind) & (v["exit"] == x) & v["enough_trades"]]
    pick = g[g["meaningful_b"]] if g["meaningful_b"].any() else g
    return pick.sort_values("cagr_1993-2023", ascending=False)["set"].head(n).tolist()


def fig_top_returns(out: Path, v: pd.DataFrame, un: pd.DataFrame, hist: pd.DataFrame, kind: str, x: str = TOP_EXIT) -> Path:
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    top = top_sets(v, kind, x)
    fig, axes = plt.subplots(len(top), 1, figsize=(11, 2.9 * max(len(top), 1)), sharex=True, squeeze=False)
    he = hist[(hist["kind"] == "e0") & (hist["exit"] == x) & (hist["period"] == "1993-2023")]
    he = he.assign(bin=he["set"].str.split("__").str[2].astype(float))
    passed = v[(v["kind"] == kind) & v["meaningful_b"] & v["enough_trades"]]["set"].nunique() > 0
    for ax, s in zip(axes[:, 0], top):
        u = un[(un["set"] == s) & (un["exit"] == x)].iloc[0]
        r = v[(v["set"] == s) & (v["exit"] == x)].iloc[0]
        hs = hist[(hist["set"] == s) & (hist["exit"] == x) & (hist["period"] == "1993-2023")][hc].to_numpy().sum(axis=0)
        _bar(ax, _density(he[he["bin"] == r["bin"]][hc].to_numpy().sum(axis=0)), color="#777", lw=1.5, label="같은 빈도 무작위 매수")
        _bar(ax, _density(hs), color=figures.COLORS[0], lw=1.6, label="이 묶음")
        ax.axvline(0, color="#bbb", lw=0.8)
        st = (f"평균 차 {u['st_diff'] * 100:+.1f}%p (95% CI {u['st_ci_lo'] * 100:+.1f} – {u['st_ci_hi'] * 100:+.1f}), "
              f"순위 검정 p = {u['st_mw_p']:.3f}, KS {u['st_ks']:.2f}") if pd.notna(u.get("st_n")) else "신호 10개 미만: 검정 생략"
        ax.set_title(f"{s.split('__', 1)[1].replace('__', ' × ')}: 신호 {int(u['signals']):,}개, 평균 {u['mean_ret_1993-2023'] * 100:+.1f}% "
                     f"(무작위 {u.get('st_ref_mean', np.nan) * 100:+.1f}%), 중앙값 {u['median_ret_1993-2023'] * 100:+.1f}%, "
                     f"승률 {u['win_1993-2023'] * 100:.0f}%, 10종목 CAGR {r['cagr_1993-2023'] * 100:.1f}%\n{st}", fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("신호 하나의 수익률 (%, 1993–2023, 편도 20bp 반영, 양 끝 칸은 −60% 미만 / +200% 초과)")
    which = "(b) 통과 묶음 중" if passed else "(b) 통과 묶음이 없어 전체 중"
    fig.suptitle(f"{which} 10종목 CAGR 상위 {len(top)}개의 신호 단위 수익률, 매도 {x}", fontsize=11)
    return figures._save(fig, out / "figures" / f"{kind}_top_returns_{x}.png")


def fig_matrix(out: Path, v: pd.DataFrame) -> list[Path]:
    plt = figures._plt()
    made = []
    pv, sv = v[v["kind"] == "pair"], v[v["kind"] == "single"]
    for x in EXITS_J:
        g = pv[pv["exit"] == x]
        gs = sv[sv["exit"] == x]
        fig, axes = plt.subplots(2, 2, figsize=(13, 12))
        for c, h in enumerate(es.HALVES):
            for r_, (col, scale, cmap, lim, title) in enumerate((
                    ("cagr", 100, "viridis", None, "10종목 CAGR (%), 대각선 = 단독"),
                    ("pct_in_e0", 100, "RdBu", (0, 100), "같은 빈도 무작위 매수 안 분위 (%)"))):
                ax = axes[r_, c]
                mat = np.full((len(RULES), len(RULES)), np.nan)
                for _, r in g.iterrows():
                    a, b = r["set"].split("__")[1:3]
                    i, j = RULES.index(a), RULES.index(b)
                    mat[i, j] = mat[j, i] = r[f"{col}_{h}"] * scale
                if col == "cagr":
                    for _, r in gs.iterrows():
                        i = RULES.index(r["set"].split("__")[1])
                        mat[i, i] = r[f"cagr_{h}"] * scale
                lo, hi = lim if lim else (np.nanmin(mat), np.nanmax(mat))
                im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi)
                for i in range(len(RULES)):
                    for j in range(len(RULES)):
                        if np.isfinite(mat[i, j]):
                            t = (mat[i, j] - lo) / max(hi - lo, 1e-9)
                            dark = t < 0.45 if cmap == "viridis" else (t < 0.2 or t > 0.8)
                            ax.text(j, i, f"{mat[i, j]:.0f}" if col == "pct_in_e0" else f"{mat[i, j]:.1f}", ha="center",
                                    va="center", fontsize=5.5, color="white" if dark else "black")
                ax.set_xticks(range(len(RULES)), RULES, fontsize=7)
                ax.set_yticks(range(len(RULES)), RULES, fontsize=7)
                ax.set_title(f"{h.replace('-', '–')}: {title}", fontsize=10)
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f"쌍별 10종목 계좌 (무작위 순 100번 중앙값), 매도 {x}; 아래 50 = 같은 빈도 무작위와 같음", fontsize=11)
        made.append(figures._save(fig, out / "figures" / f"pairs_matrix_{x}.png"))
    return made


def fig_stacking(out: Path, v: pd.DataFrame, base: pd.DataFrame, kind: str) -> Path:
    plt = figures._plt()
    spy = base.set_index("baseline").loc["SPY"]
    parts = ["single"] if kind == "pair" else ["pair", "single"]
    names = {"single": "단독", "pair": "쌍"}
    nk = {"pair": "쌍", "triple": "묶음"}[kind]
    g0 = v[v["kind"] == kind]
    fig, axes = plt.subplots(len(parts), len(EXITS_J), figsize=(15, 5 * len(parts)), sharex=True, sharey=True, squeeze=False)
    vals = g0[["cagr_1993-2023"] + [f"best_{p}_cagr" for p in parts]].to_numpy(dtype=float) * 100
    lo, hi = np.nanmin(vals) - 0.5, np.nanmax(vals) + 0.5
    for r_, part in enumerate(parts):
        for ax, x in zip(axes[r_], EXITS_J):
            g = g0[g0["exit"] == x]
            ax.plot([lo, hi], [lo, hi], color="#555", lw=1)
            ax.axhline(spy["cagr_1993-2023"] * 100, color="#c0392b", ls="--", lw=1)
            up = g[f"stacks_{part}"].to_numpy(dtype=bool)
            ax.scatter(g[f"best_{part}_cagr"][~up] * 100, g["cagr_1993-2023"][~up] * 100, s=10, color="#999",
                       label=f"더 나은 {names[part]} 이하")
            ax.scatter(g[f"best_{part}_cagr"][up] * 100, g["cagr_1993-2023"][up] * 100, s=10, color=figures.COLORS[0],
                       label=f"더 나은 {names[part]}보다 높음")
            ax.set_title(f"매도 {x}: {int(up.sum())} / {len(g)}개가 구성 {names[part]} 중 최선보다 높음", fontsize=10)
            ax.set_xlabel(f"구성 {names[part]} 중 최선의 CAGR (%)")
            ax.grid(alpha=0.3)
        axes[r_, 0].set_ylabel(f"{nk} CAGR (%)")
        axes[r_, -1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"쌓임 효과: 10종목 계좌 1993–2023 CAGR (무작위 순 중앙값), 대각선 위 = {nk}이 낫다, 빨간 점선 SPY", fontsize=11)
    return figures._save(fig, out / "figures" / f"{'pairs' if kind == 'pair' else 'triples'}_stacking.png")


def write_figures(out: Path, v: pd.DataFrame, base: pd.DataFrame, size: int, un: pd.DataFrame | None = None,
                  hist: pd.DataFrame | None = None) -> list[Path]:
    if not figures.available():
        return []
    kind = SIZES[size]
    label = {"pair": "쌍", "triple": "3개 묶음"}[kind]
    made = [fig_excess(out, v, kind, label, f"{'pairs' if size == 2 else 'triples'}_excess_hist.png")]
    if hist is not None and len(hist):
        made += fig_returns_pooled(out, v, un, hist, kind, label)
        made.append(fig_top_returns(out, v, un, hist, kind))
    if size == 2:
        made += fig_matrix(out, v)
    made.append(fig_stacking(out, v, base, kind))
    return made


# --------------------------------------------------------------------------- main

def run_sets(todo: list, sets_dir: Path, a) -> None:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs, a.cfg)) as ex:
        for i, (spec, acc, un, hist) in enumerate(ex.map(run_set, todo, chunksize=1)):
            acc.to_csv(sets_dir / f"{set_name(spec)}_acc.csv", index=False)
            un.to_csv(sets_dir / f"{set_name(spec)}_un.csv", index=False)
            hist.to_csv(sets_dir / f"{set_name(spec)}_hist.csv", index=False)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                ts.log(f"  [{i + 1}/{len(todo)}] 남은 예상 {(len(todo) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


def _with_bin(t: pd.DataFrame) -> pd.DataFrame:
    r0 = t["kind"].isin(["e0", "placebo"])
    t["bin"] = np.nan
    t.loc[r0, "bin"] = t.loc[r0, "set"].str.split("__").str[2].astype(float)
    return t


def redraw(out: Path, size: int) -> list[Path]:
    """Figures again from a finished run's tables (verdicts.csv, baselines.csv; unlimited / hist if any)."""
    v, base = pd.read_csv(out / "verdicts.csv"), pd.read_csv(out / "baselines.csv")
    un = pd.read_csv(out / "unlimited.csv") if (out / "unlimited.csv").exists() else None
    hist = pd.read_csv(out / "hist.csv") if (out / "hist.csv").exists() else None
    return write_figures(out, v, base, size, un, hist)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}")
    ap.add_argument("--size", type=int, default=2, choices=sorted(SIZES))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--redraw", default=None, help="그림만 다시 그림: 끝난 실행의 결과 폴더")
    a = ap.parse_args(argv)
    global RULES, E0_SEEDS, BINS
    if a.size == 3:
        BINS = list(BINS_J3)
    if a.redraw:
        made = redraw(Path(a.redraw), a.size)
        ts.log(f"그림 {len(made)}장")
        return
    missing = [k for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "out") if not getattr(a, k)]
    if missing:
        ap.error("필요한 인자: " + ", ".join(f"--{k.replace('_', '-')}" for k in missing))
    kind = SIZES[a.size]
    n_placebo = N_PLACEBO_J3 if a.size == 3 else N_PLACEBO_PER_BIN
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events), "r0x1": out / "r0x1"}
    a.dirs = dirs
    use_random_x1_menu(a.dry)
    with ProcessPoolExecutor(max_workers=2, initializer=_r_init, initargs=(a.panel, a.eligible, a.dry)) as ex:
        sc.run_signal_tables(ex, out / "r0x1", ts.log)
    if a.dry:
        RULES, E0_SEEDS, n_placebo = ["E2", "E9", "V3"], E0_SEEDS[:2], 2
    a.cfg = {"bins": BINS, "seeds": E0_SEEDS}
    n_orders = 5 if a.dry else N_ORDERS
    p_orders = 3 if a.dry else PLACEBO_ORDERS
    specs = ([(("single", r), n_orders, COST_BPS) for r in RULES]
             + [((SIZES[k],) + c, n_orders, COST_BPS) for k in range(2, a.size + 1) for c in combos(k)]
             + [(("e0", s, b, 0), n_orders, COST_BPS) for s in E0_SEEDS for b in BINS]
             + [(("placebo", E0_SEEDS[d % len(E0_SEEDS)], b, 1 + d), p_orders, COST_BPS) for b in BINS for d in range(n_placebo)])
    sets_dir = out / "sets"
    sets_dir.mkdir(exist_ok=True)
    todo = [j for j in specs if not (sets_dir / f"{set_name(j[0])}_acc.csv").exists()]
    ts.log(f"묶음 {len(todo)}개 남음 (단독 {len(RULES)}, " + ", ".join(f"{k}개 묶음 {len(combos(k))}" for k in range(2, a.size + 1))
           + f", E0 {len(E0_SEEDS) * len(BINS)}, 위약 {n_placebo * len(BINS)})")
    run_sets(todo, sets_dir, a)
    acc = _with_bin(pd.concat([pd.read_csv(f) for f in sorted(sets_dir.glob("*_acc.csv"))], ignore_index=True))
    un = _with_bin(pd.concat([pd.read_csv(f) for f in sorted(sets_dir.glob("*_un.csv"))], ignore_index=True))
    hist = pd.concat([pd.read_csv(f) for f in sorted(sets_dir.glob("*_hist.csv"))], ignore_index=True)
    ref_acc, ref_un = acc[acc["kind"] == "e0"], un[un["kind"] == "e0"]
    _init(a.panel, a.eligible, dirs, a.cfg)
    base = es.baselines(_W["px"], _W["wdates"], _W["w0"], _W["cash"])
    base.to_csv(out / "baselines.csv", index=False)
    v = judge(acc[acc["kind"] != "e0"], un[un["kind"] != "e0"], ref_acc, ref_un)
    v["kind"] = v["set"].str.split("__").str[0]
    v["candidate_20bp"] = candidates(v, base) & (v["kind"] != "placebo")
    v["candidate_40bp"] = False
    cand = v[v["candidate_20bp"]]
    if len(cand):  # candidates only: the same accounts at 40 bp
        d40 = out / "sets_40bp"
        d40.mkdir(exist_ok=True)
        by_name = {set_name(j[0]): j[0] for j in specs}
        run_sets([(by_name[s], n_orders, CHECK_COST_BPS) for s in sorted(cand["set"].unique())
                  if not (d40 / f"{s}_acc.csv").exists()], d40, a)
        med40 = pd.concat([pd.read_csv(d40 / f"{s}_acc.csv") for s in cand["set"].unique()]).groupby(["set", "exit"]).median(numeric_only=True)
        spy = base.set_index("baseline").loc["SPY"]
        for i, r in cand.iterrows():
            m = med40.loc[(r["set"], r["exit"])]
            v.loc[i, "candidate_40bp"] = all(m[f"cagr_{p}"] > spy[f"cagr_{p}"] and abs(m[f"mdd_{p}"]) < abs(spy[f"mdd_{p}"])
                                             for p in ("1993-2023", "2017-2023"))
    v = stacking(v)
    v.to_csv(out / "verdicts.csv", index=False)
    acc.groupby(["set", "kind", "exit"]).quantile([0.05, 0.5, 0.95], numeric_only=True).to_csv(out / "accounts_quantiles.csv")
    un.to_csv(out / "unlimited.csv", index=False)
    hist.to_csv(out / "hist.csv", index=False)
    st_cols = ["set", "kind", "exit", "signals", "st_bin"] + [f"{c}_1993-2023" for c in ("mean_ret", "median_ret", "win", "big50", "loss20")] \
        + [c for c in un.columns if c.startswith("st_") and c != "st_bin"]
    st = un[un["kind"] != "e0"].reindex(columns=st_cols)
    st.to_csv(out / "stats.csv", index=False)
    ch = {}
    for k in sorted({SIZES[k] for k in range(2, a.size + 1)}, reverse=True):
        sv = v[(v["kind"] == k) & v["enough_trades"]].drop_duplicates("set")
        passed = {"meaningful_a": int(sv["meaningful_a"].sum()), "meaningful_b": int(sv["meaningful_b"].sum())}
        ch[k] = chance(v[v["kind"] == "placebo"], len(sv), passed)
        ch[k]["excluded_few_trades"] = int((v[v["kind"] == k].drop_duplicates("set")["enough_trades"] == False).sum())  # noqa: E712
    main_ch = ch[kind]
    main_ch.update({k_: ch[k_] for k_ in ch if k_ != kind})  # Stage J layout at the top level, other sizes nested
    main_ch["candidates_20bp"] = v.loc[v["candidate_20bp"], ["set", "exit"]].to_dict("records")
    main_ch["candidates_40bp"] = v.loc[v["candidate_40bp"], ["set", "exit"]].to_dict("records")
    main_ch["stats"] = stats_summary(st)
    main_ch["top_sets"] = top_sets(v, kind)
    main_ch["trials"] = {"before": N_TRIALS_BEFORE + (N_NEW if a.size == 3 else 0), "new": N_NEW_J3 if a.size == 3 else N_NEW}
    (out / "chance.json").write_text(json.dumps(main_ch, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, v, base, a.size, un, hist)
    sm = main_ch["stats"]
    ts.log(f"완료 · 통과 {kind} (a) {main_ch['observed']['meaningful_a']} · (b) {main_ch['observed']['meaningful_b']} / "
           f"{main_ch['n_pairs_judged']} · 우연 95% 분위 {main_ch['meaningful_a']['q95']} · {main_ch['meaningful_b']['q95']} · "
           f"후보 {len(main_ch['candidates_20bp'])} · p<0.05 {kind} {sm.get(kind, {}).get('all', {}).get('p_lt_05', np.nan):.2f} "
           f"vs 가짜 {sm.get('placebo', {}).get('all', {}).get('p_lt_05', np.nan):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
