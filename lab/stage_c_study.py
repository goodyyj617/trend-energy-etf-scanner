"""Stage C: entry strategies x exit strategies on the daily top-3,000 universe (DECISIONS 81), resumable.

    python -m lab.stage_c_study --panel DIR --out DIR [--results DIR] [--workers N] [--dry]

Universe: the panel built by `lab.crsp --daily` (each day's top 3,000 by 21-day average market
value, delisted names included) with the hygiene screens of lab/screens.py evaluated daily. No
month-end list and no trend template. Ten entries (E0 always, E1-E9 fire on the first day their
condition is true) x nine exits (X1-X9), three values each (3 x 3 cells; E0 has one value).

C1 every signal: lab/single.py, each signal traded on its own (no slot limit). Trade list kept.
C2 20-slot account: max 20 positions, 1/20 of equity at purchase, no rebalancing, signals while
full are dropped. Priority (a) 20-day dollar volume on every cell; on the canonical cell of each
pair also (b) mean percentile of trend efficiency (6 months) and market value, (c) random (seed 81).
Per-signal table: every entry signal traded on its own with the same exits, overlapping signals of
one symbol allowed (engine run_backtest(outcomes=True); not for E0, where every eligible stock-day is
a signal). Missed vs taken: C2(a) trades matched to that table by (symbol, entry day).

Output under `out`: c1/<sample>/<E>__<X>.csv (+ _trades.parquet, _sleeve.parquet) and
c2/<sample>/<E>__<X>.csv (+ _equity.parquet), one pair at a time so an interrupted run resumes;
summary_c1.csv, summary_c2.csv, entry_ranks.csv, exit_ranks.csv, stage_c_summary.json, figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import explore, figures, research_log as rlog, screens, single
from .crsp_study import SAMPLES, load_panel_with_cash, memory_note, spy_stats
from .data import BENCHMARK, Panel
from .engine import SignalCache, StrategyConfig, run_backtest
from .grid import _neighbors_and_regions
from .metrics import _month_end_values
from .target import target_report

# ---- registered menu (DECISIONS 81): fixed before any result
ENTRIES: dict[str, tuple[str, list]] = {
    "E0": ("always", [None]),
    "E1": ("c_donchian", [20, 55, 100]),
    "E2": ("c_golden", [20, 50, 100]),
    "E3": ("c_high_volume", [1.2, 1.5, 2.0]),
    "E4": ("c_squeeze", [10, 20, 30]),
    "E5": ("c_rsi2_pullback", [5, 10, 15]),
    "E6": ("c_macd", [8, 12, 19]),
    "E7": ("c_di_cross", [15, 20, 25]),
    "E8": ("c_tsmom", [3, 6, 12]),
    "E9": ("c_gap_volume", [2, 4, 6]),
}
ENTRY_CANON = {"E0": 0, "E1": 1, "E2": 1, "E3": 1, "E4": 1, "E5": 1, "E6": 1, "E7": 1, "E8": 2, "E9": 1}
EXITS: dict[str, tuple[str, list]] = {
    "X1": ("trailing_pct", [15, 25, 35]),
    "X2": ("chandelier", [2.0, 3.0, 4.0]),
    "X3": ("low_break", [10, 20, 50]),
    "X4": ("below_ma", [20, 50, 100]),
    "X5": ("momentum_neg", [21, 63, 126]),
    "X6": ("psar_flip", [0.01, 0.02, 0.03]),
    "X7": ("rsi2_exit", [60, 70, 80]),
    "X8": ("market_off", [100, 150, 200]),
    "X9": ("atr_stop", [1.5, 2.0, 3.0]),  # + time_stop 60 + breakeven at +2R
}
EXIT_CANON = {"X1": 1, "X2": 1, "X3": 1, "X4": 1, "X5": 1, "X6": 1, "X7": 1, "X8": 2, "X9": 1}
X9_EXTRA = {"time_stop": 60}
X9_BREAKEVEN_R = 2.0
HYGIENE = dict(screens.HYGIENE_DEFAULT)  # no size floor, no trend template
COMMON = dict(cost_bps=20.0, min_price=5.0, min_dollar_volume=1e6, sizing="equal")
SLOTS = 20
PRIORITIES = ("dv", "eff_large", "random")  # (a) default, (b), (c)
SEED = 81
BIG_WINNER = 0.5  # a signal that returns +50% or more is a "big winner"
NO_SIGNAL_TABLE = {"E0"}  # 'always' is a state: every eligible stock-day would be a signal
C1_KEEP = ["symbols", "symbol_pass_share", "symbol_median_excess", "n_trades", "sleeve_cagr", "sleeve_mdd",
           "sleeve_dd_cagr", "sleeve_mix_cagr", "sleeve_dd_excess", "sleeve_window_share"]
_W: dict = {}


def pairs() -> list[tuple[str, str]]:
    return [(e, x) for e in ENTRIES for x in EXITS]


def cells(e: str, x: str) -> list[tuple[int, int]]:
    return [(xi, yi) for yi in range(len(EXITS[x][1])) for xi in range(len(ENTRIES[e][1]))]


def canonical(e: str, x: str) -> tuple[int, int]:
    return ENTRY_CANON[e], EXIT_CANON[x]


def rules(e: str, x: str, xi: int, yi: int) -> tuple[dict, dict, float | None]:
    ek, ev = ENTRIES[e]
    xk, xv = EXITS[x]
    exits = {xk: xv[yi]}
    if x == "X9":
        exits.update(X9_EXTRA)
    return {ek: ev[xi]}, exits, (X9_BREAKEVEN_R if x == "X9" else None)


def config(sample: str, e: str, x: str, xi: int, yi: int, **kw) -> StrategyConfig:
    s, end, _ = SAMPLES[sample]
    entries, exits, be = rules(e, x, xi, yi)
    return StrategyConfig(entries=entries, exits=exits, start=s, end=end, breakeven_r=be, **COMMON, **kw)


def pair_file(out: Path, stage: str, sample: str, e: str, x: str) -> Path:
    return out / stage / sample / f"{e}__{x}.csv"


# --------------------------------------------------------------------------- universe

def daily_eligible(panel: Panel) -> pd.DataFrame:
    """Daily top-3,000 membership minus every hygiene screen (True = may be bought that day)."""
    ok = panel.eligible.reindex(index=panel.close.index, columns=panel.close.columns).fillna(False).astype(bool)
    for key, value in HYGIENE.items():
        ok &= ~screens.ALL_SCREENS[key].compute(panel, value).fillna(False).astype(bool)
    ok[BENCHMARK] = False
    return ok


def priority_rank(panel: Panel, how: str) -> pd.DataFrame | None:
    """panel.rank for C2 (1 = bought first): None for dollar volume (the engine default)."""
    if how == "dv":
        return None
    elig = panel.eligible
    if how == "eff_large":
        parts = []
        for key, value in (("s_efficiency", 6), ("s_large", None)):
            sc = screens.SELECT_SCORES[key].compute(panel, value).where(elig)
            parts.append(sc.rank(axis=1, pct=True))
        score = (parts[0] + parts[1]) / 2.0
    else:
        rng = np.random.default_rng(SEED)
        score = pd.DataFrame(rng.random(panel.close.shape, dtype=np.float32), index=panel.close.index,
                             columns=panel.close.columns).where(elig)
    return score.rank(axis=1, ascending=False, method="first").astype("float32")


# --------------------------------------------------------------------------- workers

def _setup(panel: Panel, eligible: pd.DataFrame | None = None) -> None:
    if eligible is not None:
        panel = replace(panel, eligible=eligible)
    _W["panel"] = panel
    _W["cache"] = SignalCache(panel)
    _W["spy"] = {k: spy_stats(panel, s, e) for k, (s, e, _) in SAMPLES.items()}
    _W["rank"] = {}
    _W["pair"] = None


def _init(panel_dir: str, eligible_path: str, dry: bool = False):
    if dry:
        _dry_menu()
    _setup(load_panel_with_cash(Path(panel_dir)), pd.read_parquet(eligible_path))


def trade_stats(trades: pd.DataFrame, years: float) -> dict:
    r = trades["return"].astype(float) if len(trades) else pd.Series(dtype=float)
    win, loss = r[r > 0], r[r <= 0]
    return {"n_trades": int(len(r)), "signals_per_year": len(r) / years if years > 0 else np.nan,
            "expectancy": float(r.mean()) if len(r) else np.nan,
            "win_rate": float((r > 0).mean()) if len(r) else np.nan,
            "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else np.nan,
            "avg_hold_days": float(trades["holding_days"].mean()) if len(r) else np.nan}


def signal_stats(sig: pd.DataFrame) -> dict:
    """Per-signal results (every signal traded on its own, overlaps allowed)."""
    r = sig["return"].astype(float) if len(sig) else pd.Series(dtype=float)
    win, loss = r[r > 0], r[r <= 0]
    return {"signals": int(len(r)), "sig_expectancy": float(r.mean()) if len(r) else np.nan,
            "sig_win_rate": float((r > 0).mean()) if len(r) else np.nan,
            "sig_payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else np.nan,
            "sig_avg_hold_days": float(sig["holding_days"].mean()) if len(r) else np.nan,
            "sig_big_share": float((r >= BIG_WINNER).mean()) if len(r) else np.nan}


def missed_vs_taken(sig: pd.DataFrame, c2: pd.DataFrame, cap_rank: pd.DataFrame | None) -> dict:
    """C2 trades matched to the per-signal table by (symbol, entry day = the open after the signal day);
    per-signal returns of the signals the account took vs those it could not take."""
    if not len(sig):
        return {"signals": 0}
    key = list(zip(sig["symbol"], sig["entry_date"]))
    taken_keys = set(zip(c2["symbol"], c2["entry_date"]))
    taken = np.array([k in taken_keys for k in key])
    r = sig["return"].to_numpy(dtype=float)
    big = r >= BIG_WINNER
    out = {"taken": int(taken.sum()), "missed": int((~taken).sum()),
           "c2_unmatched": int(len(taken_keys - set(key))),
           "taken_expectancy": float(r[taken].mean()) if taken.any() else np.nan,
           "missed_expectancy": float(r[~taken].mean()) if (~taken).any() else np.nan,
           "big_winners": int(big.sum()), "big_winners_taken_share": float(taken[big].mean()) if big.any() else np.nan}
    if cap_rank is not None:
        pos = cap_rank.index.get_indexer(pd.DatetimeIndex(sig["signal_date"]))
        cols = cap_rank.columns.get_indexer(sig["symbol"])
        ok = (pos >= 0) & (cols >= 0)
        cr = np.full(len(sig), np.nan)
        cr[ok] = cap_rank.to_numpy()[pos[ok], cols[ok]]
        for name, m in (("taken", taken), ("missed", ~taken)):
            v = cr[m]
            v = v[v == v]
            for q in (25, 50, 75):
                out[f"{name}_cap_rank_p{q}"] = float(np.percentile(v, q)) if len(v) else np.nan
    return out


def _run_pair(job):
    """Every cell of one (sample, entry, exit) pair: C1, C2 (a) and, on the canonical cell, C2 (b, c)."""
    sample, e, x = job
    panel, cache = _W["panel"], _W["cache"]
    if _W["pair"] != (sample, e):  # block outputs of one entry are reused across its exits
        cache.drop_rule_data()
        _W["pair"] = (sample, e)
    s, end, _ = SAMPLES[sample]
    years = (pd.Timestamp(end) - pd.Timestamp(s)).days / 365.25
    spy = _W["spy"][sample]
    cap_rank = panel.extra.get("cap_rank")
    c1_rows, c2_rows, c1_trades, signals, sleeves, c2_eq, logs = [], [], [], [], {}, {}, []
    for xi, yi in cells(e, x):
        cfg = config(sample, e, x, xi, yi)
        o = single.evaluate(panel, cfg, cache, with_result=True)
        t1 = o["result"].trades
        sig = (run_backtest(panel, cfg, cache, outcomes=True) if e not in NO_SIGNAL_TABLE
               else pd.DataFrame(columns=["symbol", "signal_date", "entry_date", "return", "holding_days", "closed"]))
        row = {"xi": xi, "yi": yi, "entry_value": ENTRIES[e][1][xi], "exit_value": EXITS[x][1][yi],
               **{k: o[k] for k in C1_KEEP}, **trade_stats(t1, years), **spy,
               **(signal_stats(sig) if e not in NO_SIGNAL_TABLE else {})}
        row["goal_excess"] = o["sleeve_cagr"] - spy["spy_cagr"]
        row["goal_pass"] = bool(o["sleeve_cagr"] >= spy["spy_cagr"] and abs(o["sleeve_mdd"]) <= abs(spy["spy_mdd"])
                                and len(t1) >= 30)
        c1_rows.append(row)
        c1_trades.append(t1.assign(xi=xi, yi=yi)[["xi", "yi", "symbol", "entry_date", "exit_date", "return",
                                                   "holding_days", "exit_reason"]])
        if len(sig):
            signals.append(sig.assign(xi=xi, yi=yi)[["xi", "yi", "symbol", "signal_date", "entry_date", "return",
                                                     "holding_days", "closed"]])
        sleeves[f"{xi}|{yi}"] = o["sleeve"].astype("float32")
        logs.append(("stagec_c1", cfg, o["sleeve"]))
        for how in PRIORITIES if (xi, yi) == canonical(e, x) else PRIORITIES[:1]:
            if how not in _W["rank"]:
                _W["rank"][how] = priority_rank(panel, how)
            pp = replace(panel, rank=_W["rank"][how])
            c2cfg = replace(cfg, max_positions=SLOTS)
            res = run_backtest(pp, c2cfg, cache if how == "dv" else SignalCache(pp).share_from(cache))
            eq = res.equity
            v = eq.to_numpy(dtype=float)
            cagr = float(v[-1] ** (252.0 / max(len(v) - 1, 1)) - 1.0)
            mdd = float((v / np.maximum.accumulate(v) - 1.0).min())
            t = target_report(eq, panel.close[BENCHMARK], panel.cash, len(res.trades))
            r2 = {"xi": xi, "yi": yi, "priority": how, "cagr": cagr, "mdd": mdd, **trade_stats(res.trades, years), **spy,
                  "dd_excess": t["dd_excess"], "window_share": t["window_share"],
                  "exposure": float(res.exposure.mean())}
            r2["goal_excess"] = cagr - spy["spy_cagr"]
            r2["goal_pass"] = bool(cagr > spy["spy_cagr"] and abs(mdd) < abs(spy["spy_mdd"]) and len(res.trades) >= 30)
            if how == "dv" and e not in NO_SIGNAL_TABLE:
                r2.update(missed_vs_taken(sig, res.trades, cap_rank))
            c2_rows.append(r2)
            c2_eq[f"{xi}|{yi}|{how}"] = eq.astype("float32")
            logs.append(("stagec_c2", c2cfg, eq))
    sig_all = pd.concat(signals, ignore_index=True) if signals else None
    return (sample, e, x), pd.DataFrame(c1_rows), pd.concat(c1_trades, ignore_index=True), pd.DataFrame(sleeves), \
        pd.DataFrame(c2_rows), pd.DataFrame(c2_eq), logs, sig_all


# --------------------------------------------------------------------------- runs

def write_pair(out: Path, result, panel_symbols: list[str]) -> None:
    (sample, e, x), c1, trades, sleeves, c2, eq, logs, sig = result
    for stage in ("c1", "c2"):
        pair_file(out, stage, sample, e, x).parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(pair_file(out, "c1", sample, e, x).with_name(f"{e}__{x}_trades.parquet"))
    if sig is not None:
        sig.astype({"return": "float32", "holding_days": "int32"}).to_parquet(
            pair_file(out, "c1", sample, e, x).with_name(f"{e}__{x}_signals.parquet"), compression="zstd")
    sleeves.to_parquet(pair_file(out, "c1", sample, e, x).with_name(f"{e}__{x}_sleeve.parquet"))
    eq.to_parquet(pair_file(out, "c2", sample, e, x).with_name(f"{e}__{x}_equity.parquet"))
    for frame, stage in ((c1, "c1"), (c2, "c2")):
        frame = frame.copy()
        frame.insert(0, "exit", x)
        frame.insert(0, "entry", e)
        frame.to_csv(pair_file(out, stage, sample, e, x), index=False)  # written last: marks the pair done
    rlog.log_trials([(kind, cfg, panel_symbols, curve.astype(float)) for kind, cfg, curve in logs])


def todo_jobs(out: Path) -> list[tuple[str, str, str]]:
    return [(s, e, x) for s in SAMPLES for e, x in pairs()
            if not (pair_file(out, "c1", s, e, x).exists() and pair_file(out, "c2", s, e, x).exists())]


def run_pairs(ex, out: Path, panel: Panel, log) -> None:
    jobs = todo_jobs(out)
    total = len(SAMPLES) * len(pairs())
    log(f"전략 쌍 × 표본: {len(jobs)}개 남음 (전체 {total}, 칸 {sum(len(cells(e, x)) for e, x in pairs())}개/표본)")
    t0 = time.time()
    for n, result in enumerate(ex.map(_run_pair, jobs, chunksize=1)):
        write_pair(out, result, panel.tradable_symbols)
        sample, e, x = result[0]
        per = (time.time() - t0) / (n + 1)
        c2 = result[4]
        log(f"  [{n + 1}/{len(jobs)}] {SAMPLES[sample][2]} {e}×{x}: C1 통과 {int(result[1]['goal_pass'].sum())}/{len(result[1])}, "
            f"C2 통과 {int(c2['goal_pass'].sum())}/{len(c2)} · 남은 예상 {(len(jobs) - n - 1) * per / 3600:.1f}h{memory_note()}")


# --------------------------------------------------------------------------- judging (registered) and summary

def load(out: Path, stage: str) -> pd.DataFrame:
    parts = []
    for sample in SAMPLES:
        for f in sorted((out / stage / sample).glob("*__*.csv")):
            g = pd.read_csv(f)
            g["sample"] = sample
            parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def neighbours(c: pd.DataFrame) -> pd.DataFrame:
    """Pass, neighbour survival and connected region within each pair's 3 x 3 grid."""
    parts = []
    for _, g in c.groupby(["sample", "entry", "exit"], sort=False):
        g = g.copy()
        g["pass"] = g["goal_pass"].astype(bool)
        _neighbors_and_regions(g)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def side_ranks(c: pd.DataFrame, side: str, value: str = "goal_excess") -> pd.DataFrame:
    """Entry (or exit) rank per sample by the median over every cell of the other side, then pass share;
    Spearman of the ranks across samples in the attrs."""
    g = c.groupby([side, "sample"]).agg(median=(value, "median"), pass_share=("goal_pass", "mean")).reset_index()
    out = []
    for smp, h in g.groupby("sample"):
        h = h.sort_values(["median", "pass_share"], ascending=False).copy()
        h["rank"] = range(1, len(h) + 1)
        out.append(h)
    t = pd.concat(out).pivot(index=side, columns="sample", values=["rank", "median", "pass_share"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t["rank_sum"] = t[[f"rank_{k}" for k in SAMPLES if f"rank_{k}" in t]].sum(axis=1)
    t = t.sort_values("rank_sum")
    if len(SAMPLES) == 2 and all(f"rank_{k}" in t for k in SAMPLES):
        a, b = list(SAMPLES)
        t.attrs["spearman"] = float(t[f"rank_{a}"].corr(t[f"rank_{b}"]))
    return t


def pbo_all(out: Path, stage: str) -> dict:
    res = {}
    for smp in SAMPLES:
        cols = {}
        pat = "*_sleeve.parquet" if stage == "c1" else "*_equity.parquet"
        for f in sorted((out / stage / smp).glob(pat)):
            eq = pd.read_parquet(f)
            for c in eq:
                if stage == "c2" and not c.endswith("|dv"):
                    continue
                cols[f"{f.stem}|{c}"] = _month_end_values(eq[c].dropna().astype(float)).pct_change(fill_method=None).dropna()
        res[smp] = explore.pbo(pd.DataFrame(cols), blocks=12) if len(cols) > 1 else None
    return res


def summarize(out: Path) -> dict:
    c1, c2 = load(out, "c1"), load(out, "c2")
    if c1.empty or c2.empty:
        return {}
    c1 = neighbours(c1)
    c2a = neighbours(c2[c2["priority"] == "dv"])
    c1.to_csv(out / "summary_c1.csv", index=False)
    c2.to_csv(out / "summary_c2.csv", index=False)
    summ: dict = {"cells_c1": int(len(c1)), "cells_c2": int(len(c2))}
    for name, frame in (("c1", c1), ("c2", c2a)):
        for side in ("entry", "exit"):
            r = side_ranks(frame, side)
            r.to_csv(out / f"{side}_ranks_{name}.csv")
            summ[f"spearman_{side}_{name}"] = r.attrs.get("spearman")
        e0 = frame[frame["entry"] == "E0"].groupby("sample")["goal_excess"].median()
        below = frame.groupby(["entry", "sample"])["goal_excess"].median().unstack("sample")
        summ[f"entries_below_E0_{name}"] = {s: [e for e in below.index if e != "E0" and below.loc[e, s] < e0.get(s, np.nan)]
                                            for s in below.columns}
        summ[f"pass_{name}"] = {s: [int(g["goal_pass"].sum()), int(len(g))] for s, g in frame.groupby("sample")}
    pr = c2[c2["priority"].isin(PRIORITIES)].groupby(["priority", "sample"])[["goal_excess", "goal_pass"]].mean()
    summ["priority"] = {f"{p}|{s}": {"mean_excess": float(v["goal_excess"]), "pass_share": float(v["goal_pass"])}
                        for (p, s), v in pr.iterrows()}
    mt = c2a[["sample", "taken_expectancy", "missed_expectancy", "big_winners_taken_share",
              "taken_cap_rank_p50", "missed_cap_rank_p50"]].groupby("sample").median()
    summ["missed_vs_taken_median"] = mt.to_dict(orient="index")
    summ["pbo"] = {"c1": pbo_all(out, "c1"), "c2": pbo_all(out, "c2")}
    (out / "stage_c_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, c1, c2a)
    return summ


def write_figures(out: Path, c1: pd.DataFrame, c2a: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    d = out / "figures"
    names = {k: SAMPLES[k][2] for k in SAMPLES}
    made = []
    for side in ("entry", "exit"):
        for stage, frame, label in (("c1", c1, "신호 전체"), ("c2", c2a, "20종목 계좌")):
            m = frame.groupby([side, "sample"])["goal_excess"].median().unstack("sample")
            labels = [f"{k} {(ENTRIES if side == 'entry' else EXITS)[k][0]}" for k in m.index]
            made.append(figures.paired_bars(labels, {names[s]: m[s].to_numpy() for s in m.columns if s in names},
                                            f"{'매수' if side == 'entry' else '매도'} 전략별 SPY 대비 초과 CAGR ({label}, 중앙값)",
                                            "SPY 대비 초과 CAGR (%p)", d / f"{side}_{stage}.png"))
    mt = c2a.groupby("entry")[["taken_expectancy", "missed_expectancy"]].median().dropna()
    made.append(figures.paired_bars([f"{k} {ENTRIES[k][0]}" for k in mt.index],
                                    {"20종목 계좌가 산 신호": mt["taken_expectancy"].to_numpy(),
                                     "자리가 없어 놓친 신호": mt["missed_expectancy"].to_numpy()},
                                    "산 신호와 놓친 신호의 신호당 평균 수익 (신호마다 따로 매매, 중앙값)", "신호당 평균 수익 (%)",
                                    d / "missed_vs_taken.png"))
    groups = {names[k]: g[["cagr", "mdd"]] for k, g in c2a.groupby("sample")}
    spy = {names[k]: (float(g["spy_cagr"].iloc[0]), float(g["spy_mdd"].iloc[0])) for k, g in c2a.groupby("sample")}
    made.append(figures.cagr_mdd(groups, spy, "20종목 계좌 모든 칸의 CAGR과 MDD (별 = SPY)", d / "c2_cagr_mdd.png"))
    return made


# --------------------------------------------------------------------------- main

def _dry_menu() -> None:
    for k in [k for k in ENTRIES if k not in ("E0", "E1")]:
        del ENTRIES[k]
    for k in [k for k in EXITS if k not in ("X1", "X9")]:
        del EXITS[k]


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="E0, E1 x X1, X9: checks the pipeline and times it")
    a = ap.parse_args(argv)
    if a.dry:
        _dry_menu()
    out, panel_dir = Path(a.out), Path(a.panel)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(a.results) if a.results else out / "lab_results"
    rlog.RESULTS_DIR = results
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    panel = load_panel_with_cash(panel_dir)
    meta = json.loads((panel_dir / "meta.json").read_text(encoding="utf-8"))
    if "cap_rank" not in panel.extra:
        raise SystemExit("일별 유니버스 패널이 아닙니다: python -m lab.crsp --daily 로 만든 패널이 필요합니다.")
    log(f"패널: 종목 {meta['symbols']}, {meta['first']} – {meta['last']} · 작업 {a.workers}개")
    elig_path = out / "eligible_daily.parquet"
    if not elig_path.exists():
        t = time.time()
        daily_eligible(panel).to_parquet(elig_path)
        log(f"일별 매수 가능 목록(상위 3000 − 위생 필터) 계산: {time.time() - t:.0f}s")
    (out / "menu.json").write_text(json.dumps({"entries": ENTRIES, "exits": EXITS, "entry_canon": ENTRY_CANON,
                                               "exit_canon": EXIT_CANON, "x9_extra": X9_EXTRA, "x9_breakeven_r": X9_BREAKEVEN_R,
                                               "hygiene": HYGIENE, "common": COMMON, "slots": SLOTS,
                                               "priorities": PRIORITIES, "seed": SEED},
                                              ensure_ascii=False, default=str), encoding="utf-8")
    workers = a.workers
    while True:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(panel_dir), str(elig_path), a.dry)) as ex:
                run_pairs(ex, out, panel, log)
            break
        except (BrokenProcessPool, MemoryError):
            if workers <= 1:
                raise
            workers = max(1, workers * 3 // 4)
            log(f"메모리가 부족했습니다. 작업 {workers}개로 줄여 이어서 실행합니다(끝난 부분은 건너뜀).")
    summ = summarize(out)
    n, _ = rlog.trial_stats()
    n_eff, rho, _ = rlog.effective_trials()
    log(f"완료 · 시험 {n}회 (유효 {n_eff:.0f}, ρ {rho:.2f}) · C2 통과 {summ.get('pass_c2')} · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
