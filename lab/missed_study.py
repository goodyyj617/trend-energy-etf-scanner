"""Stage D: capture the signals the 20-slot account missed (DECISIONS 82), resumable.

    python -m lab.missed_study --panel DIR --signals DIR --out DIR [--stagec DIR] [--workers N] [--dry]

Stage C's 20-slot account bought mostly the largest names among the day's signals and missed the
smaller ones, which earned more per signal. Here only the choice of signals changes; entries
(E1-E9), exits (X1-X9), the daily universe, 20 slots of 1/20 of equity, 20 bp per side and cash at
T-bills stay as in Stage C (DECISIONS 81). Nine variants:
  band (signal-day market-value rank among the day's top 3,000): all, 501-3,000, 1,001-3,000
  priority when signals exceed the free slots: dollar volume (Stage C), smallest first, random
'all x dollar volume' reproduces Stage C's account (checked against its saved equity curves).
Each account is replayed from the per-signal tables (lab/signal_replay.py, stagec_signals_v2), so
no blocks or exits are recomputed. Output: <sample>/<E>__<X>.csv (+ _equity.parquet), one pair at a
time; summary.csv, variants.csv, pairs.csv, missed_summary.json, figures/.
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

from . import figures, research_log as rlog, signal_replay as sr, stage_c_study as sc
from .crsp_study import SAMPLES, load_panel_with_cash, memory_note, spy_stats
from .data import BENCHMARK, Panel
from .engine import CANCELLED, SignalCache
from .grid import _neighbors_and_regions
from .target import target_report

# ---- registered menu (DECISIONS 82): fixed before any result
BANDS: dict[str, tuple[int, int]] = {"all": (1, 3000), "ex500": (501, 3000), "ex1000": (1001, 3000)}
PRIORITIES = ("dv", "small", "random")
BAND_LABELS = {"all": "전체 1–3000위", "ex500": "대형 500 제외 (501–3000위)", "ex1000": "대형 1000 제외 (1001–3000위)"}
PRIORITY_LABELS = {"dv": "거래대금 큰 순", "small": "작은 종목 먼저", "random": "무작위"}
COSTS = (20.0, 40.0)  # bp per side: every variant at both; candidates must pass at 40 too (small caps cost more)
ACCOUNTS = (100_000.0, 1_000_000.0)  # capacity check: order size vs the name's 20-day dollar volume
CAPACITY_LIMIT = 0.05  # share of trades whose order exceeds 5% of the 20-day average dollar volume
_W: dict = {}


def risk_view(eq: pd.Series) -> dict:
    """Information only (the gate is unchanged): volatility and the worst calendar year."""
    v = eq.astype(float)
    r = v.pct_change(fill_method=None).dropna()
    ends = v.groupby(v.index.year).last()
    starts = ends.shift(1)
    starts.iloc[0] = v.iloc[0]  # the first (partial) year from the start of the sample
    yearly = ends / starts - 1.0
    return {"volatility": float(r.std() * np.sqrt(252)), "worst_year": float(yearly.min())}


def capacity(trades: pd.DataFrame, eq: pd.Series, dv20: pd.DataFrame, slots: int) -> dict:
    """Order size at purchase (equity at the previous close / slots) over the name's 20-day average
    dollar volume on the signal day, for each account size."""
    out = {}
    if not len(trades):
        return out
    idx = eq.index
    pos = idx.get_indexer(pd.DatetimeIndex(trades["entry_date"])) - 1
    ok = pos >= 0
    sig_day = idx[np.where(ok, pos, 0)]
    rows = dv20.index.get_indexer(sig_day)
    cols = dv20.columns.get_indexer(trades["symbol"].astype(str))
    good = ok & (rows >= 0) & (cols >= 0)
    dv = np.full(len(trades), np.nan)
    dv[good] = dv20.to_numpy()[rows[good], cols[good]]
    eq_prev = np.where(ok, eq.to_numpy()[np.where(ok, pos, 0)], np.nan)
    for acct in ACCOUNTS:
        share = acct * eq_prev / slots / dv
        share = share[np.isfinite(share)]
        tag = f"{int(acct / 1000)}k"
        out[f"order_share_median_{tag}"] = float(np.median(share)) if len(share) else np.nan
        out[f"order_share_over5_{tag}"] = float((share > CAPACITY_LIMIT).mean()) if len(share) else np.nan
    return out


def pairs() -> list[tuple[str, str]]:
    return [(e, x) for e, x in sc.pairs() if e not in sc.NO_SIGNAL_TABLE]


def variants() -> list[tuple[str, str]]:
    return [(b, p) for b in BANDS for p in PRIORITIES]


def priority_frame(panel: Panel, how: str) -> pd.DataFrame | None:
    """panel.rank for the replay (smaller = bought first); None = the engine's dollar volume."""
    if how == "dv":
        return None
    if how == "small":
        return -panel.extra["cap_rank"].reindex(index=panel.close.index, columns=panel.close.columns).astype("float32")
    return sc.priority_rank(panel, "random")


def band_mask(signals: pd.DataFrame, cap_rank: pd.DataFrame, band: str) -> np.ndarray:
    """Signals whose signal-day market-value rank is inside the band. Cancelled orders stay listed (the
    account meets them as candidates too) only when inside the band."""
    lo, hi = BANDS[band]
    r = sr.signal_values(None, signals, "cap_rank", frame=cap_rank).to_numpy()
    return (r >= lo) & (r <= hi)


# --------------------------------------------------------------------------- workers

def _setup(panel: Panel, signals_dir: str, stagec_dir: str | None = None) -> None:
    _W["panel"] = panel
    _W["cache"] = SignalCache(panel)
    _W["signals"] = Path(signals_dir)
    _W["stagec"] = Path(stagec_dir) if stagec_dir else None
    _W["spy"] = {k: spy_stats(panel, s, e) for k, (s, e, _) in SAMPLES.items()}
    _W["rank"] = {p: priority_frame(panel, p) for p in PRIORITIES}


def _init(panel_dir: str, eligible_path: str, signals_dir: str, stagec_dir: str | None, dry: bool):
    if dry:
        _dry_menu()
    panel = load_panel_with_cash(Path(panel_dir))
    _setup(replace(panel, eligible=pd.read_parquet(eligible_path)), signals_dir, stagec_dir)


def _run_pair(job):
    sample, e, x = job
    panel, cache = _W["panel"], _W["cache"]
    spy = _W["spy"][sample]
    sig_all = pd.read_parquet(_W["signals"] / "signals" / sample / f"{e}__{x}_signals.parquet")
    cap_rank = panel.extra["cap_rank"]
    rows, eqs, logs = [], {}, []
    check = None
    if _W["stagec"] is not None:
        f = _W["stagec"] / "c2" / sample / f"{e}__{x}_equity.parquet"
        check = pd.read_parquet(f) if f.exists() else None
    dv20 = cache.dollar_volume(20)
    for xi, yi in sc.cells(e, x):
        base = replace(sc.config(sample, e, x, xi, yi), max_positions=sc.SLOTS)
        sig = sig_all[(sig_all["xi"] == xi) & (sig_all["yi"] == yi)].reset_index(drop=True)
        for band in BANDS:
            keep = band_mask(sig, cap_rank, band)
            in_band = sig[keep]
            for how in PRIORITIES:
                pp = replace(panel, rank=_W["rank"][how])
                for cost in COSTS:
                    cfg = replace(base, cost_bps=cost)
                    eq, trades = sr.replay(pp, cfg, sig, keep=keep, cache=cache)
                    v = eq.to_numpy(dtype=float)
                    cagr = float(v[-1] ** (252.0 / max(len(v) - 1, 1)) - 1.0)
                    mdd = float((v / np.maximum.accumulate(v) - 1.0).min())
                    t = target_report(eq, panel.close[BENCHMARK], panel.cash, len(trades))
                    row = {"xi": xi, "yi": yi, "band": band, "priority": how, "cost_bps": cost, "cagr": cagr, "mdd": mdd,
                           "n_trades": int(len(trades)), "dd_cagr": t["dd_cagr"], "dd_excess": t["dd_excess"],
                           "window_share": t["window_share"], **risk_view(eq),
                           "trade_expectancy": float(trades["return"].mean()) if len(trades) else np.nan, **spy}
                    row["goal_excess"] = cagr - spy["spy_cagr"]
                    row["goal_pass"] = bool(cagr > spy["spy_cagr"] and abs(mdd) < abs(spy["spy_mdd"]) and len(trades) >= 30)
                    if cost == sc.COMMON["cost_bps"]:
                        # taken vs missed inside the band (per-signal returns at the study cost), and capacity
                        mt = sc.missed_vs_taken(in_band, trades, cap_rank)
                        row.update({k: v_ for k, v_ in mt.items() if k != "signals"})
                        row["band_signals"] = int((in_band["exit_reason"] != CANCELLED).sum())
                        row.update(capacity(trades, eq, dv20, sc.SLOTS))
                        if band == "all" and how == "dv" and check is not None and f"{xi}|{yi}|dv" in check:
                            ref = check[f"{xi}|{yi}|dv"].astype(float).to_numpy()
                            row["stagec_reldiff"] = float(np.nanmax(np.abs(v - ref) / ref))
                    rows.append(row)
                    eqs[f"{xi}|{yi}|{band}|{how}|{int(cost)}"] = eq.astype("float32")
                    logs.append((f"missed_{band}_{how}", cfg, eq))
    return job, pd.DataFrame(rows), pd.DataFrame(eqs), logs


def pair_file(out: Path, sample: str, e: str, x: str) -> Path:
    return out / sample / f"{e}__{x}.csv"


def run_pairs(ex, out: Path, panel: Panel, log) -> None:
    jobs = [(s, e, x) for s in SAMPLES for e, x in pairs() if not pair_file(out, s, e, x).exists()]
    log(f"전략 쌍 × 표본: {len(jobs)}개 남음 (쌍마다 9칸 × 변형 {len(variants())}가지)")
    t0 = time.time()
    for n, (job, rows, eqs, logs) in enumerate(ex.map(_run_pair, jobs, chunksize=1)):
        sample, e, x = job
        f = pair_file(out, sample, e, x)
        f.parent.mkdir(parents=True, exist_ok=True)
        eqs.to_parquet(f.with_name(f"{e}__{x}_equity.parquet"))
        rows.insert(0, "exit", x)
        rows.insert(0, "entry", e)
        rows.to_csv(f, index=False)  # written last: marks the pair done
        # the variants change only which signals are bought, not the rules: one trial per (rules, variant)
        rlog.log_trials([(kind, cfg, panel.tradable_symbols, curve.astype(float)) for kind, cfg, curve in logs])
        per = (time.time() - t0) / (n + 1)
        diff = rows["stagec_reldiff"].max() if "stagec_reldiff" in rows else float("nan")
        log(f"  [{n + 1}/{len(jobs)}] {SAMPLES[sample][2]} {e}×{x}: Stage C 재현 오차 {diff:.1e} · "
            f"남은 예상 {(len(jobs) - n - 1) * per / 3600:.1f}h{memory_note()}")


# --------------------------------------------------------------------------- judging (registered) and summary

def load(out: Path) -> pd.DataFrame:
    parts = []
    for s in SAMPLES:
        for f in sorted((out / s).glob("*__*.csv")):
            g = pd.read_csv(f)
            g["sample"] = s
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


KEYS = ["band", "priority", "cost_bps"]


def regions(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, e, x, b, p, cost), g in c.groupby(["sample", "entry", "exit"] + KEYS):
        g = g.copy()
        g["pass"] = g["goal_pass"].astype(bool)
        _neighbors_and_regions(g)
        rows.append({"sample": s, "entry": e, "exit": x, "band": b, "priority": p, "cost_bps": cost,
                     "pass_cells": int(g["pass"].sum()), "largest_region": int(g["region_size"].max()),
                     "median_excess": g["goal_excess"].median(), "median_cagr": g["cagr"].median(),
                     "median_mdd": g["mdd"].median()})
    return pd.DataFrame(rows)


def summarize(out: Path) -> dict:
    c = load(out)
    c.to_csv(out / "summary.csv", index=False)
    reg = regions(c)
    reg.to_csv(out / "pairs.csv", index=False)
    var = (c.groupby(KEYS + ["sample"])
           .agg(pass_cells=("goal_pass", "sum"), cells=("goal_pass", "size"), median_excess=("goal_excess", "median"),
                median_cagr=("cagr", "median"), median_mdd=("mdd", "median"), median_trades=("n_trades", "median"),
                median_dd_cagr=("dd_cagr", "median"), median_dd_excess=("dd_excess", "median"),
                median_worst_year=("worst_year", "median"), median_volatility=("volatility", "median"))
           .reset_index())
    # risk-accepting view (information only): rank of the variants by CAGR alone
    var["cagr_rank"] = var.groupby(["sample", "cost_bps"])["median_cagr"].rank(ascending=False, method="min")
    var.to_csv(out / "variants.csv", index=False)
    mt_cols = [k for k in ("taken_expectancy", "missed_expectancy", "big_winners_taken_share", "taken_cap_rank_p50",
                           "missed_cap_rank_p50", "order_share_median_100k", "order_share_over5_100k",
                           "order_share_median_1000k", "order_share_over5_1000k") if k in c]
    c[c["cost_bps"] == sc.COMMON["cost_bps"]].groupby(KEYS[:2] + ["sample"])[mt_cols].median().to_csv(out / "taken_missed_capacity.csv")
    wide = reg.pivot_table(index=KEYS + ["entry", "exit"], columns="sample", values=["pass_cells", "largest_region"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide["region_both"] = (wide[[f"largest_region_{s}" for s in SAMPLES]] >= 2).all(axis=1)
    wide["pass_both"] = (wide[[f"pass_cells_{s}" for s in SAMPLES]] > 0).all(axis=1)
    # candidate: a connected passing region in both samples at the study cost, and passes in both at 40 bp
    w = wide.reset_index()
    at_hi = w[w["cost_bps"] == COSTS[-1]].set_index(["band", "priority", "entry", "exit"])["pass_both"]
    cand = w[(w["cost_bps"] == COSTS[0]) & w["region_both"]].copy()
    cand["pass_both_40bp"] = [bool(at_hi.get((r.band, r.priority, r.entry, r.exit), False)) for r in cand.itertuples()]
    cand.to_csv(out / "candidates.csv", index=False)
    final = cand[cand["pass_both_40bp"]]
    summ = {"cells": int(len(c)), "stagec_reproduction_max_reldiff": float(c["stagec_reldiff"].max()) if "stagec_reldiff" in c else None,
            "variants": var.round(4).to_dict("records"), "candidates": cand.to_dict("records"),
            "finalists": final.to_dict("records"), "pass_both_pairs_20bp": int(w[(w["cost_bps"] == COSTS[0]) & w["pass_both"]].shape[0])}
    (out / "missed_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, c, reg)
    return summ


def write_figures(out: Path, c: pd.DataFrame, reg: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    d = out / "figures"
    names = {k: SAMPLES[k][2] for k in SAMPLES}
    made = []
    c = c[c["cost_bps"] == COSTS[0]]
    reg = reg[reg["cost_bps"] == COSTS[0]]
    labels = [f"{BAND_LABELS[b]} · {PRIORITY_LABELS[p]}" for b, p in variants()]
    for stat, title, unit, pct in (("goal_pass", "변형별 통과 칸 수 (729칸 중, 편도 20bp)", "통과 칸", False),
                                    ("goal_excess", "변형별 SPY 대비 초과 CAGR (중앙값)", "초과 CAGR (%p)", True)):
        agg = "sum" if stat == "goal_pass" else "median"
        m = c.groupby(["band", "priority", "sample"])[stat].agg(agg).unstack("sample")
        m = m.reindex(pd.MultiIndex.from_tuples(variants(), names=["band", "priority"]))
        made.append(figures.paired_bars(labels, {names[s]: m[s].to_numpy(dtype=float) for s in SAMPLES if s in m},
                                        title, unit, d / f"variants_{stat}.png", pct=pct))
    best = c.groupby(["band", "priority"])["goal_pass"].sum().idxmax()
    frames = {}
    for s in SAMPLES:
        h = reg[(reg["sample"] == s) & (reg["band"] == best[0]) & (reg["priority"] == best[1])]
        frames[names[s]] = h.pivot(index="entry", columns="exit", values="median_excess")
    made.append(figures.heatmap(frames, f"{BAND_LABELS[best[0]]} · {PRIORITY_LABELS[best[1]]}: 매수 × 매도별 초과 CAGR (9칸 중앙값, %p)",
                                d / "heatmap_best_variant.png", cbar="%p"))
    groups = {}
    for s in SAMPLES:
        g = c[(c["sample"] == s) & (c["priority"].isin(["dv", "small"])) & (c["band"].isin(["all", "ex1000"]))]
        groups[names[s]] = g.assign(group=[f"{BAND_LABELS[b].split(' (')[0]} · {PRIORITY_LABELS[p]}" for b, p in zip(g["band"], g["priority"])])
    spy = {names[s]: (float(c.loc[c["sample"] == s, "spy_cagr"].iloc[0]), float(c.loc[c["sample"] == s, "spy_mdd"].iloc[0]))
           for s in SAMPLES}
    made.append(figures.cagr_mdd(groups, spy, "20종목 계좌 CAGR과 MDD: Stage C(전체·거래대금) 대 중소형 쪽 (별 = SPY)",
                                 d / "cagr_mdd.png"))
    return made


# --------------------------------------------------------------------------- main

def _dry_menu() -> None:
    for k in [k for k in sc.ENTRIES if k not in ("E0", "E1")]:
        del sc.ENTRIES[k]
    for k in [k for k in sc.EXITS if k not in ("X1", "X9")]:
        del sc.EXITS[k]


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--signals", required=True, help="stagec_signals_v2 폴더 (signals/<표본>/…)")
    ap.add_argument("--stagec", default=None, help="Stage C 결과 폴더: 전체·거래대금 변형의 재현을 확인")
    ap.add_argument("--eligible", default=None, help="Stage C의 eligible_daily.parquet (기본: --stagec 안)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="E1 × X1, X9만")
    a = ap.parse_args(argv)
    if a.dry:
        _dry_menu()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rlog.RESULTS_DIR = Path(a.results) if a.results else out / "lab_results"
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    elig = Path(a.eligible) if a.eligible else Path(a.stagec) / "eligible_daily.parquet"
    panel = load_panel_with_cash(Path(a.panel))
    (out / "menu.json").write_text(json.dumps({"bands": BANDS, "priorities": PRIORITIES, "slots": sc.SLOTS,
                                               "common": sc.COMMON, "costs_bps": COSTS, "accounts": ACCOUNTS},
                                              ensure_ascii=False), encoding="utf-8")
    workers = a.workers
    while True:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                     initargs=(a.panel, str(elig), a.signals, a.stagec, a.dry)) as ex:
                run_pairs(ex, out, panel, log)
            break
        except (BrokenProcessPool, MemoryError):
            if workers <= 1:
                raise
            workers = max(1, workers * 3 // 4)
            log(f"메모리가 부족했습니다. 작업 {workers}개로 줄여 이어서 실행합니다(끝난 부분은 건너뜀).")
    summ = summarize(out)
    log(f"완료 · 칸×변형 {summ['cells']} · Stage C 재현 최대 오차 {summ['stagec_reproduction_max_reldiff']} · "
        f"두 표본 영역 후보 {len(summ['candidates'])} · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
