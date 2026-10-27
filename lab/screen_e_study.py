"""Stage E: pre-buy screening of the Stage C signals, unlimited and 20-slot (DECISIONS 83), resumable.

    python -m lab.screen_e_study --panel DIR --signals DIR --eligible FILE --out DIR [--workers N] [--dry]

A signal is bought only if its stock passes one screen on the signal day (data up to that close).
Sixteen screens, each alone: none (baseline), seven exclusion screens of lab/screens.py and eight
selection scores keeping the day's top 30% among the eligible names. Entries E1-E9, exits X1-X9
(3 x 3 cells), the daily universe and hygiene are Stage C's (DECISIONS 81); no trend template.
Two views per cell and screen:
  unlimited  every passing signal traded on its own (the per-signal table): count, expectancy,
             win rate, payoff, share of +50% signals
  20-slot    the Stage C account (20 slots, 1/20 of equity, dollar-volume order, 20 bp) replayed
             from the per-signal table with the screen as keep-mask (lab/signal_replay.py)
Output: <sample>/<E>__<X>.csv (+ _equity.parquet), one pair at a time; masks/ (screen masks);
summary.csv, screens.csv, pairs.csv, candidates.csv, screen_summary.json, figures/.
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

from . import figures, research_log as rlog, screens, signal_replay as sr, stage_c_study as sc
from .crsp_study import SAMPLES, load_panel_with_cash, memory_note, spy_stats
from .data import BENCHMARK, Panel
from .engine import CANCELLED, SignalCache
from .grid import _neighbors_and_regions
from .missed_study import pairs
from .target import target_report

# ---- registered menu (DECISIONS 83): fixed before any result
EXCLUDE = {"x_ivol": 20, "x_volatility": 20, "x_lottery": 20, "x_extended": 25, "x_downtrend": 0,
           "x_weak_industry": 30, "x_overextended": 10}
SELECT = ["s_rs", "s_quality", "s_efficiency", "s_smooth", "s_high52", "s_lowvol", "s_dd_shallow", "s_rs_spy"]
TOP_SHARE = 0.30  # a selection score keeps the day's top 30% of the eligible names
CHECK_COST_BPS = 40.0  # candidates are replayed again at 40 bp per side
HELP_SHARE = 0.60  # 'helps': in both samples >= 60% of the 729 cells improve on 'none' and the median paired difference > 0
BIG = 0.5
_W: dict = {}


def screen_names() -> list[str]:
    return ["none"] + list(EXCLUDE) + list(SELECT)


def label(name: str) -> str:
    if name == "none":
        return "스크린 없음"
    s = screens.ALL_SCREENS[name]
    return f"제외: {s.label}" if name in EXCLUDE else f"선별: {s.label} 상위 30%"


def screen_mask(panel: Panel, eligible: pd.DataFrame, name: str) -> pd.DataFrame:
    """True = the stock may be bought on that day under this screen (already within `eligible`)."""
    if name == "none":
        return eligible
    if name in EXCLUDE:
        hit = screens.ALL_SCREENS[name].compute(panel, EXCLUDE[name]).astype("float64").fillna(0.0) > 0.5
        return eligible & ~hit
    p = screens.SELECT_SCORES[name].param
    score = screens.SELECT_SCORES[name].compute(panel, p.default if p else None).where(eligible)
    return eligible & (score.rank(axis=1, pct=True) >= 1.0 - TOP_SHARE)


def mask_file(out: Path, name: str) -> Path:
    return out / "masks" / f"{name}.parquet"


def calendar_portfolio(entry_i: np.ndarray, sale_i: np.ndarray, closed: np.ndarray, sym_j: np.ndarray,
                       O: np.ndarray, C: np.ndarray, cash: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Calendar-time portfolio of every position (no slot limit): each day all open positions in equal
    weight, fully invested while at least one is open, else cash. A position bought at the open of
    day e and sold at the open of day s (or still open at the end) earns close/(open x (1 + cost)) - 1
    on day e, close-to-close on held days and open x (1 - cost)/previous close - 1 on day s. The daily
    re-weighting to equal weight is free (a limit). Returns (equity, positions per day)."""
    n = len(cash)
    if not len(entry_i):
        return np.cumprod(1.0 + cash), np.zeros(n, dtype=int)
    cols, local = np.unique(sym_j, return_inverse=True)
    o, c = O[:, cols].astype(float), C[:, cols].astype(float)
    lc = pd.DataFrame(c).ffill().to_numpy()
    prev = np.vstack([np.full((1, len(cols)), np.nan), lc[:-1]])
    cc = np.where(np.isfinite(c) & np.isfinite(prev), c / prev - 1.0, 0.0)
    end = np.where(closed, sale_i, n)  # still open: held to the last day
    held = np.zeros((n + 1, len(cols)), dtype=np.int32)  # held days e+1 .. end-1 earn close-to-close
    np.add.at(held, (entry_i + 1, local), 1)
    np.add.at(held, (end, local), -1)
    held = np.cumsum(held, axis=0)[:n]
    num = (held * cc).sum(axis=1)
    cnt = held.sum(axis=1).astype(float)
    r_in = np.where(np.isfinite(c[entry_i, local]) & (o[entry_i, local] > 0),
                    c[entry_i, local] / (o[entry_i, local] * (1 + cost)) - 1.0, 0.0)
    np.add.at(num, entry_i, r_in)
    np.add.at(cnt, entry_i, 1)
    ex = closed & (sale_i < n)
    si, sl = sale_i[ex], local[ex]
    r_out = np.where(np.isfinite(o[si, sl]) & np.isfinite(prev[si, sl]), o[si, sl] * (1 - cost) / prev[si, sl] - 1.0, 0.0)
    np.add.at(num, si, r_out)
    np.add.at(cnt, si, 1)
    day = np.where(cnt > 0, num / np.maximum(cnt, 1), cash)
    return np.cumprod(1.0 + day), cnt.astype(int)


def signal_stats(r: np.ndarray, years: float) -> dict:
    win, loss = r[r > 0], r[r <= 0]
    return {"signals": int(len(r)), "signals_per_year": len(r) / years if years > 0 else np.nan,
            "expectancy": float(r.mean()) if len(r) else np.nan, "win_rate": float((r > 0).mean()) if len(r) else np.nan,
            "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else np.nan,
            "big_share": float((r >= BIG).mean()) if len(r) else np.nan}


# --------------------------------------------------------------------------- workers

def _setup(panel: Panel, signals_dir: str, out: str) -> None:
    _W["panel"] = panel
    _W["cache"] = SignalCache(panel)
    _W["signals"] = Path(signals_dir)
    _W["spy"] = {k: spy_stats(panel, s, e) for k, (s, e, _) in SAMPLES.items()}
    _W["masks"] = {n: pd.read_parquet(mask_file(Path(out), n)).to_numpy(dtype=bool) for n in screen_names()}


def _init(panel_dir: str, eligible: str, signals_dir: str, out: str, dry: bool):
    if dry:
        _dry_menu()
    panel = load_panel_with_cash(Path(panel_dir))
    _setup(replace(panel, eligible=pd.read_parquet(eligible)), signals_dir, out)


def _run_pair(job):
    sample, e, x = job
    panel, cache = _W["panel"], _W["cache"]
    spy = _W["spy"][sample]
    s0, s1, _ = SAMPLES[sample]
    years = (pd.Timestamp(s1) - pd.Timestamp(s0)).days / 365.25
    sig_all = pd.read_parquet(_W["signals"] / "signals" / sample / f"{e}__{x}_signals.parquet")
    idx, cols = panel.close.index, panel.close.columns
    rows, eqs, logs = [], {}, []
    win = idx[(idx >= pd.Timestamp(s0)) & (idx <= pd.Timestamp(s1))]
    if sample not in _W.setdefault("arrays", {}):  # window prices for the calendar-time portfolio
        _W["arrays"][sample] = (panel.open.loc[win].to_numpy(dtype="float32"), panel.close.loc[win].to_numpy(dtype="float32"),
                                panel.cash.reindex(win).fillna(0.0).to_numpy())
    O, C, cash = _W["arrays"][sample]
    base_n = None
    for xi, yi in sc.cells(e, x):
        cfg = replace(sc.config(sample, e, x, xi, yi), max_positions=sc.SLOTS)
        sig = sig_all[(sig_all["xi"] == xi) & (sig_all["yi"] == yi)].reset_index(drop=True)
        di = idx.get_indexer(pd.DatetimeIndex(sig["signal_date"]))
        dj = cols.get_indexer(sig["symbol"].astype(str))
        ok = (di >= 0) & (dj >= 0)
        filled = (sig["exit_reason"] != CANCELLED).to_numpy()
        ret = sig["return"].to_numpy(dtype=float)
        canonical = (xi, yi) == sc.canonical(e, x)
        if canonical:
            ent = win.get_indexer(pd.DatetimeIndex(sig["entry_date"]))
            sale = ent + sig["holding_days"].to_numpy(dtype=int)
            closed = sig["closed"].to_numpy(dtype=bool)
        for name in screen_names():
            keep = np.zeros(len(sig), dtype=bool)
            keep[ok] = _W["masks"][name][di[ok], dj[ok]]
            row = {"xi": xi, "yi": yi, "screen": name, **signal_stats(ret[keep & filled], years), **spy}
            if name == "none":
                base_n = row["signals"]
            row["signal_share"] = row["signals"] / base_n if base_n else np.nan  # signals left vs no screen
            eq, trades = sr.replay(panel, cfg, sig, keep=keep, cache=cache)
            v = eq.to_numpy(dtype=float)
            cagr = float(v[-1] ** (252.0 / max(len(v) - 1, 1)) - 1.0)
            mdd = float((v / np.maximum.accumulate(v) - 1.0).min())
            t = target_report(eq, panel.close[BENCHMARK], panel.cash, len(trades))
            row.update({"cagr": cagr, "mdd": mdd, "n_trades": int(len(trades)), "dd_excess": t["dd_excess"],
                        "window_share": t["window_share"], "goal_excess": cagr - spy["spy_cagr"],
                        "bought_share": len(trades) / row["signals"] if row["signals"] else np.nan,
                        "goal_pass": bool(cagr > spy["spy_cagr"] and abs(mdd) < abs(spy["spy_mdd"]) and len(trades) >= 30)})
            if canonical:  # unlimited account: calendar-time portfolio of every passing signal
                m = keep & filled & (ent >= 0)
                ceq, npos = calendar_portfolio(ent[m], sale[m], closed[m], dj[m], O, C, cash, cfg.cost_bps / 10_000.0)
                row.update({"cal_cagr": float(ceq[-1] ** (252.0 / max(len(ceq) - 1, 1)) - 1.0),
                            "cal_mdd": float((ceq / np.maximum.accumulate(ceq) - 1.0).min()),
                            "cal_positions_median": float(np.median(npos[npos > 0])) if (npos > 0).any() else 0.0,
                            "cal_invested_share": float((npos > 0).mean())})
                row["cal_pass"] = bool(row["cal_cagr"] > spy["spy_cagr"] and abs(row["cal_mdd"]) < abs(spy["spy_mdd"]))
            rows.append(row)
            eqs[f"{xi}|{yi}|{name}"] = eq.astype("float32")
            logs.append((f"screen_{name}", cfg, eq))
    return job, pd.DataFrame(rows), pd.DataFrame(eqs), logs


def pair_file(out: Path, sample: str, e: str, x: str) -> Path:
    return out / sample / f"{e}__{x}.csv"


def run_pairs(ex, out: Path, panel: Panel, log) -> None:
    jobs = [(s, e, x) for s in SAMPLES for e, x in pairs() if not pair_file(out, s, e, x).exists()]
    log(f"전략 쌍 × 표본: {len(jobs)}개 남음 (쌍마다 9칸 × 스크린 {len(screen_names())}가지)")
    t0 = time.time()
    for n, (job, rows, eqs, logs) in enumerate(ex.map(_run_pair, jobs, chunksize=1)):
        sample, e, x = job
        f = pair_file(out, sample, e, x)
        f.parent.mkdir(parents=True, exist_ok=True)
        eqs.to_parquet(f.with_name(f"{e}__{x}_equity.parquet"))
        rows.insert(0, "exit", x)
        rows.insert(0, "entry", e)
        rows.to_csv(f, index=False)
        rlog.log_trials([(kind, cfg, panel.tradable_symbols, curve.astype(float)) for kind, cfg, curve in logs])
        per = (time.time() - t0) / (n + 1)
        log(f"  [{n + 1}/{len(jobs)}] {SAMPLES[sample][2]} {e}×{x} · 남은 예상 {(len(jobs) - n - 1) * per / 3600:.1f}h{memory_note()}")


def precompute_masks(panel: Panel, eligible: pd.DataFrame, out: Path, log) -> None:
    (out / "masks").mkdir(parents=True, exist_ok=True)
    for name in screen_names():
        f = mask_file(out, name)
        if not f.exists():
            t = time.time()
            screen_mask(panel, eligible, name).to_parquet(f)
            log(f"  스크린 {label(name)}: {time.time() - t:.0f}s")


# --------------------------------------------------------------------------- judging (registered) and summary

def load(out: Path) -> pd.DataFrame:
    parts = []
    for s in SAMPLES:
        for f in sorted((out / s).glob("*__*.csv")):
            g = pd.read_csv(f)
            g["sample"] = s
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def summarize(out: Path, panel: Panel | None = None) -> dict:
    c = load(out)
    c.to_csv(out / "summary.csv", index=False)
    # paired comparison with 'none' on the same cell (Stage D lesson: never compare separate medians)
    base = c[c["screen"] == "none"].set_index(["sample", "entry", "exit", "xi", "yi"])
    bcols = ["expectancy", "win_rate", "goal_excess", "cagr", "mdd"] + [k for k in ("cal_cagr", "cal_mdd") if k in c]
    c = c.join(base[bcols].rename(columns=lambda k: f"base_{k}"), on=["sample", "entry", "exit", "xi", "yi"])
    c["d_expectancy"] = c["expectancy"] - c["base_expectancy"]
    c["d_excess"] = c["goal_excess"] - c["base_goal_excess"]
    c["d_cagr"] = c["cagr"] - c["base_cagr"]
    c["d_mdd"] = c["mdd"] - c["base_mdd"]  # > 0 = shallower drawdown
    if "cal_cagr" in c:
        c["d_cal_cagr"] = c["cal_cagr"] - c["base_cal_cagr"]
        c["d_cal_mdd"] = c["cal_mdd"] - c["base_cal_mdd"]
    better = lambda d: float((d.dropna() > 0).mean()) if d.notna().any() else np.nan  # noqa: E731
    agg = dict(median_expectancy=("expectancy", "median"), median_d_expectancy=("d_expectancy", "median"),
               share_better_expectancy=("d_expectancy", better), median_d_cagr=("d_cagr", "median"),
               share_better_cagr=("d_cagr", better), median_d_mdd=("d_mdd", "median"), share_better_mdd=("d_mdd", better),
               median_signal_share=("signal_share", "median"), median_bought_share=("bought_share", "median"),
               median_signals_per_year=("signals_per_year", "median"), median_win=("win_rate", "median"),
               median_big=("big_share", "median"), pass_cells=("goal_pass", "sum"), cells=("goal_pass", "size"),
               median_cagr=("cagr", "median"), median_mdd=("mdd", "median"), median_excess=("goal_excess", "median"),
               median_d_excess=("d_excess", "median"))
    if "cal_cagr" in c:
        agg.update(cal_median_cagr=("cal_cagr", "median"), cal_median_mdd=("cal_mdd", "median"),
                   cal_pass_pairs=("cal_pass", "sum"), cal_median_positions=("cal_positions_median", "median"),
                   cal_median_d_cagr=("d_cal_cagr", "median"), cal_share_better_cagr=("d_cal_cagr", better))
    scr = c.groupby(["screen", "sample"]).agg(**agg).reset_index()
    # registered verdict: a screen helps when, in both samples, >= HELP_SHARE of the cells improve and
    # the median paired difference is > 0 (unlimited: per-signal expectancy; 20-slot: CAGR)
    for col, kind in (("expectancy", "unlimited"), ("cagr", "account")):
        ok = (scr[f"share_better_{col}"] >= HELP_SHARE) & (scr[f"median_d_{col}"] > 0)
        scr[f"helps_{kind}"] = scr["screen"].map(ok.groupby(scr["screen"]).all())
    scr.to_csv(out / "screens.csv", index=False)
    reg = []
    for (s, e, x, n), g in c.groupby(["sample", "entry", "exit", "screen"]):
        g = g.copy()
        g["pass"] = g["goal_pass"].astype(bool)
        _neighbors_and_regions(g)
        reg.append({"sample": s, "entry": e, "exit": x, "screen": n, "pass_cells": int(g["pass"].sum()),
                    "largest_region": int(g["region_size"].max()), "median_excess": g["goal_excess"].median(),
                    "median_d_expectancy": g["d_expectancy"].median()})
    reg = pd.DataFrame(reg)
    reg.to_csv(out / "pairs.csv", index=False)
    w = reg.pivot_table(index=["screen", "entry", "exit"], columns="sample", values=["pass_cells", "largest_region"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    cand = w[(w[[f"largest_region_{s}" for s in SAMPLES]] >= 2).all(axis=1)].reset_index()
    if panel is not None and len(cand):  # candidates: replay their passing cells again at 40 bp
        cand["pass_both_40bp"] = [check_cost(out, panel, r.screen, r.entry, r.exit) for r in cand.itertuples()]
    cand.to_csv(out / "candidates.csv", index=False)
    summ = {"cells": int(len(c)), "screens": scr.round(4).to_dict("records"), "candidates": cand.to_dict("records"),
            "helps_unlimited": sorted(scr.loc[scr["helps_unlimited"], "screen"].unique().tolist()),
            "helps_account": sorted(scr.loc[scr["helps_account"], "screen"].unique().tolist()),
            "pass_both_pairs": int((w[[f"pass_cells_{s}" for s in SAMPLES]] > 0).all(axis=1).sum())}
    (out / "screen_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, scr, reg)
    return summ


def check_cost(out: Path, panel: Panel, name: str, e: str, x: str) -> bool:
    """True if the pair has a passing cell in both samples when replayed at CHECK_COST_BPS."""
    if "masks" not in _W:
        _W["masks"] = {n: pd.read_parquet(mask_file(out, n)).to_numpy(dtype=bool) for n in screen_names()}
    passed = {}
    for sample in SAMPLES:
        spy = spy_stats(panel, SAMPLES[sample][0], SAMPLES[sample][1])
        sig_all = pd.read_parquet(_W["signals"] / "signals" / sample / f"{e}__{x}_signals.parquet")
        ok_any = False
        for xi, yi in sc.cells(e, x):
            cfg = replace(sc.config(sample, e, x, xi, yi), max_positions=sc.SLOTS, cost_bps=CHECK_COST_BPS)
            sig = sig_all[(sig_all["xi"] == xi) & (sig_all["yi"] == yi)].reset_index(drop=True)
            di = panel.close.index.get_indexer(pd.DatetimeIndex(sig["signal_date"]))
            dj = panel.close.columns.get_indexer(sig["symbol"].astype(str))
            ok = (di >= 0) & (dj >= 0)
            keep = np.zeros(len(sig), dtype=bool)
            keep[ok] = _W["masks"][name][di[ok], dj[ok]]
            eq, trades = sr.replay(panel, cfg, sig, keep=keep)
            v = eq.to_numpy(dtype=float)
            cagr = float(v[-1] ** (252.0 / max(len(v) - 1, 1)) - 1.0)
            mdd = float((v / np.maximum.accumulate(v) - 1.0).min())
            if cagr > spy["spy_cagr"] and abs(mdd) < abs(spy["spy_mdd"]) and len(trades) >= 30:
                ok_any = True
                break
        passed[sample] = ok_any
    return all(passed.values())


def write_figures(out: Path, scr: pd.DataFrame, reg: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    d = out / "figures"
    names = {k: SAMPLES[k][2] for k in SAMPLES}
    order = screen_names()
    made = []
    for stat, title, unit, pct in (
            ("median_d_expectancy", "종목 수 제한 없음: 스크린이 바꾼 신호당 수익 (칸별 차이 중앙값)", "스크린 없음 대비 (%p)", True),
            ("share_better_expectancy", "종목 수 제한 없음: 신호당 수익이 나아진 칸 비율 (기준 60%)", "나아진 칸 비율 (%)", True),
            ("median_d_cagr", "20종목 계좌: 스크린이 바꾼 CAGR (칸별 차이 중앙값)", "스크린 없음 대비 (%p)", True),
            ("share_better_cagr", "20종목 계좌: CAGR이 나아진 칸 비율 (기준 60%)", "나아진 칸 비율 (%)", True),
            ("pass_cells", "20종목 계좌: 스크린별 통과 칸 수 (729칸 중)", "통과 칸", False),
            ("cal_median_cagr", "종목 수 제한 없음(달력 시간 포트폴리오): CAGR 중앙값 (대표 칸 81개)", "CAGR (%)", True)):
        if stat not in scr:
            continue
        m = scr.pivot(index="screen", columns="sample", values=stat).reindex(order)
        made.append(figures.paired_bars([label(n) for n in order], {names[s]: m[s].to_numpy(dtype=float) for s in SAMPLES},
                                        title, unit, d / f"{stat}.png", pct=pct))
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
    ap.add_argument("--signals", required=True, help="stagec_signals_v2 폴더")
    ap.add_argument("--eligible", required=True, help="Stage C의 eligible_daily.parquet")
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
    panel = load_panel_with_cash(Path(a.panel))
    eligible = pd.read_parquet(a.eligible)
    panel = replace(panel, eligible=eligible)
    (out / "menu.json").write_text(json.dumps({"exclude": EXCLUDE, "select": SELECT, "top_share": TOP_SHARE,
                                               "slots": sc.SLOTS, "common": sc.COMMON, "check_cost_bps": CHECK_COST_BPS},
                                              ensure_ascii=False), encoding="utf-8")
    log("스크린 계산(한 번만)")
    precompute_masks(panel, eligible, out, log)
    workers = a.workers
    while True:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                     initargs=(a.panel, a.eligible, a.signals, str(out), a.dry)) as ex:
                run_pairs(ex, out, panel, log)
            break
        except (BrokenProcessPool, MemoryError):
            if workers <= 1:
                raise
            workers = max(1, workers * 3 // 4)
            log(f"메모리가 부족했습니다. 작업 {workers}개로 줄여 이어서 실행합니다(끝난 부분은 건너뜀).")
    _W["signals"] = Path(a.signals)
    summ = summarize(out, panel)
    log(f"완료 · 칸×스크린 {summ['cells']} · 두 표본 영역 후보 {len(summ['candidates'])} · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
