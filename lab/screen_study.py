"""Screen comparison on the CRSP point-in-time universe (DECISIONS 77), resumable.

    python -m lab.screen_study --panel DIR --out DIR [--results DIR] [--workers N] [--dry]

Stage A of the "selection -> entry -> risk management" design: every exclusion screen is paired
with every selection score (lab/screens.py); each pair is a family with a grid of exclusion value
x top-K, run at account level (max 20 positions, risk-based sizing) with a fixed trading layer,
on the two samples of lab/crsp_study.py. The trading layer is fixed on purpose: this stage asks
which screens matter, not which exits.

Output under `out`: families/<sample>/<entry>/<exclude>__<select>.csv (+ _equity.parquet), one
family at a time, so an interrupted run resumes; summary_*.csv, ranks.csv; a study for the app's
저장된 결과 page under `results` (default out/lab_results) and the trial log there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import crsp, research_log as rlog, screens, store
from .crsp_study import SAMPLES, load_panel_with_cash, memory_note, spy_stats
from .data import BENCHMARK, Panel
from .engine import SignalCache, StrategyConfig, run_backtest
from .grid import _neighbors_and_regions
from .metrics import all_metrics, deflated_sharpe
from .target import target_report

# ---- registered menus (DECISIONS 77): exclusion values and the K ladder; selection scores at their defaults
EXCLUDE_MENU: dict[str, list[float]] = {
    "x_downtrend": [-20, -10, 0, 10], "x_drawdown": [20, 30, 40, 50], "x_volatility": [10, 20, 30, 40],
    "x_lottery": [15, 20, 30, 40], "x_overextended": [5, 10, 20, 30], "x_small": [100, 300, 1000, 3000],
    "x_thin": [0.5, 1, 2, 5],
}
K_MENU = [20, 50, 100, 200]
SELECT_KEYS = [k for k in screens.SELECT_SCORES if k not in screens.STAGE_K_SCORES]  # as registered (DECISIONS 77)
ENTRY_VARIANTS: dict[str, dict] = {"always": {"always": None}, "above_ma100": {"above_ma": 100}}
# fixed trading layer: widest trailing stop of 76, an ATR initial stop for risk sizing, 20 slots
TRADING = dict(exits={"trailing_pct": 30, "atr_stop": 3.0}, max_positions=20, cost_bps=20.0, min_price=5.0,
               min_dollar_volume=1e6, sizing="risk", risk_pct=1.0, max_weight=0.10)
ORDER = ["goal_pass", "region_size", "neighbor_survival", "window_share", "goal_excess"]
KEEP = ["cagr", "mdd", "calmar", "sharpe", "n_trades", "exposure", "spy_cagr", "spy_mdd", "dd_weight", "dd_cagr", "mix_cagr",
        "dd_excess", "window_share", "worst_window_excess"]
_W: dict = {}


class ScreenedConfig:
    """A strategy config plus its screen, for the trial log (identity and description)."""

    def __init__(self, cfg: StrategyConfig, scfg: screens.ScreenConfig):
        self.cfg, self.scfg = cfg, scfg

    def to_dict(self) -> dict:
        return {**self.cfg.to_dict(), "screen": self.scfg.to_dict()}

    def describe(self) -> str:
        return f"{self.scfg.describe()} | {self.cfg.describe()}"


def base_config(sample: str) -> StrategyConfig:
    s, e, _ = SAMPLES[sample]
    return StrategyConfig(entries={}, start=s, end=e, **TRADING)


def screened(panel: Panel, cfg: screens.ScreenConfig, cache: dict) -> Panel:
    eligible, rank = screens.monthly_mask(panel, cfg, cache=cache)
    return replace(panel, eligible=eligible, rank=rank)


# --------------------------------------------------------------------------- workers

def _init(panel_dir: str):
    _W["panel"] = load_panel_with_cash(Path(panel_dir))
    _W["cache"] = SignalCache(_W["panel"])
    _W["screens"] = {}
    _W["spy"] = {k: spy_stats(_W["panel"], s, e) for k, (s, e, _) in SAMPLES.items()}


def _run(job):
    tag, sample, scfg_d, entries = job
    panel = _W["panel"]
    scfg = screens.ScreenConfig(**scfg_d)
    sp = screened(panel, scfg, _W["screens"])
    cache = SignalCache(sp).share_from(_W["cache"])
    cfg = replace(base_config(sample), entries=entries)
    res = run_backtest(sp, cfg, cache)
    m = all_metrics(res)
    t = target_report(res.equity, panel.close[BENCHMARK], panel.cash, len(res.trades))
    row = {k: m[k] for k in KEEP if k in m}
    row.update({k: t[k] for k in KEEP if k in t})
    row["goal_excess"] = m["cagr"] - m["spy_cagr"]
    row["goal_pass"] = bool(m["cagr"] >= m["spy_cagr"] and abs(m["mdd"]) <= abs(m["spy_mdd"]) and m["n_trades"] >= 30)
    row["buyable_median"] = float(sp.eligible.loc[res.equity.index].sum(axis=1).median())
    return tag, ScreenedConfig(cfg, scfg), scfg, row, res.equity.astype("float32")


# --------------------------------------------------------------------------- families

def family_jobs(sample: str, entry: str, x_key: str | None, s_key: str) -> list:
    x_vals = EXCLUDE_MENU[x_key] if x_key else [None]
    jobs = []
    for yi, k in enumerate(K_MENU):
        for xi, xv in enumerate(x_vals):
            scfg = {"exclude": {} if x_key is None else {x_key: xv}, "select": {s_key: screens.SELECT_SCORES[s_key].param.default
                    if screens.SELECT_SCORES[s_key].param else None}, "top_k": k}
            jobs.append(((x_key or "none", s_key, xi, yi, xv, k), sample, scfg, ENTRY_VARIANTS[entry]))
    return jobs


def run_families(ex, out: Path, panel: Panel, log) -> None:
    todo = []
    for sample in SAMPLES:
        for entry in ENTRY_VARIANTS:
            d = out / "families" / sample / entry
            d.mkdir(parents=True, exist_ok=True)
            for x_key in [None, *EXCLUDE_MENU]:
                for s_key in SELECT_KEYS:
                    if not (d / f"{x_key or 'none'}__{s_key}.csv").exists():
                        todo.append((sample, entry, x_key, s_key))
    log(f"스크린 조합: {len(todo)}개 남음 (전체 {len(SAMPLES) * len(ENTRY_VARIANTS) * (1 + len(EXCLUDE_MENU)) * len(SELECT_KEYS)})")
    t0, done_cells = time.time(), 0
    for n, (sample, entry, x_key, s_key) in enumerate(todo):
        t = time.time()
        res = list(ex.map(_run, family_jobs(sample, entry, x_key, s_key), chunksize=1))
        cells = pd.DataFrame([{"exclude": tag[0], "select": tag[1], "xi": tag[2], "yi": tag[3], "x": tag[4], "y": tag[5],
                               "screen": scfg.describe(), **row} for tag, _, scfg, row, _ in res])
        cells["pass"] = cells["goal_pass"]
        if len(cells) == len(K_MENU):  # no exclusion: a 1 x 4 grid
            cells["xi"] = 0
        _neighbors_and_regions(cells)
        d = out / "families" / sample / entry
        pd.DataFrame({f"{tag[2]}|{tag[3]}": eq for tag, _, _, _, eq in res}).to_parquet(d / f"{x_key or 'none'}__{s_key}_equity.parquet")
        cells.to_csv(d / f"{x_key or 'none'}__{s_key}.csv", index=False)
        rlog.log_trials([("screen", cfg, panel.tradable_symbols, eq.astype(float)) for _, cfg, _, _, eq in res])
        done_cells += len(cells)
        per = (time.time() - t0) / done_cells
        left = sum(len(K_MENU) * (len(EXCLUDE_MENU[x]) if x else 1) for _, _, x, _ in todo[n + 1:])
        log(f"  [{n + 1}/{len(todo)}] {sample} {entry} {x_key or 'none'} × {s_key}: {time.time() - t:.0f}s, "
            f"통과 {int(cells['pass'].sum())}/{len(cells)} · 칸당 {per:.1f}s · 남은 예상 {left * per / 3600:.1f}h{memory_note()}")


def family_summary(out: Path, sample: str, entry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = out / "families" / sample / entry
    cells = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*__*.csv"))], ignore_index=True)
    rows = []
    for (x, s), g in cells.groupby(["exclude", "select"], sort=False):
        rows.append({"exclude": x, "select": s, "pass_share": g["pass"].mean(), "largest_region": int(g["region_size"].max()),
                     "median_goal_excess": g["goal_excess"].median(), "median_window_share": g["window_share"].median(),
                     "median_cagr": g["cagr"].median(), "median_mdd": g["mdd"].median(), "best_goal_excess": g["goal_excess"].max(),
                     "median_dd_excess": g["dd_excess"].median(), "median_buyable": g["buyable_median"].median()})
    summ = pd.DataFrame(rows).sort_values(["pass_share", "largest_region", "median_goal_excess", "median_window_share"],
                                          ascending=False).reset_index(drop=True)
    summ.to_csv(out / f"summary_{sample}_{entry}.csv", index=False)
    return summ, cells


def label(x: str, s: str) -> str:
    xl = "제외 없음" if x == "none" else screens.EXCLUDE_SCREENS[x].label
    return f"{xl} × {screens.SELECT_SCORES[s].label}"


def save_study(out: Path, panel, meta: dict, n_eff: float, sr_var: float) -> None:
    pct = ["통과 칸 비율", "SPY 대비 초과 수익 (중앙값)", "SPY 대비 초과 수익 (최고)", "CAGR (중앙값)", "MDD (중앙값)", "3년 구간 승률 (중앙값)",
           "혼합 대비 초과 (낙폭 15%, 중앙값)"]
    tables, curves, rank_frames = [], [], {}
    for entry in ENTRY_VARIANTS:
        sums = {k: family_summary(out, k, entry) for k in SAMPLES}
        rk = {k: {(r.exclude, r.select): i + 1 for i, r in enumerate(s.itertuples())} for k, (s, _) in sums.items()}
        ranks = pd.DataFrame([{"exclude": x, "select": s, **{f"rank_{k}": rk[k][(x, s)] for k in rk}} for x, s in rk["full"]])
        ranks["rank_sum"] = ranks[[f"rank_{k}" for k in rk]].sum(axis=1)
        ranks = ranks.sort_values("rank_sum").reset_index(drop=True)
        ranks.to_csv(out / f"ranks_{entry}.csv", index=False)
        rank_frames[entry] = ranks
        for k, (s, cells) in sums.items():
            g = s.head(20)
            tables.append({"title": f"{SAMPLES[k][2]} · 진입 {entry} — 상위 20개 조합", "pct": pct,
                           "frame": pd.DataFrame({"제외 × 선별": [label(x, y) for x, y in zip(g.exclude, g.select)],
                                                  "통과 칸 비율": g.pass_share, "가장 큰 연결 영역": g.largest_region,
                                                  "SPY 대비 초과 수익 (중앙값)": g.median_goal_excess, "SPY 대비 초과 수익 (최고)": g.best_goal_excess,
                                                  "CAGR (중앙값)": g.median_cagr, "MDD (중앙값)": g.median_mdd,
                                                  "3년 구간 승률 (중앙값)": g.median_window_share,
                                                  "혼합 대비 초과 (낙폭 15%, 중앙값)": g.median_dd_excess, "매수 가능 종목 수 (중앙값)": g.median_buyable})})
            named = []
            for r in ranks.head(10).itertuples():
                c = cells[(cells.exclude == r.exclude) & (cells.select == r.select)]
                best = c.sort_values(ORDER, ascending=False).iloc[0]
                eq = pd.read_parquet(out / "families" / k / entry / f"{r.exclude}__{r.select}_equity.parquet")[f"{int(best.xi)}|{int(best.yi)}"]
                named.append((f"{label(r.exclude, r.select)} ({best.screen})", eq))
            from .crsp_study import curve_group
            cg = curve_group(panel, k, named)
            cg["group"] = f"{SAMPLES[k][2]} · 진입 {entry}"
            curves.append(cg)
        top20 = ranks.head(20)
        tables.append({"title": f"두 표본 순위 합 · 진입 {entry} (상위 20)",
                       "frame": pd.DataFrame({"순위": range(1, len(top20) + 1), "제외 × 선별": [label(x, y) for x, y in zip(top20.exclude, top20.select)],
                                              **{f"{SAMPLES[k][2]} 순위": top20[f"rank_{k}"].to_numpy() for k in SAMPLES},
                                              "순위 합": top20.rank_sum.to_numpy()})})
    store.save_study(f"CRSP 개별 종목 (시총 상위 {meta['top_n']}) 스크린 비교 · 제외 × 선별",
                     f"매월 말 제외 필터와 선별 점수로 매수 가능 목록을 만들고(상위 K), 그 안에서 계좌 단위(최대 20종목, 위험 1%/거래, "
                     f"종목 상한 10%, 트레일링 30% + ATR 손절 3, 편도 20bp)로 매매했습니다. 통과 = CAGR ≥ SPY 그리고 |MDD| ≤ |SPY MDD|, 거래 30회 이상.",
                     tables, [f"종목 누적 {meta['symbols']}개 · 상장폐지 포함 · {meta['first']} – {meta['last']}",
                              "조합마다 제외 값 4 × K {20, 50, 100, 200} = 16칸 (제외 없음은 4칸). 선별 점수는 기본값 고정.",
                              "그래프: 순위 합 상위 10개 조합의 대표 칸"],
                     "77", "docs/research/lab/", curves=curves)


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="two screens, two scores, two K values: checks the pipeline")
    a = ap.parse_args(argv)
    if a.dry:
        for k in list(EXCLUDE_MENU):
            if k not in ("x_downtrend", "x_small"):
                del EXCLUDE_MENU[k]
        EXCLUDE_MENU["x_downtrend"] = [0, 10]
        EXCLUDE_MENU["x_small"] = [300, 1000]
        K_MENU[:] = [20, 50]
        SELECT_KEYS[:] = ["s_momentum", "s_high52"]
        ENTRY_VARIANTS.pop("above_ma100")
    out, panel_dir = Path(a.out), Path(a.panel)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(a.results) if a.results else out / "lab_results"
    rlog.RESULTS_DIR = results
    store.RESULTS_DIR = results
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    panel = load_panel_with_cash(panel_dir)
    meta = json.loads((panel_dir / "meta.json").read_text(encoding="utf-8"))
    log(f"패널: 종목 {meta['symbols']}, {meta['first']} – {meta['last']} · 작업 {a.workers}개")
    (out / "menu.json").write_text(json.dumps({"exclude": EXCLUDE_MENU, "k": K_MENU, "select": SELECT_KEYS,
                                               "entries": ENTRY_VARIANTS, "trading": TRADING}, ensure_ascii=False), encoding="utf-8")
    workers = a.workers
    while True:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(panel_dir),)) as ex:
                run_families(ex, out, panel, log)
            break
        except (BrokenProcessPool, MemoryError):  # a worker killed, or one that could not allocate
            if workers <= 1:
                raise
            workers = max(1, workers * 3 // 4)
            log(f"메모리가 부족했습니다. 작업 {workers}개로 줄여 이어서 실행합니다(끝난 조합은 건너뜀).")
    n_eff, rho, _ = rlog.effective_trials()
    _, sr_var = rlog.trial_stats()
    save_study(out, panel, meta, n_eff, sr_var)
    n, _ = rlog.trial_stats()
    log(f"완료 · 시험 {n}회 (유효 {n_eff:.0f}, ρ {rho:.2f}) · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
