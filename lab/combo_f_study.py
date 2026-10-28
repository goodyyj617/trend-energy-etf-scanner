"""Stage F: combinations of the Stage E screens that helped (DECISIONS 84), resumable.

    python -m lab.combo_f_study --panel DIR --signals DIR --eligible FILE --out DIR [--stage-e DIR] [--workers N] [--dry]

Same signals, universe, hygiene, cells, costs and two views (unlimited / 20-slot) as Stage E
(lab/screen_e_study.py, whose pair runner and summary this module reuses). Each screen is judged
alone exactly as in Stage E (s_efficiency's top 30% ranked among the hygiene-eligible names, not
re-ranked after other screens); a combination buys only stocks passing all of its screens (AND).
Screens run: none, the five singles (re-run as a reproduction check against Stage E) and F1-F5.
Registered verdicts:
  1. vs none        Stage E's 'helps' rule (both samples: >= 60% of cells better, median paired diff > 0)
  2. stacking       combo vs its best single on the same cells, same rule. The best single is chosen
                    per view by the mean over the two samples of its median paired difference vs none
  3. candidates     Stage C-E rule (connected passing region >= 2 cells in both samples, re-checked at 40 bp)
Only F1-F5 are logged as new trials (the singles and none repeat Stage E).
Output: as Stage E, plus stacking.csv, reproduction.json, combo_summary.json and figures/stacking_*.png.
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

from . import figures, research_log as rlog, screen_e_study as se
from .crsp_study import SAMPLES, load_panel_with_cash, memory_note
from .data import Panel
from .missed_study import pairs

# ---- registered menu (DECISIONS 84): fixed before any result
SINGLES = ["x_weak_industry", "x_overextended", "x_ivol", "x_volatility", "s_efficiency"]
COMBOS = {"F1": ["x_weak_industry", "x_overextended"],
          "F2": ["x_weak_industry", "x_overextended", "x_ivol"],
          "F3": ["x_weak_industry", "x_overextended", "s_efficiency"],
          "F4": ["x_weak_industry", "x_overextended", "x_ivol", "s_efficiency"],
          "F5": ["x_weak_industry", "x_overextended", "x_ivol", "x_volatility"]}
SHORT = {"x_weak_industry": "약한 업종 제외", "x_overextended": "장기 과열 제외", "x_ivol": "고유 변동성 제외",
         "x_volatility": "변동성 제외", "s_efficiency": "추세 효율 상위 30%"}
VIEWS = {"unlimited": "expectancy", "account": "cagr"}  # verdict metric per view
REPRO_COLS = ["signals", "expectancy", "win_rate", "cagr", "mdd", "n_trades", "cal_cagr", "cal_mdd"]
KEYS = ["sample", "entry", "exit", "xi", "yi"]


def names() -> list[str]:
    return ["none"] + SINGLES + list(COMBOS)


TINY = {"x_weak_industry": "업종", "x_overextended": "장기 과열", "x_ivol": "고유 변동성", "x_volatility": "변동성",
        "s_efficiency": "추세 효율"}


def labels() -> dict[str, str]:
    """Short names for figures, e.g. 'F3: 업종·장기 과열·추세 효율' (full definitions in the memo)."""
    return {k: f"{k}: " + "·".join(TINY[s] for s in v) for k, v in COMBOS.items()}


def _activate() -> None:
    se._W["names"] = names()
    se._W["labels"] = labels()


def precompute_masks(panel: Panel, eligible: pd.DataFrame, out: Path, log) -> None:
    """Singles exactly as Stage E (each judged alone among the eligible names); combos = AND of singles."""
    (out / "masks").mkdir(parents=True, exist_ok=True)
    for name in ["none"] + SINGLES:
        f = se.mask_file(out, name)
        if not f.exists():
            t = time.time()
            se.screen_mask(panel, eligible, name).to_parquet(f)
            log(f"  스크린 {se.label(name)}: {time.time() - t:.0f}s")
    for name, parts in COMBOS.items():
        f = se.mask_file(out, name)
        if not f.exists():
            m = pd.read_parquet(se.mask_file(out, parts[0]))
            for p in parts[1:]:
                m = m & pd.read_parquet(se.mask_file(out, p))
            m.to_parquet(f)
            log(f"  조합 {se.label(name)}")


def _init(panel_dir: str, eligible: str, signals_dir: str, out: str, dry: bool):
    _activate()
    if dry:
        se._dry_menu()
    panel = load_panel_with_cash(Path(panel_dir))
    se._setup(replace(panel, eligible=pd.read_parquet(eligible)), signals_dir, out)


def run_pairs(ex, out: Path, panel: Panel, log) -> None:
    jobs = [(s, e, x) for s in SAMPLES for e, x in pairs() if not se.pair_file(out, s, e, x).exists()]
    log(f"전략 쌍 × 표본: {len(jobs)}개 남음 (쌍마다 9칸 × 스크린 {len(names())}가지)")
    t0 = time.time()
    for n, (job, rows, eqs, logs) in enumerate(ex.map(se._run_pair, jobs, chunksize=1)):
        sample, e, x = job
        f = se.pair_file(out, sample, e, x)
        f.parent.mkdir(parents=True, exist_ok=True)
        eqs.to_parquet(f.with_name(f"{e}__{x}_equity.parquet"))
        rows.insert(0, "exit", x)
        rows.insert(0, "entry", e)
        rows.to_csv(f, index=False)
        new = {f"screen_{k}" for k in COMBOS}  # none and singles repeat Stage E: not new trials
        rlog.log_trials([(kind, cfg, panel.tradable_symbols, curve.astype(float)) for kind, cfg, curve in logs if kind in new])
        per = (time.time() - t0) / (n + 1)
        log(f"  [{n + 1}/{len(jobs)}] {SAMPLES[sample][2]} {e}×{x} · 남은 예상 {(len(jobs) - n - 1) * per / 3600:.1f}h{memory_note()}")


# --------------------------------------------------------------------------- judging (registered)

def reproduction(c: pd.DataFrame, stage_e: Path) -> dict:
    """Max absolute difference from Stage E's summary for 'none' and the five singles."""
    e = pd.read_csv(stage_e / "summary.csv")
    m = c[c["screen"].isin(["none"] + SINGLES)].merge(e, on=KEYS + ["screen"], suffixes=("", "_e"))
    out = {"rows_compared": int(len(m)), "rows_expected": int(c["screen"].isin(["none"] + SINGLES).sum())}
    for k in REPRO_COLS:
        if k in m and f"{k}_e" in m:
            d = (m[k] - m[f"{k}_e"]).abs()
            both_nan = m[k].isna() & m[f"{k}_e"].isna()
            out[f"max_abs_diff_{k}"] = float(d[~both_nan].max()) if (~both_nan).any() else 0.0
            out[f"nan_mismatch_{k}"] = int((m[k].isna() != m[f"{k}_e"].isna()).sum())
    return out


def best_single(scr: pd.DataFrame, parts: list[str], col: str) -> str:
    """Component with the largest mean (over the two samples) of its median paired difference vs none."""
    g = scr[scr["screen"].isin(parts)].groupby("screen")[f"median_d_{col}"].mean()
    return str(g.idxmax())


def stacking(c: pd.DataFrame, scr: pd.DataFrame) -> pd.DataFrame:
    """Registered verdict 2: each combo vs its best single on the same cells, per view and sample."""
    better = lambda d: float((d.dropna() > 0).mean()) if d.notna().any() else np.nan  # noqa: E731
    rows = []
    for k, parts in COMBOS.items():
        for view, col in VIEWS.items():
            b = best_single(scr, parts, col)
            cols = list(dict.fromkeys([col, "mdd", "cagr", "dd_excess", "signal_share"]))
            j = c[c["screen"] == k].set_index(KEYS)[cols].join(
                c[c["screen"] == b].set_index(KEYS)[cols], rsuffix="_single")
            for s in SAMPLES:
                g = j.loc[s]
                d = g[col] - g[f"{col}_single"]
                rows.append({"combo": k, "view": view, "metric": col, "best_single": b, "sample": s, "cells": int(len(g)),
                             "share_better": better(d), "median_diff": float(d.median()),
                             "median_d_mdd": float((g["mdd"] - g["mdd_single"]).median()),
                             "median_d_dd_excess": float((g["dd_excess"] - g["dd_excess_single"]).median()),
                             "median_signal_share": float(g["signal_share"].median()),
                             "median_signal_share_single": float(g["signal_share_single"].median())})
    t = pd.DataFrame(rows)
    ok = (t["share_better"] >= se.HELP_SHARE) & (t["median_diff"] > 0)
    t["stacks"] = ok.groupby([t["combo"], t["view"]]).transform("all")
    return t


def summarize(out: Path, panel: Panel | None, stage_e: Path | None) -> dict:
    _activate()
    summ = se.summarize(out, panel)  # screens.csv (verdict 1), pairs.csv, candidates.csv (verdict 3), figures
    c = se.load(out)
    scr = pd.read_csv(out / "screens.csv")
    st = stacking(c, scr)
    st.to_csv(out / "stacking.csv", index=False)
    dd = c.groupby(["screen", "sample"]).agg(median_dd_excess=("dd_excess", "median"),
                                             median_window_share=("window_share", "median")).reset_index()
    dd.to_csv(out / "target.csv", index=False)
    res = {"helps_unlimited": summ["helps_unlimited"], "helps_account": summ["helps_account"],
           "stacks": st[st["stacks"]].groupby(["combo", "view"]).size().reset_index()[["combo", "view"]].to_dict("records"),
           "candidates": [r for r in summ["candidates"] if r["screen"] in COMBOS],
           "best_single": st.drop_duplicates(["combo", "view"])[["combo", "view", "best_single"]].to_dict("records")}
    if stage_e is not None and (stage_e / "summary.csv").exists():
        rep = reproduction(c, stage_e)
        (out / "reproduction.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        res["reproduction"] = rep
    (out / "combo_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, st)
    return res


def write_figures(out: Path, st: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    made = []
    for view, title, unit in (("unlimited", "종목 수 제한 없음: 조합 − 가장 좋은 단독 (신호당 수익, 칸별 차이 중앙값)", "%p"),
                              ("account", "20종목 계좌: 조합 − 가장 좋은 단독 (CAGR, 칸별 차이 중앙값)", "%p")):
        g = st[st["view"] == view]
        for stat, t2, u2 in (("median_diff", title, unit), ("share_better", title.split(":")[0] + ": 조합이 나아진 칸 비율 (기준 60%)", "나아진 칸 비율 (%)")):
            m = g.pivot(index="combo", columns="sample", values=stat).reindex(list(COMBOS))
            labs = [f"{k} (대비: {SHORT[g[g['combo'] == k]['best_single'].iloc[0]]})" for k in COMBOS]
            share = stat == "share_better"
            made.append(figures.paired_bars(labs, {SAMPLES[s][2]: m[s].to_numpy(dtype=float) for s in SAMPLES},
                                            t2, u2, out / "figures" / f"stacking_{view}_{stat}.png", pct=True, legend_out=True,
                                            ref=se.HELP_SHARE if share else None, ref_label="기준 60%" if share else ""))
    return made


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--signals", required=True, help="stagec_signals_v2 폴더")
    ap.add_argument("--eligible", required=True, help="Stage C의 eligible_daily.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-e", default=None, help="재현 점검용 Stage E 결과 폴더(summary.csv)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="E1 × X1, X9만")
    ap.add_argument("--summary-only", action="store_true", help="끝난 결과로 판정·그림만 다시")
    a = ap.parse_args(argv)
    _activate()
    if a.dry:
        se._dry_menu()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rlog.RESULTS_DIR = Path(a.results) if a.results else out / "lab_results"
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    panel = load_panel_with_cash(Path(a.panel))
    eligible = pd.read_parquet(a.eligible)
    panel = replace(panel, eligible=eligible)
    (out / "menu.json").write_text(json.dumps({"singles": SINGLES, "combos": COMBOS, "exclude": {k: se.EXCLUDE[k] for k in SINGLES if k in se.EXCLUDE},
                                               "top_share": se.TOP_SHARE, "help_share": se.HELP_SHARE,
                                               "check_cost_bps": se.CHECK_COST_BPS}, ensure_ascii=False), encoding="utf-8")
    if not a.summary_only:
        log("스크린·조합 계산(한 번만)")
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
    se._W["signals"] = Path(a.signals)
    res = summarize(out, panel, Path(a.stage_e) if a.stage_e else None)
    log(f"완료 · 쌓임 효과 {res['stacks']} · 조합 후보 {len(res['candidates'])} · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
