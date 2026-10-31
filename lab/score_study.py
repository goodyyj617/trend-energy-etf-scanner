"""Representative-filter comparison of the selection scores (DECISIONS 79), resumable.

    python -m lab.score_study --panel DIR --out DIR [--results DIR] [--workers N] [--dry]

Every selection score of lab/screens.py (42, each at its default lookback) is tested on its own:
hygiene screens always on, exclusion {hygiene only, + Minervini trend template} x top K {20, 50,
100} = a 2 x 3 grid per score, for two entries (always, above the 100-day average), two trading
layers (hold: 30% trailing + ATR stop; replace: sell when the stock leaves the monthly list + ATR
stop) and the two samples of lab/crsp_study.py. Account level, no leverage, risk 1% per position.

The screens are computed once (in parallel) and stored as month-end rows under out/screens/, so the
workers only read them. Output: families/<sample>/<entry>/<layer>/<score>.csv (+ _equity.parquet),
written per score so an interrupted run resumes; summary and ranks CSVs; a study for the app.
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

from . import research_log as rlog, screens, store
from .crsp_study import SAMPLES, curve_group, load_panel_with_cash, memory_note, spy_stats
from .data import BENCHMARK, Panel
from .engine import SignalCache, StrategyConfig, run_backtest
from .grid import _neighbors_and_regions
from .metrics import all_metrics
from .screen_study import ScreenedConfig
from .target import target_report

# ---- registered menu (DECISIONS 79)
SCORE_KEYS = [k for k in screens.SELECT_SCORES if k not in screens.STAGE_K_SCORES]  # as registered (41 scores)
EXCLUDE_VARIANTS = {"hygiene": {}, "template": {"tt_fail": 25}}  # added to screens.HYGIENE_DEFAULT
K_MENU = [20, 50, 100]
ENTRY_VARIANTS = {"always": {"always": None}, "above_ma100": {"above_ma": 100}}
COMMON = dict(max_positions=20, cost_bps=20.0, min_price=5.0, min_dollar_volume=1e6, sizing="risk", risk_pct=1.0,
              max_weight=0.10)
LAYERS = {"hold": {"trailing_pct": 30, "atr_stop": 3.0}, "replace": {"left_list": None, "atr_stop": 3.0}}
LAYER_LABELS = {"hold": "보유형 (트레일링 30% + ATR 손절 3)", "replace": "교체형 (목록 이탈 시 매도 + ATR 손절 3)"}
ENTRY_LABELS = {"always": "조건 없음", "above_ma100": "100일선 위"}
ORDER = ["goal_pass", "region_size", "neighbor_survival", "window_share", "goal_excess"]
KEEP = ["cagr", "mdd", "calmar", "sharpe", "n_trades", "exposure", "spy_cagr", "spy_mdd", "dd_weight", "dd_cagr", "mix_cagr",
        "dd_excess", "window_share", "worst_window_excess"]
_W: dict = {}


def screen_keys() -> list[tuple[str, float | None]]:
    """Every (screen, value) the menu needs: computed once before the runs."""
    keys = list(screens.HYGIENE_DEFAULT.items())
    for extra in EXCLUDE_VARIANTS.values():
        keys += [(k, v) for k, v in extra.items() if (k, v) not in keys]
    for s in SCORE_KEYS:
        p = screens.SELECT_SCORES[s].param
        keys.append((s, p.default if p else None))
    return keys


def screen_file(out: Path, key: str, value) -> Path:
    return out / "screens" / f"{key}__{'none' if value is None else f'{float(value):g}'}.parquet"


def score_config(score: str, variant: str, k: int) -> screens.ScreenConfig:
    p = screens.SELECT_SCORES[score].param
    return screens.ScreenConfig(exclude={**screens.HYGIENE_DEFAULT, **EXCLUDE_VARIANTS[variant]},
                                select={score: p.default if p else None}, top_k=k)


def base_config(sample: str, entry: str, layer: str) -> StrategyConfig:
    s, e, _ = SAMPLES[sample]
    return StrategyConfig(entries=dict(ENTRY_VARIANTS[entry]), exits=dict(LAYERS[layer]), start=s, end=e, **COMMON)


# --------------------------------------------------------------------------- workers

def _init(panel_dir: str, out_dir: str | None = None):
    _W["panel"] = load_panel_with_cash(Path(panel_dir))
    _W["cache"] = SignalCache(_W["panel"])
    _W["screens"] = {}
    if out_dir:  # precomputed month-end rows
        for key, value in screen_keys():
            f = screen_file(Path(out_dir), key, value)
            if f.exists():
                _W["screens"][(key, None if value is None else float(value))] = pd.read_parquet(f)
    _W["spy"] = {k: spy_stats(_W["panel"], s, e) for k, (s, e, _) in SAMPLES.items()}


def _compute_screen(job):
    key, value, path = job
    panel = _W["panel"]
    me = screens.month_ends(panel.close.index)
    frame = screens.ALL_SCREENS[key].compute(panel, value).loc[me]
    frame.astype("float32").to_parquet(path)
    return key


def _run(job):
    tag, sample, entry, layer, score, variant, k = job
    panel = _W["panel"]
    scfg = score_config(score, variant, k)
    eligible, rank = screens.monthly_mask(panel, scfg, cache=_W["screens"])
    sp = replace(panel, eligible=eligible, rank=rank)
    cache = SignalCache(sp).share_from(_W["cache"])
    cfg = base_config(sample, entry, layer)
    res = run_backtest(sp, cfg, cache)
    m = all_metrics(res)
    t = target_report(res.equity, panel.close[BENCHMARK], panel.cash, len(res.trades))
    row = {key: m[key] for key in KEEP if key in m}
    row.update({key: t[key] for key in KEEP if key in t})
    row["goal_excess"] = m["cagr"] - m["spy_cagr"]
    row["goal_pass"] = bool(m["cagr"] >= m["spy_cagr"] and abs(m["mdd"]) <= abs(m["spy_mdd"]) and m["n_trades"] >= 30)
    row["buyable_median"] = float(eligible.loc[res.equity.index].sum(axis=1).median())
    return tag, ScreenedConfig(cfg, scfg), row, res.equity.astype("float32")


# --------------------------------------------------------------------------- runs

def precompute(ex, out: Path, log) -> None:
    (out / "screens").mkdir(parents=True, exist_ok=True)
    todo = [(k, v, str(screen_file(out, k, v))) for k, v in screen_keys() if not screen_file(out, k, v).exists()]
    log(f"스크린 계산: {len(todo)}개 남음 (전체 {len(screen_keys())})")
    for n, key in enumerate(ex.map(_compute_screen, todo, chunksize=1)):
        log(f"  [{n + 1}/{len(todo)}] {screens.ALL_SCREENS[key].label}")


def family_dir(out: Path, sample: str, entry: str, layer: str) -> Path:
    return out / "families" / sample / entry / layer


def run_groups(ex, out: Path, panel: Panel, log) -> None:
    groups = [(s, e, l) for s in SAMPLES for e in ENTRY_VARIANTS for l in LAYERS]
    t0, done = time.time(), 0
    total_left = sum(1 for s, e, l in groups for sc in SCORE_KEYS if not (family_dir(out, s, e, l) / f"{sc}.csv").exists())
    log(f"점수 × 조합: {total_left}개 남음 (전체 {len(groups) * len(SCORE_KEYS)})")
    for sample, entry, layer in groups:
        d = family_dir(out, sample, entry, layer)
        d.mkdir(parents=True, exist_ok=True)
        todo = [sc for sc in SCORE_KEYS if not (d / f"{sc}.csv").exists()]
        if not todo:
            continue
        jobs = [((sc, xi, yi, v, k), sample, entry, layer, sc, v, k)
                for sc in todo for xi, v in enumerate(EXCLUDE_VARIANTS) for yi, k in enumerate(K_MENU)]
        t = time.time()
        res = list(ex.map(_run, jobs, chunksize=1))
        for sc in todo:
            part = [r for r in res if r[0][0] == sc]
            cells = pd.DataFrame([{"score": tag[0], "xi": tag[1], "yi": tag[2], "exclude": tag[3], "k": tag[4],
                                   "tier": screens.SELECT_SCORES[sc].tier, **row} for tag, _, row, _ in part])
            cells["pass"] = cells["goal_pass"]
            _neighbors_and_regions(cells)
            pd.DataFrame({f"{tag[1]}|{tag[2]}": eq for tag, _, _, eq in part}).to_parquet(d / f"{sc}_equity.parquet")
            cells.to_csv(d / f"{sc}.csv", index=False)
            rlog.log_trials([("score", cfg, panel.tradable_symbols, eq.astype(float)) for _, cfg, _, eq in part])
        done += len(todo)
        per = (time.time() - t0) / done
        passing = sum(int(r[2]["goal_pass"]) for r in res)
        log(f"  {sample} · {ENTRY_LABELS[entry]} · {LAYER_LABELS[layer]}: 점수 {len(todo)}개, {time.time() - t:.0f}s, "
            f"통과 {passing}/{len(res)} · 남은 예상 {(total_left - done) * per / 3600:.1f}h{memory_note()}")


# --------------------------------------------------------------------------- summary and study

def summaries(out: Path) -> pd.DataFrame:
    rows = []
    for f in (out / "families").glob("*/*/*/*.csv"):
        g = pd.read_csv(f)
        layer, entry, sample = f.parent.name, f.parent.parent.name, f.parent.parent.parent.name
        rows.append({"sample": sample, "entry": entry, "layer": layer, "score": f.stem, "tier": int(g["tier"].iloc[0]),
                     "pass_share": g["pass"].mean(), "largest_region": int(g["region_size"].max()),
                     "median_goal_excess": g["goal_excess"].median(), "best_goal_excess": g["goal_excess"].max(),
                     "median_cagr": g["cagr"].median(), "median_mdd": g["mdd"].median(),
                     "median_window_share": g["window_share"].median(), "median_dd_excess": g["dd_excess"].median()})
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    return s


def ranks(s: pd.DataFrame) -> pd.DataFrame:
    keys = ["pass_share", "largest_region", "median_goal_excess", "median_window_share"]
    parts = []
    for (entry, layer), g in s.groupby(["entry", "layer"]):
        out = pd.DataFrame(index=pd.Index(sorted(g["score"].unique()), name="score"))
        for smp in SAMPLES:
            sub = g[g["sample"] == smp].sort_values(keys, ascending=False)
            out[f"rank_{smp}"] = pd.Series(range(1, len(sub) + 1), index=sub["score"])
            out[f"pass_{smp}"] = sub.set_index("score")["pass_share"]
            out[f"excess_{smp}"] = sub.set_index("score")["median_goal_excess"]
        out["rank_sum"] = out[[f"rank_{k}" for k in SAMPLES]].sum(axis=1)
        out["both_pass"] = (out[[f"pass_{k}" for k in SAMPLES]] > 0).all(axis=1)
        out["entry"], out["layer"] = entry, layer
        parts.append(out.reset_index())
    rk = pd.concat(parts, ignore_index=True).sort_values(["entry", "layer", "both_pass", "rank_sum"],
                                                       ascending=[True, True, False, True])
    return rk


def save_study(out: Path, panel, meta: dict) -> None:
    s = summaries(out)
    rk = ranks(s)
    rk.to_csv(out / "ranks.csv", index=False)
    pct = ["1993–2023 통과 칸 비율", "2017–2023 통과 칸 비율", "1993–2023 SPY 대비 초과 (중앙값)", "2017–2023 SPY 대비 초과 (중앙값)"]
    tables, curves = [], []
    lab = lambda k: screens.SELECT_SCORES[k].label  # noqa: E731
    tier = s.groupby(["tier", "sample"])["pass_share"].mean().unstack("sample")
    tables.append({"title": "우선순위 계층별 평균 통과 칸 비율", "pct": [SAMPLES[k][2] for k in SAMPLES],
                   "frame": pd.DataFrame({"계층": [screens.TIER_LABELS[i] for i in tier.index],
                                          **{SAMPLES[k][2]: tier[k].to_numpy() for k in SAMPLES if k in tier}})})
    for (entry, layer), g in rk.groupby(["entry", "layer"], sort=False):
        top = g.head(20)
        tables.append({"title": f"진입 {ENTRY_LABELS[entry]} · {LAYER_LABELS[layer]} — 상위 20개 점수", "pct": pct,
                       "frame": pd.DataFrame({"선별 점수": [lab(k) for k in top["score"]],
                                              "계층": [screens.SELECT_SCORES[k].tier for k in top["score"]],
                                              "두 기간 모두 통과 칸 있음": top["both_pass"].to_numpy(),
                                              "1993–2023 통과 칸 비율": top["pass_full"].to_numpy(),
                                              "2017–2023 통과 칸 비율": top["pass_recent"].to_numpy(),
                                              "1993–2023 SPY 대비 초과 (중앙값)": top["excess_full"].to_numpy(),
                                              "2017–2023 SPY 대비 초과 (중앙값)": top["excess_recent"].to_numpy(),
                                              "순위 합": top["rank_sum"].to_numpy()})})
        for smp in SAMPLES:
            named = []
            for sc in top["score"].head(8):
                c = pd.read_csv(family_dir(out, smp, entry, layer) / f"{sc}.csv")
                best = c.sort_values(ORDER, ascending=False).iloc[0]
                eq = pd.read_parquet(family_dir(out, smp, entry, layer) / f"{sc}_equity.parquet")[f"{int(best.xi)}|{int(best.yi)}"]
                named.append((f"{lab(sc)} ({best.exclude}, 상위 {int(best.k)})", eq))
            cg = curve_group(panel, smp, named)
            cg["group"] = f"{SAMPLES[smp][2]} · {ENTRY_LABELS[entry]} · {LAYER_LABELS[layer]}"
            curves.append(cg)
    store.save_study(f"CRSP 개별 종목 (시총 상위 {meta['top_n']}) 선별 점수 대표 필터 비교",
                     f"선별 점수 {len(SCORE_KEYS)}개를 기본값으로 하나씩, 위생 필터 공통 + (제외 없음 / 추세 템플릿) × 상위 K {K_MENU}로 "
                     "계좌 단위(최대 20종목, 위험 1%/거래, 종목 상한 10%, 편도 20bp)에서 비교했습니다. "
                     "통과 = CAGR ≥ SPY 그리고 |MDD| ≤ |SPY MDD|, 거래 30회 이상.",
                     tables, [f"종목 누적 {meta['symbols']}개 · 상장폐지 포함 · {meta['first']} – {meta['last']}",
                              "점수마다 6칸(제외 2 × K 3). 진입 2가지 × 매매층 2가지 × 표본 2개.",
                              "그래프: 각 묶음의 순위 합 상위 8개 점수의 대표 칸"],
                     "79", "docs/research/lab/", curves=curves)


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="three scores, one entry: checks the pipeline")
    a = ap.parse_args(argv)
    if a.dry:
        SCORE_KEYS[:] = ["s_rs", "s_clv", "s_small"]
        ENTRY_VARIANTS.pop("above_ma100")
        K_MENU[:] = [20, 50]
    out, panel_dir = Path(a.out), Path(a.panel)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(a.results) if a.results else out / "lab_results"
    rlog.RESULTS_DIR = results
    store.RESULTS_DIR = results
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    panel = load_panel_with_cash(panel_dir)
    meta = json.loads((panel_dir / "meta.json").read_text(encoding="utf-8"))
    if "sic" not in panel.info.columns:
        log("주의: 이 패널에는 업종 코드(SIC)가 없어 업종 점수는 비게 됩니다. 패널을 다시 만들면 됩니다.")
    log(f"패널: 종목 {meta['symbols']}, {meta['first']} – {meta['last']} · 작업 {a.workers}개")
    (out / "menu.json").write_text(json.dumps({"scores": SCORE_KEYS, "exclude": EXCLUDE_VARIANTS, "hygiene": screens.HYGIENE_DEFAULT,
                                               "k": K_MENU, "entries": ENTRY_VARIANTS, "layers": LAYERS, "common": COMMON},
                                              ensure_ascii=False, default=str), encoding="utf-8")
    workers = a.workers
    while True:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(panel_dir),)) as ex:
                precompute(ex, out, log)
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(panel_dir), str(out))) as ex:
                run_groups(ex, out, panel, log)
            break
        except (BrokenProcessPool, MemoryError):
            if workers <= 1:
                raise
            workers = max(1, workers * 3 // 4)
            log(f"메모리가 부족했습니다. 작업 {workers}개로 줄여 이어서 실행합니다(끝난 부분은 건너뜀).")
    save_study(out, panel, meta)
    n, _ = rlog.trial_stats()
    n_eff, rho, _ = rlog.effective_trials()
    log(f"완료 · 시험 {n}회 (유효 {n_eff:.0f}, ρ {rho:.2f}) · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
