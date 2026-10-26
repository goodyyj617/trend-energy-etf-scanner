"""Stage B: blends of the consistent top selection scores x a Sobol sample of the trading layer (DECISIONS 80).

    python -m lab.combo_study --panel DIR --out DIR [--results DIR] [--workers N] [--dry]

Selection: every 2- and 3-combination of POOL (the scores that ranked high in both samples of
DECISIONS 79), each at its default lookback and combined by the mean percentile rank of
screens.monthly_mask, plus 12-1 momentum alone (does the replace layer without an ATR stop revive
it?). Trading layer: the same N_POINTS Sobol points (7 variables) for every combination and both
samples of lab/crsp_study.py: exclusion, top K, layer (hold / replace), trailing %, ATR stop,
risk %, maximum positions. Account level, no leverage, entry always.

Risk sizing needs an initial stop distance; a point without an ATR stop has none, so it is sized
with equal slots (1 / max positions) instead and its risk % is not used (registered with the menu).

The screens are computed once and stored as month-end rows under out/screens/. Output:
combos/<sample>/<combo>.csv (+ _equity.parquet), written per combination so an interrupted run
resumes; summary.csv, ranks.csv, layers.csv, combo_summary.json (PBO), figures/*.png; a study for the app.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import explore, figures, research_log as rlog, screens, store
from .crsp_study import SAMPLES, curve_group, load_panel_with_cash, memory_note
from .data import BENCHMARK, Panel
from .engine import SignalCache, StrategyConfig, run_backtest
from .metrics import _month_end_values, all_metrics
from .screen_study import ScreenedConfig
from .target import target_report

# ---- registered menu (DECISIONS 80): fixed before any result
POOL = ["s_dollar_volume", "s_efficiency", "s_large", "s_smooth", "s_accum", "s_obv_slope"]
COMBOS = [list(c) for r in (2, 3) for c in combinations(POOL, r)] + [["s_momentum"]]
N_POINTS = 32
EXCLUDE_VARIANTS = {"hygiene": {}, "template": {"tt_fail": 25}}  # added to screens.HYGIENE_DEFAULT
K_MENU = [20, 50, 100]
LAYER_MENU = ["hold", "replace"]
TRAIL_RANGE = (15, 40)  # hold layer only, whole %
ATR_MENU = [None, 2.0, 3.0, 4.0]
RISK_RANGE = (0.5, 2.0)  # % of equity at the initial stop, step 0.1
MAX_POS_MENU = [10, 15, 20, 30]
SPACE = ["exclude", "k", "layer", "trailing_pct", "atr_stop", "risk_pct", "max_positions"]
COMMON = dict(cost_bps=20.0, min_price=5.0, min_dollar_volume=1e6, max_weight=0.10)
ENTRIES = {"always": None}
LAYER_LABELS = {"hold": "보유형 (트레일링)", "replace": "교체형 (목록 이탈 시 매도)"}
KEEP = ["cagr", "mdd", "calmar", "sharpe", "n_trades", "exposure", "spy_cagr", "spy_mdd", "dd_weight", "dd_cagr", "mix_cagr",
        "dd_excess", "window_share", "worst_window_excess"]
_W: dict = {}


def combo_id(combo: list[str]) -> str:
    return "+".join(k.removeprefix("s_") for k in combo)


def combo_label(combo: list[str]) -> str:
    return " + ".join(screens.SELECT_SCORES[k].label for k in combo)


def points() -> np.ndarray:
    """The Sobol points shared by every combination and sample."""
    return explore.sobol(N_POINTS, len(SPACE))


def _pick(u: float, options: list):
    return options[min(int(u * len(options)), len(options) - 1)]


def point_settings(u: np.ndarray) -> dict:
    """One point of [0, 1)^7 -> the trading-layer settings (None = not used at this point)."""
    layer = _pick(u[2], LAYER_MENU)
    atr = _pick(u[4], ATR_MENU)
    lo, hi = TRAIL_RANGE
    rlo, rhi = RISK_RANGE
    return {"exclude": _pick(u[0], list(EXCLUDE_VARIANTS)), "k": int(_pick(u[1], K_MENU)), "layer": layer,
            "trailing_pct": float(round(lo + u[3] * (hi - lo))) if layer == "hold" else None,
            "atr_stop": atr,
            "risk_pct": float(round(rlo + u[5] * (rhi - rlo), 1)) if atr is not None else None,
            "max_positions": int(_pick(u[6], MAX_POS_MENU))}


def strategy_config(p: dict, sample: str) -> StrategyConfig:
    s, e, _ = SAMPLES[sample]
    exits = {"trailing_pct": p["trailing_pct"]} if p["layer"] == "hold" else {"left_list": None}
    if p["atr_stop"] is not None:
        exits["atr_stop"] = p["atr_stop"]
    sizing = dict(sizing="risk", risk_pct=p["risk_pct"]) if p["atr_stop"] is not None else dict(sizing="equal")
    return StrategyConfig(entries=dict(ENTRIES), exits=exits, start=s, end=e, max_positions=p["max_positions"],
                          **sizing, **COMMON)


def screen_config(combo: list[str], p: dict) -> screens.ScreenConfig:
    sel = {}
    for k in combo:
        par = screens.SELECT_SCORES[k].param
        sel[k] = par.default if par else None
    return screens.ScreenConfig(exclude={**screens.HYGIENE_DEFAULT, **EXCLUDE_VARIANTS[p["exclude"]]}, select=sel,
                                top_k=p["k"])


def screen_keys() -> list[tuple[str, float | None]]:
    """Every (screen, value) the menu needs: computed once before the runs."""
    keys = list(screens.HYGIENE_DEFAULT.items())
    for extra in EXCLUDE_VARIANTS.values():
        keys += [(k, v) for k, v in extra.items() if (k, v) not in keys]
    for s in sorted({k for c in COMBOS for k in c}):
        p = screens.SELECT_SCORES[s].param
        keys.append((s, p.default if p else None))
    return keys


def screen_file(out: Path, key: str, value) -> Path:
    return out / "screens" / f"{key}__{'none' if value is None else f'{float(value):g}'}.parquet"


# --------------------------------------------------------------------------- workers

def _setup(panel: Panel, out_dir: str | None = None) -> None:
    _W["panel"] = panel
    _W["cache"] = SignalCache(panel)
    _W["screens"] = {}
    if out_dir:  # precomputed month-end rows
        for key, value in screen_keys():
            f = screen_file(Path(out_dir), key, value)
            if f.exists():
                _W["screens"][(key, None if value is None else float(value))] = pd.read_parquet(f)


def _init(panel_dir: str, out_dir: str | None = None, dry: bool = False):
    if dry:  # the worker imports the full menu
        _dry_menu()
    _setup(load_panel_with_cash(Path(panel_dir)), out_dir)


def _compute_screen(job):
    key, value, path = job
    panel = _W["panel"]
    me = screens.month_ends(panel.close.index)
    frame = screens.ALL_SCREENS[key].compute(panel, value).loc[me]
    frame.astype("float32").to_parquet(path)
    return key


def _run(job):
    tag, sample, combo, i, p = job
    panel = _W["panel"]
    scfg = screen_config(combo, p)
    eligible, rank = screens.monthly_mask(panel, scfg, cache=_W["screens"])
    sp = replace(panel, eligible=eligible, rank=rank)
    cache = SignalCache(sp).share_from(_W["cache"])
    cfg = strategy_config(p, sample)
    res = run_backtest(sp, cfg, cache)
    m = all_metrics(res)
    t = target_report(res.equity, panel.close[BENCHMARK], panel.cash, len(res.trades))
    row = {"i": i, **p}
    row.update({key: m[key] for key in KEEP if key in m})
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


def combo_dir(out: Path, sample: str) -> Path:
    return out / "combos" / sample


def run_combos(ex, out: Path, panel: Panel, log, chunk: int = 4) -> None:
    """Every combination x point x sample; one CSV per (sample, combination), so a rerun resumes."""
    pts = [point_settings(u) for u in points()]
    total_left = sum(1 for s in SAMPLES for c in COMBOS if not (combo_dir(out, s) / f"{combo_id(c)}.csv").exists())
    log(f"조합 × 표본: {total_left}개 남음 (전체 {len(SAMPLES) * len(COMBOS)}, 각 {len(pts)}점)")
    t0, done = time.time(), 0
    for sample in SAMPLES:
        d = combo_dir(out, sample)
        d.mkdir(parents=True, exist_ok=True)
        todo = [c for c in COMBOS if not (d / f"{combo_id(c)}.csv").exists()]
        for n in range(0, len(todo), chunk):
            part = todo[n:n + chunk]
            jobs = [((combo_id(c), i), sample, c, i, p) for c in part for i, p in enumerate(pts)]
            t = time.time()
            res = list(ex.map(_run, jobs, chunksize=1))
            for c in part:
                cid = combo_id(c)
                mine = [r for r in res if r[0][0] == cid]
                cells = pd.DataFrame([{"combo": cid, "size": len(c), **row} for _, _, row, _ in mine])
                pd.DataFrame({str(tag[1]): eq for tag, _, _, eq in mine}).to_parquet(d / f"{cid}_equity.parquet")
                cells.to_csv(d / f"{cid}.csv", index=False)
                rlog.log_trials([("combo", cfg, panel.tradable_symbols, eq.astype(float)) for _, cfg, _, eq in mine])
            done += len(part)
            per = (time.time() - t0) / done
            passing = sum(int(r[2]["goal_pass"]) for r in res)
            log(f"  {SAMPLES[sample][2]}: 조합 {n + len(part)}/{len(todo)}, {time.time() - t:.0f}s, 통과 {passing}/{len(res)}"
                f" · 남은 예상 {(total_left - done) * per / 3600:.1f}h{memory_note()}")


# --------------------------------------------------------------------------- judging (registered) and study

def load_cells(out: Path) -> pd.DataFrame:
    parts = []
    for sample in SAMPLES:
        for f in sorted(combo_dir(out, sample).glob("*.csv")):
            g = pd.read_csv(f)
            g["sample"] = sample
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def summaries(cells: pd.DataFrame) -> pd.DataFrame:
    s = (cells.groupby(["combo", "sample"])
         .agg(size=("size", "first"), pass_share=("goal_pass", "mean"), median_goal_excess=("goal_excess", "median"),
              best_goal_excess=("goal_excess", "max"), median_cagr=("cagr", "median"), median_mdd=("mdd", "median"),
              median_dd_excess=("dd_excess", "median"), median_window_share=("window_share", "median"))
         .reset_index())
    return s


def ranks(s: pd.DataFrame) -> pd.DataFrame:
    """Per sample: pass share of the points, then median excess. Rank = sum of the two sample ranks;
    a combination with passes in one sample only is listed below every both-sample one."""
    out = pd.DataFrame(index=pd.Index(sorted(s["combo"].unique()), name="combo"))
    for smp in SAMPLES:
        sub = s[s["sample"] == smp].sort_values(["pass_share", "median_goal_excess"], ascending=False)
        out[f"rank_{smp}"] = pd.Series(range(1, len(sub) + 1), index=sub["combo"])
        for col in ("pass_share", "median_goal_excess", "median_mdd", "median_dd_excess", "median_window_share"):
            out[f"{col}_{smp}"] = sub.set_index("combo")[col]
    out["rank_sum"] = out[[f"rank_{k}" for k in SAMPLES]].sum(axis=1)
    out["both_pass"] = (out[[f"pass_share_{k}" for k in SAMPLES]] > 0).all(axis=1)
    return out.reset_index().sort_values(["both_pass", "rank_sum"], ascending=[False, True])


def both_sample_passes(cells: pd.DataFrame) -> pd.DataFrame:
    """Points (combination, i) that pass in both samples, with their layer settings."""
    p = cells.pivot_table(index=["combo", "i"], columns="sample", values="goal_pass", aggfunc="first")
    ok = p.reindex(columns=list(SAMPLES)).eq(True).all(axis=1)
    first = cells[cells["sample"] == next(iter(SAMPLES))].set_index(["combo", "i"])
    return first.loc[ok[ok].index, SPACE].reset_index()


def layer_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Pass share and median excess by layer x ATR stop, per sample (over every combination)."""
    c = cells.assign(atr=cells["atr_stop"].map(lambda v: "없음" if pd.isna(v) else f"{v:g}"))
    t = (c.groupby(["layer", "atr", "sample"]).agg(pass_share=("goal_pass", "mean"), median_goal_excess=("goal_excess", "median"))
         .unstack("sample"))
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    return t.reset_index()


def pbo_all(out: Path) -> dict:
    """PBO of 'pick the best run' over every run of each sample (information only)."""
    res = {}
    for smp in SAMPLES:
        cols = {}
        for f in sorted(combo_dir(out, smp).glob("*_equity.parquet")):
            eq = pd.read_parquet(f)
            for c in eq:
                cols[f"{f.stem.removesuffix('_equity')}|{c}"] = (
                    _month_end_values(eq[c].dropna().astype(float)).pct_change(fill_method=None).dropna())
        res[smp] = explore.pbo(pd.DataFrame(cols), blocks=12) if len(cols) > 1 else None
    return res


def write_figures(out: Path, cells: pd.DataFrame, rk: pd.DataFrame, lt: pd.DataFrame) -> list[Path]:
    """The memo figures (aggregates only) under out/figures/; nothing when matplotlib is missing."""
    if not figures.available():
        return []
    d = out / "figures"
    by_id = {combo_id(c): c for c in COMBOS}
    labels = [combo_label(by_id[c]) if c in by_id else c for c in rk["combo"]]
    names = {k: SAMPLES[k][2] for k in SAMPLES}
    made = [figures.paired_bars(labels, {names[k]: rk[f"median_goal_excess_{k}"].to_numpy() for k in SAMPLES},
                                "선별 조합별 SPY 대비 초과 CAGR (32점 중앙값)", "SPY 대비 초과 CAGR (%p)", d / "combo_excess.png")]
    if len(SAMPLES) == 2:
        a, b = list(SAMPLES)
        rho = rk[f"median_goal_excess_{a}"].rank().corr(rk[f"median_goal_excess_{b}"].rank())
        made.append(figures.scatter_labeled(rk[f"median_goal_excess_{a}"].to_numpy(), rk[f"median_goal_excess_{b}"].to_numpy(),
                                            labels, "두 표본의 조합 순위 일관성", f"{names[a]} 초과 (중앙값, %p)",
                                            f"{names[b]} 초과 (중앙값, %p)", d / "combo_consistency.png",
                                            note=f"순위 상관(Spearman) {rho:.2f}"))
    lab = [f"{LAYER_LABELS[l]} · ATR {a}" for l, a in zip(lt["layer"], lt["atr"])]
    made.append(figures.paired_bars(lab, {names[k]: lt[f"pass_share_{k}"].to_numpy() for k in SAMPLES},
                                    "매매층 × ATR 손절별 통과 점 비율 (모든 조합)", "통과 점 비율 (%)", d / "layer_pass.png"))
    groups = {names[k]: cells[cells["sample"] == k].assign(group=lambda g: g["layer"].map(LAYER_LABELS)) for k in SAMPLES}
    spy = {names[k]: (float(cells.loc[cells["sample"] == k, "spy_cagr"].iloc[0]),
                      float(cells.loc[cells["sample"] == k, "spy_mdd"].iloc[0])) for k in SAMPLES}
    made.append(figures.cagr_mdd(groups, spy, "모든 시험의 CAGR과 MDD (별 = SPY)", d / "cagr_mdd.png"))
    return made


def save_study(out: Path, panel, meta: dict) -> dict:
    cells = load_cells(out)
    s = summaries(cells)
    s.to_csv(out / "summary.csv", index=False)
    rk = ranks(s)
    rk.to_csv(out / "ranks.csv", index=False)
    lt = layer_table(cells)
    lt.to_csv(out / "layers.csv", index=False)
    both = both_sample_passes(cells)
    both.to_csv(out / "both_pass_points.csv", index=False)
    summ = {"pbo": pbo_all(out), "runs": int(len(cells)), "both_pass_points": int(len(both)),
            "both_pass_combos": int(rk["both_pass"].sum())}
    (out / "combo_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, cells, rk, lt)

    by_id = {combo_id(c): c for c in COMBOS}
    lab = lambda cid: combo_label(by_id[cid]) if cid in by_id else cid  # noqa: E731
    names = {k: SAMPLES[k][2] for k in SAMPLES}
    pct = [f"{names[k]} {w}" for k in SAMPLES for w in ("통과 점 비율", "SPY 대비 초과 (중앙값)", "MDD (중앙값)")]
    top = rk.head(36)
    frame = {"선별 조합": [lab(c) for c in top["combo"]], "두 기간 모두 통과 점 있음": top["both_pass"].to_numpy()}
    for k in SAMPLES:
        frame[f"{names[k]} 통과 점 비율"] = top[f"pass_share_{k}"].to_numpy()
        frame[f"{names[k]} SPY 대비 초과 (중앙값)"] = top[f"median_goal_excess_{k}"].to_numpy()
        frame[f"{names[k]} MDD (중앙값)"] = top[f"median_mdd_{k}"].to_numpy()
    frame["순위 합"] = top["rank_sum"].to_numpy()
    tables = [{"title": "선별 조합 순위 (두 표본 순위의 합)", "pct": pct, "frame": pd.DataFrame(frame)}]
    lpct = [f"{names[k]} {w}" for k in SAMPLES for w in ("통과 점 비율", "SPY 대비 초과 (중앙값)")]
    lf = {"매매층": [LAYER_LABELS[x] for x in lt["layer"]], "ATR 손절": lt["atr"].to_numpy()}
    for k in SAMPLES:
        lf[f"{names[k]} 통과 점 비율"] = lt[f"pass_share_{k}"].to_numpy()
        lf[f"{names[k]} SPY 대비 초과 (중앙값)"] = lt[f"median_goal_excess_{k}"].to_numpy()
    tables.append({"title": "매매층 × ATR 손절 (모든 조합 합산)", "pct": lpct, "frame": pd.DataFrame(lf)})
    if len(both):
        bf = both.assign(combo=[lab(c) for c in both["combo"]],
                         layer=[LAYER_LABELS[x] for x in both["layer"]])
        tables.append({"title": "두 표본 모두 통과한 점 (조합 × 매매층 설정)",
                       "frame": bf.rename(columns={"combo": "선별 조합", "i": "점 번호", "exclude": "제외", "k": "상위 K",
                                                   "layer": "매매층", "trailing_pct": "트레일링 %", "atr_stop": "ATR 손절",
                                                   "risk_pct": "위험 %", "max_positions": "최대 종목 수"})})
    curves = []
    for smp in SAMPLES:
        named = []
        for cid in rk["combo"].head(8):
            c = pd.read_csv(combo_dir(out, smp) / f"{cid}.csv")
            best = c.sort_values(["goal_pass", "goal_excess"], ascending=False).iloc[0]
            eq = pd.read_parquet(combo_dir(out, smp) / f"{cid}_equity.parquet")[str(int(best.i))]
            named.append((f"{lab(cid)} (점 {int(best.i)})", eq))
        cg = curve_group(panel, smp, named)
        cg["group"] = names[smp]
        curves.append(cg)
    pbo_note = " · ".join(f"{names[k]} {v['pbo']:.0%}" for k, v in summ["pbo"].items() if v)
    store.save_study(f"CRSP 개별 종목 (시총 상위 {meta['top_n']}) 선별 점수 조합 × 매매층 Sobol",
                     f"상위 점수 {len(POOL)}개의 2·3개 조합과 모멘텀 단독({len(COMBOS)}개)을 평균 백분위 순위로 섞고, "
                     f"모든 조합에 같은 Sobol {N_POINTS}점(제외·상위 K·매매층·트레일링·ATR 손절·위험 %·최대 종목 수)을 "
                     "계좌 단위(종목 상한 10%, 편도 20bp, 레버리지 없음)에서 비교했습니다. "
                     "통과 = CAGR ≥ SPY 그리고 |MDD| ≤ |SPY MDD|, 거래 30회 이상. ATR 손절이 없는 점은 동일 금액 슬롯으로 삽니다.",
                     tables, [f"종목 누적 {meta['symbols']}개 · 상장폐지 포함 · {meta['first']} – {meta['last']}",
                              f"조합 {len(COMBOS)} × 점 {N_POINTS} × 표본 {len(SAMPLES)} = 시험 {len(cells)}회",
                              f"PBO(가장 좋은 시험 고르기, 참고용): {pbo_note or '계산 안 됨'}",
                              "그래프: 순위 합 상위 8개 조합의 가장 좋은 점"],
                     "80", "docs/research/lab/", curves=curves)
    return summ


# --------------------------------------------------------------------------- main

def _dry_menu() -> None:
    global N_POINTS
    COMBOS[:] = [["s_dollar_volume", "s_efficiency"], ["s_momentum"]]
    N_POINTS = 4


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="two combinations, four points: checks the pipeline")
    a = ap.parse_args(argv)
    if a.dry:
        _dry_menu()
    out, panel_dir = Path(a.out), Path(a.panel)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(a.results) if a.results else out / "lab_results"
    rlog.RESULTS_DIR = results
    store.RESULTS_DIR = results
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    panel = load_panel_with_cash(panel_dir)
    meta = json.loads((panel_dir / "meta.json").read_text(encoding="utf-8"))
    log(f"패널: 종목 {meta['symbols']}, {meta['first']} – {meta['last']} · 작업 {a.workers}개")
    (out / "menu.json").write_text(json.dumps({"combos": COMBOS, "points": [point_settings(u) for u in points()],
                                               "exclude": EXCLUDE_VARIANTS, "hygiene": screens.HYGIENE_DEFAULT,
                                               "common": COMMON, "entries": ENTRIES},
                                              ensure_ascii=False, default=str), encoding="utf-8")
    workers = a.workers
    while True:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(panel_dir), None, a.dry)) as ex:
                precompute(ex, out, log)
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(panel_dir), str(out), a.dry)) as ex:
                run_combos(ex, out, panel, log)
            break
        except (BrokenProcessPool, MemoryError):
            if workers <= 1:
                raise
            workers = max(1, workers * 3 // 4)
            log(f"메모리가 부족했습니다. 작업 {workers}개로 줄여 이어서 실행합니다(끝난 부분은 건너뜀).")
    summ = save_study(out, panel, meta)
    n, _ = rlog.trial_stats()
    n_eff, rho, _ = rlog.effective_trials()
    log(f"완료 · 시험 {n}회 (유효 {n_eff:.0f}, ρ {rho:.2f}) · 두 표본 통과 점 {summ['both_pass_points']} · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
