"""Strategy-only study on the CRSP point-in-time universe (DECISIONS 70-71), resumable.

    python -m lab.crsp_study --panel DIR --out DIR [--results DIR] [--workers 3] [--sobol 1024]

Reads the panel built by lab/crsp.py, runs menu v3 (77 families x 36 cells) and the Sobol points
in the strategy-only mode of lab/single.py on two samples (1993-02 - 2023-12 and 2017-08 -
2023-12), and writes everything under `out`:
    families/<sample>/<entry>__<exit>.csv, *_sleeves.parquet   one family at a time
    sobol/<sample>.csv, <sample>_sleeves.parquet                in chunks of 64 points
    summary_<sample>.csv, ranks.csv, sobol_summary.json
Parts that already exist are skipped, so an interrupted run continues where it stopped.
Trials are logged (research_log) and a study is saved for the app's 저장된 결과 page under
`results` (default out/lab_results). The data stays where it is; nothing is written to the repo.
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

from . import crsp, explore, research_log as rlog, single, store
from .blocks import ENTRY_BLOCKS, EXIT_BLOCKS
from .data import BENCHMARK, download_panel
from .engine import SignalCache, StrategyConfig
from .families import ENTRY_MENU, EXIT_MENU, MENU_VERSION
from .grid import _neighbors_and_regions
from .metrics import _month_end_values, deflated_sharpe
from .target import _cagr, _mdd, blend, fit_weight

SAMPLES = {"full": ("1993-02-01", "2023-12-31", "CRSP 1993–2023"), "recent": ("2017-08-01", "2023-12-31", "CRSP 2017–2023")}
KEEP = ["symbols", "symbol_pass_share", "symbol_median_excess", "symbol_median_mdd", "n_trades", "sleeve_cagr", "sleeve_mdd",
        "sleeve_dd_weight", "sleeve_dd_cagr", "sleeve_mix_cagr", "sleeve_dd_excess", "sleeve_dd_pass", "sleeve_window_share",
        "sleeve_windows", "sleeve_worst_window_excess"]
ORDER = ["goal_pass", "region_size", "neighbor_survival", "sleeve_window_share", "goal_excess"]
MIX = "SPY+국채 혼합 (낙폭 15%)"
_W: dict = {}


def base_config(start: str, end: str) -> StrategyConfig:
    return StrategyConfig(entries={}, exits={}, start=start, end=end, sizing="equal")  # settings of DECISIONS 43


def load_panel_with_cash(panel_dir: Path):
    panel = crsp.load_panel(panel_dir)
    ref, _ = download_panel([BENCHMARK], str(panel.last_date.date()))  # only for its T-bill cash series
    panel.cash = ref.cash.reindex(panel.close.index).ffill().fillna(0.0)
    return panel


def spy_stats(panel, start: str, end: str) -> dict:
    spy = panel.close[BENCHMARK].loc[start:end].dropna()
    v = (spy / spy.iloc[0]).to_numpy(dtype=float)
    return {"spy_cagr": _cagr(v), "spy_mdd": _mdd(v)}


# --------------------------------------------------------------------------- workers

def _init(panel_dir: str, entry_menu: dict | None = None, exit_menu: dict | None = None):
    if entry_menu:  # a --dry run's reduced menu (the worker imports the full one)
        ENTRY_MENU.clear(); ENTRY_MENU.update(entry_menu)
        EXIT_MENU.clear(); EXIT_MENU.update(exit_menu)
    _W["panel"] = load_panel_with_cash(Path(panel_dir))
    _W["cache"] = SignalCache(_W["panel"])
    _W["spy"] = {k: spy_stats(_W["panel"], s, e) for k, (s, e, _) in SAMPLES.items()}
    idx = _W["panel"].close.index
    _W["window"] = {k: tuple(idx[(idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))][[0, -1]]) for k, (s, e, _) in SAMPLES.items()}
    _W["rules"] = None


def _run(job):
    tag, sample, entries, exits = job
    s, e, _ = SAMPLES[sample]
    # a family's 36 cells share block outputs; a new rule set frees the previous one's arrays
    # (kept, they add up to tens of GB over 154 families on 3,700 symbols)
    rules = (sample, tuple(entries), tuple(exits)) if isinstance(tag, tuple) else (sample, tag)
    if rules != _W["rules"]:
        _W["cache"].drop_rule_data(keep_window=_W["window"][sample])
        _W["rules"] = rules
    cfg = replace(base_config(s, e), entries=entries, exits=exits)
    o = single.evaluate(_W["panel"], cfg, _W["cache"])
    row = {k: o[k] for k in KEEP}
    row.update(_W["spy"][sample])
    row["goal_excess"] = o["sleeve_cagr"] - row["spy_cagr"]
    # DECISIONS 71: the user's goal, at 100% invested and without leverage
    row["goal_pass"] = bool(o["sleeve_cagr"] >= row["spy_cagr"] and abs(o["sleeve_mdd"]) <= abs(row["spy_mdd"])
                            and o["n_trades"] >= 30)
    return tag, cfg, row, o["sleeve"].astype("float32")


def run_jobs(ex, jobs):
    return list(ex.map(_run, jobs, chunksize=1))


def memory_note() -> str:
    try:
        import psutil
    except ImportError:
        return ""
    vm = psutil.virtual_memory()
    return f" · 메모리 여유 {vm.available / 1e9:.1f}/{vm.total / 1e9:.0f} GB"


# --------------------------------------------------------------------------- families

def family_jobs(sample: str, e: str, x: str):
    return [((e, x, xi, yi, xv, yv), sample, {e: xv}, {x: yv})
            for yi, yv in enumerate(EXIT_MENU[x]) for xi, xv in enumerate(ENTRY_MENU[e])]


def run_families(ex, out: Path, panel, log):
    t0, done_cells, todo = time.time(), 0, []
    for sample in SAMPLES:
        d = out / "families" / sample
        d.mkdir(parents=True, exist_ok=True)
        for e in ENTRY_MENU:
            for x in EXIT_MENU:
                if not (d / f"{e}__{x}.csv").exists():
                    todo.append((sample, e, x))
    if todo and not (out / "menu.json").exists():  # the menu the folder was built with, for resuming
        (out / "menu.json").write_text(json.dumps({"version": MENU_VERSION, "entry": ENTRY_MENU, "exit": EXIT_MENU}), encoding="utf-8")
    log(f"계열 격자: {len(todo)}개 남음 (전체 {len(SAMPLES) * len(ENTRY_MENU) * len(EXIT_MENU)})")
    for n, (sample, e, x) in enumerate(todo):
        t = time.time()
        res = run_jobs(ex, family_jobs(sample, e, x))
        cells = pd.DataFrame([{"entry": tag[0], "exit": tag[1], "xi": tag[2], "yi": tag[3], "x": tag[4], "y": tag[5], **row}
                              for tag, _, row, _ in res])
        cells["pass"] = cells["goal_pass"]
        _neighbors_and_regions(cells)
        sleeves = pd.DataFrame({f"{tag[4]:g}|{tag[5]:g}": eq for tag, _, _, eq in res})
        d = out / "families" / sample
        sleeves.to_parquet(d / f"{e}__{x}_sleeves.parquet")
        cells.to_csv(d / f"{e}__{x}.csv", index=False)  # written last: its presence marks the family done
        rlog.log_trials([("crsp_single", cfg, panel.tradable_symbols, eq.astype(float)) for _, cfg, _, eq in res])
        done_cells += len(cells)
        per_cell = (time.time() - t0) / done_cells
        remaining = (len(todo) - n - 1) * 36 * per_cell
        log(f"  [{n + 1}/{len(todo)}] {sample} {ENTRY_BLOCKS[e].label} × {EXIT_BLOCKS[x].label}: "
            f"{time.time() - t:.0f}s, 통과 {int(cells['pass'].sum())}/36 · 칸당 {per_cell:.1f}s · 남은 예상 {remaining / 3600:.1f}h"
            + memory_note())


def family_summary(out: Path, sample: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = out / "families" / sample
    cells = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*__*.csv"))], ignore_index=True)
    rows = []
    for (e, x), g in cells.groupby(["entry", "exit"], sort=False):
        rows.append({"entry": e, "exit": x, "pass_share": g["pass"].mean(), "largest_region": int(g["region_size"].max()),
                     "median_goal_excess": g["goal_excess"].median(), "median_symbol_pass": g["symbol_pass_share"].median(),
                     "median_sleeve_cagr": g["sleeve_cagr"].median(), "median_sleeve_mdd": g["sleeve_mdd"].median(),
                     "median_dd_excess": g["sleeve_dd_excess"].median(), "median_window_share": g["sleeve_window_share"].median()})
    s = pd.DataFrame(rows).sort_values(["pass_share", "largest_region", "median_goal_excess", "median_symbol_pass"],
                                       ascending=False).reset_index(drop=True)
    s.to_csv(out / f"summary_{sample}.csv", index=False)
    return s, cells


# --------------------------------------------------------------------------- Sobol

def run_sobol(ex, out: Path, panel, n_points: int, log):
    pts = explore.sobol(n_points, 8)
    d = out / "sobol"
    d.mkdir(exist_ok=True)
    for sample in SAMPLES:
        path = d / f"{sample}.csv"
        done = set(pd.read_csv(path)["i"]) if path.exists() else set()
        todo = [i for i in range(n_points) if i not in done]
        log(f"Sobol {sample}: {len(todo)}개 남음 (전체 {n_points})")
        s_, e_, _ = SAMPLES[sample]
        base = base_config(s_, e_)
        for k in range(0, len(todo), 64):
            chunk = todo[k:k + 64]
            jobs = []
            for i in chunk:
                cfg = explore.point_config(np.r_[pts[i, :6], 0.0, 0.0], base)  # variables 1-6 only (DECISIONS 68)
                jobs.append((i, sample, cfg.entries, cfg.exits))
            res = run_jobs(ex, jobs)
            rows = pd.DataFrame([{"i": i, "desc": cfg.describe().split(" | 최대", 1)[0].split(" | 종목별", 1)[0],
                                  "entries": json.dumps(cfg.entries), "exits": json.dumps(cfg.exits), **row}
                                 for i, cfg, row, _ in res])
            sl = pd.DataFrame({str(i): eq for i, _, _, eq in res})
            sp = d / f"{sample}_sleeves.parquet"
            if sp.exists():
                sl = pd.concat([pd.read_parquet(sp), sl], axis=1)
            sl.to_parquet(sp)
            rows.to_csv(path, mode="a", header=not path.exists(), index=False)
            rlog.log_trials([("crsp_sobol", cfg, panel.tradable_symbols, eq.astype(float)) for _, cfg, _, eq in res])
            log(f"  {sample}: {k + len(chunk)}/{len(todo)} · 통과 {int(rows['goal_pass'].sum())}/{len(rows)}")


def sobol_summary(out: Path) -> dict:
    d = out / "sobol"
    rows = {k: pd.read_csv(d / f"{k}.csv").set_index("i").sort_index() for k in SAMPLES}
    summ = {"pbo": {}, "pass_share": {}}
    for k in SAMPLES:
        sl = pd.read_parquet(d / f"{k}_sleeves.parquet")
        monthly = pd.DataFrame({c: _month_end_values(sl[c].dropna().astype(float)).pct_change(fill_method=None).dropna() for c in sl})
        summ["pbo"][k] = explore.pbo(monthly, blocks=12)
        summ["pass_share"][k] = float(rows[k]["goal_pass"].mean())
    both = pd.concat([rows[k]["goal_pass"] for k in SAMPLES], axis=1).all(axis=1)
    win = pd.concat([rows[k]["sleeve_window_share"] >= 0.5 for k in SAMPLES], axis=1).all(axis=1)
    worst = pd.concat([rows[k]["goal_excess"] for k in SAMPLES], axis=1).min(axis=1)
    summ["all_pass"] = int(both.sum())
    summ["robust_candidates"] = int((both & win).sum())
    top = worst[both & win].sort_values(ascending=False).head(20) if (both & win).any() else worst.sort_values(ascending=False).head(20)
    summ["top"] = [{"i": int(i), "worst_goal_excess": float(v), "desc": rows["full"].loc[i, "desc"],
                    **{f"{k}_pass": bool(rows[k].loc[i, "goal_pass"]) for k in SAMPLES}} for i, v in top.items()]
    (out / "sobol_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return summ


# --------------------------------------------------------------------------- study for the app

def _scaled(eq: pd.Series, cash: pd.Series):
    r = eq.pct_change(fill_method=None).iloc[1:].fillna(0.0).to_numpy()
    c = cash.reindex(eq.index).iloc[1:].fillna(0.0).to_numpy()
    w = fit_weight(r, c)
    return pd.Series(blend(r, c, w), index=eq.index), w


def curve_group(panel, sample: str, named: list[tuple[str, pd.Series]]) -> dict:
    raw, sc = {}, {}
    for name, eq in named:
        eq = eq.dropna().astype(float)
        s_eq, w = _scaled(eq, panel.cash)
        raw[name] = eq / eq.iloc[0]
        sc[f"{name} [비중 {w:.0%}]"] = s_eq
    idx = next(iter(raw.values())).index
    spy = panel.close[BENCHMARK].reindex(idx).ffill()
    spy = spy / spy.iloc[0]
    mix, _ = _scaled(spy, panel.cash)
    return {"group": SAMPLES[sample][2], "raw": pd.DataFrame(raw).assign(SPY=spy),
            "scaled": pd.DataFrame(sc).assign(**{MIX: mix, "SPY": spy}), "baselines": ["SPY", MIX]}


def fam(e: str, x: str) -> str:
    return f"{ENTRY_BLOCKS[e].label} × {EXIT_BLOCKS[x].label}"


def save_studies(out: Path, panel, meta: dict, summ: dict, n_eff: float, sr_var: float) -> None:
    save_family_study(out, panel, meta, n_eff, sr_var)
    save_sobol_study(out, panel, meta, summ)


def save_family_study(out: Path, panel, meta: dict, n_eff: float, sr_var: float) -> None:
    """The family comparison as a study for the app (panel: only close[SPY] and cash are read)."""
    sums = {k: family_summary(out, k) for k in SAMPLES}
    rk = {k: {(r.entry, r.exit): i + 1 for i, r in enumerate(s.itertuples())} for k, (s, _) in sums.items()}
    ranks = pd.DataFrame([{"entry": e, "exit": x, **{f"rank_{k}": rk[k][(e, x)] for k in rk}} for e, x in rk["full"]])
    ranks["rank_sum"] = ranks[[f"rank_{k}" for k in rk]].sum(axis=1)
    ranks = ranks.sort_values("rank_sum").reset_index(drop=True)
    ranks.to_csv(out / "ranks.csv", index=False)
    pct = ["통과 칸 비율", "SPY 대비 초과 수익 (중앙값)", "보유보다 나은 종목 비율 (중앙값)", "CAGR (중앙값)", "MDD (중앙값)", "3년 구간 승률 (중앙값)"]
    tables, curves, top_rows = [], [], []
    for k, (s, cells) in sums.items():
        g = s.head(20)
        tables.append({"title": f"{SAMPLES[k][2]} — 상위 20개 계열", "pct": pct,
                       "frame": pd.DataFrame({"계열": [fam(e, x) for e, x in zip(g.entry, g.exit)], "통과 칸 비율": g.pass_share,
                                              "가장 큰 연결 영역": g.largest_region, "SPY 대비 초과 수익 (중앙값)": g.median_goal_excess,
                                              "보유보다 나은 종목 비율 (중앙값)": g.median_symbol_pass, "CAGR (중앙값)": g.median_sleeve_cagr,
                                              "MDD (중앙값)": g.median_sleeve_mdd, "3년 구간 승률 (중앙값)": g.median_window_share})})
        named = []
        for r in ranks.head(10).itertuples():
            c = cells[(cells.entry == r.entry) & (cells.exit == r.exit)]
            best = c.sort_values(ORDER, ascending=False).iloc[0]
            sl = pd.read_parquet(out / "families" / k / f"{r.entry}__{r.exit}_sleeves.parquet")[f"{best.x:g}|{best.y:g}"]
            named.append((f"{fam(r.entry, r.exit)} ({best.x:g}, {best.y:g})", sl))
            top_rows.append({"계열": fam(r.entry, r.exit), "표본": SAMPLES[k][2], "칸 (진입 값, 청산 값)": f"({best.x:g}, {best.y:g})",
                             "목표 통과": bool(best.goal_pass), "CAGR": best.sleeve_cagr, "MDD": best.sleeve_mdd,
                             "SPY CAGR": best.spy_cagr, "SPY MDD": best.spy_mdd, "보유보다 나은 종목 비율": best.symbol_pass_share,
                             "3년 구간 승률": best.sleeve_window_share, "연결 영역": int(best.region_size), "거래 수": best.n_trades,
                             "DSR": deflated_sharpe(sl.dropna().astype(float), max(1, int(round(n_eff))), sr_var)})
        curves.append(curve_group(panel, k, named))
    top20 = ranks.head(20)
    tables.append({"title": "두 표본 순위 합 (상위 20)",
                   "frame": pd.DataFrame({"순위": range(1, len(top20) + 1), "계열": [fam(e, x) for e, x in zip(top20.entry, top20.exit)],
                                          **{f"{SAMPLES[k][2]} 순위": top20[f"rank_{k}"].to_numpy() for k in SAMPLES},
                                          "순위 합": top20.rank_sum.to_numpy()})})
    tables.append({"title": "상위 10개 계열의 대표 칸", "frame": pd.DataFrame(top_rows),
                   "pct": ["CAGR", "MDD", "SPY CAGR", "SPY MDD", "보유보다 나은 종목 비율", "3년 구간 승률", "DSR"]})
    store.save_study(f"CRSP 개별 종목 (시총 상위 {meta['top_n']}) 계열 비교 · 종목별 단독",
                     f"CRSP 보통주를 매년 말 시가총액 상위 {meta['top_n']}개로 제한한 시점별 유니버스에서 77개 계열(메뉴 {MENU_VERSION})을 "
                     "**종목별 단독** 방식으로 시험했습니다. 통과 = 매매 기록 포트폴리오(동일비중, 비중 100%, 레버리지 없음)의 CAGR ≥ SPY, "
                     "|MDD| ≤ |SPY MDD|, 거래 30회 이상.",
                     tables, [f"종목 누적 {meta['symbols']}개 · 상장폐지 종목 포함(폐지 수익률 반영) · {meta['first']} – {meta['last']}",
                              "매매 기록 포트폴리오 = 그날 매수 가능한 종목(당해 상위 N)과 보유 중인 종목의 단독 계좌를 동일비중으로 합친 것",
                              "그래프: 순위 합 상위 10개 계열의 대표 칸 (표본마다 따로 고른 칸)"],
                     "70–71", "docs/research/lab/", curves=curves)


def save_sobol_study(out: Path, panel, meta: dict, summ: dict) -> None:
    d = out / "sobol"
    rows = {k: pd.read_csv(d / f"{k}.csv").set_index("i").sort_index() for k in SAMPLES}
    pbo = pd.DataFrame([{"표본": SAMPLES[k][2], "통과 비율": summ["pass_share"][k], "PBO (과최적화 확률)": summ["pbo"][k]["pbo"],
                         "1위의 표본 밖 순위 (중앙값, 1=최고)": 1 - summ["pbo"][k]["median_oos_rank"]} for k in SAMPLES])
    top_i = [t["i"] for t in summ["top"]]
    tp = pd.DataFrame({"전략": [t["desc"] for t in summ["top"]],
                       **{f"{SAMPLES[k][2]} 초과 수익": [rows[k].loc[i, "goal_excess"] for i in top_i] for k in SAMPLES},
                       **{f"{SAMPLES[k][2]} 통과": [bool(rows[k].loc[i, "goal_pass"]) for i in top_i] for k in SAMPLES}})
    curves = []
    for k in SAMPLES:
        sl = pd.read_parquet(d / f"{k}_sleeves.parquet")
        curves.append(curve_group(panel, k, [(rows[k].loc[i, "desc"], sl[str(i)]) for i in top_i if str(i) in sl]))
    store.save_study(f"CRSP 개별 종목 (시총 상위 {meta['top_n']}) Sobol 다변수 탐색 + PBO",
                     "진입 블록·값, 청산 블록·값, 필터, 초기 손절 6개 변수를 Sobol 점으로 고르게 뽑아 종목별 단독 방식으로 시험했습니다. "
                     "통과 조건은 계열 비교와 같습니다(CAGR ≥ SPY, |MDD| ≤ |SPY MDD|).",
                     [{"title": "표본별 통과 비율과 PBO", "frame": pbo, "pct": ["통과 비율", "PBO (과최적화 확률)", "1위의 표본 밖 순위 (중앙값, 1=최고)"]},
                      {"title": "두 표본 최소 초과 수익 상위 20점", "frame": tp, "pct": [c for c in tp.columns if "초과" in c]}],
                     [f"두 표본 모두 통과: {summ['all_pass']}점 · 그중 3년 구간 승률 ≥ 0.5: {summ['robust_candidates']}점"],
                     "68, 70–71", "docs/research/lab/", curves=curves)


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None, help="시험 기록과 앱용 연구 요약을 쓸 폴더 (기본 out/lab_results)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sobol", type=int, default=1024)
    ap.add_argument("--skip-sobol", action="store_true")
    ap.add_argument("--dry", action="store_true", help="2x2 menu, 16 Sobol points: checks the pipeline end to end")
    a = ap.parse_args(argv)
    if a.dry:
        ENTRY_MENU.clear(); ENTRY_MENU.update({"breakout": [20, 50], "momentum": [63, 126]})
        EXIT_MENU.clear(); EXIT_MENU.update({"trailing_pct": [10, 15], "low_break": [20, 50]})
        a.sobol = 16
    out, panel_dir = Path(a.out), Path(a.panel)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(a.results) if a.results else out / "lab_results"
    rlog.RESULTS_DIR = results
    store.RESULTS_DIR = results
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    panel = load_panel_with_cash(panel_dir)
    meta = json.loads((panel_dir / "meta.json").read_text(encoding="utf-8"))
    log(f"패널: 종목 {meta['symbols']}, {meta['first']} – {meta['last']} · 작업 {a.workers}개")
    workers = a.workers
    while True:  # a worker killed for memory breaks the pool: restart with one fewer and resume
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                     initargs=(str(panel_dir), dict(ENTRY_MENU), dict(EXIT_MENU))) as ex:
                run_families(ex, out, panel, log)
                if not a.skip_sobol:
                    run_sobol(ex, out, panel, a.sobol, log)
            break
        except BrokenProcessPool:
            if workers <= 1:
                raise
            workers -= 1
            log(f"작업 프로세스가 메모리 부족으로 종료됐습니다. 작업 {workers}개로 줄여 이어서 실행합니다.")
    summ = sobol_summary(out) if not a.skip_sobol else {"pbo": {}, "pass_share": {}, "all_pass": 0, "robust_candidates": 0, "top": []}
    n_eff, rho, _ = rlog.effective_trials()
    _, sr_var = rlog.trial_stats()
    if a.skip_sobol:
        save_family_study(out, panel, meta, n_eff, sr_var)
    else:
        save_studies(out, panel, meta, summ, n_eff, sr_var)
    n, _ = rlog.trial_stats()
    log(f"완료 · 시험 {n}회 (유효 {n_eff:.0f}, ρ {rho:.2f}) · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
