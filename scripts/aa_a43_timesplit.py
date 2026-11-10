"""A43: time-split test of the rule that picks a dynamic setting (allocation DECISIONS A43).

    .venv/Scripts/python.exe scripts/aa_a43_timesplit.py --quick --out %TEMP%\\a43_quick
    .venv/Scripts/python.exe scripts/aa_a43_timesplit.py --out docs/research/allocation/2026-11-06_a43_timesplit --check-stage4
    ... --figures-only

Same data, targets, settings, satellite shares, price sets, placebos and seeds as A42 stage 4 (`scripts/aa_a42_stage4.py`,
whose `build_context` is reused). Every path is run once over the whole research window and then measured inside the
fixed windows of `lab.aa_timesplit.WINDOWS`; the stage-4 verdict is applied per window, the selection-window passers
(rule A) and the best passer by rank value (rule B) are measured in the test window of each direction.
Design fixed before the run: docs/research/allocation/2026-11-06_a43_design.md.

Work is saved per (price set, target) and per (target, signal) placebo job under <out>/_work.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import aa_canary as K  # noqa: E402
from lab import aa_layers as LY  # noqa: E402
from lab import aa_timesplit as TS  # noqa: E402


def _stage4():
    spec = importlib.util.spec_from_file_location("aa_a42_stage4", ROOT / "scripts" / "aa_a42_stage4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S4 = _stage4()
SETS, SAT, PLACEBOS, SEED, TARGET_NAMES = S4.SETS, S4.SAT, S4.PLACEBOS, S4.SEED, S4.TARGET_NAMES
FIG_DIR = ROOT / "docs" / "research" / "allocation" / "figures" / "a43_timesplit"
STAGE4_DIR = ROOT / "docs" / "research" / "allocation" / "2026-11-02_a42_stage4"

_CTX: dict = {}


def _init(ctx):
    _CTX.update(ctx)


def _windows(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {w: TS.window_rows(dates, w) for w in TS.WINDOWS}


def eq_key(st: LY.Setting) -> tuple[str, str]:
    return S4.eq_key(st)


# --------------------------------------------------------------------------- job A: real settings on one set, all windows

def job_real(args):
    sname, tname, settings = args
    c = _CTX["sets"][(sname, tname)]
    w = _CTX["targets"][tname]
    sim, cols = c["sim"], c["sim"].columns
    wins = _windows(sim.dates)
    rep = sname == "대표"
    levs = {v: (LY.vol_levels(c["base_ret"], sim.dates, v) if v else None) for v in (0.0, 0.08, 0.10)}
    base = c["base_path"][None, :]
    rows = []
    for s in SAT:
        bm = LY.mix(base, c["sat"], s)
        for wn, rws in wins.items():
            cg, md = TS.window_metrics(bm, rws)
            rows.append({"세트": sname, "대상": tname, "종류": "정적", "위성": s, "구간": wn, "cagr": float(cg[0]), "mdd": float(md[0]),
                         "순위값": float(TS.rank_value(bm, sim.dates, rws)[0]) if rep else np.nan})
    grids = {}
    for st in settings:
        key = eq_key(st)
        if key in grids:
            continue
        b0 = LY.action_tables(w, "none", "none", cols)[0]
        full = LY.full_move(w, key[0], key[1], cols)
        xs = np.linspace(0, 1, TS.EQ_GRID)
        W = np.vstack([b0 + x * (full - b0) for x in xs])
        P, _, _ = sim.run(*K.static_rows(W, sim))
        grids[key] = {(s, wn): TS.window_metrics(LY.mix(P, c["sat"], s), rws) for s in SAT for wn, rws in wins.items()}
    for st in settings:
        tab = LY.action_tables(w, st.signal, st.action, cols)
        S = c["states"][st.signal][None, :] if st.signal != "none" else np.zeros((1, sim.n_days), dtype=int)
        P = LY.paths(sim, tab, S, st, c["qrows"], levs[st.vol])
        for s in SAT:
            m = LY.mix(P, c["sat"], s)
            for wn, rws in wins.items():
                cg, md = TS.window_metrics(m, rws)
                gc, gm = grids[eq_key(st)][(s, wn)]
                x, eq_mdd = TS.equal_cost(gc, gm, float(cg[0]))
                rows.append({"세트": sname, "대상": tname, "종류": "설정", "위성": s, "구간": wn, **st.__dict__, "설정 이름": st.label(),
                             "cagr": float(cg[0]), "mdd": float(md[0]), "같은 비용 x": x, "같은 비용 mdd": eq_mdd,
                             "순위값": float(TS.rank_value(m, sim.dates, rws)[0]) if rep else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- job B: placebos on the representative set, all windows

def job_placebo(args):
    tname, signal, settings, n = args
    c = _CTX["sets"][("대표", tname)]
    w = _CTX["targets"][tname]
    sim = c["sim"]
    wins = _windows(sim.dates)
    seed = SEED + 100 * TARGET_NAMES.index(tname) + LY.SIGNALS.index(signal)
    S, _ = K.placebo_states(c["states"][signal], n, seed)
    levs = {v: (LY.vol_levels(c["base_ret"], sim.dates, v) if v else None) for v in (0.0, 0.08, 0.10)}
    rows = []
    for st in settings:
        tab = LY.action_tables(w, st.signal, st.action, sim.columns)
        P = LY.paths(sim, tab, S, st, c["qrows"], levs[st.vol])
        for s in SAT:
            m = LY.mix(P, c["sat"], s)
            for wn, rws in wins.items():
                _, md = TS.window_metrics(m, rws)
                rows.append({"대상": tname, **st.__dict__, "위성": s, "구간": wn, "가짜 mdd q95": float(np.quantile(md, TS.PLACEBO_Q)),
                             "가짜 mdd 중앙": float(np.median(md)), "가짜 수": len(md)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- stage-4 reproduction check

def check_stage4(V: pd.DataFrame) -> dict:
    """'전체' window verdicts vs stage 4's verdicts.csv.gz (same settings, targets, satellite): counts of mismatches."""
    if not (STAGE4_DIR / "verdicts.csv.gz").exists():
        return {"available": False}
    s4 = pd.read_csv(STAGE4_DIR / "verdicts.csv.gz", keep_default_na=False, na_values=[""])
    v = V[V["구간"] == "전체"]
    m = v.merge(s4[TS.KEY + ["판정", "cagr", "mdd", "가짜 mdd q95"]], on=TS.KEY, how="inner", suffixes=("", " s4"))
    return {"available": True, "rows_compared": int(len(m)), "verdict_mismatch": int((m["판정"] != m["판정 s4"]).sum()),
            "max_abs_cagr_diff": float((m["cagr"] - m["cagr s4"]).abs().max()) if len(m) else None,
            "max_abs_mdd_diff": float((m["mdd"] - m["mdd s4"]).abs().max()) if len(m) else None,
            "max_abs_placebo_q95_diff": float((m["가짜 mdd q95"] - m["가짜 mdd q95 s4"]).abs().max()) if len(m) else None}


# --------------------------------------------------------------------------- figures

def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    dest.mkdir(parents=True, exist_ok=True)
    TR = pd.read_csv(out / "transfer.csv")
    BP = pd.read_csv(out / "best_pick.csv")
    V = pd.read_csv(out / "verdicts_by_window.csv.gz", keep_default_na=False, na_values=[""])
    R3 = pd.read_csv(out / "rep3_windows.csv")
    made = []
    # ① rule A: kept share vs base rate, per direction × target
    dirs = [d[0] for d in TS.DIRECTIONS]
    tnames = [t for t in ["전체", *TARGET_NAMES] if t in set(TR["대상"])]
    fig, axes = plt.subplots(1, len(dirs), figsize=(4.2 * len(dirs), 5), sharey=True)
    for ax, (dname, sel, test) in zip(np.atleast_1d(axes), TS.DIRECTIONS):
        g = TR[TR["방향"] == f"{sel} → {test}"].set_index("대상").reindex(tnames)
        x = np.arange(len(tnames))
        ax.bar(x - 0.2, g["바탕 통과율"] * 100, 0.4, label="바탕 통과율 (모든 설정)", color="#BBBBBB")
        ax.bar(x + 0.2, g["유지율"] * 100, 0.4, label="고른 설정의 유지율", color="#4C78A8")
        for i, (n, p) in enumerate(zip(g["고르는 구간 통과 n"], g["p값 (초기하)"])):
            ax.text(i, max(float(np.nan_to_num(g["유지율"].iloc[i])), float(np.nan_to_num(g["바탕 통과율"].iloc[i]))) * 100 + 1,
                    f"n={int(n) if pd.notna(n) else 0}\np={p:.2f}" if pd.notna(p) else f"n={int(n) if pd.notna(n) else 0}", ha="center", fontsize=7)
        ax.set_xticks(x, tnames, rotation=30, ha="right", fontsize=8)
        ax.set_title(dname, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("시험 구간에서 판정 통과 (%)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("① 규칙 A: 고르는 구간에서 통과한 설정이 시험 구간에서도 통과한 비율 vs 바탕 통과율 (대표 세트)")
    made.append(str(figs._save(fig, dest / "1_transfer.png")))
    # ② rule B on (가): test-window MDD improvement of all settings, passers, and the pick
    fig, axes = plt.subplots(1, len(dirs), figsize=(4.2 * len(dirs), 4.8), sharey=True)
    for ax, (dname, sel, test) in zip(np.atleast_1d(axes), TS.DIRECTIONS):
        a = V[(V["구간"] == sel) & (V["대상"] == "F4 (가)")].set_index(TS.KEY)
        b = V[(V["구간"] == test) & (V["대상"] == "F4 (가)")].set_index(TS.KEY)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        ax.scatter(b["CAGR 손실"] * 100, b["MDD 개선"] * 100, s=5, color="#CCCCCC", label="모든 설정 (시험 구간)")
        p = (a["판정"] == "통과").to_numpy()
        ax.scatter(b.loc[p, "CAGR 손실"] * 100, b.loc[p, "MDD 개선"] * 100, s=14, color="#4C78A8", label="고르는 구간 통과")
        pk = BP[(BP["방향"] == f"{sel} → {test}") & (BP["대상"] == "F4 (가)")]
        if len(pk) and pd.notna(pk["시험 MDD 개선"].iloc[0]):
            ax.scatter(pk["시험 CAGR 손실"] * 100, pk["시험 MDD 개선"] * 100, s=90, marker="*", color="#E45756", label="규칙 B 선택")
        ax.axvline(1, color="gray", lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(dname, fontsize=10)
        ax.set_xlabel("시험 구간 CAGR 손실 (%p)")
    np.atleast_1d(axes)[0].set_ylabel("시험 구간 MDD 개선 (%p, 정적 대비)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("② 대상 (가): 고르는 구간에서 통과한 설정과 규칙 B의 선택이 시험 구간에서 어디에 있는가")
    made.append(str(figs._save(fig, dest / "2_best_pick.png")))
    # ③ representative ③ and references per window
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    wins = list(TS.WINDOWS)
    names = list(dict.fromkeys(R3["이름"]))
    x = np.arange(len(wins))
    wd = 0.8 / max(len(names), 1)
    for k, nm in enumerate(names):
        g = R3[R3["이름"] == nm].set_index("구간").reindex(wins)
        axes[0].bar(x + (k - len(names) / 2 + 0.5) * wd, g["mdd"] * 100, wd, label=nm)
        axes[1].bar(x + (k - len(names) / 2 + 0.5) * wd, g["cagr"] * 100, wd, label=nm)
    r3 = R3[R3["이름"].str.startswith("③")].set_index("구간").reindex(wins)
    axes[0].scatter(x, r3["가짜 mdd q95"] * 100, marker="_", s=300, color="black", label="③의 가짜 신호 95% 선", zorder=5)
    axes[0].axhline(-15, color="gray", lw=0.8, ls="--")
    for ax, t in zip(axes, ("MDD (%)", "CAGR (%)")):
        ax.set_xticks(x, wins, rotation=20, fontsize=8)
        ax.set_ylabel(t)
    axes[0].legend(fontsize=6)
    fig.suptitle("③ 대표 ③과 신호 없는 참고 행의 구간별 MDD·CAGR (대표 세트). 점선 −15%")
    made.append(str(figs._save(fig, dest / "3_rep3_windows.png")))
    return made


# --------------------------------------------------------------------------- main

def analyse(out: Path, R: pd.DataFrame, PL: pd.DataFrame, check: bool, meta: dict) -> dict:
    V = TS.verdicts(R, PL)
    V.to_csv(out / "verdicts_by_window.csv.gz", index=False)
    TR = pd.concat([TS.transfer(V, s, t) for _, s, t in TS.DIRECTIONS], ignore_index=True)
    BP = pd.concat([TS.best_pick(V, s, t) for _, s, t in TS.DIRECTIONS], ignore_index=True)
    R3 = TS.rep3_rows(V, R)
    TR.to_csv(out / "transfer.csv", index=False)
    BP.to_csv(out / "best_pick.csv", index=False)
    R3.to_csv(out / "rep3_windows.csv", index=False)
    J = TS.judgements(TR, BP, R3)
    summ = {**meta, "windows": TS.WINDOWS, "directions": [list(d) for d in TS.DIRECTIONS], "judgements": J,
            "pass_by_window": V.groupby("구간")["판정"].apply(lambda s: int((s == "통과").sum())).to_dict(),
            "transfer_overall": TR[TR["대상"] == "전체"].to_dict("records"),
            "best_pick_ga": BP[BP["대상"] == "F4 (가)"].to_dict("records")}
    if check:
        summ["stage4_check"] = check_stage4(V)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summ


def quick_settings(settings: list) -> list:
    """The --quick subset: every fourth setting of three signals, plus ③'s own setting, in all_settings() order."""
    pick = [s for s in settings if s.signal in ("none", "k2", "k4") and s.action in ("none", "riskcd", "safe")][::4]
    rep3 = LY.Setting(**{k: v for k, v in TS.REP3.items() if k in TS.SETTING_KEY})
    order = {s: i for i, s in enumerate(LY.all_settings())}
    return sorted(set(pick) | {rep3}, key=lambda s: order.get(s, len(order)))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--analyse-only", action="store_true", help="re-run the verdicts and tables from the saved runs")
    ap.add_argument("--check-stage4", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S4._TARGETS.update(S4.targets())
    if args.figures_only:
        print(figures(out, FIG_DIR if not args.quick else out / "figures"))
        return
    t0 = time.time()
    if args.analyse_only:
        R = pd.read_csv(out / "runs_by_window.csv.gz", keep_default_na=False, na_values=[""])
        PL = pd.read_csv(out / "placebos_by_window.csv.gz", keep_default_na=False, na_values=[""])
        summ = analyse(out, R, PL, args.check_stage4, {"quick": args.quick, "analyse_only": True})
        print(json.dumps(summ["judgements"], ensure_ascii=False))
        print(figures(out, FIG_DIR if not args.quick else out / "figures"))
        return
    work = out / "_work"
    work.mkdir(exist_ok=True)
    settings = LY.all_settings()
    n_pl = PLACEBOS
    names = list(TARGET_NAMES)
    if args.quick:
        settings = quick_settings(settings)
        n_pl, names = 8, ["F4 (가)", "F3 현재 보유"]
    ctx = S4.build_context(args.quick)
    ctx["targets"] = {n: ctx["targets"][n] for n in names}
    jobs_a = [(sn, tn, settings) for sn in SETS for tn in names if not (work / f"a_{sn}_{TARGET_NAMES.index(tn)}.pkl").exists()]
    sigs = sorted({s.signal for s in settings} - {"none"}, key=LY.SIGNALS.index)
    jobs_b = [(tn, sg, [s for s in settings if s.signal == sg], n_pl) for tn in names for sg in sigs
              if not (work / f"b_{TARGET_NAMES.index(tn)}_{sg}.pkl").exists()]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ctx,)) as ex:
        futs = {ex.submit(job_real, j): ("a", j) for j in jobs_a}
        futs.update({ex.submit(job_placebo, j): ("b", j) for j in jobs_b})
        for k, (f, (kind, j)) in enumerate(futs.items()):
            df = f.result()
            name = f"a_{j[0]}_{TARGET_NAMES.index(j[1])}.pkl" if kind == "a" else f"b_{TARGET_NAMES.index(j[0])}_{j[1]}.pkl"
            df.to_pickle(work / name)  # this script's own work files, read back only by it
            print(f"  {k + 1}/{len(futs)} {kind} {j[0]} {j[1]} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat([pd.read_pickle(f) for f in sorted(work.glob("a_*.pkl"))], ignore_index=True)
    PL = pd.concat([pd.read_pickle(f) for f in sorted(work.glob("b_*.pkl"))], ignore_index=True)
    R.to_csv(out / "runs_by_window.csv.gz", index=False)
    PL.to_csv(out / "placebos_by_window.csv.gz", index=False)
    summ = analyse(out, R, PL, args.check_stage4, {"quick": args.quick, "settings_per_target": len(settings), "targets": names,
                                                   "placebos": n_pl, "seconds": round(time.time() - t0, 1)})
    print(json.dumps({k: summ[k] for k in ("judgements", "pass_by_window", "stage4_check") if k in summ}, ensure_ascii=False, default=str))
    print(figures(out, FIG_DIR if not args.quick else out / "figures"))


if __name__ == "__main__":
    main(sys.argv[1:])
