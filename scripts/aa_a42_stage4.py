"""A42 stage 4: dynamic layers (A38 and A40 together) on six targets (allocation DECISIONS A42 §4).

    .venv/Scripts/python.exe scripts/aa_a42_stage4.py --quick --out %TEMP%\\a42s4_quick
    .venv/Scripts/python.exe scripts/aa_a42_stage4.py --out docs/research/allocation/2026-11-02_a42_stage4
    ... --figures-only

Targets (user, 2026-11-02): R4 / R5 / R6 weights on 2024-12-31 (stage 3 targets.csv) and F2 안정형 40, F4 (가),
F3 현재 보유. Settings: 6 signals × 4 actions × confirmation 2 × tranches 3 × volatility target 3 × cooldown 2, plus
'no signal' × tranches × volatility target = 873 per target (`lab.aa_layers.all_settings`), each with the HAA50 + BAA50
Korean satellite at 0 / 25 / 50 %. Price sets: representative (real · h60 · CD91) and the two extremes (model · h60 ·
base rate, real · h0 · CD91). Research window 2000-09 – 2024-12 (each target from the first day all its products
have data).

Verdict (design §4, A40 kind, fixed before the run): on the representative set ① the MDD improvement over the static
target beats 95 % of 200 placebo signals (circular shifts ≥ 1 year, same setting), ② MDD at least 3 %p shallower than
the equal-cost static (the static that moves x of the way to the action's risk-off weights for good, x matching the
setting's CAGR), ③ CAGR at most 1 %p below the static target; and ② holds on both extreme sets.

Work is saved per (price set, target) and per (target, signal) placebo job under <out>/_work.
"""
from __future__ import annotations

import argparse
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
from lab import aa_pricesets as PS  # noqa: E402
from lab import aa_tactical as TAC  # noqa: E402
from lab.aa_assets import CASH, PRESETS  # noqa: E402
from lab.aa_korea import load_usdkrw  # noqa: E402
from lab.data import download_panel  # noqa: E402

START, END = PS.START, "2024-12-31"
SETS = {"대표": PS.PriceSet("real", 60.0, "cd91"), "극단 1": PS.PriceSet("model", 60.0, "base"),
        "극단 2": PS.PriceSet("real", 0.0, "cd91")}
SAT = (0.0, 0.25, 0.50)
PLACEBOS = 200
SEED = 4242
EQ_GRID = 41
MDD_GAIN_EQ = 0.03
CAGR_LOSS = 0.01
PLACEBO_Q = 0.95
STAGE3 = ROOT / "docs" / "research" / "allocation" / "2026-11-01_a42_stage3" / "targets.csv"
FIG_DIR = ROOT / "docs" / "research" / "allocation" / "figures" / "a42_stage4"
TARGET_NAMES = ("R4 동일 비중", "R5 역변동성", "R6 주식 위험 몫 상한", "F2 안정형 40", "F4 (가)", "F3 현재 보유")


def targets() -> dict[str, dict[str, float]]:
    t = pd.read_csv(STAGE3)
    out = {}
    for n in TARGET_NAMES:
        s = t.loc[t["대상"] == n, "비중"].iloc[0]
        out[n] = {c: float(x) for c, x in (p.split(":") for p in s.split(";"))}
    return out


def columns(T: dict) -> list[str]:
    cols = sorted({c for w in T.values() for c in w} | set(K.DEFENSIVE) | {LY.CD})
    return cols


# --------------------------------------------------------------------------- context (built once in the main process)

def build_context(quick: bool) -> dict:
    T = targets()
    cols = columns(T)
    usd, failed = download_panel(K.CANARY_TICKERS, end=PS.DATA_END)
    can_close, _ = K.chained(usd.close)
    tac_usd, _ = download_panel(TAC.ALL_SYMBOLS, end=PS.DATA_END)
    fx = load_usdkrw()
    ctx = {"targets": T, "columns": cols, "sets": {}}
    for sname, ps in SETS.items():
        p = PS.build(ps.prices, ps.hedge_bp, ps.cash, quirks=True)[0]
        inp = K.Inputs(can_close, usd.cash, fx, p, None)
        spec = TAC.TacticalSpec("HAA50+BAA50 국내판", start=START, end=END)
        sat = TAC.run_tactical(p, tac_usd.close, spec, "krw", tac_usd.cash).equity
        for tname, w in T.items():
            first = p.close[list(w)].apply(lambda s: s.first_valid_index()).max()
            start = str(max(pd.Timestamp(START), first).date())
            sim = K.DynamicSim(p, sorted(set(w) | set(K.DEFENSIVE) | {LY.CD}), start, END)
            dates = sim.dates
            qrows = K.quarter_rows(dates)
            states = {s: K.states(s, inp, K.RULES[s].params[0], dates).to_numpy() for s in LY.SIGNALS[1:]}
            base_tab = LY.action_tables(w, "none", "none", sim.columns)
            base_p, _, _ = sim.run(*K.static_rows(base_tab, sim))
            base_r = base_p[0, 1:] / base_p[0, :-1] - 1.0
            base_r = np.concatenate([[0.0], base_r])
            sat_path = sat.reindex(dates).to_numpy(float)
            ctx["sets"][(sname, tname)] = {"sim": sim, "qrows": qrows, "states": states, "base_path": base_p[0],
                                           "base_ret": base_r, "sat": sat_path, "start": start}
        print(f"  준비: {sname} ({ps.label})", flush=True)
    return ctx


_CTX: dict = {}


def _init(ctx):
    _CTX.update(ctx)


def _levels(c, vol):
    return LY.vol_levels(c["base_ret"], c["sim"].dates, vol) if vol else None


def _metrics(P, base_cagr):
    return LY.cagr(P), LY.mdd(P)


# --------------------------------------------------------------------------- job A: real settings on one set

def job_real(args):
    sname, tname, settings = args
    c = _CTX["sets"][(sname, tname)]
    w = _CTX["targets"][tname]
    sim, cols = c["sim"], c["sim"].columns
    levs = {v: _levels(c, v) for v in (0.0, 0.08, 0.10)}
    base = c["base_path"][None, :]
    rows = []
    for s in SAT:
        bm = LY.mix(base, c["sat"], s)
        rows.append({"세트": sname, "대상": tname, "종류": "정적", "위성": s, "cagr": float(LY.cagr(bm)[0]), "mdd": float(LY.mdd(bm)[0])})
    # equal-cost static grids per move (signal, action), mixed with each satellite share
    grids = {}
    for st in settings:
        key = eq_key(st)
        if key in grids:
            continue
        b0 = LY.action_tables(w, "none", "none", cols)[0]
        full = LY.full_move(w, key[0], key[1], cols)
        xs = np.linspace(0, 1, EQ_GRID)
        W = np.vstack([b0 + x * (full - b0) for x in xs])
        P, _, _ = sim.run(*K.static_rows(W, sim))
        grids[key] = {s: (LY.cagr(LY.mix(P, c["sat"], s)), LY.mdd(LY.mix(P, c["sat"], s))) for s in SAT}
    for st in settings:
        tab = LY.action_tables(w, st.signal, st.action, cols)
        S = c["states"][st.signal][None, :] if st.signal != "none" else np.zeros((1, sim.n_days), dtype=int)
        P = LY.paths(sim, tab, S, st, c["qrows"], levs[st.vol])
        for s in SAT:
            m = LY.mix(P, c["sat"], s)
            cg, md = float(LY.cagr(m)[0]), float(LY.mdd(m)[0])
            gc, gm = grids[eq_key(st)][s]
            x, eq_mdd = equal_cost(gc, gm, cg)
            rows.append({"세트": sname, "대상": tname, "종류": "설정", "위성": s, **st.__dict__, "설정 이름": st.label(),
                         "cagr": cg, "mdd": md, "같은 비용 x": x, "같은 비용 mdd": eq_mdd})
    return pd.DataFrame(rows)


def eq_key(st: LY.Setting) -> tuple[str, str]:
    """The move an equal-cost static makes: toward the action's risk-off weights, or toward the CD-rate ETF when a
    volatility target is on (it moves money to CD) or there is no signal (decided after --quick, before the run)."""
    return ("none", "none") if (st.vol or st.signal == "none") else (st.signal, st.action)


def equal_cost(gc: np.ndarray, gm: np.ndarray, target: float) -> tuple[float, float]:
    """x on the grid (0 … 1) whose CAGR matches `target` (linear between grid points) and the MDD there."""
    xs = np.linspace(0, 1, len(gc))
    if target >= gc[0]:
        return 0.0, float(gm[0])
    if target <= gc.min():
        k = int(np.argmin(gc))
        return float(xs[k]), float(gm[k])
    k = int(np.flatnonzero(gc <= target)[0])
    f = (gc[k - 1] - target) / (gc[k - 1] - gc[k])
    return float(xs[k - 1] + f * (xs[k] - xs[k - 1])), float(gm[k - 1] + f * (gm[k] - gm[k - 1]))


# --------------------------------------------------------------------------- job B: placebos on the representative set

def job_placebo(args):
    tname, signal, settings, n = args
    c = _CTX["sets"][("대표", tname)]
    w = _CTX["targets"][tname]
    sim = c["sim"]
    seed = SEED + 100 * TARGET_NAMES.index(tname) + LY.SIGNALS.index(signal)
    S, shifts = K.placebo_states(c["states"][signal], n, seed)
    levs = {v: _levels(c, v) for v in (0.0, 0.08, 0.10)}
    rows = []
    for st in settings:
        tab = LY.action_tables(w, st.signal, st.action, sim.columns)
        P = LY.paths(sim, tab, S, st, c["qrows"], levs[st.vol])
        for s in SAT:
            m = LY.mdd(LY.mix(P, c["sat"], s))
            rows.append({"대상": tname, **st.__dict__, "위성": s, "가짜 mdd q95": float(np.quantile(m, PLACEBO_Q)),
                         "가짜 mdd 중앙": float(np.median(m)), "가짜 수": len(m)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- verdict

KEY = ["대상", "signal", "action", "confirm", "tranches", "vol", "cooldown", "위성"]


def verdicts(R: pd.DataFrame, PL: pd.DataFrame) -> pd.DataFrame:
    st = R[R["종류"] == "정적"]
    plain = st[st["위성"] == 0.0].set_index(["세트", "대상"])
    sat_static = st.set_index(["세트", "대상", "위성"])
    s = R[R["종류"] == "설정"].copy()
    s["정적 cagr"] = [plain.loc[(a, b), "cagr"] for a, b in zip(s["세트"], s["대상"])]
    s["정적 mdd"] = [plain.loc[(a, b), "mdd"] for a, b in zip(s["세트"], s["대상"])]
    s["같은 위성 정적 mdd"] = [sat_static.loc[(a, b, c), "mdd"] for a, b, c in zip(s["세트"], s["대상"], s["위성"])]
    s["같은 비용 대비 MDD 개선"] = s["mdd"] - s["같은 비용 mdd"]
    rep = s[s["세트"] == "대표"].copy()
    rep = rep.merge(PL[KEY + ["가짜 mdd q95", "가짜 mdd 중앙"]], on=KEY, how="left")
    rep["MDD 개선"] = rep["mdd"] - rep["정적 mdd"]
    rep["CAGR 손실"] = rep["정적 cagr"] - rep["cagr"]
    rep["① 가짜 95%보다 개선"] = rep["mdd"] > rep["가짜 mdd q95"]  # NaN (no signal) → False
    rep["② 같은 비용보다 3%p"] = rep["같은 비용 대비 MDD 개선"] >= MDD_GAIN_EQ
    rep["③ CAGR 손실 ≤ 1%p"] = rep["CAGR 손실"] <= CAGR_LOSS
    for ex in ("극단 1", "극단 2"):
        e = s[s["세트"] == ex].set_index(KEY)["같은 비용 대비 MDD 개선"]
        rep[f"② {ex}"] = (e.reindex(pd.MultiIndex.from_frame(rep[KEY])).to_numpy() >= MDD_GAIN_EQ)
    rep["극단 두 세트 ② 유지"] = rep["② 극단 1"] & rep["② 극단 2"]
    rep["행동 대상 없음"] = (rep["signal"] != "none") & (rep["action"].isin(["defensive", "safe", "tilt_nd"])) & \
        rep["대상"].map(lambda n: _TARGETS[n].get(K.NASDAQ, 0) == 0)
    rep["판정"] = np.where(rep["① 가짜 95%보다 개선"] & rep["② 같은 비용보다 3%p"] & rep["③ CAGR 손실 ≤ 1%p"] &
                         rep["극단 두 세트 ② 유지"], "통과", "탈락")
    return rep


_TARGETS: dict = {}


# --------------------------------------------------------------------------- figures

def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    dest.mkdir(parents=True, exist_ok=True)
    V = pd.read_csv(out / "verdicts.csv.gz", keep_default_na=False, na_values=[""])
    made = []
    v0 = V[V["위성"] == 0.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for ax, (tname, g) in zip(axes.ravel(), v0.groupby("대상", sort=False)):
        for sig, col in zip(LY.SIGNALS, ["#BBBBBB", "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]):
            h = g[g["signal"] == sig]
            ax.scatter(h["CAGR 손실"] * 100, h["MDD 개선"] * 100, s=6, alpha=0.5, color=col,
                       label=("신호 없음" if sig == "none" else K.RULES[sig].label))
        ok = g[g["판정"] == "통과"]
        ax.scatter(ok["CAGR 손실"] * 100, ok["MDD 개선"] * 100, s=40, facecolors="none", edgecolors="black", label="판정 통과")
        ax.axvline(1, color="gray", lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(tname)
    for ax in axes[-1]:
        ax.set_xlabel("CAGR 손실 (%p, 정적 대비)")
    for ax in axes[:, 0]:
        ax.set_ylabel("MDD 개선 (%p, 정적 대비)")
    axes[0, 0].legend(fontsize=6, markerscale=2)
    fig.suptitle("① 동적 층 873가지: 정적 대비 CAGR 손실 vs MDD 개선 (대표 세트, 위성 0%)\n점선 = CAGR 손실 1%p, 검은 원 = 판정 통과")
    made.append(str(figs._save(fig, dest / "1_tradeoff.png")))
    fig, ax = plt.subplots(figsize=(11, 5))
    sig = v0[v0["signal"] != "none"]
    piv = sig.groupby(["대상", "signal"])["① 가짜 95%보다 개선"].mean().unstack().reindex(TARGET_NAMES) * 100
    im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iat[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(piv.shape[1]), [K.RULES[k].label for k in piv.columns], rotation=15)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_title("② 판정 ①(가짜 신호 200개의 95%보다 MDD 개선)을 넘은 설정 비율 (%), 신호별, 위성 0%")
    fig.colorbar(im, ax=ax)
    made.append(str(figs._save(fig, dest / "2_placebo.png")))
    fig, ax = plt.subplots(figsize=(11, 5))
    checks = ["① 가짜 95%보다 개선", "② 같은 비용보다 3%p", "③ CAGR 손실 ≤ 1%p"]
    cnt = (V.assign(**{c: V[c].astype(str).eq("True") for c in checks}).groupby(["대상", "위성"])[checks].mean() * 100).reset_index()
    x = np.arange(len(TARGET_NAMES))
    for k, col in enumerate(["① 가짜 95%보다 개선", "② 같은 비용보다 3%p", "③ CAGR 손실 ≤ 1%p"]):
        vals = cnt[cnt["위성"] == 0.0].set_index("대상")[col].reindex(TARGET_NAMES)
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=col)
    ax.set_xticks(x, TARGET_NAMES, rotation=15)
    ax.set_ylabel("설정 비율 (%)")
    ax.set_title("③ 판정 항목별 통과 비율 (대표 세트, 위성 0%, 신호 없음 포함 873가지)")
    ax.legend(fontsize=8)
    made.append(str(figs._save(fig, dest / "3_checks.png")))
    return made


# --------------------------------------------------------------------------- main

def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _TARGETS.update(targets())
    if args.figures_only:
        print(figures(out))
        return
    work = out / "_work"
    work.mkdir(exist_ok=True)
    t0 = time.time()
    settings = LY.all_settings()
    n_pl = PLACEBOS
    names = list(TARGET_NAMES)
    if args.quick:
        settings = [s for s in settings if s.signal in ("none", "k4") and s.action in ("none", "riskcd", "safe")][::6]
        n_pl, names = 8, names[:2]
    ctx = build_context(args.quick)
    ctx["targets"] = {n: ctx["targets"][n] for n in names}
    jobs_a = [(sn, tn, settings) for sn in SETS for tn in names if not (work / f"a_{sn}_{names.index(tn)}.pkl").exists()]
    sigs = sorted({s.signal for s in settings} - {"none"}, key=LY.SIGNALS.index)
    jobs_b = [(tn, sg, [s for s in settings if s.signal == sg], n_pl) for tn in names for sg in sigs
              if not (work / f"b_{names.index(tn)}_{sg}.pkl").exists()]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ctx,)) as ex:
        futs = {ex.submit(job_real, j): ("a", j) for j in jobs_a}
        futs.update({ex.submit(job_placebo, j): ("b", j) for j in jobs_b})
        for k, (f, (kind, j)) in enumerate(futs.items()):
            df = f.result()
            name = f"a_{j[0]}_{names.index(j[1])}.pkl" if kind == "a" else f"b_{names.index(j[0])}_{j[1]}.pkl"
            df.to_pickle(work / name)  # this script's own work files, read back only by it
            print(f"  {k + 1}/{len(futs)} {kind} {j[0]} {j[1]} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat([pd.read_pickle(f) for f in sorted(work.glob("a_*.pkl"))], ignore_index=True)
    PL = pd.concat([pd.read_pickle(f) for f in sorted(work.glob("b_*.pkl"))], ignore_index=True)
    R.to_csv(out / "runs.csv.gz", index=False)
    PL.to_csv(out / "placebos.csv.gz", index=False)
    V = verdicts(R, PL)
    V.to_csv(out / "verdicts.csv.gz", index=False)
    passed = V[V["판정"] == "통과"]
    passed.to_csv(out / "passed.csv", index=False)
    summ = {"quick": args.quick, "settings_per_target": len(settings), "targets": names, "placebos": n_pl,
            "seconds": round(time.time() - t0, 1), "passed": int(len(passed)),
            "passed_by_target": passed.groupby("대상").size().to_dict(),
            "check_shares": {c: float(V[c].mean()) for c in ["① 가짜 95%보다 개선", "② 같은 비용보다 3%p", "③ CAGR 손실 ≤ 1%p",
                                                             "극단 두 세트 ② 유지"]}}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=str)[:1500])
    print(figures(out, FIG_DIR if not args.quick else out / "figures"))


if __name__ == "__main__":
    main(sys.argv[1:])
