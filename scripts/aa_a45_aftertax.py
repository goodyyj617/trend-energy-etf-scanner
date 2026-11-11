"""A45: after-tax final table of the recommended allocations, and whether the 10 % volatility-target layer survives the
account conditions (allocation DECISIONS A45, design docs/research/allocation/2026-11-11_a45_design.md).

    .venv/Scripts/python.exe scripts/aa_a45_aftertax.py --quick --out %TEMP%\\a45_quick
    .venv/Scripts/python.exe scripts/aa_a45_aftertax.py --out docs/research/allocation/2026-11-11_a45_aftertax
    ... --figures-only

Seven targets ((가), 안정형 40, R6 + vol target 10 %, R6 static, R6 at the same average exposure, 현재 보유, S&P500) ×
the baseline and the 648 stage-3b policies, representative price set, each target's own window to 2024-12; the common
window from 2003-11-07 with the baseline and the declared policy only. Before the sweep two reproduction checks must
pass (the stage-5 reference row, the stage-3b verdicts), otherwise the run stops. Each target is saved under
<out>/_work when done, so a stopped run continues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import aa_accounts as AC  # noqa: E402
from lab import aa_aftertax as AT  # noqa: E402
from lab import aa_layers as LY  # noqa: E402
from lab import aa_pricesets as PS  # noqa: E402

ALLOC = ROOT / "docs" / "research" / "allocation"
STAGE3_TARGETS = ALLOC / "2026-11-01_a42_stage3" / "targets.csv"
STAGE3B = ALLOC / "2026-11-02_a42_stage3b" / "verdicts.csv"
STAGE5 = ALLOC / "2026-11-02_a42_stage5" / "runs.csv.gz"
FIG_DIR = ALLOC / "figures" / "a45_aftertax"
REP = PS.PriceSet("real", 60.0, "cd91")
START, END = PS.START, "2024-12-31"
TOL = 1e-6
POLICY_COLS = ["accounts", "deposits", "execution", "cost", "tax", "limit", "overflow"]
REPRO_POLICY = AC.Policy("single", "lump", "full", "flat", "none", "none")  # stage 5: 10 bp, first purchase free


def load_targets(path: Path = STAGE3_TARGETS) -> dict[str, dict[str, float]]:
    t = pd.read_csv(path, dtype=str)
    return {n: {c: float(x) for c, x in (p.split(":") for p in s.split(";"))} for n, s in zip(t["대상"], t["비중"])}


def arrays(p, w: dict[str, float], start: str = START, end: str = END):
    """codes, dates, daily returns, cash, weights from the first day all products have data (and `start`) to `end`."""
    codes = list(w)
    close = p.close[codes]
    first = close.apply(lambda s: s.first_valid_index()).max()
    idx = p.close.index
    dates = idx[(idx >= max(pd.Timestamp(start), first)) & (idx <= pd.Timestamp(end))]
    rets = close.ffill().pct_change(fill_method=None).loc[dates].to_numpy()
    cash = p.cash.reindex(idx).fillna(0.0).loc[dates].to_numpy()
    return codes, dates, rets, cash, np.array([w[c] for c in codes], float)


def build(p, start: str = START) -> tuple[dict, dict]:
    """Per target: (codes, dates, rets, cash, weights, targets or None); and facts about the vol layer."""
    T3 = load_targets()
    ctx = {}
    for name in AT.ORDER:
        if name in AT.STATIC_OF:
            codes, dates, rets, cash, wv = arrays(p, T3[AT.STATIC_OF[name]], start)
            ctx[name] = (codes, dates, rets, cash, wv, None)
    w6 = T3[AT.STATIC_OF["R6 정적"]]
    codes, dates, rets, cash, wv = arrays(p, w6, start)
    vdates, lev = AT.vol_levels_of(p, w6, str(dates[0].date()), END)
    if not vdates.equals(dates):
        raise RuntimeError("변동성 목표의 날짜가 R6 정적의 날짜와 다릅니다")
    ctx[AT.VOL_NAME] = (codes, dates, rets, cash, wv, AT.vol_targets(w6, codes, lev))
    kbar = AT.mean_exposure(lev)
    mw = AT.mean_exposure_weights(w6, kbar)
    codes_m, dates_m, rets_m, cash_m, wv_m = arrays(p, mw, start)
    ctx[AT.MEAN_NAME] = (codes_m, dates_m, rets_m, cash_m, wv_m, None)
    k = LY.K_LEVELS[lev]
    facts = {"k_mean": kbar, "k_min": float(k.min()), "k_share_below_1": float(np.mean(k < 1.0)),
             "level_changes": int(np.sum(lev[1:] != lev[:-1])), "start": str(dates[0].date()),
             "mean_exposure_weights": {c: round(x, 4) for c, x in mw.items()},
             "exposure": pd.Series(k, index=dates)}
    return {n: ctx[n] for n in AT.ORDER}, facts


def run_one(x, pol: AC.Policy, **kw) -> dict:
    codes, dates, rets, cash, wv, tg = x
    return AC.run_policy(rets, cash, dates, codes, None if tg is not None else wv, pol, targets=tg, **kw)


def result_row(name: str, pol: AC.Policy, r: dict, dates, declared: AC.Policy) -> dict:
    return {"대상": name, **asdict(pol), "정책": pol.label(), "기준선": pol == AC.BASELINE, "선언 정책": pol == declared,
            **{k: r[k] for k in ("twr_cagr", "twr_mdd", "irr", "final", "deposited", "general_cash_end",
                                 "general_invested_end", "years")},
            "세금 (원)": r["tax"], "비용 (원)": r["cost"], "시작": str(dates[0].date())}


_CTX: dict = {}


def _init(ctx):
    _CTX.update(ctx)


def _job(args):
    name, pol = args
    r = run_one(_CTX[name], pol)
    return result_row(name, pol, r, _CTX[name][1], AC.DECLARED)


def reproduction(ctx: dict) -> dict:
    """(a) the vol allocation under the stage-5 condition vs stage 5's reference row; (b) statics vs stage 3b."""
    out = {"ok": True}
    s5 = pd.read_csv(STAGE5)
    ref = s5[(s5["세트"] == REP.label) & (s5["이름"] == AT.STAGE5_REF)].iloc[0]
    r = run_one(ctx[AT.VOL_NAME], REPRO_POLICY, charge_first=False)
    a = {"cagr": r["twr_cagr"], "mdd": r["twr_mdd"], "stage5_cagr": float(ref["cagr"]), "stage5_mdd": float(ref["mdd"]),
         "stage5_start": ref["시작"], "start": str(ctx[AT.VOL_NAME][1][0].date())}
    a["diff"] = max(abs(a["cagr"] - a["stage5_cagr"]), abs(a["mdd"] - a["stage5_mdd"]))
    a["ok"] = bool(a["diff"] < TOL and a["start"] == a["stage5_start"])
    out["a_stage5_vol"] = a
    v3 = pd.read_csv(STAGE3B).set_index("대상")
    b = []
    for name, s3 in AT.STATIC_OF.items():
        base, dec = run_one(ctx[name], AC.BASELINE), run_one(ctx[name], AC.DECLARED)
        row = {"대상": name, "기준선 CAGR": base["twr_cagr"], "기준선 MDD": base["twr_mdd"],
               "선언 CAGR": dec["twr_cagr"], "선언 MDD": dec["twr_mdd"]}
        ref3 = v3.loc[s3]
        row["diff"] = float(max(abs(row["기준선 CAGR"] - ref3["기준선 CAGR"]), abs(row["기준선 MDD"] - ref3["기준선 MDD"]),
                                abs(row["선언 CAGR"] - ref3["선언 CAGR (TWR)"]), abs(row["선언 MDD"] - ref3["선언 MDD (TWR)"])))
        row["ok"] = bool(row["diff"] < TOL)
        b.append(row)
    out["b_stage3b"] = b
    out["ok"] = a["ok"] and all(x["ok"] for x in b)
    return out


def sweep(ctx: dict, pols: list, work: Path, workers: int, tag: str) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for i, n in enumerate(ctx):
        f = work / f"{tag}_{i}.csv"
        if f.exists():
            rows.append(pd.read_csv(f, keep_default_na=False, na_values=[""]))
            continue
        _init(ctx)
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(ctx,)) as ex:
            part = pd.DataFrame(list(ex.map(_job, [(n, pol) for pol in pols], chunksize=4)))
        part.to_csv(f, index=False)
        rows.append(part)
        print(f"  [{tag}] {n}: 정책 {len(part)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(rows, ignore_index=True)


def _bool(R: pd.DataFrame) -> pd.DataFrame:
    for c in ("기준선", "선언 정책"):
        R[c] = R[c].astype(str).str.lower().isin(("true", "1"))
    return R


def analyse(R: pd.DataFrame, C: pd.DataFrame, facts: dict) -> dict:
    T = AT.table_rows(R)
    Tc = AT.table_rows(C)
    L = AT.layer_rows(R, POLICY_COLS)
    dec = L[L["선언 정책"]].iloc[0]
    pol = L[~L["기준선"]]
    t = T.set_index("대상")
    gap = lambda n: float(t.loc[n, "CAGR 차"])  # noqa: E731
    return {
        "table": T, "common": Tc, "layer": L,
        "summary": {
            "X1": {n: t.loc[n, "X1 세후 구역"] for n in T["대상"]},
            "X2": {n: bool(t.loc[n, "X2 표시"]) for n in T["대상"]},
            "X3": dec["X3"],
            "X3_numbers": {k: float(dec[k]) for k in ("층 CAGR", "층 MDD", "평균 노출 CAGR", "평균 노출 MDD", "정적 CAGR",
                                                      "정적 MDD", "MDD 이점 (vs 평균 노출)", "CAGR 손실 (vs 평균 노출)",
                                                      "MDD 이점 (vs 정적)", "CAGR 손실 (vs 정적)")},
            "X3_share_keep": float((pol["X3"] == "층 유지").mean()),
            "X3_share": {k: float((pol["X3"] == k).mean()) for k in ("층 유지", "층 비쌈", "층 없음")},
            "layer_extra_cost_pp": (gap(AT.VOL_NAME) - gap("R6 정적")) * 100,
            "recommended": {n: {"세후 CAGR": float(t.loc[n, "세후 CAGR (TWR)"]), "세후 MDD": float(t.loc[n, "세후 MDD (TWR)"]),
                                "세후 IRR": float(t.loc[n, "세후 IRR"]), "구역": t.loc[n, "X1 세후 구역"]}
                            for n in AT.RECOMMENDED},
            "vol_layer": {k: v for k, v in facts.items() if k != "exposure"},
        }}


def run(out: Path, quick: bool, workers: int) -> dict:
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    p = PS.build(REP.prices, REP.hedge_bp, REP.cash, quirks=True)[0]
    ctx, facts = build(p)
    facts["exposure"].rename("k").to_csv(out / "exposure.csv", index_label="date")
    rep = reproduction(ctx)
    (out / "repro.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("재현 점검:", "통과" if rep["ok"] else "실패", flush=True)
    if not rep["ok"]:
        raise SystemExit("재현 점검 실패: repro.json을 보세요. 더 돌리지 않습니다.")
    pols = [AC.BASELINE] + AC.all_policies(overflow=True)
    if quick:
        pols = [AC.BASELINE, AC.DECLARED] + [q for q in AC.all_policies(overflow=True)[::60] if q != AC.DECLARED]
    R = _bool(sweep(ctx, pols, work, workers, "own"))
    R.to_csv(out / "policies.csv.gz", index=False)
    cctx, _ = build(p, AT.COMMON_START)
    C = _bool(sweep(cctx, [AC.BASELINE, AC.DECLARED], work, workers, "common"))
    C.to_csv(out / "common.csv", index=False)
    A = analyse(R, C, facts)
    A["table"].to_csv(out / "table.csv", index=False)
    A["common"].to_csv(out / "table_common.csv", index=False)
    A["layer"].to_csv(out / "layer.csv", index=False)
    return {"repro": rep, **A["summary"], "policies": len(pols), "quick": quick}


# --------------------------------------------------------------------------- figures

def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    dest.mkdir(parents=True, exist_ok=True)
    made = []
    T = pd.read_csv(out / "table.csv")
    Tc = pd.read_csv(out / "table_common.csv")
    L = pd.read_csv(out / "layer.csv")
    R = _bool(pd.read_csv(out / "policies.csv.gz", keep_default_na=False, na_values=[""]))
    E = pd.read_csv(out / "exposure.csv", index_col=0, parse_dates=True)["k"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(T))
    for a, (b, d, title) in zip(axes, (("기준선 CAGR", "세후 CAGR (TWR)", "CAGR (%)"), ("기준선 MDD", "세후 MDD (TWR)", "MDD (%)"))):
        a.bar(x - 0.2, T[b] * 100, 0.4, label="한 계좌 · 목돈 · 비용·세금 없음", color="#4C78A8")
        a.bar(x + 0.2, T[d] * 100, 0.4, label="선언 정책 세후 (TWR)", color="#F58518")
        if "MDD" in b:
            for y, c in ((-15, "#E45756"), (-17, "#B279A2")):
                a.axhline(y, color=c, lw=0.8, ls="--")
            for i, r in T.iterrows():
                a.text(i + 0.2, r[d] * 100 - 1.0, r["X1 세후 구역"], ha="center", fontsize=8)
        a.set_xticks(x, T["대상"], rotation=20, fontsize=8)
        a.set_ylabel(title)
        a.legend(fontsize=8)
    axes[0].set_title("① CAGR: 기준선 vs 실제 계좌 조건 세후 (분리 계좌 · 매월 입금 · 상품별 비용 · ISA 일반형 · 한도)")
    axes[1].set_title("MDD와 세후 구역 (−15% 안 / −17% 경계)")
    made.append(str(figs._save(fig, dest / "1_after_tax.png")))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    a = axes[0]
    pol = L[~L["기준선"].astype(bool)]
    a.scatter(pol["MDD 이점 (vs 평균 노출)"] * 100, pol["CAGR 손실 (vs 평균 노출)"] * 100, s=8, alpha=0.4, color="#4C78A8",
              label=f"정책 {len(pol)}가지")
    d = L[L["선언 정책"].astype(bool)]
    b = L[L["기준선"].astype(bool)]
    a.scatter(d["MDD 이점 (vs 평균 노출)"] * 100, d["CAGR 손실 (vs 평균 노출)"] * 100, s=60, color="#F58518", label="선언 정책")
    a.scatter(b["MDD 이점 (vs 평균 노출)"] * 100, b["CAGR 손실 (vs 평균 노출)"] * 100, s=60, marker="s", color="#54A24B",
              label="기준선")
    a.axvline(AT.LAYER_MDD_PP * 100, color="#E45756", ls="--", lw=0.8)
    a.axhline(AT.LAYER_CAGR_PP * 100, color="#E45756", ls="--", lw=0.8)
    a.set_xlabel("MDD 이점 (%p, +면 변동성 목표가 얕음)")
    a.set_ylabel("CAGR 손실 (%p, +면 변동성 목표가 낮음)")
    a.set_title("② 변동성 목표 10% vs 같은 평균 노출의 정적 (R6 × k̄ + CD)\n오른쪽 아래 칸 = 층 유지")
    a.legend(fontsize=8)
    axes[1].plot(E.index, E.to_numpy(), color="#4C78A8", lw=0.9)
    axes[1].axhline(float(E.mean()), color="#F58518", ls="--", lw=0.8, label=f"평균 k̄ = {E.mean():.2f}")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("R6 + 변동성 목표 10%의 노출 k (매달 말 결정)")
    axes[1].legend(fontsize=8)
    made.append(str(figs._save(fig, dest / "2_layer.png")))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    p = R[~R["기준선"]]
    order = list(dict.fromkeys(p["대상"]))
    for a, col, title in zip(axes, ("twr_cagr", "twr_mdd"), ("TWR CAGR (%)", "TWR MDD (%)")):
        a.boxplot([p.loc[p["대상"] == n, col].to_numpy() * 100 for n in order], showfliers=True)
        base = [R[(R["대상"] == n) & R["기준선"]][col].iloc[0] * 100 for n in order]
        decl = [R[(R["대상"] == n) & R["선언 정책"]][col].iloc[0] * 100 for n in order]
        a.scatter(range(1, len(order) + 1), base, color="#54A24B", marker="s", zorder=3, label="기준선")
        a.scatter(range(1, len(order) + 1), decl, color="#F58518", zorder=3, label="선언 정책")
        a.set_xticks(range(1, len(order) + 1), order, rotation=20, fontsize=8)
        a.set_ylabel(title)
        a.legend(fontsize=8)
    axes[0].set_title(f"③ 정책 {p.groupby('대상').size().iloc[0]}가지에 걸친 분포 (대상별)")
    for y in (-15, -17):
        axes[1].axhline(y, color="#E45756", lw=0.8, ls="--")
    made.append(str(figs._save(fig, dest / "3_spread.png")))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(Tc))
    ax.bar(x - 0.2, Tc["세후 CAGR (TWR)"] * 100, 0.4, label="세후 CAGR (TWR)", color="#F58518")
    ax.bar(x + 0.2, -Tc["세후 MDD (TWR)"] * 100, 0.4, label="세후 |MDD| (TWR)", color="#E45756")
    ax.set_xticks(x, Tc["대상"], rotation=20, fontsize=8)
    ax.set_ylabel("%")
    ax.set_title(f"④ 공통 구간 {AT.COMMON_START} – 2024-12 (모든 대상 같은 기간, 선언 정책 세후)")
    ax.legend(fontsize=8)
    made.append(str(figs._save(fig, dest / "4_common_window.png")))
    return made


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.figures_only:
        t0 = time.time()
        summ = run(out, args.quick, args.workers)
        summ["seconds"] = round(time.time() - t0, 1)
        (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: summ[k] for k in ("X1", "X2", "X3", "X3_share", "layer_extra_cost_pp", "recommended")},
                         ensure_ascii=False, indent=1, default=str))
    print(figures(out, FIG_DIR if not args.quick else out / "figures"))


if __name__ == "__main__":
    main(sys.argv[1:])
