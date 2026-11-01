"""A42 stage 3: operating policies (accounts × deposits × execution × cost × tax × ISA limit = 324) on the stage-2
targets (allocation DECISIONS A42 §3).

    .venv/Scripts/python.exe scripts/aa_a42_stage3.py --quick --out %TEMP%\\a42s3_quick
    .venv/Scripts/python.exe scripts/aa_a42_stage3.py --out docs/research/allocation/2026-11-01_a42_stage3
    .venv/Scripts/python.exe scripts/aa_a42_stage3.py --overflow --out docs/research/allocation/2026-11-02_a42_stage3b
    ... --figures-only

--overflow (stage 3b, added after the stage-3 results): one more axis, what the general account does with the money
over the ISA limit (cash / invested in the same ETFs), so 648 policies; the declared policy becomes the 'invest' one
(post hoc, design change log). Without it the run is stage 3 as registered (324, declared policy with cash).

Targets (user, 2026-11-01): R4 / R5 / R6 weights on 2024-12-31 and F1–F4; F5 S&P500 as the comparison line. The
representative price set (real · h60 · CD91), research window 2000-09 – 2024-12 (each target from the first day all its
products have data), quarterly decisions. Each target is saved under <out>/_work when done, so a stopped run continues.
The verdict only marks: TWR CAGR of '한 계좌 · 목돈 · 비용 없음' minus the declared policy after tax > 0.5 %p, or the MDD
gap > 2 %p.
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

from lab import aa_accounts as AC  # noqa: E402
from lab import aa_pricesets as PS  # noqa: E402
from lab import aa_wide as A  # noqa: E402
from lab.aa_assets import PRESETS  # noqa: E402
from lab.aa_composition import whole  # noqa: E402

REP = PS.PriceSet("real", 60.0, "cd91")
START, END = PS.START, "2024-12-31"
CAGR_PP, MDD_PP = 0.005, 0.02
AXIS_KO = {"accounts": "계좌", "deposits": "입금", "execution": "실행", "cost": "비용", "tax": "세금", "limit": "한도",
           "overflow": "한도 밖"}
FIG_DIR = ROOT / "docs" / "research" / "allocation" / "figures" / "a42_stage3"
FIXED = {"F1 60/40": {"379800": 60, "0091C0": 40}, "F2 안정형 40": "stable40", "F3 현재 보유": "aaa_2026",
         "F4 (가)": "a41_ga", "F5 S&P500": "sp500"}


def targets(p) -> dict[str, dict[str, float]]:
    rets = p.close[A.U17].ffill().pct_change(fill_method=None)
    idx = rets.index[rets.index <= pd.Timestamp(END)]
    row = int(rets.index.get_loc(idx[-1]))
    out = {}
    names = {"R4": "R4 동일 비중", "R5": "R5 역변동성", "R6": "R6 주식 위험 몫 상한"}
    for rule in ("R4", "R5", "R6"):
        wd = A.rule_weights(rule, rets, A.U17, row)
        vec = whole(np.array([wd.get(c, 0.0) for c in A.U17]) * 100)
        out[names[rule]] = {c: float(x) for c, x in zip(A.U17, vec) if x > 0}
    for name, w in FIXED.items():
        wt = w if isinstance(w, dict) else PRESETS[w].weights
        out[name] = {c: float(x) for c, x in wt.items()}
    return out


def target_arrays(p, w: dict[str, float]):
    codes = list(w)
    close = p.close[codes]
    first = close.apply(lambda s: s.first_valid_index()).max()
    idx = p.close.index
    dates = idx[(idx >= max(pd.Timestamp(START), first)) & (idx <= pd.Timestamp(END))]
    rets = close.ffill().pct_change(fill_method=None).loc[dates].to_numpy()
    cash = p.cash.reindex(idx).fillna(0.0).loc[dates].to_numpy()
    return codes, dates, rets, cash, np.array([w[c] for c in codes], float)


_CTX: dict = {}


def _init(ctx):
    _CTX.update(ctx)


def _one(args):
    name, pol = args
    codes, dates, rets, cash, wv = _CTX[name]
    r = AC.run_policy(rets, cash, dates, codes, wv, pol)
    return {"대상": name, **AC.policy_row(pol), "기준선": pol == AC.BASELINE, "선언 정책": pol == _CTX["__declared"],
            **{k: r[k] for k in ("twr_cagr", "twr_mdd", "irr", "final", "deposited", "general_cash_end", "years")},
            **({"general_invested_end": r["general_invested_end"]} if _CTX["__overflow"] else {}),
            "세금 (원)": r["tax"], "비용 (원)": r["cost"],  # not "tax" / "cost": those are the policy's axis columns
            "시작": str(dates[0].date())}


def run(out: Path, quick: bool, workers: int, overflow: bool = False) -> dict:
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    p = PS.build(REP.prices, REP.hedge_bp, REP.cash, quirks=True)[0]
    T = targets(p)
    pd.DataFrame([{"대상": k, "비중": ";".join(f"{c}:{int(x)}" for c, x in v.items())} for k, v in T.items()]).to_csv(
        out / "targets.csv", index=False)
    declared = AC.DECLARED if overflow else AC.DECLARED_3
    pols = [AC.BASELINE] + AC.all_policies(overflow)
    names = list(T)
    if quick:
        pols = [AC.BASELINE, declared, AC.DECLARED_3] + AC.all_policies(overflow)[::40]
        names = names[:2]
    ctx = {n: target_arrays(p, T[n]) for n in names}
    ctx["__declared"], ctx["__overflow"] = declared, overflow
    rows = []
    t0 = time.time()
    for n in names:
        f = work / f"t_{names.index(n)}.csv"
        if f.exists():
            rows.append(pd.read_csv(f))
            continue
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(ctx,)) as ex:
            part = pd.DataFrame(list(ex.map(_one, [(n, pol) for pol in pols], chunksize=4)))
        part.to_csv(f, index=False)
        rows.append(part)
        print(f"  {n}: {len(part)} 정책 ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(out / "policies.csv.gz", index=False)
    V = verdicts(R)
    V.to_csv(out / "verdicts.csv", index=False)
    E = axis_effects(R)
    E.to_csv(out / "axis_effects.csv", index=False)
    return {"targets": names, "policies": len(pols), "verdicts": V.to_dict("records")}


def verdicts(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in R.groupby("대상", sort=False):
        b = g[g["기준선"]].iloc[0]
        d = g[g["선언 정책"]]
        if d.empty:
            continue
        d = d.iloc[0]
        pol = g[~g["기준선"]]
        row = {"대상": n, "기준선 CAGR": b["twr_cagr"], "기준선 MDD": b["twr_mdd"], "선언 CAGR (TWR)": d["twr_cagr"],
               "선언 MDD (TWR)": d["twr_mdd"], "선언 IRR": d["irr"], "CAGR 차": b["twr_cagr"] - d["twr_cagr"],
               "MDD 차": b["twr_mdd"] - d["twr_mdd"], "선언 세금 (원)": d["세금 (원)"], "선언 비용 (원)": d["비용 (원)"],
               "선언 입금 (원)": d["deposited"], "선언 최종 (원)": d["final"],
               f"{len(pol)} CAGR 최소": pol["twr_cagr"].min(), f"{len(pol)} CAGR 최대": pol["twr_cagr"].max(),
               f"{len(pol)} MDD 최악": pol["twr_mdd"].min(), f"{len(pol)} IRR 최소": pol["irr"].min(),
               f"{len(pol)} IRR 최대": pol["irr"].max()}
        row["CAGR 표시 (>0.5%p)"] = bool(row["CAGR 차"] > CAGR_PP)
        row["MDD 표시 (>2%p)"] = bool(abs(row["MDD 차"]) > MDD_PP)
        rows.append(row)
    return pd.DataFrame(rows)


def axis_effects(R: pd.DataFrame) -> pd.DataFrame:
    """Per target and axis level: mean TWR CAGR / MDD / IRR over the policies at that level, minus the axis mean
    (main effects; the other axes are averaged over all their levels)."""
    pol = R[~R["기준선"]]
    rows = []
    for n, g in pol.groupby("대상", sort=False):
        for ax, levels in (AC.AXES_B if "overflow" in g else AC.AXES).items():
            m = g.groupby(ax)[["twr_cagr", "twr_mdd", "irr"]].mean()
            for lv in levels:
                if lv in m.index:
                    rows.append({"대상": n, "축": ax, "수준": lv, "이름": AC.LABELS[lv],
                                 "CAGR 효과": m.loc[lv, "twr_cagr"] - m["twr_cagr"].mean(),
                                 "MDD 효과": m.loc[lv, "twr_mdd"] - m["twr_mdd"].mean(),
                                 "IRR 효과": m.loc[lv, "irr"] - m["irr"].mean()})
    return pd.DataFrame(rows)


def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    dest.mkdir(parents=True, exist_ok=True)
    made = []
    V = pd.read_csv(out / "verdicts.csv")
    R = pd.read_csv(out / "policies.csv.gz", keep_default_na=False)
    E = pd.read_csv(out / "axis_effects.csv")
    x = np.arange(len(V))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.27, V["기준선 CAGR"] * 100, 0.27, label="한 계좌 · 목돈 · 비용 없음 (TWR)", color="#4C78A8")
    ax.bar(x, V["선언 CAGR (TWR)"] * 100, 0.27, label="선언 정책 세후 (TWR)", color="#F58518")
    ax.bar(x + 0.27, V["선언 IRR"] * 100, 0.27, label="선언 정책 세후 (IRR)", color="#72B7B2")
    for i, r in V.iterrows():
        if r["CAGR 표시 (>0.5%p)"] or r["MDD 표시 (>2%p)"]:
            ax.text(i, max(r["기준선 CAGR"], r["선언 CAGR (TWR)"]) * 100 + 0.3, "표시", ha="center", fontsize=8, color="#E45756")
    ax.set_xticks(x, V["대상"], rotation=15)
    ax.set_ylabel("CAGR (%)")
    ax.set_title("① 같은 비중, 운영 방식만 다르게: 기준선 vs 선언 정책(분리 계좌 · 1,000만 + 매월 100만 · 일반형 · 한도)\n"
                 "'표시' = CAGR 차 > 0.5%p 또는 MDD 차 > 2%p")
    ax.legend(fontsize=8)
    made.append(str(figs._save(fig, dest / "1_declared.png")))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for a, col, title in zip(axes, ("CAGR 효과", "IRR 효과", "MDD 효과"), ("TWR CAGR", "IRR", "TWR MDD")):
        piv = E.pivot_table(index=["축", "이름"], columns="대상", values=col, sort=False) * 100
        lim = np.nanmax(np.abs(piv.to_numpy())) or 1.0
        im = a.imshow(piv.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        a.set_xticks(range(piv.shape[1]), piv.columns, rotation=60, fontsize=7)
        a.set_title(f"{title} 효과 (%p, 축 평균 대비)")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                a.text(j, i, f"{piv.iat[i, j]:+.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=a, shrink=0.6)
    axes[0].set_yticks(range(piv.shape[0]), [f"{AXIS_KO[k[0]]} · {k[1]}" for k in piv.index], fontsize=7)
    fig.suptitle("② 정책 축별 주효과 (다른 축은 모든 수준의 평균)")
    made.append(str(figs._save(fig, dest / "2_axes.png")))
    fig, ax = plt.subplots(figsize=(11, 5))
    pol = R[~R["기준선"]]
    order = list(dict.fromkeys(pol["대상"]))
    data = [pol.loc[pol["대상"] == n, "twr_cagr"].to_numpy() * 100 for n in order]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(order) + 1), order, rotation=15)
    base = [R[(R["대상"] == n) & R["기준선"]]["twr_cagr"].iloc[0] * 100 for n in order]
    ax.scatter(range(1, len(order) + 1), base, color="#E45756", zorder=3, label="기준선 (한 계좌 · 목돈 · 비용 없음)")
    ax.set_ylabel("TWR CAGR (%)")
    ax.set_title(f"③ 정책 {pol.groupby('대상').size().iloc[0]}가지의 TWR CAGR 분포 (대상별)")
    ax.legend(fontsize=8)
    made.append(str(figs._save(fig, dest / "3_spread.png")))
    if "overflow" in R:
        made.append(overflow_figure(R, dest, plt, figs))
    return made


def overflow_figure(R: pd.DataFrame, dest: Path, plt, figs) -> str:
    """④ stage 3b: declared policy with the money over the limit as cash vs invested, and the same without a limit."""
    order = list(dict.fromkeys(R["대상"]))
    base = R[R["기준선"]].set_index("대상")["twr_cagr"]

    def pick(limit, overflow):
        m = (R["accounts"] == "split") & (R["deposits"] == "monthly") & (R["execution"] == "full") & \
            (R["cost"] == "product") & (R["tax"] == "general") & (R["limit"] == limit) & (R["overflow"] == overflow)
        return R[m].set_index("대상")["twr_cagr"]
    series = {"기준선 (한 계좌 · 목돈 · 비용 없음)": base, "선언: 한도 밖 현금 (3단계)": pick("isa", "cash"),
              "선언: 한도 밖 일반 계좌 굴림 (3b)": pick("isa", "invest"), "선언에서 한도만 없음": pick("none", "cash")}
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, (name, sr) in enumerate(series.items()):
        ax.bar(x + (k - 1.5) * 0.2, sr.reindex(order).to_numpy() * 100, 0.2, label=name,
               color=["#4C78A8", "#BBBBBB", "#F58518", "#72B7B2"][k])
    ax.set_xticks(x, order, rotation=15)
    ax.set_ylabel("TWR CAGR (%)")
    ax.set_title("④ ISA 한도 밖의 돈: 현금으로 두기 vs 일반 계좌에서 같은 ETF로 굴리기 (선언 정책 세후)")
    ax.legend(fontsize=8)
    return str(figs._save(fig, dest / "4_overflow.png"))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--overflow", action="store_true", help="stage 3b: add the 'money over the limit' axis (648)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.figures_only:
        t0 = time.time()
        summ = run(out, args.quick, args.workers, args.overflow)
        summ["seconds"] = round(time.time() - t0, 1)
        summ["quick"] = args.quick
        summ["overflow"] = args.overflow
        (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(figures(out, FIG_DIR.with_name("a42_stage3b") if args.overflow else FIG_DIR))


if __name__ == "__main__":
    main(sys.argv[1:])
