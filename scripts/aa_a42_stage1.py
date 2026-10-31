"""A42 stage 1: factor exposures and the eight fixed scenarios (allocation DECISIONS A42 §1; memo 2026-10-31_a42_stage1.md).

    .venv/Scripts/python.exe scripts/aa_a42_stage1.py --out docs/research/allocation/2026-10-31_a42_stage1

Representative price set (real · h60 · CD91, stage 0). Writes betas.csv (every product), exposures.csv and
scenarios.csv (the six stage-0 allocations), crisis_check.csv (2008 / 2020 / 2022: linear estimate from the realized
factor moves vs the allocation's actual return over the window), summary.json, and PNG figures into
docs/research/allocation/figures/a42_stage1/. Nothing is chosen: the scenario verdict is recorded per allocation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import aa_exposure as E  # noqa: E402
from lab import aa_pricesets as PS  # noqa: E402
from lab.aa_assets import PRESETS  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation  # noqa: E402
from lab.aa_korea import load_kr_rates, load_usdkrw  # noqa: E402
from lab.aa_score import CRISIS_WINDOWS  # noqa: E402

REP = PS.PriceSet("real", 60.0, "cd91")
ALLOCS = {"(가)": "a41_ga", "안정형 40": "stable40", "B2": "gs_b2", "B4": "gs_b4", "현재 보유": "aaa_2026", "S&P500": "sp500"}
CRISES = {"2008": CRISIS_WINDOWS["금융위기 2007–09"], "2020": CRISIS_WINDOWS["코로나 2020"], "2022": CRISIS_WINDOWS["금리 급등 2022"]}
FIG_DIR = ROOT / "docs" / "research" / "allocation" / "figures" / "a42_stage1"
TNX = ROOT / "lab_data" / "yf" / "_TNX.csv.gz"


def window_return(eq: pd.Series, window: tuple[str, str]) -> float:
    a, b = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return float(eq.asof(b) / eq.asof(a) - 1)


def log_estimate(expo: pd.Series, moves: pd.Series) -> float:
    """Information only: the same exposures applied to log moves of the % factors (bp factors as they are), then back
    to a simple return. Large opposite moves (2008: S&P500 −55 %, won −40 %) compound rather than add."""
    z = sum(expo[f] * (np.log1p(moves[f]) if f in E.PCT_FACTORS else moves[f]) for f in E.FACTORS)
    return float(np.expm1(z))


def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    dest.mkdir(parents=True, exist_ok=True)
    ex = pd.read_csv(out / "exposures.csv", index_col=0)
    sc = pd.read_csv(out / "scenarios.csv", index_col=0)
    cc = pd.read_csv(out / "crisis_check.csv")
    made = []
    # ① exposures: four panels in the factor's own unit
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    panels = (("EQ", "주식 베타 (EQ)", 1.0), ("GROWTH", "성장주 베타 (GROWTH)", 1.0), ("UST", "미국 10년 금리 1bp당 (%)", 100.0),
              ("FX", "원화 약세 1%당 (%) (FX)", 1.0))
    names = list(ex.index)[::-1]
    for ax, (f, title, scale) in zip(axes, panels):
        v = ex.loc[names, f] * scale
        ax.barh(range(len(names)), v, color=["#2b6cb0" if x >= 0 else "#9b2c2c" for x in v])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    fig.suptitle("A42 1단계 ① 배분 노출 (대표 세트, 2005–2024 월간 회귀). 금 베타와 한국 금리 베타는 표(exposures.csv)")
    made.append(figs._save(fig, dest / "1_exposures.png").name)
    # ② scenario losses
    fig, ax = plt.subplots(figsize=(12, 5))
    cols = [c for c in sc.columns if c.startswith("S")]
    x = np.arange(len(cols))
    width = 0.8 / len(sc.index)
    for k, name in enumerate(sc.index):
        ax.bar(x + k * width - 0.4 + width / 2, sc.loc[name, cols] * 100, width, label=name)
    ax.axhline(E.LOSS_LINE * 100, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{E.SCENARIOS[c][0]}" for c in cols], fontsize=8)
    ax.set_ylabel("선형 손실 (%)")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("A42 1단계 ② 시나리오 8개의 선형 손실 (점선 −25%: 이보다 나쁜 것이 2개 이하면 통과)")
    made.append(figs._save(fig, dest / "2_scenarios.png").name)
    # ③ generator check
    fig, ax = plt.subplots(figsize=(7, 6))
    for crisis, marker in zip(CRISES, "os^"):
        p = cc[cc["위기"] == crisis]
        ax.scatter(p["실제"] * 100, p["선형 추정"] * 100, marker=marker, s=50, label=crisis)
        for _, r in p.iterrows():
            ax.annotate(r["배분"], (r["실제"] * 100, r["선형 추정"] * 100), fontsize=7, xytext=(3, 2), textcoords="offset points")
    lim = [min(cc["실제"].min(), cc["선형 추정"].min()) * 100 - 3, max(cc["실제"].max(), cc["선형 추정"].max()) * 100 + 3]
    ax.plot(lim, lim, color="gray", lw=0.8)
    ax.set_xlabel("실제 위기 구간 수익 (%)")
    ax.set_ylabel("노출 × 실제 요인 움직임 (%)")
    ax.legend(fontsize=8)
    ax.set_title("A42 1단계 ③ 생성기 점검: 선형 추정 vs 실제 (판정 아님)")
    made.append(figs._save(fig, dest / "3_crisis_check.png").name)
    return made


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        print(figures(out))
        return
    t0 = time.time()
    panel, _ = PS.build(REP.prices, REP.hedge_bp, REP.cash, quirks=True)
    ext, _, _ = PS.usd_extended(PS.START)
    factors = E.factor_table(ext.close)
    prod_m = E.monthly_returns(panel.close)
    beta = E.betas(prod_m, factors)
    beta.to_csv(out / "betas.csv", float_format="%.6f")
    tnx = pd.read_csv(TNX, index_col=0, parse_dates=True)["close"]
    ktb = load_kr_rates()["ktb10"]
    daily = {"SPY": ext.close["SPY"], "QQQ": ext.close["QQQ"], "GLD": ext.close["GLD"], "UST": tnx, "KTB": ktb, "FX": load_usdkrw()}
    exps, losses, checks = {}, {}, []
    for name, key in ALLOCS.items():
        w = PRESETS[key].weights
        e = E.exposure(w, beta)
        exps[name] = e
        losses[name] = E.scenario_losses(e)
        eq = run_allocation(panel, AllocConfig(name, dict(w), PS.START, "2024-12-31", rebalance="Q", cost_bps=10.0)).equity
        for crisis, win in CRISES.items():
            mv = E.realized_moves(daily, win)
            checks.append({"배분": name, "위기": crisis, "구간": f"{win[0]} – {win[1]}", "실제": window_return(eq, win),
                           "선형 추정": float((e * mv).sum()), "로그 추정 (참고)": log_estimate(e, mv),
                           **{f"움직임 {k}": v for k, v in mv.items()}})
    EX = pd.DataFrame(exps).T
    EX.to_csv(out / "exposures.csv", float_format="%.6f")
    SC = pd.DataFrame(losses).T
    SC["−25%보다 나쁜 수"] = (SC[list(E.SCENARIOS)] < E.LOSS_LINE).sum(axis=1)
    SC["판정"] = SC["−25%보다 나쁜 수"] <= E.MAX_BAD
    SC.to_csv(out / "scenarios.csv", float_format="%.6f")
    CC = pd.DataFrame(checks)
    CC.to_csv(out / "crisis_check.csv", index=False, float_format="%.6f")
    cushion = {n: {"S8 손실": float(SC.loc[n, "S8"]), "실제 2008": float(CC[(CC["배분"] == n) & (CC["위기"] == "2008")]["실제"].iloc[0]),
                   "2008 선형 추정": float(CC[(CC["배분"] == n) & (CC["위기"] == "2008")]["선형 추정"].iloc[0])} for n in ALLOCS}
    summ = {"price_set": REP.label, "window": list(E.WINDOW), "betas_products": int(beta[list(E.FACTORS)].notna().all(axis=1).sum()),
            "betas_r2_median": float(beta["R2"].median()), "verdict": {n: bool(SC.loc[n, "판정"]) for n in ALLOCS},
            "fx_cushion": cushion, "figures": figures(out), "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
