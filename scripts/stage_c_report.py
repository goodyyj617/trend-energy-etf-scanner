"""Stage C report (DECISIONS 81): the registered judging, aggregated tables and the memo figures.

    python scripts/stage_c_report.py --drive "H:\\내 드라이브\\CRSP"

Reads lab_crsp/stagec_daily3000 (summaries and the 20-slot account equity curves; never prices) plus
SPY and cash from the lab's price cache. Writes aggregated tables to docs/research/lab/2026-10-27_stage_c/
and figures to docs/research/lab/figures/stagec_daily3000/, and prints the numbers the memo cites.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import figures  # noqa: E402
from lab.crsp_study import SAMPLES  # noqa: E402
from lab.data import BENCHMARK, download_panel  # noqa: E402
from lab.grid import _neighbors_and_regions  # noqa: E402
from lab.metrics import _month_end_values, deflated_sharpe  # noqa: E402
from lab.stage_c_study import ENTRIES, ENTRY_CANON, EXIT_CANON, EXITS  # noqa: E402

OUT = ROOT / "docs" / "research" / "lab" / "2026-10-27_stage_c"
FIG = ROOT / "docs" / "research" / "lab" / "figures" / "stagec_daily3000"
NAMES = {k: SAMPLES[k][2] for k in SAMPLES}
E_LABEL = {"E0": "E0 조건 없음", "E1": "E1 돈치안 돌파", "E2": "E2 골든크로스", "E3": "E3 52주 신고가+거래량",
           "E4": "E4 볼린저 수축 돌파", "E5": "E5 추세 속 눌림목", "E6": "E6 MACD 교차", "E7": "E7 +DI 교차+ADX",
           "E8": "E8 시계열 모멘텀", "E9": "E9 갭+거래량"}
X_LABEL = {"X1": "X1 % 트레일링", "X2": "X2 샹들리에", "X3": "X3 N일 신저가", "X4": "X4 이동평균 이탈",
           "X5": "X5 모멘텀 마이너스", "X6": "X6 파라볼릭 SAR", "X7": "X7 단기 반등 청산", "X8": "X8 시장 국면 이탈",
           "X9": "X9 손절+손익분기+시간"}
PERIODS = [("1993-02", "1998-12"), ("1999-01", "2003-12"), ("2004-01", "2016-12"), ("2017-01", "2023-12")]


def grid_regions(c2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (smp, e, x), g in c2.groupby(["sample", "entry", "exit"]):
        g = g.copy()
        g["pass"] = g["goal_pass"].astype(bool)
        _neighbors_and_regions(g)
        rows.append({"sample": smp, "entry": e, "exit": x, "cells": len(g), "pass_cells": int(g["pass"].sum()),
                     "largest_region": int(g["region_size"].max()), "median_excess": g["goal_excess"].median(),
                     "best_excess": g["goal_excess"].max(), "median_cagr": g["cagr"].median(), "median_mdd": g["mdd"].median()})
    return pd.DataFrame(rows)


def index_stats(eq: pd.Series, spy: pd.Series, cash: pd.Series) -> dict:
    eq = eq.dropna().astype(float)
    s = spy.reindex(eq.index).ffill()
    r, m = eq.pct_change(fill_method=None).iloc[1:], s.pct_change(fill_method=None).iloc[1:]
    ok = r.notna() & m.notna()
    r, m = r[ok], m[ok]
    beta = float(np.cov(r, m)[0, 1] / np.var(m, ddof=1))
    rm = _month_end_values(eq).pct_change(fill_method=None).dropna()
    mm = _month_end_values(s).pct_change(fill_method=None).reindex(rm.index)
    out = {"beta": beta, "corr": float(r.corr(m)), "up_capture": float(rm[mm > 0].mean() / mm[mm > 0].mean()),
           "down_capture": float(rm[mm < 0].mean() / mm[mm < 0].mean())}
    lx = np.log(eq).diff() - np.log(s).diff()
    for a, b in PERIODS:
        out[f"excess_{a[:4]}_{b[:4]}"] = float(lx.loc[a:b].sum())
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drive", required=True)
    a = ap.parse_args(argv)
    d = Path(a.drive) / "lab_crsp" / "stagec_daily3000"
    OUT.mkdir(parents=True, exist_ok=True)
    c1 = pd.read_csv(d / "summary_c1.csv")
    c2all = pd.read_csv(d / "summary_c2.csv")
    c2 = c2all[c2all["priority"] == "dv"].copy()
    summ = json.loads((d / "stage_c_summary.json").read_text(encoding="utf-8"))
    rep: dict = {"pass_c2": summ["pass_c2"], "spearman": {k: summ[k] for k in summ if k.startswith("spearman")},
                 "below_E0": {k: summ[k] for k in summ if k.startswith("entries_below")}, "pbo": summ["pbo"]}

    # entry / exit ranks (registered): median excess over every cell of the other side
    for stage, frame in (("c1", c1), ("c2", c2)):
        for side in ("entry", "exit"):
            t = pd.read_csv(d / f"{side}_ranks_{stage}.csv")
            t.to_csv(OUT / f"{side}_ranks_{stage}.csv", index=False)
    # C1 per-signal and sleeve stats by entry and by exit
    years = {k: (pd.Timestamp(SAMPLES[k][1]) - pd.Timestamp(SAMPLES[k][0])).days / 365.25 for k in SAMPLES}
    c1 = c1.assign(signals_per_yr=c1["signals"] / c1["sample"].map(years))
    cols = ["signals_per_yr", "sig_expectancy", "sig_win_rate", "sig_payoff", "sig_avg_hold_days", "sig_big_share",
            "sleeve_cagr", "sleeve_mdd", "sleeve_dd_excess"]
    by_e = c1.groupby(["entry", "sample"])[cols].median().unstack("sample")
    by_x = c1[c1["entry"] != "E0"].groupby(["exit", "sample"])[cols].median().unstack("sample")
    by_e.to_csv(OUT / "c1_by_entry.csv")
    by_x.to_csv(OUT / "c1_by_exit.csv")
    # C2 pair grid: pass cells, connected regions, medians
    reg = grid_regions(c2)
    reg.to_csv(OUT / "c2_pairs.csv", index=False)
    both = reg.pivot_table(index=["entry", "exit"], columns="sample", values=["pass_cells", "largest_region", "median_excess"])
    both.columns = [f"{x}_{y}" for x, y in both.columns]
    both["both"] = (both["pass_cells_full"] > 0) & (both["pass_cells_recent"] > 0)
    both["region_both"] = (both["largest_region_full"] >= 2) & (both["largest_region_recent"] >= 2)
    both = both.sort_values(["both", "pass_cells_full", "pass_cells_recent"], ascending=False)
    both.to_csv(OUT / "c2_pairs_both.csv")
    rep["pairs_both_samples"] = both[both["both"]].reset_index()[["entry", "exit", "pass_cells_full", "pass_cells_recent",
                                                                  "largest_region_full", "largest_region_recent"]].to_dict("records")
    rep["pairs_region_both"] = both[both["region_both"]].reset_index()[["entry", "exit"]].to_dict("records")
    # taken vs missed (C2 dv vs per-signal table), by entry
    tm_cols = ["taken", "missed", "taken_expectancy", "missed_expectancy", "big_winners_taken_share",
               "taken_cap_rank_p25", "taken_cap_rank_p50", "taken_cap_rank_p75", "missed_cap_rank_p25", "missed_cap_rank_p50",
               "missed_cap_rank_p75"]
    tm = c2[c2["entry"] != "E0"].groupby(["entry", "sample"])[tm_cols].median().unstack("sample")
    tm.to_csv(OUT / "taken_vs_missed_by_entry.csv")
    # priority comparison on canonical cells
    canon = c2all[[(r.xi, r.yi) == (ENTRY_CANON[r.entry], EXIT_CANON[r.exit]) for r in c2all.itertuples()]]
    pr = canon.pivot_table(index=["sample", "entry", "exit"], columns="priority", values=["goal_excess", "goal_pass"])
    pr.columns = [f"{x}_{y}" for x, y in pr.columns]
    pr.to_csv(OUT / "priority_canonical.csv")
    prs = canon.groupby(["priority", "sample"]).agg(median_excess=("goal_excess", "median"), pass_share=("goal_pass", "mean"),
                                                    median_cagr=("cagr", "median"), median_mdd=("mdd", "median"))
    prs.to_csv(OUT / "priority_summary.csv")
    rep["priority"] = {f"{p}|{s}": v for (p, s), v in prs.round(4).to_dict("index").items()}
    wins = {}
    for s in SAMPLES:
        sub = pr.loc[s]
        wins[s] = {"eff_large_beats_dv": int((sub["goal_excess_eff_large"] > sub["goal_excess_dv"]).sum()),
                   "random_beats_dv": int((sub["goal_excess_random"] > sub["goal_excess_dv"]).sum()), "pairs": int(len(sub))}
    rep["priority_wins"] = wins

    # index check on the top pairs (C2 dv, best cell of each pair in the long sample) and DSR finalists
    ref, _ = download_panel([BENCHMARK], "2024-01-31")
    spy, cash = ref.close[BENCHMARK], ref.cash
    top = both.head(5).index.tolist()
    idx_rows, dsr_rows = [], []
    sr_var = {}
    for smp in SAMPLES:
        srs = []
        for f in (d / "c2" / smp).glob("*_equity.parquet"):
            e = pd.read_parquet(f)
            for c in e:
                if c.endswith("|dv"):
                    r = e[c].dropna().astype(float).pct_change(fill_method=None).dropna()
                    if r.std() > 0:
                        srs.append(r.mean() / r.std())
        sr_var[smp] = float(np.var(srs))
    for e, x in top:
        cells = c2[(c2["entry"] == e) & (c2["exit"] == x)]
        piv = cells.pivot_table(index=["xi", "yi"], columns="sample", values="goal_excess")
        best = piv.min(axis=1).idxmax()
        for smp in SAMPLES:
            eq = pd.read_parquet(d / "c2" / smp / f"{e}__{x}_equity.parquet")[f"{best[0]}|{best[1]}|dv"]
            row = cells[(cells["sample"] == smp) & (cells["xi"] == best[0]) & (cells["yi"] == best[1])].iloc[0]
            st = index_stats(eq, spy, cash)
            idx_rows.append({"entry": e, "exit": x, "xi": best[0], "yi": best[1], "sample": smp, "cagr": row["cagr"],
                             "mdd": row["mdd"], "goal_pass": bool(row["goal_pass"]), "n_trades": int(row["n_trades"]), **st})
            if both.loc[(e, x), "region_both"]:
                dsr_rows.append({"entry": e, "exit": x, "xi": best[0], "yi": best[1], "sample": smp,
                                 "dsr_n1169": deflated_sharpe(eq.dropna(), 1169, sr_var[smp]),
                                 "dsr_n17000": deflated_sharpe(eq.dropna(), 17000, sr_var[smp])})
    pd.DataFrame(idx_rows).to_csv(OUT / "index_check_top_pairs.csv", index=False)
    pd.DataFrame(dsr_rows).to_csv(OUT / "dsr_finalists.csv", index=False)
    rep["index_check"] = pd.DataFrame(idx_rows).round(3).to_dict("records")
    rep["dsr"] = pd.DataFrame(dsr_rows).round(3).to_dict("records")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    print(by_e.round(3).to_string())
    print(by_x.round(3).to_string())
    print(tm.round(3).to_string())

    if not figures.available():
        return
    # 1. entry x exit heatmap of the 20-slot account's median excess, per sample
    frames = {}
    for smp in SAMPLES:
        h = reg[reg["sample"] == smp].pivot(index="entry", columns="exit", values="median_excess")
        frames[NAMES[smp]] = h.rename(index=E_LABEL, columns=X_LABEL)
    figures.heatmap(frames, "20종목 계좌: 매수 × 매도 전략별 SPY 대비 초과 CAGR (9칸 중앙값, %p)",
                    FIG / "heatmap_c2_excess.png", cbar="%p")
    frames = {}
    for smp in SAMPLES:
        h = reg[reg["sample"] == smp].pivot(index="entry", columns="exit", values="pass_cells").astype(float)
        frames[NAMES[smp]] = h.rename(index=E_LABEL, columns=X_LABEL)
    figures.heatmap(frames, "20종목 계좌: 매수 × 매도 전략별 통과 칸 수 (9칸 중)", FIG / "heatmap_c2_pass.png",
                    fmt="{:.0f}", scale=1.0, center=0.5, cbar="통과 칸")
    # 2. rank scatter full vs recent
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, side, lab in ((axes[0], "entry", E_LABEL), (axes[1], "exit", X_LABEL)):
        t = pd.read_csv(d / f"{side}_ranks_c2.csv")
        ax.scatter(t["rank_full"], t["rank_recent"], color=figures.COLORS[0])
        for r in t.itertuples():
            ax.annotate(lab[getattr(r, side)], (r.rank_full, r.rank_recent), fontsize=7, xytext=(3, 3), textcoords="offset points")
        n = len(t)
        ax.plot([1, n], [1, n], ls="--", color="#888", lw=0.8)
        ax.set_xlabel("1993–2023 순위")
        ax.set_ylabel("2017–2023 순위")
        ax.set_title(f"{'매수' if side == 'entry' else '매도'} 전략 순위 (Spearman {summ[f'spearman_{side}_c2']:.2f})")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
    figures._save(fig, FIG / "rank_scatter.png")
    # 3. taken vs missed
    tme = tm.xs("full", axis=1, level="sample")
    figures.paired_bars([E_LABEL[k] for k in tme.index], {"20종목 계좌가 산 신호": tme["taken_expectancy"].to_numpy(),
                                                          "자리가 없어 놓친 신호": tme["missed_expectancy"].to_numpy()},
                        "산 신호와 놓친 신호의 신호당 평균 수익 (1993–2023, 9칸 중앙값)", "신호당 평균 수익 (%)",
                        FIG / "taken_vs_missed.png")
    figures.paired_bars([E_LABEL[k] for k in tme.index], {"산 신호": tme["taken_cap_rank_p50"].to_numpy(),
                                                          "놓친 신호": tme["missed_cap_rank_p50"].to_numpy()},
                        "신호일 시가총액 순위 중앙값 (1993–2023, 1 = 가장 큼)", "시총 순위", FIG / "taken_vs_missed_caprank.png",
                        pct=False)
    # 4. priority comparison (canonical cells)
    lab = {"dv": "거래대금 순", "eff_large": "추세 효율+시총 순", "random": "무작위"}
    ps = prs.reset_index()
    figures.paired_bars([lab[p] for p in ps["priority"].unique()],
                        {NAMES[s]: ps[ps["sample"] == s].set_index("priority").loc[list(ps["priority"].unique()), "median_excess"].to_numpy()
                         for s in SAMPLES},
                        "20종목 계좌의 매수 우선순위 비교 (대표 칸 90개, 초과 CAGR 중앙값)", "SPY 대비 초과 CAGR (%p)",
                        FIG / "priority.png")


if __name__ == "__main__":
    main(sys.argv[1:])
