"""Hand-picked representatives of the A31 global search, with their sensitivity (memo 2026-10-25_global_search.md §4).

    .venv/Scripts/python.exe scripts/aa_global_search_reps.py --out docs/research/allocation/2026-10-25_global_search --full-out <folder with candidates_full.csv.gz>

The picks are a post-hoc reading aid, not a selection rule: for each 1 %p MDD band the best raw sample
(A) overall, (B) without leverage and without the synthetic unhedged treasuries, (C) additionally without
나스닥100 and the S&P 500 sector products. Every row is re-simulated with monthly / annual rebalancing,
20 bp cost, the A30 timing offsets, all ±5 %p neighbours and 5-year rolling windows. Writes
representatives.csv into --out. Nothing is written to lab_results/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import aa_global_search as G  # noqa: E402
from lab import aa_search as srch  # noqa: E402
from lab.aa_validate import TIMING_OFFSETS  # noqa: E402

BANDS_B = (-14, -13, -12, -11, -10)
BANDS_C = (-14, -13, -12, -11)


def parse_weights(text: str) -> np.ndarray:
    name2code = {G.name_of(c): c for c in G.SYMBOLS}
    w = np.zeros(len(G.SYMBOLS), dtype=int)
    for part in text.split(" · "):
        name, v = part.rsplit(" ", 1)
        w[G.SYMBOLS.index(name2code[name])] = int(v)
    return w


def pick(full: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    p = full[full["mdd"] >= G.MDD_LINE].copy()
    p["band"] = np.ceil(p["mdd"] * 100).astype(int)
    syn = p["비중"].str.contains("*", regex=False)
    sec = p["비중"].str.contains("유틸리티|필수소비재|헬스케어|에너지")
    nq = p["비중"].str.contains("나스닥")
    nolev = p["레버리지"] == 0
    reps = [("A 전체 최고", p.sort_values("cagr", ascending=False).iloc[0])]
    for b in BANDS_B:
        q = p[nolev & ~syn & (p["band"] == b)].sort_values("cagr", ascending=False)
        if len(q):
            reps.append((f"B 레버·가상 없음 ({b - 1},{b}]", q.iloc[0]))
    for b in BANDS_C:
        q = p[nolev & ~syn & ~sec & ~nq & (p["band"] == b)].sort_values("cagr", ascending=False)
        if len(q):
            reps.append((f"C 나스닥·섹터도 없음 ({b - 1},{b}]", q.iloc[0]))
    return reps


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/research/allocation/2026-10-25_global_search")
    ap.add_argument("--full-out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out)
    full_out = Path(args.full_out) if args.full_out else out
    full = pd.read_csv(full_out / "candidates_full.csv.gz", usecols=["cagr", "mdd", "레버리지", "비중"])
    reps = pick(full)
    krw = G.load()
    sims: dict = {}

    def sim(reb="Q", cost=10.0, off=0):
        k = (reb, cost, off)
        if k not in sims:
            sims[k] = srch.StaticSim(krw, G.SYMBOLS, G.START, G.END, rebalance=reb, cost_bps=cost, rebalance_offset=off)
        return sims[k]

    W = np.vstack([parse_weights(r["비중"]) for _, r in reps])
    F = G.to_fraction(W)
    base = sim()
    paths, turn = base.run(F)
    s = srch.summarize(paths, turn, base.dates, base.cash, windows=G.WINDOWS, halves=(G.HALF,), extra_starts=G.EXTRA_STARTS,
                       holdout_share=G.HOLDOUT_SHARE)
    rows = []
    for i, (label, r) in enumerate(reps):
        row = {"후보": label, "CAGR": s.cagr[i], "MDD": s.mdd[i], "초과샤프": s.sharpe_excess[i], "회전": s.turnover[i],
               "CAGR 05-14": s.cagr_h1[i], "CAGR 15-26": s.cagr_h2[i], "MDD 05-14": s.mdd_h1[i], "MDD 15-26": s.mdd_h2[i],
               "dd08": s.dd_2008[i], "dd20": s.dd_2020[i], "dd22": s.dd_2022[i], "CAGR 08-": s.cagr_2008[i], "MDD 08-": s.mdd_2008[i],
               "CAGR 10-": s.cagr_2010[i], "보류 CAGR": s.cagr_holdout[i], "보류 MDD": s.mdd_holdout[i]}
        for reb in ("M", "A"):
            c, m = srch.cagr_mdd(sim(reb).run(F[i:i + 1])[0])
            row[f"{reb} CAGR"], row[f"{reb} MDD"] = c[0], m[0]
        row["20bp CAGR"] = srch.cagr_mdd(sim("Q", 20.0).run(F[i:i + 1])[0])[0][0]
        row["날짜 최악 MDD"] = min(srch.cagr_mdd(sim("Q", 10.0, off).run(F[i:i + 1])[0])[1][0] for off in TIMING_OFFSETS)
        nb = np.vstack(srch.neighbours(W[i], 5, G.MIN_PCT))
        c, m = srch.cagr_mdd(base.run(G.to_fraction(nb))[0])
        row["이웃 최악 MDD"], row["이웃 −15% 안 비율"], row["이웃 CAGR 중앙값"] = m.min(), float((m >= G.MDD_LINE).mean()), float(np.median(c))
        share, med = srch.rolling_window_share(paths[i:i + 1], base.dates, G.ROLL_YEARS, G.MDD_LINE)
        row["5년창 비율"], row["5년창 CAGR 중앙값"] = share[0], med[0]
        row["상품 수"] = int((W[i] > 0).sum())
        row["비중"] = r["비중"]
        rows.append(row)
    table = pd.DataFrame(rows)
    num = [c for c in table.columns if c not in ("후보", "비중", "상품 수")]
    table[num] = (table[num] * 100).round(2)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "representatives.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    print(table.drop(columns=["비중"]).set_index("후보").T.to_string())
    for _, r in table.iterrows():
        print(r["후보"], "→", r["비중"])


if __name__ == "__main__":
    main(sys.argv[1:])
