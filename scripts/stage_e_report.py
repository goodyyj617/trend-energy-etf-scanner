"""Stage E candidates (DECISIONS 83): index check and DSR of each candidate's representative cell.

    python scripts/stage_e_report.py --out DIR

Reads the Stage E output folder (20-slot equity curves, summary) and SPY / cash from the price cache.
Representative cell of a candidate = among its cells passing in both samples, the one whose smaller
excess over SPY is the largest. DSR trial counts: Stages B-E logged runs (55,000) and a lower
effective count (1,169, Stage C's effective N), with the variance of daily Sharpe ratios across
all of the sample's Stage E accounts.
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

from lab.crsp_study import SAMPLES  # noqa: E402
from lab.data import BENCHMARK, download_panel  # noqa: E402
from lab.metrics import _month_end_values, deflated_sharpe  # noqa: E402

PERIODS = [("1993-02", "1998-12"), ("1999-01", "2003-12"), ("2004-01", "2016-12"), ("2017-01", "2023-12")]
N_TRIALS = {"raw": 55_000, "effective": 1_169}


def index_stats(eq: pd.Series, spy: pd.Series) -> dict:
    eq = eq.dropna().astype(float)
    s = spy.reindex(eq.index).ffill()
    r, m = eq.pct_change(fill_method=None).iloc[1:], s.pct_change(fill_method=None).iloc[1:]
    ok = r.notna() & m.notna()
    r, m = r[ok], m[ok]
    rm = _month_end_values(eq).pct_change(fill_method=None).dropna()
    mm = _month_end_values(s).pct_change(fill_method=None).reindex(rm.index)
    out = {"beta": float(np.cov(r, m)[0, 1] / np.var(m, ddof=1)), "corr": float(r.corr(m)),
           "down_capture": float(rm[mm < 0].mean() / mm[mm < 0].mean()), "up_capture": float(rm[mm > 0].mean() / mm[mm > 0].mean())}
    lx = np.log(eq).diff() - np.log(s).diff()
    for a, b in PERIODS:
        out[f"excess_{a[:4]}_{b[:4]}"] = float(lx.loc[a:b].sum())
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    c = pd.read_csv(out / "summary.csv")
    cand = pd.read_csv(out / "candidates.csv")
    ref, _ = download_panel([BENCHMARK], "2024-01-31")
    spy = ref.close[BENCHMARK]
    sr_var = {}
    for smp in SAMPLES:
        srs = []
        for f in (out / smp).glob("*_equity.parquet"):
            e = pd.read_parquet(f)
            r = e.astype(float).pct_change(fill_method=None)
            s = (r.mean() / r.std()).replace([np.inf, -np.inf], np.nan).dropna()
            srs.extend(s.tolist())
        sr_var[smp] = float(np.var(srs))
    rows = []
    for r in cand.itertuples():
        g = c[(c["screen"] == r.screen) & (c["entry"] == r.entry) & (c["exit"] == r.exit)]
        piv = g.pivot_table(index=["xi", "yi"], columns="sample", values="goal_excess")
        ok = g.pivot_table(index=["xi", "yi"], columns="sample", values="goal_pass").fillna(0).astype(bool).all(axis=1)
        pool = piv[ok] if ok.any() else piv
        xi, yi = pool.min(axis=1).idxmax()
        for smp in SAMPLES:
            eq = pd.read_parquet(out / smp / f"{r.entry}__{r.exit}_equity.parquet")[f"{xi}|{yi}|{r.screen}"].astype(float)
            cell = g[(g["sample"] == smp) & (g["xi"] == xi) & (g["yi"] == yi)].iloc[0]
            row = {"screen": r.screen, "entry": r.entry, "exit": r.exit, "xi": int(xi), "yi": int(yi), "sample": smp,
                   "cagr": cell["cagr"], "mdd": cell["mdd"], "n_trades": int(cell["n_trades"]), "goal_pass": bool(cell["goal_pass"]),
                   **index_stats(eq, spy)}
            for k, n in N_TRIALS.items():
                row[f"dsr_{k}"] = deflated_sharpe(eq.dropna(), n, sr_var[smp])
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "candidates_checked.csv", index=False)
    print(t.round(3).to_string())
    (out / "candidates_checked.json").write_text(json.dumps(t.round(4).to_dict("records"), ensure_ascii=False, indent=1),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
