"""Stage 2 of the A31 global search: shake the user's chosen candidates (memo 2026-10-25_stage2.md).

    .venv/Scripts/python.exe scripts/aa_stage2.py --out docs/research/allocation/2026-10-25_stage2

Stage 1 looked at 250k allocations once each, with one setting (quarterly, 10 bp, 2005 start, fast static
simulator). Stage 2 takes a handful and changes one thing at a time:

  engine      run_allocation (the real engine) with band 0 / 3 / 5 %p, against the static number
  rebalance   monthly / quarterly / annual; 20 bp cost; rebalance date +1/+5/+10/+15 days; hedge cost 40 bp
  start       2005 / 2008 / 2010 / 2015 start, same end
  neighbours  every ±2 %p and ±5 %p move of weight between two products: share within −15%, worst MDD, median CAGR
  synthetic   the unhedged US-treasury placeholders (*) replaced by the hedged product (H) or by 달러선물

Nothing is a cut: the memo reads the table. Nothing is written to lab_results/.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import aa_global_search as G  # noqa: E402
from lab import aa_search as srch  # noqa: E402
from lab import aa_search_results as sr  # noqa: E402
from lab.aa_assets import PRESETS, PROXY_SYMBOLS  # noqa: E402
from lab.aa_engine import run_allocation  # noqa: E402
from lab.aa_korea import KrwModel, krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.aa_validate import TIMING_OFFSETS  # noqa: E402
from lab.data import download_panel  # noqa: E402

STARTS = ("2005-01-03", "2008-01-02", "2010-01-04", "2015-01-02")
BANDS = (0.0, 3.0, 5.0)
SYN_TO_HEDGED = {"SYN_IEF_U": "0091C0", "SYN_TLT_U": "484790"}  # 미국10년국채(H), 미국30년국채(H)
USD_FUT = "261240"  # 미국달러선물

# the user's stage-2 list (2026-10-25 23:35 KST): the refined best, the raw best and the five B representatives
CANDIDATES = {
    "R 다듬음 최고 (−15,−14]": "미국나스닥100 5 · 미국S&P500헬스케어 22 · 미국30년국채액티브(H) 3 · 미국10년 환노출* 13 · ACE KRX금현물 14 · CD금리액티브(합성) 18 · 미국나스닥100레버리지(합성 H) 15 · ACE 골드선물 레버리지(합성 H) 10",
}
REP_LABELS = ("A 전체 최고", "B 레버·가상 없음 (-15,-14]", "B 레버·가상 없음 (-14,-13]", "B 레버·가상 없음 (-13,-12]",
              "B 레버·가상 없음 (-12,-11]", "B 레버·가상 없음 (-11,-10]")
COMPARATORS = ("aaa_2026", "stable40")


def load_panels():
    usd, failed = download_panel(sorted(set(PROXY_SYMBOLS) | {"IEF", "TLT"}), end=G.DATA_END)
    if failed:
        print("받지 못한 종목:", failed)
    fx, bok = load_usdkrw(), load_bok_rate()
    p20, _ = krw_panel(usd, fx, bok, KrwModel(hedge_extra_bps=20.0), products=sr.ALL_PRODUCTS)
    p40, _ = krw_panel(usd, fx, bok, KrwModel(hedge_extra_bps=40.0), products=sr.ALL_PRODUCTS)
    return p20, p40


def equity_cagr_mdd(eq: pd.Series) -> tuple[float, float]:
    c, m = srch.cagr_mdd(eq.to_numpy(dtype=float)[None, :])
    return float(c[0]), float(m[0])


def substitute(w: dict[str, int], target: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for code, v in w.items():
        key = SYN_TO_HEDGED.get(code, code) if target == "hedged" else (USD_FUT if code in SYN_TO_HEDGED else code)
        out[key] = out.get(key, 0) + v
    return out


def candidates(out_dir: Path) -> list[tuple[str, dict[str, int]]]:
    reps = pd.read_csv(out_dir.parent / "2026-10-25_global_search" / "representatives.csv")
    by_label = dict(zip(reps["후보"], reps["비중"]))
    items = [(k, sr.parse_names(v)) for k, v in CANDIDATES.items()]
    items += [(lab, sr.parse_names(by_label[lab])) for lab in REP_LABELS]
    items += [(f"비교: {PRESETS[k].name}", {c: int(round(v)) for c, v in PRESETS[k].weights.items()}) for k in COMPARATORS]
    return items


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/research/allocation/2026-10-25_stage2")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = candidates(out)
    p20, p40 = load_panels()
    sims: dict = {}

    def sim(panel=p20, start=sr.START, reb="Q", cost=10.0, off=0):
        k = (id(panel), start, reb, cost, off)
        if k not in sims:
            sims[k] = srch.StaticSim(panel, sr.SYMBOLS, start, sr.END, rebalance=reb, cost_bps=cost, rebalance_offset=off)
        return sims[k]

    def static(w: dict, **kw) -> tuple[float, float]:
        s = sim(**kw)
        c, m = srch.cagr_mdd(s.run(G.to_fraction(sr.to_vector(w)[None, :]))[0])
        return float(c[0]), float(m[0])

    base = sim()
    rows, nb_rows = [], []
    for label, w in items:
        vec = sr.to_vector(w)
        paths, turn = base.run(G.to_fraction(vec[None, :]))
        s = srch.summarize(paths, turn, base.dates, base.cash, windows=G.WINDOWS, halves=(G.HALF,), holdout_share=G.HOLDOUT_SHARE)
        row = {"후보": label, "CAGR": s.cagr[0], "MDD": s.mdd[0], "초과샤프": s.sharpe_excess[0],
               "CAGR 05-14": s.cagr_h1[0], "CAGR 15-26": s.cagr_h2[0], "dd08": s.dd_2008[0], "dd20": s.dd_2020[0], "dd22": s.dd_2022[0],
               "보류 CAGR": s.cagr_holdout[0], "보류 MDD": s.mdd_holdout[0]}
        # engine with bands
        for b in BANDS:
            cfg = dataclasses.replace(sr.candidate_config(label, w), band=b)
            c, m = equity_cagr_mdd(run_allocation(p20, cfg).equity)
            row[f"엔진 밴드{int(b)} CAGR"], row[f"엔진 밴드{int(b)} MDD"] = c, m
        # rebalance / cost / timing / hedge
        for reb in ("M", "A"):
            row[f"{reb} CAGR"], row[f"{reb} MDD"] = static(w, reb=reb)
        row["20bp CAGR"], row["20bp MDD"] = static(w, cost=20.0)
        row["날짜 최악 MDD"] = min(static(w, off=o)[1] for o in TIMING_OFFSETS)
        row["헤지40 CAGR"], row["헤지40 MDD"] = static(w, panel=p40)
        # starts
        for st in STARTS[1:]:
            row[f"{st[:4]}~ CAGR"], row[f"{st[:4]}~ MDD"] = static(w, start=st)
        # neighbours
        for step in (2, 5):
            nb = np.vstack(srch.neighbours(vec, step, 1 if step == 2 else G.MIN_PCT))
            c, m = srch.cagr_mdd(base.run(G.to_fraction(nb))[0])
            row[f"이웃±{step} 수"] = len(nb)
            row[f"이웃±{step} −15% 안"] = float((m >= sr.MDD_LINE).mean())
            row[f"이웃±{step} 최악 MDD"] = float(m.min())
            row[f"이웃±{step} CAGR 중앙값"] = float(np.median(c))
            row[f"이웃±{step} CAGR 최저"] = float(c.min())
            nb_rows += [{"후보": label, "step": step, "cagr": ci, "mdd": mi, "w": sr.code_text(v)} for ci, mi, v in zip(c, m, nb)]
        # synthetic substitution
        has_syn = any(k in SYN_TO_HEDGED for k in w)
        row["가상 상품"] = "있음" if has_syn else "없음"
        if has_syn:
            row["가상→(H) CAGR"], row["가상→(H) MDD"] = static(substitute(w, "hedged"))
            row["가상→달러선물 CAGR"], row["가상→달러선물 MDD"] = static(substitute(w, "usd"))
        row["상품 수"] = int((vec > 0).sum())
        row["레버리지 %"] = int(sum(v for k, v in w.items() if k in sr.CLASSES["레버리지"]))
        row["비중"] = sr.describe(vec)
        rows.append(row)
        print(label, "done", flush=True)
    table = pd.DataFrame(rows)
    keep = {"후보", "비중", "상품 수", "레버리지 %", "초과샤프", "가상 상품", "이웃±2 수", "이웃±5 수"}
    pct = [c for c in table.columns if c not in keep]
    table[pct] = (table[pct] * 100).round(2)
    table["초과샤프"] = table["초과샤프"].round(2)
    table.to_csv(out / "stage2.csv", index=False)
    pd.DataFrame(nb_rows).to_csv(out / "neighbours.csv.gz", index=False)
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 80)
    print(table.drop(columns=["비중"]).set_index("후보").T.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
