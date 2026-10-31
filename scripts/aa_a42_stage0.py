"""A42 stage 0 check run (allocation DECISIONS A42 §0; memo 2026-10-29_a42_stage0.md).

    .venv/Scripts/python.exe scripts/aa_a42_stage0.py --out docs/research/allocation/2026-10-29_a42_stage0

Writes into --out (small tables only):
  hedge.csv        0-2  hedge-cost estimate from 267440 / 304660 against the h = 0 hedge model, block bootstrap 95 %
  calibration.csv  0-3·0-7  per product: overlap of real prices with the model, annual gap, tracking error, used or not
  quirks.csv       0-6  per affected product: CAGR of the real series before / after aa_real.QUIRKS
  fx_timing.csv    0-8  CAGR / MDD of three allocations with DEXKOUS (New York noon) vs ECOS 매매기준율 (Seoul)
  sets.csv         B2, B4, 안정형 40, (가), current holdings, S&P500 × 24 price sets (2000-09 – 2024-12, quarterly, 10 bp)
  summary.json     the numbers above plus h*, the representative set and coverage notes
Nothing is chosen from these: they record how sensitive the existing results are to the data choices.
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

from lab import aa_pricesets as PS  # noqa: E402
from lab import aa_real  # noqa: E402
from lab.aa_assets import PRESETS  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation  # noqa: E402
from lab.aa_korea import KrwModel, cd91_schedule, krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.aa_assets import PRODUCTS  # noqa: E402

START, END = PS.START, "2024-12-31"
ALLOCS = {"B2": "gs_b2", "B4": "gs_b4", "안정형 40": "stable40", "(가)": "a41_ga", "현재 보유": "aaa_2026", "S&P500": "sp500"}
FX_ECOS = ROOT / "lab" / "aa_usdkrw_ecos.csv.gz"


def metrics(eq: pd.Series) -> dict:
    v = eq.to_numpy()
    return {"cagr": float((v[-1] / v[0]) ** (252 / max(len(v) - 1, 1)) - 1), "mdd": float((v / np.maximum.accumulate(v) - 1).min())}


def run(panel, w: dict, start=START, end=END) -> dict:
    r = run_allocation(panel, AllocConfig("x", dict(w), start, end, rebalance="Q", cost_bps=10.0))
    return {**metrics(r.equity), "시작": str(r.equity.index[0].date())}


def quirks_table() -> pd.DataFrame:
    files = aa_real.load_files()
    plain = aa_real.real_series(*files)
    fixed = aa_real.real_series(*files, quirks=aa_real.QUIRKS)
    yah = files[1]
    paid = set(yah.loc[(pd.to_datetime(yah["date"]) == pd.Timestamp("2025-12-29")) & (yah["dividends"] > 0), "code"].astype(str))
    rows = []
    for q in aa_real.QUIRKS:
        codes = [q["code"]] if q["code"] != "*" else sorted(plain)
        for c in codes:
            if c not in plain:
                continue
            a, b = plain[c], fixed[c]
            if q["code"] == "*" and c not in paid:
                continue
            yrs = (a.index[-1] - a.index[0]).days / 365.25
            ca = (a.iloc[-1] / a.iloc[0]) ** (1 / yrs) - 1
            cb = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
            rows.append({"상품": c, "날짜": ",".join(q["dates"]), "처리": q["kind"], "CAGR 전": ca, "CAGR 후": cb,
                         "차이 (%p)": (cb - ca) * 100, "근거": q["reason"]})
    return pd.DataFrame(rows)


def fx_timing() -> pd.DataFrame:
    ext, _, _ = PS.usd_extended(START)
    ecos = pd.read_csv(FX_ECOS, parse_dates=["date"]).set_index("date")["krw_per_usd"]
    base = load_usdkrw()
    # Seoul fixing of day t is set before New York trades t: the US close of t meets the next Seoul fixing
    seoul = ecos.copy()
    seoul.index = seoul.index - pd.Timedelta(days=1)
    rows = []
    for tag, fx in (("DEXKOUS (뉴욕 정오)", base), ("매매기준율 (서울, 하루 당김)", seoul)):
        krw, _ = krw_panel(ext, fx, load_bok_rate(), KrwModel(), products=PRODUCTS)
        hyb, _ = aa_real.hybrid_panel(krw, aa_real.real_series(*aa_real.load_files()), chains=aa_real.A41_CHAINS)
        for name in ("B2", "(가)", "안정형 40"):
            m = run(hyb, PRESETS[ALLOCS[name]].weights)
            rows.append({"환율": tag, "배분": name, **m})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="배분", columns="환율", values="cagr")
    df.attrs["max_cagr_gap_pp"] = float((piv.iloc[:, 0] - piv.iloc[:, 1]).abs().max() * 100)
    return df


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summ: dict = {}

    print("기본값 대조: 새 경로(real·h20·base, QUIRKS 끔)와 예전 경로", flush=True)
    ext, _, _ = PS.usd_extended(START)
    old, _ = krw_panel(ext, load_usdkrw(), load_bok_rate(), KrwModel(), products=PRODUCTS)
    old, _ = aa_real.hybrid_panel(old, aa_real.real_series(*aa_real.load_files()), chains=aa_real.A41_CHAINS)
    new, _ = PS.build("real", 20.0, "base", quirks=False)
    same = all(old.close[c].equals(new.close[c]) for c in old.close.columns)
    ga = run(new, PRESETS["a41_ga"].weights)
    summ["default_bit_identical"] = bool(same)
    summ["a41_ga_default"] = ga  # A41 recorded 8.843 % / −13.706 % for this allocation
    if not same:
        raise SystemExit("기본값 경로가 예전과 다릅니다. 멈춥니다.")

    print("0-2 환헤지 비용 추정", flush=True)
    h = PS.hedge_estimate()
    h.to_csv(out / "hedge.csv", index=False)
    pick = h[(h["상품"] == "267440") & (h["원화 현금"] == "cd91")].iloc[0]
    h_star = PS.nearest_grid(float(pick["h 추정 (bp)"]))
    summ["hedge"] = {"rule": "267440(장기국채선물, TLT와 가장 가까움) · 대표 원화 현금 CD91의 추정에 가장 가까운 격자값",
                     "h_est_bp": float(pick["h 추정 (bp)"]), "ci95_bp": [float(pick["95% 하한 (bp)"]), float(pick["95% 상한 (bp)"])],
                     "h_star_bp": h_star}
    rep = PS.PriceSet("real", h_star, "cd91")
    summ["representative_set"] = rep.label

    print("0-3·0-7 보정 표", flush=True)
    model, _, _ = PS.model_panel(PS.PriceSet("model", 20.0, "base"))
    real = aa_real.real_series(*aa_real.load_files(), quirks=aa_real.QUIRKS)
    cal = PS.calibration(model, real)
    cal.to_csv(out / "calibration.csv", index=False)

    print("0-6 자료 흠", flush=True)
    q = quirks_table()
    q.to_csv(out / "quirks.csv", index=False)
    summ["quirks_max_abs_cagr_pp"] = float(q["차이 (%p)"].abs().max()) if len(q) else 0.0

    print("0-8 환율 시점", flush=True)
    fx = fx_timing()
    fx.to_csv(out / "fx_timing.csv", index=False)
    summ["fx_max_cagr_gap_pp"] = fx.attrs["max_cagr_gap_pp"]

    print("확인 실행: 6개 배분 × 24세트", flush=True)
    rows = []
    for ps in PS.all_sets():
        panel, _ = PS.build(ps.prices, ps.hedge_bp, ps.cash, quirks=True)
        for name, key in ALLOCS.items():
            m = run(panel, PRESETS[key].weights)
            rows.append({"배분": name, "가격": ps.prices, "헤지 bp": ps.hedge_bp, "원화 현금": ps.cash, "세트": ps.label,
                         "대표": ps == rep, **m})
        print(f"  {ps.label}", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "sets.csv", index=False)
    rng = S.groupby("배분", sort=False).agg(CAGR_최저=("cagr", "min"), CAGR_최고=("cagr", "max"), MDD_최악=("mdd", "min"),
                                           MDD_최선=("mdd", "max"), 안_세트수=("mdd", lambda s: int((s >= -0.15).sum())))
    repr_rows = S[S["대표"]].set_index("배분")[["cagr", "mdd"]]
    summ["sets"] = {k: {**{c: float(v) for c, v in rng.loc[k].items()}, "rep_cagr": float(repr_rows.loc[k, "cagr"]),
                        "rep_mdd": float(repr_rows.loc[k, "mdd"])} for k in rng.index}
    summ["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1:])
