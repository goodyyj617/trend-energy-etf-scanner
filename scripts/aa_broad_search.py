"""Broad search for a better low-drawdown allocation (DECISIONS A22; memo 2026-10-24_broad_search.md).

    .venv/Scripts/python.exe scripts/aa_broad_search.py                 # main study, data to 2026-10-08
    .venv/Scripts/python.exe scripts/aa_broad_search.py --snapshot      # 2016+ sector/factor check only
    .venv/Scripts/python.exe scripts/aa_broad_search.py --out results.csv

Families F1–F5 and the lenient rule ("drop only clearly dominated candidates") were registered in
A22 before the first run. Prints tables; with --out also writes every candidate's numbers to a CSV.
Nothing is written to lab_results/ and nothing is added to the lab's run log.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.aa_assets import CASH, PRESETS, PRODUCTS, PROXY_SYMBOLS, Product  # noqa: E402
from lab.aa_engine import AllocConfig, Leverage, MixConfig, Overlay, run_allocation, run_mix  # noqa: E402
from lab.aa_korea import krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.data import CACHE_DIR, download_panel, load_snapshot  # noqa: E402

START, HALF_START, STUDY_END = "2005-01-03", "2015-01-02", "2026-10-08"
WINDOWS = {"금융위기": ("2007-10-09", "2009-03-09"), "코로나": ("2020-02-19", "2020-03-23"), "2022": ("2022-01-03", "2022-10-12")}
AAA, S45, S35 = PRESETS["aaa_2026"].weights, PRESETS["stable45"].weights, PRESETS["stable35"].weights
CD, TEN, GOLD, SP, LEV = "459580", "0091C0", "411060", "379800", "225040"
SWAPS = [("468370", TEN), ("468630", TEN), ("463640", SP), ("218420", SP), ("245340", SP), ("182480", GOLD)]


# ----------------------------------------------------------------------------- helpers

def shift(base: dict, src: str, dst: str, amt: float) -> dict:
    w = dict(base)
    assert w.get(src, 0) >= amt, (src, amt)
    w[src] = w[src] - amt
    w[dst] = w.get(dst, 0) + amt
    return {k: v for k, v in w.items() if v > 0}


def mdd(e: pd.Series) -> float:
    return float((e / e.cummax() - 1).min())


def cagr(e: pd.Series) -> float:
    return float((e.iloc[-1] / e.iloc[0]) ** (252 / (len(e) - 1)) - 1)


def crisis_mdd(e: pd.Series) -> dict[str, float]:
    """Worst drawdown inside each S&P500 crash window, measured from the level at the window start."""
    out = {}
    for name, (a, b) in WINDOWS.items():
        w = e.loc[a:b]
        out[name] = mdd(w) if len(w) > 5 else np.nan
    return out


def static(name: str, w: dict, start=START, end=STUDY_END, **kw) -> AllocConfig:
    return AllocConfig(name, w, start, end, rebalance=kw.pop("rebalance", "Q"), cost_bps=10.0, **kw)


# ----------------------------------------------------------------------------- candidates

def candidates(end: str) -> list[tuple[str, str, object]]:
    """(family, name, AllocConfig | MixConfig) in registration order (A22)."""
    out: list[tuple[str, str, object]] = []
    refs = [("내 AAA", AAA), ("안정형 45", S45), ("안정형 35", S35), ("영구", PRESETS["permanent"].weights),
            ("60/40", PRESETS["sixty_forty"].weights), ("S&P500", PRESETS["sp500"].weights),
            ("골든 버터플라이", PRESETS["golden_butterfly"].weights), ("AAA 금 20", PRESETS["aaa_gold20"].weights)]
    for n, w in refs:
        out.append(("F0 참고", n, static(n, w, end=end)))
    for bname, base in (("S45", S45), ("S35", S35)):
        for p, src in SWAPS:
            for amt in (5, 10):
                n = f"{bname} {PRODUCTS[src].name.replace('KODEX ', '')} {amt} → {PRODUCTS[p].name.replace('KODEX ', '')}"
                out.append(("F1 상품 교체", n, static(n, shift(base, src, p, amt), end=end)))
        sp = base[SP]
        for p in ("463640", "245340", "218420"):
            for frac, lab in ((1 / 3, "⅓"), (1 / 2, "½")):
                amt = int(round(sp * frac))
                n = f"{bname} S&P500의 {lab} → {PRODUCTS[p].name.replace('KODEX ', '')}"
                out.append(("F2 섹터 분산", n, static(n, shift(base, SP, p, amt), end=end)))
    for bname, base in (("AAA", AAA), ("S45", S45), ("S35", S35)):
        for m in (6, 10, 12):
            for safe in (CD, TEN, GOLD):
                n = f"{bname} 추세 {m}개월 → {PRODUCTS[safe].name.replace('KODEX ', '')}"
                out.append(("F3 overlay", n, static(n, base, end=end, overlay=Overlay("sma", float(m), safe))))
        for m in (6, 12):
            for safe in (CD, GOLD):
                n = f"{bname} 절대모멘텀 {m}개월 → {PRODUCTS[safe].name.replace('KODEX ', '')}"
                out.append(("F3 overlay", n, static(n, base, end=end, overlay=Overlay("absmom", float(m), safe))))
        for d in (63, 126):
            n = f"{bname} 리스크 패리티 {d}일"
            out.append(("F3 overlay", n, static(n, base, end=end, overlay=Overlay("riskparity", float(d)))))
        for v in (6, 8, 10):
            for cap in (1.0, 1.5):
                n = f"{bname} 변동성 목표 {v}%" + ("" if cap == 1.0 else f" (상한 {cap:g}배, 가상)")
                out.append(("F3 overlay", n, static(n, base, end=end, overlay=Overlay("voltarget", float(v), CASH), leverage=Leverage(cap))))
    sats = {
        "듀얼모멘텀 주식": static("듀얼모멘텀 주식", {SP: 34, "251350": 33, "291890": 33}, end=end, overlay=Overlay("dualmom", 12.0, CD)),
        "S&P500 추세10": static("S&P500 추세10", {SP: 100}, end=end, overlay=Overlay("sma", 10.0, CD)),
        "영구": static("영구", PRESETS["permanent"].weights, end=end),
        "S&P500 2배": static("S&P500 2배", {LEV: 100}, end=end),
        "금현물": static("금현물", {GOLD: 100}, end=end),
    }
    for bname, base in (("S45", S45), ("S35", S35)):
        core = static(f"안정형 {bname[1:]}", base, end=end)
        for sname, sat in sats.items():
            for cw in (90, 80, 70, 60, 50):
                n = f"{bname} {cw} + {sname} {100 - cw}"
                out.append(("F4 믹스", n, MixConfig(n, ((core, cw), (sat, 100 - cw)))))
        for a, b, wa, wb in (("듀얼모멘텀 주식", "S&P500 2배", 20, 20), ("S&P500 추세10", "금현물", 20, 20), ("듀얼모멘텀 주식", "S&P500 2배", 15, 15)):
            n = f"{bname} {100 - wa - wb} + {a} {wa} + {b} {wb}"
            out.append(("F4 믹스", n, MixConfig(n, ((core, 100 - wa - wb), (sats[a], wa), (sats[b], wb)))))
    for bname, base in (("S45", S45), ("S35", S35)):
        sp = base[SP]
        half = sp // 2
        for dst in (GOLD, CD):
            w = shift(base, SP, LEV, half)  # half of S&P500 into the 2x product ...
            w = shift(w, SP, dst, sp - half)  # ... the rest of the S&P500 weight into gold or cash
            n = f"{bname} S&P500 절반을 2배 상품으로, 나머지 → {PRODUCTS[dst].name.replace('KODEX ', '')}"
            out.append(("F5 레버리지 적층", n, static(n, w, end=end)))
        w = shift(shift(base, SP, LEV, half), SP, GOLD, (sp - half) // 2)
        w = shift(w, SP, CD, w.get(SP, 0))
        n = f"{bname} S&P500 절반을 2배 상품으로, 나머지 → 금·CD 반반"
        out.append(("F5 레버리지 적층", n, static(n, w, end=end)))
    return out


# ----------------------------------------------------------------------------- running

def run_one(panel, obj, start: str, end: str):
    if isinstance(obj, MixConfig):
        from dataclasses import replace
        mix = MixConfig(obj.name, tuple((replace(c, start=start, end=end), w) for c, w in obj.sleeves), obj.rebalance, obj.cost_bps)
        return run_mix(panel, mix)
    from dataclasses import replace
    return run_allocation(panel, replace(obj, start=start, end=end))


def evaluate(panel, cands, end: str) -> pd.DataFrame:
    rows = []
    bench = None
    for fam, name, obj in cands:
        try:
            r = run_one(panel, obj, START, end)
            r2 = run_one(panel, obj, HALF_START, end)
        except ValueError as exc:
            print(f"  건너뜀 {name}: {exc}")
            continue
        if bench is None:
            bench = r.benchmark
        e = r.equity
        cm = crisis_mdd(e)
        rows.append({"family": fam, "후보": name, "CAGR": cagr(e), "MDD": mdd(e), **{f"위기 {k}": v for k, v in cm.items()},
                     "위기 MDD": min(v for v in cm.values() if v == v), "2015+ CAGR": cagr(r2.equity), "2015+ MDD": mdd(r2.equity),
                     "회전율": r.turnover, "노출": float(r.exposure.mean())})
    df = pd.DataFrame(rows).set_index("후보")
    b = bench
    df.attrs["bench"] = {"CAGR": cagr(b), "MDD": mdd(b), **crisis_mdd(b), "2015+ CAGR": cagr(b.loc[HALF_START:])}
    df["S&P500 대비 CAGR"] = df["CAGR"] - df.attrs["bench"]["CAGR"]
    return df


def lenient_filter(df: pd.DataFrame, d_cagr: float = 0.003, d_mdd: float = 0.01) -> pd.DataFrame:
    """A22: drop a candidate only if another beats it on CAGR by >= d_cagr and on 위기 MDD by >= d_mdd
    without a worse full-period MDD. Adds 'dominated' and 'frontier' (strict Pareto on CAGR × 위기 MDD)."""
    c, k, m = df["CAGR"].to_numpy(), df["위기 MDD"].to_numpy(), df["MDD"].to_numpy()
    dominated = np.zeros(len(df), bool)
    frontier = np.ones(len(df), bool)
    for i in range(len(df)):
        better = (c >= c[i] + d_cagr) & (k >= k[i] + d_mdd) & (m >= m[i])
        dominated[i] = better.any()
        strict = (c >= c[i]) & (k >= k[i]) & ((c > c[i]) | (k > k[i]))
        frontier[i] = not strict.any()
    out = df.copy()
    out["dominated"], out["frontier"] = dominated, frontier
    return out


# ----------------------------------------------------------------------------- 2016+ snapshot check

SNAP_EXTRA = {  # code -> Product for sleeves that only have 2016+ proxy data here (or no Korean product yet)
    "VIG_": Product("VIG_", "(국내 상품 없음) 미국 배당성장 VIG", "주식", (("VIG", 1),), "unhedged", 0.25, "2016-08-01"),
    "USMV_": Product("USMV_", "(국내 상품 없음) 미국 최소변동성 USMV", "주식", (("USMV", 1),), "unhedged", 0.25, "2016-08-01"),
    "463680": Product("463680", "KODEX 미국S&P500테크놀로지", "주식", (("XLK", 1),), "unhedged", 0.25, "2023-08-01"),
    "453650": Product("453650", "KODEX 미국S&P500금융", "주식", (("XLF", 1),), "unhedged", 0.25, "2023-03-21"),
    "200030": Product("200030", "KODEX 미국S&P500산업재(합성)", "주식", (("XLI", 1),), "unhedged", 0.25, "2014-06-12"),
}
SNAP_WINDOWS = {"2018Q4": ("2018-09-20", "2018-12-24"), "코로나": ("2020-02-19", "2020-03-23"), "2022": ("2022-01-03", "2022-10-12")}


def snapshot_check() -> pd.DataFrame:
    snap = load_snapshot()
    products = {**PRODUCTS, **SNAP_EXTRA}
    syms = sorted({s for p in products.values() for s in p.proxy_symbols if s in snap.symbols})
    usd = snap.subset(syms)
    fx, bok = load_usdkrw(), load_bok_rate()
    panel, unavailable = krw_panel(usd, fx, bok, products=products)
    start, end = "2016-09-01", str(panel.last_date.date())
    sp = S45[SP]
    variants = {"안정형 45": S45, "내 AAA": AAA, "S&P500": {SP: 100}}
    thirds = {"헬스케어": "453640", "필수소비재": "453630", "유틸리티": "463640"}
    w = dict(S45)
    w[SP] = sp - 3 * (sp // 3)
    for code in thirds.values():
        w[code] = sp // 3
    variants["S45 S&P500 → 헬스케어·필수소비재·유틸리티 ⅓씩"] = {k: v for k, v in w.items() if v > 0}
    for lab, code in {**thirds, "최소변동성(USMV)": "USMV_", "배당성장(VIG)": "VIG_", "테크놀로지": "463680", "금융": "453650", "산업재": "200030"}.items():
        variants[f"S45 S&P500 절반 → {lab}"] = shift(S45, SP, code, sp // 2)
        variants[f"S45 S&P500 전부 → {lab}"] = shift(S45, SP, code, sp)
    rows = []
    for name, wts in variants.items():
        try:
            r = run_allocation(panel, AllocConfig(name, wts, start, end, rebalance="Q", cost_bps=10.0))
        except ValueError as exc:
            print(f"  건너뜀 {name}: {exc}")
            continue
        e = r.equity
        cm = {k: mdd(e.loc[a:b]) for k, (a, b) in SNAP_WINDOWS.items()}
        rows.append({"후보": name, "CAGR": cagr(e), "MDD": mdd(e), **{f"위기 {k}": v for k, v in cm.items()}, "위기 MDD": min(cm.values())})
    return pd.DataFrame(rows).set_index("후보")


# ----------------------------------------------------------------------------- main

def load_panel(want: str):
    symbols = [s for s in PROXY_SYMBOLS if s not in ("EWY", "XLV", "XLP")]
    usd, failed = download_panel(symbols, want)
    if failed:  # offline: fall back to the cached files, as the app does
        cached = [p for p in (CACHE_DIR / "yf").glob("*.csv.gz") if p.stem.split(".")[0] in symbols]
        last = min(pd.read_csv(p, usecols=["date"])["date"].max() for p in cached)
        usd, failed = download_panel(symbols, str(last))
    panel, unavailable = krw_panel(usd, load_usdkrw(), load_bok_rate())
    return panel, str(min(panel.last_date, pd.Timestamp(want)).date()), unavailable


def main(argv: list[str]) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    if "--snapshot" in argv:
        df = snapshot_check()
        print((df * 100).round(1).to_string())
        return
    panel, end, unavailable = load_panel(STUDY_END)
    if unavailable:
        print("대용 데이터 없음:", unavailable)
    cands = [c for c in candidates(end) if all(s in panel.symbols or s == CASH for s in _codes(c[2]))]
    print(f"원화 패널 {START} – {end}, 후보 {len(cands)}개")
    df = lenient_filter(evaluate(panel, cands, end))
    b = df.attrs["bench"]
    print(f"\n기준선 S&P500(원화): CAGR {b['CAGR']:.1%}, MDD {b['MDD']:.1%}, 위기 " + ", ".join(f"{k} {v:.1%}" for k, v in b.items() if k in WINDOWS))
    pct = [c for c in df.columns if df[c].dtype == float and c not in ("노출",)]
    shown = df.copy()
    shown[pct] = (shown[pct] * 100).round(1)
    print("\n전체 후보 (위기 MDD 순)")
    print(shown.sort_values("위기 MDD", ascending=False)[["family", "CAGR", "MDD", "위기 MDD", "2015+ CAGR", "2015+ MDD", "S&P500 대비 CAGR", "회전율", "dominated", "frontier"]].to_string())
    print(f"\n명확히 열등해서 제외: {int(df['dominated'].sum())}개 / 남은 후보 {int((~df['dominated']).sum())}개 / 프런티어 {int(df['frontier'].sum())}개")
    print("\n프런티어 (CAGR × 위기 MDD)")
    print(shown[df["frontier"]].sort_values("위기 MDD", ascending=False)[["family", "CAGR", "MDD", "위기 금융위기", "위기 코로나", "위기 2022", "2015+ CAGR", "2015+ MDD", "회전율"]].to_string())
    print("\n가족별 최선 (제외되지 않은 것 중 CAGR 상위 3)")
    for fam, g in shown[~df["dominated"]].groupby("family"):
        print(f"  [{fam}]")
        print(g.sort_values("CAGR", ascending=False).head(3)[["CAGR", "MDD", "위기 MDD", "2015+ CAGR"]].to_string())
    if "--out" in argv:
        path = Path(argv[argv.index("--out") + 1])
        df.to_csv(path, encoding="utf-8")
        print("저장:", path)


def _codes(obj) -> set[str]:
    if isinstance(obj, MixConfig):
        return {s for c, _ in obj.sleeves for s in c.weights}
    return set(obj.weights) | ({obj.overlay.safe} if obj.overlay.kind not in ("none", "riskparity") else set())


if __name__ == "__main__":
    main(sys.argv[1:])
