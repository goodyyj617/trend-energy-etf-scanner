"""Re-run the 2026-10-24 lower-drawdown study (docs/research/allocation/2026-10-24_lower_drawdown.md).

    .venv/Scripts/python.exe scripts/aa_lower_drawdown_study.py              # the memo's period (to 2026-10-08)
    .venv/Scripts/python.exe scripts/aa_lower_drawdown_study.py 2027-03-31   # a later end date

Candidates and the judging rule were registered in DECISIONS A20 before the first run; the
robustness checks at the end were added after it and were not used to choose. Prints tables only:
nothing is written to lab_results/ and nothing is added to the lab's run log.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.aa_assets import PRESETS, PROXY_SYMBOLS  # noqa: E402
from lab.aa_engine import AllocConfig, Leverage, Overlay, run_allocation  # noqa: E402
from lab.aa_korea import KrwModel, krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.aa_score import crisis_table, raw_metrics  # noqa: E402
from lab.data import CACHE_DIR, download_panel  # noqa: E402

START, HALF_END, HALF_START = "2005-01-03", "2014-12-31", "2015-01-02"
STUDY_END = "2026-10-08"  # last day of the data the memo used
AAA = PRESETS["aaa_2026"].weights
GRID = {
    "E55-현재 (내 AAA)": AAA,
    "E55-분산": {"379800": 30, "251350": 15, "291890": 10, "484790": 5, "0091C0": 15, "411060": 15, "459580": 10},
    "E45-현재형": {"379800": 25, "251350": 12, "291890": 8, "484790": 12, "0091C0": 30, "411060": 13},
    "E45-분산": PRESETS["stable45"].weights,
    "E35-현재형": {"379800": 19, "251350": 10, "291890": 6, "484790": 15, "0091C0": 35, "411060": 15},
    "E35-분산": PRESETS["stable35"].weights,
    "E55-달러": {"379800": 30, "251350": 15, "291890": 10, "484790": 10, "0091C0": 15, "411060": 10, "261240": 10},
}
CRISES = ("금융위기 2007–09", "코로나 2020", "금리 급등 2022")


def candidates() -> dict[str, dict]:
    out = {n: dict(weights=w) for n, w in GRID.items()}
    out["AAA + 리스크 패리티"] = dict(weights=AAA, overlay=Overlay("riskparity", 63.0))
    out["AAA + 변동성 목표 8%"] = dict(weights=AAA, overlay=Overlay("voltarget", 8.0), leverage=Leverage(1.0))
    out["AAA + 추세 필터 10 → CD"] = dict(weights=AAA, overlay=Overlay("sma", 10.0, "459580"))
    out["AAA + 밴드 5%p"] = dict(weights=AAA, band=5.0)
    for k in ("sp500", "sixty_forty", "all_weather", "permanent", "golden_butterfly", "aaa_gold20", "aaa_2025"):
        out[f"[참고] {PRESETS[k].name}"] = dict(weights=PRESETS[k].weights)
    return out


def run(panel, spec: dict, start: str, end: str):
    cfg = AllocConfig("x", spec["weights"], start, end, rebalance="Q", band=spec.get("band", 0.0), cost_bps=10.0,
                      overlay=spec.get("overlay", Overlay()), leverage=spec.get("leverage", Leverage()))
    return run_allocation(panel, cfg)


def mdd(e: pd.Series) -> float:
    return float((e / e.cummax() - 1).min())


def cagr(e: pd.Series) -> float:
    return float((e.iloc[-1] / e.iloc[0]) ** (252 / (len(e) - 1)) - 1)


def study(panel, end: str) -> pd.DataFrame:
    rows = []
    for name, spec in candidates().items():
        row = {"후보": name}
        for tag, (a, b) in {"전체": (START, end), "전반": (START, HALF_END), "후반": (HALF_START, end)}.items():
            res = run(panel, spec, a, b)
            m = raw_metrics(res)
            row.update({f"{tag} CAGR": m["cagr"], f"{tag} MDD": m["mdd"], f"{tag} 칼마": m["calmar"]})
            if tag == "전체":
                row.update({"최악 12개월": m["worst_12m"], "최장 수면하(일)": m["underwater_days"], "회전율": m["turnover"]})
                ct = crisis_table(res).set_index("구간")
                row.update({c: ct.loc[c, "전략"] if c in ct.index else np.nan for c in CRISES})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("후보")
    df["반기 최악 MDD"] = df[["전반 MDD", "후반 MDD"]].min(axis=1)
    return df


def judge(df: pd.DataFrame) -> pd.DataFrame:
    """A20: CAGR floor = 내 AAA − 1 %p, then worse half MDD, full Calmar, longest underwater."""
    floor = df.loc["E55-현재 (내 AAA)", "전체 CAGR"] - 0.01
    ok = df[df["전체 CAGR"] >= floor]
    return ok.sort_values(["반기 최악 MDD", "전체 칼마", "최장 수면하(일)"], ascending=[False, False, True])


def robustness(panel, usd, fx, bok, end: str) -> None:
    names = {"내 AAA": AAA, "E45-분산": GRID["E45-분산"], "E35-분산": GRID["E35-분산"], "E55-달러": GRID["E55-달러"],
             "영구": PRESETS["permanent"].weights}
    eq = {n: run(panel, dict(weights=w), START, end).equity for n, w in names.items()}
    base = eq["내 AAA"]
    starts = pd.date_range("2005-01-31", pd.Timestamp(end) - pd.DateOffset(years=5), freq="ME")
    print(f"\n5년 구간 (월마다 시작, {len(starts)}개), 내 AAA 대비")
    for n in list(names)[1:]:
        dm, dc = [], []
        for s in starts:
            i, j = base.index.searchsorted(s), base.index.searchsorted(s + pd.DateOffset(years=5))
            dm.append(mdd(eq[n].iloc[i:j]) - mdd(base.iloc[i:j]))
            dc.append(cagr(eq[n].iloc[i:j]) - cagr(base.iloc[i:j]))
        dm, dc = np.array(dm), np.array(dc)
        print(f"  {n}: MDD가 더 작았던 구간 {np.mean(dm > 0):.0%}, MDD 개선 중앙값 {np.median(dm) * 100:+.1f}%p, "
              f"CAGR 차이 중앙값 {np.median(dc) * 100:+.2f}%p, 최악 {dc.min() * 100:+.1f}%p")
    print("\n금 급락기 2011-09-06 → 2015-12-17")
    for n, e in eq.items():
        w = e.loc["2011-09-06":"2015-12-17"]
        print(f"  {n}: 수익 {w.iloc[-1] / w.iloc[0] - 1:+.1%}, 구간 MDD {mdd(w):.1%}")
    p50, _ = krw_panel(usd, fx, bok, KrwModel(hedge_extra_bps=50))
    print("\n헤지 추가 비용 50bp")
    for n, w in names.items():
        e = run(p50, dict(weights=w), START, end).equity
        print(f"  {n}: CAGR {cagr(e):.2%}, MDD {mdd(e):.1%}")


def main() -> None:
    want = sys.argv[1] if len(sys.argv) > 1 else STUDY_END
    symbols = [s for s in PROXY_SYMBOLS if s != "EWY"]  # EWY: no KOSPI product is used
    usd, failed = download_panel(symbols, want)
    if failed:  # offline: fall back to the cached files, as the app does
        cached = [p for p in (CACHE_DIR / "yf").glob("*.csv.gz") if p.stem.split(".")[0] in symbols]
        last = min(pd.read_csv(p, usecols=["date"])["date"].max() for p in cached)
        usd, failed = download_panel(symbols, str(last))
    fx, bok = load_usdkrw(), load_bok_rate()
    panel, _ = krw_panel(usd, fx, bok)
    end = str(min(panel.last_date, pd.Timestamp(want)).date())
    print(f"원화 패널 {START} – {end}" + (f" (받지 못한 대용: {failed})" if failed else ""))
    df = study(panel, end)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pct = [c for c in df.columns if "CAGR" in c or "MDD" in c or c in CRISES or c in ("최악 12개월", "회전율")]
    shown = df.copy()
    shown[pct] = (shown[pct] * 100).round(1)
    print(shown.round(2).to_string())
    print("\nA20 판정 순서")
    print((judge(df)[["전체 CAGR", "반기 최악 MDD", "전체 칼마"]] * [100, 100, 1]).round(2).to_string())
    robustness(panel, usd, fx, bok, end)


if __name__ == "__main__":
    main()
