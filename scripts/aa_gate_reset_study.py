"""HAA + BAA reproduction, timing luck and the final-Gate reset (DECISIONS A24–A28, A30; memo 2026-10-25_gate_reset_and_validation.md).

    .venv/Scripts/python.exe scripts/aa_gate_reset_study.py            # everything, data to 2026-10-23
    .venv/Scripts/python.exe scripts/aa_gate_reset_study.py --quick    # skip bootstrap and lazy draws

Reads the yfinance cache in lab_data/yf (downloads what is missing). Prints tables only; nothing is
written to lab_results/ and nothing is added to the lab's run log.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.aa_assets import PRESETS, PROXY_SYMBOLS  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation  # noqa: E402
from lab.aa_korea import krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.aa_score import AllocGates, gate_results, raw_metrics  # noqa: E402
from lab.aa_tactical import TacticalSpec, krw_coverage, monthly_sharpe, run_tactical  # noqa: E402
from lab.aa_validate import (block_bootstrap, cscv, headline, holdout_split, lazy_table, split_metrics,  # noqa: E402
                             timing_summary, timing_table, timing_worst_mdd)
from lab.data import download_panel  # noqa: E402

END = "2026-10-23"
FRIEND = ("2008-01-01", "2026-08-31")  # the friend's tool: USD 2008.1 – 2026.8
KRW_START, KRW_END = "2008-01-01", "2026-10-16"  # USD/KRW snapshot ends 2026-10-16
LONG_START = "2005-01-03"
US = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT", "TIP", "LQD", "BND", "AGG", "SHY", "BIL", "GLD"]
TACTICAL = {"HAA": (("haa", 100.0),), "BAA 공격형": (("baa_g4", 100.0),), "VAA 공격형": (("vaa_g4", 100.0),),
            "HAA+BAA 50:50": (("haa", 50.0), ("baa_g4", 50.0))}
STATIC = {"안정형 35": "stable35", "안정형 40": "stable40", "안정형 45": "stable45", "내 AAA": "aaa_2026",
          "영구": "permanent", "60/40": "sixty_forty", "S&P500": "sp500"}


def pct(x: float) -> str:
    return "nan" if x != x else f"{x * 100:5.1f}"


def line(label: str, h: dict, extra: str = "") -> str:
    return f"  {label:28s} CAGR {pct(h['cagr'])}  MDD {pct(h['mdd'])}  샤프 {h['sharpe']:.2f}  {extra}"


def load() -> tuple:
    usd, failed = download_panel(sorted(set(US) | set(PROXY_SYMBOLS)), end=END)
    if failed:
        print("받지 못한 종목:", failed)
    krw, _ = krw_panel(usd, load_usdkrw(), load_bok_rate())
    return usd, krw


def tactical(usd, krw, parts, holdings="krw", start=KRW_START, end=KRW_END, **kw):
    spec = TacticalSpec("t", parts=parts, start=start, end=end, **kw)
    return run_tactical(usd if holdings == "usd" else krw, usd.close, spec, holdings, usd.cash)


def static(krw, key, start=KRW_START, end=KRW_END, offset=0):
    cfg = AllocConfig(key, PRESETS[key].weights, start, end, rebalance="Q", cost_bps=10.0, rebalance_offset=offset)
    return run_allocation(krw, cfg)


def main(argv: list[str]) -> None:
    quick = "--quick" in argv
    usd, krw = load()

    print("\n[1] 지인 도구 재현: USD, 2008-01 – 2026-08, 10bp, 월말 신호")
    for name, parts in TACTICAL.items():
        for same in (True, False):
            r = tactical(usd, krw, parts, "usd", *FRIEND, same_close=same)
            h = headline(r)
            print(line(f"{name} ({'같은 날 종가' if same else '다음 날 종가'})", h,
                       f"월간샤프(rf 0) {monthly_sharpe(r.equity):.2f}  회전 {r.turnover * 100:.0f}%/년"))
    b = usd.close["SPY"].loc[FRIEND[0]:FRIEND[1]]
    print(line("SPY", headline(b / b.iloc[0], usd.cash)))

    print("\n[2] 원화·국내 ETF 버전 (다음 날 종가, DBC→CD금리), 2008-01 – 2026-10")
    res = {}
    for name, parts in TACTICAL.items():
        res[name] = tactical(usd, krw, parts)
    for name, key in STATIC.items():
        res[name] = static(krw, key)
    for name, r in res.items():
        print(line(name, headline(r), f"회전 {r.turnover * 100:.0f}%/년"))
    gold = tactical(usd, krw, TACTICAL["HAA+BAA 50:50"], missing_to="411060")
    print(line("HAA+BAA (DBC→금현물)", headline(gold)))
    cov = krw_coverage(res["HAA+BAA 50:50"].signals)
    print(f"  HAA+BAA 국내 대체 커버리지(월평균): 정확 {cov['정확'].mean():.0%} · 근사 {cov['근사'].mean():.0%} · 없음 {cov['없음'].mean():.0%};"
          f" '없음'이 30% 이상인 달 {(cov['없음'] >= 0.3).mean():.0%}")

    print("\n[3] 리밸런싱 날짜 민감도 (원화, 다음 날 종가). 정적 배분은 분기, 전술은 매월")
    timing = {}
    for name, parts in TACTICAL.items():
        timing[name] = timing_table(lambda o, d, p=parts: tactical(usd, krw, p, offset=o, delay=d))
    for name, key in STATIC.items():
        timing[name] = timing_table(lambda o, d, k=key: static(krw, k, offset=o), delays=(0,))
    for name, t in timing.items():
        s = timing_summary(t)
        cells = "  ".join(f"{r.타이밍}: {pct(r.cagr)}/{pct(r.mdd)}" for r in t.itertuples())
        print(f"  {name:16s} MDD 최악 {pct(s['mdd_worst'])} (기준 {pct(s['mdd_base'])})  CAGR {pct(s['cagr_min'])}–{pct(s['cagr_max'])}\n      {cells}")

    if not quick:
        print("\n[4] 게으른 리밸런싱 (다음 매매가 1·2·3개월 뒤일 확률 60·30·10%, 50회, 시드 7)")
        for name in ("HAA", "BAA 공격형", "HAA+BAA 50:50"):
            months = [str(d.to_period("M")) for d in res[name].signals.index]
            lt = lazy_table(lambda skip, p=TACTICAL[name]: tactical(usd, krw, p, skip_months=skip), months)
            print(f"  {name:16s} CAGR 중앙 {pct(lt['cagr'].median())} · 하위10% {pct(lt['cagr'].quantile(0.1))}  "
                  f"MDD 중앙 {pct(lt['mdd'].median())} · 하위10% {pct(lt['mdd'].quantile(0.1))}")

    print("\n[5] 성역(마지막 15%)과 선정 구간")
    for name in ("HAA", "HAA+BAA 50:50", "안정형 40", "안정형 45", "내 AAA", "S&P500"):
        sm = split_metrics(res[name])
        print(f"  {name:16s} " + "  ".join(f"{i}: {pct(r.cagr)}/{pct(r.mdd)}/{r.sharpe:.2f}" for i, r in sm.iterrows()))

    if not quick:
        print("\n[6] 블록 부트스트랩 (21일 블록, 1000회) 5–50–95%")
        for name in ("HAA", "HAA+BAA 50:50", "안정형 40", "안정형 45"):
            bb = block_bootstrap(res[name])
            print(f"  {name:16s} 샤프 {bb['샤프'].iloc[0]:.2f}–{bb['샤프'].iloc[1]:.2f}–{bb['샤프'].iloc[2]:.2f}  "
                  f"MDD {pct(bb['MDD'].iloc[0])}–{pct(bb['MDD'].iloc[1])}–{pct(bb['MDD'].iloc[2])}")

    print("\n[7] 정직(튜닝) vs 균등: CSCV 252분할, 성역 제외 구간, 슬리브 최대 3개 · 10% 격자")
    names = ["HAA", "BAA 공격형", "VAA 공격형", "안정형 40", "내 AAA", "영구", "60/40", "S&P500"]
    eq = pd.DataFrame({n: res[n].equity for n in names}).dropna()
    cut = holdout_split(eq.index)
    rets = eq.loc[:cut].pct_change(fill_method=None).dropna()
    out = cscv(rets, krw.cash)
    print(f"  후보 {out['candidates']}개 · 분할 {out['splits']}개 · 튜닝이 균등을 이긴 분할 {out['win_rate']:.0%} · PBO {out['pbo']:.0%}")
    print(f"  표본밖 샤프 중앙값: 튜닝 {out['tuned_out_median']:.2f} · 균등 {out['equal_out_median']:.2f} · 천장 {out['ceiling_out_median']:.2f}")
    print("  자주 뽑힌 튜닝 비중:", "; ".join(f"{k} ({v})" for k, v in out["top_picks"].items()))

    print("\n[8] 최종 Gate 세 버전, 2008-01 – 2026-10: 기존(CAGR ≥ 0.8×, |MDD| ≤ 0.75×|S&P500 MDD|) · "
          "A26(CAGR ≥ 0.7×, MDD ≥ −15%) · 현행 A30(A26 + 날짜 바꾼 최악 MDD ≥ −20%)")
    old = AllocGates(min_cagr_ratio=0.80, max_mdd_abs=None, max_timing_mdd_abs=None)
    a26 = AllocGates(max_timing_mdd_abs=None)
    now = AllocGates()
    for name, r in res.items():
        m = raw_metrics(r)
        m["timing_mdd_worst"] = timing_summary(timing[name])["mdd_worst"]
        verdict = "  ".join(f"{k} {'통과' if gate_results(m, g)['최종 Gate'] else '탈락'}" for k, g in (("기존", old), ("A26", a26), ("현행", now)))
        print(f"  {name:16s} CAGR {pct(m['cagr'])} (기준 {pct(0.7 * m['spy_cagr'])})  MDD {pct(m['mdd'])}  "
              f"날짜 바꾼 최악 MDD {pct(m['timing_mdd_worst'])}  {verdict}")
    print("\n[9] 2005-01 시작(정적 배분만, 전술은 DBC 12개월 이력 때문에 2007-03부터), 현행 최종 Gate")
    long_runs = {name: (static(krw, key, start=LONG_START), lambda o, k=key: static(krw, k, start=LONG_START, offset=o))
                 for name, key in STATIC.items()}
    for name in ("HAA", "HAA+BAA 50:50"):
        long_runs[name + " (2007-03–)"] = (tactical(usd, krw, TACTICAL[name], start="2007-03-01"),
                                          lambda o, p=TACTICAL[name]: tactical(usd, krw, p, start="2007-03-01", offset=o))
    for name, (r, run) in long_runs.items():
        m = raw_metrics(r)
        m["timing_mdd_worst"] = timing_worst_mdd(run, r)
        print(line(name, headline(r), f"날짜 견고 {pct(m['timing_mdd_worst'])}  "
                                      f"A26 {'통과' if gate_results(m, a26)['최종 Gate'] else '탈락'}  "
                                      f"현행 {'통과' if gate_results(m, now)['최종 Gate'] else '탈락'}"))


if __name__ == "__main__":
    main(sys.argv[1:])
