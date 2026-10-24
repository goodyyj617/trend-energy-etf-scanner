"""Where do the presets' drawdowns come from, and could a signal have avoided them? (DECISIONS A29;
memo 2026-10-25_gate_reset_and_validation.md)

    .venv/Scripts/python.exe scripts/aa_crisis_signal_study.py

Part 1 lists the deepest drawdown episodes of the recommended presets (KRW, 2005-01 – 2026-10).
Part 2 runs each preset with a monthly risk-off switch (equity products -> CD금리 or 미국달러선물)
for the signals registered in lab/aa_signals.SIGNALS before the run, plus the hindsight "oracle"
as the upper bound. Prints tables only; nothing is written to lab_results/.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.aa_assets import PRESETS, PRODUCTS, PROXY_SYMBOLS  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation  # noqa: E402
from lab.aa_korea import krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.aa_signals import SIGNALS, drawdown_episodes, label_episode, load_macro, run_switch, signal_series  # noqa: E402
from lab.aa_validate import headline  # noqa: E402
from lab.data import download_panel  # noqa: E402

START, END, HALF = "2005-01-03", "2026-10-16", "2015-01-02"
WINDOWS = {"금융위기": ("2007-10-09", "2009-03-09"), "코로나": ("2020-02-19", "2020-03-23"), "2022": ("2022-01-03", "2022-10-12")}
BASES = {"안정형 40": "stable40", "안정형 45": "stable45", "내 AAA": "aaa_2026"}
SAFES = {"CD금리": "459580", "달러선물": "261240"}
US = ["SPY", "VEA", "EFA", "VWO", "EEM", "BND", "AGG", "TIP", "BIL"]  # VIX from lab/macro_snapshot


def pct(x: float) -> str:
    return "  nan" if x != x else f"{x * 100:5.1f}"


def mdd(e: pd.Series) -> float:
    return float((e / e.cummax() - 1).min())


def crisis(e: pd.Series) -> dict[str, float]:
    return {k: mdd(e.loc[a:b]) if len(e.loc[a:b]) > 5 else float("nan") for k, (a, b) in WINDOWS.items()}


def main() -> None:
    usd, failed = download_panel(sorted(set(US) | set(PROXY_SYMBOLS)), end="2026-10-23")
    if failed:
        print("받지 못한 종목:", failed)
    krw, _ = krw_panel(usd, load_usdkrw(), load_bok_rate())
    macro = load_macro()

    print("[1] 낙폭 에피소드 (원화, 분기 리밸런싱, 10bp, 2005-01 – 2026-10). 깊이 3% 이상, 깊은 순 6개")
    base_res = {}
    for name, key in {**BASES, "S&P500": "sp500"}.items():
        cfg = AllocConfig(name, PRESETS[key].weights, START, END, rebalance="Q", cost_bps=10.0)
        r = run_allocation(krw, cfg)
        base_res[name] = r
        ep = drawdown_episodes(r.equity, top=6)
        print(f"  {name}: MDD {pct(mdd(r.equity))}")
        for e in ep.itertuples():
            rec = "미회복" if pd.isna(e.회복) else f"{e.회복.date()} ({int(e._5)}일)"
            print(f"     {pct(e.깊이)}  {label_episode(e.바닥):14s} {e.시작.date()} → 바닥 {e.바닥.date()} → 회복 {rec}")
        dd = r.equity / r.equity.cummax() - 1
        print(f"     낙폭이 −5%보다 깊었던 날 {float((dd < -0.05).mean()):.0%}, −10%보다 깊었던 날 {float((dd < -0.10).mean()):.1%}")

    print("\n[2] 위험 회피 스위치: 신호가 켜지면 주식 상품(S&P500·선진국·EM) 전부를 안전 상품으로, 월말 판단·다음 날 종가 매매")
    equity_codes = [c for c, p in PRODUCTS.items() if p.asset_class == "주식"]
    dates = krw.close.index
    flags = {k: signal_series(k, usd.close, usd.cash, dates, macro) for k in SIGNALS}
    for k, f in flags.items():
        f2 = f.loc[START:END]
        print(f"  신호 {SIGNALS[k]}: 켜진 달 {f2.mean():.0%}, 켜짐/꺼짐 전환 {int((f2.astype(int).diff().abs() == 1).sum())}회")
    for bname, key in BASES.items():
        for sname, safe in SAFES.items():
            print(f"\n  ── {bname}, 회피처 {sname}")
            cfg = AllocConfig(bname, PRESETS[key].weights, START, END, rebalance="Q", cost_bps=10.0)
            rows = [("스위치 없음", base_res[bname])]
            for k in SIGNALS:
                rows.append((SIGNALS[k], run_switch(krw, cfg, flags[k], equity_codes, safe)))
            print(f"    {'신호':44s}  CAGR   MDD   샤프 | 금융위기 코로나  2022 | 2015+ CAGR MDD | 위기 밖 MDD")
            for label, r in rows:
                h = headline(r)
                c = crisis(r.equity)
                late = r.equity.loc[HALF:]
                outside = r.equity.copy()
                for a, b in WINDOWS.values():
                    outside = outside.drop(outside.loc[a:b].index)
                print(f"    {label[:44]:44s} {pct(h['cagr'])} {pct(h['mdd'])} {h['sharpe']:5.2f} |"
                      f" {pct(c['금융위기'])} {pct(c['코로나'])} {pct(c['2022'])} | {pct(headline(late, r.cash)['cagr'])} {pct(mdd(late))} |"
                      f" {pct(mdd(outside))}")

    print("\n[3] 날짜 민감도: 신호 판단일을 말일 / +5 / +10거래일로 바꿀 때 (안정형 40, 회피처 CD금리)")
    cfg0 = AllocConfig("안정형 40", PRESETS["stable40"].weights, START, END, rebalance="Q", cost_bps=10.0)
    for k in ("spy_sma10", "spy_sma200", "baa_canary", "tip_canary", "vix25"):
        cells = []
        for o in (0, 5, 10):
            f = signal_series(k, usd.close, usd.cash, dates, macro, offset=o)
            r = run_switch(krw, replace(cfg0, rebalance_offset=o), f, equity_codes, "459580")
            h = headline(r)
            cells.append(f"+{o}: {pct(h['cagr'])}/{pct(h['mdd'])}")
        print(f"  {SIGNALS[k][:40]:40s} " + "   ".join(cells))


if __name__ == "__main__":
    main()
