"""Asset catalog and allocation presets for the asset-allocation lab (lab/aa_*.py).

US-listed ETFs stand in for each asset class because they have the longest daily history
(2004 or earlier) and are what lab/data.py already downloads. The Korean-listed products the
user actually holds are listed next to each proxy as text only; their KRW prices, hedging
cost and fees are not modelled (see docs/research/allocation/README.md, "한계").
"""
from __future__ import annotations

from dataclasses import dataclass

CASH = "CASH"  # the panel's cash return (BIL total return, ^IRX/252 before BIL)


@dataclass(frozen=True)
class Asset:
    symbol: str
    label: str
    asset_class: str
    korea: str  # Korean-listed equivalents (information only)


ASSETS: dict[str, Asset] = {
    "SPY": Asset("SPY", "미국 대형주 (S&P 500)", "주식", "KODEX 미국S&P500, TIGER 미국S&P500 (미헤지)"),
    "QQQ": Asset("QQQ", "미국 나스닥 100", "주식", "KODEX 미국나스닥100, TIGER 미국나스닥100"),
    "IWM": Asset("IWM", "미국 소형주 (Russell 2000)", "주식", "KODEX 미국러셀2000(H)"),
    "DIA": Asset("DIA", "미국 다우 30", "주식", "TIGER 미국다우존스30"),
    "MDY": Asset("MDY", "미국 중형주 (S&P 400)", "주식", "국내 상장 없음"),
    "EFA": Asset("EFA", "선진국 주식 (미국 제외, EAFE)", "주식", "KODEX 선진국MSCI World는 미국 70% 포함 → SPY 0.7 + EFA 0.3로 근사"),
    "EEM": Asset("EEM", "신흥국 주식", "주식", "KODEX MSCI EM선물(H), ACE 신흥국MSCI"),
    "TLT": Asset("TLT", "미국 장기국채 20년+", "채권", "KODEX 미국30년국채액티브(H), TIGER 미국채30년스트립액티브(합성 H)"),
    "IEF": Asset("IEF", "미국 중기국채 7–10년", "채권", "KODEX 미국10년국채선물, KODEX 미국10년국채액티브(H)"),
    "AGG": Asset("AGG", "미국 종합채권", "채권", "KODEX 미국종합채권액티브(H)"),
    "LQD": Asset("LQD", "미국 투자등급 회사채", "채권", "ACE 미국회사채(H) 등"),
    "TIP": Asset("TIP", "미국 물가연동국채", "채권", "국내 상장 소수"),
    "GLD": Asset("GLD", "금", "실물", "KRX 금현물(증권사 금현물계좌, 비과세), ACE KRX금현물, KODEX 골드선물(H)"),
    "IYR": Asset("IYR", "미국 리츠", "실물", "TIGER 미국MSCI리츠(합성 H)"),
    "XLE": Asset("XLE", "미국 에너지 섹터", "주식", "KODEX 미국S&P에너지(합성)"),
    "XLU": Asset("XLU", "미국 유틸리티 섹터", "주식", "국내 상장 소수"),
}
DEFAULT_ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD"]
CLASS_ORDER = ["주식", "채권", "실물"]


def label(symbol: str) -> str:
    if symbol == CASH:
        return "현금 (단기국채)"
    a = ASSETS.get(symbol)
    return f"{symbol} · {a.label}" if a else symbol


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    weights: dict[str, float]  # percent, sums to 100
    note: str


# Presets are the user's own allocations and well-known static mixes mapped to the proxies above.
# MSCI World (70% US) is looked through: 15% World = 10.5% SPY + 4.5% EFA.
PRESETS: dict[str, Preset] = {
    "aaa_2026": Preset("aaa_2026", "내 AAA (2026-06 이후)",
                       {"SPY": 40.5, "EFA": 4.5, "EEM": 10, "TLT": 10, "IEF": 25, "GLD": 10},
                       "S&P500 30 · MSCI World 15 · EM 10 · 미국30년 10 · 미국10년 25 · 금 10. World는 SPY 0.7 + EFA 0.3으로 근사."),
    "aaa_2025": Preset("aaa_2025", "내 AAA 구배분 (2025-09 – 2026-06)",
                       {"SPY": 30.5, "EFA": 4.5, "TLT": 25, "IEF": 20, "GLD": 20},
                       "S&P500 20 · MSCI World 15 · 미국30년 25 · 미국10년 20 · 금 20."),
    "aaa_gold20": Preset("aaa_gold20", "내 AAA · 금 20 변형",
                         {"SPY": 40.5, "EFA": 4.5, "EEM": 10, "TLT": 10, "IEF": 15, "GLD": 20},
                         "2026-10-24 점검에서 제안한 변형: 금 10 → 20, 미국10년 25 → 15."),
    "spy": Preset("spy", "SPY 보유", {"SPY": 100}, "기준선. 미국 대형주 100%."),
    "sixty_forty": Preset("sixty_forty", "60/40", {"SPY": 60, "IEF": 40}, "주식 60 · 중기국채 40."),
    "all_weather": Preset("all_weather", "올웨더 (Dalio)", {"SPY": 30, "TLT": 40, "IEF": 15, "GLD": 15},
                          "주식 30 · 장기국채 40 · 중기국채 15 · 금 7.5 + 원자재 7.5(금으로 대체)."),
    "permanent": Preset("permanent", "영구 포트폴리오", {"SPY": 25, "TLT": 25, "GLD": 25, CASH: 25},
                        "주식 · 장기국채 · 금 · 현금 각 25."),
    "golden_butterfly": Preset("golden_butterfly", "골든 버터플라이",
                               {"SPY": 20, "IWM": 20, "TLT": 20, CASH: 20, "GLD": 20},
                               "주식 20 · 소형가치 20(IWM으로 대체) · 장기국채 20 · 단기국채 20(현금) · 금 20."),
    "gtaa5": Preset("gtaa5", "GTAA 5 동일비중", {"SPY": 20, "EFA": 20, "IEF": 20, "GLD": 20, "IYR": 20},
                    "Faber(2007)의 5자산 동일비중. 원 논문은 각 자산에 10개월 이동평균 필터를 얹음(추세 overlay '10개월 SMA'로 재현)."),
}
