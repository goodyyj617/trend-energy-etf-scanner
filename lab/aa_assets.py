"""Korean-listed ETFs and presets for the asset-allocation lab (lab/aa_*.py).

The user invests through a Korean ISA account, which can only buy Korean-listed products, so the
lab's building blocks are those products. Most of them were listed after 2015, so each one is
backed by a long-history US proxy (US ETF total return in USD) converted to a KRW return the way
the product earns it (lab/aa_korea.py):

- 환노출 (unhedged): the USD return times the change in USD/KRW.
- 환헤지 (H): the USD return minus the USD cash rate plus the KRW cash rate (what a rolled FX
  forward earns), minus an extra hedging cost.
- 원화 자산: KOSPI 200 products use EWY (USD) converted to KRW; the CD-rate ETF earns the
  Bank of Korea base rate.
- 레버리지: a daily-reset multiple of the excess return over the funding currency's cash rate.

Product facts (code, total fee, listing date, hedge) were checked on etfshopping.com, funetf.co.kr
and the issuers' pages on 2026-10-24. Fees are 총보수; the 실부담비용 can be higher.
"""
from __future__ import annotations

from dataclasses import dataclass

CASH = "CASH"  # uninvested KRW cash (Bank of Korea base rate)
KRW_CASH, USD_CASH = "KRW_CASH", "USD_CASH"  # pseudo proxies for the cash-like products
BENCHMARK_CODE = "379800"  # KODEX 미국S&P500, the KRW version of "SPY"
CLASS_ORDER = ["주식", "채권", "금·실물", "현금·달러", "레버리지"]
FX_LABELS = {"unhedged": "환노출", "hedged": "환헤지(H)", "krw": "원화", "krw_native": "원화(국내 주식)"}


@dataclass(frozen=True)
class Product:
    code: str
    name: str
    asset_class: str
    proxy: tuple  # ((US ticker or KRW_CASH / USD_CASH, weight), ...)
    fx: str  # "unhedged" | "hedged" | "krw" | "krw_native"
    ter: float  # total fee, percent per year
    listed: str  # listing date on KRX
    leverage: float = 1.0
    extra_drag: float = 0.0  # percent per year, a known gap the proxy does not capture
    note: str = ""

    @property
    def proxy_symbols(self) -> list[str]:
        return [s for s, _ in self.proxy if s not in (KRW_CASH, USD_CASH)]

    @property
    def proxy_text(self) -> str:
        names = {KRW_CASH: "한국은행 기준금리", USD_CASH: "달러 현금(미국 단기국채)"}
        parts = [f"{names.get(s, s)}" + ("" if w == 1 else f" {w:g}") for s, w in self.proxy]
        lev = "" if self.leverage == 1 else f" × {self.leverage:g}배 (일일)"
        return " + ".join(parts) + lev


def _p(code, name, cls, proxy, fx, ter, listed, **kw) -> Product:
    return Product(code, name, cls, tuple(proxy), fx, ter, listed, **kw)


PRODUCTS: dict[str, Product] = {p.code: p for p in [
    # 주식
    _p("379800", "KODEX 미국S&P500", "주식", [("SPY", 1)], "unhedged", 0.0062, "2021-04-09"),
    _p("449180", "KODEX 미국S&P500(H)", "주식", [("SPY", 1)], "hedged", 0.0099, "2022-12-02"),
    _p("379810", "KODEX 미국나스닥100", "주식", [("QQQ", 1)], "unhedged", 0.0062, "2021-04-09"),
    _p("251350", "KODEX MSCI선진국", "주식", [("SPY", 0.7), ("EFA", 0.3)], "unhedged", 0.30, "2016-08-17",
       note="MSCI World 추종(‘선진국 MSCI World’). 대용은 미국 70 · 미국 외 선진국 30 고정 비율이라 2010년 전후 미국 비중(약 50%)보다 미국을 더 담음."),
    _p("291890", "KODEX MSCI EM선물(H)", "주식", [("EEM", 1)], "hedged", 0.45, "2018-03-23",
       note="MSCI 신흥국 지수 선물(초과수익) + 원화 현금 = 환헤지 신흥국 주식."),
    _p("280930", "KODEX 미국러셀2000(H)", "주식", [("IWM", 1)], "hedged", 0.45, "2017-11-09"),
    _p("069500", "KODEX 200", "주식", [("EWY", 1)], "krw_native", 0.15, "2002-10-14",
       note="코스피 200. 대용은 EWY(MSCI 한국, 달러)를 원화로 환산한 값."),
    # 채권
    _p("484790", "KODEX 미국30년국채액티브(H)", "채권", [("TLT", 1)], "hedged", 0.015, "2024-06-18",
       note="미국 국채 20년+ 지수(TR) 환헤지."),
    _p("0091C0", "KODEX 미국10년국채액티브(H)", "채권", [("IEF", 1)], "hedged", 0.15, "2025-08-19",
       note="미국 국채 10년 지수(TR) 환헤지. 대용 IEF는 7–10년 국채라 듀레이션이 조금 짧음."),
    # 금·실물
    _p("411060", "ACE KRX금현물", "금·실물", [("GLD", 1)], "unhedged", 0.19, "2021-12-15",
       note="KRX 금시장 가격(원화)을 따름 = 국제 금값 × 환율 근사. 국내 금 프리미엄(2025년 한때 +10% 이상)은 반영 안 됨. 다른 운용사의 KRX금현물 ETF도 같은 방식."),
    _p("132030", "KODEX 골드선물(H)", "금·실물", [("GLD", 1)], "hedged", 0.68, "2010-10-01"),
    _p("352560", "KODEX 미국부동산리츠(H)", "금·실물", [("IYR", 1)], "hedged", 0.09, "2020-05-13"),
    # 현금·달러
    _p("459580", "KODEX CD금리액티브(합성)", "현금·달러", [(KRW_CASH, 1)], "krw", 0.02, "2023-06-08",
       note="CD 금리를 받음. 대용은 한국은행 기준금리(실제 CD 금리는 보통 0.1–0.5%p 높음)."),
    _p("261240", "KODEX 미국달러선물", "현금·달러", [(USD_CASH, 1)], "unhedged", 0.25, "2016-12-27",
       note="원/달러 환율 + 달러 금리. 국내 투자자에게 주식 폭락기 방패 역할(2008·2020년 원화 약세)."),
    # 레버리지 (ISA에서 매수 가능, 증권사 레버리지 ETF 사전교육·기본예탁금 필요)
    _p("225040", "TIGER 미국S&P500레버리지(합성 H)", "레버리지", [("SPY", 1)], "hedged", 0.25, "2015-07-29", leverage=2.0,
       extra_drag=3.0, note="S&P 500 가격지수(배당 제외)의 2배라 배당 2배만큼 덜 받음 → 연 3%p 차감(배당수익률 1.5% 가정)."),
    _p("409820", "KODEX 미국나스닥100레버리지(합성 H)", "레버리지", [("QQQ", 1)], "hedged", 0.30, "2021-12-09", leverage=2.0),
    _p("418660", "TIGER 미국나스닥100레버리지(합성)", "레버리지", [("QQQ", 1)], "unhedged", 0.25, "2022-02-22", leverage=2.0),
    _p("452250", "ACE 미국30년국채선물레버리지(합성 H)", "레버리지", [("TLT", 1)], "hedged", 0.25, "2023-02-07", leverage=2.0,
       note="울트라 T-본드 선물 2배. 실제 듀레이션이 TLT보다 길어 대용(TLT 2배)은 금리 민감도를 낮게 잡음."),
    _p("225130", "ACE 골드선물 레버리지(합성 H)", "레버리지", [("GLD", 1)], "hedged", 0.49, "2015-07-28", leverage=2.0),
    _p("122630", "KODEX 레버리지", "레버리지", [("EWY", 1)], "krw_native", 0.64, "2010-02-22", leverage=2.0,
       note="코스피 200 2배. 대용은 EWY 원화 환산의 2배."),
]}
PROXY_SYMBOLS = sorted({s for p in PRODUCTS.values() for s in p.proxy_symbols})
DEFAULT_PRODUCTS = ["379800", "251350", "291890", "484790", "0091C0", "411060"]


def label(code: str) -> str:
    if code == CASH:
        return "현금 (원화 단기금리)"
    p = PRODUCTS.get(code)
    return f"{p.name} ({code})" if p else code


def short(code: str) -> str:
    if code == CASH:
        return "현금"
    p = PRODUCTS.get(code)
    return p.name.replace("KODEX ", "") if p else code


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    weights: dict[str, float]  # product code -> percent, sums to 100
    note: str


# Registered 2026-10-24 as the Korean-product versions of the presets in DECISIONS A10 (A14).
PRESETS: dict[str, Preset] = {p.key: p for p in [
    Preset("aaa_2026", "내 AAA (2026-06 이후)",
           {"379800": 30, "251350": 15, "291890": 10, "484790": 10, "0091C0": 25, "411060": 10},
           "S&P500 30 · MSCI선진국 15 · EM선물(H) 10 · 미국30년(H) 10 · 미국10년(H) 25 · 금현물 10. 실제 보유 상품 그대로."),
    Preset("aaa_2025", "내 AAA 구배분 (2025-09 – 2026-06)",
           {"379800": 20, "251350": 15, "484790": 25, "0091C0": 20, "411060": 20},
           "S&P500 20 · MSCI선진국 15 · 미국30년(H) 25 · 미국10년(H) 20 · 금현물 20."),
    Preset("aaa_gold20", "내 AAA · 금 20 변형",
           {"379800": 30, "251350": 15, "291890": 10, "484790": 10, "0091C0": 15, "411060": 20},
           "2026-10-24 점검에서 제안한 변형: 금 10 → 20, 미국10년 25 → 15."),
    # Registered after the 2026-10-24 lower-drawdown study had been run (DECISIONS A20–A21).
    Preset("stable45", "안정형 45 (금·현금 분산)",
           {"379800": 25, "251350": 12, "291890": 8, "484790": 7, "0091C0": 18, "411060": 18, "459580": 12},
           "주식 45(S&P500 25 · 선진국 12 · EM 8) · 미국30년(H) 7 · 미국10년(H) 18 · 금현물 18 · CD금리 12. "
           "2026-10-24 낙폭 연구의 권고안(결과를 본 뒤 등록)."),
    Preset("stable35", "안정형 35 (금·현금 분산)",
           {"379800": 19, "251350": 10, "291890": 6, "484790": 8, "0091C0": 22, "411060": 20, "459580": 15},
           "주식 35(S&P500 19 · 선진국 10 · EM 6) · 미국30년(H) 8 · 미국10년(H) 22 · 금현물 20 · CD금리 15. "
           "2026-10-24 낙폭 연구에서 낙폭이 가장 작았던 배분(결과를 본 뒤 등록)."),
    Preset("sp500", "S&P500 보유", {"379800": 100}, "기준선. KODEX 미국S&P500 100% (환노출)."),
    Preset("sixty_forty", "60/40", {"379800": 60, "0091C0": 40}, "S&P500 60 · 미국10년국채(H) 40."),
    Preset("all_weather", "올웨더 (Dalio)", {"379800": 30, "484790": 40, "0091C0": 15, "411060": 15},
           "주식 30 · 장기국채 40 · 중기국채 15 · 금 15(원자재 7.5를 금으로 대체)."),
    Preset("permanent", "영구 포트폴리오", {"379800": 25, "484790": 25, "411060": 25, "459580": 25},
           "주식 · 장기국채 · 금 · 현금(CD금리) 각 25."),
    Preset("golden_butterfly", "골든 버터플라이", {"379800": 20, "280930": 20, "484790": 20, "459580": 20, "411060": 20},
           "주식 20 · 소형주 20(러셀2000(H)) · 장기국채 20 · 현금 20 · 금 20."),
    Preset("gtaa5", "GTAA 5 동일비중", {"379800": 20, "251350": 20, "0091C0": 20, "411060": 20, "352560": 20},
           "Faber(2007)의 5자산 동일비중(미국 외 선진국은 MSCI선진국으로 대체). 원 논문은 각 자산에 10개월 이동평균 필터(overlay ‘추세 필터 10’)."),
]}
