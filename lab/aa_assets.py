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
    # 미국 섹터 (DECISIONS A22, 2026-10-24 추가). KODEX S&P500 섹터 시리즈는 환노출로 봄(테크놀로지 상품 자료에 '원화 환산 지수,
    # 환헤지 미실시'. 나머지는 명시 없음).
    _p("245340", "TIGER 미국다우존스30", "주식", [("DIA", 1)], "unhedged", 0.35, "2016-07-01", note="다우존스 산업평균 30종목."),
    _p("463640", "KODEX 미국S&P500유틸리티", "주식", [("XLU", 1)], "unhedged", 0.25, "2023-08-01",
       note="S&P 500 유틸리티 섹터. 방어 섹터(배당·낮은 변동성)."),
    _p("453630", "KODEX 미국S&P500필수소비재", "주식", [("XLP", 1)], "unhedged", 0.25, "2023-03-21",
       note="S&P 500 필수소비재 섹터. 방어 섹터. 대용 XLP는 이 연구실 캐시에 2016년 이후만 있을 수 있음."),
    _p("453640", "KODEX 미국S&P500헬스케어", "주식", [("XLV", 1)], "unhedged", 0.25, "2023-03-21",
       note="S&P 500 헬스케어 섹터. 대용 XLV는 이 연구실 캐시에 2016년 이후만 있을 수 있음."),
    _p("218420", "KODEX 미국S&P500에너지(합성)", "주식", [("XLE", 1)], "unhedged", 0.25, "2015-04-28",
       note="S&P 500 에너지 섹터. 인플레이션 국면(2022) 방패 후보."),
    # 채권
    _p("484790", "KODEX 미국30년국채액티브(H)", "채권", [("TLT", 1)], "hedged", 0.015, "2024-06-18",
       note="미국 국채 20년+ 지수(TR) 환헤지."),
    _p("0091C0", "KODEX 미국10년국채액티브(H)", "채권", [("IEF", 1)], "hedged", 0.15, "2025-08-19",
       note="미국 국채 10년 지수(TR) 환헤지. 대용 IEF는 7–10년 국채라 듀레이션이 조금 짧음."),
    _p("468370", "KODEX iShares미국인플레이션국채액티브", "채권", [("TIP", 1)], "unhedged", 0.10, "2023-10-19",
       note="미국 물가연동국채(TIPS) 환노출. 대용 TIP(2003-12 이후)."),
    _p("468630", "KODEX iShares미국투자등급회사채액티브", "채권", [("LQD", 1)], "unhedged", 0.10, "2023-10-19",
       note="미국 투자등급 회사채 환노출. 대용 LQD(2002-07 이후)."),
    # 금·실물
    _p("411060", "ACE KRX금현물", "금·실물", [("GLD", 1)], "unhedged", 0.19, "2021-12-15",
       note="KRX 금시장 가격(원화)을 따름 = 국제 금값 × 환율 근사. 국내 금 프리미엄(2025년 한때 +10% 이상)은 반영 안 됨. 다른 운용사의 KRX금현물 ETF도 같은 방식."),
    _p("132030", "KODEX 골드선물(H)", "금·실물", [("GLD", 1)], "hedged", 0.68, "2010-10-01"),
    _p("352560", "KODEX 미국부동산리츠(H)", "금·실물", [("IYR", 1)], "hedged", 0.09, "2020-05-13"),
    _p("182480", "TIGER 미국MSCI리츠(합성 H)", "금·실물", [("IYR", 1)], "hedged", 0.24, "2013-10-10",
       note="MSCI 미국 리츠 지수 환헤지. 352560보다 상장이 7년 빠르고 보수는 높음. 대용은 같은 IYR."),
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
class PresetRule:
    """A preset's registered signal beyond its weights (allocation DECISIONS A38): an overlay of `kind` with `param`,
    the product the switched-off weight goes to, and whether only the equity products are switched. A preset with a
    rule decides monthly (the engine re-weights monthly whenever an overlay is set)."""
    kind: str = "sma"
    param: float = 10.0
    safe: str = "459580"  # KODEX CD금리액티브(합성)
    equity_only: bool = True

    def describe(self) -> str:
        what = "주식 상품만" if self.equity_only else "모든 위험자산"
        return f"{what} {self.param:g}개월 이동평균 필터 → {short(self.safe)} · 매월 결정"


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    weights: dict[str, float]  # product code -> percent, sums to 100
    note: str
    rule: PresetRule | None = None  # None = static, quarterly (every preset registered before A38)

    def describe_rule(self) -> str:
        return "정적 · 매분기 리밸런싱" if self.rule is None else self.rule.describe()


def equity_codes(codes) -> tuple[str, ...]:
    """The equity products among `codes`, in their order (the A38 '주식만' filter switches these)."""
    return tuple(c for c in codes if c in PRODUCTS and PRODUCTS[c].asset_class == "주식")


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
    Preset("stable40", "안정형 40 (금·현금 분산)",
           {"379800": 25, "251350": 8, "291890": 6, "484790": 12, "0091C0": 13, "411060": 20, "459580": 16},
           "주식 39(S&P500 25 · 선진국 8 · EM 6) · 미국30년(H) 12 · 미국10년(H) 13 · 금현물 20 · CD금리 16. "
           "2026-10-24 넓은 탐색(A23)에서 '안정형 45 70 + 영구 30' 믹스를 정적 비중으로 옮긴 것. 35와 45의 중간(결과를 본 뒤 등록)."),
    Preset("stable45_gold26", "안정형 45 · 금 26 변형",
           {"379800": 23, "251350": 11, "291890": 7, "484790": 6, "0091C0": 16, "411060": 26, "459580": 11},
           "주식 41(S&P500 23 · 선진국 11 · EM 7) · 미국30년(H) 6 · 미국10년(H) 16 · 금현물 26 · CD금리 11. "
           "2026-10-24 넓은 탐색(A23)의 '안정형 45 90 + 금현물 10' 믹스를 정적 비중으로 옮긴 것. 수익을 더 원할 때의 선택지이며, "
           "금 비중이 경고(20% 초과 비권장)를 넘는다(결과를 본 뒤 등록)."),
    # Registered after the global search and its checks had been run (A31 search, A32 stage 2, A35 real-price re-run; DECISIONS A36).
    # Each is a post-hoc pick from the search; the user asked for all three (2026-10-26 18:00 KST).
    Preset("gs_b2", "탐색 B2 (나스닥·방어 섹터·골드선물)",
           {"379810": 19, "453630": 19, "453640": 6, "0091C0": 10, "484790": 6, "468630": 5, "132030": 35},
           "나스닥100 19 · S&P500 필수소비재 19 · 헬스케어 6 · 미국10년(H) 10 · 미국30년(H) 6 · 투자등급회사채 5 · 골드선물(H) 35. "
           "전역 탐색(A31)의 MDD (−14,−13] 구간에서 레버리지·가상 상품 없이 CAGR이 가장 높았던 배분. 2단계 MDD 시험(A32)을 "
           "모형과 실제 가격(A35) 모두에서 통과(결과를 본 뒤 등록). 2005년 이후에만 확인됨: 미국 대용 자료로 2000-09부터 돌린 닷컴 시험(A37, "
           "2005년 이전은 모형)에서 MDD −16.4%(최저 2002-07)."),
    Preset("gs_b4", "탐색 B4 (나스닥·CD금리)",
           {"379810": 30, "453630": 11, "468630": 8, "411060": 15, "459580": 36},
           "나스닥100 30 · S&P500 필수소비재 11 · 투자등급회사채 8 · 금현물 15 · CD금리 36. "
           "전역 탐색(A31)의 MDD (−12,−11] 구간에서 레버리지·가상 상품 없이 CAGR이 가장 높았던 배분. 2단계 시험과 ±5%p 이웃까지 "
           "실제 가격에서 모두 −15% 안(결과를 본 뒤 등록). 2005년 이후에만 확인됨: 닷컴 시험(A37, 2000-09부터, 2005년 이전은 모형)에서 "
           "MDD −25.6%(최저 2002-07)."),
    Preset("gs_a", "탐색 A (헬스케어·나스닥 2배, 미국10년(H))",
           {"453640": 35, "0091C0": 22, "468370": 6, "411060": 14, "132030": 11, "409820": 12},
           "S&P500 헬스케어 35 · 미국10년(H) 22 · 물가채 6 · 금현물 14 · 골드선물(H) 11 · 나스닥100 2배(H) 12. "
           "전역 탐색(A31) 전체 최고 A에서 살 수 없는 가상 상품 ‘미국10년 환노출’ 8을 미국10년(H)에 합친 것(2단계 ‘가상→(H)’ 시험과 같은 배분). "
           "실제 가격 우선 12.06% / −14.48%이지만 리밸런싱 밴드 3%p에서 −15.61%로 선을 넘고, ±5%p 이웃의 53%만 −15% 안이다. "
           "2배 상품 12% 포함. 세 탐색 프리셋 중 가장 덜 튼튼하다(결과를 본 뒤 등록). 2005년 이후에만 확인됨: 닷컴 시험(A37, 2000-09부터, "
           "2005년 이전은 모형)에서 MDD −21.6%(최저 2002-07)."),
    # Registered after the A38 trend-filter test had been run (DECISIONS A38): the same weights as B2 · B3 · B4 with the
    # equity products switched to CD by a 10-month moving-average filter. The static presets above stay as they are.
    Preset("gs_b2_trend", "탐색 B2 + 주식 추세 필터",
           {"379810": 19, "453630": 19, "453640": 6, "0091C0": 10, "484790": 6, "468630": 5, "132030": 35},
           "탐색 B2와 같은 비중(나스닥100 19 · S&P500 필수소비재 19 · 헬스케어 6 · 미국10년(H) 10 · 미국30년(H) 6 · 투자등급회사채 5 · "
           "골드선물(H) 35). 매월 말 세 주식 상품의 원화 종가가 최근 10개 월말 평균보다 낮으면 그 비중을 다음 한 달 CD금리로 옮긴다. "
           "A38(실제 가격 우선, 편도 10bp): 2005–2026 8.9% / −13.0%, 2000-09–2026(2005년 이전 모형) 8.9% / −13.0%. 필터 8·12개월과 "
           "결정일 이동에서도 −15% 안. 정적 탐색 B2보다 2005–2026 연 1.8%p 낮다. 2단계 흔들기(밴드·비용·시작 연도) 전(결과를 본 뒤 등록).",
           PresetRule()),
    Preset("gs_b3_trend", "탐색 B3 + 주식 추세 필터",
           {"379810": 33, "251350": 12, "0091C0": 10, "468370": 7, "411060": 12, "459580": 20, "261240": 6},
           "B3 비중(나스닥100 33 · MSCI선진국 12 · 미국10년(H) 10 · 물가채 7 · 금현물 12 · CD금리 20 · 달러선물 6, 정적 판은 등록하지 않음). "
           "매월 말 나스닥100·MSCI선진국의 원화 종가가 최근 10개 월말 평균보다 낮으면 그 비중을 다음 한 달 CD금리로 옮긴다. "
           "A38: 2005–2026 8.2% / −13.4%, 2000-09–2026 7.7% / −13.4%, 결정일을 옮기면 최악 −14.1%. 정적 B3보다 2005–2026 연 1.8%p 낮다. "
           "2단계 흔들기 전(결과를 본 뒤 등록).",
           PresetRule()),
    Preset("gs_b4_trend", "탐색 B4 + 주식 추세 필터",
           {"379810": 30, "453630": 11, "468630": 8, "411060": 15, "459580": 36},
           "탐색 B4와 같은 비중(나스닥100 30 · S&P500 필수소비재 11 · 투자등급회사채 8 · 금현물 15 · CD금리 36). 매월 말 나스닥100·필수소비재의 "
           "원화 종가가 최근 10개 월말 평균보다 낮으면 그 비중을 다음 한 달 CD금리로 옮긴다. A38: 2005–2026 8.0% / −11.4%, "
           "2000-09–2026 7.7% / −11.4%. 정적 탐색 B4보다 2005–2026 연 1.6%p 낮다. 2단계 흔들기 전(결과를 본 뒤 등록).",
           PresetRule()),
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
