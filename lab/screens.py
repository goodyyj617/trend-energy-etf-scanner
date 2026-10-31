"""Stock screens: what may be bought (DECISIONS 77; practitioner templates added 2026-10-25).

Stages, all point-in-time and all swappable:
  0. hygiene            - tradability: history, price, dollar volume, Amihud illiquidity, size
                          (HYGIENE_SCREENS, True = excluded);
  1. exclusion screens  - "never buy this now": trend-template failure, over-extension, weak
                          industry, lottery, falling knife ... (EXCLUDE_SCREENS, True = excluded);
  2. selection scores   - "which of the rest are best": each returns a score, higher = better;
                          several scores are combined by their average percentile rank, and only
                          the top K symbols stay buyable (SELECT_SCORES; STYLE_TILTS are the
                          older single-factor tilts, kept for comparison);
  3. entry timing       - the existing entry blocks (lab/blocks.py), or none.
PROFILES are ready-made combinations that mirror what traders publish (Minervini's trend template,
IBD/CANSLIM price-volume rules, quality momentum, industry leaders ...). Each is one ScreenConfig.
Stages 1-2 are evaluated on the last trading day of each month from data up to that day, and
hold for the following month. `monthly_mask` turns them into the engine's Panel.eligible (who may
be bought on each day) and Panel.rank (buying order, 1 = best), so no engine change is needed.

Every screen has at most one numeric parameter. Inputs are prices, volume and, when the panel
carries them (lab/crsp.py), market value `panel.extra["cap"]` ($M) and daily dividend yield
`panel.extra["dividend"]`. True valuation ratios need accounting data, which CRSP does not have.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .blocks import Param, adx_di, atr, cmf, obv, rsi, sma
from .data import BENCHMARK, Panel

MONTH = 21  # trading days


@dataclass(frozen=True)
class Screen:
    key: str
    label: str
    rule: str
    help: str
    param: Param | None
    compute: Callable[[Panel, float | None], pd.DataFrame]
    family: str = "가격"  # 가격 / 거래량 / 규모 / 배당
    tier: int = 1  # 1 = price & volume directly, 2 = first derivatives (ATR, slopes, indicators), 3 = size, dividend, industry


# --------------------------------------------------------------------------- helpers

def _months(n: float) -> int:
    return int(round(float(n) * MONTH))


def _ret(close: pd.DataFrame, months: float, skip_months: float = 0.0) -> pd.DataFrame:
    """Return over the last `months`, ending `skip_months` ago (12-1 momentum: months=12, skip=1)."""
    a, b = _months(months), _months(skip_months)
    return close.shift(b) / close.shift(a) - 1.0


def _vol(close: pd.DataFrame, days: int) -> pd.DataFrame:
    return close.pct_change(fill_method=None).rolling(days, min_periods=days).std()


def _cap(p: Panel) -> pd.DataFrame:
    if "cap" not in p.extra:
        raise ValueError("이 데이터에는 시가총액이 없습니다 (CRSP 패널에서만 쓸 수 있습니다).")
    return p.extra["cap"]


def _dividend(p: Panel) -> pd.DataFrame:
    if "dividend" not in p.extra:
        raise ValueError("이 데이터에는 배당 수익률이 없습니다 (CRSP 패널에서만 쓸 수 있습니다).")
    return p.extra["dividend"]


def _top_share(score: pd.DataFrame, share_pct: float) -> pd.DataFrame:
    """True where the symbol is in the top `share_pct` percent of the cross-section that day."""
    pct = score.rank(axis=1, pct=True)
    return pct > 1.0 - share_pct / 100.0


def _industry(p: Panel) -> pd.Series:
    """2-digit SIC industry group per symbol (needs panel.info["sic"], CRSP panels)."""
    if "sic" not in p.info.columns:
        raise ValueError("이 데이터에는 업종 코드(SIC)가 없습니다 (CRSP 패널에서만 쓸 수 있습니다).")
    sic = pd.to_numeric(p.info.set_index("symbol")["sic"], errors="coerce")
    return (sic // 100).reindex(p.symbols)


def _industry_mean(p: Panel, frame: pd.DataFrame) -> pd.DataFrame:
    """Each symbol's column replaced by the mean of its industry group (same shape)."""
    grp = _industry(p)
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for g, cols in grp.dropna().groupby(grp.dropna()).groups.items():
        cols = [c for c in cols if c in frame.columns]
        if cols:
            out.loc[:, cols] = np.repeat(frame[cols].mean(axis=1).to_numpy()[:, None], len(cols), axis=1)
    return out


def _rs_rating(p: Panel) -> pd.DataFrame:
    """IBD-style relative strength: 40% weight on the last 3 months, 20% each on the three
    quarters before, then the percentile rank across the universe that day."""
    c = p.close
    q = [c / c.shift(63 * k) - 1.0 for k in (1, 2, 3, 4)]
    score = 0.4 * q[0] + 0.2 * (q[1] - q[0]) + 0.2 * (q[2] - q[1]) + 0.2 * (q[3] - q[2])
    return score.rank(axis=1, pct=True) * 100.0


def _amihud(p: Panel, days: int = 63) -> pd.DataFrame:
    """Amihud illiquidity: mean |return| / dollar volume (higher = more illiquid)."""
    r = p.close.pct_change(fill_method=None).abs()
    dv = (p.close * p.volume).replace(0, np.nan)
    return (r / dv).rolling(days, min_periods=days // 2).mean()


# --------------------------------------------------------------------------- 0. hygiene (True = not tradable)

def _h_history(p: Panel, m: float) -> pd.DataFrame:
    n = _months(m)
    return p.close.notna().astype(float).rolling(n, min_periods=1).sum() < n * 0.9


def _h_price(p: Panel, x: float) -> pd.DataFrame:
    return p.close < x


def _h_liquidity(p: Panel, x: float) -> pd.DataFrame:
    return (p.close * p.volume).rolling(MONTH, min_periods=MONTH).mean() < x * 1e6


def _h_illiquid(p: Panel, x: float) -> pd.DataFrame:
    return _top_share(_amihud(p), x)


HYGIENE_SCREENS: dict[str, Screen] = {
    s.key: s for s in [
        Screen("h_history", "상장 기간", "최근 M개월 중 가격이 있는 날이 90% 미만",
               "신규 상장·거래 공백 종목을 뺍니다. 지표가 안정되려면 1년 이상의 가격이 필요합니다.",
               Param("M (개월)", 12, 6, 36, 6, "개월", "필요한 가격 이력"), _h_history),
        Screen("h_price", "최소 주가", "종가 < X 달러", "저가주는 호가 차이와 거래 비용이 큽니다. 실무에서는 5–15달러 하한을 둡니다.",
               Param("X ($)", 10, 1, 30, 1, "$", "주가 하한"), _h_price),
        Screen("h_liquidity", "최소 거래대금", "21거래일 평균 거래대금 < X 백만 달러",
               "계좌 규모에 비해 거래가 적은 종목은 체결이 어렵습니다. 포지션의 50–100배 거래대금을 권합니다.",
               Param("X ($M)", 2, 0.5, 50, 0.5, "$M", "일평균 거래대금 하한"), _h_liquidity, "거래량"),
        Screen("h_illiquid", "유동성 하위 (Amihud)", "63일 Amihud 비유동성(|수익률| ÷ 거래대금)이 상위 X%",
               "거래대금 대비 가격 충격이 큰 종목을 뺍니다(Amihud 2002). 거래대금 하한이 놓치는 종목을 잡습니다.",
               Param("X (%)", 20, 5, 50, 5, "%", "제외할 상위 비율"), _h_illiquid, "거래량"),
        Screen("h_cap", "시가총액 하한", "시가총액 < X 백만 달러", "너무 작은 회사는 뺍니다. 소형주를 넣을지는 이 값으로 조절합니다.",
               Param("X ($M)", 300, 50, 5000, 50, "$M", "시가총액 하한"), lambda p, x: _cap(p) < x, "규모"),
    ]
}


# --------------------------------------------------------------------------- 1. exclusion (True = do not buy)

def _tt_fail(p: Panel, x: float) -> pd.DataFrame:
    """Minervini's trend template, as a failure flag. All of: close > SMA50 > SMA150 > SMA200,
    SMA200 higher than a month ago, close >= 1.3 x 52-week low, close >= (1 - X%) x 52-week high."""
    c = p.close
    m50, m150, m200 = sma(c, 50), sma(c, 150), sma(c, 200)
    hi, lo = c.rolling(252, min_periods=200).max(), c.rolling(252, min_periods=200).min()
    ok = (c > m50) & (m50 > m150) & (m150 > m200) & (m200 > m200.shift(MONTH)) & (c >= 1.3 * lo) & (c >= hi * (1 - x / 100.0))
    return ~ok.where(c.notna() & m200.notna(), False)


def _x_extended(p: Panel, x: float) -> pd.DataFrame:
    return p.close > sma(p.close, 50) * (1 + x / 100.0)


def _x_weak_industry(p: Panel, x: float) -> pd.DataFrame:
    ind = _industry_mean(p, _ret(p.close, 6, 1))
    return ind.rank(axis=1, pct=True) < x / 100.0


def _ivol(p: Panel) -> pd.DataFrame:
    """Idiosyncratic volatility: std of the residual of daily returns on SPY over 63 days."""
    r = p.close.pct_change(fill_method=None)
    m = r[BENCHMARK]
    cov = r.rolling(63, min_periods=40).cov(m)
    beta = cov.div(m.rolling(63, min_periods=40).var(), axis=0)
    resid = r - beta * m.to_numpy()[:, None]
    return resid.rolling(63, min_periods=40).std()


def _x_ivol(p: Panel, x: float) -> pd.DataFrame:
    """Idiosyncratic volatility: residual of daily returns on SPY over 63 days, top X% excluded."""
    return _top_share(_ivol(p), x)

def _x_downtrend(p: Panel, x: float) -> pd.DataFrame:
    return _ret(p.close, 12, 1) < x / 100.0


def _x_drawdown(p: Panel, x: float) -> pd.DataFrame:
    hi = p.close.rolling(252, min_periods=126).max()
    return p.close < hi * (1 - x / 100.0)


def _x_volatility(p: Panel, x: float) -> pd.DataFrame:
    return _top_share(_vol(p.close, 60), x)


def _x_lottery(p: Panel, x: float) -> pd.DataFrame:
    return p.close.pct_change(fill_method=None).rolling(MONTH, min_periods=MONTH).max() >= x / 100.0


def _x_overextended(p: Panel, x: float) -> pd.DataFrame:
    return _top_share(_ret(p.close, 36), x)


def _x_small(p: Panel, x: float) -> pd.DataFrame:
    return _cap(p) < x


def _x_thin(p: Panel, x: float) -> pd.DataFrame:
    dv = (p.close * p.volume).rolling(MONTH, min_periods=MONTH).mean()
    return dv < x * 1e6


EXCLUDE_SCREENS: dict[str, Screen] = {
    s.key: s for s in [
        Screen("tt_fail", "추세 템플릿 미충족 (Minervini)",
               "종가 > SMA50 > SMA150 > SMA200, SMA200이 1개월 전보다 위, 종가 ≥ 52주 저가 × 1.3, 종가 ≥ 52주 고가 × (1 − X%) 중 하나라도 어긋남",
               "Mark Minervini의 '2단계 상승 추세' 조건입니다(SEPA). 추세추종 실무에서 가장 널리 쓰는 종목 선별 틀로, 이 조건을 다 만족하는 종목만 삽니다.",
               Param("X (%)", 25, 10, 40, 5, "%", "52주 고가에서 허용하는 거리"), _tt_fail),
        Screen("x_extended", "단기 과열 (50일선 위로 과도)", "종가 > SMA50 × (1 + X%)",
               "50일선에서 너무 멀리 떨어진 종목은 쫓아 사지 않습니다. 눌림 뒤 진입을 기다리는 실무 규칙입니다.",
               Param("X (%)", 25, 10, 50, 5, "%", "50일선 대비 허용 괴리"), _x_extended),
        Screen("x_weak_industry", "약한 업종", "업종(2자리 SIC) 평균 6-1개월 수익률이 하위 X%",
               "IBD식 업종 순위. 주도 업종이 아닌 종목은 뺍니다(Moskowitz & Grinblatt 1999 업종 모멘텀).",
               Param("X (%)", 30, 10, 60, 10, "%", "제외할 하위 업종 비율"), _x_weak_industry),
        Screen("x_ivol", "고유 변동성 상위", "SPY 대비 잔차 변동성(63일)이 상위 X%",
               "시장과 무관하게 크게 흔들리는 종목(Ang 등 2006의 고유 변동성 효과)을 뺍니다.",
               Param("X (%)", 20, 5, 50, 5, "%", "제외할 상위 비율"), _x_ivol),
        Screen("x_downtrend", "12개월 하락 추세", "지난 12개월(최근 1개월 제외) 수익률 < X%",
               "중기 추세가 아래인 종목은 사지 않습니다. 횡단면 모멘텀(Jegadeesh & Titman 1993)의 반대쪽을 걸러냅니다.",
               Param("X (%)", 0, -30, 30, 5, "%", "12-1개월 수익률 하한"), _x_downtrend),
        Screen("x_drawdown", "고점 대비 큰 하락", "종가 < 52주 최고 종가 × (1 − X%)",
               "떨어지는 칼은 잡지 않습니다. 52주 고가에서 X% 넘게 내려온 종목을 제외합니다.",
               Param("X (%)", 30, 10, 60, 5, "%", "52주 고가 대비 허용 하락폭"), _x_drawdown),
        Screen("x_volatility", "변동성 상위", "60일 변동성이 그날 유니버스 상위 X%",
               "변동성이 가장 큰 종목은 손절에 자주 걸리고 복권형 수익(Bali·Cakici·Whitelaw 2011)과 겹칩니다.",
               Param("X (%)", 20, 5, 50, 5, "%", "제외할 상위 비율"), _x_volatility),
        Screen("x_lottery", "최근 하루 급등", "지난 21거래일 중 하루 수익률 최댓값 ≥ X%",
               "한 달 안에 하루 X% 이상 급등한 종목(복권형)은 그 뒤 평균 수익이 낮습니다(MAX 효과).",
               Param("X (%)", 20, 10, 50, 5, "%", "하루 급등 기준"), _x_lottery),
        Screen("x_overextended", "36개월 과열 (장기 반전)", "지난 36개월 수익률이 그날 유니버스 상위 X%",
               "3년 넘게 가장 많이 오른 종목은 평균적으로 되돌림이 큽니다(De Bondt & Thaler 1985). 밸류에이션의 가격 대용입니다.",
               Param("X (%)", 10, 5, 30, 5, "%", "제외할 상위 비율"), _x_overextended),
        Screen("x_small", "시가총액 하한", "시가총액 < X 백만 달러",
               "너무 작은 회사는 제외합니다. 소형주를 넣을지는 이 값으로 조절합니다.",
               Param("X ($M)", 300, 50, 5000, 50, "$M", "시가총액 하한"), _x_small, "규모"),
        Screen("x_thin", "거래대금 하한", "21거래일 평균 거래대금 < X 백만 달러",
               "실제로 사고팔 수 있을 만큼 거래되는 종목만 남깁니다.",
               Param("X ($M)", 1, 0.5, 50, 0.5, "$M", "일평균 거래대금 하한"), _x_thin, "거래량"),
    ]
}


# --------------------------------------------------------------------------- 2. selection scores (higher = better)

def _s_momentum(p: Panel, m: float) -> pd.DataFrame:
    return _ret(p.close, m, 1)


def _s_high52(p: Panel, _: float | None) -> pd.DataFrame:
    return p.close / p.close.rolling(252, min_periods=126).max()


def _s_smooth(p: Panel, m: float) -> pd.DataFrame:
    n = _months(m)
    up = (p.close.pct_change(fill_method=None) > 0).astype(float).where(p.close.notna())
    return up.rolling(n, min_periods=n).mean()


def _s_quality(p: Panel, m: float) -> pd.DataFrame:
    return _ret(p.close, m, 1) / _vol(p.close, _months(m)).replace(0, np.nan)


def _s_volume_trend(p: Panel, m: float) -> pd.DataFrame:
    dv = p.close * p.volume
    return dv.rolling(MONTH, min_periods=MONTH).mean() / dv.rolling(_months(m), min_periods=_months(m)).mean().replace(0, np.nan)


def _s_small(p: Panel, _: float | None) -> pd.DataFrame:
    return -np.log(_cap(p).where(_cap(p) > 0))


def _s_large(p: Panel, _: float | None) -> pd.DataFrame:
    return np.log(_cap(p).where(_cap(p) > 0))


def _s_dividend(p: Panel, m: float) -> pd.DataFrame:
    n = _months(m)
    return _dividend(p).rolling(n, min_periods=n).sum()


def _s_reversal(p: Panel, m: float) -> pd.DataFrame:
    return -_ret(p.close, m)


def _s_lowvol(p: Panel, _: float | None) -> pd.DataFrame:
    return -_vol(p.close, 60)


def _s_rs(p: Panel, _: float | None) -> pd.DataFrame:
    return _rs_rating(p)


def _s_accum(p: Panel, m: float) -> pd.DataFrame:
    """Accumulation/distribution: dollar volume on up days over dollar volume on down days."""
    n = _months(m)
    r = p.close.pct_change(fill_method=None)
    dv = p.close * p.volume
    up = dv.where(r > 0, 0.0).rolling(n, min_periods=n).sum()
    down = dv.where(r < 0, 0.0).rolling(n, min_periods=n).sum()
    return up / down.replace(0, np.nan)


def _s_industry_rs(p: Panel, m: float) -> pd.DataFrame:
    return _industry_mean(p, _ret(p.close, m, 1))


def _s_high52_vol(p: Panel, _: float | None) -> pd.DataFrame:
    """52-week-high proximity, with volume confirmation: proximity x (20d / 120d dollar volume)."""
    dv = p.close * p.volume
    conf = dv.rolling(MONTH, min_periods=MONTH).mean() / dv.rolling(120, min_periods=120).mean().replace(0, np.nan)
    return _s_high52(p, None) * conf.clip(upper=3.0)


SELECT_SCORES: dict[str, Screen] = {
    s.key: s for s in [
        Screen("s_rs", "상대강도 등급 (IBD식)", "최근 3개월 수익률 40% + 그 전 세 분기 각 20% 가중 → 유니버스 백분위(0–100)",
               "IBD RS Rating의 공개된 계산 방식입니다. 최근 분기에 무게를 두어 지금 강한 종목을 앞세웁니다. 파라미터 없음.",
               None, _s_rs),
        Screen("s_accum", "매집/분산 비율", "지난 M개월 오른 날 거래대금 합 ÷ 내린 날 거래대금 합",
               "오를 때 거래가 많고 내릴 때 적으면 기관 매집으로 봅니다(IBD Accumulation/Distribution). ",
               Param("M (개월)", 3, 1, 6, 1, "개월", "합산 기간"), _s_accum, "거래량"),
        Screen("s_industry_rs", "업종 상대강도", "업종(2자리 SIC) 평균 M-1개월 수익률",
               "주도 업종에 속한 종목을 앞세웁니다. 종목 모멘텀과 별개로 업종 모멘텀이 있습니다(Moskowitz & Grinblatt 1999).",
               Param("M (개월)", 6, 3, 12, 3, "개월", "기간"), _s_industry_rs),
        Screen("s_high52_vol", "52주 고가 근접 × 거래량 확인", "(종가 ÷ 52주 고가) × (20일 거래대금 ÷ 120일 거래대금, 최대 3)",
               "고가 근처이면서 거래가 늘어난 종목. 돌파를 거래량이 받쳐 주는지 봅니다(O'Neil).", None, _s_high52_vol, "거래량"),
        Screen("s_momentum", "모멘텀 (M-1개월 수익률)", "지난 M개월 수익률, 최근 1개월 제외",
               "가장 많이 오른 종목이 계속 오르는 경향(횡단면 모멘텀). 최근 1개월은 단기 반전 때문에 뺍니다.",
               Param("M (개월)", 12, 3, 12, 3, "개월", "수익률을 재는 기간"), _s_momentum),
        Screen("s_high52", "52주 고가 근접", "종가 ÷ 52주 최고 종가",
               "52주 고가에 가까울수록 높은 점수(George & Hwang 2004). 파라미터 없음.", None, _s_high52),
        Screen("s_smooth", "모멘텀 매끄러움", "지난 M개월 중 오른 날의 비율",
               "같은 수익이라도 조금씩 꾸준히 오른 종목이 더 오래 갑니다(Da·Gurun·Warachka 2014).",
               Param("M (개월)", 12, 3, 12, 3, "개월", "비율을 재는 기간"), _s_smooth),
        Screen("s_quality", "변동성 대비 모멘텀", "M-1개월 수익률 ÷ M개월 변동성",
               "흔들림 대비 얼마나 올랐나. 변동성이 큰 종목의 모멘텀을 깎아 줍니다.",
               Param("M (개월)", 12, 3, 12, 3, "개월", "기간"), _s_quality),
        Screen("s_volume_trend", "거래량 증가 추세", "최근 21일 평균 거래대금 ÷ 최근 M개월 평균 거래대금",
               "관심이 늘어나는 종목(거래대금이 평소보다 많은 종목)을 앞세웁니다.",
               Param("M (개월)", 6, 2, 12, 1, "개월", "비교 기간"), _s_volume_trend, "거래량"),
        Screen("s_small", "소형주 우선", "−log(시가총액)", "작은 회사를 앞세웁니다(소형주 효과). 파라미터 없음.", None, _s_small, "규모"),
        Screen("s_large", "대형주 우선", "log(시가총액)", "큰 회사를 앞세웁니다. 파라미터 없음.", None, _s_large, "규모"),
        Screen("s_dividend", "배당수익률", "지난 M개월 배당 수익률 합",
               "CRSP의 총수익률과 가격수익률 차이에서 계산한 배당. 밸류에이션의 가격 대용 중 하나입니다.",
               Param("M (개월)", 12, 6, 24, 6, "개월", "합산 기간"), _s_dividend, "배당"),
        Screen("s_reversal", "장기 반전 (M개월 수익률 낮은 순)", "−(지난 M개월 수익률)",
               "오래 못 오른 종목이 되돌아오는 경향(De Bondt & Thaler 1985). 밸류 대용입니다.",
               Param("M (개월)", 36, 24, 60, 12, "개월", "기간"), _s_reversal),
        Screen("s_lowvol", "저변동성 우선", "−(60일 변동성)", "덜 흔들리는 종목을 앞세웁니다. 파라미터 없음.", None, _s_lowvol),
    ]
}

# --------------------------------------------------------------------------- 2b. more selection scores
# Priority set by the user (2026-10-25): tier 1 uses open/high/low/close/volume directly; tier 2 the
# first derivatives of them (ATR, moving-average slopes, regression, RSI, ADX, CMF); tier 3 size,
# dividend and industry. Each score has one lookback parameter M (months) or none; the representative
# default is fixed here and other values are for the robustness tests.

def _n(m: float) -> int:
    return _months(m)


def _dd(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.rolling(n, min_periods=n // 2).max() - 1.0


def _s_mom_short(p: Panel, m: float) -> pd.DataFrame:
    return _ret(p.close, m)


def _s_rs_spy(p: Panel, m: float) -> pd.DataFrame:
    r = _ret(p.close, m, 1)
    return r.sub(r[BENCHMARK], axis=0)


def _s_rs_line_high(p: Panel, _: float | None) -> pd.DataFrame:
    line = p.close.div(p.close[BENCHMARK], axis=0)
    return line / line.rolling(252, min_periods=126).max()


def _s_low52_dist(p: Panel, _: float | None) -> pd.DataFrame:
    return p.close / p.close.rolling(252, min_periods=126).min() - 1.0


def _s_new_highs(p: Panel, m: float) -> pd.DataFrame:
    hi = p.close.rolling(252, min_periods=200).max()
    is_high = ((p.close >= hi) & p.close.notna()).astype(float)
    return is_high.rolling(_n(m), min_periods=_n(m)).sum()


def _s_clv(p: Panel, m: float) -> pd.DataFrame:
    rng = (p.high - p.low).replace(0, np.nan)
    clv = (((p.close - p.low) - (p.high - p.close)) / rng).fillna(0.0).where(p.close.notna())  # zero range = neutral
    return clv.rolling(_n(m), min_periods=_n(m) // 2).mean()


def _s_overnight(p: Panel, m: float) -> pd.DataFrame:
    gap = p.open / p.close.shift(1) - 1.0
    return gap.rolling(_n(m), min_periods=_n(m) // 2).sum()


def _s_obv_slope(p: Panel, m: float) -> pd.DataFrame:
    n = _n(m)
    o = obv(p)
    return (o - o.shift(n)) / p.volume.rolling(n, min_periods=n).sum().replace(0, np.nan)


def _s_dollar_volume(p: Panel, _: float | None) -> pd.DataFrame:
    dv = (p.close * p.volume).rolling(63, min_periods=63).mean()
    return np.log(dv.where(dv > 0))


def _s_range_contraction(p: Panel, m: float) -> pd.DataFrame:
    rng = (p.high - p.low) / p.close
    return rng.rolling(_n(m), min_periods=_n(m)).mean() / rng.rolling(MONTH, min_periods=MONTH).mean().replace(0, np.nan)


def _s_consistency(p: Panel, m: float) -> pd.DataFrame:
    monthly = (p.close / p.close.shift(MONTH) - 1.0 > 0).astype(float).where(p.close.notna())
    return monthly.rolling(_n(m), min_periods=_n(m)).mean()


def _s_dd_shallow(p: Panel, m: float) -> pd.DataFrame:
    n = _n(m)
    return _dd(p.close, n).rolling(n, min_periods=n // 2).min()


def _s_fresh_high(p: Panel, _: float | None) -> pd.DataFrame:
    c = p.close
    is_high = (c >= c.rolling(252, min_periods=126).max()) & c.notna()
    idx = pd.DataFrame(np.repeat(np.arange(len(c))[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    last = idx.where(is_high).ffill()
    since = (idx - last).fillna(idx)  # never made a 52-week high: count from the start of the data
    return -since.where(c.notna())


def _s_atr_momentum(p: Panel, m: float) -> pd.DataFrame:
    return _ret(p.close, m) / (atr(p, 14) / p.close).replace(0, np.nan)


def _s_atr_squeeze(p: Panel, _: float | None) -> pd.DataFrame:
    return -(atr(p, 20) / atr(p, 120).replace(0, np.nan))


def _s_bb_squeeze(p: Panel, m: float) -> pd.DataFrame:
    width = p.close.rolling(20, min_periods=20).std(ddof=0) / sma(p.close, 20)
    return -(width / width.rolling(_n(m), min_periods=_n(m) // 2).max().replace(0, np.nan))


def _s_adx(p: Panel, _: float | None) -> pd.DataFrame:
    adx, plus, minus = adx_di(p, 14)
    return adx * np.sign(plus - minus)


def _s_efficiency(p: Panel, m: float) -> pd.DataFrame:
    n = _n(m)
    path = p.close.diff().abs().rolling(n, min_periods=n).sum().replace(0, np.nan)
    return (p.close - p.close.shift(n)) / path


def _s_slope_r2(p: Panel, m: float) -> pd.DataFrame:
    """Clenow's momentum: annualised exponential regression slope x R^2 over M months."""
    n = _n(m)
    y = np.log(p.close.where(p.close > 0))
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    ty = y.mul(t, axis=0)
    cov = ty.rolling(n, min_periods=n).mean() - y.rolling(n, min_periods=n).mean().mul(t.rolling(n).mean(), axis=0)
    var_t = (n ** 2 - 1) / 12.0
    var_y = y.rolling(n, min_periods=n).var(ddof=0).replace(0, np.nan)
    slope = cov / var_t
    r2 = cov ** 2 / (var_t * var_y)
    return (np.exp(slope * 252) - 1.0) * r2


def _s_ma_slope(p: Panel, _: float | None) -> pd.DataFrame:
    m50 = sma(p.close, 50)
    return m50 / m50.shift(MONTH) - 1.0


def _s_ma_distance(p: Panel, _: float | None) -> pd.DataFrame:
    return p.close / sma(p.close, 200) - 1.0


def _s_rsi(p: Panel, _: float | None) -> pd.DataFrame:
    return rsi(p.close, 14)


def _s_cmf(p: Panel, m: float) -> pd.DataFrame:
    return cmf(p, _n(m))


def _s_downside(p: Panel, m: float) -> pd.DataFrame:
    r = p.close.pct_change(fill_method=None).clip(upper=0.0)
    return -np.sqrt((r ** 2).rolling(_n(m), min_periods=_n(m)).mean())


def _s_ulcer(p: Panel, m: float) -> pd.DataFrame:
    n = _n(m)
    return -np.sqrt((_dd(p.close, n) ** 2).rolling(n, min_periods=n // 2).mean())


def _s_beta_low(p: Panel, _: float | None) -> pd.DataFrame:
    r = p.close.pct_change(fill_method=None)
    m = r[BENCHMARK]
    return -r.rolling(63, min_periods=40).cov(m).div(m.rolling(63, min_periods=40).var(), axis=0)


def _s_turnover(p: Panel, _: float | None) -> pd.DataFrame:
    """Dollar volume relative to market value (shares traded as a share of the company)."""
    dv = (p.close * p.volume).rolling(63, min_periods=63).mean()
    return dv / (_cap(p) * 1e6).replace(0, np.nan)


_M = lambda default, lo, hi, step=3: Param("M (개월)", default, lo, hi, step, "개월", "계산 기간")  # noqa: E731
SELECT_SCORES.update({
    s.key: s for s in [
        # ---- tier 1: price and volume directly
        Screen("s_mom_short", "단기 모멘텀 (M개월 수익률)", "지난 M개월 수익률 (최근 달 포함)",
               "3개월 안팎의 짧은 추세. 1개월만 보면 단기 반전이 섞이므로 기본 3개월.", _M(3, 1, 6, 1), _s_mom_short),
        Screen("s_rs_spy", "SPY 대비 초과수익", "M-1개월 수익률 − SPY의 같은 기간 수익률",
               "시장보다 얼마나 더 올랐나(상대강도선의 기울기).", _M(6, 3, 12), _s_rs_spy),
        Screen("s_rs_line_high", "상대강도선 신고가 근접", "(종가 ÷ SPY) ÷ 그 비율의 52주 최고값",
               "O'Neil의 RS선 신고가. 시장 대비 비율이 1년 고점에 가까운 종목. 파라미터 없음.", None, _s_rs_line_high),
        Screen("s_low52_dist", "52주 저가 대비 상승폭", "종가 ÷ 52주 최저 종가 − 1",
               "바닥에서 얼마나 올라왔나(Minervini 템플릿의 '저가 대비 30% 이상' 조건의 연속형). 파라미터 없음.", None, _s_low52_dist),
        Screen("s_new_highs", "신고가 횟수", "지난 M개월 중 52주 신고가를 기록한 날 수",
               "계속 새 고가를 쓰는 종목. 한 번 튄 종목과 구분됩니다.", _M(3, 1, 12), _s_new_highs),
        Screen("s_clv", "종가 위치 (일중 범위)", "지난 M개월 평균 [(종가−저가) − (고가−종가)] ÷ (고가−저가)",
               "장중 범위의 위쪽에서 끝나는 날이 많으면 매수세가 강한 것으로 봅니다(−1~+1).", _M(3, 1, 6, 1), _s_clv),
        Screen("s_overnight", "야간 수익 누적", "지난 M개월 (시가 ÷ 전일 종가 − 1)의 합",
               "갭으로 오르는 종목은 뉴스·기관 수요가 있는 경우가 많습니다.", _M(3, 1, 12), _s_overnight),
        Screen("s_obv_slope", "OBV 기울기", "(OBV − M개월 전 OBV) ÷ 같은 기간 거래량 합",
               "오른 날 거래량이 내린 날보다 얼마나 많았나(−1~+1).", _M(3, 1, 12), _s_obv_slope, "거래량"),
        Screen("s_dollar_volume", "거래대금 (유동성)", "log(63일 평균 거래대금)",
               "거래가 활발한 종목 우선. 파라미터 없음.", None, _s_dollar_volume, "거래량"),
        Screen("s_range_contraction", "변동폭 수축 (VCP)", "지난 M개월 평균 일중 범위 ÷ 최근 21일 평균 일중 범위",
               "Minervini의 VCP: 돌파 전 일중 변동폭이 줄어드는 종목. 값이 클수록 최근이 조용함.", _M(6, 3, 12), _s_range_contraction),
        Screen("s_consistency", "월간 승률", "지난 M개월 중 21일 수익률이 양수였던 날의 비율",
               "꾸준히 오르는 종목.", _M(12, 6, 24, 6), _s_consistency),
        Screen("s_dd_shallow", "낙폭 얕음", "−(지난 M개월 최대 낙폭)",
               "같은 기간 덜 빠진 종목 우선.", _M(12, 3, 24), _s_dd_shallow),
        Screen("s_fresh_high", "최근 고가 (경과일 짧음)", "−(52주 고가 이후 지난 거래일 수)",
               "고가를 막 쓴 종목이 앞섭니다. 파라미터 없음.", None, _s_fresh_high),
        # ---- tier 2: first derivatives
        Screen("s_atr_momentum", "ATR 대비 모멘텀", "M개월 수익률 ÷ (ATR14 ÷ 종가)",
               "하루 평균 변동폭 몇 개만큼 올랐나. 변동성이 다른 종목을 같은 잣대로 비교합니다.", _M(6, 3, 12), _s_atr_momentum, "보조지표", 2),
        Screen("s_atr_squeeze", "ATR 수축", "−(ATR20 ÷ ATR120)",
               "최근 변동폭이 장기보다 작아진 종목(에너지 응축). 파라미터 없음.", None, _s_atr_squeeze, "보조지표", 2),
        Screen("s_bb_squeeze", "볼린저 밴드폭 수축", "−(20일 밴드폭 ÷ 지난 M개월 밴드폭 최댓값)",
               "밴드폭이 최근 몇 달 중 가장 좁은 종목(스퀴즈).", _M(6, 3, 12), _s_bb_squeeze, "보조지표", 2),
        Screen("s_adx", "ADX 추세 강도 (방향 포함)", "ADX(14) × 부호(+DI − −DI)",
               "추세가 뚜렷하고 위쪽인 종목. 파라미터 없음.", None, _s_adx, "보조지표", 2),
        Screen("s_efficiency", "추세 효율 (Kaufman)", "(종가 − M개월 전 종가) ÷ Σ|일간 종가 변화|",
               "같은 상승이라도 지그재그가 적을수록 1에 가깝습니다.", _M(6, 3, 12), _s_efficiency, "보조지표", 2),
        Screen("s_slope_r2", "회귀 기울기 × R² (Clenow)", "지난 M개월 log 종가의 지수회귀 기울기(연율) × 결정계수",
               "Andreas Clenow의 모멘텀 순위. 가파르고 매끄러운 상승에 높은 점수.", _M(4, 3, 12, 1), _s_slope_r2, "보조지표", 2),
        Screen("s_ma_slope", "50일선 기울기", "SMA50 ÷ 21일 전 SMA50 − 1", "중기 이동평균이 올라가는 속도. 파라미터 없음.", None, _s_ma_slope, "보조지표", 2),
        Screen("s_ma_distance", "200일선 대비 거리", "종가 ÷ SMA200 − 1", "장기 추세 위로 얼마나 올라와 있나. 파라미터 없음.", None, _s_ma_distance, "보조지표", 2),
        Screen("s_rsi", "RSI(14)", "RSI(14)", "상승 힘의 비율(0–100). 파라미터 없음.", None, _s_rsi, "보조지표", 2),
        Screen("s_cmf", "자금 흐름 (CMF)", "Chaikin Money Flow, M개월", "거래량이 실린 종가 위치. 매집/분산의 또 다른 계산.", _M(3, 1, 6, 1), _s_cmf, "거래량", 2),
        Screen("s_downside", "하방 변동성 낮음", "−√(지난 M개월 음(−) 일간 수익률 제곱 평균)", "내릴 때 덜 내리는 종목.", _M(6, 3, 12), _s_downside, "보조지표", 2),
        Screen("s_ulcer", "낙폭 고통 지수 낮음 (Ulcer)", "−√(지난 M개월 낙폭² 평균)", "낙폭의 깊이와 길이를 함께 봅니다(Martin 1987).", _M(12, 6, 24, 6), _s_ulcer, "보조지표", 2),
        Screen("s_beta_low", "저베타", "−(SPY 대비 63일 베타)", "시장에 덜 끌려다니는 종목(Frazzini & Pedersen 2014). 파라미터 없음.", None, _s_beta_low, "보조지표", 2),
        # ---- tier 3: size, dividend, industry
        Screen("s_turnover", "회전율", "63일 평균 거래대금 ÷ 시가총액", "회사 크기 대비 거래가 활발한 종목. 파라미터 없음.", None, _s_turnover, "규모", 3),
    ]
})
for key, tier in {"s_quality": 2, "s_lowvol": 2, "s_small": 3, "s_large": 3, "s_dividend": 3, "s_reversal": 3, "s_industry_rs": 3}.items():
    from dataclasses import replace as _replace
    SELECT_SCORES[key] = _replace(SELECT_SCORES[key], tier=tier)

# Stage K (DECISIONS 91) features without a score before: added here, kept out of the registered
# menus of score_study and screen_study (STAGE_K_SCORES), so earlier results stay as they were.
SELECT_SCORES.update({
    s.key: s for s in [
        Screen("s_low_price", "저가주 우선", "−log(종가)", "주가가 낮은 종목을 앞세웁니다(Stage H 계수). 파라미터 없음.",
               None, lambda p, _: -np.log(p.close.where(p.close > 0))),
        Screen("s_low_ivol", "고유 변동성 낮음", "−(SPY 대비 잔차 변동성, 63일)",
               "시장과 무관한 흔들림이 작은 종목(Ang 등 2006). '고유 변동성 상위' 제외 스크린과 같은 값입니다. 파라미터 없음.",
               None, lambda p, _: -_ivol(p), "보조지표", 2),
    ]
})
STAGE_K_SCORES = ("s_low_price", "s_low_ivol")
STYLE_TILTS = ("s_small", "s_large", "s_dividend", "s_reversal", "s_lowvol")  # single-factor tilts (kept: the user wants small caps testable)
TIER_LABELS = {1: "1순위: 가격·거래량", 2: "2순위: 파생 지표", 3: "3순위: 규모·배당·업종"}


def scores_by_tier() -> dict[int, list[str]]:
    out: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for k, sc in SELECT_SCORES.items():
        out[sc.tier].append(k)
    return out


ALL_SCREENS: dict[str, Screen] = {**HYGIENE_SCREENS, **EXCLUDE_SCREENS, **SELECT_SCORES}


# --------------------------------------------------------------------------- monthly mask and rank

@dataclass(frozen=True)
class ScreenConfig:
    exclude: dict[str, float | None]  # exclusion (and hygiene) screens and their values (all applied)
    select: dict[str, float | None]  # selection scores, combined by mean percentile rank
    top_k: int = 100  # how many of the best remain buyable each month (0 = all)

    def describe(self) -> str:
        def fmt(chosen):
            return " + ".join(ALL_SCREENS[k].label + ("" if v is None else f"({v:g})") for k, v in chosen.items()) or "없음"
        return f"제외: {fmt(self.exclude)} | 선별: {fmt(self.select)}" + (f" 상위 {self.top_k}" if self.top_k else "")

    def to_dict(self) -> dict:
        return {"exclude": dict(self.exclude), "select": dict(self.select), "top_k": int(self.top_k)}


# Ready-made combinations in the shape traders publish them. Values are the published defaults,
# fixed before any backtest; a profile is tested as a unit (its K and the trading layer vary).
HYGIENE_DEFAULT: dict[str, float | None] = {"h_history": 12, "h_price": 5, "h_liquidity": 1, "h_illiquid": 20}  # no size floor
PROFILES: dict[str, tuple[str, ScreenConfig]] = {
    "minervini": ("Minervini 추세 템플릿", ScreenConfig(
        exclude={**HYGIENE_DEFAULT, "tt_fail": 25, "x_extended": 25}, select={"s_rs": None}, top_k=50)),
    "canslim_pv": ("IBD/CANSLIM 가격·거래량", ScreenConfig(
        exclude={**HYGIENE_DEFAULT, "x_drawdown": 15, "x_weak_industry": 30},
        select={"s_rs": None, "s_accum": 3, "s_high52_vol": None}, top_k=50)),
    "quality_momentum": ("질 좋은 모멘텀", ScreenConfig(
        exclude={**HYGIENE_DEFAULT, "x_lottery": 20, "x_ivol": 20},
        select={"s_rs": None, "s_smooth": 12, "s_quality": 12}, top_k=50)),
    "industry_leaders": ("주도 업종의 주도주", ScreenConfig(
        exclude={**HYGIENE_DEFAULT, "x_weak_industry": 50, "tt_fail": 25},
        select={"s_industry_rs": 6, "s_rs": None}, top_k=50)),
    "lowvol_momentum": ("저변동성 모멘텀", ScreenConfig(
        exclude={**HYGIENE_DEFAULT, "x_ivol": 30}, select={"s_rs": None, "s_lowvol": None}, top_k=50)),
    "trend_only": ("추세 템플릿만 (순위 없음)", ScreenConfig(
        exclude={**HYGIENE_DEFAULT, "tt_fail": 25}, select={}, top_k=0)),
}


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month in the index."""
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby([index.year, index.month]).last().to_numpy())


def monthly_mask(panel: Panel, cfg: ScreenConfig, universe: pd.DataFrame | None = None,
                 cache: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(eligible, rank): bool and float frames (dates x symbols) for the engine.

    Decided on each month's last trading day from data up to that day and held through the next
    month. `universe` (default panel.eligible) is the membership the screens are applied inside.
    `cache` (optional dict) memoises screen frames across configs on the same panel.
    """
    close = panel.close
    base = universe if universe is not None else panel.eligible
    base = (base.reindex(index=close.index, columns=close.columns).astype(float).fillna(0.0).astype(bool) if base is not None
            else close.notna())
    if BENCHMARK in base.columns:
        base[BENCHMARK] = False
    cache = {} if cache is None else cache
    me = month_ends(close.index)

    def frame(key: str, value):
        """The screen's month-end rows (only those are decided on); cached per screen and value."""
        k = (key, None if value is None else float(value))
        if k not in cache:
            cache[k] = ALL_SCREENS[key].compute(panel, value).loc[me]
        return cache[k]

    allowed = base.loc[me]
    for key, value in cfg.exclude.items():
        allowed &= ~frame(key, value).astype(float).fillna(1.0).astype(bool)  # unknown = excluded
    if cfg.select:
        pct = None
        for key, value in cfg.select.items():
            s = frame(key, value).where(allowed)
            r = s.rank(axis=1, pct=True)
            pct = r if pct is None else pct + r
        score = pct / len(cfg.select)
        rank = score.where(allowed).rank(axis=1, ascending=False, method="first")
        if cfg.top_k:
            allowed &= rank <= cfg.top_k
        rank = rank.where(allowed)
    else:
        rank = pd.DataFrame(np.nan, index=me, columns=close.columns)
    # forward-filled through numbers: a reindexed bool frame would become object dtype and warn on fillna
    eligible = allowed.astype("int8").reindex(close.index, method="ffill").fillna(0).astype(bool)
    rank = rank.reindex(close.index, method="ffill")
    return eligible, rank


def apply_screens(panel: Panel, cfg: ScreenConfig, cache: dict | None = None) -> Panel:
    """A copy of the panel whose eligible/rank come from the screens (the engine then buys only
    screened symbols, best rank first)."""
    eligible, rank = monthly_mask(panel, cfg, cache=cache)
    return Panel(panel.open, panel.high, panel.low, panel.close, panel.volume, info=panel.info, source=panel.source,
                 tradable=panel.tradable, cash=panel.cash, kind=panel.kind, eligible=eligible,
                 delist_exit=panel.delist_exit, extra=panel.extra, rank=rank)
