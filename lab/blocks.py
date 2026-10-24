"""Strategy building blocks.

Entry blocks are combined with AND; exit blocks are combined with OR.
All conditions are evaluated on the day-t close and executed at the day t+1 open.
Each block has at most one numeric parameter; some blocks are fixed rules with none
(fewer knobs = less room for overfitting).

To add a block: append a Block to ENTRY_BLOCKS or EXIT_BLOCKS and give it a compute
function. Exits that depend on the open position (compute=None) are handled by key in
engine.py.

Macro blocks (family "매크로") are market-wide regime conditions read from lab/macro.py
(FRED series, one-day publication lag); the same value applies to every symbol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import Panel


@dataclass(frozen=True)
class Param:
    label: str
    default: float
    minimum: float
    maximum: float
    step: float
    unit: str
    help: str

    @property
    def is_int(self) -> bool:
        return float(self.step).is_integer() and float(self.default).is_integer()


@dataclass(frozen=True)
class Block:
    key: str
    label: str
    rule: str  # exact condition, shown in the UI
    help: str
    param: Param | None
    compute: Callable[[Panel, float | None], pd.DataFrame] | None = None  # None => position-dependent
    family: str = "가격"  # 가격 / 거래량 / 보조지표 / 매크로: grouping in the UI


# --------------------------------------------------------------------------- indicators

def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def wilder(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's smoothing (RSI / ATR convention): EMA with alpha = 1/n."""
    return df.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder(-delta.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def atr(p: Panel, n: int = 14) -> pd.DataFrame:
    prev = p.close.shift(1)
    tr = np.fmax(np.fmax(p.high - p.low, (p.high - prev).abs()), (p.low - prev).abs())  # fmax skips the NaN gap
    return wilder(tr, n)


def obv(p: Panel) -> pd.DataFrame:
    direction = np.sign(p.close.diff()).fillna(0.0)
    return (direction * p.volume.fillna(0.0)).cumsum().where(p.close.notna())


def cmf(p: Panel, n: int = 20) -> pd.DataFrame:
    """Chaikin Money Flow: n-day sum of (close location in the day's range x volume) / n-day volume."""
    rng = (p.high - p.low).replace(0, np.nan)
    mfm = (((p.close - p.low) - (p.high - p.close)) / rng).fillna(0.0)  # zero-range day = neutral
    vol = p.volume.fillna(0.0)
    return (mfm * vol).rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum().replace(0, np.nan)


def adx_di(p: Panel, n: int = 14) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wilder's ADX, +DI and -DI."""
    up, down = p.high.diff(), -p.low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    tr = atr(p, n)
    plus_di = 100 * wilder(plus_dm, n) / tr.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, n) / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, n), plus_di, minus_di


# --------------------------------------------------------------------------- entry rules

def _breakout(p: Panel, n: float) -> pd.DataFrame:
    n = int(n)
    return p.close > p.high.shift(1).rolling(n, min_periods=n).max()


def _above_ma(p: Panel, n: float) -> pd.DataFrame:
    return p.close > sma(p.close, int(n))


def _ma_cross(p: Panel, n: float) -> pd.DataFrame:
    n = int(n)
    return sma(p.close, max(2, int(round(n / 4)))) > sma(p.close, n)


def _momentum(p: Panel, n: float) -> pd.DataFrame:
    return p.close / p.close.shift(int(n)) - 1.0 > 0


def _market_trend(p: Panel, n: float) -> pd.DataFrame:
    from .data import BENCHMARK

    spy = p.close[BENCHMARK]
    up = (spy > spy.rolling(int(n), min_periods=int(n)).mean()).to_numpy()
    return pd.DataFrame(np.repeat(up[:, None], p.close.shape[1], axis=1), index=p.close.index, columns=p.close.columns)


def _broadcast(p: Panel, cond: pd.Series) -> pd.DataFrame:
    """One market-wide condition repeated for every symbol. Missing data (NaN) counts as False."""
    v = cond.fillna(0.0).to_numpy(dtype=float) > 0.5
    return pd.DataFrame(np.repeat(v[:, None], p.close.shape[1], axis=1), index=p.close.index,
                        columns=p.close.columns)


def _macro_cond(p: Panel, sid: str, test, sma: int | None = None) -> pd.DataFrame:
    from . import macro

    val, avg = macro.series(sid, p.close.index, sma)
    ok = val.notna() & (avg.notna() if avg is not None else True)
    cond = test(val, avg).astype(float).where(ok)
    return _broadcast(p, cond)


def _curve_ok(p: Panel, x: float) -> pd.DataFrame:
    return _macro_cond(p, "T10Y3M", lambda v, _: v > x)


def _credit_calm(p: Panel, n: float) -> pd.DataFrame:
    return _macro_cond(p, "BAA10Y", lambda v, a: v <= a, int(n))


def _vix_calm(p: Panel, x: float) -> pd.DataFrame:
    return _macro_cond(p, "VIXCLS", lambda v, _: v < x)


def _ma_stack(p: Panel, _: float | None) -> pd.DataFrame:
    m20, m60, m120, m200 = (sma(p.close, n) for n in (20, 60, 120, 200))
    return (m20 > m60) & (m60 > m120) & (m120 > m200)


def _rsi_min(p: Panel, x: float) -> pd.DataFrame:
    return rsi(p.close, 14) >= x


def _bollinger(p: Panel, k: float) -> pd.DataFrame:
    mid = sma(p.close, 20)
    sd = p.close.rolling(20, min_periods=20).std(ddof=0)
    return p.close > mid + k * sd


def _williams(p: Panel, x: float) -> pd.DataFrame:
    hh = p.high.rolling(14, min_periods=14).max()
    ll = p.low.rolling(14, min_periods=14).min()
    wr = -100 * (hh - p.close) / (hh - ll).replace(0, np.nan)
    return wr >= x


def _volume_surge(p: Panel, k: float) -> pd.DataFrame:
    avg = p.volume.shift(1).rolling(20, min_periods=20).mean()
    return p.volume > k * avg


def _obv_up(p: Panel, n: float) -> pd.DataFrame:
    o = obv(p)
    return o > sma(o, int(n))


def _up_candle(p: Panel, _: float | None) -> pd.DataFrame:
    return p.close > p.open


def _near_high(p: Panel, x: float) -> pd.DataFrame:
    hi = p.close.rolling(252, min_periods=252).max()
    return p.close >= hi * (1 - x / 100.0)


def _adx_trend(p: Panel, x: float) -> pd.DataFrame:
    adx, plus_di, minus_di = adx_di(p, 14)
    return (adx >= x) & (plus_di > minus_di)


def _keltner(p: Panel, k: float) -> pd.DataFrame:
    mid = p.close.ewm(span=20, adjust=False, min_periods=20).mean()
    return p.close > mid + k * atr(p, 14)


def _cmf_min(p: Panel, x: float) -> pd.DataFrame:
    return cmf(p, 20) >= x


# --------------------------------------------------------------------------- exit rules

def _momentum_neg(p: Panel, n: float) -> pd.DataFrame:
    return p.close / p.close.shift(int(n)) - 1.0 < 0


def _rsi_below(p: Panel, x: float) -> pd.DataFrame:
    return rsi(p.close, 14) < x


def _cmf_below(p: Panel, x: float) -> pd.DataFrame:
    return cmf(p, 20) < x


def _low_break(p: Panel, n: float) -> pd.DataFrame:
    n = int(n)
    return p.close < p.low.shift(1).rolling(n, min_periods=n).min()


def _below_ma(p: Panel, n: float) -> pd.DataFrame:
    return p.close < sma(p.close, int(n))


# Regime-off exits: the exact complement of the matching regime entry filter (REGIME_PAIRS).
def _market_off(p: Panel, n: float) -> pd.DataFrame:
    from .data import BENCHMARK

    spy = p.close[BENCHMARK]
    ma = spy.rolling(int(n), min_periods=int(n)).mean()
    return _broadcast(p, (spy < ma).astype(float).where(ma.notna()))


def _curve_inverted(p: Panel, x: float) -> pd.DataFrame:
    return _macro_cond(p, "T10Y3M", lambda v, _: v <= x)


def _credit_stress(p: Panel, n: float) -> pd.DataFrame:
    return _macro_cond(p, "BAA10Y", lambda v, a: v > a, int(n))


def _vix_high(p: Panel, x: float) -> pd.DataFrame:
    return _macro_cond(p, "VIXCLS", lambda v, _: v >= x)


ENTRY_BLOCKS: dict[str, Block] = {
    b.key: b
    for b in [
        Block("breakout", "N일 신고가 돌파", "오늘 종가 > 직전 N거래일 고가의 최댓값",
              "가격이 최근 N일 범위를 위로 벗어나는 순간을 잡습니다. 대표적인 돈치안(Donchian) 돌파.",
              Param("N (거래일)", 20, 5, 250, 5, "거래일", "비교할 과거 고가 구간의 길이"), _breakout, "가격"),
        Block("above_ma", "이동평균 위", "오늘 종가 > 최근 N거래일 종가 단순이동평균(SMA)",
              "가격이 중장기 평균보다 위에 있는 동안만 진입을 허용하는 추세 필터.",
              Param("N (거래일)", 200, 10, 300, 10, "거래일", "이동평균 기간"), _above_ma, "가격"),
        Block("ma_cross", "이동평균 교차 (골든크로스)", "SMA(N÷4, 반올림) > SMA(N)   (예: N=200 → SMA50 > SMA200)",
              "단기 이동평균이 장기 이동평균 위에 있는 동안 진입을 허용합니다. 단기 기간은 장기의 4분의 1로 고정해 "
              "파라미터를 하나로 줄였습니다(50/200일 골든크로스가 대표적).",
              Param("N (장기, 거래일)", 200, 20, 300, 10, "거래일", "장기 이동평균 기간. 단기 = N÷4"), _ma_cross, "가격"),
        Block("momentum", "N일 수익률 플러스", "오늘 종가 / N거래일 전 종가 − 1 > 0",
              "N일 전보다 가격이 높을 때만 진입(시계열 모멘텀).",
              Param("N (거래일)", 126, 21, 252, 21, "거래일", "수익률을 재는 기간 (21거래일 ≈ 1개월)"), _momentum, "가격"),
        Block("market_trend", "시장 추세 (SPY 이동평균 위)", "SPY 종가 > SPY의 최근 N거래일 SMA",
              "시장 전체(SPY)가 상승 추세일 때만 새로 매수합니다. 개별 종목 신호와 상관없이 하락장에서 신규 진입을 막는 국면 필터. "
              "이미 보유한 종목은 청산 조건으로만 팝니다.",
              Param("N (거래일)", 200, 50, 300, 10, "거래일", "SPY 이동평균 기간"), _market_trend, "가격"),
        Block("curve_ok", "장단기 금리차 플러스 (매크로)", "전일 미국 10년 − 3개월 국채금리차 > X %p   (FRED T10Y3M)",
              "금리차가 음수(장단기 역전)면 경기 침체 신호로 보고 새 매수를 막는 국면 필터. 모든 종목에 똑같이 적용됩니다. "
              "FRED 공표 시점을 고려해 하루 전 값을 씁니다. 이미 보유한 종목은 청산 조건으로만 팝니다.",
              Param("X (%p)", 0.0, -1.0, 2.0, 0.25, "%p", "금리차 하한 (0 = 역전만 아니면 됨)"), _curve_ok, "매크로"),
        Block("credit_calm", "신용 스프레드 안정 (매크로)", "전일 Baa 회사채 − 10년 국채 스프레드 ≤ 그 스프레드의 N일 SMA   (FRED BAA10Y)",
              "회사채 가산금리가 평소(N일 평균)보다 넓어지는 동안은 신용 위험이 커지는 국면으로 보고 새 매수를 막습니다. "
              "하루 전 값을 씁니다.",
              Param("N (관측일)", 200, 50, 300, 10, "관측일", "스프레드 이동평균 기간"), _credit_calm, "매크로"),
        Block("vix_calm", "VIX 낮음 (매크로)", "전일 VIX 종가 < X   (FRED VIXCLS)",
              "시장의 기대 변동성(공포 지수)이 X 이상이면 새 매수를 막습니다. 하루 전 값을 씁니다.",
              Param("X", 25.0, 12.0, 50.0, 1.0, "점", "VIX 상한"), _vix_calm, "매크로"),
        Block("ma_stack", "이동평균 정배열", "SMA20 > SMA60 > SMA120 > SMA200",
              "단기부터 장기까지 이동평균이 위에서 아래로 차례대로 놓인 상태. 고정 규칙이라 조정할 파라미터가 없습니다.",
              None, _ma_stack, "가격"),
        Block("up_candle", "양봉", "오늘 종가 > 오늘 시가",
              "신호가 난 날이 상승 마감한 경우만 인정합니다. 파라미터 없음.",
              None, _up_candle, "가격"),
        Block("volume_surge", "거래량 증가 확인", "오늘 거래량 > k × 직전 20거래일 평균 거래량",
              "가격 신호가 평소보다 많은 거래량을 동반할 때만 인정합니다. 평균 기간 20일은 고정입니다.",
              Param("k (배수)", 1.5, 1.0, 4.0, 0.25, "배", "평균 거래량 대비 배수 (2.0 = 200%)"), _volume_surge, "거래량"),
        Block("obv_up", "OBV 상승", "OBV > OBV의 N일 이동평균(시그널선)",
              "OBV(거래량 누적: 오른 날 +거래량, 내린 날 −거래량)가 시그널선 위 = 매수 쪽 거래량이 우세. "
              "OBV의 시작값은 비교에 영향을 주지 않습니다.",
              Param("N (시그널, 거래일)", 10, 5, 60, 5, "거래일", "OBV 시그널선 이동평균 기간"), _obv_up, "거래량"),
        Block("rsi_min", "RSI(14) 이상", "RSI(14) ≥ X   (RSI = 100 − 100 / (1 + 평균상승폭/평균하락폭), Wilder 평활 14일)",
              "최근 14일 상승 힘이 하락 힘보다 충분히 클 때만 진입. 50 초과 = 상승 우위.",
              Param("X", 55, 40, 80, 5, "점", "RSI 하한"), _rsi_min, "보조지표"),
        Block("bollinger", "볼린저밴드 상단 돌파", "오늘 종가 > SMA20 + k × 20일 표준편차",
              "최근 변동성 대비 이례적으로 강한 상승(변동성 돌파). 기간 20일은 고정입니다.",
              Param("k (표준편차 배수)", 2.0, 1.0, 3.0, 0.25, "σ", "밴드 폭"), _bollinger, "보조지표"),
        Block("williams", "Williams %R(14) 이상", "%R = −100 × (14일 최고가 − 종가) / (14일 최고가 − 14일 최저가) ≥ X",
              "종가가 최근 14일 범위의 위쪽에 있을 때만 진입 (0 = 범위 최상단, −100 = 최하단).",
              Param("X", -20, -50, -5, 5, "점", "%R 하한"), _williams, "보조지표"),
        Block("near_high", "52주 고가 근접", "오늘 종가 ≥ 최근 252거래일 최고 종가 × (1 − X%)",
              "1년 최고가에서 X% 안쪽에 있을 때만 진입. 신고가 근처 종목이 계속 강한 경향(52주 고가 효과)을 씁니다. "
              "돌파 순간이 아니라 '고가 근처에 머무는 상태'를 봅니다. 기간 252일은 고정입니다.",
              Param("X (%)", 5, 1, 30, 1, "%", "1년 최고 종가에서 허용하는 거리"), _near_high, "가격"),
        Block("adx_trend", "ADX 추세 강도", "ADX(14) ≥ X 이고 +DI(14) > −DI(14)",
              "ADX는 방향과 상관없이 추세가 얼마나 뚜렷한지(0–100), +DI > −DI는 그 추세가 위쪽이라는 뜻입니다. "
              "횡보 구간의 가짜 신호를 거르는 용도. 고가·저가·종가로 계산하고 기간 14일(Wilder)은 고정입니다.",
              Param("X", 25, 10, 50, 5, "점", "ADX 하한 (25 이상이면 흔히 '추세 있음')"), _adx_trend, "보조지표"),
        Block("keltner", "켈트너 채널 상단 돌파", "오늘 종가 > EMA20 + k × ATR(14)",
              "볼린저밴드와 비슷하지만 표준편차 대신 ATR(평균 진폭)로 폭을 잽니다. 갭과 장중 변동까지 반영합니다. "
              "기간 20·14일은 고정입니다.",
              Param("k (ATR 배수)", 2.0, 0.5, 4.0, 0.25, "ATR", "채널 폭"), _keltner, "보조지표"),
        Block("cmf_min", "자금 흐름 유입 (CMF)", "CMF(20) ≥ X   (CMF = Σ20일[(종가−저가) − (고가−종가)] / (고가−저가) × 거래량 ÷ Σ20일 거래량)",
              "Chaikin Money Flow. 거래량이 실린 날 종가가 하루 범위의 위쪽에서 끝나면 매수 쪽 자금 흐름으로 봅니다(−1–+1). "
              "기간 20일은 고정입니다.",
              Param("X", 0.05, -0.2, 0.4, 0.05, "", "CMF 하한 (0 초과 = 유입 우세)"), _cmf_min, "거래량"),
    ]
}

EXIT_BLOCKS: dict[str, Block] = {
    b.key: b
    for b in [
        Block("low_break", "N일 신저가 이탈", "오늘 종가 < 직전 N거래일 저가의 최솟값",
              "가격이 최근 N일 범위 아래로 내려가면 청산. 추세가 이어지는 동안은 계속 보유합니다.",
              Param("N (거래일)", 20, 5, 120, 5, "거래일", "비교할 과거 저가 구간의 길이"), _low_break, "가격"),
        Block("below_ma", "이동평균 이탈", "오늘 종가 < 최근 N거래일 종가 SMA",
              "가격이 이동평균 아래로 내려가면 청산.",
              Param("N (거래일)", 50, 10, 300, 10, "거래일", "이동평균 기간"), _below_ma, "가격"),
        Block("trailing_pct", "고점 대비 하락(트레일링 스톱)", "오늘 종가 < 진입 후 최고 종가 × (1 − X%)",
              "보유 중 기록한 최고 종가에서 X% 이상 떨어지면 청산. 수익을 지키면서 추세는 따라갑니다.",
              Param("X (%)", 15, 3, 40, 1, "%", "최고 종가 대비 허용 하락폭"), None, "가격"),
        Block("atr_trail", "ATR 트레일링 스톱", "오늘 종가 < 진입 후 최고 종가 − k × ATR(14)",
              "종목의 평소 변동폭(ATR)에 맞춰 손절 폭이 자동으로 넓어지고 좁아집니다(샹들리에 청산). "
              "ATR = 진폭(고가−저가, 전일 종가 대비 갭 포함)의 14일 Wilder 평균.",
              Param("k (ATR 배수)", 3.0, 1.0, 6.0, 0.5, "ATR", "최고 종가에서 몇 ATR 아래에서 청산할지"), None, "보조지표"),
        Block("stop_loss_pct", "진입가 대비 손절", "오늘 종가 < 진입가 × (1 − X%)",
              "진입 직후 신호가 틀렸을 때 손실을 X%로 제한합니다.",
              Param("X (%)", 10, 2, 30, 1, "%", "진입가 대비 허용 손실폭"), None, "가격"),
        Block("time_stop", "시간 손절", "진입 후 N거래일이 지났는데 오늘 종가 ≤ 진입가",
              "오래 지나도 오르지 않는 포지션을 정리합니다. 수익 중인 포지션은 이 조건으로 팔지 않습니다.",
              Param("N (거래일)", 20, 5, 120, 5, "거래일", "진입 후 기다리는 거래일 수"), None, "가격"),
        Block("take_profit_pct", "부분 익절 (보유의 50%)", "종가 ≥ 진입가 × (1 + X%)인 첫날 → 다음 날 시가에 보유의 50% 매도 (한 번만)",
              "수익의 일부를 확정합니다. 남은 절반은 다른 청산 조건으로 팝니다. 이 조건만으로는 전량 청산이 일어나지 않습니다.",
              Param("X (%)", 20, 5, 100, 5, "%", "진입가 대비 상승률"), None, "가격"),
        Block("atr_stop", "ATR 손절 (진입가 기준)", "오늘 종가 < 진입가 − k × ATR(14)   (ATR은 신호일 값으로 고정)",
              "손절 폭을 종목의 평소 변동폭에 맞춥니다(변동성이 큰 종목은 넓게, 작은 종목은 좁게). 이 폭이 '위험 기준' 포지션 크기의 "
              "기준 단위 R이 됩니다.",
              Param("k (ATR 배수)", 2.0, 0.5, 5.0, 0.5, "ATR", "진입가에서 몇 ATR 아래에서 손절할지"), None, "보조지표"),
        Block("market_off", "시장 추세 이탈 (SPY 이동평균 아래)", "SPY 종가 < SPY의 최근 N거래일 SMA",
              "시장 전체가 하락 추세로 바뀌면 보유 종목을 모두 팝니다. ‘시장 추세’ 진입 필터의 정반대 조건.",
              Param("N (거래일)", 200, 50, 300, 10, "거래일", "SPY 이동평균 기간"), _market_off, "매크로"),
        Block("curve_inverted", "장단기 금리 역전 (매크로)", "전일 미국 10년 − 3개월 국채금리차 ≤ X %p",
              "금리차가 X 이하로 내려가면 보유 종목을 모두 팝니다. ‘금리차 플러스’ 진입 필터의 정반대 조건.",
              Param("X (%p)", 0.0, -1.0, 2.0, 0.25, "%p", "금리차 기준"), _curve_inverted, "매크로"),
        Block("credit_stress", "신용 스프레드 확대 (매크로)", "전일 Baa − 10년 국채 스프레드 > 그 스프레드의 N일 SMA",
              "회사채 가산금리가 평소보다 넓어지면 보유 종목을 모두 팝니다. ‘신용 스프레드 안정’의 정반대 조건.",
              Param("N (관측일)", 200, 50, 300, 10, "관측일", "스프레드 이동평균 기간"), _credit_stress, "매크로"),
        Block("vix_high", "VIX 높음 (매크로)", "전일 VIX 종가 ≥ X",
              "기대 변동성이 X 이상으로 올라가면 보유 종목을 모두 팝니다. ‘VIX 낮음’의 정반대 조건.",
              Param("X", 25.0, 12.0, 50.0, 1.0, "점", "VIX 기준"), _vix_high, "매크로"),
        Block("momentum_neg", "N일 수익률 마이너스", "오늘 종가 / N거래일 전 종가 − 1 < 0",
              "N일 전보다 가격이 낮아지면 청산(시계열 모멘텀이 꺼짐). 'N일 수익률 플러스' 진입의 반대 조건.",
              Param("N (거래일)", 63, 21, 252, 21, "거래일", "수익률을 재는 기간"), _momentum_neg, "가격"),
        Block("rsi_below", "RSI(14) 하락", "RSI(14) < X",
              "최근 14일 하락 힘이 상승 힘보다 커지면 청산. X가 높을수록 일찍 팝니다.",
              Param("X", 40, 20, 60, 5, "점", "RSI 기준"), _rsi_below, "보조지표"),
        Block("cmf_below", "자금 흐름 유출 (CMF)", "CMF(20) < X",
              "거래량이 실린 날 종가가 하루 범위의 아래쪽에서 끝나는 날이 우세해지면(매도 쪽 자금 흐름) 청산. 기간 20일은 고정입니다.",
              Param("X", -0.1, -0.4, 0.2, 0.05, "", "CMF 기준 (0 미만 = 유출 우세)"), _cmf_below, "거래량"),
    ]
}

REGIME_PAIRS: dict[str, str] = {  # regime entry filter -> its complement exit (regime switch)
    "market_trend": "market_off", "curve_ok": "curve_inverted", "credit_calm": "credit_stress", "vix_calm": "vix_high",
}

ALL_BLOCKS: dict[str, Block] = {**ENTRY_BLOCKS, **EXIT_BLOCKS}
