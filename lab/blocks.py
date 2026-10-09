"""Strategy building blocks.

Entry blocks are combined with AND; exit blocks are combined with OR.
All conditions are evaluated on the day-t close and executed at the day t+1 open.
Each block has at most one numeric parameter; some blocks are fixed rules with none
(fewer knobs = less room for overfitting).

To add a block: append a Block to ENTRY_BLOCKS or EXIT_BLOCKS and give it a compute
function. Exits that depend on the open position (compute=None) are handled by key in
engine.py.
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
    family: str = "가격"  # 가격 / 거래량 / 보조지표: grouping in the UI


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


# --------------------------------------------------------------------------- entry rules

def _breakout(p: Panel, n: float) -> pd.DataFrame:
    n = int(n)
    return p.close > p.high.shift(1).rolling(n, min_periods=n).max()


def _above_ma(p: Panel, n: float) -> pd.DataFrame:
    return p.close > sma(p.close, int(n))


def _momentum(p: Panel, n: float) -> pd.DataFrame:
    return p.close / p.close.shift(int(n)) - 1.0 > 0


def _market_trend(p: Panel, n: float) -> pd.DataFrame:
    from .data import BENCHMARK

    spy = p.close[BENCHMARK]
    up = (spy > spy.rolling(int(n), min_periods=int(n)).mean()).to_numpy()
    return pd.DataFrame(np.repeat(up[:, None], p.close.shape[1], axis=1), index=p.close.index, columns=p.close.columns)


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


# --------------------------------------------------------------------------- exit rules

def _low_break(p: Panel, n: float) -> pd.DataFrame:
    n = int(n)
    return p.close < p.low.shift(1).rolling(n, min_periods=n).min()


def _below_ma(p: Panel, n: float) -> pd.DataFrame:
    return p.close < sma(p.close, int(n))


ENTRY_BLOCKS: dict[str, Block] = {
    b.key: b
    for b in [
        Block("breakout", "N일 신고가 돌파", "오늘 종가 > 직전 N거래일 고가의 최댓값",
              "가격이 최근 N일 범위를 위로 벗어나는 순간을 잡습니다. 대표적인 돈치안(Donchian) 돌파.",
              Param("N (거래일)", 20, 5, 250, 5, "거래일", "비교할 과거 고가 구간의 길이"), _breakout, "가격"),
        Block("above_ma", "이동평균 위", "오늘 종가 > 최근 N거래일 종가 단순이동평균(SMA)",
              "가격이 중장기 평균보다 위에 있는 동안만 진입을 허용하는 추세 필터.",
              Param("N (거래일)", 200, 10, 300, 10, "거래일", "이동평균 기간"), _above_ma, "가격"),
        Block("momentum", "N일 수익률 플러스", "오늘 종가 / N거래일 전 종가 − 1 > 0",
              "N일 전보다 가격이 높을 때만 진입(시계열 모멘텀).",
              Param("N (거래일)", 126, 21, 252, 21, "거래일", "수익률을 재는 기간 (21거래일 ≈ 1개월)"), _momentum, "가격"),
        Block("market_trend", "시장 추세 (SPY 이동평균 위)", "SPY 종가 > SPY의 최근 N거래일 SMA",
              "시장 전체(SPY)가 상승 추세일 때만 새로 매수합니다. 개별 종목 신호와 상관없이 하락장에서 신규 진입을 막는 국면 필터. "
              "이미 보유한 종목은 청산 조건으로만 팝니다.",
              Param("N (거래일)", 200, 50, 300, 10, "거래일", "SPY 이동평균 기간"), _market_trend, "가격"),
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
    ]
}

ALL_BLOCKS: dict[str, Block] = {**ENTRY_BLOCKS, **EXIT_BLOCKS}
