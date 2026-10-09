"""Strategy building blocks.

Every block uses only price and volume and has exactly one numeric parameter.
Entry blocks are combined with AND; exit blocks are combined with OR.
All conditions are evaluated on the day-t close and executed at the day t+1 open.

To add a block: append a Block to ENTRY_BLOCKS or EXIT_BLOCKS and give it a compute
function (for exits that depend on the position, handle the key in engine.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    param: Param
    compute: Callable[[Panel, float], pd.DataFrame] | None = None  # None => position-dependent


def _sma(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close.rolling(n, min_periods=n).mean()


def _breakout(p: Panel, n: float) -> pd.DataFrame:
    n = int(n)
    return p.close > p.high.shift(1).rolling(n, min_periods=n).max()


def _above_ma(p: Panel, n: float) -> pd.DataFrame:
    return p.close > _sma(p.close, int(n))


def _momentum(p: Panel, n: float) -> pd.DataFrame:
    return p.close / p.close.shift(int(n)) - 1.0 > 0


def _volume_surge(p: Panel, k: float) -> pd.DataFrame:
    avg = p.volume.shift(1).rolling(20, min_periods=20).mean()
    return p.volume > k * avg


def _low_break(p: Panel, n: float) -> pd.DataFrame:
    n = int(n)
    return p.close < p.low.shift(1).rolling(n, min_periods=n).min()


def _below_ma(p: Panel, n: float) -> pd.DataFrame:
    return p.close < _sma(p.close, int(n))


ENTRY_BLOCKS: dict[str, Block] = {
    b.key: b
    for b in [
        Block(
            "breakout", "N일 신고가 돌파",
            "오늘 종가 > 직전 N거래일 고가의 최댓값",
            "가격이 최근 N일 범위를 위로 벗어나는 순간을 잡습니다. 대표적인 돈치안(Donchian) 돌파.",
            Param("N (거래일)", 20, 5, 250, 5, "거래일", "비교할 과거 고가 구간의 길이"),
            _breakout,
        ),
        Block(
            "above_ma", "이동평균 위",
            "오늘 종가 > 최근 N거래일 종가 단순이동평균(SMA)",
            "가격이 중장기 평균보다 위에 있는 동안만 진입을 허용하는 추세 필터.",
            Param("N (거래일)", 200, 10, 300, 10, "거래일", "이동평균 기간"),
            _above_ma,
        ),
        Block(
            "momentum", "N일 수익률 플러스",
            "오늘 종가 / N거래일 전 종가 − 1 > 0",
            "N일 전보다 가격이 높을 때만 진입(시계열 모멘텀).",
            Param("N (거래일)", 126, 21, 252, 21, "거래일", "수익률을 재는 기간 (21거래일 ≈ 1개월)"),
            _momentum,
        ),
        Block(
            "volume_surge", "거래량 증가 확인",
            "오늘 거래량 > k × 직전 20거래일 평균 거래량",
            "가격 신호가 평소보다 많은 거래량을 동반할 때만 인정합니다. 평균 기간 20일은 고정입니다.",
            Param("k (배수)", 1.5, 1.0, 4.0, 0.25, "배", "평균 거래량 대비 배수"),
            _volume_surge,
        ),
    ]
}

EXIT_BLOCKS: dict[str, Block] = {
    b.key: b
    for b in [
        Block(
            "low_break", "N일 신저가 이탈",
            "오늘 종가 < 직전 N거래일 저가의 최솟값",
            "가격이 최근 N일 범위 아래로 내려가면 청산. 추세가 이어지는 동안은 계속 보유합니다.",
            Param("N (거래일)", 20, 5, 120, 5, "거래일", "비교할 과거 저가 구간의 길이"),
            _low_break,
        ),
        Block(
            "below_ma", "이동평균 이탈",
            "오늘 종가 < 최근 N거래일 종가 SMA",
            "가격이 이동평균 아래로 내려가면 청산.",
            Param("N (거래일)", 50, 10, 300, 10, "거래일", "이동평균 기간"),
            _below_ma,
        ),
        Block(
            "trailing_pct", "고점 대비 하락(트레일링 스톱)",
            "오늘 종가 < 진입 후 최고 종가 × (1 − X%)",
            "보유 중 기록한 최고 종가에서 X% 이상 떨어지면 청산. 수익을 지키면서 추세는 따라갑니다.",
            Param("X (%)", 15, 3, 40, 1, "%", "최고 종가 대비 허용 하락폭"),
            None,
        ),
        Block(
            "stop_loss_pct", "진입가 대비 손절",
            "오늘 종가 < 진입가 × (1 − X%)",
            "진입 직후 신호가 틀렸을 때 손실을 X%로 제한합니다.",
            Param("X (%)", 10, 2, 30, 1, "%", "진입가 대비 허용 손실폭"),
            None,
        ),
    ]
}

ALL_BLOCKS: dict[str, Block] = {**ENTRY_BLOCKS, **EXIT_BLOCKS}
