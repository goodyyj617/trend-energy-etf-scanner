"""Stock screens: what may be bought (DECISIONS 77).

Three stages, all point-in-time and all swappable:
  1. exclusion screens  - "never buy this now": each returns True where a symbol is excluded;
  2. selection scores   - "which of the rest are best": each returns a score, higher = better;
                          several scores are combined by their average percentile rank, and only
                          the top K symbols stay buyable;
  3. entry timing       - the existing entry blocks (lab/blocks.py), or none.
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

from .blocks import Param
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


# --------------------------------------------------------------------------- 1. exclusion (True = do not buy)

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


SELECT_SCORES: dict[str, Screen] = {
    s.key: s for s in [
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

ALL_SCREENS: dict[str, Screen] = {**EXCLUDE_SCREENS, **SELECT_SCORES}


# --------------------------------------------------------------------------- monthly mask and rank

@dataclass(frozen=True)
class ScreenConfig:
    exclude: dict[str, float | None]  # exclusion screens and their values (all applied)
    select: dict[str, float | None]  # selection scores, combined by mean percentile rank
    top_k: int = 100  # how many of the best remain buyable each month (0 = all)

    def describe(self) -> str:
        def fmt(screens, chosen):
            return " + ".join(screens[k].label + ("" if v is None else f"({v:g})") for k, v in chosen.items()) or "없음"
        return f"제외: {fmt(EXCLUDE_SCREENS, self.exclude)} | 선별: {fmt(SELECT_SCORES, self.select)}" + \
            (f" 상위 {self.top_k}" if self.top_k else "")

    def to_dict(self) -> dict:
        return {"exclude": dict(self.exclude), "select": dict(self.select), "top_k": int(self.top_k)}


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
    base = (base.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool) if base is not None
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
        allowed &= ~frame(key, value).fillna(True).astype(bool)  # unknown = excluded
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
    eligible = allowed.reindex(close.index, method="ffill").fillna(False).astype(bool)
    rank = rank.reindex(close.index, method="ffill")
    return eligible, rank


def apply_screens(panel: Panel, cfg: ScreenConfig, cache: dict | None = None) -> Panel:
    """A copy of the panel whose eligible/rank come from the screens (the engine then buys only
    screened symbols, best rank first)."""
    eligible, rank = monthly_mask(panel, cfg, cache=cache)
    return Panel(panel.open, panel.high, panel.low, panel.close, panel.volume, info=panel.info, source=panel.source,
                 tradable=panel.tradable, cash=panel.cash, kind=panel.kind, eligible=eligible,
                 delist_exit=panel.delist_exit, extra=panel.extra, rank=rank)
