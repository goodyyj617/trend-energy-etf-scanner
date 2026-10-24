"""Crash-avoidance switches for a static allocation (DECISIONS A29; memo 2026-10-25_gate_reset_and_validation.md).

A switch is a monthly risk-off flag computed from data known at the decision day's close. When it is
on, the allocation's equity products move to the safe product (CD금리 by default); when it turns off,
they come back. Between switches the allocation keeps its own rebalancing (quarterly by default).

Signals (all on US data, as an investor would read them; `macro` = lab/macro_snapshot.csv.gz, FRED):
- spy_sma10   SPY month-end close below its 10-month average (Faber 2007).
- spy_sma200  SPY close below its 200-day average on the decision day.
- tip_canary  TIP 13612U momentum <= 0 (the HAA canary).
- baa_canary  any of SPY VWO VEA BND with 13612W < 0 (the BAA/VAA "breadth" canary).
- vix25       VIX close above 25.
- credit      Moody's Baa − 10y Treasury spread more than 0.5 %p above its 12-month low.
- curve       10y − 3m Treasury spread below zero (yield-curve inversion).
- oracle      on exactly inside the registered S&P500 crash windows (hindsight; the upper bound of
              what avoiding them could have been worth, never a strategy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aa_assets import CASH
from .aa_engine import AllocConfig, AllocResult, benchmark_of, decision_rows, simulate
from .aa_tactical import mom_13612u, mom_13612w, sampling_rows, spliced_close

MACRO_PATH = Path(__file__).with_name("macro_snapshot.csv.gz")
EQUITY_CLASS = "주식"
SIGNALS = {
    "spy_sma10": "S&P500 10개월 이동평균 아래",
    "spy_sma200": "S&P500 200일 이동평균 아래",
    "tip_canary": "TIP 카나리아(13612U ≤ 0, HAA)",
    "baa_canary": "SPY·VWO·VEA·BND 카나리아(하나라도 13612W < 0, BAA)",
    "vix25": "VIX 25 초과",
    "credit": "신용 스프레드(Baa−10년) 12개월 최저 + 0.5%p 초과",
    "curve": "장단기 금리 역전(10년 − 3개월 < 0)",
    "oracle": "사후 정답(위기 구간에만 회피, 상한선 참고용)",
}
ORACLE_WINDOWS = {"금융위기": ("2007-10-09", "2009-03-09"), "코로나": ("2020-02-19", "2020-03-23"),
                  "2022": ("2022-01-03", "2022-10-12")}


def load_macro(path: Path = MACRO_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def signal_series(kind: str, usd_close: pd.DataFrame, usd_cash: pd.Series | None, dates: pd.DatetimeIndex,
                  macro: pd.DataFrame | None = None, offset: int = 0) -> pd.Series:
    """Risk-off flag on each decision day (the month's last trading day + offset), indexed by that day."""
    rows = sampling_rows(dates, offset)
    days = dates[rows]
    if kind == "oracle":
        on = pd.Series(False, index=days)
        for a, b in ORACLE_WINDOWS.values():
            # entered at the first decision day on/after the window start, left at the first one after its end
            on |= (days >= pd.Timestamp(a) - pd.offsets.MonthEnd(1)) & (days < pd.Timestamp(b))
        return on
    if kind in ("vix25", "credit", "curve"):
        macro = load_macro() if macro is None else macro
        col = {"vix25": "VIXCLS", "credit": "BAA10Y", "curve": "T10Y3M"}[kind]
        s = macro[col].dropna()
        if kind == "vix25" and "^VIX" in usd_close.columns:
            s = usd_close["^VIX"].dropna().combine_first(s)
        s = s.reindex(s.index.union(days)).ffill().reindex(days)
        if kind == "vix25":
            return s > 25
        if kind == "curve":
            return s < 0
        daily = macro[col].dropna()
        low = daily.rolling(252, min_periods=120).min()
        low = low.reindex(low.index.union(days)).ffill().reindex(days)
        return s > low + 0.5
    close = spliced_close(usd_close, usd_cash).ffill()
    if kind == "spy_sma200":
        sma = close["SPY"].rolling(200, min_periods=200).mean()
        return _as_flag(close["SPY"] < sma, days)
    me = close.iloc[sampling_rows(close.index, offset)]  # the full monthly history, sampled like the decisions
    flags = pd.Series(False, index=me.index)
    for i in range(len(me)):
        if kind == "spy_sma10":
            if i >= 9:
                flags.iloc[i] = me["SPY"].iloc[i] < me["SPY"].iloc[i - 9:i + 1].mean()
        elif kind == "tip_canary":
            if i >= 12:
                flags.iloc[i] = mom_13612u(me[["TIP"]], i)["TIP"] <= 0
        elif kind == "baa_canary":
            if i >= 12:
                flags.iloc[i] = bool((mom_13612w(me[["SPY", "VWO", "VEA", "BND"]], i) < 0).any())
        else:
            raise ValueError(f"unknown signal {kind}")
    return _as_flag(flags, days)


def _as_flag(flags: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Boolean flags carried forward onto `index` (False before the first flag)."""
    f = flags.astype(float)
    return f.reindex(f.index.union(index)).ffill().reindex(index).fillna(0.0) > 0.5


def run_switch(panel, cfg: AllocConfig, risk_off: pd.Series, risky: list[str], safe: str = "459580",
               share: float = 1.0) -> AllocResult:
    """Run a static allocation with a monthly risk-off switch.

    risk_off: flag per decision day (from signal_series on the same dates and offset). When on, `share`
    of each product in `risky` moves to `safe`. Trades happen at the next close, on days when the
    flag changes and on the allocation's own rebalancing days."""
    from .aa_engine import _normalise  # noqa: PLC0415

    base = _normalise(cfg.weights)
    off_w = dict(base)
    for s in risky:
        if s in off_w:
            moved = off_w[s] * share
            off_w[s] -= moved
            off_w[safe] = off_w.get(safe, 0.0) + moved
    off_w = {s: v for s, v in off_w.items() if v > 1e-12}
    syms = sorted({s for s in list(base) + list(off_w) if s != CASH})
    idx = panel.close.index
    dates = idx[(idx >= pd.Timestamp(cfg.start)) & (idx <= pd.Timestamp(cfg.end))]
    close = panel.close[syms].ffill()
    first_valid = close.loc[:dates[-1]].apply(lambda s: s.first_valid_index()).max()
    dates = dates[dates >= first_valid]
    rets = close.pct_change(fill_method=None).loc[dates]
    cash = (panel.cash.reindex(idx).fillna(0.0) if panel.cash is not None else pd.Series(0.0, index=idx)).loc[dates]
    pos = pd.Series(np.arange(len(dates)), index=dates)
    flag = _as_flag(risk_off, dates)
    reb = set((decision_rows(dates, cfg.rebalance) + int(cfg.rebalance_offset)).tolist()) if cfg.rebalance != "none" else set()
    targets: dict[int, dict[str, float]] = {}
    state = bool(flag.iloc[0])
    for d in risk_off.index:
        if d not in pos.index:
            continue
        i = int(pos.loc[d])
        new = bool(risk_off.loc[d])
        if new != state and i < len(dates) - 1:
            targets[i] = off_w if new else base
        state = new
    current = bool(flag.iloc[0])
    flags = flag.to_numpy()
    for i in sorted(reb):
        if i < len(dates) - 1 and i not in targets:
            targets[i] = off_w if flags[i] else base
    first = off_w if current else base
    eq, weights, turnover, trades = simulate(rets, cash, targets, cfg.cost_bps, cfg.band, None, 0.0, 0.0, first)
    bench = panel.close[benchmark_of(panel)].loc[dates].ffill()
    res = AllocResult(cfg, eq, bench / bench.iloc[0], cash, weights, pd.Series(1.0, index=dates), turnover, trades)
    res.risk_off = flag
    return res


def drawdown_episodes(equity: pd.Series, top: int = 5, min_depth: float = 0.03) -> pd.DataFrame:
    """The deepest separate drawdowns: peak, trough, recovery (NaT if not yet), depth, days to recover."""
    dd = equity / equity.cummax() - 1.0
    under = dd < 0
    rows = []
    start = None
    for t, u in zip(dd.index, under):
        if u and start is None:
            start = t
        if not u and start is not None:
            seg = dd.loc[start:t]
            rows.append((start, seg.idxmin(), t, float(seg.min())))
            start = None
    if start is not None:
        seg = dd.loc[start:]
        rows.append((start, seg.idxmin(), pd.NaT, float(seg.min())))
    df = pd.DataFrame(rows, columns=["시작", "바닥", "회복", "깊이"])
    df["peak"] = [equity.loc[:s].index[-2] if len(equity.loc[:s]) > 1 else s for s in df["시작"]]
    df = df[df["깊이"] <= -min_depth].sort_values("깊이").head(top)
    df["회복까지(거래일)"] = [int(((equity.index > b) & (equity.index <= r)).sum()) if pd.notna(r) else np.nan
                         for b, r in zip(df["바닥"], df["회복"])]
    return df.drop(columns="peak").reset_index(drop=True)


def label_episode(trough: pd.Timestamp) -> str:
    for name, (a, b) in {"금융위기": ("2007-06-01", "2009-06-30"), "유럽 재정위기": ("2011-04-01", "2011-12-31"),
                         "2015–16 중국·유가": ("2015-05-01", "2016-03-31"), "2018년 4분기": ("2018-09-01", "2019-01-31"),
                         "코로나": ("2020-02-01", "2020-04-30"), "2022 금리 급등": ("2021-11-01", "2023-03-31"),
                         "2025 관세 충격": ("2025-02-01", "2025-05-31")}.items():
        if pd.Timestamp(a) <= trough <= pd.Timestamp(b):
            return name
    return "기타"
