"""User goal metrics: return at a drawdown budget without leverage, and rolling-window checks.

DECISIONS 64-65. A strategy is run at a fraction w (0-1) of the account with the rest in cash
(T-bills); w is the largest value, in 0.01 steps, that keeps the research-period drawdown within
the budget (15%). A strategy whose drawdown is already within the budget keeps w = 1 (no leverage).
The same scaling applied to SPY gives the reference "SPY + T-bill mix" at the same budget.

The blend is rebalanced daily: blended return = w x strategy return + (1 - w) x cash return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS

DD_BUDGET = 0.15
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
WINDOW_YEARS = 3


def _cagr(values: np.ndarray) -> float:
    days = max(len(values) - 1, 1)
    return float((values[-1] / values[0]) ** (TRADING_DAYS / days) - 1.0) if values[-1] > 0 else -1.0


def _mdd(values: np.ndarray) -> float:
    return float((values / np.maximum.accumulate(values) - 1.0).min())


def blend(rets: np.ndarray, cash: np.ndarray, w: float) -> np.ndarray:
    """Equity (starting at 1) of w x asset + (1 - w) x cash, rebalanced daily."""
    return np.concatenate([[1.0], np.cumprod(1.0 + w * rets + (1.0 - w) * cash)])


def _blend_mdd(rets: np.ndarray, cash: np.ndarray, w: float) -> float:
    eq = np.cumprod(1.0 + w * rets + (1.0 - w) * cash)
    peak = np.maximum(np.maximum.accumulate(eq), 1.0)
    return float(min((eq / peak - 1.0).min(), 0.0)) if len(eq) else 0.0


def fit_weight(rets: np.ndarray, cash: np.ndarray, budget: float = DD_BUDGET) -> float:
    """Largest w in WEIGHTS whose blended drawdown is within the budget (0 if none).

    The drawdown of the blend grows with w, so the grid is bisected (7 evaluations instead of 101).
    """
    def ok(k):
        return _blend_mdd(rets, cash, WEIGHTS[k]) >= -budget - 1e-12

    lo, hi = 0, len(WEIGHTS) - 1
    if ok(hi):
        return float(WEIGHTS[hi])
    if not ok(lo):
        return 0.0
    while hi - lo > 1:  # invariant: ok(lo), not ok(hi)
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(WEIGHTS[lo])


def fit_weight_grid(rets: np.ndarray, cash: np.ndarray, budget: float = DD_BUDGET) -> float:
    """Reference implementation: every weight of the grid (used by the tests)."""
    eq = np.cumprod(1.0 + WEIGHTS[:, None] * rets[None, :] + (1.0 - WEIGHTS[:, None]) * cash[None, :], axis=1)
    eq = np.concatenate([np.ones((len(WEIGHTS), 1)), eq], axis=1)
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    ok = np.flatnonzero(mdd >= -budget - 1e-12)
    return float(WEIGHTS[ok[-1]]) if len(ok) else 0.0


def daily_returns(equity: pd.Series, cash: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    eq = equity.dropna()
    r = eq.pct_change(fill_method=None).iloc[1:].fillna(0.0)
    c = cash.reindex(r.index).fillna(0.0)
    return r.index, r.to_numpy(dtype=float), c.to_numpy(dtype=float)


def windows(index: pd.DatetimeIndex, years: int = WINDOW_YEARS) -> list[tuple[int, int]]:
    """Calendar-year windows of `years` full years, stepping one year, inside the index."""
    first = index[0].year + (0 if (index[0].month, index[0].day) <= (1, 5) else 1)
    last = index[-1].year if (index[-1].month, index[-1].day) >= (12, 24) else index[-1].year - 1
    return [(y, y + years - 1) for y in range(first, last - years + 2)]


def window_cagrs(index: pd.DatetimeIndex, rets: np.ndarray, cash: np.ndarray, w: float,
                 wins: list[tuple[int, int]]) -> np.ndarray:
    out = []
    years = index.year
    for a, b in wins:
        sel = (years >= a) & (years <= b)
        out.append(_cagr(blend(rets[sel], cash[sel], w)))
    return np.array(out)


def target_report(equity: pd.Series, spy: pd.Series, cash: pd.Series, n_trades: float,
                  budget: float = DD_BUDGET, min_trades: int = 30) -> dict:
    """Scaled-to-budget metrics of a strategy against the SPY + T-bill mix at the same budget.

    passes = scaled CAGR >= mix CAGR and enough trades (DECISIONS 64);
    window_share = share of rolling windows where the scaled strategy's CAGR >= the mix's,
    both kept at their full-period weights (DECISIONS 65).
    """
    idx, r, c = daily_returns(equity, cash)
    rs = spy.reindex(equity.dropna().index).pct_change(fill_method=None).iloc[1:].fillna(0.0).to_numpy(dtype=float)
    w = fit_weight(r, c, budget)
    w_mix = fit_weight(rs, c, budget)
    eq_s, eq_m = blend(r, c, w), blend(rs, c, w_mix)
    wins = windows(idx)
    ws = window_cagrs(idx, r, c, w, wins)
    wm = window_cagrs(idx, rs, c, w_mix, wins)
    return {
        "dd_weight": w, "dd_cagr": _cagr(eq_s), "dd_mdd": _mdd(eq_s),
        "mix_weight": w_mix, "mix_cagr": _cagr(eq_m),
        "dd_excess": _cagr(eq_s) - _cagr(eq_m),
        "dd_pass": bool(_cagr(eq_s) >= _cagr(eq_m) and n_trades >= min_trades),
        "windows": len(wins),
        "window_share": float((ws >= wm).mean()) if len(wins) else np.nan,
        "worst_window_excess": float((ws - wm).min()) if len(wins) else np.nan,
    }
