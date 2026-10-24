"""Multi-variable exploration: Sobol sampling of the strategy space and the probability of
backtest overfitting (PBO) by combinatorially symmetric cross-validation (DECISIONS 66).

Instead of an exhaustive grid (2 parameters at a time), a fixed budget of points is spread
evenly over several variables at once with a Sobol sequence. Every point is a backtest and is
logged as a trial. PBO then asks whether "pick the best point" generalises: the research period
is cut into S blocks; for every way to choose half of them as the in-sample part, the best point
in-sample is located and its rank in the other half is recorded. PBO = share of splits where
that point falls in the bottom half out of sample (Bailey, Borwein, Lopez de Prado, Zhu 2015).
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd

from .blocks import ENTRY_BLOCKS, EXIT_BLOCKS
from .engine import StrategyConfig
from .families import ENTRY_MENU, EXIT_MENU

# --------------------------------------------------------------------------- Sobol sequence

# Joe & Kuo (2008) direction numbers for dimensions 2-10: (degree s, coefficient a, initial m).
_JOE_KUO = [
    (1, 0, [1]), (2, 1, [1, 3]), (3, 1, [1, 3, 1]), (3, 2, [1, 1, 1]), (4, 1, [1, 1, 3, 3]),
    (4, 4, [1, 3, 5, 13]), (5, 2, [1, 1, 5, 5, 17]), (5, 4, [1, 1, 5, 5, 5]), (5, 7, [1, 1, 7, 11, 19]),
]
_BITS = 30


def _directions(dim: int) -> np.ndarray:
    v = np.zeros((dim, _BITS), dtype=np.int64)
    v[0] = [1 << (_BITS - 1 - i) for i in range(_BITS)]  # dimension 1: van der Corput
    for d in range(1, dim):
        s, a, m = _JOE_KUO[d - 1]
        for i in range(_BITS):
            if i < s:
                v[d, i] = m[i] << (_BITS - 1 - i)
            else:
                x = v[d, i - s] ^ (v[d, i - s] >> s)
                for k in range(1, s):
                    if (a >> (s - 1 - k)) & 1:
                        x ^= v[d, i - k]
                v[d, i] = x
    return v


def sobol(n: int, dim: int) -> np.ndarray:
    """First n points of the (unscrambled) Sobol sequence in [0, 1)^dim, Gray-code order."""
    if not 1 <= dim <= len(_JOE_KUO) + 1:
        raise ValueError(f"dim must be 1-{len(_JOE_KUO) + 1}")
    v = _directions(dim)
    out = np.zeros((n, dim))
    x = np.zeros(dim, dtype=np.int64)
    for i in range(1, n):
        c = ((~(i - 1)) & i).bit_length() - 1  # rightmost zero bit of i - 1 (Gray-code order)
        x ^= v[:, c]
        out[i] = x / float(1 << _BITS)
    return out


# --------------------------------------------------------------------------- the search space (DECISIONS 66)

ENTRIES = list(ENTRY_MENU)
EXITS = list(EXIT_MENU)
FILTERS = [None, "market_trend", "ma_stack", "obv_up", "volume_surge", "adx_trend", "near_high"]
SPACE = ["entry", "entry_value", "exit", "exit_value", "filter", "stop", "max_positions", "sizing"]


def _pick(u: float, options: list):
    return options[min(int(u * len(options)), len(options) - 1)]


def _value(block, lo: float, hi: float, u: float) -> float:
    """Continuous draw in [lo, hi] snapped to the block parameter's step."""
    p = block.param
    v = lo + u * (hi - lo)
    v = round(round((v - p.minimum) / p.step) * p.step + p.minimum, 6)
    return float(min(max(v, lo), hi))


def point_config(u: np.ndarray, base: StrategyConfig) -> StrategyConfig:
    """Map one point of [0, 1)^8 to a strategy. Ranges are the menu-v3 ranges of each block."""
    e = _pick(u[0], list(ENTRY_MENU))  # read at call time: a reduced menu (dry runs) is honoured
    x = _pick(u[2], list(EXIT_MENU))
    entries = {e: _value(ENTRY_BLOCKS[e], min(ENTRY_MENU[e]), max(ENTRY_MENU[e]), u[1])}
    exits = {x: _value(EXIT_BLOCKS[x], min(EXIT_MENU[x]), max(EXIT_MENU[x]), u[3])}
    f = _pick(u[4], FILTERS)
    if f and f not in entries:
        b = ENTRY_BLOCKS[f]
        entries[f] = None if b.param is None else b.param.default
    if u[5] >= 0.5:  # half of the points carry an initial stop of 5-20%
        exits["stop_loss_pct"] = float(round(5 + (u[5] - 0.5) / 0.5 * 15))
    return replace(base, entries=entries, exits=exits, max_positions=int(5 + min(int(u[6] * 16), 15)),
                   sizing="equal" if u[7] < 0.5 else "inverse_vol")


# --------------------------------------------------------------------------- PBO (CSCV)

def budget_return(monthly: np.ndarray, budget: float = 0.15) -> np.ndarray:
    """Approximate return at a drawdown budget without leverage, per column of monthly returns:
    min(CAGR, budget x CAGR / |MDD|). Used only to rank points inside the PBO splits."""
    eq = np.cumprod(1.0 + monthly, axis=0)
    eq = np.vstack([np.ones((1, monthly.shape[1])), eq])
    years = monthly.shape[0] / 12.0
    cagr = np.where(eq[-1] > 0, eq[-1] ** (1.0 / years) - 1.0, -1.0)
    mdd = (eq / np.maximum.accumulate(eq, axis=0) - 1.0).min(axis=0)
    calmar_scaled = np.where(mdd < 0, budget * cagr / np.maximum(-mdd, 1e-12), cagr)
    return np.minimum(cagr, calmar_scaled)


def pbo(monthly: pd.DataFrame, blocks: int = 12, metric=budget_return) -> dict:
    """Probability of backtest overfitting of 'choose the best column' by CSCV.

    monthly: months x strategies. Rows are cut into `blocks` contiguous blocks (blocks even);
    every half of the blocks is an in-sample set, the rest out-of-sample.
    """
    m = monthly.to_numpy(dtype=float)
    m = np.nan_to_num(m, nan=0.0)
    parts = np.array_split(np.arange(len(m)), blocks)
    n = m.shape[1]
    logits, oos_ranks = [], []
    for chosen in combinations(range(blocks), blocks // 2):
        is_rows = np.concatenate([parts[i] for i in chosen])
        oos_rows = np.concatenate([parts[i] for i in range(blocks) if i not in chosen])
        best = int(np.nanargmax(metric(m[is_rows])))
        oos = metric(m[oos_rows])
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1) + 1  # 1 = worst
        w = rank / (n + 1)
        oos_ranks.append(w)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "splits": len(logits),
            "median_oos_rank": float(np.median(oos_ranks)), "blocks": blocks}
