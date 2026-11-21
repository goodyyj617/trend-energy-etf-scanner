"""A46: probabilistic MDD verdict (allocation DECISIONS A46, design `docs/research/allocation/2026-11-11_a46_design.md`,
fixed before the run). The MDD of one historical path is replaced by MDD distributions over assumptions (M1: price
sets × decision dates × cost), resampled paths (M2: stationary block bootstrap, pre-tax) and the same paths under the
declared account policy (M3, after tax).

Pieces:

  ``stationary_indices``  n × T row numbers of a circular stationary block bootstrap (Politis–Romano): each row
                          continues the previous block with probability 1 − 1/L, otherwise starts at a random row
  ``boot_run``            `aa_canary.DynamicSim.run` with every path reading its own day's growth row
                          (`growth[idx[p, t]]`); idx None (or 0 … T−1) gives `DynamicSim.run` itself
  ``static_inputs``       decide / choice / first for static allocations at the given quarter rows
  ``vol_levels_many``     `aa_layers.vol_levels` for n return rows at once, with an optional decision offset
                          (month end + n trading days)
  ``vol_inputs``          decide / choice / first for the volatility target (as `aa_layers.compose` with no signal)
  ``mean_tabs``           per-path same-average-exposure static: static row × k̄ + the CD-rate ETF × (1 − k̄)
  ``zone_of``             the zone rule on a set of MDDs (median and lower 10 %)
  ``layer_of``            the layer verdict from the paired MDD differences of M2 and M3
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import aa_canary as K
from . import aa_layers as LY
from .aa_engine import TRADING_DAYS, decision_rows

SEED = 20261111
BLOCK = 126
BLOCKS = (63, 126, 252)
N_PATHS = 1000
LINE, NEAR, TAIL = -0.15, -0.17, -0.20
TAIL_Q = 0.10
LAYER_PP, LAYER_SHARE = 0.02, 0.75
VOL_OFFSETS = (0, 5, 10, 15)


def stationary_indices(T: int, n: int, mean_block: float, seed: int) -> np.ndarray:
    """n × T row numbers: a circular stationary bootstrap with mean block length ``mean_block`` (geometric lengths)."""
    if T < 2 or mean_block < 1:
        raise ValueError("T ≥ 2 and mean_block ≥ 1")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T, size=(n, T))
    new = rng.random((n, T)) < 1.0 / mean_block
    new[:, 0] = True
    idx = np.empty((n, T), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def identity_indices(T: int, n: int = 1) -> np.ndarray:
    return np.tile(np.arange(T, dtype=np.int64), (n, 1))


def boot_run(sim: K.DynamicSim, idx: np.ndarray | None, tabs: np.ndarray, decide: np.ndarray, choice: np.ndarray,
             first: np.ndarray) -> np.ndarray:
    """Equity paths (P × T) of `DynamicSim.run` where path p's day t has the growth of row idx[p, t] (same
    arithmetic, same order; idx None = the historical order for every path)."""
    tabs = np.asarray(tabs, dtype=float)
    P, T = tabs.shape[0], sim.n_days
    decide, choice = np.asarray(decide, dtype=bool), np.asarray(choice, dtype=int)
    if decide.shape != (P, T) or choice.shape != (P, T):
        raise ValueError("decide·choice must be (paths, days)")
    if idx is not None:
        idx = np.asarray(idx)
        if idx.shape != (P, T):
            raise ValueError("idx must be (paths, days)")
        cols = np.ascontiguousarray(idx.T)
    rows = np.arange(P)
    hold = tabs[rows, np.asarray(first, dtype=int)].copy()
    pend = np.full(P, -1)
    paths = np.empty((P, T))
    for t in range(T):
        hold *= sim.growth[t] if idx is None else sim.growth[cols[t]]
        value = hold.sum(axis=1)
        has = pend >= 0
        if has.any():
            h = np.flatnonzero(has)
            w = tabs[h, pend[h]]
            tr = np.abs(w * value[h, None] - hold[h]).sum(axis=1) / 2.0
            value[h] = value[h] - tr * sim.cost
            hold[h] = w * value[h, None]
            pend[h] = -1
        paths[:, t] = value
        d = decide[:, t]
        pend[d] = choice[d, t]
    return paths


def static_inputs(P: int, T: int, qrows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decide = np.zeros((P, T), dtype=bool)
    q = np.asarray(qrows, dtype=int)
    decide[:, q[(q >= 0) & (q < T - 1)]] = True
    return decide, np.zeros((P, T), dtype=int), np.zeros(P, dtype=int)


def base_returns(paths: np.ndarray) -> np.ndarray:
    """Daily returns of equity paths with the first day's return set to 0 (as stage 5's base_r)."""
    paths = np.atleast_2d(paths)
    return np.concatenate([np.zeros((len(paths), 1)), paths[:, 1:] / paths[:, :-1] - 1.0], axis=1)


def vol_levels_many(R: np.ndarray, dates: pd.DatetimeIndex, target: float, offset: int = 0) -> np.ndarray:
    """`aa_layers.vol_levels` for each row of R (n × T): the 60-day volatility (pandas rolling, the same arithmetic),
    a decision at each month end + ``offset`` rows, k = min(1, target / σ) in 0.05 steps, held from the next row."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    n, T = R.shape
    vol = pd.DataFrame(R.T).rolling(LY.VOL_WINDOW, min_periods=LY.VOL_WINDOW).std().to_numpy().T * np.sqrt(TRADING_DAYS)
    top = len(LY.K_LEVELS) - 1
    rows = np.unique(decision_rows(dates, "M") + int(offset))
    rows = rows[(rows >= 0) & (rows < T)]
    cur = np.full(n, top)
    after = np.empty((n, len(rows)), dtype=int)
    for j, d in enumerate(rows):
        v = vol[:, d]
        ok = np.isfinite(v) & (v > 0)
        k = np.minimum(1.0, target / np.where(ok, v, 1.0))
        new = np.clip(np.round(k / LY.K_STEP) - 1, 0, top).astype(int)
        cur = np.where(ok, new, cur)
        after[:, j] = cur
    j = np.searchsorted(rows, np.arange(T), side="left")  # decisions strictly before row t
    lev = np.full((n, T), top)
    has = j > 0
    lev[:, has] = after[:, j[has] - 1]
    return lev


def vol_inputs(lev: np.ndarray, qrows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """decide / choice / first (n × T) for the volatility target on a static allocation (scaled tables, one state)."""
    lev = np.atleast_2d(np.asarray(lev, dtype=int))
    n, T = lev.shape
    decide, _, _ = static_inputs(n, T, qrows)
    decide[:, :-1] |= lev[:, 1:] != lev[:, :-1]
    decide[:, T - 1] = False
    choice = np.concatenate([lev[:, 1:], lev[:, -1:]], axis=1)
    return decide, choice, lev[:, 0].copy()


def mean_exposure(lev: np.ndarray) -> np.ndarray:
    """Average exposure k held over the days, per row."""
    return LY.K_LEVELS[np.atleast_2d(np.asarray(lev, dtype=int))].mean(axis=1)


def mean_tabs(row: np.ndarray, columns: list[str], kbar: np.ndarray) -> np.ndarray:
    """n × 1 × columns: the static row × k̄ + the CD-rate ETF × (1 − k̄), one per k̄."""
    kbar = np.atleast_1d(np.asarray(kbar, dtype=float))
    cd = np.zeros(len(columns))
    cd[columns.index(LY.CD)] = 1.0
    return (kbar[:, None] * np.asarray(row, float)[None, :] + (1.0 - kbar)[:, None] * cd[None, :])[:, None, :]


def mean_weights(w: dict[str, float], codes: list[str], kbar: float) -> np.ndarray:
    """Same-average-exposure weights (fractions) in ``codes`` order for the account simulator."""
    tot = sum(w.values())
    v = np.array([w.get(c, 0.0) / tot for c in codes]) * kbar
    v[codes.index(LY.CD)] += 1.0 - kbar
    return v


def zone_of(mdd) -> dict:
    """The zone rule: median ≥ −15 % and lower 10 % ≥ −20 % → 안; else median ≥ −17 % → 경계; else 밖."""
    x = np.asarray(mdd, dtype=float)
    x = x[np.isfinite(x)]
    med, q10 = float(np.median(x)), float(np.quantile(x, TAIL_Q))
    z = "안" if med >= LINE and q10 >= TAIL else "경계" if med >= NEAR else "밖"
    return {"n": int(len(x)), "median": med, "q10": q10, "worst": float(x.min()), "best": float(x.max()),
            "p15": float(np.mean(x >= LINE)), "p17": float(np.mean(x >= NEAR)), "p20": float(np.mean(x >= TAIL)),
            "zone": z}


ZONE_RANK = {"안": 0, "경계": 1, "밖": 2}


def worst_zone(*zones: str | None) -> str:
    zs = [z for z in zones if z]
    return max(zs, key=ZONE_RANK.__getitem__)


def percentile_of(value: float, dist) -> float:
    """Share of the distribution at or below ``value`` (0 = the deepest, 1 = the shallowest)."""
    d = np.asarray(dist, dtype=float)
    return float(np.mean(d <= value))


def layer_stats(d_mdd, d_cagr=None) -> dict:
    d = np.asarray(d_mdd, dtype=float)
    out = {"n": int(len(d)), "median": float(np.median(d)), "share_pos": float(np.mean(d > 0)),
           "q10": float(np.quantile(d, 0.10)), "q90": float(np.quantile(d, 0.90))}
    if d_cagr is not None:
        out["cagr_median"] = float(np.median(np.asarray(d_cagr, dtype=float)))
    out["holds"] = bool(out["median"] >= LAYER_PP and out["share_pos"] >= LAYER_SHARE)
    return out


def layer_of(m2: dict, m3: dict | None) -> str:
    """층 확인 if M2 and M3 both hold (median ΔMDD ≥ 2 %p and > 0 in ≥ 75 % of paths); 층 없음 if either median
    ≤ 0; else 층 약함."""
    parts = [m2] + ([m3] if m3 is not None else [])
    if any(p["median"] <= 0 for p in parts):
        return "층 없음"
    return "층 확인" if all(p["holds"] for p in parts) else "층 약함"


__all__ = ["SEED", "BLOCK", "BLOCKS", "N_PATHS", "LINE", "NEAR", "TAIL", "VOL_OFFSETS", "stationary_indices",
           "identity_indices", "boot_run", "static_inputs", "base_returns", "vol_levels_many", "vol_inputs",
           "mean_exposure", "mean_tabs", "mean_weights", "zone_of", "worst_zone", "percentile_of", "layer_stats",
           "layer_of"]
