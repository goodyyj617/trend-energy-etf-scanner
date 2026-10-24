"""Validation tools for the allocation lab, modelled on the checks the user's friend's tool runs
(DECISIONS A26–A28): timing luck, lazy rebalancing, a never-touched holdout, tuned-vs-equal-weight
cross-validation over 252 splits (CSCV), block-bootstrap confidence intervals and leave-one-out.

Everything here works on daily equity curves, so it applies to static allocations, mixes and the
tactical strategies (lab/aa_tactical.py) alike. Conventions:
- Sharpe is the excess-over-cash Sharpe of daily returns (as in aa_score) unless named otherwise.
- The holdout ("성역") is the last `HOLDOUT_SHARE` of the trading days. Selection steps (CSCV) only
  see the days before it; the holdout row is reported, never used to choose.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from .aa_engine import TRADING_DAYS, AllocResult
from .metrics import curve_metrics

HOLDOUT_SHARE = 0.15
TIMING_OFFSETS = (0, 1, 5, 10, 15)  # trading days after the period end: month end, next day, ~1 week, mid-month, ~3 weeks
CSCV_BLOCKS = 10  # C(10, 5) = 252 splits
LAZY_PROBS = (0.6, 0.3, 0.1)  # after a trade, the next one comes 1, 2 or 3 months later


def _sharpe(r: pd.Series | np.ndarray, cash: pd.Series | np.ndarray | None = None) -> float:
    r = np.asarray(r, dtype=float)
    x = r - (np.asarray(cash, dtype=float) if cash is not None else 0.0)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if len(x) > 2 and sd > 0 else float("nan")


def headline(res_or_eq, cash: pd.Series | None = None) -> dict[str, float]:
    eq = res_or_eq.equity if isinstance(res_or_eq, AllocResult) else res_or_eq
    cash = res_or_eq.cash if isinstance(res_or_eq, AllocResult) else cash
    m = curve_metrics(eq)
    r = eq.pct_change(fill_method=None).dropna()
    c = cash.reindex(r.index).fillna(0.0) if cash is not None else None
    return {"cagr": m["cagr"], "mdd": m["mdd"], "sharpe": _sharpe(r, c), "calmar": m["calmar"]}


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """Year x month returns (columns 1..12) plus the calendar-year return in column "연간"."""
    me = equity.groupby(equity.index.to_period("M")).last()
    first = equity.iloc[0]
    r = me / me.shift(1).fillna(first) - 1.0
    df = pd.DataFrame({"year": r.index.year, "month": r.index.month, "r": r.to_numpy()})
    table = df.pivot(index="year", columns="month", values="r").reindex(columns=range(1, 13))
    ye = equity.groupby(equity.index.year).last()
    table["연간"] = ye / ye.shift(1).fillna(first) - 1.0
    return table


# --------------------------------------------------------------------------- timing luck

def timing_table(run: Callable[[int, int], AllocResult], offsets=TIMING_OFFSETS, delays=(0, 1)) -> pd.DataFrame:
    """run(offset, delay) -> result. One row per (offset, delay) with CAGR / MDD / Sharpe."""
    rows = []
    for d in delays:
        for o in offsets:
            if d and o:  # the delay row is shown at the base timing only
                continue
            try:
                h = headline(run(o, d))
            except ValueError:
                continue
            label = ("말일" if o == 0 else f"말일 +{o}거래일") + (f", {d}개월 지연" if d else "")
            rows.append({"타이밍": label, "offset": o, "delay": d, **h})
    return pd.DataFrame(rows)


def timing_summary(table: pd.DataFrame) -> dict[str, float]:
    base = table[(table["offset"] == 0) & (table["delay"] == 0)]
    same = table[table["delay"] == 0]
    return {"mdd_base": float(base["mdd"].iloc[0]) if len(base) else float("nan"),
            "mdd_worst": float(same["mdd"].min()), "mdd_best": float(same["mdd"].max()),
            "cagr_min": float(same["cagr"].min()), "cagr_max": float(same["cagr"].max()),
            "sharpe_spread": float(same["sharpe"].max() - same["sharpe"].min())}


def lazy_skip_months(months: list[str], rng: np.random.Generator, probs=LAZY_PROBS) -> tuple[str, ...]:
    """Months whose decision is skipped when the next trade comes 1/2/3 months after the last one."""
    skip, i = [], 0
    while i < len(months):
        gap = int(rng.choice([1, 2, 3], p=np.asarray(probs) / sum(probs)))
        skip.extend(months[i + 1:i + gap])
        i += gap
    return tuple(skip)


def lazy_table(run: Callable[[tuple], AllocResult], months: list[str], n: int = 50, seed: int = 7,
               probs=LAZY_PROBS) -> pd.DataFrame:
    """run(skip_months) -> result, for n random lazy schedules (fixed seed). Rows per draw."""
    rng = np.random.default_rng(seed)
    rows = [headline(run(lazy_skip_months(months, rng, probs))) for _ in range(n)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- holdout

def holdout_split(index: pd.DatetimeIndex, share: float = HOLDOUT_SHARE) -> pd.Timestamp:
    """First day of the holdout (the last `share` of trading days)."""
    return index[int(len(index) * (1.0 - share))]


def split_metrics(res: AllocResult, share: float = HOLDOUT_SHARE) -> pd.DataFrame:
    cut = holdout_split(res.equity.index, share)
    eq = res.equity
    out = {"선정 구간": headline(eq.loc[:cut], res.cash), "성역(홀드아웃)": headline(eq.loc[cut:], res.cash),
           "전체": headline(eq, res.cash)}
    df = pd.DataFrame(out).T
    df["기간"] = [f"{eq.index[0].date()} – {cut.date()}", f"{cut.date()} – {eq.index[-1].date()}",
                f"{eq.index[0].date()} – {eq.index[-1].date()}"]
    return df


# --------------------------------------------------------------------------- bootstrap and leave-one-out

def block_bootstrap(res: AllocResult, n: int = 1000, block: int = 21, seed: int = 11) -> pd.DataFrame:
    """Stationary block bootstrap (mean block length `block` days) of daily returns and cash together.
    Returns the 5 / 50 / 95 percentiles of Sharpe, CAGR and MDD."""
    r = res.equity.pct_change(fill_method=None).dropna()
    c = res.cash.reindex(r.index).fillna(0.0).to_numpy()
    x = r.to_numpy()
    T = len(x)
    rng = np.random.default_rng(seed)
    sh, cg, dd = np.empty(n), np.empty(n), np.empty(n)
    p = 1.0 / block
    t = np.arange(T)
    for k in range(n):
        new = rng.random(T) < p
        new[0] = True
        block_id = np.cumsum(new) - 1
        block_start = t[new]
        idx = (rng.integers(T, size=len(block_start))[block_id] + t - block_start[block_id]) % T
        xs, cs = x[idx], c[idx]
        sh[k] = _sharpe(xs, cs)
        eq = np.cumprod(1.0 + xs)
        cg[k] = eq[-1] ** (TRADING_DAYS / T) - 1.0
        dd[k] = (eq / np.maximum.accumulate(eq) - 1.0).min()
    q = [5, 50, 95]
    return pd.DataFrame({"샤프": np.percentile(sh, q), "CAGR": np.percentile(cg, q), "MDD": np.percentile(dd, q)},
                        index=["5%", "50%", "95%"])


def leave_one_out(parts: list[tuple[str, float]], run: Callable[[list[tuple[str, float]]], AllocResult]) -> pd.DataFrame:
    """Drop each part of a blend in turn (the rest keep their relative weights)."""
    rows = [{"제외": "(없음)", **headline(run(parts))}]
    if len(parts) > 1:
        for i, (name, _) in enumerate(parts):
            rest = [p for j, p in enumerate(parts) if j != i]
            rows.append({"제외": name, **headline(run(rest))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- tuned vs equal weight (CSCV)

def weight_grid(n_assets: int, step: float = 0.1, max_k: int = 3) -> list[np.ndarray]:
    """All weight vectors over n_assets that use 1..max_k assets with weights on a `step` grid."""
    units = int(round(1.0 / step))
    out = []
    for k in range(1, min(max_k, n_assets) + 1):
        for combo in combinations(range(n_assets), k):
            for parts in _compositions(units, k):
                w = np.zeros(n_assets)
                w[list(combo)] = np.asarray(parts) / units
                out.append(w)
    return out


def _compositions(total: int, k: int):
    if k == 1:
        yield (total,)
        return
    for first in range(1, total - k + 2):
        for rest in _compositions(total - first, k - 1):
            yield (first, *rest)


def cscv(sleeve_rets: pd.DataFrame, cash: pd.Series, step: float = 0.1, max_k: int = 3,
         blocks: int = CSCV_BLOCKS) -> dict:
    """Combinatorially symmetric cross-validation of choosing blend weights by Sharpe.

    sleeve_rets: daily returns of the candidate strategies (columns) on the selection window (the
    holdout already removed). Blends are daily-weighted averages of the sleeve returns (a close
    stand-in for monthly-rebalanced sleeves). For each of the C(blocks, blocks/2) splits the blend
    with the best Sharpe on the training half is compared on the other half with
    (a) the equal-weight blend of the same sleeves ("균등"), and (b) the best blend on that half
    ("천장", hindsight). Returns the per-split table and the summary numbers:
    win_rate = share of splits where the tuned blend beats its equal-weight version out of sample,
    pbo = share of splits where the tuned blend ranks in the bottom half of all blends out of sample.
    """
    R = sleeve_rets.fillna(0.0).to_numpy()
    C = cash.reindex(sleeve_rets.index).fillna(0.0).to_numpy()
    grid = weight_grid(R.shape[1], step, max_k)
    W = np.vstack(grid)  # candidates x sleeves
    eqw = (W > 0) / (W > 0).sum(axis=1, keepdims=True)
    T = len(R)
    edges = np.linspace(0, T, blocks + 1).astype(int)
    block_of = np.searchsorted(edges[1:], np.arange(T), side="right")
    P = R @ W.T - C[:, None]  # excess daily returns, days x candidates
    E = R @ eqw.T - C[:, None]
    rows = []
    for train in combinations(range(blocks), blocks // 2):
        mask = np.isin(block_of, train)
        sh_in = _col_sharpe(P[mask])
        sh_out = _col_sharpe(P[~mask])
        best = int(np.nanargmax(sh_in))
        eq_out = _col_sharpe(E[~mask][:, [best]])[0]
        rank = float((sh_out < sh_out[best]).mean())  # share of candidates it beats out of sample
        rows.append({"tuned_in": sh_in[best], "tuned_out": sh_out[best], "equal_out": eq_out,
                     "ceiling_out": float(np.nanmax(sh_out)), "oos_rank": rank,
                     "pick": _describe(W[best], sleeve_rets.columns)})
    df = pd.DataFrame(rows)
    picks = df["pick"].value_counts()
    return {"table": df, "candidates": len(W), "splits": len(df),
            "win_rate": float((df["tuned_out"] > df["equal_out"]).mean()),
            "pbo": float((df["oos_rank"] < 0.5).mean()),
            "tuned_out_median": float(df["tuned_out"].median()), "equal_out_median": float(df["equal_out"].median()),
            "ceiling_out_median": float(df["ceiling_out"].median()),
            "top_picks": picks.head(5)}


def _col_sharpe(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, X.mean(axis=0) / sd * np.sqrt(TRADING_DAYS), np.nan)


def _describe(w: np.ndarray, names) -> str:
    return " · ".join(f"{n} {x * 100:.0f}" for n, x in zip(names, w) if x > 0)


def blend_equity(sleeve_eq: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily-weighted blend of sleeve equity curves (for quick comparisons)."""
    r = sleeve_eq.pct_change(fill_method=None).fillna(0.0)
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    return (1.0 + (r[w.index] * w).sum(axis=1)).cumprod()


__all__ = ["HOLDOUT_SHARE", "TIMING_OFFSETS", "timing_table", "timing_summary", "lazy_table", "lazy_skip_months",
           "holdout_split", "split_metrics", "block_bootstrap", "leave_one_out", "weight_grid", "cscv",
           "blend_equity", "headline", "monthly_table"]
