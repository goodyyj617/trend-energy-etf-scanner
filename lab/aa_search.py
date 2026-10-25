"""Allocation lab: the global static search (DECISIONS A31).

Tools for scanning hundreds of thousands of static allocations quickly and judging them without
hard filters other than the full-period MDD line:

- ``StaticSim``: a vectorised static simulator that reproduces ``aa_engine.run_allocation`` for
  fixed weights (decide at the period's last close, trade at the next close, one-way cost, drift
  between trades; no overlay, no band, no leverage factor) for thousands of weight vectors at once.
- ``round_weights`` / ``sample_two_level`` / ``sample_sparse`` / ``neighbours``: the samplers of A31
  (class-level Sobol × product-level random picks; a sparse check sample; ±5 %p refinement moves).
- ``summarize``: CAGR, MDD, Sharpe, Calmar, window drawdowns, half-period and late-start figures,
  holdout figures for a block of equity paths.
- ``rolling_window_share``: share of 5-year rolling windows whose MDD stays within a line.
- ``pbo_selection``: CSCV over monthly returns with the rule "best CAGR among candidates within
  the MDD line" (PBO, how often each candidate is picked, how often it stays within the line
  out of sample).

Everything here is for research scripts (``scripts/aa_global_search.py``); the Streamlit apps do not
import it.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from lab.aa_assets import CASH, short
from lab.aa_engine import TRADING_DAYS, decision_rows
from lab.data import Panel
from lab.explore import sobol

MIN_DAYS = 30


# --------------------------------------------------------------------------- fast simulator

class StaticSim:
    """Simulate many static allocations over the same dates in one pass.

    ``syms`` are the risky columns (``CASH`` is implicit as the last weight column). Weight rows passed
    to ``run`` are fractions summing to 1 with the cash share last. Raises ``ValueError`` when a symbol's
    data begins after ``start`` (the engine would silently move the start date; a search must not mix
    windows).
    """

    def __init__(self, panel: Panel, syms: list[str], start: str, end: str, rebalance: str = "Q",
                 cost_bps: float = 10.0, rebalance_offset: int = 0):
        missing = [s for s in syms if s not in panel.symbols]
        if missing:
            raise ValueError(f"가격 데이터가 없는 자산: {', '.join(short(s) for s in missing)}")
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        close_all = panel.close[list(syms)]
        first_valid = close_all.apply(lambda s: s.first_valid_index())
        late = [s for s in syms if pd.isna(first_valid[s]) or first_valid[s] > start_ts]
        if late:
            raise ValueError("시작일 이후에 데이터가 시작되는 자산: " + ", ".join(f"{short(s)} [{s}]" for s in late))
        idx = panel.close.index
        dates = idx[(idx >= start_ts) & (idx <= end_ts)]
        if len(dates) < MIN_DAYS:
            raise ValueError("선택한 기간에 거래일이 30일 미만입니다.")
        close = close_all.loc[:dates[-1]].ffill()
        rets = close.pct_change(fill_method=None).loc[dates]
        cash_full = panel.cash if panel.cash is not None else pd.Series(0.0, index=idx)
        cash = cash_full.reindex(idx).fillna(0.0).loc[dates]
        R = np.nan_to_num(rets.to_numpy(dtype=float), nan=0.0)
        growth = np.concatenate([1.0 + R, (1.0 + cash.to_numpy(dtype=float))[:, None]], axis=1)
        # cumulative growth index: G[0] = 1 (before the first day), G[t + 1] = value after day t
        self._G = np.vstack([np.ones((1, growth.shape[1])), np.cumprod(growth, axis=0)])
        rows = decision_rows(dates, rebalance) + int(rebalance_offset)  # A27 timing check: trade later
        rows = rows[rows < len(dates) - 1]
        self._trade_rows = (rows + 1).astype(int)
        self.syms = list(syms)
        self.columns = self.syms + [CASH]
        self.dates = dates
        self.cash = cash
        self.rebalance = rebalance
        self.rebalance_offset = int(rebalance_offset)
        self.cost = cost_bps / 1e4
        self.n_days = len(dates)

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def run(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Equity paths (K × days, starting at the value after day 0) and annual one-way turnover (K,)."""
        W = np.asarray(W, dtype=float)
        if W.ndim == 1:
            W = W[None, :]
        if W.shape[1] != self.n_cols:
            raise ValueError(f"비중 열 수가 {self.n_cols}(자산 + 현금)이어야 합니다: {W.shape[1]}")
        K, T, G = W.shape[0], self.n_days, self._G
        paths = np.empty((K, T))
        turn = np.zeros(K)
        value = np.ones(K)  # value right after the latest trade (1.0 before day 0)
        starts = [-1] + self._trade_rows.tolist()
        for j, s in enumerate(starts):
            trade = j + 1 < len(starts)
            e = starts[j + 1] if trade else T - 1  # last row of this segment (a trade row, or the end)
            seg = G[s + 2:e + 2] / G[s + 1]  # growth from after day s to after each day in (s, e]
            block = (W @ seg.T) * value[:, None]  # pre-trade values
            paths[:, s + 1:e + 1] = block
            if trade:
                pre = block[:, -1]
                prev = block[:, -2] if block.shape[1] >= 2 else value  # the decision day's value
                hold = W * seg[-1] * value[:, None]
                tr = np.abs(W * pre[:, None] - hold).sum(axis=1) / 2.0
                value = pre - tr * self.cost
                paths[:, e] = value
                turn += tr / prev
        years = max(T, 1) / TRADING_DAYS
        return paths, turn / years


# --------------------------------------------------------------------------- samplers

def round_weights(w: np.ndarray, min_pct: int = 3) -> np.ndarray:
    """Whole percents summing to 100: parts under ``min_pct`` are dropped, the rest renormalised and
    rounded by largest remainder."""
    w = np.asarray(w, dtype=float).copy()
    w[~np.isfinite(w) | (w < 0)] = 0.0
    if w.sum() <= 0:
        raise ValueError("비중의 합이 0입니다.")
    w = w / w.sum()
    w[w < min_pct / 100.0] = 0.0
    if w.sum() <= 0:  # everything was tiny: keep the single largest part
        out = np.zeros(len(w), dtype=np.int64)
        out[int(np.argmax(np.asarray(w)))] = 100
        return out
    w = w / w.sum() * 100.0
    floors = np.floor(w).astype(np.int64)
    rest = int(100 - floors.sum())
    if rest > 0:
        order = np.argsort(-(w - floors), kind="stable")
        floors[order[:rest]] += 1
    return floors


def _dirichlet_from_uniform(u: np.ndarray) -> np.ndarray:
    """Dirichlet(1, …, 1) points from uniforms (rows): normalised exponential spacings."""
    e = -np.log(np.clip(1.0 - u, 1e-12, 1.0 - 1e-12))
    return e / e.sum(axis=1, keepdims=True)


def sample_two_level(classes: dict[str, list[str]], n_class: int, picks: int, seed: int, lev_class: str | None,
                     lev_cap: float, equity_classes: tuple[str, ...], min_pct: int = 3,
                     count_probs: tuple[float, float, float] = (0.5, 0.35, 0.15), max_products: int = 10
                     ) -> tuple[np.ndarray, dict]:
    """A31 two-level sampler.

    Level 1 (``n_class`` Sobol points): the leverage class gets 0 in every other point and a uniform
    share on (0, ``lev_cap``] in the rest; the remaining share is Dirichlet(1) over the other classes.
    Level 2 (``picks`` draws per point): each class with weight holds 1 / 2 / 3 products with
    probability ``count_probs`` (3 only in ``equity_classes``, never more than the class has), uniform
    picks, Dirichlet(1) split. Weights are rounded with ``round_weights``; if more than ``max_products``
    products remain, the smallest are dropped first. Returns (W in whole percents, meta) with
    ``meta["symbols"]`` in class order and ``meta["class_point"]`` the level-1 index of each row.
    """
    names = list(classes)
    symbols = [s for c in names for s in classes[c]]
    col = {s: i for i, s in enumerate(symbols)}
    others = [c for c in names if c != lev_class]
    dim = len(others) + (1 if lev_class else 0)
    U = sobol(n_class + 1, dim)[1:]  # the first Sobol point is the origin: degenerate for the simplex map
    rng = np.random.default_rng(seed)
    class_w = np.zeros((n_class, len(names)))
    base = _dirichlet_from_uniform(U[:, :len(others)]) if len(others) > 1 else np.ones((n_class, 1))
    if lev_class:
        lev_share = np.where(np.arange(n_class) % 2 == 0, 0.0, (1.0 - U[:, -1]) * lev_cap)
    else:
        lev_share = np.zeros(n_class)
    for k, c in enumerate(others):
        class_w[:, names.index(c)] = base[:, k] * (1.0 - lev_share)
    if lev_class:
        class_w[:, names.index(lev_class)] = lev_share
    probs = np.asarray(count_probs, dtype=float)
    probs = probs / probs.sum()
    rows = []
    point = []
    for i in range(n_class):
        for _ in range(picks):
            w = np.zeros(len(symbols))
            for c_idx, c in enumerate(names):
                share = class_w[i, c_idx]
                if share <= 0:
                    continue
                members = classes[c]
                cap = 3 if c in equity_classes else 2
                cap = min(cap, len(members))
                n = int(rng.choice(np.arange(1, 4), p=probs))
                n = min(n, cap)
                chosen = rng.choice(len(members), size=n, replace=False)
                split = rng.dirichlet(np.ones(n))
                for m, f in zip(chosen, split):
                    w[col[members[m]]] += share * f
            rows.append(_cap_products(w, max_products, min_pct))
            point.append(i)
    W = np.vstack(rows).astype(np.int64)
    if lev_class:
        W = _enforce_lev_cap(W, [col[s] for s in classes[lev_class]], int(round(lev_cap * 100)), min_pct)
    meta = {"symbols": symbols, "classes": names, "class_point": np.asarray(point),
            "class_weights": class_w, "class_of": {s: c for c in names for s in classes[c]}}
    return W, meta


def _cap_products(w: np.ndarray, max_products: int, min_pct: int) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    out = round_weights(w, min_pct)
    while (out > 0).sum() > max_products:
        live = np.where(out > 0)[0]
        w[live[np.argmin(out[live])]] = 0.0
        out = round_weights(w, min_pct)
    return out


def _enforce_lev_cap(W: np.ndarray, lev_cols: list[int], cap_pct: int, min_pct: int) -> np.ndarray:
    """Rounding can push the leverage total 1 %p over the cap: move the excess to the largest
    non-leverage product."""
    W = W.copy()
    other = [j for j in range(W.shape[1]) if j not in lev_cols]
    for i in np.where(W[:, lev_cols].sum(axis=1) > cap_pct)[0]:
        excess = int(W[i, lev_cols].sum() - cap_pct)
        while excess > 0:
            j = lev_cols[int(np.argmax(W[i, lev_cols]))]
            take = 1 if W[i, j] - 1 >= min_pct else int(W[i, j])
            take = min(take, excess) if take == 1 else take
            W[i, j] -= take
            W[i, other[int(np.argmax(W[i, other]))]] += take
            excess -= take
    return W


def sample_sparse(syms: list[str], n: int, seed: int, k_range: tuple[int, int] = (3, 10), min_pct: int = 3
                  ) -> np.ndarray:
    """``n`` portfolios of ``k_range`` products drawn uniformly from ``syms`` with Dirichlet(1) weights
    (no class structure; the A31 check sample)."""
    rng = np.random.default_rng(seed)
    lo, hi = k_range
    hi = min(hi, len(syms))
    W = np.zeros((n, len(syms)), dtype=np.int64)
    for i in range(n):
        k = int(rng.integers(lo, hi + 1))
        chosen = rng.choice(len(syms), size=k, replace=False)
        w = np.zeros(len(syms))
        w[chosen] = rng.dirichlet(np.ones(k))
        W[i] = round_weights(w, min_pct)
    return W


def neighbours(w: np.ndarray, step: int = 5, min_pct: int = 3) -> list[np.ndarray]:
    """All weight vectors reachable by moving ``step`` points from one product to another (a product
    with 0 may enter); parts left between 0 and ``min_pct`` are not allowed. The original is excluded."""
    w = np.asarray(w, dtype=np.int64)
    out, seen = [], set()
    for i in range(len(w)):
        if w[i] < step:
            continue
        for j in range(len(w)):
            if i == j:
                continue
            v = w.copy()
            v[i] -= step
            v[j] += step
            if 0 < v[i] < min_pct:
                continue
            key = tuple(v.tolist())
            if key in seen or key == tuple(w.tolist()):
                continue
            seen.add(key)
            out.append(v)
    return out


# --------------------------------------------------------------------------- metrics

def _cagr(paths: np.ndarray) -> np.ndarray:
    days = max(paths.shape[1] - 1, 1)
    ratio = paths[:, -1] / paths[:, 0]
    with np.errstate(invalid="ignore"):
        out = np.where(ratio > 0, np.abs(ratio) ** (TRADING_DAYS / days) - 1.0, -1.0)
    return out


def _mdd(paths: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(paths, axis=1)
    return (paths / peak - 1.0).min(axis=1)


def _sharpe(paths: np.ndarray) -> np.ndarray:
    r = paths[:, 1:] / paths[:, :-1] - 1.0
    sd = r.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, r.mean(axis=1) / sd * np.sqrt(TRADING_DAYS), np.nan)


def cagr_mdd(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CAGR and MDD of equity paths (rows), as ``metrics.curve_metrics`` defines them."""
    paths = np.asarray(paths, dtype=float)
    return _cagr(paths), _mdd(paths)


def _slice(dates: pd.DatetimeIndex, a: str | None, b: str | None) -> slice:
    lo = 0 if a is None else int(dates.searchsorted(pd.Timestamp(a), side="left"))
    hi = len(dates) if b is None else int(dates.searchsorted(pd.Timestamp(b), side="right"))
    return slice(lo, hi)


def _tag(date: str, used: set[str]) -> str:
    y = str(pd.Timestamp(date).year)
    tag = y if y not in used else pd.Timestamp(date).strftime("%Y%m%d")
    used.add(tag)
    return tag


def summarize(paths: np.ndarray, turnover: np.ndarray, dates: pd.DatetimeIndex, cash: pd.Series | None = None,
              windows: dict[str, tuple[str, str]] | None = None, halves: tuple[str, ...] = (),
              extra_starts: tuple[str, ...] = (), holdout_share: float = 0.0) -> pd.DataFrame:
    """One row per path. Columns: cagr, mdd, sharpe, calmar, vol, turnover, dd_<window> (drawdown from
    the window's first day), cagr_h1/mdd_h1/… for the pieces cut at ``halves``, cagr_<year>/mdd_<year>
    for ``extra_starts``, cagr_holdout/mdd_holdout for the last ``holdout_share`` of the days, and
    sharpe_excess when ``cash`` is given (daily return minus cash)."""
    paths = np.asarray(paths, dtype=float)
    out = {"cagr": _cagr(paths), "mdd": _mdd(paths), "sharpe": _sharpe(paths)}
    r = paths[:, 1:] / paths[:, :-1] - 1.0
    out["vol"] = r.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["calmar"] = np.where(out["mdd"] < 0, out["cagr"] / np.abs(out["mdd"]), np.nan)
    if cash is not None:
        c = cash.reindex(dates).fillna(0.0).to_numpy(dtype=float)[1:]
        ex = r - c[None, :]
        sd = ex.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["sharpe_excess"] = np.where(sd > 0, ex.mean(axis=1) / sd * np.sqrt(TRADING_DAYS), np.nan)
    out["turnover"] = np.asarray(turnover, dtype=float)
    for name, (a, b) in (windows or {}).items():
        sl = _slice(dates, a, b)
        sub = paths[:, sl]
        out[f"dd_{name}"] = _mdd(sub) if sub.shape[1] else np.full(len(paths), np.nan)
    cuts = [None, *halves, None]
    for h in range(len(cuts) - 1):
        sl = _slice(dates, cuts[h], None)
        hi = len(dates) if cuts[h + 1] is None else int(dates.searchsorted(pd.Timestamp(cuts[h + 1]), side="left"))
        sub = paths[:, sl.start:hi]
        if sub.shape[1] >= 2:
            out[f"cagr_h{h + 1}"], out[f"mdd_h{h + 1}"] = _cagr(sub), _mdd(sub)
    used: set[str] = set()
    for start in extra_starts:
        tag = _tag(start, used)
        sub = paths[:, _slice(dates, start, None)]
        if sub.shape[1] >= 2:
            out[f"cagr_{tag}"], out[f"mdd_{tag}"] = _cagr(sub), _mdd(sub)
    if holdout_share > 0:
        n = int(round(len(dates) * holdout_share))
        if n >= 2:
            sub = paths[:, -n:]
            out["cagr_holdout"], out["mdd_holdout"] = _cagr(sub), _mdd(sub)
    return pd.DataFrame(out)


def rolling_window_share(paths: np.ndarray, idx: pd.DatetimeIndex, years: int = 5, mdd_line: float = -0.15,
                         step: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Share of ``years``-long windows (moved every ``step`` trading days) whose MDD stays within
    ``mdd_line``, and the median window CAGR. Windows need the full length; paths shorter than one
    window give nan."""
    paths = np.asarray(paths, dtype=float)
    L = years * TRADING_DAYS
    T = paths.shape[1]
    if T <= L:
        return np.full(len(paths), np.nan), np.full(len(paths), np.nan)
    starts = range(0, T - L + 1, step)
    within = np.zeros(len(paths))
    cagrs = []
    for s in starts:
        sub = paths[:, s:s + L]
        within += _mdd(sub) >= mdd_line
        cagrs.append(_cagr(sub))
    return within / len(starts), np.median(np.vstack(cagrs), axis=0)


# --------------------------------------------------------------------------- CSCV with the MDD rule

def _month_metrics(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Annualised compound return and MDD of candidates (rows) from monthly returns (columns)."""
    path = np.cumprod(1.0 + R, axis=1)
    months = R.shape[1]
    with np.errstate(invalid="ignore"):
        cagr = np.where(path[:, -1] > 0, np.abs(path[:, -1]) ** (12.0 / months) - 1.0, -1.0)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(R), 1)), path], axis=1), axis=1)
    mdd = (path / peak[:, 1:] - 1.0).min(axis=1)
    return cagr, mdd


def pbo_selection(R: np.ndarray, mdd_line: float = -0.15, blocks: int = 10) -> dict:
    """CSCV (Bailey et al.) with the A31 selection rule.

    R: monthly returns, candidates × months. The months are cut into ``blocks`` contiguous blocks; for
    each of C(blocks, blocks/2) splits the training half picks the candidate with the best CAGR among
    those whose training MDD is within ``mdd_line`` (if none is, the one with the shallowest MDD).
    Out of sample the pick is ranked by CAGR among the candidates within the line there (a pick that
    breaks the line out of sample ranks at the bottom). Returns splits, pbo (share of splits with the
    pick in the bottom half), picked (per candidate: share of splits it was picked), oos_within (per
    candidate: share of splits with OOS MDD within the line), oos_rank (per split), and the pick's
    OOS CAGR / MDD per split.
    """
    R = np.asarray(R, dtype=float)
    K, M = R.shape
    edges = np.linspace(0, M, blocks + 1).astype(int)
    block_of = np.searchsorted(edges[1:], np.arange(M), side="right")
    picked = np.zeros(K)
    within = np.zeros(K)
    ranks, oos_cagr, oos_mdd = [], [], []
    n_splits = 0
    for train in combinations(range(blocks), blocks // 2):
        mask = np.isin(block_of, train)
        c_in, m_in = _month_metrics(R[:, mask])
        c_out, m_out = _month_metrics(R[:, ~mask])
        ok_in = m_in >= mdd_line
        if ok_in.any():
            score = np.where(ok_in, c_in, -np.inf)
            best = int(np.argmax(score))
        else:
            best = int(np.argmax(m_in))
        ok_out = m_out >= mdd_line
        within += ok_out
        picked[best] += 1
        if ok_out[best] and ok_out.sum() > 1:
            rank = float((c_out[ok_out] < c_out[best]).mean())
        elif ok_out[best]:
            rank = 1.0
        else:
            rank = 0.0
        ranks.append(rank)
        oos_cagr.append(float(c_out[best]))
        oos_mdd.append(float(m_out[best]))
        n_splits += 1
    ranks_arr = np.asarray(ranks)
    return {"splits": n_splits, "pbo": float((ranks_arr < 0.5).mean()), "picked": picked / n_splits,
            "oos_within": within / n_splits, "oos_rank": ranks_arr,
            "pick_oos_cagr": np.asarray(oos_cagr), "pick_oos_mdd": np.asarray(oos_mdd)}


def monthly_returns(paths: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Month-over-month returns of equity paths (candidates × months) from month-end values."""
    rows = decision_rows(dates, "M")
    me = np.asarray(paths, dtype=float)[:, rows]
    return me[:, 1:] / me[:, :-1] - 1.0


__all__ = ["StaticSim", "round_weights", "sample_two_level", "sample_sparse", "neighbours", "summarize", "cagr_mdd",
           "rolling_window_share", "pbo_selection", "monthly_returns"]
