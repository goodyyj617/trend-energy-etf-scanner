"""A42 stage 2: the wide static search v2 and the walk-forward of the picking rules (allocation DECISIONS A42 §2).

Design fixed before results: docs/research/allocation/2026-10-29_a42_design.md §2. scripts/aa_a42_stage2.py runs it.

- Universe: the 13 A41 products + KTB 3 / 10 / 30-year + US dividend (17, `U17`); the leverage arm adds 409820
  and 225040 with their sum ≤ 30 % (`U19`).
- Samples (seed 2042): T1 two-level (eight role groups on a randomly shifted Sobol set, products inside a group split
  at random), T2 free (each product in with probability 0.3 / 0.5 / 0.7 by thirds, size −log(u), min 2 %), T3
  k-uniform (k = 2 … 8 products, equal counts), TL leverage arm (T2 on 19 products).
- A point is measured from the first day all its products have data (`point_start`): KTB 10/30-year proxies begin
  2000-10, the US-dividend proxy 2003-11; such points carry their own start date.
- Extra metrics: CVaR 5 % (monthly), Ulcer index, longest time under water (months), recovery time from the MDD
  trough (months, NaN if not recovered), real CAGR (Korean CPI, ECOS 901Y009), exposures from the stage-1 betas.
- Walk-forward: at each December's last trading day 2005–2023 a rule picks weights from data up to that day
  (expanding from the point's start, or the last 10 years) and holds them for the next year. R1–R3 pick from the
  stage-2 sample using per-point training metrics computed from the full path (`yearly_windows`); R4–R8 compute
  weights from past daily returns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aa_composition import whole
from .aa_engine import TRADING_DAYS
from .aa_landscape import UNIVERSE as U13
from .explore import sobol

U17 = list(U13) + ["114260", "148070", "439870", "446720"]
LEVERAGE = ["409820", "225040"]
U19 = U17 + LEVERAGE
LEV_CAP = 30
ROLES = {
    "성장 주식": ["379810", "379800"], "방어 주식": ["453630", "453640", "463640"], "배당": ["446720"],
    "금": ["KRXGOLD"], "원화 현금": ["459580"], "달러 현금": ["0046A0", "329750"],
    "미국채": ["305080", "0091C0", "476760", "484790"], "국고채": ["114260", "148070", "439870"],
}
SEED = 2042
MIN_PCT = 2
CUT_YEARS = list(range(2005, 2024))  # selection days: last trading day of each December
ROLL_YEARS = 10
CPI_PATH = Path(__file__).with_name("aa_kr_cpi.csv")


# --------------------------------------------------------------------------- samplers

def _whole_rows(w: np.ndarray) -> np.ndarray:
    return np.vstack([whole(r) for r in w]).astype(np.int64)


def _sizes(u: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(1.0 - u, 1e-12, 1.0))


def sample_t1(n: int, seed: int = SEED, universe: list[str] = U17) -> np.ndarray:
    """Role-group weights on a randomly shifted (Cranley–Patterson) Sobol set, then a random split inside each group."""
    roles = list(ROLES)
    U = sobol(n + 1, len(roles))[1:]
    rng = np.random.default_rng(seed)
    U = np.mod(U + rng.random(len(roles)), 1.0)
    gw = _sizes(U)
    gw = gw / gw.sum(axis=1, keepdims=True)
    col = {c: i for i, c in enumerate(universe)}
    W = np.zeros((n, len(universe)))
    for g, members in enumerate(ROLES.values()):
        split = rng.dirichlet(np.ones(len(members)), size=n) if len(members) > 1 else np.ones((n, 1))
        for m, code in enumerate(members):
            W[:, col[code]] += gw[:, g] * split[:, m]
    W = np.where(W * 100 < MIN_PCT, 0.0, W)
    return _whole_rows(W)


def sample_t2(n: int, seed: int = SEED, universe: list[str] = U17, probs=(0.3, 0.5, 0.7)) -> np.ndarray:
    """Free sample: each product in with probability p (a third of the rows each), size −log(u), min 2 %."""
    rng = np.random.default_rng(seed)
    d = len(universe)
    p = np.repeat(np.asarray(probs, float), int(np.ceil(n / len(probs))))[:n]
    inc = rng.random((n, d)) < p[:, None]
    size = rng.random((n, d))
    none = ~inc.any(axis=1)
    inc[none, np.argmax(size[none], axis=1)] = True
    raw = np.where(inc, np.maximum(_sizes(size), 1e-12), 0.0)
    k = inc.sum(axis=1)
    free = (100.0 - MIN_PCT * k)[:, None]
    w = np.where(inc, MIN_PCT + free * raw / raw.sum(axis=1, keepdims=True), 0.0)
    return _whole_rows(w)


def sample_t3(n: int, seed: int = SEED, universe: list[str] = U17, ks=range(2, 9)) -> np.ndarray:
    """k-uniform: k products chosen uniformly (equal counts per k), uniform Dirichlet sizes, min 2 %."""
    rng = np.random.default_rng(seed)
    ks = list(ks)
    d = len(universe)
    k_of = np.repeat(ks, int(np.ceil(n / len(ks))))[:n]
    w = np.zeros((n, d))
    for k in ks:
        rows = np.flatnonzero(k_of == k)
        pick = np.argsort(rng.random((len(rows), d)), axis=1)[:, :k]
        sizes = rng.dirichlet(np.ones(k), size=len(rows))
        sizes = MIN_PCT / 100 + (1 - MIN_PCT * k / 100) * sizes
        w[rows[:, None], pick] = sizes
    return _whole_rows(w * 100)


def sample_tl(n: int, seed: int = SEED) -> np.ndarray:
    """Leverage arm: T2 on the 19 products, the two leverage products together capped at 30 %: the excess goes pro
    rata to the other held products (to CD금리 if none), then whole percents with the 2 % floor kept."""
    W = sample_t2(n, seed, U19).astype(float)
    lev = np.array([U19.index(c) for c in LEVERAGE])
    rest = np.setdiff1d(np.arange(len(U19)), lev)
    cd = U19.index("459580")
    out = np.zeros_like(W, dtype=np.int64)
    for i in range(n):
        w = W[i].copy()
        tot = w[lev].sum()
        if tot > LEV_CAP:
            freed = tot - LEV_CAP
            w[lev] *= LEV_CAP / tot
            base = w[rest].sum()
            if base > 0:
                w[rest] += freed * w[rest] / base
            else:
                w[cd] += freed
        r = whole(w)
        while r[lev].sum() > LEV_CAP:  # rounding can add a point: move it to the largest non-leverage holding
            j = lev[np.argmax(r[lev])]
            r[j] -= 1
            k = rest[np.argmax(np.where(r[rest] > 0, r[rest], -1))] if (r[rest] > 0).any() else cd
            r[k] += 1
        small = (r > 0) & (r < MIN_PCT)
        if small.any():  # keep the 2 % floor: give a small holding's points to the largest holding of the same row
            for j in np.flatnonzero(small):
                big = int(np.argmax(np.where(np.isin(np.arange(len(r)), lev) & (r[lev].sum() >= LEV_CAP), -1, r)))
                r[big] += r[j]
                r[j] = 0
        out[i] = r
    return out


# --------------------------------------------------------------------------- start dates and metrics

def point_start(W: np.ndarray, universe: list[str], first_valid: pd.Series) -> np.ndarray:
    """Per row: the latest first-valid date among the products it holds (as numpy datetime64)."""
    fv = np.array([first_valid[c] for c in universe], dtype="datetime64[ns]")
    held = W > 0
    return np.where(held, fv[None, :], np.datetime64("1900-01-01")).max(axis=1)


def extra_metrics(P: np.ndarray, dates: pd.DatetimeIndex, cpi: pd.Series | None = None) -> pd.DataFrame:
    """CVaR 5 % of monthly returns, Ulcer index (%), longest time under water and recovery time from the MDD trough
    (months; NaN = not recovered by the end), real CAGR with Korean CPI."""
    me = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")).last().to_numpy()
    M = P[:, me]
    mr = M[:, 1:] / M[:, :-1] - 1.0
    q = np.quantile(mr, 0.05, axis=1, keepdims=True)
    cvar = np.where(mr <= q, mr, np.nan)
    cvar = np.nanmean(cvar, axis=1)
    peak = np.maximum.accumulate(P, axis=1)
    dd = P / peak - 1.0
    ulcer = np.sqrt((dd ** 2).mean(axis=1)) * 100.0
    under = (M / np.maximum.accumulate(M, axis=1)) < 1.0 - 1e-12
    run = np.zeros(len(P))
    longest = np.zeros(len(P))
    for j in range(under.shape[1]):  # vectorised over paths: the longest run of months under the previous peak
        run = (run + 1.0) * under[:, j]
        longest = np.maximum(longest, run)
    trough = dd.argmin(axis=1)
    level = peak[np.arange(len(P)), trough]
    cols = np.arange(P.shape[1])[None, :]
    back = (P >= level[:, None] * (1 - 1e-12)) & (cols > trough[:, None])
    first = back.argmax(axis=1)
    rec = np.where(back.any(axis=1), (first - trough) / 21.0, np.nan)
    out = {"cvar5_m": cvar, "ulcer": ulcer, "underwater_m": longest, "recovery_m": rec}
    if cpi is not None:
        c = cpi.reindex(dates.to_period("M").astype(str)).to_numpy(float)
        ok = np.isfinite(c)
        if ok.sum() >= 2:
            i0, i1 = np.flatnonzero(ok)[[0, -1]]
            yrs = (i1 - i0) / TRADING_DAYS
            nominal = P[:, i1] / P[:, i0]
            out["real_cagr"] = (nominal / (c[i1] / c[i0])) ** (1 / max(yrs, 1e-9)) - 1.0
    return pd.DataFrame(out)


def load_cpi(path: Path = CPI_PATH) -> pd.Series:
    d = pd.read_csv(path)
    return d.set_index("month")["cpi"].astype(float)


def cut_rows(dates: pd.DatetimeIndex, years=CUT_YEARS) -> dict[int, int]:
    """Row of the last trading day of each December in `years` (only those inside `dates`)."""
    out = {}
    for y in years:
        d = dates[(dates.year == y) & (dates.month == 12)]
        if len(d):
            out[y] = int(dates.get_loc(d[-1]))
    return out


def _cagr(a, b, days):
    r = b / a
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(r > 0, r ** (TRADING_DAYS / np.maximum(days, 1)) - 1.0, -1.0)


def _mdd(P):
    return (P / np.maximum.accumulate(P, axis=1) - 1.0).min(axis=1)


def yearly_windows(P: np.ndarray, dates: pd.DatetimeIndex, years=CUT_YEARS, roll: int = ROLL_YEARS) -> np.ndarray:
    """(K, len(years), 6) float32: for each selection day, training CAGR / MDD over the expanding window (from the
    path's start) and the rolling `roll`-year window (NaN if the path is shorter), then the next year's return and
    MDD (from the selection day to the next December's last trading day, or the path's end)."""
    cuts = cut_rows(dates, years)
    K = len(P)
    out = np.full((K, len(years), 6), np.nan, dtype=np.float32)
    rows = sorted(cuts.values())
    for j, y in enumerate(years):
        if y not in cuts:
            continue
        c = cuts[y]
        if c < 252:
            continue
        out[:, j, 0] = _cagr(P[:, 0], P[:, c], c)
        out[:, j, 1] = _mdd(P[:, :c + 1])
        lo = int(dates.searchsorted(dates[c] - pd.DateOffset(years=roll)))
        if dates[c] - dates[0] >= pd.Timedelta(days=365 * roll - 7):
            out[:, j, 2] = _cagr(P[:, lo], P[:, c], c - lo)
            out[:, j, 3] = _mdd(P[:, lo:c + 1])
        nxt = [r for r in rows if r > c]
        e = nxt[0] if nxt else len(dates) - 1
        out[:, j, 4] = P[:, e] / P[:, c] - 1.0
        out[:, j, 5] = _mdd(P[:, c:e + 1])
    return out


# --------------------------------------------------------------------------- picking rules

def pick_r1(train_cagr: np.ndarray, train_mdd: np.ndarray, line: float = -0.15) -> int | None:
    ok = np.flatnonzero(np.isfinite(train_cagr) & (train_mdd >= line))
    return int(ok[np.argmax(train_cagr[ok])]) if len(ok) else None


def pick_r3(train_cagr: np.ndarray, train_mdd: np.ndarray) -> int | None:
    with np.errstate(divide="ignore", invalid="ignore"):
        calmar = np.where(train_mdd < 0, train_cagr / np.abs(train_mdd), np.nan)
    ok = np.flatnonzero(np.isfinite(calmar))
    return int(ok[np.argmax(calmar[ok])]) if len(ok) else None


def pick_r2(W: np.ndarray, train_cagr: np.ndarray, train_mdd: np.ndarray, line: float = -0.15, flat: float = 0.003,
            major: int = 10) -> np.ndarray | None:
    """Flat-region centre in the training window: points inside the line whose CAGR is within `flat` of the best,
    grouped by the products holding ≥ `major` %, the largest group's mean (products under 2 % dropped, whole %).
    (A41's ±5 %p neighbour test is left out here: it would need fresh simulations at every selection day.)"""
    ok = np.flatnonzero(np.isfinite(train_cagr) & (train_mdd >= line))
    if not len(ok):
        return None
    best = train_cagr[ok].max()
    reg = ok[train_cagr[ok] >= best - flat]
    keys = ["+".join(map(str, np.flatnonzero(W[i] >= major))) for i in reg]
    groups = pd.Series(range(len(reg))).groupby(pd.Series(keys)).apply(list)
    biggest = max(groups, key=len)
    mean = W[reg[biggest]].mean(axis=0)
    mean[mean < MIN_PCT] = 0.0
    return whole(mean)


def neighbours(w: np.ndarray, step: int = 5) -> np.ndarray:
    """A41's ±step %p moves for any number of products: each held product up or down by `step` (to 0 below `step`),
    each missing product added at `step` %, the rest rescaled in proportion, whole percents, duplicates dropped."""
    w = np.asarray(w, np.int64)
    D = len(w)
    out = []
    for i in range(D):
        vals = [w[i] + step, max(w[i] - step, 0)] if w[i] > 0 else [step]
        for val in vals:
            rest = np.delete(np.arange(D), i)
            other = w[rest].sum()
            if other <= 0 or not 0 <= val < 100:
                continue
            v = np.zeros(D)
            v[i] = val
            v[rest] = w[rest] * (100.0 - val) / other
            out.append(whole(v))
    if not out:
        return np.zeros((0, D), np.int64)
    V = np.unique(np.vstack(out), axis=0)
    return V[(V != w).any(axis=1)]


def _project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.flatnonzero(u * np.arange(1, len(v) + 1) > (css - 1))[-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def min_variance(S: np.ndarray, iters: int = 4000) -> np.ndarray:
    """Long-only minimum-variance weights (projected gradient on the simplex)."""
    n = len(S)
    w = np.full(n, 1.0 / n)
    step = 1.0 / (np.linalg.eigvalsh(S).max() * 2 + 1e-12)
    for _ in range(iters):
        w = _project_simplex(w - step * 2 * S @ w)
    return w


def max_diversification(S: np.ndarray, iters: int = 4000) -> np.ndarray:
    """Long-only maximum diversification ratio weights: minimise w'Cw over the simplex with C the correlation matrix,
    then rescale by 1/σ (Choueifaty–Coignard), long-only projection."""
    sd = np.sqrt(np.maximum(np.diag(S), 1e-18))
    C = S / np.outer(sd, sd)
    y = min_variance(C, iters)
    w = y / sd
    return w / w.sum()


def rule_weights(rule: str, rets: pd.DataFrame, codes: list[str], row: int, lookback: int = 3 * TRADING_DAYS,
                 min_days: int = 63) -> dict[str, float]:
    """R4 equal, R5 inverse volatility, R6 equity risk-share cap, R7 minimum variance, R8 maximum diversification,
    from daily returns up to `row` (the selection day). Fewer than `min_days` days of history: equal weights."""
    from .aa_landscape import rule_targets
    if rule in ("R4", "R5", "R6"):
        name = {"R4": "equal", "R5": "invvol", "R6": "riskcap"}[rule]
        return rule_targets(rets, name, codes, np.array([row]), lookback, min_days)[row]
    X = rets[codes].to_numpy(float)[max(0, row + 1 - lookback):row + 1]
    X = X[~np.isnan(X).any(axis=1)]
    if len(X) < min_days:
        return {c: 1.0 / len(codes) for c in codes}
    S = np.cov(X, rowvar=False) * TRADING_DAYS
    w = min_variance(S) if rule == "R7" else max_diversification(S)
    return {c: float(x) for c, x in zip(codes, w) if x > 1e-6}


__all__ = ["U17", "U19", "LEVERAGE", "ROLES", "SEED", "CUT_YEARS", "sample_t1", "sample_t2", "sample_t3", "sample_tl",
           "point_start", "extra_metrics", "yearly_windows", "cut_rows", "pick_r1", "pick_r2", "pick_r3",
           "min_variance", "max_diversification", "rule_weights", "load_cpi", "neighbours"]
