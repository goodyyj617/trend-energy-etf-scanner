"""A41: a map of the whole weight space of 13 products, read as one big function (DECISIONS A41).

Design (fixed before results): docs/research/allocation/2026-10-28_a41_design.md. This module holds the pieces;
scripts/aa_redesign_study.py runs them.

- Sample: per product an inclusion coordinate (included when < 0.5) and a size coordinate u; the k included
  products get 2 % each and share the rest in proportion to −log(u) (uniform Dirichlet within the included set),
  whole percents by largest remainder (aa_composition.whole). Coordinates are pseudo-random uniforms (see ``rqmc``
  for why not Sobol; design change log, 2026-10-28).
- Metrics per path (`path_metrics`): research window, halves, 2005– reference, crisis windows, and the start-date
  statistics: CAGR from every quarter-end start (2000-09 – 2015-12) to the end, whose 10th percentile is the rank
  value, and the MDD of every 10-year window starting at a quarter end.
- Zones: 안 (MDD ≥ −15 %), 경계 (−17 % ≤ MDD < −15 %), 밖. Nothing is dropped for crossing the line.
- Landscape readers: Scheffé quadratic surrogate (gradient map only), Sobol indices on real simulations (Saltelli
  first order, Jansen total, product = its two coordinates), exchange rates (1 / 5 %p moved between products),
  leave-one-out, product-set summaries.
- Choosing: flat region (within 0.3 %p of the best rank value among points passing MDD, all 10-year windows and
  ±5 %p neighbours ≥ 80 % inside), grouped by the products holding ≥ 10 %, one centre per group.
- Rule-based comparators: equal weight, inverse volatility, equity risk-share cap (decided from past data only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aa_assets import CASH, KRX_GOLD
from .aa_composition import whole
from .aa_engine import TRADING_DAYS, decision_rows, simulate

UNIVERSE = ["379810", "379800", "453630", "453640", "463640", KRX_GOLD, "459580", "0046A0", "329750", "305080",
            "0091C0", "476760", "484790"]
ROLE = {"379810": "성장 주식", "379800": "성장 주식", "453630": "방어 주식", "453640": "방어 주식", "463640": "방어 주식",
        KRX_GOLD: "금", "459580": "원화 현금", "0046A0": "달러 현금", "329750": "단기 달러 채권",
        "305080": "중기 미국채(환노출)", "0091C0": "중기 미국채(헤지)", "476760": "장기 미국채(환노출)",
        "484790": "장기 미국채(헤지)"}
SHORT = {"379810": "나스닥100", "379800": "S&P500", "453630": "필수소비재", "453640": "헬스케어", "463640": "유틸리티",
         KRX_GOLD: "KRX금현물", "459580": "CD금리", "0046A0": "미국초단기채", "329750": "미국단기채", "305080": "미국10년(환노출)",
         "0091C0": "미국10년(H)", "476760": "미국30년(환노출)", "484790": "미국30년(H)"}
EQUITY = ["379810", "379800", "453630", "453640", "463640"]
CASH_LIKE = ["459580", "0046A0"]
D = len(UNIVERSE)
COL = {c: i for i, c in enumerate(UNIVERSE)}

START, END = "2000-09-01", "2024-12-31"
H1 = ("2000-09-01", "2012-12-31")
H2 = ("2013-01-02", "2024-12-31")
REF_START = "2005-01-03"
LAST_START = "2015-12-31"  # quarter-end starts up to here (rank value over the full window)
DOTCOM = ("2000-09-01", "2002-12-31")
LINE, NEAR = -0.15, -0.17
MIN_PCT = 2
SEED = 2041
FLAT_PP = 0.003  # the flat region: rank value within 0.3 %p of the best
NB_SHARE = 0.80
MAJOR_PCT = 10
CENTRE_MIN_SHARE, CENTRE_MIN_N, MAX_CENTRES = 0.05, 5, 5


# --------------------------------------------------------------------------- sampling

def rqmc(n: int, dim: int, seed: int = SEED, skip: int = 0) -> np.ndarray:
    """Independent uniform coordinates in [0, 1)^dim (numpy PCG64, ``seed``); rows skip … skip + n − 1 of the stream.

    The design asked for scrambled Sobol. lab.explore.sobol stops at 10 dimensions, scipy is not installed, and a
    Kronecker (R_d) set was tried first but its step in dimension 11 is 0.7499 ≈ 3/4, which repeats every four
    points. Plain pseudo-random points have no such pattern; with 2^20 points the coverage loss is small
    (design change log, 2026-10-28)."""
    rng = np.random.default_rng(seed)
    if skip:
        rng.random((skip, dim))
    return rng.random((n, dim))


def weights_from_u(U: np.ndarray, min_pct: int = MIN_PCT) -> np.ndarray:
    """(n, 2D) coordinates [inclusion | size] -> whole-percent weights (n, D) as in the module docstring."""
    U = np.asarray(U, dtype=float)
    inc = U[:, :D] < 0.5
    size = np.clip(U[:, D:], 1e-12, 1.0)
    none = ~inc.any(axis=1)
    inc[none, np.argmax(size[none], axis=1)] = True
    raw = -np.log1p(-size + 1e-15) * inc  # −log(1 − u): u near 1 is a big share; same law as −log(u)
    raw = np.where(inc, np.maximum(raw, 1e-12), 0.0)
    k = inc.sum(axis=1)
    free = (100.0 - min_pct * k)[:, None]
    w = np.where(inc, min_pct + free * raw / raw.sum(axis=1, keepdims=True), 0.0)
    return np.vstack([whole(row) for row in w]).astype(np.int64)


def sample(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    U = rqmc(n, 2 * D, seed)
    return U, weights_from_u(U)


def to_fraction(W: np.ndarray) -> np.ndarray:
    F = np.zeros((W.shape[0], W.shape[1] + 1))
    F[:, :-1] = np.asarray(W, dtype=float) / 100.0
    return F


# --------------------------------------------------------------------------- metrics

def _cagr(a: np.ndarray, b: np.ndarray, days: np.ndarray | int) -> np.ndarray:
    ratio = b / a
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ratio > 0, ratio ** (TRADING_DAYS / np.maximum(days, 1)) - 1.0, -1.0)


def _mdd(P: np.ndarray) -> np.ndarray:
    return (P / np.maximum.accumulate(P, axis=1) - 1.0).min(axis=1)


@dataclass(frozen=True)
class Layout:
    """Row positions on the simulation dates (computed once per date index)."""
    dates: pd.DatetimeIndex
    full: slice
    h1: slice
    h2: slice
    ref: slice
    windows: dict
    starts: np.ndarray  # quarter-end rows, full window rank value
    win_starts: np.ndarray  # 10-year windows (start row, end row)
    win_ends: np.ndarray
    h1_starts: np.ndarray
    h1_end: int
    h1_win: tuple
    h2_starts: np.ndarray
    h2_end: int
    h2_win: tuple


def _sl(dates: pd.DatetimeIndex, a: str, b: str) -> slice:
    return slice(int(dates.searchsorted(pd.Timestamp(a), "left")), int(dates.searchsorted(pd.Timestamp(b), "right")))


def _quarter_starts(dates: pd.DatetimeIndex, a: str, b: str) -> np.ndarray:
    rows = decision_rows(dates, "Q")[:-1]
    first = np.array([0]) if dates[0] >= pd.Timestamp(a) else np.array([], dtype=int)
    rows = np.concatenate([first, rows])
    ok = (dates[rows] >= pd.Timestamp(a)) & (dates[rows] <= pd.Timestamp(b))
    return np.unique(rows[ok])


def _ten_year(dates: pd.DatetimeIndex, starts: np.ndarray, last: int) -> tuple[np.ndarray, np.ndarray]:
    s_out, e_out = [], []
    for s in starts:
        e = int(dates.searchsorted(dates[s] + pd.DateOffset(years=10), "left"))
        if e <= last:
            s_out.append(int(s))
            e_out.append(e)
    return np.array(s_out, dtype=int), np.array(e_out, dtype=int)


def layout(dates: pd.DatetimeIndex, windows: dict[str, tuple[str, str]]) -> Layout:
    T = len(dates)
    full = slice(0, T)
    h1, h2, ref = _sl(dates, *H1), _sl(dates, *H2), _sl(dates, REF_START, END)
    starts = _quarter_starts(dates, START, LAST_START)
    ws, we = _ten_year(dates, _quarter_starts(dates, START, END), T - 1)
    h1_end, h2_end = h1.stop - 1, h2.stop - 1
    five = pd.DateOffset(years=5)
    h1s = _quarter_starts(dates, H1[0], str((dates[h1_end] - five).date()))
    h2s = _quarter_starts(dates, H2[0], str((dates[h2_end] - five).date()))
    h1w = _ten_year(dates, _quarter_starts(dates, *H1), h1_end)
    h2w = _ten_year(dates, _quarter_starts(dates, *H2), h2_end)
    return Layout(dates, full, h1, h2, ref, {k: _sl(dates, *v) for k, v in windows.items()}, starts, ws, we,
                  h1s, h1_end, h1w, h2s, h2_end, h2w)


def _start_stats(P: np.ndarray, starts: np.ndarray, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(starts) == 0:
        nan = np.full(len(P), np.nan)
        return nan, nan, nan
    C = _cagr(P[:, starts], P[:, [end]], end - starts[None, :])
    return np.percentile(C, 10, axis=1), np.median(C, axis=1), np.percentile(C, 90, axis=1)


def _window_stats(P: np.ndarray, s: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(s) == 0:
        nan = np.full(len(P), np.nan)
        return nan, nan
    m = np.column_stack([_mdd(P[:, a:b + 1]) for a, b in zip(s, e)])
    return m.min(axis=1), (m >= LINE).mean(axis=1)


def path_metrics(P: np.ndarray, turnover: np.ndarray, L: Layout) -> pd.DataFrame:
    """One row per path (columns in fractions). P: equity paths on L.dates."""
    T = P.shape[1]
    out = {"cagr": _cagr(P[:, 0], P[:, -1], T - 1), "mdd": _mdd(P), "turnover": np.asarray(turnover, float)}
    for tag, sl in (("h1", L.h1), ("h2", L.h2), ("ref", L.ref)):
        sub = P[:, sl]
        out[f"cagr_{tag}"], out[f"mdd_{tag}"] = _cagr(sub[:, 0], sub[:, -1], sub.shape[1] - 1), _mdd(sub)
    for k, sl in L.windows.items():
        out[f"dd_{k}"] = _mdd(P[:, sl])
    out["rank"], out["start_p50"], out["start_p90"] = _start_stats(P, L.starts, T - 1)
    out["win_worst"], out["win_share"] = _window_stats(P, L.win_starts, L.win_ends)
    out["rank_h1"], _, _ = _start_stats(P, L.h1_starts, L.h1_end)
    out["win_worst_h1"], out["win_share_h1"] = _window_stats(P, *L.h1_win)
    out["rank_h2"], _, _ = _start_stats(P, L.h2_starts, L.h2_end)
    out["win_worst_h2"], out["win_share_h2"] = _window_stats(P, *L.h2_win)
    return pd.DataFrame(out)


def zone(mdd: np.ndarray | pd.Series) -> np.ndarray:
    m = np.asarray(mdd, float)
    return np.where(m >= LINE, "안", np.where(m >= NEAR, "경계", "밖"))


# --------------------------------------------------------------------------- moves around a point

def rescale(w: np.ndarray, fixed: int, value: float) -> np.ndarray | None:
    """Set product ``fixed`` to ``value`` %, the others in proportion to the rest; whole percents."""
    w = np.asarray(w, float)
    rest = np.delete(np.arange(D), fixed)
    other = w[rest].sum()
    if other <= 0 or not 0 <= value < 100:
        return None
    v = np.zeros(D)
    v[fixed] = value
    v[rest] = w[rest] * (100.0 - value) / other
    return whole(v)


def neighbours(w: np.ndarray, step: int = 5) -> np.ndarray:
    """±step %p on each held product (to 0 when below step) and each missing product added at step %, the rest
    rescaled in proportion; duplicates and the point itself removed."""
    w = np.asarray(w, np.int64)
    out = []
    for i in range(D):
        if w[i] > 0:
            for val in (w[i] + step, max(w[i] - step, 0) if w[i] >= step else 0):
                v = rescale(w, i, float(val))
                if v is not None:
                    out.append(v)
        else:
            v = rescale(w, i, float(step))
            if v is not None:
                out.append(v)
    if not out:
        return np.empty((0, D), np.int64)
    arr = np.unique(np.vstack(out), axis=0)
    return arr[~(arr == w).all(axis=1)]


def exchanges(w: np.ndarray, step: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """All moves of ``step`` %p from a held product i (w_i ≥ step) to another product j."""
    w = np.asarray(w, np.int64)
    rows, pairs = [], []
    for i in np.flatnonzero(w >= step):
        for j in range(D):
            if j != i:
                v = w.copy()
                v[i] -= step
                v[j] += step
                rows.append(v)
                pairs.append((int(i), int(j)))
    return (np.vstack(rows) if rows else np.empty((0, D), np.int64)), pairs


def drop_one(w: np.ndarray, mode: str = "pro") -> list[tuple[int, np.ndarray]]:
    """Each held product out; its weight pro rata to the rest ("pro") or to CD금리 ("cd")."""
    w = np.asarray(w, float)
    out = []
    for i in np.flatnonzero(w > 0):
        v = w.copy()
        v[i] = 0.0
        if mode == "cd":
            if UNIVERSE[i] == "459580":
                continue
            v[COL["459580"]] += w[i]
        if v.sum() > 0:
            out.append((int(i), whole(v)))
    return out


# --------------------------------------------------------------------------- landscape readers

def scheffe_features(W: np.ndarray) -> np.ndarray:
    X = np.asarray(W, float) / 100.0
    iu, ju = np.triu_indices(D, 1)
    return np.hstack([X, X[:, iu] * X[:, ju]])


def scheffe_fit(W: np.ndarray, y: np.ndarray) -> dict:
    """Least squares on the even rows, R² on the odd rows. Returns coefficients and the two R²."""
    X, y = scheffe_features(W), np.asarray(y, float)
    tr, te = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
    beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)

    def r2(idx):
        e = y[idx] - X[idx] @ beta
        return float(1.0 - (e @ e) / max(((y[idx] - y[idx].mean()) ** 2).sum(), 1e-300))

    return {"beta": beta, "r2_fit": r2(tr), "r2_test": r2(te)}


def scheffe_gradient(beta: np.ndarray, W: np.ndarray) -> np.ndarray:
    """∂f/∂w_i (per 1 %p) of the quadratic mixture model at each row of W."""
    X = np.asarray(W, float) / 100.0
    lin, quad = beta[:D], beta[D:]
    iu, ju = np.triu_indices(D, 1)
    G = np.tile(lin, (len(X), 1))
    for k, (i, j) in enumerate(zip(iu, ju)):
        G[:, i] += quad[k] * X[:, j]
        G[:, j] += quad[k] * X[:, i]
    return G / 100.0


def saltelli(N: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """A, B (N × 2D) and A_B^(i) for each product i (its two coordinates taken from B)."""
    AB = rqmc(N, 4 * D, seed + 1)
    A, B = AB[:, :2 * D], AB[:, 2 * D:]
    mixed = []
    for i in range(D):
        M = A.copy()
        M[:, [i, D + i]] = B[:, [i, D + i]]
        mixed.append(M)
    return A, B, mixed


def sobol_indices(fA: np.ndarray, fB: np.ndarray, fM: list[np.ndarray], boot: int = 200, seed: int = SEED) -> pd.DataFrame:
    """First-order (Saltelli 2010) and total (Jansen) indices with bootstrap 95 % intervals."""
    fA, fB = np.asarray(fA, float), np.asarray(fB, float)
    N = len(fA)
    rng = np.random.default_rng(seed)
    idx = [np.arange(N)] + [rng.integers(0, N, N) for _ in range(boot)]
    rows = []
    for i, fi in enumerate(fM):
        fi = np.asarray(fi, float)
        s1, st = [], []
        for k in idx:
            V = np.var(np.concatenate([fA[k], fB[k]]))
            s1.append(np.mean(fB[k] * (fi[k] - fA[k])) / V if V > 0 else np.nan)
            st.append(np.mean((fA[k] - fi[k]) ** 2) / (2 * V) if V > 0 else np.nan)
        s1, st = np.array(s1), np.array(st)
        rows.append({"상품": UNIVERSE[i], "S1": s1[0], "S1_lo": np.nanpercentile(s1[1:], 2.5), "S1_hi": np.nanpercentile(s1[1:], 97.5),
                     "ST": st[0], "ST_lo": np.nanpercentile(st[1:], 2.5), "ST_hi": np.nanpercentile(st[1:], 97.5)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- choosing: flat region and centres

def flat_region(df: pd.DataFrame, nb_share: pd.Series, near: bool = False) -> pd.DataFrame:
    """Rows of ``df`` (with rank, mdd, win_worst) in the flat region: (가) or, with near=True, (나)."""
    lo, hi = (NEAR, LINE) if near else (LINE, 1.0)
    win_line = NEAR if near else LINE
    ok = (df["mdd"] >= lo) & (df["mdd"] < hi) & (df["win_worst"] >= win_line)
    ok &= nb_share.reindex(df.index).fillna(-1.0) >= NB_SHARE
    cand = df[ok]
    if cand.empty:
        return cand
    return cand[cand["rank"] >= cand["rank"].max() - FLAT_PP]


def centres(W: np.ndarray, region: pd.DataFrame) -> list[dict]:
    """One centre per group of region points sharing the products that hold ≥ MAJOR_PCT %."""
    if region.empty:
        return []
    Wr = W[region.index.to_numpy()]
    keys = ["+".join(UNIVERSE[i] for i in np.flatnonzero(r >= MAJOR_PCT)) for r in Wr]
    groups = pd.Series(range(len(keys))).groupby(pd.Series(keys)).apply(list)
    need = max(CENTRE_MIN_N, int(np.ceil(CENTRE_MIN_SHARE * len(region))))
    out = []
    for key, pos in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(pos) < need:
            continue
        mean = Wr[pos].mean(axis=0)
        mean[mean < MIN_PCT] = 0.0
        out.append({"묶음": key, "점 수": len(pos), "비중": whole(mean)})
        if len(out) == MAX_CENTRES:
            break
    return out


# --------------------------------------------------------------------------- rule-based comparators

def rule_targets(rets: pd.DataFrame, rule: str, codes: list[str], rows: np.ndarray, lookback: int = 3 * TRADING_DAYS,
                 min_days: int = 63) -> dict[int, dict[str, float]]:
    """Target weights at each decision row from past daily returns only (rows ≤ the decision day).

    equal: 1/n. invvol: cash-like products (annual vol < 1 %) share 30 % equally, the rest ∝ 1/vol.
    riskcap: start from equal weights; if the equities' share of portfolio variance exceeds 50 %, scale the
    equities down (to cash-like products, equally; to the other products if none is held) until it is 50 %.
    Volatility and covariance use up to ``lookback`` days before the decision (fewer at the start, at least
    ``min_days``; before that, equal weights)."""
    X = rets[codes].to_numpy(float)
    n = len(codes)
    eq_idx = [k for k, c in enumerate(codes) if c in EQUITY]
    out = {}
    for r in rows:
        lo = max(0, r + 1 - lookback)
        hist = X[lo:r + 1]
        hist = hist[~np.isnan(hist).any(axis=1)]
        w = np.full(n, 1.0 / n)
        if rule != "equal" and len(hist) >= min_days:
            vol = hist.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
            if rule == "invvol":
                cash = vol < 0.01
                w = np.zeros(n)
                rest = ~cash
                if cash.any():
                    w[cash] = (0.30 if rest.any() else 1.0) / cash.sum()
                if rest.any():
                    inv = 1.0 / np.maximum(vol[rest], 1e-6)
                    w[rest] = (1.0 - w.sum()) * inv / inv.sum()
            elif rule == "riskcap":
                S = np.cov(hist, rowvar=False) * TRADING_DAYS
                w = _risk_cap(np.full(n, 1.0 / n), S, eq_idx, [k for k, c in enumerate(codes) if c in CASH_LIKE])
        out[int(r)] = {c: float(x) for c, x in zip(codes, w) if x > 0}
    return out


def _risk_cap(w: np.ndarray, S: np.ndarray, eq: list[int], cash: list[int], cap: float = 0.5) -> np.ndarray:
    if not eq:
        return w
    sink = cash or [k for k in range(len(w)) if k not in eq]
    if not sink:
        return w

    def share(x):
        tot = x @ S @ x
        return float((x[eq] @ (S @ x)[eq]) / tot) if tot > 0 else 0.0

    if share(w) <= cap:
        return w
    lo, hi = 0.0, 1.0  # fraction of the equity weight kept
    base_eq = w[eq].sum()
    for _ in range(60):
        mid = (lo + hi) / 2
        x = w.copy()
        x[eq] *= mid
        x[sink] += base_eq * (1 - mid) / len(sink)
        lo, hi = (mid, hi) if share(x) <= cap else (lo, mid)
    x = w.copy()
    x[eq] *= lo
    x[sink] += base_eq * (1 - lo) / len(sink)
    return x


def run_rule(panel, rule: str, codes: list[str], start: str, end: str, cost_bps: float = 10.0) -> tuple[pd.Series, float]:
    """Equity curve and turnover of a rule-based allocation (quarterly decisions, next close, engine `simulate`)."""
    idx = panel.close.index
    close = panel.close.loc[:pd.Timestamp(end), codes].ffill()
    rets_all = close.pct_change(fill_method=None)
    dates = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    rows_all = np.searchsorted(rets_all.index, dates[decision_rows(dates, "Q")])
    rows_all = rows_all[rows_all < rets_all.index.searchsorted(dates[-1])]
    tg_all = rule_targets(rets_all, rule, codes, np.concatenate([[rets_all.index.searchsorted(dates[0])], rows_all]))
    off = rets_all.index.searchsorted(dates[0])
    first = tg_all[int(off)]
    targets = {int(r - off): tg_all[int(r)] for r in rows_all}
    cash = panel.cash.reindex(idx).fillna(0.0).loc[dates]
    eq, _, turnover, _ = simulate(rets_all.loc[dates], cash, targets, cost_bps, 0.0, first=first)
    return eq, turnover


__all__ = ["UNIVERSE", "ROLE", "SHORT", "EQUITY", "CASH_LIKE", "START", "END", "H1", "H2", "REF_START", "DOTCOM",
           "LINE", "NEAR", "SEED", "rqmc", "weights_from_u", "sample", "to_fraction", "Layout", "layout", "path_metrics",
           "zone", "rescale", "neighbours", "exchanges", "drop_one", "scheffe_fit", "scheffe_gradient", "saltelli",
           "sobol_indices", "flat_region", "centres", "rule_targets", "run_rule", "CASH"]
