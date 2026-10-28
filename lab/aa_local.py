"""Local structure study around the search allocations B2 · B3 · B4 (allocation DECISIONS A37).

The point is not to find a better allocation near B2–B4 (that only fits 2005–2026 harder) but to ask
three things about the structure they share:

  ① how wide the good region is: random neighbours at growing distance, and same-role swaps
  ② whether each shared feature is a cause: allocations built from rules R1–R3, against copies with one
     feature taken out (X1–X4) and against unconstrained random allocations from the same products
  ③ whether it leans on one period: crisis drawdowns, half-period ranks, CSCV inside the family and a
     one-year block bootstrap
  ④ the dot-com bust, which the 2005 start never saw: the US proxies are carried back to 2000-09 with
     substitutes (``SUBSTITUTES``, ``extend_history``); everything before 2005 is model

Weights are whole percents over ``aa_search_results.SYMBOLS`` (the A31 universe, so ``describe``,
``code_text`` and ``StaticSim`` work as in stages 1–2); the samplers only use ``UNIVERSE``: the stage-1
"B" subset (no leverage, no synthetic placeholder) without KODEX 200, whose real−model gap of +3.7 %p a
year is unexplained (A35). The script is ``scripts/aa_local_study.py``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from lab import aa_search as srch
from lab import aa_search_results as sr
from lab.data import Panel

# --------------------------------------------------------------------------- products and roles

NASDAQ, SP500 = "379810", "379800"
GOLD = ("411060", "132030")  # ACE KRX금현물, 골드선물(H)
LONG = "484790"  # 미국30년국채액티브(H)
TEN, CD, USD = "0091C0", "459580", "261240"
CORP, TIPS = "468630", "468370"
SECTORS = ("463640", "453630", "453640")  # 유틸리티, 필수소비재, 헬스케어 (the defensive sectors)
REITS = ("352560", "182480")
KODEX200 = "069500"

UNIVERSE = [s for s in sr.SYMBOLS
            if sr.CLASS_OF[s] != sr.LEV_CLASS and s not in sr.SYNTHETIC and s != KODEX200]
EQUITY = [s for s in UNIVERSE if sr.CLASS_OF[s] in sr.EQUITY_CLASSES]
OTHER_EQUITY = [s for s in EQUITY if s != NASDAQ]
FAMILY_SAFE = (CD, TEN, CORP, TIPS, USD)  # where R3 puts the rest
COL = {s: i for i, s in enumerate(sr.SYMBOLS)}

MIN_PCT, MAX_PRODUCTS = 3, 10

# the three centres as stage 2 ran them on real prices (2026-10-26_stage2_real/stage2.csv)
CENTRES: dict[str, dict[str, int]] = {
    "B2": {NASDAQ: 19, "453630": 19, "453640": 6, TEN: 10, LONG: 6, CORP: 5, "132030": 35},
    "B3": {NASDAQ: 33, "251350": 12, TEN: 10, TIPS: 7, "411060": 12, CD: 20, USD: 6},
    "B4": {NASDAQ: 30, "453630": 11, CORP: 8, "411060": 15, CD: 36},
}

# same-role swaps (the whole weight of one product moves to a sibling)
ROLE_SWAPS: dict[str, tuple[str, ...]] = {
    "132030": ("411060",), "411060": ("132030",),
    CD: (USD, TEN), USD: (CD, TEN), TEN: (CD, CORP, TIPS),
    "453630": ("453640", "463640"), "453640": ("453630", "463640"), "463640": ("453630", "453640"),
    CORP: (TIPS, TEN), TIPS: (CORP, TEN),
    NASDAQ: (SP500,), "251350": (SP500,), LONG: (TEN,),
}

# rule bounds (whole percents); the doc's R1–R3
EQUITY_RANGE = (35, 50)
NASDAQ_MIN_SHARE = 0.40  # of the equity total
GOLD_RANGE = (10, 35)
LONG_MAX = 6


def vector(weights: dict[str, int]) -> np.ndarray:
    return sr.to_vector(weights)


def _sum(W: np.ndarray, codes) -> np.ndarray:
    return W[:, [COL[s] for s in codes]].sum(axis=1)


def rules(W: np.ndarray) -> pd.DataFrame:
    """R1 nasdaq-led equity (35–50 %, nasdaq ≥ 40 % of it), R2 gold 10–35 %, R3 at most 6 % long
    treasuries and no REITs; ``all`` = every rule and only universe products."""
    W = np.atleast_2d(np.asarray(W))
    eq = _sum(W, EQUITY)
    nq = W[:, COL[NASDAQ]]
    gold = _sum(W, GOLD)
    r1 = (eq >= EQUITY_RANGE[0]) & (eq <= EQUITY_RANGE[1]) & (nq >= NASDAQ_MIN_SHARE * eq) & (eq > 0)
    r2 = (gold >= GOLD_RANGE[0]) & (gold <= GOLD_RANGE[1])
    r3 = (W[:, COL[LONG]] <= LONG_MAX) & (_sum(W, REITS) == 0)
    inside = W[:, [COL[s] for s in sr.SYMBOLS if s not in UNIVERSE]].sum(axis=1) == 0
    return pd.DataFrame({"R1": r1, "R2": r2, "R3": r3, "all": r1 & r2 & r3 & inside})


def _cap(w: np.ndarray) -> np.ndarray:
    out = srch.round_weights(w, MIN_PCT)
    while (out > 0).sum() > MAX_PRODUCTS:
        live = np.where(out > 0)[0]
        w = w.copy()
        w[live[np.argmin(out[live])]] = 0.0
        out = srch.round_weights(w, MIN_PCT)
    return out


def _split(rng: np.random.Generator, w: np.ndarray, amount: float, pool, k: int) -> None:
    chosen = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    for i, f in zip(chosen, rng.dirichlet(np.ones(len(chosen)))):
        w[COL[pool[i]]] += amount * f


def sample_family(n: int, seed: int) -> np.ndarray:
    """``n`` distinct allocations that satisfy R1–R3 after rounding (rows over SYMBOLS, whole percents).

    Equity total U(35, 50) with nasdaq share U(0.4, 1) of it and the rest over 0–2 other equity products;
    gold U(10, 35) over one or both gold products; long treasuries 0 (half) or U(3, 6); the rest over 1–3
    of ``FAMILY_SAFE``. Dirichlet(1) splits."""
    rng = np.random.default_rng(seed)
    rows, seen = [], set()
    tries = 0
    while len(rows) < n:
        tries += 1
        if tries > 50 * n + 1000:
            raise RuntimeError("구조 가족 표본을 충분히 만들지 못했습니다.")
        w = np.zeros(len(sr.SYMBOLS))
        eq = rng.uniform(*EQUITY_RANGE)
        nq = eq * rng.uniform(NASDAQ_MIN_SHARE, 1.0)
        k_other = int(rng.integers(0, 3)) if eq - nq >= MIN_PCT else 0
        if k_other == 0:
            nq = eq
        w[COL[NASDAQ]] = nq
        if k_other:
            _split(rng, w, eq - nq, OTHER_EQUITY, k_other)
        gold = rng.uniform(*GOLD_RANGE)
        _split(rng, w, gold, GOLD, int(rng.integers(1, 3)))
        long = 0.0 if rng.random() < 0.5 else rng.uniform(MIN_PCT, LONG_MAX)
        w[COL[LONG]] = long
        _split(rng, w, 100.0 - eq - gold - long, FAMILY_SAFE, int(rng.integers(1, 4)))
        v = _cap(w)
        key = v.tobytes()
        if key in seen or not rules(v)["all"].iloc[0]:
            continue
        seen.add(key)
        rows.append(v)
    return np.vstack(rows).astype(np.int64)


def sample_control(n_class: int, picks: int, seed: int) -> np.ndarray:
    """Unconstrained allocations over the same products: the A31 two-level sampler on the universe's
    classes (no leverage class), rows over SYMBOLS."""
    classes = {c: [s for s in syms if s in UNIVERSE] for c, syms in sr.CLASSES.items() if c != sr.LEV_CLASS}
    classes = {c: syms for c, syms in classes.items() if syms}
    W, meta = srch.sample_two_level(classes, n_class=n_class, picks=picks, seed=seed, lev_class=None, lev_cap=0.0,
                                    equity_classes=sr.EQUITY_CLASSES, min_pct=MIN_PCT, max_products=MAX_PRODUCTS)
    out = np.zeros((len(W), len(sr.SYMBOLS)), dtype=np.int64)
    for j, s in enumerate(meta["symbols"]):
        out[:, COL[s]] = W[:, j]
    return out


# --------------------------------------------------------------------------- ablations (X1–X4)

def _spread(amount: int, weights: np.ndarray) -> np.ndarray:
    """Whole-percent split of ``amount`` in proportion to ``weights`` (largest remainder)."""
    raw = amount * weights / weights.sum()
    out = np.floor(raw).astype(np.int64)
    rest = int(amount - out.sum())
    if rest:
        out[np.argsort(-(raw - out), kind="stable")[:rest]] += 1
    return out


def ablate(w: np.ndarray, kind: str) -> np.ndarray | None:
    """One feature taken out of an allocation (whole percents, sum kept at 100); None when the
    allocation does not have that feature.

    X1 nasdaq → S&P500, X2 gold → the allocation's other safe holdings pro rata (CD if none),
    X3 CD + 10년(H) → 30년(H), X4 defensive sectors → S&P500."""
    v = np.asarray(w, dtype=np.int64).copy()
    if kind == "X1":
        m = int(v[COL[NASDAQ]])
        if not m:
            return None
        v[COL[NASDAQ]] = 0
        v[COL[SP500]] += m
    elif kind == "X2":
        cols = [COL[s] for s in GOLD]
        m = int(v[cols].sum())
        if not m:
            return None
        v[cols] = 0
        safe = [COL[s] for s in (*FAMILY_SAFE, LONG) if v[COL[s]] > 0]
        if not safe:
            safe = [COL[CD]]
        v[safe] += _spread(m, v[safe].astype(float) if v[safe].sum() else np.ones(len(safe)))
    elif kind == "X3":
        cols = [COL[CD], COL[TEN]]
        m = int(v[cols].sum())
        if not m:
            return None
        v[cols] = 0
        v[COL[LONG]] += m
    elif kind == "X4":
        cols = [COL[s] for s in SECTORS]
        m = int(v[cols].sum())
        if not m:
            return None
        v[cols] = 0
        v[COL[SP500]] += m
    else:
        raise ValueError(kind)
    return v


ABLATIONS = {"X1": "나스닥 → S&P500", "X2": "금 → 다른 안전자산", "X3": "현금·10년(H) → 30년(H)",
             "X4": "방어 섹터 → S&P500"}


# --------------------------------------------------------------------------- neighbours (①)

def perturb(center: np.ndarray, n: int, seed: int, max_dist: int = 30, new_prob: float = 0.5) -> np.ndarray:
    """``n`` distinct random neighbours of ``center``: the held products are split at random into givers
    and takers; half of an L1 distance drawn from U(2, max_dist) leaves the givers (Dirichlet shares,
    never more than a product holds) and goes to the takers plus, with probability ``new_prob`` (always
    when every held product gives), one product from ``UNIVERSE`` it does not hold (Dirichlet shares).
    Givers and takers never overlap, so the distance before rounding is the one drawn unless a giver
    runs dry. Rounded like every other sample; the centre itself is never returned."""
    rng = np.random.default_rng(seed)
    c = np.asarray(center, dtype=float)
    held = np.where(c > 0)[0]
    absent = [COL[s] for s in UNIVERSE if c[COL[s]] == 0]
    rows, seen = [], {np.asarray(center, dtype=np.int64).tobytes()}
    tries = 0
    while len(rows) < n:
        tries += 1
        if tries > 50 * n + 1000:
            raise RuntimeError("이웃 표본을 충분히 만들지 못했습니다.")
        move = rng.uniform(2, max_dist) / 2.0
        give_mask = rng.random(len(held)) < 0.5
        if not give_mask.any():
            give_mask[int(rng.integers(len(held)))] = True
        givers = held[give_mask]
        gets = list(held[~give_mask])
        if absent and (not gets or rng.random() < new_prob):
            gets.append(absent[int(rng.integers(len(absent)))])
        if not gets:
            continue
        take = np.minimum(c[givers], move * rng.dirichlet(np.ones(len(givers))))
        w = c.copy()
        w[givers] -= take
        w[gets] += take.sum() * rng.dirichlet(np.ones(len(gets)))
        v = _cap(w)
        key = v.tobytes()
        if key in seen:
            continue
        seen.add(key)
        rows.append(v)
    return np.vstack(rows).astype(np.int64)


def substitutions(center: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Every same-role swap of one held product (``ROLE_SWAPS``), labelled '이름 → 이름'."""
    c = np.asarray(center, dtype=np.int64)
    out = []
    for s, sibs in ROLE_SWAPS.items():
        if c[COL[s]] == 0:
            continue
        for q in sibs:
            v = c.copy()
            v[COL[q]] += v[COL[s]]
            v[COL[s]] = 0
            out.append((f"{sr.name_of(s)} → {sr.name_of(q)}", v))
    return out


def l1(W: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.abs(np.atleast_2d(W) - np.asarray(center)[None, :]).sum(axis=1)


DIST_BINS = (0, 5, 10, 20, 30, 200)


def dist_bin(d: np.ndarray) -> np.ndarray:
    labels = np.array([f"{a}–{b}" if b < 200 else f"{a}+" for a, b in zip(DIST_BINS[:-1], DIST_BINS[1:])])
    return labels[np.clip(np.searchsorted(DIST_BINS, d, side="left") - 1, 0, len(labels) - 1)]


# --------------------------------------------------------------------------- bootstrap (③)

def block_bootstrap_mdd(paths: np.ndarray, n: int, block: int = 252, seed: int = 11) -> np.ndarray:
    """MDD of ``n`` circular block-bootstrap histories per path (rows × n). Every path is resampled
    with the same blocks in each draw, so rows are compared on the same made-up histories."""
    paths = np.asarray(paths, dtype=float)
    r = paths[:, 1:] / paths[:, :-1] - 1.0
    T = r.shape[1]
    rng = np.random.default_rng(seed)
    n_blocks = -(-T // block)
    out = np.empty((len(paths), n))
    for b in range(n):
        starts = rng.integers(0, T, size=n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        eq = np.cumprod(1.0 + r[:, idx], axis=1)
        peak = np.maximum.accumulate(np.concatenate([np.ones((len(eq), 1)), eq], axis=1), axis=1)[:, 1:]
        out[:, b] = (eq / peak - 1.0).min(axis=1)
    return out


# --------------------------------------------------------------------------- dot-com extension (④)

DOTCOM_START = "2000-09-01"  # the gold substitute (front-month futures GC=F) starts 2000-08-30
DOTCOM_WINDOW = ("2000-09-01", "2003-12-31")  # nasdaq-100 fell about 80 % from 2000-09 to 2002-10

# proxies that start after DOTCOM_START → substitutes tried in order; the one whose monthly returns track
# the proxy best (correlation) over the proxy's first three years is used. Fixed before the run (A37).
SUBSTITUTES: dict[str, tuple[str, ...]] = {
    "IEF": ("VFITX",),  # Vanguard Intermediate-Term Treasury
    "TLT": ("VUSTX",),  # Vanguard Long-Term Treasury
    "LQD": ("VFICX", "VWESX"),  # Vanguard Intermediate-Term / Long-Term Investment-Grade
    "TIP": ("VIPSX",),  # Vanguard Inflation-Protected Securities (2000-06)
    "GLD": ("GC=F",),  # COMEX gold, front-month continuous
    "EFA": ("VGTSX", "FSIIX", "VDMIX"),  # Vanguard Total Intl / Fidelity Intl Index / Vanguard Developed Markets
    "EEM": ("VEIEX",),  # Vanguard Emerging Markets Index
}


def _monthly(s: pd.Series) -> pd.Series:
    return s.resample("ME").last().pct_change(fill_method=None).dropna()


def extend_history(usd: Panel, subs: Panel, start: str, candidates: dict[str, tuple[str, ...]] | None = None,
                   overlap_years: int = 3) -> tuple[Panel, pd.DataFrame]:
    """Carry each proxy that begins after ``start`` back with the best-tracking substitute.

    Before the proxy's first day its price is the substitute's path scaled to meet it on that day (the
    substitute is read on the proxy panel's dates, carried forward over its own missing days). Only the
    close (which the KRW model reads) is spliced. Returns the panel and one row per proxy: the substitute
    used, the splice day and the overlap statistics (monthly correlation, mean and tracking gap a year)."""
    candidates = SUBSTITUTES if candidates is None else candidates
    start_ts = pd.Timestamp(start)
    idx = usd.close.index
    close = usd.close.copy()
    sub_close = subs.close.reindex(subs.close.index.union(idx)).ffill().reindex(idx)
    rows = []
    for proxy, cands in candidates.items():
        if proxy not in close.columns:
            continue
        first = close[proxy].first_valid_index()
        if first is None or first <= start_ts:
            rows.append({"대용": proxy, "이어 붙인 상품": "", "이음 날짜": None, "비고": "시작 전부터 있음"})
            continue
        tried = []
        for c in cands:
            sc = sub_close[c] if c in sub_close.columns else None
            if sc is None or sc.first_valid_index() is None or sc.first_valid_index() > start_ts or pd.isna(sc.get(first)):
                tried.append({"c": c, "corr": np.nan, "ok": False})
                continue
            end = first + pd.DateOffset(years=overlap_years)
            a, b = _monthly(close[proxy].loc[first:end]), _monthly(sc.loc[first:end])
            a, b = a.align(b, join="inner")
            diff = a - b
            tried.append({"c": c, "ok": True, "corr": float(a.corr(b)), "gap": float(diff.mean() * 12),
                          "te": float(diff.std() * np.sqrt(12)), "months": len(a)})
        ok = [t for t in tried if t["ok"]]
        if not ok:
            rows.append({"대용": proxy, "이어 붙인 상품": "", "이음 날짜": first, "비고": "대체 상품 없음: " + ", ".join(cands)})
            continue
        best = max(ok, key=lambda t: t["corr"])
        c = best["c"]
        before = idx < first
        close.loc[before, proxy] = close.at[first, proxy] * sub_close.loc[before, c] / sub_close.at[first, c]
        rows.append({"대용": proxy, "이어 붙인 상품": c, "이음 날짜": first, "겹친 달": best["months"],
                     "월 상관": best["corr"], "연 평균 차이": best["gap"], "연 추적 차이": best["te"],
                     "비고": "후보 " + ", ".join(f"{t['c']} {t['corr']:.3f}" if t["ok"] else f"{t['c']} 없음" for t in tried)})
    ext = dataclasses.replace(usd, close=close, open=close, high=close, low=close)
    return ext, pd.DataFrame(rows)


__all__ = ["UNIVERSE", "CENTRES", "ROLE_SWAPS", "ABLATIONS", "rules", "sample_family", "sample_control", "ablate",
           "perturb", "substitutions", "l1", "dist_bin", "block_bootstrap_mdd", "vector", "SUBSTITUTES",
           "DOTCOM_START", "DOTCOM_WINDOW", "extend_history"]
