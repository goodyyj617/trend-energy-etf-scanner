"""Stage G-1: tails of the Stage C signals, SPY-relative scoring and big-winner features (DECISIONS 85).

    python -m lab.tail_study --panel DIR --signals DIR --eligible FILE --out DIR [--dry]

Descriptive, no new trials. Reads the Stage C v2 per-signal tables (one equal amount per signal, no
slot limit, filled signals only) and writes aggregate tables only:
  (a)+(b) cells.csv   per sample / period x entry x exit x cell: tail shares (top 1/2/5% of the
                      signals' net profit), expectancy without them, skew, big-winner shares B50 /
                      B5R / B2S, excess over SPY (same window, open to open) and over the equal-weight
                      universe (close index, signal-day close -> close before the sale)
  (c) features.csv    40 signal-day features of big winners vs the rest on the pooled canonical
                      E1-E9 signals (exit X1 25% main, X9 k 2 secondary): medians, univariate AUC by
                      half and period, verdict; deciles.csv (big-winner share by feature decile);
                      placebo.csv (counts of passing features under 200 year-shuffles and 200
                      month-block shuffles); entries.csv (big-winner share by entry signal)
summary.json and figures/ (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures, macro, screens, stage_c_study as sc
from .blocks import atr
from .crsp_study import SAMPLES, load_panel_with_cash
from .data import BENCHMARK, Panel
from .engine import CANCELLED

# ---- registered (DECISIONS 85): fixed before any result
PERIODS = {"p1993": ("1993-02-01", "1999-12-31"), "p2000": ("2000-01-01", "2008-12-31"),
           "p2009": ("2009-01-01", "2016-12-31"), "p2017": ("2017-01-01", "2023-12-31")}
HALVES = {"h1": ("1993-02-01", "2007-12-31"), "h2": ("2008-01-01", "2023-12-31")}
R_ATR = 2.0  # R = 2 x ATR14 (signal day) / entry price
BIG_RET = 0.5  # B50
BIG_R = 5.0  # B5R
SIGMA = 2.0  # B2S: above mean + 2 sd of the cell's signals sold in the previous 365 days
B2S_DAYS, B2S_MIN = 365, 30
TOPS = (0.01, 0.02, 0.05)
FEATURE_EXITS = {"X1": 1, "X9": 1}  # (c): exit and its cell (X1 25%, X9 k 2); X1 is the main one
MAIN_LABEL = "b5r"
AUC_HI, AUC_LO = 0.55, 0.45
N_PLACEBO = 200
SEED = 85
MARKET = ["spy_ma200", "spy_dd52", "vix", "term", "credit", "breadth200"]
FEATURES = ["r1", "r5", "r21", "r63", "r126", "r252", "z1", "z5", "vol_z", "log_dv", "atr_pct", "range_exp",
            "dist_high52", "dist_low52", "days_since_high", "efficiency", "rs", "ma_align", "dist_ma200",
            "vol_contraction", "cap_rank", "log_price", "industry_rs"] + MARKET + [
            "quality", "smooth", "dd_shallow", "rs_spy", "ivol", "lottery", "r756", "accum", "clv", "obv_slope",
            "turnover"]
LABELS = {"r1": "1일 수익", "r5": "5일 수익", "r21": "21일 수익", "r63": "63일 수익", "r126": "126일 수익",
          "r252": "252일 수익", "z1": "1일 수익 z", "z5": "5일 수익 z", "vol_z": "거래량 z", "log_dv": "log 거래대금",
          "atr_pct": "ATR%", "range_exp": "변동폭 확장비", "dist_high52": "52주 고가 거리", "dist_low52": "52주 저가 거리",
          "days_since_high": "52주 고가 이후 일수", "efficiency": "추세 효율", "rs": "IBD식 RS", "ma_align": "이평 정렬",
          "dist_ma200": "200일선 거리", "vol_contraction": "변동성 수축비", "cap_rank": "시총 순위", "log_price": "log 주가",
          "industry_rs": "업종 RS", "spy_ma200": "SPY 200일선 거리", "spy_dd52": "SPY 52주 낙폭", "vix": "VIX(전날)",
          "term": "금리차(전날)", "credit": "신용 스프레드(전날)", "breadth200": "200일선 위 비율",
          "quality": "변동성 대비 모멘텀", "smooth": "모멘텀 매끄러움", "dd_shallow": "낙폭 얕음", "rs_spy": "SPY 대비 초과 6개월",
          "ivol": "고유 변동성", "lottery": "21일 최대 1일 수익", "r756": "36개월 수익", "accum": "매집/분산",
          "clv": "종가 위치", "obv_slope": "OBV 기울기", "turnover": "회전율"}
assert len(FEATURES) == 40 and set(FEATURES) == set(LABELS)


def log(m: str) -> None:
    print(time.strftime("%H:%M:%S"), m, flush=True)


# --------------------------------------------------------------------------- per-signal measures

class Prices:
    """Arrays on the panel's full date index for the per-signal measures."""

    def __init__(self, panel: Panel, eligible: pd.DataFrame):
        self.dates = panel.close.index
        self.cols = panel.close.columns
        self.O = panel.open.to_numpy(dtype="float32")
        self.LC = panel.close.ffill().to_numpy(dtype="float32")
        self.ATR = atr(panel, 14).to_numpy(dtype="float32")
        self.spy_o = panel.open[BENCHMARK].to_numpy(dtype=float)
        self.spy_c = panel.close[BENCHMARK].ffill().to_numpy(dtype=float)
        self.ew = equal_weight_index(panel, eligible)

    def measures(self, sig: pd.DataFrame, end: str) -> pd.DataFrame:
        """Per filled signal: return, r_mult (return / R), spy, ew excess, sale index, signal index."""
        last = int(self.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
        si = self.dates.get_indexer(pd.DatetimeIndex(sig["signal_date"]))
        ei = self.dates.get_indexer(pd.DatetimeIndex(sig["entry_date"]))
        sj = self.cols.get_indexer(sig["symbol"].astype(str))
        closed = sig["closed"].to_numpy(dtype=bool)
        sale = np.where(closed, ei + sig["holding_days"].to_numpy(dtype=np.int64), last)
        ret = sig["return"].to_numpy(dtype=float)
        px = self.O[ei, sj].astype(float)
        r_unit = R_ATR * self.ATR[si, sj].astype(float) / px
        spy = np.where(closed, self.spy_o[np.minimum(sale, len(self.dates) - 1)], self.spy_c[last]) / self.spy_o[ei] - 1.0
        ew_end = np.where(closed, sale - 1, last)
        ew = self.ew[ew_end] / self.ew[si] - 1.0
        return pd.DataFrame({"si": si, "ei": ei, "sj": sj, "sale": sale, "closed": closed, "ret": ret,
                             "r_mult": np.where(r_unit > 0, ret / r_unit, np.nan), "x_spy": ret - spy, "x_ew": ret - ew})


def equal_weight_index(panel: Panel, eligible: pd.DataFrame) -> np.ndarray:
    """Close-based level of the hygiene-eligible top-3,000 universe in equal weight, re-weighted daily
    (names eligible on day t-1 earn their close-to-close return on day t)."""
    c = panel.close
    r = (c / c.ffill().shift(1) - 1.0).to_numpy(dtype=float)
    held = eligible.reindex(index=c.index, columns=c.columns, fill_value=False).shift(1, fill_value=False).to_numpy(dtype=bool)
    held &= np.isfinite(r)
    n = held.sum(axis=1)
    day = np.where(n > 0, np.where(held, r, 0.0).sum(axis=1) / np.maximum(n, 1), 0.0)
    return np.cumprod(1.0 + day)


def b2s_flags(m: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """1/0 = above / not above mean + SIGMA sd of the same cell's signals sold in the B2S_DAYS before
    the signal day (known by then); NaN when fewer than B2S_MIN such signals."""
    n = len(dates)
    ok = m["closed"].to_numpy() & np.isfinite(m["ret"].to_numpy())
    s = m["sale"].to_numpy()[ok]
    r = m["ret"].to_numpy()[ok]
    c0 = np.concatenate([[0.0], np.cumsum(np.bincount(s, minlength=n)[:n])])
    c1 = np.concatenate([[0.0], np.cumsum(np.bincount(s, weights=r, minlength=n)[:n])])
    c2 = np.concatenate([[0.0], np.cumsum(np.bincount(s, weights=r * r, minlength=n)[:n])])
    si = m["si"].to_numpy()
    lo = dates.searchsorted(dates[si] - pd.Timedelta(days=B2S_DAYS))  # sold on day lo .. si (sale at the open)
    hi = si + 1
    k = c0[hi] - c0[lo]
    mu = (c1[hi] - c1[lo]) / np.maximum(k, 1)
    var = (c2[hi] - c2[lo]) / np.maximum(k, 1) - mu ** 2
    thr = mu + SIGMA * np.sqrt(np.maximum(var, 0.0) * k / np.maximum(k - 1, 1))
    out = (m["ret"].to_numpy() > thr).astype(float)
    out[k < B2S_MIN] = np.nan
    return out


def tail_stats(m: pd.DataFrame) -> dict:
    r = m["ret"].to_numpy()
    r = r[np.isfinite(r)]
    n = len(r)
    out = {"signals": n}
    if not n:
        return out
    srt = np.sort(r)[::-1]
    tot = r.sum()
    out.update(mean=float(r.mean()), median=float(np.median(r)), win=float((r > 0).mean()),
               skew=float(pd.Series(r).skew()) if n > 2 else np.nan, net=float(tot))
    for t in TOPS:
        k = max(1, int(round(n * t)))
        out[f"top{int(t * 100)}_share"] = float(srt[:k].sum() / tot) if tot != 0 else np.nan
        out[f"mean_wo_top{int(t * 100)}"] = float(srt[k:].mean()) if n > k else np.nan
    out["b50"] = float((r >= BIG_RET).mean())
    rm = m["r_mult"].to_numpy()
    out["b5r"] = float(np.nanmean(rm >= BIG_R)) if np.isfinite(rm).any() else np.nan
    b2 = m["b2s"].to_numpy()
    out["b2s"] = float(np.nanmean(b2)) if np.isfinite(b2).any() else np.nan
    out["b2s_covered"] = float(np.isfinite(b2).mean())
    for k in ("x_spy", "x_ew"):
        x = m[k].to_numpy()
        out[f"mean_{k}"] = float(np.nanmean(x))
        out[f"median_{k}"] = float(np.nanmedian(x))
        out[f"beat_{k[2:]}"] = float(np.nanmean(x > 0))
        big = (rm >= BIG_R) & np.isfinite(x)
        out[f"big_mean_{k}"] = float(x[big].mean()) if big.any() else np.nan
    return out


def overlap_share(sig: pd.DataFrame) -> float:
    """Share of signals issued while an earlier signal of the same symbol (same cell) is still held."""
    s = sig.sort_values(["symbol", "signal_date"])
    sym = s["symbol"].to_numpy()
    sd = pd.DatetimeIndex(s["signal_date"]).to_numpy()
    sale = pd.DatetimeIndex(s["exit_signal_date"]).to_numpy()
    sale = np.where(pd.isna(sale), np.datetime64("2262-01-01"), sale)
    prev_max = pd.Series(sale).groupby(sym).cummax().groupby(sym).shift(1).to_numpy()
    over = (~pd.isna(prev_max)) & (sd < prev_max)
    return float(over.mean()) if len(over) else np.nan


def run_cells(px: Prices, signals: Path, out: Path) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    pairs_ = [(e, x) for e in sc.ENTRIES if e not in sc.NO_SIGNAL_TABLE for x in sc.EXITS]
    for n, (e, x) in enumerate(pairs_):
        for sample, (s0, s1, _) in SAMPLES.items():
            sig_all = pd.read_parquet(signals / "signals" / sample / f"{e}__{x}_signals.parquet")
            for (xi, yi), sig in sig_all.groupby(["xi", "yi"]):
                sig = sig[sig["exit_reason"] != CANCELLED].reset_index(drop=True)
                m = px.measures(sig, s1)
                m["b2s"] = b2s_flags(m, px.dates)
                base = {"entry": e, "exit": x, "xi": int(xi), "yi": int(yi), "canonical": (xi, yi) == sc.canonical(e, x)}
                rows.append({"sample": sample, **base, **tail_stats(m), "overlap": overlap_share(sig)})
                if sample == "full":
                    sd = px.dates[m["si"].to_numpy()]
                    for p, (a, b) in PERIODS.items():
                        sel = (sd >= pd.Timestamp(a)) & (sd <= pd.Timestamp(b))
                        rows.append({"sample": p, **base, **tail_stats(m[sel])})
        log(f"  꼬리 · SPY 대비 [{n + 1}/{len(pairs_)}] {e}×{x} · 남은 예상 {(len(pairs_) - n - 1) * (time.time() - t0) / (n + 1) / 60:.0f}분")
    c = pd.DataFrame(rows)
    c.to_csv(out / "cells.csv", index=False)
    return c


# --------------------------------------------------------------------------- (c) features

def pooled_signals(px: Prices, signals: Path, exit_: str) -> pd.DataFrame:
    """E1-E9 canonical-cell signals of one exit (its registered cell), one row per stock-day, with
    the entry signals that fired (outcomes depend only on the stock-day and the exit)."""
    parts = []
    for e in sc.ENTRIES:
        if e in sc.NO_SIGNAL_TABLE:
            continue
        s = pd.read_parquet(signals / "signals" / "full" / f"{e}__{exit_}_signals.parquet")
        s = s[(s["xi"] == sc.ENTRY_CANON[e]) & (s["yi"] == FEATURE_EXITS[exit_]) & (s["exit_reason"] != CANCELLED)]
        parts.append(s.assign(entry=e))
    s = pd.concat(parts, ignore_index=True)
    flags = pd.crosstab([s["symbol"], s["signal_date"]], s["entry"]).clip(upper=1)
    s = s.drop_duplicates(["symbol", "signal_date"]).set_index(["symbol", "signal_date"]).join(flags).reset_index()
    m = px.measures(s, SAMPLES["full"][1])
    m["b2s"] = b2s_flags(m, px.dates)
    s = pd.concat([s.reset_index(drop=True), m.drop(columns=["ret", "closed", "ei"])], axis=1)
    s["b50"] = (s["return"] >= BIG_RET).astype(float)
    s["b5r"] = np.where(np.isfinite(s["r_mult"]), (s["r_mult"] >= BIG_R).astype(float), np.nan)
    return s


def _rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def days_since_high(c: pd.DataFrame, hi252: pd.DataFrame, window: int = 252, chunk: int = 64) -> pd.DataFrame:
    """Sessions since the highest close of the last `window` sessions (0 = today); NaN where the
    252-day high itself is NaN. Column chunks keep the sliding windows small."""
    from numpy.lib.stride_tricks import sliding_window_view

    a = c.to_numpy(dtype="float32")
    out = np.full(a.shape, np.nan, dtype="float32")
    for k in range(0, a.shape[1], chunk):
        b = np.where(np.isfinite(a[:, k:k + chunk]), a[:, k:k + chunk], -np.inf)
        w = sliding_window_view(b, window, axis=0)  # (rows - window + 1, cols, window)
        last = window - 1 - np.argmax(w[:, :, ::-1], axis=2)  # last position of the max in each window
        out[window - 1:, k:k + chunk] = window - 1 - last
    return pd.DataFrame(out, index=c.index, columns=c.columns).where(hi252.notna())


def feature_frames(panel: Panel, eligible: pd.DataFrame):
    """Yields (name, dates x symbols frame or per-day Series) for the 40 registered features, one at a time."""
    c, o, h, lo, v = panel.close, panel.open, panel.high, panel.low, panel.volume
    lc = c.ffill()
    r = c / lc.shift(1) - 1.0
    for k, n in (("r1", 1), ("r5", 5), ("r21", 21), ("r63", 63), ("r126", 126), ("r252", 252)):
        yield k, c / lc.shift(n) - 1.0
    sd63 = r.rolling(63, min_periods=40).std()
    yield "z1", r / sd63
    yield "z5", (c / lc.shift(5) - 1.0) / (sd63 * np.sqrt(5))
    lv = np.log(v.where(v > 0))
    yield "vol_z", (lv - lv.shift(1).rolling(50, min_periods=30).mean()) / lv.shift(1).rolling(50, min_periods=30).std()
    yield "log_dv", np.log((c * v).rolling(21, min_periods=15).mean())
    a14 = atr(panel, 14)
    yield "atr_pct", a14 / c
    prev = lc.shift(1)
    tr = np.fmax(np.fmax(h - lo, (h - prev).abs()), (lo - prev).abs())
    yield "range_exp", tr / atr(panel, 20).shift(1)
    hi252 = c.rolling(252, min_periods=200).max()
    yield "dist_high52", c / hi252 - 1.0
    yield "dist_low52", c / c.rolling(252, min_periods=200).min() - 1.0
    yield "days_since_high", days_since_high(c, hi252)
    yield "efficiency", screens.SELECT_SCORES["s_efficiency"].compute(panel, 6)
    yield "rs", screens.SELECT_SCORES["s_rs"].compute(panel, None)
    m20, m50, m200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200, min_periods=180).mean()
    yield "ma_align", ((c > m20).astype(float) + (m20 > m50).astype(float) + (m50 > m200).astype(float)).where(m200.notna())
    yield "dist_ma200", c / m200 - 1.0
    del m20, m50
    yield "vol_contraction", r.rolling(20, min_periods=15).std() / r.rolling(252, min_periods=200).std()
    yield "cap_rank", panel.extra["cap_rank"]
    yield "log_price", np.log(c.where(c > 0))
    yield "industry_rs", screens.SELECT_SCORES["s_industry_rs"].compute(panel, 6)
    spy = c[BENCHMARK].ffill()
    yield "spy_ma200", spy / spy.rolling(200).mean() - 1.0
    yield "spy_dd52", spy / spy.rolling(252).max() - 1.0
    for k, sid in (("vix", "VIXCLS"), ("term", "T10Y3M"), ("credit", "BAA10Y")):
        yield k, macro.series(sid, c.index)[0]
    el = eligible.reindex(index=c.index, columns=c.columns, fill_value=False)
    above = (c > m200) & el
    yield "breadth200", above.sum(axis=1) / el.sum(axis=1).replace(0, np.nan)
    del above, m200
    for k, key, p in (("quality", "s_quality", 12), ("smooth", "s_smooth", 12), ("dd_shallow", "s_dd_shallow", 12),
                      ("rs_spy", "s_rs_spy", 6)):
        yield k, screens.SELECT_SCORES[key].compute(panel, p)
    rs = spy / spy.shift(1) - 1.0
    roll = lambda f: f.rolling(63, min_periods=40).mean()  # noqa: E731
    m_s = roll(rs)
    cov = roll(r.mul(rs, axis=0)) - roll(r).mul(m_s, axis=0)
    beta = cov.div(roll(rs * rs) - m_s * m_s, axis=0)
    resid = r - beta.mul(rs, axis=0)
    yield "ivol", resid.rolling(63, min_periods=40).std()
    del cov, beta, resid
    yield "lottery", r.rolling(21, min_periods=15).max()
    yield "r756", c / lc.shift(756) - 1.0
    for k, key, p in (("accum", "s_accum", 3), ("clv", "s_clv", 3), ("obv_slope", "s_obv_slope", 3), ("turnover", "s_turnover", None)):
        yield k, screens.SELECT_SCORES[key].compute(panel, p)


def attach_features(panel: Panel, eligible: pd.DataFrame, pooled: dict[str, pd.DataFrame]) -> None:
    """Reads every feature on each pooled signal's (signal day, symbol)."""
    idx, cols = panel.close.index, panel.close.columns
    for name, f in feature_frames(panel, eligible):
        t = time.time()
        for s in pooled.values():
            if isinstance(f, pd.Series):
                s[name] = f.reindex(idx).to_numpy(dtype=float)[s["si"].to_numpy()]
            else:
                a = f.reindex(index=idx, columns=cols).to_numpy(dtype=float)
                s[name] = a[s["si"].to_numpy(), s["sj"].to_numpy()]
        log(f"    특징 {LABELS[name]}: {time.time() - t:.0f}s")


def auc(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    rk = pd.Series(x).rank().to_numpy()
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _masks(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    out = {}
    for k, (a, b) in {**HALVES, **PERIODS}.items():
        out[k] = np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
    return out


def separates(row: dict) -> bool:
    h = [row[f"auc_{k}"] for k in HALVES]
    p = [row[f"auc_{k}"] - 0.5 for k in PERIODS]
    if any(not np.isfinite(v) for v in h + p):
        return False
    up = all(v >= AUC_HI for v in h) and all(v > 0 for v in p)
    down = all(v <= AUC_LO for v in h) and all(v < 0 for v in p)
    return bool(up or down)


def feature_table(s: pd.DataFrame, label: str) -> pd.DataFrame:
    dates = pd.DatetimeIndex(s["signal_date"])
    y = s[label].to_numpy(dtype=float)
    masks = _masks(dates)
    rows = []
    for f in FEATURES:
        x = s[f].to_numpy(dtype=float)
        big = y == 1
        rest = y == 0
        row = {"feature": f, "label_ko": LABELS[f], "market": f in MARKET, "coverage": float(np.isfinite(x).mean()),
               "median_big": float(np.nanmedian(x[big])) if big.any() else np.nan,
               "median_rest": float(np.nanmedian(x[rest])) if rest.any() else np.nan, "auc_all": auc(x, y)}
        for k, msk in masks.items():
            row[f"auc_{k}"] = auc(x[msk], y[msk])
        row["separates"] = separates(row)
        rows.append(row)
    return pd.DataFrame(rows)


def decile_table(s: pd.DataFrame, label: str) -> pd.DataFrame:
    y = s[label].to_numpy(dtype=float)
    base = np.nanmean(y)
    rows = []
    for f in FEATURES:
        x = s[f]
        ok = x.notna() & np.isfinite(y)
        if ok.sum() < 100:
            continue
        q = pd.qcut(x[ok].rank(method="first"), 10, labels=False)
        rate = pd.Series(y[ok.to_numpy()]).groupby(q.to_numpy()).mean()
        for d, v in rate.items():
            rows.append({"feature": f, "decile": int(d) + 1, "big_share": float(v), "lift": float(v / base) if base > 0 else np.nan})
    return pd.DataFrame(rows)


def placebo(s: pd.DataFrame, label: str, n: int, seed: int = SEED) -> pd.DataFrame:
    """Number of 'separating' features when the big-winner labels are shuffled: within each year
    (signal) and by month blocks (months permuted, labels drawn with replacement from the paired
    month), n times each. Ranks are computed once per feature and subset; a draw only sums the
    ranks of its positives."""
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(s["signal_date"])
    y = s[label].to_numpy(dtype=float)
    ok_y = np.isfinite(y)
    masks = _masks(dates)
    year = dates.year.to_numpy()
    month = (dates.year * 12 + dates.month).to_numpy()
    pre = {}  # (feature, subset) -> (ranks on the full index, NaN outside; count of usable rows)
    for f in FEATURES:
        x = s[f].to_numpy(dtype=float)
        for k, msk in masks.items():
            use = msk & np.isfinite(x) & ok_y
            rk = np.full(len(x), np.nan)
            rk[use] = pd.Series(x[use]).rank().to_numpy()
            pre[(f, k)] = (rk, int(use.sum()))
    months = np.unique(month[ok_y])
    by_month = {m_: np.flatnonzero((month == m_) & ok_y) for m_ in months}
    by_year = {yr: np.flatnonzero((year == yr) & ok_y) for yr in np.unique(year)}
    rows = []
    for kind in ("signal", "month_block"):
        for b in range(n):
            yp = np.full(len(y), np.nan)
            if kind == "signal":
                for p_ in by_year.values():
                    yp[p_] = y[rng.permutation(p_)]
            else:
                for m_, src in zip(months, rng.permutation(months)):
                    dst = by_month[m_]
                    yp[dst] = y[rng.choice(by_month[src], size=len(dst), replace=True)]
            pos = np.flatnonzero(yp == 1)
            cnt_all = cnt_mkt = 0
            for f in FEATURES:
                row = {}
                for k in masks:
                    rk, tot = pre[(f, k)]
                    v = rk[pos]
                    v = v[np.isfinite(v)]
                    n1, n0 = len(v), tot - len(v)
                    row[f"auc_{k}"] = (v.sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
                if separates(row):
                    cnt_all += 1
                    cnt_mkt += f in MARKET
            rows.append({"kind": kind, "draw": b, "separating": cnt_all, "separating_market": cnt_mkt,
                         "separating_other": cnt_all - cnt_mkt})
    return pd.DataFrame(rows)


def run_features(panel: Panel, eligible: pd.DataFrame, px: Prices, signals: Path, out: Path) -> dict:
    pooled = {x: pooled_signals(px, signals, x) for x in FEATURE_EXITS}
    for x, s in pooled.items():
        log(f"  (다) {x}: 신호 {len(s):,}개(종목·날짜별로 합침), 큰 승자 B5R {np.nanmean(s['b5r']):.3%}")
    attach_features(panel, eligible, pooled)
    res = {}
    for x, s in pooled.items():
        ent = []
        for e in [k for k in sc.ENTRIES if k in s]:
            g = s[s[e] == 1]
            ent.append({"exit": x, "entry": e, "signals": len(g), **{k: float(np.nanmean(g[k])) for k in ("b50", "b5r", "b2s")}})
        pd.DataFrame(ent).to_csv(out / f"entries_{x}.csv", index=False)
        for label in ("b5r", "b50", "b2s"):
            ft = feature_table(s, label)
            ft.to_csv(out / f"features_{x}_{label}.csv", index=False)
            res[(x, label)] = ft
        decile_table(s, MAIN_LABEL).to_csv(out / f"deciles_{x}.csv", index=False)
    s = pooled["X1"]
    pl = placebo(s, MAIN_LABEL, N_PLACEBO)
    pl.to_csv(out / "placebo.csv", index=False)
    ft = res[("X1", MAIN_LABEL)]
    q = lambda kind, col: float(pl.loc[pl["kind"] == kind, col].quantile(0.95))  # noqa: E731
    sep_other = ft.loc[ft["separates"] & ~ft["market"], "feature"].tolist()
    sep_mkt = ft.loc[ft["separates"] & ft["market"], "feature"].tolist()
    thr_other = q("signal", "separating_other")
    thr_mkt = max(q("signal", "separating_market"), q("month_block", "separating_market"))
    verdict = {"separating_other": sep_other, "separating_market": sep_mkt,
               "placebo95_other_signal": thr_other, "placebo95_other_month_block": q("month_block", "separating_other"),
               "placebo95_market_strict": thr_mkt,
               "placebo_mean_all_signal": float(pl.loc[pl["kind"] == "signal", "separating"].mean()),
               "placebo_mean_all_month_block": float(pl.loc[pl["kind"] == "month_block", "separating"].mean())}
    n_sep = len(sep_other) + len(sep_mkt)
    above = len(sep_other) > thr_other or len(sep_mkt) > thr_mkt
    verdict["n_separating"] = n_sep
    verdict["rule"] = "selection_model" if (n_sep >= 3 and above) else "sizing_pyramiding"
    verdict["pooled_signals"] = {x: int(len(v)) for x, v in pooled.items()}
    verdict["overlap_pooled"] = {x: float(overlap_share(v)) for x, v in pooled.items()}
    return verdict


# --------------------------------------------------------------------------- figures and main

def write_figures(out: Path, c: pd.DataFrame, ft: pd.DataFrame, ft_b50: pd.DataFrame | None = None) -> list[Path]:
    if not figures.available():
        return []
    d = out / "figures"
    made = []
    can = c[c["canonical"] & (c["sample"].isin(list(SAMPLES)))]
    names = {k: SAMPLES[k][2] for k in SAMPLES}
    for stat, title, unit in (("mean", "신호당 평균 수익 (대표 칸, 매도별 중앙값)", "%"),
                              ("mean_wo_top5", "상위 5% 신호를 뺀 신호당 평균 수익 (대표 칸, 매도별 중앙값)", "%"),
                              ("median_x_spy", "신호당 SPY 대비 초과 중앙값 (대표 칸, 매도별 중앙값)", "%p"),
                              ("b5r", "큰 승자(≥ 5R) 비율 (대표 칸, 매도별 중앙값)", "%")):
        m = can.groupby(["exit", "sample"])[stat].median().unstack("sample").reindex(list(sc.EXITS))
        made.append(figures.paired_bars([f"{x} {sc.EXITS[x][0]}" for x in m.index], {names[s]: m[s].to_numpy(dtype=float) for s in SAMPLES},
                                        title, unit, d / f"{stat}_by_exit.png", pct=True, legend_out=True))
    for f, label, name in ((ft, "≥ 5R", "feature_auc.png"), (ft_b50, "≥ +50%", "feature_auc_b50.png")):
        if f is None:
            continue
        f = f.sort_values("auc_all")
        made.append(figures.paired_bars([LABELS[k] for k in f["feature"]],
                                        {"1993–2007": (f["auc_h1"] - 0.5).to_numpy(), "2008–2023": (f["auc_h2"] - 0.5).to_numpy()},
                                        f"큰 승자({label}, 매도 X1) 가르는 힘: AUC − 0.5 (기준 ±0.05)", "AUC − 0.5", d / name,
                                        pct=False, ref=AUC_HI - 0.5, ref_label="기준 +0.05 (−0.05도 같음)", legend_out=True))
    return made


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--signals", required=True, help="stagec_signals_v2 폴더")
    ap.add_argument("--eligible", required=True, help="Stage C의 eligible_daily.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true", help="E1·E9 × X1·X9만, 위약 5번")
    ap.add_argument("--skip-cells", action="store_true", help="(가)(나)는 끝난 cells.csv를 다시 씀")
    a = ap.parse_args(argv)
    global N_PLACEBO
    if a.dry:
        for k in [k for k in sc.ENTRIES if k not in ("E0", "E1", "E9")]:
            del sc.ENTRIES[k]
        for k in [k for k in sc.EXITS if k not in ("X1", "X9")]:
            del sc.EXITS[k]
        N_PLACEBO = 5
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    signals = Path(a.signals)
    panel = load_panel_with_cash(Path(a.panel))
    eligible = pd.read_parquet(a.eligible)
    panel.eligible = eligible
    log("가격 배열 · 동일비중 지수 준비")
    px = Prices(panel, eligible)
    if a.skip_cells and (out / "cells.csv").exists():
        c = pd.read_csv(out / "cells.csv")
    else:
        log("(가)(나) 꼬리 집중도 · SPY 대비 채점")
        c = run_cells(px, signals, out)
    log("(다) 큰 승자의 신호일 특징")
    verdict = run_features(panel, eligible, px, signals, out)
    ft = pd.read_csv(out / f"features_X1_{MAIN_LABEL}.csv")
    write_figures(out, c, ft, pd.read_csv(out / "features_X1_b50.csv"))
    can = c[c["canonical"]]
    summ = {"verdict": verdict,
            "canonical_median": can.groupby("sample")[["top1_share", "top2_share", "top5_share", "mean_wo_top5", "skew", "b50", "b5r",
                                                       "b2s", "median_x_spy", "mean_x_spy", "beat_spy", "median_x_ew", "beat_ew",
                                                       "big_mean_x_spy"]].median().round(4).to_dict("index"),
            "overlap_median": float(c.loc[c["sample"] == "full", "overlap"].median()),
            "n_placebo": N_PLACEBO}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"완료 · 가르는 특징 {verdict['n_separating']}개 → {verdict['rule']} · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
