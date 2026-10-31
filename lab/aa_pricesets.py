"""A42 price sets (allocation DECISIONS A42 §0): prices × hedge cost × KRW cash, and the stage-0 estimates.

    build(prices="real", hedge_bp=20, cash="base")  # the A41 panel (bit-identical, tests/test_aa_pricesets.py)

- prices: "model" = US proxies × USD/KRW only; "real" = real Korean prices after listing (A33, A41 chains);
  "adjusted" = like "real", but before each product's real prices the model's daily return gets the product's
  measured real-minus-model gap (A42 0-7, `calibration`): overlap ≥ 1 year, reason known; KODEX 200 is excluded.
- hedge_bp: extra hedging cost of (H) products beyond the rate gap, bp a year (0, 20, 40, 60).
- cash: "base" = Bank of Korea base rate (as before), "cd91" = CD 91-day monthly average (ECOS 721Y001) — used for
  KRW cash, the CD-rate ETF before listing and the KRW leg of the hedge.
- quirks: True applies aa_real.QUIRKS to the real prices (A42 runs); False = as before.
- The research window starts 2000-09-01; earlier starts raise unless unseal=True (lab/aa_sealed.py).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import aa_local as loc
from . import aa_real
from . import aa_sealed
from .aa_assets import PRODUCTS, PROXY_SYMBOLS, Product, _p
from .aa_korea import KrwModel, cd91_schedule, krw_panel, load_bok_rate, load_usdkrw, product_returns
from .data import download_panel

DATA_END = "2026-10-27"
START = aa_sealed.RESEARCH_START
PRICES = ("model", "adjusted", "real")
HEDGE_BP = (0.0, 20.0, 40.0, 60.0)
CASH = ("base", "cd91")
NO_CALIBRATE = {"069500": "KODEX 200의 실제–모형 차이(+3.7%p)는 원인이 밝혀지지 않음(A33)"}
MIN_OVERLAP_DAYS = 252
HEDGE_PROBES = {  # 2017–18 listed hedged US treasury ETFs (A42 0-2), with the proxy the model would use
    "267440": _p("267440", "RISE 미국장기국채선물(H)", "채권", [("TLT", 1)], "hedged", 0.40, "2017-04-20"),
    "304660": _p("304660", "KODEX 미국30년국채울트라선물(H)", "채권", [("TLT", 1)], "hedged", 0.30, "2018-09-12"),
}


@dataclass(frozen=True)
class PriceSet:
    prices: str = "real"
    hedge_bp: float = 20.0
    cash: str = "base"

    @property
    def label(self) -> str:
        return f"{self.prices}·h{self.hedge_bp:g}·{self.cash}"


def all_sets() -> list[PriceSet]:
    return [PriceSet(p, h, c) for p, h, c in itertools.product(PRICES, HEDGE_BP, CASH)]


_USD_CACHE: dict = {}
LATE_CHAINS = {"SCHD": "DVY"}  # proxy → substitute that starts after the research start (partial history, A42 0-4)


def usd_extended(start: str = START, unseal: bool = False):
    """US proxy panel carried back to ``start`` (A37 extend_history), cached per process."""
    aa_sealed.guard(start, unseal)
    key = (start, unseal)
    if key not in _USD_CACHE:
        usd, failed = download_panel(sorted(set(PROXY_SYMBOLS) | {"IEF", "TLT", "SHY"}), end=DATA_END)
        subs, failed_s = download_panel(sorted({c for v in loc.SUBSTITUTES.values() for c in v}), end=DATA_END)
        ext, table = loc.extend_history(usd, subs, start)
        # A42 0-4: SCHD is carried back with DVY from DVY's own first day (2003-11), not from `start` — extend_history
        # only takes substitutes that cover `start`, so it left SCHD at 2011-10 (found in stage 2, design change log)
        for proxy, sub in LATE_CHAINS.items():
            if proxy in ext.close.columns and sub in subs.close.columns and subs.close[sub].first_valid_index() is not None:
                s0 = str(subs.close[sub].first_valid_index().date())
                ext, more = loc.extend_history(ext, subs, max(s0, start), candidates={proxy: (sub,)})
                table = pd.concat([table[table["대용"] != proxy], more], ignore_index=True)
        _USD_CACHE[key] = (ext, table, failed + failed_s)
    return _USD_CACHE[key]


def model_panel(ps: PriceSet, start: str = START, unseal: bool = False, products: dict[str, Product] | None = None):
    ext, table, _ = usd_extended(start, unseal)
    bok = load_bok_rate() if ps.cash == "base" else cd91_schedule()
    krw, unavailable = krw_panel(ext, load_usdkrw(), bok, KrwModel(hedge_extra_bps=float(ps.hedge_bp)), products=products or PRODUCTS)
    return krw, unavailable, table


CALIBRATION_END = "2024-12-31"  # the overlap stops at the research window's end: 2025– is the spent hold-out


def calibration(model, real: dict[str, pd.Series], chains=aa_real.A41_CHAINS, end: str = CALIBRATION_END) -> pd.DataFrame:
    """Per column: overlap of its own real prices with the model (from the splice to ``end``), the annual gap
    real − model (mean daily difference × 252), tracking error, monthly correlation, and whether it is used."""
    hyb, info = aa_real.hybrid_panel(model, real, chains=chains)
    rows = []
    for col, d in info.items():
        t0, last = d.get("splice"), d.get("real_last")
        if t0 is None or last is None:
            continue
        idx = model.close.index
        sl = (idx >= t0) & (idx <= min(pd.Timestamp(last), pd.Timestamp(end)))
        m = model.close.loc[sl, col].pct_change(fill_method=None)
        r = hyb.close.loc[sl, col].pct_change(fill_method=None)
        ok = m.notna() & r.notna()
        n = int(ok.sum())
        gap = float((r[ok] - m[ok]).mean() * 252) if n else np.nan
        te = float((r[ok] - m[ok]).std(ddof=1) * np.sqrt(252)) if n > 2 else np.nan
        mm = (1 + m[ok]).resample("ME").prod() - 1
        rm = (1 + r[ok]).resample("ME").prod() - 1
        corr = float(mm.corr(rm)) if len(mm) > 3 else np.nan
        use = n >= MIN_OVERLAP_DAYS and col not in NO_CALIBRATE and np.isfinite(gap)
        why = "" if use else (NO_CALIBRATE.get(col) or f"겹침 {n}일 < {MIN_OVERLAP_DAYS}일")
        rows.append({"코드": col, "실제 시작": str(pd.Timestamp(t0).date()), "겹침 일": n, "연 차이 (실제−모형)": gap,
                     "추적오차 (연)": te, "월 상관": corr, "보정에 씀": use, "안 쓰는 이유": why})
    return pd.DataFrame(rows)


def adjusted_model(model, cal: pd.DataFrame, info: dict):
    """Model panel whose returns before each used product's first real date get the measured gap added."""
    import copy
    close = model.close.copy()
    for _, row in cal[cal["보정에 씀"]].iterrows():
        col = row["코드"]
        t0 = min((t for _, t in info[col]["chain"]), default=None) if info.get(col) else None
        if t0 is None:
            continue
        r = model.close[col].pct_change(fill_method=None)
        before = r.index < t0
        r[before] = r[before] + row["연 차이 (실제−모형)"] / 252.0
        s = model.close[col].copy()
        first = s.first_valid_index()
        vals = (1 + r.loc[first:].fillna(0.0)).cumprod() * s.loc[first]
        vals.iloc[0] = s.loc[first]
        close.loc[first:, col] = vals
    out = copy.copy(model)
    out.open = out.high = out.low = out.close = close
    out.source = model.source + " · 상장 전 모형에 실제–모형 차이 보정(A42)"
    for attr in ("benchmark", "fx"):
        if hasattr(model, attr):
            setattr(out, attr, getattr(model, attr))
    return out


def build(prices: str = "real", hedge_bp: float = 20.0, cash: str = "base", start: str = START, quirks: bool = False,
          unseal: bool = False, products: dict[str, Product] | None = None):
    """(panel, info) for one price set. info = aa_real splice info (None for the model)."""
    if prices not in PRICES or cash not in CASH:
        raise ValueError(f"prices {PRICES}, cash {CASH}")
    ps = PriceSet(prices, float(hedge_bp), cash)
    model, _, _ = model_panel(ps, start, unseal, products)
    if prices == "model":
        return model, None
    files = aa_real.load_files()
    if files is None:
        raise FileNotFoundError("실제 가격 파일이 없습니다 (python -m lab.aa_real fetch)")
    real = aa_real.real_series(*files, quirks=aa_real.QUIRKS if quirks else None)
    if prices == "real":
        return aa_real.hybrid_panel(model, real, chains=aa_real.A41_CHAINS)
    _, info = aa_real.hybrid_panel(model, real, chains=aa_real.A41_CHAINS)
    cal = calibration(model, real)
    adj = adjusted_model(model, cal, info)
    return aa_real.hybrid_panel(adj, real, chains=aa_real.A41_CHAINS)


def hedge_estimate(end: str = "2024-12-31", block: int = 63, reps: int = 2000, seed: int = 2042, quirks: bool = True) -> pd.DataFrame:
    """A42 0-2: real hedged US-treasury ETFs against the hedge model with h = 0 (same proxy, fee and KRW cash = base).
    h_est = annual mean of (model − real); 95 % interval by circular block bootstrap of the daily differences."""
    ext, _, _ = usd_extended(START)
    files = aa_real.load_files()
    real = aa_real.real_series(*files, quirks=aa_real.QUIRKS if quirks else None)
    close = ext.close.ffill()
    usd_rets = close.pct_change(fill_method=None)
    idx = close.index
    from .aa_korea import krw_cash_returns
    fx_line = load_usdkrw().reindex(load_usdkrw().index.union(idx)).ffill().reindex(idx)
    fx_ret = fx_line.pct_change(fill_method=None).fillna(0.0)
    usd_cash = ext.cash.reindex(idx).fillna(0.0)
    rng = np.random.default_rng(seed)
    rows = []
    for cash_name, sched in (("base", load_bok_rate()), ("cd91", cd91_schedule())):
        krw_cash = krw_cash_returns(idx, sched)
        for code, p in HEDGE_PROBES.items():
            tr = real.get(code)
            if tr is None:
                rows.append({"상품": code, "원화 현금": cash_name, "비고": "실제 가격 없음"})
                continue
            m = product_returns(p, usd_rets, fx_ret, usd_cash, krw_cash, KrwModel(hedge_extra_bps=0.0))
            vals, used = aa_real.to_us_grid(tr, idx, same_day=False)
            t0 = aa_real.splice_date(tr, used)
            r = vals.pct_change(fill_method=None)
            sl = (idx > t0) & (idx <= pd.Timestamp(end))
            d = (m[sl] - r[sl]).dropna().to_numpy()
            est = d.mean() * 252
            n = len(d)
            boots = []
            for _ in range(reps):
                starts = rng.integers(0, n, int(np.ceil(n / block)))
                pick = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
                boots.append(d[pick].mean() * 252)
            rows.append({"상품": code, "이름": p.name, "원화 현금": cash_name, "겹침 시작": str(t0.date()), "끝": end,
                         "겹침 일": n, "h 추정 (bp)": est * 1e4, "95% 하한 (bp)": np.percentile(boots, 2.5) * 1e4,
                         "95% 상한 (bp)": np.percentile(boots, 97.5) * 1e4})
    return pd.DataFrame(rows)


def nearest_grid(h_bp: float, grid=HEDGE_BP) -> float:
    """The grid value nearest to h_bp (ties go to the larger one)."""
    g = np.array(grid, float)
    dist = np.abs(g - h_bp)
    return float(g[np.flatnonzero(dist == dist.min()).max()])


__all__ = ["PRICES", "HEDGE_BP", "CASH", "PriceSet", "all_sets", "build", "calibration", "adjusted_model",
           "hedge_estimate", "nearest_grid", "usd_extended", "model_panel", "NO_CALIBRATE"]
