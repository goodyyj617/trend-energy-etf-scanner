"""KRW returns of Korean-listed ETFs, built from US proxies, USD/KRW and the two cash rates.

Daily return of product p on day t (r = proxy USD total return, c$ = USD cash return of the US
panel (BIL / T-bill), c₩ = KRW cash return from the Bank of Korea base rate, fx = change in
KRW per USD, L = leverage, f = fee + known drag per day):

    x            = L * (r - c$)                               excess return, daily reset
    환헤지 (H)    = x + c₩ - f - hedge_extra                    forward hedge earns the rate gap
    환노출        = (1 + x + c$) * (1 + fx) - 1 - f              a USD asset held in KRW
    원화(국내 주식) = L * (r₩ - c₩) + c₩ - f,  r₩ = (1 + r) * (1 + fx) - 1
    원화 현금      = c₩ - f

Data:
- USD/KRW: lab/aa_usdkrw.csv.gz, FRED DEXKOUS (Fed H.10 noon buying rate in New York, the same
  clock as US closes) via github.com/datasets/exchange-rates. `python -m lab.aa_korea refresh`
  rewrites it. The app also splices the newer days from Yahoo (KRW=X) when the network allows.
- Bank of Korea base rate: lab/aa_bok_rate.csv, a step schedule (column `source`: "checked"
  rows were compared with global-rates.com / bluesharehub.com on 2026-10-24; "memory" rows,
  before 2018-11, still need checking against the Bank of Korea's ECOS).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .aa_assets import BENCHMARK_CODE, KRW_CASH, PRODUCTS, USD_CASH, Product
from .data import CACHE_DIR, FIELDS, Panel

TRADING_DAYS = 252
USDKRW_PATH = Path(__file__).with_name("aa_usdkrw.csv.gz")
BOK_PATH = Path(__file__).with_name("aa_bok_rate.csv")
FX_SOURCE_URL = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"
FX_TAIL_CACHE = CACHE_DIR / "yf" / "KRW_X.csv.gz"


@dataclass(frozen=True)
class KrwModel:
    hedge_extra_bps: float = 20.0  # hedging cost beyond the rate gap (basis, roll), per year
    fees: bool = True  # subtract each product's total fee and known drag


# --------------------------------------------------------------------------- inputs

def load_usdkrw(path: Path = USDKRW_PATH) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["krw_per_usd"].astype(float).sort_index()


def load_bok_rate(path: Path = BOK_PATH) -> pd.Series:
    """Base rate in percent, indexed by the date each level took effect."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["rate_pct"].astype(float).sort_index()


def splice(base: pd.Series, tail: pd.Series | None) -> pd.Series:
    """Append the days of `tail` after the end of `base`, scaled to meet it on their last common day."""
    if tail is None or tail.empty:
        return base
    tail = tail.sort_index().dropna()
    common = base.index.intersection(tail.index)
    newer = tail[tail.index > base.index[-1]]
    if newer.empty:
        return base
    scale = float(base.loc[common[-1]] / tail.loc[common[-1]]) if len(common) else 1.0
    return pd.concat([base, newer * scale])


def fetch_usdkrw_tail() -> pd.Series | None:
    """Recent USD/KRW from Yahoo (KRW=X), cached for a day. None when offline."""
    try:
        if FX_TAIL_CACHE.exists() and (pd.Timestamp.now() - pd.Timestamp(FX_TAIL_CACHE.stat().st_mtime, unit="s")) < pd.Timedelta(hours=20):
            df = pd.read_csv(FX_TAIL_CACHE, parse_dates=["date"])
        else:
            import yfinance as yf
            raw = yf.download("KRW=X", period="2y", auto_adjust=True, actions=False, progress=False, threads=False,
                              multi_level_index=False)
            if raw is None or raw.empty:
                return None
            df = raw.reset_index().rename(columns=str.lower)[["date", "close"]].dropna()
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            FX_TAIL_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(FX_TAIL_CACHE, index=False)
        return df.set_index("date")["close"].astype(float)
    except Exception:  # offline or Yahoo changed: the snapshot alone is fine
        return None


def refresh_snapshot(url: str = FX_SOURCE_URL, path: Path = USDKRW_PATH) -> int:
    """Rewrite the USD/KRW snapshot from the public FRED mirror. Returns the row count."""
    d = pd.read_csv(url)
    k = d[d["Country"] == "South Korea"][["Date", "Exchange rate"]]
    k.columns = ["date", "krw_per_usd"]
    k = k[(k["date"] >= "1993-01-01") & (k["krw_per_usd"] > 0)].dropna().sort_values("date")
    k.to_csv(path, index=False, float_format="%.2f")
    return len(k)


def krw_cash_returns(index: pd.DatetimeIndex, bok: pd.Series) -> pd.Series:
    rate = bok.reindex(bok.index.union(index)).ffill().reindex(index)
    rate = rate.fillna(bok.iloc[0])  # before the first row: its level
    return rate / 100.0 / TRADING_DAYS


# --------------------------------------------------------------------------- product returns

def product_returns(p: Product, usd_rets: pd.DataFrame, fx_ret: pd.Series, usd_cash: pd.Series, krw_cash: pd.Series,
                    model: KrwModel = KrwModel()) -> pd.Series:
    """Daily KRW return of one product on usd_rets.index (NaN where a proxy has no data)."""
    idx = usd_rets.index
    base = pd.Series(0.0, index=idx)
    for sym, w in p.proxy:
        if sym == KRW_CASH:
            base = base + w * krw_cash
        elif sym == USD_CASH:
            base = base + w * usd_cash
        else:
            base = base + w * usd_rets[sym]
    L = p.leverage
    fee = (p.ter + p.extra_drag) / 100.0 / TRADING_DAYS if model.fees else 0.0
    if p.fx == "krw":
        out = base
    elif p.fx == "hedged":
        out = L * (base - usd_cash) + krw_cash - model.hedge_extra_bps / 1e4 / TRADING_DAYS
    elif p.fx == "unhedged":
        out = (1.0 + L * (base - usd_cash) + usd_cash) * (1.0 + fx_ret) - 1.0
    elif p.fx == "krw_native":
        r_krw = (1.0 + base) * (1.0 + fx_ret) - 1.0
        out = L * (r_krw - krw_cash) + krw_cash
    else:
        raise ValueError(f"unknown fx kind {p.fx}")
    return out - fee


def _index_from_returns(r: pd.Series) -> pd.Series:
    """Price index 100 on the day before the first valid return (NaN before), gaps filled flat."""
    first = r.first_valid_index()
    out = pd.Series(np.nan, index=r.index)
    if first is None:
        return out
    pos = r.index.get_loc(first)
    start = pos - 1 if pos > 0 else pos
    rr = r.iloc[start:].copy()
    rr.iloc[0] = 0.0
    out.iloc[start:] = 100.0 * np.cumprod(1.0 + rr.fillna(0.0).to_numpy())
    return out


def krw_panel(usd_panel: Panel, fx: pd.Series, bok: pd.Series, model: KrwModel = KrwModel(),
              products: dict[str, Product] | None = None) -> tuple[Panel, dict[str, str]]:
    """Panel whose columns are product codes (KRW price indices) and whose cash is KRW cash.

    Returns (panel, unavailable) where unavailable maps a product code to the reason it is missing.
    The panel carries `benchmark` = KODEX 미국S&P500 for the engine and the scorecard."""
    products = products or PRODUCTS
    idx = usd_panel.close.index
    close = usd_panel.close.ffill()
    usd_rets = close.pct_change(fill_method=None)  # NaN on a proxy's first row: the product starts the next day
    fx_line = fx.reindex(fx.index.union(idx)).ffill().reindex(idx)
    fx_ret = fx_line.pct_change(fill_method=None).fillna(0.0)
    usd_cash = (usd_panel.cash if usd_panel.cash is not None else pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    krw_cash = krw_cash_returns(idx, bok)
    cols, unavailable = {}, {}
    for code, p in products.items():
        missing = [s for s in p.proxy_symbols if s not in usd_rets.columns or usd_panel.close[s].notna().sum() < 2]
        if missing:
            unavailable[code] = f"대용 데이터 없음 ({', '.join(missing)})"
            continue
        r = product_returns(p, usd_rets, fx_ret, usd_cash, krw_cash, model)
        cols[code] = _index_from_returns(r)
    if not cols:
        raise ValueError("국내 ETF 수익률을 하나도 만들지 못했습니다 (미국 대용 데이터 없음).")
    prices = pd.DataFrame(cols, index=idx)
    vol = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    tables = {f: (vol if f == "volume" else prices) for f in FIELDS}
    info = pd.DataFrame({"symbol": prices.columns, "name": [products[c].name for c in prices.columns],
                         "asset_group": [products[c].asset_class for c in prices.columns]})
    panel = Panel(**tables, info=info, source=f"국내 ETF 원화 합성 ({usd_panel.source} + USD/KRW + 한국은행 기준금리)",
                  cash=krw_cash, kind="krw")
    panel.benchmark = BENCHMARK_CODE if BENCHMARK_CODE in prices.columns else prices.columns[0]
    panel.fx = fx_line
    return panel, unavailable


if __name__ == "__main__":
    if sys.argv[1:2] == ["refresh"]:
        print(f"USD/KRW rows written: {refresh_snapshot()}")
    else:
        raise SystemExit("usage: python -m lab.aa_korea refresh")
