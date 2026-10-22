"""Macro / market-regime data for regime-filter blocks (FRED, free, no API key).

Series (all market prices or spreads observed daily and not revised afterwards, so a
past value is what was known on that day):
  T10Y3M  10-year minus 3-month Treasury yield, percentage points (yield curve)
  BAA10Y  Moody's Baa corporate yield minus 10-year Treasury, percentage points (credit spread)
  VIXCLS  CBOE VIX close (expected 30-day S&P 500 volatility)

Point-in-time rule: a decision on day t uses only values dated on or before the previous
trading day (t-1). FRED publishes most daily series the next business day, so this lag
covers publication timing; the order is still executed at the t+1 open.

Research runs read the frozen snapshot committed in lab/macro_snapshot.csv.gz (same
numbers every run). The daily alert fetches fresh values with `fetch_fred` and passes
them to `use`.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "lab" / "macro_snapshot.csv.gz"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SERIES: dict[str, str] = {
    "T10Y3M": "장단기 금리차 (미국 10년 − 3개월 국채, %p)",
    "BAA10Y": "신용 스프레드 (Moody's Baa 회사채 − 10년 국채, %p)",
    "VIXCLS": "VIX (S&P 500 30일 기대 변동성)",
}
MAX_STALE_DAYS = 10  # a value older than this (calendar days) is treated as missing

_data: pd.DataFrame | None = None


def fetch_fred(series: list[str] | None = None, timeout: float = 30.0) -> pd.DataFrame:
    """Download full histories from FRED. Returns a date-indexed frame, one column per series."""
    import requests

    cols = {}
    for sid in series or list(SERIES):
        r = requests.get(FRED_URL.format(sid=sid), timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), na_values=["."])
        df.columns = ["date", sid]
        cols[sid] = df.assign(date=pd.to_datetime(df["date"])).set_index("date")[sid].astype(float)
    return pd.DataFrame(cols).sort_index()


def save_snapshot(df: pd.DataFrame, path: Path = SNAPSHOT) -> None:
    df.rename_axis("date").reset_index().to_csv(path, index=False, float_format="%.4f")


def load_snapshot(path: Path = SNAPSHOT) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def use(df: pd.DataFrame | None) -> None:
    """Replace the data the blocks read (None = back to the committed snapshot)."""
    global _data
    _data = None if df is None else df.sort_index()


def data() -> pd.DataFrame:
    global _data
    if _data is None:
        if not SNAPSHOT.exists():
            raise ValueError("매크로 데이터 파일(lab/macro_snapshot.csv.gz)이 없습니다.")
        _data = load_snapshot()
    return _data


def on_dates(values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Value known for a decision on each date of index: last observation dated before that
    date's previous trading day or on it (one-day publication lag). Stale values become NaN."""
    s = values.dropna()
    if s.empty:
        return pd.Series(float("nan"), index=index)
    obs_date = pd.Series(s.index, index=s.index)
    union = index.union(s.index)
    val = s.reindex(union).ffill().reindex(index).shift(1)
    when = obs_date.reindex(union).ffill().reindex(index).shift(1)
    prev_day = pd.Series(index, index=index).shift(1)
    fresh = (prev_day - when) <= pd.Timedelta(days=MAX_STALE_DAYS)
    return val.where(fresh)


def series(sid: str, index: pd.DatetimeIndex, sma: int | None = None) -> tuple[pd.Series, pd.Series | None]:
    """(value, its N-observation SMA or None) aligned to index with the publication lag.
    The SMA is computed on the series' own observations, then aligned the same way."""
    raw = data()[sid].dropna()
    avg = raw.rolling(sma, min_periods=sma).mean() if sma else None
    return on_dates(raw, index), (on_dates(avg, index) if avg is not None else None)


if __name__ == "__main__":  # refresh the committed snapshot: python -m lab.macro
    fresh = fetch_fred()
    save_snapshot(fresh)
    print(f"saved {SNAPSHOT.name}: {len(fresh)} rows, {fresh.index[0].date()} - {fresh.index[-1].date()}")
    print(fresh.tail(3).to_string())
