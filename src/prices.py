from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
DAILY_BAR_COMPLETION_HOUR = 20
DAILY_BAR_COMPLETION_MINUTE = 15


def yahoo_symbol(symbol: str) -> str:
    # Yahoo Finance usually uses '-' for class suffixes. ETF symbols rarely need this,
    # but this normalization avoids common failures.
    return symbol.replace(".", "-").replace("/", "-")


def download_ohlcv(symbols: Iterable[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    symbols = list(dict.fromkeys([str(s).upper().strip() for s in symbols]))
    if not symbols:
        return pd.DataFrame()
    yahoo_symbols = [yahoo_symbol(s) for s in symbols]
    symbol_map = dict(zip(yahoo_symbols, symbols))

    raw = yf.download(
        yahoo_symbols,
        period=period,
        interval=interval,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ysym in yahoo_symbols:
            if ysym not in raw.columns.get_level_values(0):
                continue
            sub = raw[ysym].copy()
            sub["symbol"] = symbol_map[ysym]
            frames.append(sub)
    else:
        # Single ticker case.
        sub = raw.copy()
        sub["symbol"] = symbols[0]
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    df = df.reset_index().rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])
    return df


def filter_completed_daily_bars(
    prices: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, str | int]]:
    evaluation_time = now if now is not None else datetime.now(timezone.utc)
    if evaluation_time.tzinfo is None or evaluation_time.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")

    new_york_time = evaluation_time.astimezone(NEW_YORK_TIMEZONE)
    cutoff_reached = (
        new_york_time.hour,
        new_york_time.minute,
        new_york_time.second,
        new_york_time.microsecond,
    ) >= (
        DAILY_BAR_COMPLETION_HOUR,
        DAILY_BAR_COMPLETION_MINUTE,
        0,
        0,
    )
    safe_date = new_york_time.date()
    if not cutoff_reached:
        safe_date -= timedelta(days=1)

    if "date" not in prices.columns:
        if not prices.empty:
            raise ValueError("daily OHLCV data is missing the date column")
        filtered = prices.copy().reset_index(drop=True)
        downloaded_max_date = None
        retained_max_date = None
    else:
        parsed_dates = pd.to_datetime(prices["date"], errors="coerce").dt.date
        downloaded_max_date = parsed_dates.dropna().max() if parsed_dates.notna().any() else None
        keep = parsed_dates.notna() & parsed_dates.le(safe_date)

        filtered = prices.loc[keep].copy()
        filtered["_completed_bar_date"] = parsed_dates.loc[keep]
        retained_max_date = (
            filtered["_completed_bar_date"].max() if not filtered.empty else None
        )
        sort_columns = [
            column
            for column in ["symbol", "_completed_bar_date"]
            if column in filtered.columns
        ]
        filtered = (
            filtered.sort_values(sort_columns, kind="mergesort")
            .drop(columns="_completed_bar_date")
            .reset_index(drop=True)
        )

    provenance: dict[str, str | int] = {
        "new_york_evaluation_time": new_york_time.isoformat(),
        "safe_latest_daily_bar_date": safe_date.isoformat(),
        "downloaded_max_date": (
            downloaded_max_date.isoformat() if downloaded_max_date is not None else "unknown"
        ),
        "retained_max_date": (
            retained_max_date.isoformat() if retained_max_date is not None else "unknown"
        ),
        "rows_removed": int(len(prices) - len(filtered)),
    }
    return filtered, provenance
