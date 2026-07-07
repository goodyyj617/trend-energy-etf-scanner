from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf


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
