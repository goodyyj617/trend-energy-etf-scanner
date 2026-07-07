from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _read_pipe_table(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text
    lines = [line for line in text.splitlines() if not line.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")


def fetch_nasdaq_trader_etf_list() -> pd.DataFrame:
    """Return US-listed ETF candidates from Nasdaq Trader symbol directory."""
    nasdaq = _read_pipe_table(NASDAQ_LISTED_URL)
    other = _read_pipe_table(OTHER_LISTED_URL)

    ndf = pd.DataFrame({
        "symbol": nasdaq["Symbol"].astype(str),
        "name": nasdaq["Security Name"].astype(str),
        "exchange": "Q",
        "is_etf": nasdaq["ETF"].astype(str).eq("Y"),
        "is_test_issue": nasdaq["Test Issue"].astype(str).eq("Y"),
    })

    odf = pd.DataFrame({
        "symbol": other["ACT Symbol"].astype(str),
        "name": other["Security Name"].astype(str),
        "exchange": other["Exchange"].astype(str),
        "is_etf": other["ETF"].astype(str).eq("Y"),
        "is_test_issue": other["Test Issue"].astype(str).eq("Y"),
    })

    out = pd.concat([ndf, odf], ignore_index=True)
    out = out[out["is_etf"] & ~out["is_test_issue"]].copy()
    out["symbol"] = out["symbol"].str.strip()
    out["name"] = out["name"].str.strip()
    out = out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    return out


def load_aum_csv(path: str | Path) -> pd.DataFrame:
    """Load manual or semi-manual AUM file.

    Required columns: symbol, aum
    Optional columns: asset_group, category
    aum should be numeric USD value. Symbols absent from this file are excluded
    if aum_top_n filtering is enabled.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns or "aum" not in df.columns:
        raise ValueError("AUM CSV must contain symbol and aum columns")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["aum"] = pd.to_numeric(df["aum"], errors="coerce")
    for col in ["asset_group", "category"]:
        if col not in df.columns:
            df[col] = "unknown"
    return df[["symbol", "aum", "asset_group", "category"]]


def load_manual_overrides(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.lower().str.strip()
    return dict(zip(df["symbol"], df["action"]))
