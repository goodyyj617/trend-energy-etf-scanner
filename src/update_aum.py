from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .data_sources import fetch_nasdaq_trader_etf_list
from .universe import apply_exclusion_flags


def _load_existing_aum(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "aum", "asset_group", "category"])

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "symbol" not in df.columns:
        return pd.DataFrame(columns=["symbol", "aum", "asset_group", "category"])

    if "aum" not in df.columns:
        df["aum"] = None

    for col in ["asset_group", "category"]:
        if col not in df.columns:
            df[col] = "unknown"

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["aum"] = pd.to_numeric(df["aum"], errors="coerce")

    return df[["symbol", "aum", "asset_group", "category"]]


def _get_yfinance_aum(symbol: str) -> float | None:
    try:
        ticker = yf.Ticker(symbol)

        # For many ETFs, Yahoo exposes totalAssets in info.
        info = ticker.info or {}
        value = info.get("totalAssets")

        if value is None:
            return None

        value = float(value)
        if value <= 0:
            return None

        return value

    except Exception:
        return None


def update_aum_csv(
    aum_csv: str | Path,
    exclusions_yml: str | Path,
    max_new_per_run: int = 200,
    refresh_existing: bool = False,
) -> pd.DataFrame:
    aum_csv = Path(aum_csv)

    etfs = fetch_nasdaq_trader_etf_list()
    etfs = apply_exclusion_flags(etfs, exclusions_yml)

    candidates = etfs.loc[~etfs["exclude_by_name"], ["symbol", "name"]].copy()
    candidates["symbol"] = candidates["symbol"].astype(str).str.upper().str.strip()

    existing = _load_existing_aum(aum_csv)

    merged = candidates.merge(existing, on="symbol", how="left")

    if refresh_existing:
        missing_mask = pd.Series(True, index=merged.index)
    else:
        missing_mask = merged["aum"].isna() | (merged["aum"] <= 0)

    missing_symbols = merged.loc[missing_mask, "symbol"].drop_duplicates().tolist()
    symbols_to_fetch = missing_symbols[: int(max_new_per_run)]

    print(f"AUM cache rows before={len(existing)}")
    print(f"ETF candidates={len(candidates)}")
    print(f"AUM missing={len(missing_symbols)}")
    print(f"Fetching AUM for up to {len(symbols_to_fetch)} symbols")

    updates = {}
    for i, symbol in enumerate(symbols_to_fetch, start=1):
        value = _get_yfinance_aum(symbol)
        if value is not None:
            updates[symbol] = value
            print(f"[{i}/{len(symbols_to_fetch)}] {symbol}: {value:.0f}")
        else:
            print(f"[{i}/{len(symbols_to_fetch)}] {symbol}: AUM not found")

        # Small delay to reduce request pressure.
        time.sleep(0.05)

    if updates:
        for symbol, value in updates.items():
            merged.loc[merged["symbol"].eq(symbol), "aum"] = value

    merged["asset_group"] = merged["asset_group"].fillna("unknown")
    merged["category"] = merged["category"].fillna("unknown")

    out = merged[["symbol", "aum", "asset_group", "category"]].drop_duplicates("symbol")
    out["aum"] = pd.to_numeric(out["aum"], errors="coerce")
    out = out.sort_values("aum", ascending=False, na_position="last")

    aum_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(aum_csv, index=False)

    print(f"Wrote updated AUM cache to {aum_csv}")
    print(f"AUM available={out['aum'].notna().sum()} / {len(out)}")

    return out
