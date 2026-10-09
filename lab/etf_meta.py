"""ETF metadata (expense ratio, AUM, Morningstar category, inception) used by the universe filter.

The metadata is fetched once from yfinance and stored in lab/etf_meta.csv so that the
app runs offline and every research run uses the same, dated metadata.

Refresh:  .venv\\Scripts\\python.exe -m lab.etf_meta            (all snapshot ETFs)
          .venv\\Scripts\\python.exe -m lab.etf_meta SGOV BIL   (append / update some)
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

META_PATH = Path(__file__).with_name("etf_meta.csv")
COLUMNS = ["symbol", "name", "category", "expense_ratio", "aum", "inception", "fetched"]


def fetch(symbols: list[str], pause: float = 0.2) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for i, sym in enumerate(symbols, 1):
        try:
            info = yf.Ticker(sym.replace(".", "-")).info or {}
        except Exception as exc:  # network / unknown ticker
            print(f"[{i}/{len(symbols)}] {sym}: {exc}", file=sys.stderr)
            continue
        er = info.get("netExpenseRatio")
        if er is None and info.get("annualReportExpenseRatio") is not None:
            er = info["annualReportExpenseRatio"] * 100  # yfinance reports this one as a fraction
        inception = info.get("fundInceptionDate")
        rows.append({
            "symbol": sym,
            "name": info.get("longName") or info.get("shortName") or "",
            "category": info.get("category") or "",
            "expense_ratio": er,  # percent per year, e.g. 0.09 = 0.09%
            "aum": info.get("totalAssets"),
            "inception": pd.to_datetime(inception, unit="s").date().isoformat() if inception else "",
            "fetched": date.today().isoformat(),
        })
        if i % 25 == 0:
            print(f"[{i}/{len(symbols)}]", file=sys.stderr)
        time.sleep(pause)
    return pd.DataFrame(rows, columns=COLUMNS)


def load() -> pd.DataFrame:
    if not META_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(META_PATH, keep_default_na=False, na_values=[""])


def update(symbols: list[str]) -> pd.DataFrame:
    new = fetch(symbols)
    old = load()
    merged = pd.concat([old[~old["symbol"].isin(new["symbol"])], new], ignore_index=True)
    merged = merged.sort_values("symbol").reset_index(drop=True)
    merged.to_csv(META_PATH, index=False)
    return merged


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        from lab.data import load_snapshot

        args = load_snapshot().symbols
    out = update(args)
    print(f"saved {len(out)} rows to {META_PATH}")
