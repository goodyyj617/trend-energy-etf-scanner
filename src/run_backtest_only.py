from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import run_backtests
from .prices import download_ohlcv
from .run_daily_scan import ROOT, load_config
from .universe import build_base_universe


def main() -> None:
    cfg = load_config()

    data_dir = ROOT / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    universe = build_base_universe(
        aum_csv=ROOT / "config" / "aum.csv",
        exclusions_yml=ROOT / "config" / "exclusions.yml",
        overrides_csv=ROOT / "config" / "manual_overrides.csv",
        universe_yml=ROOT / "config" / "universe.yml",
    )

    symbols = universe.loc[universe["base_universe_eligible"], "symbol"].tolist()
    prices = download_ohlcv(
        symbols,
        period=str(cfg["lookback_period"]),
        interval=str(cfg["price_interval"]),
    )

    if prices.empty:
        raise RuntimeError("No price data downloaded. Check symbols, internet access, or yfinance availability.")

    as_of_ts = pd.to_datetime(prices["date"], errors="coerce").dropna().max()
    as_of = str(as_of_ts.date()) if pd.notna(as_of_ts) else "unknown"

    backtest_payload = run_backtests(
        prices=prices,
        universe=universe,
        cfg=cfg,
        data_dir=data_dir,
        as_of=as_of,
    )

    print(f"Wrote {data_dir / 'backtest_summary.json'}")
    print(f"strategies={len(backtest_payload.get('summary', []))}")
    print(f"diagnostics={len(backtest_payload.get('diagnostic_summary', []))}")
    print(f"recent_trades={len(backtest_payload.get('recent_trades', []))}")
    print(f"as_of={as_of}")


if __name__ == "__main__":
    main()
