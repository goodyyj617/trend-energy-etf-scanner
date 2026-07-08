from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .features import compute_latest_features
from .prices import download_ohlcv
from .universe import build_base_universe
from .update_aum import update_aum_csv

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config" / "universe.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["universe"]


def to_jsonable(df: pd.DataFrame) -> list[dict]:
    clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def main() -> None:
    try:
        update_aum_csv()
    except Exception as e:
        print(f"[AUM] update failed, continuing scan with existing config/aum.csv: {e}")
        
    cfg = load_config()
    data_dir = ROOT / "docs" / "data"
    history_dir = data_dir / "history"
    data_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    auto_aum_cfg = cfg.get("auto_aum", {})
    if auto_aum_cfg.get("enabled", False):
        update_aum_csv(
            aum_csv=ROOT / "config" / "aum.csv",
            exclusions_yml=ROOT / "config" / "exclusions.yml",
            max_new_per_run=int(auto_aum_cfg.get("max_new_per_run", 200)),
            refresh_existing=bool(auto_aum_cfg.get("refresh_existing", False)),
        )

    universe = build_base_universe(
        aum_csv=ROOT / "config" / "aum.csv",
        exclusions_yml=ROOT / "config" / "exclusions.yml",
        overrides_csv=ROOT / "config" / "manual_overrides.csv",
        universe_yml=ROOT / "config" / "universe.yml",
    )
    universe.to_csv(data_dir / "universe_current.csv", index=False)

    symbols = universe.loc[universe["base_universe_eligible"], "symbol"].tolist()
    prices = download_ohlcv(symbols, period=str(cfg["lookback_period"]), interval=str(cfg["price_interval"]))
    if prices.empty:
        raise RuntimeError("No price data downloaded. Check symbols, internet access, or yfinance availability.")

    latest = compute_latest_features(prices, universe, cfg)
    latest.to_csv(data_dir / "latest.csv", index=False)

    as_of = str(latest["date"].dropna().max())
    latest.to_csv(history_dir / f"{as_of}.csv", index=False)

    payload = {
        "as_of": as_of,
        "row_count": int(len(latest)),
        "eligible_count": int(latest["eligible_universe"].sum()),
        "signal_count": int(latest["signal_surge_v0"].sum()),
        "rows": to_jsonable(latest),
    }
    with open(data_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {data_dir / 'latest.json'}")
    print(f"as_of={as_of} rows={payload['row_count']} eligible={payload['eligible_count']} signals={payload['signal_count']}")


if __name__ == "__main__":
    main()
