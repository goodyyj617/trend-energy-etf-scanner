"""Save and reload results as plain files under lab_results/ (one folder per result).

Single backtest: meta.json, equity.csv (strategy, SPY, exposure), trades.csv
Grid          : meta.json, cells.csv
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import ROOT

RESULTS_DIR = ROOT / "lab_results"


def _folder(kind: str, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:40] or kind
    path = RESULTS_DIR / f"{stamp}_{kind}_{safe}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_meta(path: Path, meta: dict) -> None:
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_backtest(result, metrics: dict, name: str, data_source: str) -> Path:
    path = _folder("backtest", name)
    start, end = result.period
    _write_meta(path, {
        "kind": "backtest", "name": name, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": data_source, "universe_size": result.universe_size,
        "period": [str(start.date()), str(end.date())],
        "config": result.config.to_dict(), "description": result.config.describe(), "metrics": metrics,
    })
    pd.DataFrame({"strategy": result.equity, "SPY": result.benchmark, "exposure": result.exposure}).to_csv(path / "equity.csv")
    result.trades.to_csv(path / "trades.csv", index=False)
    return path


def save_grid(grid, name: str, data_source: str, period: tuple) -> Path:
    path = _folder("grid", name)
    _write_meta(path, {
        "kind": "grid", "name": name, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": data_source, "period": [str(period[0].date()), str(period[1].date())],
        "base_config": grid.base.to_dict(), "description": grid.base.describe(),
        "x_key": grid.x_key, "x_values": grid.x_values, "y_key": grid.y_key, "y_values": grid.y_values,
        "gates": grid.gates.to_dict(), "summary": grid.summary(), "seconds": grid.seconds,
    })
    grid.cells.to_csv(path / "cells.csv", index=False)
    return path


def list_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    out = []
    for meta_path in sorted(RESULTS_DIR.glob("*/meta.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta["path"] = meta_path.parent
        out.append(meta)
    return out


def load_equity(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "equity.csv", index_col=0, parse_dates=True)


def load_trades(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "trades.csv")


def load_cells(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "cells.csv")
