"""Save and reload results as plain files under lab_results/ (one folder per result).

Single backtest: meta.json, equity.csv (strategy, SPY, exposure), trades.csv
Grid          : meta.json, cells.csv
Family / refine / common add-on tests: meta.json, a summary table, cells.csv
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


def save_backtest(result, metrics: dict, name: str, panel) -> Path:
    """`panel` supplies what is needed to rebuild the run later (holdout evaluation)."""
    path = _folder("backtest", name)
    start, end = result.period
    _write_meta(path, {
        "kind": "backtest", "name": name, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": panel.source, "data_kind": panel.kind, "universe": panel.tradable_symbols,
        "universe_size": result.universe_size,
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


def save_family(fc, name: str, data_source: str, universe: list[str]) -> Path:
    """Strategy-family comparison: meta.json, summary.csv, cells.csv (every cell, with entry/exit columns)."""
    from .families import ENTRY_MENU, EXIT_MENU, MENU_VERSION

    path = _folder("family", name)
    _write_meta(path, {
        "kind": "family", "name": name, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": data_source, "universe": universe, "period": [fc.base.start, fc.base.end],
        "base_config": fc.base.to_dict(), "description": f"계열 비교 ({MENU_VERSION})",
        "menu_version": MENU_VERSION, "entry_menu": ENTRY_MENU, "exit_menu": EXIT_MENU,
        "gates": fc.gates.to_dict(), "ma200_calmar": fc.ma200_calmar, "seconds": fc.seconds,
        "extra_entries": fc.extra_entries, "extra_exits": fc.extra_exits,
    })
    fc.summary().to_csv(path / "summary.csv", index=False)
    fc.all_cells().to_csv(path / "cells.csv", index=False)
    return path


def save_addon(at, name: str, data_source: str, universe: list[str]) -> Path:
    """Common add-on test: meta.json, table.csv (one row per family), cells.csv (plain and added grids)."""
    from .families import MENU_VERSION

    base = at.plain.base
    path = _folder("addon", name)
    _write_meta(path, {
        "kind": "addon", "name": name, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": data_source, "universe": universe, "period": [base.start, base.end],
        "base_config": base.to_dict(), "description": f"공통 조건 시험 · {at.label} ({MENU_VERSION})",
        "menu_version": MENU_VERSION, "addon_kind": at.kind, "addon_key": at.key, "addon_value": at.value,
        "label": at.label, "counts": at.counts(), "gates": at.plain.gates.to_dict(),
        "ma200_calmar": at.plain.ma200_calmar, "seconds": at.plain.seconds + at.added.seconds,
    })
    at.table().to_csv(path / "table.csv", index=False)
    at.all_cells().to_csv(path / "cells.csv", index=False)
    return path


def load_addon_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "table.csv")


def load_family_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "summary.csv")


def save_refinement(ref, name: str, data_source: str, universe: list[str]) -> Path:
    """Confirmation-condition test: meta.json, table.csv, cells.csv (every cell, with an addon column)."""
    path = _folder("refine", name)
    _write_meta(path, {
        "kind": "refine", "name": name, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": data_source, "universe": universe, "period": [ref.base.start, ref.base.end],
        "base_config": ref.base.to_dict(), "entry_key": ref.entry_key, "exit_key": ref.exit_key,
        "description": f"확인 조건 시험 ({ref.entry_key} × {ref.exit_key})",
        "gates": ref.gates.to_dict(), "ma200_calmar": ref.ma200_calmar, "seconds": ref.seconds,
    })
    ref.table().to_csv(path / "table.csv", index=False)
    ref.all_cells().to_csv(path / "cells.csv", index=False)
    return path


def load_refine_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path / "table.csv")
    table["addon"] = table["addon"].fillna("")  # the base family is stored as an empty add-on name
    return table
