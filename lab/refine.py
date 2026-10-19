"""Confirmation-condition test: does adding ONE extra entry condition make a family better?

For a chosen family (one entry block + one exit block from the fixed menu), the whole 6x6
family grid is rerun once per candidate add-on, with the add-on fixed at its block's
default value (no tuning of the new parameter). Each variant is compared with the base
family on the four ordering metrics of the family comparison:

  improvement : at least as good on all four and strictly better on at least one (Pareto)
  worse       : at most as good on all four and strictly worse on at least one
  mixed       : better on some, worse on others

Only Pareto improvements are candidates to keep, and a condition should improve the family
on every universe tested before it is adopted: added complexity has to earn its place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import pandas as pd

from .blocks import ENTRY_BLOCKS, EXIT_BLOCKS
from .data import Panel
from .engine import SignalCache, StrategyConfig
from .families import ENTRY_MENU, EXIT_MENU, ORDER, VERDICT_TEXT, grid_summary, ma200_calmar, verdict  # noqa: F401
from .grid import Gates, GridResult, run_grid


def addon_candidates(entry_key: str) -> list[str]:
    """Every entry block except the family's own entry, each at its default value."""
    return [k for k in ENTRY_BLOCKS if k != entry_key]


def addon_value(key: str) -> float | None:
    p = ENTRY_BLOCKS[key].param
    return None if p is None else p.default


@dataclass
class Refinement:
    base: StrategyConfig
    entry_key: str
    exit_key: str
    gates: Gates
    grids: dict[str, GridResult]  # "" = base family, otherwise the add-on block key
    ma200_calmar: float
    seconds: float

    def table(self) -> pd.DataFrame:
        base_sum = grid_summary(self.grids[""].cells, self.ma200_calmar)
        rows = []
        for key, g in self.grids.items():
            s = grid_summary(g.cells, self.ma200_calmar)
            label = "기준 (추가 없음)" if key == "" else ENTRY_BLOCKS[key].label + (
                "" if addon_value(key) is None else f" ({addon_value(key):g})")
            rows.append({"addon": key, "condition": label, **s,
                         "verdict": "base" if key == "" else verdict(base_sum, s)})
        out = pd.DataFrame(rows)
        rank = {"base": 0, "improve": 1, "mixed": 2, "same": 3, "worse": 4}
        out["_r"] = out["verdict"].map(rank)
        out = out.sort_values(["_r", *ORDER], ascending=[True, False, False, False, False]).drop(columns="_r")
        return out.reset_index(drop=True)

    def all_cells(self) -> pd.DataFrame:
        return pd.concat([g.cells.assign(addon=k) for k, g in self.grids.items()], ignore_index=True)


def run_refinement(panel: Panel, base: StrategyConfig, entry_key: str, exit_key: str, gates: Gates,
                   progress=None, on_result=None) -> Refinement:
    if entry_key not in ENTRY_MENU or exit_key not in EXIT_MENU:
        raise ValueError("고정 메뉴에 있는 진입·청산 계열을 고르세요.")
    t0 = time.time()
    e_vals, x_vals = ENTRY_MENU[entry_key], EXIT_MENU[exit_key]
    keys = [""] + addon_candidates(entry_key)
    cache = SignalCache(panel)
    grids: dict[str, GridResult] = {}
    for n, key in enumerate(keys):
        entries = {entry_key: e_vals[0]} if key == "" else {entry_key: e_vals[0], key: addon_value(key)}
        cfg = replace(base, entries=entries, exits={exit_key: x_vals[0]})

        def step(f, n=n):
            if progress:
                progress((n + f) / len(keys))

        grids[key] = run_grid(panel, cfg, entry_key, e_vals, exit_key, x_vals, gates, progress=step,
                              on_result=on_result, cache=cache)
    return Refinement(base, entry_key, exit_key, gates, grids, ma200_calmar(panel, base), time.time() - t0)


def family_label(entry_key: str, exit_key: str) -> str:
    return f"{ENTRY_BLOCKS[entry_key].label} × {EXIT_BLOCKS[exit_key].label}"
