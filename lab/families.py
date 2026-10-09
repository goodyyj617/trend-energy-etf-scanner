"""Strategy-family comparison over a fixed, pre-registered menu.

Instead of searching every combination of blocks and numbers, a small menu of entry
families x exit families is fixed in advance. Each family (one entry block + one exit
block) is run over its own 6 x 6 parameter grid, and families are compared by how the
WHOLE grid behaves, not by its best cell:

  pass share        share of cells that pass the gates
  largest region    size of the biggest connected block of passing cells
  beats SPY 200MA   share of cells whose Calmar exceeds SPY 200-day timing (same dates/costs/cash)
  median Calmar     typical outcome across the grid
  Calmar P25        lower-quartile outcome (how bad it gets with an unlucky parameter choice)
  median trades     typical sample size

Families are ordered lexicographically by the first four columns, no weights.

The axis values were chosen before looking at any results: each covers short, medium and
long horizons with roughly geometric spacing (so neighbouring cells are comparable steps).
They are not to be tuned after seeing results; a changed menu is a new pre-registration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .blocks import ENTRY_BLOCKS, EXIT_BLOCKS
from .data import BENCHMARK, Panel
from .engine import SignalCache, StrategyConfig, run_backtest
from .grid import Gates, GridResult, run_grid
from .metrics import curve_metrics

MENU_VERSION = "family-menu-v1 (2026-10-09)"
ENTRY_MENU: dict[str, list[float]] = {
    "breakout": [20, 30, 50, 80, 120, 200],
    "above_ma": [50, 80, 100, 150, 200, 250],
    "momentum": [21, 42, 63, 126, 189, 252],
}
EXIT_MENU: dict[str, list[float]] = {
    "low_break": [10, 15, 20, 30, 50, 80],
    "below_ma": [20, 30, 50, 100, 150, 200],
    "atr_trail": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
}

SUMMARY_TEXT = {
    "pass_share": ("Gate 통과 비율", "격자 36칸 중 최소 통과 조건(Gate)을 모두 만족한 칸의 비율."),
    "largest_region": ("가장 큰 연결 영역", "Gate 통과 칸이 상하좌우로 이어진 가장 큰 덩어리의 칸 수. 클수록 파라미터를 대충 골라도 통한다는 뜻."),
    "beat_ma200": ("SPY 200일선 이긴 비율", "칼마 비율이 같은 조건의 'SPY 200일선' 전략보다 높은 칸의 비율. 규칙 1개짜리보다 나은 칸이 얼마나 되는지."),
    "median_calmar": ("중간 칼마", "격자 36칸 칼마 비율의 중앙값. 파라미터를 무작위로 골랐을 때의 전형적인 결과."),
    "p25_calmar": ("칼마 하위 25%", "격자 칸 칼마 비율의 하위 25% 값. 운 나쁜 파라미터를 골랐을 때 어느 정도까지 나빠지는지."),
    "median_cagr": ("중간 CAGR", "격자 칸 CAGR의 중앙값."),
    "median_mdd": ("중간 MDD", "격자 칸 최대 낙폭의 중앙값."),
    "median_trades": ("중간 거래 수", "격자 칸 완료 거래 수의 중앙값. 너무 적으면 결과를 믿기 어렵습니다."),
}
ORDER = ["pass_share", "largest_region", "beat_ma200", "median_calmar"]


@dataclass
class FamilyComparison:
    base: StrategyConfig
    gates: Gates
    grids: dict[tuple[str, str], GridResult]
    ma200_calmar: float
    seconds: float

    def summary(self) -> pd.DataFrame:
        rows = []
        for (e, x), g in self.grids.items():
            c = g.cells
            calmar = c["calmar"].astype(float)
            rows.append({
                "entry": e, "exit": x,
                "family": f"{ENTRY_BLOCKS[e].label} × {EXIT_BLOCKS[x].label}",
                "pass_share": float(c["pass"].mean()),
                "largest_region": int(c["region_size"].max()),
                "beat_ma200": float((calmar > self.ma200_calmar).mean()) if np.isfinite(self.ma200_calmar) else np.nan,
                "median_calmar": float(calmar.median()),
                "p25_calmar": float(calmar.quantile(0.25)),
                "median_cagr": float(c["cagr"].median()),
                "median_mdd": float(c["mdd"].median()),
                "median_trades": float(c["n_trades"].median()),
            })
        out = pd.DataFrame(rows).sort_values(ORDER, ascending=False, na_position="last").reset_index(drop=True)
        out.insert(0, "순위", range(1, len(out) + 1))
        return out

    def all_cells(self) -> pd.DataFrame:
        frames = [g.cells.assign(entry=e, exit=x) for (e, x), g in self.grids.items()]
        return pd.concat(frames, ignore_index=True)


def cell_count() -> int:
    return sum(len(ev) * len(xv) for ev in ENTRY_MENU.values() for xv in EXIT_MENU.values())


def ma200_calmar(panel: Panel, cfg: StrategyConfig) -> float:
    """Calmar of SPY 200-day timing on the same dates, costs and cash assumption."""
    ma = replace(cfg, entries={"above_ma": 200}, exits={"below_ma": 200}, max_positions=1,
                 min_price=0.0, min_dollar_volume=0.0)
    eq = run_backtest(panel.subset([BENCHMARK]), ma).equity
    return curve_metrics(eq.dropna())["calmar"]


def run_family_comparison(panel: Panel, base: StrategyConfig, gates: Gates, progress=None,
                          on_result=None) -> FamilyComparison:
    """Run every (entry family x exit family) grid of the fixed menu on one panel."""
    t0 = time.time()
    cache = SignalCache(panel)  # shared: indicator frames are reused across families
    total = cell_count()
    done = 0
    grids: dict[tuple[str, str], GridResult] = {}
    for e, e_vals in ENTRY_MENU.items():
        for x, x_vals in EXIT_MENU.items():
            cfg = replace(base, entries={e: e_vals[0]}, exits={x: x_vals[0]})
            offset = done

            def step(f, offset=offset, n=len(e_vals) * len(x_vals)):
                if progress:
                    progress((offset + f * n) / total)

            grids[(e, x)] = run_grid(panel, cfg, e, e_vals, x, x_vals, gates, progress=step,
                                     on_result=on_result, cache=cache)
            done += len(e_vals) * len(x_vals)
    return FamilyComparison(base, gates, grids, ma200_calmar(panel, base), time.time() - t0)
