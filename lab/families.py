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

Common add-on test: one extra condition (an entry filter or an extra exit, at its block's
default value) is added to EVERY family and each family's grid is compared with the plain
family on the four ordering metrics. It answers "does this component help in general?"
rather than "does it help this one family?".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .blocks import ENTRY_BLOCKS, EXIT_BLOCKS
from .data import BENCHMARK, Panel
from .engine import SignalCache, StrategyConfig, run_backtest
from .grid import Gates, GridResult, run_grid
from .metrics import curve_metrics

# v1 (2026-10-09) was the first three entries x first three exits below, with the same values.
MENU_VERSION = "family-menu-v2 (2026-10-10)"
ENTRY_MENU: dict[str, list[float]] = {
    "breakout": [20, 30, 50, 80, 120, 200],
    "above_ma": [50, 80, 100, 150, 200, 250],
    "momentum": [21, 42, 63, 126, 189, 252],
    "ma_cross": [40, 60, 100, 150, 200, 250],
    "bollinger": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    "rsi_min": [50, 55, 60, 65, 70, 75],
    "williams": [-50, -40, -30, -20, -10, -5],
}
EXIT_MENU: dict[str, list[float]] = {
    "low_break": [10, 15, 20, 30, 50, 80],
    "below_ma": [20, 30, 50, 100, 150, 200],
    "atr_trail": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "trailing_pct": [5, 8, 10, 15, 20, 30],
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
VERDICT_TEXT = {
    "base": "기준 (추가 없음)",
    "improve": "개선 — 모든 기준에서 같거나 낫고 하나 이상 나음",
    "worse": "악화 — 모든 기준에서 같거나 못하고 하나 이상 못함",
    "mixed": "엇갈림 — 나아진 기준과 나빠진 기준이 섞임",
    "same": "변화 없음",
}


def verdict(base: dict, other: dict) -> str:
    """Pareto comparison of two grid summaries on the four ordering metrics."""
    diffs = []
    for k in ORDER:
        a, b = base[k], other[k]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        diffs.append(0 if np.isclose(a, b) else (1 if b > a else -1))
    if not diffs or all(d == 0 for d in diffs):
        return "same"
    if all(d >= 0 for d in diffs):
        return "improve"
    if all(d <= 0 for d in diffs):
        return "worse"
    return "mixed"


def grid_summary(cells: pd.DataFrame, ma200: float) -> dict:
    """Whole-grid behaviour of one family (the SUMMARY_TEXT metrics)."""
    calmar = cells["calmar"].astype(float)
    return {
        "pass_share": float(cells["pass"].mean()),
        "largest_region": int(cells["region_size"].max()),
        "beat_ma200": float((calmar > ma200).mean()) if np.isfinite(ma200) else np.nan,
        "median_calmar": float(calmar.median()),
        "p25_calmar": float(calmar.quantile(0.25)),
        "median_cagr": float(cells["cagr"].median()),
        "median_mdd": float(cells["mdd"].median()),
        "median_trades": float(cells["n_trades"].median()),
    }


@dataclass
class FamilyComparison:
    base: StrategyConfig
    gates: Gates
    grids: dict[tuple[str, str], GridResult]
    ma200_calmar: float
    seconds: float
    extra_entries: dict = field(default_factory=dict)  # common add-on given to every family
    extra_exits: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = [{"entry": e, "exit": x, "family": f"{ENTRY_BLOCKS[e].label} × {EXIT_BLOCKS[x].label}",
                 **grid_summary(g.cells, self.ma200_calmar)} for (e, x), g in self.grids.items()]
        out = pd.DataFrame(rows).sort_values(ORDER, ascending=False, na_position="last").reset_index(drop=True)
        out.insert(0, "순위", range(1, len(out) + 1))
        return out

    def all_cells(self) -> pd.DataFrame:
        frames = [g.cells.assign(entry=e, exit=x) for (e, x), g in self.grids.items()]
        return pd.concat(frames, ignore_index=True)


def family_pairs(extra_entries: dict | None = None, extra_exits: dict | None = None) -> list[tuple[str, str]]:
    """Menu families; a family whose own entry/exit IS the common add-on is skipped (it cannot be added twice)."""
    return [(e, x) for e in ENTRY_MENU for x in EXIT_MENU
            if e not in (extra_entries or {}) and x not in (extra_exits or {})]


def cell_count(extra_entries: dict | None = None, extra_exits: dict | None = None) -> int:
    return sum(len(ENTRY_MENU[e]) * len(EXIT_MENU[x]) for e, x in family_pairs(extra_entries, extra_exits))


def ma200_calmar(panel: Panel, cfg: StrategyConfig) -> float:
    """Calmar of SPY 200-day timing on the same dates, costs and cash assumption."""
    ma = replace(cfg, entries={"above_ma": 200}, exits={"below_ma": 200}, max_positions=1,
                 min_price=0.0, min_dollar_volume=0.0)
    eq = run_backtest(panel.subset([BENCHMARK]), ma).equity
    return curve_metrics(eq.dropna())["calmar"]


def run_family_comparison(panel: Panel, base: StrategyConfig, gates: Gates, progress=None,
                          on_result=None, extra_entries: dict | None = None, extra_exits: dict | None = None,
                          cache: SignalCache | None = None) -> FamilyComparison:
    """Run every (entry family x exit family) grid of the fixed menu on one panel.

    extra_entries / extra_exits are added unchanged to every family (common add-on test).
    """
    t0 = time.time()
    extra_entries, extra_exits = dict(extra_entries or {}), dict(extra_exits or {})
    cache = cache or SignalCache(panel)  # shared: indicator frames are reused across families
    total = cell_count(extra_entries, extra_exits)
    done = 0
    grids: dict[tuple[str, str], GridResult] = {}
    for e, x in family_pairs(extra_entries, extra_exits):
        e_vals, x_vals = ENTRY_MENU[e], EXIT_MENU[x]
        cfg = replace(base, entries={e: e_vals[0], **extra_entries}, exits={x: x_vals[0], **extra_exits})
        offset = done

        def step(f, offset=offset, n=len(e_vals) * len(x_vals)):
            if progress:
                progress((offset + f * n) / total)

        grids[(e, x)] = run_grid(panel, cfg, e, e_vals, x, x_vals, gates, progress=step,
                                 on_result=on_result, cache=cache)
        done += len(e_vals) * len(x_vals)
    return FamilyComparison(base, gates, grids, ma200_calmar(panel, base), time.time() - t0,
                            extra_entries, extra_exits)


# --------------------------------------------------------------------------- common add-on test

def addon_choices() -> list[tuple[str, str]]:
    """(kind, key) of every block that can be added to all families at its default value."""
    return [("entry", k) for k in ENTRY_BLOCKS] + [("exit", k) for k in EXIT_BLOCKS]


def addon_default(kind: str, key: str) -> float | None:
    p = (ENTRY_BLOCKS if kind == "entry" else EXIT_BLOCKS)[key].param
    return None if p is None else p.default


def addon_label(kind: str, key: str) -> str:
    b = (ENTRY_BLOCKS if kind == "entry" else EXIT_BLOCKS)[key]
    v = addon_default(kind, key)
    return f"{'진입 조건' if kind == 'entry' else '청산 조건'} · {b.label}" + ("" if v is None else f" ({v:g})")


@dataclass
class AddonTest:
    kind: str  # "entry" | "exit"
    key: str
    value: float | None
    plain: FamilyComparison  # every family without the add-on
    added: FamilyComparison  # the same families with the add-on

    @property
    def label(self) -> str:
        return addon_label(self.kind, self.key)

    def table(self) -> pd.DataFrame:
        """One row per tested family: plain vs with add-on on the ordering metrics, and the verdict."""
        rows = []
        for (e, x), g in self.added.grids.items():
            a = grid_summary(self.plain.grids[(e, x)].cells, self.plain.ma200_calmar)
            b = grid_summary(g.cells, self.added.ma200_calmar)
            rows.append({"entry": e, "exit": x, "family": f"{ENTRY_BLOCKS[e].label} × {EXIT_BLOCKS[x].label}",
                         "verdict": verdict(a, b),
                         **{f"{k}_plain": a[k] for k in ORDER}, **{f"{k}_added": b[k] for k in ORDER}})
        out = pd.DataFrame(rows)
        rank = {"improve": 0, "mixed": 1, "same": 2, "worse": 3}
        out = out.assign(_r=out["verdict"].map(rank)).sort_values(["_r", "family"]).drop(columns="_r")
        return out.reset_index(drop=True)

    def counts(self) -> dict[str, int]:
        v = self.table()["verdict"]
        return {k: int((v == k).sum()) for k in ("improve", "mixed", "same", "worse")}

    def all_cells(self) -> pd.DataFrame:
        frames = [g.cells.assign(entry=e, exit=x, variant=variant)
                  for variant, fc in (("plain", self.plain), ("added", self.added))
                  for (e, x), g in fc.grids.items() if (e, x) in self.added.grids]
        return pd.concat(frames, ignore_index=True)


def run_addon_test(panel: Panel, base: StrategyConfig, gates: Gates, kind: str, key: str,
                   plain: FamilyComparison | None = None, progress=None, on_result=None) -> AddonTest:
    """Add one block (at its default value) to every family and compare with the plain families.

    A finished plain comparison on the same panel and settings can be passed in to skip rerunning it.
    """
    value = addon_default(kind, key)
    ee, xx = ({key: value}, {}) if kind == "entry" else ({}, {key: value})
    cache = SignalCache(panel)
    n_plain = 0 if plain is not None else cell_count()
    n_added = cell_count(ee, xx)
    total = n_plain + n_added

    def part(offset, n):
        return (lambda f: progress((offset + f * n) / total)) if progress else None

    if plain is None:
        plain = run_family_comparison(panel, base, gates, part(0, n_plain), on_result, cache=cache)
    added = run_family_comparison(panel, base, gates, part(n_plain, n_added), on_result, ee, xx, cache=cache)
    return AddonTest(kind, key, value, plain, added)
