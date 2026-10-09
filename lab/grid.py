"""Parameter-grid robustness analysis (one or two parameters at a time).

No weighted scores. The pipeline is:
1. Gate      : each grid cell passes or fails fixed minimum conditions (vs SPY, same dates).
2. Neighbors : share of the cell's adjacent cells (up/down/left/right) that also pass.
3. Region    : passing cells joined by adjacency form a region; its size shows whether
               performance survives across a connected block of parameters or only at a point.
4. LOYO      : remove one calendar year at a time and re-check the return and drawdown gates.
5. Ordering  : passing cells sorted lexicographically by
               region size -> neighbor survival -> LOYO pass share -> Calmar.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import Panel
from .engine import SignalCache, StrategyConfig, run_backtest
from .metrics import all_metrics, curve_metrics

MAX_AXIS_VALUES = 12


@dataclass(frozen=True)
class Gates:
    min_cagr_ratio: float = 0.80
    max_mdd_ratio: float = 0.75
    min_trades: int = 30

    def to_dict(self) -> dict:
        return asdict(self)


GATE_TEXT = {
    "min_cagr_ratio": ("수익 유지", "전략 CAGR ≥ 기준 × SPY CAGR (SPY CAGR ≤ 0 이면: 전략 CAGR ≥ SPY CAGR)",
                       "기본 0.80: 프로젝트 헌장(CHARTER.md)의 'SPY CAGR의 약 80% 유지' 목표."),
    "max_mdd_ratio": ("낙폭 축소", "|전략 MDD| ≤ 기준 × |SPY MDD|",
                      "기본 0.75: 헌장의 'SPY보다 최대 낙폭을 실질적으로 줄인다' 목표."),
    "min_trades": ("최소 거래 수", "완료 거래 수 ≥ 기준",
                   "기본 30: 평균의 분포가 정규에 가까워진다고 보는 관례적 최소 표본."),
}

ROBUST_TEXT = {
    "neighbor_survival": ("이웃 생존율", "격자에서 상하좌우로 맞닿은 셀 중 Gate를 통과한 셀의 비율. 격자 끝 셀은 존재하는 이웃만 셉니다."),
    "region_size": ("연결 영역 크기", "Gate 통과 셀이 상하좌우로 이어진 덩어리의 셀 수. 1이면 주변이 모두 탈락한 고립점입니다."),
    "loyo_ratio": ("LOYO 통과 비율", "한 해씩 빼고(Leave-One-Year-Out) 다시 계산했을 때 수익·낙폭 Gate를 통과한 경우의 비율. "
                   "1.00이면 어떤 한 해에도 의존하지 않습니다. 부분 연도도 한 해로 셉니다."),
}


def _return_gate(strat_cagr: float, bench_cagr: float, g: Gates) -> bool:
    if bench_cagr > 0:
        return strat_cagr >= g.min_cagr_ratio * bench_cagr
    return strat_cagr >= bench_cagr


def check_gates(m: dict, g: Gates) -> list[str]:
    """Return the labels of failed gates (empty list = pass)."""
    failed = []
    if not _return_gate(m["cagr"], m["spy_cagr"], g):
        failed.append(GATE_TEXT["min_cagr_ratio"][0])
    if abs(m["mdd"]) > g.max_mdd_ratio * abs(m["spy_mdd"]):
        failed.append(GATE_TEXT["max_mdd_ratio"][0])
    if m["n_trades"] < g.min_trades:
        failed.append(GATE_TEXT["min_trades"][0])
    return failed


def loyo(equity: pd.Series, bench: pd.Series, g: Gates) -> tuple[float, list[int]]:
    """Leave-one-year-out check of the return and drawdown gates."""
    r_s, r_b = equity.pct_change().fillna(0.0), bench.pct_change().fillna(0.0)
    years = sorted(set(equity.index.year))
    if len(years) < 2:
        return np.nan, []
    failed = []
    for y in years:
        keep = equity.index.year != y
        s = curve_metrics((1 + r_s[keep]).cumprod())
        b = curve_metrics((1 + r_b[keep]).cumprod())
        ok = _return_gate(s["cagr"], b["cagr"], g) and abs(s["mdd"]) <= g.max_mdd_ratio * abs(b["mdd"])
        if not ok:
            failed.append(int(y))
    return 1 - len(failed) / len(years), failed


@dataclass
class GridResult:
    base: StrategyConfig
    x_key: str
    x_values: list[float]
    y_key: str | None
    y_values: list[float]
    gates: Gates
    cells: pd.DataFrame
    seconds: float

    def summary(self) -> dict:
        passed = self.cells[self.cells["pass"]]
        regions = passed["region"].nunique() if len(passed) else 0
        return {
            "total": len(self.cells),
            "passed": len(passed),
            "regions": regions,
            "largest_region": int(passed["region_size"].max()) if len(passed) else 0,
        }

    def candidates(self) -> pd.DataFrame:
        passed = self.cells[self.cells["pass"]].copy()
        passed = passed.sort_values(
            ["region_size", "neighbor_survival", "loyo_ratio", "calmar"],
            ascending=False, na_position="last",
        )
        passed.insert(0, "순위", range(1, len(passed) + 1))
        return passed


def run_grid(panel: Panel, base: StrategyConfig, x_key: str, x_values: list[float],
             y_key: str | None, y_values: list[float] | None, gates: Gates, progress=None) -> GridResult:
    if y_key == x_key:
        raise ValueError("X축과 Y축에는 서로 다른 파라미터를 고르세요.")
    y_values = y_values if y_key else [None]
    if len(x_values) > MAX_AXIS_VALUES or len(y_values) > MAX_AXIS_VALUES:
        raise ValueError(f"축마다 값은 최대 {MAX_AXIS_VALUES}개입니다.")
    t0 = time.time()
    cache = SignalCache(panel)
    rows = []
    total = len(x_values) * len(y_values)
    for yi, y in enumerate(y_values):
        for xi, x in enumerate(x_values):
            cfg = base.with_param(x_key, x)
            if y_key:
                cfg = cfg.with_param(y_key, y)
            res = run_backtest(panel, cfg, cache)
            m = all_metrics(res)
            failed = check_gates(m, gates)
            lr, lf = loyo(res.equity, res.benchmark, gates) if not failed else (np.nan, [])
            rows.append({"xi": xi, "yi": yi, "x": x, "y": y, **m, "pass": not failed,
                         "fail_reasons": ", ".join(failed), "loyo_ratio": lr,
                         "loyo_fail_years": ", ".join(map(str, lf))})
            if progress:
                progress(len(rows) / total)
    cells = pd.DataFrame(rows)
    _neighbors_and_regions(cells)
    return GridResult(base, x_key, list(x_values), y_key, list(y_values) if y_key else [], gates, cells, time.time() - t0)


def _neighbors_and_regions(cells: pd.DataFrame) -> None:
    ok = {(r.xi, r.yi): r.pass_ for r in cells.rename(columns={"pass": "pass_"}).itertuples()}

    def neighbors(k):
        x, y = k
        return [n for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)) if n in ok]

    survival, region, size = [], {}, {}
    next_id = 1
    for k in ok:
        if ok[k] and k not in region:
            members, queue = [], deque([k])
            region[k] = next_id
            while queue:
                cur = queue.popleft()
                members.append(cur)
                for n in neighbors(cur):
                    if ok[n] and n not in region:
                        region[n] = next_id
                        queue.append(n)
            for mbr in members:
                size[mbr] = len(members)
            next_id += 1
    for k in ok:
        ns = neighbors(k)
        survival.append(sum(ok[n] for n in ns) / len(ns) if ns else np.nan)
    keys = list(ok)
    cells["neighbor_survival"] = survival
    cells["region"] = [region.get(k, 0) for k in keys]
    cells["region_size"] = [size.get(k, 0) for k in keys]


def axis_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0 or stop < start:
        raise ValueError("시작 ≤ 끝, 간격 > 0 이어야 합니다.")
    vals = np.arange(start, stop + step / 2, step)
    return [float(round(v, 6)) for v in vals]
