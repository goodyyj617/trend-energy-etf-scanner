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
from .blocks import ALL_BLOCKS
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


def _cagr_mdd(values: np.ndarray) -> tuple[float, float]:
    """CAGR and MDD only (fast path for LOYO and half-period checks; same formulas as curve_metrics)."""
    v = values[~np.isnan(values)]
    if len(v) < 2 or v[0] <= 0:
        return float("nan"), float("nan")
    cagr = (v[-1] / v[0]) ** (252 / (len(v) - 1)) - 1.0 if v[-1] > 0 else -1.0
    return float(cagr), float((v / np.maximum.accumulate(v) - 1.0).min())


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
    r_s, r_b = equity.pct_change(fill_method=None).fillna(0.0), bench.pct_change(fill_method=None).fillna(0.0)
    years = sorted(set(equity.index.year))
    if len(years) < 2:
        return np.nan, []
    failed = []
    for y in years:
        keep = equity.index.year != y
        s_cagr, s_mdd = _cagr_mdd(np.cumprod(1 + r_s.to_numpy()[keep]))
        b_cagr, b_mdd = _cagr_mdd(np.cumprod(1 + r_b.to_numpy()[keep]))
        ok = _return_gate(s_cagr, b_cagr, g) and abs(s_mdd) <= g.max_mdd_ratio * abs(b_mdd)
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
             y_key: str | None, y_values: list[float] | None, gates: Gates, progress=None,
             on_result=None, cache: SignalCache | None = None) -> GridResult:
    """on_result(cfg, result) is called for every cell (the app uses it to log trials).
    Pass `cache` to share indicator frames across several grids on the same panel."""
    if y_key == x_key:
        raise ValueError("X축과 Y축에는 서로 다른 파라미터를 고르세요.")
    y_values = y_values if y_key else [None]
    if len(x_values) > MAX_AXIS_VALUES or len(y_values) > MAX_AXIS_VALUES:
        raise ValueError(f"축마다 값은 최대 {MAX_AXIS_VALUES}개입니다.")
    t0 = time.time()
    cache = cache or SignalCache(panel)
    rows = []
    total = len(x_values) * len(y_values)
    for yi, y in enumerate(y_values):
        for xi, x in enumerate(x_values):
            cfg = base.with_param(x_key, x)
            if y_key:
                cfg = cfg.with_param(y_key, y)
            res = run_backtest(panel, cfg, cache)
            if on_result:
                on_result(cfg, res)
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


# --------------------------------------------------------------------------- single-strategy checklist

CHECK_TEXT = {
    "gate": ("기본 Gate", "연구 기간 전체에서 최소 통과 조건(수익 유지·낙폭 축소·최소 거래 수)을 모두 만족."),
    "neighbors": ("주변 파라미터", "각 파라미터를 한 단계씩 위아래로 바꾼 전략 중 과반(> 50%)이 Gate를 통과. "
                  "과반이면 '그 숫자에서만 우연히 좋은' 고립점이 아니라는 뜻. 파라미터가 없으면 해당 없음."),
    "cost": ("거래 비용 2배", "편도 비용을 2배로 올려도 Gate를 통과. 실제 체결이 생각보다 나빠도 버티는지."),
    "loyo": ("어떤 한 해에도 의존하지 않음 (LOYO)", "한 해씩 빼고 다시 판정해도 수익·낙폭 Gate를 모든 경우 통과."),
    "halves": ("전반부·후반부 모두 SPY보다 칼마 높음",
               "연구 기간을 반으로 나눠 각 절반에서 전략의 칼마(CAGR ÷ |MDD|)가 SPY의 칼마 이상. 위험 대비 우위가 시기에 따라 "
               "뒤집히지 않는지 봅니다. (수익 유지 0.8×SPY는 전체 기간 Gate에서 이미 요구하므로 절반마다 다시 요구하지 않습니다.)"),
    "simple": ("단순 기준선보다 나음", "칼마 비율(수익 ÷ 최대 낙폭)이 'SPY 200일선' 전략보다 높음. "
               "규칙이 하나뿐인 전략보다 못하면 복잡하게 만들 이유가 없음."),
    "dsr": ("과최적화 보정 (DSR ≥ 0.95)", "지금까지 시험한 전략 수를 감안해도 샤프 비율이 0보다 클 확률이 95% 이상."),
}


def _calmar(values: np.ndarray) -> float:
    cagr, mdd = _cagr_mdd(values)
    return cagr / abs(mdd) if mdd < 0 else float("inf") if cagr > 0 else float("nan")


def _half_ok(eq: pd.Series, bench: pd.Series) -> tuple[bool, float, float]:
    """Halves check (v2): strategy Calmar >= SPY Calmar within the half."""
    s, b = _calmar(eq.to_numpy(dtype=float)), _calmar(bench.to_numpy(dtype=float))
    return bool(s >= b), s, b


def strategy_checklist(panel: Panel, cfg: StrategyConfig, gates: Gates, result, spy200: pd.Series,
                       dsr: float, on_result=None) -> list[dict]:
    """Pass/fail checks for one strategy. Returns [{key, label, passed (True/False/None), detail}]."""
    cache = SignalCache(panel)
    m = all_metrics(result)
    out = []

    def add(key, passed, detail):
        out.append({"key": key, "label": CHECK_TEXT[key][0], "passed": passed, "detail": detail})

    failed = check_gates(m, gates)
    add("gate", not failed, "모두 통과" if not failed else "미달: " + ", ".join(failed))

    tried, passed, calmars = 0, 0, []
    for key in cfg.tunable():
        p = ALL_BLOCKS[key].param
        for delta in (-p.step, p.step):
            v = round(cfg.param_value(key) + delta, 6)
            if not (p.minimum <= v <= p.maximum):
                continue
            ncfg = cfg.with_param(key, v)
            res = run_backtest(panel, ncfg, cache)
            if on_result:
                on_result(ncfg, res)
            nm = all_metrics(res)
            tried += 1
            passed += not check_gates(nm, gates)
            calmars.append(nm["calmar"])
    if tried:
        add("neighbors", passed / tried > 0.5,
            f"{tried}개 중 {passed}개 통과 · 칼마 {np.nanmin(calmars):.2f} – {np.nanmax(calmars):.2f} (기준 {m['calmar']:.2f})")
    else:
        add("neighbors", None, "조정할 파라미터가 없음 (고정 규칙만 사용)")

    ccfg = cfg.with_param("cost_bps", cfg.cost_bps * 2)
    cres = run_backtest(panel, ccfg, cache)
    if on_result:
        on_result(ccfg, cres)
    cm = all_metrics(cres)
    add("cost", not check_gates(cm, gates), f"비용 {ccfg.cost_bps:g}bp: CAGR {cm['cagr']:.1%}, MDD {cm['mdd']:.1%}")

    ratio, fail_years = loyo(result.equity, result.benchmark, gates)
    add("loyo", None if np.isnan(ratio) else ratio == 1.0,
        "연도가 2개 미만" if np.isnan(ratio) else
        (f"통과 비율 {ratio:.2f}" + (f" · 빼면 실패하는 해: {', '.join(map(str, fail_years))}" if fail_years else "")))

    eq, bm = result.equity, result.benchmark
    mid = len(eq) // 2
    if mid >= 126:
        (ok1, s1, b1), (ok2, s2, b2) = _half_ok(eq.iloc[:mid], bm.iloc[:mid]), _half_ok(eq.iloc[mid:], bm.iloc[mid:])
        add("halves", ok1 and ok2,
            f"전반부({eq.index[0].date()} – {eq.index[mid - 1].date()}) 칼마 {s1:.2f} vs SPY {b1:.2f} · "
            f"후반부({eq.index[mid].date()} – {eq.index[-1].date()}) 칼마 {s2:.2f} vs SPY {b2:.2f}")
    else:
        add("halves", None, "기간이 1년 미만이라 나눌 수 없음")

    s200 = curve_metrics(spy200.dropna())["calmar"]
    add("simple", bool(m["calmar"] > s200), f"전략 칼마 {m['calmar']:.2f} vs SPY 200일선 {s200:.2f}")
    add("dsr", None if np.isnan(dsr) else dsr >= 0.95, "계산 불가 (기간이 너무 짧음)" if np.isnan(dsr) else f"DSR {dsr:.2f}")
    return out
