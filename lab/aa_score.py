"""Robustness scorecard for allocation strategies (several axes, each scored 0-100 by percentile
rank within the compared set; the composite is shown as a reference only, verdicts are gates).

Axes (raw value -> higher score is better):
- 단순   fewer free parameters (AllocConfig.parameter_count)
- 일관   share of rolling 36-month windows that beat cash
- 위기   mean return relative to the benchmark (S&P 500; in KRW for the Korean-ETF panel) over the
         registered crisis windows that fall in the period
- 꼬리   CVaR 5% of monthly returns (average of the worst 5% months)
- 비용   annual one-way turnover plus leverage financing drag (lower is better)
- 샤프 전반 / 후반   excess-over-cash Sharpe in the first and second half of the period
- 최대낙폭, 칼마     from lab.metrics (same formulas as the trend lab)

Gates (pass/fail, the primary verdict, same thresholds as the trend lab's charter):
- 탐색 Gate: Calmar >= SPY Calmar and |MDD| <= 0.75 x |SPY MDD|
- 최종 Gate: CAGR >= 0.80 x SPY CAGR and |MDD| <= 0.75 x |SPY MDD|
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aa_engine import AllocResult, TRADING_DAYS, decision_rows, mix_parameter_count
from .metrics import curve_metrics

# Registered before any scorecard was run (2026-10-24). Peak-to-trough windows of SPY.
CRISIS_WINDOWS = {
    "닷컴 붕괴 2000–02": ("2000-03-24", "2002-10-09"),
    "금융위기 2007–09": ("2007-10-09", "2009-03-09"),
    "유럽 재정위기 2011": ("2011-04-29", "2011-10-03"),
    "2018년 4분기": ("2018-09-20", "2018-12-24"),
    "코로나 2020": ("2020-02-19", "2020-03-23"),
    "금리 급등 2022": ("2022-01-03", "2022-10-12"),
}
ROLLING_MONTHS = 36
AXES = ["단순", "일관", "위기", "꼬리", "비용", "샤프 전반", "샤프 후반", "최대낙폭", "칼마"]
RAW_COLUMNS = {
    "단순": ("parameters", "자유 파라미터 수", "int"),
    "일관": ("rolling_beat_cash", "36개월 구간 중 현금을 이긴 비율", "pct"),
    "위기": ("crisis_vs_spy", "위기 구간 평균 (전략 수익률 − S&P500 수익률)", "pct"),
    "꼬리": ("cvar_5", "월간 수익률 CVaR 5%", "pct"),
    "비용": ("cost_drag", "연간 회전율 + 레버리지 조달 비용", "pct"),
    "샤프 전반": ("sharpe_front", "전반부 현금 초과 샤프", "ratio"),
    "샤프 후반": ("sharpe_back", "후반부 현금 초과 샤프", "ratio"),
    "최대낙폭": ("mdd", "최대 낙폭", "pct"),
    "칼마": ("calmar", "칼마 비율", "ratio"),
}
HIGHER_IS_BETTER = {"단순": False, "일관": True, "위기": True, "꼬리": True, "비용": False, "샤프 전반": True,
                    "샤프 후반": True, "최대낙폭": True, "칼마": True}  # mdd is negative: higher (closer to 0) is better
AXIS_TEXT = {
    "단순": "자유 파라미터가 적을수록 높음. 비중 벡터 1 + 리밸런싱 주기(분기 아니면) + 밴드 + overlay(종류·안전자산) + 레버리지 배수마다 1.",
    "일관": "기간 안의 모든 36개월 구간에서 전략 수익이 현금 수익보다 높았던 비율. 특정 시기에만 좋은 전략은 낮음.",
    "위기": "등록된 위기 구간(닷컴·금융위기·2011·2018Q4·코로나·2022)에서 전략 수익률 − S&P500(원화, KODEX 미국S&P500) 수익률의 평균. 기간에 포함된 구간만 셈.",
    "꼬리": "월간 수익률 중 가장 나쁜 5%의 평균(CVaR 5%). 0에 가까울수록 높음.",
    "비용": "연간 한쪽 회전율 × 거래비용 + 레버리지 조달 스프레드·보수. 낮을수록 높음.",
    "샤프 전반": "기간 전반부(거래일 기준 절반)의 현금 초과 샤프 = mean(r − 현금) / std(r − 현금) × √252.",
    "샤프 후반": "기간 후반부의 현금 초과 샤프. 전반과 후반이 비슷해야 한 시기에 의존하지 않은 것.",
    "최대낙폭": "직전 고점 대비 최대 하락. 0에 가까울수록 높음.",
    "칼마": "CAGR / |MDD|.",
}


@dataclass(frozen=True)
class AllocGates:
    min_cagr_ratio: float = 0.80
    max_mdd_ratio: float = 0.75


def excess_sharpe(equity: pd.Series, cash: pd.Series) -> float:
    r = equity.pct_change(fill_method=None).dropna()
    ex = r - cash.reindex(r.index).fillna(0.0)
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if len(ex) > 20 and sd > 0 else float("nan")


def monthly_returns(equity: pd.Series) -> pd.Series:
    rows = decision_rows(equity.index, "M")
    return equity.iloc[rows].pct_change(fill_method=None).dropna()


def rolling_beat_cash(equity: pd.Series, cash: pd.Series, months: int = ROLLING_MONTHS) -> float:
    rows = decision_rows(equity.index, "M")
    eq_m = equity.iloc[rows]
    cash_m = (1.0 + cash.fillna(0.0)).cumprod().iloc[rows]
    if len(eq_m) <= months:
        return float("nan")
    win = (eq_m / eq_m.shift(months) - 1.0) > (cash_m / cash_m.shift(months) - 1.0)
    return float(win.iloc[months:].mean())


def cvar(monthly: pd.Series, q: float = 0.05) -> float:
    if len(monthly) < 20:
        return float("nan")
    k = max(int(np.ceil(len(monthly) * q)), 1)
    return float(np.sort(monthly.to_numpy())[:k].mean())


def crisis_table(res: AllocResult) -> pd.DataFrame:
    """Return of the strategy and the benchmark in each registered crisis window inside the period."""
    rows = []
    eq, bm = res.equity, res.benchmark
    for name, (a, b) in CRISIS_WINDOWS.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a < eq.index[0] or b > eq.index[-1]:
            continue
        e = eq.loc[a:b]
        s = bm.loc[a:b]
        if len(e) < 5:
            continue
        rows.append({"구간": name, "시작": a.date(), "끝": b.date(), "전략": e.iloc[-1] / e.iloc[0] - 1,
                     "S&P500": s.iloc[-1] / s.iloc[0] - 1})
    return pd.DataFrame(rows, columns=["구간", "시작", "끝", "전략", "S&P500"])


def cost_drag(res: AllocResult) -> float:
    cfg = res.config
    drag = res.turnover * cfg.cost_bps / 1e4
    k = res.exposure
    extra = (k - 1.0).clip(lower=0.0)
    if (extra > 0).any():
        drag += float((extra * cfg.leverage.spread_bps / 1e4).mean()) + cfg.leverage.ter_bps / 1e4 * float((extra > 0).mean())
    for s, m in cfg.leverage.per_asset.items():
        if m > 1.0 and s in res.weights.columns:
            drag += float(res.weights[s].mean()) * ((m - 1.0) * cfg.leverage.spread_bps + cfg.leverage.ter_bps) / 1e4
    return drag


def raw_metrics(res: AllocResult, parameters: int | None = None) -> dict[str, float]:
    m = curve_metrics(res.equity)
    b = curve_metrics(res.benchmark)
    eq, cash = res.equity, res.cash
    half = len(eq) // 2
    monthly = monthly_returns(eq)
    ct = crisis_table(res)
    out = {
        **m,
        "spy_cagr": b["cagr"], "spy_mdd": b["mdd"], "spy_calmar": b["calmar"],
        "sharpe_excess": excess_sharpe(eq, cash),
        "sharpe_front": excess_sharpe(eq.iloc[:half], cash), "sharpe_back": excess_sharpe(eq.iloc[half:], cash),
        "rolling_beat_cash": rolling_beat_cash(eq, cash),
        "crisis_vs_spy": float((ct["전략"] - ct["S&P500"]).mean()) if len(ct) else float("nan"),
        "crisis_n": float(len(ct)),
        "cvar_5": cvar(monthly),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
        "turnover": res.turnover,
        "cost_drag": cost_drag(res),
        "parameters": float(res.config.parameter_count() if parameters is None else parameters),
        "exposure_mean": float(res.exposure.mean()),
    }
    return out


def gate_results(m: dict, gates: AllocGates = AllocGates()) -> dict[str, bool]:
    mdd_ok = abs(m["mdd"]) <= gates.max_mdd_ratio * abs(m["spy_mdd"]) if m["spy_mdd"] < 0 else False
    cagr_ok = (m["cagr"] >= gates.min_cagr_ratio * m["spy_cagr"]) if m["spy_cagr"] > 0 else m["cagr"] >= m["spy_cagr"]
    calmar_ok = np.isfinite(m.get("calmar", np.nan)) and np.isfinite(m.get("spy_calmar", np.nan)) and m["calmar"] >= m["spy_calmar"]
    return {"탐색 Gate": bool(calmar_ok and mdd_ok), "최종 Gate": bool(cagr_ok and mdd_ok)}


def percentile_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """0-100 per axis: rank among the compared strategies (average ranks for ties, NaN stays NaN).
    With one strategy every score is 50."""
    out = pd.DataFrame(index=raw.index)
    n = len(raw)
    for axis, (col, _, _) in RAW_COLUMNS.items():
        v = raw[col].astype(float)
        if n <= 1:
            out[axis] = 50.0
            continue
        ranks = v.rank(ascending=HIGHER_IS_BETTER[axis], method="average")
        out[axis] = (ranks - 1.0) / (n - 1.0) * 100.0
    return out


def scorecard(results: dict[str, AllocResult], weights: dict[str, float] | None = None,
              parameters: dict[str, int] | None = None, gates: AllocGates = AllocGates()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(table, raw): table has 견고 (weighted mean of axis scores), the axes, gates and headline metrics."""
    raw = pd.DataFrame({name: raw_metrics(r, (parameters or {}).get(name)) for name, r in results.items()}).T
    scores = percentile_scores(raw)
    w = pd.Series({a: 1.0 for a in AXES}) if not weights else pd.Series({a: float(weights.get(a, 0.0)) for a in AXES})
    w = w / w.sum() if w.sum() > 0 else pd.Series({a: 1.0 / len(AXES) for a in AXES})
    composite = (scores[AXES].fillna(scores[AXES].mean(axis=1), axis=0) * w).sum(axis=1)  # NaN axis -> the row's mean
    table = pd.DataFrame({"견고 (참고)": composite.round(0)})
    for a in AXES:
        table[a] = scores[a].round(0)
    gate_df = pd.DataFrame({n: gate_results(raw.loc[n].to_dict(), gates) for n in raw.index}).T
    table["탐색 Gate"] = gate_df["탐색 Gate"]
    table["최종 Gate"] = gate_df["최종 Gate"]
    for col in ("cagr", "mdd", "calmar", "sharpe_excess", "volatility", "turnover"):
        table[col] = raw[col]
    return table.sort_values("견고 (참고)", ascending=False), raw


def axis_raw_table(raw: pd.DataFrame) -> pd.DataFrame:
    """The raw value behind each axis, for the hover text / detail table."""
    out = pd.DataFrame(index=raw.index)
    for axis, (col, label, kind) in RAW_COLUMNS.items():
        out[axis] = raw[col]
    return out


def parameter_count_of(obj) -> int:
    from .aa_engine import AllocConfig, MixConfig
    if isinstance(obj, MixConfig):
        return mix_parameter_count(obj)
    if isinstance(obj, AllocConfig):
        return obj.parameter_count()
    raise TypeError(type(obj))
