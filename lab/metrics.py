"""Portfolio-level (primary) and trade-level (secondary) metrics.

DEFINITIONS is the single source for the UI labels, tooltips and the glossary page,
so a metric shown on screen always carries the definition used in this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Definition:
    label: str
    short: str
    formula: str
    fmt: str  # "pct", "ratio", "num", "int", "days"
    better: str  # "high", "low", ""
    group: str


DEFINITIONS: dict[str, Definition] = {
    # ---- portfolio (primary)
    "cagr": Definition("연평균 수익률 (CAGR)", "기간 전체를 1년 단위 복리로 환산한 수익률.",
                       "(최종 자산 / 시작 자산)^(252 / 거래일 수) − 1", "pct", "high", "포트폴리오"),
    "total_return": Definition("누적 수익률", "기간 시작부터 끝까지의 총 수익률.",
                               "최종 자산 / 시작 자산 − 1", "pct", "high", "포트폴리오"),
    "volatility": Definition("연 변동성", "일간 수익률의 표준편차를 연율화한 값.",
                             "std(일간 수익률) × √252", "pct", "low", "포트폴리오"),
    "sharpe": Definition("샤프 비율 (Sharpe)", "변동성 한 단위당 수익. 무위험 수익률은 0으로 가정.",
                         "mean(일간 수익률) / std(일간 수익률) × √252", "ratio", "high", "포트폴리오"),
    "mdd": Definition("최대 낙폭 (MDD)", "직전 최고점 대비 가장 크게 떨어졌던 비율.",
                      "min_t ( 자산_t / max_{s≤t} 자산_s − 1 )", "pct", "high", "포트폴리오"),
    "calmar": Definition("칼마 비율 (Calmar)", "최대 낙폭 한 단위당 연수익.",
                         "CAGR / |MDD|", "ratio", "high", "포트폴리오"),
    "sortino": Definition("소르티노 비율 (Sortino)", "하락 변동성 한 단위당 수익. 오르는 변동성은 벌점으로 치지 않음.",
                          "mean(일간 수익률) / √mean(min(일간 수익률, 0)²) × √252", "ratio", "high", "포트폴리오"),
    "worst_12m": Definition("최악의 12개월 수익률", "기간 중 어느 시점에 시작했든 1년(252거래일) 동안 겪은 가장 나쁜 수익률.",
                            "min_t ( 자산_t / 자산_{t−252} − 1 )", "pct", "high", "포트폴리오"),
    "month_win": Definition("월간 승률", "수익이 플러스였던 달의 비율.",
                            "count(월 수익률 > 0) / 전체 월 수", "pct", "high", "포트폴리오"),
    "underwater_days": Definition("최장 회복 기간", "직전 최고점을 회복하지 못한 채 머문 가장 긴 기간.",
                                  "자산 < 직전 최고점 인 연속 거래일 수의 최댓값", "days", "low", "포트폴리오"),
    "exposure": Definition("평균 투자 비중", "자산 중 주식/ETF에 투자된 비율의 평균. 나머지는 현금.",
                           "mean( 보유 평가액_t / 자산_t )", "pct", "", "포트폴리오"),
    # ---- relative to SPY (primary)
    "cagr_ratio": Definition("SPY 대비 수익 비율", "전략 CAGR이 같은 기간 SPY CAGR의 몇 배인지.",
                             "전략 CAGR / SPY CAGR  (SPY CAGR ≤ 0 이면 계산하지 않음)", "ratio", "high", "SPY 대비"),
    "mdd_ratio": Definition("SPY 대비 낙폭 비율", "전략 최대 낙폭이 SPY 최대 낙폭의 몇 배인지. 1보다 작을수록 덜 빠짐.",
                            "|전략 MDD| / |SPY MDD|", "ratio", "low", "SPY 대비"),
    "excess_cagr": Definition("SPY 대비 초과 CAGR", "전략 CAGR − SPY CAGR.",
                              "전략 CAGR − SPY CAGR", "pct", "high", "SPY 대비"),
    # ---- trades (secondary: statistical edge per trade)
    "n_trades": Definition("완료 거래 수", "기간 안에 진입과 청산이 모두 끝난 거래의 수.",
                           "count(완료 거래)", "int", "high", "거래"),
    "win_rate": Definition("승률", "비용 차감 후 수익이 0보다 큰 거래의 비율.",
                           "count(거래 수익 > 0) / 완료 거래 수", "pct", "", "거래"),
    "avg_trade": Definition("평균 거래 수익", "완료 거래 수익률(비용 차감)의 산술 평균.",
                            "mean(거래 수익률)", "pct", "high", "거래"),
    "median_trade": Definition("중앙 거래 수익", "완료 거래 수익률의 중앙값. 추세추종은 소수의 큰 수익에 의존하므로 음수일 수 있음.",
                               "median(거래 수익률)", "pct", "", "거래"),
    "profit_factor": Definition("손익비 (Profit Factor)", "이긴 거래 수익 합이 진 거래 손실 합의 몇 배인지.",
                                "Σ(양의 거래 수익률) / |Σ(음의 거래 수익률)|", "ratio", "high", "거래"),
    "payoff": Definition("평균 손익 배수", "평균 이익 거래 크기 / 평균 손실 거래 크기.",
                         "mean(양의 거래 수익률) / |mean(음의 거래 수익률)|", "ratio", "high", "거래"),
    "t_stat": Definition("평균 수익 t-통계량", "평균 거래 수익이 0과 통계적으로 다른 정도. 약 2 이상이면 우연일 가능성이 낮다는 관례적 기준. "
                         "단, 동시 보유 거래는 서로 독립이 아니므로 실제 신뢰도는 이 값보다 낮습니다.",
                         "mean(거래 수익률) / ( std(거래 수익률) / √거래 수 )", "ratio", "high", "거래"),
    "avg_hold": Definition("평균 보유 기간", "완료 거래의 평균 보유 거래일 수.",
                           "mean(청산일 − 진입일, 거래일)", "days", "", "거래"),
    "trades_per_year": Definition("연간 거래 수", "1년당 평균 완료 거래 수. 비용 부담의 대략적 척도.",
                                  "완료 거래 수 / (거래일 수 / 252)", "num", "", "거래"),
    # ---- overfitting control
    "dsr": Definition("DSR (과최적화 보정)",
                      "지금까지 시험한 전략 수(N)를 감안해도 이 전략의 진짜 샤프 비율이 0보다 클 확률 (Deflated Sharpe Ratio, "
                      "Bailey & López de Prado 2014). 많이 시험할수록 '운 좋은 최고치'의 기준선 SR₀가 올라가 확률이 낮아집니다. "
                      "0.95 이상이면 관례적으로 유의. 비슷한 전략(격자 이웃)을 여러 번 센 경우 N이 과대 → 보수적으로 나옵니다.",
                      "Φ( (SR − SR₀)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) ),  "
                      "SR₀ = √V·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))],  "
                      "SR=일간 샤프, T=일수, γ₃·γ₄=일간 수익률 왜도·첨도, V=시험한 전략들의 일간 샤프 분산, γ≈0.5772",
                      "pct", "high", "과최적화"),
}


def fmt_value(key: str, value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    kind = DEFINITIONS[key].fmt if key in DEFINITIONS else "num"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind in ("int", "days"):
        return f"{value:,.0f}" + ("일" if kind == "days" else "")
    return f"{value:,.1f}"


# --------------------------------------------------------------------------- portfolio

def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _longest_underwater(equity: pd.Series) -> int:
    under = (equity < equity.cummax()).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def curve_metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.dropna()
    rets = equity.pct_change(fill_method=None).dropna()
    days = max(len(equity) - 1, 1)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / days) - 1.0 if equity.iloc[-1] > 0 else -1.0
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = rets.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else np.nan
    downside = np.sqrt((rets.clip(upper=0) ** 2).mean())
    mdd = drawdown(equity).min()
    roll = equity / equity.shift(TRADING_DAYS) - 1.0
    monthly = equity.resample("ME").last().pct_change(fill_method=None).dropna()
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": rets.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan,
        "mdd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "worst_12m": roll.min() if roll.notna().any() else np.nan,
        "month_win": (monthly > 0).mean() if len(monthly) else np.nan,
        "underwater_days": float(_longest_underwater(equity)),
    }


def daily_sharpe(equity: pd.Series) -> float:
    """Non-annualised Sharpe of daily returns (the unit the DSR formula uses)."""
    r = equity.dropna().pct_change(fill_method=None).dropna()
    return float(r.mean() / r.std()) if len(r) > 1 and r.std() > 0 else float("nan")


def deflated_sharpe(equity: pd.Series, n_trials: int, sr_variance: float) -> float:
    """Probability that the true Sharpe is > 0 after selecting the best of n_trials (DSR).

    With n_trials <= 1 or no variance information this is the Probabilistic Sharpe Ratio
    against zero.
    """
    from statistics import NormalDist

    r = equity.dropna().pct_change(fill_method=None).dropna()
    T = len(r)
    if T < 30 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    g3 = float(((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3)
    g4 = float(((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4)
    nd = NormalDist()
    sr0 = 0.0
    if n_trials > 1 and sr_variance > 0:
        gamma = 0.5772156649
        sr0 = np.sqrt(sr_variance) * ((1 - gamma) * nd.inv_cdf(1 - 1 / n_trials)
                                      + gamma * nd.inv_cdf(1 - 1 / (n_trials * np.e)))
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(nd.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)))


def relative_metrics(strat: dict, bench: dict) -> dict[str, float]:
    return {
        "cagr_ratio": strat["cagr"] / bench["cagr"] if bench["cagr"] > 0 else np.nan,
        "mdd_ratio": abs(strat["mdd"]) / abs(bench["mdd"]) if bench["mdd"] < 0 else np.nan,
        "excess_cagr": strat["cagr"] - bench["cagr"],
    }


def trade_metrics(trades: pd.DataFrame, n_days: int) -> dict[str, float]:
    r = trades["return"].astype(float) if len(trades) else pd.Series(dtype=float)
    n = len(r)
    wins, losses = r[r > 0], r[r < 0]
    sd = r.std() if n > 1 else np.nan
    return {
        "n_trades": float(n),
        "win_rate": (r > 0).mean() if n else np.nan,
        "avg_trade": r.mean() if n else np.nan,
        "median_trade": r.median() if n else np.nan,
        "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan,
        "payoff": wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else np.nan,
        "t_stat": r.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
        "avg_hold": trades["holding_days"].mean() if n else np.nan,
        "trades_per_year": n / (max(n_days, 1) / TRADING_DAYS),
    }


def all_metrics(result) -> dict[str, float]:
    strat = curve_metrics(result.equity)
    bench = curve_metrics(result.benchmark)
    out = {**strat, **relative_metrics(strat, bench), "exposure": float(result.exposure.mean())}
    out.update(trade_metrics(result.trades, len(result.equity)))
    out.update({f"spy_{k}": v for k, v in bench.items()})
    return out


def annual_table(result) -> pd.DataFrame:
    """Calendar-year returns for strategy and SPY. Partial first/last years are flagged."""
    eq, bm = result.equity, result.benchmark
    rows = []
    for year, part in eq.groupby(eq.index.year):
        prev_eq = eq[eq.index < part.index[0]]
        prev_bm = bm[bm.index < part.index[0]]
        start_eq = prev_eq.iloc[-1] if len(prev_eq) else part.iloc[0]
        start_bm = prev_bm.iloc[-1] if len(prev_bm) else bm.loc[part.index[0]]
        s = part.iloc[-1] / start_eq - 1
        b = bm.loc[part.index[-1]] / start_bm - 1
        tr = result.trades[pd.to_datetime(result.trades["exit_date"]).dt.year == year]["return"]
        rows.append({
            "연도": int(year),
            "전략": s, "SPY": b, "차이": s - b,
            "전략 MDD": drawdown(part / start_eq).min(),
            "완료 거래": len(tr),
            "평균 거래 수익": tr.mean() if len(tr) else np.nan,
            "부분 연도": len(part) < 240,
        })
    return pd.DataFrame(rows)
