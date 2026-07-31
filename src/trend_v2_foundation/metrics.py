"""Metric metadata registry and adapters for existing reliable portfolio metrics."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from .contracts import ArtifactKind, MetricDefinition, MetricDirection


def _metric(
    key: str,
    ko: str,
    en: str,
    abbreviation: str,
    direction: MetricDirection,
    unit: str,
    annualization: str,
    *,
    source: str | None = None,
    gates: bool = True,
    pareto: bool = True,
    weighted: bool = True,
    robustness: bool = False,
    diagnostics: bool = True,
    artifacts: tuple[ArtifactKind, ...] = (ArtifactKind.DAILY_PORTFOLIO_CURVE,),
) -> MetricDefinition:
    return MetricDefinition(
        metric_key=key,
        korean_name=ko,
        english_name=en,
        abbreviation=abbreviation,
        direction=direction,
        unit=unit,
        annualization_convention=annualization,
        required_input_artifacts=artifacts,
        suitable_for_gates=gates,
        suitable_for_pareto=pareto,
        suitable_for_weighted_view=weighted,
        suitable_for_robustness=robustness,
        suitable_for_diagnostics=diagnostics,
        source_summary_key=source,
    )


_REGISTRY = {
    item.metric_key: item
    for item in (
        _metric("cagr", "연복리수익률", "Compound Annual Growth Rate", "CAGR", MetricDirection.MAXIMIZE, "ratio", "calendar days / 365.25", source="cagr"),
        _metric("cagr_spy_ratio", "SPY 대비 연복리수익률", "CAGR relative to SPY", "CAGR/SPY", MetricDirection.MAXIMIZE, "ratio", "calendar days / 365.25"),
        _metric("annualized_volatility", "연환산 변동성", "Annualized Volatility", "Vol", MetricDirection.MINIMIZE, "ratio", "daily sample standard deviation × sqrt(252)", source="annualized_volatility"),
        _metric("sharpe_ratio", "샤프지수", "Sharpe Ratio", "Sharpe", MetricDirection.MAXIMIZE, "ratio", "daily mean / daily sample standard deviation × sqrt(252)", source="sharpe_ratio"),
        _metric("sortino_ratio", "소르티노지수", "Sortino Ratio", "Sortino", MetricDirection.MAXIMIZE, "ratio", "daily mean / downside deviation × sqrt(252)", source="sortino_ratio"),
        _metric("maximum_drawdown", "최대낙폭", "Maximum Drawdown", "MDD", MetricDirection.MAXIMIZE, "ratio (non-positive)", "not annualized", source="maximum_drawdown"),
        _metric("maximum_drawdown_spy_ratio", "SPY 대비 최대낙폭", "Maximum Drawdown relative to SPY", "MDD/SPY", MetricDirection.MINIMIZE, "magnitude ratio", "not annualized"),
        _metric("cdar95", "조건부 낙폭 위험 95%", "Conditional Drawdown at Risk 95%", "CDaR95", MetricDirection.MAXIMIZE, "ratio (non-positive)", "not annualized", source="conditional_drawdown_at_risk_95"),
        _metric("cdar95_spy_ratio", "SPY 대비 CDaR95", "CDaR95 relative to SPY", "CDaR95/SPY", MetricDirection.MINIMIZE, "magnitude ratio", "not annualized"),
        _metric("calmar_ratio", "칼마지수", "Calmar Ratio", "Calmar", MetricDirection.MAXIMIZE, "ratio", "CAGR divided by absolute MDD", source="calmar_ratio"),
        _metric("calmar_spy_ratio", "SPY 대비 칼마지수", "Calmar relative to SPY", "Calmar/SPY", MetricDirection.MAXIMIZE, "ratio", "CAGR divided by absolute MDD"),
        _metric("recovery_duration_days", "회복기간", "Recovery Duration", "Recovery", MetricDirection.MINIMIZE, "calendar days", "not annualized", source="max_drawdown_duration_days"),
        _metric("annual_turnover", "연환산 회전율", "Annual Turnover", "Turnover", MetricDirection.MINIMIZE, "ratio per year", "sum of daily turnover / calendar years", source="annual_turnover"),
        _metric("walk_forward_pass_ratio", "워크포워드 통과비율", "Walk-forward Pass Ratio", "WF pass", MetricDirection.MAXIMIZE, "ratio", "not annualized", gates=True, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("walk_forward_worst_fold", "워크포워드 최악 폴드", "Walk-forward Worst Fold", "WF worst", MetricDirection.MAXIMIZE, "ratio", "not annualized", gates=True, pareto=True, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("loyo_reversing_years", "LOYO 반전 연도 수", "Leave-One-Year-Out Reversing Years", "LOYO reverse", MetricDirection.MINIMIZE, "count", "not annualized", gates=True, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("transaction_cost_stress_survival", "거래비용 스트레스 생존", "Transaction-cost Stress Survival", "Cost stress", MetricDirection.MAXIMIZE, "binary", "not annualized", gates=True, pareto=False, weighted=False, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("block_bootstrap_effect", "블록 부트스트랩 효과", "Paired Block-bootstrap Effect", "Block bootstrap", MetricDirection.MAXIMIZE, "ratio", "defined by robustness study", gates=False, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
    )
}

METRIC_REGISTRY: Mapping[str, MetricDefinition] = MappingProxyType(_REGISTRY)


def metric_registry() -> Mapping[str, MetricDefinition]:
    return METRIC_REGISTRY


def _ratio(numerator: Any, denominator: Any, *, magnitude: bool = False) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    left = abs(float(numerator)) if magnitude else float(numerator)
    right = abs(float(denominator)) if magnitude else float(denominator)
    return left / right if right != 0 else None


def metrics_from_portfolio_summaries(
    strategy_summary: Mapping[str, Any],
    benchmark_summary: Mapping[str, Any],
) -> dict[str, float | int | None]:
    """Map existing ``summarize_portfolio_curve`` output without recomputing it."""

    metrics: dict[str, float | int | None] = {}
    for definition in METRIC_REGISTRY.values():
        if definition.source_summary_key:
            metrics[definition.metric_key] = strategy_summary.get(definition.source_summary_key)
    metrics.update(
        {
            "cagr_spy_ratio": _ratio(strategy_summary.get("cagr"), benchmark_summary.get("cagr")),
            "maximum_drawdown_spy_ratio": _ratio(
                strategy_summary.get("maximum_drawdown"),
                benchmark_summary.get("maximum_drawdown"),
                magnitude=True,
            ),
            "cdar95_spy_ratio": _ratio(
                strategy_summary.get("conditional_drawdown_at_risk_95"),
                benchmark_summary.get("conditional_drawdown_at_risk_95"),
                magnitude=True,
            ),
            "calmar_spy_ratio": _ratio(
                strategy_summary.get("calmar_ratio"), benchmark_summary.get("calmar_ratio")
            ),
        }
    )
    return metrics
