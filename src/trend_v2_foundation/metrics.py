"""Metric metadata registry and adapters for existing reliable portfolio metrics."""

from __future__ import annotations

import math
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from .contracts import ArtifactKind, MetricDefinition, MetricDirection

if TYPE_CHECKING:
    from .contracts import EvaluationProfile


class EvaluationProfileValidationError(ValueError):
    """A configured profile is inconsistent with the metric registry."""


class MetricValueRepresentationError(ValueError):
    """A metric artifact uses an invalid numeric representation."""


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
    numeric_representation: str = "continuous_numeric",
    allowed_numeric_values: tuple[float, ...] = (),
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
        numeric_representation=numeric_representation,
        allowed_numeric_values=allowed_numeric_values,
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
        _metric("longest_recovery_duration_days", "최장 수중 기간", "Longest Time Under Water", "TUW", MetricDirection.MINIMIZE, "calendar days", "not annualized"),
        _metric("current_time_under_water_days", "현재 미회복 수중 기간", "Current Unfinished Time Under Water", "Current TUW", MetricDirection.MINIMIZE, "calendar days", "not annualized"),
        _metric("recovery_duration_spy_ratio", "SPY 대비 회복기간", "Recovery Duration relative to SPY", "Recovery/SPY", MetricDirection.MINIMIZE, "magnitude ratio", "not annualized"),
        _metric("worst_rolling_3_month_return", "최악 3개월 롤링 수익률", "Worst Rolling 3-month Return", "Worst 3M", MetricDirection.MAXIMIZE, "ratio", "63 trading sessions"),
        _metric("worst_rolling_12_month_return", "최악 12개월 롤링 수익률", "Worst Rolling 12-month Return", "Worst 12M", MetricDirection.MAXIMIZE, "ratio", "252 trading sessions"),
        _metric("worst_rolling_36_month_annualized_return", "최악 36개월 연환산 롤링 수익률", "Worst Rolling 36-month Annualized Return", "Worst 36M ann.", MetricDirection.MAXIMIZE, "ratio", "756 sessions annualized to 252 sessions"),
        _metric("worst_rolling_3_month_spy_ratio", "SPY 대비 최악 3개월 롤링 수익률", "Worst Rolling 3-month Return relative to SPY", "Worst 3M/SPY", MetricDirection.MINIMIZE, "magnitude ratio", "63 trading sessions"),
        _metric("worst_rolling_12_month_spy_ratio", "SPY 대비 최악 12개월 롤링 수익률", "Worst Rolling 12-month Return relative to SPY", "Worst 12M/SPY", MetricDirection.MINIMIZE, "magnitude ratio", "252 trading sessions"),
        _metric("worst_rolling_36_month_spy_ratio", "SPY 대비 최악 36개월 롤링 수익률", "Worst Rolling 36-month Return relative to SPY", "Worst 36M/SPY", MetricDirection.MINIMIZE, "magnitude ratio", "756 trading sessions"),
        _metric("annual_turnover", "연환산 회전율", "Annual Turnover", "Turnover", MetricDirection.MINIMIZE, "ratio per year", "sum of daily turnover / calendar years", source="annual_turnover"),
        _metric("transaction_cost_drag", "거래비용 수익률 잠식", "Transaction-cost Drag", "Cost drag", MetricDirection.MINIMIZE, "annual return difference", "gross CAGR minus net CAGR"),
        _metric("average_gross_exposure", "평균 총 익스포저", "Average Gross Exposure", "Gross exposure", MetricDirection.MINIMIZE, "ratio", "daily arithmetic mean"),
        _metric("average_cash_weight", "평균 현금 비중", "Average Cash Weight", "Cash weight", MetricDirection.MINIMIZE, "ratio", "daily arithmetic mean", pareto=False),
        _metric("average_position_count", "평균 보유 종목 수", "Average Position Count", "Avg positions", MetricDirection.MINIMIZE, "count", "daily arithmetic mean", pareto=False),
        _metric("maximum_position_count", "최대 보유 종목 수", "Maximum Position Count", "Max positions", MetricDirection.MINIMIZE, "count", "not annualized", pareto=False),
        _metric("walk_forward_pass_ratio", "워크포워드 통과비율", "Walk-forward Pass Ratio", "WF pass", MetricDirection.MAXIMIZE, "ratio", "not annualized", gates=True, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("walk_forward_worst_fold", "워크포워드 최악 폴드", "Walk-forward Worst Fold", "WF worst", MetricDirection.MAXIMIZE, "ratio", "not annualized", gates=True, pareto=True, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("loyo_stability_ratio", "LOYO 안정성 비율", "Leave-One-Year-Out Stability Ratio", "LOYO stability", MetricDirection.MAXIMIZE, "ratio", "not annualized", gates=True, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("loyo_reversing_years", "LOYO 반전 연도 수", "Leave-One-Year-Out Reversing Years", "LOYO reverse", MetricDirection.MINIMIZE, "count", "not annualized", gates=True, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("transaction_cost_stress_survival", "거래비용 스트레스 생존", "Transaction-cost Stress Survival", "Cost stress", MetricDirection.MAXIMIZE, "binary numeric (0.0 or 1.0)", "not annualized", gates=True, pareto=False, weighted=False, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,), numeric_representation="binary_numeric", allowed_numeric_values=(0.0, 1.0)),
        _metric("block_bootstrap_effect", "블록 부트스트랩 효과", "Paired Block-bootstrap Effect", "Block bootstrap", MetricDirection.MAXIMIZE, "ratio", "defined by robustness study", gates=False, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("block_bootstrap_ci_lower", "블록 부트스트랩 신뢰구간 하한", "Block-bootstrap Confidence Interval Lower Bound", "Bootstrap CI low", MetricDirection.MAXIMIZE, "ratio", "defined by robustness study", gates=True, pareto=False, weighted=False, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("block_bootstrap_ci_upper", "블록 부트스트랩 신뢰구간 상한", "Block-bootstrap Confidence Interval Upper Bound", "Bootstrap CI high", MetricDirection.MAXIMIZE, "ratio", "defined by robustness study", gates=False, pareto=False, weighted=False, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("adjusted_p_value", "다중검정 보정 p값", "Multiple-testing Adjusted p-value", "Adjusted p", MetricDirection.MINIMIZE, "probability", "not annualized", gates=True, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("dominant_group_share", "지배 자산군 비중", "Dominant Asset-group Share", "Dominant group", MetricDirection.MINIMIZE, "ratio", "not annualized", gates=True, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
        _metric("unclassified_group_share", "미분류 자산군 비중", "Unclassified Asset-group Share", "Unclassified group", MetricDirection.MINIMIZE, "ratio", "not annualized", gates=True, pareto=False, weighted=True, robustness=True, artifacts=(ArtifactKind.ROBUSTNESS_SUMMARY,)),
    )
}

METRIC_REGISTRY: Mapping[str, MetricDefinition] = MappingProxyType(_REGISTRY)


def metric_registry() -> Mapping[str, MetricDefinition]:
    return METRIC_REGISTRY


def _direction_text(value: Any) -> str:
    return value.value if isinstance(value, MetricDirection) else str(value)


def validate_metric_value(metric_key: str, value: Any, *, field: str) -> float | None:
    definition = METRIC_REGISTRY.get(metric_key)
    if definition is None:
        raise MetricValueRepresentationError(f"{field}: unknown metric key '{metric_key}'")
    if value is None:
        return None
    if isinstance(value, bool):
        raise MetricValueRepresentationError(
            f"{field}: metric '{metric_key}' requires numeric representation; "
            "Boolean values are not accepted"
        )
    if not isinstance(value, Real):
        raise MetricValueRepresentationError(
            f"{field}: metric '{metric_key}' must be a finite numeric value"
        )
    number = float(value)
    if not math.isfinite(number):
        raise MetricValueRepresentationError(
            f"{field}: metric '{metric_key}' must be a finite numeric value"
        )
    if (
        definition.numeric_representation == "binary_numeric"
        and number not in definition.allowed_numeric_values
    ):
        allowed = ", ".join(f"{item:.1f}" for item in definition.allowed_numeric_values)
        raise MetricValueRepresentationError(
            f"{field}: binary metric '{metric_key}' must use one of [{allowed}]"
        )
    return number


def validate_metric_artifact(metrics: Mapping[str, Any], *, field: str) -> None:
    for metric_key, value in sorted(metrics.items()):
        validate_metric_value(metric_key, value, field=f"{field}.{metric_key}")


def validate_evaluation_profile(profile: "EvaluationProfile") -> None:
    """Fail closed on any profile/registry mismatch with precise field paths."""

    errors: list[str] = []
    enabled = set(profile.enabled_metrics)
    for index, metric_key in enumerate(profile.enabled_metrics):
        if metric_key not in METRIC_REGISTRY:
            errors.append(f"enabled_metrics[{index}]: unknown metric key '{metric_key}'")

    for field_name, configured in (
        ("metric_directions", profile.metric_directions),
        ("metric_modes", profile.metric_modes),
    ):
        for metric_key in sorted(enabled - set(configured)):
            errors.append(f"{field_name}.{metric_key}: enabled metric is missing")
        for metric_key in sorted(set(configured) - enabled):
            errors.append(f"{field_name}.{metric_key}: metric is not enabled")
        for metric_key in sorted(set(configured) - set(METRIC_REGISTRY)):
            errors.append(f"{field_name}.{metric_key}: unknown metric key '{metric_key}'")

    for metric_key in sorted(enabled & set(profile.metric_directions) & set(METRIC_REGISTRY)):
        configured_direction = profile.metric_directions[metric_key]
        registry_direction = METRIC_REGISTRY[metric_key].direction
        if configured_direction != registry_direction:
            errors.append(
                f"metric_directions.{metric_key}: direction "
                f"'{_direction_text(configured_direction)}' does not match registry direction "
                f"'{registry_direction.value}'"
            )

    def validate_rules(
        field_name: str,
        rules: Any,
        suitability_field: str | None,
        *,
        direction_field: bool = False,
        validate_threshold: bool = False,
    ) -> None:
        seen: set[str] = set()
        for index, rule in enumerate(rules):
            metric_key = rule.metric_key
            path = f"{field_name}[{index}].metric_key"
            if metric_key in seen:
                errors.append(f"{path}: duplicate rule for metric '{metric_key}'")
            seen.add(metric_key)
            if metric_key not in enabled:
                errors.append(f"{path}: metric '{metric_key}' is not enabled")
                continue
            definition = METRIC_REGISTRY.get(metric_key)
            if definition is None:
                errors.append(f"{path}: unknown metric key '{metric_key}'")
                continue
            if suitability_field is not None and not getattr(definition, suitability_field):
                errors.append(
                    f"{path}: metric '{metric_key}' is not suitable for {field_name}"
                )
            if direction_field and rule.direction != definition.direction:
                errors.append(
                    f"{field_name}[{index}].direction: metric '{metric_key}' direction "
                    f"'{_direction_text(rule.direction)}' does not match registry direction "
                    f"'{definition.direction.value}'"
                )
            if validate_threshold:
                try:
                    validate_metric_value(
                        metric_key, rule.threshold, field=f"{field_name}[{index}].threshold"
                    )
                except MetricValueRepresentationError as error:
                    errors.append(str(error))

    validate_rules(
        "mandatory_gates",
        profile.mandatory_gates,
        "suitable_for_gates",
        validate_threshold=True,
    )
    validate_rules(
        "pareto_objectives",
        profile.pareto_objectives,
        "suitable_for_pareto",
        direction_field=True,
    )
    validate_rules(
        "robustness_vetoes",
        profile.robustness_vetoes,
        "suitable_for_robustness",
        validate_threshold=True,
    )
    validate_rules(
        "lexicographic_tie_break",
        profile.lexicographic_tie_break,
        None,
        direction_field=True,
    )

    for metric_key in sorted(profile.exploratory_metric_weights):
        path = f"exploratory_metric_weights.{metric_key}"
        if metric_key not in enabled:
            errors.append(f"{path}: metric '{metric_key}' is not enabled")
            continue
        definition = METRIC_REGISTRY.get(metric_key)
        if definition is None:
            errors.append(f"{path}: unknown metric key '{metric_key}'")
        elif not definition.suitable_for_weighted_view:
            errors.append(f"{path}: metric '{metric_key}' is not suitable for weighted comparison")

    if errors:
        raise EvaluationProfileValidationError(
            f"invalid EvaluationProfile '{profile.name}': " + "; ".join(errors)
        )


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
