"""Deterministic constraint/Pareto evaluation with a separate weighted view."""

from __future__ import annotations

import math
import statistics
from functools import cmp_to_key
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .contracts import (
    CandidateEvaluation,
    CheckResult,
    ComparisonMode,
    EvaluationProfile,
    EvaluationRun,
    GateRule,
    MetricDirection,
    NormalizationMethod,
    ParetoObjective,
    SortRule,
    WeightedCandidateView,
)
from .metrics import validate_metric_artifact, validate_metric_value

if TYPE_CHECKING:
    from .result_store import ResultStore


METRIC_ENGINE_VERSION = "trend_v2_metric_selection_contract_v1"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_gate(
    metrics: Mapping[str, Any],
    rule: GateRule,
    unavailable_reasons: Mapping[str, str] | None = None,
) -> CheckResult:
    value = validate_metric_value(
        rule.metric_key,
        metrics.get(rule.metric_key),
        field=f"metrics.{rule.metric_key}",
    )
    if value is None:
        return CheckResult(
            metric_key=rule.metric_key,
            operator=rule.operator,
            threshold=rule.threshold,
            value=None,
            passed=False,
            reason=(unavailable_reasons or {}).get(
                rule.metric_key, "metric_missing_or_non_finite"
            ),
        )
    operators = {
        ">=": lambda left, right: left >= right,
        ">": lambda left, right: left > right,
        "<=": lambda left, right: left <= right,
        "<": lambda left, right: left < right,
        "==": lambda left, right: left == right,
        "!=": lambda left, right: left != right,
    }
    passed = bool(operators[rule.operator](value, rule.threshold))
    return CheckResult(
        metric_key=rule.metric_key,
        operator=rule.operator,
        threshold=rule.threshold,
        value=value,
        passed=passed,
        reason="passed" if passed else "threshold_failed",
    )


def epsilon_pareto(
    metrics_by_run: Mapping[str, Mapping[str, Any]],
    objectives: Sequence[ParetoObjective],
) -> tuple[set[str], dict[str, tuple[str, ...]]]:
    """Return epsilon-Pareto members and their deterministic dominator lists."""

    keys = sorted(metrics_by_run)
    dominated_by: dict[str, tuple[str, ...]] = {}
    members: set[str] = set()
    for candidate_key in keys:
        candidate = metrics_by_run[candidate_key]
        dominators: list[str] = []
        for other_key in keys:
            if other_key == candidate_key:
                continue
            other = metrics_by_run[other_key]
            no_worse = True
            materially_better = False
            for objective in objectives:
                candidate_value = _finite(candidate.get(objective.metric_key))
                other_value = _finite(other.get(objective.metric_key))
                if candidate_value is None or other_value is None:
                    raise ValueError(f"non-finite Pareto metric: {objective.metric_key}")
                if objective.direction == MetricDirection.MAXIMIZE:
                    if other_value < candidate_value - objective.epsilon:
                        no_worse = False
                        break
                    materially_better |= other_value > candidate_value + objective.epsilon
                else:
                    if other_value > candidate_value + objective.epsilon:
                        no_worse = False
                        break
                    materially_better |= other_value < candidate_value - objective.epsilon
            if no_worse and materially_better:
                dominators.append(other_key)
        dominated_by[candidate_key] = tuple(dominators)
        if not dominators:
            members.add(candidate_key)
    return members, dominated_by


def _lexicographic_order(
    keys: Sequence[str], metrics_by_run: Mapping[str, Mapping[str, Any]], rules: Sequence[SortRule]
) -> list[str]:
    def compare(left_key: str, right_key: str) -> int:
        for rule in rules:
            left = _finite(metrics_by_run[left_key].get(rule.metric_key))
            right = _finite(metrics_by_run[right_key].get(rule.metric_key))
            if left is None and right is None:
                continue
            if left is None:
                return 1
            if right is None:
                return -1
            if left == right:
                continue
            if rule.direction == MetricDirection.MAXIMIZE:
                return -1 if left > right else 1
            return -1 if left < right else 1
        return -1 if left_key < right_key else (1 if left_key > right_key else 0)

    return sorted(keys, key=cmp_to_key(compare))


def _normalize_metric(
    values: Mapping[str, float | None],
    direction: MetricDirection,
    method: NormalizationMethod,
) -> dict[str, float | None]:
    finite = [value for value in values.values() if value is not None]
    if not finite:
        return {key: None for key in values}
    if method == NormalizationMethod.MIN_MAX:
        low, high = min(finite), max(finite)
        if high == low:
            return {key: (1.0 if value is not None else None) for key, value in values.items()}
        normalized = {
            key: ((value - low) / (high - low) if value is not None else None)
            for key, value in values.items()
        }
        if direction == MetricDirection.MINIMIZE:
            normalized = {
                key: (1.0 - value if value is not None else None)
                for key, value in normalized.items()
            }
        return normalized
    mean = statistics.fmean(finite)
    standard_deviation = statistics.pstdev(finite)
    if standard_deviation == 0:
        return {key: (0.0 if value is not None else None) for key, value in values.items()}
    sign = 1.0 if direction == MetricDirection.MAXIMIZE else -1.0
    return {
        key: (sign * (value - mean) / standard_deviation if value is not None else None)
        for key, value in values.items()
    }


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("weighted comparison requires positive total weight")
    return {key: float(weights[key]) / total for key in sorted(weights)}


def _rank_weighted_values(values: Mapping[str, float | None]) -> dict[str, int | None]:
    valid = sorted(
        ((key, value) for key, value in values.items() if value is not None),
        key=lambda item: (-float(item[1]), item[0]),
    )
    ranks = {key: None for key in values}
    for rank, (key, _) in enumerate(valid, start=1):
        ranks[key] = rank
    return ranks


def _sensitivity_weight_view(
    normalized_weights: Mapping[str, float], metric_key: str, adjustment: float
) -> dict[str, float]:
    """Adjust one normalized share and redistribute the balance proportionally."""

    baseline = {key: float(value) for key, value in sorted(normalized_weights.items())}
    other_positive = [
        key for key, value in baseline.items() if key != metric_key and value > 0.0
    ]
    if not other_positive:
        return baseline
    selected = min(1.0, max(0.0, baseline[metric_key] + adjustment))
    other_total = sum(baseline[key] for key in other_positive)
    remaining = 1.0 - selected
    result = {key: 0.0 for key in baseline}
    result[metric_key] = selected
    for key in other_positive:
        result[key] = remaining * baseline[key] / other_total
    correction_key = other_positive[-1]
    result[correction_key] += 1.0 - sum(result.values())
    return result


def _weighted_outputs(
    profile: EvaluationProfile,
    metrics_by_run: Mapping[str, Mapping[str, Any]],
    gates_passed: Mapping[str, bool],
) -> tuple[dict[str, float], dict[str, WeightedCandidateView], dict[str, Any]]:
    weights = {key: float(value) for key, value in profile.exploratory_metric_weights.items()}
    normalized_weights = _normalize_weights(weights)
    if profile.normalization_method is None:
        raise ValueError("normalization method is required for weighted output")
    normalized_values: dict[str, dict[str, float | None]] = {key: {} for key in metrics_by_run}
    for metric_key in sorted(weights):
        values = {key: _finite(metrics.get(metric_key)) for key, metrics in metrics_by_run.items()}
        direction = profile.metric_directions[metric_key]
        metric_normalized = _normalize_metric(values, direction, profile.normalization_method)
        for key, value in metric_normalized.items():
            normalized_values[key][metric_key] = value

    def weighted_values_for(weight_view: Mapping[str, float]) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for key, metric_values in normalized_values.items():
            required = [metric for metric, weight in weight_view.items() if weight > 0]
            if any(metric_values.get(metric) is None for metric in required):
                result[key] = None
            else:
                result[key] = sum(
                    float(metric_values[metric]) * weight_view[metric] for metric in required
                )
        return result

    baseline_values = weighted_values_for(normalized_weights)
    baseline_ranks = _rank_weighted_values(baseline_values)
    scenarios: dict[str, dict[str, Any]] = {
        "baseline": {"weights": normalized_weights, "ranks": baseline_ranks}
    }
    delta = profile.ranking_sensitivity_delta
    for metric_key in sorted(weights):
        for suffix, adjustment in (("plus", delta), ("minus", -delta)):
            scenario_weights = _sensitivity_weight_view(
                normalized_weights, metric_key, adjustment
            )
            scenario = f"{metric_key}:{suffix}_{delta:g}"
            scenarios[scenario] = {
                "weights": scenario_weights,
                "ranks": _rank_weighted_values(weighted_values_for(scenario_weights)),
            }

    views: dict[str, WeightedCandidateView] = {}
    for key in sorted(metrics_by_run):
        contributions = {
            metric: (
                None
                if normalized_values[key].get(metric) is None
                else float(normalized_values[key][metric]) * normalized_weights[metric]
            )
            for metric in sorted(weights)
        }
        warnings: list[str] = []
        rank = baseline_ranks[key]
        if rank is not None and rank <= profile.high_weighted_rank_cutoff and not gates_passed[key]:
            warnings.append("high_weighted_rank_but_mandatory_gate_failed")
        views[key] = WeightedCandidateView(
            exploratory_weighted_value=baseline_values[key],
            rank=rank,
            normalized_metric_values=normalized_values[key],
            weighted_contributions=contributions,
            sensitivity_ranks={
                scenario: details["ranks"][key] for scenario, details in scenarios.items()
            },
            warnings=tuple(warnings),
        )
    sensitivity = {
        "delta": delta,
        "scenarios": scenarios,
        "normalization_method": profile.normalization_method.value,
    }
    return normalized_weights, views, sensitivity


def evaluate_strategy_runs(
    profile: EvaluationProfile,
    metrics_by_run: Mapping[str, Mapping[str, Any]],
    *,
    benchmark_data_identity: str,
    metric_engine_version: str = METRIC_ENGINE_VERSION,
    behavior_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    unavailable_reasons: Mapping[str, Mapping[str, str]] | None = None,
    derived_metric_ids: Mapping[str, str] | None = None,
    behavior_pairwise_diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    simplicity_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    creation_time: str,
) -> EvaluationRun:
    """Apply the non-compensatory pipeline and optional weighted view."""

    if not metrics_by_run:
        raise ValueError("at least one stored strategy run is required")
    metrics_by_run = {key: dict(value) for key, value in sorted(metrics_by_run.items())}
    for strategy_run_id, metric_values in metrics_by_run.items():
        validate_metric_artifact(metric_values, field=f"raw_metrics[{strategy_run_id}]")
    behavior_metadata = behavior_metadata or {}
    unavailable_reasons = unavailable_reasons or {}
    gate_results = {
        key: tuple(
            evaluate_gate(metrics, rule, unavailable_reasons.get(key, {}))
            for rule in profile.mandatory_gates
        )
        for key, metrics in metrics_by_run.items()
    }
    gates_passed = {key: all(result.passed for result in results) for key, results in gate_results.items()}

    pareto_input: dict[str, Mapping[str, Any]] = {}
    pareto_missing: set[str] = set()
    for key, metrics in metrics_by_run.items():
        if not gates_passed[key]:
            continue
        if any(_finite(metrics.get(rule.metric_key)) is None for rule in profile.pareto_objectives):
            pareto_missing.add(key)
        else:
            pareto_input[key] = metrics
    pareto_members, dominated_by = epsilon_pareto(pareto_input, profile.pareto_objectives)

    robustness_results = {
        key: tuple(
            evaluate_gate(metrics, rule, unavailable_reasons.get(key, {}))
            for rule in profile.robustness_vetoes
        )
        for key, metrics in metrics_by_run.items()
    }
    robustness_passed = {
        key: all(result.passed for result in results) for key, results in robustness_results.items()
    }
    selectable = [
        key
        for key in metrics_by_run
        if gates_passed[key]
        and key in pareto_members
        and robustness_passed[key]
        and not behavior_metadata.get(key, {}).get("duplicated", False)
    ]
    ordered = _lexicographic_order(selectable, metrics_by_run, profile.lexicographic_tie_break)
    lexicographic_orders = {key: rank for rank, key in enumerate(ordered, start=1)}

    normalized_weights: dict[str, float] = {}
    weighted_views: dict[str, WeightedCandidateView] = {}
    sensitivity: dict[str, Any] = {}
    if profile.exploratory_metric_weights:
        normalized_weights, weighted_views, sensitivity = _weighted_outputs(
            profile, metrics_by_run, gates_passed
        )

    candidate_results: list[CandidateEvaluation] = []
    for key, metrics in metrics_by_run.items():
        labels: list[str] = []
        if not gates_passed[key]:
            labels.append("mandatory_gate_failed")
        elif key in pareto_missing:
            labels.append("pareto_metric_missing")
        elif key not in pareto_members:
            labels.append("epsilon_pareto_dominated")
        elif not robustness_passed[key]:
            labels.append("robustness_vetoed")
        elif behavior_metadata.get(key, {}).get("duplicated", False):
            labels.append("behavior_duplicate_non_representative")
        else:
            labels.append("constraint_pareto_selected")
        metadata = dict(behavior_metadata.get(key, {}))
        if metadata:
            metadata.setdefault("status", "available")
        else:
            metadata = {"status": "not_available", "reason": "behavior artifact not supplied"}
        candidate_results.append(
            CandidateEvaluation(
                strategy_run_id=key,
                raw_metrics=metrics,
                mandatory_gate_results=gate_results[key],
                mandatory_gates_passed=gates_passed[key],
                pareto_member=key in pareto_members,
                dominated_by=dominated_by.get(key, ()),
                robustness_results=robustness_results[key],
                robustness_passed=robustness_passed[key],
                lexicographic_order=lexicographic_orders.get(key),
                behavior_deduplication_metadata=metadata,
                weighted_view=weighted_views.get(key),
                unavailable_reasons=unavailable_reasons.get(key, {}),
                final_labels=tuple(labels),
            )
        )
    return EvaluationRun(
        strategy_run_ids=tuple(metrics_by_run),
        evaluation_profile_id=profile.evaluation_profile_id,
        metric_engine_version=metric_engine_version,
        benchmark_data_identity=benchmark_data_identity,
        profile_hash=profile.profile_hash,
        comparison_mode=profile.comparison_mode,
        results=tuple(candidate_results),
        normalized_weights=normalized_weights,
        ranking_sensitivity=sensitivity,
        derived_metric_ids=derived_metric_ids or {},
        creation_time=creation_time,
        behavior_pairwise_diagnostics=behavior_pairwise_diagnostics or {},
        simplicity_metadata=simplicity_metadata or {},
    )


def evaluate_saved_runs(
    store: "ResultStore",
    strategy_run_ids: Sequence[str],
    profile: EvaluationProfile,
    *,
    benchmark_data_identity: str,
    metric_engine_version: str = METRIC_ENGINE_VERSION,
    creation_time: str,
    persist: bool = True,
) -> EvaluationRun:
    """Re-evaluate stored summary metrics without invoking a backtest engine."""

    metrics: dict[str, Mapping[str, Any]] = {}
    behavior: dict[str, Mapping[str, Any]] = {}
    for strategy_run_id in sorted(set(strategy_run_ids)):
        store.get_strategy_run_manifest(strategy_run_id)
        metrics[strategy_run_id] = store.load_artifact_payload(
            strategy_run_id, "summary_metrics"
        )
        try:
            behavior[strategy_run_id] = store.load_artifact_payload(
                strategy_run_id, "behavior_metadata"
            )
        except KeyError:
            pass
    run = evaluate_strategy_runs(
        profile,
        metrics,
        benchmark_data_identity=benchmark_data_identity,
        metric_engine_version=metric_engine_version,
        behavior_metadata=behavior,
        creation_time=creation_time,
    )
    if persist:
        store.save_evaluation_profile(profile)
        store.save_evaluation_run(run)
    return run
