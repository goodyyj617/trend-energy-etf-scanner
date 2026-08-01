"""Deterministic metrics and reusable artifacts calculated from stored curves."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.portfolio import summarize_portfolio_curve

from .artifact_schemas import (
    DAILY_PORTFOLIO_CURVE_SCHEMA_VERSION,
    ROLLING_METRICS_SCHEMA_VERSION,
    YEARLY_METRICS_SCHEMA_VERSION,
    validate_daily_portfolio_curve,
    validate_robustness_summary,
    validate_rolling_metrics,
    validate_yearly_metrics,
)
from .canonical import canonical_data


METRIC_CALCULATION_ENGINE_VERSION = "trend_v2_stored_curve_metric_engine_v1"
METRIC_DEFINITION_VERSION = "legacy_portfolio_metric_parity_v1"
DERIVED_METRICS_SCHEMA_VERSION = "derived_metrics_v1"
DEFAULT_ROLLING_WINDOWS = (63, 252, 756)
DEFAULT_MIN_COMMON_OBSERVATIONS = 2


def calculation_settings(
    *,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    min_common_observations: int = DEFAULT_MIN_COMMON_OBSERVATIONS,
) -> dict[str, Any]:
    windows = tuple(sorted(set(int(value) for value in rolling_windows)))
    if not windows or any(value < 2 for value in windows):
        raise ValueError("rolling windows must contain unique values >= 2")
    if min_common_observations < 2:
        raise ValueError("min_common_observations must be >= 2")
    return {
        "rolling_windows": list(windows),
        "rolling_min_observations": "full_window",
        "trading_sessions_per_year": 252,
        "calendar_year_days": 365.25,
        "min_common_observations": int(min_common_observations),
    }


def _initial_value(rows: Sequence[Mapping[str, Any]], value_field: str, return_field: str) -> float:
    first = rows[0]
    first_return = float(first[return_field])
    return float(first[value_field]) / (1.0 + first_return)


def _legacy_curve(
    payload: Mapping[str, Any],
    *,
    value_field: str = "portfolio_value",
    return_field: str = "daily_return",
) -> tuple[pd.DataFrame, float]:
    rows = payload["rows"]
    if not rows:
        return pd.DataFrame(), 1.0
    initial = _initial_value(rows, value_field, return_field)
    frame = pd.DataFrame(
        {
            "date": [row["economic_date"] for row in rows],
            "observation_type": "trading_session",
            "strategy_key": "stored_strategy",
            "portfolio_equity": [float(row[value_field]) for row in rows],
            "gross_exposure": [float(row["gross_exposure"]) for row in rows],
            "active_position_count": [float(row.get("position_count", 0.0)) for row in rows],
            "daily_portfolio_return": [float(row[return_field]) for row in rows],
            "transaction_cost_paid": [float(row["transaction_cost"]) for row in rows],
            "turnover": [float(row["daily_turnover"]) for row in rows],
        }
    )
    equity = frame["portfolio_equity"].astype(float)
    frame["cash_value"] = [float(row["cash_weight"]) * value for row, value in zip(rows, equity)]
    frame["invested_value"] = equity - frame["cash_value"]
    frame["cumulative_return"] = equity / initial - 1.0
    frame["running_peak_equity"] = equity.cummax().clip(lower=initial)
    frame["drawdown"] = equity / frame["running_peak_equity"] - 1.0
    return frame, initial


def _summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    frame, initial = _legacy_curve(payload)
    return summarize_portfolio_curve(
        "stored_strategy", "Stored strategy", frame, initial_capital=initial
    )


def _window_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))
    drawdowns = np.concatenate(([1.0], wealth)) / peaks - 1.0
    return float(drawdowns.min())


def generate_yearly_metrics(daily_curve: Mapping[str, Any]) -> dict[str, Any]:
    validate_daily_portfolio_curve(daily_curve)
    rows = daily_curve["rows"]
    if not rows:
        payload = {
            "schema_version": YEARLY_METRICS_SCHEMA_VERSION,
            "source_schema_version": DAILY_PORTFOLIO_CURVE_SCHEMA_VERSION,
            "source_economic_date_range": None,
            "rows": [],
        }
        validate_yearly_metrics(payload)
        return payload
    years = sorted({int(row["economic_date"][:4]) for row in rows})
    first_year, last_year = years[0], years[-1]
    generated: list[dict[str, Any]] = []
    for year in years:
        selected = [row for row in rows if int(row["economic_date"][:4]) == year]
        returns = np.asarray([float(row["daily_return"]) for row in selected], dtype=float)
        volatility = float(np.std(returns, ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.0
        generated.append(
            {
                "calendar_year": year,
                "complete_year": bool(first_year < year < last_year),
                "start_economic_date": selected[0]["economic_date"],
                "end_economic_date": selected[-1]["economic_date"],
                "annual_return": float(np.prod(1.0 + returns) - 1.0),
                "annualized_volatility": volatility,
                "maximum_drawdown": _window_drawdown(returns),
                "turnover": float(sum(float(row["daily_turnover"]) for row in selected)),
                "observation_count": len(selected),
            }
        )
    payload = {
        "schema_version": YEARLY_METRICS_SCHEMA_VERSION,
        "source_schema_version": daily_curve["schema_version"],
        "source_economic_date_range": daily_curve["economic_date_range"],
        "complete_year_rule": "only years strictly between first and last stored years",
        "rows": generated,
    }
    validate_yearly_metrics(payload)
    return payload


def generate_rolling_metrics(
    daily_curve: Mapping[str, Any], windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS
) -> dict[str, Any]:
    validate_daily_portfolio_curve(daily_curve)
    configured = sorted(set(int(value) for value in windows))
    if not configured or any(value < 2 for value in configured):
        raise ValueError("rolling windows must contain unique values >= 2")
    source_rows = daily_curve["rows"]
    generated: list[dict[str, Any]] = []
    all_returns = np.asarray([float(row["daily_return"]) for row in source_rows], dtype=float)
    for window in configured:
        for end_index in range(window - 1, len(source_rows)):
            selected = all_returns[end_index - window + 1 : end_index + 1]
            rolling_return = float(np.prod(1.0 + selected) - 1.0)
            annualized_return = (
                float((1.0 + rolling_return) ** (252.0 / window) - 1.0)
                if rolling_return > -1.0
                else None
            )
            sample_std = float(np.std(selected, ddof=1))
            generated.append(
                {
                    "economic_date": source_rows[end_index]["economic_date"],
                    "window_sessions": window,
                    "rolling_return": rolling_return,
                    "rolling_annualized_return": annualized_return,
                    "rolling_volatility": sample_std * math.sqrt(252.0),
                    "rolling_sharpe": (
                        float(np.mean(selected) / sample_std * math.sqrt(252.0))
                        if sample_std > 0
                        else None
                    ),
                    "rolling_maximum_drawdown": _window_drawdown(selected),
                    "observation_count": window,
                }
            )
    payload = {
        "schema_version": ROLLING_METRICS_SCHEMA_VERSION,
        "source_schema_version": daily_curve["schema_version"],
        "source_economic_date_range": daily_curve["economic_date_range"],
        "configured_windows": configured,
        "minimum_observation_rule": "full_window",
        "rows": generated,
    }
    validate_rolling_metrics(payload)
    return payload


def _current_time_under_water_days(payload: Mapping[str, Any]) -> int:
    rows = payload["rows"]
    if not rows:
        return 0
    initial = _initial_value(rows, "portfolio_value", "daily_return")
    peak = initial
    episode_start: pd.Timestamp | None = None
    for row in rows:
        current_date = pd.Timestamp(row["economic_date"])
        value = float(row["portfolio_value"])
        if value >= peak - 1e-9:
            peak = max(peak, value)
            episode_start = None
        elif episode_start is None:
            episode_start = current_date
    if episode_start is None:
        return 0
    return int((pd.Timestamp(rows[-1]["economic_date"]) - episode_start).days)


def _worst_rolling(rolling: Mapping[str, Any], window: int, field: str) -> float | None:
    values = [
        row[field]
        for row in rolling["rows"]
        if row["window_sessions"] == window and row[field] is not None
    ]
    return float(min(values)) if values else None


def _availability(values: dict[str, Any], reasons: dict[str, str], key: str, value: Any, reason: str) -> None:
    if value is None:
        values[key] = None
        reasons[key] = reason
    else:
        values[key] = canonical_data(value)


def calculate_absolute_metrics(
    daily_curve: Mapping[str, Any], rolling_metrics: Mapping[str, Any]
) -> tuple[dict[str, float | int | None], dict[str, str]]:
    validate_daily_portfolio_curve(daily_curve)
    validate_rolling_metrics(rolling_metrics)
    values: dict[str, float | int | None] = {}
    reasons: dict[str, str] = {}
    if not daily_curve["rows"]:
        for key in (
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "maximum_drawdown",
            "cdar95",
            "calmar_ratio",
            "recovery_duration_days",
            "longest_recovery_duration_days",
            "current_time_under_water_days",
            "annual_turnover",
            "average_gross_exposure",
            "average_cash_weight",
        ):
            _availability(values, reasons, key, None, "daily_portfolio_curve_has_no_rows")
        return values, reasons
    summary = _summary(daily_curve)
    source_map = {
        "cagr": "cagr",
        "annualized_volatility": "annualized_volatility",
        "sharpe_ratio": "sharpe_ratio",
        "sortino_ratio": "sortino_ratio",
        "maximum_drawdown": "maximum_drawdown",
        "cdar95": "conditional_drawdown_at_risk_95",
        "calmar_ratio": "calmar_ratio",
        "recovery_duration_days": "max_drawdown_duration_days",
        "longest_recovery_duration_days": "longest_time_under_water_days",
        "annual_turnover": "annual_turnover",
        "average_gross_exposure": "average_gross_exposure",
    }
    for key, source in source_map.items():
        _availability(values, reasons, key, summary[source], f"{source}_undefined_for_stored_curve")
    values["current_time_under_water_days"] = _current_time_under_water_days(daily_curve)
    values["average_cash_weight"] = float(
        np.mean([float(row["cash_weight"]) for row in daily_curve["rows"]])
    )
    _availability(
        values,
        reasons,
        "worst_rolling_3_month_return",
        _worst_rolling(rolling_metrics, 63, "rolling_return"),
        "rolling_metrics_window_63_unavailable",
    )
    _availability(
        values,
        reasons,
        "worst_rolling_12_month_return",
        _worst_rolling(rolling_metrics, 252, "rolling_return"),
        "rolling_metrics_window_252_unavailable",
    )
    _availability(
        values,
        reasons,
        "worst_rolling_36_month_annualized_return",
        _worst_rolling(rolling_metrics, 756, "rolling_annualized_return"),
        "rolling_metrics_window_756_unavailable",
    )
    if all("position_count" in row for row in daily_curve["rows"]):
        counts = [float(row["position_count"]) for row in daily_curve["rows"]]
        values["average_position_count"] = float(np.mean(counts))
        values["maximum_position_count"] = float(max(counts))
    else:
        _availability(values, reasons, "average_position_count", None, "daily_portfolio_curve.position_count_missing")
        _availability(values, reasons, "maximum_position_count", None, "daily_portfolio_curve.position_count_missing")
    if all(
        "gross_portfolio_value" in row and "gross_daily_return" in row
        for row in daily_curve["rows"]
    ):
        gross_payload = {
            **daily_curve,
            "rows": [
                {
                    **row,
                    "portfolio_value": row["gross_portfolio_value"],
                    "daily_return": row["gross_daily_return"],
                }
                for row in daily_curve["rows"]
            ],
        }
        gross_cagr = _summary(gross_payload)["cagr"]
        values["transaction_cost_drag"] = float(gross_cagr - values["cagr"])
    else:
        _availability(
            values,
            reasons,
            "transaction_cost_drag",
            None,
            "gross_portfolio_value_and_gross_daily_return_missing",
        )
    return values, reasons


def exact_common_date_curves(
    strategy_curve: Mapping[str, Any],
    benchmark_curve: Mapping[str, Any],
    *,
    min_common_observations: int = DEFAULT_MIN_COMMON_OBSERVATIONS,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    validate_daily_portfolio_curve(strategy_curve)
    validate_daily_portfolio_curve(benchmark_curve)
    strategy_dates = {row["economic_date"] for row in strategy_curve["rows"]}
    benchmark_dates = {row["economic_date"] for row in benchmark_curve["rows"]}
    common = sorted(strategy_dates & benchmark_dates)
    alignment = {
        "strategy_input_date_range": strategy_curve["economic_date_range"],
        "benchmark_input_date_range": benchmark_curve["economic_date_range"],
        "common_retained_date_range": (
            {"start": common[0], "end": common[-1]} if common else None
        ),
        "dropped_strategy_dates": sorted(strategy_dates - benchmark_dates),
        "dropped_benchmark_dates": sorted(benchmark_dates - strategy_dates),
        "common_observation_count": len(common),
        "minimum_common_observations": min_common_observations,
        "status": "available" if len(common) >= min_common_observations else "insufficient_common_date_coverage",
    }
    if len(common) < min_common_observations:
        return None, None, alignment

    def retain(payload: Mapping[str, Any]) -> dict[str, Any]:
        retained = [row for row in payload["rows"] if row["economic_date"] in set(common)]
        return {
            **payload,
            "economic_date_range": {"start": common[0], "end": common[-1]},
            "rows": retained,
        }

    return retain(strategy_curve), retain(benchmark_curve), alignment


def _ratio(numerator: Any, denominator: Any, *, magnitude: bool = False) -> float | None:
    if numerator is None or denominator is None:
        return None
    left = abs(float(numerator)) if magnitude else float(numerator)
    right = abs(float(denominator)) if magnitude else float(denominator)
    if right == 0:
        return None
    return left / right


def calculate_benchmark_relative_metrics(
    strategy_curve: Mapping[str, Any],
    benchmark_curve: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
) -> tuple[dict[str, float | None], dict[str, str], dict[str, Any]]:
    strategy_common, benchmark_common, alignment = exact_common_date_curves(
        strategy_curve,
        benchmark_curve,
        min_common_observations=int(settings["min_common_observations"]),
    )
    keys = (
        "cagr_spy_ratio",
        "maximum_drawdown_spy_ratio",
        "cdar95_spy_ratio",
        "calmar_spy_ratio",
        "recovery_duration_spy_ratio",
        "worst_rolling_3_month_spy_ratio",
        "worst_rolling_12_month_spy_ratio",
        "worst_rolling_36_month_spy_ratio",
    )
    if strategy_common is None or benchmark_common is None:
        reason = (
            "benchmark_common_date_coverage_insufficient:"
            f"{alignment['common_observation_count']}<"
            f"{alignment['minimum_common_observations']}"
        )
        return {key: None for key in keys}, {key: reason for key in keys}, alignment
    windows = settings["rolling_windows"]
    strategy_rolling = generate_rolling_metrics(strategy_common, windows)
    benchmark_rolling = generate_rolling_metrics(benchmark_common, windows)
    strategy_values, _ = calculate_absolute_metrics(strategy_common, strategy_rolling)
    benchmark_values, _ = calculate_absolute_metrics(benchmark_common, benchmark_rolling)
    ratios = {
        "cagr_spy_ratio": _ratio(strategy_values["cagr"], benchmark_values["cagr"]),
        "maximum_drawdown_spy_ratio": _ratio(
            strategy_values["maximum_drawdown"], benchmark_values["maximum_drawdown"], magnitude=True
        ),
        "cdar95_spy_ratio": _ratio(strategy_values["cdar95"], benchmark_values["cdar95"], magnitude=True),
        "calmar_spy_ratio": _ratio(strategy_values["calmar_ratio"], benchmark_values["calmar_ratio"]),
        "recovery_duration_spy_ratio": _ratio(
            strategy_values["recovery_duration_days"], benchmark_values["recovery_duration_days"]
        ),
        "worst_rolling_3_month_spy_ratio": _ratio(
            strategy_values["worst_rolling_3_month_return"],
            benchmark_values["worst_rolling_3_month_return"],
            magnitude=True,
        ),
        "worst_rolling_12_month_spy_ratio": _ratio(
            strategy_values["worst_rolling_12_month_return"],
            benchmark_values["worst_rolling_12_month_return"],
            magnitude=True,
        ),
        "worst_rolling_36_month_spy_ratio": _ratio(
            strategy_values["worst_rolling_36_month_annualized_return"],
            benchmark_values["worst_rolling_36_month_annualized_return"],
            magnitude=True,
        ),
    }
    reasons = {
        key: "benchmark_denominator_zero_or_required_rolling_window_unavailable"
        for key, value in ratios.items()
        if value is None
    }
    return ratios, reasons, alignment


def robustness_metrics(
    robustness_summary: Mapping[str, Any] | None,
) -> tuple[dict[str, float | int | None], dict[str, str], dict[str, Any]]:
    mappings = {
        "walk_forward_pass_ratio": "walk_forward_pass_ratio",
        "walk_forward_worst_fold": "walk_forward_worst_fold",
        "loyo_stability_ratio": "loyo_stability_ratio",
        "block_bootstrap_effect": "block_bootstrap_effect",
        "adjusted_p_value": "adjusted_p_value",
        "transaction_cost_stress_survival": "transaction_cost_stress_survival",
        "dominant_group_share": "dominant_group_share",
        "unclassified_group_share": "unclassified_group_share",
    }
    result_keys = tuple(mappings) + (
        "loyo_reversing_years",
        "block_bootstrap_ci_lower",
        "block_bootstrap_ci_upper",
    )
    if robustness_summary is None:
        return (
            {key: None for key in result_keys},
            {key: "robustness_summary_artifact_missing" for key in result_keys},
            {"status": "not_available", "reason": "robustness_summary_artifact_missing"},
        )
    validate_robustness_summary(robustness_summary)
    values: dict[str, float | int | None] = {}
    reasons: dict[str, str] = {}
    source_reasons = robustness_summary["unavailable_reasons"]
    for metric_key, source_key in mappings.items():
        value = robustness_summary[source_key]
        _availability(values, reasons, metric_key, value, source_reasons.get(source_key, f"{source_key}_unavailable"))
    reversing = robustness_summary["loyo_reversing_years"]
    _availability(
        values,
        reasons,
        "loyo_reversing_years",
        len(reversing) if reversing is not None else None,
        source_reasons.get("loyo_reversing_years", "loyo_reversing_years_unavailable"),
    )
    interval = robustness_summary["bootstrap_confidence_interval"]
    for metric_key, field in (
        ("block_bootstrap_ci_lower", "lower"),
        ("block_bootstrap_ci_upper", "upper"),
    ):
        _availability(
            values,
            reasons,
            metric_key,
            interval[field] if interval is not None else None,
            source_reasons.get("bootstrap_confidence_interval", "bootstrap_confidence_interval_unavailable"),
        )
    provenance = {
        "status": "available",
        "walk_forward_fold_count": robustness_summary["walk_forward_fold_count"],
        "loyo_case_count": robustness_summary["loyo_case_count"],
        "multiple_testing_method": robustness_summary["multiple_testing_method"],
        "raw_p_value": robustness_summary["raw_p_value"],
        "dominant_asset_group": robustness_summary["dominant_asset_group"],
        "method_metadata": robustness_summary["method_metadata"],
    }
    return values, reasons, provenance


def calculate_metric_artifact(
    *,
    strategy_run_id: str,
    daily_curve: Mapping[str, Any],
    benchmark_curve: Mapping[str, Any],
    benchmark_identity: str,
    benchmark_artifact_hash: str,
    source_artifact_hashes: Mapping[str, str],
    robustness_summary: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    effective_settings = dict(settings or calculation_settings())
    yearly = generate_yearly_metrics(daily_curve)
    rolling = generate_rolling_metrics(daily_curve, effective_settings["rolling_windows"])
    values, reasons = calculate_absolute_metrics(daily_curve, rolling)
    relative, relative_reasons, alignment = calculate_benchmark_relative_metrics(
        daily_curve, benchmark_curve, settings=effective_settings
    )
    robustness, robustness_reasons, robustness_provenance = robustness_metrics(robustness_summary)
    values.update(relative)
    values.update(robustness)
    reasons.update(relative_reasons)
    reasons.update(robustness_reasons)
    artifact = {
        "schema_version": DERIVED_METRICS_SCHEMA_VERSION,
        "metric_calculation_engine_version": METRIC_CALCULATION_ENGINE_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "strategy_run_id": strategy_run_id,
        "benchmark_identity": benchmark_identity,
        "benchmark_artifact_hash": benchmark_artifact_hash,
        "source_artifact_hashes": dict(sorted(source_artifact_hashes.items())),
        "calculation_settings": canonical_data(effective_settings),
        "values": dict(sorted(values.items())),
        "unavailable_reasons": dict(sorted(reasons.items())),
        "benchmark_alignment": alignment,
        "robustness_provenance": robustness_provenance,
    }
    return artifact, yearly, rolling
