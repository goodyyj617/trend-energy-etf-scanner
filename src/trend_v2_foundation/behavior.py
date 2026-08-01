"""Deterministic behavior fingerprints, diagnostics, and non-destructive clustering."""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_schemas import (
    BEHAVIOR_METADATA_SCHEMA_VERSION,
    validate_behavior_metadata,
    validate_daily_portfolio_curve,
)
from .canonical import content_hash


BEHAVIOR_ENGINE_VERSION = "trend_v2_behavior_fingerprint_v1"


def _trade_rows(payload: Any | None) -> list[Mapping[str, Any]] | None:
    if payload is None:
        return None
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("trade_lifecycles: rows must be a list")
    normalized: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"trade_lifecycles.rows[{index}]: must be a mapping")
        for field in ("entry_date", "exit_date"):
            value = row.get(field)
            if value is None and field == "exit_date":
                continue
            if not isinstance(value, str):
                raise ValueError(f"trade_lifecycles.rows[{index}].{field}: must be an ISO date")
            try:
                date.fromisoformat(value)
            except ValueError as error:
                raise ValueError(
                    f"trade_lifecycles.rows[{index}].{field}: invalid ISO date"
                ) from error
        normalized.append(row)
    return normalized


def generate_behavior_metadata(
    daily_curve: Mapping[str, Any],
    *,
    daily_curve_hash: str,
    trade_lifecycles: Any | None = None,
    trade_lifecycles_hash: str | None = None,
) -> dict[str, Any]:
    validate_daily_portfolio_curve(daily_curve)
    rows = daily_curve["rows"]
    dates = [row["economic_date"] for row in rows]
    returns = [float(row["daily_return"]) for row in rows]
    active_dates = [
        row["economic_date"] for row in rows if abs(float(row["gross_exposure"])) > 1e-12
    ]
    active_exposure_path = [
        (
            row["economic_date"],
            float(row["gross_exposure"]),
            float(row["net_exposure"]),
        )
        for row in rows
    ]
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=float)) if returns else np.asarray([])
    normalized_path = (wealth / wealth[0]).tolist() if len(wealth) and wealth[0] != 0 else wealth.tolist()
    trades = _trade_rows(trade_lifecycles)
    if trades is None:
        entry_dates = None
        exit_dates = None
        lifecycle = None
    else:
        entry_dates = sorted({str(row["entry_date"]) for row in trades})
        exit_dates = sorted({str(row["exit_date"]) for row in trades if row.get("exit_date")})
        lifecycle = sorted(
            (
                str(row.get("symbol", "")),
                str(row["entry_date"]),
                str(row.get("exit_date") or "OPEN"),
            )
            for row in trades
        )
    payload = {
        "schema_version": BEHAVIOR_METADATA_SCHEMA_VERSION,
        "behavior_engine_version": BEHAVIOR_ENGINE_VERSION,
        "daily_return_fingerprint": content_hash(list(zip(dates, returns))),
        "active_exposure_fingerprint": content_hash(active_exposure_path),
        "trade_entry_dates_fingerprint": content_hash(entry_dates) if entry_dates is not None else None,
        "trade_exit_dates_fingerprint": content_hash(exit_dates) if exit_dates is not None else None,
        "symbol_lifecycle_fingerprint": content_hash(lifecycle) if lifecycle is not None else None,
        "comparison_inputs": {
            "economic_dates": dates,
            "daily_returns": returns,
            "normalized_wealth_path": normalized_path,
            "active_dates": active_dates,
            "active_exposure_path": active_exposure_path,
            "entry_dates": entry_dates,
            "exit_dates": exit_dates,
            "symbol_lifecycle_sequence": lifecycle,
        },
        "source_artifact_hashes": {
            "daily_portfolio_curve": daily_curve_hash,
            **(
                {"trade_lifecycles": trade_lifecycles_hash}
                if trade_lifecycles_hash is not None
                else {}
            ),
        },
    }
    validate_behavior_metadata(payload)
    return payload


def _paired_values(
    left: Mapping[str, Any], right: Mapping[str, Any], field: str
) -> tuple[np.ndarray, np.ndarray]:
    left_inputs = left["comparison_inputs"]
    right_inputs = right["comparison_inputs"]
    left_map = dict(zip(left_inputs["economic_dates"], left_inputs[field]))
    right_map = dict(zip(right_inputs["economic_dates"], right_inputs[field]))
    common = sorted(set(left_map) & set(right_map))
    return (
        np.asarray([left_map[value] for value in common], dtype=float),
        np.asarray([right_map[value] for value in common], dtype=float),
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    if np.array_equal(left, right):
        return 1.0
    if float(np.std(left)) == 0 or float(np.std(right)) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _jaccard(left: Sequence[str] | None, right: Sequence[str] | None) -> float | None:
    if left is None or right is None:
        return None
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _normalized_path_distance(left: np.ndarray, right: np.ndarray) -> float | None:
    if not len(left):
        return None
    numerator = float(np.sqrt(np.mean(np.square(left - right))))
    scale = max(
        float(np.sqrt(np.mean(np.square(left)))),
        float(np.sqrt(np.mean(np.square(right)))),
        1e-12,
    )
    return numerator / scale


def behavior_similarity(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, float | int | None]:
    validate_behavior_metadata(left)
    validate_behavior_metadata(right)
    left_returns, right_returns = _paired_values(left, right, "daily_returns")
    left_path, right_path = _paired_values(left, right, "normalized_wealth_path")
    return {
        "common_observation_count": len(left_returns),
        "daily_return_correlation": _correlation(left_returns, right_returns),
        "active_date_jaccard": _jaccard(
            left["comparison_inputs"]["active_dates"],
            right["comparison_inputs"]["active_dates"],
        ),
        "entry_date_jaccard": _jaccard(
            left["comparison_inputs"]["entry_dates"],
            right["comparison_inputs"]["entry_dates"],
        ),
        "exit_date_jaccard": _jaccard(
            left["comparison_inputs"]["exit_dates"],
            right["comparison_inputs"]["exit_dates"],
        ),
        "normalized_path_distance": _normalized_path_distance(left_path, right_path),
    }


def _condition_passes(value: Any, condition: Mapping[str, Any]) -> bool:
    if value is None or isinstance(value, bool):
        return False
    operator = condition["operator"]
    threshold = float(condition["threshold"])
    operators = {
        ">=": lambda left, right: left >= right,
        ">": lambda left, right: left > right,
        "<=": lambda left, right: left <= right,
        "<": lambda left, right: left < right,
    }
    if operator not in operators:
        raise ValueError(f"unsupported behavior condition operator: {operator}")
    return bool(operators[operator](float(value), threshold))


def _default_conditions(configuration: Mapping[str, Any]) -> list[dict[str, Any]]:
    tolerance = float(configuration.get("tolerance", 1e-8))
    return [
        {"metric_key": "daily_return_correlation", "operator": ">=", "threshold": 1.0 - tolerance},
        {"metric_key": "active_date_jaccard", "operator": ">=", "threshold": 1.0 - tolerance},
        {"metric_key": "normalized_path_distance", "operator": "<=", "threshold": tolerance},
    ]


def deduplicate_behaviors(
    metadata_by_run: Mapping[str, Mapping[str, Any]],
    configuration: Mapping[str, Any],
    *,
    simplicity_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Assign clusters while preserving every run and all pairwise diagnostics."""

    keys = sorted(metadata_by_run)
    if not configuration.get("enabled", True):
        return (
            {
                key: {
                    "status": "available",
                    "behavior_cluster_id": f"behavior_cluster_{content_hash([key])}",
                    "representative_strategy_run_id": key,
                    "is_representative": True,
                    "duplicated": False,
                    "reason": "behavior_deduplication_disabled",
                }
                for key in keys
            },
            {},
        )
    conditions = list(configuration.get("required_conditions") or _default_conditions(configuration))
    pairwise: dict[str, dict[str, Any]] = {}
    duplicate_pairs: set[frozenset[str]] = set()
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            diagnostics = behavior_similarity(metadata_by_run[left], metadata_by_run[right])
            condition_results = {
                condition["metric_key"]: _condition_passes(
                    diagnostics.get(condition["metric_key"]), condition
                )
                for condition in conditions
            }
            duplicated = all(condition_results.values())
            pair_key = f"{left}|{right}"
            pairwise[pair_key] = {
                "left_strategy_run_id": left,
                "right_strategy_run_id": right,
                "diagnostics": diagnostics,
                "condition_results": condition_results,
                "duplicated": duplicated,
            }
            if duplicated:
                duplicate_pairs.add(frozenset((left, right)))

    groups: list[list[str]] = []
    for key in keys:
        matching_group = next(
            (
                group
                for group in groups
                if all(frozenset((key, member)) in duplicate_pairs for member in group)
            ),
            None,
        )
        if matching_group is None:
            groups.append([key])
        else:
            matching_group.append(key)
    simplicity_metadata = simplicity_metadata or {}
    order = list(
        configuration.get("representative_order")
        or (
            {"field": "complexity_score", "direction": "minimize"},
            {"field": "parameter_count", "direction": "minimize"},
        )
    )

    def representative(members: Sequence[str]) -> str:
        def sort_key(run_id: str) -> tuple[Any, ...]:
            metadata = simplicity_metadata.get(run_id, {})
            values: list[Any] = []
            for rule in order:
                value = metadata.get(rule["field"])
                if value is None:
                    values.append(math.inf)
                elif rule.get("direction", "minimize") == "maximize":
                    values.append(-float(value))
                else:
                    values.append(float(value))
            values.append(run_id)
            return tuple(values)

        return min(members, key=sort_key)

    result: dict[str, dict[str, Any]] = {}
    for members in groups:
        members = sorted(members)
        selected = representative(members)
        cluster_id = f"behavior_cluster_{content_hash(members)}"
        for key in members:
            result[key] = {
                "status": "available",
                "behavior_cluster_id": cluster_id,
                "cluster_members": members,
                "representative_strategy_run_id": selected,
                "is_representative": key == selected,
                "duplicated": len(members) > 1 and key != selected,
                "underlying_strategy_run_preserved": True,
            }
    return result, pairwise
