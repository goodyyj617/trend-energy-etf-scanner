"""Controlled Foundation 5 construction, estimation, and confirmation contracts."""

from __future__ import annotations

import itertools
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import canonical_bytes, canonical_data, content_hash, deep_freeze, deterministic_id
from .contracts import StrategyRunSpec


CONSTRUCTION_REQUEST_VERSION = "strategy_construction_request_v1"
NORMALIZED_CONSTRUCTION_VERSION = "normalized_strategy_construction_v1"
CANDIDATE_ESTIMATE_VERSION = "candidate_space_estimate_v1"
EXECUTION_CONFIRMATION_VERSION = "execution_confirmation_v1"
EXECUTION_REQUEST_VERSION = "execution_request_v1"
EXECUTION_POLICY_VERSION = "local_execution_policy_v1"
CONTROLLED_ENGINE_VERSION = "trend_v2_phase_a_controlled_adapter_v1"


class Foundation5Error(ValueError):
    """Stable error translated directly by the local API boundary."""

    def __init__(
        self,
        code: str,
        diagnostic_en: str,
        *,
        object_identity: str | None = None,
        recoverable: bool = True,
        next_action_ko: str | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.diagnostic_en = diagnostic_en
        self.object_identity = object_identity
        self.recoverable = recoverable
        self.next_action_ko = next_action_ko
        self.status_code = status_code


@dataclass(frozen=True)
class LocalExecutionPolicy:
    informational_threshold: int
    explicit_confirmation_threshold: int
    hard_local_refusal_threshold: int
    maximum_economic_candidate_count: int
    maximum_evaluation_count: int
    maximum_estimated_execution_units: int
    maximum_date_span_days: int
    maximum_universe_size: int
    maximum_parameter_values_per_dimension: int
    maximum_parameter_dimensions: int
    maximum_concurrency: int
    maximum_json_body_bytes: int
    confirmation_ttl_seconds: int
    confirmation_one_time_use: bool
    candidate_count_overflow_limit: int
    schema_version: str = EXECUTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_POLICY_VERSION:
            raise ValueError("unsupported local execution policy version")
        numeric = [
            self.informational_threshold,
            self.explicit_confirmation_threshold,
            self.hard_local_refusal_threshold,
            self.maximum_economic_candidate_count,
            self.maximum_evaluation_count,
            self.maximum_estimated_execution_units,
            self.maximum_date_span_days,
            self.maximum_universe_size,
            self.maximum_parameter_values_per_dimension,
            self.maximum_parameter_dimensions,
            self.maximum_concurrency,
            self.maximum_json_body_bytes,
            self.confirmation_ttl_seconds,
            self.candidate_count_overflow_limit,
        ]
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in numeric):
            raise ValueError("local execution policy numeric limits must be positive integers")
        if not (
            self.informational_threshold
            <= self.explicit_confirmation_threshold
            <= self.hard_local_refusal_threshold
        ):
            raise ValueError("policy workload thresholds must be ordered")
        if self.maximum_concurrency != 1:
            raise ValueError("Foundation 5 permits exactly one local execution worker")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalExecutionPolicy":
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @property
    def policy_hash(self) -> str:
        return content_hash(self)


def load_execution_policy(path: str | Path) -> LocalExecutionPolicy:
    import json

    return LocalExecutionPolicy.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


SUPPORTED_OPTIONS: Mapping[str, Any] = {
    "schema_version": "controlled_strategy_options_v1",
    "engine_version": CONTROLLED_ENGINE_VERSION,
    "data_snapshots": ({
        "option_id": "phase_a2_frozen_2026_07_30",
        "snapshot_hash": "b37025ba30f5ecc2d32c53797201b8fcf43c06f539e6414ed4d4cdd9f70b82bd",
        "date_range": {"start": "2016-08-01", "end": "2026-07-30"},
        "universe_size": 470,
        "label_ko": "Phase A2 동결 스냅샷 (2026-07-30)",
    },),
    "universe": ({
        "option_id": "phase_a2_historical_eligible_v1",
        "label_ko": "Phase A2 동결 적격 유니버스",
        "parameter_schema": {},
    },),
    "benchmark": ({"option_id": "spy_adjusted_close_v1", "symbol": "SPY", "label_ko": "SPY"},),
    "trend_filter": ({
        "option_id": "price_above_rising_ma200_v0",
        "label_ko": "종가가 상승 중인 200일 이동평균 위",
        "parameter_schema": {},
    },),
    "signal": ({
        "option_id": "prior_price_high_l20_v1",
        "label_ko": "직전 20거래일 가격 고점 돌파",
        "parameter_schema": {"lookback": {"type": "integer", "minimum": 20, "maximum": 20}},
    },),
    "entry_rule": ({
        "option_id": "first_event_next_open_v1",
        "label_ko": "첫 신호 다음 거래일 시가 진입",
        "parameter_schema": {},
    },),
    "initial_stop": ({
        "option_id": "signal_day_low20_v1",
        "label_ko": "신호일 20일 저가 초기 손절",
        "parameter_schema": {},
    },),
    "trailing_exit": ({
        "option_id": "ratcheting_low20_v1",
        "label_ko": "20일 저가 추적 청산 (하향 이동 없음)",
        "parameter_schema": {},
    },),
    "position_sizing": ({
        "option_id": "canonical_equal_weight_active_v1",
        "label_ko": "활성 종목 동일 비중",
        "parameter_schema": {},
    },),
    "portfolio_constraints": ({
        "option_id": "long_only_cash_constrained_v1",
        "label_ko": "롱 전용·현금 제약·차입 없음",
        "parameter_schema": {},
    },),
    "transaction_cost": ({
        "option_id": "round_trip_bps_v1",
        "label_ko": "왕복 거래비용 (bp)",
        "parameter_schema": {"bps": {"type": "decimal", "minimum": "0", "maximum": "50"}},
    },),
    "slippage": ({
        "option_id": "round_trip_slippage_bps_v1",
        "label_ko": "왕복 슬리피지 (bp)",
        "parameter_schema": {"bps": {"type": "decimal", "minimum": "0", "maximum": "50"}},
    },),
}


_COMPONENT_FIELDS = (
    "trend_filter",
    "signal",
    "entry_rule",
    "initial_stop",
    "trailing_exit",
    "position_sizing",
    "portfolio_constraints",
    "transaction_cost",
    "slippage",
)
_ECONOMIC_FIELDS = _COMPONENT_FIELDS


def construction_options(policy: LocalExecutionPolicy) -> dict[str, Any]:
    return {**canonical_data(SUPPORTED_OPTIONS), "execution_policy": policy.to_dict()}


def _decimal(value: Any, path: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise Foundation5Error("invalid_parameter_range", f"{path} must use a decimal string or integer.")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise Foundation5Error("invalid_parameter_range", f"{path} is not a valid decimal.") from error
    if not number.is_finite():
        raise Foundation5Error("invalid_parameter_range", f"{path} must be finite.")
    return number


def _decimal_string(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _range_values(
    value: Mapping[str, Any], path: str, maximum_values: int
) -> tuple[list[Any], int]:
    kind = value.get("kind")
    if kind == "fixed":
        return [value.get("value")], 1
    if kind == "list":
        values = value.get("values")
        if not isinstance(values, list) or not values:
            raise Foundation5Error("invalid_parameter_range", f"{path}.values must be a non-empty list.")
        if len(values) > maximum_values:
            raise Foundation5Error("hard_limit_exceeded", f"{path} exceeds the per-dimension value limit.")
        return list(values), len(values)
    if kind not in {"integer_range", "decimal_range"}:
        raise Foundation5Error("invalid_parameter_range", f"{path}.kind is unsupported.")
    start = _decimal(value.get("start"), f"{path}.start")
    end = _decimal(value.get("end"), f"{path}.end")
    step = _decimal(value.get("step"), f"{path}.step")
    if step <= 0 or end < start:
        raise Foundation5Error("invalid_parameter_range", f"{path} requires end >= start and step > 0.")
    distance = end - start
    quotient = distance / step
    if quotient != quotient.to_integral_value():
        raise Foundation5Error(
            "invalid_parameter_range",
            f"{path} does not terminate exactly at end under the declared step.",
        )
    count = int(quotient) + 1
    if count > maximum_values:
        raise Foundation5Error("hard_limit_exceeded", f"{path} exceeds the per-dimension value limit.")
    values = [start + step * index for index in range(count)]
    if kind == "integer_range":
        if any(item != item.to_integral_value() for item in values):
            raise Foundation5Error("invalid_parameter_range", f"{path} requires integer values.")
        return [int(item) for item in values], count
    return [_decimal_string(item) for item in values], count


def _normalize_parameter_space(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
    policy: LocalExecutionPolicy,
) -> tuple[tuple[Any, ...], int]:
    values, raw_count = (
        _range_values(value, path, policy.maximum_parameter_values_per_dimension)
        if isinstance(value, Mapping)
        else ([value], 1)
    )
    normalized: list[Any] = []
    for item in values:
        if schema["type"] == "integer":
            if isinstance(item, bool):
                raise Foundation5Error("invalid_construction_field", f"{path} must be an integer.")
            try:
                number = int(item)
            except (TypeError, ValueError) as error:
                raise Foundation5Error("invalid_construction_field", f"{path} must be an integer.") from error
            if str(number) != str(item) and not isinstance(item, int):
                raise Foundation5Error("invalid_construction_field", f"{path} must be a canonical integer.")
            normalized.append(number)
        else:
            normalized.append(_decimal_string(_decimal(item, path)))
    if len(normalized) > policy.maximum_parameter_values_per_dimension:
        raise Foundation5Error(
            "hard_limit_exceeded",
            f"{path} exceeds maximum_parameter_values_per_dimension.",
            recoverable=True,
        )
    if len(set(normalized)) != len(normalized):
        raise Foundation5Error("invalid_parameter_range", f"{path} contains duplicate normalized values.")
    normalized.sort(key=lambda item: Decimal(str(item)))
    minimum = Decimal(str(schema["minimum"]))
    maximum = Decimal(str(schema["maximum"]))
    if any(not minimum <= Decimal(str(item)) <= maximum for item in normalized):
        raise Foundation5Error("invalid_construction_field", f"{path} is outside the supported bounds.")
    return tuple(normalized), raw_count


def _option(field_name: str, payload: Any) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(payload, Mapping) or set(payload).difference({"option_id", "parameters"}):
        raise Foundation5Error("invalid_construction_field", f"{field_name} must contain option_id and parameters only.")
    option_id = payload.get("option_id")
    matches = [item for item in SUPPORTED_OPTIONS[field_name] if item["option_id"] == option_id]
    if not matches:
        raise Foundation5Error("unsupported_option", f"Unsupported {field_name} option.", object_identity=str(option_id)[:160])
    return str(option_id), matches[0]


@dataclass(frozen=True)
class NormalizedConstruction:
    normalized: Mapping[str, Any]
    raw_dimension_counts: Mapping[str, int]
    schema_version: str = NORMALIZED_CONSTRUCTION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized", deep_freeze(self.normalized))
        object.__setattr__(self, "raw_dimension_counts", deep_freeze(self.raw_dimension_counts))

    @property
    def construction_hash(self) -> str:
        return content_hash({"schema_version": self.schema_version, "normalized": self.normalized})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalized_construction_hash": self.construction_hash,
            "normalized": canonical_data(self.normalized),
            "raw_dimension_counts": canonical_data(self.raw_dimension_counts),
        }


def normalize_construction(
    request: Mapping[str, Any], policy: LocalExecutionPolicy
) -> NormalizedConstruction:
    allowed = {
        "schema_version", "data_snapshot", "backtest_start_date", "backtest_end_date",
        "universe", "benchmark", *_COMPONENT_FIELDS, "walk_forward", "robustness",
        "evaluation_profile_ids",
    }
    unknown = set(request).difference(allowed)
    if unknown:
        raise Foundation5Error("invalid_construction_field", f"Unknown construction fields: {sorted(unknown)}.")
    if request.get("schema_version") != CONSTRUCTION_REQUEST_VERSION:
        raise Foundation5Error("invalid_construction_field", "Unsupported construction request schema version.")
    snapshot_id = request.get("data_snapshot")
    snapshot = next((item for item in SUPPORTED_OPTIONS["data_snapshots"] if item["option_id"] == snapshot_id), None)
    if snapshot is None:
        raise Foundation5Error("snapshot_unavailable", "The selected data snapshot is not allow-listed.", object_identity=str(snapshot_id)[:160])
    try:
        start = date.fromisoformat(str(request.get("backtest_start_date")))
        end = date.fromisoformat(str(request.get("backtest_end_date")))
    except ValueError as error:
        raise Foundation5Error("invalid_construction_field", "Backtest dates must be canonical YYYY-MM-DD values.") from error
    if start > end or start.isoformat() < snapshot["date_range"]["start"] or end.isoformat() > snapshot["date_range"]["end"]:
        raise Foundation5Error("invalid_construction_field", "Backtest dates are outside the selected snapshot or reversed.")
    normalized: dict[str, Any] = {
        "data_snapshot": {"option_id": snapshot_id, "snapshot_hash": snapshot["snapshot_hash"]},
        "economic_date_range": {"start": start.isoformat(), "end": end.isoformat()},
    }
    raw_counts: dict[str, int] = {}
    varying_dimensions = 0
    for field_name in ("universe", "benchmark", *_COMPONENT_FIELDS):
        option_id, definition = _option(field_name, request.get(field_name))
        supplied = request[field_name].get("parameters", {})
        schemas = definition.get("parameter_schema", {})
        if not isinstance(supplied, Mapping) or set(supplied) != set(schemas):
            raise Foundation5Error("invalid_construction_field", f"{field_name}.parameters must exactly match its schema.")
        parameters: dict[str, tuple[Any, ...]] = {}
        for parameter_name, schema in schemas.items():
            path = f"{field_name}.parameters.{parameter_name}"
            values, raw_count = _normalize_parameter_space(supplied[parameter_name], schema, path, policy)
            parameters[parameter_name] = values
            raw_counts[path] = raw_count
            if len(values) > 1:
                varying_dimensions += 1
        normalized[field_name] = {"option_id": option_id, "parameters": parameters}
    if varying_dimensions > policy.maximum_parameter_dimensions:
        raise Foundation5Error("hard_limit_exceeded", "Too many varying parameter dimensions.")
    walk_forward = request.get("walk_forward")
    robustness = request.get("robustness")
    if not isinstance(walk_forward, Mapping) or set(walk_forward) != {"enabled", "fold_count"}:
        raise Foundation5Error("invalid_construction_field", "walk_forward requires enabled and fold_count.")
    if not isinstance(robustness, Mapping) or set(robustness) != {"scenario_count"}:
        raise Foundation5Error("invalid_construction_field", "robustness requires scenario_count.")
    fold_count = walk_forward["fold_count"]
    scenario_count = robustness["scenario_count"]
    if not isinstance(walk_forward["enabled"], bool) or isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise Foundation5Error("invalid_construction_field", "walk_forward values have invalid types.")
    if isinstance(scenario_count, bool) or not isinstance(scenario_count, int):
        raise Foundation5Error("invalid_construction_field", "robustness.scenario_count must be an integer.")
    if fold_count < 0 or fold_count > 20 or scenario_count < 0 or scenario_count > 20:
        raise Foundation5Error("invalid_construction_field", "Walk-forward folds and robustness scenarios must be between 0 and 20.")
    if not walk_forward["enabled"] and fold_count != 0:
        raise Foundation5Error("invalid_construction_field", "Disabled walk-forward requires fold_count 0.")
    if walk_forward["enabled"] and fold_count < 2:
        raise Foundation5Error("invalid_construction_field", "Enabled walk-forward requires at least two folds.")
    profile_ids = request.get("evaluation_profile_ids")
    if not isinstance(profile_ids, list) or not profile_ids or any(not isinstance(item, str) or not item for item in profile_ids):
        raise Foundation5Error("invalid_construction_field", "At least one evaluation profile ID is required.")
    if len(profile_ids) != len(set(profile_ids)):
        raise Foundation5Error("invalid_construction_field", "Evaluation profile IDs cannot contain duplicates.")
    normalized["walk_forward"] = {"enabled": walk_forward["enabled"], "fold_count": fold_count}
    normalized["robustness"] = {"scenario_count": scenario_count}
    normalized["evaluation_profile_ids"] = tuple(sorted(profile_ids))
    normalized["universe_size"] = int(snapshot["universe_size"])
    return NormalizedConstruction(normalized, raw_counts)


def _component_spec(option: Mapping[str, Any], parameter_values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "option_id": option["option_id"],
        "schema_version": "controlled_component_v1",
        "parameters": dict(parameter_values),
    }


def expand_strategy_candidates(normalized: NormalizedConstruction) -> tuple[StrategyRunSpec, ...]:
    dimensions: list[tuple[str, str, tuple[Any, ...]]] = []
    for field_name in _ECONOMIC_FIELDS:
        for parameter, values in normalized.normalized[field_name]["parameters"].items():
            dimensions.append((field_name, parameter, tuple(values)))
    combinations = itertools.product(*(item[2] for item in dimensions)) if dimensions else [()]
    candidates: dict[str, StrategyRunSpec] = {}
    for combination in combinations:
        selected: dict[str, dict[str, Any]] = {field: {} for field in _ECONOMIC_FIELDS}
        for (field_name, parameter, _), value in zip(dimensions, combination):
            selected[field_name][parameter] = value
        source = normalized.normalized
        spec = StrategyRunSpec(
            data_snapshot_hash=source["data_snapshot"]["snapshot_hash"],
            economic_date_range=source["economic_date_range"],
            universe_specification=_component_spec(source["universe"], {}),
            benchmark={**_component_spec(source["benchmark"], {}), "symbol": "SPY", "identity": "spy_adjusted_close_v1"},
            trend_filter=_component_spec(source["trend_filter"], selected["trend_filter"]),
            signal=_component_spec(source["signal"], selected["signal"]),
            entry_rule=_component_spec(source["entry_rule"], selected["entry_rule"]),
            initial_stop=_component_spec(source["initial_stop"], selected["initial_stop"]),
            trailing_exit=_component_spec(source["trailing_exit"], selected["trailing_exit"]),
            position_sizing=_component_spec(source["position_sizing"], selected["position_sizing"]),
            portfolio_constraints=_component_spec(source["portfolio_constraints"], selected["portfolio_constraints"]),
            transaction_costs=_component_spec(source["transaction_cost"], selected["transaction_cost"]),
            slippage=_component_spec(source["slippage"], selected["slippage"]),
            engine_version=CONTROLLED_ENGINE_VERSION,
        )
        candidates[spec.strategy_run_id] = spec
    return tuple(candidates[key] for key in sorted(candidates))


@dataclass(frozen=True)
class CandidateSpaceEstimate:
    normalized_construction_hash: str
    raw_cartesian_candidate_count: int
    deduplicated_normalized_candidate_count: int
    economic_strategy_run_candidate_count: int
    evaluation_profile_application_count: int
    walk_forward_fold_execution_count: int
    robustness_scenario_count: int
    benchmark_calculation_count: int
    derived_metric_calculation_count: int
    estimated_total_execution_units: int
    estimated_reuse_count: int
    estimated_new_backtest_count: int
    threshold_results: tuple[Mapping[str, Any], ...]
    confirmation_required: bool
    hard_limit_exceeded: bool
    policy_version: str
    policy_hash: str
    schema_version: str = CANDIDATE_ESTIMATE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold_results", tuple(deep_freeze(item) for item in self.threshold_results))

    @property
    def estimate_hash(self) -> str:
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_data(self)
        payload["candidate_estimate_hash"] = self.estimate_hash
        return payload


def _safe_product(values: Sequence[int], limit: int) -> int:
    result = 1
    for value in values:
        if value and result > limit // value:
            raise Foundation5Error("candidate_estimate_overflow", "Candidate count exceeds the deterministic overflow bound.")
        result *= value
    return result


def estimate_candidate_space(
    normalized: NormalizedConstruction,
    policy: LocalExecutionPolicy,
    *,
    reusable: Callable[[StrategyRunSpec], bool] | None = None,
) -> tuple[CandidateSpaceEstimate, tuple[StrategyRunSpec, ...]]:
    raw = _safe_product(tuple(normalized.raw_dimension_counts.values()), policy.candidate_count_overflow_limit)
    candidates = expand_strategy_candidates(normalized)
    economic = len(candidates)
    profile_count = len(normalized.normalized["evaluation_profile_ids"])
    evaluation = economic * profile_count
    folds = economic * int(normalized.normalized["walk_forward"]["fold_count"])
    robustness = economic * int(normalized.normalized["robustness"]["scenario_count"])
    benchmark = economic
    derived = economic
    total = economic + evaluation + folds + robustness + benchmark + derived
    reuse = sum(bool(reusable(candidate)) for candidate in candidates) if reusable else 0
    span = (
        date.fromisoformat(normalized.normalized["economic_date_range"]["end"])
        - date.fromisoformat(normalized.normalized["economic_date_range"]["start"])
    ).days + 1
    checks = (
        ("informational_threshold", total, policy.informational_threshold, "informational"),
        ("explicit_confirmation_threshold", total, policy.explicit_confirmation_threshold, "confirmation"),
        ("hard_local_refusal_threshold", total, policy.hard_local_refusal_threshold, "hard"),
        ("maximum_economic_candidate_count", economic, policy.maximum_economic_candidate_count, "hard"),
        ("maximum_evaluation_count", evaluation, policy.maximum_evaluation_count, "hard"),
        ("maximum_estimated_execution_units", total, policy.maximum_estimated_execution_units, "hard"),
        ("maximum_date_span_days", span, policy.maximum_date_span_days, "hard"),
        ("maximum_universe_size", int(normalized.normalized["universe_size"]), policy.maximum_universe_size, "hard"),
    )
    results = tuple(
        {
            "threshold": name,
            "observed": observed,
            "limit": limit,
            "severity": severity,
            "triggered": observed >= limit if name.endswith("threshold") else observed > limit,
        }
        for name, observed, limit, severity in checks
    )
    hard = any(item["triggered"] and item["severity"] == "hard" for item in results)
    confirmation = not hard and any(
        item["triggered"] and item["severity"] == "confirmation" for item in results
    )
    return CandidateSpaceEstimate(
        normalized_construction_hash=normalized.construction_hash,
        raw_cartesian_candidate_count=raw,
        deduplicated_normalized_candidate_count=economic,
        economic_strategy_run_candidate_count=economic,
        evaluation_profile_application_count=evaluation,
        walk_forward_fold_execution_count=folds,
        robustness_scenario_count=robustness,
        benchmark_calculation_count=benchmark,
        derived_metric_calculation_count=derived,
        estimated_total_execution_units=total,
        estimated_reuse_count=reuse,
        estimated_new_backtest_count=economic - reuse,
        threshold_results=results,
        confirmation_required=confirmation,
        hard_limit_exceeded=hard,
        policy_version=policy.schema_version,
        policy_hash=policy.policy_hash,
    ), candidates


@dataclass(frozen=True)
class ExecutionConfirmation:
    normalized_construction_hash: str
    candidate_estimate_hash: str
    policy_version: str
    policy_hash: str
    threshold_result_hash: str
    created_timestamp: str
    expires_timestamp: str
    one_time_use: bool
    schema_version: str = EXECUTION_CONFIRMATION_VERSION

    @property
    def confirmation_id(self) -> str:
        return deterministic_id("execution_confirmation", self)

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_data(self)
        payload["confirmation_id"] = self.confirmation_id
        return payload

    @classmethod
    def create(
        cls,
        normalized: NormalizedConstruction,
        estimate: CandidateSpaceEstimate,
        policy: LocalExecutionPolicy,
        *,
        created_timestamp: str,
    ) -> "ExecutionConfirmation":
        if estimate.hard_limit_exceeded:
            raise Foundation5Error("hard_limit_exceeded", "Hard-limit violations cannot be confirmed.", recoverable=True)
        if not estimate.confirmation_required:
            raise Foundation5Error("confirmation_invalid", "This request does not require large-request confirmation.")
        created = _parse_timestamp(created_timestamp)
        expires = created + timedelta(seconds=policy.confirmation_ttl_seconds)
        return cls(
            normalized_construction_hash=normalized.construction_hash,
            candidate_estimate_hash=estimate.estimate_hash,
            policy_version=policy.schema_version,
            policy_hash=policy.policy_hash,
            threshold_result_hash=content_hash(estimate.threshold_results),
            created_timestamp=_timestamp(created),
            expires_timestamp=_timestamp(expires),
            one_time_use=policy.confirmation_one_time_use,
        )


def _parse_timestamp(value: str) -> datetime:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ExecutionRequest:
    normalized_construction: Mapping[str, Any]
    normalized_construction_hash: str
    candidate_estimate: Mapping[str, Any]
    candidate_estimate_hash: str
    confirmation_id: str | None
    request_timestamp: str
    requested_strategy_run_candidates: tuple[Mapping[str, Any], ...]
    selected_evaluation_profile_ids: tuple[str, ...]
    execution_policy_version: str
    execution_policy_hash: str
    source_commit: str
    engine_version: str
    data_snapshot_identity: str
    expected_output_contracts: tuple[str, ...]
    schema_version: str = EXECUTION_REQUEST_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized_construction", deep_freeze(self.normalized_construction))
        object.__setattr__(self, "candidate_estimate", deep_freeze(self.candidate_estimate))
        object.__setattr__(self, "requested_strategy_run_candidates", tuple(deep_freeze(item) for item in self.requested_strategy_run_candidates))
        object.__setattr__(self, "selected_evaluation_profile_ids", tuple(self.selected_evaluation_profile_ids))
        object.__setattr__(self, "expected_output_contracts", tuple(self.expected_output_contracts))

    @property
    def execution_request_id(self) -> str:
        return deterministic_id("execution_request", self)

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_data(self)
        payload["execution_request_id"] = self.execution_request_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionRequest":
        payload = dict(value)
        expected = payload.pop("execution_request_id", None)
        payload["requested_strategy_run_candidates"] = tuple(payload["requested_strategy_run_candidates"])
        payload["selected_evaluation_profile_ids"] = tuple(payload["selected_evaluation_profile_ids"])
        payload["expected_output_contracts"] = tuple(payload["expected_output_contracts"])
        request = cls(**payload)
        if expected and expected != request.execution_request_id:
            raise ValueError("execution_request_id does not match immutable content")
        return request


def create_execution_request(
    normalized: NormalizedConstruction,
    estimate: CandidateSpaceEstimate,
    candidates: Sequence[StrategyRunSpec],
    policy: LocalExecutionPolicy,
    *,
    request_timestamp: str,
    source_commit: str,
    confirmation: ExecutionConfirmation | None = None,
) -> ExecutionRequest:
    if estimate.hard_limit_exceeded:
        raise Foundation5Error("hard_limit_exceeded", "Execution request exceeds a hard local limit.")
    if estimate.normalized_construction_hash != normalized.construction_hash:
        raise Foundation5Error("confirmation_stale", "Estimate is not bound to the normalized construction.")
    if estimate.policy_hash != policy.policy_hash:
        raise Foundation5Error("confirmation_stale", "Execution policy changed after estimation.")
    if estimate.confirmation_required:
        if confirmation is None:
            raise Foundation5Error("confirmation_required", "This request requires an exact bound confirmation.", status_code=409)
        validate_confirmation(confirmation, normalized, estimate, policy, now=request_timestamp)
    elif confirmation is not None:
        validate_confirmation(confirmation, normalized, estimate, policy, now=request_timestamp)
    return ExecutionRequest(
        normalized_construction=normalized.to_dict(),
        normalized_construction_hash=normalized.construction_hash,
        candidate_estimate=estimate.to_dict(),
        candidate_estimate_hash=estimate.estimate_hash,
        confirmation_id=confirmation.confirmation_id if confirmation else None,
        request_timestamp=request_timestamp,
        requested_strategy_run_candidates=tuple(candidate.to_dict() for candidate in candidates),
        selected_evaluation_profile_ids=tuple(normalized.normalized["evaluation_profile_ids"]),
        execution_policy_version=policy.schema_version,
        execution_policy_hash=policy.policy_hash,
        source_commit=source_commit,
        engine_version=CONTROLLED_ENGINE_VERSION,
        data_snapshot_identity=normalized.normalized["data_snapshot"]["snapshot_hash"],
        expected_output_contracts=(
            "strategy_run_manifest_v1", "daily_portfolio_curve_v1", "trade_lifecycles",
            "derived_metric_manifest_v1", "evaluation_run_v2", "execution_attempt_v1",
        ),
    )


def validate_confirmation(
    confirmation: ExecutionConfirmation,
    normalized: NormalizedConstruction,
    estimate: CandidateSpaceEstimate,
    policy: LocalExecutionPolicy,
    *,
    now: str,
) -> None:
    expected = (
        confirmation.normalized_construction_hash == normalized.construction_hash
        and confirmation.candidate_estimate_hash == estimate.estimate_hash
        and confirmation.policy_version == policy.schema_version
        and confirmation.policy_hash == policy.policy_hash
        and confirmation.threshold_result_hash == content_hash(estimate.threshold_results)
    )
    if not expected:
        raise Foundation5Error("confirmation_stale", "Confirmation no longer matches construction, estimate, or policy.", status_code=409)
    if _parse_timestamp(now) > _parse_timestamp(confirmation.expires_timestamp):
        raise Foundation5Error("confirmation_invalid", "Confirmation has expired.", status_code=409)


class ImmutableJsonRepository:
    """Small immutable local repository with idempotency-key bindings."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _write_immutable(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError("immutable object already exists with different content")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_bytes(payload)
        temporary.replace(path)

    def save(self, kind: str, identity: str, value: Mapping[str, Any]) -> None:
        with self._lock:
            self._write_immutable(self.root / kind / f"{identity}.json", canonical_bytes(value))

    def get(self, kind: str, identity: str) -> Mapping[str, Any]:
        import json

        path = self.root / kind / f"{identity}.json"
        if not path.is_file():
            raise KeyError(identity)
        return json.loads(path.read_text(encoding="utf-8"))

    def bind_idempotency(self, operation: str, key: str, request_hash: str, object_id: str) -> str:
        with self._lock:
            binding_id = content_hash({"operation": operation, "key": key})
            payload = {"operation": operation, "key_hash": content_hash(key), "request_hash": request_hash, "object_id": object_id}
            path = self.root / "idempotency" / f"{binding_id}.json"
            if path.exists():
                existing = self.get("idempotency", binding_id)
                if existing["request_hash"] != request_hash:
                    raise Foundation5Error("invalid_construction_field", "Idempotency key was reused with different content.", status_code=409)
                return str(existing["object_id"])
            self.save("idempotency", binding_id, payload)
            return object_id
