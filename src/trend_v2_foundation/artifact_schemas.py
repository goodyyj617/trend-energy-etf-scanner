"""Versioned stored-artifact schemas used by the Foundation 2 metric engine."""

from __future__ import annotations

import math
from datetime import date
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


DAILY_PORTFOLIO_CURVE_SCHEMA_VERSION = "daily_portfolio_curve_v1"
YEARLY_METRICS_SCHEMA_VERSION = "yearly_metrics_v1"
ROLLING_METRICS_SCHEMA_VERSION = "rolling_metrics_v1"
ROBUSTNESS_SUMMARY_SCHEMA_VERSION = "robustness_summary_v1"
BEHAVIOR_METADATA_SCHEMA_VERSION = "behavior_metadata_v1"


class ArtifactSchemaError(ValueError):
    """A stored artifact is unsafe to use for deterministic calculation."""


DAILY_REQUIRED_FIELDS = (
    "economic_date",
    "portfolio_value",
    "daily_return",
    "gross_exposure",
    "net_exposure",
    "cash_weight",
    "daily_turnover",
    "transaction_cost",
)
DAILY_OPTIONAL_NUMERIC_FIELDS = (
    "position_count",
    "gross_portfolio_value",
    "gross_daily_return",
)
YEARLY_REQUIRED_FIELDS = (
    "calendar_year",
    "complete_year",
    "start_economic_date",
    "end_economic_date",
    "annual_return",
    "annualized_volatility",
    "maximum_drawdown",
    "turnover",
    "observation_count",
)
ROLLING_REQUIRED_FIELDS = (
    "economic_date",
    "window_sessions",
    "rolling_return",
    "rolling_annualized_return",
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_maximum_drawdown",
    "observation_count",
)
ROBUSTNESS_REQUIRED_FIELDS = (
    "walk_forward_fold_count",
    "walk_forward_pass_ratio",
    "walk_forward_worst_fold",
    "loyo_case_count",
    "loyo_stability_ratio",
    "loyo_reversing_years",
    "block_bootstrap_effect",
    "bootstrap_confidence_interval",
    "raw_p_value",
    "adjusted_p_value",
    "multiple_testing_method",
    "transaction_cost_stress_survival",
    "dominant_asset_group",
    "dominant_group_share",
    "unclassified_group_share",
    "method_metadata",
    "unavailable_reasons",
)
BEHAVIOR_REQUIRED_FIELDS = (
    "daily_return_fingerprint",
    "active_exposure_fingerprint",
    "trade_entry_dates_fingerprint",
    "trade_exit_dates_fingerprint",
    "symbol_lifecycle_fingerprint",
    "comparison_inputs",
    "source_artifact_hashes",
)


def _require_mapping(payload: Any, artifact: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ArtifactSchemaError(f"{artifact}: payload must be a mapping")
    return payload


def _require_version(payload: Mapping[str, Any], expected: str, artifact: str) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        raise ArtifactSchemaError(
            f"{artifact}.schema_version: expected '{expected}', received {actual!r}"
        )


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], path: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ArtifactSchemaError(f"{path}: missing required fields: {', '.join(missing)}")


def _finite(value: Any, path: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ArtifactSchemaError(f"{path}: must be a finite numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactSchemaError(f"{path}: must be a finite numeric value")
    return number


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ArtifactSchemaError(f"{path}: must be an integer")
    result = int(value)
    if result < minimum:
        raise ArtifactSchemaError(f"{path}: must be >= {minimum}")
    return result


def _economic_date(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ArtifactSchemaError(f"{path}: must be an ISO economic date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ArtifactSchemaError(f"{path}: invalid ISO economic date '{value}'") from error
    if parsed.isoformat() != value:
        raise ArtifactSchemaError(f"{path}: must use canonical YYYY-MM-DD format")
    return value


def _rows(payload: Mapping[str, Any], artifact: str) -> list[Mapping[str, Any]]:
    value = payload.get("rows")
    if not isinstance(value, list):
        raise ArtifactSchemaError(f"{artifact}.rows: must be a list")
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ArtifactSchemaError(f"{artifact}.rows[{index}]: must be a mapping")
    return value


def _validate_date_sequence(dates: Sequence[str], artifact: str) -> None:
    if len(dates) != len(set(dates)):
        raise ArtifactSchemaError(f"{artifact}: duplicate economic dates are not allowed")
    if list(dates) != sorted(dates):
        raise ArtifactSchemaError(f"{artifact}: economic dates must be sorted ascending")


def _validate_envelope_range(
    payload: Mapping[str, Any], dates: Sequence[str], artifact: str
) -> None:
    expected = None if not dates else {"start": dates[0], "end": dates[-1]}
    actual = payload.get("economic_date_range")
    if actual != expected:
        raise ArtifactSchemaError(
            f"{artifact}.economic_date_range: expected {expected!r}, received {actual!r}"
        )


def validate_daily_portfolio_curve(payload: Any) -> Mapping[str, Any]:
    artifact = "daily_portfolio_curve"
    source = _require_mapping(payload, artifact)
    _require_version(source, DAILY_PORTFOLIO_CURVE_SCHEMA_VERSION, artifact)
    rows = _rows(source, artifact)
    dates: list[str] = []
    for index, row in enumerate(rows):
        path = f"{artifact}.rows[{index}]"
        _require_fields(row, DAILY_REQUIRED_FIELDS, path)
        dates.append(_economic_date(row["economic_date"], f"{path}.economic_date"))
        portfolio_value = _finite(row["portfolio_value"], f"{path}.portfolio_value")
        if portfolio_value is None or portfolio_value <= 0:
            raise ArtifactSchemaError(f"{path}.portfolio_value: must be greater than zero")
        for field in DAILY_REQUIRED_FIELDS[2:]:
            _finite(row[field], f"{path}.{field}")
        if row["daily_return"] <= -1:
            raise ArtifactSchemaError(f"{path}.daily_return: cannot be <= -1")
        if row["gross_exposure"] < 0:
            raise ArtifactSchemaError(f"{path}.gross_exposure: cannot be negative")
        if row["daily_turnover"] < 0 or row["transaction_cost"] < 0:
            raise ArtifactSchemaError(f"{path}: turnover and transaction cost cannot be negative")
        for field in DAILY_OPTIONAL_NUMERIC_FIELDS:
            if field not in row:
                continue
            number = _finite(row[field], f"{path}.{field}")
            if field in {"position_count", "gross_portfolio_value"} and number is not None and number < 0:
                raise ArtifactSchemaError(f"{path}.{field}: cannot be negative")
            if field == "gross_daily_return" and number is not None and number <= -1:
                raise ArtifactSchemaError(f"{path}.{field}: cannot be <= -1")
    _validate_date_sequence(dates, artifact)
    _validate_envelope_range(source, dates, artifact)
    return source


def validate_yearly_metrics(payload: Any) -> Mapping[str, Any]:
    artifact = "yearly_metrics"
    source = _require_mapping(payload, artifact)
    _require_version(source, YEARLY_METRICS_SCHEMA_VERSION, artifact)
    rows = _rows(source, artifact)
    years: list[int] = []
    for index, row in enumerate(rows):
        path = f"{artifact}.rows[{index}]"
        _require_fields(row, YEARLY_REQUIRED_FIELDS, path)
        year = _integer(row["calendar_year"], f"{path}.calendar_year", minimum=1)
        years.append(year)
        if not isinstance(row["complete_year"], bool):
            raise ArtifactSchemaError(f"{path}.complete_year: must be a JSON boolean")
        start = _economic_date(row["start_economic_date"], f"{path}.start_economic_date")
        end = _economic_date(row["end_economic_date"], f"{path}.end_economic_date")
        if start > end or int(start[:4]) != year or int(end[:4]) != year:
            raise ArtifactSchemaError(f"{path}: inconsistent calendar-year date range")
        for field in ("annual_return", "annualized_volatility", "maximum_drawdown", "turnover"):
            _finite(row[field], f"{path}.{field}")
        _integer(row["observation_count"], f"{path}.observation_count", minimum=1)
    if years != sorted(years) or len(years) != len(set(years)):
        raise ArtifactSchemaError(f"{artifact}: calendar years must be unique and sorted")
    return source


def validate_rolling_metrics(payload: Any) -> Mapping[str, Any]:
    artifact = "rolling_metrics"
    source = _require_mapping(payload, artifact)
    _require_version(source, ROLLING_METRICS_SCHEMA_VERSION, artifact)
    windows = source.get("configured_windows")
    if not isinstance(windows, list) or not windows:
        raise ArtifactSchemaError(f"{artifact}.configured_windows: must be a non-empty list")
    normalized_windows = [_integer(value, f"{artifact}.configured_windows", minimum=2) for value in windows]
    if normalized_windows != sorted(set(normalized_windows)):
        raise ArtifactSchemaError(f"{artifact}.configured_windows: must be unique and sorted")
    rows = _rows(source, artifact)
    keys: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        path = f"{artifact}.rows[{index}]"
        _require_fields(row, ROLLING_REQUIRED_FIELDS, path)
        economic_date = _economic_date(row["economic_date"], f"{path}.economic_date")
        window = _integer(row["window_sessions"], f"{path}.window_sessions", minimum=2)
        if window not in normalized_windows:
            raise ArtifactSchemaError(f"{path}.window_sessions: not configured")
        count = _integer(row["observation_count"], f"{path}.observation_count", minimum=1)
        if count != window:
            raise ArtifactSchemaError(f"{path}.observation_count: must equal window_sessions")
        for field in ROLLING_REQUIRED_FIELDS[2:7]:
            _finite(row[field], f"{path}.{field}", allow_none=True)
        keys.append((window, economic_date))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ArtifactSchemaError(f"{artifact}: rows must be unique and sorted by window then date")
    return source


def validate_robustness_summary(payload: Any) -> Mapping[str, Any]:
    artifact = "robustness_summary"
    source = _require_mapping(payload, artifact)
    _require_version(source, ROBUSTNESS_SUMMARY_SCHEMA_VERSION, artifact)
    _require_fields(source, ROBUSTNESS_REQUIRED_FIELDS, artifact)
    unavailable = source["unavailable_reasons"]
    if not isinstance(unavailable, Mapping):
        raise ArtifactSchemaError(f"{artifact}.unavailable_reasons: must be a mapping")

    for field in ("walk_forward_fold_count", "loyo_case_count"):
        value = source[field]
        if value is not None:
            _integer(value, f"{artifact}.{field}")
    for field in (
        "walk_forward_pass_ratio",
        "walk_forward_worst_fold",
        "loyo_stability_ratio",
        "block_bootstrap_effect",
        "raw_p_value",
        "adjusted_p_value",
        "dominant_group_share",
        "unclassified_group_share",
    ):
        value = source[field]
        if value is not None:
            number = _finite(value, f"{artifact}.{field}")
            if field.endswith("ratio") or field.endswith("share") or field.endswith("p_value"):
                if number is None or not 0 <= number <= 1:
                    raise ArtifactSchemaError(f"{artifact}.{field}: must be between zero and one")
    reversing = source["loyo_reversing_years"]
    if reversing is not None:
        if not isinstance(reversing, list):
            raise ArtifactSchemaError(f"{artifact}.loyo_reversing_years: must be a list")
        normalized = [_integer(value, f"{artifact}.loyo_reversing_years", minimum=1) for value in reversing]
        if normalized != sorted(set(normalized)):
            raise ArtifactSchemaError(f"{artifact}.loyo_reversing_years: must be unique and sorted")
    interval = source["bootstrap_confidence_interval"]
    if interval is not None:
        interval = _require_mapping(interval, f"{artifact}.bootstrap_confidence_interval")
        _require_fields(interval, ("lower", "upper", "confidence_level"), f"{artifact}.bootstrap_confidence_interval")
        lower = _finite(interval["lower"], f"{artifact}.bootstrap_confidence_interval.lower")
        upper = _finite(interval["upper"], f"{artifact}.bootstrap_confidence_interval.upper")
        confidence = _finite(interval["confidence_level"], f"{artifact}.bootstrap_confidence_interval.confidence_level")
        if lower is None or upper is None or lower > upper:
            raise ArtifactSchemaError(f"{artifact}.bootstrap_confidence_interval: lower cannot exceed upper")
        if confidence is None or not 0 < confidence < 1:
            raise ArtifactSchemaError(f"{artifact}.bootstrap_confidence_interval.confidence_level: must be in (0, 1)")
    survival = source["transaction_cost_stress_survival"]
    if survival is not None:
        if isinstance(survival, bool):
            raise ArtifactSchemaError(
                f"{artifact}.transaction_cost_stress_survival: binary representation must be 0.0 or 1.0, not Boolean"
            )
        numeric = _finite(survival, f"{artifact}.transaction_cost_stress_survival")
        if numeric not in (0.0, 1.0):
            raise ArtifactSchemaError(
                f"{artifact}.transaction_cost_stress_survival: binary representation must be 0.0 or 1.0"
            )
    for field in ("multiple_testing_method", "dominant_asset_group"):
        value = source[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ArtifactSchemaError(f"{artifact}.{field}: must be a non-empty string or null")
    if not isinstance(source["method_metadata"], Mapping):
        raise ArtifactSchemaError(f"{artifact}.method_metadata: must be a mapping")
    for field in ROBUSTNESS_REQUIRED_FIELDS[:-2]:
        if source[field] is None and not unavailable.get(field):
            raise ArtifactSchemaError(f"{artifact}.{field}: null values require an unavailable reason")
    return source


def validate_behavior_metadata(payload: Any) -> Mapping[str, Any]:
    artifact = "behavior_metadata"
    source = _require_mapping(payload, artifact)
    _require_version(source, BEHAVIOR_METADATA_SCHEMA_VERSION, artifact)
    _require_fields(source, BEHAVIOR_REQUIRED_FIELDS, artifact)
    for field in BEHAVIOR_REQUIRED_FIELDS[:5]:
        value = source[field]
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ArtifactSchemaError(f"{artifact}.{field}: must be a lowercase SHA-256 or null")
    if not isinstance(source["comparison_inputs"], Mapping):
        raise ArtifactSchemaError(f"{artifact}.comparison_inputs: must be a mapping")
    comparison = source["comparison_inputs"]
    _require_fields(
        comparison,
        (
            "economic_dates",
            "daily_returns",
            "normalized_wealth_path",
            "active_dates",
            "active_exposure_path",
            "entry_dates",
            "exit_dates",
            "symbol_lifecycle_sequence",
        ),
        f"{artifact}.comparison_inputs",
    )
    dates = comparison["economic_dates"]
    returns = comparison["daily_returns"]
    paths = comparison["normalized_wealth_path"]
    if not isinstance(dates, list) or not isinstance(returns, list) or not isinstance(paths, list):
        raise ArtifactSchemaError(f"{artifact}.comparison_inputs: path inputs must be lists")
    if len(dates) != len(returns) or len(dates) != len(paths):
        raise ArtifactSchemaError(f"{artifact}.comparison_inputs: path input lengths must match")
    normalized_dates = [_economic_date(value, f"{artifact}.comparison_inputs.economic_dates") for value in dates]
    _validate_date_sequence(normalized_dates, f"{artifact}.comparison_inputs")
    for index, value in enumerate(returns):
        _finite(value, f"{artifact}.comparison_inputs.daily_returns[{index}]")
    for index, value in enumerate(paths):
        _finite(value, f"{artifact}.comparison_inputs.normalized_wealth_path[{index}]")
    for field in ("active_dates", "entry_dates", "exit_dates"):
        values = comparison[field]
        if values is None and field != "active_dates":
            continue
        if not isinstance(values, list):
            raise ArtifactSchemaError(f"{artifact}.comparison_inputs.{field}: must be a list or null")
        normalized = [_economic_date(value, f"{artifact}.comparison_inputs.{field}") for value in values]
        if normalized != sorted(set(normalized)):
            raise ArtifactSchemaError(f"{artifact}.comparison_inputs.{field}: must be unique and sorted")
    exposure_path = comparison["active_exposure_path"]
    if not isinstance(exposure_path, list) or len(exposure_path) != len(dates):
        raise ArtifactSchemaError(f"{artifact}.comparison_inputs.active_exposure_path: length must match dates")
    for index, item in enumerate(exposure_path):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ArtifactSchemaError(f"{artifact}.comparison_inputs.active_exposure_path[{index}]: invalid row")
        _economic_date(item[0], f"{artifact}.comparison_inputs.active_exposure_path[{index}]")
        _finite(item[1], f"{artifact}.comparison_inputs.active_exposure_path[{index}].gross")
        _finite(item[2], f"{artifact}.comparison_inputs.active_exposure_path[{index}].net")
    if not isinstance(source["source_artifact_hashes"], Mapping):
        raise ArtifactSchemaError(f"{artifact}.source_artifact_hashes: must be a mapping")
    return source


def artifact_payload_row_count(payload: Any) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        return len(payload["rows"])
    return None
