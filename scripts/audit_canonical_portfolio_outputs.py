"""Bounded audit for committed canonical portfolio outputs.

This script reads only committed aggregate/matrix/curve files. It does not
download prices, simulate trades, or run the full Backtest Only workflow.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import (
    _attach_parameter_stability,
    _attach_robustness_tiers,
    _attach_time_stability,
    _gate_fields,
    rank_strategy_summary,
)
from src.portfolio import (
    CDAR_DEFINITION_VERSION,
    DIAGNOSTIC_LEADER_FIELDS,
    PORTFOLIO_CURVE_SCHEMA_VERSION,
    conditional_drawdown_at_risk,
    select_published_curve_keys,
)

DATA = ROOT / "docs" / "data"
ATOL = 1e-9
RTOL = 1e-9


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def close(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if not finite(left) or not finite(right):
        return False
    return bool(np.isclose(float(left), float(right), atol=ATOL, rtol=RTOL))


def independent_cdar(drawdowns: np.ndarray, confidence: float = 0.95) -> float:
    clean = np.asarray(drawdowns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return 0.0
    if np.any(clean > ATOL):
        raise ValueError("drawdown values cannot be positive")
    clean = np.minimum(clean, 0.0)
    clean.sort()
    tail_count = max(1, len(clean) - math.floor(confidence * len(clean) + 1e-12))
    return float(clean[:tail_count].mean())


def expected_shortfall(values: np.ndarray, confidence: float) -> float:
    clean = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    cutoff = clean.quantile(1.0 - confidence, interpolation="linear")
    tail = clean.loc[clean <= cutoff]
    return float(tail.mean()) if len(tail) else float(cutoff)


def drawdown_dates(dates: pd.DatetimeIndex, equity: np.ndarray, drawdown: np.ndarray) -> dict[str, Any]:
    trough = int(np.argmin(drawdown))
    if float(drawdown[trough]) >= -ATOL:
        first = dates[0].date().isoformat()
        return {
            "max_drawdown_peak_date": first,
            "max_drawdown_trough_date": first,
            "max_drawdown_recovery_date": None if False else first,
            "max_drawdown_duration_days": 0,
            "longest_time_under_water_days": 0,
        }

    peak_value = max(1000.0, float(np.max(equity[: trough + 1])))
    if peak_value == 1000.0 and np.all(equity[: trough + 1] < peak_value):
        peak_index = 0
    else:
        peak_index = int(np.flatnonzero(np.isclose(
            equity[: trough + 1], peak_value, atol=ATOL, rtol=0
        ))[-1])
    recovery_indexes = np.flatnonzero(equity[trough + 1 :] >= peak_value - ATOL)
    recovery_index = int(trough + 1 + recovery_indexes[0]) if len(recovery_indexes) else None
    duration_end = dates[recovery_index] if recovery_index is not None else dates[-1]

    longest = 0
    running_peak = 1000.0
    episode_start = None
    for index, value in enumerate(equity):
        if value >= running_peak - ATOL:
            running_peak = max(running_peak, float(value))
            if episode_start is not None:
                longest = max(longest, int((dates[index] - dates[episode_start]).days))
                episode_start = None
        elif episode_start is None:
            episode_start = index
    if episode_start is not None:
        longest = max(longest, int((dates[-1] - dates[episode_start]).days))

    return {
        "max_drawdown_peak_date": dates[peak_index].date().isoformat(),
        "max_drawdown_trough_date": dates[trough].date().isoformat(),
        "max_drawdown_recovery_date": (
            dates[recovery_index].date().isoformat() if recovery_index is not None else None
        ),
        "max_drawdown_duration_days": int((duration_end - dates[peak_index]).days),
        "longest_time_under_water_days": longest,
    }


def add_mismatch(store: dict[str, list[str]], field: str, key: str) -> None:
    store.setdefault(field, []).append(key)


def audit() -> dict[str, Any]:
    event = pd.read_csv(DATA / "backtest_strategy_summary.csv")
    annual = pd.read_csv(DATA / "backtest_strategy_year_summary.csv")
    portfolio = pd.read_csv(DATA / "backtest_portfolio_strategy_summary.csv")
    matrix = pd.read_csv(DATA / "backtest_portfolio_daily_returns.csv.gz")
    payload = json.loads((DATA / "backtest_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (DATA / "backtest_portfolio_curve_manifest.json").read_text(encoding="utf-8")
    )
    benchmark = json.loads((DATA / "backtest_benchmark_spy.json").read_text(encoding="utf-8"))

    event_by_key = event.set_index("strategy_key")
    portfolio_by_key = portfolio.set_index("strategy_key")
    strategy_columns = list(matrix.columns[1:])
    manifest_entries = manifest.get("strategies", [])
    manifest_keys = [str(item["strategy_key"]) for item in manifest_entries]
    manifest_files = {str(item["file"]) for item in manifest_entries}
    curve_directory = DATA / "backtest_portfolio_curves"
    curve_files = {path.name for path in curve_directory.glob("*.json")}
    qualified_keys = set(event.loc[event["qualification_tier"].eq("Qualified"), "strategy_key"])
    expected_publication = set(select_published_curve_keys(event, int(manifest["curve_cap"])))
    primary_key = str(event.sort_values(["qualification_rank", "strategy_key"]).iloc[0].strategy_key)
    required_leaders: set[str] = set()
    qualified = event.loc[event["qualification_tier"].eq("Qualified")]
    for field in DIAGNOSTIC_LEADER_FIELDS:
        values = pd.to_numeric(qualified[field], errors="coerce")
        if values.notna().any():
            required_leaders.update(qualified.loc[values.eq(values.max()), "strategy_key"].astype(str))

    matrix_dates = pd.to_datetime(matrix["date"], format="mixed", utc=True)
    return_matrix = matrix[strategy_columns].apply(pd.to_numeric, errors="coerce")
    economic_dates = pd.DatetimeIndex(matrix_dates[1:])
    economic_returns = return_matrix.iloc[1:].reset_index(drop=True)
    equity = 1000.0 * (1.0 + return_matrix).cumprod(axis=0)
    running_peak = equity.cummax().clip(lower=1000.0)
    drawdowns = equity / running_peak - 1.0

    completeness = {
        "as_of": payload.get("as_of"),
        "portfolio_summary_rows": int(len(portfolio)),
        "portfolio_unique_strategy_keys": int(portfolio.strategy_key.nunique()),
        "event_summary_rows": int(len(event)),
        "daily_return_strategy_count": len(strategy_columns),
        "daily_return_date_count": int(len(matrix)),
        "daily_return_economic_date_count": int(len(matrix) - 1),
        "daily_return_date_range": [str(matrix.date.iloc[0]), str(matrix.date.iloc[-1])],
        "economic_date_range": [
            economic_dates[0].date().isoformat(),
            economic_dates[-1].date().isoformat(),
        ],
        "duplicate_dates": int(matrix.date.duplicated().sum()),
        "duplicate_date_strategy_observations": (
            int(matrix.date.duplicated().sum()) * len(strategy_columns)
        ),
        "duplicate_strategy_columns": len(strategy_columns) - len(set(strategy_columns)),
        "manifest_count": int(manifest.get("published_curve_count", -1)),
        "manifest_unique_strategy_count": len(set(manifest_keys)),
        "existing_curve_file_count": len(curve_files),
        "missing_manifest_files": sorted(manifest_files - curve_files),
        "orphan_curve_files": sorted(curve_files - manifest_files),
        "qualified_count": len(qualified_keys),
        "qualified_not_published": sorted(qualified_keys - set(manifest_keys)),
        "primary_key": primary_key,
        "primary_published": primary_key in manifest_keys,
        "required_diagnostic_leader_count": len(required_leaders),
        "diagnostic_leaders_not_published": sorted(required_leaders - set(manifest_keys)),
        "publication_set_mismatch": sorted(set(manifest_keys) ^ expected_publication),
        "event_vs_portfolio_strategy_key_mismatch": sorted(
            set(event.strategy_key) ^ set(portfolio.strategy_key)
        ),
        "event_vs_matrix_strategy_key_mismatch": sorted(
            set(event.strategy_key) ^ set(strategy_columns)
        ),
        "matrix_t0_nonzero_strategy_count": int(
            (return_matrix.iloc[0].abs() > ATOL).sum()
        ),
        "schema_versions": {
            "summary_curve": payload.get("portfolio_curve_schema_version"),
            "manifest_curve": manifest.get("curve_schema_version"),
            "summary_cdar": payload.get("portfolio_cdar_definition_version"),
            "manifest_cdar": manifest.get("cdar_definition_version"),
        },
    }

    size_names = [
        "backtest_summary.json",
        "backtest_strategy_summary.csv",
        "backtest_strategy_year_summary.csv",
        "backtest_portfolio_strategy_summary.csv",
        "backtest_portfolio_curve_manifest.json",
        "backtest_benchmark_spy.json",
        "backtest_portfolio_daily_returns.csv.gz",
        "backtest_recent_trades.csv",
        "backtest_skipped_summary.csv",
    ]
    completeness["file_sizes_bytes"] = {
        name: (DATA / name).stat().st_size for name in size_names
    }
    completeness["curve_files_total_bytes"] = sum(
        (curve_directory / name).stat().st_size for name in curve_files
    )

    published_mismatches: dict[str, list[str]] = {}
    published_max_errors = {
        "accounting": 0.0,
        "drawdown": 0.0,
        "daily_return": 0.0,
        "cumulative_return": 0.0,
        "matrix_return": 0.0,
        "transaction_cost": 0.0,
    }
    published_rows = 0
    published_observed = {
        "min_daily_return": math.inf,
        "max_daily_return": -math.inf,
        "min_gross_exposure": math.inf,
        "max_gross_exposure": -math.inf,
        "min_active_positions": math.inf,
        "max_active_positions": -math.inf,
        "min_cash": math.inf,
        "min_invested_value": math.inf,
    }
    manifest_summary_mismatches: dict[str, list[str]] = {}

    for entry in manifest_entries:
        key = str(entry["strategy_key"])
        curve_payload = json.loads(
            (curve_directory / str(entry["file"])).read_text(encoding="utf-8")
        )
        curve = pd.DataFrame(curve_payload.get("series", []))
        published_rows += len(curve)
        if curve_payload.get("curve_schema_version") != PORTFOLIO_CURVE_SCHEMA_VERSION:
            add_mismatch(published_mismatches, "curve_schema_version", key)
        if curve_payload.get("cdar_definition_version") != CDAR_DEFINITION_VERSION:
            add_mismatch(published_mismatches, "cdar_definition_version", key)
        if curve.empty:
            add_mismatch(published_mismatches, "empty_curve", key)
            continue

        first = curve.iloc[0]
        initial_checks = {
            "observation_type": first.get("observation_type") == "initialization",
            "portfolio_equity": close(first.get("portfolio_equity"), 1000.0),
            "cash_plus_invested": close(
                float(first.get("cash_value")) + float(first.get("invested_value")), 1000.0
            ),
            "cash_value": close(first.get("cash_value"), 1000.0),
            "invested_value": close(first.get("invested_value"), 0.0),
            "gross_exposure": close(first.get("gross_exposure"), 0.0),
            "active_position_count": close(first.get("active_position_count"), 0.0),
            "cumulative_return": close(first.get("cumulative_return"), 0.0),
            "running_peak_equity": close(first.get("running_peak_equity"), 1000.0),
            "drawdown": close(first.get("drawdown"), 0.0),
            "daily_portfolio_return": close(first.get("daily_portfolio_return"), 0.0),
            "transaction_cost_paid": close(first.get("transaction_cost_paid"), 0.0),
            "turnover": close(first.get("turnover"), 0.0),
        }
        for field, passed in initial_checks.items():
            if not passed:
                add_mismatch(published_mismatches, f"t0_{field}", key)

        economic = curve.loc[curve["observation_type"].ne("initialization")].reset_index(drop=True)
        if economic.empty:
            add_mismatch(published_mismatches, "empty_economic_curve", key)
            continue
        curve_dates = pd.to_datetime(curve["date"], format="mixed", utc=True)
        dates = pd.DatetimeIndex(pd.to_datetime(economic["date"], format="mixed", utc=True))
        if not curve_dates.iloc[0] < dates[0]:
            add_mismatch(published_mismatches, "t0_not_before_economic", key)
        if not economic["observation_type"].eq("trading_session").all():
            add_mismatch(published_mismatches, "economic_observation_type", key)

        numeric = {
            column: pd.to_numeric(curve[column], errors="coerce").to_numpy(dtype=float)
            for column in [
                "portfolio_equity", "cash_value", "invested_value", "gross_exposure",
                "active_position_count", "daily_portfolio_return", "cumulative_return",
                "running_peak_equity", "drawdown", "transaction_cost_paid", "turnover",
            ]
        }
        for column, values in numeric.items():
            if not np.isfinite(values).all():
                add_mismatch(published_mismatches, f"non_finite_{column}", key)

        account_error = np.max(np.abs(
            numeric["portfolio_equity"] - numeric["cash_value"] - numeric["invested_value"]
        ))
        published_max_errors["accounting"] = max(
            published_max_errors["accounting"], float(account_error)
        )
        if account_error > ATOL:
            add_mismatch(published_mismatches, "accounting", key)
        if np.min(numeric["cash_value"]) < -ATOL:
            add_mismatch(published_mismatches, "negative_cash", key)
        if np.min(numeric["invested_value"]) < -ATOL:
            add_mismatch(published_mismatches, "negative_invested_value", key)
        if np.min(numeric["gross_exposure"]) < -ATOL or np.max(numeric["gross_exposure"]) > 1 + ATOL:
            add_mismatch(published_mismatches, "gross_exposure_bounds", key)
        positions = numeric["active_position_count"]
        if np.min(positions) < -ATOL or not np.allclose(positions, np.rint(positions), atol=ATOL):
            add_mismatch(published_mismatches, "active_position_count", key)

        reconstructed_peak = np.maximum.accumulate(numeric["portfolio_equity"])
        reconstructed_peak = np.maximum(reconstructed_peak, 1000.0)
        if np.any(np.diff(numeric["running_peak_equity"]) < -ATOL):
            add_mismatch(published_mismatches, "decreasing_running_peak", key)
        if not np.allclose(
            numeric["running_peak_equity"], reconstructed_peak, atol=ATOL, rtol=RTOL
        ):
            add_mismatch(published_mismatches, "running_peak_reconstruction", key)
        reconstructed_drawdown = numeric["portfolio_equity"] / reconstructed_peak - 1.0
        drawdown_error = np.max(np.abs(numeric["drawdown"] - reconstructed_drawdown))
        published_max_errors["drawdown"] = max(
            published_max_errors["drawdown"], float(drawdown_error)
        )
        if drawdown_error > ATOL or np.max(numeric["drawdown"]) > ATOL:
            add_mismatch(published_mismatches, "drawdown", key)

        reconstructed_returns = np.r_[
            0.0,
            numeric["portfolio_equity"][1:] / numeric["portfolio_equity"][:-1] - 1.0,
        ]
        return_error = np.max(np.abs(
            numeric["daily_portfolio_return"] - reconstructed_returns
        ))
        published_max_errors["daily_return"] = max(
            published_max_errors["daily_return"], float(return_error)
        )
        if return_error > ATOL:
            add_mismatch(published_mismatches, "daily_return", key)
        reconstructed_cumulative = numeric["portfolio_equity"] / 1000.0 - 1.0
        cumulative_error = np.max(np.abs(
            numeric["cumulative_return"] - reconstructed_cumulative
        ))
        published_max_errors["cumulative_return"] = max(
            published_max_errors["cumulative_return"], float(cumulative_error)
        )
        if cumulative_error > ATOL:
            add_mismatch(published_mismatches, "cumulative_return", key)
        reconstructed_cost = (
            numeric["turnover"] * np.r_[1000.0, numeric["portfolio_equity"][:-1]] * 0.001
        )
        cost_error = np.max(np.abs(
            numeric["transaction_cost_paid"] - reconstructed_cost
        ))
        published_max_errors["transaction_cost"] = max(
            published_max_errors["transaction_cost"], float(cost_error)
        )
        if cost_error > ATOL:
            add_mismatch(
                published_mismatches,
                "transaction_cost_turnover_reconciliation",
                key,
            )

        if key not in matrix:
            add_mismatch(published_mismatches, "missing_matrix_column", key)
        else:
            matrix_error = np.max(np.abs(
                numeric["daily_portfolio_return"]
                - pd.to_numeric(matrix[key], errors="coerce").to_numpy(dtype=float)
            ))
            published_max_errors["matrix_return"] = max(
                published_max_errors["matrix_return"], float(matrix_error)
            )
            if matrix_error > ATOL:
                add_mismatch(published_mismatches, "matrix_return", key)

        summary_row = portfolio_by_key.loc[key]
        elapsed_days = max(int((dates[-1] - dates[0]).days), 0)
        years = elapsed_days / 365.25 if elapsed_days else 0.0
        economic_equity = pd.to_numeric(
            economic["portfolio_equity"], errors="coerce"
        ).to_numpy(dtype=float)
        economic_returns_for_key = pd.to_numeric(
            economic["daily_portfolio_return"], errors="coerce"
        ).to_numpy(dtype=float)
        economic_drawdown = pd.to_numeric(
            economic["drawdown"], errors="coerce"
        ).to_numpy(dtype=float)
        curve_checks = {
            "portfolio_start_date": str(summary_row.portfolio_start_date) == dates[0].date().isoformat(),
            "portfolio_end_date": str(summary_row.portfolio_end_date) == dates[-1].date().isoformat(),
            "ending_equity": close(summary_row.ending_equity, economic_equity[-1]),
            "economic_return_reconstruction": close(
                economic_equity[-1],
                1000.0 * float(np.prod(1.0 + economic_returns_for_key)),
            ),
            "cagr_timing": close(
                summary_row.cagr,
                (economic_equity[-1] / 1000.0) ** (1.0 / years) - 1.0 if years else 0.0,
            ),
            "maximum_drawdown": close(summary_row.maximum_drawdown, economic_drawdown.min()),
            "conditional_drawdown_at_risk_95": close(
                summary_row.conditional_drawdown_at_risk_95,
                independent_cdar(economic_drawdown),
            ),
            "total_transaction_cost": close(
                summary_row.total_transaction_cost,
                pd.to_numeric(economic["transaction_cost_paid"], errors="coerce").sum(),
            ),
            "annual_turnover": close(
                summary_row.annual_turnover,
                pd.to_numeric(economic["turnover"], errors="coerce").sum() / years if years else 0.0,
            ),
            "average_gross_exposure": close(
                summary_row.average_gross_exposure,
                pd.to_numeric(economic["gross_exposure"], errors="coerce").mean(),
            ),
            "median_gross_exposure": close(
                summary_row.median_gross_exposure,
                pd.to_numeric(economic["gross_exposure"], errors="coerce").median(),
            ),
            "maximum_gross_exposure": close(
                summary_row.maximum_gross_exposure,
                pd.to_numeric(economic["gross_exposure"], errors="coerce").max(),
            ),
            "average_active_positions": close(
                summary_row.average_active_positions,
                pd.to_numeric(economic["active_position_count"], errors="coerce").mean(),
            ),
            "maximum_active_positions": close(
                summary_row.maximum_active_positions,
                pd.to_numeric(economic["active_position_count"], errors="coerce").max(),
            ),
            "percent_days_in_cash": close(
                summary_row.percent_days_in_cash,
                pd.to_numeric(economic["active_position_count"], errors="coerce").eq(0).mean(),
            ),
        }
        episode = drawdown_dates(dates, economic_equity, economic_drawdown)
        for field, expected in episode.items():
            actual = summary_row[field]
            if field.endswith("_date"):
                actual_value = None if pd.isna(actual) else str(actual)
                curve_checks[field] = actual_value == expected
            else:
                curve_checks[field] = close(actual, expected)
        for field, passed in curve_checks.items():
            if not passed:
                add_mismatch(published_mismatches, field, key)

        embedded = entry.get("summary", {})
        for field in portfolio.columns.drop("strategy_key"):
            expected = summary_row[field]
            actual = embedded.get(field)
            passed = (
                (pd.isna(expected) and actual is None)
                or (finite(expected) and finite(actual) and close(expected, actual))
                or (str(expected) == str(actual))
            )
            if not passed:
                add_mismatch(manifest_summary_mismatches, field, key)

        published_observed["min_daily_return"] = min(
            published_observed["min_daily_return"], float(np.min(economic_returns_for_key))
        )
        published_observed["max_daily_return"] = max(
            published_observed["max_daily_return"], float(np.max(economic_returns_for_key))
        )
        for result_name, column, reducer in [
            ("min_gross_exposure", "gross_exposure", np.min),
            ("max_gross_exposure", "gross_exposure", np.max),
            ("min_active_positions", "active_position_count", np.min),
            ("max_active_positions", "active_position_count", np.max),
            ("min_cash", "cash_value", np.min),
            ("min_invested_value", "invested_value", np.min),
        ]:
            published_observed[result_name] = (
                min(published_observed[result_name], float(reducer(numeric[column])))
                if result_name.startswith("min_")
                else max(published_observed[result_name], float(reducer(numeric[column])))
            )

    # Reconstruct all path-observable summary fields from the 540-column matrix.
    all_strategy_mismatches: dict[str, list[str]] = {}
    max_discrepancy = 0.0
    weekly = (1.0 + economic_returns.set_axis(economic_dates)).resample("W-FRI").prod() - 1.0
    monthly = (1.0 + economic_returns.set_axis(economic_dates)).resample("ME").prod() - 1.0
    elapsed_days = int((economic_dates[-1] - economic_dates[0]).days)
    years = elapsed_days / 365.25
    path_fields = [
        "ending_equity", "total_portfolio_return", "cagr", "annualized_volatility",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "maximum_drawdown",
        "ulcer_index", "conditional_drawdown_at_risk_95", "worst_daily_return",
        "worst_weekly_return", "worst_monthly_return", "daily_expected_shortfall_95",
        "daily_expected_shortfall_99", "daily_return_skewness",
        "daily_return_excess_kurtosis",
    ]
    for key in strategy_columns:
        returns = economic_returns[key].to_numpy(dtype=float)
        values = equity[key].iloc[1:].to_numpy(dtype=float)
        dd = drawdowns[key].iloc[1:].to_numpy(dtype=float)
        std = float(pd.Series(returns).std(ddof=1))
        downside = float(np.sqrt(np.mean(np.square(np.minimum(returns, 0.0)))))
        ending = float(values[-1])
        total = ending / 1000.0 - 1.0
        cagr = (ending / 1000.0) ** (1.0 / years) - 1.0
        mdd = float(np.min(dd))
        calculated = {
            "ending_equity": ending,
            "total_portfolio_return": total,
            "cagr": cagr,
            "annualized_volatility": std * math.sqrt(252.0),
            "sharpe_ratio": (
                float(np.mean(returns) / std * math.sqrt(252.0)) if std > 0 else None
            ),
            "sortino_ratio": (
                float(np.mean(returns) / downside * math.sqrt(252.0)) if downside > 0 else None
            ),
            "calmar_ratio": cagr / abs(mdd) if mdd < 0 else None,
            "maximum_drawdown": mdd,
            "ulcer_index": float(np.sqrt(np.mean(np.square(dd)))),
            "conditional_drawdown_at_risk_95": independent_cdar(dd),
            "worst_daily_return": float(np.min(returns)),
            "worst_weekly_return": float(weekly[key].min()),
            "worst_monthly_return": float(monthly[key].min()),
            "daily_expected_shortfall_95": expected_shortfall(returns, 0.95),
            "daily_expected_shortfall_99": expected_shortfall(returns, 0.99),
            "daily_return_skewness": float(pd.Series(returns).skew()),
            "daily_return_excess_kurtosis": float(pd.Series(returns).kurt()),
        }
        summary_row = portfolio_by_key.loc[key]
        if str(summary_row.portfolio_start_date) != economic_dates[0].date().isoformat():
            add_mismatch(all_strategy_mismatches, "portfolio_start_date", key)
        if str(summary_row.portfolio_end_date) != economic_dates[-1].date().isoformat():
            add_mismatch(all_strategy_mismatches, "portfolio_end_date", key)
        for field in path_fields:
            actual = summary_row[field]
            expected = calculated[field]
            if not close(actual, expected):
                add_mismatch(all_strategy_mismatches, field, key)
            elif finite(actual) and finite(expected):
                max_discrepancy = max(max_discrepancy, abs(float(actual) - float(expected)))
        episode = drawdown_dates(economic_dates, values, dd)
        for field, expected in episode.items():
            actual = summary_row[field]
            passed = (
                (None if pd.isna(actual) else str(actual)) == expected
                if field.endswith("_date")
                else close(actual, expected)
            )
            if not passed:
                add_mismatch(all_strategy_mismatches, field, key)

    summary_cdar = portfolio_by_key.loc[strategy_columns, "conditional_drawdown_at_risk_95"].astype(float)
    independent_cdars = pd.Series({
        key: independent_cdar(drawdowns[key].iloc[1:].to_numpy(dtype=float))
        for key in strategy_columns
    })
    cdar_discrepancies = (summary_cdar - independent_cdars).abs()
    production_cdar_discrepancies = pd.Series({
        key: abs(
            conditional_drawdown_at_risk(drawdowns[key].iloc[1:], 0.95)
            - independent_cdars[key]
        )
        for key in strategy_columns
    })

    gate_fields = pd.DataFrame([
        _gate_fields({
            "completed_trades": row.completed_trades,
            "profit_factor": row.profit_factor,
            "avg_trade_return": row.avg_trade_return,
            "median_trade_return": row.median_trade_return,
            "trade_win_rate": row.trade_win_rate,
        })
        for row in event.itertuples(index=False)
    ])
    reconstructed = pd.concat([
        event.drop(
            columns=[column for column in gate_fields.columns if column in event],
            errors="ignore",
        ).reset_index(drop=True),
        gate_fields,
    ], axis=1)
    reconstructed = _attach_time_stability(reconstructed, annual)
    reconstructed = _attach_parameter_stability(reconstructed, annual)
    reconstructed = _attach_robustness_tiers(reconstructed)
    reconstructed = reconstructed.loc[:, ~reconstructed.columns.duplicated(keep="last")]
    reconstructed = rank_strategy_summary(reconstructed)
    stored_ranked = event.sort_values(["qualification_rank", "strategy_key"]).reset_index(drop=True)
    gate_mismatches: dict[str, list[str]] = {}
    gate_compare_fields = [
        "sample_gate_pass", "edge_gate_pass", "time_gate_pass", "parameter_gate_pass",
        "mandatory_gates_pass", "qualification_tier", "qualification_rank",
    ]
    stored_by_key = event.set_index("strategy_key")
    reconstructed_by_key = reconstructed.set_index("strategy_key")
    for key in event.strategy_key:
        for field in gate_compare_fields:
            if stored_by_key.loc[key, field] != reconstructed_by_key.loc[key, field]:
                add_mismatch(gate_mismatches, field, str(key))
    conjunction = (
        reconstructed.sample_gate_pass
        & reconstructed.edge_gate_pass
        & reconstructed.time_gate_pass
        & reconstructed.parameter_gate_pass
    )
    shuffled = reconstructed.sample(frac=1, random_state=20260727)
    shuffled_reranked = rank_strategy_summary(shuffled)
    qualification = {
        "qualified_count": int(event.qualification_tier.eq("Qualified").sum()),
        "reconstructed_qualified_key_mismatch": sorted(
            qualified_keys ^ set(reconstructed.loc[conjunction, "strategy_key"])
        ),
        "gate_field_mismatches": gate_mismatches,
        "conjunction_mismatch_count": int(
            (reconstructed.mandatory_gates_pass.ne(conjunction)).sum()
        ),
        "tier_mismatch_count": int(
            (reconstructed.qualification_tier.eq("Qualified").ne(conjunction)).sum()
        ),
        "rank_order_mismatch_count": int(
            sum(
                left != right
                for left, right in zip(
                    reconstructed.strategy_key.tolist(),
                    stored_ranked.strategy_key.tolist(),
                )
            )
        ),
        "contiguous_ranks": reconstructed.qualification_rank.tolist()
        == list(range(1, len(reconstructed) + 1)),
        "primary_key": str(reconstructed.iloc[0].strategy_key),
        "shuffled_rank_order_mismatch_count": int(
            sum(
                left != right
                for left, right in zip(
                    reconstructed.strategy_key.tolist(),
                    shuffled_reranked.strategy_key.tolist(),
                )
            )
        ),
    }

    benchmark_series = pd.DataFrame(benchmark.get("series", []))
    benchmark_dates = pd.DatetimeIndex(
        pd.to_datetime(benchmark_series["date"], format="mixed", utc=True)
    )
    benchmark_equity = pd.to_numeric(
        benchmark_series["benchmark_equity"], errors="coerce"
    ).to_numpy(dtype=float)
    benchmark_returns = pd.to_numeric(
        benchmark_series["benchmark_daily_return"], errors="coerce"
    ).to_numpy(dtype=float)
    benchmark_drawdown = pd.to_numeric(
        benchmark_series["benchmark_drawdown"], errors="coerce"
    ).to_numpy(dtype=float)
    benchmark_peak = np.maximum.accumulate(benchmark_equity)
    benchmark_reconstructed_returns = np.r_[
        0.0, benchmark_equity[1:] / benchmark_equity[:-1] - 1.0
    ]
    benchmark_reconstructed_drawdown = benchmark_equity / benchmark_peak - 1.0
    benchmark_result = {
        "status": benchmark.get("status"),
        "schema_version": benchmark.get("curve_schema_version"),
        "cdar_definition_version": benchmark.get("cdar_definition_version"),
        "row_count": len(benchmark_series),
        "economic_row_count": int(
            benchmark_series.observation_type.ne("initialization").sum()
        ),
        "first_observation_type": benchmark_series.iloc[0].observation_type,
        "first_equity": float(benchmark_equity[0]),
        "first_return": float(benchmark_returns[0]),
        "first_drawdown": float(benchmark_drawdown[0]),
        "t0_before_economic": bool(benchmark_dates[0] < benchmark_dates[1]),
        "economic_date_alignment": (
            benchmark_series.loc[
                benchmark_series.observation_type.ne("initialization"), "date"
            ].tolist()
            == [date.date().isoformat() for date in economic_dates]
        ),
        "return_reconstruction_max_error": float(
            np.max(np.abs(benchmark_returns - benchmark_reconstructed_returns))
        ),
        "drawdown_reconstruction_max_error": float(
            np.max(np.abs(benchmark_drawdown - benchmark_reconstructed_drawdown))
        ),
        "price_convention": benchmark.get("price_convention"),
        "missing_value_count": int(
            benchmark_series[[
                "benchmark_equity", "benchmark_daily_return", "benchmark_drawdown"
            ]].isna().sum().sum()
        ),
        "stress_observations": {
            "below_minus_10": int(np.sum(benchmark_drawdown <= -0.10)),
            "below_minus_15": int(np.sum(benchmark_drawdown <= -0.15)),
            "below_minus_20": int(np.sum(benchmark_drawdown <= -0.20)),
        },
    }

    portfolio_numeric = portfolio.select_dtypes(include=[np.number])
    anomaly = {
        "ending_equity_range": [
            float(portfolio.ending_equity.min()),
            float(portfolio.ending_equity.max()),
        ],
        "cagr_range": [float(portfolio.cagr.min()), float(portfolio.cagr.max())],
        "maximum_drawdown_range": [
            float(portfolio.maximum_drawdown.min()),
            float(portfolio.maximum_drawdown.max()),
        ],
        "cdar_range": [
            float(portfolio.conditional_drawdown_at_risk_95.min()),
            float(portfolio.conditional_drawdown_at_risk_95.max()),
        ],
        "daily_return_range": [
            float(economic_returns.min().min()),
            float(economic_returns.max().max()),
        ],
        "published_observed": published_observed,
        "maximum_gross_exposure_summary_range": [
            float(portfolio.maximum_gross_exposure.min()),
            float(portfolio.maximum_gross_exposure.max()),
        ],
        "maximum_active_positions_summary_range": [
            int(portfolio.maximum_active_positions.min()),
            int(portfolio.maximum_active_positions.max()),
        ],
        "annual_turnover_range": [
            float(portfolio.annual_turnover.min()),
            float(portfolio.annual_turnover.max()),
        ],
        "total_transaction_cost_range": [
            float(portfolio.total_transaction_cost.min()),
            float(portfolio.total_transaction_cost.max()),
        ],
        "non_finite_summary_strategies": sorted(
            portfolio.loc[
                ~np.isfinite(portfolio_numeric).all(axis=1), "strategy_key"
            ].astype(str).tolist()
        ),
        "non_finite_matrix_strategies": sorted(
            [
                key for key in strategy_columns
                if not np.isfinite(return_matrix[key].to_numpy(dtype=float)).all()
            ]
        ),
        "ending_equity_non_positive": sorted(
            portfolio.loc[portfolio.ending_equity.le(0), "strategy_key"].astype(str)
        ),
        "daily_return_at_or_below_minus_one": sorted(
            [
                key for key in strategy_columns
                if economic_returns[key].min() <= -1.0
            ]
        ),
        "empty_economic_series": sorted(
            [
                key for key in strategy_columns
                if economic_returns[key].dropna().empty
            ]
        ),
    }

    return {
        "tolerance": {"absolute": ATOL, "relative": RTOL},
        "completeness": completeness,
        "published_curve_rows": published_rows,
        "published_curve_mismatches": published_mismatches,
        "published_curve_max_errors": published_max_errors,
        "manifest_summary_mismatches": manifest_summary_mismatches,
        "all_strategy_path_mismatches": all_strategy_mismatches,
        "all_strategy_path_max_absolute_discrepancy": max_discrepancy,
        "cdar": {
            "strategies_tested": len(strategy_columns),
            "summary_mismatch_count": int((cdar_discrepancies > ATOL).sum()),
            "summary_mismatch_keys": sorted(cdar_discrepancies[cdar_discrepancies > ATOL].index),
            "published_curve_mismatch_count": len(
                published_mismatches.get("conditional_drawdown_at_risk_95", [])
            ),
            "production_vs_independent_mismatch_count": int(
                (production_cdar_discrepancies > ATOL).sum()
            ),
            "maximum_summary_absolute_discrepancy": float(cdar_discrepancies.max()),
            "maximum_production_absolute_discrepancy": float(
                production_cdar_discrepancies.max()
            ),
        },
        "qualification": qualification,
        "benchmark": benchmark_result,
        "anomaly": anomaly,
        "bounded_limitations": [
            "Duplicate concurrent symbols cannot be reconstructed from aggregate curve rows.",
            "Cash, invested value, exposure, active positions, turnover, and costs are independently observable only for published curves.",
            "Full raw completed/skipped event records are intentionally not committed.",
        ],
    }


def critical_failures(result: dict[str, Any]) -> list[str]:
    completeness = result["completeness"]
    qualification = result["qualification"]
    benchmark = result["benchmark"]
    failures = []
    for field in [
        "duplicate_dates", "duplicate_date_strategy_observations",
        "duplicate_strategy_columns", "matrix_t0_nonzero_strategy_count",
    ]:
        if completeness[field]:
            failures.append(field)
    for field in [
        "missing_manifest_files", "orphan_curve_files", "qualified_not_published",
        "diagnostic_leaders_not_published", "publication_set_mismatch",
        "event_vs_portfolio_strategy_key_mismatch",
        "event_vs_matrix_strategy_key_mismatch",
    ]:
        if completeness[field]:
            failures.append(field)
    if result["published_curve_mismatches"]:
        failures.append("published_curve_mismatches")
    if result["manifest_summary_mismatches"]:
        failures.append("manifest_summary_mismatches")
    if result["all_strategy_path_mismatches"]:
        failures.append("all_strategy_path_mismatches")
    for field in [
        "summary_mismatch_count", "published_curve_mismatch_count",
        "production_vs_independent_mismatch_count",
    ]:
        if result["cdar"][field]:
            failures.append(f"cdar.{field}")
    for field in [
        "conjunction_mismatch_count", "tier_mismatch_count",
        "rank_order_mismatch_count", "shuffled_rank_order_mismatch_count",
    ]:
        if qualification[field]:
            failures.append(f"qualification.{field}")
    if qualification["reconstructed_qualified_key_mismatch"]:
        failures.append("qualification.reconstructed_qualified_key_mismatch")
    if qualification["gate_field_mismatches"]:
        failures.append("qualification.gate_field_mismatches")
    if not qualification["contiguous_ranks"]:
        failures.append("qualification.contiguous_ranks")
    for field in ["return_reconstruction_max_error", "drawdown_reconstruction_max_error"]:
        if benchmark[field] > ATOL:
            failures.append(f"benchmark.{field}")
    if not benchmark["economic_date_alignment"]:
        failures.append("benchmark.economic_date_alignment")
    return failures


if __name__ == "__main__":
    result = audit()
    result["audit_failures"] = critical_failures(result)
    result["audit_passed"] = not result["audit_failures"]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["audit_passed"] else 1)
