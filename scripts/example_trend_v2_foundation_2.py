"""Synthetic-only Foundation 2 integration example; writes no repository data."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trend_v2_foundation import (
    ArtifactKind,
    ArtifactRetentionPolicy,
    ExecutionStatus,
    LocalResultStore,
    StrategyRunManifest,
    StrategyRunSpec,
    calculate_and_evaluate_saved_runs,
    load_evaluation_profiles,
    load_terminology_source,
)


CREATED_AT = "2026-08-01T00:00:00Z"


def synthetic_curve(returns: list[float], exposure: float) -> dict:
    dates = [value.date().isoformat() for value in pd.bdate_range("2020-01-02", periods=len(returns))]
    values = 1_000.0 * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    rows = [
        {
            "economic_date": economic_date,
            "portfolio_value": float(value),
            "daily_return": float(daily_return),
            "gross_exposure": exposure,
            "net_exposure": exposure,
            "cash_weight": 1.0 - exposure,
            "daily_turnover": 0.01,
            "transaction_cost": 0.05,
            "position_count": 4,
        }
        for economic_date, value, daily_return in zip(dates, values, returns)
    ]
    return {
        "schema_version": "daily_portfolio_curve_v1",
        "economic_date_range": {"start": dates[0], "end": dates[-1]},
        "rows": rows,
    }


def strategy_spec(threshold: float, date_range: dict) -> StrategyRunSpec:
    return StrategyRunSpec(
        data_snapshot_hash="a" * 64,
        economic_date_range=date_range,
        universe_specification={"name": "synthetic"},
        benchmark={"symbol": "SPY", "identity": "synthetic_spy_v1"},
        trend_filter={"family": "synthetic"},
        signal={"family": "synthetic", "threshold": threshold},
        entry_rule={"family": "next_open"},
        initial_stop={"family": "atr"},
        trailing_exit={"family": "atr"},
        position_sizing={"family": "equal"},
        portfolio_constraints={"max_positions": 5},
        transaction_costs={"bps": 5},
        slippage={"bps": 2},
        engine_version="synthetic_artifact_writer_v1",
    )


def save_run(
    store: LocalResultStore, threshold: float, daily: dict, benchmark: dict
) -> StrategyRunManifest:
    daily_record = store.put_artifact(
        "daily_portfolio_curve",
        ArtifactKind.DAILY_PORTFOLIO_CURVE,
        daily,
        row_count=len(daily["rows"]),
    ).record
    benchmark_record = store.put_artifact(
        "benchmark_daily_portfolio_curve",
        ArtifactKind.DAILY_PORTFOLIO_CURVE,
        benchmark,
        row_count=len(benchmark["rows"]),
    ).record
    manifest = StrategyRunManifest.create(
        strategy_spec(threshold, daily["economic_date_range"]),
        source_code_commit="b" * 40,
        artifacts=(daily_record, benchmark_record),
        creation_time=CREATED_AT,
        execution_status=ExecutionStatus.SUCCEEDED,
        warnings=("synthetic_example_only",),
    )
    store.save_strategy_run(manifest)
    return manifest


def main() -> None:
    economic_backtest_calls = 0
    benchmark_returns = [0.0002] * 800
    benchmark_returns[300] = -0.10
    first_returns = [0.0003] * 800
    first_returns[300] = -0.03
    second_returns = [0.00028] * 800
    second_returns[300] = -0.04
    benchmark = synthetic_curve(benchmark_returns, 1.0)
    profiles = load_evaluation_profiles(ROOT / "config" / "trend_v2" / "evaluation_profiles")
    terminology = load_terminology_source(ROOT / "config" / "trend_v2" / "terminology_ko.json")
    policy = ArtifactRetentionPolicy(
        max_store_bytes=20_000_000,
        max_artifact_bytes=5_000_000,
        max_strategy_runs=10,
        max_evaluation_runs=10,
    )
    with tempfile.TemporaryDirectory() as directory:
        store = LocalResultStore(directory, policy)
        first = save_run(store, 1.0, synthetic_curve(first_returns, 0.8), benchmark)
        second = save_run(store, 1.1, synthetic_curve(second_returns, 0.7), benchmark)
        run_ids = (first.strategy_run_id, second.strategy_run_id)
        research = calculate_and_evaluate_saved_runs(
            store, run_ids, profiles["research_default"], creation_time=CREATED_AT
        )
        weighted = calculate_and_evaluate_saved_runs(
            store,
            run_ids,
            profiles["exploratory_weighted_example"],
            creation_time="2026-08-01T00:01:00Z",
        )
        output = {
            "strategy_run_ids": run_ids,
            "research_default": research.evaluation_run.to_dict(),
            "exploratory_weighted_example": weighted.evaluation_run.to_dict(),
            "derived_metric_ids_equal": research.derived_metric_ids == weighted.derived_metric_ids,
            "first_cache_status": research.cache_status,
            "second_cache_status": weighted.cache_status,
            "economic_backtest_calls": economic_backtest_calls,
            "derived_artifact_count": len(store.derived_metric_history()),
            "korean_metric_metadata": terminology["entries"],
        }
        print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
