"""Synthetic-only Foundation 3 registry and read-only API demonstration."""

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
    AttemptOperationalStatus,
    AttemptTerminalOutcome,
    ExecutionAttempt,
    FileExecutionAttemptRepository,
    LocalResultStore,
    ReadOnlyTrendApi,
    SavedRunRegistryBuilder,
    StrategyRunManifest,
    StrategyRunSpec,
    calculate_and_evaluate_saved_runs,
    load_evaluation_profiles,
    load_terminology_source,
)


CREATED_AT = "2026-08-01T00:00:00Z"


def policy() -> ArtifactRetentionPolicy:
    return ArtifactRetentionPolicy(
        max_store_bytes=30_000_000,
        max_artifact_bytes=5_000_000,
        max_strategy_runs=20,
        max_evaluation_runs=20,
    )


def curve(returns: list[float], exposure: float = 0.8) -> dict:
    dates = [item.date().isoformat() for item in pd.bdate_range("2024-01-02", periods=len(returns))]
    values = 1_000.0 * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    return {
        "schema_version": "daily_portfolio_curve_v1",
        "economic_date_range": {"start": dates[0], "end": dates[-1]},
        "rows": [
            {
                "economic_date": economic_date,
                "portfolio_value": float(value),
                "daily_return": float(daily_return),
                "gross_exposure": exposure,
                "net_exposure": exposure,
                "cash_weight": 1.0 - exposure,
                "daily_turnover": 0.01,
                "transaction_cost": 0.01,
                "position_count": 3,
            }
            for economic_date, value, daily_return in zip(dates, values, returns)
        ],
    }


def strategy_spec(threshold: float, date_range: dict, snapshot_character: str) -> StrategyRunSpec:
    return StrategyRunSpec(
        data_snapshot_hash=snapshot_character * 64,
        economic_date_range=date_range,
        universe_specification={"name": "synthetic"},
        benchmark={"symbol": "SPY", "identity": "synthetic_spy_v1"},
        trend_filter={"family": "price_above_ma", "days": 20},
        signal={"family": "synthetic_breakout", "threshold": threshold},
        entry_rule={"family": "next_open"},
        initial_stop={"family": "atr", "multiple": 2.0},
        trailing_exit={"family": "channel", "days": 10},
        position_sizing={"family": "equal"},
        portfolio_constraints={"max_positions": 5},
        transaction_costs={"bps": 5},
        slippage={"bps": 2},
        engine_version="synthetic_engine_v1",
    )


def robustness() -> dict:
    return {
        "schema_version": "robustness_summary_v1",
        "walk_forward_fold_count": 3,
        "walk_forward_pass_ratio": 2 / 3,
        "walk_forward_worst_fold": -0.01,
        "loyo_case_count": 2,
        "loyo_stability_ratio": 1.0,
        "loyo_reversing_years": [],
        "block_bootstrap_effect": 0.02,
        "bootstrap_confidence_interval": {
            "lower": 0.001,
            "upper": 0.04,
            "confidence_level": 0.95,
        },
        "raw_p_value": 0.02,
        "adjusted_p_value": 0.04,
        "multiple_testing_method": "holm",
        "transaction_cost_stress_survival": 1.0,
        "dominant_asset_group": "equity",
        "dominant_group_share": 0.5,
        "unclassified_group_share": 0.0,
        "method_metadata": {"seeds": [11], "sample_counts": {"bootstrap": 100}},
        "unavailable_reasons": {},
    }


def save_run(
    store: LocalResultStore,
    daily: dict,
    benchmark: dict,
    *,
    threshold: float,
    snapshot_character: str,
) -> tuple[StrategyRunManifest, object]:
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
    robustness_record = store.put_artifact(
        "robustness_summary", ArtifactKind.ROBUSTNESS_SUMMARY, robustness(), row_count=1
    ).record
    manifest = StrategyRunManifest.create(
        strategy_spec(threshold, daily["economic_date_range"], snapshot_character),
        source_code_commit="c" * 40,
        artifacts=(daily_record, benchmark_record, robustness_record),
        creation_time=CREATED_AT,
        warnings=("synthetic_demonstration_only",),
    )
    store.save_strategy_run(manifest)
    return manifest, daily_record


def artifact_error(mode: str, root: Path, terminology: dict) -> dict:
    store = LocalResultStore(root / mode, policy())
    payload = curve([0.0, 0.01, -0.005])
    record = store.put_artifact(
        "daily_portfolio_curve",
        ArtifactKind.DAILY_PORTFOLIO_CURVE,
        payload,
        row_count=len(payload["rows"]),
    ).record
    manifest = StrategyRunManifest.create(
        strategy_spec(
            2.0,
            payload["economic_date_range"],
            {"missing": "1", "corrupt": "2", "pruned": "3"}[mode],
        ),
        source_code_commit="d" * 40,
        artifacts=(record,),
        creation_time=CREATED_AT,
    )
    store.save_strategy_run(manifest)
    path = store.object_path_for_hash(record.content_hash)
    if mode == "missing":
        path.unlink()
    elif mode == "corrupt":
        path.write_bytes(b"invalid-gzip")
    else:
        store.mark_artifact_pruned(
            record.content_hash,
            pruned_at="2026-08-01T01:00:00Z",
            reason="synthetic_demonstration",
        )
    response = ReadOnlyTrendApi(store, terminology_source=terminology).dispatch(
        "GET", f"/api/v1/runs/{manifest.strategy_run_id}/curve"
    )
    return {"http_status": response.status_code, "error": response.body["error"]}


def main() -> None:
    economic_backtest_calls = 0
    profiles = load_evaluation_profiles(ROOT / "config" / "trend_v2" / "evaluation_profiles")
    terminology = load_terminology_source(ROOT / "config" / "trend_v2" / "terminology_ko.json")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = LocalResultStore(root / "main", policy())
        benchmark_returns = [0.0002] * 90
        benchmark_returns[30] = -0.04
        benchmark = curve(benchmark_returns, 1.0)
        first_returns = [0.00035] * 90
        first_returns[30] = -0.015
        second_returns = [0.00030] * 90
        second_returns[30] = -0.02
        first, _ = save_run(
            store,
            curve(first_returns),
            benchmark,
            threshold=1.0,
            snapshot_character="a",
        )
        second, _ = save_run(
            store,
            curve(second_returns, 0.7),
            benchmark,
            threshold=1.1,
            snapshot_character="b",
        )
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

        attempts = FileExecutionAttemptRepository(store.root / "execution_attempts")
        attempt = ExecutionAttempt.create(
            StrategyRunSpec.from_dict(first.canonical_specification),
            attempt_number=1,
            created_timestamp=CREATED_AT,
            source_commit="c" * 40,
            engine_version="synthetic_engine_v1",
        )
        attempts.save(attempt)
        attempt = attempts.transition(
            attempt.execution_attempt_id,
            operational_status=AttemptOperationalStatus.RUNNING,
            started_timestamp="2026-08-01T00:00:01Z",
        )
        attempt = attempts.transition(
            attempt.execution_attempt_id,
            operational_status=AttemptOperationalStatus.COMPLETED,
            terminal_outcome=AttemptTerminalOutcome.SUCCEEDED,
            completed_timestamp="2026-08-01T00:00:02Z",
        )

        builder = SavedRunRegistryBuilder(store, attempts)
        rebuilt_first = builder.rebuild()
        rebuilt_second = builder.rebuild()
        api = ReadOnlyTrendApi(
            store,
            registry_builder=builder,
            attempt_repository=attempts,
            terminology_source=terminology,
        )
        run_list = api.dispatch("GET", "/api/v1/runs")
        run_detail = api.dispatch("GET", f"/api/v1/runs/{first.strategy_run_id}")
        provenance = api.dispatch("GET", f"/api/v1/runs/{first.strategy_run_id}/provenance")
        evaluations = api.dispatch(
            "GET", f"/api/v1/evaluation-runs?strategy_run_id={first.strategy_run_id}"
        )
        derived = api.dispatch("GET", f"/api/v1/runs/{first.strategy_run_id}/derived-metrics")
        attempt_response = api.dispatch(
            "GET", f"/api/v1/execution-attempts/{attempt.execution_attempt_id}"
        )
        manifest_response = api.dispatch(
            "GET", f"/api/v1/runs/{first.strategy_run_id}/manifest"
        )
        output = {
            "registry_reconstruction": {
                "equivalent": rebuilt_first.to_dict() == rebuilt_second.to_dict(),
                "registry_id": rebuilt_first.registry_id,
            },
            "saved_strategy_runs": [item["strategy_run_id"] for item in run_list.body["items"]],
            "one_run": {
                "strategy_run_id": run_detail.body["strategy_run_id"],
                "integrity_status": run_detail.body["integrity_status"],
                "manifest_hash": provenance.body["manifest_hash"],
                "artifact_hash_count": len(provenance.body["artifact_hashes"]),
            },
            "two_evaluations_for_same_run": [
                {
                    "evaluation_run_id": item["evaluation_run_id"],
                    "evaluation_profile_id": item["evaluation_profile_id"],
                    "profile_hash": item["profile_hash"],
                }
                for item in evaluations.body["items"]
            ],
            "derived_artifact_read": {
                "http_status": derived.status_code,
                "schema_version": derived.body["payload"]["schema_version"],
                "first_cache_status": research.cache_status,
                "second_cache_status": weighted.cache_status,
                "economic_backtest_calls": economic_backtest_calls,
            },
            "separate_execution_lifecycle": {
                "execution_attempt_id": attempt_response.body["execution_attempt_id"],
                "operational_status": attempt_response.body["operational_status"],
                "immutable_strategy_run_status": manifest_response.body["execution_status"],
            },
            "artifact_error_distinctions": {
                mode: artifact_error(mode, root / "errors", terminology)
                for mode in ("missing", "corrupt", "pruned")
            },
        }
        print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
