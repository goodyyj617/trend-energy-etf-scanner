"""Synthetic-only Foundation 4 API and Korean web UI demonstration."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.example_trend_v2_foundation_3 import (  # noqa: E402
    CREATED_AT,
    curve,
    policy,
    save_run,
    strategy_spec,
)
from src.trend_v2_foundation import (  # noqa: E402
    ApiServerConfig,
    ArtifactKind,
    AttemptOperationalStatus,
    AttemptTerminalOutcome,
    ExecutionAttempt,
    FileExecutionAttemptRepository,
    LocalResultStore,
    ReadOnlyTrendApi,
    StrategyRunManifest,
    StrategyRunSpec,
    TrendWebApplication,
    build_web_server,
    calculate_and_evaluate_saved_runs,
    load_evaluation_profiles,
    load_terminology_source,
)


def _state_run(store: LocalResultStore, state: str, index: int) -> str:
    payload = curve([0.0001 * (index + 1), 0.001, -0.0002, 0.0003])
    spec = strategy_spec(2.0 + index / 10, payload["economic_date_range"], str(index + 2))
    artifacts = ()
    record = None
    if state != "never_generated":
        record = store.put_artifact(
            "daily_portfolio_curve",
            ArtifactKind.DAILY_PORTFOLIO_CURVE,
            payload,
            row_count=len(payload["rows"]),
        ).record
        artifacts = (record,)
    manifest = StrategyRunManifest.create(
        spec,
        source_code_commit="d" * 40,
        artifacts=artifacts,
        creation_time=f"2026-08-01T00:0{index + 2}:00Z",
        warnings=("synthetic_demonstration_only", state),
    )
    store.save_strategy_run(manifest)
    if record is not None and state == "missing":
        store.object_path_for_hash(record.content_hash).unlink()
    elif record is not None and state == "corrupt":
        store.object_path_for_hash(record.content_hash).write_bytes(b"synthetic-corrupt")
    elif record is not None and state == "pruned":
        store.mark_artifact_pruned(
            record.content_hash,
            pruned_at="2026-08-01T01:00:00Z",
            reason="synthetic_demonstration",
        )
    return manifest.strategy_run_id


def build_demo(root: Path, *, port: int) -> tuple[ReadOnlyTrendApi, dict[str, object]]:
    store = LocalResultStore(root, policy())
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
    profiles = load_evaluation_profiles(ROOT / "config" / "trend_v2" / "evaluation_profiles")
    research = calculate_and_evaluate_saved_runs(
        store,
        (first.strategy_run_id, second.strategy_run_id),
        profiles["research_default"],
        creation_time=CREATED_AT,
    )
    weighted = calculate_and_evaluate_saved_runs(
        store,
        (first.strategy_run_id, second.strategy_run_id),
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
        current_stage="stored_artifact_validation",
        progress_summary={"completed_units": 1, "total_units": 2},
    )
    attempt = attempts.transition(
        attempt.execution_attempt_id,
        operational_status=AttemptOperationalStatus.COMPLETED,
        terminal_outcome=AttemptTerminalOutcome.SUCCEEDED,
        completed_timestamp="2026-08-01T00:00:02Z",
        current_stage="complete",
        progress_summary={"completed_units": 2, "total_units": 2},
    )
    state_runs = {
        state: _state_run(store, state, index)
        for index, state in enumerate(("missing", "corrupt", "pruned", "never_generated"))
    }
    terminology = load_terminology_source(ROOT / "config" / "trend_v2" / "terminology_ko.json")
    api = ReadOnlyTrendApi(
        store,
        attempt_repository=attempts,
        terminology_source=terminology,
        server_config=ApiServerConfig(port=port),
    )
    identities = {
        "primary_strategy_run_id": first.strategy_run_id,
        "comparison_strategy_run_id": second.strategy_run_id,
        "evaluation_run_ids": [
            research.evaluation_run.evaluation_run_id,
            weighted.evaluation_run.evaluation_run_id,
        ],
        "execution_attempt_id": attempt.execution_attempt_id,
        "artifact_state_run_ids": state_runs,
        "store_root": str(store.root),
    }
    return api, identities


def request_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="검토용 서버를 종료할 때까지 실행")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory()
    try:
        api, ids = build_demo(Path(temporary.name), port=args.port if args.serve else 0)
        server = build_web_server(TrendWebApplication(api))
        host, port = server.server_address
        address = f"http://{host}:{port}"
        if args.serve:
            print(f"합성 Foundation 4 UI: {address}/")
            print(json.dumps(ids, ensure_ascii=False, indent=2))
            server.serve_forever()
            return
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"{address}/", timeout=5) as response:
                ui_started = response.status == 200 and "저장된 전략 실행" in response.read().decode("utf-8")
            overview = request_json(f"{address}/api/v1/overview")
            run_id = ids["primary_strategy_run_id"]
            run_list = request_json(f"{address}/api/v1/runs?page_size=20")
            curve_page = request_json(f"{address}/api/v1/runs/{run_id}/curve?page_size=25")
            derived = request_json(f"{address}/api/v1/runs/{run_id}/derived-metrics")
            evaluations = request_json(
                f"{address}/api/v1/evaluation-runs?strategy_run_id={run_id}&page_size=20"
            )
            outputs = [
                request_json(f"{address}/api/v1/evaluation-runs/{evaluation_id}/outputs?page_size=20")
                for evaluation_id in ids["evaluation_run_ids"]
            ]
            attempts = request_json(
                f"{address}/api/v1/execution-attempts?intended_strategy_run_id={run_id}"
            )
            terminology = request_json(f"{address}/api/v1/terminology")
            demonstration = {
                "local_start": {"ui": ui_started, "api": overview["versions"]["api_version"]},
                "overview_registry_metadata": overview["last_registry_rebuild_identity"],
                "saved_run_listed_and_openable": run_id in {
                    item["strategy_run_id"] for item in run_list["items"]
                },
                "bounded_curve_and_derived_metrics": {
                    "requested_page_size": 25,
                    "returned_rows": curve_page["page"]["returned"],
                    "derived_schema": derived["payload"]["schema_version"],
                },
                "two_profiles_same_strategy_run": [
                    {"profile_id": item["evaluation_profile_id"], "profile_hash": item["profile_hash"]}
                    for item in evaluations["items"]
                ],
                "separate_evaluation_stages": sorted(outputs[0]["items"][0]),
                "artifact_state_run_ids": ids["artifact_state_run_ids"],
                "separate_execution_attempt": {
                    "operational_status": attempts["items"][0]["operational_status"],
                    "strategy_run_terminal_status": next(
                        item["terminal_status"] for item in run_list["items"]
                        if item["strategy_run_id"] == run_id
                    ),
                },
                "korean_explanation_count": len(terminology["entries"]),
                "economic_backtest_calls": 0,
            }
            print(json.dumps(demonstration, ensure_ascii=False, sort_keys=True, indent=2))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    main()
