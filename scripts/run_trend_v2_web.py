"""Start the loopback-only Trend Strategy v2 saved-result UI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trend_v2_foundation import (  # noqa: E402
    ApiServerConfig,
    ArtifactRetentionPolicy,
    CanonicalCostStressAdapter,
    ControlledExecutionService,
    FileExecutionAttemptRepository,
    LocalResultStore,
    PhaseAControlledExecutionAdapter,
    ReadOnlyTrendApi,
    RobustnessExecutionService,
    RobustnessPolicy,
    TrendWebApplication,
    WorkflowCoordinator,
    build_web_server,
    load_retention_policy,
    load_execution_policy,
    load_evaluation_profiles,
    load_robustness_catalog,
    load_terminology_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 Trend Strategy v2 결과 UI")
    parser.add_argument("--store", required=True, type=Path, help="기존 로컬 ResultStore 디렉터리")
    parser.add_argument("--port", type=int, default=8765, help="루프백 포트 (기본 8765)")
    args = parser.parse_args()

    store_root = args.store.resolve()
    policy = ArtifactRetentionPolicy.from_dict(load_retention_policy(store_root))
    store = LocalResultStore(store_root, policy)
    terminology = load_terminology_source(ROOT / "config" / "trend_v2" / "terminology_ko.json")
    attempt_repository = FileExecutionAttemptRepository(store.root / "execution_attempts")
    execution_policy = load_execution_policy(
        ROOT / "config" / "trend_v2" / "local_execution_policy_v1.json"
    )
    loaded_profiles = load_evaluation_profiles(ROOT / "config" / "trend_v2" / "evaluation_profiles")
    profiles = {profile.evaluation_profile_id: profile for profile in loaded_profiles.values()}
    source_commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    economic_adapter = PhaseAControlledExecutionAdapter(ROOT / "docs" / "research" / "trend_v2" / "phase_a2")
    execution_service = ControlledExecutionService(
        store,
        attempt_repository,
        economic_adapter,
        execution_policy,
        profiles,
        source_commit=source_commit,
    )
    robustness_policy, robustness_catalog = RobustnessPolicy.load(
        ROOT / "config" / "trend_v2" / "robustness_execution_policy_v1.json"
    ), load_robustness_catalog(ROOT / "config" / "trend_v2" / "robustness_option_catalog_v1.json")
    robustness_service = RobustnessExecutionService(
        store, robustness_policy, robustness_catalog, source_commit=source_commit,
        cost_stress_runner=CanonicalCostStressAdapter(store, economic_adapter, source_commit=source_commit),
    )
    api = ReadOnlyTrendApi(
        store,
        attempt_repository=attempt_repository,
        terminology_source=terminology,
        server_config=ApiServerConfig(port=args.port),
        controlled_execution_service=execution_service,
        robustness_execution_service=robustness_service,
        workflow_coordinator=WorkflowCoordinator(execution_service, robustness_service),
    )
    server = build_web_server(TrendWebApplication(api))
    address = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(f"Trend Strategy v2 저장 결과 UI: {address}")
    print("읽기 전용입니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        execution_service.close()


if __name__ == "__main__":
    main()
