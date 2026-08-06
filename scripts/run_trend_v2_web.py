"""The single local launcher for Trend Strategy v2's persisted workflow UI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

'''Deferred application imports: init, preflight, and help stay dependency-light.
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
    load_evaluation_profiles,
    load_execution_policy,
    load_retention_policy,
    load_robustness_catalog,
    load_terminology_source,
)
from src.trend_v2_foundation.foundation_6 import OptionCatalog, PersistedExecutionManager  # noqa: E402
from src.trend_v2_foundation.local_operability import (  # noqa: E402
    local_status,
    reconcile_local_state,
    run_preflight,
)
'''


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "local-unavailable"


def _print_preflight(report: dict) -> None:
    for check in report["checks"]:
        marker = {"pass": "통과", "warning": "경고", "blocking": "차단"}[check["status"]]
        print(f"[{marker}] {check['code']}: {check['message_ko']}")
        if check["status"] == "blocking" and " init --store " in check["suggested_action_ko"]:
            print(check["suggested_action_ko"])


def _services(store_root: Path, port: int):
    from src.trend_v2_foundation import (ApiServerConfig, ArtifactRetentionPolicy, CanonicalCostStressAdapter, ControlledExecutionService, FileExecutionAttemptRepository, LocalResultStore, PhaseAControlledExecutionAdapter, RobustnessExecutionService, RobustnessPolicy, WorkflowCoordinator, load_evaluation_profiles, load_execution_policy, load_retention_policy, load_robustness_catalog, load_terminology_source)
    from src.trend_v2_foundation.foundation_6 import OptionCatalog, PersistedExecutionManager
    policy = ArtifactRetentionPolicy.from_dict(load_retention_policy(store_root))
    store = LocalResultStore(store_root, policy)
    terminology = load_terminology_source(ROOT / "config" / "trend_v2" / "terminology_ko.json")
    attempts = FileExecutionAttemptRepository(store.root / "execution_attempts")
    execution_policy = load_execution_policy(ROOT / "config" / "trend_v2" / "local_execution_policy_v1.json")
    profiles = {item.evaluation_profile_id: item for item in load_evaluation_profiles(ROOT / "config" / "trend_v2" / "evaluation_profiles").values()}
    source_commit = _source_commit()
    economic_adapter = PhaseAControlledExecutionAdapter(ROOT / "docs" / "research" / "trend_v2" / "phase_a2")
    execution = ControlledExecutionService(store, attempts, economic_adapter, execution_policy, profiles, source_commit=source_commit)
    robustness = RobustnessExecutionService(store, RobustnessPolicy.load(ROOT / "config" / "trend_v2" / "robustness_execution_policy_v1.json"), load_robustness_catalog(ROOT / "config" / "trend_v2" / "robustness_option_catalog_v1.json"), source_commit=source_commit, cost_stress_runner=CanonicalCostStressAdapter(store, economic_adapter, source_commit=source_commit))
    manager = PersistedExecutionManager(store.root / "execution_management_v1", OptionCatalog.load(ROOT / "config" / "trend_v2" / "strategy_option_catalog_v2.json"))
    workflows = WorkflowCoordinator(execution, robustness, manager=manager)
    return store, terminology, attempts, execution, robustness, manager, workflows, source_commit


def _shutdown(execution: ControlledExecutionService, attempts: FileExecutionAttemptRepository) -> int:
    requested = 0
    for attempt in attempts.list():
        if attempt.operational_status.value in {"pending", "queued", "running", "cancelling"}:
            execution.cancel(attempt.execution_attempt_id, idempotency_key=f"local-shutdown-{attempt.execution_attempt_id[-32:]}")
            requested += 1
    execution.close()
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trend Strategy v2 로컬 워크플로 도구")
    parser.add_argument("command", choices=("init", "start", "preflight", "status"), nargs="?", default="start")
    parser.add_argument("--store", required=True, type=Path, help="기존 로컬 ResultStore 디렉터리")
    parser.add_argument("--port", type=int, default=8765, help="루프백 포트 (기본 8765)")
    parser.add_argument("--launcher-instance-id", help=argparse.SUPPRESS)
    parser.add_argument("--launcher-token-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    store_root = args.store.resolve()
    from src.trend_v2_foundation.local_operability import initialize_result_store, run_preflight
    if args.command == "init":
        try:
            summary = initialize_result_store(store_root, ROOT / "config" / "trend_v2" / "evaluation_profiles")
            created = bool(summary["created"])
        except ValueError as error:
            print(f"[차단] ResultStore 초기화 실패: {error}")
            return 1
        print("[완료] ResultStore를 초기화했습니다." if created else "[완료] 호환되는 ResultStore가 이미 초기화되어 있습니다.")
        print(f"기본 평가 프로파일: 생성 {summary['seeded']}개, 재사용 {summary['reused']}개")
        print("다음 명령: python scripts/run_trend_v2_web.py preflight --store <경로>")
        return 0
    report = run_preflight(ROOT, store_root, port=args.port)
    if args.command == "preflight":
        _print_preflight(report)
        return 1 if report["blocking"] else 0
    if report["blocking"]:
        print("[차단] 시작할 수 없습니다. 아래 사전 점검 결과를 해결하세요.")
        _print_preflight(report)
        return 1
    if report["warning_count"]:
        print(f"[경고] 사전 점검 경고 {report['warning_count']}건이 있습니다.")
    from src.trend_v2_foundation import ApiServerConfig, ReadOnlyTrendApi, TrendWebApplication, build_web_server
    from src.trend_v2_foundation.local_operability import local_status, reconcile_local_state
    store, terminology, attempts, execution, robustness, manager, workflows, source_commit = _services(store_root, args.port)
    recovery = reconcile_local_state(store.root, source_commit=source_commit, manager=manager, attempts=attempts, robustness=robustness, workflows=workflows)
    if args.command == "status":
        status = local_status(store.root, manager=manager, attempts=attempts)
        print("로컬 상태: 저장소 준비됨")
        print(f"워크플로 {status['workflow_count_by_stage']['persisted']}개 · 활성 시도 {len(status['active_attempts'])}개 · 재개 가능 {status['resumable_workflow_count']}개")
        print(f"마지막 복구: {recovery['recovery_id'][:12]}")
        execution.close()
        return 0
    api = ReadOnlyTrendApi(store, attempt_repository=attempts, terminology_source=terminology, server_config=ApiServerConfig(port=args.port), controlled_execution_service=execution, persisted_execution_manager=manager, robustness_execution_service=robustness, workflow_coordinator=workflows, local_status_provider=lambda: local_status(store.root, manager=manager, attempts=attempts))
    shutdown_token = None
    if args.launcher_instance_id or args.launcher_token_file:
        if not args.launcher_instance_id or args.launcher_token_file is None:
            execution.close()
            print("[차단] Windows 런처 제어 정보가 불완전합니다.")
            return 1
        try:
            shutdown_token = args.launcher_token_file.read_text(encoding="utf-8").strip()
        except OSError:
            execution.close()
            print("[차단] Windows 런처 종료 토큰을 읽을 수 없습니다.")
            return 1
        if len(shutdown_token) < 32:
            execution.close()
            print("[차단] Windows 런처 종료 토큰이 올바르지 않습니다.")
            return 1
    try:
        server = build_web_server(
            TrendWebApplication(api),
            launcher_instance_id=args.launcher_instance_id,
            launcher_shutdown_token=shutdown_token,
        )
    except OSError:
        execution.close()
        print("[차단] 다른 로컬 서비스가 이미 이 포트를 사용 중입니다.")
        return 1
    address = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(f"Trend Strategy v2 로컬 UI: {address}")
    print(f"저장소 준비됨 · 복구 워크플로 {recovery['scanned_workflow_count']}개 · 중단/차단 항목 {len(recovery['blocked_items'])}개")
    print("종료하려면 Ctrl+C를 누르세요. 새 작업은 중단 요청 후 상태를 보존합니다.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("종료 중: 새 요청을 받지 않고 활성 작업에 중단을 요청합니다.")
    finally:
        server.server_close()
        requested = _shutdown(execution, attempts)
        print(f"종료 완료: 활성 작업 {requested}개에 중단을 요청했고 저장 상태를 보존했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
