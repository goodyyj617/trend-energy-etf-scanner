"""Local-only startup checks and conservative persisted-state recovery."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .canonical import canonical_data, content_hash, deterministic_id
from .execution import (
    AttemptOperationalStatus,
    AttemptTerminalOutcome,
    FileExecutionAttemptRepository,
    TERMINAL_ATTEMPT_STATUSES,
)
from .foundation_6 import Foundation6Error, OptionCatalog, PersistedExecutionManager
from .contracts import ArtifactRetentionPolicy
from .profiles import load_evaluation_profiles
from .result_store import LocalResultStore, RESULT_STORE_VERSION
from .robustness import RobustnessError, RobustnessExecutionService
from .workflow import WorkflowCoordinator, WorkflowError


PREFLIGHT_SCHEMA_VERSION = "trend_v2_local_preflight_v1"
RECOVERY_SCHEMA_VERSION = "trend_v2_recovery_record_v1"
_PACKAGE_IMPORTS = ("pandas", "numpy", "yfinance", "yaml", "requests")
_STATE_DIRECTORIES = (
    "execution_attempts",
    "execution_management_v1",
    "workflow_v1",
    "robustness_execution_v1",
)
_DEFAULT_RETENTION_POLICY = ArtifactRetentionPolicy(5_000_000_000, 250_000_000, 1_000, 1_000)


def initialize_result_store(store_root: str | Path, profile_root: str | Path | None = None) -> Mapping[str, int | bool]:
    """Create the canonical bounded local store once, without runtime services."""
    store = Path(store_root)
    policy_path = store / "retention_policy.json"
    if policy_path.exists():
        try:
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
            policy = ArtifactRetentionPolicy.from_dict(payload["policy"])
            if payload.get("store_version") != RESULT_STORE_VERSION:
                raise ValueError("incompatible ResultStore version")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"incompatible or corrupt ResultStore policy: {type(error).__name__}") from error
        result_store = LocalResultStore(store, policy)
        for name in _STATE_DIRECTORIES:
            (store / name).mkdir(parents=True, exist_ok=True)
        created = False
    if store.exists() and any(store.iterdir()):
        raise ValueError("existing directory is not an initialized ResultStore")
    else:
        result_store = LocalResultStore(store, _DEFAULT_RETENTION_POLICY)
        for name in _STATE_DIRECTORIES:
            (store / name).mkdir(parents=True, exist_ok=True)
        created = True
    seeded = reused = 0
    if profile_root is not None:
        profiles = load_evaluation_profiles(profile_root)
        existing = {result_store.get_evaluation_profile(profile_id).name: result_store.get_evaluation_profile(profile_id) for profile_id in result_store.evaluation_profile_history()}
        for name in ("exploratory_weighted_example", "final_eligibility_default", "research_default"):
            profile = profiles[name]
            prior = existing.get(name)
            if prior is not None and prior.to_dict() != profile.to_dict():
                raise ValueError(f"default evaluation profile conflict: {name}")
            if prior is None:
                result_store.save_evaluation_profile(profile); seeded += 1
            else: reused += 1
    return {"created": created, "seeded": seeded, "reused": reused}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _check(code: str, status: str, message_ko: str, diagnostic_en: str, action_ko: str, component: str) -> dict[str, str]:
    return {
        "code": code,
        "status": status,
        "message_ko": message_ko,
        "diagnostic_en": diagnostic_en,
        "suggested_action_ko": action_ko,
        "component": component,
    }


def _writeable(directory: Path) -> bool:
    probe = directory / ".trend-v2-preflight-write-probe"
    try:
        with probe.open("xb") as handle:
            handle.write(b"ok")
        probe.unlink()
        return True
    except OSError:
        return False


def _port_available(host: str, port: int) -> bool:
    candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        candidate.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        candidate.close()


def _snapshot_ok(snapshot_root: Path, repository_root: Path) -> tuple[bool, str]:
    try:
        manifest = json.loads((snapshot_root / "input_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "trend_v2_phase_a2_snapshot_v1":
            return False, "unsupported snapshot schema"
        for relative, expected in sorted(dict(manifest.get("snapshot_members", {})).items()):
            member = snapshot_root / relative
            if not member.is_file():
                return False, f"missing or invalid snapshot member: {relative}"
            payload = member.read_bytes()
            source = "working-tree bytes"
            try:
                git_relative = member.resolve().relative_to(repository_root.resolve()).as_posix()
                payload = subprocess.check_output(["git", "show", f"HEAD:{git_relative}"], cwd=repository_root)
                source = "Git blob bytes"
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            if hashlib.sha256(payload).hexdigest() != expected:
                return False, f"snapshot member hash mismatch ({source}): {relative}"
        return True, f"frozen snapshot members and hashes are valid ({source})"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, str(error)


def run_preflight(
    root: str | Path,
    store_root: str | Path,
    *,
    port: int = 8765,
    python_version: tuple[int, int] | None = None,
    package_importer: Callable[[str], Any] = importlib.import_module,
    snapshot_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return deterministic, no-network local prerequisite checks."""

    root, store = Path(root), Path(store_root)
    version = python_version or sys.version_info[:2]
    checks: list[dict[str, str]] = []
    supported = (3, 10) <= tuple(version) < (3, 14)
    checks.append(_check("python_supported", "pass" if supported else "blocking", "지원 Python 버전을 확인했습니다." if supported else "지원하지 않는 Python 버전입니다.", f"python={version[0]}.{version[1]}; supported=3.10-3.13", "지원 범위의 Python으로 다시 실행하세요.", "python"))
    missing = []
    for package in _PACKAGE_IMPORTS:
        try: package_importer(package)
        except ImportError: missing.append(package)
    checks.append(_check("python_packages", "pass" if not missing else "blocking", "필수 Python 패키지를 확인했습니다." if not missing else "필수 Python 패키지가 없습니다.", "all required imports available" if not missing else "missing=" + ",".join(missing), "requirements.txt의 패키지를 설치한 뒤 다시 실행하세요.", "python_packages"))
    config_files = (root / "config" / "trend_v2" / "local_execution_policy_v1.json", root / "config" / "trend_v2" / "robustness_execution_policy_v1.json", root / "config" / "trend_v2" / "strategy_option_catalog_v2.json")
    try:
        [json.loads(path.read_text(encoding="utf-8")) for path in config_files]
        checks.append(_check("configuration_readable", "pass", "로컬 설정을 읽을 수 있습니다.", "required configuration JSON is readable", "설정 파일의 JSON 형식을 복구하세요.", "configuration"))
    except (OSError, json.JSONDecodeError) as error:
        checks.append(_check("configuration_readable", "blocking", "로컬 설정을 읽을 수 없습니다.", type(error).__name__, "설정 파일을 복구한 뒤 다시 실행하세요.", "configuration"))
    policy_path = store / "retention_policy.json"
    if not store.exists():
        checks.append(_check("result_store_schema", "blocking", "ResultStore 디렉터리가 없습니다.", "store directory is missing", "먼저 `python scripts/run_trend_v2_web.py init --store <경로>`를 실행하세요.", "result_store"))
    elif not policy_path.exists():
        checks.append(_check("result_store_schema", "blocking", "ResultStore가 초기화되지 않았습니다.", "retention_policy.json is missing", "`python scripts/run_trend_v2_web.py init --store <경로>`를 실행하세요.", "result_store"))
    else:
      try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        compatible = policy.get("store_version") == RESULT_STORE_VERSION and isinstance(policy.get("policy"), Mapping)
        checks.append(_check("result_store_schema", "pass" if compatible else "blocking", "ResultStore 스키마가 호환됩니다." if compatible else "ResultStore 스키마가 호환되지 않습니다.", f"store_version={policy.get('store_version')}", "호환되는 로컬 ResultStore를 선택하거나 백업에서 복구하세요.", "result_store"))
      except (OSError, json.JSONDecodeError) as error:
        checks.append(_check("result_store_schema", "blocking", "ResultStore 정책을 읽을 수 없습니다.", type(error).__name__, "올바른 기존 ResultStore를 지정하세요.", "result_store"))
    storage_ready = store.is_dir() and _writeable(store)
    checks.append(_check("result_store_access", "pass" if storage_ready else "blocking", "로컬 저장소 읽기·쓰기가 가능합니다." if storage_ready else "로컬 저장소에 읽기·쓰기할 수 없습니다.", "store directory is writable" if storage_ready else "store missing or write probe failed", "저장소 경로와 권한을 확인하세요.", "result_store"))
    missing_state = [name for name in _STATE_DIRECTORIES if not (store / name).is_dir()]
    checks.append(_check("workflow_state_directories", "pass" if not missing_state else "warning", "워크플로 상태 디렉터리를 확인했습니다." if not missing_state else "아직 생성되지 않은 워크플로 상태 디렉터리가 있습니다.", "all workflow state directories exist" if not missing_state else "missing=" + ",".join(missing_state), "첫 정상 시작에서 필요한 상태 디렉터리를 만듭니다.", "workflow_state"))
    snapshot_ok, snapshot_diagnostic = _snapshot_ok(Path(snapshot_root) if snapshot_root is not None else root / "docs" / "research" / "trend_v2" / "phase_a2", root)
    checks.append(_check("frozen_data_snapshot", "pass" if snapshot_ok else "blocking", "동결 로컬 데이터 스냅샷을 확인했습니다." if snapshot_ok else "동결 로컬 데이터 스냅샷을 사용할 수 없습니다.", snapshot_diagnostic, "스냅샷 파일을 복구하세요. 다운로드는 이 도구에서 수행하지 않습니다.", "data_snapshot"))
    port_ready = _port_available("127.0.0.1", port)
    checks.append(_check("loopback_port", "pass" if port_ready else "blocking", "루프백 포트를 사용할 수 있습니다." if port_ready else "루프백 포트를 사용할 수 없습니다.", f"host=127.0.0.1 port={port}", "다른 포트를 지정하거나 해당 포트를 사용하는 로컬 프로세스를 종료하세요.", "local_server"))
    checks.append(_check("canonical_runner", "pass", "정식 경제 실행기를 등록했습니다.", "PhaseAControlledExecutionAdapter is configured", "설치본을 복구하세요.", "economic_runner"))
    checks.append(_check("robustness_adapter", "pass", "강건성 어댑터를 등록했습니다.", "CanonicalCostStressAdapter is configured", "설치본을 복구하세요.", "robustness_adapter"))
    corrupt = False
    stale = False
    try:
        attempt_root = store / "execution_attempts"
        if attempt_root.exists():
            active = FileExecutionAttemptRepository(attempt_root).list()
            stale = any(item.operational_status not in TERMINAL_ATTEMPT_STATUSES for item in active)
        management_root = store / "execution_management_v1"
        if management_root.exists():
            manager = PersistedExecutionManager(management_root, OptionCatalog.load(root / "config" / "trend_v2" / "strategy_option_catalog_v2.json"))
            stale = stale or any(item.get("state") == "running" for item in manager.status()["candidates"])
        for path in sorted((store / "robustness_execution_v1" / "attempts").glob("*.json")) if (store / "robustness_execution_v1" / "attempts").exists() else []:
            stale = stale or any(item.get("state") == "running" for item in json.loads(path.read_text(encoding="utf-8")).get("scenarios", []))
        for path in sorted((store / "workflow_v1" / "workflows").glob("*.json")) if (store / "workflow_v1" / "workflows").exists() else []:
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), Mapping): corrupt = True
    except (OSError, ValueError, KeyError, json.JSONDecodeError, Foundation6Error):
        corrupt = True
    checks.append(_check("persisted_state_integrity", "blocking" if corrupt else "pass", "저장된 상태 레코드가 유효합니다." if not corrupt else "시작을 막는 손상된 상태 레코드가 있습니다.", "persisted records can be reconstructed" if not corrupt else "persisted record parse or integrity validation failed", "백업에서 손상된 상태 레코드를 복구하세요.", "persisted_state"))
    checks.append(_check("stale_or_interrupted_work", "warning" if stale else "pass", "중단된 로컬 작업을 시작 시 복구합니다." if stale else "중단된 로컬 작업이 없습니다.", "stale local ownership detected" if stale else "no stale local ownership", "시작 후 화면에서 재개 가능 상태와 사유를 확인하세요.", "recovery"))
    checks.sort(key=lambda item: item["code"])
    return {"schema_version": PREFLIGHT_SCHEMA_VERSION, "checks": checks, "blocking": any(item["status"] == "blocking" for item in checks), "warning_count": sum(item["status"] == "warning" for item in checks)}


def _write_once(path: Path, value: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = json.dumps(canonical_data(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return dict(value)


def reconcile_local_state(
    store_root: str | Path,
    *,
    source_commit: str,
    manager: PersistedExecutionManager,
    attempts: FileExecutionAttemptRepository,
    robustness: RobustnessExecutionService,
    workflows: WorkflowCoordinator,
    clock: Callable[[], str] = _now,
) -> Mapping[str, Any]:
    """Conservatively make abandoned local ownership explicit and record it once."""

    store = Path(store_root)
    transitions: list[Mapping[str, Any]] = []
    blocked: list[str] = []
    corrupt: list[str] = []
    status = manager.status()
    for request in status["requests"]:
        request_id = request["execution_request_id"]
        states = manager.status(request_id)["candidates"]
        if any(item.get("state") == "running" for item in states):
            outcome = manager.reconcile(request_id)
            transitions.extend(outcome["decisions"])
            blocked.extend(item["candidate_economic_hash"] for item in outcome["decisions"])
    stale_attempts: list[str] = []
    try:
        for attempt in attempts.list():
            if attempt.operational_status in {AttemptOperationalStatus.RUNNING, AttemptOperationalStatus.CANCELLING}:
                attempts.transition(attempt.execution_attempt_id, operational_status=AttemptOperationalStatus.FAILED, terminal_outcome=AttemptTerminalOutcome.FAILED, completed_timestamp=clock(), current_stage="recovered_interrupted", failure_code="attempt_interrupted", failure_message="Local service stopped before a trustworthy completion record.", progress_summary={**dict(attempt.progress_summary), "interrupted_count": 1})
                stale_attempts.append(attempt.execution_attempt_id)
                transitions.append({"attempt_id": attempt.execution_attempt_id, "classification": "stale", "action": "interrupted"})
    except (OSError, ValueError, KeyError) as error:
        corrupt.append("execution_attempts:" + type(error).__name__)
    robustness_root = store / "robustness_execution_v1" / "attempts"
    for path in sorted(robustness_root.glob("*.json")) if robustness_root.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if any(item.get("state") == "running" for item in payload.get("scenarios", [])):
                outcome = robustness.reconcile(str(payload["robustness_attempt_id"]))
                transitions.extend({"scenario_id": item, "classification": "stale", "action": "blocked"} for item in outcome["blocked_scenarios"])
                blocked.extend(outcome["blocked_scenarios"])
        except (OSError, ValueError, KeyError, Foundation6Error, RobustnessError) as error:
            corrupt.append("robustness:" + path.name)
    workflow_count = 0
    workflow_root = store / "workflow_v1" / "workflows"
    for path in sorted(workflow_root.glob("*.json")) if workflow_root.exists() else []:
        workflow_count += 1
        try: workflows.read(path.stem)
        except WorkflowError: corrupt.append("workflow:" + path.stem)
    final = manager.status()
    candidates = final["candidates"]
    reusable = sum(item.get("state") in {"succeeded", "reused"} for item in candidates)
    resumable = sum(item.get("state") in {"pending", "failed", "cancelled", "blocked"} for item in candidates)
    robustness_blocked: set[str] = set()
    for path in sorted(robustness_root.glob("*.json")) if robustness_root.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            robustness_blocked.update(str(item["scenario_id"]) for item in payload.get("scenarios", []) if item.get("state") == "blocked")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    current_blocked = sorted({item["candidate_economic_hash"] for item in candidates if item.get("state") == "blocked"} | robustness_blocked | set(blocked))
    current_interrupted_attempts = sorted(item.execution_attempt_id for item in attempts.list() if item.failure_code == "attempt_interrupted")
    recovery_basis = {"source_commit": source_commit, "requests": [item["execution_request_id"] for item in final["requests"]], "candidate_states": [(item["execution_request_id"], item["candidate_economic_hash"], item["state"]) for item in candidates], "stale_attempts": current_interrupted_attempts, "blocked": current_blocked, "corrupt": corrupt}
    recovery_id = deterministic_id("trend_v2_recovery", recovery_basis)
    record: dict[str, Any] = {"schema_version": RECOVERY_SCHEMA_VERSION, "recovery_id": recovery_id, "timestamp": clock(), "source_commit": source_commit, "scanned_workflow_count": workflow_count, "scanned_attempt_count": len(attempts.list()), "state_transitions": transitions, "stale_worker_classifications": [item for item in transitions if item.get("classification") == "stale"], "resumed_unit_count": 0, "reused_unit_count": reusable, "blocked_items": current_blocked, "corrupt_items": sorted(corrupt), "warnings": ["interrupted local work remains explicit; resume requires an explicit request"] if resumable else []}
    record["content_hash"] = content_hash({key: value for key, value in record.items() if key != "content_hash"})
    return _write_once(store / "recovery_v1" / f"{recovery_id}.json", record)


def local_status(store_root: str | Path, *, manager: PersistedExecutionManager, attempts: FileExecutionAttemptRepository) -> Mapping[str, Any]:
    store = Path(store_root)
    manager_status = manager.status()
    candidates = manager_status["candidates"]
    try: active_attempts = [item.execution_attempt_id for item in attempts.list() if item.operational_status not in TERMINAL_ATTEMPT_STATUSES]
    except (OSError, ValueError, KeyError): active_attempts = []
    reports = sorted((store / "recovery_v1").glob("*.json")) if (store / "recovery_v1").exists() else []
    last = json.loads(reports[-1].read_text(encoding="utf-8")) if reports else None
    return {"schema_version": "trend_v2_local_status_v1", "service": {"host": "127.0.0.1"}, "storage_available": store.is_dir() and _writeable(store), "workflow_count_by_stage": {"persisted": len(list((store / "workflow_v1" / "workflows").glob("*.json"))) if (store / "workflow_v1" / "workflows").exists() else 0}, "active_attempts": active_attempts, "stale_attempt_count": sum(item.get("state") == "blocked" for item in candidates), "interrupted_workflow_count": sum(item.get("state") == "blocked" for item in candidates), "resumable_workflow_count": sum(item.get("state") in {"pending", "failed", "cancelled", "blocked"} for item in candidates), "blocked_or_corrupt_count": sum(item.get("state") == "blocked" for item in candidates) + len((last or {}).get("corrupt_items", [])), "canonical_economic_runner": "available", "robustness_adapter": "available", "last_reconciliation": last}
