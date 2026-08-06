"""Foundation 8 persisted coordinator over the existing local contracts.

The coordinator deliberately owns no economic, robustness, or evaluation data.
It records immutable intent/event references and derives presentation state from
those authoritative records after a browser refresh or process restart.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_data, content_hash
from .execution_service import ControlledExecutionService
from .execution import AttemptOperationalStatus
from .integration import calculate_and_evaluate_saved_runs
from .robustness import RobustnessExecutionService
from .foundation_6 import PersistedExecutionManager


WORKFLOW_SCHEMA_VERSION = "trend_v2_workflow_v1"


class WorkflowError(ValueError):
    def __init__(self, code: str, diagnostic_en: str, *, object_identity: str | None = None, recoverable: bool = True) -> None:
        super().__init__(diagnostic_en)
        self.code, self.diagnostic_en = code, diagnostic_en
        self.object_identity, self.recoverable = object_identity, recoverable

    def to_dict(self, request_id: str = "local") -> dict[str, Any]:
        messages = {
            "workflow_not_found": "워크플로를 찾을 수 없습니다.",
            "workflow_integrity_invalid": "워크플로 상태의 무결성을 확인할 수 없습니다.",
            "workflow_construction_invalid": "전략 구성이 유효하지 않습니다.",
            "workflow_resume_unavailable": "이 워크플로는 재개할 수 없습니다.",
            "workflow_economic_incomplete": "경제 백테스트 완료 결과가 아직 없습니다.",
        }
        return {"error_code": self.code, "message_ko": messages.get(self.code, "워크플로 요청을 처리할 수 없습니다."), "diagnostic_en": self.diagnostic_en, "object_identity": self.object_identity, "recoverable": self.recoverable, "next_action_ko": "표시된 상태와 참조를 확인한 뒤 다시 시도하세요.", "request_id": request_id}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class WorkflowCoordinator:
    """Append-only workflow references with deterministic idempotency bindings."""

    def __init__(self, execution: ControlledExecutionService, robustness: RobustnessExecutionService | None = None, *, manager: PersistedExecutionManager | None = None, clock: callable = _now) -> None:
        self.execution, self.robustness, self.manager, self.clock = execution, robustness, manager, clock
        self.root = execution.store.root / "workflow_v1"
        for name in ("workflows", "events", "idempotency"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, identity: str) -> Path:
        return self.root / kind / f"{identity}.json"

    def _write_once(self, kind: str, identity: str, value: Mapping[str, Any]) -> None:
        path = self._path(kind, identity)
        payload = json.dumps(canonical_data(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if path.exists():
            if path.read_bytes() != payload:
                raise WorkflowError("workflow_integrity_invalid", "Immutable workflow record conflicts with existing content.", object_identity=identity, recoverable=False)
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def _read(self, kind: str, identity: str) -> Mapping[str, Any]:
        try:
            return json.loads(self._path(kind, identity).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise WorkflowError("workflow_not_found", "Workflow identity was not found.", object_identity=identity) from error
        except (OSError, ValueError) as error:
            raise WorkflowError("workflow_integrity_invalid", "Workflow record is corrupt.", object_identity=identity, recoverable=False) from error

    def _bind(self, operation: str, key: str, request: Mapping[str, Any], identity: str) -> str:
        binding_id = content_hash({"operation": operation, "key": key})
        path = self._path("idempotency", binding_id)
        if path.exists():
            value = self._read("idempotency", binding_id)
            if value["request_hash"] != content_hash(request):
                raise WorkflowError("workflow_integrity_invalid", "Idempotency key was reused for different content.", object_identity=key)
            return str(value["object_identity"])
        self._write_once("idempotency", binding_id, {"schema_version": WORKFLOW_SCHEMA_VERSION, "operation": operation, "key": key, "request_hash": content_hash(request), "object_identity": identity, "created_timestamp": self.clock()})
        return identity

    def create(self, construction: Mapping[str, Any], *, label_ko: str, idempotency_key: str) -> Mapping[str, Any]:
        if not isinstance(construction, Mapping) or not isinstance(label_ko, str) or not label_ko.strip() or len(label_ko) > 120:
            raise WorkflowError("workflow_construction_invalid", "Construction and bounded Korean label are required.")
        request = {"construction": canonical_data(construction), "label_ko": label_ko.strip()}
        workflow_id = "workflow_" + content_hash(request)[:32]
        bound = self._bind("create", idempotency_key, request, workflow_id)
        if bound != workflow_id:
            return self.read(bound)
        if self._path("workflows", workflow_id).exists():
            return self.read(workflow_id)
        created = self.clock()
        record = {"schema_version": WORKFLOW_SCHEMA_VERSION, "workflow_id": workflow_id, "label_ko": label_ko.strip(), "construction": canonical_data(construction), "created_timestamp": created, "provenance": {"source_commit": self.execution.source_commit, "execution_policy_hash": self.execution.policy.policy_hash}, "recoverable": True}
        record["integrity_hash"] = content_hash(record)
        self._write_once("workflows", workflow_id, record)
        return self.read(workflow_id)

    def _events(self, workflow_id: str) -> list[Mapping[str, Any]]:
        prefix = f"{workflow_id}_"
        values: list[Mapping[str, Any]] = []
        for path in sorted((self.root / "events").glob(f"{prefix}*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("integrity_hash") != content_hash({key: value for key, value in item.items() if key != "integrity_hash"}):
                    raise ValueError("hash mismatch")
                values.append(item)
            except (OSError, ValueError, TypeError) as error:
                raise WorkflowError("workflow_integrity_invalid", "Workflow event is corrupt.", object_identity=workflow_id, recoverable=False) from error
        return values

    def _event(self, workflow_id: str, action: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._read("workflows", workflow_id)
        identity = content_hash({"workflow_id": workflow_id, "action": action, "payload": canonical_data(payload)})
        value = {"schema_version": WORKFLOW_SCHEMA_VERSION, "workflow_id": workflow_id, "event_id": identity, "action": action, "payload": canonical_data(payload), "created_timestamp": self.clock()}
        value["integrity_hash"] = content_hash(value)
        self._write_once("events", f"{workflow_id}_{identity}", value)
        return value

    @staticmethod
    def _latest(events: list[Mapping[str, Any]], action: str) -> Mapping[str, Any] | None:
        found = [item for item in events if item["action"] == action]
        return found[-1]["payload"] if found else None

    def list(self) -> Mapping[str, Any]:
        """Return persisted workflows without treating client storage as authority."""
        items = []
        for path in sorted((self.root / "workflows").glob("*.json")):
            state = self.read(path.stem)
            items.append({
                "workflow_id": state["workflow_id"], "label_ko": state["label_ko"],
                "stage": state["stage"], "created_timestamp": state["created_timestamp"],
                "recoverability": state["recoverability"], "last_updated_timestamp": state["last_updated_timestamp"],
            })
        return {"schema_version": WORKFLOW_SCHEMA_VERSION, "items": sorted(items, key=lambda item: (item["last_updated_timestamp"], item["workflow_id"]), reverse=True)}

    def _economic_progress(self, reference: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if reference is None:
            return None
        attempt_ids = list(reference.get("execution_attempt_ids") or [item.get("execution_attempt_id") for item in reference.get("attempts", [])])
        attempts = []
        for attempt_id in attempt_ids:
            if not isinstance(attempt_id, str):
                return {"status": "incompatible", "error_code": "workflow_integrity_invalid", "attempts": []}
            try:
                attempt = self.execution.attempt_repository.get(attempt_id)
            except (KeyError, ValueError):
                return {"status": "missing", "error_code": "workflow_reference_missing", "attempts": []}
            attempts.append(attempt)
        if not attempts:
            return {"status": "missing", "error_code": "workflow_reference_missing", "attempts": []}
        statuses = {item.operational_status.value for item in attempts}
        if statuses & {"running", "cancelling"}:
            status = "running"
        elif statuses & {"queued", "pending"}:
            status = "pending"
        elif statuses == {"completed"}:
            status = "completed"
            for attempt in attempts:
                validation = self.execution.store.validate_manifest(attempt.intended_strategy_run_id)
                if not validation.valid:
                    status = "stale"
                    break
        elif statuses & {"failed"}:
            status = "failed"
        elif statuses & {"cancelled"}:
            status = "cancelled"
        else:
            status = "blocked"
        return {"status": status, "execution_request_id": reference.get("execution_request_id"), "attempts": [item.to_dict() for item in attempts], "strategy_run_ids": sorted(item.intended_strategy_run_id for item in attempts)}

    def _robustness_progress(self, reference: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if reference is None:
            return None
        if self.robustness is None:
            return {"status": "incompatible", "error_code": "workflow_resume_unavailable"}
        attempt_id = reference.get("robustness_attempt_id")
        if not isinstance(attempt_id, str):
            return {"status": "missing", "error_code": "workflow_reference_missing"}
        try:
            attempt = self.robustness._load("attempts", attempt_id)
        except (KeyError, ValueError, OSError):
            return {"status": "missing", "error_code": "workflow_reference_missing"}
        states = {item.get("state") for item in attempt.get("scenarios", [])}
        if not states:
            status = "pending"
        elif states & {"running"}:
            status = "running"
        elif states <= {"succeeded", "reused", "skipped"}:
            status = "completed"
        elif states & {"failed"}:
            status = "failed"
        elif states & {"cancelled"}:
            status = "cancelled"
        else:
            status = "blocked"
        return {"status": status, "robustness_plan_id": reference.get("robustness_plan_id"), "robustness_attempt_id": attempt_id, "attempt": attempt}

    def read(self, workflow_id: str) -> Mapping[str, Any]:
        record = self._read("workflows", workflow_id)
        if record.get("integrity_hash") != content_hash({key: value for key, value in record.items() if key != "integrity_hash"}):
            raise WorkflowError("workflow_integrity_invalid", "Workflow identity record is corrupt.", object_identity=workflow_id, recoverable=False)
        events = self._events(workflow_id)
        normalized, estimate = self._latest(events, "normalized"), self._latest(events, "estimated")
        economic, robustness, evaluation = self._latest(events, "economic_started"), self._latest(events, "robustness_started"), self._latest(events, "evaluated")
        economic_progress, robustness_progress = self._economic_progress(economic), self._robustness_progress(robustness)
        stage = "draft"
        if normalized: stage = "normalized"
        if estimate: stage = "estimated"
        if estimate and estimate.get("confirmation_required") and not self._latest(events, "confirmed"): stage = "confirmation_required"
        if economic: stage = "economic_pending"
        if economic_progress and economic_progress["status"] == "running": stage = "economic_running"
        if economic_progress and economic_progress["status"] == "completed": stage = "economic_completed"
        if economic_progress and economic_progress["status"] in {"failed", "cancelled", "blocked", "stale", "missing", "incompatible"}: stage = "economic_" + economic_progress["status"]
        if self._latest(events, "robustness_configured"): stage = "robustness_configuration_required"
        if robustness: stage = "robustness_pending"
        if robustness_progress and robustness_progress["status"] == "running": stage = "robustness_running"
        if robustness_progress and robustness_progress["status"] == "completed": stage = "robustness_completed"
        if robustness_progress and robustness_progress["status"] in {"failed", "cancelled", "blocked", "stale", "missing", "incompatible"}: stage = "robustness_" + robustness_progress["status"]
        if evaluation: stage = "completed"
        latest_timestamp = events[-1]["created_timestamp"] if events else record["created_timestamp"]
        return {"schema_version": WORKFLOW_SCHEMA_VERSION, "workflow_id": workflow_id, "label_ko": record["label_ko"], "stage": stage, "created_timestamp": record["created_timestamp"], "last_updated_timestamp": latest_timestamp, "construction": record["construction"], "references": {"normalized_construction": normalized, "candidate_estimate": estimate, "confirmation": self._latest(events, "confirmed"), "economic": economic, "economic_progress": economic_progress, "robustness_plan": self._latest(events, "robustness_configured"), "robustness": robustness, "robustness_progress": robustness_progress, "evaluation": evaluation}, "provenance": record["provenance"], "recoverability": {"resumable": bool(economic or robustness), "reason": None if economic or robustness else "no_started_work"}, "events": [{"event_id": item["event_id"], "action": item["action"], "created_timestamp": item["created_timestamp"]} for item in events]}

    def normalize(self, workflow_id: str) -> Mapping[str, Any]:
        record = self._read("workflows", workflow_id)
        normalized = self.execution.normalize(record["construction"]).to_dict()
        self._event(workflow_id, "normalized", normalized)
        return self.read(workflow_id)

    def estimate(self, workflow_id: str) -> Mapping[str, Any]:
        record = self._read("workflows", workflow_id)
        normalized, estimate, candidates = self.execution.estimate(record["construction"])
        payload = {"normalized_construction": normalized.to_dict(), **estimate.to_dict(), "candidate_strategy_run_ids": [item.strategy_run_id for item in candidates]}
        self._event(workflow_id, "estimated", payload)
        return self.read(workflow_id)

    def confirm(self, workflow_id: str, *, idempotency_key: str) -> Mapping[str, Any]:
        record = self._read("workflows", workflow_id)
        confirmation = self.execution.confirm(record["construction"], idempotency_key=idempotency_key).to_dict()
        self._event(workflow_id, "confirmed", confirmation)
        return self.read(workflow_id)

    def start_economic(self, workflow_id: str, *, idempotency_key: str) -> Mapping[str, Any]:
        record, state = self._read("workflows", workflow_id), self.read(workflow_id)
        confirmation = state["references"]["confirmation"] or {}
        request = self.execution.create_request(record["construction"], confirmation_id=confirmation.get("confirmation_id"), idempotency_key=idempotency_key)
        status = self.execution.start(request.execution_request_id, idempotency_key=idempotency_key)
        attempt_ids = list(status.get("execution_attempt_ids", []))
        if self.manager is not None:
            self.manager.track_controlled_request(request.to_dict(), self.execution.attempt_repository.list())
        self._event(workflow_id, "economic_started", {"execution_request_id": request.execution_request_id, "execution_attempt_ids": attempt_ids, "strategy_run_ids": [item["strategy_run_id"] for item in request.requested_strategy_run_candidates]})
        return self.read(workflow_id)

    def configure_robustness(self, workflow_id: str, request: Mapping[str, Any], *, confirmation_id: str | None = None) -> Mapping[str, Any]:
        if self.robustness is None: raise WorkflowError("workflow_resume_unavailable", "Robustness service is disabled.")
        plan = self.robustness.create_plan(request, confirmation_id=confirmation_id)
        self._event(workflow_id, "robustness_configured", {"robustness_plan_id": plan["robustness_plan_id"], "plan_hash": plan["plan_hash"], "estimate": plan["estimate"]})
        return self.read(workflow_id)

    def start_robustness(self, workflow_id: str) -> Mapping[str, Any]:
        if self.robustness is None: raise WorkflowError("workflow_resume_unavailable", "Robustness service is disabled.")
        plan = self._latest(self._events(workflow_id), "robustness_configured")
        if plan is None: raise WorkflowError("workflow_economic_incomplete", "No persisted robustness plan is configured.", object_identity=workflow_id)
        attempt = self.robustness.start(str(plan["robustness_plan_id"]))
        self._event(workflow_id, "robustness_started", {"robustness_plan_id": plan["robustness_plan_id"], "robustness_attempt_id": attempt["robustness_attempt_id"], "status": attempt["status"]})
        return self.read(workflow_id)

    def resume(self, workflow_id: str, *, idempotency_key: str) -> Mapping[str, Any]:
        """Resume only explicitly incomplete persisted units; completed evidence is reused."""
        resume_request = {"workflow_id": workflow_id}
        binding_id = content_hash({"operation": "resume", "key": idempotency_key})
        if self._path("idempotency", binding_id).exists():
            binding = self._read("idempotency", binding_id)
            if binding.get("request_hash") != content_hash(resume_request):
                raise WorkflowError("workflow_integrity_invalid", "Idempotency key was reused for another workflow.", object_identity=idempotency_key, recoverable=False)
            return self.read(workflow_id)
        state = self.read(workflow_id)
        economic = state["references"]["economic"] or {}
        economic_progress = state["references"].get("economic_progress") or {}
        robustness_reference = state["references"]["robustness"] or {}
        if not economic and not robustness_reference:
            raise WorkflowError("workflow_resume_unavailable", "No persisted economic or robustness attempt can be resumed.", object_identity=workflow_id)
        resumed_economic: list[str] = []
        reused_economic: list[str] = []
        for attempt_id in economic.get("execution_attempt_ids", [item.get("execution_attempt_id") for item in economic.get("attempts", [])]):
            if not isinstance(attempt_id, str):
                raise WorkflowError("workflow_integrity_invalid", "Economic attempt reference is missing its identity.", object_identity=workflow_id, recoverable=False)
            try:
                attempt = self.execution.attempt_repository.get(attempt_id)
            except (KeyError, ValueError) as error:
                raise WorkflowError("workflow_integrity_invalid", "Referenced economic attempt is corrupt or missing.", object_identity=attempt_id, recoverable=False) from error
            if attempt.operational_status == AttemptOperationalStatus.COMPLETED:
                validation = self.execution.store.validate_manifest(attempt.intended_strategy_run_id)
                if not validation.valid:
                    raise WorkflowError("workflow_integrity_invalid", "Completed economic attempt has invalid artifacts.", object_identity=attempt_id, recoverable=False)
                reused_economic.append(attempt.intended_strategy_run_id)
            elif attempt.operational_status in {AttemptOperationalStatus.FAILED, AttemptOperationalStatus.CANCELLED} and not resumed_economic:
                retry = self.execution.retry(attempt_id, idempotency_key=f"workflow-resume-{workflow_id[:20]}-{idempotency_key}")
                resumed_economic.append(retry.execution_attempt_id)
            elif attempt.operational_status in {AttemptOperationalStatus.RUNNING, AttemptOperationalStatus.CANCELLING}:
                raise WorkflowError("workflow_resume_unavailable", "A referenced economic attempt still has an active owner.", object_identity=attempt_id)
        resumed_robustness: list[str] = []
        reused_robustness: list[str] = []
        if robustness_reference:
            if self.robustness is None:
                raise WorkflowError("workflow_resume_unavailable", "Robustness service is disabled.", object_identity=workflow_id)
            attempt_id = robustness_reference.get("robustness_attempt_id")
            if not isinstance(attempt_id, str):
                raise WorkflowError("workflow_integrity_invalid", "Robustness attempt reference is missing its identity.", object_identity=workflow_id, recoverable=False)
            try:
                persisted = self.robustness._load("attempts", attempt_id)
            except (KeyError, ValueError, OSError) as error:
                raise WorkflowError("workflow_integrity_invalid", "Referenced robustness attempt is corrupt or missing.", object_identity=attempt_id, recoverable=False) from error
            scenarios = list(persisted.get("scenarios", []))
            if any(item.get("state") in {"failed", "cancelled", "blocked", "incomplete"} for item in scenarios):
                resumed_robustness = list(self.robustness.resume(attempt_id)["resumed_scenarios"])
            reused_robustness = [str(item["scenario_id"]) for item in scenarios if item.get("state") in {"succeeded", "reused"}]
        payload = {"economic_attempt_ids": resumed_economic, "reused_strategy_run_ids": sorted(reused_economic), "robustness_scenario_ids": resumed_robustness, "reused_robustness_scenario_ids": sorted(reused_robustness)}
        self._bind("resume", idempotency_key, resume_request, content_hash(payload))
        self._event(workflow_id, "resumed", payload)
        return self.read(workflow_id)

    def evaluate(self, workflow_id: str, *, evaluation_profile_id: str, idempotency_key: str | None = None) -> Mapping[str, Any]:
        state = self.read(workflow_id)
        request = {"workflow_id": workflow_id, "evaluation_profile_id": evaluation_profile_id}
        if idempotency_key:
            binding_id = content_hash({"operation": "evaluate", "key": idempotency_key})
            if self._path("idempotency", binding_id).exists():
                self._bind("evaluate", idempotency_key, request, workflow_id)
                return self.read(workflow_id)
        economic = state["references"]["economic"] or {}
        progress = state["references"].get("economic_progress") or {}
        if progress.get("status") != "completed":
            raise WorkflowError("workflow_economic_incomplete", "Completed StrategyRun references are required before evaluation.", object_identity=workflow_id)
        run_ids = list(progress.get("strategy_run_ids", []))
        if not run_ids: raise WorkflowError("workflow_economic_incomplete", "No completed StrategyRun reference is available.", object_identity=workflow_id)
        try:
            profile = self.execution.store.get_evaluation_profile(evaluation_profile_id)
        except KeyError as error:
            raise WorkflowError("workflow_construction_invalid", "EvaluationProfile is unknown.", object_identity=evaluation_profile_id) from error
        result = calculate_and_evaluate_saved_runs(self.execution.store, run_ids, profile, creation_time=self.clock())
        self._event(workflow_id, "evaluated", {"evaluation_profile_id": evaluation_profile_id, "evaluation_run_id": result.evaluation_run.evaluation_run_id, "strategy_run_ids": run_ids, "economic_backtest_started": False})
        if idempotency_key:
            self._bind("evaluate", idempotency_key, request, workflow_id)
        return self.read(workflow_id)
