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
from .integration import calculate_and_evaluate_saved_runs
from .robustness import RobustnessExecutionService


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

    def __init__(self, execution: ControlledExecutionService, robustness: RobustnessExecutionService | None = None, *, clock: callable = _now) -> None:
        self.execution, self.robustness, self.clock = execution, robustness, clock
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

    def read(self, workflow_id: str) -> Mapping[str, Any]:
        record = self._read("workflows", workflow_id)
        if record.get("integrity_hash") != content_hash({key: value for key, value in record.items() if key != "integrity_hash"}):
            raise WorkflowError("workflow_integrity_invalid", "Workflow identity record is corrupt.", object_identity=workflow_id, recoverable=False)
        events = self._events(workflow_id)
        normalized, estimate = self._latest(events, "normalized"), self._latest(events, "estimated")
        economic, robustness, evaluation = self._latest(events, "economic_started"), self._latest(events, "robustness_started"), self._latest(events, "evaluated")
        stage = "draft"
        if normalized: stage = "normalized"
        if estimate: stage = "estimated"
        if estimate and estimate.get("confirmation_required") and not self._latest(events, "confirmed"): stage = "confirmation_required"
        if economic: stage = "economic_queued"
        if economic and economic.get("status") in {"completed", "reused"}: stage = "economic_completed"
        if self._latest(events, "robustness_configured"): stage = "robustness_configuration_required"
        if robustness: stage = "robustness_queued"
        if robustness and robustness.get("status") == "completed": stage = "robustness_completed"
        if evaluation: stage = "completed"
        return {"schema_version": WORKFLOW_SCHEMA_VERSION, "workflow_id": workflow_id, "label_ko": record["label_ko"], "stage": stage, "created_timestamp": record["created_timestamp"], "construction": record["construction"], "references": {"normalized_construction": normalized, "candidate_estimate": estimate, "confirmation": self._latest(events, "confirmed"), "economic": economic, "robustness_plan": self._latest(events, "robustness_configured"), "robustness": robustness, "evaluation": evaluation}, "provenance": record["provenance"], "recoverability": {"resumable": bool(economic or robustness), "reason": None if economic or robustness else "no_started_work"}, "events": [{"event_id": item["event_id"], "action": item["action"], "created_timestamp": item["created_timestamp"]} for item in events]}

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
        self._event(workflow_id, "economic_started", {"execution_request_id": request.execution_request_id, "execution_request": request.to_dict(), "status": status.get("status", "queued"), "attempts": status.get("attempts", [])})
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

    def evaluate(self, workflow_id: str, *, evaluation_profile_id: str) -> Mapping[str, Any]:
        state = self.read(workflow_id)
        economic = state["references"]["economic"] or {}
        run_ids = [item.get("intended_strategy_run_id") for item in economic.get("attempts", []) if item.get("intended_strategy_run_id")]
        if not run_ids: raise WorkflowError("workflow_economic_incomplete", "No completed StrategyRun reference is available.", object_identity=workflow_id)
        profile = self.execution.profiles.get(evaluation_profile_id)
        if profile is None: raise WorkflowError("workflow_construction_invalid", "EvaluationProfile is unknown.", object_identity=evaluation_profile_id)
        result = calculate_and_evaluate_saved_runs(self.execution.store, run_ids, profile, creation_time=self.clock())
        self._event(workflow_id, "evaluated", {"evaluation_profile_id": evaluation_profile_id, "evaluation_run_id": result.evaluation_run.evaluation_run_id, "strategy_run_ids": run_ids})
        return self.read(workflow_id)
