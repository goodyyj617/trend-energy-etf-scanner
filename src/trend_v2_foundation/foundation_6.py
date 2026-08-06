"""Foundation 6 controlled catalog and restart-safe local execution records.

The files in ``execution_management_v1`` are the authority.  The projection is
deliberately disposable: it is rebuilt from canonical request and event files
on every manager construction.  This keeps recovery independent from process
memory and makes a stopped local process visible rather than accidentally live.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_bytes, canonical_data, content_hash, deterministic_id


CATALOG_SCHEMA_VERSION = "controlled_strategy_option_catalog_v2"
MANAGER_SCHEMA_VERSION = "persisted_local_execution_manager_v1"
EVENT_SCHEMA_VERSION = "execution_management_event_v1"
_TERMINAL = {"succeeded", "failed", "cancelled", "reused", "skipped", "blocked"}
_CANDIDATE_STATES = {"pending", "reused", "running", "succeeded", "failed", "cancelled", "skipped", "blocked"}


class Foundation6Error(ValueError):
    def __init__(self, code: str, message_ko: str, diagnostic_en: str, *, object_identity: str | None = None,
                 recoverable: bool = True, suggested_action: str | None = None) -> None:
        super().__init__(code)
        self.code, self.message_ko, self.diagnostic_en = code, message_ko, diagnostic_en
        self.object_identity, self.recoverable, self.suggested_action = object_identity, recoverable, suggested_action

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message_ko": self.message_ko, "diagnostic_en": self.diagnostic_en,
                "object_identity": self.object_identity, "recoverable": self.recoverable,
                "suggested_action": self.suggested_action}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(payload)
    if path.exists():
        if path.read_bytes() != data:
            raise Foundation6Error("execution_request_corrupt", "불변 레코드의 내용이 충돌합니다.", "Immutable record content conflicts.", object_identity=path.stem, recoverable=False)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _read_record(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Foundation6Error("attempt_event_corrupt", "실행 상태 레코드가 손상되었습니다.", "Execution-state record is corrupt.", object_identity=path.name, recoverable=False) from error
    if not isinstance(value, Mapping) or value.get("content_hash") != content_hash({key: val for key, val in value.items() if key != "content_hash"}):
        raise Foundation6Error("attempt_event_corrupt", "실행 상태 레코드의 해시가 일치하지 않습니다.", "Execution-state record hash is invalid.", object_identity=path.name, recoverable=False)
    return value


def _hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = canonical_data(payload)
    value["content_hash"] = content_hash(value)
    return value


@dataclass(frozen=True)
class OptionCatalog:
    document: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "OptionCatalog":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Foundation6Error("catalog_version_mismatch", "전략 옵션 카탈로그를 읽을 수 없습니다.", "Cannot read strategy option catalog.", recoverable=False) from error
        if document.get("schema_version") != CATALOG_SCHEMA_VERSION or not isinstance(document.get("categories"), Mapping):
            raise Foundation6Error("catalog_version_mismatch", "지원하지 않는 전략 옵션 카탈로그 버전입니다.", "Unsupported strategy option catalog schema.", recoverable=False)
        for category, entries in document["categories"].items():
            if not isinstance(category, str) or not isinstance(entries, list) or not entries:
                raise Foundation6Error("catalog_version_mismatch", "전략 옵션 카탈로그가 완전하지 않습니다.", "Strategy option catalog is incomplete.", recoverable=False)
            for entry in entries:
                required = {"option_id", "name_ko", "name_en", "category", "schema_version", "parameters", "default", "units", "compatibility", "economic_identity", "explanation_ref", "engine_adapter_support"}
                if not isinstance(entry, Mapping) or set(entry) != required or entry["category"] != category:
                    raise Foundation6Error("catalog_version_mismatch", "전략 옵션 메타데이터가 완전하지 않습니다.", "Strategy option metadata is incomplete.", recoverable=False)
        return cls(canonical_data(document))

    @property
    def catalog_hash(self) -> str:
        return content_hash(self.document)

    def to_dict(self) -> dict[str, Any]:
        return {**canonical_data(self.document), "catalog_hash": self.catalog_hash}

    def option(self, category: str, option_id: str) -> Mapping[str, Any]:
        for item in self.document["categories"].get(category, []):
            if item["option_id"] == option_id:
                return item
        raise Foundation6Error("unsupported_strategy_combination", "허용되지 않은 전략 옵션입니다.", "Option is not allow-listed.", object_identity=option_id)

    def defaults(self) -> dict[str, Mapping[str, Any]]:
        return {category: next(item for item in entries if item["default"]) for category, entries in self.document["categories"].items()}


def _values(value: Any, definition: Mapping[str, Any], field: str) -> tuple[Any, ...]:
    """Normalize finite scalar/list/range input without accepting expressions."""
    if not isinstance(value, Mapping) or set(value).difference({"kind", "value", "values", "start", "end", "step"}):
        raise Foundation6Error("unsupported_strategy_combination", "옵션 매개변수 형식이 올바르지 않습니다.", "Parameter input must be scalar, list, or range.", object_identity=field)
    kind = value.get("kind", "fixed")
    if kind == "fixed": raw = [value.get("value", definition["default"])]
    elif kind == "list": raw = value.get("values")
    elif kind == "range":
        try:
            start, end, step = (Decimal(str(value[key])) for key in ("start", "end", "step"))
        except (InvalidOperation, KeyError) as error:
            raise Foundation6Error("unsupported_strategy_combination", "범위 매개변수가 올바르지 않습니다.", "Range parameter is invalid.", object_identity=field) from error
        if not all(item.is_finite() for item in (start, end, step)) or step <= 0 or end < start or (end - start) % step:
            raise Foundation6Error("unsupported_strategy_combination", "범위의 끝과 간격이 정확히 일치해야 합니다.", "Range must have a positive exact endpoint and step.", object_identity=field)
        raw = [start + index * step for index in range(int((end - start) / step) + 1)]
    else:
        raise Foundation6Error("unsupported_strategy_combination", "허용되지 않은 매개변수 형식입니다.", "Parameter kind is not allowed.", object_identity=field)
    if not isinstance(raw, list) or not raw or len(raw) > 8:
        raise Foundation6Error("unsupported_strategy_combination", "매개변수 값은 1~8개여야 합니다.", "Parameter values must contain 1 to 8 values.", object_identity=field)
    try:
        parsed = [Decimal(str(item)) for item in raw]
    except InvalidOperation as error:
        raise Foundation6Error("unsupported_strategy_combination", "숫자 매개변수가 올바르지 않습니다.", "Numeric parameter is invalid.", object_identity=field) from error
    if any(not item.is_finite() for item in parsed) or any(item < Decimal(str(definition["minimum"])) or item > Decimal(str(definition["maximum"])) for item in parsed):
        raise Foundation6Error("unsupported_strategy_combination", "매개변수 값이 허용 범위를 벗어났습니다.", "Parameter is outside allow-listed bounds.", object_identity=field)
    if definition["type"] == "integer" and any(item != item.to_integral_value() for item in parsed):
        raise Foundation6Error("unsupported_strategy_combination", "정수 매개변수가 필요합니다.", "Integer parameter is required.", object_identity=field)
    result = tuple(int(item) if definition["type"] == "integer" else format(item.normalize(), "f") for item in parsed)
    if len(set(result)) != len(result):
        raise Foundation6Error("unsupported_strategy_combination", "중복 매개변수 값은 허용되지 않습니다.", "Duplicate normalized values are not allowed.", object_identity=field)
    return tuple(sorted(result, key=lambda item: Decimal(str(item))))


def normalize_selection(catalog: OptionCatalog, request: Mapping[str, Any]) -> dict[str, Any]:
    if set(request).difference({"catalog_schema_version", "components", "universe_size", "asset_group_data_available", "history_sessions", "evaluation_profile_ids"}):
        raise Foundation6Error("unsupported_strategy_combination", "알 수 없는 전략 구성 필드입니다.", "Unknown strategy construction field.")
    if request.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION or not isinstance(request.get("components"), Mapping):
        raise Foundation6Error("catalog_version_mismatch", "카탈로그 버전 또는 구성 형식이 일치하지 않습니다.", "Catalog version or construction shape does not match.")
    components: dict[str, Any] = {}
    for category, default in catalog.defaults().items():
        supplied = request["components"].get(category)
        if supplied is None:
            supplied = {
                "option_id": default["option_id"],
                "parameters": {
                    name: {"kind": "fixed", "value": definition["default"]}
                    for name, definition in default["parameters"].items()
                },
            }
        if not isinstance(supplied, Mapping) or set(supplied).difference({"option_id", "parameters"}):
            raise Foundation6Error("unsupported_strategy_combination", "구성 요소 형식이 올바르지 않습니다.", "Component shape is invalid.", object_identity=category)
        option = catalog.option(category, str(supplied.get("option_id")))
        parameters = supplied.get("parameters", {})
        if not isinstance(parameters, Mapping) or set(parameters) != set(option["parameters"]):
            raise Foundation6Error("unsupported_strategy_combination", "옵션 매개변수가 카탈로그 정의와 일치하지 않습니다.", "Option parameters do not match the catalog definition.", object_identity=category)
        components[category] = {"option_id": option["option_id"], "parameters": {key: _values(parameters[key], definition, f"{category}.{key}") for key, definition in option["parameters"].items()}, "engine_adapter_support": option["engine_adapter_support"]}
    normalized = {"catalog_schema_version": CATALOG_SCHEMA_VERSION, "catalog_hash": catalog.catalog_hash, "components": components,
                  "universe_size": int(request.get("universe_size", 470)), "asset_group_data_available": bool(request.get("asset_group_data_available", True)),
                  "history_sessions": int(request.get("history_sessions", 252)), "evaluation_profile_ids": tuple(sorted(set(request.get("evaluation_profile_ids", ["default"])) ))}
    validate_compatibility(catalog, normalized)
    normalized["normalized_construction_hash"] = content_hash(normalized)
    return canonical_data(normalized)


def validate_compatibility(catalog: OptionCatalog, normalized: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    components = normalized["components"]
    violations: list[Mapping[str, Any]] = []
    rules = {item["rule_id"]: item for item in catalog.document["compatibility_rules"]}
    if int(normalized["history_sessions"]) < 20 and any(components[key]["option_id"].endswith("low20_v1") for key in ("initial_stop", "trailing_exit")):
        violations.append(rules["requires_low20_history"])
    max_positions = int(components["portfolio_constraints"]["parameters"]["maximum_position_count"][0])
    if max_positions > int(normalized["universe_size"]): violations.append(rules["position_count_within_universe"])
    group_cap = Decimal(str(components["portfolio_constraints"]["parameters"]["asset_group_exposure_cap_pct"][0]))
    if group_cap < 1 and not normalized["asset_group_data_available"]: violations.append(rules["group_cap_requires_group_data"])
    if violations:
        first = violations[0]
        raise Foundation6Error("unsupported_strategy_combination", first["message_ko"], first["diagnostic_en"], object_identity=first["rule_id"])
    return tuple(canonical_data(item) for item in rules.values())


def estimate_candidates(catalog: OptionCatalog, request: Mapping[str, Any], *, reusable_hashes: Iterable[str] = ()) -> dict[str, Any]:
    normalized = normalize_selection(catalog, request)
    dimensions: list[tuple[str, tuple[Any, ...]]] = []
    for category, component in normalized["components"].items():
        for name, values in component["parameters"].items(): dimensions.append((f"{category}.{name}", tuple(values)))
    raw = 1
    for _, values in dimensions: raw *= len(values)
    candidates: dict[str, Mapping[str, Any]] = {}
    import itertools
    for values in itertools.product(*(values for _, values in dimensions)) if dimensions else [()]:
        candidate = canonical_data(normalized)
        for (path, _), value in zip(dimensions, values):
            category, parameter = path.split(".", 1); candidate["components"][category]["parameters"][parameter] = value
        identity = content_hash({"catalog_hash": catalog.catalog_hash, "economic": candidate["components"]})
        candidates[identity] = candidate
    ordered = tuple(sorted(candidates))
    reusable = set(reusable_hashes)
    reused = sum(item in reusable for item in ordered)
    profile_count = len(normalized["evaluation_profile_ids"])
    result = {"schema_version": "candidate_space_estimate_v2", "catalog_hash": catalog.catalog_hash,
              "raw_cartesian_combinations": raw, "invalid_incompatible_combinations": 0, "canonical_duplicates": raw - len(ordered),
              "valid_unique_economic_candidates": len(ordered), "reusable_completed_candidates": reused,
              "new_candidates_requiring_execution": len(ordered) - reused, "evaluation_only_applications": len(ordered) * profile_count,
              "robustness_workload": 0, "total_estimated_work": (len(ordered) - reused) + len(ordered) * profile_count,
              "rejected_by_rule": {}, "candidate_economic_hashes": ordered, "normalized_construction": normalized}
    result["candidate_estimate_hash"] = content_hash(result)
    return result


class PersistedExecutionManager:
    """Single-host durable queue and candidate lifecycle projector."""
    def __init__(self, root: str | Path, catalog: OptionCatalog, *, host_identity: str | None = None) -> None:
        self.root, self.catalog = Path(root), catalog
        self.host_identity = host_identity or socket.gethostname()
        self._lock = threading.RLock(); self.root.mkdir(parents=True, exist_ok=True)
        self._projection = self._rebuild()

    def _request_path(self, request_id: str) -> Path: return self.root / "requests" / f"{request_id}.json"
    def _event_paths(self) -> list[Path]: return sorted((self.root / "events").glob("*.json")) if (self.root / "events").exists() else []
    def _rebuild(self) -> dict[str, Any]:
        projection: dict[str, Any] = {"requests": {}, "candidates": {}, "workers": {}, "history": []}
        for path in sorted((self.root / "requests").glob("*.json")) if (self.root / "requests").exists() else []:
            record = _read_record(path); projection["requests"][record["execution_request_id"]] = record
        for path in self._event_paths():
            event = _read_record(path); projection["history"].append(event); typ, payload = event["event_type"], event["payload"]
            if typ == "candidate_state": projection["candidates"][(payload["execution_request_id"], payload["candidate_economic_hash"])] = payload
            elif typ == "worker": projection["workers"][payload["worker_id"]] = payload
        return projection

    def _event(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            ordinal = len(self._event_paths()) + 1
            event = _hashed({"schema_version": EVENT_SCHEMA_VERSION, "event_id": deterministic_id("execution_management_event", {"ordinal": ordinal, "event_type": event_type, "payload": payload}), "ordinal": ordinal, "timestamp": _now(), "event_type": event_type, "payload": canonical_data(payload)})
            _atomic_write(self.root / "events" / f"{ordinal:020d}-{event['event_id']}.json", event); self._projection = self._rebuild(); return event

    def create_request(self, estimate: Mapping[str, Any]) -> Mapping[str, Any]:
        if estimate.get("catalog_hash") != self.catalog.catalog_hash: raise Foundation6Error("catalog_version_mismatch", "카탈로그가 추정 이후 변경되었습니다.", "Catalog changed after estimation.")
        request_id = deterministic_id("persisted_execution_request", {"estimate_hash": estimate.get("candidate_estimate_hash"), "catalog_hash": self.catalog.catalog_hash})
        record = _hashed({"schema_version": MANAGER_SCHEMA_VERSION, "execution_request_id": request_id, "catalog_hash": self.catalog.catalog_hash, "candidate_estimate_hash": estimate["candidate_estimate_hash"], "candidate_economic_hashes": list(estimate["candidate_economic_hashes"]), "created_timestamp": _now(), "normalized_construction": estimate["normalized_construction"]})
        _atomic_write(self._request_path(request_id), record); self._projection = self._rebuild()
        for ordinal, candidate_hash in enumerate(record["candidate_economic_hashes"], 1): self._candidate(request_id, candidate_hash, ordinal, "pending")
        return record

    def track_controlled_request(self, request: Mapping[str, Any], attempts: Iterable[Any]) -> Mapping[str, Any]:
        """Project an existing Foundation-5 request without creating another queue.

        The controlled executor and ExecutionAttempt repository remain authoritative
        for economic work.  This manager merely persists the candidate identity
        projection that the workspace needs for restart-safe progress inspection.
        """
        request_id = request.get("execution_request_id")
        candidates = request.get("requested_strategy_run_candidates")
        if not isinstance(request_id, str) or not isinstance(candidates, (list, tuple)):
            raise Foundation6Error("execution_request_corrupt", "기존 실행 요청 참조가 올바르지 않습니다.", "Controlled execution request is invalid.")
        candidate_ids = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("strategy_run_id"), str):
                raise Foundation6Error("execution_request_corrupt", "후보 StrategyRun 참조가 올바르지 않습니다.", "Controlled candidate identity is invalid.")
            candidate_ids.append(candidate["strategy_run_id"])
        record = _hashed({
            "schema_version": MANAGER_SCHEMA_VERSION,
            "execution_request_id": request_id,
            "catalog_hash": self.catalog.catalog_hash,
            "candidate_estimate_hash": request.get("candidate_estimate_hash"),
            "candidate_economic_hashes": candidate_ids,
            "created_timestamp": request.get("request_timestamp", _now()),
            "normalized_construction": request.get("normalized_construction", {}),
            "projection_source": "controlled_execution_request_v1",
        })
        _atomic_write(self._request_path(request_id), record)
        self._projection = self._rebuild()
        by_run = {getattr(item, "intended_strategy_run_id", None): item for item in attempts}
        for ordinal, candidate_id in enumerate(candidate_ids, 1):
            attempt = by_run.get(candidate_id)
            state = "pending"
            extra: dict[str, Any] = {"strategy_run_id": candidate_id, "projection_source": "controlled_execution_request_v1"}
            if attempt is not None:
                operational = getattr(getattr(attempt, "operational_status", None), "value", None)
                state = {"queued": "pending", "pending": "pending", "running": "running", "cancelling": "running", "completed": "succeeded", "failed": "failed", "cancelled": "cancelled"}.get(operational, "blocked")
                extra["execution_attempt_id"] = getattr(attempt, "execution_attempt_id", None)
            previous = self._projection["candidates"].get((request_id, candidate_id))
            comparable = {key: value for key, value in (previous or {}).items() if key not in {"timestamp"}}
            expected = {"execution_request_id": request_id, "candidate_economic_hash": candidate_id, "candidate_ordinal": ordinal, "state": state, "economic_specification_hash": candidate_id, **extra}
            if not previous or any(comparable.get(key) != value for key, value in expected.items()):
                self._candidate(request_id, candidate_id, ordinal, state, **extra)
        return self.status(request_id)

    def _candidate(self, request_id: str, candidate_hash: str, ordinal: int, state: str, **extra: Any) -> Mapping[str, Any]:
        if state not in _CANDIDATE_STATES: raise ValueError("invalid candidate state")
        return self._event("candidate_state", {"execution_request_id": request_id, "candidate_economic_hash": candidate_hash, "candidate_ordinal": ordinal, "state": state, "economic_specification_hash": candidate_hash, "timestamp": _now(), **extra})

    def register_worker(self, worker_id: str, process_id: int, attempt_id: str | None = None, current_candidate: str | None = None) -> Mapping[str, Any]:
        if not worker_id or process_id <= 0: raise Foundation6Error("worker_stale", "작업자 식별 정보가 올바르지 않습니다.", "Worker identity is invalid.")
        return self._event("worker", {"worker_id": worker_id, "process_id": process_id, "host_identity": self.host_identity, "started_timestamp": _now(), "heartbeat_timestamp": _now(), "attempt_id": attempt_id, "current_candidate": current_candidate, "engine_version": "trend_v2_controlled_library_v2", "source_commit": "local"})

    def heartbeat(self, worker_id: str) -> Mapping[str, Any]:
        worker = self._projection["workers"].get(worker_id)
        if worker is None: raise Foundation6Error("worker_stale", "등록되지 않은 작업자입니다.", "Worker is not registered.", object_identity=worker_id)
        return self._event("worker", {**worker, "heartbeat_timestamp": _now()})

    def lease_next(self, request_id: str, worker_id: str) -> Mapping[str, Any] | None:
        with self._lock:
            if worker_id not in self._projection["workers"]: raise Foundation6Error("worker_stale", "작업자가 등록되지 않았습니다.", "Worker is not registered.")
            request = self._projection["requests"].get(request_id)
            if request is None: raise Foundation6Error("execution_request_corrupt", "실행 요청을 찾을 수 없습니다.", "Execution request was not found.", object_identity=request_id)
            for ordinal, candidate_hash in enumerate(request["candidate_economic_hashes"], 1):
                state = self._projection["candidates"].get((request_id, candidate_hash), {})
                if state.get("state") == "pending":
                    self._candidate(request_id, candidate_hash, ordinal, "running", worker_id=worker_id, lease_id=deterministic_id("candidate_lease", {"request": request_id, "candidate": candidate_hash, "worker": worker_id}), started_timestamp=_now())
                    return self._projection["candidates"][(request_id, candidate_hash)]
            return None

    def complete_candidate(self, request_id: str, candidate_hash: str, *, succeeded: bool, artifact_references: Iterable[Mapping[str, Any]] = (), failure_code: str | None = None, failure_message: str | None = None) -> None:
        current = self._projection["candidates"].get((request_id, candidate_hash))
        if not current or current["state"] != "running": raise Foundation6Error("candidate_lease_conflict", "후보 실행 임대가 유효하지 않습니다.", "Candidate lease is not active.", object_identity=candidate_hash)
        self._candidate(request_id, candidate_hash, int(current["candidate_ordinal"]), "succeeded" if succeeded else "failed", completed_timestamp=_now(), artifact_references=list(artifact_references), failure_code=failure_code, failure_message=failure_message, provenance={"catalog_hash": self.catalog.catalog_hash})

    def reconcile(self, request_id: str) -> Mapping[str, Any]:
        request = self._projection["requests"].get(request_id)
        if request is None: raise Foundation6Error("execution_request_corrupt", "실행 요청을 찾을 수 없습니다.", "Execution request was not found.", object_identity=request_id)
        decisions = []
        for ordinal, candidate_hash in enumerate(request["candidate_economic_hashes"], 1):
            state = self._projection["candidates"].get((request_id, candidate_hash), {})
            if state.get("state") == "running":
                self._candidate(request_id, candidate_hash, ordinal, "blocked", failure_code="attempt_interrupted", failure_message="로컬 프로세스 재시작 후 실행 중 후보를 보수적으로 중단 처리했습니다.", recovery="interrupted_running_no_live_worker")
                decisions.append({"candidate_economic_hash": candidate_hash, "classification": "running_with_no_live_worker", "action": "blocked"})
        self._event("recovery", {"execution_request_id": request_id, "decisions": decisions, "reconciled_timestamp": _now()})
        return {"execution_request_id": request_id, "decisions": decisions, "reconciled": True}

    def resume(self, request_id: str) -> Mapping[str, Any]:
        request = self._projection["requests"].get(request_id)
        if request is None: raise Foundation6Error("execution_request_corrupt", "실행 요청을 찾을 수 없습니다.", "Execution request was not found.")
        requeued = []
        for ordinal, candidate_hash in enumerate(request["candidate_economic_hashes"], 1):
            state = self._projection["candidates"].get((request_id, candidate_hash), {})
            if state.get("state") in {"pending", "failed", "cancelled", "blocked"}:
                self._candidate(request_id, candidate_hash, ordinal, "pending", retry_of_state=state.get("state"), resume_timestamp=_now()); requeued.append(candidate_hash)
        self._event("recovery", {"execution_request_id": request_id, "action": "resume", "requeued": requeued})
        return {"execution_request_id": request_id, "requeued": requeued, "reused": [key for key in request["candidate_economic_hashes"] if self._projection["candidates"].get((request_id, key), {}).get("state") in {"succeeded", "reused"}]}

    def status(self, request_id: str | None = None) -> Mapping[str, Any]:
        candidates = [value for (rid, _), value in self._projection["candidates"].items() if request_id is None or rid == request_id]
        return {"schema_version": MANAGER_SCHEMA_VERSION, "source_of_truth": "append_only_request_and_event_records", "request_count": len(self._projection["requests"]), "requests": [value for key, value in self._projection["requests"].items() if request_id in (None, key)], "candidates": sorted(candidates, key=lambda item: (item["execution_request_id"], item["candidate_ordinal"])), "workers": list(self._projection["workers"].values()), "recovery_history": [event["payload"] for event in self._projection["history"] if event["event_type"] == "recovery"]}
