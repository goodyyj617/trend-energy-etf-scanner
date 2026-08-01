"""Operational execution-attempt contracts kept separate from economic runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .canonical import canonical_bytes, canonical_data, deep_freeze, deterministic_id
from .contracts import StrategyRunSpec


EXECUTION_ATTEMPT_SCHEMA_VERSION = "execution_attempt_v1"
EXECUTION_ATTEMPT_REPOSITORY_VERSION = "file_execution_attempt_repository_v1"


class AttemptOperationalStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"


class AttemptTerminalOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


TERMINAL_ATTEMPT_STATUSES = frozenset(
    {
        AttemptOperationalStatus.CANCELLED,
        AttemptOperationalStatus.FAILED,
        AttemptOperationalStatus.COMPLETED,
    }
)


def _timestamp(value: str | None, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an ISO-8601 timestamp or null")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ExecutionAttempt:
    execution_attempt_id: str
    intended_strategy_run_id: str
    requested_strategy_specification: Mapping[str, Any]
    attempt_number: int
    retry_of_execution_attempt_id: str | None
    created_timestamp: str
    started_timestamp: str | None
    completed_timestamp: str | None
    operational_status: AttemptOperationalStatus
    terminal_outcome: AttemptTerminalOutcome | None
    failure_code: str | None
    failure_message: str | None
    progress_summary: Mapping[str, Any]
    current_stage: str | None
    artifact_references: tuple[Mapping[str, Any], ...]
    source_commit: str
    engine_version: str
    worker_metadata: Mapping[str, Any]
    schema_version: str = EXECUTION_ATTEMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_ATTEMPT_SCHEMA_VERSION:
            raise ValueError(f"unsupported execution-attempt schema: {self.schema_version}")
        if (
            isinstance(self.attempt_number, bool)
            or not isinstance(self.attempt_number, int)
            or self.attempt_number < 1
        ):
            raise ValueError("attempt_number must be a positive integer")
        if self.attempt_number == 1 and self.retry_of_execution_attempt_id is not None:
            raise ValueError("the first attempt cannot identify a retry parent")
        if self.attempt_number > 1 and not self.retry_of_execution_attempt_id:
            raise ValueError("retry attempts require retry_of_execution_attempt_id")
        if not self.source_commit or not self.engine_version:
            raise ValueError("source_commit and engine_version are required")
        specification = StrategyRunSpec.from_dict(self.requested_strategy_specification)
        if specification.strategy_run_id != self.intended_strategy_run_id:
            raise ValueError(
                "intended_strategy_run_id does not match requested strategy specification"
            )
        object.__setattr__(
            self, "requested_strategy_specification", deep_freeze(self.requested_strategy_specification)
        )
        object.__setattr__(self, "progress_summary", deep_freeze(self.progress_summary))
        object.__setattr__(
            self,
            "artifact_references",
            tuple(deep_freeze(item) for item in self.artifact_references),
        )
        object.__setattr__(self, "worker_metadata", deep_freeze(self.worker_metadata))
        if not isinstance(self.progress_summary, Mapping):
            raise ValueError("progress_summary must be a mapping")
        if not isinstance(self.worker_metadata, Mapping):
            raise ValueError("worker_metadata must be a mapping")

        created = _timestamp(self.created_timestamp, "created_timestamp")
        started = _timestamp(self.started_timestamp, "started_timestamp")
        completed = _timestamp(self.completed_timestamp, "completed_timestamp")
        if started is not None and created is not None and started < created:
            raise ValueError("started_timestamp cannot precede created_timestamp")
        if completed is not None and created is not None and completed < created:
            raise ValueError("completed_timestamp cannot precede created_timestamp")
        if completed is not None and started is not None and completed < started:
            raise ValueError("completed_timestamp cannot precede started_timestamp")

        if self.operational_status in {
            AttemptOperationalStatus.PENDING,
            AttemptOperationalStatus.QUEUED,
        }:
            if started is not None or completed is not None:
                raise ValueError("pending or queued attempts cannot have start/completion timestamps")
        elif self.operational_status in {
            AttemptOperationalStatus.RUNNING,
            AttemptOperationalStatus.CANCELLING,
        }:
            if started is None or completed is not None:
                raise ValueError("running or cancelling attempts require only a start timestamp")
        elif completed is None:
            raise ValueError("terminal attempts require a completion timestamp")

        if self.operational_status not in TERMINAL_ATTEMPT_STATUSES:
            if self.terminal_outcome is not None:
                raise ValueError("non-terminal attempts cannot have a terminal outcome")
            if self.failure_code is not None or self.failure_message is not None:
                raise ValueError("non-terminal attempts cannot have failure details")
        elif self.operational_status == AttemptOperationalStatus.COMPLETED:
            if started is None:
                raise ValueError("completed attempts require a start timestamp")
            if self.terminal_outcome not in {
                AttemptTerminalOutcome.SUCCEEDED,
                AttemptTerminalOutcome.PARTIAL,
            }:
                raise ValueError("completed attempts require succeeded or partial outcome")
            if self.failure_code is not None or self.failure_message is not None:
                raise ValueError("completed attempts cannot have failure details")
        elif self.operational_status == AttemptOperationalStatus.FAILED:
            if self.terminal_outcome != AttemptTerminalOutcome.FAILED:
                raise ValueError("failed attempts require failed terminal outcome")
            if not self.failure_code or not self.failure_message:
                raise ValueError("failed attempts require failure code and message")
        elif self.terminal_outcome != AttemptTerminalOutcome.CANCELLED:
            raise ValueError("cancelled attempts require cancelled terminal outcome")

        if self.execution_attempt_id != deterministic_id("execution_attempt", self.identity_content):
            raise ValueError("execution_attempt_id does not match immutable identity content")

    @property
    def identity_content(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intended_strategy_run_id": self.intended_strategy_run_id,
            "requested_strategy_specification": self.requested_strategy_specification,
            "attempt_number": self.attempt_number,
            "retry_of_execution_attempt_id": self.retry_of_execution_attempt_id,
            "created_timestamp": self.created_timestamp,
            "source_commit": self.source_commit,
            "engine_version": self.engine_version,
        }

    @classmethod
    def create(
        cls,
        specification: StrategyRunSpec,
        *,
        attempt_number: int,
        created_timestamp: str,
        source_commit: str,
        engine_version: str,
        retry_of_execution_attempt_id: str | None = None,
        operational_status: AttemptOperationalStatus = AttemptOperationalStatus.PENDING,
        progress_summary: Mapping[str, Any] | None = None,
        worker_metadata: Mapping[str, Any] | None = None,
    ) -> "ExecutionAttempt":
        identity = {
            "schema_version": EXECUTION_ATTEMPT_SCHEMA_VERSION,
            "intended_strategy_run_id": specification.strategy_run_id,
            "requested_strategy_specification": specification.to_dict(),
            "attempt_number": attempt_number,
            "retry_of_execution_attempt_id": retry_of_execution_attempt_id,
            "created_timestamp": created_timestamp,
            "source_commit": source_commit,
            "engine_version": engine_version,
        }
        return cls(
            execution_attempt_id=deterministic_id("execution_attempt", identity),
            intended_strategy_run_id=specification.strategy_run_id,
            requested_strategy_specification=specification.to_dict(),
            attempt_number=attempt_number,
            retry_of_execution_attempt_id=retry_of_execution_attempt_id,
            created_timestamp=created_timestamp,
            started_timestamp=None,
            completed_timestamp=None,
            operational_status=operational_status,
            terminal_outcome=None,
            failure_code=None,
            failure_message=None,
            progress_summary=progress_summary or {},
            current_stage=None,
            artifact_references=(),
            source_commit=source_commit,
            engine_version=engine_version,
            worker_metadata=worker_metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionAttempt":
        payload = dict(value)
        payload["operational_status"] = AttemptOperationalStatus(payload["operational_status"])
        outcome = payload.get("terminal_outcome")
        payload["terminal_outcome"] = AttemptTerminalOutcome(outcome) if outcome else None
        payload["artifact_references"] = tuple(payload.get("artifact_references", ()))
        return cls(**payload)


_ALLOWED_TRANSITIONS = {
    AttemptOperationalStatus.PENDING: {
        AttemptOperationalStatus.PENDING,
        AttemptOperationalStatus.QUEUED,
        AttemptOperationalStatus.RUNNING,
        AttemptOperationalStatus.CANCELLED,
        AttemptOperationalStatus.FAILED,
    },
    AttemptOperationalStatus.QUEUED: {
        AttemptOperationalStatus.QUEUED,
        AttemptOperationalStatus.RUNNING,
        AttemptOperationalStatus.CANCELLING,
        AttemptOperationalStatus.CANCELLED,
        AttemptOperationalStatus.FAILED,
    },
    AttemptOperationalStatus.RUNNING: {
        AttemptOperationalStatus.RUNNING,
        AttemptOperationalStatus.CANCELLING,
        AttemptOperationalStatus.COMPLETED,
        AttemptOperationalStatus.FAILED,
    },
    AttemptOperationalStatus.CANCELLING: {
        AttemptOperationalStatus.CANCELLING,
        AttemptOperationalStatus.CANCELLED,
        AttemptOperationalStatus.COMPLETED,
        AttemptOperationalStatus.FAILED,
    },
}


def validate_attempt_transition(previous: ExecutionAttempt, current: ExecutionAttempt) -> None:
    if previous.execution_attempt_id != current.execution_attempt_id:
        raise ValueError("execution-attempt identity cannot change during a transition")
    if previous.identity_content != current.identity_content:
        raise ValueError("execution-attempt immutable identity content cannot change")
    if previous.operational_status in TERMINAL_ATTEMPT_STATUSES:
        raise ValueError("terminal execution attempts are immutable")
    allowed = _ALLOWED_TRANSITIONS.get(previous.operational_status, set())
    if current.operational_status not in allowed:
        raise ValueError(
            f"invalid operational transition: {previous.operational_status.value} -> "
            f"{current.operational_status.value}"
        )


class ExecutionAttemptRepository(Protocol):
    def save(self, attempt: ExecutionAttempt) -> None: ...

    def get(self, execution_attempt_id: str) -> ExecutionAttempt: ...

    def list(self) -> tuple[ExecutionAttempt, ...]: ...


class FileExecutionAttemptRepository:
    """Append-only full-snapshot event history for local operational state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _event_paths(self, attempt_id: str) -> list[Path]:
        return sorted((self.root / attempt_id / "events").glob("*.json"))

    @staticmethod
    def _write_immutable(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError("execution-attempt event already exists with different content")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def save(self, attempt: ExecutionAttempt) -> None:
        events = self._event_paths(attempt.execution_attempt_id)
        if not events:
            if attempt.operational_status not in {
                AttemptOperationalStatus.PENDING,
                AttemptOperationalStatus.QUEUED,
            }:
                raise ValueError("new execution attempts must start pending or queued")
            index = 0
        else:
            previous = ExecutionAttempt.from_dict(
                json.loads(events[-1].read_text(encoding="utf-8"))
            )
            if previous == attempt:
                return
            validate_attempt_transition(previous, attempt)
            index = len(events)
        self._write_immutable(
            self.root / attempt.execution_attempt_id / "events" / f"{index:08d}.json",
            canonical_bytes(attempt.to_dict()),
        )

    def get(self, execution_attempt_id: str) -> ExecutionAttempt:
        return self.history(execution_attempt_id)[-1]

    def history(self, execution_attempt_id: str) -> tuple[ExecutionAttempt, ...]:
        events = self._event_paths(execution_attempt_id)
        if not events:
            raise KeyError(execution_attempt_id)
        snapshots = tuple(
            ExecutionAttempt.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in events
        )
        for previous, current in zip(snapshots, snapshots[1:]):
            validate_attempt_transition(previous, current)
        return snapshots

    def list(self) -> tuple[ExecutionAttempt, ...]:
        attempts = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            attempts.append(self.get(directory.name))
        return tuple(sorted(attempts, key=lambda item: item.execution_attempt_id))

    def transition(self, execution_attempt_id: str, **changes: Any) -> ExecutionAttempt:
        current = self.get(execution_attempt_id)
        updated = replace(current, **changes)
        self.save(updated)
        return updated
