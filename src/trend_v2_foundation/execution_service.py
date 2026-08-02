"""Immutable request repository and conservative one-worker execution service."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .construction import (
    CandidateSpaceEstimate,
    ExecutionConfirmation,
    ExecutionRequest,
    Foundation5Error,
    ImmutableJsonRepository,
    LocalExecutionPolicy,
    NormalizedConstruction,
    create_execution_request,
    estimate_candidate_space,
    normalize_construction,
    validate_confirmation,
)
from .contracts import EvaluationProfile, ExecutionStatus, StrategyRunManifest, StrategyRunSpec
from .engine_adapter import EconomicExecutionAdapter
from .execution import (
    AttemptOperationalStatus,
    AttemptTerminalOutcome,
    ExecutionAttempt,
    FileExecutionAttemptRepository,
    TERMINAL_ATTEMPT_STATUSES,
)
from .integration import calculate_and_evaluate_saved_runs
from .result_store import LocalResultStore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ControlledExecutionService:
    """Control construction and bounded local execution without a queue service."""

    def __init__(
        self,
        store: LocalResultStore,
        attempt_repository: FileExecutionAttemptRepository,
        adapter: EconomicExecutionAdapter,
        policy: LocalExecutionPolicy,
        profiles: Mapping[str, EvaluationProfile],
        *,
        source_commit: str,
        clock: Callable[[], str] = utc_now,
        background: bool = True,
    ) -> None:
        self.store = store
        self.attempt_repository = attempt_repository
        self.adapter = adapter
        self.policy = policy
        self.profiles = dict(profiles)
        self.source_commit = source_commit
        self.clock = clock
        self.background = background
        self.repository = ImmutableJsonRepository(store.root / "controlled_execution")
        self._executor = ThreadPoolExecutor(
            max_workers=policy.maximum_concurrency,
            thread_name_prefix="trend-v2-controlled",
        )
        self._lock = threading.RLock()
        self._cancel_events: dict[str, threading.Event] = {}

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def normalize(self, construction: Mapping[str, Any]) -> NormalizedConstruction:
        return normalize_construction(construction, self.policy)

    def _existing_reusable(self, specification: StrategyRunSpec) -> bool:
        try:
            manifest = self.store.get_strategy_run_manifest(specification.strategy_run_id)
        except KeyError:
            return False
        validation = self.store.validate_manifest(specification.strategy_run_id)
        if not validation.valid or manifest.execution_status != ExecutionStatus.SUCCEEDED:
            return False
        keys = {artifact.artifact_key for artifact in manifest.artifacts}
        return {"daily_portfolio_curve", "benchmark_daily_portfolio_curve"} <= keys

    def estimate(
        self, construction: Mapping[str, Any]
    ) -> tuple[NormalizedConstruction, CandidateSpaceEstimate, tuple[StrategyRunSpec, ...]]:
        normalized = self.normalize(construction)
        missing_profiles = sorted(set(normalized.normalized["evaluation_profile_ids"]) - set(self.profiles))
        if missing_profiles:
            raise Foundation5Error(
                "unsupported_option",
                f"Unknown evaluation profile IDs: {missing_profiles}.",
                object_identity=missing_profiles[0],
            )
        estimate, candidates = estimate_candidate_space(
            normalized, self.policy, reusable=self._existing_reusable
        )
        return normalized, estimate, candidates

    def confirm(
        self,
        construction: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> ExecutionConfirmation:
        normalized, estimate, _ = self.estimate(construction)
        created = self.clock()
        confirmation = ExecutionConfirmation.create(
            normalized, estimate, self.policy, created_timestamp=created
        )
        bound = self.repository.bind_idempotency(
            "confirmation",
            idempotency_key,
            content_hash_payload({"construction": construction}),
            confirmation.confirmation_id,
        )
        if bound != confirmation.confirmation_id:
            return ExecutionConfirmation(**{
                key: value
                for key, value in self.repository.get("confirmations", bound).items()
                if key != "confirmation_id"
            })
        self.repository.save("confirmations", confirmation.confirmation_id, confirmation.to_dict())
        return confirmation

    def create_request(
        self,
        construction: Mapping[str, Any],
        *,
        confirmation_id: str | None,
        idempotency_key: str,
    ) -> ExecutionRequest:
        normalized, estimate, candidates = self.estimate(construction)
        confirmation: ExecutionConfirmation | None = None
        if confirmation_id:
            try:
                payload = self.repository.get("confirmations", confirmation_id)
            except KeyError as error:
                raise Foundation5Error("confirmation_invalid", "Confirmation identity was not found.", status_code=409) from error
            confirmation = ExecutionConfirmation(**{key: value for key, value in payload.items() if key != "confirmation_id"})
        request = create_execution_request(
            normalized,
            estimate,
            candidates,
            self.policy,
            request_timestamp=self.clock(),
            source_commit=self.source_commit,
            confirmation=confirmation,
        )
        if (
            normalized.normalized["walk_forward"]["fold_count"] > 0
            or normalized.normalized["robustness"]["scenario_count"] > 0
        ):
            raise Foundation5Error(
                "engine_unsupported",
                "The initial controlled adapter estimates fold and robustness work but does not execute it.",
                next_action_ko="워크포워드 fold와 강건성 시나리오를 0으로 설정해 주세요.",
            )
        request_hash = content_hash_payload({"construction": construction, "confirmation_id": confirmation_id})
        bound = self.repository.bind_idempotency(
            "execution_request", idempotency_key, request_hash, request.execution_request_id
        )
        if bound != request.execution_request_id:
            return ExecutionRequest.from_dict(self.repository.get("execution_requests", bound))
        self.repository.save("execution_requests", request.execution_request_id, request.to_dict())
        return request

    def get_request(self, execution_request_id: str) -> ExecutionRequest:
        try:
            return ExecutionRequest.from_dict(
                self.repository.get("execution_requests", execution_request_id)
            )
        except KeyError as error:
            raise Foundation5Error(
                "not_found", "Execution request was not found.", object_identity=execution_request_id, status_code=404
            ) from error

    def _start_record(self, request_id: str) -> Mapping[str, Any] | None:
        try:
            return self.repository.get("request_starts", request_id)
        except KeyError:
            return None

    def _active_duplicate(self, strategy_run_id: str) -> ExecutionAttempt | None:
        for attempt in self.attempt_repository.list():
            if (
                attempt.intended_strategy_run_id == strategy_run_id
                and attempt.operational_status not in TERMINAL_ATTEMPT_STATUSES
            ):
                return attempt
        return None

    def _validate_request_confirmation(self, request: ExecutionRequest) -> None:
        estimate = _estimate_from_dict(request.candidate_estimate)
        if not estimate.confirmation_required:
            return
        if not request.confirmation_id:
            raise Foundation5Error("confirmation_required", "Execution request lacks its required confirmation.", status_code=409)
        payload = self.repository.get("confirmations", request.confirmation_id)
        confirmation = ExecutionConfirmation(**{key: value for key, value in payload.items() if key != "confirmation_id"})
        normalized = _normalized_from_dict(request.normalized_construction)
        validate_confirmation(confirmation, normalized, estimate, self.policy, now=self.clock())
        try:
            usage = self.repository.get("confirmation_usage", confirmation.confirmation_id)
        except KeyError:
            usage = None
        if usage and usage["execution_request_id"] != request.execution_request_id:
            raise Foundation5Error("confirmation_invalid", "One-time confirmation was already used.", status_code=409)
        self.repository.save(
            "confirmation_usage",
            confirmation.confirmation_id,
            {"confirmation_id": confirmation.confirmation_id, "execution_request_id": request.execution_request_id},
        )

    def start(self, execution_request_id: str, *, idempotency_key: str) -> Mapping[str, Any]:
        request = self.get_request(execution_request_id)
        request_hash = content_hash_payload({"execution_request_id": execution_request_id})
        with self._lock:
            previous = self._start_record(execution_request_id)
            if previous is not None:
                self.repository.bind_idempotency("start", idempotency_key, request_hash, execution_request_id)
                return previous
            self._validate_request_confirmation(request)
            candidates = tuple(StrategyRunSpec.from_dict(item) for item in request.requested_strategy_run_candidates)
            for candidate in candidates:
                duplicate = self._active_duplicate(candidate.strategy_run_id)
                if duplicate is not None:
                    raise Foundation5Error(
                        "duplicate_active_execution",
                        "An equivalent StrategyRun is already active in another attempt.",
                        object_identity=duplicate.execution_attempt_id,
                        status_code=409,
                    )
            attempts = []
            total = len(candidates)
            for ordinal, candidate in enumerate(candidates, start=1):
                previous_attempts = sorted(
                    (
                        item
                        for item in self.attempt_repository.list()
                        if item.intended_strategy_run_id == candidate.strategy_run_id
                    ),
                    key=lambda item: (item.attempt_number, item.execution_attempt_id),
                )
                attempt_number = previous_attempts[-1].attempt_number + 1 if previous_attempts else 1
                retry_parent = previous_attempts[-1].execution_attempt_id if previous_attempts else None
                attempt = ExecutionAttempt.create(
                    candidate,
                    attempt_number=attempt_number,
                    retry_of_execution_attempt_id=retry_parent,
                    created_timestamp=self.clock(),
                    source_commit=request.source_commit,
                    engine_version=request.engine_version,
                    operational_status=AttemptOperationalStatus.QUEUED,
                    progress_summary={
                        "execution_request_id": execution_request_id,
                        "candidate_ordinal": ordinal,
                        "candidate_total": total,
                        "completed_count": 0,
                        "reused_count": 0,
                        "failed_count": 0,
                        "cancelled_count": 0,
                    },
                    worker_metadata={"mode": "bounded_local", "maximum_concurrency": self.policy.maximum_concurrency},
                )
                self.attempt_repository.save(attempt)
                self._cancel_events[attempt.execution_attempt_id] = threading.Event()
                attempts.append(attempt)
            record = {
                "execution_request_id": execution_request_id,
                "execution_attempt_ids": [item.execution_attempt_id for item in attempts],
                "candidate_count": total,
                "start_idempotent": True,
            }
            self.repository.save("request_starts", execution_request_id, record)
            self.repository.bind_idempotency("start", idempotency_key, request_hash, execution_request_id)
            if self.background:
                self._executor.submit(self._execute_request, request, tuple(attempts))
            else:
                self._execute_request(request, tuple(attempts))
            return record

    def _assert_reusable(self, specification: StrategyRunSpec) -> StrategyRunManifest | None:
        try:
            manifest = self.store.get_strategy_run_manifest(specification.strategy_run_id)
        except KeyError:
            return None
        validation = self.store.validate_manifest(specification.strategy_run_id)
        if not validation.valid:
            raise Foundation5Error(
                "stored_equivalent_run_corrupt",
                "Equivalent stored StrategyRun failed provenance or artifact validation.",
                object_identity=specification.strategy_run_id,
                status_code=409,
            )
        if manifest.execution_status != ExecutionStatus.SUCCEEDED:
            raise Foundation5Error(
                "stored_equivalent_run_corrupt",
                "Equivalent stored StrategyRun is not a reusable successful result.",
                object_identity=specification.strategy_run_id,
                status_code=409,
            )
        keys = {artifact.artifact_key for artifact in manifest.artifacts}
        if not {"daily_portfolio_curve", "benchmark_daily_portfolio_curve"} <= keys:
            raise Foundation5Error(
                "stored_equivalent_run_corrupt", "Equivalent run is missing required economic artifacts.", status_code=409
            )
        return manifest

    def _execute_request(
        self, request: ExecutionRequest, attempts: Sequence[ExecutionAttempt]
    ) -> None:
        successful: list[tuple[ExecutionAttempt, bool, tuple[Mapping[str, Any], ...]]] = []
        for initial in attempts:
            current = self.attempt_repository.get(initial.execution_attempt_id)
            cancel = self._cancel_events.setdefault(current.execution_attempt_id, threading.Event())
            if current.operational_status == AttemptOperationalStatus.CANCELLED:
                continue
            if cancel.is_set():
                self.attempt_repository.transition(
                    current.execution_attempt_id,
                    operational_status=AttemptOperationalStatus.CANCELLED,
                    terminal_outcome=AttemptTerminalOutcome.CANCELLED,
                    completed_timestamp=self.clock(),
                    current_stage="cancelled_before_start",
                    progress_summary={**dict(current.progress_summary), "cancelled_count": 1},
                )
                continue
            try:
                current = self.attempt_repository.transition(
                    current.execution_attempt_id,
                    operational_status=AttemptOperationalStatus.RUNNING,
                    started_timestamp=self.clock(),
                    current_stage="cache_validation",
                )
                specification = StrategyRunSpec.from_dict(current.requested_strategy_specification)
                manifest = self._assert_reusable(specification)
                reused = manifest is not None
                artifact_refs: tuple[Mapping[str, Any], ...]
                if reused:
                    artifact_refs = tuple(
                        {
                            "strategy_run_id": specification.strategy_run_id,
                            "artifact_key": artifact.artifact_key,
                            "content_hash": artifact.content_hash,
                        }
                        for artifact in manifest.artifacts
                    )
                else:
                    current = self.attempt_repository.transition(
                        current.execution_attempt_id, current_stage="economic_backtest"
                    )
                    result = self.adapter.execute(specification)
                    if cancel.is_set():
                        current = self.attempt_repository.transition(
                            current.execution_attempt_id,
                            operational_status=AttemptOperationalStatus.CANCELLING,
                            current_stage="cooperative_cancellation_boundary",
                        )
                        self.attempt_repository.transition(
                            current.execution_attempt_id,
                            operational_status=AttemptOperationalStatus.CANCELLED,
                            terminal_outcome=AttemptTerminalOutcome.CANCELLED,
                            completed_timestamp=self.clock(),
                            current_stage="cancelled_before_artifact_commit",
                            progress_summary={**dict(current.progress_summary), "cancelled_count": 1},
                        )
                        continue
                    records = tuple(
                        self.store.put_artifact(
                            artifact.artifact_key,
                            artifact.kind,
                            artifact.payload,
                            row_count=artifact.row_count,
                        ).record
                        for artifact in result.artifacts
                    )
                    manifest = StrategyRunManifest.create(
                        specification,
                        source_code_commit=request.source_commit,
                        artifacts=records,
                        creation_time=self.clock(),
                        execution_status=ExecutionStatus.SUCCEEDED,
                        warnings=result.warnings,
                        limitations=result.limitations,
                    )
                    self.store.save_strategy_run(manifest)
                    artifact_refs = tuple(
                        {
                            "strategy_run_id": specification.strategy_run_id,
                            "artifact_key": artifact.artifact_key,
                            "content_hash": artifact.content_hash,
                        }
                        for artifact in records
                    )
                successful.append((current, reused, artifact_refs))
            except Foundation5Error as error:
                latest = self.attempt_repository.get(current.execution_attempt_id)
                self.attempt_repository.transition(
                    latest.execution_attempt_id,
                    operational_status=AttemptOperationalStatus.FAILED,
                    terminal_outcome=AttemptTerminalOutcome.FAILED,
                    completed_timestamp=self.clock(),
                    current_stage="candidate_failed",
                    failure_code=error.code,
                    failure_message=error.diagnostic_en,
                    progress_summary={**dict(latest.progress_summary), "failed_count": 1},
                )
            except Exception:
                latest = self.attempt_repository.get(current.execution_attempt_id)
                self.attempt_repository.transition(
                    latest.execution_attempt_id,
                    operational_status=AttemptOperationalStatus.FAILED,
                    terminal_outcome=AttemptTerminalOutcome.FAILED,
                    completed_timestamp=self.clock(),
                    current_stage="candidate_failed",
                    failure_code="internal_execution_failure",
                    failure_message="Unexpected controlled execution failure.",
                    progress_summary={**dict(latest.progress_summary), "failed_count": 1},
                )

        run_ids = tuple(sorted(item[0].intended_strategy_run_id for item in successful))
        evaluation_ids: list[str] = []
        evaluation_failure: Exception | None = None
        if run_ids:
            try:
                for profile_id in request.selected_evaluation_profile_ids:
                    existing_evaluation = next(
                        (
                            self.store.get_evaluation_run(identity)
                            for identity in self.store.evaluation_history()
                            if (
                                self.store.get_evaluation_run(identity).evaluation_profile_id == profile_id
                                and self.store.get_evaluation_run(identity).strategy_run_ids == run_ids
                            )
                        ),
                        None,
                    )
                    if existing_evaluation is not None:
                        evaluation_ids.append(existing_evaluation.evaluation_run_id)
                        continue
                    outcome = calculate_and_evaluate_saved_runs(
                        self.store,
                        run_ids,
                        self.profiles[profile_id],
                        creation_time=self.clock(),
                    )
                    evaluation_ids.append(outcome.evaluation_run.evaluation_run_id)
            except Exception as error:  # Economic artifacts remain valid and immutable.
                evaluation_failure = error
        for attempt, reused, artifact_refs in successful:
            latest = self.attempt_repository.get(attempt.execution_attempt_id)
            if latest.operational_status in TERMINAL_ATTEMPT_STATUSES:
                continue
            if evaluation_failure is not None:
                self.attempt_repository.transition(
                    latest.execution_attempt_id,
                    operational_status=AttemptOperationalStatus.FAILED,
                    terminal_outcome=AttemptTerminalOutcome.FAILED,
                    completed_timestamp=self.clock(),
                    current_stage="evaluation_failed",
                    failure_code="internal_execution_failure",
                    failure_message="Economic artifacts were stored, but derived evaluation failed.",
                    progress_summary={**dict(latest.progress_summary), "failed_count": 1},
                    artifact_references=artifact_refs,
                )
                continue
            refs = (*artifact_refs, *(
                {"evaluation_run_id": identity, "artifact_key": "evaluation_run"}
                for identity in evaluation_ids
            ))
            self.attempt_repository.transition(
                latest.execution_attempt_id,
                operational_status=AttemptOperationalStatus.COMPLETED,
                terminal_outcome=AttemptTerminalOutcome.SUCCEEDED,
                completed_timestamp=self.clock(),
                current_stage="complete",
                progress_summary={
                    **dict(latest.progress_summary),
                    "completed_count": 1,
                    "reused_count": int(reused),
                    "evaluation_run_count": len(evaluation_ids),
                },
                artifact_references=tuple(refs),
            )

    def cancel(self, execution_attempt_id: str, *, idempotency_key: str) -> ExecutionAttempt:
        with self._lock:
            try:
                current = self.attempt_repository.get(execution_attempt_id)
            except KeyError as error:
                raise Foundation5Error("not_found", "Execution attempt was not found.", status_code=404) from error
            self.repository.bind_idempotency(
                "cancel", idempotency_key, content_hash_payload({"attempt": execution_attempt_id}), execution_attempt_id
            )
            if current.operational_status == AttemptOperationalStatus.CANCELLED:
                return current
            if current.operational_status in TERMINAL_ATTEMPT_STATUSES:
                raise Foundation5Error("attempt_not_cancellable", "Terminal attempt cannot be cancelled.", status_code=409)
            event = self._cancel_events.setdefault(execution_attempt_id, threading.Event())
            event.set()
            if current.operational_status in {AttemptOperationalStatus.PENDING, AttemptOperationalStatus.QUEUED}:
                return self.attempt_repository.transition(
                    execution_attempt_id,
                    operational_status=AttemptOperationalStatus.CANCELLED,
                    terminal_outcome=AttemptTerminalOutcome.CANCELLED,
                    completed_timestamp=self.clock(),
                    current_stage="cancelled_before_start",
                    progress_summary={**dict(current.progress_summary), "cancelled_count": 1},
                )
            if current.operational_status == AttemptOperationalStatus.RUNNING:
                return self.attempt_repository.transition(
                    execution_attempt_id,
                    operational_status=AttemptOperationalStatus.CANCELLING,
                    current_stage="cooperative_cancellation_requested",
                )
            return current

    def retry(self, execution_attempt_id: str, *, idempotency_key: str) -> ExecutionAttempt:
        with self._lock:
            try:
                parent = self.attempt_repository.get(execution_attempt_id)
            except KeyError as error:
                raise Foundation5Error("not_found", "Execution attempt was not found.", status_code=404) from error
            if parent.operational_status not in {AttemptOperationalStatus.FAILED, AttemptOperationalStatus.CANCELLED}:
                raise Foundation5Error("retry_not_allowed", "Only failed or cancelled attempts may be retried.", status_code=409)
            request_id = str(parent.progress_summary.get("execution_request_id", ""))
            request = self.get_request(request_id)
            siblings = [
                item for item in self.attempt_repository.list()
                if item.intended_strategy_run_id == parent.intended_strategy_run_id
            ]
            next_number = max(item.attempt_number for item in siblings) + 1
            retry = ExecutionAttempt.create(
                StrategyRunSpec.from_dict(parent.requested_strategy_specification),
                attempt_number=next_number,
                retry_of_execution_attempt_id=parent.execution_attempt_id,
                created_timestamp=self.clock(),
                source_commit=parent.source_commit,
                engine_version=parent.engine_version,
                operational_status=AttemptOperationalStatus.QUEUED,
                progress_summary={**dict(parent.progress_summary), "completed_count": 0, "failed_count": 0, "cancelled_count": 0},
                worker_metadata=parent.worker_metadata,
            )
            bound = self.repository.bind_idempotency(
                "retry",
                idempotency_key,
                content_hash_payload({"attempt": execution_attempt_id}),
                retry.execution_attempt_id,
            )
            if bound != retry.execution_attempt_id:
                return self.attempt_repository.get(bound)
            self.attempt_repository.save(retry)
            self._cancel_events[retry.execution_attempt_id] = threading.Event()
            if self.background:
                self._executor.submit(self._execute_request, request, (retry,))
            else:
                self._execute_request(request, (retry,))
            return retry

    def request_status(self, execution_request_id: str) -> Mapping[str, Any]:
        request = self.get_request(execution_request_id)
        start = self._start_record(execution_request_id)
        attempts = [
            item.to_dict()
            for item in self.attempt_repository.list()
            if item.progress_summary.get("execution_request_id") == execution_request_id
        ]
        counts: dict[str, int] = {}
        for attempt in attempts:
            status = str(attempt["operational_status"])
            counts[status] = counts.get(status, 0) + 1
        return {**request.to_dict(), "start": start, "attempts": attempts, "attempt_status_counts": counts}


def content_hash_payload(value: Any) -> str:
    from .canonical import content_hash

    return content_hash(value)


def _normalized_from_dict(value: Mapping[str, Any]) -> NormalizedConstruction:
    return NormalizedConstruction(
        normalized=value["normalized"],
        raw_dimension_counts=value["raw_dimension_counts"],
        schema_version=value["schema_version"],
    )


def _estimate_from_dict(value: Mapping[str, Any]) -> CandidateSpaceEstimate:
    payload = dict(value)
    expected = payload.pop("candidate_estimate_hash", None)
    payload["threshold_results"] = tuple(payload["threshold_results"])
    estimate = CandidateSpaceEstimate(**payload)
    if expected and expected != estimate.estimate_hash:
        raise Foundation5Error("confirmation_stale", "Stored estimate hash is invalid.", status_code=409)
    return estimate
