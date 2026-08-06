"""Deterministic, rebuildable registry over local ResultStore artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_schemas import (
    ArtifactSchemaError,
    artifact_payload_row_count,
    validate_behavior_metadata,
    validate_daily_portfolio_curve,
    validate_robustness_summary,
    validate_rolling_metrics,
    validate_yearly_metrics,
)
from .canonical import canonical_bytes, canonical_data, content_hash, deep_freeze, deterministic_id
from .contracts import DecisionReport, DerivedMetricManifest, EvaluationProfile, EvaluationRun, StrategyRunManifest
from .execution import ExecutionAttempt, FileExecutionAttemptRepository
from .result_store import LocalResultStore


SAVED_RUN_REGISTRY_SCHEMA_VERSION = "saved_run_registry_v2"
REGISTRY_REBUILD_VERSION = "result_store_registry_rebuild_v1"


class ArtifactAvailability(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    CORRUPT = "corrupt"
    PRUNED = "pruned"
    NEVER_GENERATED = "never_generated"
    UNSUPPORTED_SCHEMA = "unsupported_schema"


class IntegrityStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    CORRUPT = "corrupt"
    PRUNED = "pruned"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    INTEGRITY_FAILED = "integrity_failed"
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class RegistryIssue:
    issue_code: str
    object_identity: str
    relative_location: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryIssue":
        return cls(**dict(value))


@dataclass(frozen=True)
class RegistryArtifact:
    artifact_key: str
    artifact_kind: str
    owner_kind: str
    owner_id: str
    content_hash: str | None
    availability: ArtifactAvailability
    integrity_status: IntegrityStatus
    retention_state: str
    row_count: int | None
    logical_bytes: int | None
    stored_bytes: int | None
    schema_version: str | None
    validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "validation_errors", tuple(self.validation_errors))

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryArtifact":
        payload = dict(value)
        payload["availability"] = ArtifactAvailability(payload["availability"])
        payload["integrity_status"] = IntegrityStatus(payload["integrity_status"])
        payload["validation_errors"] = tuple(payload.get("validation_errors", ()))
        return cls(**payload)


@dataclass(frozen=True)
class RegistryStrategyRun:
    strategy_run_id: str
    specification_hash: str
    manifest_hash: str
    canonical_specification: Mapping[str, Any]
    source_data_snapshot_id: str
    engine_version: str
    source_commit: str
    terminal_status: str
    creation_time: str
    economic_date_range: Mapping[str, str]
    artifacts: tuple[RegistryArtifact, ...]
    derived_metric_ids: tuple[str, ...]
    evaluation_profile_ids: tuple[str, ...]
    evaluation_run_ids: tuple[str, ...]
    execution_attempt_ids: tuple[str, ...]
    benchmark_provenance: tuple[Mapping[str, Any], ...]
    calculation_provenance: tuple[Mapping[str, Any], ...]
    integrity_status: IntegrityStatus
    retention_status: str
    validation_errors: tuple[str, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_specification", deep_freeze(self.canonical_specification))
        object.__setattr__(self, "economic_date_range", deep_freeze(self.economic_date_range))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "derived_metric_ids", tuple(self.derived_metric_ids))
        object.__setattr__(self, "evaluation_profile_ids", tuple(self.evaluation_profile_ids))
        object.__setattr__(self, "evaluation_run_ids", tuple(self.evaluation_run_ids))
        object.__setattr__(self, "execution_attempt_ids", tuple(self.execution_attempt_ids))
        object.__setattr__(
            self,
            "benchmark_provenance",
            tuple(deep_freeze(item) for item in self.benchmark_provenance),
        )
        object.__setattr__(
            self,
            "calculation_provenance",
            tuple(deep_freeze(item) for item in self.calculation_provenance),
        )
        object.__setattr__(self, "validation_errors", tuple(self.validation_errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryStrategyRun":
        payload = dict(value)
        payload["artifacts"] = tuple(
            RegistryArtifact.from_dict(item) for item in payload.get("artifacts", ())
        )
        payload["integrity_status"] = IntegrityStatus(payload["integrity_status"])
        for field in (
            "derived_metric_ids",
            "evaluation_profile_ids",
            "evaluation_run_ids",
            "execution_attempt_ids",
            "benchmark_provenance",
            "calculation_provenance",
            "validation_errors",
            "warnings",
            "limitations",
        ):
            payload[field] = tuple(payload.get(field, ()))
        return cls(**payload)


@dataclass(frozen=True)
class RegistryEvaluationProfile:
    evaluation_profile_id: str
    profile_hash: str
    name: str
    comparison_mode: str
    approval_status: str
    integrity_status: IntegrityStatus

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryEvaluationProfile":
        payload = dict(value)
        payload["integrity_status"] = IntegrityStatus(payload["integrity_status"])
        return cls(**payload)


@dataclass(frozen=True)
class RegistryEvaluationRun:
    evaluation_run_id: str
    evaluation_profile_id: str
    profile_hash: str
    strategy_run_ids: tuple[str, ...]
    creation_time: str
    status: str
    comparison_mode: str
    benchmark_data_identity: str
    metric_engine_version: str
    derived_metric_ids: Mapping[str, str]
    integrity_status: IntegrityStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_run_ids", tuple(self.strategy_run_ids))
        object.__setattr__(self, "derived_metric_ids", deep_freeze(self.derived_metric_ids))

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryEvaluationRun":
        payload = dict(value)
        payload["strategy_run_ids"] = tuple(payload["strategy_run_ids"])
        payload["integrity_status"] = IntegrityStatus(payload["integrity_status"])
        return cls(**payload)


@dataclass(frozen=True)
class RegistryDecisionReport:
    decision_report_id: str
    strategy_run_id: str
    evaluation_run_id: str
    evaluation_profile_id: str
    profile_hash: str
    creation_time: str
    integrity_status: IntegrityStatus

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryDecisionReport":
        payload = dict(value)
        payload["integrity_status"] = IntegrityStatus(payload["integrity_status"])
        return cls(**payload)


@dataclass(frozen=True)
class SavedRunRegistry:
    source_fingerprint: str
    strategy_runs: tuple[RegistryStrategyRun, ...]
    evaluation_profiles: tuple[RegistryEvaluationProfile, ...]
    evaluation_runs: tuple[RegistryEvaluationRun, ...]
    execution_attempts: tuple[ExecutionAttempt, ...]
    orphan_object_hashes: tuple[str, ...]
    issues: tuple[RegistryIssue, ...]
    decision_reports: tuple[RegistryDecisionReport, ...] = ()
    schema_version: str = SAVED_RUN_REGISTRY_SCHEMA_VERSION
    rebuild_version: str = REGISTRY_REBUILD_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "strategy_runs", tuple(sorted(self.strategy_runs, key=lambda item: item.strategy_run_id))
        )
        object.__setattr__(
            self,
            "evaluation_profiles",
            tuple(sorted(self.evaluation_profiles, key=lambda item: item.evaluation_profile_id)),
        )
        object.__setattr__(
            self,
            "evaluation_runs",
            tuple(sorted(self.evaluation_runs, key=lambda item: item.evaluation_run_id)),
        )
        object.__setattr__(
            self,
            "decision_reports",
            tuple(sorted(self.decision_reports, key=lambda item: item.decision_report_id)),
        )
        object.__setattr__(
            self,
            "execution_attempts",
            tuple(sorted(self.execution_attempts, key=lambda item: item.execution_attempt_id)),
        )
        object.__setattr__(self, "orphan_object_hashes", tuple(sorted(self.orphan_object_hashes)))
        object.__setattr__(
            self,
            "issues",
            tuple(
                sorted(
                    self.issues,
                    key=lambda item: (
                        item.issue_code,
                        item.object_identity,
                        item.relative_location,
                        item.detail,
                    ),
                )
            ),
        )

    @property
    def identity_content(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rebuild_version": self.rebuild_version,
            "source_fingerprint": self.source_fingerprint,
            "strategy_runs": self.strategy_runs,
            "evaluation_profiles": self.evaluation_profiles,
            "evaluation_runs": self.evaluation_runs,
            "decision_reports": self.decision_reports,
            "execution_attempts": self.execution_attempts,
            "orphan_object_hashes": self.orphan_object_hashes,
            "issues": self.issues,
        }

    @property
    def registry_id(self) -> str:
        return deterministic_id("saved_run_registry", self.identity_content)

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_data(self)
        payload["registry_id"] = self.registry_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SavedRunRegistry":
        payload = dict(value)
        expected_id = payload.pop("registry_id", None)
        payload["strategy_runs"] = tuple(
            RegistryStrategyRun.from_dict(item) for item in payload["strategy_runs"]
        )
        payload["evaluation_profiles"] = tuple(
            RegistryEvaluationProfile.from_dict(item) for item in payload["evaluation_profiles"]
        )
        payload["evaluation_runs"] = tuple(
            RegistryEvaluationRun.from_dict(item) for item in payload["evaluation_runs"]
        )
        payload["decision_reports"] = tuple(
            RegistryDecisionReport.from_dict(item)
            for item in payload.get("decision_reports", ())
        )
        payload["execution_attempts"] = tuple(
            ExecutionAttempt.from_dict(item) for item in payload["execution_attempts"]
        )
        payload["orphan_object_hashes"] = tuple(payload["orphan_object_hashes"])
        payload["issues"] = tuple(RegistryIssue.from_dict(item) for item in payload["issues"])
        registry = cls(**payload)
        if expected_id is not None and expected_id != registry.registry_id:
            raise ValueError("registry_id does not match canonical registry content")
        return registry


_ARTIFACT_VALIDATORS = {
    "daily_portfolio_curve": validate_daily_portfolio_curve,
    "benchmark_daily_portfolio_curve": validate_daily_portfolio_curve,
    "yearly_metrics": validate_yearly_metrics,
    "rolling_metrics": validate_rolling_metrics,
    "robustness_summary": validate_robustness_summary,
    "behavior_metadata": validate_behavior_metadata,
}
_DISCOVERABLE_ARTIFACT_KEYS = (
    "daily_portfolio_curve",
    "yearly_metrics",
    "rolling_metrics",
    "derived_metrics",
    "robustness_summary",
    "behavior_metadata",
)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SavedRunRegistryBuilder:
    def __init__(
        self,
        store: LocalResultStore,
        attempts: FileExecutionAttemptRepository | None = None,
    ) -> None:
        self.store = store
        self.attempts = attempts or FileExecutionAttemptRepository(
            store.root / "execution_attempts"
        )
        self.registry_dir = store.root / "registry"
        self.registry_path = self.registry_dir / f"{SAVED_RUN_REGISTRY_SCHEMA_VERSION}.json"

    def source_fingerprint(self) -> str:
        rows = []
        for path in sorted(self.store.root.rglob("*")):
            if not path.is_file() or self.registry_dir in path.parents or path.name.endswith(".tmp"):
                continue
            rows.append(
                {
                    "relative_path": path.relative_to(self.store.root).as_posix(),
                    "stored_bytes": path.stat().st_size,
                    "content_hash": _file_digest(path),
                }
            )
        return content_hash(rows)

    def _artifact(
        self,
        record: Any,
        *,
        owner_kind: str,
        owner_id: str,
        evidence_status: IntegrityStatus = IntegrityStatus.VALID,
    ) -> RegistryArtifact:
        errors: list[str] = []
        schema_version: str | None = None
        try:
            retention_event = self.store.artifact_retention_event(record.content_hash)
        except (ValueError, json.JSONDecodeError):
            retention_event = None
            errors.append("retention_event_invalid")
        if retention_event is not None:
            return RegistryArtifact(
                artifact_key=record.artifact_key,
                artifact_kind=record.kind.value,
                owner_kind=owner_kind,
                owner_id=owner_id,
                content_hash=record.content_hash,
                availability=ArtifactAvailability.PRUNED,
                integrity_status=IntegrityStatus.PRUNED,
                retention_state="pruned",
                row_count=record.row_count,
                logical_bytes=record.logical_bytes,
                stored_bytes=record.stored_bytes,
                schema_version=None,
                validation_errors=tuple(errors),
            )
        path = self.store.object_path_for_hash(record.content_hash)
        if not path.exists():
            return RegistryArtifact(
                artifact_key=record.artifact_key,
                artifact_kind=record.kind.value,
                owner_kind=owner_kind,
                owner_id=owner_id,
                content_hash=record.content_hash,
                availability=ArtifactAvailability.MISSING,
                integrity_status=IntegrityStatus.MISSING,
                retention_state="retained_reference_missing",
                row_count=record.row_count,
                logical_bytes=record.logical_bytes,
                stored_bytes=record.stored_bytes,
                schema_version=None,
                validation_errors=("artifact_missing",),
            )
        payload: Any = None
        try:
            stored = path.read_bytes()
            if len(stored) != record.stored_bytes:
                errors.append("stored_size_mismatch")
            logical = gzip.decompress(stored)
            if len(logical) != record.logical_bytes:
                errors.append("logical_size_mismatch")
            if content_hash(logical) != record.content_hash:
                errors.append("content_hash_mismatch")
            payload = json.loads(logical.decode("utf-8"))
            actual_rows = artifact_payload_row_count(payload)
            if actual_rows is not None and actual_rows != record.row_count:
                errors.append("row_count_mismatch")
            if isinstance(payload, Mapping):
                candidate = payload.get("schema_version")
                schema_version = str(candidate) if candidate is not None else None
            validator = _ARTIFACT_VALIDATORS.get(record.artifact_key)
            if validator is not None:
                validator(payload)
        except ArtifactSchemaError as error:
            if "schema_version" in str(error):
                return RegistryArtifact(
                    artifact_key=record.artifact_key,
                    artifact_kind=record.kind.value,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                    content_hash=record.content_hash,
                    availability=ArtifactAvailability.UNSUPPORTED_SCHEMA,
                    integrity_status=IntegrityStatus.UNSUPPORTED_SCHEMA,
                    retention_state="retained",
                    row_count=record.row_count,
                    logical_bytes=record.logical_bytes,
                    stored_bytes=record.stored_bytes,
                    schema_version=schema_version,
                    validation_errors=("schema_unsupported",),
                )
            errors.append("schema_validation_failed")
        except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            errors.append("artifact_corrupt")
        if errors:
            return RegistryArtifact(
                artifact_key=record.artifact_key,
                artifact_kind=record.kind.value,
                owner_kind=owner_kind,
                owner_id=owner_id,
                content_hash=record.content_hash,
                availability=ArtifactAvailability.CORRUPT,
                integrity_status=IntegrityStatus.CORRUPT,
                retention_state="retained",
                row_count=record.row_count,
                logical_bytes=record.logical_bytes,
                stored_bytes=record.stored_bytes,
                schema_version=schema_version,
                validation_errors=tuple(sorted(set(errors))),
            )
        integrity = (
            IntegrityStatus.VALID
            if evidence_status == IntegrityStatus.VALID
            else IntegrityStatus.INTEGRITY_FAILED
        )
        return RegistryArtifact(
            artifact_key=record.artifact_key,
            artifact_kind=record.kind.value,
            owner_kind=owner_kind,
            owner_id=owner_id,
            content_hash=record.content_hash,
            availability=ArtifactAvailability.AVAILABLE,
            integrity_status=integrity,
            retention_state="retained",
            row_count=record.row_count,
            logical_bytes=record.logical_bytes,
            stored_bytes=record.stored_bytes,
            schema_version=schema_version,
            validation_errors=(
                () if integrity == IntegrityStatus.VALID else ("referenced_evidence_unavailable",)
            ),
        )

    @staticmethod
    def _aggregate_integrity(artifacts: Sequence[RegistryArtifact]) -> IntegrityStatus:
        priority = (
            IntegrityStatus.CORRUPT,
            IntegrityStatus.UNSUPPORTED_SCHEMA,
            IntegrityStatus.INTEGRITY_FAILED,
            IntegrityStatus.MISSING,
            IntegrityStatus.PRUNED,
        )
        values = {item.integrity_status for item in artifacts}
        return next((status for status in priority if status in values), IntegrityStatus.VALID)

    def _read_unique(
        self,
        paths: Sequence[Path],
        parser: Any,
        identity: Any,
        object_kind: str,
        issues: list[RegistryIssue],
    ) -> dict[str, Any]:
        grouped: dict[str, list[tuple[Path, Any, bytes]]] = {}
        for path in sorted(paths):
            relative = path.relative_to(self.store.root).as_posix()
            try:
                raw = path.read_bytes()
                parsed = parser(json.loads(raw.decode("utf-8")))
                object_id = identity(parsed)
            except Exception as error:  # registry records local corruption and continues
                issues.append(
                    RegistryIssue(
                        issue_code=f"{object_kind}_corrupt",
                        object_identity=path.parent.name or path.stem,
                        relative_location=relative,
                        detail=type(error).__name__,
                    )
                )
                continue
            grouped.setdefault(object_id, []).append((path, parsed, raw))
        result: dict[str, Any] = {}
        for object_id, candidates in sorted(grouped.items()):
            candidates.sort(key=lambda item: item[0].relative_to(self.store.root).as_posix())
            selected_path, selected, selected_raw = candidates[0]
            result[object_id] = selected
            for duplicate_path, _duplicate, duplicate_raw in candidates[1:]:
                equivalent = canonical_data(json.loads(selected_raw)) == canonical_data(
                    json.loads(duplicate_raw)
                )
                issues.append(
                    RegistryIssue(
                        issue_code=(
                            f"duplicate_{object_kind}_equivalent"
                            if equivalent
                            else f"duplicate_{object_kind}_conflict"
                        ),
                        object_identity=object_id,
                        relative_location=duplicate_path.relative_to(self.store.root).as_posix(),
                        detail=(
                            "ignored_equivalent_duplicate"
                            if equivalent
                            else "selected_lexicographically_first_record"
                        ),
                    )
                )
        return result

    def rebuild(self, *, persist: bool = True) -> SavedRunRegistry:
        fingerprint = self.source_fingerprint()
        issues: list[RegistryIssue] = []
        manifests = self._read_unique(
            list(self.store.strategy_runs_dir.rglob("manifest.json")),
            StrategyRunManifest.from_dict,
            lambda item: item.strategy_run_id,
            "strategy_run",
            issues,
        )
        derived = self._read_unique(
            list(self.store.derived_metrics_dir.rglob("manifest.json")),
            DerivedMetricManifest.from_dict,
            lambda item: item.derived_metric_id,
            "derived_metric",
            issues,
        )
        profiles = self._read_unique(
            list(self.store.profiles_dir.rglob("*.json")),
            EvaluationProfile.from_dict,
            lambda item: item.evaluation_profile_id,
            "evaluation_profile",
            issues,
        )
        evaluations = self._read_unique(
            list(self.store.evaluation_runs_dir.rglob("*.json")),
            EvaluationRun.from_dict,
            lambda item: item.evaluation_run_id,
            "evaluation_run",
            issues,
        )
        reports = self._read_unique(
            list(self.store.decision_reports_dir.rglob("*.json")),
            DecisionReport.from_dict,
            lambda item: item.decision_report_id,
            "decision_report",
            issues,
        )

        attempts: dict[str, ExecutionAttempt] = {}
        if self.attempts.root.exists():
            for directory in sorted(path for path in self.attempts.root.iterdir() if path.is_dir()):
                try:
                    attempt = self.attempts.get(directory.name)
                    attempts[attempt.execution_attempt_id] = attempt
                except Exception as error:
                    issues.append(
                        RegistryIssue(
                            issue_code="execution_attempt_corrupt",
                            object_identity=directory.name,
                            relative_location=directory.relative_to(self.store.root).as_posix(),
                            detail=type(error).__name__,
                        )
                    )

        derived_by_run: dict[str, list[DerivedMetricManifest]] = {}
        for item in derived.values():
            if item.strategy_run_id not in manifests:
                issues.append(
                    RegistryIssue(
                        issue_code="orphan_derived_metric",
                        object_identity=item.derived_metric_id,
                        relative_location=f"derived_metrics/{item.derived_metric_id}",
                        detail="referenced_strategy_run_missing",
                    )
                )
            derived_by_run.setdefault(item.strategy_run_id, []).append(item)

        evals_by_run: dict[str, list[EvaluationRun]] = {}
        for run in evaluations.values():
            if run.evaluation_profile_id not in profiles:
                issues.append(
                    RegistryIssue(
                        issue_code="evaluation_profile_reference_missing",
                        object_identity=run.evaluation_run_id,
                        relative_location=f"evaluation_runs/{run.evaluation_run_id}.json",
                        detail=run.evaluation_profile_id,
                    )
                )
            for strategy_run_id in run.strategy_run_ids:
                if strategy_run_id not in manifests:
                    issues.append(
                        RegistryIssue(
                            issue_code="evaluation_strategy_run_reference_missing",
                            object_identity=run.evaluation_run_id,
                            relative_location=f"evaluation_runs/{run.evaluation_run_id}.json",
                            detail=strategy_run_id,
                        )
                    )
                evals_by_run.setdefault(strategy_run_id, []).append(run)

        attempts_by_run: dict[str, list[ExecutionAttempt]] = {}
        for attempt in attempts.values():
            attempts_by_run.setdefault(attempt.intended_strategy_run_id, []).append(attempt)

        run_entries: list[RegistryStrategyRun] = []
        for strategy_run_id, manifest in sorted(manifests.items()):
            direct_artifacts = [
                self._artifact(
                    record,
                    owner_kind="strategy_run",
                    owner_id=strategy_run_id,
                )
                for record in manifest.artifacts
            ]
            derived_artifacts: list[RegistryArtifact] = []
            benchmark_provenance: list[Mapping[str, Any]] = []
            calculation_provenance: list[Mapping[str, Any]] = []
            for derived_manifest in sorted(
                derived_by_run.get(strategy_run_id, ()), key=lambda item: item.derived_metric_id
            ):
                evidence_status = IntegrityStatus.VALID
                evidence_hashes = list(derived_manifest.source_artifact_hashes.values())
                if derived_manifest.benchmark_artifact_hash is not None:
                    evidence_hashes.append(derived_manifest.benchmark_artifact_hash)
                for source_hash in evidence_hashes:
                    try:
                        event = self.store.artifact_retention_event(source_hash)
                    except (ValueError, json.JSONDecodeError):
                        event = None
                        evidence_status = IntegrityStatus.INTEGRITY_FAILED
                    source_path = self.store.object_path_for_hash(source_hash)
                    if event is not None or not source_path.exists():
                        evidence_status = IntegrityStatus.INTEGRITY_FAILED
                        break
                    try:
                        logical = gzip.decompress(source_path.read_bytes())
                        if content_hash(logical) != source_hash:
                            evidence_status = IntegrityStatus.INTEGRITY_FAILED
                            break
                    except (OSError, EOFError):
                        evidence_status = IntegrityStatus.INTEGRITY_FAILED
                        break
                derived_artifacts.extend(
                    self._artifact(
                        record,
                        owner_kind="derived_metric",
                        owner_id=derived_manifest.derived_metric_id,
                        evidence_status=evidence_status,
                    )
                    for record in derived_manifest.artifacts
                )
                benchmark_provenance.append(
                    {
                        "derived_metric_id": derived_manifest.derived_metric_id,
                        "benchmark_identity": derived_manifest.benchmark_identity,
                        "benchmark_artifact_hash": derived_manifest.benchmark_artifact_hash,
                    }
                )
                calculation_provenance.append(
                    {
                        "derived_metric_id": derived_manifest.derived_metric_id,
                        "metric_calculation_engine_version": (
                            derived_manifest.metric_calculation_engine_version
                        ),
                        "metric_definition_version": derived_manifest.metric_definition_version,
                        "source_artifact_hashes": derived_manifest.source_artifact_hashes,
                        "calculation_settings": derived_manifest.calculation_settings,
                        "integrity_status": evidence_status.value,
                    }
                )
            artifacts = direct_artifacts + derived_artifacts
            present_keys = {item.artifact_key for item in artifacts}
            for artifact_key in _DISCOVERABLE_ARTIFACT_KEYS:
                if artifact_key not in present_keys:
                    artifacts.append(
                        RegistryArtifact(
                            artifact_key=artifact_key,
                            artifact_kind=artifact_key,
                            owner_kind="strategy_run",
                            owner_id=strategy_run_id,
                            content_hash=None,
                            availability=ArtifactAvailability.NEVER_GENERATED,
                            integrity_status=IntegrityStatus.NOT_CHECKED,
                            retention_state="never_generated",
                            row_count=None,
                            logical_bytes=None,
                            stored_bytes=None,
                            schema_version=None,
                        )
                    )
            artifacts.sort(key=lambda item: (item.artifact_key, item.owner_kind, item.owner_id))
            integrity = self._aggregate_integrity(
                [item for item in artifacts if item.availability != ArtifactAvailability.NEVER_GENERATED]
            )
            evaluation_items = sorted(
                evals_by_run.get(strategy_run_id, ()), key=lambda item: item.evaluation_run_id
            )
            retention = (
                "pruned"
                if any(item.retention_state == "pruned" for item in artifacts)
                else "retained"
            )
            run_entries.append(
                RegistryStrategyRun(
                    strategy_run_id=strategy_run_id,
                    specification_hash=content_hash(manifest.canonical_specification),
                    manifest_hash=content_hash(manifest.to_dict()),
                    canonical_specification=manifest.canonical_specification,
                    source_data_snapshot_id=manifest.snapshot_hash,
                    engine_version=str(manifest.canonical_specification["engine_version"]),
                    source_commit=manifest.source_code_commit,
                    terminal_status=manifest.execution_status.value,
                    creation_time=manifest.creation_time,
                    economic_date_range=manifest.canonical_specification["economic_date_range"],
                    artifacts=tuple(artifacts),
                    derived_metric_ids=tuple(
                        sorted(item.derived_metric_id for item in derived_by_run.get(strategy_run_id, ()))
                    ),
                    evaluation_profile_ids=tuple(
                        sorted({item.evaluation_profile_id for item in evaluation_items})
                    ),
                    evaluation_run_ids=tuple(item.evaluation_run_id for item in evaluation_items),
                    execution_attempt_ids=tuple(
                        sorted(
                            item.execution_attempt_id
                            for item in attempts_by_run.get(strategy_run_id, ())
                        )
                    ),
                    benchmark_provenance=tuple(benchmark_provenance),
                    calculation_provenance=tuple(calculation_provenance),
                    integrity_status=integrity,
                    retention_status=retention,
                    validation_errors=tuple(
                        sorted(
                            {
                                f"{item.artifact_key}:{error}"
                                for item in artifacts
                                for error in item.validation_errors
                            }
                        )
                    ),
                    warnings=manifest.warnings,
                    limitations=manifest.limitations,
                )
            )

        profile_entries = tuple(
            RegistryEvaluationProfile(
                evaluation_profile_id=profile.evaluation_profile_id,
                profile_hash=profile.profile_hash,
                name=profile.name,
                comparison_mode=profile.comparison_mode.value,
                approval_status=profile.approval_status,
                integrity_status=IntegrityStatus.VALID,
            )
            for profile in profiles.values()
        )
        evaluation_entries = tuple(
            RegistryEvaluationRun(
                evaluation_run_id=run.evaluation_run_id,
                evaluation_profile_id=run.evaluation_profile_id,
                profile_hash=run.profile_hash,
                strategy_run_ids=run.strategy_run_ids,
                creation_time=run.creation_time,
                status="completed",
                comparison_mode=run.comparison_mode.value,
                benchmark_data_identity=run.benchmark_data_identity,
                metric_engine_version=run.metric_engine_version,
                derived_metric_ids=run.derived_metric_ids,
                integrity_status=(
                    IntegrityStatus.VALID
                    if run.evaluation_profile_id in profiles
                    and all(item in manifests for item in run.strategy_run_ids)
                    else IntegrityStatus.INTEGRITY_FAILED
                ),
            )
            for run in evaluations.values()
        )
        report_entries = []
        for report in reports.values():
            integrity = IntegrityStatus.VALID
            evaluation = evaluations.get(report.evaluation_run_id)
            profile = profiles.get(report.evaluation_profile_id)
            if (
                report.strategy_run_id not in manifests
                or evaluation is None
                or profile is None
                or report.strategy_run_id not in evaluation.strategy_run_ids
                or evaluation.evaluation_profile_id != report.evaluation_profile_id
                or evaluation.profile_hash != report.profile_hash
                or profile.profile_hash != report.profile_hash
            ):
                integrity = IntegrityStatus.INTEGRITY_FAILED
                issues.append(
                    RegistryIssue(
                        issue_code="decision_report_reference_invalid",
                        object_identity=report.decision_report_id,
                        relative_location=f"decision_reports/{report.decision_report_id}.json",
                        detail="strategy_run/evaluation/profile reference mismatch",
                    )
                )
            report_entries.append(
                RegistryDecisionReport(
                    decision_report_id=report.decision_report_id,
                    strategy_run_id=report.strategy_run_id,
                    evaluation_run_id=report.evaluation_run_id,
                    evaluation_profile_id=report.evaluation_profile_id,
                    profile_hash=report.profile_hash,
                    creation_time=report.creation_time,
                    integrity_status=integrity,
                )
            )
        orphan_hashes = self.store.orphan_hashes()
        issues.extend(
            RegistryIssue(
                issue_code="orphan_object",
                object_identity=artifact_hash,
                relative_location=f"objects/sha256/{artifact_hash}.json.gz",
                detail="not_referenced_by_any_manifest",
            )
            for artifact_hash in orphan_hashes
        )
        registry = SavedRunRegistry(
            source_fingerprint=fingerprint,
            strategy_runs=tuple(run_entries),
            evaluation_profiles=profile_entries,
            evaluation_runs=evaluation_entries,
            decision_reports=tuple(report_entries),
            execution_attempts=tuple(attempts.values()),
            orphan_object_hashes=orphan_hashes,
            issues=tuple(issues),
        )
        if persist:
            self.registry_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.registry_path.with_suffix(".tmp")
            temporary.write_bytes(canonical_bytes(registry.to_dict()))
            temporary.replace(self.registry_path)
        return registry

    def load_or_rebuild(self) -> SavedRunRegistry:
        fingerprint = self.source_fingerprint()
        if self.registry_path.exists():
            try:
                registry = SavedRunRegistry.from_dict(
                    json.loads(self.registry_path.read_text(encoding="utf-8"))
                )
                if registry.source_fingerprint == fingerprint:
                    return registry
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return self.rebuild(persist=True)
