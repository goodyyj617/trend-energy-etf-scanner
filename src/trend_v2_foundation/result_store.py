"""Bounded content-addressed result storage with a local disk backend."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .canonical import canonical_bytes, canonical_data, content_hash
from .artifact_schemas import artifact_payload_row_count
from .contracts import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactRetentionPolicy,
    DerivedMetricManifest,
    EvaluationProfile,
    EvaluationRun,
    StrategyRunManifest,
)


RESULT_STORE_VERSION = "bounded_local_result_store_v1"


@dataclass(frozen=True)
class ArtifactSizeEstimate:
    logical_bytes: int
    stored_bytes: int
    fits_artifact_limit: bool
    fits_store_limit: bool


@dataclass(frozen=True)
class ArtifactPutResult:
    record: ArtifactRecord
    retention_status: str


@dataclass(frozen=True)
class ManifestValidation:
    valid: bool
    errors: tuple[str, ...]
    checked_artifacts: int


@dataclass(frozen=True)
class StoreRetentionStatus:
    store_version: str
    current_bytes: int
    max_store_bytes: int
    remaining_bytes: int
    strategy_run_count: int
    max_strategy_runs: int
    evaluation_run_count: int
    max_evaluation_runs: int
    orphan_object_count: int


class ResultStore(Protocol):
    """Backend-neutral interface; Foundation 1 implements only local disk."""

    def put_artifact(
        self,
        artifact_key: str,
        kind: ArtifactKind,
        payload: Any,
        *,
        row_count: int,
    ) -> ArtifactPutResult: ...

    def save_strategy_run(self, manifest: StrategyRunManifest) -> None: ...

    def get_strategy_run_manifest(self, strategy_run_id: str) -> StrategyRunManifest: ...

    def load_artifact_payload(self, strategy_run_id: str, artifact_key: str) -> Any: ...

    def save_evaluation_profile(self, profile: EvaluationProfile) -> None: ...

    def save_evaluation_run(self, run: EvaluationRun) -> None: ...

    def save_derived_metric_manifest(self, manifest: DerivedMetricManifest) -> None: ...

    def get_derived_metric_manifest(self, derived_metric_id: str) -> DerivedMetricManifest: ...


class LocalResultStore:
    """Immutable manifests plus gzip JSON objects addressed by SHA-256."""

    def __init__(self, root: str | Path, policy: ArtifactRetentionPolicy) -> None:
        self.root = Path(root)
        self.policy = policy
        self.objects_dir = self.root / "objects" / "sha256"
        self.strategy_runs_dir = self.root / "strategy_runs"
        self.profiles_dir = self.root / "evaluation_profiles"
        self.evaluation_runs_dir = self.root / "evaluation_runs"
        self.derived_metrics_dir = self.root / "derived_metrics"
        for path in (
            self.objects_dir,
            self.strategy_runs_dir,
            self.profiles_dir,
            self.evaluation_runs_dir,
            self.derived_metrics_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        policy_payload = {
            "store_version": RESULT_STORE_VERSION,
            "policy_id": policy.artifact_retention_policy_id,
            "policy": canonical_data(policy),
        }
        self._write_immutable(self.root / "retention_policy.json", canonical_bytes(policy_payload))

    def _current_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def _assert_capacity(self, additional_bytes: int) -> None:
        if self._current_bytes() + additional_bytes > self.policy.max_store_bytes:
            raise OverflowError("result store byte limit would be exceeded")

    def _write_immutable(self, path: Path, payload: bytes) -> bool:
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(f"immutable record already exists with different content: {path.name}")
            return False
        self._assert_capacity(len(payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        return True

    @staticmethod
    def _compressed(payload: Any) -> tuple[bytes, bytes]:
        logical = canonical_bytes(payload)
        return logical, gzip.compress(logical, compresslevel=9, mtime=0)

    def estimate_artifact_size(self, payload: Any) -> ArtifactSizeEstimate:
        logical, stored = self._compressed(payload)
        return ArtifactSizeEstimate(
            logical_bytes=len(logical),
            stored_bytes=len(stored),
            fits_artifact_limit=len(stored) <= self.policy.max_artifact_bytes,
            fits_store_limit=self._current_bytes() + len(stored) <= self.policy.max_store_bytes,
        )

    def put_artifact(
        self,
        artifact_key: str,
        kind: ArtifactKind,
        payload: Any,
        *,
        row_count: int,
    ) -> ArtifactPutResult:
        if not artifact_key:
            raise ValueError("artifact_key is required")
        if kind not in self.policy.retained_artifact_kinds:
            raise ValueError(f"artifact kind is not retained by policy: {kind.value}")
        if row_count < 0:
            raise ValueError("row_count cannot be negative")
        logical, stored = self._compressed(payload)
        if len(stored) > self.policy.max_artifact_bytes:
            raise OverflowError("artifact exceeds per-artifact retention limit")
        digest = content_hash(logical)
        path = self.objects_dir / f"{digest}.json.gz"
        created = self._write_immutable(path, stored)
        record = ArtifactRecord(
            artifact_key=artifact_key,
            kind=kind,
            content_hash=digest,
            media_type="application/json",
            encoding="gzip",
            logical_bytes=len(logical),
            stored_bytes=len(stored),
            row_count=row_count,
        )
        return ArtifactPutResult(
            record=record,
            retention_status="retained" if created else "deduplicated",
        )

    def _object_path(self, record: ArtifactRecord) -> Path:
        return self.objects_dir / f"{record.content_hash}.json.gz"

    def _strategy_manifest_path(self, strategy_run_id: str) -> Path:
        return self.strategy_runs_dir / strategy_run_id / "manifest.json"

    def save_strategy_run(self, manifest: StrategyRunManifest) -> None:
        path = self._strategy_manifest_path(manifest.strategy_run_id)
        if not path.exists() and len(list(self.strategy_runs_dir.glob("*/manifest.json"))) >= self.policy.max_strategy_runs:
            raise OverflowError("strategy-run retention count would be exceeded")
        validation = self.validate_manifest(manifest)
        if not validation.valid:
            raise ValueError("invalid strategy manifest: " + "; ".join(validation.errors))
        self._write_immutable(path, canonical_bytes(manifest.to_dict()))

    def get_strategy_run_manifest(self, strategy_run_id: str) -> StrategyRunManifest:
        path = self._strategy_manifest_path(strategy_run_id)
        if not path.exists():
            raise KeyError(strategy_run_id)
        return StrategyRunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def validate_manifest(
        self, manifest_or_id: StrategyRunManifest | str
    ) -> ManifestValidation:
        try:
            manifest = (
                manifest_or_id
                if isinstance(manifest_or_id, StrategyRunManifest)
                else self.get_strategy_run_manifest(manifest_or_id)
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return ManifestValidation(False, (f"manifest_unreadable:{error}",), 0)
        errors: list[str] = []
        for artifact in manifest.artifacts:
            path = self._object_path(artifact)
            if not path.exists():
                errors.append(f"artifact_missing:{artifact.artifact_key}")
                continue
            stored = path.read_bytes()
            if len(stored) != artifact.stored_bytes:
                errors.append(f"stored_size_mismatch:{artifact.artifact_key}")
            try:
                logical = gzip.decompress(stored)
            except (gzip.BadGzipFile, EOFError, OSError):
                errors.append(f"gzip_invalid:{artifact.artifact_key}")
                continue
            if len(logical) != artifact.logical_bytes:
                errors.append(f"logical_size_mismatch:{artifact.artifact_key}")
            if content_hash(logical) != artifact.content_hash:
                errors.append(f"content_hash_mismatch:{artifact.artifact_key}")
            try:
                payload = json.loads(logical.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                errors.append(f"json_invalid:{artifact.artifact_key}")
                continue
            actual_row_count = artifact_payload_row_count(payload)
            if actual_row_count is not None and actual_row_count != artifact.row_count:
                errors.append(
                    f"row_count_mismatch:{artifact.artifact_key}:"
                    f"expected={artifact.row_count}:actual={actual_row_count}"
                )
        return ManifestValidation(not errors, tuple(errors), len(manifest.artifacts))

    def load_artifact_payload(self, strategy_run_id: str, artifact_key: str) -> Any:
        manifest = self.get_strategy_run_manifest(strategy_run_id)
        matching = [artifact for artifact in manifest.artifacts if artifact.artifact_key == artifact_key]
        if not matching:
            raise KeyError(f"{strategy_run_id}:{artifact_key}")
        record = matching[0]
        path = self._object_path(record)
        if not path.exists():
            raise FileNotFoundError(path)
        logical = gzip.decompress(path.read_bytes())
        if content_hash(logical) != record.content_hash:
            raise ValueError(f"artifact hash validation failed: {artifact_key}")
        payload = json.loads(logical.decode("utf-8"))
        actual_row_count = artifact_payload_row_count(payload)
        if actual_row_count is not None and actual_row_count != record.row_count:
            raise ValueError(
                f"manifest row-count mismatch:{artifact_key}:"
                f"expected={record.row_count}:actual={actual_row_count}"
            )
        return payload

    def get_strategy_artifact_record(
        self, strategy_run_id: str, artifact_key: str
    ) -> ArtifactRecord:
        manifest = self.get_strategy_run_manifest(strategy_run_id)
        matching = [artifact for artifact in manifest.artifacts if artifact.artifact_key == artifact_key]
        if not matching:
            raise KeyError(f"{strategy_run_id}:{artifact_key}")
        return matching[0]

    def load_and_validate_artifact(
        self, strategy_run_id: str, artifact_key: str, validator: Any
    ) -> Any:
        payload = self.load_artifact_payload(strategy_run_id, artifact_key)
        validator(payload)
        return payload

    def _load_record_payload(self, record: ArtifactRecord) -> Any:
        path = self._object_path(record)
        if not path.exists():
            raise FileNotFoundError(path)
        logical = gzip.decompress(path.read_bytes())
        if content_hash(logical) != record.content_hash:
            raise ValueError(f"artifact hash validation failed: {record.artifact_key}")
        payload = json.loads(logical.decode("utf-8"))
        actual_row_count = artifact_payload_row_count(payload)
        if actual_row_count is not None and actual_row_count != record.row_count:
            raise ValueError(
                f"manifest row-count mismatch:{record.artifact_key}:"
                f"expected={record.row_count}:actual={actual_row_count}"
            )
        return payload

    def save_derived_metric_manifest(self, manifest: DerivedMetricManifest) -> None:
        self.get_strategy_run_manifest(manifest.strategy_run_id)
        for artifact in manifest.artifacts:
            path = self._object_path(artifact)
            if not path.exists():
                raise ValueError(f"derived metric artifact missing: {artifact.artifact_key}")
            self._load_record_payload(artifact)
        path = self.derived_metrics_dir / manifest.derived_metric_id / "manifest.json"
        self._write_immutable(path, canonical_bytes(manifest.to_dict()))

    def get_derived_metric_manifest(self, derived_metric_id: str) -> DerivedMetricManifest:
        path = self.derived_metrics_dir / derived_metric_id / "manifest.json"
        if not path.exists():
            raise KeyError(derived_metric_id)
        return DerivedMetricManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_derived_metric_artifact(self, derived_metric_id: str, artifact_key: str) -> Any:
        manifest = self.get_derived_metric_manifest(derived_metric_id)
        matching = [artifact for artifact in manifest.artifacts if artifact.artifact_key == artifact_key]
        if not matching:
            raise KeyError(f"{derived_metric_id}:{artifact_key}")
        return self._load_record_payload(matching[0])

    def derived_metric_history(self, strategy_run_id: str | None = None) -> tuple[str, ...]:
        result: list[str] = []
        for path in sorted(self.derived_metrics_dir.glob("*/manifest.json")):
            manifest = DerivedMetricManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if strategy_run_id is None or manifest.strategy_run_id == strategy_run_id:
                result.append(manifest.derived_metric_id)
        return tuple(result)

    def provenance_for_strategy_run(self, strategy_run_id: str) -> Mapping[str, Any]:
        manifest = self.get_strategy_run_manifest(strategy_run_id)
        derived = [
            self.get_derived_metric_manifest(derived_id).to_dict()
            for derived_id in self.derived_metric_history(strategy_run_id)
        ]
        evaluations = []
        for path in sorted(self.evaluation_runs_dir.glob("*.json")):
            run = EvaluationRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if strategy_run_id in run.strategy_run_ids:
                evaluations.append(run.evaluation_run_id)
        return {
            "strategy_run_manifest": manifest.to_dict(),
            "derived_metric_manifests": derived,
            "evaluation_run_ids": evaluations,
        }

    def save_evaluation_profile(self, profile: EvaluationProfile) -> None:
        path = self.profiles_dir / f"{profile.evaluation_profile_id}.json"
        self._write_immutable(path, canonical_bytes(profile.to_dict()))

    def get_evaluation_profile(self, evaluation_profile_id: str) -> EvaluationProfile:
        path = self.profiles_dir / f"{evaluation_profile_id}.json"
        if not path.exists():
            raise KeyError(evaluation_profile_id)
        return EvaluationProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_evaluation_run(self, run: EvaluationRun) -> None:
        path = self.evaluation_runs_dir / f"{run.evaluation_run_id}.json"
        if not path.exists() and len(list(self.evaluation_runs_dir.glob("*.json"))) >= self.policy.max_evaluation_runs:
            raise OverflowError("evaluation-run retention count would be exceeded")
        self.get_evaluation_profile(run.evaluation_profile_id)
        for strategy_run_id in run.strategy_run_ids:
            self.get_strategy_run_manifest(strategy_run_id)
        self._write_immutable(path, canonical_bytes(run.to_dict()))

    def get_evaluation_run(self, evaluation_run_id: str) -> EvaluationRun:
        path = self.evaluation_runs_dir / f"{evaluation_run_id}.json"
        if not path.exists():
            raise KeyError(evaluation_run_id)
        return EvaluationRun.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def evaluation_history(self) -> tuple[str, ...]:
        return tuple(path.stem for path in sorted(self.evaluation_runs_dir.glob("*.json")))

    def orphan_hashes(self) -> tuple[str, ...]:
        referenced: set[str] = set()
        for path in self.strategy_runs_dir.glob("*/manifest.json"):
            try:
                manifest = StrategyRunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            referenced.update(artifact.content_hash for artifact in manifest.artifacts)
        for path in self.derived_metrics_dir.glob("*/manifest.json"):
            try:
                manifest = DerivedMetricManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            referenced.update(artifact.content_hash for artifact in manifest.artifacts)
        objects = {path.name.removesuffix(".json.gz") for path in self.objects_dir.glob("*.json.gz")}
        return tuple(sorted(objects - referenced))

    def retention_status(self) -> StoreRetentionStatus:
        current = self._current_bytes()
        return StoreRetentionStatus(
            store_version=RESULT_STORE_VERSION,
            current_bytes=current,
            max_store_bytes=self.policy.max_store_bytes,
            remaining_bytes=max(0, self.policy.max_store_bytes - current),
            strategy_run_count=len(list(self.strategy_runs_dir.glob("*/manifest.json"))),
            max_strategy_runs=self.policy.max_strategy_runs,
            evaluation_run_count=len(list(self.evaluation_runs_dir.glob("*.json"))),
            max_evaluation_runs=self.policy.max_evaluation_runs,
            orphan_object_count=len(self.orphan_hashes()),
        )
