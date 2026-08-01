"""Small typed read-only local API over the Foundation 3 saved-run registry."""

from __future__ import annotations

import base64
from collections import Counter
import ipaddress
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

from .artifact_schemas import (
    BEHAVIOR_METADATA_SCHEMA_VERSION,
    DAILY_PORTFOLIO_CURVE_SCHEMA_VERSION,
    ROBUSTNESS_SUMMARY_SCHEMA_VERSION,
    ROLLING_METRICS_SCHEMA_VERSION,
    YEARLY_METRICS_SCHEMA_VERSION,
)
from .behavior import BEHAVIOR_ENGINE_VERSION
from .calculation import (
    DERIVED_METRICS_SCHEMA_VERSION,
    METRIC_CALCULATION_ENGINE_VERSION,
    METRIC_DEFINITION_VERSION,
)
from .canonical import canonical_bytes, canonical_data, content_hash
from .contracts import (
    DERIVED_METRIC_MANIFEST_VERSION,
    EVALUATION_PROFILE_VERSION,
    EVALUATION_RUN_VERSION,
    METRIC_REGISTRY_VERSION,
    STRATEGY_RUN_MANIFEST_VERSION,
    STRATEGY_RUN_SPEC_VERSION,
)
from .execution import (
    EXECUTION_ATTEMPT_REPOSITORY_VERSION,
    EXECUTION_ATTEMPT_SCHEMA_VERSION,
    AttemptOperationalStatus,
    FileExecutionAttemptRepository,
)
from .registry import (
    REGISTRY_REBUILD_VERSION,
    SAVED_RUN_REGISTRY_SCHEMA_VERSION,
    ArtifactAvailability,
    IntegrityStatus,
    RegistryArtifact,
    SavedRunRegistry,
    SavedRunRegistryBuilder,
)
from .result_store import LocalResultStore, RESULT_STORE_VERSION


API_VERSION = "trend_v2_local_read_api_v1"
API_PATH_PREFIX = "/api/v1"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
DEFAULT_TIME_SERIES_PAGE_SIZE = 250
MAX_TIME_SERIES_PAGE_SIZE = 1_000


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{2,160}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SECRET_FIELD = re.compile(r"(?:^|_)(?:secret|password|token|api_key|authorization)(?:$|_)", re.I)
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"']+")
_POSIX_LOCAL_PATH = re.compile(r"/(?:Users|home|tmp|var/tmp)/[^\s\"']+")


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]" if _SECRET_FIELD.search(str(key)) else _redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _POSIX_LOCAL_PATH.sub(
            "[LOCAL_PATH]", _WINDOWS_PATH.sub("[LOCAL_PATH]", value)
        )
    return value


@dataclass(frozen=True)
class ApiServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: tuple[str, ...] = ()
    allow_non_loopback: bool = False
    max_response_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as error:
            raise ValueError("API host must be an explicit IP address") from error
        if not self.allow_non_loopback and not address.is_loopback:
            raise ValueError("local API binds to a loopback address unless explicitly overridden")
        if isinstance(self.port, bool) or not 0 <= self.port <= 65535:
            raise ValueError("API port must be between 0 and 65535")
        if self.max_response_bytes < 1_024:
            raise ValueError("max_response_bytes must be at least 1024")
        for origin in self.cors_origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                raise ValueError("CORS origins must be explicit localhost origins")
        object.__setattr__(self, "cors_origins", tuple(self.cors_origins))


@dataclass(frozen=True)
class ApiResponse:
    status_code: int
    body: Mapping[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)


class ApiContractError(Exception):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        diagnostic_en: str,
        *,
        object_identity: str | None = None,
        recoverable: bool = True,
        next_action_ko: str | None = None,
    ) -> None:
        super().__init__(error_code)
        self.status_code = status_code
        self.error_code = error_code
        self.diagnostic_en = diagnostic_en
        self.object_identity = object_identity
        self.recoverable = recoverable
        self.next_action_ko = next_action_ko


def _encoded_cursor(payload: Mapping[str, Any]) -> str:
    return base64.urlsafe_b64encode(canonical_bytes(payload)).decode("ascii").rstrip("=")


def _decoded_cursor(value: str) -> Mapping[str, Any]:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApiContractError(400, "invalid_query", "Cursor is malformed.") from error
    if not isinstance(payload, Mapping):
        raise ApiContractError(400, "invalid_query", "Cursor payload is invalid.")
    return payload


class ReadOnlyTrendApi:
    def __init__(
        self,
        store: LocalResultStore,
        *,
        registry_builder: SavedRunRegistryBuilder | None = None,
        attempt_repository: FileExecutionAttemptRepository | None = None,
        terminology_source: Mapping[str, Any] | None = None,
        server_config: ApiServerConfig | None = None,
    ) -> None:
        self.store = store
        self.attempt_repository = attempt_repository or FileExecutionAttemptRepository(
            store.root / "execution_attempts"
        )
        self.registry_builder = registry_builder or SavedRunRegistryBuilder(
            store, self.attempt_repository
        )
        self.terminology_source = terminology_source or {}
        api_terms = self.terminology_source.get("api", {})
        self.status_labels = dict(api_terms.get("status_labels", {}))
        self.error_messages = dict(api_terms.get("error_messages", {}))
        self.server_config = server_config or ApiServerConfig()

    def _request_id(self, headers: Mapping[str, str]) -> str:
        supplied = headers.get("X-Request-ID") or headers.get("x-request-id")
        return supplied if supplied and _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex

    def _error_response(self, error: ApiContractError, request_id: str) -> ApiResponse:
        message = self.error_messages.get(error.error_code, "요청을 처리할 수 없습니다.")
        return ApiResponse(
            status_code=error.status_code,
            body={
                "error": {
                    "code": error.error_code,
                    "message_ko": message,
                    "diagnostic_en": error.diagnostic_en,
                    "request_id": request_id,
                    "object_identity": error.object_identity,
                    "recoverable": error.recoverable,
                    "next_action_ko": error.next_action_ko,
                }
            },
            headers={"X-Request-ID": request_id},
        )

    @staticmethod
    def _validate_path(raw_path: str) -> str:
        decoded = unquote(unquote(raw_path))
        segments = decoded.split("/")
        if (
            "\\" in decoded
            or "\x00" in decoded
            or any(segment in {".", ".."} for segment in segments)
            or any(".." in segment for segment in segments)
        ):
            raise ApiContractError(400, "invalid_identifier", "Path traversal is not allowed.")
        return decoded.rstrip("/") or "/"

    @staticmethod
    def _query(target: str) -> tuple[str, dict[str, list[str]]]:
        parsed = urlsplit(target)
        path = ReadOnlyTrendApi._validate_path(parsed.path)
        return path, parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)

    @staticmethod
    def _allow_query(query: Mapping[str, Sequence[str]], allowed: set[str]) -> None:
        unknown = sorted(set(query) - allowed)
        if unknown:
            raise ApiContractError(
                400,
                "invalid_query",
                f"Unknown query fields: {', '.join(unknown)}.",
            )
        repeated = sorted(key for key, values in query.items() if len(values) != 1)
        if repeated:
            raise ApiContractError(
                400,
                "invalid_query",
                f"Query fields may appear once: {', '.join(repeated)}.",
            )

    @staticmethod
    def _value(query: Mapping[str, Sequence[str]], key: str) -> str | None:
        values = query.get(key)
        return values[0] if values else None

    @staticmethod
    def _identifier(value: str, kind: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ApiContractError(
                400,
                "invalid_identifier",
                f"Invalid {kind} identifier.",
                object_identity=value[:160],
            )
        return value

    @staticmethod
    def _date(value: str | None, field: str) -> str | None:
        if value is None:
            return None
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ApiContractError(400, "invalid_query", f"{field} must be YYYY-MM-DD.") from error
        if parsed.isoformat() != value:
            raise ApiContractError(400, "invalid_query", f"{field} must be canonical YYYY-MM-DD.")
        return value

    def _page(
        self,
        items: Sequence[Any],
        *,
        query: Mapping[str, Sequence[str]],
        registry_id: str,
        resource: str,
        signature_fields: Mapping[str, Any],
        default_size: int = DEFAULT_PAGE_SIZE,
        max_size: int = MAX_PAGE_SIZE,
    ) -> tuple[list[Any], Mapping[str, Any]]:
        raw_size = self._value(query, "page_size")
        try:
            page_size = default_size if raw_size is None else int(raw_size)
        except ValueError as error:
            raise ApiContractError(400, "invalid_query", "page_size must be an integer.") from error
        if not 1 <= page_size <= max_size:
            raise ApiContractError(
                400,
                "invalid_query",
                f"page_size must be between 1 and {max_size}.",
            )
        signature = content_hash(signature_fields)
        offset = 0
        raw_cursor = self._value(query, "cursor")
        if raw_cursor:
            cursor = _decoded_cursor(raw_cursor)
            if (
                cursor.get("registry_id") != registry_id
                or cursor.get("resource") != resource
                or cursor.get("signature") != signature
                or not isinstance(cursor.get("offset"), int)
                or cursor["offset"] < 0
            ):
                raise ApiContractError(400, "invalid_query", "Cursor is stale or does not match this query.")
            offset = cursor["offset"]
        if offset > len(items):
            raise ApiContractError(400, "invalid_query", "Cursor offset is outside the result set.")
        selected = list(items[offset : offset + page_size])
        next_offset = offset + len(selected)
        next_cursor = None
        if next_offset < len(items):
            next_cursor = _encoded_cursor(
                {
                    "registry_id": registry_id,
                    "resource": resource,
                    "signature": signature,
                    "offset": next_offset,
                }
            )
        return selected, {
            "page_size": page_size,
            "returned": len(selected),
            "total": len(items),
            "next_cursor": next_cursor,
        }

    @staticmethod
    def _sort(items: Sequence[Any], sort: str, allowed: set[str], identity: str) -> list[Any]:
        descending = sort.startswith("-")
        field_name = sort[1:] if descending else sort
        if field_name not in allowed:
            raise ApiContractError(400, "invalid_query", f"Unsupported sort key: {sort}.")
        result = sorted(items, key=lambda item: str(getattr(item, identity)))
        return sorted(
            result,
            key=lambda item: getattr(item, field_name),
            reverse=descending,
        )

    @staticmethod
    def _one(items: Sequence[Any], identity_field: str, value: str, kind: str) -> Any:
        match = next((item for item in items if getattr(item, identity_field) == value), None)
        if match is None:
            raise ApiContractError(404, "not_found", f"{kind} was not found.", object_identity=value)
        return match

    def _health(self, registry: SavedRunRegistry) -> Mapping[str, Any]:
        return {
            "status": "ok",
            "status_ko": "정상",
            "api_version": API_VERSION,
            "registry_id": registry.registry_id,
            "registry_schema_version": registry.schema_version,
            "read_only": True,
        }

    def _metadata(self, registry: SavedRunRegistry) -> Mapping[str, Any]:
        return {
            "api_version": API_VERSION,
            "api_path_prefix": API_PATH_PREFIX,
            "registry_id": registry.registry_id,
            "registry_schema_version": SAVED_RUN_REGISTRY_SCHEMA_VERSION,
            "registry_rebuild_version": REGISTRY_REBUILD_VERSION,
            "result_store_version": RESULT_STORE_VERSION,
            "supported_artifact_schema_versions": [
                STRATEGY_RUN_SPEC_VERSION,
                STRATEGY_RUN_MANIFEST_VERSION,
                EVALUATION_PROFILE_VERSION,
                EVALUATION_RUN_VERSION,
                DERIVED_METRIC_MANIFEST_VERSION,
                DAILY_PORTFOLIO_CURVE_SCHEMA_VERSION,
                YEARLY_METRICS_SCHEMA_VERSION,
                ROLLING_METRICS_SCHEMA_VERSION,
                ROBUSTNESS_SUMMARY_SCHEMA_VERSION,
                BEHAVIOR_METADATA_SCHEMA_VERSION,
                DERIVED_METRICS_SCHEMA_VERSION,
                EXECUTION_ATTEMPT_SCHEMA_VERSION,
            ],
            "metric_registry_version": METRIC_REGISTRY_VERSION,
            "calculation_engine_versions": [
                METRIC_CALCULATION_ENGINE_VERSION,
                METRIC_DEFINITION_VERSION,
            ],
            "behavior_engine_versions": [BEHAVIOR_ENGINE_VERSION],
            "execution_repository_version": EXECUTION_ATTEMPT_REPOSITORY_VERSION,
            "default_page_size": DEFAULT_PAGE_SIZE,
            "maximum_page_size": MAX_PAGE_SIZE,
            "maximum_time_series_page_size": MAX_TIME_SERIES_PAGE_SIZE,
            "cors_origins": list(self.server_config.cors_origins),
            "error_codes": sorted(self.error_messages),
        }

    def _overview(self, registry: SavedRunRegistry) -> Mapping[str, Any]:
        """Return bounded aggregate counts without exposing registry internals."""

        artifact_counts = Counter(
            artifact.availability.value
            for run in registry.strategy_runs
            for artifact in run.artifacts
        )
        attempt_counts = Counter(
            attempt.operational_status.value for attempt in registry.execution_attempts
        )
        return {
            "strategy_run_count": len(registry.strategy_runs),
            "evaluation_profile_count": len(registry.evaluation_profiles),
            "evaluation_run_count": len(registry.evaluation_runs),
            "artifact_availability_counts": {
                item.value: artifact_counts[item.value] for item in ArtifactAvailability
            },
            "execution_attempt_status_counts": {
                item.value: attempt_counts[item.value] for item in AttemptOperationalStatus
            },
            "versions": self._metadata(registry),
            "last_registry_rebuild_identity": {
                "registry_id": registry.registry_id,
                "source_fingerprint": registry.source_fingerprint,
                "rebuild_version": registry.rebuild_version,
            },
            "registry_issue_count": len(registry.issues),
            "orphan_object_count": len(registry.orphan_object_hashes),
            "evidence_quality_note_ko": (
                "이 요약은 저장된 근거의 상태만 보여 주며 운영 또는 프로덕션 승인 상태를 의미하지 않습니다."
            ),
        }

    def _terminology(self, registry: SavedRunRegistry) -> Mapping[str, Any]:
        """Expose the centralized Korean terminology source as read-only data."""

        source = self.terminology_source
        return {
            "schema_version": source.get("schema_version"),
            "language": source.get("language", "ko-KR"),
            "registry_id": registry.registry_id,
            "status_labels": canonical_data(source.get("api", {}).get("status_labels", {})),
            "entries": canonical_data(source.get("entries", {})),
        }

    def _list_runs(
        self, registry: SavedRunRegistry, query: Mapping[str, Sequence[str]]
    ) -> Mapping[str, Any]:
        allowed = {
            "status",
            "profile_id",
            "data_snapshot_id",
            "engine_version",
            "source_commit",
            "start_date",
            "end_date",
            "artifact_key",
            "artifact_availability",
            "integrity_status",
            "retention_status",
            "sort",
            "page_size",
            "cursor",
        }
        self._allow_query(query, allowed)
        filters = {key: self._value(query, key) for key in allowed - {"sort", "page_size", "cursor"}}
        start = self._date(filters["start_date"], "start_date")
        end = self._date(filters["end_date"], "end_date")
        if start and end and start > end:
            raise ApiContractError(400, "invalid_query", "start_date cannot follow end_date.")
        if filters["status"] and filters["status"] not in {"succeeded", "failed", "partial"}:
            raise ApiContractError(400, "invalid_query", "Unknown StrategyRun terminal status.")
        if filters["integrity_status"] and filters["integrity_status"] not in {
            item.value for item in IntegrityStatus
        }:
            raise ApiContractError(400, "invalid_query", "Unknown integrity status.")
        if filters["artifact_availability"] and filters["artifact_availability"] not in {
            item.value for item in ArtifactAvailability
        }:
            raise ApiContractError(400, "invalid_query", "Unknown artifact availability.")
        items = list(registry.strategy_runs)
        if filters["status"]:
            items = [item for item in items if item.terminal_status == filters["status"]]
        if filters["profile_id"]:
            items = [item for item in items if filters["profile_id"] in item.evaluation_profile_ids]
        if filters["data_snapshot_id"]:
            items = [item for item in items if item.source_data_snapshot_id == filters["data_snapshot_id"]]
        if filters["engine_version"]:
            items = [item for item in items if item.engine_version == filters["engine_version"]]
        if filters["source_commit"]:
            items = [item for item in items if item.source_commit == filters["source_commit"]]
        if start:
            items = [item for item in items if item.economic_date_range["end"] >= start]
        if end:
            items = [item for item in items if item.economic_date_range["start"] <= end]
        if filters["integrity_status"]:
            items = [item for item in items if item.integrity_status.value == filters["integrity_status"]]
        if filters["retention_status"]:
            if filters["retention_status"] not in {"retained", "pruned"}:
                raise ApiContractError(400, "invalid_query", "Unknown retention status.")
            items = [item for item in items if item.retention_status == filters["retention_status"]]
        if filters["artifact_key"] or filters["artifact_availability"]:
            items = [
                item
                for item in items
                if any(
                    (not filters["artifact_key"] or artifact.artifact_key == filters["artifact_key"])
                    and (
                        not filters["artifact_availability"]
                        or artifact.availability.value == filters["artifact_availability"]
                    )
                    for artifact in item.artifacts
                )
            ]
        sort = self._value(query, "sort") or "-creation_time"
        items = self._sort(
            items,
            sort,
            {"creation_time", "strategy_run_id", "terminal_status", "engine_version"},
            "strategy_run_id",
        )
        selected, page = self._page(
            items,
            query=query,
            registry_id=registry.registry_id,
            resource="strategy_runs",
            signature_fields={"filters": filters, "sort": sort},
        )
        return {
            "items": [
                {
                    "strategy_run_id": item.strategy_run_id,
                    "creation_time": item.creation_time,
                    "terminal_status": item.terminal_status,
                    "terminal_status_ko": self.status_labels.get(item.terminal_status),
                    "source_data_snapshot_id": item.source_data_snapshot_id,
                    "engine_version": item.engine_version,
                    "source_commit": item.source_commit,
                    "economic_date_range": canonical_data(item.economic_date_range),
                    "integrity_status": item.integrity_status.value,
                    "integrity_status_ko": self.status_labels.get(item.integrity_status.value),
                    "retention_status": item.retention_status,
                    "benchmark_identity": canonical_data(
                        item.canonical_specification.get("benchmark", {})
                    ),
                    "artifact_availability_counts": dict(
                        sorted(
                            Counter(
                                artifact.availability.value for artifact in item.artifacts
                            ).items()
                        )
                    ),
                    "available_artifact_keys": sorted(
                        artifact.artifact_key
                        for artifact in item.artifacts
                        if artifact.availability == ArtifactAvailability.AVAILABLE
                    ),
                    "evaluation_run_count": len(item.evaluation_run_ids),
                    "execution_attempt_count": len(item.execution_attempt_ids),
                }
                for item in selected
            ],
            "page": page,
            "sort": sort,
        }

    def _artifact_entry(
        self,
        run: Any,
        artifact_key: str,
        derived_metric_id: str | None,
    ) -> RegistryArtifact:
        candidates = [item for item in run.artifacts if item.artifact_key == artifact_key]
        if derived_metric_id is not None:
            candidates = [item for item in candidates if item.owner_id == derived_metric_id]
        if not candidates:
            raise ApiContractError(404, "artifact_missing", "Artifact is not registered.", object_identity=artifact_key)
        prefer_derived = artifact_key in {
            "yearly_metrics",
            "rolling_metrics",
            "derived_metrics",
            "behavior_metadata",
        }
        candidates.sort(
            key=lambda item: (
                item.availability == ArtifactAvailability.AVAILABLE,
                item.owner_kind == ("derived_metric" if prefer_derived else "strategy_run"),
                item.owner_id,
            ),
            reverse=True,
        )
        return candidates[0]

    @staticmethod
    def _assert_artifact_readable(entry: RegistryArtifact) -> None:
        mapping = {
            ArtifactAvailability.MISSING: (404, "artifact_missing", "Referenced artifact is missing."),
            ArtifactAvailability.CORRUPT: (409, "artifact_corrupt", "Artifact integrity check failed."),
            ArtifactAvailability.PRUNED: (410, "retention_pruned_artifact", "Artifact was retention-pruned."),
            ArtifactAvailability.UNSUPPORTED_SCHEMA: (422, "schema_unsupported", "Artifact schema is unsupported."),
            ArtifactAvailability.NEVER_GENERATED: (404, "artifact_missing", "Artifact was never generated."),
        }
        if entry.availability in mapping:
            status, code, diagnostic = mapping[entry.availability]
            if entry.artifact_key == "robustness_summary" and entry.availability in {
                ArtifactAvailability.MISSING,
                ArtifactAvailability.NEVER_GENERATED,
            }:
                code = "robustness_evidence_missing"
            raise ApiContractError(status, code, diagnostic, object_identity=entry.artifact_key)
        if entry.integrity_status == IntegrityStatus.INTEGRITY_FAILED:
            raise ApiContractError(
                409,
                "integrity_validation_failed",
                "Referenced source evidence is unavailable or invalid.",
                object_identity=entry.artifact_key,
            )

    def _load_artifact(self, run: Any, entry: RegistryArtifact) -> Any:
        self._assert_artifact_readable(entry)
        try:
            if entry.owner_kind == "derived_metric":
                return self.store.load_derived_metric_artifact(entry.owner_id, entry.artifact_key)
            return self.store.load_artifact_payload(run.strategy_run_id, entry.artifact_key)
        except FileNotFoundError as error:
            raise ApiContractError(
                404, "artifact_missing", "Referenced artifact is missing.", object_identity=entry.artifact_key
            ) from error
        except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ApiContractError(
                409, "artifact_corrupt", "Artifact failed validation.", object_identity=entry.artifact_key
            ) from error

    def _artifact_response(
        self,
        registry: SavedRunRegistry,
        run: Any,
        artifact_key: str,
        query: Mapping[str, Sequence[str]],
    ) -> Mapping[str, Any]:
        allowed = {"derived_metric_id", "start_date", "end_date", "window_sessions", "page_size", "cursor"}
        self._allow_query(query, allowed)
        derived_id = self._value(query, "derived_metric_id")
        if derived_id:
            self._identifier(derived_id, "derived metric")
        entry = self._artifact_entry(run, artifact_key, derived_id)
        payload = self._load_artifact(run, entry)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), list):
            if any(self._value(query, key) is not None for key in ("start_date", "end_date", "window_sessions", "page_size", "cursor")):
                raise ApiContractError(400, "invalid_query", "Pagination/range fields apply only to row artifacts.")
            if len(canonical_bytes(payload)) > self.server_config.max_response_bytes:
                raise ApiContractError(400, "invalid_query", "Artifact response exceeds the configured bound.")
            return {
                "artifact": entry.to_dict(),
                "payload": canonical_data(payload),
            }
        start = self._date(self._value(query, "start_date"), "start_date")
        end = self._date(self._value(query, "end_date"), "end_date")
        if start and end and start > end:
            raise ApiContractError(400, "invalid_query", "start_date cannot follow end_date.")
        window_raw = self._value(query, "window_sessions")
        try:
            window = int(window_raw) if window_raw is not None else None
        except ValueError as error:
            raise ApiContractError(400, "invalid_query", "window_sessions must be an integer.") from error
        if window is not None and artifact_key != "rolling_metrics":
            raise ApiContractError(400, "invalid_query", "window_sessions applies only to rolling metrics.")
        rows = []
        for row in payload["rows"]:
            economic_date = row.get("economic_date") or row.get("end_economic_date")
            if start and economic_date and economic_date < start:
                continue
            if end and economic_date and economic_date > end:
                continue
            if window is not None and row.get("window_sessions") != window:
                continue
            rows.append(row)
        selected, page = self._page(
            rows,
            query=query,
            registry_id=registry.registry_id,
            resource=f"artifact:{run.strategy_run_id}:{entry.owner_id}:{artifact_key}",
            signature_fields={
                "artifact_hash": entry.content_hash,
                "start_date": start,
                "end_date": end,
                "window_sessions": window,
            },
            default_size=DEFAULT_TIME_SERIES_PAGE_SIZE,
            max_size=MAX_TIME_SERIES_PAGE_SIZE,
        )
        metadata = {key: value for key, value in payload.items() if key != "rows"}
        return {
            "artifact": entry.to_dict(),
            "metadata": canonical_data(metadata),
            "items": canonical_data(selected),
            "page": page,
        }

    def _behavior_summary(self, run: Any, query: Mapping[str, Sequence[str]]) -> Mapping[str, Any]:
        """Expose fingerprints and bounded counts, never the full comparison paths."""

        self._allow_query(query, set())
        entry = self._artifact_entry(run, "behavior_metadata", None)
        payload = self._load_artifact(run, entry)
        if not isinstance(payload, Mapping):
            raise ApiContractError(
                409,
                "artifact_corrupt",
                "Behavior metadata is not a mapping.",
                object_identity="behavior_metadata",
            )
        inputs = payload.get("comparison_inputs", {})
        if not isinstance(inputs, Mapping):
            inputs = {}
        count_fields = {
            "economic_date_count": len(inputs.get("economic_dates", ())),
            "daily_return_count": len(inputs.get("daily_returns", ())),
            "active_date_count": len(inputs.get("active_dates", ())),
            "entry_date_count": (
                None if inputs.get("entry_dates") is None else len(inputs.get("entry_dates", ()))
            ),
            "exit_date_count": (
                None if inputs.get("exit_dates") is None else len(inputs.get("exit_dates", ()))
            ),
        }
        summary = {
            key: value for key, value in payload.items() if key != "comparison_inputs"
        }
        summary["comparison_input_counts"] = count_fields
        return {"artifact": entry.to_dict(), "payload": canonical_data(summary)}

    def _list_profiles(
        self, registry: SavedRunRegistry, query: Mapping[str, Sequence[str]]
    ) -> Mapping[str, Any]:
        self._allow_query(query, {"sort", "page_size", "cursor"})
        sort = self._value(query, "sort") or "evaluation_profile_id"
        items = self._sort(
            registry.evaluation_profiles,
            sort,
            {"evaluation_profile_id", "name", "comparison_mode", "approval_status"},
            "evaluation_profile_id",
        )
        selected, page = self._page(
            items,
            query=query,
            registry_id=registry.registry_id,
            resource="evaluation_profiles",
            signature_fields={"sort": sort},
        )
        return {"items": [item.to_dict() for item in selected], "page": page, "sort": sort}

    def _list_evaluations(
        self, registry: SavedRunRegistry, query: Mapping[str, Sequence[str]]
    ) -> Mapping[str, Any]:
        allowed = {
            "profile_id",
            "strategy_run_id",
            "status",
            "integrity_status",
            "start_date",
            "end_date",
            "sort",
            "page_size",
            "cursor",
        }
        self._allow_query(query, allowed)
        filters = {key: self._value(query, key) for key in allowed - {"sort", "page_size", "cursor"}}
        start = self._date(filters["start_date"], "start_date")
        end = self._date(filters["end_date"], "end_date")
        if start and end and start > end:
            raise ApiContractError(400, "invalid_query", "start_date cannot follow end_date.")
        if filters["status"] and filters["status"] != "completed":
            raise ApiContractError(400, "invalid_query", "Unknown EvaluationRun status.")
        if filters["integrity_status"] and filters["integrity_status"] not in {
            item.value for item in IntegrityStatus
        }:
            raise ApiContractError(400, "invalid_query", "Unknown integrity status.")
        items = list(registry.evaluation_runs)
        if filters["profile_id"]:
            items = [item for item in items if item.evaluation_profile_id == filters["profile_id"]]
        if filters["strategy_run_id"]:
            items = [item for item in items if filters["strategy_run_id"] in item.strategy_run_ids]
        if filters["status"]:
            items = [item for item in items if item.status == filters["status"]]
        if filters["integrity_status"]:
            items = [item for item in items if item.integrity_status.value == filters["integrity_status"]]
        if start:
            items = [item for item in items if item.creation_time[:10] >= start]
        if end:
            items = [item for item in items if item.creation_time[:10] <= end]
        sort = self._value(query, "sort") or "-creation_time"
        items = self._sort(
            items,
            sort,
            {"creation_time", "evaluation_run_id", "evaluation_profile_id", "status"},
            "evaluation_run_id",
        )
        selected, page = self._page(
            items,
            query=query,
            registry_id=registry.registry_id,
            resource="evaluation_runs",
            signature_fields={"filters": filters, "sort": sort},
        )
        return {"items": [item.to_dict() for item in selected], "page": page, "sort": sort}

    def _evaluation_detail(
        self,
        registry: SavedRunRegistry,
        evaluation_id: str,
        query: Mapping[str, Sequence[str]],
        *,
        outputs_only: bool = False,
    ) -> Mapping[str, Any]:
        self._allow_query(query, {"page_size", "cursor"})
        entry = self._one(
            registry.evaluation_runs, "evaluation_run_id", evaluation_id, "EvaluationRun"
        )
        if entry.integrity_status != IntegrityStatus.VALID:
            raise ApiContractError(
                409,
                "integrity_validation_failed",
                "EvaluationRun references are incomplete.",
                object_identity=evaluation_id,
            )
        try:
            run = self.store.get_evaluation_run(evaluation_id)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            raise ApiContractError(
                409, "integrity_validation_failed", "EvaluationRun could not be validated.", object_identity=evaluation_id
            ) from error
        selected, page = self._page(
            list(run.results),
            query=query,
            registry_id=registry.registry_id,
            resource=f"evaluation_results:{evaluation_id}",
            signature_fields={"evaluation_run_id": evaluation_id},
        )
        if outputs_only:
            items = [
                {
                    "strategy_run_id": item.strategy_run_id,
                    "mandatory_gates": canonical_data(item.mandatory_gate_results),
                    "mandatory_gates_passed": item.mandatory_gates_passed,
                    "pareto": {
                        "member": item.pareto_member,
                        "dominated_by": list(item.dominated_by),
                    },
                    "robustness_vetoes": canonical_data(item.robustness_results),
                    "robustness_passed": item.robustness_passed,
                    "tie_break_order": item.lexicographic_order,
                    "exploratory_weighted": canonical_data(item.weighted_view),
                    "decision_labels": list(item.final_labels),
                    "behavior": canonical_data(item.behavior_deduplication_metadata),
                }
                for item in selected
            ]
        else:
            items = canonical_data(selected)
        return {
            "evaluation_run_id": run.evaluation_run_id,
            "evaluation_profile_id": run.evaluation_profile_id,
            "profile_hash": run.profile_hash,
            "comparison_mode": run.comparison_mode.value,
            "metric_engine_version": run.metric_engine_version,
            "benchmark_data_identity": run.benchmark_data_identity,
            "derived_metric_ids": canonical_data(run.derived_metric_ids),
            "creation_time": run.creation_time,
            "items": items,
            "page": page,
        }

    def _evaluation_behavior(
        self,
        registry: SavedRunRegistry,
        evaluation_id: str,
        query: Mapping[str, Sequence[str]],
    ) -> Mapping[str, Any]:
        self._allow_query(query, {"page_size", "cursor"})
        self._one(registry.evaluation_runs, "evaluation_run_id", evaluation_id, "EvaluationRun")
        run = self.store.get_evaluation_run(evaluation_id)
        pairs = [
            {"pair_id": pair_id, **canonical_data(value)}
            for pair_id, value in sorted(run.behavior_pairwise_diagnostics.items())
        ]
        selected, page = self._page(
            pairs,
            query=query,
            registry_id=registry.registry_id,
            resource=f"evaluation_behavior:{evaluation_id}",
            signature_fields={"evaluation_run_id": evaluation_id},
        )
        return {
            "evaluation_run_id": evaluation_id,
            "pairwise_diagnostics": selected,
            "simplicity_metadata": canonical_data(run.simplicity_metadata),
            "candidate_clusters": {
                item.strategy_run_id: canonical_data(item.behavior_deduplication_metadata)
                for item in run.results
            },
            "page": page,
        }

    def _list_attempts(
        self, registry: SavedRunRegistry, query: Mapping[str, Sequence[str]]
    ) -> Mapping[str, Any]:
        allowed = {
            "intended_strategy_run_id",
            "operational_status",
            "source_commit",
            "engine_version",
            "start_date",
            "end_date",
            "sort",
            "page_size",
            "cursor",
        }
        self._allow_query(query, allowed)
        filters = {key: self._value(query, key) for key in allowed - {"sort", "page_size", "cursor"}}
        start = self._date(filters["start_date"], "start_date")
        end = self._date(filters["end_date"], "end_date")
        if start and end and start > end:
            raise ApiContractError(400, "invalid_query", "start_date cannot follow end_date.")
        if filters["operational_status"] and filters["operational_status"] not in {
            item.value for item in AttemptOperationalStatus
        }:
            raise ApiContractError(400, "invalid_query", "Unknown execution-attempt status.")
        items = list(registry.execution_attempts)
        for field_name in ("intended_strategy_run_id", "source_commit", "engine_version"):
            if filters[field_name]:
                items = [item for item in items if getattr(item, field_name) == filters[field_name]]
        if filters["operational_status"]:
            items = [
                item for item in items if item.operational_status.value == filters["operational_status"]
            ]
        if start:
            items = [item for item in items if item.created_timestamp[:10] >= start]
        if end:
            items = [item for item in items if item.created_timestamp[:10] <= end]
        sort = self._value(query, "sort") or "-created_timestamp"
        items = self._sort(
            items,
            sort,
            {"created_timestamp", "execution_attempt_id", "operational_status", "attempt_number"},
            "execution_attempt_id",
        )
        selected, page = self._page(
            items,
            query=query,
            registry_id=registry.registry_id,
            resource="execution_attempts",
            signature_fields={"filters": filters, "sort": sort},
        )
        return {
            "items": [
                {
                    **item.to_dict(),
                    "operational_status_ko": self.status_labels.get(item.operational_status.value),
                }
                for item in selected
            ],
            "page": page,
            "sort": sort,
        }

    def _route(
        self,
        path: str,
        query: Mapping[str, Sequence[str]],
        registry: SavedRunRegistry,
    ) -> Mapping[str, Any]:
        if path == f"{API_PATH_PREFIX}/health":
            self._allow_query(query, set())
            return self._health(registry)
        if path == f"{API_PATH_PREFIX}/metadata":
            self._allow_query(query, set())
            return self._metadata(registry)
        if path == f"{API_PATH_PREFIX}/overview":
            self._allow_query(query, set())
            return self._overview(registry)
        if path == f"{API_PATH_PREFIX}/terminology":
            self._allow_query(query, set())
            return self._terminology(registry)
        if path == f"{API_PATH_PREFIX}/runs":
            return self._list_runs(registry, query)
        if path == f"{API_PATH_PREFIX}/evaluation-profiles":
            return self._list_profiles(registry, query)
        if path == f"{API_PATH_PREFIX}/evaluation-runs":
            return self._list_evaluations(registry, query)
        if path == f"{API_PATH_PREFIX}/execution-attempts":
            return self._list_attempts(registry, query)

        parts = path.removeprefix(f"{API_PATH_PREFIX}/").split("/")
        if parts[0] == "runs" and len(parts) >= 2:
            run_id = self._identifier(parts[1], "StrategyRun")
            run = self._one(registry.strategy_runs, "strategy_run_id", run_id, "StrategyRun")
            if len(parts) == 2:
                self._allow_query(query, set())
                return run.to_dict()
            if len(parts) == 3:
                resource = parts[2]
                if resource == "manifest":
                    self._allow_query(query, set())
                    return self.store.get_strategy_run_manifest(run_id).to_dict()
                if resource == "specification":
                    self._allow_query(query, set())
                    return {
                        "strategy_run_id": run_id,
                        "specification_hash": run.specification_hash,
                        "specification": canonical_data(run.canonical_specification),
                    }
                if resource == "provenance":
                    self._allow_query(query, set())
                    return {
                        "strategy_run_id": run_id,
                        "manifest_hash": run.manifest_hash,
                        "source_data_snapshot_id": run.source_data_snapshot_id,
                        "source_commit": run.source_commit,
                        "engine_version": run.engine_version,
                        "artifact_hashes": {
                            f"{item.owner_kind}:{item.owner_id}:{item.artifact_key}": item.content_hash
                            for item in run.artifacts
                            if item.content_hash is not None
                        },
                        "benchmark_provenance": canonical_data(run.benchmark_provenance),
                        "calculation_provenance": canonical_data(run.calculation_provenance),
                        "evaluation_run_ids": list(run.evaluation_run_ids),
                    }
                if resource == "status":
                    self._allow_query(query, set())
                    return {
                        "strategy_run_id": run_id,
                        "terminal_status": run.terminal_status,
                        "terminal_status_ko": self.status_labels.get(run.terminal_status),
                        "integrity_status": run.integrity_status.value,
                        "integrity_status_ko": self.status_labels.get(run.integrity_status.value),
                        "retention_status": run.retention_status,
                        "validation_errors": list(run.validation_errors),
                    }
                if resource == "artifacts":
                    self._allow_query(query, set())
                    return {"strategy_run_id": run_id, "items": [item.to_dict() for item in run.artifacts]}
                if resource == "behavior-summary":
                    return self._behavior_summary(run, query)
                artifact_routes = {
                    "curve": "daily_portfolio_curve",
                    "benchmark-curve": "benchmark_daily_portfolio_curve",
                    "yearly-metrics": "yearly_metrics",
                    "rolling-metrics": "rolling_metrics",
                    "derived-metrics": "derived_metrics",
                    "robustness-summary": "robustness_summary",
                    "behavior": "behavior_metadata",
                }
                if resource in artifact_routes:
                    return self._artifact_response(registry, run, artifact_routes[resource], query)
        if parts[0] == "evaluation-profiles" and len(parts) == 2:
            profile_id = self._identifier(parts[1], "EvaluationProfile")
            self._allow_query(query, set())
            self._one(
                registry.evaluation_profiles,
                "evaluation_profile_id",
                profile_id,
                "EvaluationProfile",
            )
            return self.store.get_evaluation_profile(profile_id).to_dict()
        if parts[0] == "evaluation-runs" and len(parts) in {2, 3}:
            evaluation_id = self._identifier(parts[1], "EvaluationRun")
            if len(parts) == 2:
                return self._evaluation_detail(registry, evaluation_id, query)
            if parts[2] == "outputs":
                return self._evaluation_detail(registry, evaluation_id, query, outputs_only=True)
            if parts[2] == "behavior":
                return self._evaluation_behavior(registry, evaluation_id, query)
        if parts[0] == "execution-attempts" and len(parts) == 2:
            attempt_id = self._identifier(parts[1], "execution attempt")
            self._allow_query(query, set())
            attempt = self._one(
                registry.execution_attempts,
                "execution_attempt_id",
                attempt_id,
                "ExecutionAttempt",
            )
            return {
                **attempt.to_dict(),
                "operational_status_ko": self.status_labels.get(attempt.operational_status.value),
            }
        raise ApiContractError(404, "not_found", "API route was not found.")

    def dispatch(
        self,
        method: str,
        target: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> ApiResponse:
        request_headers = headers or {}
        request_id = self._request_id(request_headers)
        try:
            if method.upper() not in {"GET", "HEAD"}:
                raise ApiContractError(405, "method_not_allowed", "Production API is read-only.")
            path, query = self._query(target)
            registry = self.registry_builder.load_or_rebuild()
            body = self._route(path, query, registry)
            response_body = (
                {} if method.upper() == "HEAD" else _redact_secrets(canonical_data(body))
            )
            if len(canonical_bytes(response_body)) > self.server_config.max_response_bytes:
                raise ApiContractError(400, "invalid_query", "Response exceeds the configured bound.")
            return ApiResponse(
                status_code=200,
                body=response_body,
                headers={"X-Request-ID": request_id, "Content-Type": "application/json; charset=utf-8"},
            )
        except ApiContractError as error:
            return self._error_response(error, request_id)
        except Exception:
            return self._error_response(
                ApiContractError(
                    500,
                    "internal_error",
                    "Unexpected local API error.",
                    recoverable=False,
                ),
                request_id,
            )


def build_http_server(api: ReadOnlyTrendApi) -> ThreadingHTTPServer:
    """Create a local stdlib server; the caller controls serve/shutdown lifecycle."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "TrendV2LocalAPI/1"

        def _send(self, method: str) -> None:
            response = api.dispatch(method, self.path, headers=dict(self.headers.items()))
            payload = canonical_bytes(response.body)
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                self.send_header(key, value)
            origin = self.headers.get("Origin")
            if origin and origin in api.server_config.cors_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Content-Length", str(0 if method == "HEAD" else len(payload)))
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            self._send("GET")

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler contract
            self._send("HEAD")

        def do_POST(self) -> None:  # noqa: N802 - explicit read-only response
            self._send("POST")

        def do_PUT(self) -> None:  # noqa: N802 - explicit read-only response
            self._send("PUT")

        def do_DELETE(self) -> None:  # noqa: N802 - explicit read-only response
            self._send("DELETE")

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return ThreadingHTTPServer((api.server_config.host, api.server_config.port), Handler)
