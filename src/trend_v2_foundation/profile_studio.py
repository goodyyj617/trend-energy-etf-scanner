"""Bounded local authoring for immutable EvaluationProfile revisions."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .canonical import canonical_bytes, canonical_data, content_hash
from .contracts import EVALUATION_PROFILE_V2_VERSION, EvaluationProfile
from .integration import calculate_and_evaluate_saved_runs
from .metrics import METRIC_REGISTRY, EvaluationProfileValidationError
from .result_store import LocalResultStore


PROFILE_STUDIO_OPTIONS_VERSION = "evaluation_profile_studio_options_v1"
_CONFIG_FIELDS = frozenset(
    {
        "comparison_mode", "enabled_metrics", "metric_directions", "metric_modes",
        "mandatory_gates", "pareto_objectives", "robustness_vetoes",
        "lexicographic_tie_break", "exploratory_metric_weights",
        "normalization_method", "ranking_sensitivity_delta",
        "high_weighted_rank_cutoff", "behavior_deduplication", "description",
    }
)
_BEHAVIOR_METRICS = {
    "daily_return_correlation": ">=",
    "active_date_jaccard": ">=",
    "entry_date_jaccard": ">=",
    "exit_date_jaccard": ">=",
    "normalized_path_distance": "<=",
}
_REPRESENTATIVE_FIELDS = ("complexity_score", "parameter_count")


class ProfileStudioError(ValueError):
    def __init__(self, code: str, diagnostic_en: str) -> None:
        super().__init__(code)
        self.code = code
        self.diagnostic_en = diagnostic_en


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ProfileStudioService:
    """Create and apply profile revisions without touching economic execution."""

    def __init__(self, store: LocalResultStore, *, clock: Callable[[], str] = _now) -> None:
        self.store = store
        self.clock = clock
        self.idempotency_dir = store.root / "profile_studio_v1" / "idempotency"
        self.idempotency_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def options(self) -> Mapping[str, Any]:
        metrics = []
        for key, definition in sorted(METRIC_REGISTRY.items()):
            item = canonical_data(definition)
            item["capabilities"] = {
                "mandatory_gate": definition.suitable_for_gates,
                "pareto": definition.suitable_for_pareto,
                "robustness_veto": definition.suitable_for_robustness,
                "exploratory_weight": definition.suitable_for_weighted_view,
            }
            metrics.append(item)
        return {
            "schema_version": PROFILE_STUDIO_OPTIONS_VERSION,
            "metrics": metrics,
            "comparison_modes": ["constraint_pareto", "exploratory_weighted"],
            "normalization_methods": ["min_max", "z_score"],
            "gate_operators": [">=", "<="],
            "behavior_deduplication": {
                "source_artifact": "behavior_metadata",
                "condition_options": [
                    {"metric_key": key, "operator": operator}
                    for key, operator in sorted(_BEHAVIOR_METRICS.items())
                ],
                "representative_fields": list(_REPRESENTATIVE_FIELDS),
            },
        }

    def _source(self, profile_id: str) -> EvaluationProfile:
        try:
            return self.store.get_evaluation_profile(profile_id)
        except KeyError as error:
            raise ProfileStudioError("profile_source_not_found", "Source EvaluationProfile does not exist.") from error

    @staticmethod
    def _plain_text(value: Any, field: str, *, required: bool = False) -> str:
        if not isinstance(value, str) or (required and not value.strip()) or "\x00" in str(value):
            raise ProfileStudioError("profile_studio_invalid_payload", f"{field} must be plain text.")
        return value.strip() if required else value

    def _candidate(self, payload: Mapping[str, Any]) -> tuple[EvaluationProfile, EvaluationProfile, str, str]:
        if set(payload) != {"source_profile_id", "change_summary_ko", "profile"}:
            raise ProfileStudioError("profile_studio_invalid_payload", "Studio draft has unsupported fields.")
        source_id = payload.get("source_profile_id")
        if not isinstance(source_id, str):
            raise ProfileStudioError("profile_studio_invalid_payload", "source_profile_id is required.")
        source = self._source(source_id)
        summary = self._plain_text(payload.get("change_summary_ko"), "change_summary_ko", required=True)
        if len(summary) > 500:
            raise ProfileStudioError("profile_studio_invalid_payload", "change_summary_ko exceeds its bound.")
        draft = payload.get("profile")
        if not isinstance(draft, Mapping) or set(draft) != _CONFIG_FIELDS:
            raise ProfileStudioError("profile_studio_invalid_payload", "Profile draft fields are not allow-listed.")
        if not isinstance(draft.get("description"), str) or len(draft["description"]) > 2000 or "\x00" in draft["description"]:
            raise ProfileStudioError("profile_studio_invalid_payload", "description must be bounded plain text.")
        self._assert_safe_text(draft)
        self._validate_epsilons(draft)
        behavior = draft.get("behavior_deduplication")
        if not isinstance(behavior, Mapping):
            raise ProfileStudioError("profile_behavior_option_invalid", "behavior_deduplication must be an object.")
        self._validate_behavior(behavior)
        profile_payload = {
            **canonical_data(draft),
            "name": "profile_studio_draft",
            "approval_status": source.approval_status,
            "schema_version": "evaluation_profile_v1",
        }
        try:
            candidate = EvaluationProfile.from_dict(profile_payload)
        except (TypeError, ValueError, EvaluationProfileValidationError) as error:
            code = "profile_metric_not_allowed" if "metric key" in str(error) or "suitable" in str(error) else "profile_validation_failed"
            raise ProfileStudioError(code, str(error)) from error
        draft_hash = content_hash({"source_profile_id": source_id, "change_summary_ko": summary, "profile": candidate.to_dict()})
        return source, candidate, summary, draft_hash

    @staticmethod
    def _validate_behavior(behavior: Mapping[str, Any]) -> None:
        allowed = {"enabled", "source_artifact", "required_conditions", "representative_order"}
        if set(behavior).difference(allowed) or behavior.get("source_artifact") != "behavior_metadata":
            raise ProfileStudioError("profile_behavior_option_invalid", "Behavior configuration contains unsupported fields.")
        if not isinstance(behavior.get("enabled"), bool):
            raise ProfileStudioError("profile_behavior_option_invalid", "Behavior enabled flag is required.")
        conditions = behavior.get("required_conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ProfileStudioError("profile_behavior_option_invalid", "At least one behavior condition is required.")
        seen: set[str] = set()
        for condition in conditions:
            if not isinstance(condition, Mapping) or set(condition) != {"metric_key", "operator", "threshold"}:
                raise ProfileStudioError("profile_behavior_option_invalid", "Behavior condition shape is invalid.")
            key = condition.get("metric_key")
            if key in seen or key not in _BEHAVIOR_METRICS or condition.get("operator") != _BEHAVIOR_METRICS[key]:
                raise ProfileStudioError("profile_behavior_option_invalid", "Behavior diagnostic is not allowed.")
            threshold = condition.get("threshold")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold != threshold or threshold in (float("inf"), float("-inf")):
                raise ProfileStudioError("profile_behavior_option_invalid", "Behavior threshold must be finite.")
            seen.add(key)
        order = behavior.get("representative_order")
        if not isinstance(order, list) or {item.get("field") for item in order if isinstance(item, Mapping)} != set(_REPRESENTATIVE_FIELDS):
            raise ProfileStudioError("profile_behavior_option_invalid", "Representative order must use the fixed fields.")
        for item in order:
            if not isinstance(item, Mapping) or set(item) != {"field", "direction"} or item.get("direction") != "minimize":
                raise ProfileStudioError("profile_behavior_option_invalid", "Representative order direction is invalid.")

    @staticmethod
    def _assert_safe_text(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                ProfileStudioService._assert_safe_text(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                ProfileStudioService._assert_safe_text(item)
        elif isinstance(value, str):
            lowered = value.casefold()
            if "\x00" in value or "http://" in lowered or "https://" in lowered or "file:" in lowered or "javascript:" in lowered or "import " in lowered or "\\" in value:
                raise ProfileStudioError("profile_studio_invalid_payload", "Profile text cannot contain code, paths, or URLs.")

    @staticmethod
    def _validate_epsilons(draft: Mapping[str, Any]) -> None:
        objectives = draft.get("pareto_objectives")
        if not isinstance(objectives, (list, tuple)):
            raise ProfileStudioError("profile_validation_failed", "pareto_objectives must be a list.")
        for index, objective in enumerate(objectives):
            if not isinstance(objective, Mapping):
                raise ProfileStudioError("profile_validation_failed", "Pareto objective shape is invalid.")
            epsilon = objective.get("epsilon")
            if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon < 0:
                raise ProfileStudioError("profile_validation_failed", f"pareto_objectives[{index}].epsilon must be finite and non-negative.")

    def validate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            _, candidate, _, draft_hash = self._candidate(payload)
        except ProfileStudioError as error:
            return {"valid": False, "draft_hash": None, "errors": [{"code": error.code, "field": "profile", "diagnostic_en": error.diagnostic_en}]}
        return {"valid": True, "draft_hash": draft_hash, "errors": [], "normalized_profile": candidate.to_dict()}

    @staticmethod
    def _lineage_for(profile: EvaluationProfile) -> Mapping[str, Any]:
        if profile.schema_version == EVALUATION_PROFILE_V2_VERSION:
            return canonical_data(profile.lineage)
        return {
            "root_profile_id": profile.evaluation_profile_id,
            "parent_profile_id": None,
            "revision": 0,
            "technical_name": profile.name,
            "created_at": None,
            "change_summary_ko": "초기 프로필",
            "changed_field_groups": [],
        }

    def history(self, profile_id: str) -> Mapping[str, Any]:
        profile = self._source(profile_id)
        root = str(self._lineage_for(profile)["root_profile_id"])
        items = []
        for identity in self.store.evaluation_profile_history():
            item = self.store.get_evaluation_profile(identity)
            lineage = self._lineage_for(item)
            if lineage["root_profile_id"] == root:
                items.append({"evaluation_profile_id": item.evaluation_profile_id, "profile_hash": item.profile_hash, "name": item.name, "approval_status": item.approval_status, "lineage": lineage})
        return {"root_profile_id": root, "items": sorted(items, key=lambda item: (item["lineage"]["revision"], item["evaluation_profile_id"]))}

    def _idempotent(self, operation: str, key: str, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if not key or len(key) > 128:
            raise ProfileStudioError("profile_studio_invalid_payload", "Idempotency-Key is required.")
        request_hash = content_hash({"operation": operation, "key": key, "request": request})
        path = self._idempotency_path(operation, key)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("request_hash") != request_hash:
            raise ProfileStudioError("profile_idempotency_conflict", "Idempotency-Key was previously used for another request.")
        return record["response"]

    def _remember(self, operation: str, key: str, request: Mapping[str, Any], response: Mapping[str, Any]) -> None:
        request_hash = content_hash({"operation": operation, "key": key, "request": request})
        path = self._idempotency_path(operation, key)
        self.store._write_immutable(path, canonical_bytes({"request_hash": request_hash, "response": canonical_data(response)}))

    def _idempotency_path(self, operation: str, key: str) -> Path:
        identity = content_hash({"operation": operation, "key": key})
        return self.idempotency_dir / f"{identity}.json"

    def save(self, payload: Mapping[str, Any], *, idempotency_key: str) -> tuple[Mapping[str, Any], bool]:
        if set(payload) != {"source_profile_id", "change_summary_ko", "profile", "validated_draft_hash"}:
            raise ProfileStudioError("profile_studio_invalid_payload", "Save payload has unsupported fields.")
        replay = self._idempotent("save", idempotency_key, payload)
        if replay is not None:
            return replay, True
        draft_payload = {key: payload[key] for key in ("source_profile_id", "change_summary_ko", "profile")}
        source, candidate, summary, draft_hash = self._candidate(draft_payload)
        if payload.get("validated_draft_hash") != draft_hash:
            raise ProfileStudioError("profile_validation_stale", "Draft differs from the validated payload.")
        with self._lock:
            parent_lineage = self._lineage_for(source)
            root = str(parent_lineage["root_profile_id"])
            revisions = [item["lineage"]["revision"] for item in self.history(source.evaluation_profile_id)["items"]]
            revision = max(revisions) + 1
            lineage = {
                "root_profile_id": root,
                "parent_profile_id": source.evaluation_profile_id,
                "revision": revision,
                "technical_name": f"{self.store.get_evaluation_profile(root).name}__v{revision}",
                "created_at": self.clock(),
                "change_summary_ko": summary,
                "changed_field_groups": sorted(
                    key for key in _CONFIG_FIELDS if candidate.to_dict().get(key) != source.to_dict().get(key)
                ),
            }
            version_payload = candidate.to_dict()
            version_payload.update({"name": lineage["technical_name"], "schema_version": EVALUATION_PROFILE_V2_VERSION, "lineage": lineage})
            version = EvaluationProfile.from_dict(version_payload)
            self.store.save_evaluation_profile(version)
        response = {"profile": version.to_dict(), "evaluation_profile_id": version.evaluation_profile_id, "profile_hash": version.profile_hash, "lineage": canonical_data(lineage)}
        self._remember("save", idempotency_key, payload, response)
        return response, False

    def apply(self, profile_id: str, payload: Mapping[str, Any], *, idempotency_key: str) -> tuple[Mapping[str, Any], bool]:
        if set(payload) != {"strategy_run_id"} or not isinstance(payload.get("strategy_run_id"), str):
            raise ProfileStudioError("profile_studio_invalid_payload", "strategy_run_id is required.")
        request = {"profile_id": profile_id, **dict(payload)}
        replay = self._idempotent("apply", idempotency_key, request)
        if replay is not None:
            return replay, True
        profile = self._source(profile_id)
        try:
            result = calculate_and_evaluate_saved_runs(self.store, [payload["strategy_run_id"]], profile, creation_time=self.clock())
        except KeyError as error:
            raise ProfileStudioError("profile_apply_strategy_run_not_found", "StrategyRun does not exist.") from error
        except (ValueError, FileNotFoundError) as error:
            raise ProfileStudioError("profile_apply_evidence_unavailable", str(error)) from error
        response = {"evaluation_run": result.evaluation_run.to_dict(), "cache_status": canonical_data(result.cache_status), "economic_backtest_started": False}
        self._remember("apply", idempotency_key, request, response)
        return response, False
