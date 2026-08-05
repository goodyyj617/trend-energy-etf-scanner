"""Immutable, versioned domain contracts for Trend Strategy v2 Foundation 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .canonical import canonical_data, content_hash, deep_freeze, deterministic_id


STRATEGY_RUN_SPEC_VERSION = "strategy_run_spec_v1"
STRATEGY_RUN_MANIFEST_VERSION = "strategy_run_manifest_v1"
EVALUATION_PROFILE_VERSION = "evaluation_profile_v1"
EVALUATION_PROFILE_V2_VERSION = "evaluation_profile_v2"
EVALUATION_RUN_VERSION = "evaluation_run_v2"
RETENTION_POLICY_VERSION = "artifact_retention_policy_v1"
METRIC_REGISTRY_VERSION = "metric_registry_v2"
DERIVED_METRIC_MANIFEST_VERSION = "derived_metric_manifest_v1"


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MetricMode(str, Enum):
    ABSOLUTE = "absolute"
    BENCHMARK_RELATIVE = "benchmark_relative"


class ComparisonMode(str, Enum):
    CONSTRAINT_PARETO = "constraint_pareto"
    EXPLORATORY_WEIGHTED = "exploratory_weighted"


class NormalizationMethod(str, Enum):
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


TERMINAL_EXECUTION_STATUSES = frozenset(
    {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.PARTIAL}
)


class ArtifactKind(str, Enum):
    SUMMARY_METRICS = "summary_metrics"
    DAILY_PORTFOLIO_CURVE = "daily_portfolio_curve"
    SIGNAL_EXECUTION_EVENTS = "signal_execution_events"
    TRADE_LIFECYCLES = "trade_lifecycles"
    YEARLY_METRICS = "yearly_metrics"
    ROLLING_METRICS = "rolling_metrics"
    ROBUSTNESS_SUMMARY = "robustness_summary"
    BEHAVIOR_METADATA = "behavior_metadata"
    DERIVED_METRICS = "derived_metrics"


@dataclass(frozen=True)
class StrategyRunSpec:
    data_snapshot_hash: str
    economic_date_range: Mapping[str, str]
    universe_specification: Mapping[str, Any]
    benchmark: Mapping[str, Any]
    trend_filter: Mapping[str, Any]
    signal: Mapping[str, Any]
    entry_rule: Mapping[str, Any]
    initial_stop: Mapping[str, Any]
    trailing_exit: Mapping[str, Any]
    position_sizing: Mapping[str, Any]
    portfolio_constraints: Mapping[str, Any]
    transaction_costs: Mapping[str, Any]
    slippage: Mapping[str, Any]
    engine_version: str
    schema_version: str = STRATEGY_RUN_SPEC_VERSION

    def __post_init__(self) -> None:
        mapping_fields = (
            "economic_date_range",
            "universe_specification",
            "benchmark",
            "trend_filter",
            "signal",
            "entry_rule",
            "initial_stop",
            "trailing_exit",
            "position_sizing",
            "portfolio_constraints",
            "transaction_costs",
            "slippage",
        )
        if not self.data_snapshot_hash or not self.engine_version:
            raise ValueError("snapshot hash and engine version are required")
        for name in mapping_fields:
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty mapping")
            object.__setattr__(self, name, deep_freeze(value))
        if set(self.economic_date_range) != {"start", "end"}:
            raise ValueError("economic_date_range must contain start and end")

    @property
    def strategy_run_id(self) -> str:
        return deterministic_id("strategy_run", self)

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyRunSpec":
        return cls(**dict(value))


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_key: str
    kind: ArtifactKind
    content_hash: str
    media_type: str
    encoding: str
    logical_bytes: int
    stored_bytes: int
    row_count: int

    def __post_init__(self) -> None:
        if not self.artifact_key or not self.content_hash:
            raise ValueError("artifact key and content hash are required")
        if min(self.logical_bytes, self.stored_bytes, self.row_count) < 0:
            raise ValueError("artifact sizes and row count cannot be negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRecord":
        payload = dict(value)
        payload["kind"] = ArtifactKind(payload["kind"])
        return cls(**payload)


@dataclass(frozen=True)
class StrategyRunManifest:
    strategy_run_id: str
    canonical_specification: Mapping[str, Any]
    source_code_commit: str
    snapshot_hash: str
    artifacts: tuple[ArtifactRecord, ...]
    row_counts: Mapping[str, int]
    creation_time: str
    execution_status: ExecutionStatus
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    schema_version: str = STRATEGY_RUN_MANIFEST_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_specification", deep_freeze(self.canonical_specification))
        object.__setattr__(self, "row_counts", deep_freeze(self.row_counts))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if self.execution_status not in TERMINAL_EXECUTION_STATUSES:
            raise ValueError(
                "StrategyRunManifest.execution_status must be terminal: "
                "succeeded, failed, or partial"
            )
        spec = StrategyRunSpec.from_dict(self.canonical_specification)
        if self.strategy_run_id != spec.strategy_run_id:
            raise ValueError("strategy_run_id does not match canonical specification")
        if self.snapshot_hash != spec.data_snapshot_hash:
            raise ValueError("manifest snapshot hash does not match specification")
        expected_counts = {artifact.artifact_key: artifact.row_count for artifact in self.artifacts}
        if dict(self.row_counts) != expected_counts:
            raise ValueError("row_counts must exactly match artifact records")

    @classmethod
    def create(
        cls,
        spec: StrategyRunSpec,
        *,
        source_code_commit: str,
        artifacts: Sequence[ArtifactRecord],
        creation_time: str,
        execution_status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
        warnings: Sequence[str] = (),
        limitations: Sequence[str] = (),
    ) -> "StrategyRunManifest":
        artifact_tuple = tuple(sorted(artifacts, key=lambda item: item.artifact_key))
        return cls(
            strategy_run_id=spec.strategy_run_id,
            canonical_specification=spec.to_dict(),
            source_code_commit=source_code_commit,
            snapshot_hash=spec.data_snapshot_hash,
            artifacts=artifact_tuple,
            row_counts={item.artifact_key: item.row_count for item in artifact_tuple},
            creation_time=creation_time,
            execution_status=execution_status,
            warnings=tuple(warnings),
            limitations=tuple(limitations),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyRunManifest":
        payload = dict(value)
        payload["artifacts"] = tuple(ArtifactRecord.from_dict(item) for item in payload["artifacts"])
        payload["execution_status"] = ExecutionStatus(payload["execution_status"])
        payload["warnings"] = tuple(payload.get("warnings", ()))
        payload["limitations"] = tuple(payload.get("limitations", ()))
        return cls(**payload)


@dataclass(frozen=True)
class GateRule:
    metric_key: str
    operator: str
    threshold: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.operator not in {">=", ">", "<=", "<", "==", "!="}:
            raise ValueError(f"unsupported gate operator: {self.operator}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateRule":
        return cls(**dict(value))


@dataclass(frozen=True)
class ParetoObjective:
    metric_key: str
    direction: MetricDirection
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError("epsilon cannot be negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParetoObjective":
        payload = dict(value)
        payload["direction"] = MetricDirection(payload["direction"])
        return cls(**payload)


@dataclass(frozen=True)
class SortRule:
    metric_key: str
    direction: MetricDirection

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SortRule":
        payload = dict(value)
        payload["direction"] = MetricDirection(payload["direction"])
        return cls(**payload)


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    comparison_mode: ComparisonMode
    enabled_metrics: tuple[str, ...]
    metric_directions: Mapping[str, MetricDirection]
    metric_modes: Mapping[str, MetricMode]
    mandatory_gates: tuple[GateRule, ...]
    pareto_objectives: tuple[ParetoObjective, ...]
    robustness_vetoes: tuple[GateRule, ...]
    lexicographic_tie_break: tuple[SortRule, ...]
    exploratory_metric_weights: Mapping[str, float] = field(default_factory=dict)
    normalization_method: NormalizationMethod | None = None
    ranking_sensitivity_delta: float = 0.10
    high_weighted_rank_cutoff: int = 3
    behavior_deduplication: Mapping[str, Any] = field(
        default_factory=lambda: {"enabled": True, "tolerance": 1e-8}
    )
    description: str = ""
    approval_status: str = "example_not_production_approved"
    schema_version: str = EVALUATION_PROFILE_VERSION
    lineage: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled_metrics", tuple(self.enabled_metrics))
        object.__setattr__(self, "mandatory_gates", tuple(self.mandatory_gates))
        object.__setattr__(self, "pareto_objectives", tuple(self.pareto_objectives))
        object.__setattr__(self, "robustness_vetoes", tuple(self.robustness_vetoes))
        object.__setattr__(self, "lexicographic_tie_break", tuple(self.lexicographic_tie_break))
        object.__setattr__(self, "metric_directions", deep_freeze(self.metric_directions))
        object.__setattr__(self, "metric_modes", deep_freeze(self.metric_modes))
        object.__setattr__(self, "exploratory_metric_weights", deep_freeze(self.exploratory_metric_weights))
        object.__setattr__(self, "behavior_deduplication", deep_freeze(self.behavior_deduplication))
        if self.lineage is not None:
            object.__setattr__(self, "lineage", deep_freeze(self.lineage))
        if self.schema_version not in {EVALUATION_PROFILE_VERSION, EVALUATION_PROFILE_V2_VERSION}:
            raise ValueError("unsupported EvaluationProfile schema_version")
        if self.schema_version == EVALUATION_PROFILE_V2_VERSION and not self.lineage:
            raise ValueError("evaluation_profile_v2 requires lineage")
        if self.schema_version == EVALUATION_PROFILE_VERSION and self.lineage is not None:
            raise ValueError("evaluation_profile_v1 cannot contain lineage")
        if len(set(self.enabled_metrics)) != len(self.enabled_metrics):
            raise ValueError("enabled_metrics cannot contain duplicates")
        if any(
            isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0
            for weight in self.exploratory_metric_weights.values()
        ):
            raise ValueError("exploratory weights must be non-negative numeric values")
        if self.comparison_mode == ComparisonMode.EXPLORATORY_WEIGHTED:
            if self.normalization_method is None:
                raise ValueError("weighted comparison requires an explicit normalization method")
            if not self.exploratory_metric_weights or sum(self.exploratory_metric_weights.values()) <= 0:
                raise ValueError("weighted comparison requires a positive total weight")
        if not 0 < self.ranking_sensitivity_delta <= 1:
            raise ValueError("ranking_sensitivity_delta must be in (0, 1]")
        if self.high_weighted_rank_cutoff < 1:
            raise ValueError("high_weighted_rank_cutoff must be positive")
        from .metrics import validate_evaluation_profile

        validate_evaluation_profile(self)

    @property
    def profile_hash(self) -> str:
        return content_hash(self.to_dict())

    @property
    def evaluation_profile_id(self) -> str:
        return f"evaluation_profile_{self.profile_hash}"

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_data(self)
        # Preserve the original v1 canonical payload byte-for-byte: existing
        # content-addressed profile IDs and EvaluationRuns depend on it.
        if self.schema_version == EVALUATION_PROFILE_VERSION:
            payload.pop("lineage", None)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationProfile":
        payload = dict(value)
        payload.setdefault("schema_version", EVALUATION_PROFILE_VERSION)
        if payload["schema_version"] == EVALUATION_PROFILE_VERSION:
            payload.pop("lineage", None)
        payload["comparison_mode"] = ComparisonMode(payload["comparison_mode"])
        payload["enabled_metrics"] = tuple(payload["enabled_metrics"])
        payload["metric_directions"] = {
            key: MetricDirection(direction) for key, direction in payload["metric_directions"].items()
        }
        payload["metric_modes"] = {
            key: MetricMode(mode) for key, mode in payload["metric_modes"].items()
        }
        payload["mandatory_gates"] = tuple(GateRule.from_dict(item) for item in payload["mandatory_gates"])
        payload["pareto_objectives"] = tuple(
            ParetoObjective.from_dict(item) for item in payload["pareto_objectives"]
        )
        payload["robustness_vetoes"] = tuple(
            GateRule.from_dict(item) for item in payload["robustness_vetoes"]
        )
        payload["lexicographic_tie_break"] = tuple(
            SortRule.from_dict(item) for item in payload["lexicographic_tie_break"]
        )
        method = payload.get("normalization_method")
        payload["normalization_method"] = NormalizationMethod(method) if method else None
        return cls(**payload)


@dataclass(frozen=True)
class CheckResult:
    metric_key: str
    operator: str
    threshold: float
    value: float | None
    passed: bool
    reason: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckResult":
        return cls(**dict(value))


@dataclass(frozen=True)
class WeightedCandidateView:
    exploratory_weighted_value: float | None
    rank: int | None
    normalized_metric_values: Mapping[str, float | None]
    weighted_contributions: Mapping[str, float | None]
    sensitivity_ranks: Mapping[str, int | None]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized_metric_values", deep_freeze(self.normalized_metric_values))
        object.__setattr__(self, "weighted_contributions", deep_freeze(self.weighted_contributions))
        object.__setattr__(self, "sensitivity_ranks", deep_freeze(self.sensitivity_ranks))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WeightedCandidateView":
        payload = dict(value)
        payload["warnings"] = tuple(payload.get("warnings", ()))
        return cls(**payload)


@dataclass(frozen=True)
class CandidateEvaluation:
    strategy_run_id: str
    raw_metrics: Mapping[str, float | int | None]
    mandatory_gate_results: tuple[CheckResult, ...]
    mandatory_gates_passed: bool
    pareto_member: bool
    dominated_by: tuple[str, ...]
    robustness_results: tuple[CheckResult, ...]
    robustness_passed: bool
    lexicographic_order: int | None
    behavior_deduplication_metadata: Mapping[str, Any]
    weighted_view: WeightedCandidateView | None
    unavailable_reasons: Mapping[str, str]
    final_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_metrics", deep_freeze(self.raw_metrics))
        object.__setattr__(self, "mandatory_gate_results", tuple(self.mandatory_gate_results))
        object.__setattr__(self, "dominated_by", tuple(self.dominated_by))
        object.__setattr__(self, "robustness_results", tuple(self.robustness_results))
        object.__setattr__(self, "behavior_deduplication_metadata", deep_freeze(self.behavior_deduplication_metadata))
        object.__setattr__(self, "unavailable_reasons", deep_freeze(self.unavailable_reasons))
        object.__setattr__(self, "final_labels", tuple(self.final_labels))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateEvaluation":
        payload = dict(value)
        payload["mandatory_gate_results"] = tuple(
            CheckResult.from_dict(item) for item in payload["mandatory_gate_results"]
        )
        payload["robustness_results"] = tuple(
            CheckResult.from_dict(item) for item in payload["robustness_results"]
        )
        payload["dominated_by"] = tuple(payload["dominated_by"])
        payload["final_labels"] = tuple(payload["final_labels"])
        payload.setdefault("unavailable_reasons", {})
        if payload.get("weighted_view") is not None:
            payload["weighted_view"] = WeightedCandidateView.from_dict(payload["weighted_view"])
        return cls(**payload)


@dataclass(frozen=True)
class EvaluationRun:
    strategy_run_ids: tuple[str, ...]
    evaluation_profile_id: str
    metric_engine_version: str
    benchmark_data_identity: str
    profile_hash: str
    comparison_mode: ComparisonMode
    results: tuple[CandidateEvaluation, ...]
    normalized_weights: Mapping[str, float]
    ranking_sensitivity: Mapping[str, Any]
    derived_metric_ids: Mapping[str, str]
    creation_time: str
    behavior_pairwise_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    simplicity_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    schema_version: str = EVALUATION_RUN_VERSION

    def __post_init__(self) -> None:
        ordered_ids = tuple(sorted(set(self.strategy_run_ids)))
        if not ordered_ids:
            raise ValueError("at least one strategy run is required")
        object.__setattr__(self, "strategy_run_ids", ordered_ids)
        object.__setattr__(self, "results", tuple(sorted(self.results, key=lambda item: item.strategy_run_id)))
        object.__setattr__(self, "normalized_weights", deep_freeze(self.normalized_weights))
        object.__setattr__(self, "ranking_sensitivity", deep_freeze(self.ranking_sensitivity))
        object.__setattr__(self, "derived_metric_ids", deep_freeze(self.derived_metric_ids))
        object.__setattr__(
            self,
            "behavior_pairwise_diagnostics",
            deep_freeze(self.behavior_pairwise_diagnostics),
        )
        object.__setattr__(self, "simplicity_metadata", deep_freeze(self.simplicity_metadata))
        if {item.strategy_run_id for item in self.results} != set(ordered_ids):
            raise ValueError("evaluation results must exactly match strategy_run_ids")
        if self.derived_metric_ids and set(self.derived_metric_ids) != set(ordered_ids):
            raise ValueError("derived_metric_ids must be empty or exactly match strategy_run_ids")

    @property
    def identity_content(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_run_ids": self.strategy_run_ids,
            "evaluation_profile_id": self.evaluation_profile_id,
            "metric_engine_version": self.metric_engine_version,
            "benchmark_data_identity": self.benchmark_data_identity,
            "derived_metric_ids": self.derived_metric_ids,
        }

    @property
    def evaluation_run_id(self) -> str:
        return deterministic_id("evaluation_run", self.identity_content)

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_data(self)
        payload["evaluation_run_id"] = self.evaluation_run_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationRun":
        payload = dict(value)
        expected_id = payload.pop("evaluation_run_id", None)
        payload["strategy_run_ids"] = tuple(payload["strategy_run_ids"])
        payload["comparison_mode"] = ComparisonMode(payload["comparison_mode"])
        payload["results"] = tuple(CandidateEvaluation.from_dict(item) for item in payload["results"])
        payload.setdefault("derived_metric_ids", {})
        payload.setdefault("behavior_pairwise_diagnostics", {})
        payload.setdefault("simplicity_metadata", {})
        run = cls(**payload)
        if expected_id is not None and expected_id != run.evaluation_run_id:
            raise ValueError("evaluation_run_id does not match identity content")
        return run


@dataclass(frozen=True)
class ArtifactRetentionPolicy:
    max_store_bytes: int
    max_artifact_bytes: int
    max_strategy_runs: int
    max_evaluation_runs: int
    retained_artifact_kinds: tuple[ArtifactKind, ...] = tuple(ArtifactKind)
    dense_candidate_symbol_date_matrices_default: bool = False
    schema_version: str = RETENTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if min(
            self.max_store_bytes,
            self.max_artifact_bytes,
            self.max_strategy_runs,
            self.max_evaluation_runs,
        ) <= 0:
            raise ValueError("retention limits must be positive")
        object.__setattr__(self, "retained_artifact_kinds", tuple(self.retained_artifact_kinds))
        if self.dense_candidate_symbol_date_matrices_default:
            raise ValueError("dense candidate-by-symbol-by-date storage cannot be enabled by default")

    @property
    def artifact_retention_policy_id(self) -> str:
        return deterministic_id("artifact_retention_policy", self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRetentionPolicy":
        payload = dict(value)
        payload["retained_artifact_kinds"] = tuple(
            ArtifactKind(item) for item in payload.get("retained_artifact_kinds", tuple(ArtifactKind))
        )
        return cls(**payload)


@dataclass(frozen=True)
class DerivedMetricManifest:
    """Immutable index for one cached calculation over stored economic artifacts."""

    strategy_run_id: str
    source_artifact_hashes: Mapping[str, str]
    benchmark_identity: str
    benchmark_artifact_hash: str | None
    metric_calculation_engine_version: str
    metric_definition_version: str
    calculation_settings: Mapping[str, Any]
    artifacts: tuple[ArtifactRecord, ...]
    creation_time: str
    schema_version: str = DERIVED_METRIC_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not self.strategy_run_id:
            raise ValueError("strategy_run_id is required")
        if not self.source_artifact_hashes:
            raise ValueError("source_artifact_hashes cannot be empty")
        if not self.metric_calculation_engine_version or not self.metric_definition_version:
            raise ValueError("metric calculation engine and definition versions are required")
        if not self.benchmark_identity:
            raise ValueError("benchmark_identity is required")
        object.__setattr__(self, "source_artifact_hashes", deep_freeze(self.source_artifact_hashes))
        object.__setattr__(self, "calculation_settings", deep_freeze(self.calculation_settings))
        object.__setattr__(self, "artifacts", tuple(sorted(self.artifacts, key=lambda item: item.artifact_key)))
        keys = [artifact.artifact_key for artifact in self.artifacts]
        if len(keys) != len(set(keys)):
            raise ValueError("derived metric artifact keys must be unique")

    @property
    def identity_content(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_run_id": self.strategy_run_id,
            "source_artifact_hashes": self.source_artifact_hashes,
            "benchmark_identity": self.benchmark_identity,
            "benchmark_artifact_hash": self.benchmark_artifact_hash,
            "metric_calculation_engine_version": self.metric_calculation_engine_version,
            "metric_definition_version": self.metric_definition_version,
            "calculation_settings": self.calculation_settings,
        }

    @property
    def derived_metric_id(self) -> str:
        return deterministic_id("derived_metric", self.identity_content)

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_data(self)
        payload["derived_metric_id"] = self.derived_metric_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DerivedMetricManifest":
        payload = dict(value)
        expected_id = payload.pop("derived_metric_id", None)
        payload["artifacts"] = tuple(ArtifactRecord.from_dict(item) for item in payload["artifacts"])
        manifest = cls(**payload)
        if expected_id is not None and expected_id != manifest.derived_metric_id:
            raise ValueError("derived_metric_id does not match identity content")
        return manifest


@dataclass(frozen=True)
class MetricDefinition:
    metric_key: str
    korean_name: str
    english_name: str
    abbreviation: str
    direction: MetricDirection
    unit: str
    annualization_convention: str
    required_input_artifacts: tuple[ArtifactKind, ...]
    suitable_for_gates: bool
    suitable_for_pareto: bool
    suitable_for_weighted_view: bool
    suitable_for_robustness: bool
    suitable_for_diagnostics: bool
    numeric_representation: str = "continuous_numeric"
    allowed_numeric_values: tuple[float, ...] = ()
    source_summary_key: str | None = None
    schema_version: str = METRIC_REGISTRY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_input_artifacts", tuple(self.required_input_artifacts))
        object.__setattr__(self, "allowed_numeric_values", tuple(self.allowed_numeric_values))
        if self.numeric_representation not in {"continuous_numeric", "binary_numeric"}:
            raise ValueError(f"unsupported numeric representation: {self.numeric_representation}")
        if self.numeric_representation == "binary_numeric" and self.allowed_numeric_values != (
            0.0,
            1.0,
        ):
            raise ValueError("binary numeric metrics must allow exactly 0.0 and 1.0")
