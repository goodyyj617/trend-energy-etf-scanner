"""Public Foundation 1 contracts for reusable Trend Strategy v2 tooling."""

from .canonical import canonical_bytes, canonical_data, canonical_json, content_hash
from .contracts import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactRetentionPolicy,
    CandidateEvaluation,
    CheckResult,
    ComparisonMode,
    EvaluationProfile,
    EvaluationRun,
    ExecutionStatus,
    GateRule,
    MetricDefinition,
    MetricDirection,
    MetricMode,
    NormalizationMethod,
    ParetoObjective,
    SortRule,
    StrategyRunManifest,
    StrategyRunSpec,
    WeightedCandidateView,
)
from .evaluation import METRIC_ENGINE_VERSION, epsilon_pareto, evaluate_saved_runs, evaluate_strategy_runs
from .metrics import METRIC_REGISTRY, metric_registry, metrics_from_portfolio_summaries
from .profiles import load_evaluation_profile, load_evaluation_profiles
from .result_store import LocalResultStore, ResultStore
from .terminology import load_terminology_source, validate_terminology_source

__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "ArtifactRetentionPolicy",
    "CandidateEvaluation",
    "CheckResult",
    "ComparisonMode",
    "EvaluationProfile",
    "EvaluationRun",
    "ExecutionStatus",
    "GateRule",
    "LocalResultStore",
    "METRIC_ENGINE_VERSION",
    "METRIC_REGISTRY",
    "MetricDefinition",
    "MetricDirection",
    "MetricMode",
    "NormalizationMethod",
    "ParetoObjective",
    "ResultStore",
    "SortRule",
    "StrategyRunManifest",
    "StrategyRunSpec",
    "WeightedCandidateView",
    "canonical_bytes",
    "canonical_data",
    "canonical_json",
    "content_hash",
    "epsilon_pareto",
    "evaluate_saved_runs",
    "evaluate_strategy_runs",
    "load_evaluation_profile",
    "load_evaluation_profiles",
    "load_terminology_source",
    "metric_registry",
    "metrics_from_portfolio_summaries",
    "validate_terminology_source",
]
