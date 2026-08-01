"""Primary Foundation 2 path from stored economic artifacts to EvaluationRun."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .artifact_schemas import (
    validate_behavior_metadata,
    validate_daily_portfolio_curve,
    validate_robustness_summary,
    validate_rolling_metrics,
    validate_yearly_metrics,
)
from .behavior import deduplicate_behaviors, generate_behavior_metadata
from .calculation import (
    METRIC_CALCULATION_ENGINE_VERSION,
    METRIC_DEFINITION_VERSION,
    calculate_metric_artifact,
    calculation_settings,
)
from .canonical import canonical_data, content_hash
from .contracts import (
    ArtifactKind,
    DerivedMetricManifest,
    EvaluationProfile,
    EvaluationRun,
)
from .evaluation import evaluate_strategy_runs
from .result_store import ResultStore


@dataclass(frozen=True)
class SavedRunEvaluationResult:
    evaluation_run: EvaluationRun
    derived_metric_ids: Mapping[str, str]
    cache_status: Mapping[str, str]
    unavailable_reasons: Mapping[str, Mapping[str, str]]
    benchmark_alignments: Mapping[str, Mapping[str, Any]]
    behavior_pairwise_diagnostics: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Mapping[str, Any]]


def _optional_artifact(
    store: ResultStore, strategy_run_id: str, artifact_key: str, validator: Any | None = None
) -> tuple[Any | None, str | None]:
    try:
        record = store.get_strategy_artifact_record(strategy_run_id, artifact_key)
    except KeyError:
        return None, None
    payload = store.load_artifact_payload(strategy_run_id, artifact_key)
    if validator is not None:
        validator(payload)
    return payload, record.content_hash


def _numeric_leaf_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_numeric_leaf_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_numeric_leaf_count(item) for item in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


def _default_simplicity(manifest: Any) -> dict[str, float]:
    specification = manifest.canonical_specification
    component_fields = (
        "trend_filter",
        "signal",
        "entry_rule",
        "initial_stop",
        "trailing_exit",
        "position_sizing",
        "portfolio_constraints",
    )
    parameter_count = sum(_numeric_leaf_count(specification[field]) for field in component_fields)
    nonempty_components = sum(bool(specification[field]) for field in component_fields)
    return {
        "parameter_count": float(parameter_count),
        "complexity_score": float(nonempty_components + parameter_count),
    }


def calculate_and_evaluate_saved_runs(
    store: ResultStore,
    strategy_run_ids: Sequence[str],
    profile: EvaluationProfile,
    *,
    creation_time: str,
    benchmark_curve: Mapping[str, Any] | None = None,
    benchmark_data_identity: str | None = None,
    calculation_settings_override: Mapping[str, Any] | None = None,
    simplicity_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    persist: bool = True,
) -> SavedRunEvaluationResult:
    """Calculate/cache metrics from stored artifacts, then apply one profile.

    This operation never invokes the economic backtest engine.  When no explicit
    benchmark curve is supplied, each StrategyRun must reference the same stored
    ``benchmark_daily_portfolio_curve`` artifact.
    """

    run_ids = tuple(sorted(set(strategy_run_ids)))
    if not run_ids:
        raise ValueError("at least one strategy_run_id is required")
    settings = dict(calculation_settings_override or calculation_settings())
    if "rolling_windows" not in settings or "min_common_observations" not in settings:
        raise ValueError("calculation settings require rolling_windows and min_common_observations")
    settings = canonical_data(settings)
    explicit_benchmark_hash: str | None = None
    if benchmark_curve is not None:
        validate_daily_portfolio_curve(benchmark_curve)
        explicit_benchmark_hash = content_hash(benchmark_curve)

    metrics_by_run: dict[str, Mapping[str, Any]] = {}
    reasons_by_run: dict[str, Mapping[str, str]] = {}
    behavior_by_run: dict[str, Mapping[str, Any]] = {}
    derived_ids: dict[str, str] = {}
    cache_status: dict[str, str] = {}
    alignments: dict[str, Mapping[str, Any]] = {}
    provenance: dict[str, Mapping[str, Any]] = {}
    default_simplicity: dict[str, Mapping[str, Any]] = {}
    observed_benchmark_hashes: set[str] = set()
    observed_benchmark_identities: set[str] = set()

    for strategy_run_id in run_ids:
        validation = store.validate_manifest(strategy_run_id)
        if not validation.valid:
            raise ValueError(
                f"stored StrategyRun manifest is invalid:{strategy_run_id}:"
                + ";".join(validation.errors)
            )
        strategy_manifest = store.get_strategy_run_manifest(strategy_run_id)
        daily_record = store.get_strategy_artifact_record(
            strategy_run_id, "daily_portfolio_curve"
        )
        daily = store.load_and_validate_artifact(
            strategy_run_id, "daily_portfolio_curve", validate_daily_portfolio_curve
        )
        if daily["economic_date_range"] != dict(
            strategy_manifest.canonical_specification["economic_date_range"]
        ):
            raise ValueError(
                f"daily_portfolio_curve economic date range does not match StrategyRun:"
                f"{strategy_run_id}"
            )
        if benchmark_curve is None:
            benchmark_record = store.get_strategy_artifact_record(
                strategy_run_id, "benchmark_daily_portfolio_curve"
            )
            selected_benchmark = store.load_and_validate_artifact(
                strategy_run_id,
                "benchmark_daily_portfolio_curve",
                validate_daily_portfolio_curve,
            )
            selected_benchmark_hash = benchmark_record.content_hash
        else:
            selected_benchmark = benchmark_curve
            selected_benchmark_hash = str(explicit_benchmark_hash)
        benchmark_spec = strategy_manifest.canonical_specification["benchmark"]
        selected_benchmark_identity = benchmark_data_identity or str(
            benchmark_spec.get("identity")
            or benchmark_spec.get("symbol")
            or f"benchmark_{content_hash(benchmark_spec)}"
        )
        observed_benchmark_hashes.add(selected_benchmark_hash)
        observed_benchmark_identities.add(selected_benchmark_identity)

        robustness, robustness_hash = _optional_artifact(
            store, strategy_run_id, "robustness_summary", validate_robustness_summary
        )
        trades, trades_hash = _optional_artifact(
            store, strategy_run_id, "trade_lifecycles"
        )
        source_hashes = {"daily_portfolio_curve": daily_record.content_hash}
        if robustness_hash is not None:
            source_hashes["robustness_summary"] = robustness_hash
        if trades_hash is not None:
            source_hashes["trade_lifecycles"] = trades_hash

        identity_manifest = DerivedMetricManifest(
            strategy_run_id=strategy_run_id,
            source_artifact_hashes=source_hashes,
            benchmark_identity=selected_benchmark_identity,
            benchmark_artifact_hash=selected_benchmark_hash,
            metric_calculation_engine_version=METRIC_CALCULATION_ENGINE_VERSION,
            metric_definition_version=METRIC_DEFINITION_VERSION,
            calculation_settings=settings,
            artifacts=(),
            creation_time=creation_time,
        )
        derived_id = identity_manifest.derived_metric_id
        derived_ids[strategy_run_id] = derived_id
        try:
            stored_manifest = store.get_derived_metric_manifest(derived_id)
            metric_artifact = store.load_derived_metric_artifact(
                derived_id, "derived_metrics"
            )
            yearly = store.load_derived_metric_artifact(derived_id, "yearly_metrics")
            rolling = store.load_derived_metric_artifact(derived_id, "rolling_metrics")
            behavior = store.load_derived_metric_artifact(derived_id, "behavior_metadata")
            validate_yearly_metrics(yearly)
            validate_rolling_metrics(rolling)
            validate_behavior_metadata(behavior)
            cache_status[strategy_run_id] = "reused"
        except KeyError:
            metric_artifact, yearly, rolling = calculate_metric_artifact(
                strategy_run_id=strategy_run_id,
                daily_curve=daily,
                benchmark_curve=selected_benchmark,
                benchmark_identity=selected_benchmark_identity,
                benchmark_artifact_hash=selected_benchmark_hash,
                source_artifact_hashes=source_hashes,
                robustness_summary=robustness,
                settings=settings,
            )
            behavior = generate_behavior_metadata(
                daily,
                daily_curve_hash=daily_record.content_hash,
                trade_lifecycles=trades,
                trade_lifecycles_hash=trades_hash,
            )
            metric_put = store.put_artifact(
                "derived_metrics", ArtifactKind.DERIVED_METRICS, metric_artifact, row_count=1
            )
            yearly_put = store.put_artifact(
                "yearly_metrics", ArtifactKind.YEARLY_METRICS, yearly, row_count=len(yearly["rows"])
            )
            rolling_put = store.put_artifact(
                "rolling_metrics", ArtifactKind.ROLLING_METRICS, rolling, row_count=len(rolling["rows"])
            )
            behavior_put = store.put_artifact(
                "behavior_metadata", ArtifactKind.BEHAVIOR_METADATA, behavior, row_count=1
            )
            stored_manifest = DerivedMetricManifest(
                strategy_run_id=strategy_run_id,
                source_artifact_hashes=source_hashes,
                benchmark_identity=selected_benchmark_identity,
                benchmark_artifact_hash=selected_benchmark_hash,
                metric_calculation_engine_version=METRIC_CALCULATION_ENGINE_VERSION,
                metric_definition_version=METRIC_DEFINITION_VERSION,
                calculation_settings=settings,
                artifacts=(
                    metric_put.record,
                    yearly_put.record,
                    rolling_put.record,
                    behavior_put.record,
                ),
                creation_time=creation_time,
            )
            if persist:
                store.save_derived_metric_manifest(stored_manifest)
            cache_status[strategy_run_id] = "calculated"
        metrics_by_run[strategy_run_id] = metric_artifact["values"]
        reasons_by_run[strategy_run_id] = metric_artifact["unavailable_reasons"]
        behavior_by_run[strategy_run_id] = behavior
        alignments[strategy_run_id] = metric_artifact["benchmark_alignment"]
        provenance[strategy_run_id] = {
            "strategy_run_manifest_hash": content_hash(strategy_manifest.to_dict()),
            "derived_metric_manifest": stored_manifest.to_dict(),
            "metric_artifact_hash": next(
                artifact.content_hash
                for artifact in stored_manifest.artifacts
                if artifact.artifact_key == "derived_metrics"
            ),
            "source_artifact_hashes": source_hashes,
        }
        default_simplicity[strategy_run_id] = _default_simplicity(strategy_manifest)

    if len(observed_benchmark_hashes) != 1 or len(observed_benchmark_identities) != 1:
        raise ValueError(
            "all candidates in one EvaluationRun must use the same benchmark identity and artifact hash"
        )
    effective_simplicity = dict(default_simplicity)
    effective_simplicity.update(simplicity_metadata or {})
    behavior_clusters, pairwise = deduplicate_behaviors(
        behavior_by_run,
        profile.behavior_deduplication,
        simplicity_metadata=effective_simplicity,
    )
    benchmark_identity = next(iter(observed_benchmark_identities))
    benchmark_hash = next(iter(observed_benchmark_hashes))
    evaluation_run = evaluate_strategy_runs(
        profile,
        metrics_by_run,
        benchmark_data_identity=f"{benchmark_identity}:{benchmark_hash}",
        metric_engine_version=METRIC_CALCULATION_ENGINE_VERSION,
        behavior_metadata=behavior_clusters,
        unavailable_reasons=reasons_by_run,
        derived_metric_ids=derived_ids,
        creation_time=creation_time,
    )
    if persist:
        store.save_evaluation_profile(profile)
        store.save_evaluation_run(evaluation_run)
    return SavedRunEvaluationResult(
        evaluation_run=evaluation_run,
        derived_metric_ids=derived_ids,
        cache_status=cache_status,
        unavailable_reasons=reasons_by_run,
        benchmark_alignments=alignments,
        behavior_pairwise_diagnostics=pairwise,
        provenance=provenance,
    )
