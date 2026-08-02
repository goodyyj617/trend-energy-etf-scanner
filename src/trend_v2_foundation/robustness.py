"""Foundation 7 bounded robustness plans and persisted scenario execution.

Economic StrategyRuns are deliberately read-only inputs.  This module writes
append-only plan/attempt/scenario records below the ResultStore and stores a
separate content-addressed evidence summary; it never alters a run manifest.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import canonical_bytes, canonical_data, content_hash, deterministic_id
from .contracts import StrategyRunSpec
from .result_store import LocalResultStore

ROBUSTNESS_PLAN_VERSION = "robustness_execution_plan_v1"
ROBUSTNESS_ATTEMPT_VERSION = "robustness_execution_manifest_v1"
ROBUSTNESS_SCENARIO_VERSION = "robustness_scenario_v1"
ROBUSTNESS_RESULT_VERSION = "robustness_scenario_result_v1"
ROBUSTNESS_SUMMARY_VERSION = "robustness_summary_v2"
ROBUSTNESS_ENGINE_VERSION = "trend_v2_bounded_robustness_engine_v1"
_METHODS = {"walk_forward_fixed_v1", "leave_one_year_out_v1", "paired_moving_block_bootstrap_v1", "canonical_cost_stress_v1"}
_TERMINAL = {"reused", "succeeded", "failed", "cancelled", "skipped", "blocked", "incomplete"}


class RobustnessError(ValueError):
    def __init__(self, code: str, diagnostic_en: str, *, object_identity: str | None = None, recoverable: bool = True) -> None:
        super().__init__(code)
        self.code, self.diagnostic_en, self.object_identity, self.recoverable = code, diagnostic_en, object_identity, recoverable
        self.message_ko = {
            "robustness_method_unsupported": "지원하지 않는 강건성 방법입니다.",
            "robustness_plan_invalid": "강건성 계획이 유효하지 않습니다.",
            "robustness_confirmation_required": "강건성 실행 전 확인이 필요합니다.",
            "robustness_confirmation_stale": "강건성 확인이 현재 계획과 일치하지 않습니다.",
            "robustness_hard_limit_exceeded": "강건성 실행 한도를 초과했습니다.",
            "walk_forward_fold_invalid": "워크포워드 폴드가 유효하지 않습니다.",
            "loyo_year_ineligible": "제외 연도가 적격하지 않습니다.",
            "bootstrap_block_length_invalid": "부트스트랩 블록 길이가 유효하지 않습니다.",
            "bootstrap_alignment_failed": "전략과 벤치마크 공통 날짜 정렬에 실패했습니다.",
            "bootstrap_sample_limit_exceeded": "부트스트랩 표본 수 한도를 초과했습니다.",
            "cost_stress_scenario_invalid": "비용 스트레스 시나리오가 유효하지 않습니다.",
            "robustness_resume_not_allowed": "이 강건성 시나리오는 재개할 수 없습니다.",
            "robustness_provenance_invalid": "강건성 증거의 출처가 유효하지 않습니다.",
        }.get(code, "강건성 실행 오류가 발생했습니다.")

    def to_dict(self, request_id: str = "local") -> dict[str, Any]:
        return {"code": self.code, "message_ko": self.message_ko, "diagnostic_en": self.diagnostic_en,
                "object_identity": self.object_identity, "request_id": request_id,
                "recoverable": self.recoverable, "suggested_action": "허용된 설정과 증거 상태를 확인하세요."}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RobustnessError("robustness_provenance_invalid", "Immutable robustness record conflicts.", object_identity=path.stem, recoverable=False)
        return
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RobustnessError("robustness_scenario_artifact_corrupt", "Robustness persistence record is corrupt.", object_identity=path.name, recoverable=False) from error
    if not isinstance(value, Mapping):
        raise RobustnessError("robustness_scenario_artifact_corrupt", "Robustness persistence record is not an object.", object_identity=path.name, recoverable=False)
    return value


@dataclass(frozen=True)
class RobustnessPolicy:
    document: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "RobustnessPolicy":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"schema_version", "maximum_walk_forward_folds", "maximum_loyo_scenarios", "maximum_bootstrap_samples", "minimum_block_length", "maximum_block_length", "maximum_cost_stress_scenarios", "maximum_combined_robustness_units", "explicit_confirmation_threshold", "hard_refusal_threshold", "maximum_economic_reruns", "maximum_date_span_days", "maximum_concurrent_robustness_tasks", "maximum_json_body_bytes", "confirmation_ttl_seconds"}
        if not isinstance(value, Mapping) or value.get("schema_version") != "robustness_execution_policy_v1" or set(value) != required:
            raise RobustnessError("robustness_plan_invalid", "Unsupported robustness execution policy.", recoverable=False)
        return cls(canonical_data(value))

    @property
    def policy_hash(self) -> str: return content_hash(self.document)


def load_robustness_catalog(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != "robustness_option_catalog_v1":
        raise RobustnessError("robustness_plan_invalid", "Unsupported robustness option catalog.", recoverable=False)
    ids = [entry.get("method_id") for entry in value.get("methods", []) if isinstance(entry, Mapping)]
    if set(ids) != _METHODS or len(ids) != len(set(ids)):
        raise RobustnessError("robustness_plan_invalid", "Robustness option catalog is incomplete.", recoverable=False)
    result = canonical_data(value); result["catalog_hash"] = content_hash(result); return result


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str): raise RobustnessError("robustness_plan_invalid", f"{field} must be ISO date.")
    try: return date.fromisoformat(value)
    except ValueError as error: raise RobustnessError("robustness_plan_invalid", f"{field} must be ISO date.") from error


def _dates(curve: Mapping[str, Any]) -> list[str]:
    rows = curve.get("rows")
    if not isinstance(rows, list) or not rows: raise RobustnessError("robustness_plan_invalid", "Base curve is missing.")
    dates = [str(row.get("economic_date", "")) for row in rows]
    if dates != sorted(set(dates)): raise RobustnessError("robustness_plan_invalid", "Base curve dates are not sorted and unique.")
    for item in dates: _parse_date(item, "curve date")
    return dates


def generate_walk_forward_folds(dates: Sequence[str], settings: Mapping[str, Any], policy: RobustnessPolicy) -> tuple[dict[str, Any], ...]:
    required = {"window", "minimum_training_observations", "minimum_test_observations", "folds", "incomplete_boundary_fold"}
    if set(settings) != required or settings["window"] not in {"expanding", "rolling"} or settings["incomplete_boundary_fold"] not in {"incomplete", "skip"}:
        raise RobustnessError("walk_forward_fold_invalid", "Walk-forward settings are not allow-listed.")
    minimum_train, minimum_test = int(settings["minimum_training_observations"]), int(settings["minimum_test_observations"])
    supplied = settings["folds"]
    if not isinstance(supplied, list) or not supplied or len(supplied) > int(policy.document["maximum_walk_forward_folds"]):
        raise RobustnessError("walk_forward_fold_invalid", "Walk-forward fold count is invalid.")
    index = {item: ordinal for ordinal, item in enumerate(dates)}; result = []
    last_test = -1
    for ordinal, fold in enumerate(supplied, 1):
        if not isinstance(fold, Mapping) or set(fold).difference({"training_start", "training_end", "test_start", "test_end", "gap_sessions"}):
            raise RobustnessError("walk_forward_fold_invalid", "Fold definition has unknown fields.")
        try:
            train_start, train_end, test_start, test_end = (index[str(fold[key])] for key in ("training_start", "training_end", "test_start", "test_end"))
            gap = int(fold.get("gap_sessions", 0))
        except (KeyError, ValueError, TypeError) as error:
            raise RobustnessError("walk_forward_fold_invalid", "Fold dates must be exact stored economic dates.") from error
        if gap < 0 or train_start > train_end or test_start > test_end or train_end + gap >= test_start or test_start <= last_test:
            raise RobustnessError("walk_forward_fold_invalid", "Fold ordering or embargo is invalid.")
        incomplete = (train_end - train_start + 1 < minimum_train or test_end - test_start + 1 < minimum_test)
        if incomplete and settings["incomplete_boundary_fold"] == "skip": continue
        result.append({"fold_id": f"wf-{ordinal:03d}", "ordinal": ordinal, "training_range": {"start": dates[train_start], "end": dates[train_end]}, "test_range": {"start": dates[test_start], "end": dates[test_end]}, "gap_sessions": gap, "window": settings["window"], "training_observations": train_end - train_start + 1, "test_observations": test_end - test_start + 1, "incomplete": incomplete})
        last_test = test_end
    if not result: raise RobustnessError("walk_forward_fold_invalid", "No eligible walk-forward folds remain.")
    return tuple(result)


def generate_loyo_years(dates: Sequence[str], settings: Mapping[str, Any], policy: RobustnessPolicy) -> tuple[dict[str, Any], ...]:
    allowed = {"included_years", "minimum_observations", "partial_year_eligibility", "minimum_trades"}
    if set(settings).difference(allowed) or settings.get("partial_year_eligibility", "exclude") not in {"exclude", "allow_flagged"}:
        raise RobustnessError("loyo_year_ineligible", "LOYO settings are invalid.")
    groups: dict[int, list[str]] = {}
    for item in dates: groups.setdefault(int(item[:4]), []).append(item)
    included = settings.get("included_years", sorted(groups))
    if not isinstance(included, list) or included != sorted(set(included)):
        raise RobustnessError("loyo_year_ineligible", "LOYO years must be ordered unique integers.")
    first, last = dates[0], dates[-1]; output = []
    for year in included:
        observations = groups.get(int(year), [])
        partial = bool(observations and (observations[0] == first or observations[-1] == last))
        eligible = bool(observations) and len(observations) >= int(settings.get("minimum_observations", 1)) and (not partial or settings.get("partial_year_eligibility", "exclude") == "allow_flagged")
        output.append({"year": int(year), "exclusion_range": None if not observations else {"start": observations[0], "end": observations[-1]}, "observation_count": len(observations), "partial_year": partial, "eligible": eligible, "incomplete_reason": None if eligible else "partial_or_insufficient_observations"})
    if len(output) > int(policy.document["maximum_loyo_scenarios"]): raise RobustnessError("loyo_year_ineligible", "LOYO scenario limit exceeded.")
    return tuple(output)


def aligned_paired_returns(strategy_curve: Mapping[str, Any], benchmark_curve: Mapping[str, Any]) -> tuple[tuple[str, float, float], ...]:
    strategy = {str(row["economic_date"]): float(row["daily_return"]) for row in strategy_curve.get("rows", [])}
    benchmark = {str(row["economic_date"]): float(row["daily_return"]) for row in benchmark_curve.get("rows", [])}
    common = sorted(set(strategy) & set(benchmark))
    if len(common) < 2: raise RobustnessError("bootstrap_alignment_failed", "At least two exact common economic dates are required.")
    return tuple((item, strategy[item], benchmark[item]) for item in common)


def paired_block_bootstrap(pairs: Sequence[tuple[str, float, float]], *, seed: int, sample_count: int, block_length: int, confidence_level: float) -> dict[str, Any]:
    if sample_count < 1: raise RobustnessError("bootstrap_sample_limit_exceeded", "Bootstrap sample count must be positive.")
    if block_length < 1 or block_length > len(pairs): raise RobustnessError("bootstrap_block_length_invalid", "Block length is outside aligned observations.")
    if not 0 < confidence_level < 1: raise RobustnessError("robustness_plan_invalid", "Confidence level must be in (0, 1).")
    rng, n = random.Random(seed), len(pairs); values: list[float] = []
    for _ in range(sample_count):
        sample: list[tuple[str, float, float]] = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(pairs[(start + offset) % n] for offset in range(block_length))
        values.append(statistics.fmean(item[1] - item[2] for item in sample[:n]))
    values.sort(); alpha = (1 - confidence_level) / 2
    lower, upper = values[max(0, math.floor(alpha * (sample_count - 1)))], values[min(sample_count - 1, math.ceil((1 - alpha) * (sample_count - 1)))]
    raw = min(1.0, 2 * min(sum(value <= 0 for value in values) / sample_count, sum(value >= 0 for value in values) / sample_count))
    return {"effect": statistics.fmean(item[1] - item[2] for item in pairs), "confidence_interval": {"lower": lower, "upper": upper, "confidence_level": confidence_level}, "raw_p_value": raw, "adjusted_p_value": None, "sample_count": sample_count, "block_length": block_length, "seed": seed, "block_method": "moving_circular_block", "effect_definition": "paired_mean_daily_return_difference", "test": "two_sided"}


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, Any]:
    if not raw_p_values or any(not 0 <= value <= 1 for value in raw_p_values.values()): raise RobustnessError("robustness_plan_invalid", "Holm family must contain finite p-values.")
    ordered = sorted(raw_p_values.items(), key=lambda item: (item[1], item[0])); prior = 0.0; adjusted = {}
    for index, (identity, value) in enumerate(ordered):
        prior = max(prior, min(1.0, (len(ordered) - index) * value)); adjusted[identity] = prior
    family = {"method": "holm", "hypothesis_ids": [key for key, _ in ordered], "raw_p_values": dict(ordered), "adjusted_p_values": adjusted}
    family["correction_identity"] = content_hash(family); return family


def estimate_work(plan: Mapping[str, Any], policy: RobustnessPolicy) -> dict[str, Any]:
    methods = plan["methods"]; fold = len(methods.get("walk_forward_fixed_v1", {}).get("folds", [])); loyo = len(methods.get("leave_one_year_out_v1", {}).get("years", [])); bootstrap = int(methods.get("paired_moving_block_bootstrap_v1", {}).get("sample_count", 0)); cost = len(methods.get("canonical_cost_stress_v1", {}).get("multipliers", [])); economic = fold + loyo + cost; total = economic + bootstrap + int(plan.get("evaluation_units", 0))
    values = {"economic_backtest_units": economic, "fold_backtest_units": fold, "loyo_backtest_units": loyo, "cost_stress_backtest_units": cost, "bootstrap_resample_units": bootstrap, "deterministic_metric_units": fold + loyo + bootstrap + cost, "evaluation_units": int(plan.get("evaluation_units", 0)), "total_policy_units": total}
    values["hard_limit_exceeded"] = total > int(policy.document["hard_refusal_threshold"]) or total > int(policy.document["maximum_combined_robustness_units"]) or economic > int(policy.document["maximum_economic_reruns"])
    values["confirmation_required"] = not values["hard_limit_exceeded"] and total >= int(policy.document["explicit_confirmation_threshold"])
    values["estimate_hash"] = content_hash({key: value for key, value in values.items() if key != "estimate_hash"}); return values


class RobustnessExecutionService:
    """Bounded, restart-safe scenario executor using Foundation-6-style append-only records."""
    def __init__(self, store: LocalResultStore, policy: RobustnessPolicy, catalog: Mapping[str, Any], *, source_commit: str, cost_stress_runner: Callable[[str, float], Mapping[str, Any]] | None = None, clock: Callable[[], str] = _now) -> None:
        self.store, self.policy, self.catalog, self.source_commit, self.cost_stress_runner, self.clock = store, policy, catalog, source_commit, cost_stress_runner, clock
        self.root = store.root / "robustness_execution_v1"; self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, identity: str) -> Path: return self.root / kind / f"{identity}.json"
    def _write(self, kind: str, identity: str, value: Mapping[str, Any]) -> None: _atomic(self._path(kind, identity), value)
    def _load(self, kind: str, identity: str) -> Mapping[str, Any]: return _read(self._path(kind, identity))

    def normalize(self, request: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"base_strategy_run_id", "methods", "seed", "evaluation_units", "confirmation_id"}
        if set(request).difference(allowed) or not isinstance(request.get("methods"), Mapping): raise RobustnessError("robustness_plan_invalid", "Robustness request fields are invalid.")
        base_id = str(request.get("base_strategy_run_id", ""))
        try:
            manifest = self.store.get_strategy_run_manifest(base_id); daily = self.store.load_artifact_payload(base_id, "daily_portfolio_curve"); benchmark = self.store.load_artifact_payload(base_id, "benchmark_daily_portfolio_curve")
            daily_record = self.store.get_strategy_artifact_record(base_id, "daily_portfolio_curve"); benchmark_record = self.store.get_strategy_artifact_record(base_id, "benchmark_daily_portfolio_curve")
        except KeyError as error: raise RobustnessError("robustness_plan_invalid", "A valid base StrategyRun and aligned benchmark are required.", object_identity=base_id) from error
        dates = _dates(daily); methods: dict[str, Any] = {}
        for method, settings in request["methods"].items():
            if method not in _METHODS: raise RobustnessError("robustness_method_unsupported", "Method is not in the bounded catalog.", object_identity=str(method))
            if not isinstance(settings, Mapping): raise RobustnessError("robustness_plan_invalid", "Method settings must be an object.")
            if method == "walk_forward_fixed_v1": methods[method] = {"folds": list(generate_walk_forward_folds(dates, settings, self.policy)), "settings": canonical_data(settings)}
            elif method == "leave_one_year_out_v1": methods[method] = {"years": list(generate_loyo_years(dates, settings, self.policy)), "settings": canonical_data(settings)}
            elif method == "paired_moving_block_bootstrap_v1":
                count, length, confidence = int(settings.get("sample_count", 0)), int(settings.get("block_length", 0)), float(settings.get("confidence_level", 0))
                if count > int(self.policy.document["maximum_bootstrap_samples"]): raise RobustnessError("bootstrap_sample_limit_exceeded", "Bootstrap sample limit exceeded.")
                if not int(self.policy.document["minimum_block_length"]) <= length <= int(self.policy.document["maximum_block_length"]): raise RobustnessError("bootstrap_block_length_invalid", "Bootstrap block length violates policy.")
                pairs = aligned_paired_returns(daily, benchmark); methods[method] = {"sample_count": count, "block_length": length, "confidence_level": confidence, "aligned_observations": len(pairs), "seed": int(settings.get("seed", request.get("seed", 0)))}
            else:
                multipliers = settings.get("multipliers")
                if not isinstance(multipliers, list) or not multipliers or len(multipliers) > int(self.policy.document["maximum_cost_stress_scenarios"]) or any(float(item) < 1 for item in multipliers): raise RobustnessError("cost_stress_scenario_invalid", "Cost stress multipliers must be finite allow-listed non-negative stress.")
                methods[method] = {"multipliers": sorted(set(float(item) for item in multipliers)), "survival_metric": str(settings.get("survival_metric", "net_cagr_nonnegative"))}
        specification = StrategyRunSpec.from_dict(manifest.canonical_specification)
        plan = {"schema_version": ROBUSTNESS_PLAN_VERSION, "base_strategy_run_id": base_id, "economic_specification_identity": manifest.strategy_run_id, "source_data_snapshot_identity": specification.data_snapshot_hash, "source_data_snapshot_hash": specification.data_snapshot_hash, "benchmark_identity": specification.benchmark.get("option_id", "stored_benchmark"), "benchmark_hash": benchmark_record.content_hash, "economic_artifact_hash": daily_record.content_hash, "calculation_engine_version": specification.engine_version, "robustness_engine_version": ROBUSTNESS_ENGINE_VERSION, "catalog_hash": self.catalog["catalog_hash"], "methods": methods, "seed": int(request.get("seed", 0)), "evaluation_units": int(request.get("evaluation_units", 0)), "policy_version": self.policy.document["schema_version"], "policy_hash": self.policy.policy_hash, "source_commit": self.source_commit, "created_timestamp": self.clock()}
        plan["plan_hash"] = content_hash({key: value for key, value in plan.items() if key not in {"created_timestamp", "plan_hash"}}); plan["robustness_plan_id"] = deterministic_id("robustness_plan", {"plan_hash": plan["plan_hash"]}); plan["estimate"] = estimate_work(plan, self.policy); return canonical_data(plan)

    def confirm(self, plan: Mapping[str, Any], *, confirmation_id: str) -> Mapping[str, Any]:
        if plan["estimate"]["hard_limit_exceeded"]: raise RobustnessError("robustness_hard_limit_exceeded", "Hard policy limits cannot be confirmed.")
        confirmation = {"schema_version": "robustness_confirmation_v1", "confirmation_id": confirmation_id, "plan_hash": plan["plan_hash"], "estimate_hash": plan["estimate"]["estimate_hash"], "policy_hash": self.policy.policy_hash, "created_timestamp": self.clock(), "expires_timestamp": (datetime.fromisoformat(self.clock().replace("Z", "+00:00")) + timedelta(seconds=int(self.policy.document["confirmation_ttl_seconds"]))).isoformat().replace("+00:00", "Z")}
        self._write("confirmations", confirmation_id, confirmation); return confirmation

    def create_plan(self, request: Mapping[str, Any], *, confirmation_id: str | None = None) -> Mapping[str, Any]:
        plan = self.normalize(request)
        if plan["estimate"]["hard_limit_exceeded"]: raise RobustnessError("robustness_hard_limit_exceeded", "Robustness plan exceeds hard policy limit.")
        if plan["estimate"]["confirmation_required"]:
            if not confirmation_id: raise RobustnessError("robustness_confirmation_required", "Confirmation is required for this plan.")
            confirmation = self._load("confirmations", confirmation_id)
            if confirmation.get("plan_hash") != plan["plan_hash"] or confirmation.get("estimate_hash") != plan["estimate"]["estimate_hash"] or confirmation.get("policy_hash") != self.policy.policy_hash: raise RobustnessError("robustness_confirmation_stale", "Confirmation does not bind this plan.")
        self._write("plans", plan["robustness_plan_id"], plan); return plan

    def start(self, plan_id: str) -> Mapping[str, Any]:
        plan = self._load("plans", plan_id); attempt_id = deterministic_id("robustness_attempt", {"plan_hash": plan["plan_hash"]})
        if self._path("attempts", attempt_id).exists(): return self._load("attempts", attempt_id)
        scenarios = []
        for method, settings in plan["methods"].items():
            items = settings.get("folds") or settings.get("years") or settings.get("multipliers") or [settings]
            for ordinal, setting in enumerate(items, 1):
                scenario_id = deterministic_id("robustness_scenario", {"plan_hash": plan["plan_hash"], "method": method, "ordinal": ordinal, "settings": setting})
                scenarios.append({"schema_version": ROBUSTNESS_SCENARIO_VERSION, "scenario_id": scenario_id, "attempt_id": attempt_id, "method": method, "ordinal": ordinal, "base_strategy_run_id": plan["base_strategy_run_id"], "scenario_settings": setting, "scenario_settings_hash": content_hash(setting), "seed": settings.get("seed", plan["seed"]), "state": "pending", "created_timestamp": self.clock(), "worker_ownership": {"mode": "single_local_foundation_6_manager"}, "failure_code": None, "failure_message": None, "artifact_references": [], "provenance": {"plan_hash": plan["plan_hash"], "economic_artifact_hash": plan["economic_artifact_hash"]}, "reuse_source": None, "incomplete_reason": None})
        attempt = {"schema_version": ROBUSTNESS_ATTEMPT_VERSION, "robustness_attempt_id": attempt_id, "robustness_plan_id": plan_id, "plan_hash": plan["plan_hash"], "created_timestamp": self.clock(), "scenarios": scenarios, "status": "pending"}; self._write("attempts", attempt_id, attempt); return attempt

    def _save_attempt(self, attempt: Mapping[str, Any]) -> None:
        path = self._path("attempts", str(attempt["robustness_attempt_id"])); path.unlink(missing_ok=True); _atomic(path, attempt)

    def execute(self, attempt_id: str) -> Mapping[str, Any]:
        attempt = dict(self._load("attempts", attempt_id)); plan = self._load("plans", str(attempt["robustness_plan_id"])); daily = self.store.load_artifact_payload(plan["base_strategy_run_id"], "daily_portfolio_curve"); benchmark = self.store.load_artifact_payload(plan["base_strategy_run_id"], "benchmark_daily_portfolio_curve"); dates = _dates(daily); scenarios = []
        for source in attempt["scenarios"]:
            scenario = dict(source)
            if scenario["state"] in _TERMINAL: scenarios.append(scenario); continue
            scenario["state"], scenario["started_timestamp"] = "running", self.clock()
            try:
                method, setting = scenario["method"], scenario["scenario_settings"]
                if method == "walk_forward_fixed_v1":
                    start, end = setting["test_range"]["start"], setting["test_range"]["end"]; values = [float(row["daily_return"]) for row in daily["rows"] if start <= row["economic_date"] <= end]; result = {"metric": statistics.fmean(values) if values else None, "range": setting["test_range"], "incomplete": setting["incomplete"]}
                    scenario["state"] = "incomplete" if setting["incomplete"] else "succeeded"
                elif method == "leave_one_year_out_v1":
                    year = int(setting["year"])
                    if not setting["eligible"]: result = {"year": year, "incomplete": True, "reason": setting["incomplete_reason"]}; scenario["state"] = "incomplete"
                    else:
                        values = [float(row["daily_return"]) for row in daily["rows"] if int(row["economic_date"][:4]) != year]
                        if not values:
                            result = {"year": year, "incomplete": True, "reason": "removed_year_leaves_no_observations"}; scenario["state"] = "incomplete"
                        else:
                            result = {"year": year, "metric": statistics.fmean(values), "partial_year": setting["partial_year"], "incomplete": False}; scenario["state"] = "succeeded"
                elif method == "paired_moving_block_bootstrap_v1":
                    result = paired_block_bootstrap(aligned_paired_returns(daily, benchmark), seed=int(scenario["seed"]), sample_count=int(plan["methods"][method]["sample_count"]), block_length=int(plan["methods"][method]["block_length"]), confidence_level=float(plan["methods"][method]["confidence_level"])); scenario["state"] = "succeeded"
                else:
                    if self.cost_stress_runner is None: raise RobustnessError("robustness_method_unsupported", "Cost stress requires the registered canonical economic runner.")
                    result = dict(self.cost_stress_runner(plan["base_strategy_run_id"], float(setting))); result["multiplier"] = float(setting); scenario["state"] = "succeeded"
                result.update({"schema_version": ROBUSTNESS_RESULT_VERSION, "scenario_id": scenario["scenario_id"], "method": method, "produced_timestamp": self.clock()}); scenario["artifact_references"] = [{"result_hash": content_hash(result)}]; self._write("results", scenario["scenario_id"], result)
            except RobustnessError as error:
                scenario["state"], scenario["failure_code"], scenario["failure_message"] = "failed", error.code, error.diagnostic_en
            scenario["completed_timestamp"] = self.clock(); scenarios.append(scenario)
        attempt["scenarios"] = scenarios; attempt["status"] = "completed" if all(item["state"] in _TERMINAL for item in scenarios) else "running"; self._save_attempt(attempt); return attempt

    def evidence(self, plan_id: str) -> Mapping[str, Any]:
        plan = self._load("plans", plan_id); attempt = self.start(plan_id); attempt = self.execute(attempt["robustness_attempt_id"])
        by_method: dict[str, list[Mapping[str, Any]]] = {}
        for scenario in attempt["scenarios"]:
            try: result = self._load("results", scenario["scenario_id"])
            except RobustnessError: result = {"incomplete": True, "reason": scenario.get("failure_code") or scenario.get("incomplete_reason")}
            by_method.setdefault(scenario["method"], []).append({"scenario": scenario, "result": result})
        wf = by_method.get("walk_forward_fixed_v1", []); loyo = by_method.get("leave_one_year_out_v1", []); bootstrap = by_method.get("paired_moving_block_bootstrap_v1", []); cost = by_method.get("canonical_cost_stress_v1", [])
        passed_wf = [float(item["result"]["metric"]) for item in wf if item["scenario"]["state"] == "succeeded" and item["result"].get("metric") is not None]; loyo_values = [float(item["result"]["metric"]) for item in loyo if item["scenario"]["state"] == "succeeded"]
        full_metric = statistics.fmean(float(row["daily_return"]) for row in self.store.load_artifact_payload(plan["base_strategy_run_id"], "daily_portfolio_curve")["rows"])
        boot = bootstrap[0]["result"] if bootstrap and bootstrap[0]["scenario"]["state"] == "succeeded" else {}
        survival = None if not cost else float(all(bool(item["result"].get("survives", False)) for item in cost if item["scenario"]["state"] == "succeeded"))
        summary = {"schema_version": ROBUSTNESS_SUMMARY_VERSION, "compatibility_schema_version": "robustness_summary_v1", "base_strategy_run_id": plan["base_strategy_run_id"], "plan_id": plan_id, "plan_hash": plan["plan_hash"], "attempt_id": attempt["robustness_attempt_id"], "provenance": {"economic_artifact_hash": plan["economic_artifact_hash"], "benchmark_hash": plan["benchmark_hash"], "robustness_engine_version": ROBUSTNESS_ENGINE_VERSION, "source_commit": self.source_commit}, "scenario_results": by_method, "walk_forward": {"fold_count": len(wf), "eligible_fold_count": len(wf), "completed_fold_count": len(passed_wf), "passed_fold_count": len([item for item in passed_wf if item >= 0]), "pass_ratio": None if not wf else len([item for item in passed_wf if item >= 0]) / len(wf), "worst_fold": min(passed_wf) if passed_wf else None, "median_fold": statistics.median(passed_wf) if passed_wf else None, "incomplete_fold_count": len([item for item in wf if item["scenario"]["state"] == "incomplete"])}, "loyo": {"evaluated_year_count": len(loyo_values), "reversing_years": sorted(int(item["result"]["year"]) for item in loyo if item["scenario"]["state"] == "succeeded" and float(item["result"]["metric"]) * full_metric < 0), "stability_ratio": None if not loyo else len([item for item in loyo_values if item * full_metric >= 0]) / len(loyo), "incomplete_years": [item["scenario"]["scenario_settings"]["year"] for item in loyo if item["scenario"]["state"] != "succeeded"]}, "bootstrap": boot or None, "cost_stress": {"survival": survival, "scenarios": cost}, "multiple_testing": None, "evidence_hash": ""}
        summary["evidence_hash"] = content_hash({key: value for key, value in summary.items() if key != "evidence_hash"}); self._write("evidence", plan_id, summary); return summary

    def reconcile(self, attempt_id: str) -> Mapping[str, Any]:
        attempt = dict(self._load("attempts", attempt_id)); changed = []
        for scenario in attempt["scenarios"]:
            if scenario["state"] == "running": scenario.update({"state": "blocked", "failure_code": "robustness_scenario_interrupted", "failure_message": "No trustworthy live owner after restart.", "incomplete_reason": "interrupted_running_no_live_owner"}); changed.append(scenario["scenario_id"])
        attempt["status"] = "reconciled"; self._save_attempt(attempt); return {"robustness_attempt_id": attempt_id, "blocked_scenarios": changed}

    def resume(self, attempt_id: str) -> Mapping[str, Any]:
        attempt = dict(self._load("attempts", attempt_id)); resumed = []
        for scenario in attempt["scenarios"]:
            if scenario["state"] in {"pending", "failed", "cancelled", "blocked", "incomplete"}:
                scenario.update({"state": "pending", "retry_of_state": scenario["state"], "failure_code": None, "failure_message": None}); resumed.append(scenario["scenario_id"])
        self._save_attempt(attempt); return {"robustness_attempt_id": attempt_id, "resumed_scenarios": resumed, "bootstrap_resume": "restart_from_seed_new_attempt_state"}
