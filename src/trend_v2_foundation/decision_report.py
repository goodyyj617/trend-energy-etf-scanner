"""Immutable, reference-only Decision Reports over existing stored evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifact_schemas import validate_robustness_summary_v2
from .canonical import canonical_data, content_hash
from .contracts import DecisionReport, EvaluationRun
from .result_store import LocalResultStore


DECISION_REPORT_TEMPLATE_VERSION = "trend_v2_decision_report_template_v1"
DECISION_REPORT_SERVICE_VERSION = "trend_v2_decision_report_service_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class DecisionReportError(ValueError):
    _MESSAGES = {
        "decision_report_invalid_request": "결정 보고서 요청이 올바르지 않습니다.",
        "decision_report_reference_missing": "결정 보고서에 필요한 저장 근거가 없습니다.",
        "decision_report_reference_incompatible": "선택한 근거가 서로 호환되지 않습니다.",
        "decision_report_reference_stale": "결정 보고서의 저장 근거가 현재 상태와 일치하지 않습니다.",
        "decision_report_idempotency_conflict": "같은 중복 방지 키가 다른 보고서 요청에 사용되었습니다.",
    }

    def __init__(self, code: str, diagnostic_en: str, *, object_identity: str | None = None) -> None:
        super().__init__(diagnostic_en)
        self.code = code
        self.diagnostic_en = diagnostic_en
        self.object_identity = object_identity
        self.message_ko = self._MESSAGES.get(code, "결정 보고서를 처리할 수 없습니다.")

    def to_dict(self, request_id: str = "local") -> dict[str, Any]:
        return {
            "code": self.code,
            "message_ko": self.message_ko,
            "diagnostic_en": self.diagnostic_en,
            "object_identity": self.object_identity,
            "recoverable": self.code != "decision_report_reference_stale",
            "request_id": request_id,
        }


class DecisionReportService:
    """Creates reports from references only; never invokes economic execution."""

    def __init__(
        self,
        store: LocalResultStore,
        *,
        robustness_service: Any | None = None,
        source_commit: str = "unknown",
        clock: Callable[[], str] = _now,
    ) -> None:
        self.store = store
        self.robustness_service = robustness_service
        self.source_commit = source_commit
        self.clock = clock
        self.idempotency_dir = store.root / "decision_report_v1" / "idempotency"
        self.idempotency_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _identifier(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise DecisionReportError("decision_report_invalid_request", f"{field} is required.")
        return value

    def _artifact_hash(self, strategy_run_id: str, artifact_key: str) -> str | None:
        try:
            return self.store.get_strategy_artifact_record(strategy_run_id, artifact_key).content_hash
        except KeyError:
            return None

    def _evidence_references(
        self,
        strategy_run_id: str,
        evaluation: EvaluationRun,
        candidate: Any,
        *,
        robustness_plan_id: str | None,
    ) -> Mapping[str, Any]:
        manifest = self.store.get_strategy_run_manifest(strategy_run_id)
        behavior_hash = self._artifact_hash(strategy_run_id, "behavior_metadata")
        references: dict[str, Any] = {
            "strategy_run": {
                "strategy_run_id": strategy_run_id,
                "manifest_hash": content_hash(manifest.to_dict()),
                "summary_metrics_hash": self._artifact_hash(strategy_run_id, "summary_metrics"),
                "behavior_metadata_hash": behavior_hash,
            },
            "evaluation": {
                "evaluation_run_id": evaluation.evaluation_run_id,
                "evaluation_run_hash": content_hash(evaluation.to_dict()),
                "candidate_hash": content_hash(canonical_data(candidate)),
                "behavior_comparison_hash": content_hash(
                    {
                        "candidate_behavior": candidate.behavior_deduplication_metadata,
                        "pairwise": evaluation.behavior_pairwise_diagnostics,
                    }
                ),
            },
            "profile": {
                "evaluation_profile_id": evaluation.evaluation_profile_id,
                "profile_hash": evaluation.profile_hash,
            },
            "robustness": {"status": "not_attached"},
        }
        if robustness_plan_id is not None:
            if self.robustness_service is None:
                raise DecisionReportError(
                    "decision_report_reference_incompatible",
                    "Robustness service is unavailable.",
                    object_identity=robustness_plan_id,
                )
            evidence = self.robustness_service.read_evidence(robustness_plan_id)
            if evidence.get("base_strategy_run_id") != strategy_run_id:
                raise DecisionReportError(
                    "decision_report_reference_incompatible",
                    "Robustness evidence targets another StrategyRun.",
                    object_identity=robustness_plan_id,
                )
            references["robustness"] = {
                "status": "attached",
                "robustness_plan_id": robustness_plan_id,
                "plan_hash": evidence["plan_hash"],
                "attempt_id": evidence["attempt_id"],
                "evidence_hash": evidence["evidence_hash"],
            }
        return references

    def _resolve(self, payload: Mapping[str, Any]) -> tuple[str, EvaluationRun, Any, Mapping[str, Any]]:
        if set(payload).difference({"strategy_run_id", "evaluation_run_id", "robustness_plan_id"}):
            raise DecisionReportError("decision_report_invalid_request", "Unsupported DecisionReport request field.")
        strategy_run_id = self._identifier(payload.get("strategy_run_id"), "strategy_run_id")
        evaluation_run_id = self._identifier(payload.get("evaluation_run_id"), "evaluation_run_id")
        plan_id = payload.get("robustness_plan_id")
        if plan_id is not None:
            plan_id = self._identifier(plan_id, "robustness_plan_id")
        validation = self.store.validate_manifest(strategy_run_id)
        if not validation.valid:
            raise DecisionReportError(
                "decision_report_reference_stale",
                "StrategyRun manifest is not valid.",
                object_identity=strategy_run_id,
            )
        try:
            evaluation = self.store.get_evaluation_run(evaluation_run_id)
            profile = self.store.get_evaluation_profile(evaluation.evaluation_profile_id)
        except KeyError as error:
            raise DecisionReportError(
                "decision_report_reference_missing",
                "Evaluation evidence is missing.",
                object_identity=evaluation_run_id,
            ) from error
        if evaluation.profile_hash != profile.profile_hash:
            raise DecisionReportError(
                "decision_report_reference_stale",
                "EvaluationRun profile hash differs from its stored profile.",
                object_identity=evaluation_run_id,
            )
        matches = [item for item in evaluation.results if item.strategy_run_id == strategy_run_id]
        if len(matches) != 1:
            raise DecisionReportError(
                "decision_report_reference_incompatible",
                "StrategyRun is not a member of EvaluationRun.",
                object_identity=strategy_run_id,
            )
        return strategy_run_id, evaluation, matches[0], self._evidence_references(
            strategy_run_id, evaluation, matches[0], robustness_plan_id=plan_id
        )

    def _binding_path(self, key: str) -> Path:
        return self.idempotency_dir / f"{content_hash({'key': key})}.json"

    def create(self, payload: Mapping[str, Any], *, idempotency_key: str) -> tuple[Mapping[str, Any], bool]:
        if not isinstance(payload, Mapping) or not isinstance(idempotency_key, str) or not idempotency_key:
            raise DecisionReportError("decision_report_invalid_request", "A bounded idempotency key is required.")
        request = canonical_data(payload)
        binding = self._binding_path(idempotency_key)
        if binding.exists():
            saved = json.loads(binding.read_text(encoding="utf-8"))
            if saved.get("request_hash") != content_hash(request):
                raise DecisionReportError("decision_report_idempotency_conflict", "Idempotency key conflicts.")
            report = self.store.get_decision_report(saved["decision_report_id"])
            return self.detail(report.decision_report_id), True
        strategy_run_id, evaluation, _candidate, references = self._resolve(request)
        report = DecisionReport(
            strategy_run_id=strategy_run_id,
            evaluation_run_id=evaluation.evaluation_run_id,
            evaluation_profile_id=evaluation.evaluation_profile_id,
            profile_hash=evaluation.profile_hash,
            evidence_references=references,
            creation_time=self.clock(),
            template_version=DECISION_REPORT_TEMPLATE_VERSION,
            source_commit=self.source_commit,
        )
        try:
            report = self.store.get_decision_report(report.decision_report_id)
            replayed = True
        except KeyError:
            self.store.save_decision_report(report)
            replayed = False
        binding.write_text(
            json.dumps(
                {
                    "request_hash": content_hash(request),
                    "decision_report_id": report.decision_report_id,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return self.detail(report.decision_report_id), replayed

    def _current_candidate(self, report: DecisionReport) -> tuple[EvaluationRun, Any, Any]:
        evaluation = self.store.get_evaluation_run(report.evaluation_run_id)
        profile = self.store.get_evaluation_profile(report.evaluation_profile_id)
        candidate = next(
            item for item in evaluation.results if item.strategy_run_id == report.strategy_run_id
        )
        return evaluation, profile, candidate

    def _evidence_state(self, report: DecisionReport, evaluation: EvaluationRun, profile: Any, candidate: Any) -> tuple[str, list[Mapping[str, str]]]:
        references = report.evidence_references
        issues: list[Mapping[str, str]] = []
        try:
            manifest = self.store.get_strategy_run_manifest(report.strategy_run_id)
            if not self.store.validate_manifest(manifest).valid:
                issues.append({"state": "stale", "code": "strategy_manifest_invalid"})
            expected_manifest_hash = references["strategy_run"]["manifest_hash"]
            if content_hash(manifest.to_dict()) != expected_manifest_hash:
                issues.append({"state": "stale", "code": "strategy_manifest_hash_mismatch"})
        except (KeyError, ValueError):
            issues.append({"state": "missing", "code": "strategy_manifest_missing"})
        if evaluation.profile_hash != report.profile_hash or profile.profile_hash != report.profile_hash:
            issues.append({"state": "stale", "code": "evaluation_profile_hash_mismatch"})
        if content_hash(evaluation.to_dict()) != references["evaluation"]["evaluation_run_hash"]:
            issues.append({"state": "stale", "code": "evaluation_run_hash_mismatch"})
        if content_hash(canonical_data(candidate)) != references["evaluation"]["candidate_hash"]:
            issues.append({"state": "stale", "code": "candidate_hash_mismatch"})
        behavior_enabled = bool(profile.behavior_deduplication.get("enabled", False))
        behavior = candidate.behavior_deduplication_metadata
        if behavior_enabled and behavior.get("status") == "not_available":
            issues.append({"state": "incomplete", "code": "behavior_evidence_missing"})
        robustness = references["robustness"]
        if robustness.get("status") == "attached":
            try:
                evidence = self.robustness_service.read_evidence(str(robustness["robustness_plan_id"])) if self.robustness_service else None
                if evidence is None:
                    issues.append({"state": "missing", "code": "robustness_service_unavailable"})
                elif evidence.get("evidence_hash") != robustness.get("evidence_hash"):
                    issues.append({"state": "stale", "code": "robustness_evidence_hash_mismatch"})
                elif evidence.get("base_strategy_run_id") != report.strategy_run_id:
                    issues.append({"state": "incompatible", "code": "robustness_target_mismatch"})
                elif any(
                    item.get("scenario", {}).get("state") not in {"succeeded", "reused", "skipped"}
                    for rows in evidence.get("scenario_results", {}).values()
                    for item in rows
                ):
                    issues.append({"state": "incomplete", "code": "robustness_evidence_incomplete"})
            except Exception:
                issues.append({"state": "missing", "code": "robustness_evidence_missing"})
        elif not profile.robustness_vetoes:
            issues.append({"state": "warning", "code": "robustness_evidence_not_attached"})
        states = {item["state"] for item in issues}
        for state in ("stale", "incompatible", "missing", "incomplete", "warning"):
            if state in states:
                return state, issues
        return "ready", issues

    def detail(self, decision_report_id: str) -> Mapping[str, Any]:
        try:
            report = self.store.get_decision_report(decision_report_id)
            evaluation, profile, candidate = self._current_candidate(report)
        except (KeyError, StopIteration, ValueError) as error:
            raise DecisionReportError(
                "decision_report_reference_missing",
                "DecisionReport reference is missing.",
                object_identity=decision_report_id,
            ) from error
        evidence_state, issues = self._evidence_state(report, evaluation, profile, candidate)
        labels = set(candidate.final_labels)
        if evidence_state in {"missing", "incomplete", "stale", "incompatible"}:
            decision_state = "incomplete"
        elif "constraint_pareto_selected" in labels:
            decision_state = "pass"
        else:
            decision_state = "fail"
        metric_keys = ("cagr", "cagr_spy_ratio", "maximum_drawdown")
        metrics = {
            key: candidate.raw_metrics.get(key)
            for key in metric_keys
            if key in candidate.raw_metrics
        }
        return {
            "decision_report": report.to_dict(),
            "decision_state": decision_state,
            "evidence_state": evidence_state,
            "issues": issues,
            "economic_metrics": metrics,
            "evaluation": {
                "final_labels": list(candidate.final_labels),
                "mandatory_gates": canonical_data(candidate.mandatory_gate_results),
                "pareto_member": candidate.pareto_member,
                "dominated_by": list(candidate.dominated_by),
                "robustness_vetoes": canonical_data(candidate.robustness_results),
                "behavior": canonical_data(candidate.behavior_deduplication_metadata),
                "unavailable_reasons": canonical_data(candidate.unavailable_reasons),
            },
            "evidence_links": {
                "strategy_run": f"/api/v1/runs/{report.strategy_run_id}/manifest",
                "evaluation_run": f"/api/v1/evaluation-runs/{report.evaluation_run_id}/outputs",
                "behavior": f"/api/v1/evaluation-runs/{report.evaluation_run_id}/behavior",
                "profile": f"/api/v1/evaluation-profiles/{report.evaluation_profile_id}",
                "robustness": (
                    None
                    if report.evidence_references["robustness"].get("status") != "attached"
                    else f"/api/v1/robustness/plans/{report.evidence_references['robustness']['robustness_plan_id']}/stored-evidence"
                ),
            },
        }

    def list(self, *, strategy_run_id: str | None = None, evaluation_run_id: str | None = None) -> list[Mapping[str, Any]]:
        reports = []
        for report_id in self.store.decision_report_history():
            report = self.store.get_decision_report(report_id)
            if strategy_run_id and report.strategy_run_id != strategy_run_id:
                continue
            if evaluation_run_id and report.evaluation_run_id != evaluation_run_id:
                continue
            detail = self.detail(report_id)
            reports.append(
                {
                    "decision_report_id": report_id,
                    "strategy_run_id": report.strategy_run_id,
                    "evaluation_run_id": report.evaluation_run_id,
                    "creation_time": report.creation_time,
                    "decision_state": detail["decision_state"],
                    "evidence_state": detail["evidence_state"],
                }
            )
        return sorted(reports, key=lambda item: (item["creation_time"], item["decision_report_id"]), reverse=True)
