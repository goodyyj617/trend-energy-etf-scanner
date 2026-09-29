"""Focused Foundation 11A robustness confirmation and workspace lifecycle coverage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from src.trend_v2_foundation.api import ReadOnlyTrendApi
from src.trend_v2_foundation.canonical import content_hash
from src.trend_v2_foundation.contracts import StrategyRunSpec
from src.trend_v2_foundation.execution import AttemptOperationalStatus
from src.trend_v2_foundation.robustness import (
    RobustnessError,
    RobustnessExecutionService,
    RobustnessPolicy,
    load_robustness_catalog,
)
from src.trend_v2_foundation.workflow import WorkflowCoordinator, WorkflowError


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "trend_v2" / "robustness_execution_policy_v1.json"
CATALOG = ROOT / "config" / "trend_v2" / "robustness_option_catalog_v1.json"


class _Clock:
    def __init__(self) -> None:
        self.tick = 0

    def __call__(self) -> str:
        self.tick += 1
        return f"2026-09-29T00:{self.tick // 60:02d}:{self.tick % 60:02d}Z"


class _Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        dates = [f"2024-01-{day:02d}" for day in range(2, 10)]
        self.daily = {"rows": [{"economic_date": day, "daily_return": value} for day, value in zip(dates, [.01, .01, -.005, .004, .002, -.003, .006, .001])]}
        self.benchmark = {"rows": [{"economic_date": day, "daily_return": value} for day, value in zip(dates, [.005, .003, -.004, .002, .001, -.002, .003, 0.0])]}
        self.spec = StrategyRunSpec(
            data_snapshot_hash="b" * 64,
            economic_date_range={"start": dates[0], "end": dates[-1]},
            universe_specification={"id": "u"}, benchmark={"option_id": "spy"},
            trend_filter={"id": "t"}, signal={"id": "s"}, entry_rule={"id": "e"},
            initial_stop={"id": "i"}, trailing_exit={"id": "x"},
            position_sizing={"id": "p"}, portfolio_constraints={"id": "c"},
            transaction_costs={"id": "cost"}, slippage={"id": "slip"},
            engine_version="engine",
        )

    def get_strategy_run_manifest(self, run_id: str):
        if run_id != self.spec.strategy_run_id:
            raise KeyError(run_id)
        return SimpleNamespace(strategy_run_id=run_id, canonical_specification=self.spec.to_dict())

    def load_artifact_payload(self, run_id: str, key: str):
        self.get_strategy_run_manifest(run_id)
        return {"daily_portfolio_curve": self.daily, "benchmark_daily_portfolio_curve": self.benchmark}[key]

    def get_strategy_artifact_record(self, run_id: str, key: str):
        return SimpleNamespace(content_hash=content_hash(self.load_artifact_payload(run_id, key)))

    def validate_manifest(self, run_id: str):
        return SimpleNamespace(valid=run_id == self.spec.strategy_run_id)


class _EconomicAttempt:
    def __init__(self, run_id: str) -> None:
        self.execution_attempt_id = "attempt_economic"
        self.intended_strategy_run_id = run_id
        self.operational_status = AttemptOperationalStatus.COMPLETED

    def to_dict(self):
        return {"execution_attempt_id": self.execution_attempt_id, "intended_strategy_run_id": self.intended_strategy_run_id, "operational_status": self.operational_status.value}


class Foundation11ABackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = _Store(Path(self.temp.name))
        self.clock = _Clock()
        self.service = RobustnessExecutionService(
            self.store, RobustnessPolicy.load(POLICY), load_robustness_catalog(CATALOG),
            source_commit="a" * 40, clock=self.clock,
        )

    def tearDown(self) -> None:
        worker = self.service._worker
        if worker is not None:
            worker.join(timeout=3)
        self.temp.cleanup()

    def request(self) -> dict:
        return {"base_strategy_run_id": self.store.spec.strategy_run_id, "seed": 3,
                "methods": {"paired_moving_block_bootstrap_v1": {"sample_count": 8, "block_length": 2, "confidence_level": .95}}}

    def confirm(self, request: dict) -> dict:
        preview = self.service.normalize(request)
        return self.service.confirm_request(
            request, plan_hash=preview["plan_hash"],
            estimate_hash=preview["estimate"]["estimate_hash"], idempotency_key="confirm-one",
        )

    def api(self) -> ReadOnlyTrendApi:
        return ReadOnlyTrendApi(
            self.store, registry_builder=SimpleNamespace(load_or_rebuild=lambda: None),
            attempt_repository=SimpleNamespace(), robustness_execution_service=self.service,
            profile_studio_service=SimpleNamespace(), decision_report_service=SimpleNamespace(),
        )

    def workflow(self) -> tuple[WorkflowCoordinator, str]:
        attempt = _EconomicAttempt(self.store.spec.strategy_run_id)
        execution = SimpleNamespace(
            store=self.store, source_commit="a" * 40,
            policy=SimpleNamespace(policy_hash="p" * 64),
            attempt_repository=SimpleNamespace(get=lambda identity: attempt if identity == attempt.execution_attempt_id else (_ for _ in ()).throw(KeyError(identity))),
        )
        workflow = WorkflowCoordinator(execution, self.service, clock=self.clock)
        created = workflow.create({"strategy": "controlled"}, label_ko="강건성 연구", idempotency_key="create")
        workflow._event(created["workflow_id"], "economic_started", {"execution_request_id": "request_economic", "execution_attempt_ids": [attempt.execution_attempt_id]})
        return workflow, created["workflow_id"]

    def test_preview_hashes_confirmation_and_key_binding(self) -> None:
        request = self.request()
        preview = self.service.normalize(request)
        with self.assertRaises(RobustnessError) as stale:
            self.service.confirm_request(request, plan_hash="stale", estimate_hash=preview["estimate"]["estimate_hash"], idempotency_key="confirm-one")
        self.assertEqual(stale.exception.code, "robustness_confirmation_stale")
        confirmation = self.confirm(request)
        self.assertEqual(self.confirm(request), confirmation)
        changed = {**request, "seed": 4}
        changed_preview = self.service.normalize(changed)
        with self.assertRaises(RobustnessError) as conflict:
            self.service.confirm_request(changed, plan_hash=changed_preview["plan_hash"], estimate_hash=changed_preview["estimate"]["estimate_hash"], idempotency_key="confirm-one")
        self.assertEqual(conflict.exception.code, "robustness_confirmation_stale")
        plan = self.service.create_plan(request, confirmation_id=confirmation["confirmation_id"])
        self.assertEqual(self.service.create_plan(request, confirmation_id=confirmation["confirmation_id"]), plan)

    def test_api_get_is_read_only_and_start_is_idempotent_background_work(self) -> None:
        api = self.api()
        request = self.request()
        preview = self.service.normalize(request)
        confirmed = api.dispatch("POST", "/api/v1/robustness/confirm", headers={"Idempotency-Key": "confirm-api"}, body={"request": request, "plan_hash": preview["plan_hash"], "estimate_hash": preview["estimate"]["estimate_hash"]})
        self.assertEqual(confirmed.status_code, 201)
        planned = api.dispatch("POST", "/api/v1/robustness/plans", body={"request": request, "confirmation_id": confirmed.body["confirmation_id"]})
        self.assertEqual(planned.status_code, 201)
        plan_id = planned.body["robustness_plan_id"]
        self.assertEqual(api.dispatch("GET", f"/api/v1/robustness/plans/{plan_id}/evidence").status_code, 409)
        self.assertFalse((self.service.root / "attempts").exists())

        original_evidence = self.service.evidence
        entered, release = Event(), Event()

        def delayed_evidence(identity: str):
            entered.set()
            if not release.wait(3):
                raise AssertionError("background worker was not released")
            return original_evidence(identity)

        with patch.object(self.service, "evidence", side_effect=delayed_evidence) as execute:
            try:
                first = api.dispatch("POST", f"/api/v1/robustness/plans/{plan_id}/start", body={})
                self.assertEqual(first.status_code, 202)
                self.assertTrue(entered.wait(2))
                replay = api.dispatch("POST", f"/api/v1/robustness/plans/{plan_id}/start", body={})
                self.assertEqual(replay.status_code, 202)
                self.assertEqual(replay.body["robustness_attempt_id"], first.body["robustness_attempt_id"])
                self.assertEqual(api.dispatch("GET", f"/api/v1/robustness/plans/{plan_id}/evidence").status_code, 409)
                self.assertEqual(execute.call_count, 1)
            finally:
                worker = self.service._worker
                release.set()
                if worker is not None:
                    worker.join(timeout=3)
        self.assertEqual(api.dispatch("GET", f"/api/v1/robustness/plans/{plan_id}/evidence").status_code, 200)
        self.assertEqual(api.dispatch("POST", f"/api/v1/robustness/plans/{plan_id}/start", body={}).body["status"], "completed")

    def test_workflow_owns_candidate_and_retains_evaluation_reference(self) -> None:
        workflow, workflow_id = self.workflow()
        workflow._event(workflow_id, "evaluated", {"evaluation_run_id": "evaluation_prior"})
        request = self.request()
        confirmation = self.confirm(request)
        with self.assertRaises(WorkflowError) as foreign:
            workflow.configure_robustness(workflow_id, {**request, "base_strategy_run_id": "strategy_run_foreign"}, confirmation_id=confirmation["confirmation_id"])
        self.assertEqual(foreign.exception.code, "workflow_robustness_candidate_invalid")
        with self.assertRaises(RobustnessError) as missing:
            workflow.configure_robustness(workflow_id, request)
        self.assertEqual(missing.exception.code, "robustness_confirmation_required")
        state = workflow.configure_robustness(workflow_id, request, confirmation_id=confirmation["confirmation_id"])
        self.assertEqual(state["stage"], "robustness_configuration_required")
        with patch("src.trend_v2_foundation.workflow.calculate_and_evaluate_saved_runs", side_effect=AssertionError("evaluation must remain explicit")):
            started = workflow.start_robustness(workflow_id)
            replay = workflow.start_robustness(workflow_id)
            self.assertEqual(replay["references"]["robustness"]["robustness_attempt_id"], started["references"]["robustness"]["robustness_attempt_id"])
            worker = self.service._worker
            if worker is not None:
                worker.join(timeout=3)
        complete = workflow.read(workflow_id)
        self.assertEqual(complete["references"]["robustness_progress"]["status"], "completed", complete["references"]["robustness_progress"])
        self.assertEqual(complete["stage"], "robustness_completed")
        self.assertEqual(complete["references"]["evaluation"]["evaluation_run_id"], "evaluation_prior")
        self.assertEqual(len([item for item in complete["events"] if item["action"] == "robustness_started"]), 1)

    def test_explicit_workflow_resume_retries_failed_worker_once(self) -> None:
        workflow, workflow_id = self.workflow()
        request = self.request()
        confirmation = self.confirm(request)
        workflow.configure_robustness(workflow_id, request, confirmation_id=confirmation["confirmation_id"])
        original_evidence = self.service.evidence
        calls = 0

        def fail_once(identity: str):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("deliberate worker failure")
            return original_evidence(identity)

        with patch.object(self.service, "evidence", side_effect=fail_once):
            workflow.start_robustness(workflow_id)
            worker = self.service._worker
            if worker is not None:
                worker.join(timeout=3)
            self.assertEqual(workflow.read(workflow_id)["references"]["robustness_progress"]["status"], "failed")
            workflow.resume(workflow_id, idempotency_key="retry-once")
            worker = self.service._worker
            if worker is not None:
                worker.join(timeout=3)
            self.assertEqual(workflow.read(workflow_id)["references"]["robustness_progress"]["status"], "completed")
            workflow.resume(workflow_id, idempotency_key="retry-once")
        self.assertEqual(calls, 2)
