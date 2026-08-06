"""Focused Research Workspace orchestration and persistence coverage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

from src.trend_v2_foundation.foundation_6 import OptionCatalog, PersistedExecutionManager
from src.trend_v2_foundation.workflow import WorkflowCoordinator


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "config" / "trend_v2" / "strategy_option_catalog_v2.json"


class _Attempt:
    def __init__(self, identity: str, run_id: str, status: str) -> None:
        self.execution_attempt_id = identity
        self.intended_strategy_run_id = run_id
        self.operational_status = SimpleNamespace(value=status)

    def to_dict(self):
        return {"execution_attempt_id": self.execution_attempt_id, "intended_strategy_run_id": self.intended_strategy_run_id, "operational_status": self.operational_status.value, "current_stage": "complete"}


class _Attempts:
    def __init__(self, attempts): self.attempts = {item.execution_attempt_id: item for item in attempts}
    def get(self, identity): return self.attempts[identity]
    def list(self): return list(self.attempts.values())


class _Execution:
    def __init__(self, root: Path, attempts: _Attempts) -> None:
        self.store = SimpleNamespace(root=root, validate_manifest=lambda _run: SimpleNamespace(valid=True))
        self.attempt_repository = attempts
        self.source_commit = "c" * 40
        self.policy = SimpleNamespace(policy_hash="p" * 64)


class WorkspaceWorkflowTests(unittest.TestCase):
    def test_list_and_restart_rebuild_use_persisted_workflow_not_browser_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempts = _Attempts([])
            workflow = WorkflowCoordinator(_Execution(Path(temporary), attempts))
            created = workflow.create({"strategy": "controlled"}, label_ko="연구 절차", idempotency_key="create")
            listing = workflow.list()
            self.assertEqual(listing["items"][0]["workflow_id"], created["workflow_id"])
            rebuilt = WorkflowCoordinator(_Execution(Path(temporary), attempts)).list()
            self.assertEqual(rebuilt, listing)

    def test_completed_attempt_reference_is_resolved_without_restarting_economics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = _Attempt("attempt_a", "strategy_run_a", "completed")
            workflow = WorkflowCoordinator(_Execution(Path(temporary), _Attempts([attempt])))
            created = workflow.create({"strategy": "controlled"}, label_ko="완료 확인", idempotency_key="create")
            workflow._event(created["workflow_id"], "economic_started", {"execution_request_id": "request_a", "execution_attempt_ids": ["attempt_a"], "strategy_run_ids": ["strategy_run_a"]})
            state = workflow.read(created["workflow_id"])
            self.assertEqual(state["references"]["economic_progress"]["status"], "completed")
            self.assertEqual(state["references"]["economic_progress"]["strategy_run_ids"], ["strategy_run_a"])

    def test_profile_re_evaluation_uses_stored_profile_without_economic_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = _Attempt("attempt_a", "strategy_run_a", "completed")
            execution = _Execution(Path(temporary), _Attempts([attempt]))
            profile = SimpleNamespace(evaluation_profile_id="evaluation_profile_new")
            execution.store.get_evaluation_profile = lambda identity: profile if identity == profile.evaluation_profile_id else (_ for _ in ()).throw(KeyError(identity))
            workflow = WorkflowCoordinator(execution)
            created = workflow.create({"strategy": "controlled"}, label_ko="재평가", idempotency_key="create")
            workflow._event(created["workflow_id"], "economic_started", {"execution_request_id": "request_a", "execution_attempt_ids": ["attempt_a"], "strategy_run_ids": ["strategy_run_a"]})
            outcome = SimpleNamespace(evaluation_run=SimpleNamespace(evaluation_run_id="evaluation_run_a"))
            with patch("src.trend_v2_foundation.workflow.calculate_and_evaluate_saved_runs", return_value=outcome) as evaluate:
                state = workflow.evaluate(created["workflow_id"], evaluation_profile_id="evaluation_profile_new", idempotency_key="evaluate")
                replay = workflow.evaluate(created["workflow_id"], evaluation_profile_id="evaluation_profile_new", idempotency_key="evaluate")
            evaluate.assert_called_once_with(execution.store, ["strategy_run_a"], profile, creation_time=ANY)
            self.assertFalse(state["references"]["evaluation"]["economic_backtest_started"])
            self.assertEqual(replay["references"]["evaluation"], state["references"]["evaluation"])

    def test_decision_report_reference_is_an_append_only_workflow_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = _Attempt("attempt_a", "strategy_run_a", "completed")
            workflow = WorkflowCoordinator(_Execution(Path(temporary), _Attempts([attempt])))
            created = workflow.create({"strategy": "controlled"}, label_ko="보고서", idempotency_key="create")
            workflow._event(created["workflow_id"], "economic_started", {"execution_request_id": "request_a", "execution_attempt_ids": ["attempt_a"], "strategy_run_ids": ["strategy_run_a"]})
            workflow._event(created["workflow_id"], "evaluated", {"evaluation_run_id": "evaluation_run_a", "strategy_run_ids": ["strategy_run_a"]})
            state = workflow.record_decision_report(
                created["workflow_id"], decision_report_id="decision_report_a",
                strategy_run_id="strategy_run_a", evaluation_run_id="evaluation_run_a",
            )
            self.assertEqual(state["stage"], "decision_report_ready")
            self.assertEqual(state["references"]["decision_report"]["decision_report_id"], "decision_report_a")


class WorkspaceManagerProjectionTests(unittest.TestCase):
    def test_existing_execution_request_is_projected_without_creating_a_second_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = PersistedExecutionManager(temporary, OptionCatalog.load(CATALOG))
            request = {"execution_request_id": "execution_request_a", "candidate_estimate_hash": "e" * 64, "request_timestamp": "2026-08-06T00:00:00Z", "normalized_construction": {}, "requested_strategy_run_candidates": [{"strategy_run_id": "strategy_run_a"}]}
            projected = manager.track_controlled_request(request, [_Attempt("attempt_a", "strategy_run_a", "queued")])
            self.assertEqual(projected["request_count"], 1)
            self.assertEqual(projected["candidates"][0]["execution_attempt_id"], "attempt_a")
            self.assertEqual(projected["candidates"][0]["state"], "pending")


class WorkspaceUiTests(unittest.TestCase):
    def test_workspace_route_is_persisted_and_links_to_specialist_views(self) -> None:
        source = (ROOT / "src" / "trend_v2_foundation" / "ui_assets" / "app.js").read_text(encoding="utf-8")
        self.assertIn('api("/workflows")', source)
        self.assertIn("workspaceKey", source)
        self.assertIn("경제 실행 및 후보 진행", source)
        self.assertIn('href="#robustness"', source)
        self.assertIn('href="#evaluations"', source)
        self.assertIn('route==="decision-reports"', source)
        self.assertIn('workspace-decision-report', source)
        self.assertIn("개인화된 투자 조언이 아닙니다.", source)


if __name__ == "__main__":
    unittest.main()
