"""Focused Foundation 10D quick-workflow and laptop-bound UI coverage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.trend_v2_foundation import ArtifactRetentionPolicy, LocalResultStore
from src.trend_v2_foundation.api import ReadOnlyTrendApi
from src.trend_v2_foundation.workflow import WorkflowCoordinator


ROOT = Path(__file__).resolve().parents[1]


class _Attempt:
    def __init__(
        self,
        identity: str,
        run_id: str,
        status: str = "completed",
        artifact_references: list[dict[str, str]] | None = None,
    ) -> None:
        self.execution_attempt_id = identity
        self.intended_strategy_run_id = run_id
        self.operational_status = SimpleNamespace(value=status)
        self.artifact_references = artifact_references or []

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_attempt_id": self.execution_attempt_id,
            "intended_strategy_run_id": self.intended_strategy_run_id,
            "operational_status": self.operational_status.value,
            "current_stage": "complete",
            "artifact_references": self.artifact_references,
        }


class _Attempts:
    def __init__(self, attempts: list[_Attempt]) -> None:
        self.attempts = {item.execution_attempt_id: item for item in attempts}

    def get(self, identity: str) -> _Attempt:
        return self.attempts[identity]

    def list(self) -> list[_Attempt]:
        return list(self.attempts.values())


class _Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.profiles: dict[str, object] = {}
        self.evaluations: dict[str, object] = {}

    @staticmethod
    def validate_manifest(_run_id: str) -> object:
        return SimpleNamespace(valid=True)

    def get_evaluation_profile(self, identity: str) -> object:
        return self.profiles[identity]

    def evaluation_history(self) -> list[str]:
        return list(self.evaluations)

    def get_evaluation_run(self, identity: str) -> object:
        return self.evaluations[identity]


class _Execution:
    def __init__(self, store: object, attempts: _Attempts) -> None:
        self.store = store
        self.attempt_repository = attempts
        self.source_commit = "c" * 40
        self.policy = SimpleNamespace(policy_hash="p" * 64)


class Foundation10DWorkflowTests(unittest.TestCase):
    def test_identical_event_replay_reuses_original_timestamp_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = WorkflowCoordinator(_Execution(_Store(Path(temporary)), _Attempts([])))
            created = workflow.create({"strategy": "controlled"}, label_ko="빠른 연구", idempotency_key="create")
            workflow.clock = lambda: "2026-09-23T01:00:00Z"
            first = workflow._event(created["workflow_id"], "normalized", {"hash": "same"})
            workflow.clock = lambda: "2026-09-23T02:00:00Z"
            replay = workflow._event(created["workflow_id"], "normalized", {"hash": "same"})
            self.assertEqual(replay, first)
            self.assertEqual(len(workflow._events(created["workflow_id"])), 1)

    def test_latest_event_uses_timestamp_instead_of_hash_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = WorkflowCoordinator(_Execution(_Store(Path(temporary)), _Attempts([])))
            created = workflow.create({"strategy": "controlled"}, label_ko="최신 평가", idempotency_key="create")
            workflow.clock = lambda: "2026-09-23T01:00:00.500000Z"
            workflow._event(created["workflow_id"], "evaluated", {"evaluation_run_id": "evaluation_later"})
            workflow.clock = lambda: "2026-09-23T01:00:00Z"
            workflow._event(created["workflow_id"], "evaluated", {"evaluation_run_id": "evaluation_earlier"})
            state = workflow.read(created["workflow_id"])
            self.assertEqual(state["references"]["evaluation"]["evaluation_run_id"], "evaluation_later")

    def test_completed_execution_reuses_its_existing_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = _Store(Path(temporary))
            profile = SimpleNamespace(evaluation_profile_id="profile_a", profile_hash="h" * 64)
            evaluation = SimpleNamespace(
                evaluation_run_id="evaluation_a",
                evaluation_profile_id=profile.evaluation_profile_id,
                profile_hash=profile.profile_hash,
                strategy_run_ids=("strategy_run_a",),
            )
            store.profiles[profile.evaluation_profile_id] = profile
            store.evaluations[evaluation.evaluation_run_id] = evaluation
            store.evaluation_history = Mock(
                side_effect=AssertionError("global history must not drive workflow reuse")
            )
            attempts = _Attempts(
                [
                    _Attempt(
                        "attempt_a",
                        "strategy_run_a",
                        artifact_references=[
                            {
                                "artifact_key": "evaluation_run",
                                "evaluation_run_id": evaluation.evaluation_run_id,
                            }
                        ],
                    )
                ]
            )
            workflow = WorkflowCoordinator(_Execution(store, attempts))
            created = workflow.create({"strategy": "controlled"}, label_ko="평가 재사용", idempotency_key="create")
            workflow._event(
                created["workflow_id"],
                "economic_started",
                {
                    "execution_request_id": "request_a",
                    "execution_attempt_ids": ["attempt_a"],
                    "strategy_run_ids": ["strategy_run_a"],
                },
            )
            with patch("src.trend_v2_foundation.workflow.calculate_and_evaluate_saved_runs") as calculate:
                state = workflow.evaluate(
                    created["workflow_id"],
                    evaluation_profile_id=profile.evaluation_profile_id,
                    idempotency_key="evaluate",
                )
                replay = workflow.evaluate(
                    created["workflow_id"],
                    evaluation_profile_id=profile.evaluation_profile_id,
                    idempotency_key="evaluate",
                )
            calculate.assert_not_called()
            self.assertEqual(state["references"]["evaluation"]["evaluation_run_id"], evaluation.evaluation_run_id)
            self.assertEqual(replay["references"]["evaluation"], state["references"]["evaluation"])

    def test_economic_start_records_run_ids_from_created_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempts = _Attempts([_Attempt("attempt_a", "strategy_run_a", status="queued")])
            execution = _Execution(_Store(Path(temporary)), attempts)
            execution.create_request = Mock(
                return_value=SimpleNamespace(
                    execution_request_id="request_a",
                    requested_strategy_run_candidates=({"schema_version": "strategy_run_spec_v1"},),
                )
            )
            execution.start = Mock(
                return_value={"execution_attempt_ids": ["attempt_a"]}
            )
            workflow = WorkflowCoordinator(execution)
            created = workflow.create(
                {"strategy": "controlled"},
                label_ko="실행 연결",
                idempotency_key="create",
            )

            state = workflow.start_economic(
                created["workflow_id"], idempotency_key="start"
            )

            self.assertEqual(
                state["references"]["economic"]["strategy_run_ids"],
                ["strategy_run_a"],
            )

    def test_workflow_polling_gets_do_not_rebuild_the_saved_run_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalResultStore(
                temporary,
                ArtifactRetentionPolicy(5_000_000, 1_000_000, 10, 10),
            )
            execution = _Execution(store, _Attempts([]))
            workflow = WorkflowCoordinator(execution)
            created = workflow.create({"strategy": "controlled"}, label_ko="저부하 조회", idempotency_key="create")
            registry_builder = Mock()
            registry_builder.load_or_rebuild.side_effect = AssertionError("registry rebuild is too expensive for polling")
            api = ReadOnlyTrendApi(
                store,
                registry_builder=registry_builder,
                workflow_coordinator=workflow,
            )
            listing = api.dispatch("GET", "/api/v1/workflows")
            detail = api.dispatch("GET", f"/api/v1/workflows/{created['workflow_id']}")
            self.assertEqual(listing.status_code, 200)
            self.assertEqual(detail.status_code, 200)
            registry_builder.load_or_rebuild.assert_not_called()


class Foundation10DQuickUiTests(unittest.TestCase):
    def test_default_workspace_is_bounded_numeric_first_and_keeps_expansion_open(self) -> None:
        source = (
            ROOT / "src" / "trend_v2_foundation" / "ui_assets" / "app.js"
        ).read_text(encoding="utf-8")
        index = (
            ROOT / "src" / "trend_v2_foundation" / "ui_assets" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("WORKSPACE_AUTO_ECONOMIC_LIMIT = 8", source)
        self.assertIn("WORKSPACE_AUTO_TOTAL_LIMIT = 16", source)
        self.assertIn('value="20,40,55"', source)
        self.assertIn("await advanceWorkspace(saved.workflow_id, saved)", source)
        self.assertIn("?page_size=64", source)
        self.assertIn("현재 후보군에서 적격 전략이 없습니다", source)
        self.assertIn("새로운 신호·청산·사이징 전략군", source)
        self.assertLess(index.index('data-route="workflow"'), index.index('data-route="construction"'))


if __name__ == "__main__":
    unittest.main()
