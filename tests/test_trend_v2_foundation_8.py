"""Focused persistence/idempotency tests for the Foundation 8 coordinator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.trend_v2_foundation.workflow import WorkflowCoordinator, WorkflowError


class _Normalized:
    def to_dict(self): return {"construction_hash": "a" * 64, "schema_version": "normalized_strategy_construction_v1"}


class _Execution:
    def __init__(self, root: Path) -> None:
        self.store = SimpleNamespace(root=root)
        self.source_commit = "c" * 40
        self.policy = SimpleNamespace(policy_hash="p" * 64)
    def normalize(self, construction): return _Normalized()


class WorkflowPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.coordinator = WorkflowCoordinator(_Execution(Path(self.temp.name)))

    def tearDown(self): self.temp.cleanup()

    def test_identity_is_idempotent_and_state_rebuilds_from_events(self):
        first = self.coordinator.create({"strategy": "controlled"}, label_ko="합성 워크플로", idempotency_key="create-1")
        second = self.coordinator.create({"strategy": "controlled"}, label_ko="합성 워크플로", idempotency_key="create-1")
        self.assertEqual(first["workflow_id"], second["workflow_id"])
        normalized = self.coordinator.normalize(first["workflow_id"])
        rebuilt = WorkflowCoordinator(_Execution(Path(self.temp.name))).read(first["workflow_id"])
        self.assertEqual(normalized["stage"], "normalized")
        self.assertEqual(rebuilt["stage"], "normalized")

    def test_corrupt_event_fails_closed(self):
        created = self.coordinator.create({"strategy": "controlled"}, label_ko="합성 워크플로", idempotency_key="create-2")
        self.coordinator.normalize(created["workflow_id"])
        event = next((Path(self.temp.name) / "workflow_v1" / "events").glob("*.json"))
        event.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "corrupt"):
            self.coordinator.read(created["workflow_id"])

