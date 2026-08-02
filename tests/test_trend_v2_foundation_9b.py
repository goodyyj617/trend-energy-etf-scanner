"""Focused local-operability checks; all fixtures are local and synthetic."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.trend_v2_foundation import ArtifactRetentionPolicy, FileExecutionAttemptRepository, LocalResultStore
from src.trend_v2_foundation.foundation_6 import OptionCatalog, PersistedExecutionManager, estimate_candidates
from src.trend_v2_foundation.local_operability import reconcile_local_state, run_preflight
from src.trend_v2_foundation.workflow import WorkflowCoordinator, WorkflowError


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "config" / "trend_v2" / "strategy_option_catalog_v2.json"


def _snapshot(root: Path) -> Path:
    root.mkdir()
    member = root / "prices.csv.gz"
    member.write_bytes(b"frozen")
    (root / "input_manifest.json").write_text(json.dumps({"schema_version": "trend_v2_phase_a2_snapshot_v1", "snapshot_members": {"prices.csv.gz": hashlib.sha256(b"frozen").hexdigest()}}), encoding="utf-8")
    return root


def _request() -> dict:
    return {"catalog_schema_version": "controlled_strategy_option_catalog_v2", "components": {"signal": {"option_id": "prior_price_high_v2", "parameters": {"lookback": {"kind": "fixed", "value": 20}}}, "transaction_cost": {"option_id": "round_trip_bps_v1", "parameters": {"bps": {"kind": "fixed", "value": "0"}}}}, "evaluation_profile_ids": ["risk"], "history_sessions": 252, "universe_size": 470, "asset_group_data_available": True}


class _NoRobustness:
    def reconcile(self, _attempt_id): raise AssertionError("no robustness attempt should be reconciled")


class _NoWorkflows:
    def read(self, _workflow_id): raise AssertionError("no workflow should be read")


class LocalPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = self.root / "store"
        LocalResultStore(self.store, ArtifactRetentionPolicy(1_000_000, 100_000, 5, 5))
        self.snapshot = _snapshot(self.root / "snapshot")

    def tearDown(self): self.temp.cleanup()

    def _report(self, **overrides):
        snapshot = overrides.pop("snapshot_root", self.snapshot)
        return run_preflight(ROOT, self.store, port=0, package_importer=lambda _name: object(), snapshot_root=snapshot, **overrides)

    def test_preflight_is_deterministic_and_successful(self):
        first, second = self._report(), self._report()
        self.assertFalse(first["blocking"])
        self.assertEqual(first["checks"], second["checks"])
        self.assertEqual([item["code"] for item in first["checks"]], sorted(item["code"] for item in first["checks"]))

    def test_python_and_dependency_failures_are_blocking(self):
        unsupported = self._report(python_version=(3, 9))
        self.assertEqual(next(item for item in unsupported["checks"] if item["code"] == "python_supported")["status"], "blocking")
        missing = run_preflight(ROOT, self.store, port=0, package_importer=lambda name: (_ for _ in ()).throw(ImportError()) if name == "numpy" else object(), snapshot_root=self.snapshot)
        self.assertEqual(next(item for item in missing["checks"] if item["code"] == "python_packages")["status"], "blocking")

    def test_schema_storage_snapshot_and_port_fail_closed(self):
        (self.store / "retention_policy.json").write_text("{}", encoding="utf-8")
        with patch("src.trend_v2_foundation.local_operability._writeable", return_value=False), patch("src.trend_v2_foundation.local_operability._port_available", return_value=False):
            report = self._report(snapshot_root=self.root / "missing")
        statuses = {item["code"]: item["status"] for item in report["checks"]}
        self.assertEqual(statuses["result_store_schema"], "blocking")
        self.assertEqual(statuses["result_store_access"], "blocking")
        self.assertEqual(statuses["frozen_data_snapshot"], "blocking")
        self.assertEqual(statuses["loopback_port"], "blocking")

    def test_corrupt_state_blocks_and_stale_state_warns(self):
        broken = self.store / "workflow_v1" / "workflows" / "broken.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{", encoding="utf-8")
        report = self._report()
        self.assertEqual(next(item for item in report["checks"] if item["code"] == "persisted_state_integrity")["status"], "blocking")
        broken.unlink()
        catalog = OptionCatalog.load(CATALOG)
        manager = PersistedExecutionManager(self.store / "execution_management_v1", catalog)
        request = manager.create_request(estimate_candidates(catalog, _request()))
        manager.register_worker("worker", 1)
        manager.lease_next(request["execution_request_id"], "worker")
        report = self._report()
        self.assertEqual(next(item for item in report["checks"] if item["code"] == "stale_or_interrupted_work")["status"], "warning")


class RecoveryTests(unittest.TestCase):
    def test_stale_worker_reconciliation_is_idempotent_and_preserves_completion(self):
        catalog = OptionCatalog.load(CATALOG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = PersistedExecutionManager(root / "execution_management_v1", catalog)
            request = manager.create_request(estimate_candidates(catalog, _request()))
            manager.register_worker("worker", 1)
            leased = manager.lease_next(request["execution_request_id"], "worker")
            attempts = FileExecutionAttemptRepository(root / "execution_attempts")
            report = reconcile_local_state(root, source_commit="a" * 40, manager=manager, attempts=attempts, robustness=_NoRobustness(), workflows=_NoWorkflows(), clock=lambda: "2026-08-02T00:00:00Z")
            repeated = reconcile_local_state(root, source_commit="a" * 40, manager=manager, attempts=attempts, robustness=_NoRobustness(), workflows=_NoWorkflows(), clock=lambda: "2026-08-02T00:00:00Z")
            self.assertEqual(report, repeated)
            state = manager.status(request["execution_request_id"])["candidates"][0]
            self.assertEqual(state["state"], "blocked")
            self.assertEqual(state["candidate_economic_hash"], leased["candidate_economic_hash"])

    def test_workflow_resume_without_persisted_work_is_rejected(self):
        class Execution:
            def __init__(self, root):
                self.store = type("Store", (), {"root": root})()
                self.source_commit = "a" * 40
                self.policy = type("Policy", (), {"policy_hash": "b" * 64})()
            def normalize(self, _construction):
                return type("Normalized", (), {"to_dict": lambda self: {"construction_hash": "c" * 64}})()
        with tempfile.TemporaryDirectory() as temporary:
            workflow = WorkflowCoordinator(Execution(Path(temporary)))
            created = workflow.create({"strategy": "controlled"}, label_ko="복구 테스트", idempotency_key="create")
            with self.assertRaisesRegex(WorkflowError, "No persisted"):
                workflow.resume(created["workflow_id"], idempotency_key="resume")


if __name__ == "__main__":
    unittest.main()
