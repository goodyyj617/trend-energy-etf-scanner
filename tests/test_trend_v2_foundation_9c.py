"""Foundation 9C bootstrap coverage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.trend_v2_foundation.local_operability import _STATE_DIRECTORIES, initialize_result_store, run_preflight


ROOT = Path(__file__).parents[1]


class Foundation9CBootstrapTests(unittest.TestCase):
    def test_init_creates_canonical_store_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Path(temporary) / "store"
            self.assertTrue(initialize_result_store(store))
            self.assertFalse(initialize_result_store(store))
            self.assertTrue((store / "retention_policy.json").is_file())
            self.assertTrue(all((store / name).is_dir() for name in _STATE_DIRECTORIES))

    def test_init_refuses_nonempty_or_corrupt_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Path(temporary) / "store"
            store.mkdir(); (store / "foreign.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not an initialized"):
                initialize_result_store(store)
            (store / "foreign.txt").unlink(); (store / "retention_policy.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                initialize_result_store(store)

    def test_preflight_distinguishes_missing_and_empty_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for store, expected in ((root / "missing", "디렉터리가 없습니다"), (root / "empty", "초기화되지 않았습니다")):
                if store.name == "empty": store.mkdir()
                report = run_preflight(ROOT, store, port=0, package_importer=lambda _name: object())
                check = next(item for item in report["checks"] if item["code"] == "result_store_schema")
                self.assertEqual(check["status"], "blocking")
                self.assertIn(expected, check["message_ko"])
                self.assertIn(" init --store ", check["suggested_action_ko"])

    def test_snapshot_falls_back_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"; snapshot.mkdir()
            member = snapshot / "frozen.csv"; member.write_bytes(b"frozen")
            import hashlib
            (snapshot / "input_manifest.json").write_text(json.dumps({"schema_version": "trend_v2_phase_a2_snapshot_v1", "snapshot_members": {"frozen.csv": hashlib.sha256(b"frozen").hexdigest()}}), encoding="utf-8")
            report = run_preflight(ROOT, Path(temporary) / "missing", port=0, package_importer=lambda _name: object(), snapshot_root=snapshot)
            self.assertIn("working-tree bytes", next(item for item in report["checks"] if item["code"] == "frozen_data_snapshot")["diagnostic_en"])
            member.write_bytes(b"changed")
            report = run_preflight(ROOT, Path(temporary) / "missing", port=0, package_importer=lambda _name: object(), snapshot_root=snapshot)
            self.assertEqual(next(item for item in report["checks"] if item["code"] == "frozen_data_snapshot")["status"], "blocking")
