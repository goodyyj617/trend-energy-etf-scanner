"""Synthetic Foundation 6 catalog and restart-recovery tests; no market data."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.trend_v2_foundation.foundation_6 import (
    CATALOG_SCHEMA_VERSION,
    Foundation6Error,
    OptionCatalog,
    PersistedExecutionManager,
    estimate_candidates,
    normalize_selection,
)


CATALOG_PATH = Path(__file__).parents[1] / "config" / "trend_v2" / "strategy_option_catalog_v2.json"


def request(**overrides):
    value = {
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "components": {
            "signal": {"option_id": "prior_price_high_v2", "parameters": {"lookback": {"kind": "list", "values": [20, 55]}}},
            "transaction_cost": {"option_id": "round_trip_bps_v1", "parameters": {"bps": {"kind": "list", "values": ["0", "10"]}}},
        },
        "evaluation_profile_ids": ["risk", "return"],
        "history_sessions": 252,
        "universe_size": 470,
        "asset_group_data_available": True,
    }
    value.update(overrides)
    return value


class Foundation6CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = OptionCatalog.load(CATALOG_PATH)

    def test_catalog_is_deterministic_korean_and_complete(self):
        first, second = self.catalog.to_dict(), OptionCatalog.load(CATALOG_PATH).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], CATALOG_SCHEMA_VERSION)
        self.assertEqual(first["categories"]["signal"][0]["name_ko"], "직전 가격 고점 돌파")
        self.assertIn("engine_adapter_support", first["categories"]["trend_filter"][0])

    def test_exact_pruning_counts_and_profile_only_work(self):
        estimate = estimate_candidates(self.catalog, request())
        self.assertEqual(estimate["raw_cartesian_combinations"], 4)
        self.assertEqual(estimate["invalid_incompatible_combinations"], 0)
        self.assertEqual(estimate["canonical_duplicates"], 0)
        self.assertEqual(estimate["valid_unique_economic_candidates"], 4)
        self.assertEqual(estimate["evaluation_only_applications"], 8)
        reused = estimate_candidates(self.catalog, request(), reusable_hashes=[estimate["candidate_economic_hashes"][0]])
        self.assertEqual(reused["reusable_completed_candidates"], 1)
        self.assertEqual(reused["new_candidates_requiring_execution"], 3)

    def test_incompatible_history_is_rejected_in_korean(self):
        with self.assertRaises(Foundation6Error) as caught:
            normalize_selection(self.catalog, request(history_sessions=19))
        self.assertEqual(caught.exception.code, "unsupported_strategy_combination")
        self.assertEqual(caught.exception.message_ko, "Low20 손절·청산에는 최소 20개 거래일의 이력이 필요합니다.")


class Foundation6PersistenceTests(unittest.TestCase):
    def test_restart_recovery_lease_exclusivity_and_resume(self):
        catalog = OptionCatalog.load(CATALOG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            manager = PersistedExecutionManager(temporary, catalog, host_identity="synthetic-host")
            one = request(components={
                "signal": {"option_id": "prior_price_high_v2", "parameters": {"lookback": {"kind": "fixed", "value": 20}}},
                "transaction_cost": {"option_id": "round_trip_bps_v1", "parameters": {"bps": {"kind": "fixed", "value": "0"}}},
            })
            persisted = manager.create_request(estimate_candidates(catalog, one))
            manager.register_worker("worker-a", 101)
            manager.register_worker("worker-b", 102)
            leased = manager.lease_next(persisted["execution_request_id"], "worker-a")
            self.assertEqual(leased["state"], "running")
            self.assertIsNone(manager.lease_next(persisted["execution_request_id"], "worker-b"))
            # A reconstructed manager has no in-memory liveness proof.  It blocks the
            # interrupted lease, records the decision, then resumes only incomplete work.
            restarted = PersistedExecutionManager(temporary, catalog, host_identity="synthetic-host")
            recovered = restarted.reconcile(persisted["execution_request_id"])
            self.assertEqual(recovered["decisions"][0]["classification"], "running_with_no_live_worker")
            resumed = restarted.resume(persisted["execution_request_id"])
            self.assertIn(leased["candidate_economic_hash"], resumed["requeued"])
            state = restarted.status(persisted["execution_request_id"])["candidates"]
            self.assertEqual(state[0]["state"], "pending")

    def test_corrupt_event_fails_closed(self):
        catalog = OptionCatalog.load(CATALOG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            manager = PersistedExecutionManager(temporary, catalog)
            manager.create_request(estimate_candidates(catalog, request()))
            event = next((Path(temporary) / "events").glob("*.json"))
            event.write_text("{}", encoding="utf-8")
            with self.assertRaises(Foundation6Error) as caught:
                PersistedExecutionManager(temporary, catalog)
            self.assertEqual(caught.exception.code, "attempt_event_corrupt")
