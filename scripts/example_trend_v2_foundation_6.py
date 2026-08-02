"""Synthetic, no-download Foundation 6 catalog and restart-recovery demonstration."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trend_v2_foundation.foundation_6 import (
    CATALOG_SCHEMA_VERSION,
    Foundation6Error,
    OptionCatalog,
    PersistedExecutionManager,
    estimate_candidates,
)


def construction(history_sessions: int = 252) -> dict:
    return {
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "components": {
            "trend_filter": {"option_id": "close_above_ma200_v1", "parameters": {}},
            "signal": {"option_id": "prior_price_high_v2", "parameters": {"lookback": {"kind": "list", "values": [20, 55]}}},
            "transaction_cost": {"option_id": "round_trip_bps_v1", "parameters": {"bps": {"kind": "list", "values": ["0", "10"]}}},
        },
        "history_sessions": history_sessions,
        "universe_size": 470,
        "asset_group_data_available": True,
        "evaluation_profile_ids": ["conservative", "exploratory"],
    }


def main() -> None:
    catalog = OptionCatalog.load(ROOT / "config" / "trend_v2" / "strategy_option_catalog_v2.json")
    estimate = estimate_candidates(catalog, construction())
    try:
        estimate_candidates(catalog, construction(history_sessions=19))
    except Foundation6Error as error:
        incompatibility = error.to_dict()
    with tempfile.TemporaryDirectory() as directory:
        manager = PersistedExecutionManager(directory, catalog, host_identity="foundation-6-synthetic")
        request = manager.create_request(estimate)
        manager.register_worker("synthetic-worker", 12345)
        lease = manager.lease_next(request["execution_request_id"], "synthetic-worker")
        # This models a local restart: no in-memory worker liveness is trusted.
        restarted = PersistedExecutionManager(directory, catalog, host_identity="foundation-6-synthetic")
        recovery = restarted.reconcile(request["execution_request_id"])
        resume = restarted.resume(request["execution_request_id"])
        output = {
            "catalog": {"schema_version": catalog.document["schema_version"], "catalog_hash": catalog.catalog_hash},
            "estimate": {key: estimate[key] for key in ("raw_cartesian_combinations", "invalid_incompatible_combinations", "canonical_duplicates", "valid_unique_economic_candidates", "reusable_completed_candidates", "new_candidates_requiring_execution", "evaluation_only_applications", "total_estimated_work")},
            "incompatible_combination": incompatibility,
            "interrupted_lease": lease,
            "recovery": recovery,
            "resume": resume,
            "reconstructed_candidate_states": restarted.status(request["execution_request_id"])["candidates"],
            "safety": "No market-data download, remote execution, arbitrary formula, or unrestricted search is used.",
        }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
