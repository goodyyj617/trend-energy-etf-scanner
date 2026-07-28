from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "oos_evaluation_manifest.json"
AUTHORITATIVE_STRATEGY_PATH = ROOT / "docs" / "data" / "backtest_summary.json"
REQUIRED_FIELDS = {
    "manifest_version",
    "cohort_id",
    "status",
    "repository",
    "source_main_commit",
    "created_at_utc",
    "activation_rule",
    "strategy_source",
    "strategy_keys",
    "strategy_parameter_fingerprints",
    "universe_definition",
    "transaction_cost_definition",
    "benchmark_definition",
    "portfolio_model_definition",
    "data_conventions",
    "no_backfill",
    "append_only",
    "correction_policy",
    "provenance",
    "unresolved_items",
}


def test_oos_manifest_schema_and_determinism() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS.issubset(manifest)
    assert manifest["no_backfill"] is True
    assert manifest["append_only"] is True
    assert re.fullmatch(r"[1-9]\d*\.\d+\.\d+", manifest["manifest_version"])
    assert re.fullmatch(r"oos-\d{4}", manifest["cohort_id"])
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source_main_commit"])

    strategy_keys = manifest["strategy_keys"]
    assert strategy_keys
    assert len(strategy_keys) == len(set(strategy_keys))

    authoritative = json.loads(
        AUTHORITATIVE_STRATEGY_PATH.read_text(encoding="utf-8")
    )
    production_primary = [
        row for row in authoritative["summary"]
        if row["qualification_rank"] == 1
        and row["qualification_tier"] == "Qualified"
    ]
    assert len(production_primary) == 1
    assert strategy_keys == [production_primary[0]["strategy_key"]]

    blockers = [
        item for item in manifest["unresolved_items"]
        if item.get("blocking") is True
    ]
    assert not blockers or manifest["status"] != "active"

    fingerprints = manifest["strategy_parameter_fingerprints"]
    assert set(fingerprints) == set(strategy_keys)
    for strategy_key in strategy_keys:
        fingerprint = fingerprints[strategy_key]
        assert fingerprint["algorithm"] == "sha256"
        assert re.fullmatch(r"[0-9a-f]{64}", fingerprint["sha256"])
        snapshot = fingerprint["parameter_snapshot"]
        assert snapshot["strategy_key"] == strategy_key
        assert snapshot["signal_key"] == production_primary[0]["signal_key"]
        assert snapshot["entry_key"] == production_primary[0]["entry_key"]
        assert snapshot["exit_key"] == production_primary[0]["exit_key"]
        assert snapshot["signal_params"] == json.loads(
            production_primary[0]["signal_params"]
        )
        assert snapshot["max_holding_days"] == authoritative["max_holding_days"]
        canonical_snapshot = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert hashlib.sha256(canonical_snapshot).hexdigest() == fingerprint["sha256"]

    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    round_tripped = json.dumps(
        json.loads(canonical),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert canonical == round_tripped
