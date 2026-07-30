from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "oos_evaluation_manifest.json"
PROTOCOL_PATH = ROOT / "config" / "oos_evaluation_protocol_v1.json"
DESIGN_PATH = ROOT / "docs" / "tasks" / "append_only_oos_evaluation_design.md"

GENERATED_DATA_COMMIT = "e844f557820c0987eeea96424e261c6fde085a51"
CALCULATION_SOURCE_COMMIT = "5b23b5d6070f4924e1afc53e7561c007663a0f0b"
STRATEGY_KEY = "score_bo_l40_rm002_erp010__signal_3d_confirm__ma50"
CONTRACT_MERGE_COMMIT = "f60f46e9c7bb4006ea8be22e76b5230b71dde1d5"
CONTRACT_MERGED_AT_UTC = "2026-07-30T15:47:40Z"
EXPECTED_SELECTION_ARTIFACTS = {
    "docs/data/backtest_summary.json": {
        "git_blob_sha": "24e0fb2fed68f642d3a48b899949c62ffd157de1",
        "content_sha256": (
            "69453fe12be7da4da3db94991392ff8f90f177def96cfab5dd204e09e1c5ada8"
        ),
    },
    "docs/data/backtest_strategy_summary.csv": {
        "git_blob_sha": "5b3d4ce8a4a969a7ef658a049adce96a65655f53",
        "content_sha256": (
            "20e8c6abc7519a5d5cfaa69baf8200bcdf9a0d063758189f583ced577345f927"
        ),
    },
    "docs/data/backtest_portfolio_curve_manifest.json": {
        "git_blob_sha": "a4bb4c16e89608566f4fc040a59d2390a303a2a0",
        "content_sha256": (
            "5b0956a714dea7ca5277f3eb6506a2a094c80766f7a474a02e390f92531e6572"
        ),
    },
}
EXPECTED_FROZEN_SOURCE_BLOBS = {
    "src/backtest.py": "984bcbc40fe6356d3d09109d381983ec49a5fb45",
    "src/features.py": "ebc82a6fe3b8f37a00b6ca25f27d607940ca40d5",
    "src/portfolio.py": "ac770dfa22b50952adbea66a28edb4173afde41c",
    "src/universe.py": "760f388fa98fd6194c57c3c41b7f90704694efb8",
    "src/prices.py": "d2707fb66570e9397b9eb51ec45074228e4cd65a",
    "config/universe.yml": "decb4071c5c78918004b0b8fb9902d08d12bf595",
    "config/exclusions.yml": "f20ad32321852911b1ee3b879133e8a770170a87",
    "config/manual_overrides.csv": "a0fe92cc43079612525daa43861dc4fd671d3f3b",
}
EXPECTED_OPERATIONAL_SOURCE_BLOBS = {
    ".github/workflows/daily_scan.yml": "f7f90158433408d691bb5d785bbf3c6cdfbe1cbf",
    ".github/workflows/backtest-only.yml": (
        "512650574bfca27068f088f0ee788fc982d5e740"
    ),
    "scripts/verify_data_publish_base.py": (
        "e630394f5946901746c315654c95f5bd1bd3188c"
    ),
    "src/run_daily_scan.py": "4ab81ec06afb98fdd622645f04c9de6013c26683",
    "src/run_backtest_only.py": "0ea7023b833fdbbff89b9fd5681301375c095839",
}
REQUIRED_FIELDS = {
    "manifest_version",
    "cohort_id",
    "status",
    "repository",
    "initial_design_created_at_utc",
    "provenance_timeline_interpretation",
    "candidate_designation",
    "candidate",
    "activation",
    "evaluation_protocol",
    "append_only",
    "no_backfill",
    "correction_policy",
    "provenance_categories",
    "contract_merge_provenance",
    "unresolved_items",
}
PROVENANCE_CATEGORIES = {
    "frozen_semantic_definitions",
    "immutable_candidate_selection_evidence",
    "dynamic_observation_inputs",
    "operational_baselines",
}


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    return parsed


def test_schema_and_exact_pinned_provenance() -> None:
    manifest = load_manifest()

    assert REQUIRED_FIELDS.issubset(manifest)
    assert manifest["manifest_version"] == "2.1.0"
    assert manifest["cohort_id"] == "oos-0001"
    assert manifest["status"] == "proposed"
    assert manifest["append_only"] is True
    assert manifest["no_backfill"] is True
    assert "created_at_utc" not in manifest

    categories = manifest["provenance_categories"]
    assert set(categories) == PROVENANCE_CATEGORIES

    frozen = categories["frozen_semantic_definitions"]
    evidence = categories["immutable_candidate_selection_evidence"]
    operational = categories["operational_baselines"]
    assert frozen["calculation_source_commit"] == CALCULATION_SOURCE_COMMIT
    assert operational["baseline_source_commit"] == CALCULATION_SOURCE_COMMIT
    assert evidence["generated_data_commit"] == GENERATED_DATA_COMMIT
    assert evidence["generated_data_commit_created_at_utc"] == (
        "2026-07-30T15:09:14Z"
    )
    assert evidence["generated_from_source_commit"] == CALCULATION_SOURCE_COMMIT
    assert evidence["generated_commit_direct_parent_verified"] is True
    assert evidence["artifact_as_of"] == "2026-07-29"

    for commit in (
        frozen["calculation_source_commit"],
        evidence["generated_data_commit"],
        evidence["generated_from_source_commit"],
        operational["baseline_source_commit"],
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", commit)

    contract_merge = manifest["contract_merge_provenance"]
    assert contract_merge["commit"] == CONTRACT_MERGE_COMMIT
    assert contract_merge["merged_at_utc"] == CONTRACT_MERGED_AT_UTC
    assert "authoritative PR #18 preregistration-contract merge" in contract_merge[
        "role"
    ]
    assert "neither the collector implementation commit nor the collector activation commit" in (
        contract_merge["role"]
    )
    assert "did not start OOS collection" in contract_merge["role"]
    assert "did not create an eligible ex-ante decision" in contract_merge["role"]


def test_provenance_timeline_does_not_imply_premature_activation() -> None:
    manifest = load_manifest()
    evidence = manifest["provenance_categories"][
        "immutable_candidate_selection_evidence"
    ]
    activation = manifest["activation"]

    initial_design_time = parse_utc(manifest["initial_design_created_at_utc"])
    generated_data_time = parse_utc(
        evidence["generated_data_commit_created_at_utc"]
    )
    artifact_as_of = date.fromisoformat(evidence["artifact_as_of"])

    assert manifest["initial_design_created_at_utc"] == "2026-07-28T17:30:13Z"
    assert initial_design_time < generated_data_time
    assert artifact_as_of <= generated_data_time.date()

    timeline = manifest["provenance_timeline_interpretation"]
    assert "initial design existed on 2026-07-28" in timeline["initial_design"]
    assert "could not have been pinned before" in timeline[
        "selection_snapshot_pinning"
    ]
    assert "became authoritative when PR #18 merged" in timeline[
        "contract_authority"
    ]
    assert CONTRACT_MERGE_COMMIT in timeline["contract_merge_fact"]
    assert CONTRACT_MERGED_AT_UTC in timeline["contract_merge_fact"]
    assert "without activating OOS collection" in timeline[
        "contract_merge_fact"
    ]

    assert manifest["contract_merge_provenance"]["merged_at_utc"] == (
        CONTRACT_MERGED_AT_UTC
    )
    assert activation["contract_merge"]["merged_at_utc"] == CONTRACT_MERGED_AT_UTC
    assert activation["collector_implementation_activation"][
        "activated_at_utc"
    ] is None
    assert activation["first_eligible_ex_ante_decision"][
        "recorded_at_utc"
    ] is None
    assert activation["activation_event_recorded"] is False


def test_candidate_identity_and_parameter_fingerprint_are_consistent() -> None:
    manifest = load_manifest()
    candidate = manifest["candidate"]
    parameter_fingerprint = candidate["parameter_fingerprint"]
    snapshot = parameter_fingerprint["snapshot"]

    assert candidate["strategy_keys"] == [STRATEGY_KEY]
    assert len(candidate["strategy_keys"]) == len(set(candidate["strategy_keys"]))
    assert snapshot == {
        "strategy_key": STRATEGY_KEY,
        "signal_key": "score_bo_l40_rm002_erp010",
        "signal_params": {
            "family": "score_breakout",
            "score_lookback": 40,
            "r20_min": -0.02,
            "er20_min": 0.1,
            "close_filter": "close > ma50",
        },
        "entry_key": "signal_3d_confirm",
        "exit_key": "ma50",
        "max_holding_days": 63,
        "round_trip_cost": 0.002,
    }
    assert parameter_fingerprint["algorithm"] == "sha256"
    assert parameter_fingerprint["sha256"] == (
        "5f7364c0cd3fb5c327db30d7b881970ed8dcefb87bdb06697bdbbd47729353aa"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", parameter_fingerprint["sha256"])
    assert canonical_sha256(snapshot) == parameter_fingerprint["sha256"]

    evidence = manifest["provenance_categories"][
        "immutable_candidate_selection_evidence"
    ]
    assert (
        evidence["candidate_parameter_fingerprint_sha256"]
        == parameter_fingerprint["sha256"]
    )


def test_embedded_selection_row_is_the_immutable_source_of_truth() -> None:
    manifest = load_manifest()
    evidence = manifest["provenance_categories"][
        "immutable_candidate_selection_evidence"
    ]
    fingerprint = evidence["selected_row_fingerprint"]
    selected = fingerprint["snapshot"]
    parameter = manifest["candidate"]["parameter_fingerprint"]["snapshot"]

    assert selected["artifact_as_of"] == evidence["artifact_as_of"]
    assert selected["strategy_key"] == STRATEGY_KEY
    assert selected["strategy_key"] == parameter["strategy_key"]
    assert selected["signal_key"] == parameter["signal_key"]
    assert selected["signal_params"] == parameter["signal_params"]
    assert selected["entry_key"] == parameter["entry_key"]
    assert selected["exit_key"] == parameter["exit_key"]
    assert selected["qualification_tier"] == "Qualified"
    assert selected["qualification_rank"] == 1
    assert selected["time_gate_pass"] is True
    assert selected["parameter_gate_pass"] is True
    assert fingerprint["algorithm"] == "sha256"
    assert fingerprint["sha256"] == (
        "29a22ac630f3acfe462e5c3cbb9037e1563fc1498f761c8fdd1826dcfd23162d"
    )
    assert canonical_sha256(selected) == fingerprint["sha256"]

    artifacts = evidence["selection_artifacts"]
    assert set(artifacts) == set(EXPECTED_SELECTION_ARTIFACTS)
    for path, expected in EXPECTED_SELECTION_ARTIFACTS.items():
        artifact = artifacts[path]
        assert artifact["git_blob_sha"] == expected["git_blob_sha"]
        assert artifact["content_sha256"] == expected["content_sha256"]
        assert re.fullmatch(r"[0-9a-f]{40}", artifact["git_blob_sha"])
        assert re.fullmatch(r"[0-9a-f]{64}", artifact["content_sha256"])
        assert artifact["role"]


def test_semantic_snapshot_fingerprint_and_source_blobs_are_valid() -> None:
    manifest = load_manifest()
    frozen = manifest["provenance_categories"]["frozen_semantic_definitions"]
    fingerprint = frozen["semantic_fingerprint"]
    snapshot = fingerprint["snapshot"]

    assert fingerprint["algorithm"] == "sha256"
    assert fingerprint["sha256"] == (
        "1d31be17d656cd86eedb34d1ca64e5172e1491a371d4907bfa6f18fffb551b92"
    )
    assert canonical_sha256(snapshot) == fingerprint["sha256"]
    assert snapshot["strategy_identity"]["strategy_key"] == STRATEGY_KEY
    assert snapshot["transaction_cost"]["round_trip_rate"] == 0.002
    assert snapshot["canonical_portfolio"]["model"] == (
        "canonical_equal_weight_active_v1"
    )
    assert snapshot["price_adjustment"] == {
        "vendor": "Yahoo Finance via yfinance",
        "interval": "1d",
        "auto_adjust": True,
        "actions": False,
    }

    assert frozen["source_blobs"]
    assert {
        path: source["git_blob_sha"]
        for path, source in frozen["source_blobs"].items()
    } == EXPECTED_FROZEN_SOURCE_BLOBS
    for path, source in frozen["source_blobs"].items():
        assert path
        assert source["commit"] == CALCULATION_SOURCE_COMMIT
        assert re.fullmatch(r"[0-9a-f]{40}", source["git_blob_sha"])
        assert source["role"]


def test_dynamic_inputs_are_not_hard_frozen_semantics() -> None:
    manifest = load_manifest()
    categories = manifest["provenance_categories"]
    frozen_sources = categories["frozen_semantic_definitions"]["source_blobs"]
    dynamic = categories["dynamic_observation_inputs"]

    assert "config/aum.csv" not in frozen_sources
    assert "config/aum.csv" in dynamic["expected_dynamic_inputs"]
    selection_aum = dynamic["historical_selection_inputs"]["config/aum.csv"]
    assert selection_aum["commit"] == CALCULATION_SOURCE_COMMIT
    assert selection_aum["git_blob_sha"] == (
        "0c13b8c6bf21fc9a68f89efa74c804b242304959"
    )
    assert selection_aum["content_sha256"] == (
        "dcf9e735e5c2e8e81e9978353fe571bac7af9e455e61df201387a6f9b0053336"
    )
    assert "not a permanently frozen semantic blob" in selection_aum[
        "classification"
    ]
    for required in (
        "downloaded price data",
        "realized daily eligible universe",
        "daily universe membership",
        "daily decision inputs",
        "vendor data revisions",
    ):
        assert required in dynamic["expected_dynamic_inputs"]
    assert "input_artifact_hashes" in dynamic["required_per_observation"]
    assert "realized_universe_snapshot_hash" in dynamic[
        "required_per_observation"
    ]


def test_operational_baselines_require_explicit_compatibility_check() -> None:
    manifest = load_manifest()
    operational = manifest["provenance_categories"]["operational_baselines"]

    assert "explicit compatibility check" in operational["compatibility_policy"]
    expected_paths = {
        ".github/workflows/daily_scan.yml",
        ".github/workflows/backtest-only.yml",
        "scripts/verify_data_publish_base.py",
        "src/run_daily_scan.py",
        "src/run_backtest_only.py",
    }
    assert set(operational["source_blobs"]) == expected_paths
    assert {
        path: source["git_blob_sha"]
        for path, source in operational["source_blobs"].items()
    } == EXPECTED_OPERATIONAL_SOURCE_BLOBS
    for source in operational["source_blobs"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", source["git_blob_sha"])
        assert source["role"]


def test_proposed_status_keeps_every_activation_fact_unresolved() -> None:
    manifest = load_manifest()
    activation = manifest["activation"]

    assert manifest["status"] == "proposed"
    assert activation["model"] == (
        "first_eligible_ex_ante_decision_recorded_after_collector_activation"
    )
    assert activation["contract_merge"] == {
        "commit": CONTRACT_MERGE_COMMIT,
        "merged_at_utc": CONTRACT_MERGED_AT_UTC,
    }
    assert activation["collector_implementation_activation"] == {
        "commit": None,
        "activated_at_utc": None,
    }
    assert activation["approved_evaluation_protocol_version"] == "oos-eval-v1.0.0"
    assert activation["first_eligible_ex_ante_decision"] == {
        "record_id": None,
        "economic_date": None,
        "recorded_at_utc": None,
    }
    assert activation["activation_event_recorded"] is False
    assert activation["pre_activation_decisions_are_eligible"] is False
    assert activation["pre_activation_positions_are_eligible"] is False
    assert activation["no_retrospective_backfill"] is True

    blockers = manifest["unresolved_items"]
    assert blockers
    assert all(item["blocking"] is True for item in blockers)
    assert {item["id"] for item in blockers} == {
        "collector-implementation-and-activation",
        "first-eligible-ex-ante-decision",
    }


def test_manifest_references_exact_approved_protocol_fingerprint() -> None:
    manifest = load_manifest()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    reference = manifest["evaluation_protocol"]

    assert reference == {
        "protocol_version": "oos-eval-v1.0.0",
        "path": "config/oos_evaluation_protocol_v1.json",
        "status": "approved_pre_activation",
        "canonical_sha256": (
            "e2b84b905c513ee73dfd36f918ce6723aa847570849ee65ddcba4a862b4ab5f5"
        ),
        "canonicalization": (
            "UTF-8 JSON with sorted keys, no insignificant whitespace, "
            "ensure_ascii=false, and allow_nan=false"
        ),
    }
    assert canonical_sha256(protocol) == reference["canonical_sha256"]


def test_candidate_terminology_cannot_imply_live_approval() -> None:
    manifest = load_manifest()
    design = DESIGN_PATH.read_text(encoding="utf-8")
    combined = (
        json.dumps(manifest, ensure_ascii=False) + "\n" + design
    ).casefold()

    for prohibited in (
        "current production primary",
        "production primary",
        "production candidate",
        "production selector",
        "production-approved strategy",
    ):
        assert prohibited not in combined

    for required in (
        "current backtest qualified rank-1 candidate at the pinned selection snapshot",
        "frozen oos evaluation candidate",
        "not production-approved",
        "not validated for out-of-sample profitability",
        "selected by the backtest ranking function",
    ):
        assert required in combined


def test_manifest_serialization_is_deterministic_and_snapshot_stable() -> None:
    manifest = load_manifest()
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

    test_source = Path(__file__).read_text(encoding="utf-8")
    assert ("AUTHORITATIVE_" + "STRATEGY_PATH") not in test_source
    assert ('ROOT / "docs" / ' + '"data"') not in test_source
    assert ("pandas." + "read_csv") not in test_source
