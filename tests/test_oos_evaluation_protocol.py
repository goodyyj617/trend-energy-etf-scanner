from __future__ import annotations

import calendar
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "config" / "oos_evaluation_protocol_v1.json"
MANIFEST_PATH = ROOT / "config" / "oos_evaluation_manifest.json"
DESIGN_PATH = ROOT / "docs" / "tasks" / "oos_evaluation_protocol_v1.md"

PROTOCOL_VERSION = "oos-eval-v1.0.0"
COHORT_ID = "oos-0001"
CONTRACT_MERGE_COMMIT = "f60f46e9c7bb4006ea8be22e76b5230b71dde1d5"
CONTRACT_MERGED_AT_UTC = "2026-07-30T15:47:40Z"
PROTOCOL_SHA256 = "e2b84b905c513ee73dfd36f918ce6723aa847570849ee65ddcba4a862b4ab5f5"


def load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def add_calendar_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def calendar_month_end(value: date) -> date:
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def mature_inputs(**overrides: object) -> dict:
    values: dict[str, object] = {
        "completed_trades": 100,
        "valid_eligible_sessions": 252,
        "expected_market_sessions": 257,
        "first_eligible_decision_date": date(2026, 1, 15),
        "cutoff": date(2027, 1, 31),
        "unresolved_duplicate_conflicts": 0,
        "unresolved_correction_conflicts": 0,
        "unresolved_identity_conflicts": 0,
        "open_positions": 0,
    }
    values.update(overrides)
    return values


def maturity_label(protocol: dict, inputs: dict) -> str:
    maturity = protocol["maturity"]
    requirements = maturity["requirements"]
    expected_sessions = int(inputs["expected_market_sessions"])
    valid_sessions = int(inputs["valid_eligible_sessions"])
    coverage = valid_sessions / expected_sessions if expected_sessions else 0.0
    first_decision = inputs["first_eligible_decision_date"]
    cutoff = inputs["cutoff"]
    assert isinstance(first_decision, date)
    assert isinstance(cutoff, date)

    conditions = [
        int(inputs["completed_trades"])
        >= requirements["completed_trades"]["minimum_completed_trades"],
        valid_sessions
        >= requirements["eligible_sessions"]["minimum_eligible_sessions"],
        cutoff
        >= add_calendar_months(
            first_decision,
            requirements["elapsed_time"]["minimum_elapsed_calendar_months"],
        ),
        coverage
        >= requirements["session_coverage"][
            "minimum_valid_session_coverage_ratio"
        ],
        cutoff == calendar_month_end(cutoff),
        int(inputs["unresolved_duplicate_conflicts"]) == 0,
        int(inputs["unresolved_correction_conflicts"]) == 0,
        int(inputs["unresolved_identity_conflicts"]) == 0,
    ]
    assert maturity["all_conditions_required"] is True
    return maturity["allowed_labels"][1] if all(conditions) else maturity[
        "allowed_labels"
    ][0]


def test_protocol_identity_contract_baseline_and_inactive_state() -> None:
    protocol = load_protocol()
    baseline = protocol["contract_baseline"]
    activation = protocol["activation_state"]

    assert protocol["protocol_version"] == PROTOCOL_VERSION
    assert protocol["cohort_id"] == COHORT_ID
    assert protocol["status"] == "approved_pre_activation"
    assert baseline["contract_merge_commit"] == CONTRACT_MERGE_COMMIT
    assert baseline["contract_merged_at_utc"] == CONTRACT_MERGED_AT_UTC

    merged_at = datetime.fromisoformat(
        baseline["contract_merged_at_utc"].replace("Z", "+00:00")
    )
    assert merged_at.tzinfo is not None
    assert merged_at.utcoffset() == timezone.utc.utcoffset(merged_at)

    assert baseline["contract_became_authoritative"] is True
    assert baseline["oos_collection_activated"] is False
    assert baseline["is_collector_activation_commit"] is False
    assert baseline["eligible_oos_decision_created"] is False
    assert all(value is None for key, value in activation.items() if key != "oos_collection_started")
    assert activation["oos_collection_started"] is False


def test_exact_conjunctive_maturity_thresholds_and_labels() -> None:
    protocol = load_protocol()
    maturity = protocol["maturity"]
    requirements = maturity["requirements"]

    assert maturity["all_conditions_required"] is True
    assert requirements["completed_trades"]["minimum_completed_trades"] == 100
    assert requirements["eligible_sessions"]["minimum_eligible_sessions"] == 252
    assert requirements["elapsed_time"]["minimum_elapsed_calendar_months"] == 12
    assert requirements["session_coverage"][
        "minimum_valid_session_coverage_ratio"
    ] == 0.98
    assert maturity["allowed_labels"] == ["Immature", "Mature for review"]
    assert maturity["forbidden_labels"] == [
        "Pass",
        "Fail",
        "Qualified",
        "Production approved",
        "Deploy",
        "Reject",
    ]


def test_99_completed_trades_never_matures() -> None:
    assert maturity_label(load_protocol(), mature_inputs(completed_trades=99)) == (
        "Immature"
    )


def test_251_eligible_sessions_never_matures() -> None:
    inputs = mature_inputs(
        valid_eligible_sessions=251,
        expected_market_sessions=256,
    )
    assert maturity_label(load_protocol(), inputs) == "Immature"


def test_less_than_12_calendar_months_never_matures() -> None:
    inputs = mature_inputs(cutoff=date(2026, 12, 31))
    assert maturity_label(load_protocol(), inputs) == "Immature"


def test_coverage_below_point_98_never_matures() -> None:
    inputs = mature_inputs(expected_market_sessions=258)
    assert 252 / 258 < 0.98
    assert maturity_label(load_protocol(), inputs) == "Immature"


def test_all_thresholds_produce_mature_for_review() -> None:
    assert maturity_label(load_protocol(), mature_inputs()) == "Mature for review"


def test_high_trade_count_reached_early_remains_immature() -> None:
    inputs = mature_inputs(
        completed_trades=10000,
        first_eligible_decision_date=date(2026, 7, 1),
        cutoff=date(2026, 12, 31),
    )
    assert maturity_label(load_protocol(), inputs) == "Immature"


def test_first_evaluation_trigger_resolves_to_first_eligible_month_end() -> None:
    protocol = load_protocol()
    trigger = protocol["evaluation_trigger"]
    thresholds_first_met = date(2027, 3, 14)

    assert trigger["type"] == (
        "first_month_end_after_all_maturity_conditions_are_met"
    )
    assert trigger["maturity_evaluated_only_at_calendar_month_end"] is True
    assert trigger["discretionary_evaluation_date_allowed"] is False
    assert calendar_month_end(thresholds_first_met) == date(2027, 3, 31)
    assert trigger["primary_snapshot_immutable"] is True
    assert trigger["follow_up_snapshots_must_be_identified"] is True


def test_open_positions_are_neither_completed_nor_synthetically_closed() -> None:
    protocol = load_protocol()
    policy = protocol["open_position_policy"]

    assert policy["remain_open_at_cutoff"] is True
    assert policy["synthetic_or_forced_exit_allowed"] is False
    assert policy["count_as_completed_trade"] is False
    assert policy["included_in_completed_trade_metrics"] is False
    assert policy["included_in_portfolio_mark_to_market_metrics"] is True
    assert maturity_label(
        protocol,
        mature_inputs(completed_trades=99, open_positions=1000),
    ) == "Immature"


def test_missing_sessions_are_visible_and_never_count_as_valid() -> None:
    protocol = load_protocol()
    coverage = protocol["maturity"]["requirements"]["session_coverage"]
    policy = protocol["missing_session_policy"]

    assert coverage["missing_sessions_count_as_valid"] is False
    assert coverage["interpolation_allowed"] is False
    assert coverage["reconstruction_allowed"] is False
    assert coverage["perfect_coverage_required"] is False
    assert coverage["market_calendar_must_be_frozen_before_activation"] is True
    assert coverage["market_calendar_source"] is None
    assert coverage["market_calendar_version"] is None
    assert policy["counts_as_eligible_session"] is False
    assert policy["decision_backfill_allowed"] is False
    assert policy["remains_in_coverage_denominator"] is True
    assert policy["later_record_cannot_claim_ex_ante_capture"] is True
    assert policy["original_observation_rewrite_allowed"] is False


def test_interim_reporting_cannot_imply_approval() -> None:
    protocol = load_protocol()
    interim = protocol["interim_reporting"]
    label = interim["required_label"]

    assert label == "Immature — descriptive monitoring only"
    assert "Pass" not in label
    assert "production approved" not in label.casefold()
    assert "trigger production approval" in interim["prohibited_uses"]
    assert "present results as OOS validation" in interim["prohibited_uses"]


def test_locked_reporting_set_is_exact() -> None:
    reporting = load_protocol()["locked_reporting_set"]

    assert reporting["data_sufficiency_and_integrity"] == [
        "completed-trade count",
        "eligible-session count",
        "elapsed calendar months",
        "expected-session count",
        "valid-session coverage ratio",
        "missing-session count",
        "longest missing-session gap",
        "correction count",
        "unresolved-conflict count",
        "open-position count",
    ]
    assert reporting["canonical_portfolio_metrics"] == [
        "initial equity",
        "ending equity",
        "net total return",
        "CAGR when at least 252 eligible sessions exist",
        "annualized volatility",
        "Sharpe ratio under the existing repository convention",
        "maximum drawdown",
        "existing CDaR 95 definition",
        "Calmar ratio when mathematically defined",
        "turnover",
        "total transaction costs",
        "average gross exposure",
        "maximum gross exposure",
        "SPY return on exact common economic dates",
        "net excess return versus SPY",
    ]
    assert reporting["completed_trade_diagnostics"] == [
        "completed trades",
        "trade win rate",
        "mean net trade return",
        "median net trade return",
        "Profit Factor",
        "worst trade",
        "10th-percentile trade return",
        "average holding period",
        "stop-exit rate",
        "maximum-hold exit rate",
    ]


def test_protocol_change_control_is_locked_before_outcomes() -> None:
    control = load_protocol()["protocol_change_control"]

    assert all(control["before_activation"].values())
    assert control["after_activation"] == {
        "thresholds_immutable_for_cohort": True,
        "primary_evaluation_trigger_immutable_for_cohort": True,
        "retroactive_primary_evaluation_change_allowed": False,
        "exploratory_or_sensitivity_analysis_requires_new_protocol_version": True,
        "material_strategy_or_economic_change_requires_new_cohort": True,
        "later_backtest_rankings_may_modify_cohort_or_protocol": False,
    }


def test_protocol_serialization_and_manifest_fingerprint_are_deterministic() -> None:
    protocol = load_protocol()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    first = canonical_sha256(protocol)
    second = canonical_sha256(json.loads(json.dumps(protocol)))

    assert first == second == PROTOCOL_SHA256
    assert manifest["evaluation_protocol"] == {
        "protocol_version": PROTOCOL_VERSION,
        "path": "config/oos_evaluation_protocol_v1.json",
        "status": "approved_pre_activation",
        "canonical_sha256": PROTOCOL_SHA256,
        "canonicalization": (
            "UTF-8 JSON with sorted keys, no insignificant whitespace, "
            "ensure_ascii=false, and allow_nan=false"
        ),
    }


def test_manifest_remains_proposed_with_only_required_blockers() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol = load_protocol()
    activation = manifest["activation"]
    blocker_ids = {item["id"] for item in manifest["unresolved_items"]}
    protocol_merge = protocol["contract_baseline"]

    assert manifest["manifest_version"] == "2.1.0"
    assert manifest["status"] == "proposed"
    assert activation["approved_evaluation_protocol_version"] == PROTOCOL_VERSION
    assert activation["contract_merge"] == {
        "commit": CONTRACT_MERGE_COMMIT,
        "merged_at_utc": CONTRACT_MERGED_AT_UTC,
    }
    assert manifest["contract_merge_provenance"]["commit"] == (
        activation["contract_merge"]["commit"]
    )
    assert manifest["contract_merge_provenance"]["merged_at_utc"] == (
        activation["contract_merge"]["merged_at_utc"]
    )
    assert protocol_merge["contract_merge_commit"] == (
        activation["contract_merge"]["commit"]
    )
    assert protocol_merge["contract_merged_at_utc"] == (
        activation["contract_merge"]["merged_at_utc"]
    )
    assert activation["collector_implementation_activation"] == {
        "commit": None,
        "activated_at_utc": None,
    }
    assert activation["first_eligible_ex_ante_decision"] == {
        "record_id": None,
        "economic_date": None,
        "recorded_at_utc": None,
    }
    assert activation["activation_event_recorded"] is False
    assert protocol["activation_state"]["oos_collection_started"] is False
    assert "evaluation-protocol-and-maturity-thresholds" not in blocker_ids
    assert "contract-merge-fact" not in blocker_ids
    assert blocker_ids == {
        "collector-implementation-and-activation",
        "first-eligible-ex-ante-decision",
    }


def test_current_manifest_and_protocol_document_do_not_claim_merge_is_future() -> None:
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8").casefold()
    design_text = DESIGN_PATH.read_text(encoding="utf-8").casefold()
    current_text = manifest_text + "\n" + design_text

    for stale_claim in (
        "merge commit and timestamp do not exist yet",
        "merge commit and timestamp remain null",
        "pr #18 contract merge: unresolved",
        "future pr #18 merge",
    ):
        assert stale_claim not in current_text
    assert CONTRACT_MERGE_COMMIT in current_text
    assert CONTRACT_MERGED_AT_UTC.casefold() in current_text


def test_protocol_contract_is_design_only_and_does_not_read_mutable_outputs() -> None:
    protocol = load_protocol()
    design = DESIGN_PATH.read_text(encoding="utf-8")
    scope = protocol["scope"]

    assert scope["design_and_configuration_only"] is True
    assert scope["collector_implemented"] is False
    assert scope["collector_activated"] is False
    assert scope["oos_ledger_created"] is False
    assert scope["oos_outcomes_inspected"] is False
    assert "OOS has still not started" in design

    test_source = Path(__file__).read_text(encoding="utf-8")
    assert ('ROOT / "docs" / ' + '"data"') not in test_source
    assert ("pandas." + "read_csv") not in test_source
    assert ("backtest_" + "summary.json").casefold() not in test_source.casefold()
