"""Synthetic-only demonstration of all Foundation 5 control boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_trend_v2_foundation_5 import Fixture, construction  # noqa: E402
from src.trend_v2_foundation import Foundation5Error, StrategyRunSpec  # noqa: E402


def main() -> None:
    fixture = Fixture()
    try:
        profile_ids = list(fixture.profiles)
        report: dict[str, object] = {"schema_version": "foundation_5_synthetic_demo_v1"}

        options = fixture.api.dispatch("GET", "/api/v1/construction/options").body
        report["1_ui_supported_options"] = {
            "loaded": options["schema_version"] == "controlled_strategy_options_v1",
            "maximum_concurrency": options["execution_policy"]["maximum_concurrency"],
        }

        fixed = construction(profile_ids[:1])
        fixed_first = fixture.service.normalize(fixed)
        fixed_second = fixture.service.normalize(fixed)
        report["2_fixed_normalization"] = {
            "deterministic": fixed_first.construction_hash == fixed_second.construction_hash,
            "hash": fixed_first.construction_hash,
        }

        multiple = construction(
            profile_ids[:2],
            cost={"kind": "list", "values": ["0", "5"]},
        )
        _, multiple_estimate, candidates = fixture.service.estimate(multiple)
        report["3_exact_multi_parameter_count"] = {
            "raw": multiple_estimate.raw_cartesian_candidate_count,
            "deduplicated": multiple_estimate.economic_strategy_run_candidate_count,
            "ordered_candidate_ids": [item.strategy_run_id for item in candidates],
        }
        report["4_economic_vs_evaluation"] = {
            "economic": multiple_estimate.economic_strategy_run_candidate_count,
            "evaluation_applications": multiple_estimate.evaluation_profile_application_count,
            "derived_calculations": multiple_estimate.derived_metric_calculation_count,
        }

        small_request = fixture.service.create_request(
            fixed, confirmation_id=None, idempotency_key="demo-small-request"
        )
        small_start = fixture.service.start(
            small_request.execution_request_id, idempotency_key="demo-small-start"
        )
        small_attempt = fixture.attempts.get(small_start["execution_attempt_ids"][0])
        report["5_small_without_confirmation"] = {
            "confirmation_id": small_request.confirmation_id,
            "attempt_status": small_attempt.operational_status.value,
        }

        large = construction(
            profile_ids[:1],
            cost={"kind": "list", "values": ["0", "1", "2", "3", "4", "5"]},
        )
        _, large_estimate, _ = fixture.service.estimate(large)
        large_confirmation = fixture.service.confirm(
            large, idempotency_key="demo-large-confirmation"
        )
        report["6_large_requires_confirmation"] = {
            "required": large_estimate.confirmation_required,
            "confirmation_id": large_confirmation.confirmation_id,
        }

        changed = construction(
            profile_ids[:1],
            cost={"kind": "list", "values": ["0", "1", "2", "3", "4"]},
        )
        stale_code = None
        try:
            fixture.service.create_request(
                changed,
                confirmation_id=large_confirmation.confirmation_id,
                idempotency_key="demo-stale-request",
            )
        except Foundation5Error as error:
            stale_code = error.code
        report["7_change_invalidates_confirmation"] = stale_code

        hard = construction(
            profile_ids[:1],
            cost={"kind": "list", "values": [str(value) for value in range(8)]},
            folds=20,
            scenarios=20,
        )
        _, hard_estimate, _ = fixture.service.estimate(hard)
        report["8_hard_limit_rejected"] = {
            "hard_limit_exceeded": hard_estimate.hard_limit_exceeded,
            "triggered": [
                item["threshold"] for item in hard_estimate.threshold_results if item["triggered"]
            ],
        }

        repeated_start = fixture.service.start(
            small_request.execution_request_id,
            idempotency_key="demo-small-start-repeat",
        )
        report["9_start_idempotency"] = {
            "same_attempt_ids": repeated_start["execution_attempt_ids"] == small_start["execution_attempt_ids"],
            "economic_adapter_calls": len(fixture.adapter.calls),
        }

        profile_request = fixture.service.create_request(
            construction(profile_ids[:2]),
            confirmation_id=None,
            idempotency_key="demo-profile-request",
        )
        profile_start = fixture.service.start(
            profile_request.execution_request_id,
            idempotency_key="demo-profile-start",
        )
        profile_attempt = fixture.attempts.get(profile_start["execution_attempt_ids"][0])
        report["10_equivalent_completed_reuse"] = profile_attempt.progress_summary["reused_count"] == 1
        report["11_two_profiles_one_economic_path"] = {
            "economic_adapter_calls": len(fixture.adapter.calls),
            "evaluation_run_count": profile_attempt.progress_summary["evaluation_run_count"],
        }
        run_id = StrategyRunSpec.from_dict(
            profile_request.requested_strategy_run_candidates[0]
        ).strategy_run_id
        report["12_attempt_separate_from_run"] = {
            "attempt_operational_status": profile_attempt.operational_status.value,
            "strategy_run_terminal_status": fixture.store.get_strategy_run_manifest(run_id).execution_status.value,
        }

        fixture.adapter.fail_bps.add("10")
        mixed = construction(
            profile_ids[:1],
            cost={"kind": "list", "values": ["5", "10"]},
        )
        mixed_request = fixture.service.create_request(
            mixed, confirmation_id=None, idempotency_key="demo-mixed-request"
        )
        mixed_start = fixture.service.start(
            mixed_request.execution_request_id, idempotency_key="demo-mixed-start"
        )
        mixed_attempts = [fixture.attempts.get(item) for item in mixed_start["execution_attempt_ids"]]
        report["13_failure_isolation"] = {
            "statuses": sorted(item.operational_status.value for item in mixed_attempts),
            "successful_strategy_runs": len(fixture.store.strategy_run_history()),
        }
        failed = next(item for item in mixed_attempts if item.operational_status.value == "failed")
        fixture.adapter.fail_bps.clear()
        retry = fixture.service.retry(failed.execution_attempt_id, idempotency_key="demo-retry")
        retry_final = fixture.attempts.get(retry.execution_attempt_id)
        report["14_retry_new_attempt"] = {
            "new_identity": retry_final.execution_attempt_id != failed.execution_attempt_id,
            "retry_parent": retry_final.retry_of_execution_attempt_id,
            "status": retry_final.operational_status.value,
        }
        report["15_bounded_local_only"] = {
            "maximum_concurrency": fixture.service.policy.maximum_concurrency,
            "remote_fetch": False,
            "dynamic_python": False,
            "unrestricted_search": False,
            "supported_signal_ids": [item["option_id"] for item in options["signal"]],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        fixture.close()


if __name__ == "__main__":
    main()
