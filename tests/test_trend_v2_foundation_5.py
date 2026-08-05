from __future__ import annotations

import json
import itertools
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.trend_v2_foundation import (
    AdapterArtifact,
    AdapterResult,
    ArtifactKind,
    ArtifactRetentionPolicy,
    ControlledExecutionService,
    ExecutionStatus,
    FileExecutionAttemptRepository,
    Foundation5Error,
    LocalResultStore,
    ReadOnlyTrendApi,
    StrategyRunSpec,
    load_evaluation_profiles,
    load_execution_policy,
    load_terminology_source,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "trend_v2" / "local_execution_policy_v1.json"
PROFILE_DIR = ROOT / "config" / "trend_v2" / "evaluation_profiles"
TERMINOLOGY_PATH = ROOT / "config" / "trend_v2" / "terminology_ko.json"
CREATED = "2026-08-02T00:00:00Z"


def result_policy() -> ArtifactRetentionPolicy:
    return ArtifactRetentionPolicy(
        max_store_bytes=50_000_000,
        max_artifact_bytes=5_000_000,
        max_strategy_runs=100,
        max_evaluation_runs=100,
    )


def profiles():
    loaded = load_evaluation_profiles(PROFILE_DIR)
    return {profile.evaluation_profile_id: profile for profile in loaded.values()}


def construction(
    profile_ids: list[str],
    *,
    cost=5,
    slippage=2,
    folds: int = 0,
    scenarios: int = 0,
) -> dict:
    return {
        "schema_version": "strategy_construction_request_v1",
        "data_snapshot": "phase_a2_frozen_2026_07_30",
        "backtest_start_date": "2024-01-02",
        "backtest_end_date": "2024-05-06",
        "universe": {"option_id": "phase_a2_historical_eligible_v1", "parameters": {}},
        "benchmark": {"option_id": "spy_adjusted_close_v1", "parameters": {}},
        "trend_filter": {"option_id": "price_above_rising_ma200_v0", "parameters": {}},
        "signal": {
            "option_id": "prior_price_high_l20_v1",
            "parameters": {"lookback": {"kind": "fixed", "value": 20}},
        },
        "entry_rule": {"option_id": "first_event_next_open_v1", "parameters": {}},
        "initial_stop": {"option_id": "signal_day_low20_v1", "parameters": {}},
        "trailing_exit": {"option_id": "ratcheting_low20_v1", "parameters": {}},
        "position_sizing": {"option_id": "canonical_equal_weight_active_v1", "parameters": {}},
        "portfolio_constraints": {"option_id": "long_only_cash_constrained_v1", "parameters": {}},
        "transaction_cost": {"option_id": "round_trip_bps_v1", "parameters": {"bps": cost}},
        "slippage": {"option_id": "round_trip_slippage_bps_v1", "parameters": {"bps": slippage}},
        "walk_forward": {"enabled": folds > 0, "fold_count": folds},
        "robustness": {"scenario_count": scenarios},
        "evaluation_profile_ids": profile_ids,
    }


def curve(specification: StrategyRunSpec, daily_return: float) -> dict:
    dates = pd.bdate_range(
        specification.economic_date_range["start"],
        specification.economic_date_range["end"],
    )
    values = []
    value = 1000.0
    rows = []
    for index, timestamp in enumerate(dates):
        observed = 0.0 if index == 0 else daily_return
        value *= 1.0 + observed
        values.append(value)
        rows.append(
            {
                "economic_date": timestamp.date().isoformat(),
                "portfolio_value": value,
                "daily_return": observed,
                "gross_exposure": 0.8,
                "net_exposure": 0.8,
                "cash_weight": 0.2,
                "daily_turnover": 0.01,
                "transaction_cost": 0.0001,
                "position_count": 3,
            }
        )
    return {
        "schema_version": "daily_portfolio_curve_v1",
        "economic_date_range": dict(specification.economic_date_range),
        "rows": rows,
    }


class SyntheticAdapter:
    engine_version = "trend_v2_phase_a_controlled_adapter_v1"

    def __init__(self, fail_bps: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_bps = fail_bps or set()

    def execute(self, specification: StrategyRunSpec) -> AdapterResult:
        self.calls.append(specification.strategy_run_id)
        bps = str(specification.transaction_costs["parameters"]["bps"])
        if bps in self.fail_bps:
            raise Foundation5Error("internal_execution_failure", "Synthetic candidate failure.")
        daily = curve(specification, 0.0005)
        benchmark = curve(specification, 0.0003)
        return AdapterResult(
            artifacts=(
                AdapterArtifact("daily_portfolio_curve", ArtifactKind.DAILY_PORTFOLIO_CURVE, daily, len(daily["rows"])),
                AdapterArtifact("benchmark_daily_portfolio_curve", ArtifactKind.DAILY_PORTFOLIO_CURVE, benchmark, len(benchmark["rows"])),
                AdapterArtifact("trade_lifecycles", ArtifactKind.TRADE_LIFECYCLES, {"schema_version": "trade_lifecycles_v1", "rows": []}, 0),
                AdapterArtifact("signal_execution_events", ArtifactKind.SIGNAL_EXECUTION_EVENTS, {"schema_version": "signal_execution_events_v1", "rows": []}, 0),
            )
        )


class Fixture:
    def __init__(self, adapter: SyntheticAdapter | None = None, *, policy=None) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = LocalResultStore(self.temp.name, result_policy())
        self.attempts = FileExecutionAttemptRepository(Path(self.temp.name) / "execution_attempts")
        self.adapter = adapter or SyntheticAdapter()
        self.profiles = profiles()
        ticks = itertools.count()
        def clock() -> str:
            value = datetime(2026, 8, 2, tzinfo=timezone.utc) + timedelta(seconds=next(ticks))
            return value.isoformat().replace("+00:00", "Z")
        self.service = ControlledExecutionService(
            self.store,
            self.attempts,
            self.adapter,
            policy or load_execution_policy(POLICY_PATH),
            self.profiles,
            source_commit="c" * 40,
            clock=clock,
            background=False,
        )
        self.api = ReadOnlyTrendApi(
            self.store,
            attempt_repository=self.attempts,
            terminology_source=load_terminology_source(TERMINOLOGY_PATH),
            controlled_execution_service=self.service,
        )

    def close(self) -> None:
        self.service.close()
        self.temp.cleanup()


class ConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()
        self.profile_ids = list(self.fixture.profiles)

    def tearDown(self) -> None:
        self.fixture.close()

    def test_canonical_scalar_list_and_singleton_range_normalize_equivalently(self) -> None:
        requests = [
            construction(self.profile_ids[:1], cost=5),
            construction(self.profile_ids[:1], cost={"kind": "list", "values": ["5"]}),
            construction(
                self.profile_ids[:1],
                cost={"kind": "decimal_range", "start": "5.0", "end": "5.00", "step": "0.5"},
            ),
        ]
        normalized = [self.fixture.service.normalize(item) for item in requests]
        self.assertEqual(len({item.construction_hash for item in normalized}), 1)
        self.assertEqual(normalized[0].normalized["transaction_cost"]["parameters"]["bps"], ("5",))

    def test_decimal_range_is_exact_and_candidate_order_is_deterministic(self) -> None:
        request = construction(
            self.profile_ids[:1],
            cost={"kind": "decimal_range", "start": "0", "end": "0.3", "step": "0.1"},
        )
        _, estimate, candidates = self.fixture.service.estimate(request)
        self.assertEqual(estimate.raw_cartesian_candidate_count, 4)
        self.assertEqual(estimate.economic_strategy_run_candidate_count, 4)
        self.assertEqual(
            [item.strategy_run_id for item in candidates],
            sorted(item.strategy_run_id for item in candidates),
        )
        self.assertEqual(
            {item.transaction_costs["parameters"]["bps"] for item in candidates},
            {"0", "0.1", "0.2", "0.3"},
        )

    def test_duplicate_values_and_nonterminating_or_nonpositive_ranges_are_rejected(self) -> None:
        invalid = [
            {"kind": "list", "values": ["5", "5.0"]},
            {"kind": "decimal_range", "start": "0", "end": "1", "step": "0.3"},
            {"kind": "decimal_range", "start": "0", "end": "1", "step": "0"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(Foundation5Error):
                self.fixture.service.normalize(construction(self.profile_ids[:1], cost=value))

    def test_exact_workload_separates_economic_evaluation_robustness_and_derived(self) -> None:
        request = construction(
            self.profile_ids[:2],
            cost={"kind": "list", "values": ["0", "5"]},
            folds=3,
            scenarios=2,
        )
        _, estimate, _ = self.fixture.service.estimate(request)
        self.assertEqual(estimate.economic_strategy_run_candidate_count, 2)
        self.assertEqual(estimate.evaluation_profile_application_count, 4)
        self.assertEqual(estimate.walk_forward_fold_execution_count, 6)
        self.assertEqual(estimate.robustness_scenario_count, 4)
        self.assertEqual(estimate.benchmark_calculation_count, 2)
        self.assertEqual(estimate.derived_metric_calculation_count, 2)
        self.assertEqual(estimate.estimated_total_execution_units, 20)

    def test_confirmation_binding_staleness_and_hard_refusal(self) -> None:
        large = construction(
            self.profile_ids[:1],
            cost={"kind": "list", "values": ["0", "1", "2", "3", "4", "5"]},
        )
        normalized, estimate, _ = self.fixture.service.estimate(large)
        self.assertTrue(estimate.confirmation_required)
        confirmation = self.fixture.service.confirm(large, idempotency_key="confirm-large")
        changed = construction(
            self.profile_ids[:1],
            cost={"kind": "list", "values": ["0", "1", "2", "3", "4"]},
        )
        with self.assertRaisesRegex(Foundation5Error, "confirmation_stale"):
            changed_normalized, changed_estimate, changed_candidates = self.fixture.service.estimate(changed)
            from src.trend_v2_foundation import create_execution_request

            create_execution_request(
                changed_normalized,
                changed_estimate,
                changed_candidates,
                self.fixture.service.policy,
                request_timestamp=CREATED,
                source_commit="c" * 40,
                confirmation=confirmation,
            )
        hard = construction(
            self.profile_ids[:1],
            cost={"kind": "list", "values": [str(value) for value in range(8)]},
            folds=20,
            scenarios=20,
        )
        _, hard_estimate, _ = self.fixture.service.estimate(hard)
        self.assertTrue(hard_estimate.hard_limit_exceeded)
        with self.assertRaisesRegex(Foundation5Error, "hard_limit_exceeded"):
            self.fixture.service.confirm(hard, idempotency_key="hard")

    def test_candidate_overflow_protection(self) -> None:
        limited = replace(self.fixture.service.policy, candidate_count_overflow_limit=5)
        other = Fixture(policy=limited)
        try:
            request = construction(
                list(other.profiles)[:1],
                cost={"kind": "list", "values": ["0", "1", "2", "3", "4", "5"]},
            )
            with self.assertRaisesRegex(Foundation5Error, "candidate_estimate_overflow"):
                other.service.estimate(request)
        finally:
            other.close()

    def test_parameter_dimension_limit_is_checked_before_expansion(self) -> None:
        request = construction(
            self.profile_ids[:1],
            cost={
                "kind": "decimal_range",
                "start": "0",
                "end": "1000000000",
                "step": "0.000000001",
            },
        )
        with self.assertRaisesRegex(Foundation5Error, "hard_limit_exceeded"):
            self.fixture.service.normalize(request)

    def test_confirmation_expiry_and_one_time_use(self) -> None:
        confirmation_policy = replace(
            self.fixture.service.policy,
            informational_threshold=2,
            explicit_confirmation_threshold=4,
        )
        other = Fixture(policy=confirmation_policy)
        try:
            request = construction(list(other.profiles)[:1])
            confirmation = other.service.confirm(request, idempotency_key="one-time-confirm")
            first = other.service.create_request(
                request,
                confirmation_id=confirmation.confirmation_id,
                idempotency_key="one-time-first",
            )
            second = other.service.create_request(
                request,
                confirmation_id=confirmation.confirmation_id,
                idempotency_key="one-time-second",
            )
            other.service.start(first.execution_request_id, idempotency_key="one-time-start")
            with self.assertRaisesRegex(Foundation5Error, "confirmation_invalid"):
                other.service.start(second.execution_request_id, idempotency_key="one-time-reuse")
        finally:
            other.close()

        expiring_policy = replace(
            self.fixture.service.policy,
            informational_threshold=2,
            explicit_confirmation_threshold=4,
            confirmation_ttl_seconds=1,
        )
        expiring = Fixture(policy=expiring_policy)
        try:
            request = construction(list(expiring.profiles)[:1])
            confirmation = expiring.service.confirm(request, idempotency_key="expiry-confirm")
            accepted = expiring.service.create_request(
                request,
                confirmation_id=confirmation.confirmation_id,
                idempotency_key="expiry-request",
            )
            with self.assertRaisesRegex(Foundation5Error, "confirmation_invalid"):
                expiring.service.start(accepted.execution_request_id, idempotency_key="expiry-start")
        finally:
            expiring.close()


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()
        self.profile_ids = list(self.fixture.profiles)

    def tearDown(self) -> None:
        self.fixture.close()

    def _run(self, request: dict, key: str = "one"):
        execution = self.fixture.service.create_request(
            request, confirmation_id=None, idempotency_key=f"request-{key}"
        )
        start = self.fixture.service.start(execution.execution_request_id, idempotency_key=f"start-{key}")
        return execution, start

    def test_small_request_stores_terminal_run_evaluation_and_separate_attempt(self) -> None:
        execution, start = self._run(construction(self.profile_ids[:1]))
        attempt = self.fixture.attempts.get(start["execution_attempt_ids"][0])
        candidate = StrategyRunSpec.from_dict(execution.requested_strategy_run_candidates[0])
        manifest = self.fixture.store.get_strategy_run_manifest(candidate.strategy_run_id)
        self.assertEqual(attempt.operational_status.value, "completed")
        self.assertEqual(manifest.execution_status, ExecutionStatus.SUCCEEDED)
        self.assertNotIn(attempt.operational_status.value, {"succeeded", "partial"})
        self.assertEqual(len(self.fixture.store.evaluation_history()), 1)
        self.assertEqual(len(self.fixture.adapter.calls), 1)

    def test_unimplemented_fold_and_robustness_execution_is_refused(self) -> None:
        for suffix, request in (
            ("fold", construction(self.profile_ids[:1], folds=2)),
            ("robustness", construction(self.profile_ids[:1], scenarios=1)),
        ):
            with self.subTest(kind=suffix), self.assertRaisesRegex(
                Foundation5Error, "engine_unsupported"
            ):
                self.fixture.service.create_request(
                    request,
                    confirmation_id=None,
                    idempotency_key=f"unsupported-{suffix}",
                )

    def test_start_is_idempotent_and_profile_only_change_reuses_economic_artifacts(self) -> None:
        first, first_start = self._run(construction(self.profile_ids[:1]), "first")
        repeated = self.fixture.service.start(first.execution_request_id, idempotency_key="start-first-repeat")
        self.assertEqual(repeated, first_start)
        second, second_start = self._run(construction(self.profile_ids[:2]), "second")
        attempt = self.fixture.attempts.get(second_start["execution_attempt_ids"][0])
        self.assertEqual(attempt.progress_summary["reused_count"], 1)
        self.assertEqual(len(self.fixture.adapter.calls), 1)
        self.assertEqual(
            StrategyRunSpec.from_dict(first.requested_strategy_run_candidates[0]).strategy_run_id,
            StrategyRunSpec.from_dict(second.requested_strategy_run_candidates[0]).strategy_run_id,
        )

    def test_per_candidate_failure_isolated_and_retry_has_new_identity(self) -> None:
        self.fixture.adapter.fail_bps.add("10")
        request = construction(
            self.profile_ids[:1],
            cost={"kind": "list", "values": ["5", "10"]},
        )
        execution, start = self._run(request, "mixed")
        attempts = [self.fixture.attempts.get(item) for item in start["execution_attempt_ids"]]
        self.assertEqual({item.operational_status.value for item in attempts}, {"completed", "failed"})
        self.assertEqual(len(self.fixture.store.strategy_run_history()), 1)
        failed = next(item for item in attempts if item.operational_status.value == "failed")
        self.fixture.adapter.fail_bps.clear()
        retry = self.fixture.service.retry(failed.execution_attempt_id, idempotency_key="retry-mixed")
        completed_retry = self.fixture.attempts.get(retry.execution_attempt_id)
        self.assertNotEqual(completed_retry.execution_attempt_id, failed.execution_attempt_id)
        self.assertEqual(completed_retry.retry_of_execution_attempt_id, failed.execution_attempt_id)
        self.assertEqual(completed_retry.operational_status.value, "completed")
        self.assertEqual(len(self.fixture.store.strategy_run_history()), 2)

    def test_corrupt_equivalent_run_is_not_reused(self) -> None:
        execution, _ = self._run(construction(self.profile_ids[:1]), "valid")
        candidate = StrategyRunSpec.from_dict(execution.requested_strategy_run_candidates[0])
        record = self.fixture.store.get_strategy_artifact_record(candidate.strategy_run_id, "daily_portfolio_curve")
        self.fixture.store.object_path_for_hash(record.content_hash).write_bytes(b"corrupt")
        second, start = self._run(construction(self.profile_ids[:1]), "corrupt")
        attempt = self.fixture.attempts.get(start["execution_attempt_ids"][0])
        self.assertEqual(attempt.operational_status.value, "failed")
        self.assertEqual(attempt.failure_code, "stored_equivalent_run_corrupt")

    def test_cancel_and_retry_state_validation(self) -> None:
        execution, start = self._run(construction(self.profile_ids[:1]), "terminal")
        attempt_id = start["execution_attempt_ids"][0]
        with self.assertRaisesRegex(Foundation5Error, "attempt_not_cancellable"):
            self.fixture.service.cancel(attempt_id, idempotency_key="cancel-terminal")
        with self.assertRaisesRegex(Foundation5Error, "retry_not_allowed"):
            self.fixture.service.retry(attempt_id, idempotency_key="retry-terminal")


class ApiAndUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()
        self.profile_id = next(iter(self.fixture.profiles))

    def tearDown(self) -> None:
        self.fixture.close()

    def test_controlled_api_validation_idempotency_and_korean_errors(self) -> None:
        request = construction([self.profile_id])
        options = self.fixture.api.dispatch("GET", "/api/v1/construction/options")
        self.assertEqual(options.status_code, 200)
        self.assertEqual(options.body["execution_policy"]["maximum_concurrency"], 1)
        estimate = self.fixture.api.dispatch("POST", "/api/v1/construction/estimate", body=request)
        self.assertEqual(estimate.status_code, 200)
        created = self.fixture.api.dispatch(
            "POST",
            "/api/v1/execution-requests",
            headers={"Idempotency-Key": "api-request"},
            body={"construction": request, "confirmation_id": None},
        )
        repeated = self.fixture.api.dispatch(
            "POST",
            "/api/v1/execution-requests",
            headers={"Idempotency-Key": "api-request"},
            body={"construction": request, "confirmation_id": None},
        )
        self.assertEqual(created.body, repeated.body)
        invalid = self.fixture.api.dispatch("POST", "/api/v1/construction/estimate", body={"path": "C:\\secret"})
        self.assertEqual(invalid.body["error"]["code"], "invalid_construction_field")
        self.assertTrue(invalid.body["error"]["message_ko"])
        self.assertNotIn(str(self.fixture.store.root), json.dumps(invalid.body, ensure_ascii=False))

    def test_candidate_estimate_uses_one_envelope_for_foundation_6_requests(self) -> None:
        response = self.fixture.api.dispatch(
            "POST",
            "/api/v1/construction/estimate",
            body={
                "catalog_schema_version": "controlled_strategy_option_catalog_v2",
                "components": {"signal": {"option_id": "prior_price_high_v2", "parameters": {"lookback": {"kind": "fixed", "value": 20}}}},
                "evaluation_profile_ids": ["risk"],
                "history_sessions": 252,
                "universe_size": 470,
                "asset_group_data_available": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("normalized_construction", response.body)
        self.assertIn("candidate_estimate", response.body)
        self.assertNotIn("normalized_construction", response.body["candidate_estimate"])
        self.assertEqual(response.body["strategy_run_candidate_ids"], response.body["candidate_estimate"]["candidate_economic_hashes"])

    def test_request_size_path_traversal_and_unsupported_methods_fail_closed(self) -> None:
        maximum = self.fixture.service.policy.maximum_json_body_bytes
        oversized = self.fixture.api.dispatch(
            "POST", "/api/v1/construction/estimate", body=b"x" * (maximum + 1)
        )
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(oversized.body["error"]["code"], "request_too_large")
        traversal = self.fixture.api.dispatch("GET", "/api/v1/execution-requests/..%2Fsecret")
        self.assertEqual(traversal.status_code, 400)
        delete = self.fixture.api.dispatch("DELETE", "/api/v1/execution-requests")
        self.assertEqual(delete.status_code, 405)

    def test_korean_ui_contains_builder_preview_confirmation_progress_and_noncolor_status(self) -> None:
        html = (ROOT / "src" / "trend_v2_foundation" / "ui_assets" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "src" / "trend_v2_foundation" / "ui_assets" / "app.js").read_text(encoding="utf-8")
        combined = html + javascript
        for label in (
            "전략 구성", "실행 요청", "백테스트 시작일", "데이터 스냅샷", "유니버스",
            "벤치마크", "추세 필터", "신호", "진입 규칙", "초기 손절", "추적 청산",
            "포지션 크기", "포트폴리오 제약", "거래비용", "슬리피지", "강건성",
            "평가 프로필", "원시 Cartesian 후보", "중복 제거 경제 후보", "총 실행 단위",
            "명시적으로 확인", "ExecutionRequest", "ExecutionAttempt", "재시도",
        ):
            self.assertIn(label, combined)
        self.assertIn("statusBadge", javascript)
        self.assertIn("aria-hidden=\"true\"", javascript)
        self.assertNotIn("strategy score", javascript.casefold())
        self.assertNotRegex(combined, r"[A-Za-z]:[\\/]")


if __name__ == "__main__":
    unittest.main()
