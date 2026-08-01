from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from src.trend_v2_foundation import (
    TrendWebApplication,
    build_web_server,
    load_terminology_source,
)
from tests.test_trend_v2_foundation_3 import Foundation3Fixture, TERMINOLOGY_PATH


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "src" / "trend_v2_foundation" / "ui_assets"
REQUIRED_EXPLANATIONS = {
    "cagr",
    "cagr_spy_ratio",
    "annualized_volatility",
    "sharpe_ratio",
    "hac_adjusted_sharpe",
    "sortino_ratio",
    "maximum_drawdown",
    "maximum_drawdown_spy_ratio",
    "cdar95",
    "cdar95_spy_ratio",
    "calmar_ratio",
    "calmar_spy_ratio",
    "recovery_duration",
    "time_under_water",
    "rolling_returns",
    "expected_shortfall",
    "downside_deviation",
    "turnover",
    "transaction_cost_drag",
    "gross_exposure",
    "net_exposure",
    "cash_weight",
    "position_count",
    "walk_forward",
    "loyo",
    "paired_block_bootstrap",
    "confidence_interval",
    "raw_p_value",
    "adjusted_p_value",
    "holm_correction",
    "white_reality_check",
    "hansen_spa",
    "deflated_sharpe_ratio",
    "pbo",
    "cost_stress",
    "asset_group_concentration",
    "pareto_dominance",
    "epsilon_dominance",
    "mandatory_gate",
    "robustness_veto",
    "lexicographic_tie_break",
    "behavior_deduplication",
    "exploratory_weighted_value",
}


class Foundation4UiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Foundation3Fixture()
        self.application = TrendWebApplication(self.fixture.api)
        self.html = (ASSET_ROOT / "index.html").read_text(encoding="utf-8")
        self.javascript = (ASSET_ROOT / "app.js").read_text(encoding="utf-8")
        self.css = (ASSET_ROOT / "style.css").read_text(encoding="utf-8")

    def tearDown(self) -> None:
        self.fixture.close()

    def test_korean_first_navigation_headings_and_read_only_scope(self) -> None:
        for label in (
            "개요",
            "저장된 전략 실행",
            "평가 결과",
            "성과 및 위험",
            "강건성",
            "행동 유사도",
            "실행 이력",
            "설명",
            "시스템 정보",
        ):
            self.assertIn(label, self.html)
        self.assertIn("읽기 전용", self.html)
        for forbidden in ("전략 실행 시작", "취소 실행", "재시도 실행", "worker-control"):
            self.assertNotIn(forbidden, self.html + self.javascript)

    def test_static_assets_are_same_origin_fixed_and_security_hardened(self) -> None:
        page = self.application.dispatch("GET", "/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["Content-Type"])
        self.assertIn("default-src 'self'", page.headers["Content-Security-Policy"])
        self.assertIn("connect-src 'self'", page.headers["Content-Security-Policy"])
        self.assertEqual(self.application.dispatch("GET", "/assets/app.js").status_code, 200)
        self.assertEqual(self.application.dispatch("GET", "/../secret").status_code, 404)
        self.assertEqual(self.application.dispatch("POST", "/").status_code, 405)
        self.assertNotIn("http://", self.javascript)
        self.assertNotIn("https://", self.javascript)

    def test_overview_contains_counts_versions_and_registry_identity(self) -> None:
        response = self.fixture.api.dispatch("GET", "/api/v1/overview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body["strategy_run_count"], 2)
        self.assertEqual(response.body["evaluation_profile_count"], 2)
        self.assertEqual(response.body["evaluation_run_count"], 2)
        self.assertGreater(response.body["artifact_availability_counts"]["available"], 0)
        self.assertEqual(response.body["execution_attempt_status_counts"]["completed"], 1)
        self.assertIn("metric_registry_version", response.body["versions"])
        self.assertIn("registry_id", response.body["last_registry_rebuild_identity"])
        self.assertIn("프로덕션", response.body["evidence_quality_note_ko"])

    def test_saved_run_listing_is_deterministic_paginated_and_has_artifact_summary(self) -> None:
        first = self.fixture.api.dispatch("GET", "/api/v1/runs?page_size=1&sort=-creation_time")
        repeated = self.fixture.api.dispatch("GET", "/api/v1/runs?page_size=1&sort=-creation_time")
        self.assertEqual(first.body, repeated.body)
        self.assertIsNotNone(first.body["page"]["next_cursor"])
        item = first.body["items"][0]
        self.assertIn("available_artifact_keys", item)
        self.assertIn("artifact_availability_counts", item)
        self.assertIn("benchmark_identity", item)
        invalid = self.fixture.api.dispatch("GET", "/api/v1/runs?sort=made_up")
        self.assertEqual(invalid.status_code, 400)
        self.assertTrue(invalid.body["error"]["message_ko"])

    def test_run_detail_and_all_curve_reads_are_bounded(self) -> None:
        run_id = self.fixture.first.strategy_run_id
        detail = self.fixture.api.dispatch("GET", f"/api/v1/runs/{run_id}")
        curve = self.fixture.api.dispatch(
            "GET", f"/api/v1/runs/{run_id}/curve?page_size=250"
        )
        benchmark = self.fixture.api.dispatch(
            "GET", f"/api/v1/runs/{run_id}/benchmark-curve?page_size=250"
        )
        behavior = self.fixture.api.dispatch(
            "GET", f"/api/v1/runs/{run_id}/behavior-summary"
        )
        self.assertEqual(detail.status_code, 200)
        self.assertLessEqual(len(curve.body["items"]), 250)
        self.assertLessEqual(len(benchmark.body["items"]), 250)
        self.assertNotIn("comparison_inputs", behavior.body["payload"])
        self.assertEqual(behavior.body["payload"]["comparison_input_counts"]["economic_date_count"], 90)
        self.assertIn("CURVE_PAGE_SIZE = 250", self.javascript)
        self.assertIn('key:"rolling_return"', self.javascript)
        self.assertIn('key:"annual_return"', self.javascript)
        self.assertIn("if (value === null || value === undefined", self.javascript)
        self.assertNotRegex(self.javascript, r"/curve(?:`|\")\s*[),]")

    def test_evaluation_stages_two_profiles_hash_and_weighted_terminology(self) -> None:
        run_id = self.fixture.first.strategy_run_id
        evaluations = self.fixture.api.dispatch(
            "GET", f"/api/v1/evaluation-runs?strategy_run_id={run_id}"
        )
        self.assertEqual(len(evaluations.body["items"]), 2)
        self.assertTrue(all(item["profile_hash"] for item in evaluations.body["items"]))
        for token in (
            "mandatory gates",
            "epsilon-Pareto",
            "robustness veto",
            "lexicographic tie-break",
            "behavior deduplication",
            "exploratory_weighted_value",
        ):
            self.assertIn(token, self.javascript)
        self.assertIn("profile hash", self.javascript)
        self.assertNotIn("strategy score", self.javascript.casefold())
        self.assertNotIn("전략 점수", self.javascript)

    def test_mandatory_gate_and_missing_robustness_reason_are_visible(self) -> None:
        evaluation_id = self.fixture.research.evaluation_run.evaluation_run_id
        output = self.fixture.api.dispatch(
            "GET", f"/api/v1/evaluation-runs/{evaluation_id}/outputs"
        ).body["items"][0]
        gate = output["mandatory_gates"][0]
        for field in ("metric_key", "operator", "threshold", "value", "passed", "reason"):
            self.assertIn(field, gate)
        self.assertIn("evidence.__error.code", self.javascript)
        self.assertIn("근거 누락", self.javascript)
        self.assertIn("사용 설정되지 않음", self.javascript)
        self.assertIn("해당 없음", self.javascript)

    def test_artifact_states_and_execution_attempt_separation(self) -> None:
        terminology = self.fixture.api.dispatch("GET", "/api/v1/terminology").body
        labels = terminology["status_labels"]
        for state in (
            "available",
            "missing",
            "pruned",
            "corrupt",
            "never_generated",
            "unsupported_schema",
            "integrity_failed",
        ):
            self.assertTrue(labels[state])
            self.assertIn(state, self.javascript)
        attempts = self.fixture.api.dispatch("GET", "/api/v1/execution-attempts").body
        manifest = self.fixture.api.dispatch(
            "GET", f"/api/v1/runs/{self.fixture.first.strategy_run_id}/manifest"
        ).body
        self.assertEqual(attempts["items"][0]["operational_status"], "completed")
        self.assertEqual(manifest["execution_status"], "succeeded")
        self.assertIn("StrategyRun의 불변 경제 결과", self.javascript)

    def test_api_error_mapping_and_assets_do_not_leak_absolute_paths(self) -> None:
        invalid = self.fixture.api.dispatch("GET", "/api/v1/runs?unknown=value")
        self.assertEqual(invalid.body["error"]["code"], "invalid_query")
        self.assertTrue(invalid.body["error"]["message_ko"])
        combined = self.html + self.javascript + self.css
        self.assertNotRegex(combined, r"[A-Za-z]:[\\/]")
        self.assertNotIn("Traceback", combined)
        self.assertIn("textContent", self.javascript)
        self.assertIn("escapeHtml", self.javascript)

    def test_explanation_source_is_complete_and_formula_rendering_is_safe(self) -> None:
        source = load_terminology_source(TERMINOLOGY_PATH)
        self.assertTrue(REQUIRED_EXPLANATIONS <= set(source["entries"]))
        for key in REQUIRED_EXPLANATIONS:
            entry = source["entries"][key]
            self.assertTrue(entry["formula_text"], key)
            self.assertTrue(entry["variable_definitions"], key)
            self.assertTrue(entry["worked_numerical_example"], key)
            self.assertTrue(entry["interpretation"], key)
            self.assertTrue(entry["unit"], key)
            self.assertTrue(entry["annualization_convention"], key)
            self.assertTrue(entry["assumptions"], key)
            self.assertTrue(entry["limitations"], key)
            self.assertTrue(entry["misleading_cases"], key)
            self.assertTrue(entry["applicable_decision_modes"], key)
        self.assertIn('class="formula"', self.javascript)
        self.assertIn("escapeHtml(entry.formula_text)", self.javascript)

    def test_keyboard_accessibility_and_non_color_status_indicators(self) -> None:
        self.assertIn('class="skip-link"', self.html)
        self.assertIn('tabindex="-1"', self.html)
        self.assertIn('aria-live="assertive"', self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn("<nav", self.html)
        self.assertIn("<label", self.javascript)
        self.assertIn(":focus-visible", self.css)
        self.assertIn("aria-hidden=\"true\"", self.javascript)
        self.assertIn('const icon = good.has(code) ? "✓"', self.javascript)
        self.assertIn('role="img"', self.javascript)
        self.assertIn("차트 요약 표", self.javascript)

    def test_ui_reads_do_not_call_economic_backtest_and_legacy_assets_stay_separate(self) -> None:
        with patch("src.backtest.run_backtests") as backtest:
            for target in ("/", "/api/v1/overview", "/api/v1/runs", "/api/v1/terminology"):
                self.assertEqual(self.application.dispatch("GET", target).status_code, 200)
            backtest.assert_not_called()
        self.assertTrue((ROOT / "web" / "index.html").is_file())
        self.assertTrue((ROOT / "docs" / "backtest_dashboard.js").is_file())
        self.assertNotIn("trend_v2_foundation/ui_assets", self.html)

    def test_local_server_starts_and_serves_api_and_ui(self) -> None:
        original = self.fixture.api.server_config
        object.__setattr__(self.fixture.api, "server_config", type(original)(port=0))
        server = build_web_server(self.application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            with urllib.request.urlopen(f"http://{host}:{port}/", timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("저장된 전략 실행", response.read().decode("utf-8"))
            with urllib.request.urlopen(
                f"http://{host}:{port}/api/v1/overview", timeout=5
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["strategy_run_count"], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_javascript_has_valid_syntax(self) -> None:
        result = subprocess.run(
            [shutil.which("node"), "--check", str(ASSET_ROOT / "app.js")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
