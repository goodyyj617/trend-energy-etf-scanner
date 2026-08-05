"""Focused immutable Evaluation Profile Studio coverage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.trend_v2_foundation.api import ReadOnlyTrendApi
from src.trend_v2_foundation.contracts import ArtifactRetentionPolicy, EVALUATION_PROFILE_V2_VERSION
from src.trend_v2_foundation.profile_studio import ProfileStudioService
from src.trend_v2_foundation.profiles import load_evaluation_profiles
from src.trend_v2_foundation.result_store import LocalResultStore


ROOT = Path(__file__).parents[1]
PROFILE_ROOT = ROOT / "config" / "trend_v2" / "evaluation_profiles"
TERMINOLOGY = json.loads((ROOT / "config" / "trend_v2" / "terminology_ko.json").read_text(encoding="utf-8"))


class ProfileStudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = LocalResultStore(Path(self.temporary.name), ArtifactRetentionPolicy(10_000_000, 1_000_000, 100, 100))
        self.source = load_evaluation_profiles(PROFILE_ROOT)["research_default"]
        self.store.save_evaluation_profile(self.source)
        self.studio = ProfileStudioService(self.store, clock=lambda: "2026-08-05T00:00:00Z")
        self.api = ReadOnlyTrendApi(self.store, terminology_source=TERMINOLOGY, profile_studio_service=self.studio)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def draft(self) -> dict:
        profile = self.source.to_dict()
        for key in ("name", "approval_status", "schema_version", "lineage"):
            profile.pop(key, None)
        profile["description"] = "Studio revision"
        return {"source_profile_id": self.source.evaluation_profile_id, "change_summary_ko": "회복기간 기준을 검토했습니다.", "profile": profile}

    def test_v1_identity_is_preserved_and_v2_lineage_is_immutable(self) -> None:
        original = self.source.to_dict()
        payload = self.draft()
        validated = self.studio.validate(payload)
        self.assertTrue(validated["valid"])
        saved, replayed = self.studio.save({**payload, "validated_draft_hash": validated["draft_hash"]}, idempotency_key="save-1")
        self.assertFalse(replayed)
        version = self.store.get_evaluation_profile(saved["evaluation_profile_id"])
        self.assertEqual(version.schema_version, EVALUATION_PROFILE_V2_VERSION)
        self.assertEqual(version.lineage["parent_profile_id"], self.source.evaluation_profile_id)
        self.assertEqual(version.lineage["root_profile_id"], self.source.evaluation_profile_id)
        self.assertEqual(version.lineage["revision"], 1)
        self.assertEqual(self.store.get_evaluation_profile(self.source.evaluation_profile_id).to_dict(), original)
        history = self.studio.history(version.evaluation_profile_id)
        self.assertEqual([item["lineage"]["revision"] for item in history["items"]], [0, 1])
        same, replayed = self.studio.save({**payload, "validated_draft_hash": validated["draft_hash"]}, idempotency_key="save-1")
        self.assertTrue(replayed)
        self.assertEqual(same, saved)

    def test_registry_options_and_validation_are_bounded(self) -> None:
        options = self.api.dispatch("GET", "/api/v1/evaluation-profile-studio/options")
        self.assertEqual(options.status_code, 200)
        self.assertIn("cagr_spy_ratio", {item["metric_key"] for item in options.body["metrics"]})
        invalid = self.draft()
        invalid["profile"]["enabled_metrics"].append("python_formula")
        invalid["profile"]["metric_directions"]["python_formula"] = "maximize"
        invalid["profile"]["metric_modes"]["python_formula"] = "absolute"
        response = self.api.dispatch("POST", "/api/v1/evaluation-profiles/validate", body=invalid)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.body["valid"])
        self.assertTrue(response.body["errors"][0]["message_ko"])
        epsilon = self.draft()
        epsilon["profile"]["pareto_objectives"][0]["epsilon"] = -0.01
        self.assertFalse(self.studio.validate(epsilon)["valid"])

    def test_api_save_lineage_and_apply_do_not_start_economic_backtest(self) -> None:
        draft = self.draft()
        validated = self.api.dispatch("POST", "/api/v1/evaluation-profiles/validate", body=draft).body
        saved = self.api.dispatch("POST", "/api/v1/evaluation-profiles", headers={"Idempotency-Key": "studio-save"}, body={**draft, "validated_draft_hash": validated["draft_hash"]})
        self.assertEqual(saved.status_code, 201)
        profile_id = saved.body["evaluation_profile_id"]
        lineage = self.api.dispatch("GET", f"/api/v1/evaluation-profiles/{profile_id}/lineage")
        self.assertEqual(lineage.status_code, 200)
        self.assertEqual(lineage.body["total"], 2)
        self.assertEqual(lineage.body["returned"], 2)
        fake = SimpleNamespace(evaluation_run=SimpleNamespace(to_dict=lambda: {"evaluation_run_id": "evaluation_run_test"}), cache_status={"strategy_run_test": "reused"})
        with patch("src.trend_v2_foundation.profile_studio.calculate_and_evaluate_saved_runs", return_value=fake) as evaluate:
            applied = self.api.dispatch("POST", f"/api/v1/evaluation-profiles/{profile_id}/apply", headers={"Idempotency-Key": "studio-apply"}, body={"strategy_run_id": "strategy_run_test"})
        self.assertEqual(applied.status_code, 201)
        self.assertFalse(applied.body["economic_backtest_started"])
        evaluate.assert_called_once()

    def test_ui_exposes_korean_studio_and_no_raw_server_path(self) -> None:
        source = (ROOT / "src" / "trend_v2_foundation" / "ui_assets" / "app.js").read_text(encoding="utf-8")
        self.assertIn("평가 프로필 스튜디오", source)
        self.assertIn("/evaluation-profiles/validate", source)
        self.assertIn("경제 백테스트 시작: 아니오", source)
        self.assertIn("탐색 가중 결과는 기본 비보상형 판단을 바꾸지 않습니다", source)


if __name__ == "__main__":
    unittest.main()
