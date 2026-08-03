"""Focused Foundation 9D first-run profile coverage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.trend_v2_foundation.local_operability import initialize_result_store
from src.trend_v2_foundation.profiles import load_evaluation_profiles
from src.trend_v2_foundation.result_store import LocalResultStore
from src.trend_v2_foundation.web import load_retention_policy
from src.trend_v2_foundation.contracts import ArtifactRetentionPolicy


ROOT = Path(__file__).parents[1]
PROFILE_ROOT = ROOT / "config" / "trend_v2" / "evaluation_profiles"


class Foundation9DTests(unittest.TestCase):
    def _store(self, root: Path) -> LocalResultStore:
        return LocalResultStore(root, ArtifactRetentionPolicy.from_dict(load_retention_policy(root)))

    def test_init_seeds_repository_profiles_and_reruns_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            first = initialize_result_store(root, PROFILE_ROOT)
            store = self._store(root)
            expected = load_evaluation_profiles(PROFILE_ROOT)
            self.assertEqual(first["seeded"], 3)
            self.assertEqual(tuple(sorted(store.evaluation_profile_history())), store.evaluation_profile_history())
            self.assertEqual({store.get_evaluation_profile(item).name for item in store.evaluation_profile_history()}, set(expected))
            second = initialize_result_store(root, PROFILE_ROOT)
            self.assertEqual((second["seeded"], second["reused"]), (0, 3))

    def test_ui_profile_dropdown_contract(self):
        source = (ROOT / "src" / "trend_v2_foundation" / "ui_assets" / "app.js").read_text(encoding="utf-8")
        self.assertIn('profileSelect.multiple = false', source)
        self.assertIn('research_default', source)
        self.assertIn('연구용 기본 평가', source)
        self.assertIn('최종 적격성 평가', source)
        self.assertIn('탐색용 가중 평가 예시', source)
        self.assertIn('사용 가능한 평가 프로필이 없습니다. ResultStore 기본 프로필을 초기화하세요.', source)
        self.assertIn('button.disabled=true', source)
