"""Focused Foundation 9E candidate-estimate response and UI rendering coverage."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
APP = ROOT / "src" / "trend_v2_foundation" / "ui_assets" / "app.js"


class Foundation9ETests(unittest.TestCase):
    def test_candidate_estimate_presentation_handles_canonical_v1_and_v2_envelopes(self) -> None:
        script = """
const fs = require('fs');
global.document = { getElementById: () => ({ textContent: '', className: '', innerHTML: '', focus: () => {} }), querySelectorAll: () => [] };
global.window = { addEventListener: () => {} };
global.location = { hash: '' };
global.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
global.AbortController = class { constructor() { this.signal = {}; } abort() {} };
eval(fs.readFileSync(process.argv[1], 'utf8'));
const ui = globalThis.TrendV2Ui;
const v1 = { normalized_construction: { walk_forward: { enabled: false, fold_count: 0 }, robustness: { scenario_count: 0 } }, candidate_estimate: { raw_cartesian_candidate_count: 1, economic_strategy_run_candidate_count: 1, evaluation_profile_application_count: 1, robustness_scenario_count: 0, estimated_total_execution_units: 3, estimated_reuse_count: 0, estimated_new_backtest_count: 1, threshold_results: [], confirmation_required: false, hard_limit_exceeded: false }, strategy_run_candidate_ids: ['one'] };
const v2 = { normalized_construction: { components: {} }, candidate_estimate: { schema_version: 'candidate_space_estimate_v2', raw_cartesian_combinations: 1, valid_unique_economic_candidates: 1, evaluation_only_applications: 1, robustness_workload: 0, total_estimated_work: 2, reusable_completed_candidates: 0, new_candidates_requiring_execution: 1, candidate_economic_hashes: ['one'] }, strategy_run_candidate_ids: ['one'] };
if (!ui.estimateSummary(v1).includes('1') || !ui.estimateSummary(v2).includes('1')) process.exit(2);
if (ui.candidateEstimatePresentation(v2).normalized.walk_forward.fold_count !== 0) process.exit(3);
try { ui.estimateSummary({ candidate_estimate: {} }); process.exit(4); } catch (error) { if (!error.message.includes('API 응답 계약')) process.exit(5); }
"""
        completed = subprocess.run(["node", "-e", script, str(APP)], text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_estimate_error_is_rendered_in_the_construction_preview(self) -> None:
        source = APP.read_text(encoding="utf-8")
        self.assertIn("candidateEstimatePresentation", source)
        self.assertIn("API 응답 계약을 확인하세요.", source)
        self.assertIn('document.getElementById("construction-preview").innerHTML = `<p class="notice danger">', source)
        self.assertIn('profileSelect.multiple = false', source)
        self.assertIn('research_default', source)


if __name__ == "__main__":
    unittest.main()
