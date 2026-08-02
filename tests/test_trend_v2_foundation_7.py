from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.trend_v2_foundation import (
    ArtifactKind, ArtifactRetentionPolicy, LocalResultStore, RobustnessError,
    RobustnessExecutionService, RobustnessPolicy, StrategyRunManifest,
    StrategyRunSpec, aligned_paired_returns, generate_loyo_years,
    generate_walk_forward_folds, holm_adjust, load_robustness_catalog,
    paired_block_bootstrap, validate_robustness_summary_v2,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "trend_v2" / "robustness_execution_policy_v1.json"
CATALOG = ROOT / "config" / "trend_v2" / "robustness_option_catalog_v1.json"


def curve(start: str, returns: list[float]) -> dict:
    dates = [f"2024-01-{day:02d}" for day in range(2, 2 + len(returns))]
    value, rows = 100.0, []
    for index, (day, daily_return) in enumerate(zip(dates, returns)):
        value *= 1 + daily_return
        rows.append({"economic_date": day, "portfolio_value": value, "daily_return": daily_return,
                     "gross_exposure": 1.0, "net_exposure": 1.0, "cash_weight": 0.0,
                     "daily_turnover": 0.01, "transaction_cost": 0.0001})
    return {"schema_version": "daily_portfolio_curve_v1", "economic_date_range": {"start": dates[0], "end": dates[-1]}, "rows": rows}


class Foundation7PureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = RobustnessPolicy.load(POLICY)
        self.dates = [f"2024-01-{day:02d}" for day in range(2, 12)]

    def test_explicit_folds_are_ordered_and_boundary_incomplete(self) -> None:
        settings = {"window": "expanding", "minimum_training_observations": 4,
                    "minimum_test_observations": 3, "incomplete_boundary_fold": "incomplete",
                    "folds": [{"training_start": self.dates[0], "training_end": self.dates[3], "test_start": self.dates[4], "test_end": self.dates[6], "gap_sessions": 0},
                              {"training_start": self.dates[0], "training_end": self.dates[6], "test_start": self.dates[7], "test_end": self.dates[8], "gap_sessions": 0}]}
        folds = generate_walk_forward_folds(self.dates, settings, self.policy)
        self.assertEqual([item["fold_id"] for item in folds], ["wf-001", "wf-002"])
        self.assertFalse(folds[0]["incomplete"])
        self.assertTrue(folds[1]["incomplete"])
        with self.assertRaises(RobustnessError):
            generate_walk_forward_folds(self.dates, {**settings, "folds": list(reversed(settings["folds"]))}, self.policy)

    def test_loyo_partial_year_is_explicit(self) -> None:
        years = generate_loyo_years(["2023-12-29", "2024-01-02", "2024-12-30"], {"included_years": [2023, 2024], "minimum_observations": 1, "partial_year_eligibility": "exclude"}, self.policy)
        self.assertTrue(all(item["partial_year"] for item in years))
        self.assertTrue(all(not item["eligible"] for item in years))

    def test_paired_bootstrap_is_seed_deterministic_and_holm_is_explicit(self) -> None:
        strategy, benchmark = curve("", [0.01, -0.01, 0.02, 0.0]), curve("", [0.005, -0.004, 0.01, 0.001])
        pairs = aligned_paired_returns(strategy, benchmark)
        first = paired_block_bootstrap(pairs, seed=7, sample_count=20, block_length=2, confidence_level=.95)
        self.assertEqual(first, paired_block_bootstrap(pairs, seed=7, sample_count=20, block_length=2, confidence_level=.95))
        self.assertIsNone(first["adjusted_p_value"])
        correction = holm_adjust({"a": .01, "b": .04})
        self.assertEqual(correction["method"], "holm")
        self.assertIn("correction_identity", correction)


class Foundation7ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = LocalResultStore(self.temp.name, ArtifactRetentionPolicy(5_000_000, 1_000_000, 10, 10))
        self.policy, self.catalog = RobustnessPolicy.load(POLICY), load_robustness_catalog(CATALOG)
        self.service = RobustnessExecutionService(self.store, self.policy, self.catalog, source_commit="a" * 40, cost_stress_runner=lambda run_id, multiple: {"survives": multiple <= 2.0, "canonical_engine": True})
        self.daily, self.benchmark = curve("", [.01, .01, -.005, .004, .002, -.003, .006, .001]), curve("", [.005, .003, -.004, .002, .001, -.002, .003, .0])
        spec = StrategyRunSpec(data_snapshot_hash="b" * 64, economic_date_range=self.daily["economic_date_range"], universe_specification={"id": "u"}, benchmark={"option_id": "spy"}, trend_filter={"id": "t"}, signal={"id": "s"}, entry_rule={"id": "e"}, initial_stop={"id": "i"}, trailing_exit={"id": "x"}, position_sizing={"id": "p"}, portfolio_constraints={"id": "c"}, transaction_costs={"id": "cost"}, slippage={"id": "slip"}, engine_version="engine")
        daily_record = self.store.put_artifact("daily_portfolio_curve", ArtifactKind.DAILY_PORTFOLIO_CURVE, self.daily, row_count=8).record
        benchmark_record = self.store.put_artifact("benchmark_daily_portfolio_curve", ArtifactKind.DAILY_PORTFOLIO_CURVE, self.benchmark, row_count=8).record
        self.store.save_strategy_run(StrategyRunManifest.create(spec, source_code_commit="a" * 40, artifacts=(daily_record, benchmark_record), creation_time="2026-08-02T00:00:00Z"))
        self.run_id = spec.strategy_run_id

    def tearDown(self) -> None: self.temp.cleanup()

    def request(self) -> dict:
        dates = [row["economic_date"] for row in self.daily["rows"]]
        return {"base_strategy_run_id": self.run_id, "seed": 3, "methods": {
            "walk_forward_fixed_v1": {"window": "rolling", "minimum_training_observations": 3, "minimum_test_observations": 2, "incomplete_boundary_fold": "incomplete", "folds": [{"training_start": dates[0], "training_end": dates[2], "test_start": dates[3], "test_end": dates[4], "gap_sessions": 0}]},
            "leave_one_year_out_v1": {"included_years": [2024], "minimum_observations": 1, "partial_year_eligibility": "allow_flagged"},
            "paired_moving_block_bootstrap_v1": {"sample_count": 10, "block_length": 2, "confidence_level": .95},
            "canonical_cost_stress_v1": {"multipliers": [1.5, 2.0]}}}

    def test_plan_is_deterministic_and_evidence_is_persisted(self) -> None:
        normalized = self.service.normalize(self.request())
        self.assertEqual(normalized["estimate"]["fold_backtest_units"], 1)
        self.assertEqual(normalized["estimate"]["bootstrap_resample_units"], 10)
        plan = self.service.create_plan(self.request())
        evidence = self.service.evidence(plan["robustness_plan_id"])
        validate_robustness_summary_v2(evidence)
        self.assertEqual(evidence["walk_forward"]["fold_count"], 1)
        self.assertTrue(evidence["cost_stress"]["survival"])
        self.assertEqual(self.service.evidence(plan["robustness_plan_id"])["evidence_hash"], evidence["evidence_hash"])
