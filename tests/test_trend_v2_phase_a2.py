from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.trend_v2 import (
    classify_score_breakout,
    default_phase_a_components,
    signal_event_counts,
    within_symbol_circular_shifted_events,
)
from src.trend_v2_phase_a2 import (
    ANALYSIS_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    annualized_paired_effect,
    bootstrap_index_hash,
    build_loyo_evidence,
    build_walk_forward_evidence,
    deterministic_gzip_csv_bytes,
    holm_adjust,
    load_frozen_inputs,
    paired_daily_returns,
    paired_stationary_bootstrap,
    reconstruct_snapshot_sha256,
    render_current_state,
    run_empirical_phase_a2,
    sha256_bytes,
    stationary_bootstrap_indices,
    summarize_asset_group_effects,
    transition_to_phase_b1,
    validate_snapshot_manifest,
    write_yearly_price_shards,
)


def price_fixture() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2019-12-30", "2021-01-04", freq="B"):
        for symbol, offset in (("AAA", 0.0), ("SPY", 20.0)):
            value = 100.0 + offset + (date - pd.Timestamp("2019-12-30")).days * 0.01
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "symbol": symbol,
                    "open": value,
                    "high": value + 1,
                    "low": value - 1,
                    "close": value + 0.5,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


def write_frozen_fixture(root: Path) -> dict:
    universe = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "name": "AAA ETF",
                "base_universe_eligible": True,
                "aum_rank": 1,
                "asset_group": "US Equity",
            },
            {
                "symbol": "SPY",
                "name": "SPY ETF",
                "base_universe_eligible": False,
                "aum_rank": 2,
                "asset_group": "US Equity",
            },
        ]
    )
    universe_payload = universe.to_csv(index=False, lineterminator="\n").encode()
    (root / "universe_snapshot.csv").write_bytes(universe_payload)
    shards = write_yearly_price_shards(price_fixture(), root / "prices")
    members = {"universe_snapshot.csv": sha256_bytes(universe_payload)}
    members.update({row["path"]: row["sha256"] for row in shards})
    prices = price_fixture()
    counts = {key: int(value) for key, value in prices.groupby("symbol").size().items()}
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_code_commit": "a" * 40,
        "utc_collection_time": "2026-07-31T00:00:00+00:00",
        "retained_minimum_economic_date": prices["date"].min(),
        "retained_maximum_economic_date": prices["date"].max(),
        "final_as_of": prices["date"].max(),
        "versions": {"yfinance": "fixture", "pandas": pd.__version__, "numpy": np.__version__},
        "adjusted_price_convention": "fixture adjusted OHLC",
        "requested_symbols": ["AAA", "SPY"],
        "retained_symbols": ["AAA", "SPY"],
        "failed_or_empty_symbols": [],
        "row_count": len(prices),
        "row_counts_by_symbol": counts,
        "row_counts_by_symbol_and_year": {},
        "input_hashes": {"aum": "b" * 64},
        "universe_snapshot": {
            "path": "universe_snapshot.csv",
            "row_count": len(universe),
            "sha256": members["universe_snapshot.csv"],
        },
        "price_shards": shards,
        "snapshot_members": members,
        "complete_snapshot_sha256": reconstruct_snapshot_sha256(members),
        "completed_bar_filtering_provenance": {"rows_removed": 0},
        "universe_parameters": {
            "min_history_days": 1,
            "min_close": 1,
            "min_avg_dollar_vol_20": 0,
            "min_avg_dollar_vol_63": 0,
            "dollar_volume_top_n": 500,
        },
        "known_limitations": [],
    }
    (root / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def curve_fixture() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2014-01-02", "2025-12-30", freq="B"):
        year = date.year
        values = {
            "trend_filter_only": 0.00020,
            "prior_price_high_l20": 0.00025 if year <= 2020 else 0.00015,
            "score_breakout_l10": 0.00030,
            "score_breakout_l20": 0.00018,
            "score_breakout_l40": 0.00010,
        }
        for key, value in values.items():
            rows.append(
                {
                    "strategy_key": key,
                    "date": date.date().isoformat(),
                    "observation_type": "trading_session",
                    "daily_portfolio_return": value,
                }
            )
    return pd.DataFrame(rows)


class SnapshotTests(unittest.TestCase):
    def test_deterministic_yearly_shards_and_snapshot_hash(self) -> None:
        shuffled = price_fixture().sample(frac=1.0, random_state=9)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            left = write_yearly_price_shards(shuffled, Path(first))
            right = write_yearly_price_shards(shuffled, Path(second))
            self.assertEqual(left, right)
            self.assertEqual(
                reconstruct_snapshot_sha256({row["path"]: row["sha256"] for row in left}),
                reconstruct_snapshot_sha256({row["path"]: row["sha256"] for row in right}),
            )
            loaded = pd.concat(
                [pd.read_csv(Path(first) / f"{row['year']}.csv.gz") for row in left],
                ignore_index=True,
            )
            self.assertEqual(
                list(zip(loaded["date"], loaded["symbol"])),
                sorted(zip(loaded["date"], loaded["symbol"])),
            )

    def test_gzip_bytes_do_not_contain_time_variance(self) -> None:
        frame = price_fixture().head(4)
        self.assertEqual(deterministic_gzip_csv_bytes(frame), deterministic_gzip_csv_bytes(frame))

    def test_manifest_validation_and_frozen_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = write_frozen_fixture(path)
            self.assertEqual(
                validate_snapshot_manifest(path)["complete_snapshot_sha256"],
                manifest["complete_snapshot_sha256"],
            )
            frozen = load_frozen_inputs(path)
            self.assertEqual(len(frozen.prices), manifest["row_count"])
            shard = path / manifest["price_shards"][0]["path"]
            shard.write_bytes(shard.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_snapshot_manifest(path)

    def test_empirical_runner_reaches_feature_build_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_frozen_fixture(path)
            with patch("src.trend_v2_phase_a2.download_ohlcv") as downloader, patch(
                "src.trend_v2_phase_a2.build_historical_features",
                side_effect=RuntimeError("frozen-input-sentinel"),
            ):
                with self.assertRaisesRegex(RuntimeError, "frozen-input-sentinel"):
                    run_empirical_phase_a2(path, path)
            downloader.assert_not_called()


class RobustnessTests(unittest.TestCase):
    def test_walk_forward_is_expanding_and_chronological(self) -> None:
        curves = curve_fixture()
        years = list(range(2015, 2025))
        result = build_walk_forward_evidence(curves, ["score_breakout_l10"], years)
        folds = result["score_breakout_l10"]["folds"]
        self.assertGreaterEqual(len(folds), 3)
        for fold in folds:
            self.assertLess(fold["training_end_date"], fold["test_start_date"])
            self.assertEqual(fold["test_year"], int(fold["test_start_date"][:4]))
        self.assertEqual([row["test_year"] for row in folds], list(range(2018, 2025)))

    def test_loyo_removes_each_complete_year(self) -> None:
        curves = curve_fixture()
        years = [2018, 2019, 2020]
        result = build_loyo_evidence(curves, ["score_breakout_l10"], years)["score_breakout_l10"]
        self.assertEqual([row["omitted_year"] for row in result["cases"]], years)
        full = paired_daily_returns(curves, "score_breakout_l10", result["full_sample_comparator"])
        expected = annualized_paired_effect(full.loc[~full["date"].dt.year.eq(2018)])
        self.assertAlmostEqual(result["cases"][0]["annualized_mean_daily_return_effect"], expected)

    def test_stationary_bootstrap_is_reproducible(self) -> None:
        left = stationary_bootstrap_indices(80, 100, 20, seed=42)
        right = stationary_bootstrap_indices(80, 100, 20, seed=42)
        np.testing.assert_array_equal(left, right)
        self.assertEqual(bootstrap_index_hash(left), bootstrap_index_hash(right))
        common = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=80, freq="B"),
                "paired_difference": np.linspace(-0.001, 0.002, 80),
            }
        )
        self.assertEqual(
            paired_stationary_bootstrap(common, path_count=100, seed=42),
            paired_stationary_bootstrap(common, path_count=100, seed=42),
        )

    def test_holm_correction_uses_whole_score_family(self) -> None:
        result = holm_adjust({"l10": 0.01, "l20": 0.03, "l40": 0.20})
        self.assertAlmostEqual(result["l10"]["adjusted_p_value"], 0.03)
        self.assertAlmostEqual(result["l20"]["adjusted_p_value"], 0.06)
        self.assertFalse(result["l20"]["pass"])

    def test_asset_group_concentration_is_labeled_non_additive(self) -> None:
        result = summarize_asset_group_effects(
            [
                {"asset_group": "US Equity", "annualized_mean_daily_return_effect": 0.03},
                {"asset_group": "Bond", "annualized_mean_daily_return_effect": 0.01},
                {"asset_group": "Other", "annualized_mean_daily_return_effect": -0.01},
            ],
            pd.Series(["US Equity", "Other", "unknown", "Bond"]),
            "trend_filter_only",
        )
        self.assertAlmostEqual(result["dominant_effect_share"], 0.75)
        self.assertAlmostEqual(result["missing_or_unclassified_group_share"], 0.5)
        self.assertFalse(result["additivity_claimed"])

    def test_circular_placebo_preserves_per_symbol_trigger_counts(self) -> None:
        rows = []
        for symbol in ("AAA", "BBB"):
            for date in pd.date_range("2020-01-01", periods=20, freq="B"):
                rows.append({"symbol": symbol, "date": date})
        frame = pd.DataFrame(rows)
        events = pd.Series(False, index=frame.index)
        eligible = pd.Series(True, index=frame.index)
        for _, group in frame.groupby("symbol"):
            events.loc[group.index[[1, 2, 7, 13]]] = True
        shifted, offsets = within_symbol_circular_shifted_events(
            frame, events, eligible, shift_bars=7
        )
        self.assertEqual(set(offsets), {"AAA", "BBB"})
        for _, group in frame.groupby("symbol"):
            source = signal_event_counts(frame.loc[group.index], events.loc[group.index], default_phase_a_components().entry)
            placebo = signal_event_counts(frame.loc[group.index], shifted.loc[group.index], default_phase_a_components().entry)
            self.assertEqual(source["executable_trigger_count"], placebo["executable_trigger_count"])


class ClassificationTransitionTests(unittest.TestCase):
    @staticmethod
    def metrics() -> pd.DataFrame:
        rows = [
            {"strategy_key": "trend_filter_only", "variant": "trend_filter_only", "strategy_cagr": 0.08},
            {"strategy_key": "prior_price_high_l20", "variant": "prior_price_high", "strategy_cagr": 0.09},
        ]
        for lookback, cagr in ((10, 0.10), (20, 0.11), (40, 0.07)):
            rows.append({"strategy_key": f"score_breakout_l{lookback}", "variant": "score_breakout", "strategy_cagr": cagr})
        return pd.DataFrame(rows)

    @staticmethod
    def robustness() -> dict:
        results = {}
        bootstrap = {}
        for lookback, p_value, effect in ((10, 0.04, 0.01), (20, 0.02, 0.02), (40, 0.20, -0.01)):
            key = f"score_breakout_l{lookback}"
            results[key] = {"adjusted_p_value": p_value}
            bootstrap[key] = {"paired_annualized_effect_estimate": effect}
        return {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "multiple_testing": {"results": results},
            "date_block_bootstrap": bootstrap,
        }

    def test_result_transition_limits_phase_b1_signal_set(self) -> None:
        inconclusive = transition_to_phase_b1(
            {"classification": "Inconclusive", "reason": "fixture"},
            self.metrics(),
            self.robustness(),
        )
        self.assertEqual(
            inconclusive["signal_candidates"],
            ["score_breakout_l20", "prior_price_high_l20"],
        )
        self.assertEqual(inconclusive["maximum_signal_candidates"], 2)
        rejected = transition_to_phase_b1(
            {"classification": "Reject", "reason": "fixture"},
            self.metrics(),
            self.robustness(),
        )
        self.assertEqual(rejected["signal_candidates"], ["prior_price_high_l20"])

    def test_current_state_records_exact_transition(self) -> None:
        summary = {
            "classification": {"classification": "Inconclusive", "reason": "fixture"},
            "snapshot_date_range": {"minimum": "2016-01-01", "maximum": "2025-12-31"},
            "complete_snapshot_sha256": "f" * 64,
            "phase_b1_transition": {
                "next_phase": "Phase B1 -- entry-family screening",
                "score_breakout_status": "exploratory_and_unresolved",
                "signal_candidates": ["score_breakout_l20", "prior_price_high_l20"],
                "maximum_signal_candidates": 2,
            },
        }
        rendered = render_current_state(summary)
        self.assertIn("score_breakout_l20", rendered)
        self.assertIn("prior_price_high_l20", rendered)
        self.assertIn("Phase B1", rendered)


if __name__ == "__main__":
    unittest.main()
