import json
import unittest
from pathlib import Path

import pandas as pd

from src.backtest import (
    ANALYSIS_SCHEMA_VERSION,
    EntryRule,
    ExitRule,
    SignalRule,
    StrategyRule,
    _first_signal_indices,
    _signal_2d_indices,
    _simulate_one_symbol,
    _simulate_symbol_strategies,
)


class ExitReuseEquivalenceTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2026-01-01", periods=14, freq="B")
        self.frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": "TEST",
                "name": "Test ETF",
                "asset_group": "fixture",
                "open": [100, 101, 102, 103, 104, 105, 106, 94, 96, 97, 98, 99, 100, 101],
                "high": [101, 102, 103, 104, 105, 106, 107, 96, 98, 99, 100, 101, 102, 103],
                "low": [99, 100, 101, 102, 103, 104, 105, 93, 95, 96, 97, 98, 99, 100],
                "close": [100, 101, 102, 103, 104, 105, 106, 95, 97, 98, 99, 100, 101, 102],
                "low10": [95, 95, 95, 95, 95, 95, 95, 95, 95, 100, 95, 95, 95, 95],
                "low20": [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90],
                "score": range(14),
                "score_pct": [0.5] * 14,
                "r20": [0.1] * 14,
                "er20": [0.2] * 14,
                "fixture_signal": [False, True, True, False, False, True, True, False, True, True, False, False, False, False],
            }
        ).sample(frac=1, random_state=7).reset_index(drop=True)

        self.signal_rules = [
            SignalRule(
                key="fixture_signal",
                label="Fixture signal",
                description="Fixture signal",
                params={"family": "fixture"},
                signal_fn=lambda df: df["fixture_signal"],
            )
        ]
        self.entry_rules = [
            EntryRule("first", "First", "First fixture entry", _first_signal_indices),
            EntryRule("confirm_2d", "Confirm 2D", "Two-day fixture entry", _signal_2d_indices),
        ]
        self.exit_rules = [
            ExitRule("low10", "Low10", "Fixture Low10", lambda row: float(row["low10"])),
            ExitRule("low20", "Low20", "Fixture Low20", lambda row: float(row["low20"])),
        ]
        self.strategy_rules = [
            StrategyRule(
                key=f"{signal.key}__{entry.key}__{exit_rule.key}",
                label=f"{signal.label} / {entry.label} / {exit_rule.label}",
                signal=signal,
                entry=entry,
                exit=exit_rule,
            )
            for signal in self.signal_rules
            for entry in self.entry_rules
            for exit_rule in self.exit_rules
        ]

    @staticmethod
    def _canonical(records: list[dict]) -> list[str]:
        return sorted(json.dumps(record, sort_keys=True) for record in records)

    def test_reused_path_matches_independent_strategy_path(self) -> None:
        old_trades: list[dict] = []
        old_skipped: list[dict] = []
        for strategy in self.strategy_rules:
            trades, skipped = _simulate_one_symbol(self.frame, strategy)
            old_trades.extend(trades)
            old_skipped.extend(skipped)

        new_trades, new_skipped, pair_count, exit_count = _simulate_symbol_strategies(
            self.frame,
            signal_rules=self.signal_rules,
            entry_rules=self.entry_rules,
            exit_rules=self.exit_rules,
            strategy_rules=self.strategy_rules,
        )

        self.assertGreater(len(old_trades), 0)
        self.assertGreater(len(old_skipped), 0)
        self.assertEqual(len(old_trades), len(new_trades))
        self.assertEqual(len(old_skipped), len(new_skipped))
        self.assertEqual(self._canonical(old_trades), self._canonical(new_trades))
        self.assertEqual(self._canonical(old_skipped), self._canonical(new_skipped))
        self.assertEqual(pair_count, len(self.signal_rules) * len(self.entry_rules))
        self.assertEqual(exit_count, len(self.strategy_rules))

    def test_diagnostics_default_and_schema_version_are_unchanged(self) -> None:
        config_path = Path(__file__).parents[1] / "config" / "universe.yml"
        config_text = config_path.read_text(encoding="utf-8")
        self.assertIn("backtest_generate_signal_diagnostics: false", config_text)
        self.assertEqual(ANALYSIS_SCHEMA_VERSION, 3)


if __name__ == "__main__":
    unittest.main()
