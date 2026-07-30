from __future__ import annotations

import ast
import contextlib
import io
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import run_daily_scan


ROOT = Path(__file__).parents[1]


def _daily_workflow_text() -> str:
    matches: list[str] = []
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        name_line = next(
            (line for line in text.splitlines() if line.startswith("name:")),
            "",
        )
        if name_line.partition(":")[2].strip() == "Daily ETF Scan":
            matches.append(text)
    if len(matches) != 1:
        raise AssertionError(f"Expected one Daily ETF Scan workflow, found {len(matches)}")
    return matches[0]


class DailyScanOnlyTest(unittest.TestCase):
    @staticmethod
    def _universe() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "base_universe_eligible": True,
                    "display_name": "Alpha ETF",
                },
                {
                    "symbol": "ZZZ",
                    "base_universe_eligible": False,
                    "display_name": "Excluded ETF",
                },
            ]
        )

    @staticmethod
    def _income_exclusions() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": "ZZZ",
                    "reason_category": "Income-oriented",
                }
            ]
        )

    @staticmethod
    def _latest() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2026-07-29",
                    "symbol": "AAA",
                    "eligible_universe": True,
                    "signal_surge_v0": True,
                    "score": 2.5,
                    "suggested_stop": 91.25,
                    "signal_streak_days": 2,
                },
                {
                    "date": "2026-07-29",
                    "symbol": "ZZZ",
                    "eligible_universe": False,
                    "signal_surge_v0": False,
                    "score": 0.0,
                    "suggested_stop": None,
                    "signal_streak_days": 0,
                },
            ]
        )

    def test_entry_point_has_no_run_backtests_import_or_call(self) -> None:
        source = (ROOT / "src" / "run_daily_scan.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        self.assertNotIn("run_backtests", imported_names)
        self.assertNotIn("run_backtests", called_names)
        self.assertNotIn("run_backtests", called_attributes)

    def test_main_writes_only_scan_owned_outputs_and_preserves_backtest_artifact(
        self,
    ) -> None:
        universe = self._universe()
        exclusions = self._income_exclusions()
        feature_rows = self._latest().drop(columns=["signal_streak_days"])
        latest = self._latest()
        prices = pd.DataFrame(
            {
                "date": ["2026-07-28", "2026-07-29"],
                "symbol": ["AAA", "AAA"],
                "close": [100.0, 101.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "docs" / "data"
            data_dir.mkdir(parents=True)
            sentinel = data_dir / "backtest_summary.json"
            sentinel_bytes = b'{"owner":"Backtest Only","sentinel":true}\n'
            sentinel.write_bytes(sentinel_bytes)
            files_before = {
                path.relative_to(data_dir)
                for path in data_dir.rglob("*")
                if path.is_file()
            }

            output = io.StringIO()
            with (
                patch.object(run_daily_scan, "ROOT", root),
                patch.object(
                    run_daily_scan,
                    "load_config",
                    return_value={"lookback_period": "5y", "price_interval": "1d"},
                ),
                patch.object(run_daily_scan, "update_aum_csv") as update_aum,
                patch.object(
                    run_daily_scan,
                    "build_base_universe",
                    return_value=universe,
                ) as build_universe,
                patch.object(
                    run_daily_scan,
                    "build_income_exclusion_review",
                    return_value=exclusions,
                ) as build_exclusions,
                patch.object(
                    run_daily_scan,
                    "download_ohlcv",
                    return_value=prices,
                ) as download,
                patch.object(
                    run_daily_scan,
                    "compute_latest_features",
                    return_value=feature_rows,
                ) as compute_features,
                patch.object(
                    run_daily_scan,
                    "add_signal_history",
                    return_value=latest,
                ) as add_history,
                contextlib.redirect_stdout(output),
            ):
                run_daily_scan.main()

            update_aum.assert_called_once_with()
            build_universe.assert_called_once_with(
                aum_csv=root / "config" / "aum.csv",
                exclusions_yml=root / "config" / "exclusions.yml",
                overrides_csv=root / "config" / "manual_overrides.csv",
                universe_yml=root / "config" / "universe.yml",
            )
            build_exclusions.assert_called_once_with(universe)
            download.assert_called_once_with(["AAA"], period="5y", interval="1d")
            compute_features.assert_called_once_with(
                prices,
                universe,
                {"lookback_period": "5y", "price_interval": "1d"},
            )
            add_history.assert_called_once_with(
                latest=feature_rows,
                history_dir=data_dir / "history",
                as_of="2026-07-29",
            )

            payload = json.loads(
                (data_dir / "latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(payload),
                {"as_of", "row_count", "eligible_count", "signal_count", "rows"},
            )
            self.assertEqual(payload["as_of"], "2026-07-29")
            self.assertEqual(payload["row_count"], 2)
            self.assertEqual(payload["eligible_count"], 1)
            self.assertEqual(payload["signal_count"], 1)
            self.assertEqual(len(payload["rows"]), 2)

            pd.testing.assert_frame_equal(pd.read_csv(data_dir / "latest.csv"), latest)
            pd.testing.assert_frame_equal(
                pd.read_csv(data_dir / "history" / "2026-07-29.csv"),
                latest,
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(data_dir / "universe_current.csv"),
                universe,
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(data_dir / "excluded_etfs_summary.csv"),
                exclusions,
            )

            self.assertEqual(sentinel.read_bytes(), sentinel_bytes)
            files_after = {
                path.relative_to(data_dir)
                for path in data_dir.rglob("*")
                if path.is_file()
            }
            new_files = files_after - files_before
            self.assertFalse(
                any(path.name.startswith("backtest_") for path in new_files),
                new_files,
            )
            self.assertFalse(
                any(path.name.startswith("signal_diagnostics") for path in new_files),
                new_files,
            )
            self.assertFalse(
                any("portfolio_curve" in path.as_posix() for path in new_files),
                new_files,
            )

            log = output.getvalue()
            self.assertIn("as_of=2026-07-29 rows=2 eligible=1 signals=1", log)
            self.assertNotIn("backtest", log.lower())

    def test_aum_failure_remains_fail_soft(self) -> None:
        output = io.StringIO()
        with (
            patch.object(
                run_daily_scan,
                "update_aum_csv",
                side_effect=RuntimeError("fixture failure"),
            ),
            contextlib.redirect_stdout(output),
        ):
            run_daily_scan._update_aum_cache_safely()

        self.assertIn(
            "[AUM] update failed, continuing scan with existing config/aum.csv: "
            "fixture failure",
            output.getvalue(),
        )

    def test_empty_downloaded_prices_fail_clearly(self) -> None:
        universe = self._universe()
        exclusions = self._income_exclusions()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.object(run_daily_scan, "ROOT", root),
                patch.object(
                    run_daily_scan,
                    "load_config",
                    return_value={"lookback_period": "5y", "price_interval": "1d"},
                ),
                patch.object(run_daily_scan, "update_aum_csv") as update_aum,
                patch.object(
                    run_daily_scan,
                    "build_base_universe",
                    return_value=universe,
                ),
                patch.object(
                    run_daily_scan,
                    "build_income_exclusion_review",
                    return_value=exclusions,
                ),
                patch.object(
                    run_daily_scan,
                    "download_ohlcv",
                    return_value=pd.DataFrame(),
                ),
                patch.object(run_daily_scan, "compute_latest_features") as compute,
                self.assertRaisesRegex(
                    RuntimeError,
                    "No price data downloaded",
                ),
            ):
                run_daily_scan.main()

            update_aum.assert_called_once_with()
            compute.assert_not_called()


class DailyScanWorkflowContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = _daily_workflow_text()

    def test_scan_commands_and_exact_owned_staging_paths(self) -> None:
        self.assertIn("python -m src.run_daily_scan", self.workflow)
        self.assertIn("python src/postprocess_groups.py", self.workflow)

        staging_commands = [
            line.strip()
            for line in self.workflow.splitlines()
            if line.strip().startswith("git add ")
        ]
        staged_paths: list[str] = []
        for command in staging_commands:
            command = command.removesuffix(" || true")
            tokens = shlex.split(command)
            self.assertEqual(tokens[:2], ["git", "add"])
            staged_paths.extend(tokens[2:])

        self.assertEqual(
            staged_paths,
            [
                "config/aum.csv",
                "docs/data/latest.json",
                "docs/data/latest.csv",
                "docs/data/universe_current.csv",
                "docs/data/excluded_etfs_summary.csv",
                "docs/data/history",
            ],
        )

    def test_backtest_cleanup_restore_and_staging_are_absent(self) -> None:
        self.assertNotIn("git restore", self.workflow)
        self.assertNotIn("rm -f", self.workflow)
        for fragment in (
            "backtest_",
            "signal_diagnostics",
            "portfolio_curve",
        ):
            self.assertNotIn(fragment, self.workflow)

    def test_pr22_publication_contract_is_preserved(self) -> None:
        required_fragments = [
            "workflow_dispatch:",
            'cron: "30 21 * * 1-5"',
            "timeout-minutes: 90",
            "group: data-publish-main",
            "cancel-in-progress: false",
            'SOURCE_SHA="$(git rev-parse HEAD)"',
            'echo "SOURCE_SHA=${SOURCE_SHA}" >> "${GITHUB_ENV}"',
            "if git diff --cached --quiet; then",
            "exit 0",
            "python scripts/verify_data_publish_base.py "
            '--source-sha "${SOURCE_SHA}"',
            'git commit -m "Update ETF scan data" '
            '-m "Generated-From: ${SOURCE_SHA}"',
            'GENERATED_PARENT_SHA="$(git rev-parse HEAD^)"',
            'if [ "${GENERATED_PARENT_SHA}" != "${SOURCE_SHA}" ]; then',
            "git push origin HEAD:main",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.workflow)

        for fragment in (
            "git pull",
            "git rebase",
            "git merge",
            "--autostash",
            "git push --force",
            "--force-with-lease",
        ):
            with self.subTest(forbidden=fragment):
                self.assertNotIn(fragment, self.workflow)

        push_lines = [
            line.strip()
            for line in self.workflow.splitlines()
            if line.strip().startswith("git push ")
        ]
        self.assertEqual(push_lines, ["git push origin HEAD:main"])


if __name__ == "__main__":
    unittest.main()
