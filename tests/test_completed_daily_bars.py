import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from src import run_backtest_only
from src.prices import NEW_YORK_TIMEZONE, filter_completed_daily_bars


class CompletedDailyBarFilterTest(unittest.TestCase):
    @staticmethod
    def prices() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2026-07-16",
                    "symbol": "AAA",
                    "open": 13.0,
                    "high": 14.0,
                    "low": 12.0,
                    "close": 13.5,
                    "volume": 130,
                },
                {
                    "date": "2026-07-14",
                    "symbol": "BBB",
                    "open": 20.0,
                    "high": 21.0,
                    "low": 19.0,
                    "close": 20.5,
                    "volume": 200,
                },
                {
                    "date": "2026-07-15",
                    "symbol": "AAA",
                    "open": 12.0,
                    "high": 13.0,
                    "low": 11.0,
                    "close": 12.5,
                    "volume": 120,
                },
                {
                    "date": "2026-07-14",
                    "symbol": "AAA",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 100,
                },
            ]
        )

    def test_before_cutoff_excludes_current_and_future_rows(self) -> None:
        now = datetime(2026, 7, 15, 20, 14, tzinfo=NEW_YORK_TIMEZONE)
        filtered, provenance = filter_completed_daily_bars(self.prices(), now=now)

        expected = self.prices().loc[
            self.prices()["date"].eq("2026-07-14")
        ].sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(filtered, expected)
        self.assertEqual(provenance["new_york_evaluation_time"], now.isoformat())
        self.assertEqual(provenance["safe_latest_daily_bar_date"], "2026-07-14")
        self.assertEqual(provenance["downloaded_max_date"], "2026-07-16")
        self.assertEqual(provenance["retained_max_date"], "2026-07-14")
        self.assertEqual(provenance["rows_removed"], 2)

    def test_at_cutoff_retains_current_row_but_excludes_future_row(self) -> None:
        now = datetime(2026, 7, 15, 20, 15, tzinfo=NEW_YORK_TIMEZONE)
        filtered, provenance = filter_completed_daily_bars(self.prices(), now=now)

        self.assertEqual(
            filtered["date"].tolist(),
            ["2026-07-14", "2026-07-15", "2026-07-14"],
        )
        self.assertNotIn("2026-07-16", filtered["date"].tolist())
        self.assertEqual(provenance["safe_latest_daily_bar_date"], "2026-07-15")
        self.assertEqual(provenance["retained_max_date"], "2026-07-15")
        self.assertEqual(provenance["rows_removed"], 1)

    def test_standard_time_result_is_independent_of_input_timezone(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": "2026-01-14", "symbol": "AAA", "close": 10.0},
                {"date": "2026-01-15", "symbol": "AAA", "close": 11.0},
                {"date": "2026-01-16", "symbol": "AAA", "close": 12.0},
            ]
        )
        utc_now = datetime(2026, 1, 16, 1, 15, tzinfo=timezone.utc)
        seoul_now = utc_now.astimezone(ZoneInfo("Asia/Seoul"))

        utc_result, utc_provenance = filter_completed_daily_bars(prices, now=utc_now)
        seoul_result, seoul_provenance = filter_completed_daily_bars(
            prices, now=seoul_now
        )

        pd.testing.assert_frame_equal(utc_result, seoul_result)
        self.assertEqual(utc_result["date"].tolist(), ["2026-01-14", "2026-01-15"])
        self.assertEqual(
            utc_provenance["safe_latest_daily_bar_date"],
            seoul_provenance["safe_latest_daily_bar_date"],
        )
        self.assertEqual(utc_provenance["safe_latest_daily_bar_date"], "2026-01-15")

    def test_naive_now_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            filter_completed_daily_bars(
                self.prices(),
                now=datetime(2026, 7, 15, 20, 15),
            )


class BacktestOnlyCompletedBarGuardTest(unittest.TestCase):
    @staticmethod
    def config() -> dict:
        return {
            "lookback_period": "1y",
            "backtest_lookback_period": "10y",
            "price_interval": "1d",
        }

    @staticmethod
    def universe() -> pd.DataFrame:
        return pd.DataFrame(
            {"symbol": ["AAA"], "base_universe_eligible": [True]}
        )

    def test_backtest_as_of_comes_from_filtered_prices(self) -> None:
        downloaded = pd.DataFrame(
            [
                {"date": "2026-07-14", "symbol": "AAA", "close": 10.0},
                {"date": "2026-07-15", "symbol": "AAA", "close": 11.0},
            ]
        )
        now = datetime(2026, 7, 15, 20, 14, tzinfo=NEW_YORK_TIMEZONE)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(run_backtest_only, "ROOT", Path(temp_dir)),
                patch.object(
                    run_backtest_only,
                    "load_config",
                    return_value=self.config(),
                ),
                patch.object(
                    run_backtest_only,
                    "build_base_universe",
                    return_value=self.universe(),
                ),
                patch.object(
                    run_backtest_only,
                    "download_ohlcv",
                    return_value=downloaded,
                ),
                patch.object(
                    run_backtest_only,
                    "run_backtests",
                    return_value={"summary": []},
                ) as run_backtests,
            ):
                run_backtest_only.main(now=now)

        call = run_backtests.call_args.kwargs
        self.assertEqual(call["as_of"], "2026-07-14")
        self.assertEqual(call["prices"]["date"].tolist(), ["2026-07-14"])

    def test_empty_filtered_result_fails_before_backtest(self) -> None:
        downloaded = pd.DataFrame(
            [{"date": "2026-07-15", "symbol": "AAA", "close": 11.0}]
        )
        now = datetime(2026, 7, 15, 20, 14, tzinfo=NEW_YORK_TIMEZONE)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(run_backtest_only, "ROOT", Path(temp_dir)),
                patch.object(
                    run_backtest_only,
                    "load_config",
                    return_value=self.config(),
                ),
                patch.object(
                    run_backtest_only,
                    "build_base_universe",
                    return_value=self.universe(),
                ),
                patch.object(
                    run_backtest_only,
                    "download_ohlcv",
                    return_value=downloaded,
                ),
                patch.object(run_backtest_only, "run_backtests") as run_backtests,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "No completed daily price bars remain after filtering",
                ):
                    run_backtest_only.main(now=now)

        run_backtests.assert_not_called()


class BacktestOnlyWorkflowScheduleTest(unittest.TestCase):
    def test_workflow_uses_new_york_schedule_and_keeps_manual_dispatch(self) -> None:
        workflow = (
            Path(__file__).parents[1] / ".github" / "workflows" / "backtest-only.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn('cron: "30 20 * * 1-5"', workflow)
        self.assertIn('timezone: "America/New_York"', workflow)
        self.assertNotIn('cron: "0 18 * * 1-5"', workflow)


if __name__ == "__main__":
    unittest.main()
