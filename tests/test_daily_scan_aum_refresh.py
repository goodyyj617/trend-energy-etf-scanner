import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import run_daily_scan, update_aum


class DailyScanAumRefreshTest(unittest.TestCase):
    def test_daily_scan_invokes_aum_update_once_without_arguments(self) -> None:
        with patch.object(run_daily_scan, "update_aum_csv") as update:
            run_daily_scan._update_aum_cache_safely()

        update.assert_called_once_with()

    def test_aum_update_failure_is_caught_and_scan_can_continue(self) -> None:
        error = RuntimeError("fixture failure")
        with (
            patch.object(run_daily_scan, "update_aum_csv", side_effect=error),
            patch("builtins.print") as print_mock,
        ):
            run_daily_scan._update_aum_cache_safely()

        print_mock.assert_called_once_with(
            "[AUM] update failed, continuing scan with existing config/aum.csv: fixture failure"
        )

    def test_auto_aum_config_is_read_from_top_level_yaml_section(self) -> None:
        config_text = """
universe:
  lookback_period: 5y
auto_aum:
  enabled: true
  max_new_per_run: 17
  time_budget_minutes: 12.5
  sleep_seconds: 0.25
  refresh_existing_days: 45
  skip_obvious_exclusions: false
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "universe.yml"
            config_path.write_text(config_text, encoding="utf-8")

            with patch.object(update_aum, "UNIVERSE_YML_PATH", config_path):
                cfg = update_aum.get_auto_aum_config()

        self.assertEqual(
            cfg,
            {
                "enabled": True,
                "max_new_per_run": 17,
                "time_budget_minutes": 12.5,
                "sleep_seconds": 0.25,
                "refresh_existing_days": 45,
                "skip_obvious_exclusions": False,
            },
        )

    def test_disabled_config_is_handled_by_update_aum_module(self) -> None:
        cached = pd.DataFrame({"symbol": ["SPY"]})
        with (
            patch.object(
                update_aum,
                "get_auto_aum_config",
                return_value={"enabled": False},
            ) as get_config,
            patch.object(update_aum, "load_aum_cache", return_value=cached) as load_cache,
            patch.object(update_aum, "fetch_nasdaq_etf_candidates") as fetch_candidates,
        ):
            result = update_aum.update_aum_csv()

        get_config.assert_called_once_with()
        load_cache.assert_called_once_with()
        fetch_candidates.assert_not_called()
        self.assertIs(result, cached)


if __name__ == "__main__":
    unittest.main()
