"""Stage E (lab/screen_e_study.py, DECISIONS 83) on small synthetic data."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lab import research_log as rlog, screen_e_study as se, signal_replay as sr, stage_c_study as sc
from lab.engine import run_backtest
from tests.test_stage_c import _study_panel


class Serial:
    def map(self, fn, jobs, chunksize=1):
        return [fn(j) for j in jobs]


def test_menu_is_registered():
    assert se.screen_names()[0] == "none" and len(se.screen_names()) == 16
    assert "tt_fail" not in se.EXCLUDE and se.TOP_SHARE == 0.30 and se.HELP_SHARE == 0.60


def test_calendar_portfolio_by_hand():
    # 4 days, 2 symbols; A bought at day-1 open, sold at day-3 open; B bought at day-2 open, still open
    O = np.array([[10, 20], [10, 20], [11, 20], [12, 22]], dtype=float)
    C = np.array([[10, 20], [11, 20], [12, 21], [12, 22]], dtype=float)
    cash = np.full(4, 0.001)
    eq, n = se.calendar_portfolio(np.array([1, 2]), np.array([3, 2]), np.array([True, False]), np.array([0, 1]),
                                  O, C, cash, 0.0)
    r = [0.001, 11 / 10 - 1, ((12 / 11 - 1) + (21 / 20 - 1)) / 2, ((12 / 12 - 1) + (22 / 21 - 1)) / 2]
    np.testing.assert_allclose(eq, np.cumprod(1 + np.array(r)))
    assert list(n) == [0, 1, 2, 2]
    # cost on the entry and exit days only
    eq_c, _ = se.calendar_portfolio(np.array([1]), np.array([3]), np.array([True]), np.array([0]), O, C, cash, 0.01)
    r_c = [0.001, 11 / (10 * 1.01) - 1, 12 / 11 - 1, 12 * 0.99 / 12 - 1]
    np.testing.assert_allclose(eq_c, np.cumprod(1 + np.array(r_c)))


def test_pipeline_masks_replays_and_paired_summary(tmp_path, monkeypatch):
    p, dates = _study_panel()
    monkeypatch.setattr(sc, "SAMPLES", {"a": (str(dates[260].date()), str(dates[-1].date()), "A")})
    monkeypatch.setattr(se, "SAMPLES", sc.SAMPLES)
    monkeypatch.setattr(sc, "ENTRIES", {"E0": sc.ENTRIES["E0"], "E1": sc.ENTRIES["E1"]})
    monkeypatch.setattr(sc, "EXITS", {"X1": sc.EXITS["X1"]})
    monkeypatch.setattr(sc, "HYGIENE", {"h_price": 5})
    monkeypatch.setattr(sc, "SLOTS", 2)
    monkeypatch.setattr(se, "EXCLUDE", {"x_volatility": 20})
    monkeypatch.setattr(se, "SELECT", ["s_efficiency"])
    monkeypatch.setattr(rlog, "RESULTS_DIR", tmp_path / "res")
    elig = sc.daily_eligible(p)
    sc._setup(p, elig)
    sc.run_signal_tables(Serial(), tmp_path / "sig", lambda m: None)
    pe = replace(p, eligible=elig)
    se.precompute_masks(pe, elig, tmp_path / "out", lambda m: None)
    m = pd.read_parquet(se.mask_file(tmp_path / "out", "s_efficiency"))
    assert (m.sum(axis=1) <= np.ceil(elig.sum(axis=1) * 0.3) + 1).all() and not (m & ~elig).any().any()
    se._setup(pe, str(tmp_path / "sig"), str(tmp_path / "out"))
    se.run_pairs(Serial(), tmp_path / "out", pe, lambda m: None)
    rows = pd.read_csv(tmp_path / "out" / "a" / "E1__X1.csv")
    assert len(rows) == 9 * 3 and {"signal_share", "bought_share", "expectancy"} <= set(rows.columns)
    none = rows[rows["screen"] == "none"]
    assert (none["signal_share"] == 1.0).all()
    canon = rows[(rows["xi"] == 1) & (rows["yi"] == 1)]
    assert canon["cal_cagr"].notna().all() and rows.loc[~rows.index.isin(canon.index), "cal_cagr"].isna().all()
    # 'none' replay = the engine's 20-slot account
    cfg = replace(sc.config("a", "E1", "X1", 1, 1), max_positions=2)
    eqs = pd.read_parquet(tmp_path / "out" / "a" / "E1__X1_equity.parquet")
    np.testing.assert_allclose(eqs["1|1|none"].to_numpy(dtype=float), run_backtest(pe, cfg).equity.to_numpy(), rtol=1e-6)
    summ = se.summarize(tmp_path / "out")
    scr = pd.read_csv(tmp_path / "out" / "screens.csv")
    assert {"share_better_expectancy", "median_d_cagr", "helps_unlimited", "helps_account", "cal_median_cagr"} <= set(scr.columns)
    base = scr[scr["screen"] == "none"].iloc[0]
    assert base["median_d_expectancy"] == 0 and base["share_better_expectancy"] == 0  # 'none' never beats itself
    assert summ["helps_unlimited"] == sorted(summ["helps_unlimited"])
