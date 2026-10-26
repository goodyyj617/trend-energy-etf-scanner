"""Stage B runner (lab/combo_study.py, DECISIONS 80) on a tiny synthetic panel."""
import numpy as np
import pandas as pd
import pytest

from lab import combo_study as cs, research_log as rlog, screens
from lab.engine import StrategyConfig
from tests.test_lab import make_panel


def _panel():
    rng = np.random.default_rng(5)
    n = 600
    drift = {"UP1": 0.0012, "UP2": 0.0009, "FLAT": 0.0, "DOWN": -0.0008, "WILD": 0.0005, "SPY": 0.0004}
    vol = {"UP1": 0.015, "UP2": 0.012, "FLAT": 0.012, "DOWN": 0.012, "WILD": 0.04, "SPY": 0.01}
    prices = {s: list(50 * np.cumprod(1 + rng.normal(drift[s], vol[s], n))) for s in drift}
    panel, dates = make_panel(prices, start="2018-01-01")
    panel.extra = {"cap": pd.DataFrame({s: 1000.0 + 100 * i for i, s in enumerate(panel.symbols)}, index=dates),
                   "dividend": pd.DataFrame(0.0, index=dates, columns=panel.symbols)}
    panel.eligible = pd.DataFrame(True, index=dates, columns=panel.symbols)
    panel.cash = pd.Series(0.0001, index=dates)
    return panel, dates


class Serial:
    def map(self, fn, jobs, chunksize=1):
        return [fn(j) for j in jobs]


def test_menu_is_the_registered_one():
    assert len(cs.COMBOS) == 36 and sum(len(c) == 2 for c in cs.COMBOS) == 15 and sum(len(c) == 3 for c in cs.COMBOS) == 20
    assert cs.COMBOS[-1] == ["s_momentum"]
    assert all(k in screens.SELECT_SCORES for c in cs.COMBOS for k in c)
    assert len({cs.combo_id(c) for c in cs.COMBOS}) == 36
    assert cs.points().shape == (32, 7)
    np.testing.assert_array_equal(cs.points(), cs.points())  # the same points for every combination


def test_sobol_points_map_to_valid_configs():
    seen = set()
    for u in cs.points():
        p = cs.point_settings(u)
        cfg = cs.strategy_config(p, "full")
        StrategyConfig.from_dict(cfg.to_dict())
        assert p["exclude"] in cs.EXCLUDE_VARIANTS and p["k"] in cs.K_MENU and p["max_positions"] in cs.MAX_POS_MENU
        if p["layer"] == "hold":
            assert 15 <= p["trailing_pct"] <= 40 and "trailing_pct" in cfg.exits and "left_list" not in cfg.exits
        else:
            assert p["trailing_pct"] is None and "left_list" in cfg.exits and "trailing_pct" not in cfg.exits
        if p["atr_stop"] is None:  # no initial stop: equal slots, risk % unused
            assert cfg.sizing == "equal" and p["risk_pct"] is None and "atr_stop" not in cfg.exits
        else:
            assert cfg.sizing == "risk" and 0.5 <= cfg.risk_pct <= 2.0 and cfg.exits["atr_stop"] == p["atr_stop"]
        assert cfg.entries == {"always": None} and cfg.max_weight == 0.10 and cfg.cost_bps == 20.0
        seen.add((p["layer"], p["atr_stop"]))
    assert len(seen) == 8  # every layer x ATR option appears among the 32 points


def test_run_writes_per_combo_and_resumes(tmp_path, monkeypatch):
    panel, dates = _panel()
    monkeypatch.setattr(cs, "SAMPLES", {"a": (str(dates[300].date()), str(dates[-1].date()), "A"),
                                        "b": (str(dates[400].date()), str(dates[-1].date()), "B")})
    monkeypatch.setattr(cs, "COMBOS", [["s_dollar_volume", "s_efficiency"], ["s_momentum"]])
    monkeypatch.setattr(cs, "N_POINTS", 4)
    monkeypatch.setattr(rlog, "RESULTS_DIR", tmp_path / "res")
    cs._setup(panel)
    cs.precompute(Serial(), tmp_path, lambda m: None)
    cs._setup(panel, str(tmp_path))
    assert len(cs._W["screens"]) == len(cs.screen_keys())
    cs.run_combos(Serial(), tmp_path, panel, lambda m: None, chunk=1)
    files = sorted((tmp_path / "combos").glob("*/*.csv"))
    assert len(files) == 4
    g = pd.read_csv(files[0])
    assert list(g["i"]) == [0, 1, 2, 3] and {"goal_pass", "dd_excess", "window_share", "layer"} <= set(g.columns)
    n, _ = rlog.trial_stats()
    assert n == 16
    # resume: one file removed -> only that combination runs again
    files[0].unlink()
    calls = []

    class Counting(Serial):
        def map(self, fn, jobs, chunksize=1):
            calls.extend(jobs)
            return super().map(fn, jobs, chunksize)

    cs.run_combos(Counting(), tmp_path, panel, lambda m: None, chunk=1)
    assert len(calls) == 4 and len(sorted((tmp_path / "combos").glob("*/*.csv"))) == 4
    cells = cs.load_cells(tmp_path)
    rk = cs.ranks(cs.summaries(cells))
    assert set(rk["combo"]) == {"dollar_volume+efficiency", "momentum"}
    assert {"rank_a", "rank_b", "rank_sum", "both_pass"} <= set(rk.columns)
    lt = cs.layer_table(cells)
    assert {"pass_share_a", "pass_share_b"} <= set(lt.columns)
    both = cs.both_sample_passes(cells)
    assert list(both.columns[:2]) == ["combo", "i"]
    pb = cs.pbo_all(tmp_path)
    assert set(pb) == {"a", "b"}
    # the study for the app, the summary files and (with matplotlib) the figures
    from lab import crsp_study, figures, store
    monkeypatch.setattr(store, "RESULTS_DIR", tmp_path / "res")
    monkeypatch.setattr(crsp_study, "SAMPLES", cs.SAMPLES)  # curve_group reads the sample dates there
    summ = cs.save_study(tmp_path, panel, {"top_n": 10, "symbols": 6, "first": "2018", "last": "2020"})
    assert summ["runs"] == 16
    for f in ("summary.csv", "ranks.csv", "layers.csv", "both_pass_points.csv", "combo_summary.json"):
        assert (tmp_path / f).exists()
    if figures.available():
        assert len(list((tmp_path / "figures").glob("*.png"))) == 4
