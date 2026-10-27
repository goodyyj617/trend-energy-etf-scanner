"""Stage D (lab/missed_study.py, DECISIONS 82) on a tiny synthetic panel."""
from dataclasses import replace

import numpy as np
import pandas as pd

from lab import missed_study as md, research_log as rlog, stage_c_study as sc
from lab.engine import run_backtest
from tests.test_stage_c import _study_panel


class Serial:
    def map(self, fn, jobs, chunksize=1):
        return [fn(j) for j in jobs]


def test_menu_and_capacity():
    assert md.variants() == [(b, p) for b in ("all", "ex500", "ex1000") for p in ("dv", "small", "random")]
    assert md.COSTS == (20.0, 40.0)
    d = pd.bdate_range("2020-01-01", periods=5)
    eq = pd.Series([1.0, 1.0, 2.0, 2.0, 2.0], index=d)
    dv20 = pd.DataFrame({"A": 1e6, "B": 1e5}, index=d)
    trades = pd.DataFrame({"symbol": ["A", "B"], "entry_date": d[[2, 3]]})
    cap = md.capacity(trades, eq, dv20, 20)
    # A: 100k x 1.0 / 20 = 5k of 1M = 0.5%; B: 100k x 2.0 / 20 = 10k of 100k = 10%
    assert cap["order_share_median_100k"] == np.median([0.005, 0.1]) and cap["order_share_over5_100k"] == 0.5
    assert cap["order_share_median_1000k"] == np.median([0.05, 1.0]) and cap["order_share_over5_1000k"] == 0.5  # 5% is not above 5%


def test_pair_replays_match_the_engine_and_summarize(tmp_path, monkeypatch):
    p, dates = _study_panel()
    monkeypatch.setattr(sc, "SAMPLES", {"a": (str(dates[260].date()), str(dates[-1].date()), "A")})
    monkeypatch.setattr(md, "SAMPLES", sc.SAMPLES)
    monkeypatch.setattr(sc, "ENTRIES", {"E0": sc.ENTRIES["E0"], "E1": sc.ENTRIES["E1"]})
    monkeypatch.setattr(sc, "EXITS", {"X1": sc.EXITS["X1"]})
    monkeypatch.setattr(sc, "HYGIENE", {"h_price": 5})
    monkeypatch.setattr(sc, "SLOTS", 2)
    monkeypatch.setattr(md, "BANDS", {"all": (1, 3000), "ex500": (3, 3000), "ex1000": (5, 3000)})  # 6 names: rank cuts
    monkeypatch.setattr(rlog, "RESULTS_DIR", tmp_path / "res")
    elig = sc.daily_eligible(p)
    sc._setup(p, elig)
    sc.run_signal_tables(Serial(), tmp_path / "sig", lambda m: None)
    pe = replace(p, eligible=elig)
    md._setup(pe, str(tmp_path / "sig"))
    md.run_pairs(Serial(), tmp_path / "out", pe, lambda m: None)
    rows = pd.read_csv(tmp_path / "out" / "a" / "E1__X1.csv")
    assert len(rows) == 9 * 3 * 3 * 2 and set(rows["cost_bps"]) == {20.0, 40.0}
    eqs = pd.read_parquet(tmp_path / "out" / "a" / "E1__X1_equity.parquet")
    # all x dollar volume x 20 bp = the engine's 20-slot account on the same cell
    cfg = replace(sc.config("a", "E1", "X1", 1, 1), max_positions=2)
    res = run_backtest(pe, cfg)
    np.testing.assert_allclose(eqs["1|1|all|dv|20"].to_numpy(dtype=float), res.equity.to_numpy(), rtol=1e-6)
    # a band only removes the larger names' signals: the 'small first' and band accounts are different accounts
    assert not np.allclose(eqs["1|1|all|dv|20"], eqs["1|1|ex1000|small|20"])
    r20 = rows[rows["cost_bps"] == 20.0]
    assert r20["band_signals"].notna().all() and (r20["taken"] + r20["missed"] == r20["band_signals"]).all()
    assert {"worst_year", "volatility", "dd_cagr", "order_share_median_100k"} <= set(rows.columns)
    hi = rows.set_index(["xi", "yi", "band", "priority", "cost_bps"])["cagr"]
    assert (hi.xs(40.0, level="cost_bps") <= hi.xs(20.0, level="cost_bps") + 1e-12).all()  # costs only hurt
    summ = md.summarize(tmp_path / "out")
    assert summ["cells"] == len(rows) and (tmp_path / "out" / "variants.csv").exists()
    assert "cagr_rank" in pd.read_csv(tmp_path / "out" / "variants.csv").columns
