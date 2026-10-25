"""Allocation lab: the global static search (DECISIONS A31) — fast simulator, samplers, metrics, PBO."""
import numpy as np
import pandas as pd
import pytest

from lab import aa_search as srch
from lab.aa_assets import CASH
from lab.aa_engine import AllocConfig, run_allocation
from lab.data import panel_from_long
from lab.metrics import curve_metrics


def make_panel(closes: dict[str, np.ndarray], start="2005-01-03", cash_rate=0.02):
    dates = pd.bdate_range(start, periods=len(next(iter(closes.values()))))
    rows = [{"date": d, "symbol": s, "open": v, "high": v, "low": v, "close": v, "volume": 1}
            for s, c in closes.items() for d, v in zip(dates, c)]
    panel = panel_from_long(pd.DataFrame(rows))
    panel.cash = pd.Series(cash_rate / 252, index=panel.close.index)
    return panel


def random_walks(n_days: int, n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        out[f"A{i}"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008 + 0.004 * i, n_days))
    out["SPY"] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n_days))
    return out


# --------------------------------------------------------------------------- fast simulator

@pytest.mark.parametrize("rebalance", ["Q", "M", "A"])
def test_static_sim_matches_the_engine(rebalance):
    panel = make_panel(random_walks(1400, 4, 1))
    syms = ["A0", "A1", "A2", "A3"]
    sim = srch.StaticSim(panel, syms, "2005-06-01", "2010-06-30", rebalance=rebalance, cost_bps=10.0)
    W = np.array([[40, 30, 20, 10, 0], [0, 50, 0, 20, 30], [100, 0, 0, 0, 0]], dtype=float) / 100.0  # last col = CASH
    paths, turnover = sim.run(W)
    assert paths.shape == (3, len(sim.dates))
    for k, w in enumerate(W):
        weights = {s: v * 100 for s, v in zip(syms + [CASH], w) if v > 0}
        res = run_allocation(panel, AllocConfig("t", weights, "2005-06-01", "2010-06-30", rebalance=rebalance, cost_bps=10.0))
        assert res.equity.index.equals(sim.dates)
        np.testing.assert_allclose(paths[k], res.equity.to_numpy(), rtol=1e-9, atol=0)
        assert turnover[k] == pytest.approx(res.turnover, rel=1e-9, abs=1e-12)


def test_static_sim_rejects_an_asset_that_starts_after_the_window():
    closes = random_walks(600, 2, 2)
    closes["LATE"] = closes["A0"].copy()
    closes["LATE"][:300] = np.nan
    panel = make_panel(closes)
    with pytest.raises(ValueError, match="LATE"):
        srch.StaticSim(panel, ["A0", "LATE"], "2005-01-03", "2007-12-31")


# --------------------------------------------------------------------------- samplers

def test_round_weights_gives_whole_percents_summing_to_100_and_drops_small_parts():
    w = np.array([0.334, 0.333, 0.02, 0.313])
    out = srch.round_weights(w, min_pct=3)
    assert out.dtype.kind in "iu" and out.sum() == 100 and out[2] == 0
    assert np.all((out == 0) | (out >= 3))
    assert srch.round_weights(np.array([0.5, 0.5]), min_pct=3).tolist() == [50, 50]


def test_two_level_sampler_respects_the_leverage_cap_and_product_counts():
    classes = {"미국 주식": ["S", "Q", "D"], "채권": ["B1", "B2"], "금": ["G"], "현금": ["C"], "레버리지": ["L1", "L2"]}
    W, meta = srch.sample_two_level(classes, n_class=64, picks=3, seed=5, lev_class="레버리지", lev_cap=0.2,
                                    equity_classes=("미국 주식",), min_pct=3)
    cols = list(meta["symbols"])
    assert W.shape == (64 * 3, len(cols)) and cols == ["S", "Q", "D", "B1", "B2", "G", "C", "L1", "L2"]
    assert np.all(W.sum(axis=1) == 100)
    lev = W[:, [cols.index("L1"), cols.index("L2")]].sum(axis=1)
    assert lev.max() <= 20 and (lev == 0).mean() > 0.3  # half of the class points carry no leverage
    assert np.all((W == 0) | (W >= 3))
    per_class = [(W[:, [cols.index(s) for s in syms]] > 0).sum(axis=1) for syms in classes.values()]
    assert max(per_class[0].max(), 1) <= 3 and all(p.max() <= 2 for p in per_class[1:])
    assert (W > 0).sum(axis=1).max() <= 10
    W2, _ = srch.sample_two_level(classes, n_class=64, picks=3, seed=5, lev_class="레버리지", lev_cap=0.2,
                                  equity_classes=("미국 주식",), min_pct=3)
    assert np.array_equal(W, W2)  # deterministic


def test_sparse_sampler_draws_3_to_10_products_without_class_structure():
    syms = [f"P{i}" for i in range(30)]
    W = srch.sample_sparse(syms, n=500, seed=3, k_range=(3, 10), min_pct=3)
    assert W.shape == (500, 30) and np.all(W.sum(axis=1) == 100)
    k = (W > 0).sum(axis=1)
    assert k.min() >= 1 and k.max() <= 10 and np.median(k) >= 4


def test_neighbours_move_five_points_between_two_products():
    w = np.array([50, 30, 20, 0])
    nb = srch.neighbours(w, step=5, min_pct=3)
    assert all(v.sum() == 100 for v in nb)
    assert any(v.tolist() == [45, 35, 20, 0] for v in nb)
    assert any(v.tolist() == [45, 30, 20, 5] for v in nb)  # a new product may enter
    assert not any(v.tolist() == [50, 30, 20, 0] for v in nb)
    assert all(np.all((v == 0) | (v >= 3)) for v in nb)


# --------------------------------------------------------------------------- metrics and dials

def test_summarize_matches_curve_metrics_and_window_drawdowns():
    panel = make_panel(random_walks(2600, 3, 7))
    syms = ["A0", "A1", "A2"]
    sim = srch.StaticSim(panel, syms, "2005-01-03", "2015-12-31")
    W = np.array([[50, 30, 20, 0]]) / 100.0
    paths, turnover = sim.run(W)
    windows = {"w1": ("2007-10-09", "2009-03-09"), "w2": ("2011-04-29", "2011-10-03")}
    df = srch.summarize(paths, turnover, sim.dates, sim.cash, windows=windows, halves=("2010-01-01",),
                        extra_starts=("2008-01-01",), holdout_share=0.15)
    eq = pd.Series(paths[0], index=sim.dates)
    m = curve_metrics(eq)
    assert df.loc[0, "cagr"] == pytest.approx(m["cagr"]) and df.loc[0, "mdd"] == pytest.approx(m["mdd"])
    w = eq.loc["2007-10-09":"2009-03-09"]
    assert df.loc[0, "dd_w1"] == pytest.approx(float((w / w.cummax() - 1).min()))
    h1 = eq.loc[:"2009-12-31"]
    assert df.loc[0, "mdd_h1"] == pytest.approx(float((h1 / h1.cummax() - 1).min()))
    s8 = eq.loc["2008-01-01":]
    assert df.loc[0, "cagr_2008"] == pytest.approx(curve_metrics(s8)["cagr"])
    ho = eq.iloc[-int(round(len(eq) * 0.15)):]
    assert df.loc[0, "cagr_holdout"] == pytest.approx(curve_metrics(ho)["cagr"])
    assert df.loc[0, "turnover"] == pytest.approx(turnover[0])


def test_rolling_window_share_counts_windows_within_the_mdd_line():
    idx = pd.bdate_range("2005-01-03", periods=252 * 12)
    flat = np.ones(len(idx))
    dip = flat.copy()
    dip[252 * 3:252 * 3 + 20] = 0.8  # one -20% hole in year 4: windows starting after it stay within the line
    share_flat, _ = srch.rolling_window_share(flat[None, :], idx, years=5, mdd_line=-0.15)
    share_dip, med = srch.rolling_window_share(dip[None, :], idx, years=5, mdd_line=-0.15)
    assert share_flat[0] == 1.0 and 0 < share_dip[0] < 1 and med.shape == (1,)


def test_pbo_selection_is_low_with_a_planted_edge_and_respects_the_mdd_rule():
    rng = np.random.default_rng(11)
    months = 240
    R = rng.normal(0.004, 0.02, (300, months))
    R[0] += 0.01  # a real edge
    R[1] = rng.normal(0.02, 0.09, months)  # high return, deep drawdowns: must not be picked under the MDD rule
    out = srch.pbo_selection(R, mdd_line=-0.15, blocks=10)
    assert out["splits"] == 252 and out["pbo"] < 0.2
    assert out["picked"][0] > 0.8 and out["picked"][1] == 0.0
    assert out["oos_within"].shape == (300,) and 0 <= out["oos_within"][0] <= 1


def test_static_sim_offset_matches_the_engine_timing_check():
    panel = make_panel(random_walks(900, 3, 4))
    syms = ["A0", "A1", "A2"]
    sim = srch.StaticSim(panel, syms, "2005-03-01", "2008-06-30", rebalance="Q", cost_bps=10.0, rebalance_offset=5)
    W = np.array([[50, 30, 20, 0]]) / 100.0
    paths, _ = sim.run(W)
    res = run_allocation(panel, AllocConfig("t", {"A0": 50, "A1": 30, "A2": 20}, "2005-03-01", "2008-06-30",
                                            rebalance="Q", cost_bps=10.0, rebalance_offset=5))
    np.testing.assert_allclose(paths[0], res.equity.to_numpy(), rtol=1e-9, atol=0)


def test_cagr_mdd_helper_matches_curve_metrics():
    rng = np.random.default_rng(9)
    paths = np.cumprod(1 + rng.normal(0.0003, 0.01, (3, 800)), axis=1)
    cagr, mdd = srch.cagr_mdd(paths)
    for k in range(3):
        m = curve_metrics(pd.Series(paths[k], index=pd.bdate_range("2005-01-03", periods=800)))
        assert cagr[k] == pytest.approx(m["cagr"]) and mdd[k] == pytest.approx(m["mdd"])


def test_round_weights_keeps_the_invariants_on_random_simplex_points():
    rng = np.random.default_rng(0)
    for _ in range(5000):
        k = int(rng.integers(2, 11))
        out = srch.round_weights(rng.dirichlet(np.ones(k)), min_pct=3)
        assert out.sum() == 100 and np.all((out == 0) | (out >= 3))
    for d in (3, 7, 10):  # parts that sit exactly on the 3 % line after renormalisation
        out = srch.round_weights(np.array([0.03 * d, 0.97 * d]), min_pct=3)
        assert out.tolist() == [3, 97]
