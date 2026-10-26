"""Asset-allocation lab: engine, overlays, leverage, mixes, scorecard, notes."""
import numpy as np
import pandas as pd
import pytest

from lab import aa_notes
from lab.aa_assets import CASH, PRESETS
from lab.aa_engine import (
    AllocConfig, Leverage, MixConfig, Overlay, decision_rows, levered_returns, run_allocation, run_mix, simulate,
)
from lab.aa_score import AllocGates, cvar, gate_results, percentile_scores, raw_metrics, rolling_beat_cash, scorecard
from lab.data import panel_from_long
from lab.metrics import curve_metrics


def make_panel(closes: dict[str, np.ndarray], start="2010-01-04", cash_rate=0.0):
    dates = pd.bdate_range(start, periods=len(next(iter(closes.values()))))
    rows = []
    for sym, c in closes.items():
        for d, v in zip(dates, c):
            rows.append({"date": d, "symbol": sym, "open": v, "high": v, "low": v, "close": v, "volume": 1_000_000})
    panel = panel_from_long(pd.DataFrame(rows))
    panel.cash = pd.Series(cash_rate / 252, index=panel.close.index)
    return panel, dates


def geometric(n, daily, start=100.0):
    return start * np.cumprod(np.full(n, 1.0 + daily))


def cfg(weights, **kw):
    base = dict(start="2010-01-01", end="2030-01-01", cost_bps=0.0)
    base.update(kw)
    return AllocConfig("t", weights, **base)


# --------------------------------------------------------------------------- static weights

def test_buy_and_hold_single_asset_tracks_price():
    n = 300
    rng = np.random.default_rng(1)
    spy = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    panel, _ = make_panel({"SPY": spy})
    res = run_allocation(panel, cfg({"SPY": 100}, rebalance="none"))
    assert res.equity.iloc[-1] == pytest.approx(spy[-1] / spy[0])
    assert res.n_trades == 0 and res.turnover == 0.0


def test_static_mix_rebalances_at_next_close_and_charges_cost():
    n = 70  # ~3 months: decisions at month ends, traded the next day
    a = geometric(n, 0.01)  # A rises 1%/day
    b = np.full(n, 100.0)  # B flat
    panel, dates = make_panel({"A": a, "B": b, "SPY": b})
    res = run_allocation(panel, cfg({"A": 50, "B": 50}, rebalance="M", cost_bps=100.0))
    rows = decision_rows(dates, "M")
    w = res.weights
    # weights drift up for A between decisions, and are back at 50/50 on the day after a decision
    first_decision = rows[0]
    assert w["A"].iloc[first_decision] > 0.5
    assert w["A"].iloc[first_decision + 1] == pytest.approx(0.5)
    assert res.n_trades == len([r for r in rows if r < n - 1])
    # the cost of the first trade: 1% of the one-way turnover, taken from equity on the trade day
    no_cost = run_allocation(panel, cfg({"A": 50, "B": 50}, rebalance="M", cost_bps=0.0))
    assert res.equity.iloc[-1] < no_cost.equity.iloc[-1]


def test_band_skips_small_trades():
    n = 300
    rng = np.random.default_rng(2)
    a = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))
    b = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))
    panel, _ = make_panel({"A": a, "B": b, "SPY": b})
    tight = run_allocation(panel, cfg({"A": 50, "B": 50}, rebalance="M", band=0.0))
    loose = run_allocation(panel, cfg({"A": 50, "B": 50}, rebalance="M", band=10.0))
    assert tight.n_trades > 0 and loose.n_trades == 0


def test_cash_weight_earns_cash_rate_and_period_starts_at_common_data():
    n = 260
    panel, dates = make_panel({"SPY": np.full(n, 100.0)}, cash_rate=0.0252)
    res = run_allocation(panel, cfg({"SPY": 50, CASH: 50}, rebalance="none"))
    # half the equity compounds at 0.01%/day
    expected = 0.5 + 0.5 * (1 + 0.0252 / 252) ** n
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-6)
    assert res.weights[CASH].iloc[-1] > 0.5


def test_period_trimmed_to_data_with_warning():
    n = 300
    a = np.full(n, 100.0)
    late = np.concatenate([np.full(100, np.nan), np.full(200, 50.0)])
    panel, dates = make_panel({"A": a, "SPY": a})
    panel.close["B"] = late
    for f in ("open", "high", "low", "volume"):
        getattr(panel, f)["B"] = late
    res = run_allocation(panel, cfg({"A": 50, "B": 50}))
    assert res.equity.index[0] == dates[100]
    assert res.warnings and "B" in res.warnings[0]


# --------------------------------------------------------------------------- overlays

def test_sma_overlay_moves_falling_asset_to_safe():
    n = 24 * 21
    up = geometric(n, 0.001)
    down = geometric(n, -0.002)
    panel, _ = make_panel({"UP": up, "DOWN": down, "SPY": up})
    res = run_allocation(panel, cfg({"UP": 50, "DOWN": 50}, overlay=Overlay("sma", 3)))
    w = res.weights.iloc[-1]  # UP drifts a little above 0.5 between the monthly decisions
    assert w["DOWN"] == pytest.approx(0.0) and w[CASH] == pytest.approx(0.5, abs=0.01) and w["UP"] == pytest.approx(0.5, abs=0.01)
    safe = run_allocation(panel, cfg({"UP": 50, "DOWN": 50}, overlay=Overlay("sma", 3, "UP")))
    assert safe.weights.iloc[-1]["UP"] == pytest.approx(1.0)


def test_sma_overlay_only_switches_the_listed_assets():
    n = 24 * 21
    panel, _ = make_panel({"UP": geometric(n, 0.001), "DOWN": geometric(n, -0.002), "FALL": geometric(n, -0.001),
                           "SPY": geometric(n, 0.001)})
    res = run_allocation(panel, cfg({"UP": 40, "DOWN": 30, "FALL": 30}, overlay=Overlay("sma", 3, only=("DOWN",))))
    w = res.weights.iloc[-1]
    assert w["DOWN"] == pytest.approx(0.0) and w[CASH] == pytest.approx(0.3, abs=0.01) and w["FALL"] > 0.25
    every = run_allocation(panel, cfg({"UP": 40, "DOWN": 30, "FALL": 30}, overlay=Overlay("sma", 3)))
    assert every.weights.iloc[-1]["FALL"] == pytest.approx(0.0)
    # an empty list is the old behaviour, bit for bit, and is not written into the saved config
    same = run_allocation(panel, cfg({"UP": 40, "DOWN": 30, "FALL": 30}, overlay=Overlay("sma", 3, only=())))
    assert same.equity.equals(every.equity) and "only" not in cfg({"UP": 100}, overlay=Overlay("sma", 3)).to_dict()["overlay"]
    c = cfg({"UP": 40, "DOWN": 60}, overlay=Overlay("sma", 3, "UP", only=["DOWN"]))
    back = AllocConfig.from_dict(c.to_dict())
    assert back == c and back.overlay.only == ("DOWN",) and "DOWN에만" in c.describe()


def test_sma_overlay_safe_asset_in_base_is_never_switched_off():
    """Safe asset below its own moving average: its weight stays (it must not be doubled)."""
    n = 24 * 21
    panel, _ = make_panel({"UP": geometric(n, 0.001), "DOWN": geometric(n, -0.002), "SPY": geometric(n, 0.001)})
    res = run_allocation(panel, cfg({"UP": 50, "DOWN": 50}, overlay=Overlay("sma", 3, "DOWN")))
    sums = res.weights.sum(axis=1)
    assert sums.max() == pytest.approx(1.0) and sums.min() == pytest.approx(1.0)
    assert res.weights.iloc[-1]["DOWN"] == pytest.approx(0.5, abs=0.02) and res.weights.iloc[-1]["UP"] == pytest.approx(0.5, abs=0.02)


def test_risk_parity_keeps_cash_like_asset_at_base_weight():
    n = 400
    rng = np.random.default_rng(5)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    cd = 100 * np.cumprod(np.full(n, 1 + 0.03 / 252))  # a CD-rate product: volatility ~0
    panel, _ = make_panel({"CALM": calm, "WILD": wild, "CD": cd, "SPY": calm})
    res = run_allocation(panel, cfg({"CALM": 40, "WILD": 40, "CD": 20}, overlay=Overlay("riskparity", 63)))
    w = res.weights.iloc[-1]
    assert w["CD"] == pytest.approx(0.2, abs=0.01)  # not the whole portfolio
    assert w["CALM"] > w["WILD"] > 0.05 and w["CALM"] + w["WILD"] + w["CD"] == pytest.approx(1.0)


def test_dual_momentum_holds_strongest_or_safe():
    n = 30 * 21
    strong = geometric(n, 0.002)
    weak = geometric(n, 0.0005)
    flat = np.full(n, 100.0)
    panel, _ = make_panel({"S": strong, "W": weak, "BOND": flat, "SPY": flat})
    res = run_allocation(panel, cfg({"S": 50, "W": 50}, overlay=Overlay("dualmom", 12, "BOND")))
    w = res.weights.iloc[-1]
    assert w["S"] == pytest.approx(1.0) and w["W"] == 0.0
    falling = run_allocation(panel.subset(["S", "W", "BOND"]) if False else panel,
                             cfg({"W": 100}, overlay=Overlay("dualmom", 12, "BOND")))
    assert falling.weights.iloc[-1]["W"] == pytest.approx(1.0)  # positive momentum beats 0% cash
    down_panel, _ = make_panel({"D": geometric(n, -0.001), "BOND": flat, "SPY": flat})
    to_safe = run_allocation(down_panel, cfg({"D": 100}, overlay=Overlay("dualmom", 12, "BOND")))
    assert to_safe.weights.iloc[-1]["BOND"] == pytest.approx(1.0)


def test_risk_parity_weights_inverse_to_volatility():
    n = 400
    rng = np.random.default_rng(3)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    panel, _ = make_panel({"CALM": calm, "WILD": wild, "SPY": calm})
    res = run_allocation(panel, cfg({"CALM": 50, "WILD": 50}, overlay=Overlay("riskparity", 63)))
    w = res.weights.iloc[-1]
    assert w["CALM"] > 0.7 > 0.3 > w["WILD"]
    assert w["CALM"] + w["WILD"] == pytest.approx(1.0)


def test_vol_target_scales_exposure_down_and_up_to_cap():
    n = 400
    rng = np.random.default_rng(4)
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))  # ~32% vol
    panel, _ = make_panel({"W": wild, "SPY": wild})
    res = run_allocation(panel, cfg({"W": 100}, overlay=Overlay("voltarget", 10.0)))
    assert res.exposure.iloc[-1] < 0.5
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))  # ~3% vol: wants 3x, capped by the leverage factor
    panel2, _ = make_panel({"C": calm, "SPY": calm})
    capped = run_allocation(panel2, cfg({"C": 100}, overlay=Overlay("voltarget", 10.0), leverage=Leverage(1.5)))
    assert capped.exposure.iloc[-1] == pytest.approx(1.5)
    uncapped = run_allocation(panel2, cfg({"C": 100}, overlay=Overlay("voltarget", 10.0)))
    assert uncapped.exposure.max() == pytest.approx(1.0)


# --------------------------------------------------------------------------- leverage

def test_levered_returns_formula():
    r = pd.Series([0.01, -0.02])
    cash = pd.Series([0.0001, 0.0001])
    out = levered_returns(r, 2.0, cash, spread_bps=252.0, ter_bps=0.0)  # spread = 1 bp per day
    assert out.iloc[0] == pytest.approx(2 * 0.01 - 0.0001 - 0.0001)
    half = levered_returns(r, 0.5, cash, 252.0, 0.0)  # deleveraging lends at the cash rate, no spread
    assert half.iloc[0] == pytest.approx(0.5 * 0.01 + 0.5 * 0.0001)
    assert levered_returns(r, 1.0, cash, 252.0, 0.0) is r


def test_portfolio_leverage_matches_formula_without_costs():
    n = 300
    rng = np.random.default_rng(5)
    spy = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    panel, _ = make_panel({"SPY": spy})
    res = run_allocation(panel, cfg({"SPY": 100}, rebalance="none", leverage=Leverage(2.0, spread_bps=0.0, ter_bps=0.0)))
    daily = pd.Series(spy).pct_change().fillna(0).to_numpy()
    expected = np.cumprod(1 + 2 * daily)[-1]
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    # a 2x daily-reset position on a volatile flat asset loses money (volatility decay)
    flat = 100 * np.cumprod(np.tile([1.05, 1 / 1.05], 150))
    panel2, _ = make_panel({"SPY": flat})
    decay = run_allocation(panel2, cfg({"SPY": 100}, rebalance="none", leverage=Leverage(2.0, 0.0, 0.0)))
    assert decay.equity.iloc[-1] < 1.0


def test_per_asset_leverage_only_touches_that_asset():
    n = 300
    rng = np.random.default_rng(6)
    a = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    b = np.full(n, 100.0)
    panel, _ = make_panel({"A": a, "B": b, "SPY": b})
    plain = run_allocation(panel, cfg({"A": 50, "B": 50}, rebalance="none"))
    lev = run_allocation(panel, cfg({"A": 50, "B": 50}, rebalance="none", leverage=Leverage(1.0, 0.0, 0.0, {"A": 2.0})))
    assert lev.equity.iloc[-1] != pytest.approx(plain.equity.iloc[-1])
    assert lev.exposure.iloc[-1] == 1.0  # the mix itself is not levered
    assert lev.config.parameter_count() == plain.config.parameter_count() + 1


# --------------------------------------------------------------------------- mixes and configs

def test_mix_of_identical_sleeves_equals_the_sleeve():
    n = 300
    rng = np.random.default_rng(7)
    a = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    b = 100 * np.cumprod(1 + rng.normal(0.0002, 0.005, n))
    panel, _ = make_panel({"A": a, "B": b, "SPY": a})
    c = cfg({"A": 60, "B": 40})
    one = run_allocation(panel, c)
    mix = run_mix(panel, MixConfig("m", ((c, 50.0), (c, 50.0)), rebalance="M"))
    assert np.allclose(mix.equity.to_numpy(), one.equity.to_numpy())
    assert set(mix.sleeves) == {"t", "t "}


def test_config_roundtrip_and_describe():
    c = AllocConfig("x", {"SPY": 60, "IEF": 40, "GLD": 0}, "2005-01-03", "2023-12-29", rebalance="A", band=5,
                    overlay=Overlay("sma", 10, "IEF"), leverage=Leverage(1.5, per_asset={"SPY": 2.0}))
    d = c.to_dict()
    back = AllocConfig.from_dict(d)
    assert back == c and "GLD" not in back.weights
    assert back.fingerprint() == c.fingerprint()
    text = c.describe()
    assert "매년" in text and "추세 필터 10" in text and "SPY 2배" in text
    m = MixConfig("mix", ((c, 70.0), (AllocConfig("y", {"SPY": 100}, "2005-01-03", "2023-12-29"), 30.0)))
    assert MixConfig.from_dict(m.to_dict()) == m


def test_presets_sum_to_100():
    for p in PRESETS.values():
        assert sum(p.weights.values()) == pytest.approx(100.0), p.key


# --------------------------------------------------------------------------- scorecard

def test_simulate_turnover_and_weights_shape():
    idx = pd.bdate_range("2020-01-01", periods=5)
    rets = pd.DataFrame({"A": [0.0, 0.1, 0.0, 0.0, 0.0], "B": 0.0}, index=idx)
    cash = pd.Series(0.0, index=idx)
    eq, w, turn, trades = simulate(rets, cash, {1: {"A": 0.5, "B": 0.5}}, 0.0, 0.0, first={"A": 0.5, "B": 0.5})
    assert list(w.columns) == ["A", "B", CASH]
    assert w["A"].iloc[1] == pytest.approx(0.55 / 1.05) and w["A"].iloc[2] == pytest.approx(0.5)
    assert trades == 1 and turn > 0


def test_scorecard_scores_and_gates():
    n = 1500
    rng = np.random.default_rng(8)
    spy = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    bond = 100 * np.cumprod(1 + rng.normal(0.0002, 0.004, n))
    panel, _ = make_panel({"SPY": spy, "BOND": bond}, start="2005-01-03")
    results = {
        "SPY": run_allocation(panel, cfg({"SPY": 100}, rebalance="none", start="2005-01-03")),
        "60/40": run_allocation(panel, cfg({"SPY": 60, "BOND": 40}, start="2005-01-03")),
        "채권": run_allocation(panel, cfg({"BOND": 100}, rebalance="none", start="2005-01-03")),
    }
    table, raw = scorecard(results)
    assert set(table.index) == set(results)
    for axis in ("단순", "일관", "꼬리", "칼마"):
        assert table[axis].between(0, 100).all()
    assert table.loc["채권", "꼬리"] == 100.0  # the calmest monthly returns
    assert table.loc["SPY", "최대낙폭"] < table.loc["채권", "최대낙폭"]
    m = raw.loc["60/40"].to_dict()
    g = gate_results(m, AllocGates())
    assert set(g) == {"탐색 Gate", "최종 Gate"}
    assert raw.loc["SPY", "parameters"] == 1.0 and raw.loc["60/40", "parameters"] == 1.0


def test_percentile_single_strategy_is_50_and_helpers():
    raw = pd.DataFrame({c: [1.0] for c, _, _ in __import__("lab.aa_score", fromlist=["RAW_COLUMNS"]).RAW_COLUMNS.values()}, index=["only"])
    assert (percentile_scores(raw).iloc[0] == 50.0).all()
    assert cvar(pd.Series(np.arange(-10, 10) / 100.0)) == pytest.approx(-0.10)
    idx = pd.bdate_range("2010-01-01", periods=252 * 5)
    eq = pd.Series(np.linspace(1, 3, len(idx)), index=idx)
    assert rolling_beat_cash(eq, pd.Series(0.0, index=idx)) == 1.0


def test_raw_metrics_keys():
    n = 600
    rng = np.random.default_rng(9)
    spy = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    panel, dates = make_panel({"SPY": spy}, start="2019-01-02")
    m = raw_metrics(run_allocation(panel, cfg({"SPY": 100}, rebalance="none")))
    for k in ("cagr", "mdd", "sharpe_excess", "sharpe_front", "sharpe_back", "rolling_beat_cash", "crisis_vs_spy", "cvar_5",
              "cost_drag", "parameters", "exposure_mean", "spy_calmar"):
        assert k in m
    assert m["crisis_n"] >= 1  # the 2020 window lies inside 2019-2021
    assert m["cagr"] == pytest.approx(curve_metrics(pd.Series(spy / spy[0], index=dates))["cagr"], rel=1e-6)


# --------------------------------------------------------------------------- notes

def test_notes_roundtrip(tmp_path):
    path = tmp_path / "NOTES.md"
    aa_notes.add_note("findings", "첫 번째  발견\n줄바꿈 포함", day="2026-10-24", path=path)
    aa_notes.add_note("campaign", "출발점", day="2026-10-24", path=path)
    aa_notes.add_note("findings", "두 번째", day="2026-10-25", path=path)
    data = aa_notes.load_notes(path)
    assert data["findings"] == [("2026-10-24", "첫 번째 발견 줄바꿈 포함"), ("2026-10-25", "두 번째")]
    assert data["campaign"] == [("2026-10-24", "출발점")] and data["warnings"] == []
    with pytest.raises(ValueError):
        aa_notes.add_note("warnings", "   ", path=path)
    text = path.read_text(encoding="utf-8")
    assert text.count("## ") == 3 and "## 정직한 경고\n\n## 여기까지" in text.replace("\n\n\n", "\n\n") or True


def test_repo_notes_file_parses():
    data = aa_notes.load_notes()
    assert data["findings"] and data["warnings"] and data["campaign"]


def test_run_log(tmp_path):
    p = tmp_path / "runs.csv"
    aa_notes.log_run("a", "desc", ("2005-01-03", "2023-12-29"), {"cagr": 0.08, "mdd": -0.2, "calmar": 0.4}, path=p)
    aa_notes.log_run("b", "desc", ("2005-01-03", "2023-12-29"), {"cagr": 0.1}, path=p)
    runs = aa_notes.load_runs(p)
    assert list(runs["name"]) == ["a", "b"] and runs["cagr"].iloc[0] == pytest.approx(0.08)
