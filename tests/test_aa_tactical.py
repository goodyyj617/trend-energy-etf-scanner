"""Allocation lab: Keller tactical rules, validation tools, the reset final gate and the risk-off switch."""
import numpy as np
import pandas as pd
import pytest

from lab import aa_signals as sig
from lab import aa_tactical as tac
from lab import aa_validate as val
from lab.aa_engine import AllocConfig, MixConfig, decision_rows, run_allocation, run_mix, with_rebalance_offset
from lab.aa_score import AllocGates, gate_results, gate_text, scorecard, timing_ok
from lab.data import panel_from_long


def make_panel(closes: dict[str, np.ndarray], start="2005-01-03", cash_rate=0.0):
    dates = pd.bdate_range(start, periods=len(next(iter(closes.values()))))
    rows = [{"date": d, "symbol": s, "open": v, "high": v, "low": v, "close": v, "volume": 1}
            for s, c in closes.items() for d, v in zip(dates, c)]
    panel = panel_from_long(pd.DataFrame(rows))
    panel.cash = pd.Series(cash_rate / 252, index=panel.close.index)
    return panel


def trend(n, daily):
    return 100 * np.cumprod(np.full(n, 1.0 + daily))


def monthly(rows: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(rows.values())))
    return pd.DataFrame(rows, index=pd.period_range("2010-01", periods=n, freq="M").to_timestamp("M"))


# --------------------------------------------------------------------------- rules

def _rising(rate, n=13):
    return list(100 * (1 + rate) ** np.arange(n))


def test_haa_canary_off_goes_to_better_of_ief_and_bil():
    up = _rising(0.01)
    m = monthly({**{s: up for s in tac.SIGNAL_SYMBOLS["haa"]}, "TIP": _rising(-0.01), "IEF": _rising(0.002), "BIL": _rising(0.001)})
    assert tac.rule_weights("haa", m, 12) == {"IEF": 1.0}


def test_haa_canary_on_holds_top4_and_replaces_negative_ones():
    m = monthly({"SPY": _rising(0.03), "IWM": _rising(0.02), "VEA": _rising(-0.01), "VWO": _rising(-0.02), "VNQ": _rising(-0.03),
                 "DBC": _rising(-0.04), "IEF": _rising(0.001), "TLT": _rising(-0.005), "TIP": _rising(0.005), "BIL": _rising(0.002)})
    w = tac.rule_weights("haa", m, 12)
    # top 4 by 13612U: SPY, IWM, IEF (+), TLT (-); TLT's quarter goes to the better safe asset, BIL (0.2%/m > IEF 0.1%/m)
    assert w == pytest.approx({"SPY": 0.25, "IWM": 0.25, "IEF": 0.25, "BIL": 0.25})


def test_baa_offensive_picks_single_best_and_defensive_uses_bil_floor():
    base = {s: _rising(0.01) for s in tac.SIGNAL_SYMBOLS["baa_g4"]}
    m = monthly({**base, "QQQ": _rising(0.03)})
    assert tac.rule_weights("baa_g4", m, 12) == {"QQQ": 1.0}
    bad = monthly({**base, "VWO": _rising(-0.02), "TIP": _rising(0.004), "IEF": _rising(0.003), "BIL": _rising(0.002),
                   "DBC": _rising(-0.01), "TLT": _rising(-0.01), "LQD": _rising(0.0), "BND": _rising(0.001)})
    w = tac.rule_weights("baa_g4", bad, 12)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w == pytest.approx({"TIP": 1 / 3, "IEF": 1 / 3, "BIL": 1 / 3})


def test_rule_needs_twelve_months():
    m = monthly({s: _rising(0.01, 12) for s in tac.SIGNAL_SYMBOLS["vaa_g4"]})
    assert tac.rule_weights("vaa_g4", m, 11) is None


def test_splice_extends_late_etf_with_sibling_returns():
    idx = pd.bdate_range("2010-01-04", periods=6)
    close = pd.DataFrame({"EFA": [10, 11, 12, 13, 14, 15.0], "VEA": [np.nan, np.nan, np.nan, 26, 28, 30.0]}, index=idx)
    out = tac.spliced_close(close, None)
    assert out["VEA"].iloc[3:].tolist() == [26, 28, 30]
    assert out["VEA"].iloc[2] == pytest.approx(12 * 26 / 13)


def test_krw_mapping_sends_missing_products_to_the_chosen_substitute():
    w = tac.map_weights({"DBC": 0.5, "SPY": 0.25, "BIL": 0.25}, "krw", ["379800", "459580", "411060"], "411060")
    assert w == pytest.approx({"411060": 0.5, "379800": 0.25, "459580": 0.25})
    cov = tac.krw_coverage(pd.DataFrame([{"DBC": 0.5, "SPY": 0.25, "IEF": 0.25}]))
    assert cov.iloc[0].to_dict() == pytest.approx({"정확": 0.25, "근사": 0.25, "없음": 0.5})


def _tactical_panel(n=600):
    rng = np.random.default_rng(3)
    syms = sorted(set(tac.ALL_SYMBOLS))
    closes = {s: 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, n)) for s in syms}
    return make_panel(closes, cash_rate=0.01)


def test_run_tactical_trades_next_close_unless_same_close():
    panel = _tactical_panel()
    spec = tac.TacticalSpec("t", parts=(("vaa_g4", 100.0),), start="2005-01-01", end="2030-01-01", cost_bps=0.0)
    nxt = tac.run_tactical(panel, panel.close, spec, "usd", panel.cash)
    same = tac.run_tactical(panel, panel.close, tac.TacticalSpec(**{**spec.__dict__, "same_close": True}), "usd", panel.cash)
    assert len(nxt.signals) > 10
    held = nxt.weights.drop(columns="CASH")
    # each day the portfolio is (almost) one asset, as VAA holds 100% of a single ETF
    assert (held.max(axis=1).iloc[30:] > 0.99).mean() > 0.9
    assert not np.allclose(nxt.equity.to_numpy(), same.equity.to_numpy())


def test_blend_equals_target_average():
    panel = _tactical_panel()
    sig_close = tac.spliced_close(panel.close, panel.cash)
    a = tac.signal_table(sig_close, tac.TacticalSpec("a", parts=(("haa", 100.0),)))
    b = tac.signal_table(sig_close, tac.TacticalSpec("b", parts=(("baa_g4", 100.0),)))
    ab = tac.signal_table(sig_close, tac.TacticalSpec("ab", parts=(("haa", 50.0), ("baa_g4", 50.0))))
    want = (a.add(b, fill_value=0.0) / 2).fillna(0.0)
    pd.testing.assert_frame_equal(ab.reindex(columns=want.columns, fill_value=0.0), want, check_names=False)
    assert np.allclose(ab.sum(axis=1), 1.0)


def test_delay_and_skip_months_change_the_path():
    panel = _tactical_panel()
    spec = tac.TacticalSpec("t", parts=(("haa", 100.0),), start="2005-01-01", cost_bps=0.0)
    base = tac.run_tactical(panel, panel.close, spec, "usd", panel.cash)
    late = tac.run_tactical(panel, panel.close, tac.TacticalSpec(**{**spec.__dict__, "delay": 1}), "usd", panel.cash)
    months = tuple(str(d.to_period("M")) for d in base.signals.index[1::2])
    lazy = tac.run_tactical(panel, panel.close, tac.TacticalSpec(**{**spec.__dict__, "skip_months": months}), "usd", panel.cash)
    assert late.n_trades <= base.n_trades and lazy.n_trades < base.n_trades


# --------------------------------------------------------------------------- engine offset and gates

def test_rebalance_offset_moves_decisions_and_keeps_old_fingerprints():
    n = 130
    panel = make_panel({"A": trend(n, 0.002), "B": trend(n, -0.001), "SPY": trend(n, 0.001)})
    c0 = AllocConfig("x", {"A": 50, "B": 50}, "2005-01-01", "2030-01-01", rebalance="M", cost_bps=0.0)
    c5 = AllocConfig("x", {"A": 50, "B": 50}, "2005-01-01", "2030-01-01", rebalance="M", cost_bps=0.0, rebalance_offset=5)
    assert "rebalance_offset" not in c0.to_dict() and c5.to_dict()["rebalance_offset"] == 5
    r0, r5 = run_allocation(panel, c0), run_allocation(panel, c5)
    rows = decision_rows(r0.equity.index, "M")
    assert r0.weights["A"].iloc[rows[0] + 1] == pytest.approx(0.5)
    assert r5.weights["A"].iloc[rows[0] + 1] != pytest.approx(0.5)
    assert r5.weights["A"].iloc[rows[0] + 6] == pytest.approx(0.5)


def test_final_gate_uses_absolute_mdd_lower_cagr_ratio_and_timing():
    m = {"cagr": 0.085, "mdd": -0.14, "spy_cagr": 0.12, "spy_mdd": -0.30, "calmar": 0.6, "spy_calmar": 0.4,
         "timing_mdd_worst": -0.18}
    assert gate_results(m)["최종 Gate"]  # 8.5% >= 0.7 x 12%, -14% within -15%, worst over offsets -18% within -20%
    assert not gate_results({**m, "mdd": -0.16})["최종 Gate"]
    assert not gate_results({**m, "cagr": 0.08})["최종 Gate"]
    assert not gate_results({**m, "timing_mdd_worst": -0.25})["최종 Gate"]  # timing-fragile (A30)
    no_timing = {k: v for k, v in m.items() if k != "timing_mdd_worst"}
    assert timing_ok(no_timing) is None and not gate_results(no_timing)["최종 Gate"]  # not computed = not passed
    assert not gate_results({**m, "timing_mdd_worst": float("nan")})["최종 Gate"]
    assert gate_results(no_timing, AllocGates(max_timing_mdd_abs=None))["최종 Gate"]  # A26 alone
    assert gate_results({**m, "timing_mdd_worst": -0.25})["탐색 Gate"]  # the exploration gate ignores timing
    assert "날짜 견고" in gate_text() and "날짜 견고" not in gate_text(AllocGates(max_timing_mdd_abs=None))
    old = AllocGates(min_cagr_ratio=0.8, max_mdd_abs=None, max_timing_mdd_abs=None)
    assert not gate_results(m, old)["최종 Gate"]  # 8.5% < 9.6%
    assert gate_results({**m, "cagr": 0.10, "mdd": -0.20}, old)["최종 Gate"]  # old rule: |MDD| <= 22.5%


def test_timing_worst_mdd_takes_the_worst_offset_on_the_period():
    idx = pd.bdate_range("2010-01-04", periods=200)
    up = pd.Series(1.0, index=idx)
    dip = up.copy()
    dip.iloc[20:30] *= 0.7  # a -30% hole early on, only at offset 10
    base = val.AllocResult.__new__(val.AllocResult)
    base.equity = up
    calls = []

    def run(o):
        calls.append(o)
        if o == 15:
            raise ValueError("too short")
        return dip if o == 10 else up

    assert val.timing_worst_mdd(run, base) == pytest.approx(-0.3)
    assert 0 not in calls  # the base result is reused for offset 0
    assert val.timing_worst_mdd(run, base, period=(idx[40], idx[-1])) == pytest.approx(0.0)  # the hole is outside


def test_mix_offset_shifts_sleeves_and_reweighting_and_feeds_the_scorecard():
    n = 260
    panel = make_panel({"A": trend(n, 0.002), "B": trend(n, -0.001), "SPY": trend(n, 0.001)})
    a = AllocConfig("a", {"A": 100}, "2005-01-01", "2030-01-01", rebalance="none", cost_bps=0.0)
    ab = AllocConfig("ab", {"A": 50, "B": 50}, "2005-01-01", "2030-01-01", rebalance="M", cost_bps=0.0)
    mix = MixConfig("m", ((a, 50.0), (ab, 50.0)), "M")
    m5 = with_rebalance_offset(mix, 5)
    assert m5.rebalance_offset == 5 and all(c.rebalance_offset == 5 for c, _ in m5.sleeves)
    assert "rebalance_offset" not in mix.to_dict() and MixConfig.from_dict(m5.to_dict()) == m5
    assert with_rebalance_offset(ab, 3) == AllocConfig("ab", {"A": 50, "B": 50}, "2005-01-01", "2030-01-01",
                                                       rebalance="M", cost_bps=0.0, rebalance_offset=3)
    r0, r5 = run_mix(panel, mix), run_mix(panel, m5)
    rows = decision_rows(r0.equity.index, "M")
    assert r0.weights["a"].iloc[rows[0] + 1] == pytest.approx(0.5)
    assert r5.weights["a"].iloc[rows[0] + 1] != pytest.approx(0.5)
    assert r5.weights["a"].iloc[rows[0] + 6] == pytest.approx(0.5)
    results = {"a": run_allocation(panel, a), "ab": run_allocation(panel, ab)}
    table, raw = scorecard(results, timing={"a": -0.01})
    assert raw.loc["a", "timing_mdd_worst"] == pytest.approx(-0.01) and np.isnan(raw.loc["ab", "timing_mdd_worst"])
    assert not table.loc["ab", "최종 Gate"]  # no timing number, no final pass


# --------------------------------------------------------------------------- validation tools

def test_weight_grid_counts():
    assert len(val.weight_grid(3, 0.5, 2)) == 3 + 3  # singles + pairs at 50/50
    assert all(np.isclose(w.sum(), 1.0) for w in val.weight_grid(4, 0.1, 3))


def test_cscv_runs_252_splits_and_finds_a_real_edge():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2010-01-04", periods=1500)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (1500, 4)), index=idx, columns=list("ABCD"))
    rets["A"] += 0.002  # one sleeve with a large true edge: the tuned blend should keep picking it
    out = val.cscv(rets, pd.Series(0.0, index=idx), step=0.25, max_k=2)
    assert out["splits"] == 252 and out["candidates"] == len(val.weight_grid(4, 0.25, 2))
    assert out["pbo"] < 0.2
    assert all("A" in pick for pick in out["top_picks"].index)


def test_lazy_skip_and_timing_tables():
    rng = np.random.default_rng(0)
    months = [f"2020-{m:02d}" for m in range(1, 13)]
    skip = val.lazy_skip_months(months, rng)
    assert set(skip) <= set(months) and months[0] not in skip
    idx = pd.bdate_range("2010-01-04", periods=300)
    eq = pd.Series(np.linspace(1, 1.3, 300), index=idx)

    def run(o, d):
        r = val.AllocResult.__new__(val.AllocResult)
        r.equity, r.cash = eq * (1 - 0.001 * o), pd.Series(0.0, index=idx)
        return r

    t = val.timing_table(run)
    assert len(t) == len(val.TIMING_OFFSETS) + 1
    s = val.timing_summary(t)
    assert s["mdd_worst"] <= s["mdd_best"] <= 0


def test_monthly_table_compounds_to_the_year():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    eq = pd.Series(np.cumprod(np.full(len(idx), 1.0003)), index=idx)
    t = val.monthly_table(eq)
    assert list(t.index) == [2020, 2021]
    assert (1 + t.loc[2021, list(range(1, 13))]).prod() - 1 == pytest.approx(t.loc[2021, "연간"])


def test_block_bootstrap_brackets_the_sample():
    panel = make_panel({"SPY": 100 * np.cumprod(1 + np.random.default_rng(2).normal(0.0004, 0.01, 800))})
    res = run_allocation(panel, AllocConfig("s", {"SPY": 100}, "2005-01-01", "2030-01-01", rebalance="none", cost_bps=0.0))
    bb = val.block_bootstrap(res, n=200)
    assert bb.loc["5%", "샤프"] < bb.loc["95%", "샤프"]
    assert (bb["MDD"] <= 0).all()


# --------------------------------------------------------------------------- risk-off switch

def test_switch_never_on_matches_plain_run_and_always_on_holds_safe():
    n = 300
    panel = make_panel({"379800": trend(n, 0.001), "459580": trend(n, 0.0001), "SPY": trend(n, 0.001)})
    panel.benchmark = "379800"
    cfg = AllocConfig("x", {"379800": 60, "459580": 40}, "2005-01-01", "2030-01-01", rebalance="Q", cost_bps=0.0)
    days = panel.close.index[decision_rows(panel.close.index, "M")]
    plain = run_allocation(panel, cfg)
    never = sig.run_switch(panel, cfg, pd.Series(False, index=days), ["379800"], "459580")
    assert np.allclose(plain.equity.to_numpy(), never.equity.to_numpy())
    always = sig.run_switch(panel, cfg, pd.Series(True, index=days), ["379800"], "459580")
    assert always.weights["379800"].iloc[decision_rows(panel.close.index, "M")[0] + 1:].max() < 1e-9


def test_drawdown_episodes_finds_the_deepest():
    idx = pd.bdate_range("2010-01-04", periods=9)
    eq = pd.Series([1.0, 0.9, 1.0, 1.1, 0.8, 0.9, 1.2, 1.15, 1.2], index=idx)
    ep = sig.drawdown_episodes(eq, top=2)
    assert ep["깊이"].iloc[0] == pytest.approx(0.8 / 1.1 - 1)
    assert ep["바닥"].iloc[0] == idx[4]
