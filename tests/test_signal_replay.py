"""lab/signal_replay.py: the slot-limited account rebuilt from a per-signal table equals the engine's."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lab import signal_replay as sr
from lab.engine import SignalCache, StrategyConfig, run_backtest
from tests.test_stage_c import _study_panel


def _cfg(dates, entries, exits, k, **kw):
    return StrategyConfig(entries=entries, exits=exits, start=str(dates[200].date()), end=str(dates[-1].date()),
                          max_positions=k, cost_bps=20.0, min_price=0.0, min_dollar_volume=0.0, sizing="equal", **kw)


@pytest.mark.parametrize("entries,exits,be,k", [
    ({"c_donchian": 20}, {"trailing_pct": 15}, None, 2),
    ({"c_donchian": 10}, {"atr_stop": 2.0, "time_stop": 30}, 2.0, 3),
    ({"c_tsmom": 1}, {"chandelier": 2.0}, None, 2),
    ({"c_golden": 20}, {"low_break": 10}, None, 1),
])
def test_replay_reproduces_engine_account(entries, exits, be, k):
    p, dates = _study_panel()
    cfg = _cfg(dates, entries, exits, k, breakeven_r=be)
    cache = SignalCache(p)
    res = run_backtest(p, cfg, cache)
    sig = run_backtest(p, cfg, cache, outcomes=True)
    assert len(res.trades) >= 5 and len(sig) > len(res.trades)  # the slots were binding
    eq, trades = sr.replay(p, cfg, sig, cache=cache)
    np.testing.assert_allclose(eq.to_numpy(), res.equity.to_numpy(), rtol=0, atol=1e-9)
    cols = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return", "holding_days"]
    pd.testing.assert_frame_equal(trades[cols].reset_index(drop=True), res.trades[cols].reset_index(drop=True))
    # an older table without exit_signal_date gives the same account here (every exit had a next open)
    eq2, _ = sr.replay(p, cfg, sig.drop(columns="exit_signal_date"), cache=cache)
    np.testing.assert_allclose(eq2.to_numpy(), res.equity.to_numpy(), rtol=0, atol=1e-9)


def test_replay_with_rank_priority_and_keep_mask_equals_screened_engine():
    p, dates = _study_panel()
    cfg = _cfg(dates, {"c_donchian": 20}, {"trailing_pct": 15}, 2)
    rng = np.random.default_rng(5)
    rank = pd.DataFrame(rng.random(p.close.shape), index=p.close.index, columns=p.close.columns).rank(axis=1)
    pr = replace(p, rank=rank)
    sig = run_backtest(pr, cfg, outcomes=True)
    res = run_backtest(pr, cfg)
    eq, trades = sr.replay(pr, cfg, sig)
    np.testing.assert_allclose(eq.to_numpy(), res.equity.to_numpy(), rtol=0, atol=1e-9)
    # a pre-buy screen as a keep-mask on the signals = the engine on the screened universe
    screened = p.eligible.copy()
    screened["A"] = False
    screened.loc[dates[400]:, "B"] = False
    keep = sr.signal_values(pr, sig, "x", frame=(~screened).astype(float)).to_numpy() < 0.5
    eq_s, _ = sr.replay(pr, cfg, sig, keep=keep)
    res_s = run_backtest(replace(pr, eligible=screened), cfg)
    np.testing.assert_allclose(eq_s.to_numpy(), res_s.equity.to_numpy(), rtol=0, atol=1e-9)
    assert not np.allclose(eq_s.to_numpy(), eq.to_numpy())


def test_daily_screen_option_defaults_to_none():
    from lab import stage_c_study as sc
    p, dates = _study_panel()
    base = sc.daily_eligible(p)
    pd.testing.assert_frame_equal(sc.daily_eligible(p, {}, {}, 0), base)
    assert sc.parse_screens("x_ivol=20, s_large") == {"x_ivol": 20.0, "s_large": None}
    with pytest.raises(ValueError):
        sc.parse_screens("nope=1")
    top = sc.daily_eligible(p, select={"s_large": None}, top_k=2)
    assert (top.sum(axis=1) <= 2).all() and top.loc[dates[-1]].sum() == 2
    cap = p.extra["cap"].loc[dates[-1]][base.loc[dates[-1]]]  # among the names that pass hygiene that day
    assert set(top.columns[top.loc[dates[-1]]]) == set(cap.nlargest(2).index)  # the two largest of them
    excl = sc.daily_eligible(p, exclude={"h_price": 1e9})
    assert not excl.any().any()


def test_signal_values_read_the_signal_day_only():
    p, dates = _study_panel()
    sig = pd.DataFrame({"symbol": ["A", "B", "Z"], "signal_date": dates[[300, 301, 302]]})
    v = sr.signal_values(p, sig, "s_efficiency", 6)
    from lab import screens
    full = screens.ALL_SCREENS["s_efficiency"].compute(p, 6)
    assert v.iloc[0] == pytest.approx(full.loc[dates[300], "A"]) and v.iloc[1] == pytest.approx(full.loc[dates[301], "B"])
    assert np.isnan(v.iloc[2])  # unknown symbol
    # no lookahead: changing prices after the signal day leaves the value unchanged
    p2, _ = _study_panel()
    p2.close.loc[dates[301]:, "A"] *= 3.0
    assert sr.signal_values(p2, sig.iloc[:1], "s_efficiency", 6).iloc[0] == pytest.approx(v.iloc[0])
    h = sr.signal_values(p, sig, "h_price", 60.0)  # hygiene: 1.0 = would be excluded
    assert set(h.dropna().unique()) <= {0.0, 1.0}
    keep = sr.keep_mask(p, sig.assign(exit_reason=["", "cancelled", ""]), {"h_price": 1e9})
    assert not keep[0] and not keep[1] and keep[2]  # A excluded by price, B cancelled, Z has no value


def test_replay_log_classifies_every_signal():
    p, dates = _study_panel()
    cfg = _cfg(dates, {"c_donchian": 20}, {"trailing_pct": 15}, 2)
    sig = run_backtest(p, cfg, outcomes=True)
    log = {}
    eq, trades = sr.replay(p, cfg, sig, log=log)
    eq2, _ = sr.replay(p, cfg, sig)
    np.testing.assert_array_equal(eq.to_numpy(), eq2.to_numpy())  # logging changes nothing
    st = pd.Series(log["status"])
    assert set(st.unique()) <= {"held", "queued", "full"} and {"held", "full"} & set(st.unique())
    assert (st == "queued").sum() >= len(trades)  # every buy was queued (the last day's queue is never filled)
    free = np.asarray(log["free"])
    full_days = {i for (i, j), v in log["status"].items() if v == "full"}
    assert all(free[i] < sum(1 for (k, _), v in log["status"].items() if k == i and v != "held") for i in full_days)
