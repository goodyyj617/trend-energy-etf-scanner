"""Allocation lab A40: canary signals on B2–B5 (lab/aa_canary.py)."""
import numpy as np
import pandas as pd
import pytest

from lab import aa_canary as K
from lab import aa_composition as C
from lab.aa_assets import CASH
from lab.aa_engine import simulate
from lab.aa_search import StaticSim
from lab.data import Panel


def _panel(syms=("A", "B", "C"), start="2001-01-01", end="2008-12-31", seed=3):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates))) for s in syms}, index=dates)
    close.iloc[5:9, 1] = np.nan  # a few missing closes, carried forward like the engine does
    p = Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1, tradable=list(syms))
    p.cash = pd.Series(0.02 / 252, index=dates)
    return p


def _inputs(start="2000-01-03", end="2008-12-31", seed=11):
    """Made-up canary closes, FRED series, USD/KRW and a KRW panel holding every product of B2–B5."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    cols = sorted({"SPY", "VWO", "BND", "EFA", "EEM", "AGG", "TIP", *K.SECTORS})
    usd = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, len(dates))) for s in cols}, index=dates)
    fred = pd.DataFrame({"BAA10Y": 2 + np.cumsum(rng.normal(0, 0.02, len(dates))),
                         "T10Y3M": np.cumsum(rng.normal(0, 0.03, len(dates))),
                         "VIXCLS": 20 + 8 * np.sin(np.arange(len(dates)) / 40)}, index=dates)
    fx = pd.Series(1100 * np.cumprod(1 + rng.normal(0, 0.005, len(dates))), index=dates)
    held = sorted({s for w in C.CENTRES.values() for s in w} | set(K.DEFENSIVE))
    close = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(dates))) for s in held}, index=dates)
    krw = Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1, tradable=held)
    krw.cash = pd.Series(0.02 / 252, index=dates)
    return K.Inputs(usd=usd, usd_cash=pd.Series(0.03 / 252, index=dates), fx=fx, krw=krw, macro_df=fred)


def _cut(inp: K.Inputs, end) -> K.Inputs:
    c = inp.krw.close.loc[:end]
    krw = Panel(open=c, high=c, low=c, close=c, volume=c * 0 + 1, tradable=list(c.columns))
    krw.cash = inp.krw.cash.loc[:end]
    return K.Inputs(inp.usd.loc[:end], inp.usd_cash.loc[:end], inp.fx.loc[:end], krw, inp.macro_df.loc[:end], inp.currency)


# --------------------------------------------------------------------------- the simulator

def test_dynamic_sim_is_the_engine_for_any_decision_path():
    panel = _panel()
    syms = ["A", "B", "C"]
    sim = K.DynamicSim(panel, syms, "2002-01-01", "2008-12-31", cost_bps=10)
    rng = np.random.default_rng(5)
    P, T = 4, sim.n_days
    tabs = rng.dirichlet(np.ones(4), size=(P, 3))  # 3 states × (A, B, C, cash)
    st = np.vstack([np.repeat(rng.integers(0, 3, T // 15 + 1), 15)[:T] for _ in range(P)])
    st[3] = 0  # never changes: only the quarter ends trade
    scheds = [K.schedule(s, K.quarter_rows(sim.dates), cooldown=21 * (p % 2)) for p, s in enumerate(st)]
    paths, turn, trades = sim.run(tabs, np.vstack([s.decide for s in scheds]), np.vstack([s.choice for s in scheds]),
                                  np.array([s.first for s in scheds]))
    close = panel.close.loc[:sim.dates[-1], syms].ffill()
    rets = close.pct_change(fill_method=None).loc[sim.dates]
    cols = syms + [CASH]
    for p, s in enumerate(scheds):
        targets = {int(t): dict(zip(cols, tabs[p, s.choice[t]])) for t in np.flatnonzero(s.decide)}
        eq, _, t_eng, n_eng = simulate(rets, sim.cash, targets, 10, 0.0, first=dict(zip(cols, tabs[p, s.first])))
        assert np.allclose(paths[p], eq.to_numpy(), rtol=1e-12, atol=0)
        assert turn[p] == pytest.approx(t_eng, rel=1e-12) and trades[p] == n_eng
    assert trades[3] == len(K.quarter_rows(sim.dates))


@pytest.mark.parametrize("offset", [0, 5])
def test_dynamic_sim_with_static_rows_is_the_static_simulator(offset):
    panel = _panel()
    syms = ["A", "B", "C"]
    W = np.random.default_rng(1).dirichlet(np.ones(4), size=6)
    sim = K.DynamicSim(panel, syms, "2002-01-01", "2008-12-31")
    paths, turn, _ = sim.run(*K.static_rows(W, sim, offset))
    ref_paths, ref_turn = StaticSim(panel, syms, "2002-01-01", "2008-12-31", rebalance_offset=offset).run(W)
    assert np.allclose(paths, ref_paths, rtol=1e-9) and np.allclose(turn, ref_turn, rtol=1e-9)


def test_dynamic_sim_refuses_late_data_and_bad_shapes():
    panel = _panel()
    panel.close.loc[:"2003-01-01", "C"] = np.nan
    with pytest.raises(ValueError):
        K.DynamicSim(panel, ["A", "C"], "2002-01-01", "2008-12-31")
    sim = K.DynamicSim(panel, ["A", "B"], "2002-01-01", "2008-12-31")
    with pytest.raises(ValueError):
        sim.run(np.ones((1, 1, 3)) / 3, np.zeros((1, 5), bool), np.zeros((1, 5), int), np.zeros(1, int))


# --------------------------------------------------------------------------- when to trade

def test_schedule_trades_the_day_after_a_change_and_the_cooldown_waits():
    st = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    s = K.schedule(st, quarter_rows=np.array([9]), cooldown=0)
    assert s.signal_rows.tolist() == [2, 3, 6] and s.first == 0
    assert np.flatnonzero(s.decide).tolist() == [2, 3, 6, 9]  # row 11 (the last) never decides
    assert s.choice.tolist() == [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert s.whipsaws(within=2) == 1 and s.whipsaws(within=21) == 2
    c = K.schedule(st, quarter_rows=np.array([9]), cooldown=3)
    # row 3 is blocked (one row after the trade at 2); at row 5 the block ends and the signal still says 0;
    # the change at row 6 waits for row 8
    assert c.signal_rows.tolist() == [2, 5, 8]
    assert c.choice[2:5].tolist() == [1, 1, 1] and c.choice[5:8].tolist() == [0, 0, 0] and c.choice[8] == 1
    held = K.schedule(np.zeros(12, int), quarter_rows=np.array([3, 7]), cooldown=0)
    assert held.n_signal == 0 and np.flatnonzero(held.decide).tolist() == [3, 7] and held.whipsaws() == 0


def test_the_fast_schedule_is_the_loop():
    """With no cooldown every change is followed; a cooldown of one row is the same rule through the loop."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        st = np.repeat(rng.integers(0, 3, 60), rng.integers(1, 6, 60))[:200]
        q = np.sort(rng.choice(len(st), 6, replace=False))
        a, b = K.schedule(st, q, cooldown=0), K.schedule(st, q, cooldown=1)
        assert (a.decide == b.decide).all() and (a.choice == b.choice).all() and (a.signal_rows == b.signal_rows).all()


def test_a_signal_that_never_warns_is_the_static_allocation():
    panel = _panel()
    sim = K.DynamicSim(panel, ["A", "B", "C"], "2002-01-01", "2008-12-31")
    base = np.array([0.5, 0.2, 0.2, 0.1])
    tab = np.vstack([base, [0.1, 0.4, 0.4, 0.1]])  # state 1 would move A away, but it never comes
    s = K.schedule(np.zeros(sim.n_days, int), K.quarter_rows(sim.dates), cooldown=21)
    paths, turn, _ = sim.run(tab[None], s.decide[None], s.choice[None], np.array([s.first]))
    ref, ref_turn = StaticSim(panel, ["A", "B", "C"], "2002-01-01", "2008-12-31").run(base[None])
    assert np.allclose(paths, ref, rtol=1e-12) and np.allclose(turn, ref_turn, rtol=1e-12)


def test_placebo_keeps_the_spells_and_moves_them():
    st = np.zeros(2000, int)
    st[300:420] = 1
    st[1500:1530] = 1
    P, shifts = K.placebo_states(st, 50, seed=0)
    assert P.shape == (50, 2000) and (P.sum(axis=1) == st.sum()).all()
    assert ((shifts >= 252) & (shifts <= 2000 - 252)).all()
    assert (P[0] == np.roll(st, -int(shifts[0]))).all()
    with pytest.raises(ValueError):
        K.placebo_states(st[:400], 5, seed=0)


# --------------------------------------------------------------------------- targets per state

COLUMNS = sorted({s for w in C.CENTRES.values() for s in w} | set(K.DEFENSIVE)) + [CASH]


def _w(row):
    return {c: round(float(x) * 100, 9) for c, x in zip(COLUMNS, row) if abs(x) > 1e-12}


@pytest.mark.parametrize("name", C.NAMES)
@pytest.mark.parametrize("rule, action", [("k4", "defensive"), ("k4", "safe"), ("k1", "defensive"), ("k1", "safe"),
                                          ("d2", "tilt_nd"), ("d3", "tilt_es")])
def test_tables_are_full_allocations_and_row_0_is_the_base(name, rule, action):
    base = C.CENTRES[name]
    tab = K.tables(base, rule, action, COLUMNS)
    assert tab.shape == ((3 if rule in ("k1", "d2", "d3") else 2), len(COLUMNS))
    assert np.allclose(tab.sum(axis=1), 1.0) and (tab >= -1e-12).all()
    assert _w(tab[0]) == {k: float(v) for k, v in base.items()}
    assert tab[:, COLUMNS.index(CASH)].max() == 0  # no product ever moves to idle cash


def test_tables_move_nasdaq_where_the_design_says():
    b2, b3 = C.CENTRES["B2"], C.CENTRES["B3"]
    d = _w(K.tables(b2, "k4", "defensive", COLUMNS)[1])  # 19 nasdaq to staples : health = 19 : 6
    assert C.NASDAQ not in d and d["453630"] == pytest.approx(19 + 19 * 19 / 25) and d["453640"] == pytest.approx(6 + 19 * 6 / 25)
    d = _w(K.tables(b3, "k1", "defensive", COLUMNS)[1])  # B3 holds neither: half of nasdaq, half each
    assert d[C.NASDAQ] == pytest.approx(16.5) and d["453630"] == d["453640"] == pytest.approx(8.25)
    s = _w(K.tables(b3, "k4", "safe", COLUMNS)[1])  # 33 nasdaq to the safe sleeve (everything but equities)
    safe = {k: v for k, v in b3.items() if k not in (C.NASDAQ, "251350")}
    assert s["251350"] == 12 and C.NASDAQ not in s
    for k, v in safe.items():
        assert s[k] == pytest.approx(v + 33 * v / sum(safe.values()))
    t = K.tables(b2, "d2", "tilt_nd", COLUMNS)
    assert _w(t[1])[C.NASDAQ] == pytest.approx(9) and _w(t[1])["453630"] == pytest.approx(19 + 10 * 19 / 25)
    assert _w(t[2])[C.NASDAQ] == pytest.approx(29) and _w(t[2])["453640"] == pytest.approx(6 - 10 * 6 / 25)
    assert _w(K.tables(b3, "d2", "tilt_nd", COLUMNS)[2]) == {k: float(v) for k, v in b3.items()}  # no D to take back
    e = K.tables(C.CENTRES["B4"], "d3", "tilt_es", COLUMNS)  # equities 41 (30 + 11), safe 59
    assert _w(e[1])[C.NASDAQ] == pytest.approx(30 - 10 * 30 / 41) and _w(e[1])[C.CD] == pytest.approx(36 + 10 * 36 / 59)
    assert _w(e[2])["453630"] == pytest.approx(11 + 10 * 11 / 41)
    with pytest.raises(KeyError):
        K.tables(b2, "k4", "x", COLUMNS)


def test_sleeves_follow_the_user_split():
    sl = K.sleeves(C.CENTRES["B3"])
    assert sl["N"] == {C.NASDAQ: 33} and sl["D"] == {"453630": 1.0, "453640": 1.0} and sl["O"] == {"251350": 12}
    assert sum(sl["S"].values()) == 100 - 33 - 12
    assert K.sleeves(C.CENTRES["B5"])["O"] == {"069500": 4}


def test_equal_cost_static_finds_the_matching_move():
    base = C.CENTRES["B4"]
    i = COLUMNS.index(C.NASDAQ)

    def cagr_of(W):  # 10 % a year, one point less for each 10 %p of nasdaq moved
        return 0.10 - (0.30 - W[:, i]) * 0.1

    x, w = K.equal_cost_static(base, "k4", "safe", COLUMNS, 0.085, cagr_of)
    assert x == pytest.approx(0.5) and w[i] == pytest.approx(0.15)
    assert K.equal_cost_static(base, "k4", "safe", COLUMNS, 0.12, cagr_of)[0] == 0.0  # beats the base: no move
    assert K.equal_cost_static(base, "k4", "safe", COLUMNS, 0.01, cagr_of)[0] == 1.0  # worse than the full move


# --------------------------------------------------------------------------- signals

def test_momentum_and_turbulence_match_hand_numbers():
    px = pd.DataFrame({"x": np.arange(1.0, 401.0)})
    m = K.momentum(px, "13612W")["x"]
    r = {k: 400 / (400 - K.MONTH[k]) - 1 for k in (1, 3, 6, 12)}
    assert m.iloc[-1] == pytest.approx((12 * r[1] + 4 * r[3] + 2 * r[6] + r[12]) / 4)
    assert K.momentum(px, "13612U")["x"].iloc[-1] == pytest.approx(sum(r.values()) / 4)
    assert K.momentum(px, "13612W")["x"].iloc[:252].isna().all()
    with pytest.raises(ValueError):
        K.momentum(px, "x")
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (900, 3)))
    tb = K.turbulence(rets, window=756, smooth=1)
    X, x = rets.iloc[100:856].to_numpy(), rets.iloc[856].to_numpy() - rets.iloc[100:856].to_numpy().mean(axis=0)
    assert tb.iloc[856] == pytest.approx(float(x @ np.linalg.inv(np.cov(X, rowvar=False)) @ x))
    assert tb.iloc[:756].isna().all()


@pytest.mark.parametrize("key", list(K.RULES))
def test_every_rule_reads_only_the_past(key):
    """States up to a date do not change when the data after it is cut off."""
    inp = _inputs()
    dates = inp.krw.close.index[inp.krw.close.index >= "2003-01-01"]
    base = C.CENTRES["B2"]
    for param in K.RULES[key].params:
        full = K.states(key, inp, param, dates, base)
        assert full.index.equals(dates) and set(full.unique()) <= ({0, 1, 2} if key in ("k1", "d2", "d3") else {0, 1})
        cut = "2006-06-30"
        part = K.states(key, _cut(inp, cut), param, dates[dates <= cut], base)
        assert (part == full.loc[:cut]).all(), (key, param)
    assert K.states(key, inp, K.RULES[key].params[0], dates, base).diff().abs().sum() > 0, "never moves"


def test_the_look_ahead_check_catches_a_signal_moved_one_day_early():
    inp = _inputs()
    dates = inp.krw.close.index[inp.krw.close.index >= "2003-01-01"]
    cut = "2006-06-30"
    leak = lambda i, d: K.states("k4", i, 200, d).shift(-1).fillna(0)  # tomorrow's state today
    full, part = leak(inp, dates), leak(_cut(inp, cut), dates[dates <= cut])
    assert not (part == full.loc[:cut]).all()


def test_signal_start_is_where_the_inputs_begin():
    inp = _inputs()
    dates = inp.krw.close.index
    inp.usd.loc[:"2001-12-31", "TIP"] = np.nan  # TIP data from 2002: the 12-month momentum 252 rows later
    raw = K.raw_states("k3", inp, "13612U", dates)
    start = K.signal_start(raw)
    first = inp.usd.index.get_loc(inp.usd["TIP"].first_valid_index())
    assert start == inp.usd.index[first + 252]
    assert raw.loc[:start].iloc[:-1].isna().all() and K.states("k3", inp, "13612U", dates).loc[:start].iloc[:-1].eq(0).all()
    assert K.signal_start(K.raw_states("k8", inp, 25.0, dates)) == dates[1]  # FRED values one day late


def test_k4_and_k11_on_a_built_path():
    inp = _inputs()
    d = inp.usd.index
    inp.usd["SPY"] = np.where(d < "2004-01-01", 100.0 + np.arange(len(d)) * 0.1, 50.0)  # a fall on 2004-01-01
    st = K.states("k4", inp, 200, d)
    assert st.loc[:"2003-12-31"].sum() == 0 and st.loc["2004-01-01":"2004-06-30"].eq(1).all()
    inp.fx = pd.Series(np.where(d < "2004-01-01", 1200.0, 1300.0), index=d)
    st = K.states("k11", inp, 200, d)
    assert st.loc["2004-01-01"] == 0 and st.loc["2004-01-02"] == 1  # FRED values are read one day later


def test_currency_check_converts_price_canaries_at_the_same_day_rate():
    inp = _inputs()
    krw = K.Inputs(inp.usd, inp.usd_cash, inp.fx, inp.krw, inp.macro_df, currency="krw")
    assert np.allclose(krw.closes(["SPY"])["SPY"], inp.usd["SPY"] * inp.fx)
    assert not (K.states("k4", krw, 200, inp.usd.index) == K.states("k4", inp, 200, inp.usd.index)).all()


def test_chained_carries_late_etfs_back_with_older_funds():
    d = pd.bdate_range("2000-01-03", periods=10)
    close = pd.DataFrame({"VWO": [np.nan] * 4 + [50.0, 51, 52, 53, 54, 55],
                          "VEIEX": [10.0, 11, 12, 13, 14, 15, 16, 17, 18, 19]}, index=d)
    out, table = K.chained(close)
    assert out["VWO"].iloc[4:].tolist() == close["VWO"].iloc[4:].tolist()
    assert out["VWO"].iloc[0] == pytest.approx(10 * 50 / 14)
    row = table.set_index("신호 자산").loc["VWO"]
    assert row["이어 붙인 순서"] == "VWO ← VEIEX" and row["시작"] == d[0]
    assert "BND" not in set(table["신호 자산"])  # nothing to chain
    assert set(K.CANARY_TICKERS) >= {"SPY", "VBMFX", "VIPSX", *K.SECTORS}


# --------------------------------------------------------------------------- the script on made-up prices

def test_canary_script_runs_on_made_up_prices(tmp_path, monkeypatch):
    import json
    import sys

    from lab import aa_search_results as sr

    sys.path.insert(0, str(sr.ROOT / "scripts"))
    import aa_canary_study as S
    from test_aa_local import _synthetic_krw

    krw, krwx = _synthetic_krw(), _synthetic_krw(start="2000-06-01")
    inp = _inputs(start="1997-01-01", end="2026-10-20")
    monkeypatch.setitem(S.LOADERS, "2005", lambda prices: krw)
    monkeypatch.setitem(S.LOADERS, "2000-09", lambda prices: krwx)
    monkeypatch.setattr(S, "load_canaries", lambda: {"usd": inp.usd, "usd_cash": inp.usd_cash, "fx": inp.fx,
                                                     "macro_df": inp.macro_df, "chains": pd.DataFrame(), "failed": []})
    S.main(["--quick", "--prices", "model", "--out", str(tmp_path)])
    for f in ("rules.csv", "comparators.csv", "runs.csv.gz", "equal_cost.csv", "placebo.csv.gz", "placebo_summary.csv",
              "verdicts.csv", "bands.csv.gz", "summary.json"):
        assert (tmp_path / f).exists(), f
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine_check_max_rel"] < 1e-6
    n_pairs = len(S.pairs())
    assert n_pairs == 26 and summary["combos"] == 26 * 4 * 2
    # canaries 1,056 + held 192 + 20 bp 416 + currency 48, as the design counts them
    assert summary["paths"]["신호"] == 1056 + 192 + 416 + 48
    v = pd.read_csv(tmp_path / "verdicts.csv")
    assert len(v) == 208 and not v.duplicated(["배분", "신호", "행동", "판"]).any()
    assert set(v["판정"]) <= {"후보", "탈락"} and v["가짜보다 얕은 비율"].between(0, 1).all()
    passed = v[["1 2005 −15% 안", "2 2000-09 −15% 안"]].all(axis=1)
    assert v.loc[passed, "날짜 이동 최악 MDD"].notna().all() and v.loc[~passed, "날짜 이동 최악 MDD"].isna().all()
    assert (v.loc[passed, "날짜 이동 최악 MDD"] <= v.loc[passed, ["mdd_meas_2005", "mdd_meas_2000-09"]].min(axis=1) + 1e-12).all()
    assert v["같은 비용 x"].between(0, 1).all()
    plac = pd.read_csv(tmp_path / "placebo.csv.gz")
    assert len(plac) == 26 * 4 * 2 * S.QUICK["placebos"] and (plac["밀기"] >= 252).all()
    assert len(pd.read_csv(tmp_path / "equal_cost.csv")) == 2 * 208
    rules = pd.read_csv(tmp_path / "rules.csv")
    assert len(rules) == 14 * 3 and set(rules["신호"]) == set(K.RULES)
    comp = pd.read_csv(tmp_path / "comparators.csv")
    assert (comp["종류"] == "정적 분기").sum() == 10 and (comp["종류"] == "A38 주식 10개월선").sum() == 8
    runs = pd.read_csv(tmp_path / "runs.csv.gz")
    assert set(runs["통화"]) == {"usd", "krw"} and set(runs["비용"]) == {10, 20}
    cool = runs[(runs["판"] == "1개월 금지") & (runs["날짜 이동"] == 0)]
    assert (cool["21일 안 되돌림"] == 0).all()  # the cooldown is the rule
    assert pd.read_csv(tmp_path / "bands.csv.gz", index_col=0).shape[1] == 14
