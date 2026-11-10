"""A44: satellite decision-date, cost and mixing test (lab/aa_satellite.py). Synthetic data, no network."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab import aa_layers as LY
from lab import aa_satellite as SA
from lab import aa_tactical as TAC
from lab.data import Panel

ROOT = Path(__file__).resolve().parents[1]


def _paths(n=3000, seed=3, nan_head=400):
    rng = np.random.default_rng(seed)
    core = np.cumprod(1 + rng.normal(0.0003, 0.008, n))
    sat = np.cumprod(1 + rng.normal(0.0004, 0.006, n))
    sat[:nan_head] = np.nan
    return core / core[0], sat


def _wide(cols, n, seed, start="2004-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    drift = rng.normal(0.0003, 0.0003, len(cols))
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(drift, 0.01, (n, len(cols))), axis=0), index=dates, columns=cols)


def _panels(n=5400):
    usd = _wide(TAC.ALL_SYMBOLS, n, 1)
    codes = sorted({v[0] for v in TAC.KRW_MAP.values() if v[0]})
    krw = _wide(codes, n, 2)
    cash = pd.Series(0.02 / 252, index=krw.index)
    p = Panel(open=krw, high=krw, low=krw, close=krw, volume=krw * 0 + 1e6, tradable=codes, cash=cash)
    p.benchmark = "379800"
    return p, usd, cash


def test_free_daily_restoring_equals_the_stage5_mix_and_cost_only_lowers_it():
    core, sat = _paths()
    free = SA.mix_rebalanced(core, sat, 0.5, range(len(core)), 0.0)
    assert np.allclose(free, LY.mix(core, sat, 0.5)[0], rtol=0, atol=1e-10)
    assert np.allclose(free[:400], core[:400])  # all in the core until the satellite exists
    dates = pd.bdate_range("2000-09-01", periods=len(core))
    q0 = SA.mix_rebalanced(core, sat, 0.5, SA.quarter_rows(dates), 0.0)
    q10 = SA.mix_rebalanced(core, sat, 0.5, SA.quarter_rows(dates), 10.0)
    assert q10[-1] < q0[-1] and q10[-1] > q0[-1] * 0.99
    assert np.allclose(SA.mix_rebalanced(core, sat, 0.0, [], 10.0), core)
    assert np.allclose(SA.combine(core, sat, 0.5, "daily", dates), LY.mix(core, sat, 0.5)[0])
    assert np.allclose(SA.combine(core, None, 0.5, "monthly", dates), core)


def test_drifting_sleeves_follow_the_two_legs_between_restoring_days():
    core = np.array([1.0, 1.1, 1.21, 1.21])
    sat = np.array([1.0, 1.0, 0.9, 0.9])
    out = SA.mix_rebalanced(core, sat, 0.5, [], 0.0)  # split once on day 0, then drift
    assert out == pytest.approx([1.0, 0.5 * 1.1 + 0.5, 0.5 * 1.21 + 0.45, 0.5 * 1.21 + 0.45])
    fee = SA.mix_rebalanced(core, sat, 0.5, [], 100.0)
    assert fee[0] == pytest.approx(1 - 0.5 * 0.01)  # 1 % on the half that moved


def test_rows_and_tranche_average():
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    m, q = SA.month_rows(dates), SA.quarter_rows(dates)
    assert len(m) == 12 and [dates[i].month for i in q] == [3, 6, 9, 12]
    a = pd.Series([np.nan, 1.0, 2.0, 4.0])
    b = pd.Series([5.0, 10.0, 10.0, 10.0])
    t = SA.tranche_equity([a, b])
    assert t.index.tolist() == [1, 2, 3] and t.tolist() == pytest.approx([1.0, 1.5, 2.5])


def test_satellite_equity_moves_with_the_decision_day_and_the_cost():
    p, usd, cash = _panels()
    e0 = SA.satellite_equity(p, usd, cash, 0, 10.0, "2000-09-01", "2024-12-31")
    e10 = SA.satellite_equity(p, usd, cash, 10, 10.0, "2000-09-01", "2024-12-31")
    e0x = SA.satellite_equity(p, usd, cash, 0, 20.0, "2000-09-01", "2024-12-31")
    assert e0.index[0] < e10.index[0] and not np.isclose(e0.iloc[-1], e10.reindex(e0.index).iloc[-1] / e10.iloc[0])
    assert e0x.iloc[-1] < e0.iloc[-1]
    spec = TAC.TacticalSpec("x", start="2000-09-01", end="2024-12-31")
    assert np.allclose(e0, TAC.run_tactical(p, usd, spec, "krw", cash).equity)  # the registered satellite


def _row(combo, core, share, off, cost, mix, cagr, mdd, window="전체"):
    return {"조합": combo, "본체": core, "구간": window, "위성 몫": share, "오프셋": off, "위성 비용 bp": cost, "섞기": mix,
            "cagr": cagr, "mdd": mdd}


def _table(mdds, cagr20, base=(0.08, -0.13), tr=-0.11):
    rows = [_row("C", "C", 0.0, "—", np.nan, "—", *base)]
    for o, m, c in zip(SA.OFFSETS, mdds, cagr20):
        rows.append(_row("X", "C", 0.5, str(o), 10.0, "quarterly", c + 0.002, m))
        rows.append(_row("X", "C", 0.5, str(o), 20.0, "quarterly", c, m - 0.002))
    rows.append(_row("X", "C", 0.5, SA.TRANCHE, 10.0, "quarterly", 0.09, tr))
    rows.append(_row("X", "C", 0.5, "0", 10.0, "daily", 0.091, -0.10))
    return pd.DataFrame(rows)


def test_verdicts_follow_the_fixed_rules():
    keep = SA.verdict_table(_table([-0.10, -0.11, -0.12, -0.12, -0.125], [0.09] * 5)).iloc[0]
    assert keep["종합"] == "유지" and keep["W2 구역"] == "안" and keep["W1 날짜 견고"] and keep["W3 위성 기여"]
    assert not keep["날짜 민감"] and keep["최악 오프셋"] == "15" and keep["4분할 구역"] == "안"
    assert keep["등록 조건 MDD"] == pytest.approx(-0.10)
    edge = SA.verdict_table(_table([-0.10, -0.11, -0.12, -0.12, -0.137], [0.09] * 5, base=(0.08, -0.16))).iloc[0]
    assert edge["종합"] == "유지" and edge["날짜 민감"]
    edge = SA.verdict_table(_table([-0.12, -0.13, -0.14, -0.15, -0.16], [0.09] * 5, base=(0.08, -0.16))).iloc[0]
    assert edge["종합"] == "경계" and edge["W2 구역"] == "경계" and edge["날짜 민감"]
    out = SA.verdict_table(_table([-0.12, -0.13, -0.14, -0.15, -0.18], [0.09] * 5, base=(0.08, -0.18))).iloc[0]
    assert out["종합"] == "탈락" and out["W2 구역"] == "밖" and out["W1 날짜 견고"]
    w1 = SA.verdict_table(_table([-0.12, -0.13, -0.14, -0.15, -0.21], [0.09] * 5, base=(0.08, -0.25))).iloc[0]
    assert w1["종합"] == "탈락" and not w1["W1 날짜 견고"]
    few = SA.verdict_table(_table([-0.10] * 5, [0.09, 0.09, 0.09, 0.07, 0.07])).iloc[0]  # adds CAGR in 3 of 5 only
    assert few["종합"] == "탈락" and not few["W3 위성 기여"] and few["20bp에서 본체보다 CAGR 높은 오프셋 수"] == 3
    deep = SA.verdict_table(_table([-0.10, -0.10, -0.10, -0.10, -0.142], [0.09] * 5, base=(0.08, -0.13))).iloc[0]
    assert not deep["W3 위성 기여"]  # 20 bp worst −14.4 % is more than 1 %p deeper than the core's −13 %
    assert SA.zone(-0.15) == "안" and SA.zone(-0.1501) == "경계" and SA.zone(-0.1701) == "밖"


def _script():
    spec = importlib.util.spec_from_file_location("aa_a44_satellite", ROOT / "scripts" / "aa_a44_satellite.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_end_to_end_on_synthetic_paths(tmp_path, monkeypatch):
    A = _script()
    p, usd, cash = _panels(6500)
    idx = p.close.index
    dates = idx[(idx >= "2004-01-01")]
    rng = np.random.default_rng(5)
    cores = {n: np.cumprod(1 + rng.normal(0.0003, 0.006, len(dates))) for n in A.CORES}
    monkeypatch.setattr(A, "core_paths", lambda panel, inp: (dates, cores))
    monkeypatch.setitem(SA.WINDOWS, "전체", (str(dates[0].date()), str(dates[-1].date())))
    monkeypatch.setitem(SA.WINDOWS, "위성 구간", (str(dates[600].date()), str(dates[-1].date())))
    R = A.run_set("대표", p, {"tac_close": usd, "tac_cash": cash})
    n_combo = len(A.COMBOS) * len(SA.SHARES) * len(SA.MIXES) * (len(SA.OFFSETS) + 1) * len(SA.COSTS) * 2
    assert len(R) == len(A.CORES) * 2 + n_combo + (len(SA.OFFSETS) + 1) * len(SA.COSTS)
    assert R["cagr"].notna().all() and (R["mdd"] <= 0).all()
    V = SA.verdict_table(R, "전체")
    assert sorted(V["조합"]) == sorted(A.COMBOS) and set(V["종합"]) <= {"유지", "경계", "탈락"}
    # the registered condition of the table is the stage-5 computation: daily fixed-ratio mix of offset 0 at 10 bp
    sat0 = SA.satellite_equity(p, usd, cash, 0, 10.0, "2000-09-01", str(dates[-1].date())).reindex(dates).to_numpy(float)
    name = "③ = (가) + K2 + 변동성 8% + 위성"
    want = LY.mix(cores[A.COMBOS[name]], sat0, 0.5)
    got = R[(R["조합"] == name) & (R["구간"] == "전체") & (R["위성 몫"] == 0.5) & (R["오프셋"] == "0")
            & (R["위성 비용 bp"] == 10.0) & (R["섞기"] == "daily")].iloc[0]
    assert got["cagr"] == pytest.approx(float(LY.cagr(want)[0])) and got["mdd"] == pytest.approx(float(LY.mdd(want)[0]))
    R.to_csv(tmp_path / "runs.csv", index=False)
    back = pd.read_csv(tmp_path / "runs.csv", keep_default_na=False, na_values=[""])
    assert SA.verdict_table(back, "전체")["종합"].tolist() == V["종합"].tolist()  # the saved file gives the same verdicts
    assert isinstance(A.figures(tmp_path, tmp_path / "fig"), list)


def test_registered_presets_are_the_ones_tested_and_the_design_is_on_file():
    A = _script()
    from lab import aa_dynamic as D
    assert set(A.REGISTERED.values()) == {"a42_dyn2", "a42_dyn3"}
    assert all(D.DYNAMIC_PRESETS[k].satellite == 0.5 for k in A.REGISTERED.values())
    assert SA.OFFSETS == (0, 1, 5, 10, 15) and SA.JUDGED == {"mix": "quarterly", "cost": 10.0, "share": 0.5}
    memo = (ROOT / "docs" / "research" / "allocation" / "2026-11-11_a44_design.md").read_text(encoding="utf-8")
    assert "실행 전에 고정" in memo and "W1" in memo and "W3" in memo and "변경 기록" in memo
    dec = (ROOT / "docs" / "research" / "allocation" / "DECISIONS.md").read_text(encoding="utf-8")
    assert "\nA44. " in dec
