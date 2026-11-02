"""A42 stage 3: deposits, taxes and two accounts (lab/aa_accounts.py, aa_frictions rules). Synthetic panels, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import aa_accounts as AC
from lab.aa_assets import CASH
from lab.aa_engine import AllocConfig, run_allocation
from lab.aa_frictions import DepositRule, TaxRule, after_tax
from lab.data import Panel


def _panel(n=900, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-01", periods=n)
    codes = ["379800", "KRXGOLD", "114260"]
    r = rng.normal([0.0004, 0.0003, 0.0001], [0.012, 0.010, 0.002], size=(n, 3))
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=codes)
    cash = pd.Series(0.03 / 252, index=dates)
    p = Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6, tradable=codes, cash=cash)
    p.benchmark = "379800"
    return p, codes


W = {"379800": 0.5, "KRXGOLD": 0.3, "114260": 0.2}


def _run(p, codes, policy, **kw):
    dates = p.close.index
    rets = p.close[codes].pct_change(fill_method=None).to_numpy()
    return AC.run_policy(rets, p.cash.to_numpy(), dates, codes, np.array([W[c] for c in codes]), policy, **kw)


@pytest.mark.parametrize("cost", ["zero", "flat"])
def test_single_lump_full_no_tax_matches_the_engine(cost):
    p, codes = _panel()
    pol = AC.Policy("single", "lump", "full", cost, "none", "none")
    res = _run(p, codes, pol, charge_first=False)
    eng = run_allocation(p, AllocConfig("x", W, str(p.close.index[0].date()), str(p.close.index[-1].date()), rebalance="Q",
                                        cost_bps=0.0 if cost == "zero" else AC.FLAT_BPS))
    e = eng.equity.to_numpy()
    assert np.allclose(res["twr"], e / e[0], rtol=0, atol=1e-10)
    assert np.allclose(res["value"] / res["value"][0], res["twr"], atol=1e-12)  # no flows after day 0


def test_twr_ignores_deposits_and_irr_matches_a_constant_return():
    dates = pd.bdate_range("2010-01-01", periods=756)
    g = 0.0003
    rets = np.full((756, 1), g)
    cash = np.zeros(756)
    pol = AC.Policy("single", "monthly", "full", "zero", "none", "none")
    res = AC.run_policy(rets, cash, dates, ["379800"], np.array([1.0]), pol)
    assert np.isclose(res["twr_cagr"], (1 + g) ** 252 - 1, rtol=1e-9)
    assert res["twr_mdd"] == 0.0
    assert res["deposited"] == AC.FIRST_KRW + AC.MONTHLY_KRW * (len(set(dates.to_period("M"))) - 1)
    # money-weighted return of a constant daily return, on calendar time (252 trading days ≈ 365.25 days, roughly)
    assert 0.06 < res["irr"] < 0.10


def test_tax_rules_and_after_tax():
    assert AC.ISA_GENERAL.tax(1_000_000) == 0.0
    assert np.isclose(AC.ISA_GENERAL.tax(12_000_000), 0.099 * 10_000_000)
    assert np.isclose(AC.ISA_LOW.tax(12_000_000), 0.099 * 8_000_000)
    assert after_tax(150.0, 100.0, TaxRule("x", 10.0, 0.5)) == 130.0
    assert after_tax(150.0, 100.0, None) == 150.0
    assert DepositRule().first_krw == 0.0  # old construction still works


def test_tax_lowers_value_and_low_income_pays_less():
    p, codes = _panel(1500)
    vals = {t: _run(p, codes, AC.Policy("single", "lump", "full", "flat", t, "none")) for t in ("none", "general", "low")}
    assert vals["none"]["tax"] == 0.0
    assert vals["general"]["tax"] > vals["low"]["tax"] > 0
    assert vals["none"]["final"] > vals["low"]["final"] > vals["general"]["final"]


def test_isa_limit_parks_money_in_general_cash():
    p, codes = _panel(700)
    res = _run(p, codes, AC.Policy("single", "lump", "full", "flat", "none", "isa"))
    # 1억 lump, 2,000만 room a year: after ~2.8 years about 4,000만 – 6,000만 still waits
    assert 30e6 < res["general_cash_end"] < 70e6
    free = _run(p, codes, AC.Policy("single", "lump", "full", "flat", "none", "none"))
    assert free["general_cash_end"] == 0.0


def test_split_account_never_sells_gold():
    p, codes = _panel(900)
    res = _run(p, codes, AC.Policy("split", "monthly", "full", "product", "none", "none"))
    single = _run(p, codes, AC.Policy("single", "monthly", "full", "product", "none", "none"))
    assert np.isfinite(res["twr_cagr"]) and res["final"] > 0
    assert not np.allclose(res["value"], single["value"])


def test_policy_grid_has_324_and_defaults():
    pols = AC.all_policies()
    assert len(pols) == 324 and len(set(pols)) == 324
    assert AC.DECLARED_3 in pols and AC.BASELINE not in pols  # the baseline has no cost: run separately
    both = AC.all_policies(overflow=True)
    assert len(both) == 648 and AC.DECLARED in both and AC.DECLARED.overflow == "invest"
    assert set(pols) == {q for q in both if q.overflow == "cash"}  # stage 3's 324 are the 'cash' half
    assert CASH not in AC.PRODUCT_FEES


def test_stage3_result_files_for_the_app():
    from lab import aa_experiments as ex
    d = ex.a42_3_declared()
    assert len(d) == 8 and d["표시"].dtype == bool
    a = ex.a42_3_axes()
    assert a.shape == (16, 8)  # 2 + 3 + 3 + 3 + 3 + 2 levels


def test_taxable_in_general():
    assert AC.taxable_in_general("379800") and AC.taxable_in_general("114260") and AC.taxable_in_general("411060")
    assert not AC.taxable_in_general("KRXGOLD") and not AC.taxable_in_general("069500")


def test_overflow_invest_changes_nothing_without_a_limit():
    p, codes = _panel(900)
    for tax in ("none", "general"):
        a = _run(p, codes, AC.Policy("split", "monthly", "full", "product", tax, "none", "cash"))
        b = _run(p, codes, AC.Policy("split", "monthly", "full", "product", tax, "none", "invest"))
        assert np.array_equal(a["value"], b["value"]) and a["tax"] == b["tax"]


def test_overflow_invest_keeps_the_money_working():
    p, codes = _panel(1500)
    cash = _run(p, codes, AC.Policy("single", "lump", "full", "product", "none", "isa", "cash"))
    inv = _run(p, codes, AC.Policy("single", "lump", "full", "product", "none", "isa", "invest"))
    assert cash["general_cash_end"] > 0 and inv["general_cash_end"] == 0.0 and inv["general_invested_end"] > 0
    ideal = _run(p, codes, AC.Policy("single", "lump", "full", "product", "none", "none"))
    # no tax: the invested general account follows the same targets, so the result is close to having no limit
    assert abs(inv["twr_cagr"] - ideal["twr_cagr"]) < abs(cash["twr_cagr"] - ideal["twr_cagr"])
    taxed = _run(p, codes, AC.Policy("single", "lump", "full", "product", "general", "isa", "invest"))
    assert taxed["tax"] > 0 and taxed["final"] < inv["final"]


def test_stage3b_result_file_for_the_app():
    from lab import aa_experiments as ex
    t = ex.a42_3_overflow()
    assert len(t) == 8 and not t.isna().any().any()
    assert (t["선언 · 한도 밖 굴림"] > t["선언 · 한도 밖 현금"]).all()
