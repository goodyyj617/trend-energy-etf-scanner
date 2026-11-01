"""Trading frictions, taxes and the two-account setup for the allocation lab (DECISIONS A41 §7). Off by default.

What is here is the place to put the numbers later, not a model that is in use. With ``Frictions()`` (the default,
``enabled=False``) `aa_engine.run_allocation` and `aa_search.StaticSim` give bit-for-bit the results they gave
before this module existed (tests/test_aa_frictions.py).

- Per-product one-way costs: ``fee_bps`` (brokerage, e.g. the 금현물 계좌 fee) and ``spread_bps`` (half the bid-ask
  spread). When enabled they are added to the common ``cost_bps`` of each trade, product by product.
- Fund fees (총보수) are not here: the model already subtracts them before listing and real prices include them.
- Accounts: every product belongs to an account (``aa_assets.Product.account``: "isa" or "gold"); ``account``
  overrides that map. ``TaxRule`` is a tax on net gain: ``tax(gain)`` = rate × (gain − exempt), never negative
  (A42 §3, rules as of 2026-10: ISA 일반형 200만 원 비과세 then 9.9 % at closing, 서민형 400만 원; 금현물 계좌 매매차익
  비과세; general-account interest 15.4 %). Check the law again before using the numbers for a real decision.
- ``DepositRule`` is the deposit schedule (first amount, then a fixed amount each month) and the products that are
  never sold ('gold by new deposits only': new money goes to the most underweight side first).
- Taxes and deposits move money in and out, so the weight simulators below (`run_allocation`, `StaticSim`) do not take
  them and still raise when they are switched on; `lab/aa_accounts.py` (`run_policy`) runs them (A42 §3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aa_assets import CASH, PRODUCTS

ACCOUNTS = {"isa": "ISA", "gold": "금현물 계좌", "general": "일반 계좌"}


@dataclass(frozen=True)
class TaxRule:
    """Numbers to fill in when taxes are switched on (all 0 = no tax). Amounts in KRW, rates as fractions."""
    account: str
    exempt_krw: float = 0.0  # tax-free amount of net gain (ISA)
    rate: float = 0.0  # rate on the taxable part
    note: str = ""

    def tax(self, gain: float) -> float:
        return max(0.0, float(gain) - self.exempt_krw) * self.rate


@dataclass(frozen=True)
class DepositRule:
    """Deposit schedule and the products that are only bought with new money (aa_accounts.run_policy)."""
    every: str = "M"  # "M" | "Q"
    amount_pct: float = 0.0  # each deposit as a percent of the starting value (when amount_krw is 0)
    never_sell: tuple[str, ...] = ()
    first_krw: float = 0.0  # deposit on the first day, KRW
    amount_krw: float = 0.0  # each later deposit, KRW


@dataclass(frozen=True)
class Frictions:
    enabled: bool = False
    fee_bps: dict = field(default_factory=dict)  # code -> one-way brokerage fee, bp
    spread_bps: dict = field(default_factory=dict)  # code -> half spread, bp
    account: dict = field(default_factory=dict)  # code -> account key, overriding Product.account
    taxes: tuple[TaxRule, ...] = ()
    deposits: DepositRule | None = None

    @property
    def active(self) -> bool:
        return bool(self.enabled)

    def account_of(self, code: str) -> str:
        if code in self.account:
            return self.account[code]
        p = PRODUCTS.get(code)
        return p.account if p is not None else "isa"

    def cost_vector(self, columns: list[str], cost_bps: float) -> np.ndarray:
        """One-way cost per column as a fraction: the common cost plus the product's fee and half spread."""
        return np.array([(cost_bps + float(self.fee_bps.get(c, 0.0)) + float(self.spread_bps.get(c, 0.0))) / 1e4
                         for c in columns], dtype=float)

    def check_supported(self) -> None:
        """Raise for settings the simulators do not implement yet (taxes with numbers, deposit-only rebalancing)."""
        if not self.active:
            return
        if any(t.rate or t.exempt_krw for t in self.taxes):
            raise NotImplementedError("세금은 입금·인출이 있는 계좌 시뮬레이터(lab/aa_accounts.py)에서만 계산합니다.")
        if self.deposits is not None:
            raise NotImplementedError("입금 규칙은 계좌 시뮬레이터(lab/aa_accounts.py)에서만 계산합니다.")


OFF = Frictions()


def resolve(fr: Frictions | None) -> Frictions | None:
    """None when the frictions change nothing (the simulators then take their old code path)."""
    if fr is None or not fr.active:
        return None
    fr.check_supported()
    return fr


def after_tax(value: float, basis: float, rule: TaxRule | None) -> float:
    """Value after the tax due when an account with this cost basis is closed (no rule = no tax)."""
    return float(value) - (rule.tax(float(value) - float(basis)) if rule is not None else 0.0)


__all__ = ["ACCOUNTS", "TaxRule", "DepositRule", "Frictions", "OFF", "resolve", "after_tax", "CASH"]
