"""Trading frictions, taxes and the two-account setup for the allocation lab (DECISIONS A41 §7). Off by default.

What is here is the place to put the numbers later, not a model that is in use. With ``Frictions()`` (the default,
``enabled=False``) `aa_engine.run_allocation` and `aa_search.StaticSim` give bit-for-bit the results they gave
before this module existed (tests/test_aa_frictions.py).

- Per-product one-way costs: ``fee_bps`` (brokerage, e.g. the 금현물 계좌 fee) and ``spread_bps`` (half the bid-ask
  spread). When enabled they are added to the common ``cost_bps`` of each trade, product by product.
- Fund fees (총보수) are not here: the model already subtracts them before listing and real prices include them.
- Accounts: every product belongs to an account (``aa_assets.Product.account``: "isa" or "gold"); ``account``
  overrides that map. Taxes are named per account (``TaxRule``); the numbers default to 0 and nothing computes a tax
  yet: ``after_tax`` raises until the rules are written and checked against the law at the time they are switched on.
  Known examples to check then (2026-10): ISA 일반형 비과세 200만 원 then 9.9 % 분리과세 on the net gain at maturity;
  금현물 계좌 매매차익 비과세; a general account taxes 국내 상장 해외 ETF gains and distributions at 15.4 %.
- ``DepositRule`` describes the 'gold by new deposits only' version (new money goes to the most underweight side
  first, gold is never sold). It is a setting only; running it raises until the deposit simulation exists.
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


@dataclass(frozen=True)
class DepositRule:
    """'New deposits only' rebalancing for products that cannot be sold or moved freely (setting only)."""
    every: str = "M"  # "M" | "Q"
    amount_pct: float = 0.0  # each deposit as a percent of the starting value
    never_sell: tuple[str, ...] = ()


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
            raise NotImplementedError("세금 계산은 아직 없습니다. 켜기 전에 그 시점의 제도를 확인해 규칙을 넣어야 합니다.")
        if self.deposits is not None:
            raise NotImplementedError("'새 입금으로만' 리밸런싱은 설정만 있고 계산은 아직 없습니다.")


OFF = Frictions()


def resolve(fr: Frictions | None) -> Frictions | None:
    """None when the frictions change nothing (the simulators then take their old code path)."""
    if fr is None or not fr.active:
        return None
    fr.check_supported()
    return fr


def after_tax(*_args, **_kwargs):
    raise NotImplementedError("세금 계산은 아직 없습니다(DECISIONS A41 §7).")


__all__ = ["ACCOUNTS", "TaxRule", "DepositRule", "Frictions", "OFF", "resolve", "after_tax", "CASH"]
