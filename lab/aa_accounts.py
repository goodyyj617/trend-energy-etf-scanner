"""Operating policies with deposits, taxes and two accounts (allocation DECISIONS A42 §3).

`run_policy` follows one allocation day by day in money (KRW) under a `Policy`:

- accounts: "single" (everything in one ISA, gold included) or "split" (ISA + 금현물 계좌: KRX gold lives in the gold
  account, is filled only by new deposits and never sold; nothing moves between the accounts).
- deposits: "lump" (1억 원 on day 0), "monthly" (1,000만 원 on day 0, then 100만 원 on each month's first trading day),
  "lump_monthly" (1억 원, then 100만 원 a month).
- execution at the quarterly decisions (decided at the close, traded at the next close, like aa_engine): "full" (back to
  target), "deposit" (new money goes to the most underweight products; a full correction only at the December
  decision), "band" (full only if some weight is off target by more than 5 %p).
- cost: "zero", "flat" (10 bp), "product" (brokerage fee + half spread per product, `PRODUCT_FEES`), "double".
  Like aa_engine the cost is charged on one-way turnover: half the gross traded at a rebalance, the full amount of a
  purchase with new money.
- tax: "none", "general" (ISA 일반형: 200만 원 of net gain tax-free, 9.9 % above), "low" (서민형: 400만 원). The ISA is
  taxed when it is closed: every three years (closed and re-opened, the after-tax balance re-deposited) and at the end.
  The gold account is tax-free; cash parked in a general account earns the panel's cash rate less 15.4 % tax.
- limit: "none" or "isa" (2,000만 원 a year, unused room carried over, 1억 원 per account). Money over the limit waits
  as general-account cash and moves in when room opens (each January, or at a re-opening). With the limit on the ISA
  is re-opened every three years even without tax (the way a holder resets the 1억 원 room).

Returns money-weighted (IRR) and time-weighted (TWR) results; MDD is on the TWR index. With a single account, one lump
sum, full rebalancing, no tax and no limit the TWR index equals `aa_engine.run_allocation`'s equity (tests).
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .aa_engine import TRADING_DAYS, decision_rows
from .aa_frictions import DepositRule, TaxRule

GOLD = "KRXGOLD"
LUMP_KRW = 100_000_000
FIRST_KRW = 10_000_000
MONTHLY_KRW = 1_000_000
ISA_ANNUAL, ISA_TOTAL, ISA_YEARS = 20_000_000, 100_000_000, 3
BAND_PP = 5.0
FLAT_BPS = 10.0
GENERAL_RATE = 0.154

ISA_GENERAL = TaxRule("isa", 2_000_000, 0.099, "ISA 일반형: 순이익 200만 원 비과세, 초과분 9.9% 분리과세 (만기·해지 때)")
ISA_LOW = TaxRule("isa", 4_000_000, 0.099, "ISA 서민형: 순이익 400만 원 비과세, 초과분 9.9%")
GENERAL_CASH = TaxRule("general", 0.0, GENERAL_RATE, "일반 계좌 이자 15.4%")
TAXES = {"none": None, "general": ISA_GENERAL, "low": ISA_LOW}

# one-way brokerage fee and half spread, bp. Spread = half of one tick (KRX ETF tick 5원, KRX gold 10원/g) over the
# 2026-10-27 close; fee: online ISA 1.5 bp, 금현물 계좌 20 bp. Assumed, not measured quotes (design 0-2 measured none).
_PRICE = {"379810": 25512, "379800": 22830, "453630": 12240, "453640": 14249, "463640": 13035, GOLD: 170390,
          "459580": 1073000, "0046A0": 9150, "329750": 12560, "305080": 11595, "0091C0": 9390, "476760": 8514,
          "484790": 8144, "114260": 62430, "148070": 107775, "439870": 94515, "446720": 13555,
          "251350": 36990, "291890": 11365, "411060": 23765}  # the last three: F2 / F3 presets
PRODUCT_FEES = {c: {"fee": 20.0 if c == GOLD else 1.5, "half_spread": (5.0 if c == GOLD else 2.5) / p * 1e4}
                for c, p in _PRICE.items()}

AXES = {"accounts": ("single", "split"), "deposits": ("lump", "monthly", "lump_monthly"),
        "execution": ("full", "deposit", "band"), "cost": ("flat", "product", "double"),
        "tax": ("none", "general", "low"), "limit": ("none", "isa")}
LABELS = {"single": "한 계좌", "split": "ISA + 금현물 분리", "lump": "목돈 1억", "monthly": "1,000만 + 매월 100만",
          "lump_monthly": "목돈 1억 + 매월 100만", "full": "전체", "deposit": "새 입금 + 연 1회 보정", "band": "밴드 5%p",
          "zero": "비용 없음", "flat": "10bp", "product": "상품별 수수료 + 반 스프레드", "double": "상품별 × 2",
          "none": "없음", "general": "ISA 일반형", "low": "ISA 서민형", "isa": "연 2,000만 · 총 1억"}


@dataclass(frozen=True)
class Policy:
    accounts: str = "single"
    deposits: str = "lump"
    execution: str = "full"
    cost: str = "flat"
    tax: str = "none"
    limit: str = "none"

    def label(self) -> str:
        lim = "한도 없음" if self.limit == "none" else "한도 " + LABELS["isa"]
        return " · ".join([LABELS[self.accounts], LABELS[self.deposits], LABELS[self.execution], LABELS[self.cost],
                           "세금 " + LABELS[self.tax], lim])

    def deposit_rule(self) -> DepositRule:
        first = LUMP_KRW if self.deposits in ("lump", "lump_monthly") else FIRST_KRW
        monthly = 0.0 if self.deposits == "lump" else MONTHLY_KRW
        return DepositRule("M", 0.0, (GOLD,) if self.accounts == "split" else (), first_krw=first, amount_krw=monthly)


BASELINE = Policy("single", "lump", "full", "zero", "none", "none")  # 한 계좌 · 목돈 · 비용 없음
DECLARED = Policy("split", "monthly", "full", "product", "general", "isa")  # design §3 default (cost / limit: change log)


def all_policies() -> list[Policy]:
    return [Policy(*v) for v in itertools.product(*AXES.values())]


def cost_vector(codes: list[str], cost: str) -> np.ndarray:
    if cost == "zero":
        return np.zeros(len(codes))
    if cost == "flat":
        return np.full(len(codes), FLAT_BPS / 1e4)
    k = 2.0 if cost == "double" else 1.0
    return np.array([k * (PRODUCT_FEES[c]["fee"] + PRODUCT_FEES[c]["half_spread"]) / 1e4 for c in codes])


def _rows_after(dates: pd.DatetimeIndex, t: int, years: int) -> int:
    return int(dates.searchsorted(dates[t] + pd.DateOffset(years=years)))


def irr(flows: np.ndarray, times: np.ndarray) -> float:
    """Annual money-weighted return: flows (deposits negative, final value positive) at times in years."""
    def npv(r):
        return float(np.sum(flows / (1.0 + r) ** times))
    lo, hi = -0.99, 1.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def run_policy(rets: np.ndarray, cash: np.ndarray, dates: pd.DatetimeIndex, codes: list[str], weights: np.ndarray,
               policy: Policy, rebalance: str = "Q", charge_first: bool = True) -> dict:
    """rets: daily product returns (T × N, NaN = 0), cash: daily cash return (T,), weights: target fractions (N,).
    Returns the daily value, flows, TWR index and the summary metrics."""
    R = np.nan_to_num(np.asarray(rets, float), nan=0.0)
    C = np.nan_to_num(np.asarray(cash, float), nan=0.0)
    T, N = R.shape
    w = np.asarray(weights, float) / np.sum(weights)
    cv = cost_vector(codes, policy.cost)
    gi = codes.index(GOLD) if GOLD in codes else None
    split = policy.accounts == "split" and gi is not None and w[gi] > 0
    isa_w = w.copy()
    if split:
        isa_w[gi] = 0.0
        isa_w = isa_w / isa_w.sum() if isa_w.sum() > 0 else isa_w
    rule = TAXES[policy.tax]
    taxed = rule is not None
    limit = policy.limit == "isa"
    dep = policy.deposit_rule()
    deposits = {0: dep.first_krw}
    if dep.amount_krw:
        months = pd.Series(np.arange(T), index=dates).groupby(dates.to_period("M")).first().to_numpy()
        for r in months[1:]:
            deposits[int(r)] = deposits.get(int(r), 0.0) + dep.amount_krw
    dec = set(int(i) for i in decision_rows(dates, rebalance) if i < T - 1)
    first_of_year = set(pd.Series(np.arange(T), index=dates).groupby(dates.year).first().to_numpy()[1:].tolist())

    st = {"h": np.zeros(N), "gold": 0.0, "G": 0.0, "basis": 0.0, "allow": np.inf, "left": np.inf, "roll": None,
          "cost": 0.0, "tax": 0.0}
    if limit:
        st["allow"], st["left"] = 0.0, float(ISA_TOTAL)

    def buy_isa(x: float, t: int, free: bool) -> None:
        if x <= 0:
            return
        h = st["h"]
        tgt = isa_w * (h.sum() + x)
        gap = np.maximum(tgt - h, 0.0)
        b = gap / gap.sum() * min(x, gap.sum()) if gap.sum() > 0 else np.zeros(N)
        rest = x - b.sum()
        if rest > 1e-9:
            b = b + isa_w * rest
        c = 0.0 if free else float(b @ cv)
        if c:
            b = b * x / (x + c)
            c = x - b.sum()
        st["h"] = h + b
        st["cost"] += c

    def into_isa(x: float, t: int, free: bool = False) -> None:
        """New money for the ISA: up to the room, the rest waits as general cash."""
        if limit:
            room = min(st["allow"], st["left"])
            y = min(x, room)
            st["G"] += x - y
            st["allow"] -= y
            st["left"] -= y
            x = y
        if x <= 0:
            return
        if st["roll"] is None and (limit or taxed):
            st["roll"] = _rows_after(dates, t, ISA_YEARS)
        st["basis"] += x
        buy_isa(x, t, free)

    def route(x: float, t: int, free: bool) -> None:
        if split:
            h_total = st["h"].sum() + st["gold"] + x
            gap = max(0.0, w[gi] * h_total - st["gold"])
            g = min(x, gap)
            if g > 0:
                c = 0.0 if free else g * cv[gi] / (1 + cv[gi])
                st["gold"] += g - c
                st["cost"] += c
            x -= g
        into_isa(x, t, free)

    def rebalance_isa() -> None:
        h = st["h"]
        v = h.sum()
        if v <= 0:
            return
        tgt = isa_w * v
        c = float(np.abs(tgt - h) @ cv) / 2.0
        st["h"] = isa_w * (v - c)
        st["cost"] += c

    def off_band() -> bool:
        h = st["h"].copy()
        if split:
            h[gi] = st["gold"]
        v = h.sum()
        return v > 0 and np.abs(h / v - w).max() > BAND_PP / 100.0

    V = np.empty(T)
    F = np.zeros(T)
    pending = False
    gen_r = 1.0 - (GENERAL_RATE if taxed else 0.0)
    for t in range(T):
        st["h"] = st["h"] * (1.0 + R[t])
        if split:
            st["gold"] *= 1.0 + R[t, gi]
        st["G"] *= 1.0 + C[t] * gen_r
        if pending:
            if policy.execution == "full" or (policy.execution == "band" and off_band()) or \
                    (policy.execution == "deposit" and dates[t - 1].month == 12):
                rebalance_isa()
            pending = False
        if limit and t in first_of_year:
            st["allow"] = min(st["allow"] + ISA_ANNUAL, st["left"])
            if st["G"] > 0:
                g = st["G"]
                st["G"] = 0.0
                into_isa(g, t)
        if st["roll"] is not None and t >= st["roll"]:
            v = st["h"].sum()
            tax = rule.tax(v - st["basis"]) if taxed else 0.0
            sell = float(st["h"] @ cv)  # close and re-open: sell all and buy back = one-way turnover v
            st["tax"] += tax
            st["cost"] += sell
            st["h"] = np.zeros(N)
            st["basis"], st["roll"] = 0.0, None
            if limit:
                st["left"], st["allow"] = float(ISA_TOTAL), float(ISA_ANNUAL)
            into_isa(v - tax - sell, t, free=True)  # the buy-back cost is in `sell`
        if t in deposits:
            F[t] = deposits[t]
            route(deposits[t], t, free=(t == 0 and not charge_first))
        if t in dec:
            pending = True
        V[t] = st["h"].sum() + st["gold"] + st["G"]
    if taxed:
        final_tax = rule.tax(st["h"].sum() - st["basis"])
        st["tax"] += final_tax
        V[-1] -= final_tax
    prev = np.concatenate([[np.nan], V[:-1]])
    r = np.where(np.arange(T) == 0, 0.0, (V - F) / prev - 1.0)
    twr = np.cumprod(1.0 + r)
    years = (T - 1) / TRADING_DAYS
    times = np.asarray((dates - dates[0]).days, float) / 365.25
    flows = -F.copy()
    flows[-1] += V[-1]
    peak = np.maximum.accumulate(twr)
    out = {"value": V, "flows": F, "twr": twr,
           "twr_cagr": float(twr[-1] ** (1.0 / max(years, 1e-9)) - 1.0), "twr_mdd": float((twr / peak - 1.0).min()),
           "irr": irr(flows, times), "final": float(V[-1]), "deposited": float(F.sum()), "tax": float(st["tax"]),
           "cost": float(st["cost"]), "general_cash_end": float(st["G"]), "years": years}
    return out


def policy_row(policy: Policy) -> dict:
    return {**asdict(policy), "정책": policy.label()}


__all__ = ["Policy", "BASELINE", "DECLARED", "AXES", "LABELS", "all_policies", "run_policy", "irr", "cost_vector",
           "PRODUCT_FEES", "ISA_GENERAL", "ISA_LOW", "GENERAL_CASH", "policy_row"]
