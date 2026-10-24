"""Rebalancing calculator for the real ISA portfolio (amounts in KRW).

Given what is held now (amount per product, optional price per share), the target weights of a
strategy and a new contribution, it returns the buy/sell amount per product:

- "full": trade every product to its target amount (sells allowed).
- "buy_only": no sells; the contribution goes to the products below target, in proportion to
  their shortfall (whatever is left after every shortfall is filled goes by target weight).
- band (percent points): when every product is within its target ± band after the contribution
  is counted, "full" behaves like "buy_only" (only the new money is placed).

Holdings can be saved to lab_results/aa_holdings.json (local, not committed) so the next quarter
starts from the last entry.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .aa_assets import label
from .store import RESULTS_DIR

HOLDINGS_PATH = RESULTS_DIR / "aa_holdings.json"


def rebalance_orders(holdings: dict[str, float], targets: dict[str, float], contribution: float = 0.0,
                     mode: str = "full", band: float = 0.0, prices: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per product (targets first, then other holdings). Amounts in KRW, weights in fractions."""
    if mode not in ("full", "buy_only"):
        raise ValueError(mode)
    tw = {k: float(v) for k, v in targets.items() if v > 0}
    total_w = sum(tw.values())
    if total_w <= 0:
        raise ValueError("목표 비중의 합이 0입니다.")
    tw = {k: v / total_w for k, v in tw.items()}
    codes = list(tw) + [c for c in holdings if c not in tw]
    cur = np.array([max(float(holdings.get(c, 0.0)), 0.0) for c in codes])
    tgt_w = np.array([tw.get(c, 0.0) for c in codes])
    total = cur.sum() + contribution
    if total <= 0:
        raise ValueError("보유 금액과 입금액이 모두 0입니다.")
    target_amt = tgt_w * total
    within_band = band > 0 and np.all(np.abs(cur / total - tgt_w) <= band / 100.0)
    effective = "buy_only" if (mode == "buy_only" or within_band) else "full"
    if effective == "full":
        trade = target_amt - cur
    else:
        shortfall = np.clip(target_amt - cur, 0.0, None)
        need = shortfall.sum()
        money = max(contribution, 0.0)
        if need <= 0:
            trade = money * tgt_w
        elif need >= money:
            trade = money * shortfall / need
        else:
            trade = shortfall + (money - need) * tgt_w
    shares = np.full(len(codes), np.nan)
    executed = trade.copy()
    if prices:
        for i, c in enumerate(codes):
            px = float(prices.get(c) or 0.0)
            if px > 0:
                shares[i] = np.trunc(trade[i] / px)
                executed[i] = shares[i] * px
    after = cur + executed
    after_total = after.sum() + (contribution - executed.sum())  # unspent money stays as cash
    side = np.where(executed > 0.5, "매수", np.where(executed < -0.5, "매도", "유지"))
    return pd.DataFrame({
        "code": codes, "상품": [label(c) for c in codes], "현재 금액": cur, "현재 비중": cur / total,
        "목표 비중": tgt_w, "목표 금액": target_amt, "구분": side, "매매 금액": executed, "주 수": shares,
        "매매 후 금액": after, "매매 후 비중": after / after_total if after_total > 0 else after,
    }).assign(**{"계산 방식": "입금액만 배분" if effective == "buy_only" else "전체 리밸런싱"})


def load_holdings(path: Path = HOLDINGS_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_holdings(rows: dict[str, dict], path: Path = HOLDINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"updated": str(date.today()), "holdings": rows}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
