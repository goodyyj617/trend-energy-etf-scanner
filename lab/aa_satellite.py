"""A44: decision-date, cost and mixing test of the HAA50+BAA50 satellite inside the registered dynamic presets
(allocation DECISIONS A44, design `docs/research/allocation/2026-11-11_a44_design.md`).

Stage 5 of A42 shifted the rebalancing dates of the core only; the satellite (50 % of representatives ② and ③) was
computed once, decided on the month's last trading day, and mixed with the core at a fixed daily ratio without any
trading cost. A28 had shown that the satellite alone moves from −16 % to −33 % MDD with the decision day. This module
re-measures the combinations with

- the satellite decided `OFFSETS` trading days after month end (`TacticalSpec.offset`), and a four-tranche satellite
  (`tranche_equity`: capital split over offsets 0/5/10/15, tranches never rebalanced against each other);
- the satellite's own switching cost at `COSTS` bp;
- the core/satellite split either restored every day for free (`LY.mix`, what stage 5 did) or held as two drifting
  sleeves and restored on fixed days with a cost on the amount moved (`mix_rebalanced`).

The verdicts (`verdict_table`) are fixed in the design memo before the run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import aa_layers as LY
from . import aa_tactical as TAC

OFFSETS = (0, 1, 5, 10, 15)  # A27/A30 decision-day offsets
TRANCHE_OFFSETS = (0, 5, 10, 15)
TRANCHE = "4분할"
COSTS = (10.0, 20.0)
MIXES = ("daily", "monthly", "quarterly")  # daily = stage 5 (free); the other two pay MIX_COST on the amount moved
MIX_KO = {"daily": "매일 고정 비율 (비용 없음, 5단계 방식)", "monthly": "월말 복원 (10bp)", "quarterly": "분기 말 복원 (10bp)"}
MIX_COST = 10.0
SHARES = (0.25, 0.5)
JUDGED = {"mix": "quarterly", "cost": 10.0, "share": 0.5}  # the condition the verdicts read
STRESS_COST = 20.0
WINDOWS = {"전체": ("2000-09-01", "2024-12-31"), "위성 구간": ("2007-04-02", "2024-12-31")}
MDD_GATE, MDD_EDGE, TIMING_GATE = -0.15, -0.17, -0.20  # A26 zones, A30 date gate
SPREAD_FLAG = 0.03  # best − worst offset MDD above this = '날짜 민감'
CAGR_MIN_OFFSETS = 4  # W3: the satellite must add CAGR in at least this many of the five offsets (20 bp)
MDD_SLACK = 0.01  # W3: worst-offset MDD may be at most this much deeper than the same core without satellite


# --------------------------------------------------------------------------- satellite paths

def satellite_equity(panel, tac_close: pd.DataFrame, tac_cash, offset: int, cost_bps: float, start: str, end: str) -> pd.Series:
    """KRW equity of HAA50+BAA50 (Korean products) decided `offset` trading days after month end."""
    spec = TAC.TacticalSpec("HAA50+BAA50 국내판", start=start, end=end, cost_bps=float(cost_bps), offset=int(offset))
    return TAC.run_tactical(panel, tac_close, spec, "krw", tac_cash).equity


def tranche_equity(parts: list[pd.Series]) -> pd.Series:
    """Equal capital in each part from the first day all of them exist; the parts are never rebalanced against
    each other (four accounts' worth of the same rule read on different days)."""
    df = pd.concat(parts, axis=1).dropna()
    return (df / df.iloc[0]).mean(axis=1)


def month_rows(dates: pd.DatetimeIndex) -> np.ndarray:
    return TAC.sampling_rows(dates, 0)


def quarter_rows(dates: pd.DatetimeIndex) -> np.ndarray:
    rows = month_rows(dates)
    return rows[np.isin(dates[rows].month, (3, 6, 9, 12))]


def mix_rebalanced(core: np.ndarray, sat: np.ndarray, share: float, rows, cost_bps: float = MIX_COST) -> np.ndarray:
    """Two sleeves that drift apart and are restored to `share` at the close of the days in `rows`, paying `cost_bps`
    on the amount moved (the lab's one-way convention). The money is all in the core until the satellite exists
    (NaN); it is split, with the same cost, at the close before the satellite's first return. With every day in
    `rows` and no cost this equals `LY.mix`."""
    core = np.asarray(core, float)
    sat = np.asarray(sat, float)
    n = len(core)
    out = np.ones(n)
    if share <= 0:
        return core / core[0]
    rc = core[1:] / core[:-1] - 1.0
    rs = sat[1:] / sat[:-1] - 1.0
    live = np.isfinite(rs)  # live[j]: the satellite has a return from day j to day j + 1
    rowset = set(int(r) for r in rows)
    c = cost_bps / 1e4
    a, b, started = 1.0, 0.0, False
    for t in range(n):
        if t > 0:
            a *= 1.0 + rc[t - 1]
            if started:
                if not live[t - 1]:  # the satellite stopped: fold it back into the core
                    a, b, started = a + b, 0.0, False
                else:
                    b *= 1.0 + rs[t - 1]
        can = t < n - 1 and live[t]
        if can and (not started or t in rowset):
            tot = a + b
            tot -= abs(share * tot - b) * c
            a, b, started = (1.0 - share) * tot, share * tot, True
        out[t] = a + b
    return out


def combine(core: np.ndarray, sat: np.ndarray | None, share: float, mix: str, dates: pd.DatetimeIndex) -> np.ndarray:
    if sat is None or share <= 0:
        return np.asarray(core, float) / core[0]
    if mix == "daily":
        return LY.mix(core, sat, share)[0]
    rows = month_rows(dates) if mix == "monthly" else quarter_rows(dates)
    return mix_rebalanced(core, sat, share, rows, MIX_COST)


def window_metrics(path: np.ndarray, dates: pd.DatetimeIndex, window: str) -> tuple[float, float]:
    """CAGR and MDD of the path cut (not re-started) to the window."""
    a, b = WINDOWS[window]
    m = np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
    if m.sum() < 30:
        return float("nan"), float("nan")
    p = np.asarray(path, float)[m]
    return float(LY.cagr(p[None, :])[0]), float(LY.mdd(p[None, :])[0])


# --------------------------------------------------------------------------- verdicts

def zone(mdd: float) -> str:
    if not np.isfinite(mdd):
        return "—"
    return "안" if mdd >= MDD_GATE else ("경계" if mdd >= MDD_EDGE else "밖")


def verdict_table(R: pd.DataFrame, window: str = "전체") -> pd.DataFrame:
    """One row per combination. R: long table with columns 조합 · 본체 · 구간 · 위성 몫 · 오프셋 · 위성 비용 bp · 섞기 ·
    cagr · mdd, where the rows with 위성 몫 0 are the same core without the satellite.

    W1 worst MDD over the five offsets ≥ −20 % (judged condition); W2 zone of that worst MDD; W3 the satellite adds:
    at 20 bp its CAGR beats the same core without satellite in ≥ 4 of 5 offsets and the worst-offset MDD is at most
    1 %p deeper than the core's; 날짜 민감 when best − worst MDD > 3 %p. 종합: 유지 (W1 · 안 · W3), 경계 (W1 · 경계 · W3),
    탈락 otherwise."""
    R = R[R["구간"] == window]
    single = R[R["오프셋"].isin([str(o) for o in OFFSETS])]
    rows = []
    for name, g in R[R["위성 몫"] == JUDGED["share"]].groupby("조합", sort=False):
        core_name = g["본체"].iloc[0]
        base = R[(R["본체"] == core_name) & (R["위성 몫"] == 0.0)]
        if base.empty:
            continue
        base_cagr, base_mdd = float(base["cagr"].iloc[0]), float(base["mdd"].iloc[0])
        s = single[(single["조합"] == name) & (single["위성 몫"] == JUDGED["share"]) & (single["섞기"] == JUDGED["mix"])]
        j = s[s["위성 비용 bp"] == JUDGED["cost"]]
        st = s[s["위성 비용 bp"] == STRESS_COST]
        tr = R[(R["조합"] == name) & (R["위성 몫"] == JUDGED["share"]) & (R["섞기"] == JUDGED["mix"])
               & (R["위성 비용 bp"] == JUDGED["cost"]) & (R["오프셋"] == TRANCHE)]
        reg = R[(R["조합"] == name) & (R["위성 몫"] == JUDGED["share"]) & (R["섞기"] == "daily")
                & (R["위성 비용 bp"] == 10.0) & (R["오프셋"] == "0")]
        if len(j) < len(OFFSETS) or len(st) < len(OFFSETS):
            continue
        worst, best = float(j["mdd"].min()), float(j["mdd"].max())
        w1 = worst >= TIMING_GATE
        w2 = zone(worst)
        n_add = int((st["cagr"] > base_cagr).sum())
        worst_st = float(st["mdd"].min())
        w3 = n_add >= CAGR_MIN_OFFSETS and worst_st >= base_mdd - MDD_SLACK
        overall = "유지" if (w1 and w2 == "안" and w3) else ("경계" if (w1 and w2 == "경계" and w3) else "탈락")
        rows.append({
            "조합": name, "구간": window, "본체만 CAGR": base_cagr, "본체만 MDD": base_mdd,
            "등록 조건 CAGR": float(reg["cagr"].iloc[0]) if len(reg) else np.nan,
            "등록 조건 MDD": float(reg["mdd"].iloc[0]) if len(reg) else np.nan,
            "CAGR 최저 (오프셋)": float(j["cagr"].min()), "CAGR 최고 (오프셋)": float(j["cagr"].max()),
            "MDD 최악 (오프셋)": worst, "MDD 최선 (오프셋)": best, "MDD 폭": best - worst,
            "최악 오프셋": str(j.loc[j["mdd"].idxmin(), "오프셋"]),
            "W1 날짜 견고": bool(w1), "W2 구역": w2,
            "20bp에서 본체보다 CAGR 높은 오프셋 수": n_add, "20bp 최악 MDD": worst_st, "W3 위성 기여": bool(w3),
            "날짜 민감": bool(best - worst > SPREAD_FLAG),
            "4분할 CAGR": float(tr["cagr"].iloc[0]) if len(tr) else np.nan,
            "4분할 MDD": float(tr["mdd"].iloc[0]) if len(tr) else np.nan,
            "4분할 구역": zone(float(tr["mdd"].iloc[0])) if len(tr) else "—",
            "종합": overall,
        })
    return pd.DataFrame(rows)


__all__ = ["OFFSETS", "TRANCHE_OFFSETS", "TRANCHE", "COSTS", "MIXES", "MIX_KO", "MIX_COST", "SHARES", "JUDGED", "STRESS_COST",
           "WINDOWS", "satellite_equity", "tranche_equity", "month_rows", "quarter_rows", "mix_rebalanced", "combine",
           "window_metrics", "zone", "verdict_table"]
