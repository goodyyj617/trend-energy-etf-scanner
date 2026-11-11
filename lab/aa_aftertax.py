"""A45: after-tax final table of the recommended allocations, and whether the volatility-target layer survives taxes
(allocation DECISIONS A45, design `docs/research/allocation/2026-11-11_a45_design.md`, fixed before the run).

Pieces:

  ``vol_levels_of``       the stage-5 reference computation of the 10 % volatility target on a static allocation:
                          the static mix run by `aa_canary.DynamicSim` (columns = its products ∪ the two defensive
                          equities ∪ the CD-rate ETF, 10 bp, quarterly), 60-day volatility, month-end k in 0.05 steps
  ``vol_targets``         T × N targets for `aa_accounts.run_policy(targets=)`: row t = the target a decision at the
                          close of t picks (the level held from t+1), i.e. `aa_layers.scaled_tables` rows
  ``mean_exposure``       the average exposure k held over the days (for the same-average-exposure static)
  ``mean_exposure_weights``  static weights × k̄, the rest in the CD-rate ETF
  ``layer_verdict``       X3 (vol layer vs the same-average-exposure static under one policy)
  ``zone``                X1 (안 / 경계 / 밖)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import aa_canary as K
from . import aa_layers as LY

VOL = 0.10
CD = LY.CD
COMMON_START = "2003-11-07"
LINE, NEAR = -0.15, -0.17
CAGR_PP, MDD_PP = 0.005, 0.02  # X2 marks (stage 3b)
LAYER_MDD_PP, LAYER_CAGR_PP = 0.02, 0.005  # X3

STATIC_OF = {"(가)": "F4 (가)", "안정형 40": "F2 안정형 40", "R6 정적": "R6 주식 위험 몫 상한",
             "현재 보유": "F3 현재 보유", "S&P500": "F5 S&P500"}  # name here → stage-3 target name
VOL_NAME = "R6 + 변동성 목표 10%"
MEAN_NAME = "R6 평균 노출 정적"
ORDER = ("(가)", "안정형 40", VOL_NAME, "R6 정적", MEAN_NAME, "현재 보유", "S&P500")
RECOMMENDED = ("(가)", "안정형 40", VOL_NAME)
STAGE5_REF = "R6 주식 위험 몫 상한 + 변동성 목표 10% (참고)"


def vol_levels_of(panel, w: dict[str, float], start: str, end: str, target: float = VOL
                  ) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(dates, level index per row) exactly as stage 5 computes the reference row's volatility target."""
    cols = sorted(set(w) | set(K.DEFENSIVE) | {CD})
    sim = K.DynamicSim(panel, cols, start, end, cost_bps=10.0)
    base_tab = LY.action_tables(w, "none", "none", sim.columns)
    bp, _, _ = sim.run(*K.static_rows(base_tab, sim))
    base_r = np.concatenate([[0.0], bp[0, 1:] / bp[0, :-1] - 1.0])
    return sim.dates, LY.vol_levels(base_r, sim.dates, target)


def vol_targets(w: dict[str, float], codes: list[str], lev: np.ndarray) -> np.ndarray:
    """T × len(codes): row t = static weights × k + CD × (1 − k) for the level held from row t+1 (the last row
    repeats). The first row is also the starting target, so the level must not change between rows 0 and 1."""
    if CD not in codes:
        raise ValueError("the CD-rate ETF must be one of the codes")
    lev = np.asarray(lev, int)
    if len(lev) > 1 and lev[0] != lev[1]:
        raise ValueError("the exposure level changes on the first day")
    tab = LY.scaled_tables(LY.action_tables(w, "none", "none", list(codes)), list(codes), True)
    lev_next = np.concatenate([lev[1:], lev[-1:]])
    return tab[lev_next]


def mean_exposure(lev: np.ndarray) -> float:
    """Average exposure k over the days (the level each day holds)."""
    return float(np.mean(LY.K_LEVELS[np.asarray(lev, int)]))


def mean_exposure_weights(w: dict[str, float], kbar: float) -> dict[str, float]:
    """Static weights (in %) scaled by k̄, the rest in the CD-rate ETF."""
    tot = sum(w.values())
    out = {c: 100.0 * x / tot * kbar for c, x in w.items()}
    out[CD] = out.get(CD, 0.0) + 100.0 * (1.0 - kbar)
    return out


def zone(mdd: float) -> str:
    return "안" if mdd >= LINE else "경계" if mdd >= NEAR else "밖"


def layer_verdict(vol_cagr: float, vol_mdd: float, ref_cagr: float, ref_mdd: float) -> str:
    """X3: the vol-target allocation against the same-average-exposure static under one policy."""
    if vol_mdd - ref_mdd < LAYER_MDD_PP:
        return "층 없음"
    return "층 유지" if ref_cagr - vol_cagr <= LAYER_CAGR_PP else "층 비쌈"


def table_rows(R: pd.DataFrame, baseline: str = "기준선", declared: str = "선언 정책") -> pd.DataFrame:
    """Per target: baseline and declared after-tax numbers, X1 zone and X2 marks. R has one row per target × policy
    with the boolean columns `baseline` / `declared` and twr_cagr / twr_mdd / irr / final / deposited / 세금 / 비용."""
    rows = []
    for n, g in R.groupby("대상", sort=False):
        b = g[g[baseline]].iloc[0]
        d = g[g[declared]].iloc[0]
        pol = g[~g[baseline]]
        row = {"대상": n, "시작": b["시작"], "기준선 CAGR": b["twr_cagr"], "기준선 MDD": b["twr_mdd"],
               "세후 CAGR (TWR)": d["twr_cagr"], "세후 MDD (TWR)": d["twr_mdd"], "세후 IRR": d["irr"],
               "CAGR 차": b["twr_cagr"] - d["twr_cagr"], "MDD 차": b["twr_mdd"] - d["twr_mdd"],
               "세금 (원)": d["세금 (원)"], "비용 (원)": d["비용 (원)"], "입금 (원)": d["deposited"], "최종 (원)": d["final"],
               "정책 CAGR 최소": pol["twr_cagr"].min(), "정책 CAGR 최대": pol["twr_cagr"].max(),
               "정책 MDD 최악": pol["twr_mdd"].min(), "정책 수": len(pol)}
        row["X1 세후 구역"] = zone(row["세후 MDD (TWR)"])
        row["X2 표시"] = bool(row["CAGR 차"] > CAGR_PP or abs(row["MDD 차"]) > MDD_PP)
        rows.append(row)
    return pd.DataFrame(rows)


def layer_rows(R: pd.DataFrame, policy_cols: list[str]) -> pd.DataFrame:
    """X3 under every policy: the vol allocation vs the same-average-exposure static and vs the static R6."""
    key = lambda n: R[R["대상"] == n].set_index(policy_cols)  # noqa: E731
    v, m, s = key(VOL_NAME), key(MEAN_NAME), key("R6 정적")
    out = pd.DataFrame(index=v.index)
    out["기준선"] = v["기준선"]
    out["선언 정책"] = v["선언 정책"]
    out["층 CAGR"], out["층 MDD"] = v["twr_cagr"], v["twr_mdd"]
    out["평균 노출 CAGR"], out["평균 노출 MDD"] = m["twr_cagr"].reindex(v.index), m["twr_mdd"].reindex(v.index)
    out["정적 CAGR"], out["정적 MDD"] = s["twr_cagr"].reindex(v.index), s["twr_mdd"].reindex(v.index)
    out["MDD 이점 (vs 평균 노출)"] = out["층 MDD"] - out["평균 노출 MDD"]
    out["CAGR 손실 (vs 평균 노출)"] = out["평균 노출 CAGR"] - out["층 CAGR"]
    out["MDD 이점 (vs 정적)"] = out["층 MDD"] - out["정적 MDD"]
    out["CAGR 손실 (vs 정적)"] = out["정적 CAGR"] - out["층 CAGR"]
    out["X3"] = [layer_verdict(a, b, c, d) for a, b, c, d in
                 zip(out["층 CAGR"], out["층 MDD"], out["평균 노출 CAGR"], out["평균 노출 MDD"])]
    return out.reset_index()


__all__ = ["VOL", "CD", "COMMON_START", "STATIC_OF", "VOL_NAME", "MEAN_NAME", "ORDER", "RECOMMENDED", "STAGE5_REF",
           "vol_levels_of", "vol_targets", "mean_exposure", "mean_exposure_weights", "zone", "layer_verdict",
           "table_rows", "layer_rows"]
