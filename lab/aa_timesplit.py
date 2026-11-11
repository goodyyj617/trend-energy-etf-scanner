"""A43: does the rule that picks a dynamic setting hold out of time? (allocation DECISIONS A43, design
`docs/research/allocation/2026-11-06_a43_design.md`, fixed before the run).

Stage 4 of A42 judged 873 settings × 6 targets × 3 satellite shares on the whole research window. Here the same
paths are cut into windows: the stage-4 verdict (①②③ + ② on the extreme sets) is applied inside a *selection*
window, and the settings it picks are measured inside the *test* window. Pure functions on arrays and frames; the
data context and the jobs live in `scripts/aa_a43_timesplit.py`.

  ``WINDOWS`` / ``DIRECTIONS``   the fixed windows and the four select → test directions
  ``window_rows``                 rows of a date index inside a window
  ``window_metrics``              CAGR / MDD of paths re-based at the window start
  ``rank_value``                  10th percentile of CAGR from month-end starts with ≥ RANK_MIN_YEARS left
  ``equal_cost``                  x on the equal-cost grid matching a CAGR, and the MDD there (stage 4's)
  ``verdicts``                    the stage-4 verdict per window from the runs / placebo frames
  ``transfer``                    rule A: selection-window passers measured in the test window + hypergeometric p
  ``best_pick``                   rule B: the passer with the best selection-window rank value, in the test window
  ``judgements``                  J1 / J2 / J3 of the design
"""
from __future__ import annotations

from math import comb, log, exp
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aa_engine import TRADING_DAYS, decision_rows

RANK_MIN_YEARS = 5
MDD_GAIN_EQ = 0.03
CAGR_LOSS = 0.01
PLACEBO_Q = 0.95
EQ_GRID = 41

WINDOWS: dict[str, tuple[str, str]] = {
    "전체": ("2000-09-01", "2024-12-31"),
    "전반": ("2000-09-01", "2012-12-31"),
    "후반": ("2013-01-01", "2024-12-31"),
    "1구간": ("2000-09-01", "2008-12-31"),
    "2구간": ("2009-01-01", "2016-12-31"),
    "3구간": ("2017-01-01", "2024-12-31"),
    "1+2구간": ("2000-09-01", "2016-12-31"),
    "2+3구간": ("2009-01-01", "2024-12-31"),
}
# (name, selection window, test window)
DIRECTIONS: tuple[tuple[str, str, str], ...] = (
    ("㉮ 전반 → 후반", "전반", "후반"),
    ("㉯ 후반 → 전반", "후반", "전반"),
    ("㉰ 1+2 → 3구간", "1+2구간", "3구간"),
    ("㉱ 2+3 → 1구간", "2+3구간", "1구간"),
)
SETTING_KEY = ["signal", "action", "confirm", "tranches", "vol", "cooldown"]
KEY = ["대상"] + SETTING_KEY + ["위성"]
CHECKS = ["① 가짜 95%보다 개선", "② 같은 비용보다 3%p", "③ CAGR 손실 ≤ 1%p", "극단 두 세트 ② 유지"]
REP3 = {"대상": "F4 (가)", "signal": "k2", "action": "safe", "confirm": 0, "tranches": 1, "vol": 0.08, "cooldown": 21, "위성": 0.5}


# --------------------------------------------------------------------------- windows and metrics

def window_rows(dates: pd.DatetimeIndex, window: str) -> np.ndarray:
    a, b = WINDOWS[window]
    return np.flatnonzero((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def window_metrics(P: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(CAGR, MDD) of each path over the rows, re-based at the first row (empty rows → NaN)."""
    P = np.atleast_2d(P)
    if len(rows) < 2:
        n = len(P)
        return np.full(n, np.nan), np.full(n, np.nan)
    S = P[:, rows]
    S = S / S[:, :1]
    cagr = S[:, -1] ** (TRADING_DAYS / (S.shape[1] - 1)) - 1.0
    mdd = (S / np.maximum.accumulate(S, axis=1) - 1.0).min(axis=1)
    return cagr, mdd


def rank_value(P: np.ndarray, dates: pd.DatetimeIndex, rows: np.ndarray, min_years: float = RANK_MIN_YEARS) -> np.ndarray:
    """10th percentile of CAGR from each month-end start inside the window that leaves ≥ min_years to the end."""
    P = np.atleast_2d(P)
    if len(rows) < 2:
        return np.full(len(P), np.nan)
    sub = dates[rows]
    me = decision_rows(sub, "M")
    last = sub[-1] - pd.DateOffset(years=int(min_years))
    me = me[sub[me] <= last]
    me = np.concatenate([[0], me]) if (len(me) == 0 or me[0] != 0) else me
    S = P[:, rows]
    T = len(rows) - 1
    c = (S[:, -1:] / S[:, me]) ** (TRADING_DAYS / np.maximum(T - me, 1)) - 1.0
    return np.quantile(c, 0.10, axis=1)


def equal_cost(gc: np.ndarray, gm: np.ndarray, target: float) -> tuple[float, float]:
    """x on the grid (0 … 1) whose CAGR matches `target` (linear between grid points) and the MDD there (stage 4)."""
    xs = np.linspace(0, 1, len(gc))
    if not np.isfinite(target) or not np.all(np.isfinite(gc)):
        return float("nan"), float("nan")
    if target >= gc[0]:
        return 0.0, float(gm[0])
    if target <= gc.min():
        k = int(np.argmin(gc))
        return float(xs[k]), float(gm[k])
    k = int(np.flatnonzero(gc <= target)[0])
    f = (gc[k - 1] - target) / (gc[k - 1] - gc[k])
    return float(xs[k - 1] + f * (xs[k] - xs[k - 1])), float(gm[k - 1] + f * (gm[k] - gm[k - 1]))


# --------------------------------------------------------------------------- verdict per window

def verdicts(R: pd.DataFrame, PL: pd.DataFrame) -> pd.DataFrame:
    """Stage-4 verdict inside every window. R: rows 종류 ∈ {정적, 설정} × 세트 × 대상 × 구간 × 위성 with cagr, mdd,
    같은 비용 mdd (settings), 순위값 (representative set). PL: 대상 × setting × 위성 × 구간 with 가짜 mdd q95."""
    st = R[R["종류"] == "정적"]
    plain = st[st["위성"] == 0.0].set_index(["세트", "대상", "구간"])
    s = R[R["종류"] == "설정"].copy()
    idx = pd.MultiIndex.from_frame(s[["세트", "대상", "구간"]])
    s["정적 cagr"] = plain["cagr"].reindex(idx).to_numpy()
    s["정적 mdd"] = plain["mdd"].reindex(idx).to_numpy()
    s["같은 비용 대비 MDD 개선"] = s["mdd"] - s["같은 비용 mdd"]
    rep = s[s["세트"] == "대표"].copy()
    keys = KEY + ["구간"]
    rep = rep.merge(PL[keys + ["가짜 mdd q95", "가짜 mdd 중앙"]], on=keys, how="left")
    rep["MDD 개선"] = rep["mdd"] - rep["정적 mdd"]
    rep["CAGR 손실"] = rep["정적 cagr"] - rep["cagr"]
    rep["① 가짜 95%보다 개선"] = rep["mdd"] > rep["가짜 mdd q95"]  # NaN (no signal) → False
    rep["② 같은 비용보다 3%p"] = rep["같은 비용 대비 MDD 개선"] >= MDD_GAIN_EQ
    rep["③ CAGR 손실 ≤ 1%p"] = rep["CAGR 손실"] <= CAGR_LOSS
    ridx = pd.MultiIndex.from_frame(rep[keys])
    for ex in ("극단 1", "극단 2"):
        e = s[s["세트"] == ex].set_index(keys)["같은 비용 대비 MDD 개선"]
        rep[f"② {ex}"] = e.reindex(ridx).to_numpy() >= MDD_GAIN_EQ
    rep["극단 두 세트 ② 유지"] = rep["② 극단 1"] & rep["② 극단 2"]
    rep["판정"] = np.where(rep[CHECKS].all(axis=1), "통과", "탈락")
    rep["①③"] = rep["① 가짜 95%보다 개선"] & rep["③ CAGR 손실 ≤ 1%p"]
    return rep


# --------------------------------------------------------------------------- selection → test

def hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """P[X ≥ k] for X ~ Hypergeometric(N population, K successes, n draws), in logs (no scipy)."""
    if n <= 0 or k <= 0:
        return 1.0
    lo, hi = max(k, 0), min(n, K)
    if lo > hi:
        return 0.0
    ltot = log(comb(N, n))
    return float(min(1.0, sum(exp(log(comb(K, j)) + log(comb(N - K, n - j)) - ltot) for j in range(lo, hi + 1))))


def _by(V: pd.DataFrame, window: str) -> pd.DataFrame:
    return V[V["구간"] == window].set_index(KEY)


def transfer(V: pd.DataFrame, sel: str, test: str) -> pd.DataFrame:
    """Rule A per target and overall: passers of the selection window measured in the test window."""
    a, b = _by(V, sel), _by(V, test)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    rows = []
    groups = [("전체", slice(None))] + [(t, a.index.get_level_values("대상") == t) for t in a.index.get_level_values("대상").unique()]
    for name, m in groups:
        aa, bb = a[m] if name != "전체" else a, b[m] if name != "전체" else b
        passed = aa["판정"] == "통과"
        N, n = len(aa), int(passed.sum())
        Kp = int((bb["판정"] == "통과").sum())
        k = int((bb.loc[passed.to_numpy(), "판정"] == "통과").sum())
        base = Kp / N if N else float("nan")
        keep = k / n if n else float("nan")
        rows.append({"방향": f"{sel} → {test}", "대상": name, "설정 수": N, "고르는 구간 통과 n": n, "시험 구간 유지 k": k,
                     "유지율": keep, "바탕 통과율": base, "비": (keep / base) if (n and base) else float("nan"),
                     "p값 (초기하)": hypergeom_sf(k, N, Kp, n) if n else float("nan"),
                     "통과 집합 시험 MDD 개선 중앙": float(bb.loc[passed.to_numpy(), "MDD 개선"].median()) if n else float("nan"),
                     "나머지 시험 MDD 개선 중앙": float(bb.loc[~passed.to_numpy(), "MDD 개선"].median()) if N > n else float("nan"),
                     "통과 집합 시험 CAGR 손실 중앙": float(bb.loc[passed.to_numpy(), "CAGR 손실"].median()) if n else float("nan"),
                     "나머지 시험 CAGR 손실 중앙": float(bb.loc[~passed.to_numpy(), "CAGR 손실"].median()) if N > n else float("nan"),
                     "①③ 유지율": float((bb.loc[passed.to_numpy(), "①③"]).mean()) if n else float("nan"),
                     "①③ 바탕": float(bb["①③"].mean()) if N else float("nan")})
    return pd.DataFrame(rows)


def best_pick(V: pd.DataFrame, sel: str, test: str) -> pd.DataFrame:
    """Rule B per target: among selection-window passers the highest selection-window rank value, in the test window."""
    a, b = _by(V, sel), _by(V, test)
    rows = []
    for t in a.index.get_level_values("대상").unique():
        at = a[a.index.get_level_values("대상") == t]
        bt = b[b.index.get_level_values("대상") == t]
        p = at[at["판정"] == "통과"]
        row = {"방향": f"{sel} → {test}", "대상": t, "고르는 구간 통과 n": int(len(p))}
        if len(p) == 0 or p["순위값"].isna().all():
            rows.append({**row, "선택": "없음"})
            continue
        key = p["순위값"].idxmax()
        if key not in bt.index:
            rows.append({**row, "선택": "시험 구간 없음"})
            continue
        s, u = p.loc[key], bt.loc[key]
        same = bt[bt.index.get_level_values("위성") == key[-1]]
        pct = float((same["MDD 개선"] <= u["MDD 개선"]).mean())
        rows.append({**row, "선택": s["설정 이름"], **dict(zip(KEY, key)),
                     "고르는 구간 순위값": float(s["순위값"]), "고르는 구간 cagr": float(s["cagr"]), "고르는 구간 mdd": float(s["mdd"]),
                     "시험 cagr": float(u["cagr"]), "시험 mdd": float(u["mdd"]), "시험 정적 cagr": float(u["정적 cagr"]),
                     "시험 정적 mdd": float(u["정적 mdd"]), "시험 MDD 개선": float(u["MDD 개선"]), "시험 CAGR 손실": float(u["CAGR 손실"]),
                     "시험 가짜 q95": float(u["가짜 mdd q95"]), "시험 MDD 개선 백분위 (같은 위성 873)": pct,
                     "시험 ①": bool(u["① 가짜 95%보다 개선"]), "시험 ②": bool(u["② 같은 비용보다 3%p"]), "시험 ③": bool(u["③ CAGR 손실 ≤ 1%p"]),
                     "시험 극단 ②": bool(u["극단 두 세트 ② 유지"]), "시험 엄격 통과": u["판정"] == "통과", "시험 완화 ①③": bool(u["①③"])})
    return pd.DataFrame(rows)


def rep3_rows(V: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """Representative ③ and the no-signal references, per window (representative set)."""
    v = V.copy()
    m = np.ones(len(v), dtype=bool)
    for k, x in REP3.items():
        m &= (v[k] == x).to_numpy()
    r3 = v[m][["구간", "cagr", "mdd", "가짜 mdd q95", "같은 비용 mdd", "MDD 개선", "CAGR 손실"] + CHECKS + ["판정"]].assign(이름="③ (가) + K2 → 안전 묶음 · 쿨다운 21 · 변동성 8% · 위성 50%")
    refs = []
    ga = v[v["대상"] == "F4 (가)"]
    for name, q in (("(가) + 변동성 목표 8%", dict(signal="none", tranches=1, vol=0.08, 위성=0.0)),
                    ("(가) 50 + 위성 50 + 변동성 목표 8%", dict(signal="none", tranches=1, vol=0.08, 위성=0.5))):
        mm = np.ones(len(ga), dtype=bool)
        for k, x in q.items():
            mm &= (ga[k] == x).to_numpy()
        refs.append(ga[mm][["구간", "cagr", "mdd", "가짜 mdd q95", "같은 비용 mdd", "MDD 개선", "CAGR 손실"] + CHECKS + ["판정"]].assign(이름=name))
    st = R[(R["종류"] == "정적") & (R["세트"] == "대표") & (R["대상"] == "F4 (가)")]
    for name, sat in (("(가) 정적", 0.0), ("(가) 50 + 위성 50 (신호 없음)", 0.5)):
        refs.append(st[st["위성"] == sat][["구간", "cagr", "mdd"]].assign(이름=name))
    out = pd.concat([r3] + refs, ignore_index=True)
    order = {w: i for i, w in enumerate(WINDOWS)}
    return out.assign(_o=out["구간"].map(order)).sort_values(["이름", "_o"]).drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------------- judgements

def judgements(TR: pd.DataFrame, BP: pd.DataFrame, R3: pd.DataFrame) -> dict:
    t = TR[TR["대상"] == "전체"]
    ok = (t["비"] >= 2.0) & (t["p값 (초기하)"] < 0.05)
    fails = int((~ok.fillna(False)).sum())
    j1 = "있음" if fails == 0 else ("약함" if fails == 1 else "없음")
    b = BP[BP["대상"] == "F4 (가)"]
    strict = b["시험 엄격 통과"].map(lambda x: bool(x) if pd.notna(x) else False).astype(bool) if "시험 엄격 통과" in b else pd.Series(dtype=bool)
    loose = b["시험 완화 ①③"].map(lambda x: bool(x) if pd.notna(x) else False).astype(bool) if "시험 완화 ①③" in b else pd.Series(dtype=bool)
    picked = (b["선택"] != "없음") & (b["선택"] != "시험 구간 없음") if "선택" in b else pd.Series(dtype=bool)
    if len(b) == 0 or not picked.any():
        j2 = "선택 없음"
    elif bool(strict.all()) and bool(picked.all()):
        j2 = "버팀"
    elif bool(loose.all()) and bool(picked.all()):
        j2 = "약함"
    else:
        j2 = "무너짐"
    r3 = R3[R3["이름"].str.startswith("③")].set_index("구간")
    halves = {w: (r3.loc[w, "판정"] == "통과") if w in r3.index else None for w in ("전반", "후반")}
    if all(halves.values()):
        j3 = "양 구간 통과"
    else:
        bad = []
        for w in ("전반", "후반"):
            if w in r3.index and r3.loc[w, "판정"] != "통과":
                bad.append(w + ": " + ", ".join(c[:1] for c in CHECKS if not bool(r3.loc[w, c])))
        j3 = "탈락 (" + "; ".join(bad) + ")"
    return {"J1 선택 규칙 A": j1, "J1 방향별": {str(r["방향"]): bool(o) for r, o in zip(t.to_dict("records"), ok.fillna(False))},
            "J2 최선 하나 (가)": j2, "J2 방향별 엄격": {str(r["방향"]): bool(s) for r, s in zip(b.to_dict("records"), strict)},
            "J3 ③ 자체": j3}


__all__ = ["WINDOWS", "DIRECTIONS", "KEY", "SETTING_KEY", "CHECKS", "REP3", "window_rows", "window_metrics", "rank_value",
           "equal_cost", "verdicts", "hypergeom_sf", "transfer", "best_pick", "rep3_rows", "judgements"]
