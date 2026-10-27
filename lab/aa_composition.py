"""Is the B2–B5 composition itself good? (allocation DECISIONS A39; design https://claude.ai/artifact/7AyPtvA6372TwXaSJFYxPq)

The four search allocations all hold nasdaq-100 (19–33 %) and differ in everything else. A31 picked them
out of tens of thousands with similar records, so the question is whether the rest is a real choice or
an interchangeable accident. The pieces here build the weight sets and read them with the rules fixed
in the design before the run:

  ① ``leave_one_out``  one product out; its weight goes pro rata to the rest ("pro") or to CD금리 ("cd")
  ② ``swap_rest``      the row allocation's nasdaq weight + the column allocation's other products
  ③ ``b_average``, ``skeleton_grid``  the mean of the four, and nasdaq-100 × KRX gold with CD금리 as the rest
  ④ ``reselect``       the best allocations chosen on one half of 2005–2026, judged on the other half
  ⑤ ``model_flags``    the A33 real-vs-model table of each product, read against fixed thresholds

Weights are whole percents over ``aa_search_results.SYMBOLS`` (the A31 universe) so ``StaticSim``,
``describe`` and ``code_text`` work as in stages 1–2. The script is ``scripts/aa_composition_study.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab import aa_local as loc
from lab import aa_search_results as sr

NAMES = ("B2", "B3", "B4", "B5")
NASDAQ, CD, KRX_GOLD = loc.NASDAQ, loc.CD, "411060"
COL = loc.COL

# B2–B4 as A37 ran them; B5 = the stage-2 row "B 레버·가상 없음 (-11,-10]" (2026-10-26_stage2_real/stage2.csv)
CENTRES: dict[str, dict[str, int]] = {
    **loc.CENTRES,
    "B5": {NASDAQ: 32, loc.KODEX200: 4, "132030": 17, CD: 35, loc.USD: 12},
}

# role of each product in the composition tables (the user's sleeves: N nasdaq, D defensive equity, S safe)
DEFENSIVE = loc.SECTORS  # 유틸리티, 필수소비재, 헬스케어
ROLES = ("나스닥", "방어 주식", "다른 주식", "채권", "금", "현금·달러")


def role_of(code: str) -> str:
    if code == NASDAQ:
        return "나스닥"
    if code in DEFENSIVE:
        return "방어 주식"
    cls = sr.CLASS_OF[code]
    if cls in sr.EQUITY_CLASSES or cls == sr.LEV_CLASS:
        return "다른 주식" if cls != sr.LEV_CLASS else "레버리지"
    if code in loc.GOLD:
        return "금"
    if cls == "금·실물":
        return "리츠"
    if cls == "현금·달러":
        return "현금·달러"
    return "채권"


def vector(weights: dict[str, int]) -> np.ndarray:
    return sr.to_vector(weights)


def whole(w: np.ndarray) -> np.ndarray:
    """Whole percents summing to 100 by largest remainder; ties go to the larger raw weight, then column
    order. Small parts are kept (unlike ``aa_search.round_weights``, which drops parts under 3 %)."""
    w = np.asarray(w, dtype=float).copy()
    w[~np.isfinite(w) | (w < 0)] = 0.0
    if w.sum() <= 0:
        raise ValueError("비중의 합이 0입니다.")
    w = np.round(w / w.sum() * 100.0, 9)  # 28.5 must stay 28.5, not 28.499999999999996, or ties break wrongly
    floors = np.floor(w).astype(np.int64)
    rest = int(100 - floors.sum())
    if rest > 0:
        order = np.lexsort((np.arange(len(w)), -w, -np.round(w - floors, 9)))  # last key first: remainder, weight, column
        floors[order[:rest]] += 1
    return floors


# --------------------------------------------------------------------------- ① leave one out

LOO_MODES = {"pro": "나머지에 비례 배분", "cd": "CD금리로"}


def leave_one_out(w: np.ndarray, mode: str) -> list[tuple[str, np.ndarray]]:
    """One row per held product: the allocation without it. ``pro`` rescales the rest to 100 %; ``cd`` adds
    the weight to CD금리 (skipped for CD금리 itself, which would be the same allocation)."""
    if mode not in LOO_MODES:
        raise ValueError(mode)
    w = np.asarray(w, dtype=float)
    out = []
    for i in np.flatnonzero(w > 0):
        code = sr.SYMBOLS[i]
        v = w.copy()
        v[i] = 0.0
        if mode == "cd":
            if code == CD:
                continue
            v[COL[CD]] += w[i]
        if v.sum() <= 0:
            continue
        out.append((code, whole(v)))
    return out


def loo_verdict(d_cagr: tuple[float, float], d_mdd: tuple[float, float],
                cagr_tol: float = 0.003, mdd_tol: float = 0.005) -> str:
    """Change when the product is taken out (without − with), per half (2005–2015, 2016–2026), as fractions.
    Fixed before the run: 제 몫 = in both halves MDD at least 0.5 %p deeper or CAGR at least 0.3 %p lower;
    있으나 마나 = both halves move less than that either way; 빼면 나아짐 = no half gets worse by the
    thresholds and at least one gets better by them; anything else 기간 따라 다름."""
    worse = [dm <= -mdd_tol or dc <= -cagr_tol for dc, dm in zip(d_cagr, d_mdd)]
    better = [dm >= mdd_tol or dc >= cagr_tol for dc, dm in zip(d_cagr, d_mdd)]
    if all(worse):
        return "제 몫"
    if not any(worse) and not any(better):
        return "있으나 마나"
    if not any(worse) and any(better):
        return "빼면 나아짐"
    return "기간 따라 다름"


# --------------------------------------------------------------------------- ② swap the rest

def swap_rest(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """``row``'s nasdaq weight with ``col``'s other products in their own ratios, filling 100 − nasdaq."""
    row, col = np.asarray(row, dtype=float), np.asarray(col, dtype=float)
    n = row[COL[NASDAQ]]
    rest = col.copy()
    rest[COL[NASDAQ]] = 0.0
    if rest.sum() <= 0:
        raise ValueError("나스닥 밖 상품이 없습니다.")
    v = rest / rest.sum() * (100.0 - n)
    v[COL[NASDAQ]] = n
    return whole(v)


def swap_matrix(centres: dict[str, dict[str, int]] | None = None) -> list[tuple[str, str, np.ndarray]]:
    centres = CENTRES if centres is None else centres
    vec = {k: vector(v) for k, v in centres.items()}
    return [(r, c, vec[r] if r == c else swap_rest(vec[r], vec[c])) for r in vec for c in vec]


def swap_verdict(table: pd.DataFrame, cagr_tol: float = 0.005, line: float = sr.MDD_LINE) -> dict:
    """``table`` rows (행, 열, cagr, mdd) on 2005–2026. 바꿔 써도 됨 = every off-diagonal cell inside the
    line and no more than 0.5 %p below its row's own allocation (the diagonal)."""
    diag = table[table["행"] == table["열"]].set_index("행")["cagr"]
    off = table[table["행"] != table["열"]].copy()
    off["선 안"] = off["mdd"] >= line
    off["원래보다"] = off["cagr"] - off["행"].map(diag)
    ok = bool(off["선 안"].all() and (off["원래보다"] >= -cagr_tol).all())
    return {"판정": "바꿔 써도 됨" if ok else "나머지에 의미 있음", "선 밖": int((~off["선 안"]).sum()),
            "0.5%p 넘게 낮음": int((off["원래보다"] < -cagr_tol).sum()), "바꾼 수": int(len(off))}


# --------------------------------------------------------------------------- ③ simpler allocations

SKELETON_NASDAQ = (20, 25, 30, 35)
SKELETON_GOLD = (10, 15, 20, 25, 30, 35)


def b_average(centres: dict[str, dict[str, int]] | None = None) -> np.ndarray:
    centres = CENTRES if centres is None else centres
    return whole(np.mean([vector(v) for v in centres.values()], axis=0))


def skeleton_grid() -> list[tuple[str, np.ndarray]]:
    out = []
    for n in SKELETON_NASDAQ:
        for g in SKELETON_GOLD:
            out.append((f"나스닥100 {n} · KRX금 {g} · CD금리 {100 - n - g}", vector({NASDAQ: n, KRX_GOLD: g, CD: 100 - n - g})))
    return out


def simple_verdict(cagr: float, mdd: float, centre_cagrs, line: float = sr.MDD_LINE) -> str:
    """같은 일을 한다 = inside the line on 2005–2026 and CAGR at least the lowest of B2–B5."""
    return "같은 일을 한다" if mdd >= line and cagr >= min(centre_cagrs) - 1e-12 else "못 미침"


# --------------------------------------------------------------------------- ④ choose on one half

RESELECT_TOP = 100


def subset_b(W: np.ndarray) -> np.ndarray:
    """The stage-1 "B" subset the four came from: no leveraged and no synthetic products."""
    bad = [COL[s] for s in sr.SYMBOLS if sr.CLASS_OF[s] == sr.LEV_CLASS or s in sr.SYNTHETIC]
    return W[:, bad].sum(axis=1) == 0


def reselect(df: pd.DataFrame, train: str, test: str, top: int = RESELECT_TOP, line: float = sr.MDD_LINE) -> np.ndarray:
    """Row positions of the ``top`` allocations by CAGR on half ``train`` ("h1" or "h2") among those inside
    the line on that half, best first. ``test`` is only checked to exist (the caller reads it)."""
    for h in (train, test):
        if f"cagr_{h}" not in df or f"mdd_{h}" not in df:
            raise KeyError(h)
    ok = np.flatnonzero(df[f"mdd_{train}"].to_numpy() >= line)
    order = np.argsort(-df[f"cagr_{train}"].to_numpy()[ok], kind="stable")
    return ok[order[:top]]


def role_shares(W: np.ndarray) -> pd.DataFrame:
    W = np.atleast_2d(np.asarray(W))
    roles = [role_of(s) for s in sr.SYMBOLS]
    out = {r: W[:, [i for i, x in enumerate(roles) if x == r]].sum(axis=1) for r in dict.fromkeys(roles)}
    for r in ROLES:
        out.setdefault(r, np.zeros(len(W), dtype=W.dtype))
    return pd.DataFrame(out)[[*ROLES, *[r for r in out if r not in ROLES]]]


RESELECT_RANGES = {"나스닥": (19, 33), "금": (12, 35)}  # the span of B2–B5


def reselect_verdict(shares: pd.DataFrame) -> str:
    """다시 뽑힘 = the median nasdaq and gold shares of the chosen allocations fall in the B2–B5 span."""
    med = shares.median()
    ok = all(lo <= med[r] <= hi for r, (lo, hi) in RESELECT_RANGES.items())
    return "다시 뽑힘" if ok else "다른 구성이 뽑힘"


# --------------------------------------------------------------------------- ⑤ model vs real prices

def model_flags(products: pd.DataFrame, codes, gap_pp: float = 1.0, corr_min: float = 0.95,
                min_years: float = 1.0) -> pd.DataFrame:
    """Read the A33 table (``2026-10-26_real_prices/products.csv``: real vs model after listing) for the
    given product codes. Fixed before the run: 모형 경고 = yearly return gap ≥ 1 %p or monthly correlation
    < 0.95; 비교 불가 = under a year of overlap. ``gap_se`` (tracking error / √years) is shown beside it so a
    short overlap can be read as noise; it does not change the label."""
    p = products[products["kind"] == "product"].set_index("real")
    rows = []
    for code in codes:
        if code not in p.index:
            rows.append({"코드": code, "상품": sr.name_of(code), "판정": "자료 없음"})
            continue
        r = p.loc[code]
        years = float(r["years"])
        label = ("비교 불가" if years < min_years else
                 "모형 경고" if abs(float(r["diff_pp"])) >= gap_pp or float(r["corr_monthly"]) < corr_min else "괜찮음")
        rows.append({"코드": code, "상품": sr.name_of(code), "상장": r["start"], "겹친 해": years,
                     "실제 CAGR": float(r["cagr_real"]), "모형 CAGR": float(r["cagr_model"]), "차이 %p": float(r["diff_pp"]),
                     "차이 표준오차 %p": float(r["te_ann"]) / np.sqrt(years), "월 상관": float(r["corr_monthly"]), "판정": label})
    return pd.DataFrame(rows)


def products_of(centres: dict[str, dict[str, int]] | None = None) -> list[str]:
    centres = CENTRES if centres is None else centres
    return [s for s in sr.SYMBOLS if any(s in v for v in centres.values())]


__all__ = ["NAMES", "CENTRES", "ROLES", "role_of", "whole", "leave_one_out", "loo_verdict", "swap_rest", "swap_matrix",
           "swap_verdict", "b_average", "skeleton_grid", "simple_verdict", "subset_b", "reselect", "role_shares",
           "reselect_verdict", "model_flags", "products_of"]
