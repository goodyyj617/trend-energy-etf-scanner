"""ETF universe selection as a transparent filter funnel.

Steps (each removal is recorded with its reason so the UI can show the funnel):
  1. cash-like ETFs (T-bill / ultrashort) are never traded: they are the cash vehicle
  2. asset classes chosen by the user
  3. minimum fund size (AUM)
  4. maximum expense ratio
  5. near-duplicates: when two ETFs' daily returns correlate >= DEDUP_CORR, keep the larger
     fund (they track practically the same thing; holding both doubles one bet)

Point-in-time liquidity (average dollar volume on each signal day) is applied later by
the engine, not here. AUM and expense ratio are today's values (see limitations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import etf_meta

DEDUP_CORR = 0.98
CASH_CLASS = "현금성"
CLASSES = ["미국 주식", "섹터·테마 주식", "해외 주식", "채권", "원자재", "부동산", "기타"]
DEFAULT_CLASSES = ["미국 주식", "섹터·테마 주식", "해외 주식", "원자재", "부동산"]
CLASS_HELP = {
    "미국 주식": "미국 시장 전체·대형/중형/소형·스타일(성장/가치) ETF",
    "섹터·테마 주식": "업종(기술, 에너지, 헬스케어…)·테마·금광주 등",
    "해외 주식": "선진국·신흥국·국가별 주식 ETF",
    "채권": "만기 1년 이상 국채·회사채·하이일드·물가연동채 등. 변동성이 낮아 같은 금액 슬롯을 쓰면 수익 기여가 작고, "
            "대신 주식 급락기에 오르는 경우가 있어 분산 효과가 있습니다. 연구 질문에 따라 넣거나 빼서 비교하세요.",
    "원자재": "금·은·원자재 지수 ETF",
    "부동산": "리츠(REIT)·부동산 ETF",
    "기타": "가상자산, 자산배분·전술형, 옵션/파생 전략, 분류 불명",
    CASH_CLASS: "만기 1년 미만 단기국채·초단기채. 매매 대상이 아니라 현금 대용(현금 수익률)으로만 씁니다.",
}

_FALLBACK_GROUP = {
    "US Equity": "미국 주식", "Factor / Style": "미국 주식", "Sector": "섹터·테마 주식",
    "Industry / Theme": "섹터·테마 주식", "Country / Region": "해외 주식", "Global Equity": "해외 주식",
    "Bond": "채권", "Commodity": "원자재", "REIT / Infra": "부동산", "Crypto": "기타", "Other": "기타",
}


def asset_class(category: str, fallback_group: str = "") -> str:
    """Map a Morningstar category (from yfinance) to one of the lab's asset classes."""
    c = (category or "").lower()
    if not c:
        return _FALLBACK_GROUP.get(fallback_group or "", "기타")
    if "ultrashort" in c or "money market" in c:
        return CASH_CLASS
    if any(w in c for w in ("digital", "trading", "allocation", "tactical", "derivative", "options", "long-short",
                             "macro", "systematic", "multistrategy", "event driven", "relative value", "preferred")):
        return "기타"
    if "real estate" in c:
        return "부동산"
    if "equity precious metals" in c or "natural resources" in c:
        return "섹터·테마 주식"
    if "commodit" in c or "precious metals" in c:
        return "원자재"
    if any(w in c for w in ("bond", "government", "treasury", "corporate", "high yield", "inflation", "muni",
                             "bank loan", "securitized", "multisector", "nontraditional", "intermediate core",
                             "target maturity")):
        return "채권"
    if any(w in c for w in ("foreign", "emerging", "europe", "japan", "china", "india", "latin", "pacific", "asia",
                             "world", "global", "region", "diversified emerging")):
        return "해외 주식"
    if any(w in c for w in ("large", "mid-cap", "small", "mid cap")):
        return "미국 주식"
    if any(w in c for w in ("technology", "financial", "health", "energy", "industrials", "consumer", "utilities",
                             "communications", "infrastructure", "miscellaneous sector", "equity")):
        return "섹터·테마 주식"
    return "기타"


@dataclass(frozen=True)
class UniverseFilter:
    classes: tuple[str, ...] = tuple(DEFAULT_CLASSES)
    min_aum_musd: float = 500.0  # $ million
    max_expense_pct: float = 0.75  # percent per year; unknown values pass
    dedup: bool = True


def candidate_table(symbols: list[str], fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    """symbol, name, category, asset_class, aum, expense_ratio for the given symbols."""
    meta = etf_meta.load().set_index("symbol")
    fb = fallback.set_index("symbol") if fallback is not None and len(fallback) else pd.DataFrame()
    rows = []
    for s in symbols:
        m = meta.loc[s] if s in meta.index else None
        f = fb.loc[s] if s in fb.index else None
        category = (m["category"] if m is not None and isinstance(m["category"], str) else "") or ""
        aum = m["aum"] if m is not None and pd.notna(m["aum"]) else (f["aum"] if f is not None and "aum" in f else np.nan)
        rows.append({
            "symbol": s,
            "name": (m["name"] if m is not None and isinstance(m["name"], str) else None)
                    or (f["name"] if f is not None and "name" in f else ""),
            "category": category,
            "asset_class": asset_class(category, f["asset_group"] if f is not None and "asset_group" in f else ""),
            "aum": float(aum) if pd.notna(aum) else np.nan,
            "expense_ratio": float(m["expense_ratio"]) if m is not None and pd.notna(m["expense_ratio"]) else np.nan,
        })
    return pd.DataFrame(rows)


def apply_filter(table: pd.DataFrame, flt: UniverseFilter, close: pd.DataFrame | None = None,
                 corr_end: str | None = None) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Returns (table with a 'status' column, funnel steps [(label, count remaining)])."""
    t = table.copy()
    t["status"] = "선택"
    funnel = [("후보 전체", len(t))]

    def drop(mask: pd.Series, reason: str, label: str) -> None:
        t.loc[mask & (t["status"] == "선택"), "status"] = reason
        funnel.append((label, int((t["status"] == "선택").sum())))

    drop(t["asset_class"] == CASH_CLASS, "현금성 (현금 대용으로만 사용)", "현금성 ETF 제외")
    drop(~t["asset_class"].isin(flt.classes), "선택하지 않은 자산군", "자산군 선택")
    drop(t["aum"].isna() | (t["aum"] < flt.min_aum_musd * 1e6), f"운용 규모 < ${flt.min_aum_musd:,.0f}M",
         f"운용 규모 ≥ ${flt.min_aum_musd:,.0f}M")
    drop(t["expense_ratio"] > flt.max_expense_pct, f"총보수 > {flt.max_expense_pct:.2f}%",
         f"총보수 ≤ {flt.max_expense_pct:.2f}%")
    if flt.dedup and close is not None:
        keep = t.loc[t["status"] == "선택"].sort_values("aum", ascending=False)["symbol"].tolist()
        dup_of = near_duplicates(close, keep, corr_end)
        for s, kept in dup_of.items():
            t.loc[t["symbol"] == s, "status"] = f"{kept}와 사실상 동일 (상관 ≥ {DEDUP_CORR})"
        funnel.append(("중복 ETF 정리", int((t["status"] == "선택").sum())))
    return t, funnel


def near_duplicates(close: pd.DataFrame, ordered: list[str], end: str | None = None,
                    threshold: float = DEDUP_CORR, min_overlap: int = 250) -> dict[str, str]:
    """Greedy: walk symbols in priority order (largest AUM first); drop any symbol whose
    daily returns correlate >= threshold with an already kept one. Uses data before `end`."""
    data = close[[s for s in ordered if s in close.columns]]
    if end is not None:
        data = data[data.index < pd.Timestamp(end)]
    rets = data.pct_change(fill_method=None)
    corr = rets.corr(min_periods=min_overlap)
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for s in ordered:
        if s not in corr.columns:
            kept.append(s)
            continue
        match = next((k for k in kept if k in corr.columns and corr.at[s, k] >= threshold), None)
        if match:
            dropped[s] = match
        else:
            kept.append(s)
    return dropped
