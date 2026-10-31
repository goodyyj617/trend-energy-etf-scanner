"""A42 stage 1: factor exposures and fixed scenarios (allocation DECISIONS A42 §1).

Six monthly factors (2005-01 – 2024-12):
  EQ     S&P500 USD total return (SPY)                         fraction
  GROWTH nasdaq-100 minus S&P500 (QQQ − SPY)                    fraction
  UST    change of the US 10-year yield (FRED DGS10, month end) bp
  KTB    change of the Korean 10-year yield (ECOS 721Y001, monthly average) bp
  FX     change of KRW per USD (DEXKOUS, month end; + = weaker won)  fraction
  GOLD   gold USD return (GLD; GC=F before 2004-11)              fraction

Each product's monthly KRW return (a price set, default the representative real · h60 · CD91) is regressed on the six
factors with an intercept (OLS). An allocation's exposure is Σ weight × beta. A scenario loss is Σ exposure × shock
(residual 0): % shocks enter as fractions, rate shocks in bp. The generator is linear: leverage path dependence and
crisis-time correlation changes are not in it. The KTB factor is built from monthly averages, so it lags month-end
moves (a limit noted in the stage-0 memo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aa_korea import load_kr_rates, load_usdkrw

FACTORS = ("EQ", "GROWTH", "UST", "KTB", "FX", "GOLD")
PCT_FACTORS = ("EQ", "GROWTH", "FX", "GOLD")
UST_PATH = Path(__file__).with_name("aa_ust10_monthly.csv")
WINDOW = ("2005-01-01", "2024-12-31")
MIN_MONTHS = 36

# shocks: % for EQ / GROWTH / FX / GOLD, bp for UST / KTB (design §1, fixed before any run)
SCENARIOS = {
    "S1": ("2022형 동반 하락 + 원화 강세", {"EQ": -30, "GROWTH": -15, "UST": 200, "KTB": 150, "FX": -10, "GOLD": -10}),
    "S2": ("원화 초강세", {"EQ": 0, "GROWTH": 0, "UST": 0, "KTB": 0, "FX": -25, "GOLD": 0}),
    "S3": ("금 폭락", {"EQ": 0, "GROWTH": 0, "UST": 0, "KTB": 0, "FX": 0, "GOLD": -40}),
    "S4": ("금리 급등", {"EQ": -15, "GROWTH": -10, "UST": 300, "KTB": 200, "FX": 0, "GOLD": -10}),
    "S5": ("성장주 역전", {"EQ": -10, "GROWTH": -35, "UST": 0, "KTB": 0, "FX": 0, "GOLD": 0}),
    "S6": ("스태그플레이션", {"EQ": -25, "GROWTH": -10, "UST": 150, "KTB": 150, "FX": 10, "GOLD": 25}),
    "S7": ("닷컴형 성장주 붕괴", {"EQ": -40, "GROWTH": -40, "UST": -150, "KTB": -100, "FX": 15, "GOLD": 0}),
    "S8": ("2008형, 원화 완충 없음", {"EQ": -45, "GROWTH": 0, "UST": -150, "KTB": -100, "FX": 0, "GOLD": 5}),
}
LOSS_LINE = -0.25
MAX_BAD = 2


def shock_vector(shocks: dict) -> pd.Series:
    """Scenario shocks in model units: fractions for % factors, bp for rate factors."""
    return pd.Series({f: shocks[f] / 100.0 if f in PCT_FACTORS else float(shocks[f]) for f in FACTORS})


def monthly_returns(close: pd.DataFrame) -> pd.DataFrame:
    m = close.resample("ME").last()
    return m.pct_change(fill_method=None)


def factor_table(usd_close: pd.DataFrame, ust: pd.Series | None = None, kr_rates: pd.DataFrame | None = None,
                 fx: pd.Series | None = None) -> pd.DataFrame:
    """Monthly factor values (month-end index). usd_close: daily USD closes with SPY, QQQ and GLD."""
    r = monthly_returns(usd_close[["SPY", "QQQ", "GLD"]])
    ust = load_ust() if ust is None else ust
    kr = load_kr_rates() if kr_rates is None else kr_rates
    fx = load_usdkrw() if fx is None else fx
    ktb = kr["ktb10"].copy()
    ktb.index = ktb.index + pd.offsets.MonthEnd(0)  # monthly average labelled at its month's end
    fxm = fx.resample("ME").last()
    out = pd.DataFrame({"EQ": r["SPY"], "GROWTH": r["QQQ"] - r["SPY"], "UST": ust.diff() * 100.0,
                        "KTB": ktb.diff() * 100.0, "FX": fxm.pct_change(fill_method=None), "GOLD": r["GLD"]})
    return out


def load_ust(path: Path = UST_PATH) -> pd.Series:
    """US 10-year yield (%) at each month end, 2000-09 – (FRED DGS10)."""
    s = pd.read_csv(path)
    idx = pd.PeriodIndex(s["month"], freq="M").to_timestamp() + pd.offsets.MonthEnd(0)
    return pd.Series(s["dgs10"].to_numpy(float), index=idx)


def betas(product_monthly: pd.DataFrame, factors: pd.DataFrame, window=WINDOW, min_months: int = MIN_MONTHS) -> pd.DataFrame:
    """OLS of each product's monthly return on the six factors (intercept included). One row per product: alpha (a
    month), the six betas, R², months used. Products with fewer than min_months complete months get NaN betas."""
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    F = factors.loc[(factors.index >= lo) & (factors.index <= hi), list(FACTORS)]
    rows = []
    for code in product_monthly.columns:
        y = product_monthly[code].reindex(F.index)
        ok = y.notna() & F.notna().all(axis=1)
        n = int(ok.sum())
        row = {"상품": code, "개월": n}
        if n >= min_months:
            X = np.column_stack([np.ones(n), F[ok].to_numpy(float)])
            coef, *_ = np.linalg.lstsq(X, y[ok].to_numpy(float), rcond=None)
            fit = X @ coef
            resid = y[ok].to_numpy(float) - fit
            ss = ((y[ok] - y[ok].mean()) ** 2).sum()
            row.update({"alpha": coef[0], **{f: c for f, c in zip(FACTORS, coef[1:])}, "R2": float(1 - (resid @ resid) / ss) if ss > 0 else np.nan,
                        "잔차 변동성 (연)": float(resid.std(ddof=len(coef)) * np.sqrt(12))})
        else:
            row.update({"alpha": np.nan, **{f: np.nan for f in FACTORS}, "R2": np.nan, "잔차 변동성 (연)": np.nan})
        rows.append(row)
    return pd.DataFrame(rows).set_index("상품")


def exposure(weights: dict, beta: pd.DataFrame) -> pd.Series:
    """Σ weight × beta over the six factors (weights in percent or fractions; normalised to 1)."""
    w = pd.Series({k: float(v) for k, v in weights.items() if k != "CASH"})
    w = w / w.sum()
    missing = [c for c in w.index if c not in beta.index or beta.loc[c, list(FACTORS)].isna().any()]
    if missing:
        raise KeyError(f"베타가 없는 상품: {missing}")
    return (beta.loc[w.index, list(FACTORS)].mul(w, axis=0)).sum()


def scenario_losses(expo: pd.Series) -> pd.Series:
    """Linear loss (fraction) of an exposure vector in each scenario S1–S8."""
    return pd.Series({k: float((expo * shock_vector(v[1])).sum()) for k, v in SCENARIOS.items()})


def scenario_verdict(losses: pd.Series) -> bool:
    """Design §1: at most two of the eight scenarios worse than −25 %."""
    return int((losses < LOSS_LINE).sum()) <= MAX_BAD


def realized_moves(factors_daily: dict, window: tuple[str, str]) -> pd.Series:
    """Cumulative factor moves over a crisis window, in model units (fraction / bp).

    factors_daily: {"SPY", "QQQ", "GLD": daily USD close series, "UST": daily 10y yield %, "KTB": monthly 10y %, "FX": daily KRW per USD}."""
    a, b = pd.Timestamp(window[0]), pd.Timestamp(window[1])

    def ret(s):
        s = s.dropna()
        s = s[(s.index >= a - pd.Timedelta(days=7)) & (s.index <= b)]
        return float(s.iloc[-1] / s.asof(a) - 1)

    def chg(s):
        s = s.dropna()
        return float((s.asof(b) - s.asof(a)) * 100.0)

    eq, qq = ret(factors_daily["SPY"]), ret(factors_daily["QQQ"])
    return pd.Series({"EQ": eq, "GROWTH": qq - eq, "UST": chg(factors_daily["UST"]), "KTB": chg(factors_daily["KTB"]),
                      "FX": ret(factors_daily["FX"]), "GOLD": ret(factors_daily["GLD"])})


__all__ = ["FACTORS", "SCENARIOS", "LOSS_LINE", "MAX_BAD", "factor_table", "betas", "exposure", "scenario_losses",
           "scenario_verdict", "shock_vector", "realized_moves", "monthly_returns", "load_ust"]
