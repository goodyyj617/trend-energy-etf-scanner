from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def compute_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c = g["close"]
    h = g["high"]
    l = g["low"]
    v = g["volume"]

    g["history_days"] = np.arange(1, len(g) + 1)
    g["dollar_volume"] = c * v
    g["avg_dollar_vol_20"] = g["dollar_volume"].rolling(20).mean()
    g["avg_dollar_vol_63"] = g["dollar_volume"].rolling(63).mean()

    g["r20"] = c / c.shift(20