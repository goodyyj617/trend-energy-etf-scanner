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

    g["r63"] = c / c.shift(63) - 1
    g["r126"] = c / c.shift(126) - 1

    abs_path_63 = c.diff().abs().rolling(63).sum()
    abs_path_126 = c.diff().abs().rolling(126).sum()
    g["er63"] = _safe_div((c - c.shift(63)).clip(lower=0), abs_path_63)
    g["er126"] = _safe_div((c - c.shift(126)).clip(lower=0), abs_path_126)

    g["te63"] = g["r63"] * g["er63"]
    g["te126"] = g["r126"] * g["er126"]
    g["score"] = 0.65 * g["te63"] + 0.35 * g["te126"]

    sma_te63_20 = g["te63"].rolling(20).mean()
    g["surge_ratio"] = _safe_div(g["te63"], sma_te63_20)

    g["ma50"] = c.rolling(50).mean()
    g["ma150"] = c.rolling(150).mean()
    g["hhv20"] = h.rolling(20).max()
    g["hhv126"] = h.rolling(126).max()
    g["hhv126_ratio"] = _safe_div(c, g["hhv126"])

    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    g["atr20"] = tr.rolling(20).mean()
    g["atr20_pct"] = _safe_div(g["atr20"], c)

    g["low10"] = l.rolling(10).min()
    g["low20"] = l.rolling(20).min()
    g["suggested_stop"] = g["low20"]
    g["stop_distance_pct"] = _safe_div(g["suggested_stop"], c) - 1

    return g


def add_signal_surge_v0(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the scanner's current TRUE/FALSE signal column.

    Backtests should call this same helper so that signal entry logic stays aligned
    with the live scanner when the scanner condition is edited later.
    """
    out = df.copy()
    out["signal_surge_v0"] = (
        out["eligible_universe"].fillna(False)
        & (out["te63"] > out["te126"])
        & (out["te63"] > 0)
        & (out["surge_ratio"] > 1.25)
        & (out["close"] > out["ma50"])
        & (out["ma50"] > out["ma150"])
        & (out["r63"] > 0.03)
        & (out["r126"] > 0)
        & (out["er63"] > 0.20)
        & (out["atr20_pct"] < 0.06)
        & (out["hhv126_ratio"] > 0.80)
    )
    return out


def compute_latest_features(prices: pd.DataFrame, universe: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    frames = []
    for symbol, g in prices.groupby("symbol", sort=False):
        out = compute_symbol_features(g.copy())
        out["symbol"] = symbol
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    features = pd.concat(frames, ignore_index=True)
    latest = features.sort_values("date").groupby("symbol", as_index=False).tail(1).copy()

    latest = latest.merge(universe, on="symbol", how="left")
    latest["dollar_vol_rank"] = latest["avg_dollar_vol_63"].rank(ascending=False, method="first")

    latest["liquidity_eligible"] = (
        (latest["history_days"] >= int(cfg["min_history_days"]))
        & (latest["close"] >= float(cfg["min_close"]))
        & (latest["avg_dollar_vol_20"] >= float(cfg["min_avg_dollar_vol_20"]))
        & (latest["avg_dollar_vol_63"] >= float(cfg["min_avg_dollar_vol_63"]))
        & (latest["dollar_vol_rank"] <= int(cfg["dollar_volume_top_n"]))
    )
    latest["eligible_universe"] = latest["base_universe_eligible"].fillna(False) & latest["liquidity_eligible"].fillna(False)

    latest = add_signal_surge_v0(latest)
    latest["rank_score"] = latest["score"].rank(ascending=False, method="first")
    latest = latest.sort_values(["signal_surge_v0", "score"], ascending=[False, False]).reset_index(drop=True)
    return latest
