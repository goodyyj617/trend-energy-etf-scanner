from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .features import add_signal_surge_v0, compute_symbol_features

SIGNAL_COL = "signal_surge_v0"
ROUND_TRIP_COST = 0.002
MAX_HOLDING_DAYS = 63


@dataclass(frozen=True)
class ExitRule:
    key: str
    label: str
    description: str
    stop_fn: Optional[Callable[[pd.Series], float]] = None
    use_signal_false_exit: bool = True
    use_trailing_max: bool = True


def _num(row: pd.Series, key: str) -> float:
    value = row.get(key, np.nan)
    try:
        return float(value)
    except Exception:
        return np.nan


def _low10(row: pd.Series) -> float:
    return _num(row, "low10")


def _low20(row: pd.Series) -> float:
    return _num(row, "low20")


def _low20_minus_half_atr(row: pd.Series) -> float:
    return _num(row, "low20") - 0.5 * _num(row, "atr20")


def _chandelier20_25atr(row: pd.Series) -> float:
    return _num(row, "hhv20") - 2.5 * _num(row, "atr20")


def _ma50(row: pd.Series) -> float:
    return _num(row, "ma50")


EXIT_RULES: list[ExitRule] = [
    ExitRule(
        key="low10",
        label="Low10 trailing",
        description="Initial stop is signal-day Low10. While holding, stop only moves up when the latest Low10 rises.",
        stop_fn=_low10,
    ),
    ExitRule(
        key="low20",
        label="Low20 trailing",
        description="Initial stop is signal-day Low20. This matches the current Suggested Stop idea.",
        stop_fn=_low20,
    ),
    ExitRule(
        key="low20_minus_0_5atr",
        label="Low20 - 0.5ATR trailing",
        description="Low20 with a half-ATR buffer, intended to reduce small whipsaw stop-outs.",
        stop_fn=_low20_minus_half_atr,
    ),
    ExitRule(
        key="chandelier20_2_5atr",
        label="Chandelier20 2.5ATR",
        description="HHV20 minus 2.5 x ATR20, trailed upward only.",
        stop_fn=_chandelier20_25atr,
    ),
    ExitRule(
        key="ma50",
        label="MA50 trailing",
        description="Uses MA50 as the price stop, trailed upward only. This is slower but often less noisy.",
        stop_fn=_ma50,
    ),
    ExitRule(
        key="signal_false_only",
        label="Signal FALSE only",
        description="No intraday price stop. Exit only when the scanner signal turns FALSE or max holding is reached.",
        stop_fn=None,
        use_signal_false_exit=True,
    ),
]


def build_historical_features(prices: pd.DataFrame, universe: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    frames = []
    for symbol, g in prices.groupby("symbol", sort=False):
        out = compute_symbol_features(g.copy())
        out["symbol"] = symbol
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["symbol", "date"])
    df = df.merge(universe, on="symbol", how="left")

    df["dollar_vol_rank"] = df.groupby("date")["avg_dollar_vol_63"].rank(ascending=False, method="first")
    df["liquidity_eligible"] = (
        (df["history_days"] >= int(cfg["min_history_days"]))
        & (df["close"] >= float(cfg["min_close"]))
        & (df["avg_dollar_vol_20"] >= float(cfg["min_avg_dollar_vol_20"]))
        & (df["avg_dollar_vol_63"] >= float(cfg["min_avg_dollar_vol_63"]))
        & (df["dollar_vol_rank"] <= int(cfg["dollar_volume_top_n"]))
    )
    df["eligible_universe"] = df["base_universe_eligible"].fillna(False) & df["liquidity_eligible"].fillna(False)
    df = add_signal_surge_v0(df)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _valid_price(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def _simulate_one_symbol(g: pd.DataFrame, rule: ExitRule) -> list[dict]:
    g = g.sort_values("date").reset_index(drop=True).copy()
    if len(g) < 2 or SIGNAL_COL not in g.columns:
        return []

    g["prev_signal"] = g[SIGNAL_COL].shift(1).fillna(False).astype(bool)
    entry_signal_indices = g.index[(g[SIGNAL_COL].astype(bool)) & (~g["prev_signal"])].tolist()

    trades = []
    next_allowed_idx = 0

    for signal_idx in entry_signal_indices:
        entry_idx = signal_idx + 1
        if entry_idx >= len(g) or entry_idx < next_allowed_idx:
            continue

        signal_row = g.iloc[signal_idx]
        entry_row = g.iloc[entry_idx]
        entry_price = _num(entry_row, "open")
        if not _valid_price(entry_price):
            continue

        active_stop = np.nan
        if rule.stop_fn is not None:
            active_stop = rule.stop_fn(signal_row)
            if not _valid_price(active_stop):
                continue

        exit_idx = None
        exit_price = np.nan
        exit_reason = ""
        stop_at_exit = active_stop

        for j in range(entry_idx, len(g)):
            row = g.iloc[j]
            holding_days = j - entry_idx + 1

            if j > entry_idx and rule.use_signal_false_exit:
                prev_row = g.iloc[j - 1]
                if not bool(prev_row[SIGNAL_COL]):
                    open_price = _num(row, "open")
                    if _valid_price(open_price):
                        exit_idx = j
                        exit_price = open_price
                        exit_reason = "signal_false_next_open"
                        stop_at_exit = active_stop
                        break

            if rule.stop_fn is not None and _valid_price(active_stop):
                day_open = _num(row, "open")
                day_low = _num(row, "low")
                if _valid_price(day_low) and day_low <= active_stop:
                    exit_idx = j
                    exit_price = day_open if _valid_price(day_open) and day_open < active_stop else active_stop
                    exit_reason = "stop_hit"
                    stop_at_exit = active_stop
                    break

            if holding_days >= MAX_HOLDING_DAYS:
                close_price = _num(row, "close")
                if _valid_price(close_price):
                    exit_idx = j
                    exit_price = close_price
                    exit_reason = "max_holding_days"
                    stop_at_exit = active_stop
                    break

            if rule.stop_fn is not None:
                raw_next_stop = rule.stop_fn(row)
                if _valid_price(raw_next_stop):
                    if rule.use_trailing_max and _valid_price(active_stop):
                        active_stop = max(active_stop, raw_next_stop)
                    else:
                        active_stop = raw_next_stop

        if exit_idx is None:
            continue

        exit_row = g.iloc[exit_idx]
        gross_return = exit_price / entry_price - 1
        net_return = gross_return - ROUND_TRIP_COST

        trades.append({
            "rule_key": rule.key,
            "rule_label": rule.label,
            "symbol": str(entry_row.get("symbol", "")),
            "name": str(entry_row.get("name", "")),
            "asset_group": str(entry_row.get("asset_group", entry_row.get("group", ""))),
            "entry_signal_date": signal_row["date"].date().isoformat(),
            "entry_date": entry_row["date"].date().isoformat(),
            "entry_price": float(entry_price),
            "exit_date": exit_row["date"].date().isoformat(),
            "exit_price": float(exit_price),
            "gross_return": float(gross_return),
            "net_return": float(net_return),
            "holding_days": int(exit_idx - entry_idx + 1),
            "exit_reason": exit_reason,
            "stop_at_exit": None if not _valid_price(stop_at_exit) else float(stop_at_exit),
            "entry_score": None if not np.isfinite(_num(signal_row, "score")) else float(_num(signal_row, "score")),
            "entry_r63": None if not np.isfinite(_num(signal_row, "r63")) else float(_num(signal_row, "r63")),
            "entry_er63": None if not np.isfinite(_num(signal_row, "er63")) else float(_num(signal_row, "er63")),
            "entry_surge_ratio": None if not np.isfinite(_num(signal_row, "surge_ratio")) else float(_num(signal_row, "surge_ratio")),
        })

        next_allowed_idx = exit_idx + 1

    return trades


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns.fillna(0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in EXIT_RULES:
        g = trades[trades["rule_key"] == rule.key].sort_values("entry_date") if not trades.empty else pd.DataFrame()
        if g.empty:
            rows.append({
                "rule_key": rule.key,
                "rule_label": rule.label,
                "description": rule.description,
                "trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "median_return": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "avg_holding_days": 0.0,
                "profit_factor": 0.0,
                "stop_hit_rate": 0.0,
                "signal_false_exit_rate": 0.0,
                "max_hold_exit_rate": 0.0,
            })
            continue

        ret = pd.to_numeric(g["net_return"], errors="coerce").dropna()
        wins = ret[ret > 0]
        losses = ret[ret < 0]
        gross_profit = wins.sum()
        gross_loss = losses.sum()

        rows.append({
            "rule_key": rule.key,
            "rule_label": rule.label,
            "description": rule.description,
            "trades": int(len(g)),
            "win_rate": float((ret > 0).mean()) if len(ret) else 0.0,
            "avg_return": float(ret.mean()) if len(ret) else 0.0,
            "median_return": float(ret.median()) if len(ret) else 0.0,
            "total_return": float((1 + ret).prod() - 1) if len(ret) else 0.0,
            "max_drawdown": _max_drawdown(ret),
            "avg_holding_days": float(pd.to_numeric(g["holding_days"], errors="coerce").mean()),
            "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0,
            "stop_hit_rate": float((g["exit_reason"] == "stop_hit").mean()),
            "signal_false_exit_rate": float((g["exit_reason"] == "signal_false_next_open").mean()),
            "max_hold_exit_rate": float((g["exit_reason"] == "max_holding_days").mean()),
        })

    return pd.DataFrame(rows)


def run_backtests(prices: pd.DataFrame, universe: pd.DataFrame, cfg: dict, data_dir: str | Path, as_of: str) -> dict:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    features = build_historical_features(prices, universe, cfg)
    all_trades: list[dict] = []

    if not features.empty:
        for rule in EXIT_RULES:
            for _, g in features.groupby("symbol", sort=False):
                all_trades.extend(_simulate_one_symbol(g, rule))

    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values(["entry_date", "rule_key", "symbol"]).reset_index(drop=True)

    summary = summarize_trades(trades)

    trades_path = data_path / "backtest_trades.csv"
    summary_path = data_path / "backtest_summary.json"

    trades.to_csv(trades_path, index=False)

    recent_trades = []
    if not trades.empty:
        recent_trades = trades.sort_values("entry_date", ascending=False).head(250).replace({np.nan: None}).to_dict(orient="records")

    payload = {
        "as_of": as_of,
        "signal_column": SIGNAL_COL,
        "entry_rule": "first TRUE day of the current signal streak, enter next trading day's open",
        "exit_model": "price stop uses intraday low; gap below stop exits at open; scanner FALSE exits next trading day's open",
        "round_trip_cost": ROUND_TRIP_COST,
        "max_holding_days": MAX_HOLDING_DAYS,
        "summary": summary.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "recent_trades": recent_trades,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload
