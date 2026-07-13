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
class EntryRule:
    key: str
    label: str
    description: str
    indices_fn: Callable[[pd.DataFrame], list[int]]


@dataclass(frozen=True)
class ExitRule:
    key: str
    label: str
    description: str
    stop_fn: Optional[Callable[[pd.Series], float]] = None
    use_trailing_max: bool = True


@dataclass(frozen=True)
class StrategyRule:
    key: str
    label: str
    entry: EntryRule
    exit: ExitRule


def _num(row: pd.Series, key: str) -> float:
    value = row.get(key, np.nan)
    try:
        return float(value)
    except Exception:
        return np.nan


def _valid_price(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


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


def _first_signal_indices(g: pd.DataFrame) -> list[int]:
    signal = g[SIGNAL_COL].fillna(False).astype(bool)
    prev = signal.shift(1).fillna(False).astype(bool)
    return g.index[signal & (~prev)].tolist()


def _signal_2d_confirm_indices(g: pd.DataFrame) -> list[int]:
    signal = g[SIGNAL_COL].fillna(False).astype(bool)
    streak = signal.groupby((signal != signal.shift()).cumsum()).cumcount() + 1
    streak = streak.where(signal, 0)
    return g.index[signal & (streak == 2)].tolist()


def _signal_3d_confirm_indices(g: pd.DataFrame) -> list[int]:
    signal = g[SIGNAL_COL].fillna(False).astype(bool)
    streak = signal.groupby((signal != signal.shift()).cumsum()).cumcount() + 1
    streak = streak.where(signal, 0)
    return g.index[signal & (streak == 3)].tolist()


def _strong_first_signal_indices(g: pd.DataFrame) -> list[int]:
    first = set(_first_signal_indices(g))
    out = []
    for idx in first:
        row = g.loc[idx]
        if (
            bool(row.get(SIGNAL_COL, False))
            and _num(row, "score_pct") >= 0.70
            and _num(row, "er63") >= 0.25
            and _num(row, "r63") >= 0.05
        ):
            out.append(idx)
    return out


def _breakout_5d_after_signal_indices(g: pd.DataFrame) -> list[int]:
    first_signals = _first_signal_indices(g)
    out = []
    used_until = -1
    for signal_idx in first_signals:
        if signal_idx <= used_until:
            continue
        end_idx = min(signal_idx + 5, len(g) - 1)
        for idx in range(signal_idx + 1, end_idx + 1):
            row = g.iloc[idx]
            prev_high5 = g.iloc[max(0, idx - 5):idx]["high"].max()
            if _valid_price(prev_high5) and _num(row, "close") > prev_high5:
                out.append(idx)
                used_until = idx
                break
    return out


ENTRY_RULES: list[EntryRule] = [
    EntryRule(
        key="first_signal",
        label="First signal",
        description="Enter next open after the scanner signal first turns TRUE.",
        indices_fn=_first_signal_indices,
    ),
    EntryRule(
        key="signal_2d_confirm",
        label="Signal 2D confirm",
        description="Enter next open after the scanner signal stays TRUE for 2 consecutive trading days.",
        indices_fn=_signal_2d_confirm_indices,
    ),
    EntryRule(
        key="signal_3d_confirm",
        label="Signal 3D confirm",
        description="Enter next open after the scanner signal stays TRUE for 3 consecutive trading days.",
        indices_fn=_signal_3d_confirm_indices,
    ),
    EntryRule(
        key="breakout_5d_after_signal",
        label="5D breakout after signal",
        description="After a first signal, enter next open if close breaks the prior 5-day high within 5 trading days.",
        indices_fn=_breakout_5d_after_signal_indices,
    ),
    EntryRule(
        key="strong_first_signal",
        label="Strong first signal",
        description="First signal plus top-30% score, ER63 >= 0.25, and R63 >= 0.05.",
        indices_fn=_strong_first_signal_indices,
    ),
]


EXIT_RULES: list[ExitRule] = [
    ExitRule(
        key="low10",
        label="Low10 trailing",
        description="Initial stop is entry-signal-day Low10. Stop only moves up when the latest Low10 rises.",
        stop_fn=_low10,
    ),
    ExitRule(
        key="low20",
        label="Low20 trailing",
        description="Initial stop is entry-signal-day Low20. This matches the current Suggested Stop idea.",
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
        key="max_hold_only",
        label="MaxHold only",
        description="No price stop. Exit only at max holding days. Included as a loose benchmark.",
        stop_fn=None,
    ),
]


STRATEGY_RULES: list[StrategyRule] = [
    StrategyRule(
        key=f"{entry.key}__{exit_rule.key}",
        label=f"{entry.label} / {exit_rule.label}",
        entry=entry,
        exit=exit_rule,
    )
    for entry in ENTRY_RULES
    for exit_rule in EXIT_RULES
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
    df["score_pct"] = df.groupby("date")["score"].rank(pct=True, ascending=True)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _simulate_one_symbol(g: pd.DataFrame, strategy: StrategyRule) -> list[dict]:
    g = g.sort_values("date").reset_index(drop=True).copy()
    if len(g) < 2 or SIGNAL_COL not in g.columns:
        return []

    signal_indices = strategy.entry.indices_fn(g)
    trades = []
    next_allowed_idx = 0

    for signal_idx in signal_indices:
        entry_idx = signal_idx + 1
        if entry_idx >= len(g) or entry_idx < next_allowed_idx:
            continue

        signal_row = g.iloc[signal_idx]
        entry_row = g.iloc[entry_idx]
        entry_price = _num(entry_row, "open")
        if not _valid_price(entry_price):
            continue

        active_stop = np.nan
        if strategy.exit.stop_fn is not None:
            active_stop = strategy.exit.stop_fn(signal_row)
            if not _valid_price(active_stop):
                continue

        exit_idx = None
        exit_price = np.nan
        exit_reason = ""
        stop_at_exit = active_stop

        for j in range(entry_idx, len(g)):
            row = g.iloc[j]
            holding_days = j - entry_idx + 1

            if strategy.exit.stop_fn is not None and _valid_price(active_stop):
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

            if strategy.exit.stop_fn is not None:
                raw_next_stop = strategy.exit.stop_fn(row)
                if _valid_price(raw_next_stop):
                    if strategy.exit.use_trailing_max and _valid_price(active_stop):
                        active_stop = max(active_stop, raw_next_stop)
                    else:
                        active_stop = raw_next_stop

        if exit_idx is None:
            continue

        exit_row = g.iloc[exit_idx]
        gross_return = exit_price / entry_price - 1
        net_return = gross_return - ROUND_TRIP_COST

        trades.append({
            "strategy_key": strategy.key,
            "strategy_label": strategy.label,
            "entry_key": strategy.entry.key,
            "entry_label": strategy.entry.label,
            "exit_key": strategy.exit.key,
            "exit_label": strategy.exit.label,
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
            "entry_score_pct": None if not np.isfinite(_num(signal_row, "score_pct")) else float(_num(signal_row, "score_pct")),
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
    for strategy in STRATEGY_RULES:
        g = trades[trades["strategy_key"] == strategy.key].sort_values("entry_date") if not trades.empty else pd.DataFrame()
        if g.empty:
            rows.append({
                "strategy_key": strategy.key,
                "strategy_label": strategy.label,
                "entry_label": strategy.entry.label,
                "exit_label": strategy.exit.label,
                "description": f"Entry: {strategy.entry.description} Exit: {strategy.exit.description}",
                "trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "median_return": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "avg_holding_days": 0.0,
                "profit_factor": 0.0,
                "stop_hit_rate": 0.0,
                "max_hold_exit_rate": 0.0,
            })
            continue

        ret = pd.to_numeric(g["net_return"], errors="coerce").dropna()
        wins = ret[ret > 0]
        losses = ret[ret < 0]
        gross_profit = wins.sum()
        gross_loss = losses.sum()

        rows.append({
            "strategy_key": strategy.key,
            "strategy_label": strategy.label,
            "entry_label": strategy.entry.label,
            "exit_label": strategy.exit.label,
            "description": f"Entry: {strategy.entry.description} Exit: {strategy.exit.description}",
            "trades": int(len(g)),
            "win_rate": float((ret > 0).mean()) if len(ret) else 0.0,
            "avg_return": float(ret.mean()) if len(ret) else 0.0,
            "median_return": float(ret.median()) if len(ret) else 0.0,
            "total_return": float((1 + ret).prod() - 1) if len(ret) else 0.0,
            "max_drawdown": _max_drawdown(ret),
            "avg_holding_days": float(pd.to_numeric(g["holding_days"], errors="coerce").mean()),
            "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0,
            "stop_hit_rate": float((g["exit_reason"] == "stop_hit").mean()),
            "max_hold_exit_rate": float((g["exit_reason"] == "max_holding_days").mean()),
        })

    return pd.DataFrame(rows)


def run_backtests(prices: pd.DataFrame, universe: pd.DataFrame, cfg: dict, data_dir: str | Path, as_of: str) -> dict:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    features = build_historical_features(prices, universe, cfg)
    all_trades: list[dict] = []

    if not features.empty:
        for strategy in STRATEGY_RULES:
            for _, g in features.groupby("symbol", sort=False):
                all_trades.extend(_simulate_one_symbol(g, strategy))

    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values(["entry_date", "strategy_key", "symbol"]).reset_index(drop=True)

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
        "entry_model": "Scanner TRUE is used only for entry candidate logic. It is not used as an exit signal.",
        "exit_model": "Exits use price stops or max holding only. Price stop uses intraday low; gap below stop exits at open.",
        "round_trip_cost": ROUND_TRIP_COST,
        "max_holding_days": MAX_HOLDING_DAYS,
        "summary": summary.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "recent_trades": recent_trades,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload
