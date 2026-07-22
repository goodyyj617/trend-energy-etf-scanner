from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .features import add_signal_surge_v0, compute_symbol_features

BASE_SIGNAL_COL = "signal_surge_v0"
ACTIVE_SIGNAL_COL = "__active_signal"
ROUND_TRIP_COST = 0.002
MAX_HOLDING_DAYS = 63
DIAGNOSTIC_HORIZONS = [1, 3, 5, 10, 20]
DIAGNOSTIC_SUMMARY_COLUMNS = [
    "signal_key",
    "signal_label",
    "entry_key",
    "entry_label",
    "signal_params",
    "description",
    "signals",
    *[
        column
        for horizon in DIAGNOSTIC_HORIZONS
        for column in (
            f"avg_fwd_{horizon}d",
            f"median_fwd_{horizon}d",
            f"win_rate_{horizon}d",
        )
    ],
    "avg_mfe_20d",
    "median_mfe_20d",
    "avg_mae_20d",
    "median_mae_20d",
]
RECENT_TRADE_LIMIT = 250
ANALYSIS_SCHEMA_VERSION = 3
STRATEGY_YEAR_BASIS = "entry_year"
STRATEGY_YEAR_SUMMARY_COLUMNS = [
    "strategy_key",
    "strategy_label",
    "score_lookback",
    "r20_min",
    "er20_min",
    "entry_rule",
    "exit_rule",
    "entry_year",
    "year_basis",
    "year_period_start",
    "year_period_end",
    "first_entry_date",
    "last_entry_date",
    "first_exit_date",
    "last_exit_date",
    "is_partial_year",
    "is_full_calendar_year",
    "completed_trades",
    "winning_trades",
    "losing_trades",
    "flat_trades",
    "gross_profit",
    "gross_loss_abs",
    "sum_trade_returns",
    "avg_trade_return",
    "median_trade_return",
    "profit_factor",
    "win_rate",
    "p10_trade_return",
    "worst_trade_return",
    "best_trade_return",
    "avg_holding_days",
]

# Transparent research decision thresholds. These are serialized with every
# backtest payload so the static UI never maintains a second threshold copy.
GATE_CONFIG = {
    "version": "provisional-trend-following-v1",
    "min_completed_trades": 100,
    "min_annual_completed_trades": 100,
    "min_eligible_years": 5,
    "min_eligible_neighbors": 2,
    "min_profit_factor": 1.2,
    "min_avg_trade_return": 0.0,
    "min_annual_profit_factor": 1.0,
    "min_joint_positive_year_ratio": 0.60,
    "min_loyo_pass_ratio": 0.80,
    "min_effective_neighbor_pass_ratio": 0.60,
    # Compatibility aliases for older bounded consumers. Median return is no
    # longer a qualification gate.
    "min_bucket_trades": 100,
    "min_neighbor_pass_ratio": 0.60,
    "min_positive_year_ratio": 0.60,
    "min_median_trade_return": None,
}

ENTRY_PERIOD_PRESETS = [
    ("all", "All available", None),
    ("last_1y", "Recent 1 year", 1),
    ("last_3y", "Recent 3 years", 3),
    ("last_5y", "Recent 5 years", 5),
]

# Score breakout robustness grid. This is compact enough for the daily GitHub
# Actions job, but structured so that more parameters can be added later for
# robustness checks and statistical tests.
SCORE_LOOKBACK_GRID = [10, 20, 40]
R20_MIN_GRID = [-0.02, 0.00, 0.02]
ER20_MIN_GRID = [0.05, 0.10, 0.15]


@dataclass(frozen=True)
class SignalRule:
    key: str
    label: str
    description: str
    params: dict
    signal_fn: Callable[[pd.DataFrame], pd.Series]


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
    stop_fn: Callable[[pd.Series], float]
    use_trailing_max: bool = True


@dataclass(frozen=True)
class StrategyRule:
    key: str
    label: str
    signal: SignalRule
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


def make_score_breakout_signal(r20_min: float, er20_min: float, score_lookback: int) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Score breakout signal.

    score = 0.65 * TE63 + 0.35 * TE126, where TE = return x efficiency ratio.
    A signal fires when today's score exceeds the highest score from the prior
    score_lookback trading days. This is not a price breakout; it is a breakout
    in the strategy's own trend-energy score.
    """
    def _fn(df: pd.DataFrame) -> pd.Series:
        score = pd.to_numeric(df["score"], errors="coerce")
        prev_high = score.shift(1).rolling(score_lookback).max()
        breakout = score > prev_high
        return (
            df["eligible_universe"].fillna(False)
            & breakout.fillna(False)
            & (score > 0)
            & (pd.to_numeric(df["r20"], errors="coerce") >= r20_min)
            & (pd.to_numeric(df["er20"], errors="coerce") >= er20_min)
            & (pd.to_numeric(df["close"], errors="coerce") > pd.to_numeric(df["ma50"], errors="coerce"))
        )
    return _fn


def _fmt_param(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    sign = "m" if value < 0 else "p"
    return f"{sign}{abs(value):.2f}".replace(".", "")


def build_score_breakout_signal_rules() -> list[SignalRule]:
    rules: list[SignalRule] = []
    for lookback in SCORE_LOOKBACK_GRID:
        for r20_min in R20_MIN_GRID:
            for er20_min in ER20_MIN_GRID:
                key = f"score_bo_l{lookback}_r{_fmt_param(r20_min)}_er{_fmt_param(er20_min)}"
                label = f"Score BO L{lookback} R20>={r20_min:.2f} ER20>={er20_min:.2f}"
                params = {
                    "family": "score_breakout",
                    "score_lookback": lookback,
                    "r20_min": r20_min,
                    "er20_min": er20_min,
                    "close_filter": "close > ma50",
                }
                description = (
                    "Score breakout signal. Score가 직전 lookback 거래일 고점을 돌파하고, "
                    "R20/ER20 최소 조건과 close > MA50 조건을 만족할 때 TRUE입니다. "
                    "Score는 0.65*TE63 + 0.35*TE126이며, TE는 return과 efficiency ratio를 곱한 trend-energy 지표입니다."
                )
                rules.append(
                    SignalRule(
                        key=key,
                        label=label,
                        description=description,
                        params=params,
                        signal_fn=make_score_breakout_signal(r20_min, er20_min, lookback),
                    )
                )
    return rules


SIGNAL_RULES: list[SignalRule] = build_score_breakout_signal_rules()


def _first_signal_indices(g: pd.DataFrame) -> list[int]:
    signal = g[ACTIVE_SIGNAL_COL].fillna(False).astype(bool)
    prev = signal.shift(1).fillna(False).astype(bool)
    return g.index[signal & (~prev)].tolist()


def _signal_2d_indices(g: pd.DataFrame) -> list[int]:
    signal = g[ACTIVE_SIGNAL_COL].fillna(False).astype(bool)
    prev1 = signal.shift(1).fillna(False).astype(bool)
    prev2 = signal.shift(2).fillna(False).astype(bool)
    return g.index[signal & prev1 & (~prev2)].tolist()


def _signal_3d_indices(g: pd.DataFrame) -> list[int]:
    signal = g[ACTIVE_SIGNAL_COL].fillna(False).astype(bool)
    prev1 = signal.shift(1).fillna(False).astype(bool)
    prev2 = signal.shift(2).fillna(False).astype(bool)
    prev3 = signal.shift(3).fillna(False).astype(bool)
    return g.index[signal & prev1 & prev2 & (~prev3)].tolist()


def _breakout_after_signal_indices(g: pd.DataFrame) -> list[int]:
    signal = g[ACTIVE_SIGNAL_COL].fillna(False).astype(bool)
    prev = signal.shift(1).fillna(False).astype(bool)
    first_signal_indices = g.index[signal & (~prev)].tolist()
    close = pd.to_numeric(g["close"], errors="coerce")
    prior_high5 = pd.to_numeric(g["high"], errors="coerce").shift(1).rolling(5).max()

    out: list[int] = []
    last_used = -1
    for start_idx in first_signal_indices:
        if start_idx <= last_used:
            continue
        end_idx = min(start_idx + 5, len(g) - 1)
        for j in range(start_idx + 1, end_idx + 1):
            if bool(signal.iloc[j]) and np.isfinite(close.iloc[j]) and np.isfinite(prior_high5.iloc[j]) and close.iloc[j] > prior_high5.iloc[j]:
                out.append(j)
                last_used = j
                break
    return out


ENTRY_RULES: list[EntryRule] = [
    EntryRule("first_signal", "First signal", "Signal이 FALSE에서 TRUE로 처음 바뀐 날을 확인하고 다음 거래일 open에 진입합니다.", _first_signal_indices),
    EntryRule("signal_2d_confirm", "Signal 2D confirm", "Signal TRUE가 2거래일 연속 유지된 날을 확인하고 다음 거래일 open에 진입합니다.", _signal_2d_indices),
    EntryRule("signal_3d_confirm", "Signal 3D confirm", "Signal TRUE가 3거래일 연속 유지된 날을 확인하고 다음 거래일 open에 진입합니다.", _signal_3d_indices),
    EntryRule("breakout_5d_after_signal", "5D breakout after signal", "첫 signal 이후 5거래일 안에 종가가 직전 5일 고점을 돌파하면 다음 거래일 open에 진입합니다.", _breakout_after_signal_indices),
]


EXIT_RULES: list[ExitRule] = [
    ExitRule("low10", "Low10 trailing", "Initial stop은 signal-day Low10입니다. 보유 중 stop은 위로만 이동합니다.", _low10),
    ExitRule("low20", "Low20 trailing", "Initial stop은 signal-day Low20입니다. 현재 Suggested Stop 아이디어와 가장 직접적으로 대응합니다.", _low20),
    ExitRule("low20_minus_0_5atr", "Low20 - 0.5ATR trailing", "Low20에서 0.5 ATR buffer를 둔 stop입니다. 작은 흔들림에 의한 stop-out 감소를 테스트합니다.", _low20_minus_half_atr),
    ExitRule("chandelier20_2_5atr", "Chandelier20 2.5ATR", "HHV20 - 2.5 x ATR20 기준의 trailing stop입니다.", _chandelier20_25atr),
    ExitRule("ma50", "MA50 trailing", "MA50을 price stop으로 사용합니다. 느리지만 noise가 적은 stop입니다.", _ma50),
]


STRATEGY_RULES: list[StrategyRule] = [
    StrategyRule(
        key=f"{signal.key}__{entry.key}__{exit_rule.key}",
        label=f"{signal.label} / {entry.label} / {exit_rule.label}",
        signal=signal,
        entry=entry,
        exit=exit_rule,
    )
    for signal in SIGNAL_RULES
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


def _apply_signal_rule(g: pd.DataFrame, signal_rule: SignalRule) -> pd.DataFrame:
    out = g.copy()
    out[ACTIVE_SIGNAL_COL] = signal_rule.signal_fn(out).fillna(False).astype(bool)
    return out


def _simulate_prepared_symbol(
    g: pd.DataFrame,
    strategy: StrategyRule,
    signal_indices: list[int],
) -> tuple[list[dict], list[dict]]:
    """Simulate one exit rule using a prepared signal frame and entry indices."""
    if len(g) < 2:
        return [], []

    trades = []
    skipped = []
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

        active_stop = strategy.exit.stop_fn(signal_row)
        if not _valid_price(active_stop):
            continue
        if entry_price <= active_stop:
            skipped.append({
                "strategy_key": strategy.key,
                "strategy_label": strategy.label,
                "signal_key": strategy.signal.key,
                "entry_key": strategy.entry.key,
                "exit_key": strategy.exit.key,
                "symbol": str(entry_row.get("symbol", "")),
                "entry_signal_date": signal_row["date"].date().isoformat(),
                "entry_date": entry_row["date"].date().isoformat(),
                "entry_price": float(entry_price),
                "initial_stop": float(active_stop),
                "skip_reason": "entry_open_at_or_below_initial_stop",
            })
            continue

        exit_idx = None
        exit_price = np.nan
        exit_reason = ""
        stop_at_exit = active_stop

        for j in range(entry_idx, len(g)):
            row = g.iloc[j]
            holding_days = j - entry_idx + 1
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

            raw_next_stop = strategy.exit.stop_fn(row)
            if _valid_price(raw_next_stop):
                active_stop = max(active_stop, raw_next_stop) if strategy.exit.use_trailing_max else raw_next_stop

        if exit_idx is None:
            continue

        exit_row = g.iloc[exit_idx]
        gross_return = exit_price / entry_price - 1
        net_return = gross_return - ROUND_TRIP_COST
        trades.append({
            "strategy_key": strategy.key,
            "strategy_label": strategy.label,
            "signal_key": strategy.signal.key,
            "signal_label": strategy.signal.label,
            "signal_params": json.dumps(strategy.signal.params, sort_keys=True),
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
            "stop_at_exit": float(stop_at_exit),
            "entry_score": None if not np.isfinite(_num(signal_row, "score")) else float(_num(signal_row, "score")),
            "entry_score_pct": None if not np.isfinite(_num(signal_row, "score_pct")) else float(_num(signal_row, "score_pct")),
            "entry_r20": None if not np.isfinite(_num(signal_row, "r20")) else float(_num(signal_row, "r20")),
            "entry_er20": None if not np.isfinite(_num(signal_row, "er20")) else float(_num(signal_row, "er20")),
        })
        next_allowed_idx = exit_idx + 1

    return trades, skipped


def _simulate_one_symbol(g: pd.DataFrame, strategy: StrategyRule) -> tuple[list[dict], list[dict]]:
    """Compatibility path that prepares one complete strategy independently."""
    prepared = _apply_signal_rule(g.sort_values("date").reset_index(drop=True), strategy.signal)
    signal_indices = strategy.entry.indices_fn(prepared)
    return _simulate_prepared_symbol(prepared, strategy, signal_indices)


def _simulate_symbol_strategies(
    g: pd.DataFrame,
    signal_rules: list[SignalRule] = SIGNAL_RULES,
    entry_rules: list[EntryRule] = ENTRY_RULES,
    exit_rules: list[ExitRule] = EXIT_RULES,
    strategy_rules: list[StrategyRule] = STRATEGY_RULES,
) -> tuple[list[dict], list[dict], int, int]:
    """Reuse one symbol's sorted frame, signals, and entry indices across exits."""
    prepared_base = g.sort_values("date").reset_index(drop=True)
    if len(prepared_base) < 2:
        return [], [], 0, 0

    strategy_by_components = {
        (strategy.signal.key, strategy.entry.key, strategy.exit.key): strategy
        for strategy in strategy_rules
    }
    trades: list[dict] = []
    skipped: list[dict] = []
    signal_entry_pairs_evaluated = 0
    exit_simulations_evaluated = 0

    for signal_rule in signal_rules:
        signal_frame = _apply_signal_rule(prepared_base, signal_rule)
        for entry_rule in entry_rules:
            signal_indices = entry_rule.indices_fn(signal_frame)
            signal_entry_pairs_evaluated += 1
            for exit_rule in exit_rules:
                strategy = strategy_by_components[(signal_rule.key, entry_rule.key, exit_rule.key)]
                trades_i, skipped_i = _simulate_prepared_symbol(signal_frame, strategy, signal_indices)
                trades.extend(trades_i)
                skipped.extend(skipped_i)
                exit_simulations_evaluated += 1

    return (
        trades,
        skipped,
        signal_entry_pairs_evaluated,
        exit_simulations_evaluated,
    )


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns.fillna(0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def _trade_metric_values(g: pd.DataFrame) -> dict:
    ordered = g.sort_values(["entry_date", "exit_date", "symbol"]) if not g.empty else g
    ret = pd.to_numeric(ordered["net_return"], errors="coerce").dropna() if not ordered.empty else pd.Series(dtype=float)
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    profit_factor = float(wins.sum() / abs(gross_loss)) if gross_loss < 0 else (np.nan if len(wins) else 0.0)
    avg_days = float(pd.to_numeric(ordered["holding_days"], errors="coerce").mean()) if not ordered.empty else 0.0
    trade_sequence_dd = _max_drawdown(ret)
    return {
        "trades": int(len(ordered)),
        "completed_trades": int(len(ordered)),
        "win_rate": float((ret > 0).mean()) if len(ret) else 0.0,
        "trade_win_rate": float((ret > 0).mean()) if len(ret) else 0.0,
        "avg_return": float(ret.mean()) if len(ret) else 0.0,
        "avg_trade_return": float(ret.mean()) if len(ret) else 0.0,
        "median_return": float(ret.median()) if len(ret) else 0.0,
        "median_trade_return": float(ret.median()) if len(ret) else 0.0,
        # Compatibility diagnostic: compounded event sequence, not a portfolio return.
        "total_return": float((1 + ret).prod() - 1) if len(ret) else 0.0,
        "sum_trade_returns": float(ret.sum()) if len(ret) else 0.0,
        "max_drawdown": trade_sequence_dd,
        "trade_sequence_drawdown": trade_sequence_dd,
        "worst_trade_return": float(ret.min()) if len(ret) else 0.0,
        "tail_return_10": float(ret.quantile(0.10)) if len(ret) else 0.0,
        "avg_holding_days": avg_days,
        "avg_days": avg_days,
        "profit_factor": profit_factor,
        "stop_hit_rate": float((ordered["exit_reason"] == "stop_hit").mean()) if not ordered.empty else 0.0,
        "max_hold_exit_rate": float((ordered["exit_reason"] == "max_holding_days").mean()) if not ordered.empty else 0.0,
    }


def _signal_params(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_strategy_year_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Build one bounded row per observed strategy and completed-trade entry year."""
    if trades.empty:
        return pd.DataFrame(columns=STRATEGY_YEAR_SUMMARY_COLUMNS)

    required = [
        "strategy_key", "strategy_label", "signal_params", "entry_key", "exit_key",
        "entry_date", "exit_date", "net_return", "holding_days",
    ]
    missing = set(required).difference(trades.columns)
    if missing:
        raise ValueError(f"Cannot build strategy-year summary; missing columns: {sorted(missing)}")

    work = trades[required].copy()
    work["entry_date"] = pd.to_datetime(work["entry_date"], errors="coerce")
    work["exit_date"] = pd.to_datetime(work["exit_date"], errors="coerce")
    work["net_return"] = pd.to_numeric(work["net_return"], errors="coerce")
    work["holding_days"] = pd.to_numeric(work["holding_days"], errors="coerce")
    invalid = work[["entry_date", "exit_date", "net_return", "holding_days"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"Cannot build strategy-year summary; {int(invalid.sum())} completed trades have invalid dates, returns, or holding days"
        )

    available_start = work["entry_date"].min().normalize()
    available_end = work["entry_date"].max().normalize()
    work["entry_year"] = work["entry_date"].dt.year.astype(int)
    work["winning_trades"] = (work["net_return"] > 0).astype(int)
    work["losing_trades"] = (work["net_return"] < 0).astype(int)
    work["flat_trades"] = (work["net_return"] == 0).astype(int)
    work["gross_profit"] = work["net_return"].where(work["net_return"] > 0, 0.0)
    work["gross_loss_abs"] = -work["net_return"].where(work["net_return"] < 0, 0.0)

    identity = work.drop_duplicates("strategy_key")[
        ["strategy_key", "strategy_label", "signal_params", "entry_key", "exit_key"]
    ].copy()
    params = identity["signal_params"].map(_signal_params)
    identity["score_lookback"] = params.map(lambda value: value.get("score_lookback"))
    identity["r20_min"] = params.map(lambda value: value.get("r20_min"))
    identity["er20_min"] = params.map(lambda value: value.get("er20_min"))
    identity = identity.rename(columns={"entry_key": "entry_rule", "exit_key": "exit_rule"})

    grouped = work.groupby(["strategy_key", "entry_year"], sort=True, observed=True)
    result = grouped.agg(
        first_entry_date=("entry_date", "min"),
        last_entry_date=("entry_date", "max"),
        first_exit_date=("exit_date", "min"),
        last_exit_date=("exit_date", "max"),
        completed_trades=("net_return", "size"),
        winning_trades=("winning_trades", "sum"),
        losing_trades=("losing_trades", "sum"),
        flat_trades=("flat_trades", "sum"),
        gross_profit=("gross_profit", "sum"),
        gross_loss_abs=("gross_loss_abs", "sum"),
        sum_trade_returns=("net_return", "sum"),
        avg_trade_return=("net_return", "mean"),
        median_trade_return=("net_return", "median"),
        worst_trade_return=("net_return", "min"),
        best_trade_return=("net_return", "max"),
        avg_holding_days=("holding_days", "mean"),
    ).reset_index()
    p10 = grouped["net_return"].quantile(0.10).rename("p10_trade_return").reset_index()
    result = result.merge(p10, on=["strategy_key", "entry_year"], how="left", validate="one_to_one")
    result = result.merge(identity, on="strategy_key", how="left", validate="many_to_one")

    with np.errstate(divide="ignore", invalid="ignore"):
        result["profit_factor"] = np.where(
            result["gross_loss_abs"] > 0,
            result["gross_profit"] / result["gross_loss_abs"],
            np.where(result["gross_profit"] > 0, np.nan, 0.0),
        )
    result["win_rate"] = result["winning_trades"] / result["completed_trades"]
    result["year_basis"] = STRATEGY_YEAR_BASIS

    year_starts = pd.to_datetime(result["entry_year"].astype(str) + "-01-01")
    year_ends = pd.to_datetime(result["entry_year"].astype(str) + "-12-31")
    result["year_period_start"] = year_starts.where(year_starts >= available_start, available_start)
    result["year_period_end"] = year_ends.where(year_ends <= available_end, available_end)
    result["is_full_calendar_year"] = (
        (result["year_period_start"] == year_starts)
        & (result["year_period_end"] == year_ends)
    )
    result["is_partial_year"] = ~result["is_full_calendar_year"]

    for column in [
        "year_period_start", "year_period_end", "first_entry_date", "last_entry_date",
        "first_exit_date", "last_exit_date",
    ]:
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")

    integer_columns = ["entry_year", "completed_trades", "winning_trades", "losing_trades", "flat_trades"]
    result[integer_columns] = result[integer_columns].astype(int)
    return result[STRATEGY_YEAR_SUMMARY_COLUMNS].sort_values(
        ["strategy_key", "entry_year"]
    ).reset_index(drop=True)


def _strategy_year_metadata(strategy_year_summary: pd.DataFrame) -> dict:
    if strategy_year_summary.empty:
        return {
            "strategy_year_summary_row_count": 0,
            "strategy_year_min": None,
            "strategy_year_max": None,
            "strategy_year_basis": STRATEGY_YEAR_BASIS,
            "partial_years": [],
            "full_calendar_years": [],
        }
    return {
        "strategy_year_summary_row_count": int(len(strategy_year_summary)),
        "strategy_year_min": int(strategy_year_summary["entry_year"].min()),
        "strategy_year_max": int(strategy_year_summary["entry_year"].max()),
        "strategy_year_basis": STRATEGY_YEAR_BASIS,
        "partial_years": sorted(
            strategy_year_summary.loc[strategy_year_summary["is_partial_year"], "entry_year"].unique().astype(int).tolist()
        ),
        "full_calendar_years": sorted(
            strategy_year_summary.loc[strategy_year_summary["is_full_calendar_year"], "entry_year"].unique().astype(int).tolist()
        ),
    }


def _gate_fields(metrics: dict) -> dict:
    sample_pass = int(metrics["completed_trades"]) >= int(GATE_CONFIG["min_completed_trades"])
    profit_factor = metrics["profit_factor"]
    edge_pass = bool(
        not pd.isna(profit_factor)
        and float(profit_factor) >= float(GATE_CONFIG["min_profit_factor"])
        and float(metrics["avg_trade_return"]) > float(GATE_CONFIG["min_avg_trade_return"])
    )
    return {
        "sample_gate_pass": bool(sample_pass),
        "edge_gate_pass": edge_pass,
        "sample_gate_status": "Pass" if sample_pass else "Insufficient",
        "edge_gate_status": "Pass" if edge_pass else "Fail",
        "gate_sample": "Pass" if sample_pass else "Insufficient",
        "gate_median_trade_return": "Diagnostic",
        "gate_profit_factor": "Pass" if edge_pass else "Fail",
    }


def summarize_trades(trades: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trade_groups = {key: group for key, group in trades.groupby("strategy_key", sort=False)} if not trades.empty else {}
    skipped_groups = {key: group for key, group in skipped.groupby("strategy_key", sort=False)} if not skipped.empty else {}
    for strategy in STRATEGY_RULES:
        g = trade_groups.get(strategy.key, pd.DataFrame())
        sg = skipped_groups.get(strategy.key, pd.DataFrame())
        metrics = _trade_metric_values(g)
        rows.append({
            "strategy_key": strategy.key,
            "strategy_label": strategy.label,
            "signal_key": strategy.signal.key,
            "signal_label": strategy.signal.label,
            "entry_key": strategy.entry.key,
            "entry_label": strategy.entry.label,
            "exit_key": strategy.exit.key,
            "exit_label": strategy.exit.label,
            "signal_params": json.dumps(strategy.signal.params, sort_keys=True),
            "description": f"Signal: {strategy.signal.description} Entry: {strategy.entry.description} Exit: {strategy.exit.description}",
            "skipped_stop_broken": int(len(sg)) if not sg.empty else 0,
            **metrics,
            **_gate_fields(metrics),
        })
    return pd.DataFrame(rows)


def _filter_by_entry_period(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if df.empty or "entry_date" not in df:
        return df.copy()
    entry_dates = pd.to_datetime(df["entry_date"], errors="coerce")
    mask = entry_dates.notna()
    if start is not None:
        mask &= entry_dates >= start
    if end is not None:
        mask &= entry_dates <= end
    return df.loc[mask].copy()


def _entry_period_specs(trades: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return [{"key": key, "label": label, "start": None, "end": None} for key, label, _ in ENTRY_PERIOD_PRESETS]
    entry_dates = pd.to_datetime(trades["entry_date"], errors="coerce").dropna()
    if entry_dates.empty:
        return [{"key": key, "label": label, "start": None, "end": None} for key, label, _ in ENTRY_PERIOD_PRESETS]
    available_start = entry_dates.min().normalize()
    available_end = entry_dates.max().normalize()
    specs = []
    for key, label, years in ENTRY_PERIOD_PRESETS:
        start = available_start if years is None else (available_end - pd.DateOffset(years=years) + pd.Timedelta(days=1)).normalize()
        specs.append({"key": key, "label": label, "start": start, "end": available_end})
    return specs


def _coerce_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "false": False}).fillna(False).astype(bool)


def validate_strategy_year_summary(
    strategy_year_summary: pd.DataFrame,
    overall_summary: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Return a clear unavailability reason, or None when the bounded aggregate is valid."""
    if strategy_year_summary is None or strategy_year_summary.empty:
        return "Strategy-year aggregate is unavailable."
    required = {
        "strategy_key", "entry_year", "year_basis", "is_partial_year", "is_full_calendar_year",
        "completed_trades", "gross_profit", "gross_loss_abs", "sum_trade_returns",
        "avg_trade_return", "median_trade_return", "profit_factor", "win_rate",
    }
    missing = sorted(required.difference(strategy_year_summary.columns))
    if missing:
        return f"Strategy-year aggregate is invalid; missing columns: {missing}."
    if strategy_year_summary.duplicated(["strategy_key", "entry_year"]).any():
        return "Strategy-year aggregate is invalid; duplicate strategy_key and entry_year rows were found."
    if set(strategy_year_summary["year_basis"].dropna().astype(str)) != {STRATEGY_YEAR_BASIS}:
        return "Strategy-year aggregate is invalid; year_basis must be entry_year."
    numeric_columns = [
        "entry_year", "completed_trades", "gross_profit", "gross_loss_abs", "sum_trade_returns",
        "avg_trade_return", "median_trade_return", "win_rate",
    ]
    numeric = strategy_year_summary[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or (numeric["completed_trades"] <= 0).any()
        or (numeric["gross_profit"] < 0).any()
        or (numeric["gross_loss_abs"] < 0).any()
    ):
        return "Strategy-year aggregate is invalid; required numeric values are missing or invalid."
    annual_pf = strategy_year_summary.apply(
        lambda row: _profit_factor_from_totals(float(row.gross_profit), float(row.gross_loss_abs)), axis=1
    )
    if not (
        np.allclose(
            numeric["gross_profit"] - numeric["gross_loss_abs"], numeric["sum_trade_returns"],
            rtol=1e-12, atol=1e-10,
        )
        and np.allclose(
            numeric["sum_trade_returns"] / numeric["completed_trades"], numeric["avg_trade_return"],
            rtol=1e-12, atol=1e-10,
        )
        and np.allclose(
            annual_pf, pd.to_numeric(strategy_year_summary["profit_factor"], errors="coerce"),
            rtol=1e-12, atol=1e-10, equal_nan=True,
        )
    ):
        return "Strategy-year aggregate is invalid; annual numerators and derived metrics disagree."
    for flag in ["is_full_calendar_year", "is_partial_year"]:
        normalized = strategy_year_summary[flag].astype(str).str.strip().str.lower()
        if not normalized.isin(["true", "false"]).all():
            return f"Strategy-year aggregate is invalid; {flag} contains a non-boolean value."
    full = _coerce_bool_series(strategy_year_summary["is_full_calendar_year"])
    partial = _coerce_bool_series(strategy_year_summary["is_partial_year"])
    if not (full == ~partial).all():
        return "Strategy-year aggregate is invalid; full and partial year flags disagree."
    if overall_summary is not None and not overall_summary.empty:
        required_overall = {"strategy_key", "completed_trades", "sum_trade_returns", "avg_trade_return", "profit_factor"}
        if not required_overall.issubset(overall_summary.columns):
            return "Strategy-year aggregate cannot be validated; overall summary fields are missing."
        annual = strategy_year_summary.groupby("strategy_key", sort=False).agg(
            annual_completed=("completed_trades", "sum"),
            annual_sum=("sum_trade_returns", "sum"),
            annual_gross_profit=("gross_profit", "sum"),
            annual_gross_loss=("gross_loss_abs", "sum"),
        ).reset_index()
        expected = overall_summary.loc[
            pd.to_numeric(overall_summary["completed_trades"], errors="coerce") > 0,
            ["strategy_key", "completed_trades", "sum_trade_returns", "avg_trade_return", "profit_factor"],
        ]
        reconstructed = expected.merge(annual, on="strategy_key", how="left", validate="one_to_one")
        if reconstructed[["annual_completed", "annual_sum", "annual_gross_profit", "annual_gross_loss"]].isna().any().any():
            return "Strategy-year aggregate is invalid; rows are missing for strategies with completed trades."
        reconstructed_avg = reconstructed["annual_sum"] / reconstructed["annual_completed"]
        reconstructed_pf = reconstructed.apply(
            lambda row: _profit_factor_from_totals(row.annual_gross_profit, row.annual_gross_loss), axis=1
        )
        if not (
            np.array_equal(reconstructed["annual_completed"].to_numpy(), reconstructed["completed_trades"].to_numpy())
            and np.allclose(reconstructed["annual_sum"], reconstructed["sum_trade_returns"], rtol=1e-12, atol=1e-10)
            and np.allclose(reconstructed_avg, reconstructed["avg_trade_return"], rtol=1e-12, atol=1e-10)
            and np.allclose(reconstructed_pf, reconstructed["profit_factor"], rtol=1e-12, atol=1e-10, equal_nan=True)
        ):
            return "Strategy-year aggregate is invalid; annual numerators do not reconstruct the overall summary."
    return None


def _profit_factor_from_totals(gross_profit: float, gross_loss_abs: float) -> float:
    if gross_loss_abs > 0:
        return float(gross_profit / gross_loss_abs)
    return np.nan if gross_profit > 0 else 0.0


def _strategy_year_details(strategy_year_summary: pd.DataFrame, validation_reason: Optional[str]) -> pd.DataFrame:
    columns = [
        "strategy_key", "year", "entry_year", "is_partial_year", "is_full_calendar_year",
        "completed_trades", "avg_trade_return", "median_trade_return", "trade_win_rate",
        "profit_factor", "sum_trade_returns", "eligible_for_time_gate", "joint_positive",
        "joint_positive_status", "status",
    ]
    if validation_reason or strategy_year_summary is None or strategy_year_summary.empty:
        return pd.DataFrame(columns=columns)
    rows = strategy_year_summary.copy()
    rows["is_full_calendar_year"] = _coerce_bool_series(rows["is_full_calendar_year"])
    rows["is_partial_year"] = _coerce_bool_series(rows["is_partial_year"])
    rows["eligible_for_time_gate"] = (
        rows["is_full_calendar_year"]
        & (pd.to_numeric(rows["completed_trades"], errors="coerce") >= int(GATE_CONFIG["min_annual_completed_trades"]))
    )
    rows["joint_positive"] = (
        rows["eligible_for_time_gate"]
        & (pd.to_numeric(rows["avg_trade_return"], errors="coerce") > 0)
        & (pd.to_numeric(rows["profit_factor"], errors="coerce") > float(GATE_CONFIG["min_annual_profit_factor"]))
    ).astype(object)
    rows.loc[~rows["eligible_for_time_gate"], "joint_positive"] = None
    rows["joint_positive_status"] = np.where(
        ~rows["eligible_for_time_gate"],
        "Not eligible",
        np.where(rows["joint_positive"] == True, "Yes", "No"),  # noqa: E712
    )
    rows["status"] = np.where(
        ~rows["eligible_for_time_gate"],
        "Not eligible",
        np.where(rows["joint_positive"] == True, "Pass", "Fail"),  # noqa: E712
    )
    rows["year"] = pd.to_numeric(rows["entry_year"], errors="coerce").astype(int)
    rows["trade_win_rate"] = rows["win_rate"]
    return rows[columns].sort_values(["strategy_key", "entry_year"]).reset_index(drop=True)


def _attach_time_stability(
    summary: pd.DataFrame,
    strategy_year_summary: pd.DataFrame,
    validation_reason: Optional[str] = None,
) -> pd.DataFrame:
    out = summary.copy()
    groups = (
        {key: group.sort_values("entry_year") for key, group in strategy_year_summary.groupby("strategy_key", sort=False)}
        if not validation_reason and strategy_year_summary is not None and not strategy_year_summary.empty
        else {}
    )
    records = []
    for strategy_key in out["strategy_key"]:
        reason = validation_reason
        rows = groups.get(strategy_key)
        if reason is None and (rows is None or rows.empty):
            reason = "No strategy-year aggregate rows are available for this strategy."
        if reason:
            records.append({
                "time_gate_pass": None,
                "time_gate_status": "Not available",
                "time_gate_unavailable_reason": reason,
                "time_stability_status": "Not available",
                "eligible_years": None,
                "joint_positive_years": None,
                "joint_positive_year_ratio": None,
                "positive_year_ratio": None,
                "median_annual_avg_trade_return": None,
                "minimum_annual_avg_trade_return": None,
                "median_annual_profit_factor": None,
                "minimum_annual_profit_factor": None,
                "annual_median_trade_return": None,
                "loyo_fold_count": None,
                "loyo_pass_count": None,
                "loyo_pass_ratio": None,
                "worst_loyo_avg_trade_return": None,
                "worst_loyo_profit_factor": None,
                "omitted_year_causing_worst_avg": None,
                "omitted_year_causing_worst_pf": None,
            })
            continue

        work = rows.copy()
        full = _coerce_bool_series(work["is_full_calendar_year"])
        eligible = work.loc[
            full & (pd.to_numeric(work["completed_trades"], errors="coerce") >= int(GATE_CONFIG["min_annual_completed_trades"]))
        ].copy()
        eligible_count = int(len(eligible))
        joint_mask = (
            (pd.to_numeric(eligible["avg_trade_return"], errors="coerce") > 0)
            & (pd.to_numeric(eligible["profit_factor"], errors="coerce") > float(GATE_CONFIG["min_annual_profit_factor"]))
        )
        joint_count = int(joint_mask.sum())
        joint_ratio = joint_count / eligible_count if eligible_count else 0.0

        folds = []
        if eligible_count:
            totals = eligible[["completed_trades", "sum_trade_returns", "gross_profit", "gross_loss_abs"]].sum()
            for annual_row in eligible.itertuples(index=False):
                pooled_count = int(totals.completed_trades - annual_row.completed_trades)
                pooled_sum = float(totals.sum_trade_returns - annual_row.sum_trade_returns)
                pooled_gp = float(totals.gross_profit - annual_row.gross_profit)
                pooled_gl = float(totals.gross_loss_abs - annual_row.gross_loss_abs)
                pooled_avg = pooled_sum / pooled_count if pooled_count > 0 else np.nan
                pooled_pf = _profit_factor_from_totals(pooled_gp, pooled_gl)
                fold_pass = bool(
                    pooled_count >= int(GATE_CONFIG["min_completed_trades"])
                    and not pd.isna(pooled_avg) and pooled_avg > 0
                    and not pd.isna(pooled_pf) and pooled_pf >= float(GATE_CONFIG["min_profit_factor"])
                )
                folds.append({
                    "omitted_year": int(annual_row.entry_year),
                    "pooled_avg_trade_return": pooled_avg,
                    "pooled_profit_factor": pooled_pf,
                    "pass": fold_pass,
                })
        loyo_count = len(folds)
        loyo_pass_count = sum(int(fold["pass"]) for fold in folds)
        loyo_ratio = loyo_pass_count / loyo_count if loyo_count else 0.0
        finite_avg_folds = [fold for fold in folds if not pd.isna(fold["pooled_avg_trade_return"])]
        finite_pf_folds = [fold for fold in folds if not pd.isna(fold["pooled_profit_factor"])]
        worst_avg = min(finite_avg_folds, key=lambda fold: (fold["pooled_avg_trade_return"], fold["omitted_year"])) if finite_avg_folds else None
        worst_pf = min(finite_pf_folds, key=lambda fold: (fold["pooled_profit_factor"], fold["omitted_year"])) if finite_pf_folds else None
        time_pass = bool(
            eligible_count >= int(GATE_CONFIG["min_eligible_years"])
            and joint_ratio >= float(GATE_CONFIG["min_joint_positive_year_ratio"])
            and loyo_ratio >= float(GATE_CONFIG["min_loyo_pass_ratio"])
        )
        time_status = "Insufficient" if eligible_count < int(GATE_CONFIG["min_eligible_years"]) else ("Pass" if time_pass else "Fail")
        records.append({
            "time_gate_pass": time_pass,
            "time_gate_status": time_status,
            "time_gate_unavailable_reason": None,
            "time_stability_status": time_status,
            "eligible_years": eligible_count,
            "joint_positive_years": joint_count,
            "joint_positive_year_ratio": joint_ratio,
            "positive_year_ratio": joint_ratio,
            "median_annual_avg_trade_return": float(eligible["avg_trade_return"].median()) if eligible_count else None,
            "minimum_annual_avg_trade_return": float(eligible["avg_trade_return"].min()) if eligible_count else None,
            "median_annual_profit_factor": float(eligible["profit_factor"].median()) if eligible_count else None,
            "minimum_annual_profit_factor": float(eligible["profit_factor"].min()) if eligible_count else None,
            "annual_median_trade_return": float(eligible["median_trade_return"].median()) if eligible_count else None,
            "loyo_fold_count": loyo_count,
            "loyo_pass_count": loyo_pass_count,
            "loyo_pass_ratio": loyo_ratio,
            "worst_loyo_avg_trade_return": worst_avg["pooled_avg_trade_return"] if worst_avg else None,
            "worst_loyo_profit_factor": worst_pf["pooled_profit_factor"] if worst_pf else None,
            "omitted_year_causing_worst_avg": worst_avg["omitted_year"] if worst_avg else None,
            "omitted_year_causing_worst_pf": worst_pf["omitted_year"] if worst_pf else None,
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _adjacent_values(value: float | int, grid: list) -> list:
    try:
        idx = grid.index(value)
    except ValueError:
        return []
    return [grid[i] for i in (idx - 1, idx + 1) if 0 <= i < len(grid)]


def _canonical_outcome_value(value: object) -> object:
    return None if pd.isna(value) else float(value)


def _outcome_signature(row: object, annual_rows: Optional[pd.DataFrame]) -> Optional[tuple]:
    if annual_rows is None or annual_rows.empty:
        return None
    overall = (
        int(row.completed_trades),
        _canonical_outcome_value(row.avg_trade_return),
        _canonical_outcome_value(row.median_trade_return),
        _canonical_outcome_value(row.trade_win_rate),
        _canonical_outcome_value(row.profit_factor),
    )
    annual = tuple(
        (
            int(annual_row.entry_year),
            int(annual_row.completed_trades),
            _canonical_outcome_value(annual_row.avg_trade_return),
            _canonical_outcome_value(annual_row.median_trade_return),
            _canonical_outcome_value(annual_row.profit_factor),
        )
        for annual_row in annual_rows.sort_values("entry_year").itertuples(index=False)
    )
    return overall + annual


def _row_edge_pass(row: object) -> bool:
    return bool(
        int(row.completed_trades) >= int(GATE_CONFIG["min_completed_trades"])
        and not pd.isna(row.profit_factor)
        and float(row.profit_factor) >= float(GATE_CONFIG["min_profit_factor"])
        and float(row.avg_trade_return) > float(GATE_CONFIG["min_avg_trade_return"])
    )


def _attach_parameter_stability(
    summary: pd.DataFrame,
    strategy_year_summary: pd.DataFrame,
    validation_reason: Optional[str] = None,
) -> pd.DataFrame:
    out = summary.copy()
    signal_by_params = {
        (int(rule.params["score_lookback"]), float(rule.params["r20_min"]), float(rule.params["er20_min"])): rule.key
        for rule in SIGNAL_RULES
    }
    lookup = {(row.signal_key, row.entry_key, row.exit_key): row for row in out.itertuples(index=False)}
    annual_groups = (
        {key: group for key, group in strategy_year_summary.groupby("strategy_key", sort=False)}
        if not validation_reason and strategy_year_summary is not None and not strategy_year_summary.empty
        else {}
    )
    signatures = {
        row.strategy_key: _outcome_signature(row, annual_groups.get(row.strategy_key))
        for row in out.itertuples(index=False)
    }
    records = []
    for row in out.itertuples(index=False):
        params = json.loads(row.signal_params)
        lookback = int(params["score_lookback"])
        r20_min = float(params["r20_min"])
        er20_min = float(params["er20_min"])
        neighbor_params = [
            *((v, r20_min, er20_min) for v in _adjacent_values(lookback, SCORE_LOOKBACK_GRID)),
            *((lookback, v, er20_min) for v in _adjacent_values(r20_min, R20_MIN_GRID)),
            *((lookback, r20_min, v) for v in _adjacent_values(er20_min, ER20_MIN_GRID)),
        ]
        neighbors = []
        for param_tuple in neighbor_params:
            signal_key = signal_by_params.get(param_tuple)
            neighbor = lookup.get((signal_key, row.entry_key, row.exit_key))
            if neighbor is not None:
                neighbors.append(neighbor)
        eligible = [n for n in neighbors if int(n.completed_trades) >= int(GATE_CONFIG["min_completed_trades"])]
        raw_pass_count = sum(int(_row_edge_pass(n)) for n in eligible)
        raw_ratio = raw_pass_count / len(eligible) if eligible else 0.0
        reason = validation_reason
        candidate_signature = signatures.get(row.strategy_key)
        if reason is None and candidate_signature is None:
            reason = "Effective neighbor outcomes are unavailable because annual strategy rows are missing."
        effective = {}
        if reason is None:
            for neighbor in eligible:
                signature = signatures.get(neighbor.strategy_key)
                if signature is None or signature == candidate_signature:
                    continue
                effective.setdefault(signature, neighbor)
        effective_count = len(effective)
        effective_pass_count = sum(int(_row_edge_pass(n)) for n in effective.values())
        effective_ratio = effective_pass_count / effective_count if effective_count else 0.0
        parameter_pass = None if reason else bool(
            effective_count >= int(GATE_CONFIG["min_eligible_neighbors"])
            and effective_ratio >= float(GATE_CONFIG["min_effective_neighbor_pass_ratio"])
        )
        status = "Not available" if reason else (
            "Insufficient" if effective_count < int(GATE_CONFIG["min_eligible_neighbors"])
            else ("Pass" if parameter_pass else "Fail")
        )
        medians = [float(n.median_trade_return) for n in eligible if not pd.isna(n.median_trade_return)]
        drawdowns = [float(n.trade_sequence_drawdown) for n in eligible if not pd.isna(n.trade_sequence_drawdown)]
        records.append({
            "available_neighbors": len(neighbors),
            "eligible_neighbors": len(eligible),
            "neighbor_pass_ratio": raw_ratio,
            "raw_eligible_neighbors": len(eligible),
            "raw_neighbor_edge_pass_ratio": raw_ratio,
            "effective_eligible_neighbors": effective_count if reason is None else None,
            "effective_neighbor_edge_pass_ratio": effective_ratio if reason is None else None,
            "parameter_gate_pass": parameter_pass,
            "parameter_gate_status": status,
            "parameter_gate_unavailable_reason": reason,
            "neighbor_median_return_min": min(medians) if medians else None,
            "neighbor_median_return_max": max(medians) if medians else None,
            "neighbor_drawdown_min": min(drawdowns) if drawdowns else None,
            "neighbor_drawdown_max": max(drawdowns) if drawdowns else None,
            "parameter_stability_status": status,
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _attach_robustness_tiers(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    records = []
    for row in out.itertuples(index=False):
        gate_values = [
            bool(row.sample_gate_pass),
            bool(row.edge_gate_pass),
            row.time_gate_pass is True,
            row.parameter_gate_pass is True,
        ]
        qualified = all(gate_values)
        unavailable = [
            str(reason) for reason in (
                getattr(row, "time_gate_unavailable_reason", None),
                getattr(row, "parameter_gate_unavailable_reason", None),
            ) if isinstance(reason, str) and reason.strip()
        ]
        failed_names = [
            name for name, passed in zip(["Sample", "Edge", "Time", "Parameter"], gate_values) if not passed
        ]
        if qualified:
            status = "Pass"
            reason = "All mandatory provisional gates pass."
        elif unavailable:
            status = "Not available"
            reason = " ".join(dict.fromkeys(unavailable))
        elif not row.sample_gate_pass:
            status = "Insufficient"
            reason = f"Failed mandatory gates: {', '.join(failed_names)}."
        else:
            status = "Fail"
            reason = f"Failed mandatory gates: {', '.join(failed_names)}."
        tier = "Qualified" if qualified else "Not qualified"
        records.append({
            "mandatory_gates_pass": qualified,
            "mandatory_gates_passed": sum(int(value) for value in gate_values),
            "mandatory_gate_status": status,
            "qualification_tier": tier,
            "qualification_reason": reason,
            "qualification_tier_rank": 1 if qualified else 0,
            "robustness_tier": tier,
            "robustness_tier_rank": 1 if qualified else 0,
        })
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    out["regime_stability_status"] = "Not available"
    return out


def rank_strategy_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply the approved deterministic lexicographic production ranking."""
    if summary.empty:
        out = summary.copy()
        out["qualification_rank"] = pd.Series(dtype=int)
        return out
    out = summary.copy()
    out["__qualified"] = out["qualification_tier"].eq("Qualified")
    out["__time"] = out["time_gate_pass"].map(lambda value: value is True)
    out["__parameter"] = out["parameter_gate_pass"].map(lambda value: value is True)
    columns = [
        "__qualified", "__time", "__parameter", "loyo_pass_ratio", "joint_positive_year_ratio",
        "effective_neighbor_edge_pass_ratio", "profit_factor", "avg_trade_return", "completed_trades",
        "strategy_key",
    ]
    ascending = [False, False, False, False, False, False, False, False, False, True]
    out = out.sort_values(columns, ascending=ascending, na_position="last", kind="mergesort").reset_index(drop=True)
    out["qualification_rank"] = np.arange(1, len(out) + 1)
    return out.drop(columns=["__qualified", "__time", "__parameter"])


def build_period_analysis(
    trades: pd.DataFrame,
    skipped: pd.DataFrame,
    strategy_year_summary: Optional[pd.DataFrame] = None,
) -> list[dict]:
    periods = []
    available_entry = pd.to_datetime(trades.get("entry_date", pd.Series(dtype=str)), errors="coerce").dropna()
    available_start = available_entry.min().date().isoformat() if len(available_entry) else None
    available_end = available_entry.max().date().isoformat() if len(available_entry) else None
    for spec in _entry_period_specs(trades):
        period_trades = _filter_by_entry_period(trades, spec["start"], spec["end"])
        period_skipped = _filter_by_entry_period(skipped, spec["start"], spec["end"])
        period_year_summary = (
            strategy_year_summary.copy()
            if spec["key"] == "all" and strategy_year_summary is not None
            else build_strategy_year_summary(period_trades)
        )
        summary = summarize_trades(period_trades, period_skipped)
        annual_validation_reason = validate_strategy_year_summary(period_year_summary, summary)
        yearly = _strategy_year_details(period_year_summary, annual_validation_reason)
        summary = _attach_time_stability(summary, period_year_summary, annual_validation_reason)
        summary = _attach_parameter_stability(summary, period_year_summary, annual_validation_reason)
        summary = _attach_robustness_tiers(summary)
        summary = rank_strategy_summary(summary)
        included_entry = pd.to_datetime(period_trades.get("entry_date", pd.Series(dtype=str)), errors="coerce").dropna()
        included_exit = pd.to_datetime(period_trades.get("exit_date", pd.Series(dtype=str)), errors="coerce").dropna()
        periods.append({
            "key": spec["key"],
            "label": spec["label"],
            "filter_mode": "entry_date_inclusive",
            "requested_entry_start": spec["start"].date().isoformat() if spec["start"] is not None else None,
            "requested_entry_end": spec["end"].date().isoformat() if spec["end"] is not None else None,
            "available_entry_start": available_start,
            "available_entry_end": available_end,
            "included_entry_start": included_entry.min().date().isoformat() if len(included_entry) else None,
            "included_entry_end": included_entry.max().date().isoformat() if len(included_entry) else None,
            "realized_exit_start": included_exit.min().date().isoformat() if len(included_exit) else None,
            "realized_exit_end": included_exit.max().date().isoformat() if len(included_exit) else None,
            "included_completed_trades": int(len(period_trades)),
            "qualification_data_status": "Available" if annual_validation_reason is None else "Not available",
            "qualification_unavailable_reason": annual_validation_reason,
            "summary": _safe_records(summary),
            "yearly_details": _safe_records(yearly),
        })
    return periods


def _diagnose_one_symbol(g: pd.DataFrame, signal_rule: SignalRule, entry_rule: EntryRule) -> list[dict]:
    g = _apply_signal_rule(g.sort_values("date").reset_index(drop=True), signal_rule)
    signal_indices = entry_rule.indices_fn(g)
    out = []
    for signal_idx in signal_indices:
        entry_idx = signal_idx + 1
        if entry_idx >= len(g):
            continue
        entry_row = g.iloc[entry_idx]
        entry_price = _num(entry_row, "open")
        if not _valid_price(entry_price):
            continue
        record = {
            "signal_key": signal_rule.key,
            "signal_label": signal_rule.label,
            "signal_params": json.dumps(signal_rule.params, sort_keys=True),
            "entry_key": entry_rule.key,
            "entry_label": entry_rule.label,
            "symbol": str(entry_row.get("symbol", "")),
            "entry_signal_date": g.iloc[signal_idx]["date"].date().isoformat(),
            "entry_date": entry_row["date"].date().isoformat(),
            "entry_price": float(entry_price),
        }
        for horizon in DIAGNOSTIC_HORIZONS:
            target_idx = entry_idx + horizon - 1
            key = f"fwd_{horizon}d"
            if target_idx < len(g):
                target_close = _num(g.iloc[target_idx], "close")
                record[key] = float(target_close / entry_price - 1) if _valid_price(target_close) else np.nan
            else:
                record[key] = np.nan
        end_idx = min(entry_idx + 20 - 1, len(g) - 1)
        window = g.iloc[entry_idx : end_idx + 1]
        highs = pd.to_numeric(window["high"], errors="coerce")
        lows = pd.to_numeric(window["low"], errors="coerce")
        record["mfe_20d"] = float(highs.max() / entry_price - 1) if len(highs.dropna()) else np.nan
        record["mae_20d"] = float(lows.min() / entry_price - 1) if len(lows.dropna()) else np.nan
        out.append(record)
    return out


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_rule in SIGNAL_RULES:
        for entry_rule in ENTRY_RULES:
            g = diagnostics[(diagnostics["signal_key"] == signal_rule.key) & (diagnostics["entry_key"] == entry_rule.key)] if not diagnostics.empty else pd.DataFrame()
            row = {
                "signal_key": signal_rule.key,
                "signal_label": signal_rule.label,
                "entry_key": entry_rule.key,
                "entry_label": entry_rule.label,
                "signal_params": json.dumps(signal_rule.params, sort_keys=True),
                "description": f"Signal: {signal_rule.description} Entry: {entry_rule.description}",
                "signals": int(len(g)),
            }
            for horizon in DIAGNOSTIC_HORIZONS:
                col = f"fwd_{horizon}d"
                series = pd.to_numeric(g[col], errors="coerce").dropna() if col in g else pd.Series(dtype=float)
                row[f"avg_{col}"] = float(series.mean()) if len(series) else 0.0
                row[f"median_{col}"] = float(series.median()) if len(series) else 0.0
                row[f"win_rate_{horizon}d"] = float((series > 0).mean()) if len(series) else 0.0
            mfe = pd.to_numeric(g["mfe_20d"], errors="coerce").dropna() if "mfe_20d" in g else pd.Series(dtype=float)
            mae = pd.to_numeric(g["mae_20d"], errors="coerce").dropna() if "mae_20d" in g else pd.Series(dtype=float)
            row["avg_mfe_20d"] = float(mfe.mean()) if len(mfe) else 0.0
            row["median_mfe_20d"] = float(mfe.median()) if len(mfe) else 0.0
            row["avg_mae_20d"] = float(mae.mean()) if len(mae) else 0.0
            row["median_mae_20d"] = float(mae.median()) if len(mae) else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def empty_diagnostic_summary() -> pd.DataFrame:
    """Return a header-only compatibility table when diagnostics are disabled."""
    return pd.DataFrame(columns=DIAGNOSTIC_SUMMARY_COLUMNS)


def _safe_records(df: pd.DataFrame) -> list[dict]:
    return df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")


def _remove_oversized_legacy_outputs(data_path: Path) -> None:
    for name in ["backtest_trades.csv", "signal_diagnostics.csv", "backtest_skipped.csv"]:
        path = data_path / name
        if path.exists():
            path.unlink()


def run_backtests(prices: pd.DataFrame, universe: pd.DataFrame, cfg: dict, data_dir: str | Path, as_of: str) -> dict:
    total_started = time.perf_counter()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    _remove_oversized_legacy_outputs(data_path)

    stage_started = time.perf_counter()
    features = build_historical_features(prices, universe, cfg)
    print(f"backtest_timing build_historical_features_sec={time.perf_counter() - stage_started:.3f}")
    all_trades: list[dict] = []
    all_skipped: list[dict] = []
    all_diagnostics: list[dict] = []
    grouped = list(features.groupby("symbol", sort=False)) if not features.empty else []

    stage_started = time.perf_counter()
    signal_entry_pairs_evaluated = 0
    exit_simulations_evaluated = 0
    if grouped:
        for _, g in grouped:
            trades_i, skipped_i, signal_entry_pairs_i, exit_simulations_i = _simulate_symbol_strategies(g)
            all_trades.extend(trades_i)
            all_skipped.extend(skipped_i)
            signal_entry_pairs_evaluated += signal_entry_pairs_i
            exit_simulations_evaluated += exit_simulations_i
    trades = pd.DataFrame(all_trades)
    skipped = pd.DataFrame(all_skipped)
    if not trades.empty:
        trades = trades.sort_values(["entry_date", "strategy_key", "symbol"]).reset_index(drop=True)
    if not skipped.empty:
        skipped = skipped.sort_values(["entry_date", "strategy_key", "symbol"]).reset_index(drop=True)
    print(f"backtest_timing simulate_completed_trades_sec={time.perf_counter() - stage_started:.3f}")

    generate_signal_diagnostics = cfg.get("backtest_generate_signal_diagnostics", False) is True
    stage_started = time.perf_counter()
    if generate_signal_diagnostics and grouped:
        for signal_rule in SIGNAL_RULES:
            for entry_rule in ENTRY_RULES:
                for _, g in grouped:
                    all_diagnostics.extend(_diagnose_one_symbol(g, signal_rule, entry_rule))
    diagnostics = pd.DataFrame(all_diagnostics)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(["entry_date", "signal_key", "entry_key", "symbol"]).reset_index(drop=True)
    print(
        f"backtest_timing signal_diagnostics_sec={time.perf_counter() - stage_started:.3f} "
        f"enabled={str(generate_signal_diagnostics).lower()}"
    )

    stage_started = time.perf_counter()
    strategy_year_summary = build_strategy_year_summary(trades)
    strategy_year_aggregation_sec = time.perf_counter() - stage_started
    print(f"backtest_timing strategy_year_aggregation_sec={strategy_year_aggregation_sec:.3f}")

    stage_started = time.perf_counter()
    period_analysis = build_period_analysis(trades, skipped, strategy_year_summary)
    gate_analysis_sec = time.perf_counter() - stage_started
    summary = pd.DataFrame(period_analysis[0]["summary"]) if period_analysis else summarize_trades(trades, skipped)
    diagnostic_summary = (
        summarize_diagnostics(diagnostics)
        if generate_signal_diagnostics
        else empty_diagnostic_summary()
    )
    skipped_summary = (
        skipped.groupby(["strategy_key", "strategy_label", "signal_key", "entry_key", "exit_key", "skip_reason"])
        .size()
        .reset_index(name="skipped")
        if not skipped.empty else pd.DataFrame()
    )

    recent_trades_df = trades.sort_values("entry_date", ascending=False).head(RECENT_TRADE_LIMIT) if not trades.empty else pd.DataFrame()
    print(f"backtest_timing gate_analysis_sec={gate_analysis_sec:.3f}")

    print(
        f"backtest_counts symbols={len(grouped)} features_rows={len(features)} "
        f"signal_entry_pairs_evaluated={signal_entry_pairs_evaluated} "
        f"exit_simulations_evaluated={exit_simulations_evaluated} "
        f"completed_trade_rows={len(trades)} skipped_rows={len(skipped)} "
        f"diagnostic_rows={len(diagnostics)} strategies={len(STRATEGY_RULES)}"
    )

    stage_started = time.perf_counter()
    summary.to_csv(data_path / "backtest_strategy_summary.csv", index=False)
    strategy_year_summary.to_csv(data_path / "backtest_strategy_year_summary.csv", index=False)
    diagnostic_summary.to_csv(data_path / "signal_diagnostics_summary.csv", index=False)
    recent_trades_df.to_csv(data_path / "backtest_recent_trades.csv", index=False)
    skipped_summary.to_csv(data_path / "backtest_skipped_summary.csv", index=False)

    payload = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "as_of": as_of,
        "entry_model": "Score breakout signal rules are entry candidate/trigger logic only. They are never used as exit signals.",
        "exit_model": "Every strategy uses a price stop plus a max holding cap. Stopless MaxHold-only strategies are intentionally disabled.",
        "analysis_model": "Completed strategy trades are analyzed as independent events. This is not a portfolio equity backtest.",
        "period_filter_model": "Static preset summaries include completed trades whose entry_date is inside the requested inclusive period. Their realized exits may occur after the requested end date.",
        "round_trip_cost": ROUND_TRIP_COST,
        "max_holding_days": MAX_HOLDING_DAYS,
        "gate_config": GATE_CONFIG,
        "qualification_model": "Provisional trend-following Sample, Edge, Time, and effective Parameter gates. Median trade return and bootstrap analysis are diagnostic only.",
        "qualification_data_status": period_analysis[0]["qualification_data_status"] if period_analysis else "Not available",
        "qualification_unavailable_reason": period_analysis[0]["qualification_unavailable_reason"] if period_analysis else "Period analysis is unavailable.",
        "trade_count_total": int(len(trades)),
        "diagnostic_event_count": int(len(diagnostics)),
        "recent_trade_limit": RECENT_TRADE_LIMIT,
        **_strategy_year_metadata(strategy_year_summary),
        "timing_metadata": {
            "strategy_year_aggregation_sec": strategy_year_aggregation_sec,
            "gate_analysis_sec": gate_analysis_sec,
        },
        "diagnostic_definitions": {
            "fwd_Nd": "Entry next open 기준으로 N거래일 뒤 close까지 보유했을 때의 단순 수익률입니다.",
            "mfe_20d": "Maximum Favorable Excursion. Entry 이후 20거래일 동안 경험한 최대 유리한 가격 이동입니다. 높을수록 진입 후 위로 열렸던 공간이 큽니다.",
            "mae_20d": "Maximum Adverse Excursion. Entry 이후 20거래일 동안 경험한 최대 불리한 가격 이동입니다. 더 음수일수록 진입 후 아래로 많이 흔들렸다는 뜻입니다.",
            "skipped_stop_broken": "Entry open이 initial stop 이하라서 진입하지 않은 거래 수입니다. 롱 전략에서는 이미 손절선이 깨진 상태이므로 제외합니다.",
        },
        "signal_rules": [{"key": r.key, "label": r.label, "description": r.description, "params": r.params} for r in SIGNAL_RULES],
        "entry_rules": [{"key": r.key, "label": r.label, "description": r.description} for r in ENTRY_RULES],
        "exit_rules": [{"key": r.key, "label": r.label, "description": r.description} for r in EXIT_RULES],
        "metric_definitions": {
            "r20": "Close-to-close return over 20 trading days: close / close.shift(20) - 1.",
            "er20": "Upside efficiency over 20 trading days: max(close - close.shift(20), 0) divided by the sum of absolute daily close changes over the same window.",
            "te63": "The 63-trading-day return multiplied by the 63-day upside efficiency ratio.",
            "te126": "The 126-trading-day return multiplied by the 126-day upside efficiency ratio.",
            "score": "0.65 * TE63 + 0.35 * TE126.",
            "score_breakout": "Score exceeds its highest value over the prior score_lookback days, Score is positive, R20 and ER20 meet their thresholds, and Close is above MA50.",
            "sum_trade_returns": "Arithmetic sum of completed net trade returns after round-trip cost. It is not a portfolio return.",
            "trade_sequence_drawdown": "Drawdown of the sequentially compounded completed-trade return series ordered by entry date. It is not portfolio drawdown and does not model overlapping positions.",
            "profit_factor": "Sum of positive completed net trade returns divided by the absolute sum of negative completed net trade returns.",
            "eligible_entry_year": "A full calendar entry year with at least 100 completed trades. Partial years remain descriptive and are excluded from the Time Gate.",
            "annual_joint_positive_year": "An eligible entry year with annual average trade return above zero and annual Profit Factor above 1.0. Annual median return is diagnostic only.",
            "joint_positive_year_ratio": "Joint-positive eligible entry years divided by eligible entry years.",
            "loyo_fold": "One eligible entry year is omitted and pooled completed trades, sum returns, gross profit, and gross loss are reconstructed from the remaining annual aggregates.",
            "loyo_pass_ratio": "Passing leave-one-year-out folds divided by eligible folds. A fold passes with at least 100 trades, positive pooled average return, and pooled Profit Factor at least 1.2.",
            "effective_neighbor": "A direct adjacent parameter neighbor after candidate-identical outcomes are removed and duplicate aggregate-plus-annual outcome signatures are counted once.",
            "effective_neighbor_edge_pass_ratio": "Effective eligible neighbors passing completed trades at least 100, overall Profit Factor at least 1.2, and positive overall average return, divided by effective eligible neighbors.",
            "parameter_stability": "Effective direct-neighbor edge consistency while holding entry and exit rules fixed and changing one adjacent Score Breakout grid value. Time is not part of the neighbor edge test.",
            "time_stability": "Provisional in-sample consistency across eligible full entry-calendar-year cohorts and leave-one-year-out pooled reconstructions.",
            "strategy_year_summary": "Bounded completed-trade aggregates grouped by strategy_key and trade entry year. Partial years and annual median return remain descriptive only.",
            "qualification_limit": "Qualification uses provisional in-sample robustness gates and does not establish out-of-sample profitability. Bootstrap analysis is diagnostic only.",
        },
        "summary": _safe_records(summary),
        "period_analysis": period_analysis,
        "diagnostic_summary": _safe_records(diagnostic_summary),
        "recent_trades": _safe_records(recent_trades_df),
    }
    with open(data_path / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"backtest_timing output_write_sec={time.perf_counter() - stage_started:.3f}")
    print(f"backtest_timing total_sec={time.perf_counter() - total_started:.3f}")
    return payload
