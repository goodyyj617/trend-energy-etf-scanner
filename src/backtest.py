from __future__ import annotations

import json
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
    stop_fn: Optional[Callable[[pd.Series], float]] = None
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


def _scanner_signal(df: pd.DataFrame) -> pd.Series:
    return df[BASE_SIGNAL_COL].fillna(False).astype(bool)


def make_early_te_cross_signal(r20_min: float, er20_min: float, ma50_slope_min: float) -> Callable[[pd.DataFrame], pd.Series]:
    def _fn(df: pd.DataFrame) -> pd.Series:
        te63 = pd.to_numeric(df["te63"], errors="coerce")
        te63_sma20 = te63.rolling(20).mean()
        crosses_up = (te63 > te63_sma20) & (te63.shift(1) <= te63_sma20.shift(1))
        return (
            df["eligible_universe"].fillna(False)
            & crosses_up.fillna(False)
            & (te63 > 0)
            & (pd.to_numeric(df["r20"], errors="coerce") >= r20_min)
            & (pd.to_numeric(df["er20"], errors="coerce") >= er20_min)
            & (pd.to_numeric(df["close"], errors="coerce") > pd.to_numeric(df["ma50"], errors="coerce"))
            & (pd.to_numeric(df["ma50_slope_10"], errors="coerce") >= ma50_slope_min)
        )
    return _fn


def make_score_breakout_signal(r20_min: float, er20_min: float, score_lookback: int) -> Callable[[pd.DataFrame], pd.Series]:
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


SIGNAL_RULES: list[SignalRule] = [
    SignalRule(
        key="scanner_surge_v0",
        label="Scanner surge v0",
        description="현재 웹앱의 signal_surge_v0 조건입니다. 기존 scanner TRUE와 정확히 같은 조건을 entry 후보로 사용합니다.",
        params={"source": "features.add_signal_surge_v0"},
        signal_fn=_scanner_signal,
    ),
    SignalRule(
        key="early_te_cross_base",
        label="Early TE cross base",
        description="TE63가 자기 20일 평균을 상향 돌파하고, R20/ER20/MA50 기울기가 최소 조건을 만족할 때의 초기 추세 전환형 signal입니다.",
        params={"r20_min": 0.00, "er20_min": 0.10, "ma50_slope_10_min": 0.00},
        signal_fn=make_early_te_cross_signal(r20_min=0.00, er20_min=0.10, ma50_slope_min=0.00),
    ),
    SignalRule(
        key="early_te_cross_strict",
        label="Early TE cross strict",
        description="Early TE cross base보다 R20/ER20 조건을 조금 더 강하게 둔 robustness variant입니다.",
        params={"r20_min": 0.02, "er20_min": 0.15, "ma50_slope_10_min": 0.00},
        signal_fn=make_early_te_cross_signal(r20_min=0.02, er20_min=0.15, ma50_slope_min=0.00),
    ),
    SignalRule(
        key="score_breakout_base",
        label="Score breakout base",
        description="Score가 직전 20거래일 고점을 돌파하고 R20/ER20 조건을 만족할 때의 초기 모멘텀 전환형 signal입니다.",
        params={"r20_min": 0.00, "er20_min": 0.10, "score_lookback": 20},
        signal_fn=make_score_breakout_signal(r20_min=0.00, er20_min=0.10, score_lookback=20),
    ),
]


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
    EntryRule(
        key="first_signal",
        label="First signal",
        description="Signal이 FALSE에서 TRUE로 처음 바뀐 날을 확인하고 다음 거래일 open에 진입합니다.",
        indices_fn=_first_signal_indices,
    ),
    EntryRule(
        key="signal_2d_confirm",
        label="Signal 2D confirm",
        description="Signal TRUE가 2거래일 연속 유지된 날을 확인하고 다음 거래일 open에 진입합니다.",
        indices_fn=_signal_2d_indices,
    ),
    EntryRule(
        key="signal_3d_confirm",
        label="Signal 3D confirm",
        description="Signal TRUE가 3거래일 연속 유지된 날을 확인하고 다음 거래일 open에 진입합니다.",
        indices_fn=_signal_3d_indices,
    ),
    EntryRule(
        key="breakout_5d_after_signal",
        label="5D breakout after signal",
        description="첫 signal 이후 5거래일 안에 종가가 직전 5일 고점을 돌파하면 다음 거래일 open에 진입합니다.",
        indices_fn=_breakout_after_signal_indices,
    ),
]


EXIT_RULES: list[ExitRule] = [
    ExitRule(
        key="low10",
        label="Low10 trailing",
        description="Initial stop은 signal-day Low10입니다. 보유 중 stop은 위로만 이동합니다.",
        stop_fn=_low10,
    ),
    ExitRule(
        key="low20",
        label="Low20 trailing",
        description="Initial stop은 signal-day Low20입니다. 현재 Suggested Stop 아이디어와 가장 직접적으로 대응합니다.",
        stop_fn=_low20,
    ),
    ExitRule(
        key="low20_minus_0_5atr",
        label="Low20 - 0.5ATR trailing",
        description="Low20에서 0.5 ATR buffer를 둔 stop입니다. 작은 흔들림에 의한 stop-out 감소를 테스트합니다.",
        stop_fn=_low20_minus_half_atr,
    ),
    ExitRule(
        key="chandelier20_2_5atr",
        label="Chandelier20 2.5ATR",
        description="HHV20 - 2.5 x ATR20 기준의 trailing stop입니다.",
        stop_fn=_chandelier20_25atr,
    ),
    ExitRule(
        key="ma50",
        label="MA50 trailing",
        description="MA50을 price stop으로 사용합니다. 느리지만 noise가 적은 stop입니다.",
        stop_fn=_ma50,
    ),
    ExitRule(
        key="maxhold_only",
        label="MaxHold only",
        description="가격 stop 없이 max holding 기간이 끝나면 close로 청산합니다. Stop rule의 기준점 역할을 합니다.",
        stop_fn=None,
    ),
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


def _simulate_one_symbol(g: pd.DataFrame, strategy: StrategyRule) -> tuple[list[dict], list[dict]]:
    g = _apply_signal_rule(g.sort_values("date").reset_index(drop=True), strategy.signal)
    if len(g) < 2:
        return [], []

    signal_indices = strategy.entry.indices_fn(g)
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

        active_stop = np.nan
        if strategy.exit.stop_fn is not None:
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
            "stop_at_exit": None if not _valid_price(stop_at_exit) else float(stop_at_exit),
            "entry_score": None if not np.isfinite(_num(signal_row, "score")) else float(_num(signal_row, "score")),
            "entry_score_pct": None if not np.isfinite(_num(signal_row, "score_pct")) else float(_num(signal_row, "score_pct")),
            "entry_r20": None if not np.isfinite(_num(signal_row, "r20")) else float(_num(signal_row, "r20")),
            "entry_r63": None if not np.isfinite(_num(signal_row, "r63")) else float(_num(signal_row, "r63")),
            "entry_er20": None if not np.isfinite(_num(signal_row, "er20")) else float(_num(signal_row, "er20")),
            "entry_er63": None if not np.isfinite(_num(signal_row, "er63")) else float(_num(signal_row, "er63")),
            "entry_surge_ratio": None if not np.isfinite(_num(signal_row, "surge_ratio")) else float(_num(signal_row, "surge_ratio")),
        })

        next_allowed_idx = exit_idx + 1

    return trades, skipped


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns.fillna(0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def summarize_trades(trades: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGY_RULES:
        g = trades[trades["strategy_key"] == strategy.key].sort_values("entry_date") if not trades.empty else pd.DataFrame()
        sg = skipped[skipped["strategy_key"] == strategy.key] if not skipped.empty else pd.DataFrame()
        skipped_stop_broken = int(len(sg)) if not sg.empty else 0
        if g.empty:
            rows.append({
                "strategy_key": strategy.key,
                "strategy_label": strategy.label,
                "signal_label": strategy.signal.label,
                "entry_label": strategy.entry.label,
                "exit_label": strategy.exit.label,
                "signal_params": json.dumps(strategy.signal.params, sort_keys=True),
                "description": f"Signal: {strategy.signal.description} Entry: {strategy.entry.description} Exit: {strategy.exit.description}",
                "trades": 0,
                "skipped_stop_broken": skipped_stop_broken,
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
            "signal_label": strategy.signal.label,
            "entry_label": strategy.entry.label,
            "exit_label": strategy.exit.label,
            "signal_params": json.dumps(strategy.signal.params, sort_keys=True),
            "description": f"Signal: {strategy.signal.description} Entry: {strategy.entry.description} Exit: {strategy.exit.description}",
            "trades": int(len(g)),
            "skipped_stop_broken": skipped_stop_broken,
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
    groups = [(s, e) for s in SIGNAL_RULES for e in ENTRY_RULES]
    for signal_rule, entry_rule in groups:
        if diagnostics.empty:
            g = pd.DataFrame()
        else:
            g = diagnostics[(diagnostics["signal_key"] == signal_rule.key) & (diagnostics["entry_key"] == entry_rule.key)]
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


def run_backtests(prices: pd.DataFrame, universe: pd.DataFrame, cfg: dict, data_dir: str | Path, as_of: str) -> dict:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    features = build_historical_features(prices, universe, cfg)
    all_trades: list[dict] = []
    all_skipped: list[dict] = []
    all_diagnostics: list[dict] = []

    if not features.empty:
        grouped = list(features.groupby("symbol", sort=False))
        for strategy in STRATEGY_RULES:
            for _, g in grouped:
                trades_i, skipped_i = _simulate_one_symbol(g, strategy)
                all_trades.extend(trades_i)
                all_skipped.extend(skipped_i)

        for signal_rule in SIGNAL_RULES:
            for entry_rule in ENTRY_RULES:
                for _, g in grouped:
                    all_diagnostics.extend(_diagnose_one_symbol(g, signal_rule, entry_rule))

    trades = pd.DataFrame(all_trades)
    skipped = pd.DataFrame(all_skipped)
    diagnostics = pd.DataFrame(all_diagnostics)

    if not trades.empty:
        trades = trades.sort_values(["entry_date", "strategy_key", "symbol"]).reset_index(drop=True)
    if not skipped.empty:
        skipped = skipped.sort_values(["entry_date", "strategy_key", "symbol"]).reset_index(drop=True)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(["entry_date", "signal_key", "entry_key", "symbol"]).reset_index(drop=True)

    summary = summarize_trades(trades, skipped)
    diagnostic_summary = summarize_diagnostics(diagnostics)

    trades.to_csv(data_path / "backtest_trades.csv", index=False)
    skipped.to_csv(data_path / "backtest_skipped.csv", index=False)
    diagnostics.to_csv(data_path / "signal_diagnostics.csv", index=False)

    recent_trades = []
    if not trades.empty:
        recent_trades = trades.sort_values("entry_date", ascending=False).head(250).replace({np.nan: None}).to_dict(orient="records")

    payload = {
        "as_of": as_of,
        "entry_model": "Signal rules are entry candidate/trigger logic only. They are never used as exit signals.",
        "exit_model": "Exits use price stops or max holding only. Price stop uses intraday low; gap below stop exits at open.",
        "round_trip_cost": ROUND_TRIP_COST,
        "max_holding_days": MAX_HOLDING_DAYS,
        "diagnostic_definitions": {
            "fwd_Nd": "Entry next open 기준으로 N거래일 뒤 close까지 보유했을 때의 단순 수익률입니다.",
            "mfe_20d": "Maximum Favorable Excursion. Entry 이후 20거래일 동안 경험한 최대 유리한 가격 이동입니다. 높을수록 진입 후 위로 열렸던 공간이 큽니다.",
            "mae_20d": "Maximum Adverse Excursion. Entry 이후 20거래일 동안 경험한 최대 불리한 가격 이동입니다. 더 음수일수록 진입 후 아래로 많이 흔들렸다는 뜻입니다.",
            "skipped_stop_broken": "Entry open이 initial stop 이하라서 진입하지 않은 거래 수입니다. 롱 전략에서는 이미 손절선이 깨진 상태이므로 제외합니다."
        },
        "signal_rules": [
            {"key": r.key, "label": r.label, "description": r.description, "params": r.params}
            for r in SIGNAL_RULES
        ],
        "entry_rules": [
            {"key": r.key, "label": r.label, "description": r.description}
            for r in ENTRY_RULES
        ],
        "exit_rules": [
            {"key": r.key, "label": r.label, "description": r.description}
            for r in EXIT_RULES
        ],
        "summary": summary.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "diagnostic_summary": diagnostic_summary.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "recent_trades": recent_trades,
    }

    with open(data_path / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload
