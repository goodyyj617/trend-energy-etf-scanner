from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PORTFOLIO_MODEL_NAME = "canonical_equal_weight_active_v1"
PORTFOLIO_INITIAL_CAPITAL = 1_000.0
PORTFOLIO_CURVE_CAP = 100
PORTFOLIO_DAILY_MATRIX_NAME = "backtest_portfolio_daily_returns.csv.gz"
PORTFOLIO_CURVE_SCHEMA_VERSION = 2
CDAR_DEFINITION_VERSION = "negative_drawdown_fixed_tail_count_v1"
INITIALIZATION_OBSERVATION = "initialization"
TRADING_SESSION_OBSERVATION = "trading_session"

PORTFOLIO_CURVE_COLUMNS = [
    "date",
    "observation_type",
    "strategy_key",
    "portfolio_equity",
    "cash_value",
    "invested_value",
    "gross_exposure",
    "active_position_count",
    "daily_portfolio_return",
    "cumulative_return",
    "running_peak_equity",
    "drawdown",
    "transaction_cost_paid",
    "turnover",
]

PORTFOLIO_SUMMARY_COLUMNS = [
    "strategy_key",
    "strategy_label",
    "portfolio_model",
    "portfolio_start_date",
    "portfolio_end_date",
    "initial_equity",
    "ending_equity",
    "total_portfolio_return",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "maximum_drawdown",
    "max_drawdown_peak_date",
    "max_drawdown_trough_date",
    "max_drawdown_recovery_date",
    "max_drawdown_duration_days",
    "longest_time_under_water_days",
    "ulcer_index",
    "conditional_drawdown_at_risk_95",
    "worst_daily_return",
    "worst_weekly_return",
    "worst_monthly_return",
    "daily_expected_shortfall_95",
    "daily_expected_shortfall_99",
    "daily_return_skewness",
    "daily_return_excess_kurtosis",
    "average_gross_exposure",
    "median_gross_exposure",
    "maximum_gross_exposure",
    "average_active_positions",
    "maximum_active_positions",
    "percent_days_in_cash",
    "annual_turnover",
    "total_transaction_cost",
]

DIAGNOSTIC_LEADER_FIELDS = [
    "profit_factor",
    "avg_trade_return",
    "joint_positive_year_ratio",
    "loyo_pass_ratio",
    "effective_neighbor_edge_pass_ratio",
    "median_trade_return",
]

FLOAT_EPSILON = 1e-9


class PortfolioInvariantError(RuntimeError):
    pass


def initialization_timestamp(first_session: object) -> str:
    """Return a non-session timestamp immediately before the first session."""
    timestamp = pd.Timestamp(first_session)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    timestamp = timestamp.normalize() - pd.Timedelta(microseconds=1)
    return timestamp.isoformat().replace("+00:00", "Z")


def _initialization_row(strategy_key: Optional[str], first_session: object, initial_capital: float) -> dict:
    return {
        "date": initialization_timestamp(first_session),
        "observation_type": INITIALIZATION_OBSERVATION,
        "strategy_key": strategy_key,
        "portfolio_equity": float(initial_capital),
        "cash_value": float(initial_capital),
        "invested_value": 0.0,
        "gross_exposure": 0.0,
        "active_position_count": 0,
        "daily_portfolio_return": 0.0,
        "transaction_cost_paid": 0.0,
        "turnover": 0.0,
    }


def economic_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty or "observation_type" not in curve.columns:
        return curve.copy()
    return curve.loc[curve["observation_type"] != INITIALIZATION_OBSERVATION].copy()


@dataclass(frozen=True)
class PricePanel:
    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    opens: np.ndarray
    closes: np.ndarray
    symbol_to_column: dict[str, int]
    date_to_row: dict[pd.Timestamp, int]


def build_price_panel(
    prices: pd.DataFrame,
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
) -> PricePanel:
    required = {"date", "symbol", "open", "close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Cannot build portfolio price panel; missing columns: {sorted(missing)}")

    work = prices[["date", "symbol", "open", "close"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype(str)
    work["open"] = pd.to_numeric(work["open"], errors="coerce")
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"])
    if start_date is not None:
        work = work.loc[work["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        work = work.loc[work["date"] <= pd.Timestamp(end_date).normalize()]
    if work.empty:
        return PricePanel(
            dates=pd.DatetimeIndex([]),
            symbols=(),
            opens=np.empty((0, 0), dtype=float),
            closes=np.empty((0, 0), dtype=float),
            symbol_to_column={},
            date_to_row={},
        )

    open_frame = work.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
    close_frame = work.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    symbols = sorted(set(open_frame.columns).union(close_frame.columns))
    dates = open_frame.index.union(close_frame.index).sort_values()
    open_frame = open_frame.reindex(index=dates, columns=symbols)
    # Close marks may carry forward for valuation only. Open decisions always use
    # the unfilled open matrix.
    close_frame = close_frame.reindex(index=dates, columns=symbols).ffill()
    return PricePanel(
        dates=pd.DatetimeIndex(dates),
        symbols=tuple(str(symbol) for symbol in symbols),
        opens=open_frame.to_numpy(dtype=float, copy=True),
        closes=close_frame.to_numpy(dtype=float, copy=True),
        symbol_to_column={str(symbol): index for index, symbol in enumerate(symbols)},
        date_to_row={pd.Timestamp(date).normalize(): index for index, date in enumerate(dates)},
    )


def _valid_price(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _event_records(lifecycles: pd.DataFrame, panel: PricePanel) -> list[dict]:
    if lifecycles is None or lifecycles.empty:
        return []
    required = {"strategy_key", "symbol", "entry_date", "entry_price"}
    missing = required.difference(lifecycles.columns)
    if missing:
        raise ValueError(f"Cannot simulate canonical portfolio; missing lifecycle columns: {sorted(missing)}")

    records = []
    ordered = lifecycles.sort_values(["entry_date", "symbol", "exit_date"], na_position="last")
    for row in ordered.to_dict(orient="records"):
        symbol = str(row.get("symbol", ""))
        column = panel.symbol_to_column.get(symbol)
        entry_date = pd.to_datetime(row.get("entry_date"), errors="coerce")
        if column is None or pd.isna(entry_date) or not _valid_price(row.get("entry_price")):
            continue
        entry_date = pd.Timestamp(entry_date).normalize()
        entry_index = panel.date_to_row.get(entry_date)
        if entry_index is None:
            continue

        exit_date = pd.to_datetime(row.get("exit_date"), errors="coerce")
        exit_index = None if pd.isna(exit_date) else panel.date_to_row.get(pd.Timestamp(exit_date).normalize())
        event = {
            "strategy_key": str(row["strategy_key"]),
            "symbol": symbol,
            "column": column,
            "entry_date": entry_date,
            "entry_index": entry_index,
            "entry_price": float(row["entry_price"]),
            "exit_date": None if pd.isna(exit_date) else pd.Timestamp(exit_date).normalize(),
            "exit_index": exit_index,
            "exit_price": _safe_float(row.get("exit_price")),
            "exit_reason": str(row.get("exit_reason") or "open_at_end"),
            "stop_at_exit": _safe_float(row.get("stop_at_exit")),
        }
        event["exit_phase"] = _exit_phase(event, panel)
        records.append(event)
    return records


def _exit_phase(event: dict, panel: PricePanel) -> Optional[str]:
    if event["exit_index"] is None or event["exit_price"] is None:
        return None
    if event["exit_reason"] == "max_holding_days":
        return "close_max_hold"
    if event["exit_reason"] == "stop_hit":
        day_open = panel.opens[event["exit_index"], event["column"]]
        stop = event.get("stop_at_exit")
        if (
            _valid_price(day_open)
            and stop is not None
            and day_open < stop
            and math.isclose(float(event["exit_price"]), float(day_open), rel_tol=1e-12, abs_tol=1e-10)
        ):
            return "gap_open"
        return "intraday_stop"
    return "close_other"


def _target_equal_weight(
    cash: float,
    current_values: dict[int, float],
    half_cost_rate: float,
) -> float:
    count = len(current_values)
    if count == 0:
        return 0.0
    equity_before_cost = cash + sum(current_values.values())
    if equity_before_cost <= 0:
        return 0.0

    def residual(target: float) -> float:
        turnover = sum(abs(target - value) for value in current_values.values())
        return equity_before_cost - count * target - half_cost_rate * turnover

    low = 0.0
    high = equity_before_cost / count
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if residual(midpoint) >= 0:
            low = midpoint
        else:
            high = midpoint
    return low


def simulate_canonical_portfolio(
    lifecycles: pd.DataFrame,
    panel: PricePanel,
    *,
    initial_capital: float = PORTFOLIO_INITIAL_CAPITAL,
    round_trip_cost: float,
) -> pd.DataFrame:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if round_trip_cost < 0:
        raise ValueError("round_trip_cost cannot be negative")
    if len(panel.dates) == 0:
        return pd.DataFrame(columns=PORTFOLIO_CURVE_COLUMNS)

    half_cost_rate = float(round_trip_cost) / 2.0
    events = _event_records(lifecycles, panel)
    strategy_key = (
        str(lifecycles["strategy_key"].iloc[0])
        if lifecycles is not None and not lifecycles.empty and "strategy_key" in lifecycles
        else None
    )
    entries_by_day: dict[int, list[dict]] = {}
    exits_by_day: dict[tuple[int, str], list[dict]] = {}
    for event in events:
        entries_by_day.setdefault(event["entry_index"], []).append(event)
        if event["exit_index"] is not None and event["exit_phase"] is not None:
            exits_by_day.setdefault((event["exit_index"], event["exit_phase"]), []).append(event)

    positions: dict[int, dict] = {}
    cash = float(initial_capital)
    previous_equity = float(initial_capital)
    pending_membership_rebalance = False
    rows = [_initialization_row(strategy_key, panel.dates[0], initial_capital)]

    def execute_exit(event: dict) -> tuple[float, float]:
        nonlocal cash
        position = positions.get(event["column"])
        if position is None or position["event"]["entry_index"] != event["entry_index"]:
            return 0.0, 0.0
        exit_price = event["exit_price"]
        if exit_price is None or not _valid_price(exit_price):
            raise PortfolioInvariantError(f"Invalid exit price for {event['strategy_key']} {event['symbol']}")
        gross_proceeds = position["quantity"] * float(exit_price)
        cost = gross_proceeds * half_cost_rate
        cash += gross_proceeds - cost
        del positions[event["column"]]
        return gross_proceeds, cost

    for day_index, date in enumerate(panel.dates):
        day_turnover_notional = 0.0
        day_cost = 0.0
        membership_changed = pending_membership_rebalance
        pending_membership_rebalance = False

        # 1. Gap/open stop exits use the current open and the already-known stop.
        for event in exits_by_day.get((day_index, "gap_open"), []):
            turnover, cost = execute_exit(event)
            day_turnover_notional += turnover
            day_cost += cost
            membership_changed = True

        # 2. Max-hold exits are scheduled now, but the existing strategy convention
        # executes them at this day's close. Their proceeds are not used at the open.

        # 3. New next-open entries become portfolio members. Duplicate symbol
        # membership is rejected deterministically; the event simulator already
        # prevents this in production data.
        for event in entries_by_day.get(day_index, []):
            if event["column"] in positions:
                continue
            positions[event["column"]] = {"quantity": 0.0, "event": event}
            membership_changed = True

        # 4. Equal-weight open rebalance happens only after a membership change.
        if membership_changed and positions:
            open_prices = {}
            for column, position in positions.items():
                if position["quantity"] <= FLOAT_EPSILON and position["event"]["entry_index"] == day_index:
                    price = position["event"]["entry_price"]
                else:
                    price = panel.opens[day_index, column]
                if not _valid_price(price):
                    open_prices = {}
                    break
                open_prices[column] = float(price)

            if open_prices:
                current_values = {
                    column: positions[column]["quantity"] * price
                    for column, price in open_prices.items()
                }
                target_value = _target_equal_weight(cash, current_values, half_cost_rate)
                for column in sorted(positions):
                    current_value = current_values[column]
                    delta = target_value - current_value
                    trade_notional = abs(delta)
                    cost = trade_notional * half_cost_rate
                    if delta >= 0:
                        cash -= delta + cost
                    else:
                        cash += -delta - cost
                    day_turnover_notional += trade_notional
                    day_cost += cost
                    positions[column]["quantity"] = target_value / open_prices[column]
                if cash < -FLOAT_EPSILON:
                    raise PortfolioInvariantError(f"Negative cash after rebalance: {cash}")
                cash = max(cash, 0.0)
            else:
                pending_membership_rebalance = True
        elif membership_changed and not positions:
            pending_membership_rebalance = False

        # 5. Intraday stop exits occur after the open rebalance. Their proceeds stay
        # in cash until the following tradable open.
        for event in exits_by_day.get((day_index, "intraday_stop"), []):
            turnover, cost = execute_exit(event)
            day_turnover_notional += turnover
            day_cost += cost
            if turnover > 0:
                pending_membership_rebalance = True

        # 6. Existing max-hold and other close exits execute at their recorded close
        # price, then all remaining positions are marked to the adjusted close.
        for phase in ("close_max_hold", "close_other"):
            for event in exits_by_day.get((day_index, phase), []):
                turnover, cost = execute_exit(event)
                day_turnover_notional += turnover
                day_cost += cost
                if turnover > 0:
                    pending_membership_rebalance = True

        invested_value = 0.0
        active_count = 0
        for column, position in positions.items():
            close_price = panel.closes[day_index, column]
            if not _valid_price(close_price):
                raise PortfolioInvariantError(
                    f"Missing close mark for active symbol {position['event']['symbol']} on {date.date().isoformat()}"
                )
            value = position["quantity"] * float(close_price)
            invested_value += value
            if position["quantity"] > FLOAT_EPSILON:
                active_count += 1

        equity = cash + invested_value
        if cash < -FLOAT_EPSILON:
            raise PortfolioInvariantError(f"Negative cash at close: {cash}")
        if invested_value > equity + FLOAT_EPSILON:
            raise PortfolioInvariantError("Gross exposure exceeds one without borrowing")
        daily_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
        turnover = day_turnover_notional / previous_equity if previous_equity > 0 else 0.0
        gross_exposure = invested_value / equity if equity > 0 else 0.0
        rows.append({
            "date": pd.Timestamp(date).date().isoformat(),
            "observation_type": TRADING_SESSION_OBSERVATION,
            "strategy_key": strategy_key,
            "portfolio_equity": equity,
            "cash_value": cash,
            "invested_value": invested_value,
            "gross_exposure": gross_exposure,
            "active_position_count": active_count,
            "daily_portfolio_return": daily_return,
            "transaction_cost_paid": day_cost,
            "turnover": turnover,
        })
        previous_equity = equity

    curve = pd.DataFrame(rows)
    curve["cumulative_return"] = curve["portfolio_equity"] / float(initial_capital) - 1.0
    curve["running_peak_equity"] = curve["portfolio_equity"].cummax().clip(lower=float(initial_capital))
    curve["drawdown"] = curve["portfolio_equity"] / curve["running_peak_equity"] - 1.0
    return curve[PORTFOLIO_CURVE_COLUMNS]


def expected_shortfall(returns: pd.Series, confidence: float) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    cutoff = clean.quantile(1.0 - confidence, interpolation="linear")
    tail = clean.loc[clean <= cutoff]
    return float(tail.mean()) if len(tail) else float(cutoff)


def conditional_drawdown_at_risk(drawdowns: pd.Series, confidence: float = 0.95) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    clean = pd.to_numeric(drawdowns, errors="coerce")
    clean = clean.loc[np.isfinite(clean)].astype(float)
    if clean.empty:
        return 0.0
    if (clean > FLOAT_EPSILON).any():
        raise ValueError("drawdown values cannot be positive")
    clean = clean.clip(upper=0.0)
    ordered = np.sort(clean.to_numpy(dtype=float))
    tail_count = max(1, len(ordered) - math.floor(confidence * len(ordered) + 1e-12))
    return float(ordered[:tail_count].mean())


def _period_returns(curve: pd.DataFrame, frequency: str) -> pd.Series:
    if curve.empty:
        return pd.Series(dtype=float)
    work = curve.copy()
    work.index = pd.to_datetime(work["date"], format="mixed", utc=True)
    return (1.0 + work["daily_portfolio_return"]).resample(frequency).prod() - 1.0


def _drawdown_episode_metrics(curve: pd.DataFrame, initial_capital: float) -> dict:
    curve = economic_curve(curve)
    if curve.empty:
        return {
            "maximum_drawdown": 0.0,
            "max_drawdown_peak_date": None,
            "max_drawdown_trough_date": None,
            "max_drawdown_recovery_date": None,
            "max_drawdown_duration_days": 0,
            "longest_time_under_water_days": 0,
        }
    dates = pd.to_datetime(curve["date"], format="mixed", utc=True).reset_index(drop=True)
    equity = pd.to_numeric(curve["portfolio_equity"], errors="coerce").reset_index(drop=True)
    drawdown = pd.to_numeric(curve["drawdown"], errors="coerce").reset_index(drop=True)
    trough_index = int(drawdown.idxmin())
    if float(drawdown.iloc[trough_index]) >= -FLOAT_EPSILON:
        first_date = dates.iloc[0].date().isoformat()
        return {
            "maximum_drawdown": 0.0,
            "max_drawdown_peak_date": first_date,
            "max_drawdown_trough_date": first_date,
            "max_drawdown_recovery_date": first_date,
            "max_drawdown_duration_days": 0,
            "longest_time_under_water_days": 0,
        }
    peak_value = max(initial_capital, float(equity.iloc[: trough_index + 1].max()))
    if peak_value == initial_capital and (equity.iloc[: trough_index + 1] < peak_value).all():
        peak_date = dates.iloc[0]
    else:
        peak_index = int(equity.iloc[: trough_index + 1][equity.iloc[: trough_index + 1] == peak_value].index[-1])
        peak_date = dates.iloc[peak_index]
    recovery_candidates = equity.iloc[trough_index + 1 :]
    recovery_candidates = recovery_candidates.loc[recovery_candidates >= peak_value - FLOAT_EPSILON]
    recovery_date = dates.iloc[int(recovery_candidates.index[0])] if len(recovery_candidates) else None
    duration_end = recovery_date if recovery_date is not None else dates.iloc[-1]

    longest_underwater = 0
    running_peak = initial_capital
    episode_start = None
    for date, value in zip(dates, equity):
        if value >= running_peak - FLOAT_EPSILON:
            running_peak = max(running_peak, float(value))
            if episode_start is not None:
                longest_underwater = max(longest_underwater, int((date - episode_start).days))
                episode_start = None
        elif episode_start is None:
            episode_start = date
    if episode_start is not None:
        longest_underwater = max(longest_underwater, int((dates.iloc[-1] - episode_start).days))

    return {
        "maximum_drawdown": float(drawdown.min()),
        "max_drawdown_peak_date": peak_date.date().isoformat(),
        "max_drawdown_trough_date": dates.iloc[trough_index].date().isoformat(),
        "max_drawdown_recovery_date": recovery_date.date().isoformat() if recovery_date is not None else None,
        "max_drawdown_duration_days": int((duration_end - peak_date).days),
        "longest_time_under_water_days": longest_underwater,
    }


def summarize_portfolio_curve(
    strategy_key: str,
    strategy_label: str,
    curve: pd.DataFrame,
    *,
    initial_capital: float = PORTFOLIO_INITIAL_CAPITAL,
) -> dict:
    economic = economic_curve(curve)
    if economic.empty:
        return {
            **{column: None for column in PORTFOLIO_SUMMARY_COLUMNS},
            "strategy_key": strategy_key,
            "strategy_label": strategy_label,
            "portfolio_model": PORTFOLIO_MODEL_NAME,
            "initial_equity": initial_capital,
        }
    dates = pd.to_datetime(economic["date"], format="mixed", utc=True)
    daily = pd.to_numeric(economic["daily_portfolio_return"], errors="coerce").fillna(0.0)
    ending = float(economic["portfolio_equity"].iloc[-1])
    elapsed_days = max(int((dates.iloc[-1] - dates.iloc[0]).days), 0)
    years = elapsed_days / 365.25 if elapsed_days > 0 else 0.0
    total_return = ending / initial_capital - 1.0
    cagr = (ending / initial_capital) ** (1.0 / years) - 1.0 if years > 0 and ending > 0 else 0.0
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    annualized_volatility = daily_std * math.sqrt(252.0)
    sharpe = float(daily.mean() / daily_std * math.sqrt(252.0)) if daily_std > 0 else None
    downside_deviation = float(np.sqrt(np.mean(np.square(np.minimum(daily.to_numpy(dtype=float), 0.0)))))
    sortino = float(daily.mean() / downside_deviation * math.sqrt(252.0)) if downside_deviation > 0 else None
    drawdown_metrics = _drawdown_episode_metrics(economic, initial_capital)
    maximum_drawdown = drawdown_metrics["maximum_drawdown"]
    calmar = cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else None
    weekly = _period_returns(economic, "W-FRI")
    monthly = _period_returns(economic, "ME")
    gross_exposure = pd.to_numeric(economic["gross_exposure"], errors="coerce").fillna(0.0)
    active_positions = pd.to_numeric(economic["active_position_count"], errors="coerce").fillna(0.0)
    return {
        "strategy_key": strategy_key,
        "strategy_label": strategy_label,
        "portfolio_model": PORTFOLIO_MODEL_NAME,
        "portfolio_start_date": dates.iloc[0].date().isoformat(),
        "portfolio_end_date": dates.iloc[-1].date().isoformat(),
        "initial_equity": initial_capital,
        "ending_equity": ending,
        "total_portfolio_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        **drawdown_metrics,
        "ulcer_index": float(np.sqrt(np.mean(np.square(economic["drawdown"].to_numpy(dtype=float))))),
        "conditional_drawdown_at_risk_95": conditional_drawdown_at_risk(economic["drawdown"], 0.95),
        "worst_daily_return": float(daily.min()),
        "worst_weekly_return": float(weekly.min()) if len(weekly) else 0.0,
        "worst_monthly_return": float(monthly.min()) if len(monthly) else 0.0,
        "daily_expected_shortfall_95": expected_shortfall(daily, 0.95),
        "daily_expected_shortfall_99": expected_shortfall(daily, 0.99),
        "daily_return_skewness": float(daily.skew()) if len(daily) > 2 else None,
        "daily_return_excess_kurtosis": float(daily.kurt()) if len(daily) > 3 else None,
        "average_gross_exposure": float(gross_exposure.mean()),
        "median_gross_exposure": float(gross_exposure.median()),
        "maximum_gross_exposure": float(gross_exposure.max()),
        "average_active_positions": float(active_positions.mean()),
        "maximum_active_positions": int(active_positions.max()),
        "percent_days_in_cash": float((active_positions == 0).mean()),
        "annual_turnover": float(economic["turnover"].sum() / years) if years > 0 else 0.0,
        "total_transaction_cost": float(economic["transaction_cost_paid"].sum()),
    }


def build_spy_benchmark(panel: PricePanel, symbol: str = "SPY") -> dict:
    column = panel.symbol_to_column.get(symbol)
    if column is None or len(panel.dates) == 0:
        return {
            "status": "Not available",
            "reason": f"{symbol} adjusted close history is unavailable.",
            "symbol": symbol,
            "initial_equity": PORTFOLIO_INITIAL_CAPITAL,
            "curve_schema_version": PORTFOLIO_CURVE_SCHEMA_VERSION,
            "cdar_definition_version": CDAR_DEFINITION_VERSION,
            "publication_state": "not_generated",
            "series": [],
        }
    closes = pd.Series(panel.closes[:, column], index=panel.dates, dtype=float).dropna()
    closes = closes.loc[closes > 0]
    if closes.empty:
        return {
            "status": "Not available",
            "reason": f"{symbol} adjusted close history has no valid observations.",
            "symbol": symbol,
            "initial_equity": PORTFOLIO_INITIAL_CAPITAL,
            "curve_schema_version": PORTFOLIO_CURVE_SCHEMA_VERSION,
            "cdar_definition_version": CDAR_DEFINITION_VERSION,
            "publication_state": "not_generated",
            "series": [],
        }
    normalized = closes / closes.iloc[0] * PORTFOLIO_INITIAL_CAPITAL
    daily = normalized.pct_change().fillna(0.0)
    peak = normalized.cummax().clip(lower=PORTFOLIO_INITIAL_CAPITAL)
    drawdown = normalized / peak - 1.0
    return {
        "status": "Available",
        "reason": None,
        "symbol": symbol,
        "price_convention": "Yahoo Finance auto_adjust=True adjusted close proxy; dividends are vendor-adjusted where available.",
        "initial_equity": PORTFOLIO_INITIAL_CAPITAL,
        "curve_schema_version": PORTFOLIO_CURVE_SCHEMA_VERSION,
        "cdar_definition_version": CDAR_DEFINITION_VERSION,
        "publication_state": "benchmark",
        "initialization_timestamp": initialization_timestamp(closes.index[0]),
        "start_date": closes.index[0].date().isoformat(),
        "end_date": closes.index[-1].date().isoformat(),
        "series": [
            {
                "date": initialization_timestamp(closes.index[0]),
                "observation_type": INITIALIZATION_OBSERVATION,
                "benchmark_equity": PORTFOLIO_INITIAL_CAPITAL,
                "benchmark_daily_return": 0.0,
                "benchmark_drawdown": 0.0,
            },
            *[
            {
                "date": date.date().isoformat(),
                "observation_type": TRADING_SESSION_OBSERVATION,
                "benchmark_equity": float(equity),
                "benchmark_daily_return": float(day_return),
                "benchmark_drawdown": float(dd),
            }
            for date, equity, day_return, dd in zip(closes.index, normalized, daily, drawdown)
            ],
        ],
    }


def select_published_curve_keys(
    strategy_summary: pd.DataFrame,
    cap: int = PORTFOLIO_CURVE_CAP,
) -> list[str]:
    if cap <= 0 or strategy_summary.empty:
        return []
    ordered = strategy_summary.sort_values(
        ["qualification_rank", "strategy_key"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    )
    primary = ordered.iloc[0]["strategy_key"] if len(ordered) else None
    qualified = ordered.loc[ordered["qualification_tier"] == "Qualified", "strategy_key"].tolist()
    leaders = []
    qualified_rows = ordered.loc[ordered["qualification_tier"] == "Qualified"]
    for field in DIAGNOSTIC_LEADER_FIELDS:
        values = pd.to_numeric(qualified_rows.get(field), errors="coerce")
        if values.notna().any():
            maximum = values.max()
            leaders.extend(qualified_rows.loc[values == maximum, "strategy_key"].tolist())

    required_keys = {str(key) for key in [primary, *leaders, *qualified] if key}
    return [
        str(key)
        for key in ordered.loc[ordered["strategy_key"].astype(str).isin(required_keys), "strategy_key"].head(cap)
    ]


def _safe_records(frame: pd.DataFrame) -> list[dict]:
    return frame.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")


def build_portfolio_outputs(
    completed_trades: pd.DataFrame,
    open_positions: pd.DataFrame,
    prices: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    *,
    round_trip_cost: float,
    curve_cap: int = PORTFOLIO_CURVE_CAP,
) -> dict:
    entry_dates = []
    for frame in (completed_trades, open_positions):
        if frame is not None and not frame.empty and "entry_date" in frame:
            entry_dates.append(pd.to_datetime(frame["entry_date"], errors="coerce").dropna())
    combined_entry_dates = pd.concat(entry_dates, ignore_index=True) if entry_dates else pd.Series(dtype="datetime64[ns]")
    start_date = combined_entry_dates.min() if len(combined_entry_dates) else None
    panel = build_price_panel(prices, start_date=start_date)
    published_keys = select_published_curve_keys(strategy_summary, curve_cap)
    published_key_set = set(published_keys)

    completed_groups = completed_trades.groupby("strategy_key", sort=False) if completed_trades is not None and not completed_trades.empty else None
    open_groups = open_positions.groupby("strategy_key", sort=False) if open_positions is not None and not open_positions.empty else None
    completed_group_keys = set(completed_groups.groups) if completed_groups is not None else set()
    open_group_keys = set(open_groups.groups) if open_groups is not None else set()

    summaries = []
    curves = {}
    matrix_dates = (
        [initialization_timestamp(panel.dates[0]), *[date.date().isoformat() for date in panel.dates]]
        if len(panel.dates)
        else []
    )
    return_columns = {"date": matrix_dates}
    for strategy_row in strategy_summary.sort_values("qualification_rank").itertuples(index=False):
        parts = []
        if strategy_row.strategy_key in completed_group_keys:
            parts.append(completed_groups.get_group(strategy_row.strategy_key))
        if strategy_row.strategy_key in open_group_keys:
            parts.append(open_groups.get_group(strategy_row.strategy_key))
        lifecycles = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        curve = simulate_canonical_portfolio(
            lifecycles,
            panel,
            initial_capital=PORTFOLIO_INITIAL_CAPITAL,
            round_trip_cost=round_trip_cost,
        )
        summaries.append(summarize_portfolio_curve(
            strategy_row.strategy_key,
            strategy_row.strategy_label,
            curve,
            initial_capital=PORTFOLIO_INITIAL_CAPITAL,
        ))
        return_columns[strategy_row.strategy_key] = (
            curve["daily_portfolio_return"].to_numpy(dtype=float)
            if len(curve) == len(panel.dates) + 1
            else np.zeros(len(matrix_dates), dtype=float)
        )
        if strategy_row.strategy_key in published_key_set:
            curves[strategy_row.strategy_key] = curve

    return {
        "model": PORTFOLIO_MODEL_NAME,
        "curve_schema_version": PORTFOLIO_CURVE_SCHEMA_VERSION,
        "cdar_definition_version": CDAR_DEFINITION_VERSION,
        "initial_capital": PORTFOLIO_INITIAL_CAPITAL,
        "round_trip_cost": round_trip_cost,
        "half_turnover_cost": round_trip_cost / 2.0,
        "curve_cap": curve_cap,
        "published_keys": published_keys,
        "summary": pd.DataFrame(summaries, columns=PORTFOLIO_SUMMARY_COLUMNS),
        "curves": curves,
        "daily_returns": pd.DataFrame(return_columns),
        "qualification_ranks": {
            str(row.strategy_key): int(row.qualification_rank)
            for row in strategy_summary[["strategy_key", "qualification_rank"]].itertuples(index=False)
            if pd.notna(row.qualification_rank)
        },
        "benchmark": build_spy_benchmark(panel),
        "start_date": panel.dates[0].date().isoformat() if len(panel.dates) else None,
        "end_date": panel.dates[-1].date().isoformat() if len(panel.dates) else None,
    }


def write_portfolio_outputs(outputs: dict, data_path: Path) -> dict:
    data_path.mkdir(parents=True, exist_ok=True)
    summary_path = data_path / "backtest_portfolio_strategy_summary.csv"
    matrix_path = data_path / PORTFOLIO_DAILY_MATRIX_NAME
    benchmark_path = data_path / "backtest_benchmark_spy.json"
    manifest_path = data_path / "backtest_portfolio_curve_manifest.json"
    curve_directory = data_path / "backtest_portfolio_curves"
    curve_directory.mkdir(parents=True, exist_ok=True)

    for stale in curve_directory.glob("*.json"):
        stale.unlink()

    summary = outputs["summary"]
    summary.to_csv(summary_path, index=False)
    outputs["daily_returns"].to_csv(
        matrix_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    with open(benchmark_path, "w", encoding="utf-8") as file:
        json.dump(outputs["benchmark"], file, ensure_ascii=False, separators=(",", ":"))

    summary_by_key = summary.set_index("strategy_key").to_dict(orient="index") if not summary.empty else {}
    strategy_entries = []
    for rank, strategy_key in enumerate(outputs["published_keys"], start=1):
        filename = f"{strategy_key}.json"
        curve = outputs["curves"][strategy_key]
        curve_payload = {
            "status": "Available",
            "portfolio_model": outputs["model"],
            "curve_schema_version": outputs["curve_schema_version"],
            "cdar_definition_version": outputs["cdar_definition_version"],
            "publication_state": "published",
            "strategy_key": strategy_key,
            "initial_equity": outputs["initial_capital"],
            "series": _safe_records(curve[PORTFOLIO_CURVE_COLUMNS]),
        }
        with open(curve_directory / filename, "w", encoding="utf-8") as file:
            json.dump(curve_payload, file, ensure_ascii=False, separators=(",", ":"))
        strategy_entries.append({
            "strategy_key": strategy_key,
            "qualification_rank": outputs.get("qualification_ranks", {}).get(strategy_key, rank),
            "file": filename,
            "summary": {
                key: (None if pd.isna(value) else value)
                for key, value in summary_by_key.get(strategy_key, {}).items()
            },
        })

    manifest = {
        "status": "Available",
        "reason": None,
        "portfolio_model": outputs["model"],
        "curve_schema_version": outputs["curve_schema_version"],
        "cdar_definition_version": outputs["cdar_definition_version"],
        "publication_state": "bounded",
        "initialization_timestamp": (
            outputs["daily_returns"]["date"].iloc[0] if len(outputs["daily_returns"]) else None
        ),
        "initial_equity": outputs["initial_capital"],
        "start_date": outputs["start_date"],
        "end_date": outputs["end_date"],
        "curve_cap": outputs["curve_cap"],
        "published_curve_count": len(strategy_entries),
        "daily_return_matrix": {
            "file": PORTFOLIO_DAILY_MATRIX_NAME,
            "format": "gzip-compressed CSV",
            "row_count": int(len(outputs["daily_returns"])),
            "strategy_count": max(int(len(outputs["daily_returns"].columns) - 1), 0),
            "size_bytes": int(matrix_path.stat().st_size),
        },
        "strategies": strategy_entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest


def write_unavailable_portfolio_outputs(data_path: Path, reason: str) -> dict:
    data_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=PORTFOLIO_SUMMARY_COLUMNS).to_csv(
        data_path / "backtest_portfolio_strategy_summary.csv", index=False
    )
    with open(data_path / PORTFOLIO_DAILY_MATRIX_NAME, "wb") as raw_file:
        with gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", compresslevel=9, mtime=0) as file:
            file.write(b"date\n")
    benchmark = {
        "status": "Not available",
        "reason": reason,
        "symbol": "SPY",
        "initial_equity": PORTFOLIO_INITIAL_CAPITAL,
        "curve_schema_version": PORTFOLIO_CURVE_SCHEMA_VERSION,
        "cdar_definition_version": CDAR_DEFINITION_VERSION,
        "publication_state": "not_generated",
        "series": [],
    }
    manifest = {
        "status": "Not available",
        "reason": reason,
        "portfolio_model": PORTFOLIO_MODEL_NAME,
        "curve_schema_version": PORTFOLIO_CURVE_SCHEMA_VERSION,
        "cdar_definition_version": CDAR_DEFINITION_VERSION,
        "publication_state": "not_generated",
        "initial_equity": PORTFOLIO_INITIAL_CAPITAL,
        "curve_cap": PORTFOLIO_CURVE_CAP,
        "published_curve_count": 0,
        "strategies": [],
    }
    with open(data_path / "backtest_benchmark_spy.json", "w", encoding="utf-8") as file:
        json.dump(benchmark, file, ensure_ascii=False, indent=2)
    with open(data_path / "backtest_portfolio_curve_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    (data_path / "backtest_portfolio_curves").mkdir(parents=True, exist_ok=True)
    return manifest
