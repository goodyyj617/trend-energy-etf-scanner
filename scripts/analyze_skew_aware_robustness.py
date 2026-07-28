"""Bounded Phase B skew-aware portfolio robustness research.

This module reads only the committed canonical portfolio outputs.  It does not
download prices, simulate trades, or alter production qualification/ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
TASKS = ROOT / "docs" / "tasks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_canonical_portfolio_outputs import audit as canonical_audit
from scripts.audit_canonical_portfolio_outputs import critical_failures
from src.backtest import (
    _attach_parameter_stability,
    _attach_robustness_tiers,
    _attach_time_stability,
    _gate_fields,
    rank_strategy_summary,
)


INITIAL_EQUITY = 1000.0
TRADING_DAYS = 252.0
SEED = 20260728
PRIMARY_BLOCK_LENGTH = 20
PRIMARY_BOOTSTRAP_PATHS = 5000
PRIMARY_CANDIDATE_PATHS = 10000
BEHAVIOR_TOLERANCE = 1e-12
NUMERIC_ATOL = 1e-9
NUMERIC_RTOL = 1e-9
POSITIVE_SHARPE_BENCHMARK = 0.50
NORMAL = NormalDist()


SUMMARY_OUTPUT = TASKS / "skew_aware_robustness_summary.csv"
BOOTSTRAP_OUTPUT = TASKS / "skew_aware_bootstrap_summary.csv"
PSR_DSR_OUTPUT = TASKS / "skew_aware_psr_dsr_summary.csv"
CONCENTRATION_OUTPUT = TASKS / "skew_aware_concentration_summary.csv"
CRASH_EPISODE_OUTPUT = TASKS / "crash_avoidance_episode_summary.csv"
CRASH_STRATEGY_OUTPUT = TASKS / "crash_avoidance_strategy_summary.csv"
PARETO_OUTPUT = TASKS / "skew_aware_pareto_candidates.csv"
REPORT_OUTPUT = TASKS / "skew_aware_robustness_analysis.md"


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return array[np.isfinite(array)]


def unbiased_skewness(values: Iterable[float]) -> float:
    x = finite_array(values)
    n = len(x)
    if n < 3:
        return math.nan
    std = float(np.std(x, ddof=1))
    if std <= 0:
        return 0.0
    z3 = float(np.sum(((x - float(np.mean(x))) / std) ** 3))
    return float(n * z3 / ((n - 1) * (n - 2)))


def unbiased_excess_kurtosis(values: Iterable[float]) -> float:
    x = finite_array(values)
    n = len(x)
    if n < 4:
        return math.nan
    std = float(np.std(x, ddof=1))
    if std <= 0:
        return 0.0
    z4 = float(np.sum(((x - float(np.mean(x))) / std) ** 4))
    first = n * (n + 1) * z4 / ((n - 1) * (n - 2) * (n - 3))
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float(first - correction)


def expected_shortfall(values: Iterable[float], confidence: float = 0.95) -> float:
    x = finite_array(values)
    if not len(x):
        return 0.0
    cutoff = float(np.quantile(x, 1.0 - confidence, method="linear"))
    tail = x[x <= cutoff]
    return float(np.mean(tail)) if len(tail) else cutoff


def conditional_drawdown_at_risk(
    drawdowns: Iterable[float], confidence: float = 0.95
) -> float:
    x = finite_array(drawdowns)
    if not len(x):
        return 0.0
    if np.any(x > NUMERIC_ATOL):
        raise ValueError("drawdowns cannot be materially positive")
    x = np.minimum(x, 0.0)
    ordered = np.sort(x)
    k = max(1, len(ordered) - math.floor(confidence * len(ordered) + 1e-12))
    return float(np.mean(ordered[:k]))


def equity_from_returns(
    returns: Iterable[float], initial_equity: float = INITIAL_EQUITY
) -> np.ndarray:
    x = finite_array(returns)
    if np.any(x <= -1):
        raise ValueError("return at or below -100%")
    return initial_equity * np.cumprod(1.0 + x)


def drawdown_from_returns(returns: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    equity = equity_from_returns(returns)
    peak = np.maximum.accumulate(np.maximum(equity, INITIAL_EQUITY))
    return equity, equity / peak - 1.0


def longest_underwater_observations(drawdowns: Iterable[float]) -> int:
    longest = 0
    current = 0
    for value in finite_array(drawdowns):
        if value < -NUMERIC_ATOL:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _iso_date(value: pd.Timestamp | None) -> str | None:
    return value.date().isoformat() if value is not None else None


def drawdown_episodes(
    dates: pd.DatetimeIndex,
    returns: Iterable[float],
    *,
    exposure: np.ndarray | None = None,
    spy_threshold_episodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    x = finite_array(returns)
    if len(x) != len(dates):
        raise ValueError("date/return length mismatch")
    equity = equity_from_returns(x)
    episodes: list[dict[str, Any]] = []
    running_peak = INITIAL_EQUITY
    peak_index = -1
    start_peak_index: int | None = None
    trough_index: int | None = None
    trough_depth = 0.0

    for index, value in enumerate(equity):
        if value >= running_peak - NUMERIC_ATOL:
            if start_peak_index is not None and trough_index is not None:
                recovery_index = index
                episodes.append(
                    _episode_record(
                        dates,
                        equity,
                        start_peak_index,
                        trough_index,
                        recovery_index,
                        trough_depth,
                        exposure,
                        spy_threshold_episodes,
                    )
                )
                start_peak_index = None
                trough_index = None
                trough_depth = 0.0
            if value > running_peak + NUMERIC_ATOL or peak_index < 0:
                running_peak = max(float(value), running_peak)
                peak_index = index
            continue

        depth = float(value / running_peak - 1.0)
        if start_peak_index is None:
            start_peak_index = peak_index
            trough_index = index
            trough_depth = depth
        elif depth < trough_depth:
            trough_index = index
            trough_depth = depth

    if start_peak_index is not None and trough_index is not None:
        episodes.append(
            _episode_record(
                dates,
                equity,
                start_peak_index,
                trough_index,
                None,
                trough_depth,
                exposure,
                spy_threshold_episodes,
            )
        )
    return episodes


def _episode_record(
    dates: pd.DatetimeIndex,
    equity: np.ndarray,
    peak_index: int,
    trough_index: int,
    recovery_index: int | None,
    depth: float,
    exposure: np.ndarray | None,
    spy_threshold_episodes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    peak_date = dates[max(peak_index, 0)]
    trough_date = dates[trough_index]
    end_index = recovery_index if recovery_index is not None else len(dates) - 1
    recovery_date = dates[recovery_index] if recovery_index is not None else None
    overlap_thresholds: set[int] = set()
    for episode in spy_threshold_episodes or []:
        spy_start = pd.Timestamp(episode["prior_peak_date"], tz="UTC")
        spy_end = pd.Timestamp(episode["recovery_date"] or episode["data_end_date"], tz="UTC")
        if peak_date <= spy_end and dates[end_index] >= spy_start:
            overlap_thresholds.add(int(episode["threshold_percent"]))
    return {
        "peak_index": int(max(peak_index, 0)),
        "trough_index": int(trough_index),
        "recovery_index": int(recovery_index) if recovery_index is not None else None,
        "peak_date": _iso_date(peak_date),
        "trough_date": _iso_date(trough_date),
        "recovery_date": _iso_date(recovery_date),
        "is_open": recovery_index is None,
        "depth": float(depth),
        "peak_to_trough_days": int((trough_date - peak_date).days),
        "trough_to_recovery_days": (
            int((recovery_date - trough_date).days) if recovery_date is not None else None
        ),
        "time_under_water_days": int((dates[end_index] - peak_date).days),
        "exposure_at_peak": (
            float(exposure[max(peak_index, 0)]) if exposure is not None else math.nan
        ),
        "exposure_at_trough": (
            float(exposure[trough_index]) if exposure is not None else math.nan
        ),
        "overlap_spy_thresholds": ";".join(map(str, sorted(overlap_thresholds))),
    }


def period_compounded_returns(
    returns: pd.Series, frequency: str
) -> pd.Series:
    return (1.0 + returns).resample(frequency).prod() - 1.0


def baseline_metrics(
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    *,
    exposure: np.ndarray | None = None,
    active_positions: np.ndarray | None = None,
    transaction_cost: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    spy_episodes: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(returns) != len(dates):
        raise ValueError("date/return length mismatch")
    if len(returns) < 2:
        raise ValueError("insufficient daily returns")
    equity, drawdowns = drawdown_from_returns(returns)
    years = max((dates[-1] - dates[0]).days / 365.25, 1.0 / 365.25)
    daily_std = float(np.std(returns, ddof=1))
    downside = np.minimum(returns, 0.0)
    upside = np.maximum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) * math.sqrt(TRADING_DAYS)
    upside_semidev = float(np.sqrt(np.mean(np.square(upside)))) * math.sqrt(TRADING_DAYS)
    annual_arithmetic = float(np.mean(returns) * TRADING_DAYS)
    annual_geometric = float(np.expm1(np.mean(np.log1p(returns)) * TRADING_DAYS))
    cagr = float((equity[-1] / INITIAL_EQUITY) ** (1.0 / years) - 1.0)
    mdd = float(np.min(drawdowns))
    ulcer = float(np.sqrt(np.mean(np.square(drawdowns))))
    daily_series = pd.Series(returns, index=dates)
    monthly = period_compounded_returns(daily_series, "ME")
    yearly = period_compounded_returns(daily_series, "YE")
    rolling5 = np.expm1(
        pd.Series(np.log1p(returns)).rolling(5).sum().dropna().to_numpy(dtype=float)
    )
    rolling21 = np.expm1(
        pd.Series(np.log1p(returns)).rolling(21).sum().dropna().to_numpy(dtype=float)
    )
    negative = returns[returns < 0]
    episodes = drawdown_episodes(
        dates, returns, exposure=exposure, spy_threshold_episodes=spy_episodes
    )
    ordered_episodes = sorted(
        episodes, key=lambda row: (row["depth"], row["peak_date"], row["trough_date"])
    )
    completed = [row for row in episodes if not row["is_open"]]
    gross_gain = float(np.sum(returns[returns > 0]))
    gross_loss = abs(float(np.sum(returns[returns < 0])))
    omega = gross_gain / gross_loss if gross_loss > 0 else math.nan
    gain_to_loss = (
        float(np.mean(returns[returns > 0])) / abs(float(np.mean(returns[returns < 0])))
        if np.any(returns > 0) and np.any(returns < 0)
        else math.nan
    )
    metrics: dict[str, Any] = {
        "ending_equity": float(equity[-1]),
        "total_return": float(equity[-1] / INITIAL_EQUITY - 1.0),
        "cagr": cagr,
        "annualized_arithmetic_mean_return": annual_arithmetic,
        "annualized_geometric_return": annual_geometric,
        "annualized_volatility": daily_std * math.sqrt(TRADING_DAYS),
        "daily_return_skewness": unbiased_skewness(returns),
        "daily_return_excess_kurtosis": unbiased_excess_kurtosis(returns),
        "downside_deviation": downside_dev,
        "upside_semideviation": upside_semidev,
        "omega_ratio_zero": omega,
        "gain_to_loss_ratio": gain_to_loss,
        "positive_day_ratio": float(np.mean(returns > 0)),
        "positive_month_ratio": float(np.mean(monthly > 0)),
        "positive_calendar_year_ratio": float(np.mean(yearly > 0)),
        "sharpe_ratio": (
            float(np.mean(returns) / daily_std * math.sqrt(TRADING_DAYS))
            if daily_std > 0
            else math.nan
        ),
        "sortino_ratio": (
            float(np.mean(returns) * math.sqrt(TRADING_DAYS) / (downside_dev / math.sqrt(TRADING_DAYS)))
            if downside_dev > 0
            else math.nan
        ),
        "calmar_ratio": cagr / abs(mdd) if mdd < 0 else math.nan,
        "martin_ratio": cagr / ulcer if ulcer > 0 else math.nan,
        "maximum_drawdown": mdd,
        "second_largest_drawdown": (
            float(ordered_episodes[1]["depth"]) if len(ordered_episodes) > 1 else 0.0
        ),
        "third_largest_drawdown": (
            float(ordered_episodes[2]["depth"]) if len(ordered_episodes) > 2 else 0.0
        ),
        "cdar_90": conditional_drawdown_at_risk(drawdowns, 0.90),
        "cdar_95": conditional_drawdown_at_risk(drawdowns, 0.95),
        "cdar_99": conditional_drawdown_at_risk(drawdowns, 0.99),
        "ulcer_index": ulcer,
        "longest_time_under_water_days": max(
            [int(row["time_under_water_days"]) for row in episodes] or [0]
        ),
        "median_completed_recovery_days": (
            float(np.median([row["trough_to_recovery_days"] for row in completed]))
            if completed
            else math.nan
        ),
        "maximum_completed_recovery_days": (
            max([int(row["trough_to_recovery_days"]) for row in completed])
            if completed
            else math.nan
        ),
        "proportion_days_below_prior_peak": float(np.mean(drawdowns < -NUMERIC_ATOL)),
        "drawdown_episode_count": len(episodes),
        "drawdown_episodes_over_5pct": sum(row["depth"] <= -0.05 for row in episodes),
        "drawdown_episodes_over_10pct": sum(row["depth"] <= -0.10 for row in episodes),
        "drawdown_episodes_over_15pct": sum(row["depth"] <= -0.15 for row in episodes),
        "drawdown_episodes_over_20pct": sum(row["depth"] <= -0.20 for row in episodes),
        "drawdown_episodes_over_30pct": sum(row["depth"] <= -0.30 for row in episodes),
        "worst_day": float(np.min(returns)),
        "worst_rolling_5d_return": float(np.min(rolling5)),
        "worst_rolling_21d_return": float(np.min(rolling21)),
        "expected_shortfall_95": expected_shortfall(returns, 0.95),
        "expected_shortfall_99": expected_shortfall(returns, 0.99),
        "downside_tail_skewness": (
            unbiased_skewness(negative) if len(negative) >= 20 else math.nan
        ),
    }
    if exposure is not None:
        metrics.update(
            {
                "average_gross_exposure": float(np.mean(exposure)),
                "median_gross_exposure": float(np.median(exposure)),
                "maximum_gross_exposure": float(np.max(exposure)),
                "percent_days_fully_in_cash": float(np.mean(exposure <= NUMERIC_ATOL)),
                "average_active_positions": float(np.mean(active_positions)),
                "maximum_active_positions": int(np.max(active_positions)),
            }
        )
    if turnover is not None and transaction_cost is not None:
        rebalances = turnover > NUMERIC_ATOL
        metrics.update(
            {
                "annual_turnover": float(np.sum(turnover) / years),
                "total_transaction_cost": float(np.sum(transaction_cost)),
                "transaction_cost_percent_initial_capital": float(
                    np.sum(transaction_cost) / INITIAL_EQUITY
                ),
                "membership_change_rebalance_days": int(np.sum(rebalances)),
                "average_turnover_per_rebalance": (
                    float(np.mean(turnover[rebalances])) if np.any(rebalances) else 0.0
                ),
                "transaction_cost_per_year": float(np.sum(transaction_cost) / years),
            }
        )
    for rank in range(5):
        episode = ordered_episodes[rank] if rank < len(ordered_episodes) else None
        prefix = f"drawdown_{rank + 1}"
        for field in [
            "peak_date",
            "trough_date",
            "recovery_date",
            "depth",
            "peak_to_trough_days",
            "trough_to_recovery_days",
            "time_under_water_days",
            "exposure_at_peak",
            "exposure_at_trough",
            "overlap_spy_thresholds",
        ]:
            metrics[f"{prefix}_{field}"] = episode[field] if episode else math.nan
    return metrics, episodes


def psr_probability(
    annualized_sharpe: float,
    benchmark_annualized_sharpe: float,
    sample_length: int,
    skewness: float,
    excess_kurtosis: float,
) -> float:
    if sample_length < 2 or not all(
        np.isfinite(
            [annualized_sharpe, benchmark_annualized_sharpe, skewness, excess_kurtosis]
        )
    ):
        return math.nan
    observed = annualized_sharpe / math.sqrt(TRADING_DAYS)
    benchmark = benchmark_annualized_sharpe / math.sqrt(TRADING_DAYS)
    kurtosis = excess_kurtosis + 3.0
    variance_term = 1.0 - skewness * observed + ((kurtosis - 1.0) / 4.0) * observed**2
    if variance_term <= 0:
        return math.nan
    z = (observed - benchmark) * math.sqrt(sample_length - 1) / math.sqrt(variance_term)
    return float(NORMAL.cdf(z))


def expected_maximum_sharpe(
    sharpe_standard_deviation: float, trial_count: int
) -> float:
    if trial_count <= 1 or sharpe_standard_deviation <= 0:
        return 0.0
    gamma = 0.5772156649015329
    first = NORMAL.inv_cdf(1.0 - 1.0 / trial_count)
    second = NORMAL.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
    return float(sharpe_standard_deviation * ((1.0 - gamma) * first + gamma * second))


def normalized_return_hash(
    returns: Iterable[float], tolerance: float = BEHAVIOR_TOLERANCE
) -> str:
    x = finite_array(returns)
    if tolerance <= 0:
        payload = x.astype("<f8", copy=False).tobytes()
    else:
        quantized = np.rint(x / tolerance).astype("<i8")
        payload = quantized.tobytes()
    return hashlib.sha256(payload).hexdigest()


def behavior_groups(
    return_matrix: pd.DataFrame, tolerance: float = BEHAVIOR_TOLERANCE
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key in sorted(return_matrix.columns):
        groups[normalized_return_hash(return_matrix[key].to_numpy(dtype=float), tolerance)].append(
            key
        )
    ordered = sorted(groups.items(), key=lambda item: (item[1][0], item[0]))
    rows = []
    mapping: dict[str, list[str]] = {}
    for number, (path_hash, keys) in enumerate(ordered, start=1):
        group_id = f"B{number:03d}"
        mapping[group_id] = keys
        for key in keys:
            rows.append(
                {
                    "strategy_key": key,
                    "behavior_group_id": group_id,
                    "behavior_group_size": len(keys),
                    "behavior_group_representative": keys[0],
                    "normalized_return_hash": path_hash,
                }
            )
    return pd.DataFrame(rows).sort_values("strategy_key").reset_index(drop=True), mapping


def stable_rng(strategy_key: str, mean_block_length: int) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{SEED}|{strategy_key}|{mean_block_length}".encode("utf-8")
    ).digest()
    words = np.frombuffer(digest[:16], dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([SEED, *map(int, words)]))


def stationary_bootstrap_indices(
    observation_count: int,
    path_count: int,
    mean_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if observation_count <= 0 or path_count <= 0 or mean_block_length <= 0:
        raise ValueError("positive observation/path/block counts required")
    restart_probability = 1.0 / float(mean_block_length)
    indices = np.empty((path_count, observation_count), dtype=np.int32)
    indices[:, 0] = rng.integers(0, observation_count, size=path_count, dtype=np.int32)
    for column in range(1, observation_count):
        restart = rng.random(path_count) < restart_probability
        continuation = (indices[:, column - 1] + 1) % observation_count
        fresh = rng.integers(0, observation_count, size=path_count, dtype=np.int32)
        indices[:, column] = np.where(restart, fresh, continuation)
    return indices


def _bootstrap_chunk_metrics(
    strategy_returns: np.ndarray,
    spy_returns: np.ndarray,
    indices: np.ndarray,
    years: float,
) -> dict[str, np.ndarray]:
    sample = strategy_returns[indices]
    spy_sample = spy_returns[indices]
    log_sample = np.log1p(sample)
    cumulative_log = np.cumsum(log_sample, axis=1)
    ending = INITIAL_EQUITY * np.exp(cumulative_log[:, -1])
    cagr = np.expm1(cumulative_log[:, -1] / years)
    spy_cagr = np.expm1(np.sum(np.log1p(spy_sample), axis=1) / years)
    volatility = np.std(sample, axis=1, ddof=1) * math.sqrt(TRADING_DAYS)
    running_peak_log = np.maximum.accumulate(np.maximum(cumulative_log, 0.0), axis=1)
    drawdown = np.expm1(cumulative_log - running_peak_log)
    mdd_magnitude = -np.min(drawdown, axis=1)
    tail_count = max(
        1,
        drawdown.shape[1]
        - math.floor(0.95 * drawdown.shape[1] + 1e-12),
    )
    cdar = -np.mean(
        np.partition(drawdown, tail_count - 1, axis=1)[:, :tail_count], axis=1
    )
    cutoff = np.quantile(sample, 0.05, axis=1, method="linear")
    es_mask = sample <= cutoff[:, None]
    es = -np.sum(np.where(es_mask, sample, 0.0), axis=1) / np.sum(es_mask, axis=1)
    calmar = np.divide(
        cagr,
        mdd_magnitude,
        out=np.full_like(cagr, np.nan),
        where=mdd_magnitude > 0,
    )
    current = np.zeros(len(sample), dtype=np.int32)
    longest = np.zeros(len(sample), dtype=np.int32)
    for column in range(drawdown.shape[1]):
        current = np.where(drawdown[:, column] < -NUMERIC_ATOL, current + 1, 0)
        longest = np.maximum(longest, current)
    return {
        "ending_equity": ending,
        "cagr": cagr,
        "spy_cagr": spy_cagr,
        "volatility": volatility,
        "mdd_magnitude": mdd_magnitude,
        "cdar_magnitude": cdar,
        "es95_magnitude": es,
        "calmar": calmar,
        "longest_underwater_observations": longest.astype(float),
    }


def stationary_bootstrap_summary(
    strategy_key: str,
    strategy_returns: np.ndarray,
    spy_returns: np.ndarray,
    years: float,
    *,
    mean_block_length: int = PRIMARY_BLOCK_LENGTH,
    path_count: int = PRIMARY_BOOTSTRAP_PATHS,
    chunk_size: int = 250,
) -> dict[str, Any]:
    if len(strategy_returns) != len(spy_returns):
        raise ValueError("strategy/SPY bootstrap length mismatch")
    rng = stable_rng(strategy_key, mean_block_length)
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    all_indices = stationary_bootstrap_indices(
        len(strategy_returns), path_count, mean_block_length, rng
    )
    for start in range(0, path_count, chunk_size):
        indices = all_indices[start : start + chunk_size]
        values = _bootstrap_chunk_metrics(
            strategy_returns, spy_returns, indices, years
        )
        for field, array in values.items():
            collected[field].append(array)
    result = {field: np.concatenate(arrays) for field, arrays in collected.items()}
    ending = result["ending_equity"]
    cagr = result["cagr"]
    mdd = result["mdd_magnitude"]
    cdar = result["cdar_magnitude"]
    calmar = result["calmar"]
    row: dict[str, Any] = {
        "strategy_key": strategy_key,
        "seed": SEED,
        "mean_block_length": mean_block_length,
        "bootstrap_paths": path_count,
        "path_observations": len(strategy_returns),
        "prob_ending_equity_above_1000": float(np.mean(ending > INITIAL_EQUITY)),
        "prob_cagr_above_zero": float(np.mean(cagr > 0)),
        "prob_cagr_above_spy": float(np.mean(cagr > result["spy_cagr"])),
        "prob_mdd_above_10pct": float(np.mean(mdd > 0.10)),
        "prob_mdd_above_15pct": float(np.mean(mdd > 0.15)),
        "prob_mdd_above_20pct": float(np.mean(mdd > 0.20)),
        "prob_mdd_above_30pct": float(np.mean(mdd > 0.30)),
        "prob_mdd_above_40pct": float(np.mean(mdd > 0.40)),
        "prob_ending_equity_below_750": float(np.mean(ending < 750.0)),
        "prob_ending_equity_below_500": float(np.mean(ending < 500.0)),
        "calmar_5pct": float(np.nanquantile(calmar, 0.05)),
    }
    for field, values, quantiles in [
        ("ending_equity", ending, [5, 25, 50, 75, 95]),
        ("cagr", cagr, [5, 25, 50, 75, 95]),
        ("mdd_magnitude", mdd, [50, 75, 90, 95, 99]),
        ("cdar_magnitude", cdar, [50, 90, 95, 99]),
        ("es95_magnitude", result["es95_magnitude"], [50, 90, 95, 99]),
        (
            "longest_underwater_observations",
            result["longest_underwater_observations"],
            [50, 90, 95, 99],
        ),
    ]:
        for quantile in quantiles:
            row[f"{field}_{quantile}pct"] = float(
                np.nanquantile(values, quantile / 100.0)
            )
    return row


def stress_path_metrics(
    returns: np.ndarray, *, years: float | None = None
) -> dict[str, float]:
    if not len(returns):
        return {
            "ending_equity": INITIAL_EQUITY,
            "cagr": 0.0,
            "maximum_drawdown": 0.0,
            "cdar_95": 0.0,
            "sharpe_ratio": math.nan,
            "calmar_ratio": math.nan,
        }
    equity, drawdown = drawdown_from_returns(returns)
    years = years if years is not None else len(returns) / TRADING_DAYS
    years = max(float(years), 1.0 / TRADING_DAYS)
    cagr = float((equity[-1] / INITIAL_EQUITY) ** (1.0 / years) - 1.0)
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    mdd = float(np.min(drawdown))
    return {
        "ending_equity": float(equity[-1]),
        "cagr": cagr,
        "maximum_drawdown": mdd,
        "cdar_95": conditional_drawdown_at_risk(drawdown, 0.95),
        "sharpe_ratio": (
            float(np.mean(returns) / std * math.sqrt(TRADING_DAYS))
            if std > 0
            else math.nan
        ),
        "calmar_ratio": cagr / abs(mdd) if mdd < 0 else math.nan,
    }


def concentration_and_stress_rows(
    strategy_key: str,
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logs = np.log1p(returns)
    total_log_growth = float(np.sum(logs))
    daily_order = np.argsort(logs)[::-1]
    series = pd.Series(returns, index=dates)
    monthly_returns = period_compounded_returns(series, "ME")
    yearly_returns = period_compounded_returns(series, "YE")
    monthly_logs = np.log1p(monthly_returns)
    yearly_logs = np.log1p(yearly_returns)

    def contribution(value: float) -> float:
        return float(value / total_log_growth) if abs(total_log_growth) > 1e-15 else math.nan

    top_months = monthly_logs.sort_values(ascending=False)
    top_years = yearly_logs.sort_values(ascending=False)
    concentration = {
        "strategy_key": strategy_key,
        "total_log_growth": total_log_growth,
        "best_day_log_contribution": float(logs[daily_order[0]]),
        "best_day_contribution_ratio": contribution(float(logs[daily_order[0]])),
        "best_5_days_log_contribution": float(np.sum(logs[daily_order[:5]])),
        "best_5_days_contribution_ratio": contribution(
            float(np.sum(logs[daily_order[:5]]))
        ),
        "best_10_days_log_contribution": float(np.sum(logs[daily_order[:10]])),
        "best_10_days_contribution_ratio": contribution(
            float(np.sum(logs[daily_order[:10]]))
        ),
        "best_month": top_months.index[0].strftime("%Y-%m"),
        "best_month_log_contribution": float(top_months.iloc[0]),
        "best_month_contribution_ratio": contribution(float(top_months.iloc[0])),
        "top_3_months_log_contribution": float(top_months.iloc[:3].sum()),
        "top_3_months_contribution_ratio": contribution(float(top_months.iloc[:3].sum())),
        "best_calendar_year": int(top_years.index[0].year),
        "best_year_log_contribution": float(top_years.iloc[0]),
        "best_year_contribution_ratio": contribution(float(top_years.iloc[0])),
        "top_2_years_log_contribution": float(top_years.iloc[:2].sum()),
        "top_2_years_contribution_ratio": contribution(float(top_years.iloc[:2].sum())),
        "worst_day_log_contribution": float(np.min(logs)),
        "worst_day_contribution_ratio": contribution(float(np.min(logs))),
        "worst_month": monthly_logs.idxmin().strftime("%Y-%m"),
        "worst_month_log_contribution": float(monthly_logs.min()),
        "worst_month_contribution_ratio": contribution(float(monthly_logs.min())),
        "worst_calendar_year": int(yearly_logs.idxmin().year),
        "worst_year_log_contribution": float(yearly_logs.min()),
        "worst_year_contribution_ratio": contribution(float(yearly_logs.min())),
    }
    rows: list[dict[str, Any]] = []

    def append_scenario(
        scenario_type: str,
        scenario_label: str,
        stressed: np.ndarray,
        years: float | None = None,
    ) -> None:
        rows.append(
            {
                "strategy_key": strategy_key,
                "scenario_type": scenario_type,
                "scenario_label": scenario_label,
                "removed_observations": int(len(returns) - len(stressed)),
                **stress_path_metrics(stressed, years=years),
            }
        )

    observed_years = max((dates[-1] - dates[0]).days / 365.25, 1.0 / 365.25)
    append_scenario("baseline", "observed", returns, years=observed_years)
    one = returns.copy()
    one[daily_order[0]] = 0.0
    append_scenario(
        "remove_best_day",
        dates[daily_order[0]].date().isoformat(),
        one,
        years=observed_years,
    )
    five = returns.copy()
    five[daily_order[:5]] = 0.0
    append_scenario(
        "zero_best_5_days",
        "top_5_daily_returns",
        five,
        years=observed_years,
    )
    ten = returns.copy()
    ten[daily_order[:10]] = 0.0
    append_scenario(
        "zero_best_10_days",
        "top_10_daily_returns",
        ten,
        years=observed_years,
    )
    best_month_period = monthly_returns.idxmax().strftime("%Y-%m")
    month_mask = dates.strftime("%Y-%m") == best_month_period
    month_zero = returns.copy()
    month_zero[month_mask] = 0.0
    append_scenario(
        "zero_best_month", best_month_period, month_zero, years=observed_years
    )
    best_year = int(yearly_returns.idxmax().year)
    year_zero = returns.copy()
    year_zero[dates.year == best_year] = 0.0
    append_scenario(
        "zero_best_calendar_year",
        str(best_year),
        year_zero,
        years=observed_years,
    )
    for year in sorted(set(dates.year)):
        if year in (dates[0].year, dates[-1].year):
            continue
        mask = dates.year != year
        append_scenario(
            "leave_one_full_calendar_year_out",
            str(year),
            returns[mask],
            years=float(np.sum(mask) / TRADING_DAYS),
        )
    for number, episode in enumerate(episodes, start=1):
        start = int(episode["peak_index"]) + 1
        stop = (
            int(episode["recovery_index"]) + 1
            if episode["recovery_index"] is not None
            else len(returns)
        )
        keep = np.ones(len(returns), dtype=bool)
        keep[start:stop] = False
        append_scenario(
            "leave_drawdown_episode_out",
            f"{number}:{episode['peak_date']}:{episode['trough_date']}:{episode['recovery_date'] or 'open'}",
            returns[keep],
            years=float(np.sum(keep) / TRADING_DAYS),
        )
    return concentration, rows


def cost_sensitivity_rows(
    strategy_key: str,
    returns: np.ndarray,
    equity: np.ndarray,
    daily_cost: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prior_equity = np.concatenate(([INITIAL_EQUITY], equity[:-1]))
    cost_rate = np.divide(
        daily_cost,
        prior_equity,
        out=np.zeros_like(daily_cost),
        where=prior_equity > 0,
    )
    gross_returns = returns + cost_rate
    pre_cost_ending = float(equity_from_returns(gross_returns)[-1])
    rows = []
    for factor in [1.0, 1.5, 2.0]:
        adjusted = gross_returns - factor * cost_rate
        metrics = stress_path_metrics(adjusted)
        rows.append(
            {
                "strategy_key": strategy_key,
                "cost_multiplier": factor,
                **metrics,
                "ending_equity_above_1000": metrics["ending_equity"] > INITIAL_EQUITY,
            }
        )
    by_factor = {row["cost_multiplier"]: row for row in rows}
    total_cost = float(np.sum(daily_cost))
    observed_profit = float(equity[-1] - INITIAL_EQUITY)
    gross_gain = pre_cost_ending - INITIAL_EQUITY
    summary = {
        "pre_cost_ending_equity_same_path": pre_cost_ending,
        "pre_cost_total_return_same_path": pre_cost_ending / INITIAL_EQUITY - 1.0,
        "pre_cost_minus_post_cost_ending_equity": pre_cost_ending - float(equity[-1]),
        "transaction_cost_percent_ending_profit": (
            total_cost / observed_profit if observed_profit > 0 else math.nan
        ),
        "transaction_cost_percent_gross_positive_growth": (
            total_cost / gross_gain if gross_gain > 0 else math.nan
        ),
        "cost_1_5x_ending_equity": by_factor[1.5]["ending_equity"],
        "cost_1_5x_cagr": by_factor[1.5]["cagr"],
        "cost_1_5x_maximum_drawdown": by_factor[1.5]["maximum_drawdown"],
        "cost_1_5x_calmar": by_factor[1.5]["calmar_ratio"],
        "cost_1_5x_profitable": by_factor[1.5]["ending_equity_above_1000"],
        "cost_2x_ending_equity": by_factor[2.0]["ending_equity"],
        "cost_2x_cagr": by_factor[2.0]["cagr"],
        "cost_2x_maximum_drawdown": by_factor[2.0]["maximum_drawdown"],
        "cost_2x_calmar": by_factor[2.0]["calmar_ratio"],
        "cost_2x_profitable": by_factor[2.0]["ending_equity_above_1000"],
        "cost_2x_cagr_drop": by_factor[1.0]["cagr"] - by_factor[2.0]["cagr"],
    }
    return summary, rows


def construct_spy_crash_episodes(
    dates: pd.DatetimeIndex,
    spy_equity: np.ndarray,
    thresholds: Iterable[float] = (0.10, 0.15, 0.20),
) -> list[dict[str, Any]]:
    if len(dates) != len(spy_equity):
        raise ValueError("SPY date/equity length mismatch")
    peak = np.maximum.accumulate(spy_equity)
    drawdown = spy_equity / peak - 1.0
    rows: list[dict[str, Any]] = []
    threshold_groups: dict[int, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        threshold_pct = int(round(threshold * 100))
        episodes = []
        index = 0
        while index < len(dates):
            prior = drawdown[index - 1] if index else 0.0
            if drawdown[index] <= -threshold and prior > -threshold:
                onset = index
                peak_value = peak[onset]
                peak_candidates = np.flatnonzero(
                    np.isclose(spy_equity[: onset + 1], peak_value, atol=NUMERIC_ATOL)
                )
                peak_index = int(peak_candidates[-1])
                recovery_candidates = np.flatnonzero(
                    spy_equity[onset + 1 :] >= peak_value - NUMERIC_ATOL
                )
                recovery = (
                    int(onset + 1 + recovery_candidates[0])
                    if len(recovery_candidates)
                    else None
                )
                end = recovery if recovery is not None else len(dates) - 1
                trough = int(onset + np.argmin(drawdown[onset : end + 1]))
                episode_id = f"SPY{threshold_pct}_{len(episodes) + 1:02d}"
                row = {
                    "episode_id": episode_id,
                    "threshold_percent": threshold_pct,
                    "prior_peak_index": peak_index,
                    "onset_index": onset,
                    "trough_index": trough,
                    "recovery_index": recovery,
                    "prior_peak_date": _iso_date(dates[peak_index]),
                    "onset_date": _iso_date(dates[onset]),
                    "trough_date": _iso_date(dates[trough]),
                    "recovery_date": (
                        _iso_date(dates[recovery]) if recovery is not None else None
                    ),
                    "data_end_date": _iso_date(dates[-1]),
                    "is_open": recovery is None,
                    "spy_peak_to_trough_return": float(
                        spy_equity[trough] / spy_equity[peak_index] - 1.0
                    ),
                    "spy_onset_to_trough_return": float(
                        spy_equity[trough]
                        / (spy_equity[onset - 1] if onset else INITIAL_EQUITY)
                        - 1.0
                    ),
                    "spy_onset_to_recovery_return": float(
                        spy_equity[end]
                        / (spy_equity[onset - 1] if onset else INITIAL_EQUITY)
                        - 1.0
                    ),
                }
                episodes.append(row)
                rows.append(row)
                index = end + 1 if recovery is not None else len(dates)
                continue
            index += 1
        threshold_groups[threshold_pct] = episodes
    for row in rows:
        parents = []
        for threshold_pct, episodes in threshold_groups.items():
            if threshold_pct >= row["threshold_percent"]:
                continue
            for parent in episodes:
                parent_end = parent["recovery_index"]
                if parent_end is None:
                    parent_end = len(dates) - 1
                if (
                    parent["onset_index"] <= row["onset_index"] <= parent_end
                    and parent["prior_peak_index"] <= row["prior_peak_index"]
                ):
                    parents.append(parent["episode_id"])
        row["nested_within_episode_ids"] = ";".join(sorted(parents))
    return sorted(rows, key=lambda row: (row["threshold_percent"], row["onset_date"]))


def compounded_window(returns: np.ndarray, start: int, end: int) -> float:
    if end < start:
        return 0.0
    return float(np.prod(1.0 + returns[start : end + 1]) - 1.0)


def first_threshold_delay(
    exposure: np.ndarray,
    onset: int,
    end: int,
    threshold: float,
) -> tuple[float, str]:
    if exposure[onset] < threshold - NUMERIC_ATOL:
        return 0.0, "already_below_at_onset"
    hits = np.flatnonzero(exposure[onset : end + 1] < threshold - NUMERIC_ATOL)
    if len(hits):
        return float(hits[0]), "reached_before_episode_end"
    return math.nan, "not_reached_before_recovery_or_data_end"


def crash_strategy_episode_row(
    strategy_key: str,
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    spy_returns: np.ndarray,
    episode: dict[str, Any],
    *,
    exposure: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    transaction_cost: np.ndarray | None = None,
) -> dict[str, Any]:
    peak = int(episode["prior_peak_index"])
    onset = int(episode["onset_index"])
    trough = int(episode["trough_index"])
    end = (
        int(episode["recovery_index"])
        if episode["recovery_index"] is not None
        else len(dates) - 1
    )
    onset_start = max(onset, 0)
    peak_start = min(peak + 1, trough)
    strategy_onset_trough = compounded_window(returns, onset_start, trough)
    strategy_peak_trough = compounded_window(returns, peak_start, trough)
    strategy_onset_recovery = compounded_window(returns, onset_start, end)
    spy_onset_trough = compounded_window(spy_returns, onset_start, trough)
    spy_peak_trough = compounded_window(spy_returns, peak_start, trough)
    spy_onset_recovery = compounded_window(spy_returns, onset_start, end)
    episode_returns = returns[onset_start : end + 1]
    _, episode_dd = drawdown_from_returns(episode_returns)
    rolling5 = np.expm1(
        pd.Series(np.log1p(episode_returns)).rolling(5).sum().dropna().to_numpy()
    )
    strategy_equity = equity_from_returns(returns)
    target = strategy_equity[peak]
    crash_segment = strategy_equity[peak : trough + 1]
    local_trough_index = int(peak + np.argmin(crash_segment))
    if float(np.min(crash_segment)) >= target - NUMERIC_ATOL:
        recovery_index = peak
    else:
        post_local_trough = np.flatnonzero(
            strategy_equity[local_trough_index + 1 :] >= target - NUMERIC_ATOL
        )
        recovery_index = (
            int(local_trough_index + 1 + post_local_trough[0])
            if len(post_local_trough)
            else None
        )
    row: dict[str, Any] = {
        "strategy_key": strategy_key,
        "episode_id": episode["episode_id"],
        "threshold_percent": episode["threshold_percent"],
        "prior_peak_date": episode["prior_peak_date"],
        "onset_date": episode["onset_date"],
        "trough_date": episode["trough_date"],
        "spy_recovery_date": episode["recovery_date"],
        "episode_is_open": episode["is_open"],
        "nested_within_episode_ids": episode["nested_within_episode_ids"],
        "strategy_onset_to_spy_trough_return": strategy_onset_trough,
        "strategy_spy_peak_to_trough_return": strategy_peak_trough,
        "strategy_onset_to_spy_recovery_return": strategy_onset_recovery,
        "spy_onset_to_trough_return": spy_onset_trough,
        "spy_peak_to_trough_return": spy_peak_trough,
        "spy_onset_to_recovery_return": spy_onset_recovery,
        "strategy_episode_maximum_drawdown": float(np.min(episode_dd)),
        "strategy_episode_cdar_95": (
            conditional_drawdown_at_risk(episode_dd, 0.95)
            if len(episode_returns) >= 20
            else math.nan
        ),
        "strategy_episode_cdar_available": len(episode_returns) >= 20,
        "worst_strategy_day": float(np.min(episode_returns)),
        "worst_strategy_rolling_5d_return": (
            float(np.min(rolling5)) if len(rolling5) else math.nan
        ),
        "strategy_recovery_date": (
            _iso_date(dates[recovery_index]) if recovery_index is not None else None
        ),
        "strategy_recovery_basis_trough_date": _iso_date(
            dates[local_trough_index]
        ),
        "strategy_recovery_duration_days": (
            int((dates[recovery_index] - dates[local_trough_index]).days)
            if recovery_index is not None
            else math.nan
        ),
        "strategy_relative_return_vs_spy": strategy_onset_trough - spy_onset_trough,
        "crash_loss_capture": max(-strategy_onset_trough, 0.0)
        / max(-spy_onset_trough, 1e-12),
        "relative_crash_protection": strategy_onset_trough - spy_onset_trough,
        "strategy_positive_over_episode": strategy_onset_recovery > 0,
        "strategy_loss_less_than_half_spy": (
            max(-strategy_onset_trough, 0.0) < 0.5 * max(-spy_onset_trough, 0.0)
        ),
        "strategy_recovered_before_spy": (
            recovery_index is not None
            and (
                episode["recovery_index"] is None
                or recovery_index < int(episode["recovery_index"])
            )
        ),
    }
    if exposure is not None:
        def at_offset(offset: int) -> float:
            return float(exposure[min(onset + offset, len(exposure) - 1)])

        row.update(
            {
                "gross_exposure_at_prior_spy_peak": float(exposure[peak]),
                "gross_exposure_at_onset": float(exposure[onset]),
                "gross_exposure_at_spy_trough": float(exposure[trough]),
                "gross_exposure_5d_after_onset": at_offset(5),
                "gross_exposure_10d_after_onset": at_offset(10),
                "gross_exposure_20d_after_onset": at_offset(20),
                "exposure_change_5d": at_offset(5) - float(exposure[onset]),
                "exposure_change_10d": at_offset(10) - float(exposure[onset]),
                "exposure_change_20d": at_offset(20) - float(exposure[onset]),
                "minimum_exposure_before_spy_trough": float(
                    np.min(exposure[onset : trough + 1])
                ),
                "average_exposure_onset_to_trough": float(
                    np.mean(exposure[onset : trough + 1])
                ),
                "percent_onset_to_trough_days_fully_in_cash": float(
                    np.mean(exposure[onset : trough + 1] <= NUMERIC_ATOL)
                ),
                "maximum_active_positions_during_episode": int(
                    np.max(positions[onset : end + 1])
                ),
                "minimum_active_positions_during_episode": int(
                    np.min(positions[onset : end + 1])
                ),
                "transaction_cost_onset_to_trough": float(
                    np.sum(transaction_cost[onset : trough + 1])
                ),
            }
        )
        for threshold in [0.75, 0.50, 0.25]:
            delay, reason = first_threshold_delay(exposure, onset, end, threshold)
            label = int(threshold * 100)
            row[f"days_to_exposure_below_{label}pct"] = delay
            row[f"exposure_below_{label}pct_reason"] = reason
    return row


def aggregate_crash_rows(crash_rows: pd.DataFrame) -> pd.DataFrame:
    published = crash_rows.loc[crash_rows["gross_exposure_at_onset"].notna()].copy()
    rows = []
    for (strategy_key, threshold), group in published.groupby(
        ["strategy_key", "threshold_percent"], sort=True
    ):
        delays = pd.to_numeric(group["days_to_exposure_below_50pct"], errors="coerce")
        rows.append(
            {
                "strategy_key": strategy_key,
                "threshold_percent": int(threshold),
                "episode_count": len(group),
                "mean_crash_loss_capture": group["crash_loss_capture"].mean(),
                "median_crash_loss_capture": group["crash_loss_capture"].median(),
                "worst_crash_loss_capture": group["crash_loss_capture"].max(),
                "mean_relative_crash_protection": group[
                    "relative_crash_protection"
                ].mean(),
                "worst_relative_crash_protection": group[
                    "relative_crash_protection"
                ].min(),
                "mean_strategy_crash_maximum_drawdown": group[
                    "strategy_episode_maximum_drawdown"
                ].mean(),
                "worst_strategy_crash_maximum_drawdown": group[
                    "strategy_episode_maximum_drawdown"
                ].min(),
                "mean_exposure_at_onset": group["gross_exposure_at_onset"].mean(),
                "mean_exposure_after_10d": group[
                    "gross_exposure_10d_after_onset"
                ].mean(),
                "mean_10d_exposure_reduction": -group["exposure_change_10d"].mean(),
                "mean_days_to_exposure_below_50pct": (
                    delays.mean() if delays.notna().any() else math.nan
                ),
                "proportion_positive_strategy_return": group[
                    "strategy_positive_over_episode"
                ].mean(),
                "proportion_loss_less_than_half_spy": group[
                    "strategy_loss_less_than_half_spy"
                ].mean(),
                "proportion_recovered_before_spy": group[
                    "strategy_recovered_before_spy"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["threshold_percent", "strategy_key"], kind="mergesort"
    )


def pareto_frontier(
    frame: pd.DataFrame,
    dimensions: dict[str, str],
    tolerances: dict[str, float] | None = None,
) -> tuple[set[str], dict[str, int]]:
    tolerances = tolerances or {field: 0.0 for field in dimensions}
    rows = frame.set_index("strategy_key")
    frontier: set[str] = set()
    dominator_counts: dict[str, int] = {}
    for key, candidate in rows.iterrows():
        dominators = 0
        for other_key, other in rows.iterrows():
            if other_key == key:
                continue
            no_worse = True
            materially_better = False
            for field, direction in dimensions.items():
                a = float(other[field])
                b = float(candidate[field])
                tolerance = float(tolerances.get(field, 0.0))
                if direction == "max":
                    if a < b - tolerance:
                        no_worse = False
                        break
                    materially_better |= a > b + tolerance
                else:
                    if a > b + tolerance:
                        no_worse = False
                        break
                    materially_better |= a < b - tolerance
            if no_worse and materially_better:
                dominators += 1
        dominator_counts[str(key)] = dominators
        if dominators == 0:
            frontier.add(str(key))
    return frontier, dominator_counts


def _safe_csv(frame: pd.DataFrame, path: Path, sort_columns: list[str]) -> None:
    ordered = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    ordered.to_csv(path, index=False, float_format="%.12g", na_rep="")


def reconstruct_qualification(
    event: pd.DataFrame, annual: pd.DataFrame
) -> dict[str, Any]:
    gate_fields = pd.DataFrame(
        [
            _gate_fields(
                {
                    "completed_trades": row.completed_trades,
                    "profit_factor": row.profit_factor,
                    "avg_trade_return": row.avg_trade_return,
                    "median_trade_return": row.median_trade_return,
                    "trade_win_rate": row.trade_win_rate,
                }
            )
            for row in event.itertuples(index=False)
        ]
    )
    reconstructed = pd.concat(
        [
            event.drop(
                columns=[column for column in gate_fields.columns if column in event],
                errors="ignore",
            ).reset_index(drop=True),
            gate_fields,
        ],
        axis=1,
    )
    reconstructed = _attach_time_stability(reconstructed, annual)
    reconstructed = _attach_parameter_stability(reconstructed, annual)
    reconstructed = _attach_robustness_tiers(reconstructed)
    reconstructed = reconstructed.loc[:, ~reconstructed.columns.duplicated(keep="last")]
    reconstructed = rank_strategy_summary(reconstructed)
    stored = event.sort_values(["qualification_rank", "strategy_key"], kind="mergesort")
    shuffled = rank_strategy_summary(
        reconstructed.sample(frac=1.0, random_state=SEED)
    )
    return {
        "qualified_keys": sorted(
            reconstructed.loc[
                reconstructed["qualification_tier"].eq("Qualified"), "strategy_key"
            ]
        ),
        "ranked_keys": reconstructed["strategy_key"].tolist(),
        "stored_ranked_keys": stored["strategy_key"].tolist(),
        "shuffled_ranked_keys": shuffled["strategy_key"].tolist(),
    }


def load_curve(
    strategy_key: str, manifest_by_key: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    file_name = manifest_by_key[strategy_key]["file"]
    payload = json.loads(
        (DATA / "backtest_portfolio_curves" / file_name).read_text(encoding="utf-8")
    )
    curve = pd.DataFrame(payload["series"])
    if curve.iloc[0]["observation_type"] != "initialization":
        raise ValueError(f"{strategy_key}: missing initialization row")
    return curve.iloc[1:].reset_index(drop=True)


def load_inputs() -> dict[str, Any]:
    event = pd.read_csv(DATA / "backtest_strategy_summary.csv")
    annual = pd.read_csv(DATA / "backtest_strategy_year_summary.csv")
    portfolio = pd.read_csv(DATA / "backtest_portfolio_strategy_summary.csv")
    matrix = pd.read_csv(DATA / "backtest_portfolio_daily_returns.csv.gz")
    manifest = json.loads(
        (DATA / "backtest_portfolio_curve_manifest.json").read_text(encoding="utf-8")
    )
    benchmark = json.loads(
        (DATA / "backtest_benchmark_spy.json").read_text(encoding="utf-8")
    )
    matrix_dates = pd.to_datetime(matrix.pop("date"), format="mixed", utc=True)
    if not np.allclose(matrix.iloc[0].to_numpy(dtype=float), 0.0, atol=NUMERIC_ATOL):
        raise ValueError("matrix t0 must be zero")
    economic_matrix = matrix.iloc[1:].reset_index(drop=True)
    economic_dates = pd.DatetimeIndex(matrix_dates[1:])
    benchmark_frame = pd.DataFrame(benchmark["series"])
    benchmark_dates = pd.to_datetime(
        benchmark_frame["date"], format="mixed", utc=True
    )
    benchmark_economic = benchmark_frame.loc[
        benchmark_frame["observation_type"].eq("trading_session")
    ].reset_index(drop=True)
    benchmark_economic_dates = pd.DatetimeIndex(
        pd.to_datetime(benchmark_economic["date"], format="mixed", utc=True)
    )
    if not economic_dates.equals(benchmark_economic_dates):
        raise ValueError("strategy/SPY economic dates do not align")
    return {
        "event": event,
        "annual": annual,
        "portfolio": portfolio,
        "matrix": economic_matrix,
        "dates": economic_dates,
        "manifest": manifest,
        "manifest_by_key": {
            row["strategy_key"]: row for row in manifest["strategies"]
        },
        "benchmark": benchmark,
        "benchmark_frame": benchmark_frame,
        "benchmark_dates": benchmark_dates,
        "spy_returns": benchmark_economic["benchmark_daily_return"].to_numpy(
            dtype=float
        ),
        "spy_equity": benchmark_economic["benchmark_equity"].to_numpy(dtype=float),
    }


def validate_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    event = inputs["event"]
    matrix = inputs["matrix"]
    dates = inputs["dates"]
    qualified = sorted(
        event.loc[event["qualification_tier"].eq("Qualified"), "strategy_key"]
    )
    manifest_keys = sorted(inputs["manifest_by_key"])
    reconstructed = reconstruct_qualification(event, inputs["annual"])
    canonical = canonical_audit()
    canonical_failures = critical_failures(canonical)
    ending = INITIAL_EQUITY * (1.0 + matrix).prod(axis=0)
    stored = inputs["portfolio"].set_index("strategy_key").loc[
        matrix.columns, "ending_equity"
    ]
    reconstruction_error = float(
        np.max(np.abs(ending.to_numpy(dtype=float) - stored.to_numpy(dtype=float)))
    )
    errors = []
    if matrix.shape[1] != 540 or len(set(matrix.columns)) != 540:
        errors.append("strategy_count")
    if len(dates) != len(set(dates)):
        errors.append("duplicate_dates")
    if not dates.is_monotonic_increasing:
        errors.append("unsorted_dates")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        errors.append("nonfinite_returns")
    if (matrix.to_numpy(dtype=float) <= -1).any():
        errors.append("return_at_or_below_minus_one")
    if len(qualified) != 42 or sorted(set(qualified) - set(matrix.columns)):
        errors.append("qualified_matrix_membership")
    if sorted(qualified) != manifest_keys:
        errors.append("qualified_curve_set")
    if reconstructed["qualified_keys"] != qualified:
        errors.append("qualification_reconstruction")
    if reconstructed["ranked_keys"] != reconstructed["stored_ranked_keys"]:
        errors.append("ranking_reconstruction")
    if reconstructed["ranked_keys"] != reconstructed["shuffled_ranked_keys"]:
        errors.append("ranking_row_order")
    if reconstruction_error > NUMERIC_ATOL + NUMERIC_RTOL * float(stored.abs().max()):
        errors.append("ending_equity_reconstruction")
    errors.extend(canonical_failures)
    return {
        "material_failures": sorted(set(errors)),
        "strategy_count": matrix.shape[1],
        "qualified_count": len(qualified),
        "manifest_curve_count": len(manifest_keys),
        "t0_excluded_rows": 1,
        "economic_observations": len(dates),
        "economic_start_date": dates[0].date().isoformat(),
        "economic_end_date": dates[-1].date().isoformat(),
        "duplicate_date_strategy_rows": int(
            dates.duplicated().sum() * matrix.shape[1]
        ),
        "nonfinite_return_count": int(
            np.size(matrix) - np.isfinite(matrix.to_numpy(dtype=float)).sum()
        ),
        "return_at_or_below_minus_one_count": int(
            np.sum(matrix.to_numpy(dtype=float) <= -1)
        ),
        "maximum_ending_equity_reconstruction_error": reconstruction_error,
        "qualification_rank_primary": reconstructed["ranked_keys"][0],
        "portfolio_statistics_in_ranking": False,
    }


def build_analysis(*, bootstrap_paths: int = PRIMARY_BOOTSTRAP_PATHS) -> dict[str, Any]:
    inputs = load_inputs()
    validation = validate_inputs(inputs)
    if validation["material_failures"]:
        raise RuntimeError(
            "Portfolio input correction required: "
            + ", ".join(validation["material_failures"])
        )
    event = inputs["event"]
    matrix = inputs["matrix"]
    dates = inputs["dates"]
    spy_returns = inputs["spy_returns"]
    years = max((dates[-1] - dates[0]).days / 365.25, 1.0 / 365.25)
    qualified_keys = event.loc[
        event["qualification_tier"].eq("Qualified"), "strategy_key"
    ].tolist()
    primary_key = str(
        event.sort_values(["qualification_rank", "strategy_key"], kind="mergesort")
        .iloc[0]["strategy_key"]
    )
    labels = event.set_index("strategy_key")["strategy_label"].to_dict()
    portfolio_stored = inputs["portfolio"].set_index("strategy_key")
    spy_episodes = construct_spy_crash_episodes(
        dates, inputs["spy_equity"], (0.10, 0.15, 0.20)
    )

    behavior, behavior_mapping = behavior_groups(matrix)
    summary_rows = []
    episode_cache: dict[str, list[dict[str, Any]]] = {}
    concentration_summary = []
    concentration_rows = []
    curve_cache: dict[str, pd.DataFrame] = {}
    cost_sensitivity: dict[str, list[dict[str, Any]]] = {}

    for key in matrix.columns:
        returns = matrix[key].to_numpy(dtype=float)
        curve = None
        exposure = positions = costs = turnover = equity = None
        if key in inputs["manifest_by_key"]:
            curve = load_curve(key, inputs["manifest_by_key"])
            curve_dates = pd.DatetimeIndex(
                pd.to_datetime(curve["date"], format="mixed", utc=True)
            )
            if not dates.equals(curve_dates):
                raise ValueError(f"{key}: curve date mismatch")
            curve_cache[key] = curve
            exposure = curve["gross_exposure"].to_numpy(dtype=float)
            positions = curve["active_position_count"].to_numpy(dtype=float)
            costs = curve["transaction_cost_paid"].to_numpy(dtype=float)
            turnover = curve["turnover"].to_numpy(dtype=float)
            equity = curve["portfolio_equity"].to_numpy(dtype=float)
        metrics, episodes = baseline_metrics(
            dates,
            returns,
            exposure=exposure,
            active_positions=positions,
            transaction_cost=costs,
            turnover=turnover,
            spy_episodes=spy_episodes,
        )
        episode_cache[key] = episodes
        row = {
            "strategy_key": key,
            "strategy_label": labels[key],
            "qualification_tier": event.set_index("strategy_key").loc[
                key, "qualification_tier"
            ],
            "qualification_rank": int(
                event.set_index("strategy_key").loc[key, "qualification_rank"]
            ),
            **metrics,
        }
        if costs is not None and equity is not None:
            cost_summary, cost_rows = cost_sensitivity_rows(
                key, returns, equity, costs
            )
            cost_sensitivity[key] = cost_rows
            row.update(cost_summary)
            full_year_count = len(
                [
                    year
                    for year in sorted(set(dates.year))
                    if year not in (dates[0].year, dates[-1].year)
                ]
            )
            row["transaction_cost_per_completed_portfolio_year"] = (
                metrics["total_transaction_cost"] / full_year_count
            )
        summary_rows.append(row)
        if key in qualified_keys:
            concentration, stress_rows = concentration_and_stress_rows(
                key, dates, returns, episodes
            )
            concentration_summary.append(concentration)
            concentration_rows.extend(stress_rows)

    summary = pd.DataFrame(summary_rows)
    behavior = behavior.set_index("strategy_key")
    summary = summary.join(behavior, on="strategy_key")

    spy_std = float(np.std(spy_returns, ddof=1))
    spy_sharpe = float(
        np.mean(spy_returns) / spy_std * math.sqrt(TRADING_DAYS)
    )
    sharpe_std = float(np.std(summary["sharpe_ratio"].to_numpy(dtype=float), ddof=1))
    raw_expected_max = expected_maximum_sharpe(sharpe_std, len(summary))
    unique_count = len(behavior_mapping)
    dedup_expected_max = expected_maximum_sharpe(sharpe_std, unique_count)
    psr_rows = []
    for row in summary.itertuples(index=False):
        psr_rows.append(
            {
                "strategy_key": row.strategy_key,
                "qualification_tier": row.qualification_tier,
                "sample_length": len(dates),
                "annualized_sharpe": row.sharpe_ratio,
                "daily_return_skewness": row.daily_return_skewness,
                "daily_return_excess_kurtosis": row.daily_return_excess_kurtosis,
                "psr_vs_zero": psr_probability(
                    row.sharpe_ratio,
                    0.0,
                    len(dates),
                    row.daily_return_skewness,
                    row.daily_return_excess_kurtosis,
                ),
                "psr_vs_spy": psr_probability(
                    row.sharpe_ratio,
                    spy_sharpe,
                    len(dates),
                    row.daily_return_skewness,
                    row.daily_return_excess_kurtosis,
                ),
                "psr_vs_annual_sharpe_0_50": psr_probability(
                    row.sharpe_ratio,
                    POSITIVE_SHARPE_BENCHMARK,
                    len(dates),
                    row.daily_return_skewness,
                    row.daily_return_excess_kurtosis,
                ),
                "dsr_raw_n_540": psr_probability(
                    row.sharpe_ratio,
                    raw_expected_max,
                    len(dates),
                    row.daily_return_skewness,
                    row.daily_return_excess_kurtosis,
                ),
                "dsr_exact_deduplicated": psr_probability(
                    row.sharpe_ratio,
                    dedup_expected_max,
                    len(dates),
                    row.daily_return_skewness,
                    row.daily_return_excess_kurtosis,
                ),
                "raw_trial_count": len(summary),
                "exact_unique_path_count": unique_count,
                "cross_strategy_sharpe_std": sharpe_std,
                "expected_max_sharpe_raw": raw_expected_max,
                "expected_max_sharpe_exact_deduplicated": dedup_expected_max,
                "behavior_group_id": row.behavior_group_id,
                "behavior_group_size": row.behavior_group_size,
                "behavior_group_representative": row.behavior_group_representative,
                "normalized_return_hash": row.normalized_return_hash,
            }
        )
    psr_dsr = pd.DataFrame(psr_rows)

    crash_rows = []
    qualified_set = set(qualified_keys)
    for key in matrix.columns:
        returns = matrix[key].to_numpy(dtype=float)
        curve = curve_cache.get(key)
        kwargs = {}
        if curve is not None:
            kwargs = {
                "exposure": curve["gross_exposure"].to_numpy(dtype=float),
                "positions": curve["active_position_count"].to_numpy(dtype=float),
                "transaction_cost": curve["transaction_cost_paid"].to_numpy(dtype=float),
            }
        for episode in spy_episodes:
            crash_rows.append(
                crash_strategy_episode_row(
                    key, dates, returns, spy_returns, episode, **kwargs
                )
            )
    crash_detail = pd.DataFrame(crash_rows)
    crash_aggregate = aggregate_crash_rows(crash_detail)

    bootstrap_rows = []
    for number, key in enumerate(sorted(qualified_keys), start=1):
        paths = (
            max(bootstrap_paths, PRIMARY_CANDIDATE_PATHS)
            if key == primary_key
            else bootstrap_paths
        )
        print(
            f"bootstrap block=20 strategy={number}/{len(qualified_keys)} "
            f"paths={paths} key={key}",
            flush=True,
        )
        bootstrap_rows.append(
            stationary_bootstrap_summary(
                key,
                matrix[key].to_numpy(dtype=float),
                spy_returns,
                years,
                mean_block_length=20,
                path_count=paths,
            )
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    concentration_frame = pd.DataFrame(concentration_summary)
    summary = summary.merge(psr_dsr[[
        "strategy_key",
        "psr_vs_zero",
        "psr_vs_spy",
        "psr_vs_annual_sharpe_0_50",
        "dsr_raw_n_540",
        "dsr_exact_deduplicated",
    ]], on="strategy_key", how="left")
    summary = summary.merge(
        concentration_frame[[
            "strategy_key",
            "best_day_contribution_ratio",
            "best_5_days_contribution_ratio",
            "best_10_days_contribution_ratio",
            "best_month_contribution_ratio",
            "best_year_contribution_ratio",
        ]],
        on="strategy_key",
        how="left",
    )
    summary = summary.merge(
        bootstrap.loc[bootstrap["mean_block_length"].eq(20), [
            "strategy_key",
            "prob_cagr_above_zero",
            "cagr_5pct",
            "mdd_magnitude_95pct",
        ]],
        on="strategy_key",
        how="left",
    )
    crash10 = crash_aggregate.loc[
        crash_aggregate["threshold_percent"].eq(10),
        [
            "strategy_key",
            "worst_crash_loss_capture",
            "worst_relative_crash_protection",
            "mean_relative_crash_protection",
            "worst_strategy_crash_maximum_drawdown",
            "mean_10d_exposure_reduction",
            "mean_days_to_exposure_below_50pct",
        ],
    ]
    summary = summary.merge(crash10, on="strategy_key", how="left")

    qualified_summary = summary.loc[
        summary["strategy_key"].isin(qualified_set)
    ].copy()
    pareto_dimensions = {
        "cagr": "max",
        "cagr_5pct": "max",
        "dsr_raw_n_540": "max",
        "mean_relative_crash_protection": "max",
        "maximum_drawdown_magnitude": "min",
        "cdar_95_magnitude": "min",
        "mdd_magnitude_95pct": "min",
        "worst_crash_loss_capture": "min",
        "longest_time_under_water_days": "min",
        "cost_2x_cagr_drop": "min",
    }
    qualified_summary["maximum_drawdown_magnitude"] = -qualified_summary[
        "maximum_drawdown"
    ]
    qualified_summary["cdar_95_magnitude"] = -qualified_summary["cdar_95"]
    strict_frontier, strict_dominators = pareto_frontier(
        qualified_summary, pareto_dimensions
    )
    tolerances = {
        "cagr": 0.0001,
        "cagr_5pct": 0.0001,
        "dsr_raw_n_540": 0.001,
        "mean_relative_crash_protection": 0.001,
        "maximum_drawdown_magnitude": 0.001,
        "cdar_95_magnitude": 0.001,
        "mdd_magnitude_95pct": 0.001,
        "worst_crash_loss_capture": 0.01,
        "longest_time_under_water_days": 5.0,
        "cost_2x_cagr_drop": 0.0001,
    }
    tolerant_frontier, tolerant_dominators = pareto_frontier(
        qualified_summary, pareto_dimensions, tolerances
    )

    sensitivity_keys = sorted(strict_frontier | tolerant_frontier | {primary_key})
    for key in sensitivity_keys:
        for block in [5, 60]:
            print(
                f"bootstrap sensitivity block={block} key={key}",
                flush=True,
            )
            bootstrap_rows.append(
                stationary_bootstrap_summary(
                    key,
                    matrix[key].to_numpy(dtype=float),
                    spy_returns,
                    years,
                    mean_block_length=block,
                    path_count=bootstrap_paths,
                )
            )
    bootstrap = pd.DataFrame(bootstrap_rows)

    leader_specs = {
        "highest_cagr": ("cagr", "max"),
        "lowest_mdd": ("maximum_drawdown_magnitude", "min"),
        "lowest_cdar95": ("cdar_95_magnitude", "min"),
        "lowest_es95": ("expected_shortfall_95", "max"),
        "highest_calmar": ("calmar_ratio", "max"),
        "shortest_longest_time_under_water": ("longest_time_under_water_days", "min"),
        "highest_bootstrap_prob_cagr_positive": ("prob_cagr_above_zero", "max"),
        "highest_bootstrap_5pct_cagr": ("cagr_5pct", "max"),
        "lowest_bootstrap_95pct_mdd": ("mdd_magnitude_95pct", "min"),
        "highest_psr_vs_zero": ("psr_vs_zero", "max"),
        "highest_dsr_raw": ("dsr_raw_n_540", "max"),
        "lowest_best_day_concentration": ("best_day_contribution_ratio", "min"),
        "lowest_worst_crash_loss_capture": ("worst_crash_loss_capture", "min"),
        "best_worst_crash_drawdown": (
            "worst_strategy_crash_maximum_drawdown",
            "max",
        ),
        "fastest_mean_exposure_reduction": ("mean_10d_exposure_reduction", "max"),
        "lowest_cost_sensitivity": ("cost_2x_cagr_drop", "min"),
        "largest_behavioral_equivalence_group": ("behavior_group_size", "max"),
    }
    leader_tags: dict[str, list[str]] = defaultdict(list)
    for name, (field, direction) in leader_specs.items():
        values = pd.to_numeric(qualified_summary[field], errors="coerce")
        target = values.max() if direction == "max" else values.min()
        for key in qualified_summary.loc[
            np.isclose(values, target, atol=1e-12, rtol=1e-12), "strategy_key"
        ]:
            leader_tags[str(key)].append(name)

    only_dimension: dict[str, list[str]] = defaultdict(list)
    for removed in pareto_dimensions:
        reduced = {
            field: direction
            for field, direction in pareto_dimensions.items()
            if field != removed
        }
        reduced_frontier, _ = pareto_frontier(qualified_summary, reduced)
        for key in reduced_frontier - strict_frontier:
            only_dimension[key].append(removed)

    pareto_rows = []
    for row in qualified_summary.itertuples(index=False):
        key = str(row.strategy_key)
        pareto_rows.append(
            {
                "strategy_key": key,
                "qualification_rank": row.qualification_rank,
                "strict_pareto_frontier": key in strict_frontier,
                "tolerance_pareto_frontier": key in tolerant_frontier,
                "strict_dominator_count": strict_dominators[key],
                "tolerance_dominator_count": tolerant_dominators[key],
                "frontier_if_dimension_removed": ";".join(sorted(only_dimension[key])),
                "leader_objectives": ";".join(sorted(leader_tags[key])),
                "behavior_group_id": row.behavior_group_id,
                "behavior_group_size": row.behavior_group_size,
                **{
                    field: getattr(row, field)
                    for field in pareto_dimensions
                },
            }
        )
    pareto = pd.DataFrame(pareto_rows)

    # Worst full-year LOO leader derives from the long-form stress table.
    concentration_long = pd.DataFrame(concentration_rows)
    year_loo = concentration_long.loc[
        concentration_long["scenario_type"].eq("leave_one_full_calendar_year_out")
    ]
    worst_year_loo = (
        year_loo.groupby("strategy_key", sort=True)["cagr"].min().rename(
            "worst_full_year_loo_cagr"
        )
    )
    summary = summary.merge(worst_year_loo, on="strategy_key", how="left")
    if len(worst_year_loo):
        best_worst = float(worst_year_loo.max())
        for key in worst_year_loo.index[
            np.isclose(worst_year_loo, best_worst, atol=1e-12, rtol=1e-12)
        ]:
            leader_tags[str(key)].append("best_worst_full_year_loo")
        pareto["leader_objectives"] = pareto.apply(
            lambda row: ";".join(sorted(set(filter(None, [
                *str(row["leader_objectives"]).split(";"),
                *(
                    ["best_worst_full_year_loo"]
                    if row["strategy_key"] in worst_year_loo.index[
                        np.isclose(worst_year_loo, best_worst, atol=1e-12, rtol=1e-12)
                    ]
                    else []
                ),
            ])))),
            axis=1,
        )

    return {
        "inputs": inputs,
        "validation": validation,
        "summary": summary,
        "bootstrap": bootstrap,
        "psr_dsr": psr_dsr,
        "concentration_summary": concentration_frame,
        "concentration_long": concentration_long,
        "crash_detail": crash_detail,
        "crash_aggregate": crash_aggregate,
        "pareto": pareto,
        "spy_episodes": spy_episodes,
        "behavior_mapping": behavior_mapping,
        "primary_key": primary_key,
        "strict_frontier": strict_frontier,
        "tolerant_frontier": tolerant_frontier,
        "cost_sensitivity": cost_sensitivity,
        "pareto_tolerances": tolerances,
        "spy_sharpe": spy_sharpe,
        "raw_expected_max_sharpe": raw_expected_max,
        "dedup_expected_max_sharpe": dedup_expected_max,
    }


def write_outputs(result: dict[str, Any]) -> None:
    summary = result["summary"]
    bootstrap = result["bootstrap"]
    psr_dsr = result["psr_dsr"]
    concentration_summary = result["concentration_summary"]
    concentration_long = result["concentration_long"]
    crash_detail = result["crash_detail"]
    crash_aggregate = result["crash_aggregate"]
    pareto = result["pareto"]

    _safe_csv(summary, SUMMARY_OUTPUT, ["qualification_rank", "strategy_key"])
    _safe_csv(
        bootstrap,
        BOOTSTRAP_OUTPUT,
        ["mean_block_length", "strategy_key"],
    )
    _safe_csv(psr_dsr, PSR_DSR_OUTPUT, ["strategy_key"])
    concentration = concentration_long.merge(
        concentration_summary, on="strategy_key", how="left"
    )
    _safe_csv(
        concentration,
        CONCENTRATION_OUTPUT,
        ["strategy_key", "scenario_type", "scenario_label"],
    )
    _safe_csv(
        crash_detail,
        CRASH_EPISODE_OUTPUT,
        ["threshold_percent", "episode_id", "strategy_key"],
    )
    _safe_csv(
        crash_aggregate,
        CRASH_STRATEGY_OUTPUT,
        ["threshold_percent", "strategy_key"],
    )
    _safe_csv(
        pareto,
        PARETO_OUTPUT,
        ["strict_pareto_frontier", "tolerance_pareto_frontier", "strategy_key"],
    )
    write_report(result)


def _pct(value: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _leader_table(pareto: pd.DataFrame) -> str:
    rows = []
    for row in pareto.loc[pareto["leader_objectives"].astype(str).ne("")].sort_values(
        ["strategy_key"], kind="mergesort"
    ).itertuples(index=False):
        rows.append(
            f"| `{row.strategy_key}` | {str(row.leader_objectives).replace(';', ', ')} |"
        )
    return "\n".join(rows) if rows else "| None | None |"


def write_report(result: dict[str, Any]) -> None:
    validation = result["validation"]
    summary = result["summary"].set_index("strategy_key")
    bootstrap = result["bootstrap"]
    psr = result["psr_dsr"].set_index("strategy_key")
    concentration = result["concentration_summary"].set_index("strategy_key")
    crash_detail = result["crash_detail"]
    crash_aggregate = result["crash_aggregate"]
    pareto = result["pareto"]
    primary_key = result["primary_key"]
    primary = summary.loc[primary_key]
    primary_boot = bootstrap.loc[
        bootstrap["strategy_key"].eq(primary_key)
        & bootstrap["mean_block_length"].eq(20)
    ].iloc[0]
    primary_psr = psr.loc[primary_key]
    primary_concentration = concentration.loc[primary_key]
    primary_crash = crash_detail.loc[crash_detail["strategy_key"].eq(primary_key)]
    primary_crash_lines = []
    for row in primary_crash.itertuples(index=False):
        primary_crash_lines.append(
            f"| {row.episode_id} | {row.onset_date} | {row.trough_date} | "
            f"{_pct(row.strategy_onset_to_spy_trough_return)} | "
            f"{_pct(row.spy_onset_to_trough_return)} | "
            f"{_num(row.crash_loss_capture, 2)} | "
            f"{_pct(row.gross_exposure_at_onset)} | "
            f"{_pct(row.gross_exposure_10d_after_onset)} |"
        )
    episode_lines = []
    for row in result["spy_episodes"]:
        episode_lines.append(
            f"| {row['episode_id']} | {row['threshold_percent']}% | "
            f"{row['prior_peak_date']} | {row['onset_date']} | {row['trough_date']} | "
            f"{row['recovery_date'] or 'open'} | {row['nested_within_episode_ids'] or 'none'} |"
        )
    strict = sorted(result["strict_frontier"])
    tolerant = sorted(result["tolerant_frontier"])
    sensitivity = bootstrap.loc[
        bootstrap["strategy_key"].eq(primary_key)
    ].sort_values("mean_block_length")
    sensitivity_lines = []
    for row in sensitivity.itertuples(index=False):
        sensitivity_lines.append(
            f"| {row.mean_block_length} | {row.bootstrap_paths:,} | "
            f"{_pct(row.prob_cagr_above_zero)} | {_pct(row.cagr_5pct)} | "
            f"{_pct(row.mdd_magnitude_95pct)} |"
        )
    qualified = summary.loc[summary["qualification_tier"].eq("Qualified")]
    nonqualified = summary.loc[summary["qualification_tier"].ne("Qualified")]
    categories = []
    for key, row in qualified.iterrows():
        evidence = []
        if row["psr_vs_zero"] >= 0.95 and row["daily_return_skewness"] > 0:
            evidence.append("statistically credible positive-skew trend follower")
        if -row["maximum_drawdown"] >= 0.30:
            evidence.append("credible edge but severe drawdown risk")
        crash10 = crash_aggregate.loc[
            crash_aggregate["strategy_key"].eq(key)
            & crash_aggregate["threshold_percent"].eq(10)
        ]
        if len(crash10) and crash10.iloc[0]["worst_crash_loss_capture"] < 0.5:
            evidence.append("strong historical crash avoidance")
        if len(crash10) and (
            not np.isfinite(crash10.iloc[0]["mean_days_to_exposure_below_50pct"])
            or crash10.iloc[0]["mean_days_to_exposure_below_50pct"] > 10
        ):
            evidence.append("slow crash-response strategy")
        if row["best_10_days_contribution_ratio"] > 0.5:
            evidence.append("high winner-period concentration")
        if row["cost_2x_cagr_drop"] > 0.02:
            evidence.append("high turnover/cost fragility")
        if row["dsr_raw_n_540"] < 0.95:
            evidence.append("likely selection-bias concern")
        if row["behavior_group_size"] > 1:
            evidence.append("behaviorally redundant parameter variant")
        evidence.append("insufficient crash observations")
        categories.append(
            f"| `{key}` | {', '.join(dict.fromkeys(evidence))} |"
        )
    report = f"""# Phase B skew-aware robustness and crash-avoidance analysis

## Decision

The bounded canonical portfolio data are internally valid and statistically informative, but they cover one in-sample history, only eight full calendar years, and a small number of nested historical SPY crash episodes. The results support provisional *diagnostic ranges*, not mandatory production portfolio-risk gates. No strategy, portfolio allocation rule, qualification gate, or ranking was changed.

## 1. Input validation

| Check | Result |
| --- | ---: |
| Unique daily strategy series | {validation['strategy_count']} |
| Qualified strategies / published curves | {validation['qualified_count']} / {validation['manifest_curve_count']} |
| Economic observations per strategy | {validation['economic_observations']:,} |
| Economic date range | {validation['economic_start_date']} to {validation['economic_end_date']} |
| Explicit t0 rows excluded from estimation | {validation['t0_excluded_rows']} |
| Duplicate date x strategy rows | {validation['duplicate_date_strategy_rows']} |
| Non-finite returns | {validation['nonfinite_return_count']} |
| Returns <= -100% | {validation['return_at_or_below_minus_one_count']} |
| Maximum ending-equity reconstruction error | {validation['maximum_ending_equity_reconstruction_error']:.3g} |
| Independently reproduced production primary | `{validation['qualification_rank_primary']}` |
| Portfolio statistics in production rank | no |
| Material validation failures | {len(validation['material_failures'])} |

The all-strategy input is the documented `backtest_portfolio_daily_returns.csv.gz`, not an assumed Parquet file. The schema-v2 initialization row remains available for USD 1,000 normalization but is excluded from every estimate. Strategy and SPY economic dates are joined by exact equality; no forward fill is used. The final canonical audit script also passes.

## 2. Populations and baseline results

The primary population is the 42 current Qualified strategies. The multiple-testing population is all 540 tested strategies; no poor strategy was removed. All 540 receive baseline return, distribution, drawdown, tail, PSR/DSR, behavioral-deduplication, and crash-return diagnostics. Computationally intensive bootstrap and detailed exposure/cost diagnostics use the 42 published Qualified curves.

Qualified versus non-Qualified medians:

| Metric | Qualified | Non-Qualified |
| --- | ---: | ---: |
| CAGR | {_pct(float(qualified['cagr'].median()))} | {_pct(float(nonqualified['cagr'].median()))} |
| Maximum drawdown | {_pct(float(qualified['maximum_drawdown'].median()))} | {_pct(float(nonqualified['maximum_drawdown'].median()))} |
| CDaR 95 | {_pct(float(qualified['cdar_95'].median()))} | {_pct(float(nonqualified['cdar_95'].median()))} |
| Daily skewness | {_num(float(qualified['daily_return_skewness'].median()))} | {_num(float(nonqualified['daily_return_skewness'].median()))} |
| Annualized Sharpe | {_num(float(qualified['sharpe_ratio'].median()))} | {_num(float(nonqualified['sharpe_ratio'].median()))} |

The robustness CSV reports arithmetic and geometric growth, volatility, unbiased skewness/excess kurtosis, downside/upside semideviation, Omega, gain/loss, positive-day/month/year rates, Sharpe/Sortino/Calmar/Martin, three largest drawdowns, CDaR 90/95/99, recovery statistics, drawdown-frequency thresholds, rolling losses, ES, and the five largest distinct drawdown paths. Exposure and implementation fields are populated only for published Qualified curves.

The positive-calendar-year ratio in this Phase B descriptive table includes every observed calendar-year bucket, including partial 2017 and 2026. It is not the production Time Gate ratio, which continues to use only eligible full entry years.

## 3. Time concentration and removal diagnostics

Contributions use additive log returns because every canonical equity path remains positive. Ratios are contribution divided by total observed log growth; negative or near-zero total growth is reported without converting the ratio into a gate. The long-form concentration table includes baseline, best-day removal, five/ten best-day zeroing, best-month/year zeroing, leave-one-full-year-out, and every distinct drawdown-episode removal, each with ending equity, CAGR, MDD, CDaR 95, Sharpe, and Calmar.

For the primary candidate, the best day contributes {_pct(float(primary_concentration['best_day_contribution_ratio']))} of total log growth, the best five days {_pct(float(primary_concentration['best_5_days_contribution_ratio']))}, the best ten days {_pct(float(primary_concentration['best_10_days_contribution_ratio']))}, the best month {_pct(float(primary_concentration['best_month_contribution_ratio']))}, and the best year {_pct(float(primary_concentration['best_year_contribution_ratio']))}. These are fragility diagnostics, not pass conditions; positive skew naturally concentrates some growth in a minority of positive periods.

Raw trade-level winner concentration and ETF-level profit attribution are unavailable and were not fabricated.

## 4. Stationary block bootstrap

The stationary bootstrap uses economic daily returns, geometric block lengths with restart probability `1 / mean_block_length`, fixed root seed {SEED}, observed path length {validation['economic_observations']}, and percentile intervals. Strategy and SPY use identical block indices within every relative-performance path, preserving contemporaneous dependence. Missing observations are not filled because validation requires a complete common matrix. Partial calendar years remain in the daily bootstrap as observed economic returns; annual gate eligibility is not reused here. CAGR uses the observed calendar span. Block-boundary splicing and regime non-stationarity remain limitations, and reported probabilities are not guaranteed future probabilities.

Every Qualified strategy uses at least {PRIMARY_BOOTSTRAP_PATHS:,} paths at mean block length 20. The primary uses {int(primary_boot['bootstrap_paths']):,}. Primary sensitivity:

| Mean block | Paths | P(CAGR > 0) | 5th pct CAGR | 95th pct MDD magnitude |
| ---: | ---: | ---: | ---: | ---: |
{chr(10).join(sensitivity_lines)}

Primary block-20 results: P(ending equity > USD 1,000) {_pct(float(primary_boot['prob_ending_equity_above_1000']))}, P(CAGR > 0) {_pct(float(primary_boot['prob_cagr_above_zero']))}, jointly resampled P(CAGR > SPY) {_pct(float(primary_boot['prob_cagr_above_spy']))}, 5th-percentile ending equity USD {float(primary_boot['ending_equity_5pct']):,.2f}, 5th-percentile CAGR {_pct(float(primary_boot['cagr_5pct']))}, 95th-percentile MDD magnitude {_pct(float(primary_boot['mdd_magnitude_95pct']))}, and 5th-percentile Calmar {_num(float(primary_boot['calmar_5pct']))}.

## 5. PSR, DSR, and selection bias

Daily Sharpe is the sample mean divided by sample standard deviation; annualized values multiply by sqrt(252). PSR uses the finite-sample skewness/kurtosis-adjusted denominator `sqrt(1 - skew*SR + ((kurtosis-1)/4)*SR^2)`, with unbiased sample skewness and excess kurtosis converted to ordinary kurtosis. Benchmarks are annual Sharpe 0, observed SPY Sharpe {_num(result['spy_sharpe'])}, and the predeclared positive annual Sharpe 0.50, each divided by sqrt(252) before evaluation. PSR is a probability that the population Sharpe exceeds the benchmark under the formula's assumptions; it is not labeled a generic p-value.

DSR uses the same PSR adjustment against the expected maximum Sharpe, cross-sectional observed-Sharpe standard deviation, Euler-Mascheroni expected-maximum approximation, and two trial treatments: raw N=540 and exact/numerically equivalent daily paths hashed after deterministic 1e-12 quantization. The exact-deduplicated count is {int(primary_psr['exact_unique_path_count'])}; raw and deduplicated expected maximum annual Sharpe are {_num(result['raw_expected_max_sharpe'])} and {_num(result['dedup_expected_max_sharpe'])}. Correlation clusters are not called independent trials.

Primary: annualized Sharpe {_num(float(primary_psr['annualized_sharpe']))}, PSR vs zero {_pct(float(primary_psr['psr_vs_zero']))}, PSR vs SPY {_pct(float(primary_psr['psr_vs_spy']))}, PSR vs 0.50 {_pct(float(primary_psr['psr_vs_annual_sharpe_0_50']))}, DSR N=540 {_pct(float(primary_psr['dsr_raw_n_540']))}, and exact-deduplicated DSR {_pct(float(primary_psr['dsr_exact_deduplicated']))}. PSR/DSR reduce neither data reuse nor model-selection history to out-of-sample evidence.

## 6. Multiple-testing feasibility

The common 540-column daily matrix is sufficient in principle for a stationary-bootstrap White Reality Check or Hansen SPA loss-differential design against cash (`strategy_return - 0`) and SPY (`strategy_return - SPY_return`) with shared block indices. It is not implemented here because a defensible SPA requires an explicitly selected studentization and weak-model trimming rule, while a 540-strategy Reality Check would answer a different familywise-null question than the requested per-candidate bootstrap and DSR. Adding a superficially precise p-value without those predeclared choices would be less reliable than the supplied feasibility design.

PBO/CSCV is not authoritative here: only eight full calendar years are available, strategy returns share regimes, and symmetric temporal partitions would be few and highly dependent. A later frozen walk-forward protocol is recommended.

## 7. Historical SPY crash episodes

Episodes are independently constructed at 10%, 15%, and 20%. Onset is the first threshold crossing from above, each threshold is non-overlapping until recovery of its prior peak, severe episodes may nest within lower-threshold episodes, and unrecovered episodes remain open. t0 is excluded.

| Episode | Threshold | Prior peak | Onset | Trough | Recovery | Nested within |
| --- | ---: | --- | --- | --- | --- | --- |
{chr(10).join(episode_lines)}

There are {sum(row['threshold_percent'] == 10 for row in result['spy_episodes'])} 10%, {sum(row['threshold_percent'] == 15 for row in result['spy_episodes'])} 15%, and {sum(row['threshold_percent'] == 20 for row in result['spy_episodes'])} 20% episodes. This small, nested sample cannot establish general crash protection.

Crash loss capture is `max(-strategy onset-to-trough return, 0) / max(-SPY onset-to-trough return, epsilon)`; lower is better. Relative protection is the signed strategy return minus SPY return. The episode CSV also reports peak-to-trough/onset-to-recovery windows, local MDD/CDaR/rolling losses, recovery, and relative outcomes for all 540 strategies. Exposure decay, positions, and cost are limited to the 42 published curves.

Exposure below 100% at onset is evidence of avoiding some risk before the threshold crossing. The change from onset to days +5/+10/+20 describes reaction after the crash begins. Peak-to-trough versus onset-to-trough loss separates losses already suffered before the threshold from losses after it. A low onset exposure can therefore coexist with a slow post-onset reduction, and neither is silently interpreted as a stop-system pass.

## 8. Current primary candidate

`{primary_key}` remains production rank 1; Phase B does not change that rank.

- Ending equity USD {float(primary['ending_equity']):,.2f}; CAGR {_pct(float(primary['cagr']))}.
- MDD {_pct(float(primary['maximum_drawdown']))}; CDaR 95 {_pct(float(primary['cdar_95']))}; ES95 {_pct(float(primary['expected_shortfall_95']))}; longest time under water {int(primary['longest_time_under_water_days'])} days.
- Daily skewness {_num(float(primary['daily_return_skewness']))}; excess kurtosis {_num(float(primary['daily_return_excess_kurtosis']))}.
- Behavioral group `{primary['behavior_group_id']}` contains {int(primary['behavior_group_size'])} numerically equivalent parameter paths.
- Cost sensitivity: observed CAGR {_pct(float(primary['cagr']))}, 1.5x-cost CAGR {_pct(float(primary['cost_1_5x_cagr']))}, 2x-cost CAGR {_pct(float(primary['cost_2x_cagr']))}; 2x ending equity USD {float(primary['cost_2x_ending_equity']):,.2f}, MDD {_pct(float(primary['cost_2x_maximum_drawdown']))}, and Calmar {_num(float(primary['cost_2x_calmar']))}. This is a deterministic fixed-membership/turnover-path reconstruction using the exact daily cost load, not a new portfolio simulation.
- Strict Pareto frontier: {'yes' if primary_key in result['strict_frontier'] else 'no'}; tolerance frontier: {'yes' if primary_key in result['tolerant_frontier'] else 'no'}.

Primary crash table:

| Episode | Onset | Trough | Strategy onset-to-trough | SPY onset-to-trough | Loss capture | Exposure onset | Exposure +10d |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(primary_crash_lines)}

It remains attractive where its deterministic qualification evidence, positive historical growth, skew-aware Sharpe probability, and relative crash behavior agree. It may be unsafe where bootstrap lower tails, prolonged underwater periods, winner-period concentration, turnover/cost sensitivity, or slow exposure decay remain material. Its first-place qualification rank is supported by event-level gate evidence, but portfolio-risk evidence is mixed and was never part of that rank.

## 9. Turnover and cost robustness

The analysis preserves the high observed turnover/cost values. For each Qualified curve it reports annual turnover, total and annual cost, cost per full portfolio year, cost relative to initial capital/ending profit/de-costed same-path growth, turnover-bearing rebalance days, and average turnover per rebalance. The 1x/1.5x/2x scenarios reconstruct daily gross return as observed net return plus exact daily cost divided by prior equity, then scale that same daily cost load. They do not claim that higher costs would leave future membership or equal-weight solver allocations unchanged.

No cost threshold is introduced and no production cost assumption changes.

## 10. Candidate comparison and Pareto analysis

Separate leaders:

| Strategy | Leader objective(s) |
| --- | --- |
{_leader_table(pareto)}

Strict dominance requires no worse performance in all ten declared dimensions and strict improvement in at least one. The strict frontier contains {len(strict)} strategies:

{', '.join(f'`{key}`' for key in strict)}

The tolerance frontier uses 1 bp for CAGR/bootstrap CAGR/cost sensitivity, 0.001 for DSR and portfolio/crash-return risk rates, 0.01 for crash capture, and 5 days for time under water. It contains {len(tolerant)} strategies:

{', '.join(f'`{key}`' for key in tolerant)}

The Pareto CSV reports every Qualified strategy, strict/tolerance frontier flags, dominator counts, dimensions whose removal alone would restore frontier membership, separate leader tags, and behavioral-equivalence groups. No weighted or lexicographic robustness score is constructed.

The crash dimensions use the aggregate 10% SPY-episode population: mean relative protection is maximized and worst loss capture is minimized. This avoids counting nested 15% and 20% observations again in the same Pareto dimension. Block-length sensitivity is nevertheless reported for the union of strict and tolerance frontier strategies plus the production primary.

### Provisional diagnostic ranges, not gates

| Qualified-strategy diagnostic | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| CAGR | {_pct(float(qualified['cagr'].min()))} | {_pct(float(qualified['cagr'].median()))} | {_pct(float(qualified['cagr'].max()))} |
| MDD magnitude | {_pct(float((-qualified['maximum_drawdown']).min()))} | {_pct(float((-qualified['maximum_drawdown']).median()))} | {_pct(float((-qualified['maximum_drawdown']).max()))} |
| CDaR 95 magnitude | {_pct(float((-qualified['cdar_95']).min()))} | {_pct(float((-qualified['cdar_95']).median()))} | {_pct(float((-qualified['cdar_95']).max()))} |
| Bootstrap 5th-percentile CAGR | {_pct(float(qualified['cagr_5pct'].min()))} | {_pct(float(qualified['cagr_5pct'].median()))} | {_pct(float(qualified['cagr_5pct'].max()))} |
| Raw-N DSR | {_pct(float(qualified['dsr_raw_n_540'].min()))} | {_pct(float(qualified['dsr_raw_n_540'].median()))} | {_pct(float(qualified['dsr_raw_n_540'].max()))} |
| Worst 10% crash loss capture | {_num(float(qualified['worst_crash_loss_capture'].min()))} | {_num(float(qualified['worst_crash_loss_capture'].median()))} | {_num(float(qualified['worst_crash_loss_capture'].max()))} |
| 2x-cost CAGR drop | {_pct(float(qualified['cost_2x_cagr_drop'].min()))} | {_pct(float(qualified['cost_2x_cagr_drop'].median()))} | {_pct(float(qualified['cost_2x_cagr_drop'].max()))} |

These observed ranges can frame later preregistration, but none is proposed as a mandatory cutoff from this in-sample run.

## 11. Descriptive interpretation categories

| Strategy | Evidence-based categories |
| --- | --- |
{chr(10).join(categories)}

These categories are overlapping diagnostics, not production qualification tiers.

Category evidence rules are explicit: PSR-vs-zero at least 95% plus positive skew; MDD magnitude at least 30%; worst 10% crash loss capture below 0.5; mean time to exposure below 50% above ten sessions or unreached; best-ten-day share above 50% of log growth; 2x-cost CAGR drop above two percentage points; raw-N DSR below 95%; behavioral group size above one; and the universal small-crash-sample warning. They are descriptive labels only.

## 12. Out-of-sample limitations and next protocol

- The same historical data influenced strategy design and selection; these are not pure out-of-sample results.
- Stationary bootstrap assumes a sufficiently stable dependent return process and can create artificial block boundaries; it cannot manufacture unseen regimes.
- PSR and DSR reduce but do not eliminate selection bias.
- Historical SPY crashes are a small, nested episode sample.
- Long-only ETF trend following is not diversified long/short managed-futures trend following.
- Future crash gaps can exceed historical stop behavior.
- Yahoo/yfinance-quality adjusted prices and vendor revisions impose research limitations.
- Detailed accounting is bounded to published curves; raw trade and ETF attribution are unavailable.

A later protocol should freeze signals, gates, portfolio rules, candidate set, and code hash before a shadow period; record every daily decision; prohibit retrospective parameter replacement; and evaluate predeclared walk-forward windows only after sufficient new regimes accumulate. That protocol is recommended but not implemented here.

## 13. Reproducibility and boundaries

All result tables are produced by `scripts/analyze_skew_aware_robustness.py` from committed bounded outputs. Deterministic tests cover bootstrap indices and shared SPY sampling, t0 exclusion, equity/MDD/CDaR/ES/time-under-water, drawdown/crash episodes and nesting, loss capture, exposure decay, concentration/removal, PSR/DSR, behavioral deduplication, Pareto tolerance, cost sensitivity, deterministic ordering, and input row-order invariance.

No full Backtest Only workflow was run. No raw event-level data were published. No signal, parameter grid, entry, exit, stop, max-hold, transaction-cost assumption, universe, allocation model, backtest period, Sample/Edge/Time/Parameter Gate, qualification tier, production ranking, UI, or risk gate changed.

Additional out-of-sample data required
"""
    REPORT_OUTPUT.write_text(report, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-paths",
        type=int,
        default=PRIMARY_BOOTSTRAP_PATHS,
        help="base paths per Qualified strategy; production research uses 5000",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate bounded inputs without generating Phase B outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        validation = validate_inputs(load_inputs())
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return 1 if validation["material_failures"] else 0
    if args.bootstrap_paths < PRIMARY_BOOTSTRAP_PATHS:
        raise ValueError("final analysis requires at least 5000 bootstrap paths")
    result = build_analysis(bootstrap_paths=args.bootstrap_paths)
    write_outputs(result)
    print(
        json.dumps(
            {
                "outcome": "Additional out-of-sample data required",
                "qualified_count": result["validation"]["qualified_count"],
                "strict_frontier_count": len(result["strict_frontier"]),
                "tolerance_frontier_count": len(result["tolerant_frontier"]),
                "outputs": [
                    str(path.relative_to(ROOT))
                    for path in [
                        REPORT_OUTPUT,
                        SUMMARY_OUTPUT,
                        BOOTSTRAP_OUTPUT,
                        PSR_DSR_OUTPUT,
                        CONCENTRATION_OUTPUT,
                        CRASH_EPISODE_OUTPUT,
                        CRASH_STRATEGY_OUTPUT,
                        PARETO_OUTPUT,
                    ]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
