from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import add_signal_surge_v0
from .portfolio import (
    PORTFOLIO_INITIAL_CAPITAL,
    build_price_panel,
    simulate_canonical_portfolio,
    summarize_portfolio_curve,
)


SCORE_LOOKBACK_GRID = (10, 20, 40)
PRICE_BREAKOUT_LOOKBACK = 20
PLACEBO_SHIFT_BARS = 63
FREQUENCY_MATCHED_RANDOM_SEEDS = (1729, 3253, 5003, 7919, 10427)
FORWARD_RETURN_HORIZONS = (1, 3, 5, 10, 20)
EXCURSION_HORIZON = 20
ROUND_TRIP_COST = 0.002
PRIMARY_TREND_MA_LENGTH = 200
PRIMARY_TREND_SLOPE_LENGTH = 20
CANONICAL_EQUAL_WEIGHT_KEY = "canonical_equal_weight_active_v1"
MIN_WALK_FORWARD_FOLDS = 3
MIN_LOYO_RESULTS = 3
MIN_LOYO_STABILITY_RATIO = 1.0
BOOTSTRAP_MATERIAL_EFFECT_TOLERANCE = 0.002
MULTIPLE_TESTING_ADJUSTED_ALPHA = 0.05
MAX_DOMINANT_ASSET_GROUP_EFFECT_SHARE = 0.50
REQUIRED_ROBUSTNESS_EVIDENCE_FIELDS = (
    "walk_forward_fold_count",
    "walk_forward_improvement_ratio",
    "leave_one_year_out_result_count",
    "leave_one_year_out_stability_ratio",
    "date_block_bootstrap_paired_effect_confidence_interval",
    "asset_group_concentration_diagnostics",
    "event_count_comparability",
    "executable_trigger_count_comparability",
)


FrameRule = Callable[[pd.DataFrame], pd.Series]
StopRule = Callable[[pd.Series], float]
EntryRuleFn = Callable[[pd.Series], list[int]]
SizingRuleFn = Callable[[Sequence[str]], dict[str, float]]
PortfolioRuleFn = Callable[[pd.DataFrame, pd.DataFrame, float], pd.DataFrame]


def _bool_series(values: object, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    return series.astype("boolean").fillna(False).astype(bool)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _number(row: pd.Series, column: str) -> float:
    try:
        value = float(row.get(column, np.nan))
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


@dataclass(frozen=True)
class UniverseEligibilityRule:
    key: str
    description: str
    eligibility_fn: FrameRule

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return _bool_series(self.eligibility_fn(frame), frame.index)


@dataclass(frozen=True)
class FilterRule:
    key: str
    description: str
    filter_fn: FrameRule

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return _bool_series(self.filter_fn(frame), frame.index)


@dataclass(frozen=True)
class SignalRule:
    key: str
    label: str
    description: str
    params: Mapping[str, object]
    signal_fn: FrameRule
    role: str = "v2_candidate"

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return _bool_series(self.signal_fn(frame), frame.index)


@dataclass(frozen=True)
class EntryRule:
    key: str
    description: str
    signal_indices_fn: EntryRuleFn

    def signal_indices(self, signal: pd.Series) -> list[int]:
        return self.signal_indices_fn(signal)


@dataclass(frozen=True)
class InitialStopRule:
    key: str
    description: str
    stop_fn: StopRule

    def initial_stop(self, signal_row: pd.Series) -> float:
        return self.stop_fn(signal_row)


@dataclass(frozen=True)
class TrailingExitRule:
    key: str
    description: str
    stop_fn: StopRule
    ratchet_only: bool = True

    def candidate_stop(self, row: pd.Series) -> float:
        return self.stop_fn(row)


@dataclass(frozen=True)
class PositionSizingRule:
    key: str
    description: str
    weights_fn: SizingRuleFn

    def target_weights(self, symbols: Sequence[str]) -> dict[str, float]:
        return self.weights_fn(symbols)


@dataclass(frozen=True)
class PortfolioConstructionRule:
    key: str
    description: str
    construction_fn: PortfolioRuleFn

    def construct(
        self,
        lifecycles: pd.DataFrame,
        prices: pd.DataFrame,
        round_trip_cost: float,
    ) -> pd.DataFrame:
        return self.construction_fn(lifecycles, prices, round_trip_cost)


@dataclass(frozen=True)
class PhaseAComponents:
    universe: UniverseEligibilityRule
    trend_filter: FilterRule
    entry: EntryRule
    initial_stop: InitialStopRule
    trailing_exit: TrailingExitRule
    position_sizing: PositionSizingRule
    portfolio_construction: PortfolioConstructionRule


@dataclass(frozen=True)
class PhaseAComparisonResult:
    signal_observations: pd.DataFrame
    signal_diagnostics: pd.DataFrame
    signal_diagnostic_summary: pd.DataFrame
    lifecycles: pd.DataFrame
    trade_diagnostics: pd.DataFrame
    portfolio_metrics: pd.DataFrame
    classification: Mapping[str, object]
    methodology: Mapping[str, object]


def legacy_scanner_trend_filter(frame: pd.DataFrame) -> pd.Series:
    return (
        (_numeric(frame, "te63") > _numeric(frame, "te126"))
        & (_numeric(frame, "te63") > 0)
        & (_numeric(frame, "close") > _numeric(frame, "ma50"))
        & (_numeric(frame, "ma50") > _numeric(frame, "ma150"))
        & (_numeric(frame, "r63") > 0.03)
        & (_numeric(frame, "r126") > 0)
        & (_numeric(frame, "hhv126_ratio") > 0.80)
    )


LEGACY_SCANNER_TREND_FILTER_CONTROL_V0 = FilterRule(
    "legacy_scanner_trend_filter_control_v0",
    "Sensitivity-only control decomposed from the legacy signal_surge_v0 Boolean.",
    legacy_scanner_trend_filter,
)


def add_phase_a_price_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the fixed price-only Phase A filter without touching v1 features."""
    if "symbol" not in frame or "date" not in frame or "close" not in frame:
        raise ValueError("Phase A price trend features require symbol, date, and close")
    out = frame.copy()
    out["phase_a_ma200"] = np.nan
    out["phase_a_ma200_slope_20"] = np.nan
    for _, group in out.groupby("symbol", sort=False):
        ordered = group.sort_values("date")
        close = _numeric(ordered, "close")
        ma200 = close.rolling(
            PRIMARY_TREND_MA_LENGTH,
            min_periods=PRIMARY_TREND_MA_LENGTH,
        ).mean()
        slope = ma200 / ma200.shift(PRIMARY_TREND_SLOPE_LENGTH) - 1.0
        out.loc[ordered.index, "phase_a_ma200"] = ma200.to_numpy(dtype=float)
        out.loc[ordered.index, "phase_a_ma200_slope_20"] = slope.to_numpy(dtype=float)
    out["phase_a_price_trend_filter"] = (
        (_numeric(out, "close") > _numeric(out, "phase_a_ma200"))
        & (_numeric(out, "phase_a_ma200_slope_20") > 0)
    )
    return out


def phase_a_price_only_trend_filter(frame: pd.DataFrame) -> pd.Series:
    prepared = add_phase_a_price_trend_features(frame)
    return prepared["phase_a_price_trend_filter"]


PRIMARY_PHASE_A_TREND_FILTER = FilterRule(
    "price_above_rising_ma200_v0",
    "Fixed price-only control: close above MA200 and MA200 20-session slope above zero.",
    phase_a_price_only_trend_filter,
)


def decompose_legacy_scanner(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose the legacy Boolean's filter, trigger, ranking, and risk pieces."""
    out = add_signal_surge_v0(frame)
    out["v2_universe_eligible"] = _bool_series(
        out.get("eligible_universe", False), out.index
    )
    out["v2_trend_filter"] = legacy_scanner_trend_filter(out)
    out["v2_trigger_signal"] = _numeric(out, "surge_ratio") > 1.25
    out["v2_continuous_ranking"] = _numeric(out, "score")
    out["v2_risk_er63"] = _numeric(out, "er63")
    out["v2_risk_atr20_pct"] = _numeric(out, "atr20_pct")
    out["v2_risk_stop_distance_pct"] = _numeric(out, "stop_distance_pct")
    out["v2_risk_gate"] = (
        (out["v2_risk_er63"] > 0.20) & (out["v2_risk_atr20_pct"] < 0.06)
    )
    out["v2_legacy_signal_reconstructed"] = (
        out["v2_universe_eligible"]
        & out["v2_trend_filter"]
        & out["v2_trigger_signal"]
        & out["v2_risk_gate"]
    )
    return out


def prepare_phase_a_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_phase_a_price_trend_features(decompose_legacy_scanner(frame))
    out[LEGACY_SCANNER_TREND_FILTER_CONTROL_V0.key] = (
        LEGACY_SCANNER_TREND_FILTER_CONTROL_V0.evaluate(out)
    )
    return out


def _groupwise_prior_max(
    frame: pd.DataFrame,
    column: str,
    lookback: int,
) -> pd.Series:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if "symbol" not in frame:
        raise ValueError("signal evaluation requires a symbol column")
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("symbol", sort=False):
        ordered = group.sort_values("date")
        values = _numeric(ordered, column)
        out.loc[ordered.index] = values.shift(1).rolling(lookback, min_periods=lookback).max()
    return out


def make_score_breakout_rule(score_lookback: int) -> SignalRule:
    if score_lookback <= 0:
        raise ValueError("score_lookback must be positive")

    def _signal(frame: pd.DataFrame) -> pd.Series:
        score = _numeric(frame, "score")
        return (score > 0) & (score > _groupwise_prior_max(frame, "score", score_lookback))

    return SignalRule(
        key=f"score_breakout_l{score_lookback}",
        label=f"Score breakout L{score_lookback}",
        description=(
            "Positive score exceeds its highest value over the prior score_lookback "
            "sessions. R20 and ER20 are observable diagnostics, not thresholds."
        ),
        params={"family": "score_breakout", "score_lookback": score_lookback},
        signal_fn=_signal,
    )


def build_score_breakout_rules(
    score_lookbacks: Iterable[int] = SCORE_LOOKBACK_GRID,
) -> list[SignalRule]:
    return [make_score_breakout_rule(int(lookback)) for lookback in score_lookbacks]


def make_prior_price_high_rule(lookback: int = PRICE_BREAKOUT_LOOKBACK) -> SignalRule:
    def _signal(frame: pd.DataFrame) -> pd.Series:
        close = _numeric(frame, "close")
        return close > _groupwise_prior_max(frame, "high", lookback)

    return SignalRule(
        key=f"prior_price_high_l{lookback}",
        label=f"Prior price high L{lookback}",
        description=f"Close exceeds the highest high of the prior {lookback} sessions.",
        params={"family": "prior_price_high", "lookback": lookback},
        signal_fn=_signal,
        role="comparator",
    )


TREND_FILTER_ONLY_RULE = SignalRule(
    key="trend_filter_only",
    label="Trend filter only",
    description="The shared Phase A trend filter is the event state; no added trigger.",
    params={"family": "trend_filter_only"},
    signal_fn=lambda frame: pd.Series(True, index=frame.index, dtype=bool),
    role="comparator",
)


LEGACY_SIGNAL_SURGE_V0_RULE = SignalRule(
    key="signal_surge_v0",
    label="Legacy signal_surge_v0",
    description="Unchanged v1 scanner Boolean retained only as historical baseline evidence.",
    params={"family": "legacy", "version": "signal_surge_v0"},
    signal_fn=lambda frame: add_signal_surge_v0(frame)["signal_surge_v0"],
    role="legacy_baseline",
)


def evaluate_signal_observations(
    frame: pd.DataFrame,
    signal_rules: Sequence[SignalRule],
    components: PhaseAComponents,
) -> pd.DataFrame:
    required = {"date", "symbol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"signal observations missing columns: {sorted(missing)}")
    work = prepare_phase_a_features(frame).sort_values(["symbol", "date"]).copy()
    eligible = components.universe.evaluate(work)
    trend = components.trend_filter.evaluate(work)
    observations = work[["date", "symbol"]].copy()
    observations["universe_eligible"] = eligible
    observations["trend_filter"] = trend
    observations[LEGACY_SCANNER_TREND_FILTER_CONTROL_V0.key] = work[
        LEGACY_SCANNER_TREND_FILTER_CONTROL_V0.key
    ]
    for rule in signal_rules:
        trigger = rule.evaluate(work)
        if rule.role == "legacy_baseline":
            observations[rule.key] = trigger
        else:
            observations[rule.key] = eligible & trend & trigger
    return observations


def frequency_matched_random_events(
    frame: pd.DataFrame,
    target_events: pd.Series,
    eligible_events: pd.Series,
    *,
    seed: int,
) -> pd.Series:
    """Match per-symbol first-event executable triggers with isolated random dates."""
    target = _bool_series(target_events, frame.index)
    eligible = _bool_series(eligible_events, frame.index)
    result = pd.Series(False, index=frame.index, dtype=bool)
    for symbol, group in frame.groupby("symbol", sort=True):
        ordered = group.sort_values("date")
        target_values = target.loc[ordered.index].reset_index(drop=True)
        target_count = len(_first_true_indices(target_values))
        if target_count == 0:
            continue
        digest = hashlib.sha256(f"{seed}:{symbol}".encode("utf-8")).digest()
        symbol_seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(symbol_seed)
        eligible_positions = np.flatnonzero(
            eligible.loc[ordered.index].to_numpy(dtype=bool)
        )
        maximum_isolated_positions: list[int] = []
        run_start = 0
        while run_start < len(eligible_positions):
            run_end = run_start + 1
            while (
                run_end < len(eligible_positions)
                and eligible_positions[run_end] == eligible_positions[run_end - 1] + 1
            ):
                run_end += 1
            run = eligible_positions[run_start:run_end]
            offset = int(rng.integers(0, 2)) if len(run) % 2 == 0 else 0
            maximum_isolated_positions.extend(run[offset::2].tolist())
            run_start = run_end
        if target_count > len(maximum_isolated_positions):
            raise ValueError(
                f"cannot frequency-match executable triggers for {symbol}: "
                f"requested {target_count}, maximum isolated eligible dates "
                f"{len(maximum_isolated_positions)}"
            )
        rng.shuffle(maximum_isolated_positions)
        chosen_positions = sorted(maximum_isolated_positions[:target_count])
        chosen_indices = ordered.index.to_numpy()[chosen_positions]
        result.loc[chosen_indices] = True
        matched_count = len(
            _first_true_indices(result.loc[ordered.index].reset_index(drop=True))
        )
        if matched_count != target_count:
            raise RuntimeError(
                f"random executable-trigger matching failed for {symbol}: "
                f"expected {target_count}, observed {matched_count}"
            )
    return result


def within_symbol_shifted_events(
    frame: pd.DataFrame,
    events: pd.Series,
    *,
    shift_bars: int = PLACEBO_SHIFT_BARS,
) -> pd.Series:
    if shift_bars <= 0:
        raise ValueError("shift_bars must be positive")
    event_mask = _bool_series(events, frame.index)
    result = pd.Series(False, index=frame.index, dtype=bool)
    for _, group in frame.groupby("symbol", sort=False):
        ordered = group.sort_values("date")
        shifted = event_mask.loc[ordered.index].shift(shift_bars, fill_value=False)
        result.loc[ordered.index] = shifted.to_numpy(dtype=bool)
    return result


def _first_true_indices(signal: pd.Series) -> list[int]:
    values = _bool_series(signal, signal.index).reset_index(drop=True)
    previous = values.shift(1, fill_value=False).astype(bool)
    return values.index[values & ~previous].tolist()


def signal_event_counts(
    frame: pd.DataFrame,
    events: pd.Series,
    entry_rule: EntryRule,
) -> dict[str, int]:
    event_mask = _bool_series(events, frame.index)
    executable_count = 0
    for _, group in frame.groupby("symbol", sort=False):
        ordered = group.sort_values("date")
        executable_count += len(
            entry_rule.signal_indices(event_mask.loc[ordered.index].reset_index(drop=True))
        )
    return {
        "raw_boolean_signal_count": int(event_mask.sum()),
        "executable_trigger_count": int(executable_count),
    }


def _equal_weights(symbols: Sequence[str]) -> dict[str, float]:
    unique = tuple(dict.fromkeys(str(symbol) for symbol in symbols))
    if not unique:
        return {}
    weight = 1.0 / len(unique)
    return {symbol: weight for symbol in unique}


def _canonical_portfolio(
    lifecycles: pd.DataFrame,
    prices: pd.DataFrame,
    round_trip_cost: float,
) -> pd.DataFrame:
    panel = build_price_panel(prices)
    return simulate_canonical_portfolio(
        lifecycles,
        panel,
        round_trip_cost=round_trip_cost,
    )


def default_phase_a_components() -> PhaseAComponents:
    return PhaseAComponents(
        universe=UniverseEligibilityRule(
            "historical_eligible_universe",
            "Use the supplied historical eligible_universe state without reconstruction.",
            lambda frame: frame.get("eligible_universe", False),
        ),
        trend_filter=PRIMARY_PHASE_A_TREND_FILTER,
        entry=EntryRule(
            "first_event_next_open",
            "Enter at the next valid session open after the first event in a run.",
            _first_true_indices,
        ),
        initial_stop=InitialStopRule(
            "signal_day_low20",
            "Use signal-day Low20 as the initial stop.",
            lambda row: _number(row, "low20"),
        ),
        trailing_exit=TrailingExitRule(
            "ratcheting_low20",
            "Ratchet the Low20 stop upward; exit on a low breach with gap-aware pricing.",
            lambda row: _number(row, "low20"),
        ),
        position_sizing=PositionSizingRule(
            CANONICAL_EQUAL_WEIGHT_KEY,
            "Fixed canonical equal weight; not an optimized or interchangeable Phase A component.",
            _equal_weights,
        ),
        portfolio_construction=PortfolioConstructionRule(
            CANONICAL_EQUAL_WEIGHT_KEY,
            "Reuse the canonical equal-weight active portfolio simulator.",
            _canonical_portfolio,
        ),
    )


def validate_phase_a_components(components: PhaseAComponents) -> None:
    if components.position_sizing.key != CANONICAL_EQUAL_WEIGHT_KEY:
        raise ValueError(
            "Phase A supports only fixed canonical equal weight; unsupported position sizing "
            f"rule: {components.position_sizing.key}"
        )
    if components.portfolio_construction.key != CANONICAL_EQUAL_WEIGHT_KEY:
        raise ValueError(
            "Phase A supports only canonical_equal_weight_active_v1 portfolio construction"
        )
    probe = components.position_sizing.target_weights(("A", "B"))
    if probe != {"A": 0.5, "B": 0.5}:
        raise ValueError("Phase A canonical sizing declaration does not produce equal weights")


def build_signal_diagnostics(
    features: pd.DataFrame,
    signal_events: Mapping[str, pd.Series],
    *,
    horizons: Sequence[int] = FORWARD_RETURN_HORIZONS,
    excursion_horizon: int = EXCURSION_HORIZON,
) -> pd.DataFrame:
    if any(int(horizon) <= 0 for horizon in horizons) or excursion_horizon <= 0:
        raise ValueError("diagnostic horizons must be positive")
    rows: list[dict[str, object]] = []
    for symbol, group in features.groupby("symbol", sort=True):
        ordered = group.sort_values("date")
        close = _numeric(ordered, "close").reset_index(drop=True)
        high = _numeric(ordered, "high").reset_index(drop=True)
        low = _numeric(ordered, "low").reset_index(drop=True)
        original_indices = ordered.index.to_list()
        for signal_key, event_series in signal_events.items():
            events = _bool_series(event_series, features.index).loc[original_indices].to_numpy(dtype=bool)
            for position in np.flatnonzero(events):
                base_close = close.iloc[position]
                if not np.isfinite(base_close) or base_close <= 0:
                    continue
                row: dict[str, object] = {
                    "signal_key": signal_key,
                    "symbol": str(symbol),
                    "signal_date": pd.Timestamp(ordered.iloc[position]["date"]).date().isoformat(),
                    "score": _number(ordered.iloc[position], "score"),
                    "r20": _number(ordered.iloc[position], "r20"),
                    "er20": _number(ordered.iloc[position], "er20"),
                    "atr20_pct": _number(ordered.iloc[position], "atr20_pct"),
                }
                for horizon in horizons:
                    target = position + int(horizon)
                    value = close.iloc[target] if target < len(close) else np.nan
                    row[f"forward_return_{int(horizon)}d"] = (
                        float(value / base_close - 1.0) if np.isfinite(value) else np.nan
                    )
                end = min(position + excursion_horizon, len(ordered) - 1)
                future_high = high.iloc[position + 1 : end + 1]
                future_low = low.iloc[position + 1 : end + 1]
                row[f"mfe_{excursion_horizon}d"] = (
                    float(future_high.max() / base_close - 1.0)
                    if future_high.notna().any()
                    else np.nan
                )
                row[f"mae_{excursion_horizon}d"] = (
                    float(future_low.min() / base_close - 1.0)
                    if future_low.notna().any()
                    else np.nan
                )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_signal_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    horizons: Sequence[int] = FORWARD_RETURN_HORIZONS,
    excursion_horizon: int = EXCURSION_HORIZON,
) -> pd.DataFrame:
    columns = ["signal_key", "signal_count"]
    for horizon in horizons:
        columns.extend(
            [
                f"mean_forward_return_{int(horizon)}d",
                f"median_forward_return_{int(horizon)}d",
                f"positive_forward_return_rate_{int(horizon)}d",
            ]
        )
    columns.extend(
        [
            f"mean_mfe_{excursion_horizon}d",
            f"median_mfe_{excursion_horizon}d",
            f"mean_mae_{excursion_horizon}d",
            f"median_mae_{excursion_horizon}d",
        ]
    )
    if diagnostics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for signal_key, group in diagnostics.groupby("signal_key", sort=True):
        row: dict[str, object] = {"signal_key": signal_key, "signal_count": int(len(group))}
        for horizon in horizons:
            values = pd.to_numeric(group[f"forward_return_{int(horizon)}d"], errors="coerce").dropna()
            row[f"mean_forward_return_{int(horizon)}d"] = float(values.mean()) if len(values) else np.nan
            row[f"median_forward_return_{int(horizon)}d"] = float(values.median()) if len(values) else np.nan
            row[f"positive_forward_return_rate_{int(horizon)}d"] = (
                float((values > 0).mean()) if len(values) else np.nan
            )
        mfe = pd.to_numeric(group[f"mfe_{excursion_horizon}d"], errors="coerce").dropna()
        mae = pd.to_numeric(group[f"mae_{excursion_horizon}d"], errors="coerce").dropna()
        row[f"mean_mfe_{excursion_horizon}d"] = float(mfe.mean()) if len(mfe) else np.nan
        row[f"median_mfe_{excursion_horizon}d"] = float(mfe.median()) if len(mfe) else np.nan
        row[f"mean_mae_{excursion_horizon}d"] = float(mae.mean()) if len(mae) else np.nan
        row[f"median_mae_{excursion_horizon}d"] = float(mae.median()) if len(mae) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def simulate_signal_lifecycles(
    features: pd.DataFrame,
    events: pd.Series,
    *,
    strategy_key: str,
    components: PhaseAComponents,
    round_trip_cost: float = ROUND_TRIP_COST,
) -> pd.DataFrame:
    """Apply the common Phase A rules without a fixed holding-period exit."""
    validate_phase_a_components(components)
    if round_trip_cost < 0:
        raise ValueError("round_trip_cost cannot be negative")
    work = features.copy()
    work["__event"] = _bool_series(events, features.index)
    records: list[dict[str, object]] = []
    for symbol, group in work.groupby("symbol", sort=True):
        ordered = group.sort_values("date").reset_index(drop=True)
        signal_indices = components.entry.signal_indices(ordered["__event"])
        next_allowed = 0
        for signal_index in signal_indices:
            entry_index = signal_index + 1
            if entry_index >= len(ordered) or entry_index < next_allowed:
                continue
            signal_row = ordered.iloc[signal_index]
            entry_row = ordered.iloc[entry_index]
            entry_price = _number(entry_row, "open")
            active_stop = components.initial_stop.initial_stop(signal_row)
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            if not np.isfinite(active_stop) or active_stop <= 0 or entry_price <= active_stop:
                continue
            exit_index: int | None = None
            exit_price = np.nan
            stop_at_exit = active_stop
            for position in range(entry_index, len(ordered)):
                row = ordered.iloc[position]
                day_open = _number(row, "open")
                day_low = _number(row, "low")
                if np.isfinite(day_low) and day_low <= active_stop:
                    exit_index = position
                    exit_price = day_open if np.isfinite(day_open) and day_open < active_stop else active_stop
                    stop_at_exit = active_stop
                    break
                candidate = components.trailing_exit.candidate_stop(row)
                if np.isfinite(candidate) and candidate > 0:
                    active_stop = (
                        max(active_stop, candidate)
                        if components.trailing_exit.ratchet_only
                        else candidate
                    )
            entry_date = pd.Timestamp(entry_row["date"]).date().isoformat()
            base = {
                "strategy_key": strategy_key,
                "symbol": str(symbol),
                "entry_signal_date": pd.Timestamp(signal_row["date"]).date().isoformat(),
                "entry_date": entry_date,
                "entry_price": float(entry_price),
                "initial_stop": float(components.initial_stop.initial_stop(signal_row)),
                "entry_rule": components.entry.key,
                "initial_stop_rule": components.initial_stop.key,
                "trailing_exit_rule": components.trailing_exit.key,
                "position_sizing_rule": components.position_sizing.key,
                "portfolio_construction_rule": components.portfolio_construction.key,
            }
            if exit_index is None:
                records.append(
                    {
                        **base,
                        "exit_date": None,
                        "exit_price": None,
                        "exit_reason": "open_at_end",
                        "stop_at_exit": float(active_stop),
                        "gross_return": np.nan,
                        "net_return": np.nan,
                        "holding_days": int(len(ordered) - entry_index),
                    }
                )
                next_allowed = len(ordered)
                continue
            gross_return = float(exit_price / entry_price - 1.0)
            records.append(
                {
                    **base,
                    "exit_date": pd.Timestamp(ordered.iloc[exit_index]["date"]).date().isoformat(),
                    "exit_price": float(exit_price),
                    "exit_reason": "stop_hit",
                    "stop_at_exit": float(stop_at_exit),
                    "gross_return": gross_return,
                    "net_return": gross_return - float(round_trip_cost),
                    "holding_days": int(exit_index - entry_index + 1),
                }
            )
            next_allowed = exit_index + 1
    return pd.DataFrame(records)


def summarize_trade_diagnostics(lifecycles: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "strategy_key",
        "completed_trades",
        "profit_factor",
        "win_rate",
        "mean_trade_return",
        "median_trade_return",
    ]
    if lifecycles.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for strategy_key, group in lifecycles.groupby("strategy_key", sort=True):
        values = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        gains = float(values.loc[values > 0].sum())
        losses = abs(float(values.loc[values < 0].sum()))
        rows.append(
            {
                "strategy_key": strategy_key,
                "completed_trades": int(len(values)),
                "profit_factor": gains / losses if losses > 0 else (math.inf if gains > 0 else np.nan),
                "win_rate": float((values > 0).mean()) if len(values) else np.nan,
                "mean_trade_return": float(values.mean()) if len(values) else np.nan,
                "median_trade_return": float(values.median()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _curve_from_equity(
    dates: pd.Series,
    equity: pd.Series,
    *,
    exposure: pd.Series | None = None,
    positions: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> pd.DataFrame:
    clean = pd.DataFrame({"date": pd.to_datetime(dates), "portfolio_equity": pd.to_numeric(equity)})
    clean = clean.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if clean.empty:
        return pd.DataFrame()
    normalized = clean["portfolio_equity"] / float(clean["portfolio_equity"].iloc[0]) * PORTFOLIO_INITIAL_CAPITAL
    daily = normalized.pct_change().fillna(0.0)
    peak = normalized.cummax().clip(lower=PORTFOLIO_INITIAL_CAPITAL)
    economic = pd.DataFrame(
        {
            "date": clean["date"].dt.date.astype(str),
            "observation_type": "trading_session",
            "portfolio_equity": normalized,
            "daily_portfolio_return": daily,
            "drawdown": normalized / peak - 1.0,
            "gross_exposure": 0.0 if exposure is None else pd.to_numeric(exposure, errors="coerce").fillna(0.0).to_numpy(),
            "active_position_count": 0 if positions is None else pd.to_numeric(positions, errors="coerce").fillna(0.0).to_numpy(),
            "turnover": 0.0 if turnover is None else pd.to_numeric(turnover, errors="coerce").fillna(0.0).to_numpy(),
            "transaction_cost_paid": 0.0 if costs is None else pd.to_numeric(costs, errors="coerce").fillna(0.0).to_numpy(),
        }
    )
    initialization = pd.DataFrame(
        [
            {
                "date": (clean["date"].iloc[0].normalize() - pd.Timedelta(microseconds=1)).isoformat() + "Z",
                "observation_type": "initialization",
                "portfolio_equity": PORTFOLIO_INITIAL_CAPITAL,
                "daily_portfolio_return": 0.0,
                "drawdown": 0.0,
                "gross_exposure": 0.0,
                "active_position_count": 0,
                "turnover": 0.0,
                "transaction_cost_paid": 0.0,
            }
        ]
    )
    return pd.concat([initialization, economic], ignore_index=True)


def summarize_relative_portfolio(
    strategy_key: str,
    curve: pd.DataFrame,
    spy_prices: pd.DataFrame,
) -> dict[str, object]:
    strategy = curve.loc[curve["observation_type"] == "trading_session"].copy()
    strategy["economic_date"] = pd.to_datetime(strategy["date"]).dt.normalize()
    spy = spy_prices.loc[spy_prices["symbol"].astype(str) == "SPY", ["date", "close"]].copy()
    spy["economic_date"] = pd.to_datetime(spy["date"], errors="coerce").dt.normalize()
    spy["spy_close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna(subset=["economic_date", "spy_close"]).drop_duplicates("economic_date", keep="last")
    common = strategy.merge(spy[["economic_date", "spy_close"]], on="economic_date", how="inner")
    if len(common) < 2:
        raise ValueError("SPY comparison requires at least two exact common economic dates")
    strategy_common = _curve_from_equity(
        common["economic_date"],
        common["portfolio_equity"],
        exposure=common["gross_exposure"],
        positions=common["active_position_count"],
        turnover=common["turnover"],
        costs=common["transaction_cost_paid"],
    )
    spy_common = _curve_from_equity(common["economic_date"], common["spy_close"])
    strategy_summary = summarize_portfolio_curve(strategy_key, strategy_key, strategy_common)
    spy_summary = summarize_portfolio_curve("SPY", "SPY", spy_common)

    def ratio(numerator: object, denominator: object, *, magnitude: bool = False) -> float:
        if not _finite(numerator) or not _finite(denominator):
            return np.nan
        left = float(numerator)
        right = float(denominator)
        if magnitude:
            left, right = abs(left), abs(right)
        return left / right if np.isfinite(right) and abs(right) > 1e-12 else np.nan

    start = pd.Timestamp(common["economic_date"].min())
    end = pd.Timestamp(common["economic_date"].max())
    common_years = max((end - start).days / 365.25, 0.0)
    return {
        "strategy_key": strategy_key,
        "common_start_date": start.date().isoformat(),
        "common_end_date": end.date().isoformat(),
        "common_date_count": int(len(common)),
        "common_years": float(common_years),
        "strategy_cagr": strategy_summary["cagr"],
        "spy_cagr": spy_summary["cagr"],
        "strategy_cagr_spy_ratio": ratio(strategy_summary["cagr"], spy_summary["cagr"]),
        "strategy_maximum_drawdown": strategy_summary["maximum_drawdown"],
        "spy_maximum_drawdown": spy_summary["maximum_drawdown"],
        "maximum_drawdown_spy_ratio": ratio(
            strategy_summary["maximum_drawdown"], spy_summary["maximum_drawdown"], magnitude=True
        ),
        "strategy_cdar95": strategy_summary["conditional_drawdown_at_risk_95"],
        "spy_cdar95": spy_summary["conditional_drawdown_at_risk_95"],
        "cdar95_spy_ratio": ratio(
            strategy_summary["conditional_drawdown_at_risk_95"],
            spy_summary["conditional_drawdown_at_risk_95"],
            magnitude=True,
        ),
        "strategy_calmar": strategy_summary["calmar_ratio"],
        "spy_calmar": spy_summary["calmar_ratio"],
        "calmar_spy_ratio": ratio(strategy_summary["calmar_ratio"], spy_summary["calmar_ratio"]),
        "strategy_recovery_duration_days": strategy_summary["longest_time_under_water_days"],
        "spy_recovery_duration_days": spy_summary["longest_time_under_water_days"],
        "annual_turnover": strategy_summary["annual_turnover"],
        "total_transaction_cost": strategy_summary["total_transaction_cost"],
    }


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _dominates(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    higher = ("strategy_cagr", "strategy_calmar")
    lower = ("strategy_maximum_drawdown", "strategy_cdar95", "strategy_recovery_duration_days")
    if not all(_finite(left.get(key)) and _finite(right.get(key)) for key in (*higher, *lower)):
        return False
    no_worse = all(float(left[key]) >= float(right[key]) - 1e-12 for key in higher)
    no_worse &= all(abs(float(left[key])) <= abs(float(right[key])) + 1e-12 for key in lower[:2])
    no_worse &= float(left[lower[2]]) <= float(right[lower[2]]) + 1e-12
    strict = any(float(left[key]) > float(right[key]) + 1e-12 for key in higher)
    strict |= any(abs(float(left[key])) < abs(float(right[key])) - 1e-12 for key in lower[:2])
    strict |= float(left[lower[2]]) < float(right[lower[2]]) - 1e-12
    return bool(no_worse and strict)


def _candidate_robustness_evidence(
    robustness_evidence: Mapping[str, object] | None,
    strategy_key: str,
) -> Mapping[str, object]:
    if robustness_evidence is None:
        return {}
    candidate = robustness_evidence.get(strategy_key)
    return candidate if isinstance(candidate, Mapping) else robustness_evidence


def _missing_or_invalid_robustness_fields(evidence: Mapping[str, object]) -> list[str]:
    missing = [
        field
        for field in REQUIRED_ROBUSTNESS_EVIDENCE_FIELDS
        if field not in evidence or evidence.get(field) is None
    ]
    if (
        "multiple_testing_adjusted_p_value" not in evidence
        and "corrected_evidence_pass" not in evidence
    ):
        missing.append("multiple_testing_adjusted_p_value_or_corrected_evidence_pass")
    if (
        "multiple_testing_adjusted_p_value" in evidence
        and not _finite(evidence.get("multiple_testing_adjusted_p_value"))
    ):
        missing.append("multiple_testing_adjusted_p_value (invalid)")
    if "corrected_evidence_pass" in evidence and not isinstance(
        evidence.get("corrected_evidence_pass"), (bool, np.bool_)
    ):
        missing.append("corrected_evidence_pass (invalid)")
    numeric_fields = (
        "walk_forward_fold_count",
        "walk_forward_improvement_ratio",
        "leave_one_year_out_result_count",
        "leave_one_year_out_stability_ratio",
    )
    for field in numeric_fields:
        if field in evidence and not _finite(evidence.get(field)):
            missing.append(f"{field} (invalid)")
    interval = evidence.get("date_block_bootstrap_paired_effect_confidence_interval")
    if interval is not None:
        valid_interval = (
            isinstance(interval, Sequence)
            and not isinstance(interval, (str, bytes))
            and len(interval) == 2
            and all(_finite(value) for value in interval)
            and float(interval[0]) <= float(interval[1])
        )
        if not valid_interval:
            missing.append(
                "date_block_bootstrap_paired_effect_confidence_interval (invalid)"
            )
    asset_groups = evidence.get("asset_group_concentration_diagnostics")
    if asset_groups is not None:
        if not isinstance(asset_groups, Mapping) or not _finite(
            asset_groups.get("dominant_effect_share")
        ):
            missing.append("asset_group_concentration_diagnostics (invalid)")
    for field in ("event_count_comparability", "executable_trigger_count_comparability"):
        if field in evidence and not isinstance(evidence.get(field), (bool, np.bool_)):
            missing.append(f"{field} (invalid)")
    return list(dict.fromkeys(missing))


def _robustness_outcomes(evidence: Mapping[str, object]) -> tuple[bool, bool]:
    interval = evidence["date_block_bootstrap_paired_effect_confidence_interval"]
    lower, upper = float(interval[0]), float(interval[1])
    corrected_pass = (
        bool(evidence["corrected_evidence_pass"])
        if "corrected_evidence_pass" in evidence
        else float(evidence["multiple_testing_adjusted_p_value"])
        <= MULTIPLE_TESTING_ADJUSTED_ALPHA
    )
    asset_share = abs(
        float(evidence["asset_group_concentration_diagnostics"]["dominant_effect_share"])
    )
    common_pass = bool(
        int(evidence["walk_forward_fold_count"]) >= MIN_WALK_FORWARD_FOLDS
        and int(evidence["leave_one_year_out_result_count"]) >= MIN_LOYO_RESULTS
        and float(evidence["leave_one_year_out_stability_ratio"])
        >= MIN_LOYO_STABILITY_RATIO
        and corrected_pass
        and asset_share <= MAX_DOMINANT_ASSET_GROUP_EFFECT_SHARE
        and bool(evidence["event_count_comparability"])
        and bool(evidence["executable_trigger_count_comparability"])
    )
    improvement_ratio = float(evidence["walk_forward_improvement_ratio"])
    retain_support = bool(
        common_pass
        and improvement_ratio > 0.50
        and lower >= -BOOTSTRAP_MATERIAL_EFFECT_TOLERANCE
    )
    reject_support = bool(
        common_pass
        and improvement_ratio <= 0.50
        and upper <= BOOTSTRAP_MATERIAL_EFFECT_TOLERANCE
    )
    return retain_support, reject_support


def classify_score_breakout(
    portfolio_metrics: pd.DataFrame,
    signal_summary: pd.DataFrame,
    *,
    robustness_evidence: Mapping[str, object] | None = None,
    minimum_common_years: float = 5.0,
    minimum_score_signals: int = 100,
) -> dict[str, object]:
    """Return a deterministic research classification, never production approval."""
    score_rows = portfolio_metrics.loc[portfolio_metrics["variant"] == "score_breakout"]
    if score_rows.empty:
        return {"classification": "Inconclusive", "reason": "No score-breakout result was supplied."}
    evidence_by_key: dict[str, Mapping[str, object]] = {}
    for key in score_rows["strategy_key"].astype(str):
        evidence = _candidate_robustness_evidence(robustness_evidence, key)
        missing = _missing_or_invalid_robustness_fields(evidence)
        if missing:
            return {
                "classification": "Inconclusive",
                "reason": (
                    f"Missing or invalid robustness evidence for {key}: "
                    f"{', '.join(missing)}."
                ),
                "missing_robustness_evidence": missing,
            }
        evidence_by_key[key] = evidence
    counts = signal_summary.set_index("signal_key")["signal_count"].to_dict() if not signal_summary.empty else {}
    adequate = score_rows.loc[
        (pd.to_numeric(score_rows["common_years"], errors="coerce") >= minimum_common_years)
        & score_rows["strategy_key"].map(lambda key: int(counts.get(key, 0)) >= minimum_score_signals)
    ]
    if adequate.empty:
        return {
            "classification": "Inconclusive",
            "reason": (
                f"No score lookback has both {minimum_common_years:g} common SPY years and "
                f"{minimum_score_signals} signal observations."
            ),
        }
    records = {str(row["strategy_key"]): row for row in portfolio_metrics.to_dict(orient="records")}
    trend = next((row for row in records.values() if row.get("variant") == "trend_filter_only"), None)
    price = next((row for row in records.values() if row.get("variant") == "prior_price_high"), None)
    if trend is None or price is None:
        return {"classification": "Inconclusive", "reason": "A deterministic comparator is missing."}
    count_fields = (
        "raw_boolean_signal_count",
        "executable_trigger_count",
        "completed_lifecycle_count",
    )
    for comparator in (trend, price):
        if any(not _finite(comparator.get(field)) for field in count_fields):
            return {
                "classification": "Inconclusive",
                "reason": (
                    "Signal/execution count diagnostics are incomplete for "
                    f"{comparator.get('strategy_key')}."
                ),
            }

    retain_candidates: list[tuple[int, str]] = []
    reject_flags: list[bool] = []
    for score in adequate.to_dict(orient="records"):
        key = str(score["strategy_key"])
        if any(not _finite(score.get(field)) for field in count_fields):
            return {
                "classification": "Inconclusive",
                "reason": f"Signal/execution count diagnostics are incomplete for {key}.",
            }
        placebo = portfolio_metrics.loc[
            (portfolio_metrics["parent_signal_key"] == key)
            & portfolio_metrics["variant"].isin(["frequency_matched_random", "shifted_placebo"])
        ]
        if placebo.empty or set(placebo["variant"]) != {"frequency_matched_random", "shifted_placebo"}:
            return {"classification": "Inconclusive", "reason": f"Placebo evidence is incomplete for {key}."}
        if any(
            not _finite(row.get(field))
            for row in placebo.to_dict(orient="records")
            for field in count_fields
        ):
            return {
                "classification": "Inconclusive",
                "reason": f"Placebo count diagnostics are incomplete for {key}.",
            }
        score_executable = int(score["executable_trigger_count"])
        mismatched = placebo.loc[
            pd.to_numeric(placebo["executable_trigger_count"], errors="coerce")
            != score_executable
        ]
        if not mismatched.empty:
            mismatch_keys = sorted(mismatched["strategy_key"].astype(str).tolist())
            return {
                "classification": "Inconclusive",
                "reason": (
                    f"Executable-trigger comparability failed for {key}: "
                    f"{', '.join(mismatch_keys)}."
                ),
            }
        placebo_medians = []
        for _, group in placebo.groupby("variant", sort=True):
            median = group.select_dtypes(include=[np.number]).median(numeric_only=True).to_dict()
            placebo_medians.append(median)
        passes_spy = bool(
            float(score["strategy_cagr_spy_ratio"]) >= 0.80
            and float(score["maximum_drawdown_spy_ratio"]) <= 0.75
            and float(score["cdar95_spy_ratio"]) <= 0.80
            and float(score["calmar_spy_ratio"]) >= 1.0
        )
        dominates_all = _dominates(score, trend) and _dominates(score, price)
        dominates_all &= all(_dominates(score, median) for median in placebo_medians)
        retain_support, reject_support = _robustness_outcomes(evidence_by_key[key])
        if passes_spy and dominates_all and retain_support:
            retain_candidates.append((int(score["score_lookback"]), key))
        reject_flags.append(
            reject_support
            and
            (_dominates(trend, score) or _dominates(price, score))
            and any(_dominates(median, score) for median in placebo_medians)
        )
    if retain_candidates:
        lookback, key = sorted(retain_candidates)[0]
        return {
            "classification": "Retain",
            "reason": "A score candidate passes every provisional SPY gate and Pareto-dominates all controls.",
            "score_lookback": lookback,
            "strategy_key": key,
        }
    if reject_flags and all(reject_flags):
        return {
            "classification": "Reject",
            "reason": "Every adequately observed score candidate is dominated by a deterministic control and a placebo control.",
        }
    return {
        "classification": "Inconclusive",
        "reason": (
            "Complete evidence was supplied, but the preregistered portfolio, "
            "robustness, and control-comparability conditions do not jointly "
            "support Retain or Reject."
        ),
    }


def phase_a_methodology(
    components: PhaseAComponents,
    *,
    score_lookbacks: Sequence[int],
    random_seeds: Sequence[int],
    round_trip_cost: float,
) -> dict[str, object]:
    return {
        "phase": "Phase A1 comparison framework",
        "empirical_score_breakout_classification": "not_run_requires_phase_a2",
        "universe_eligibility": components.universe.key,
        "trend_filter": components.trend_filter.key,
        "trend_filter_conditions": [
            "close > phase_a_ma200",
            "phase_a_ma200_slope_20 > 0",
        ],
        "trend_filter_status": "fixed_score_independent_phase_a_primary",
        "trend_filter_sensitivity_control": LEGACY_SCANNER_TREND_FILTER_CONTROL_V0.key,
        "score_breakout_numeric_parameter": "score_lookback",
        "score_lookbacks": [int(value) for value in score_lookbacks],
        "r20_er20_role": "diagnostics_and_ranking_only",
        "comparators": [
            "trend_filter_only",
            f"prior_price_high_l{PRICE_BREAKOUT_LOOKBACK}",
            "frequency_matched_random_within_symbol",
            f"within_symbol_shifted_signal_{PLACEBO_SHIFT_BARS}_bars",
        ],
        "random_seeds": [int(seed) for seed in random_seeds],
        "entry": components.entry.key,
        "initial_stop": components.initial_stop.key,
        "trailing_exit": components.trailing_exit.key,
        "fixed_holding_period_exit": None,
        "fixed_profit_target": None,
        "position_sizing": components.position_sizing.key,
        "position_sizing_status": "fixed_canonical_equal_weight_not_optimized",
        "portfolio_construction": components.portfolio_construction.key,
        "round_trip_cost": float(round_trip_cost),
        "comparison_dates": "exact_common_strategy_and_spy_economic_dates",
        "primary_portfolio_metrics": [
            "strategy_cagr_spy_ratio",
            "maximum_drawdown_spy_ratio",
            "cdar95_spy_ratio",
            "calmar_spy_ratio",
            "strategy_recovery_duration_days",
            "annual_turnover",
            "total_transaction_cost",
        ],
        "trade_metric_role": "diagnostic_only",
        "classification_requires_external_robustness_evidence": [
            *REQUIRED_ROBUSTNESS_EVIDENCE_FIELDS,
            "multiple_testing_adjusted_p_value_or_corrected_evidence_pass",
        ],
        "robustness_thresholds": {
            "minimum_walk_forward_folds": MIN_WALK_FORWARD_FOLDS,
            "minimum_walk_forward_improvement_ratio_exclusive": 0.50,
            "minimum_leave_one_year_out_results": MIN_LOYO_RESULTS,
            "minimum_leave_one_year_out_stability_ratio": MIN_LOYO_STABILITY_RATIO,
            "bootstrap_material_effect_tolerance": BOOTSTRAP_MATERIAL_EFFECT_TOLERANCE,
            "multiple_testing_adjusted_alpha": MULTIPLE_TESTING_ADJUSTED_ALPHA,
            "maximum_dominant_asset_group_effect_share": MAX_DOMINANT_ASSET_GROUP_EFFECT_SHARE,
        },
        "signal_only_diagnostics": {
            "forward_return_horizons": list(FORWARD_RETURN_HORIZONS),
            "mfe_horizon": EXCURSION_HORIZON,
            "mae_horizon": EXCURSION_HORIZON,
            "exit_rule": None,
        },
    }


def run_phase_a_signal_comparison(
    features: pd.DataFrame,
    *,
    components: PhaseAComponents | None = None,
    score_lookbacks: Sequence[int] = SCORE_LOOKBACK_GRID,
    random_seeds: Sequence[int] = FREQUENCY_MATCHED_RANDOM_SEEDS,
    round_trip_cost: float = ROUND_TRIP_COST,
    robustness_evidence: Mapping[str, object] | None = None,
) -> PhaseAComparisonResult:
    """Run every Phase A signal with one shared execution and portfolio path."""
    selected = components or default_phase_a_components()
    validate_phase_a_components(selected)
    ordered = prepare_phase_a_features(features).sort_values(["symbol", "date"]).copy()
    score_rules = build_score_breakout_rules(score_lookbacks)
    base_rules = [TREND_FILTER_ONLY_RULE, make_prior_price_high_rule(), *score_rules, LEGACY_SIGNAL_SURGE_V0_RULE]
    observations = evaluate_signal_observations(ordered, base_rules, selected)
    signal_events: dict[str, pd.Series] = {
        rule.key: _bool_series(observations[rule.key], ordered.index) for rule in base_rules
    }
    eligible_control = _bool_series(observations["universe_eligible"], ordered.index) & _bool_series(
        observations["trend_filter"], ordered.index
    )
    metadata: dict[str, dict[str, object]] = {
        TREND_FILTER_ONLY_RULE.key: {"variant": "trend_filter_only", "score_lookback": np.nan, "parent_signal_key": None},
        make_prior_price_high_rule().key: {"variant": "prior_price_high", "score_lookback": np.nan, "parent_signal_key": None},
        LEGACY_SIGNAL_SURGE_V0_RULE.key: {"variant": "legacy_baseline", "score_lookback": np.nan, "parent_signal_key": None},
    }
    for rule in score_rules:
        metadata[rule.key] = {
            "variant": "score_breakout",
            "score_lookback": int(rule.params["score_lookback"]),
            "parent_signal_key": None,
        }
        for seed in random_seeds:
            key = f"{rule.key}__random_{int(seed)}"
            signal_events[key] = frequency_matched_random_events(
                ordered,
                signal_events[rule.key],
                eligible_control,
                seed=int(seed),
            )
            metadata[key] = {
                "variant": "frequency_matched_random",
                "score_lookback": int(rule.params["score_lookback"]),
                "parent_signal_key": rule.key,
                "shift_edge_loss_raw_count": np.nan,
                "shift_edge_loss_executable_trigger_count": np.nan,
            }
        shifted_key = f"{rule.key}__shift_{PLACEBO_SHIFT_BARS}"
        shifted_events = within_symbol_shifted_events(
            ordered, signal_events[rule.key], shift_bars=PLACEBO_SHIFT_BARS
        ) & eligible_control
        signal_events[shifted_key] = shifted_events
        source_counts = signal_event_counts(
            ordered, signal_events[rule.key], selected.entry
        )
        shifted_counts = signal_event_counts(ordered, shifted_events, selected.entry)
        metadata[shifted_key] = {
            "variant": "shifted_placebo",
            "score_lookback": int(rule.params["score_lookback"]),
            "parent_signal_key": rule.key,
            "shift_edge_loss_raw_count": max(
                source_counts["raw_boolean_signal_count"]
                - shifted_counts["raw_boolean_signal_count"],
                0,
            ),
            "shift_edge_loss_executable_trigger_count": (
                max(
                    source_counts["executable_trigger_count"]
                    - shifted_counts["executable_trigger_count"],
                    0,
                )
            ),
            "executable_trigger_count_matches_parent": (
                source_counts["executable_trigger_count"]
                == shifted_counts["executable_trigger_count"]
            ),
        }
    for key, values in signal_events.items():
        if key not in observations:
            observations[key] = values

    diagnostics = build_signal_diagnostics(ordered, signal_events)
    diagnostic_summary = summarize_signal_diagnostics(diagnostics)
    lifecycle_frames = []
    metric_rows = []
    for key, events in signal_events.items():
        lifecycle = simulate_signal_lifecycles(
            ordered,
            events,
            strategy_key=key,
            components=selected,
            round_trip_cost=round_trip_cost,
        )
        if not lifecycle.empty:
            lifecycle_frames.append(lifecycle)
        curve = selected.portfolio_construction.construct(
            lifecycle,
            ordered,
            round_trip_cost,
        )
        counts = signal_event_counts(ordered, events, selected.entry)
        completed_lifecycles = (
            int(lifecycle["exit_date"].notna().sum())
            if not lifecycle.empty and "exit_date" in lifecycle
            else 0
        )
        metric_rows.append(
            {
                **summarize_relative_portfolio(key, curve, ordered),
                **metadata[key],
                **counts,
                "completed_lifecycle_count": completed_lifecycles,
            }
        )
    lifecycles = pd.concat(lifecycle_frames, ignore_index=True) if lifecycle_frames else pd.DataFrame()
    portfolio_metrics = pd.DataFrame(metric_rows)
    classification = classify_score_breakout(
        portfolio_metrics,
        diagnostic_summary,
        robustness_evidence=robustness_evidence,
    )
    methodology = phase_a_methodology(
        selected,
        score_lookbacks=score_lookbacks,
        random_seeds=random_seeds,
        round_trip_cost=round_trip_cost,
    )
    # A JSON round-trip asserts that the research contract is persistable without
    # writing generated artifacts in Phase A.
    json.dumps({"classification": classification, "methodology": methodology}, allow_nan=False)
    return PhaseAComparisonResult(
        signal_observations=observations,
        signal_diagnostics=diagnostics,
        signal_diagnostic_summary=diagnostic_summary,
        lifecycles=lifecycles,
        trade_diagnostics=summarize_trade_diagnostics(lifecycles),
        portfolio_metrics=portfolio_metrics,
        classification=classification,
        methodology=methodology,
    )
