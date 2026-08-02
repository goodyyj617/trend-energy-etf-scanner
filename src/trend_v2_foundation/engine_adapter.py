"""Bounded adapter around the existing score-independent Phase A execution path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from src.backtest import build_historical_features
from src.portfolio import build_price_panel, simulate_canonical_portfolio
from src.trend_v2 import (
    default_phase_a_components,
    evaluate_signal_observations,
    make_prior_price_high_rule,
    simulate_signal_lifecycles,
)
from src.trend_v2_phase_a2 import (
    FrozenInputs,
    SNAPSHOT_SCHEMA_VERSION,
    _canonical_prices,
    load_frozen_inputs,
    reconstruct_snapshot_sha256,
)

from .artifact_schemas import validate_daily_portfolio_curve
from .construction import CONTROLLED_ENGINE_VERSION, Foundation5Error
from .contracts import ArtifactKind, StrategyRunSpec


@dataclass(frozen=True)
class AdapterArtifact:
    artifact_key: str
    kind: ArtifactKind
    payload: Mapping[str, Any]
    row_count: int


@dataclass(frozen=True)
class AdapterResult:
    artifacts: tuple[AdapterArtifact, ...]
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class EconomicExecutionAdapter(Protocol):
    engine_version: str

    def execute(self, specification: StrategyRunSpec) -> AdapterResult: ...


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notna(frame), None)
    rows: list[dict[str, Any]] = []
    for source in clean.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in source.items():
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, (np.bool_,)):
                value = bool(value)
            elif isinstance(value, pd.Timestamp):
                value = value.isoformat()
            row[str(key)] = value
        rows.append(row)
    return rows


def _daily_curve_payload(curve: pd.DataFrame) -> dict[str, Any]:
    economic = curve.loc[curve["observation_type"] == "trading_session"].copy()
    rows = []
    for row in economic.itertuples(index=False):
        value = float(row.portfolio_equity)
        rows.append(
            {
                "economic_date": str(row.date),
                "portfolio_value": value,
                "daily_return": float(row.daily_portfolio_return),
                "gross_exposure": float(row.gross_exposure),
                "net_exposure": float(row.gross_exposure),
                "cash_weight": max(0.0, 1.0 - float(row.gross_exposure)),
                "daily_turnover": float(row.turnover),
                "transaction_cost": float(row.transaction_cost_paid) / value if value > 0 else 0.0,
                "position_count": int(row.active_position_count),
            }
        )
    if not rows:
        raise Foundation5Error("internal_execution_failure", "Economic adapter produced no trading-session curve rows.", recoverable=False)
    payload = {
        "schema_version": "daily_portfolio_curve_v1",
        "economic_date_range": {"start": rows[0]["economic_date"], "end": rows[-1]["economic_date"]},
        "rows": rows,
    }
    validate_daily_portfolio_curve(payload)
    return payload


def _benchmark_curve_payload(prices: pd.DataFrame) -> dict[str, Any]:
    spy = prices.loc[prices["symbol"].astype(str) == "SPY", ["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy = spy.dropna().sort_values("date").drop_duplicates("date", keep="last")
    if len(spy) < 2:
        raise Foundation5Error("benchmark_unavailable", "SPY has insufficient exact-date observations.")
    normalized = spy["close"] / float(spy["close"].iloc[0]) * 1000.0
    daily = normalized.pct_change().fillna(0.0)
    rows = [
        {
            "economic_date": timestamp.date().isoformat(),
            "portfolio_value": float(value),
            "daily_return": float(return_value),
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
            "cash_weight": 0.0,
            "daily_turnover": 0.0,
            "transaction_cost": 0.0,
            "position_count": 1,
        }
        for timestamp, value, return_value in zip(spy["date"], normalized, daily)
    ]
    payload = {
        "schema_version": "daily_portfolio_curve_v1",
        "economic_date_range": {"start": rows[0]["economic_date"], "end": rows[-1]["economic_date"]},
        "rows": rows,
    }
    validate_daily_portfolio_curve(payload)
    return payload


class PhaseAControlledExecutionAdapter:
    """Execute only the accepted price-only Phase A baseline from frozen files."""

    engine_version = CONTROLLED_ENGINE_VERSION

    def __init__(self, snapshot_directory: str | Path) -> None:
        self.snapshot_directory = Path(snapshot_directory)
        self._lock = Lock()
        self._inputs: Any | None = None
        self._features: pd.DataFrame | None = None

    def _load(self) -> tuple[Any, pd.DataFrame]:
        with self._lock:
            if self._inputs is None:
                try:
                    self._inputs = load_frozen_inputs(self.snapshot_directory)
                except ValueError as error:
                    if str(error) != "snapshot member hash mismatch: universe_snapshot.csv":
                        raise
                    self._inputs = self._load_windows_text_compatible()
                self._features = build_historical_features(
                    self._inputs.prices,
                    self._inputs.universe,
                    dict(self._inputs.manifest["universe_parameters"]),
                )
        assert self._features is not None
        return self._inputs, self._features

    def _load_windows_text_compatible(self) -> FrozenInputs:
        """Accept only Git CRLF checkout equivalence for the frozen universe CSV."""

        manifest = json.loads(
            (self.snapshot_directory / "input_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported snapshot schema")
        members = dict(manifest.get("snapshot_members", {}))
        root = self.snapshot_directory.resolve()
        for relative_path, expected in members.items():
            path = (root / relative_path).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"snapshot member is missing or unsafe: {relative_path}")
            payload = path.read_bytes()
            observed = hashlib.sha256(payload).hexdigest()
            if observed == expected:
                continue
            if relative_path != "universe_snapshot.csv":
                raise ValueError(f"snapshot member hash mismatch: {relative_path}")
            canonical_lf = payload.replace(b"\r\n", b"\n")
            if hashlib.sha256(canonical_lf).hexdigest() != expected:
                raise ValueError(f"snapshot member hash mismatch: {relative_path}")
        if reconstruct_snapshot_sha256(members) != manifest.get("complete_snapshot_sha256"):
            raise ValueError("complete snapshot hash mismatch")
        shard_paths = [str(item["path"]) for item in manifest.get("price_shards", [])]
        if shard_paths != sorted(shard_paths) or set(shard_paths).difference(members):
            raise ValueError("price shard manifest is invalid")
        universe = pd.read_csv(root / manifest["universe_snapshot"]["path"])
        frames = [pd.read_csv(root / item["path"], compression="gzip") for item in manifest["price_shards"]]
        prices = _canonical_prices(pd.concat(frames, ignore_index=True))
        if len(prices) != int(manifest["row_count"]):
            raise ValueError("frozen price row count does not match manifest")
        actual_counts = {
            str(key): int(value)
            for key, value in prices.groupby("symbol", sort=True).size().items()
        }
        if actual_counts != manifest["row_counts_by_symbol"]:
            raise ValueError("frozen per-symbol row counts do not match manifest")
        return FrozenInputs(manifest=manifest, universe=universe, prices=prices)

    @staticmethod
    def _require(specification: StrategyRunSpec) -> None:
        expected = {
            "universe_specification": "phase_a2_historical_eligible_v1",
            "benchmark": "spy_adjusted_close_v1",
            "trend_filter": "price_above_rising_ma200_v0",
            "signal": "prior_price_high_l20_v1",
            "entry_rule": "first_event_next_open_v1",
            "initial_stop": "signal_day_low20_v1",
            "trailing_exit": "ratcheting_low20_v1",
            "position_sizing": "canonical_equal_weight_active_v1",
            "portfolio_constraints": "long_only_cash_constrained_v1",
            "transaction_costs": "round_trip_bps_v1",
            "slippage": "round_trip_slippage_bps_v1",
        }
        if specification.engine_version != CONTROLLED_ENGINE_VERSION:
            raise Foundation5Error("engine_unsupported", "StrategyRunSpec names an unsupported engine version.")
        for field, option_id in expected.items():
            if getattr(specification, field).get("option_id") != option_id:
                raise Foundation5Error("engine_unsupported", f"The adapter does not support {field}.")
        if specification.signal["parameters"].get("lookback") != 20:
            raise Foundation5Error("engine_unsupported", "Only the established prior-price-high L20 baseline is supported.")

    def execute(self, specification: StrategyRunSpec) -> AdapterResult:
        self._require(specification)
        frozen, full_features = self._load()
        if specification.data_snapshot_hash != frozen.manifest["complete_snapshot_sha256"]:
            raise Foundation5Error("snapshot_unavailable", "StrategyRunSpec snapshot hash does not match the frozen local snapshot.")
        start = specification.economic_date_range["start"]
        end = specification.economic_date_range["end"]
        full_dates = pd.to_datetime(full_features["date"], errors="coerce")
        selected_features = full_features.loc[(full_dates >= start) & (full_dates <= end)].copy()
        price_dates = pd.to_datetime(frozen.prices["date"], errors="coerce")
        selected_prices = frozen.prices.loc[(price_dates >= start) & (price_dates <= end)].copy()
        if selected_features.empty or selected_prices.empty:
            raise Foundation5Error("snapshot_unavailable", "No frozen observations exist in the requested date range.")
        components = default_phase_a_components()
        signal_rule = make_prior_price_high_rule(20)
        observations = evaluate_signal_observations(full_features, [signal_rule], components)
        observation_dates = pd.to_datetime(full_features["date"], errors="coerce")
        events = observations.loc[(observation_dates >= start) & (observation_dates <= end), signal_rule.key]
        events.index = selected_features.index
        cost_bps = float(specification.transaction_costs["parameters"]["bps"])
        slippage_bps = float(specification.slippage["parameters"]["bps"])
        round_trip_cost = (cost_bps + slippage_bps) / 10_000.0
        lifecycles = simulate_signal_lifecycles(
            selected_features,
            events,
            strategy_key=specification.strategy_run_id,
            components=components,
            round_trip_cost=round_trip_cost,
        )
        curve = simulate_canonical_portfolio(
            lifecycles,
            build_price_panel(selected_prices),
            round_trip_cost=round_trip_cost,
        )
        daily = _daily_curve_payload(curve)
        benchmark = _benchmark_curve_payload(selected_prices)
        trade_rows = _json_records(lifecycles)
        event_frame = selected_features.loc[events.astype(bool), ["date", "symbol"]].copy()
        if not event_frame.empty:
            event_frame["date"] = pd.to_datetime(event_frame["date"]).dt.date.astype(str)
            event_frame["event"] = "prior_price_high_l20"
        event_rows = _json_records(event_frame)
        return AdapterResult(
            artifacts=(
                AdapterArtifact("daily_portfolio_curve", ArtifactKind.DAILY_PORTFOLIO_CURVE, daily, len(daily["rows"])),
                AdapterArtifact("benchmark_daily_portfolio_curve", ArtifactKind.DAILY_PORTFOLIO_CURVE, benchmark, len(benchmark["rows"])),
                AdapterArtifact("trade_lifecycles", ArtifactKind.TRADE_LIFECYCLES, {"schema_version": "trade_lifecycles_v1", "rows": trade_rows}, len(trade_rows)),
                AdapterArtifact("signal_execution_events", ArtifactKind.SIGNAL_EXECUTION_EVENTS, {"schema_version": "signal_execution_events_v1", "rows": event_rows}, len(event_rows)),
            ),
            warnings=("controlled_phase_a_baseline_only",),
            limitations=(
                "No fixed holding-period exit or fixed profit target is used.",
                "Walk-forward and robustness simulations are estimated but not generated by this adapter.",
                "Historical universe retains the documented survivorship and current-product limitations.",
            ),
        )
