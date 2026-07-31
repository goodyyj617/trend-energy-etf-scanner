from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .backtest import build_historical_features
from .portfolio import PricePanel, build_price_panel, simulate_canonical_portfolio
from .postprocess_groups import classify_group
from .prices import download_ohlcv, filter_completed_daily_bars
from .trend_v2 import (
    FREQUENCY_MATCHED_RANDOM_SEEDS,
    PLACEBO_SHIFT_BARS,
    ROUND_TRIP_COST,
    SCORE_LOOKBACK_GRID,
    PhaseAComparisonResult,
    classify_score_breakout,
    run_phase_a_signal_comparison,
)
from .universe import build_base_universe


SNAPSHOT_SCHEMA_VERSION = "trend_v2_phase_a2_snapshot_v1"
ANALYSIS_SCHEMA_VERSION = "trend_v2_phase_a2_analysis_v1"
PRICE_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
BOOTSTRAP_SEED = 260731
BOOTSTRAP_PATHS = 5000
BOOTSTRAP_MEAN_BLOCK_LENGTH = 20
ANNUALIZATION_SESSIONS = 252
MINIMUM_TRAINING_COMPLETE_YEARS = 3
HOLM_ALPHA = 0.05
UNCLASSIFIED_GROUPS = {"", "unknown", "other", "nan", "none"}


@dataclass(frozen=True)
class FrozenInputs:
    manifest: Mapping[str, Any]
    universe: pd.DataFrame
    prices: pd.DataFrame


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _load_universe_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict((yaml.safe_load(handle) or {})["universe"])


def _canonical_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = set(PRICE_COLUMNS).difference(prices.columns)
    if missing:
        raise ValueError(f"price snapshot is missing columns: {sorted(missing)}")
    out = prices[list(PRICE_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    out["symbol"] = out["symbol"].astype("string").str.upper().str.strip()
    for column in PRICE_COLUMNS[2:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "symbol", "open", "high", "low", "close"])
    out = out.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("price snapshot contains duplicate date/symbol rows")
    return out


def deterministic_gzip_csv_bytes(frame: pd.DataFrame) -> bytes:
    csv_payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.10g",
        na_rep="",
    ).encode("utf-8")
    output = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as handle:
        handle.write(csv_payload)
    return output.getvalue()


def write_yearly_price_shards(prices: pd.DataFrame, price_dir: Path) -> list[dict[str, Any]]:
    canonical = _canonical_prices(prices)
    price_dir.mkdir(parents=True, exist_ok=True)
    years = pd.to_datetime(canonical["date"], errors="raise").dt.year
    records: list[dict[str, Any]] = []
    for year in sorted(years.unique()):
        shard = canonical.loc[years.eq(year), list(PRICE_COLUMNS)].reset_index(drop=True)
        relative_path = f"prices/{int(year)}.csv.gz"
        payload = deterministic_gzip_csv_bytes(shard)
        path = price_dir / f"{int(year)}.csv.gz"
        path.write_bytes(payload)
        records.append(
            {
                "year": int(year),
                "path": relative_path,
                "row_count": int(len(shard)),
                "sha256": sha256_bytes(payload),
                "byte_count": int(len(payload)),
            }
        )
    return records


def reconstruct_snapshot_sha256(members: Mapping[str, str]) -> str:
    normalized = {str(path).replace("\\", "/"): str(digest) for path, digest in members.items()}
    return sha256_bytes(_canonical_json_bytes(normalized))


def _universe_csv_bytes(universe: pd.DataFrame) -> bytes:
    ordered = universe.sort_values(["base_universe_eligible", "aum_rank", "symbol"],
                                   ascending=[False, True, True], kind="mergesort")
    return ordered.to_csv(index=False, lineterminator="\n", float_format="%.10g", na_rep="").encode("utf-8")


def collect_and_freeze_snapshot(
    root: Path,
    snapshot_dir: Path,
    *,
    now: datetime | None = None,
    source_code_commit: str | None = None,
) -> dict[str, Any]:
    """Download once, validate, and freeze all Phase A2 inputs."""
    root = root.resolve()
    snapshot_dir = snapshot_dir.resolve()
    manifest_path = snapshot_dir / "input_manifest.json"
    if manifest_path.exists() or (snapshot_dir / "prices").exists():
        raise FileExistsError(f"snapshot already exists: {snapshot_dir}")
    collection_time = now or datetime.now(timezone.utc)
    if collection_time.tzinfo is None or collection_time.utcoffset() is None:
        raise ValueError("collection time must be timezone-aware")
    collector_commit = source_code_commit or _git_head(root)

    config_dir = root / "config"
    paths = {
        "aum": config_dir / "aum.csv",
        "universe_config": config_dir / "universe.yml",
        "exclusions": config_dir / "exclusions.yml",
        "overrides": config_dir / "manual_overrides.csv",
    }
    universe = build_base_universe(
        aum_csv=paths["aum"],
        exclusions_yml=paths["exclusions"],
        overrides_csv=paths["overrides"],
        universe_yml=paths["universe_config"],
    )
    if universe.empty:
        raise RuntimeError("current configured universe is empty")
    derived_groups = [classify_group(row.symbol, row.name) for row in universe.itertuples()]
    current_groups = universe.get("asset_group", pd.Series("unknown", index=universe.index))
    missing_group = current_groups.fillna("unknown").astype(str).str.lower().isin(UNCLASSIFIED_GROUPS)
    universe = universe.copy()
    universe.loc[missing_group, "asset_group"] = np.asarray(derived_groups, dtype=object)[missing_group]
    universe = universe.sort_values(
        ["base_universe_eligible", "aum_rank", "symbol"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    requested_symbols = universe.loc[
        universe["base_universe_eligible"].fillna(False), "symbol"
    ].astype(str).tolist()
    requested_symbols = list(dict.fromkeys([*requested_symbols, "SPY"]))
    prices = download_ohlcv(requested_symbols, period="10y", interval="1d")
    prices, completed_bar_provenance = filter_completed_daily_bars(
        prices, now=collection_time
    )
    prices = _canonical_prices(prices)
    if prices.empty or "SPY" not in set(prices["symbol"].astype(str)):
        raise RuntimeError("downloaded snapshot is empty or lacks the SPY benchmark")

    retained_symbols = sorted(prices["symbol"].astype(str).unique().tolist())
    failed_symbols = sorted(set(requested_symbols).difference(retained_symbols))
    dates = pd.to_datetime(prices["date"], errors="raise")
    if (dates.max() - dates.min()).days < 365 * 8:
        raise RuntimeError("accepted Phase A2 snapshot must retain at least eight years")

    snapshot_dir.mkdir(parents=True, exist_ok=False)
    universe_path = snapshot_dir / "universe_snapshot.csv"
    universe_payload = _universe_csv_bytes(universe)
    universe_path.write_bytes(universe_payload)
    shards = write_yearly_price_shards(prices, snapshot_dir / "prices")
    members = {"universe_snapshot.csv": sha256_bytes(universe_payload)}
    members.update({row["path"]: row["sha256"] for row in shards})
    row_counts_by_symbol = {
        str(key): int(value)
        for key, value in prices.groupby("symbol", sort=True).size().items()
    }
    row_counts_by_symbol_and_year: dict[str, dict[str, int]] = {}
    count_frame = prices.assign(year=dates.dt.year).groupby(
        ["symbol", "year"], sort=True
    ).size()
    for (symbol, year), count in count_frame.items():
        row_counts_by_symbol_and_year.setdefault(str(symbol), {})[str(int(year))] = int(count)

    import yfinance as yf

    cfg = _load_universe_config(paths["universe_config"])
    manifest: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_code_commit": collector_commit,
        "utc_collection_time": collection_time.astimezone(timezone.utc).isoformat(),
        "retained_minimum_economic_date": dates.min().date().isoformat(),
        "retained_maximum_economic_date": dates.max().date().isoformat(),
        "final_as_of": dates.max().date().isoformat(),
        "versions": {
            "yfinance": str(yf.__version__),
            "pandas": str(pd.__version__),
            "numpy": str(np.__version__),
        },
        "adjusted_price_convention": "yfinance auto_adjust=True; adjusted OHLC and raw volume",
        "requested_period": "10y",
        "requested_interval": "1d",
        "requested_symbols": requested_symbols,
        "retained_symbols": retained_symbols,
        "failed_or_empty_symbols": failed_symbols,
        "row_count": int(len(prices)),
        "row_counts_by_symbol": row_counts_by_symbol,
        "row_counts_by_symbol_and_year": row_counts_by_symbol_and_year,
        "input_hashes": {key: sha256_file(path) for key, path in paths.items()},
        "current_aum_file_sha256": sha256_file(paths["aum"]),
        "universe_snapshot": {
            "path": "universe_snapshot.csv",
            "row_count": int(len(universe)),
            "sha256": members["universe_snapshot.csv"],
        },
        "price_shards": shards,
        "snapshot_members": members,
        "complete_snapshot_sha256": reconstruct_snapshot_sha256(members),
        "completed_bar_filtering_provenance": completed_bar_provenance,
        "universe_parameters": cfg,
        "known_limitations": [
            "Universe eligibility uses present-day AUM and present-day listed products.",
            "The snapshot is provisional in-sample research evidence, not genuine OOS evidence.",
        ],
    }
    _write_json(manifest_path, manifest)
    validate_snapshot_manifest(snapshot_dir)
    return manifest


def validate_snapshot_manifest(snapshot_dir: Path) -> dict[str, Any]:
    snapshot_dir = snapshot_dir.resolve()
    manifest_path = snapshot_dir / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source_code_commit",
        "utc_collection_time",
        "retained_minimum_economic_date",
        "retained_maximum_economic_date",
        "final_as_of",
        "versions",
        "adjusted_price_convention",
        "requested_symbols",
        "retained_symbols",
        "failed_or_empty_symbols",
        "row_counts_by_symbol_and_year",
        "input_hashes",
        "universe_snapshot",
        "price_shards",
        "snapshot_members",
        "complete_snapshot_sha256",
        "completed_bar_filtering_provenance",
        "universe_parameters",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"snapshot manifest missing fields: {sorted(missing)}")
    if manifest["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported snapshot schema")
    members = dict(manifest["snapshot_members"])
    for relative_path, expected in members.items():
        path = snapshot_dir / relative_path
        if not path.is_file():
            raise ValueError(f"snapshot member is missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"snapshot member hash mismatch: {relative_path}")
    reconstructed = reconstruct_snapshot_sha256(members)
    if reconstructed != manifest["complete_snapshot_sha256"]:
        raise ValueError("complete snapshot hash mismatch")
    shard_paths = [str(row["path"]) for row in manifest["price_shards"]]
    if shard_paths != sorted(shard_paths):
        raise ValueError("price shard manifest is not deterministically ordered")
    if set(shard_paths).difference(members):
        raise ValueError("price shard is absent from snapshot member hashes")
    return manifest


def load_frozen_inputs(snapshot_dir: Path) -> FrozenInputs:
    manifest = validate_snapshot_manifest(snapshot_dir)
    universe = pd.read_csv(snapshot_dir / manifest["universe_snapshot"]["path"])
    frames = [pd.read_csv(snapshot_dir / row["path"], compression="gzip")
              for row in manifest["price_shards"]]
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


def complete_calendar_years(dates: pd.Series) -> list[int]:
    parsed = pd.to_datetime(dates, errors="coerce").dropna()
    if parsed.empty or parsed.dt.year.nunique() < 3:
        return []
    minimum_year = int(parsed.min().year)
    maximum_year = int(parsed.max().year)
    return list(range(minimum_year + 1, maximum_year))


def _strategy_returns(curves: pd.DataFrame, strategy_key: str) -> pd.DataFrame:
    selected = curves.loc[
        curves["strategy_key"].astype(str).eq(strategy_key)
        & curves["observation_type"].astype(str).eq("trading_session"),
        ["date", "daily_portfolio_return"],
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.normalize()
    selected["return"] = pd.to_numeric(
        selected["daily_portfolio_return"], errors="coerce"
    )
    return selected.dropna().drop_duplicates("date", keep="last")[["date", "return"]]


def paired_daily_returns(
    curves: pd.DataFrame, left_key: str, right_key: str
) -> pd.DataFrame:
    left = _strategy_returns(curves, left_key).rename(columns={"return": "left_return"})
    right = _strategy_returns(curves, right_key).rename(columns={"return": "right_return"})
    common = left.merge(right, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if len(common) < 2:
        raise ValueError(f"insufficient exact common returns for {left_key} and {right_key}")
    common["paired_difference"] = common["left_return"] - common["right_return"]
    return common


def annualized_paired_effect(common: pd.DataFrame) -> float:
    return float(common["paired_difference"].mean() * ANNUALIZATION_SESSIONS)


def _annualized_mean_return(frame: pd.DataFrame) -> float:
    return float(pd.to_numeric(frame["return"], errors="coerce").mean() * ANNUALIZATION_SESSIONS)


def select_stronger_deterministic_comparator(
    curves: pd.DataFrame,
    *,
    dates: pd.Series | None = None,
) -> str:
    candidates = ("trend_filter_only", "prior_price_high_l20")
    allowed = None if dates is None else set(pd.to_datetime(dates).dt.normalize())
    scored: list[tuple[float, str]] = []
    for key in candidates:
        returns = _strategy_returns(curves, key)
        if allowed is not None:
            returns = returns.loc[returns["date"].isin(allowed)]
        scored.append((_annualized_mean_return(returns), key))
    return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]


def build_walk_forward_evidence(
    curves: pd.DataFrame,
    score_keys: Sequence[str],
    complete_years: Sequence[int],
) -> dict[str, dict[str, Any]]:
    folds: dict[str, list[dict[str, Any]]] = {key: [] for key in score_keys}
    years = sorted(int(year) for year in complete_years)
    for position in range(MINIMUM_TRAINING_COMPLETE_YEARS, len(years)):
        test_year = years[position]
        train_years = years[:position]
        reference = _strategy_returns(curves, "trend_filter_only")
        train_dates = reference.loc[reference["date"].dt.year.isin(train_years), "date"]
        comparator = select_stronger_deterministic_comparator(curves, dates=train_dates)
        for score_key in score_keys:
            common = paired_daily_returns(curves, score_key, comparator)
            test = common.loc[common["date"].dt.year.eq(test_year)]
            if test.empty:
                raise ValueError(f"walk-forward test year {test_year} has no common returns")
            effect = annualized_paired_effect(test)
            folds[score_key].append(
                {
                    "training_start_year": train_years[0],
                    "training_end_year": train_years[-1],
                    "training_end_date": f"{train_years[-1]}-12-31",
                    "test_year": test_year,
                    "test_start_date": test["date"].min().date().isoformat(),
                    "test_end_date": test["date"].max().date().isoformat(),
                    "comparator_selected_using_training_only": comparator,
                    "annualized_mean_daily_return_effect": effect,
                    "improved": bool(effect > 0),
                }
            )
    return {
        key: {
            "fold_count": len(rows),
            "improving_fold_count": int(sum(bool(row["improved"]) for row in rows)),
            "improvement_ratio": float(np.mean([row["improved"] for row in rows])) if rows else 0.0,
            "folds": rows,
        }
        for key, rows in folds.items()
    }


def build_loyo_evidence(
    curves: pd.DataFrame,
    score_keys: Sequence[str],
    complete_years: Sequence[int],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    reference_dates = _strategy_returns(curves, "trend_filter_only")["date"]
    for score_key in score_keys:
        full_comparator = select_stronger_deterministic_comparator(curves)
        full_effect = annualized_paired_effect(
            paired_daily_returns(curves, score_key, full_comparator)
        )
        expected_positive = full_effect >= 0
        cases: list[dict[str, Any]] = []
        for omitted_year in sorted(int(year) for year in complete_years):
            retained_dates = reference_dates.loc[~reference_dates.dt.year.eq(omitted_year)]
            comparator = select_stronger_deterministic_comparator(
                curves, dates=retained_dates
            )
            common = paired_daily_returns(curves, score_key, comparator)
            common = common.loc[~common["date"].dt.year.eq(omitted_year)]
            effect = annualized_paired_effect(common)
            stable = bool((effect >= 0) == expected_positive)
            cases.append(
                {
                    "omitted_year": omitted_year,
                    "comparator_recomputed_without_year": comparator,
                    "annualized_mean_daily_return_effect": effect,
                    "directionally_stable": stable,
                }
            )
        output[score_key] = {
            "full_sample_comparator": full_comparator,
            "full_sample_effect": full_effect,
            "case_count": len(cases),
            "passing_case_count": int(sum(row["directionally_stable"] for row in cases)),
            "stability_ratio": float(np.mean([row["directionally_stable"] for row in cases])) if cases else 0.0,
            "reversing_omitted_years": [
                int(row["omitted_year"]) for row in cases if not row["directionally_stable"]
            ],
            "cases": cases,
        }
    return output


def stationary_bootstrap_indices(
    observation_count: int,
    path_count: int,
    mean_block_length: int,
    *,
    seed: int,
) -> np.ndarray:
    if observation_count <= 0 or path_count <= 0 or mean_block_length <= 0:
        raise ValueError("stationary bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    indices = np.empty((path_count, observation_count), dtype=np.int32)
    restart_probability = 1.0 / float(mean_block_length)
    indices[:, 0] = rng.integers(0, observation_count, size=path_count, dtype=np.int32)
    for column in range(1, observation_count):
        restart = rng.random(path_count) < restart_probability
        continued = (indices[:, column - 1] + 1) % observation_count
        fresh = rng.integers(0, observation_count, size=path_count, dtype=np.int32)
        indices[:, column] = np.where(restart, fresh, continued)
    return indices


def bootstrap_index_hash(indices: np.ndarray) -> str:
    normalized = np.asarray(indices, dtype="<i4", order="C")
    shape = np.asarray(normalized.shape, dtype="<i8").tobytes()
    return hashlib.sha256(b"phase-a2-stationary-bootstrap-v1|" + shape + normalized.tobytes()).hexdigest()


def paired_stationary_bootstrap(
    common: pd.DataFrame,
    *,
    path_count: int = BOOTSTRAP_PATHS,
    mean_block_length: int = BOOTSTRAP_MEAN_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    differences = common["paired_difference"].to_numpy(dtype=float)
    indices = stationary_bootstrap_indices(
        len(differences), path_count, mean_block_length, seed=seed
    )
    effects = differences[indices].mean(axis=1) * ANNUALIZATION_SESSIONS
    lower, upper = np.quantile(effects, [0.025, 0.975])
    return {
        "paired_annualized_effect_estimate": float(differences.mean() * ANNUALIZATION_SESSIONS),
        "confidence_interval_95": [float(lower), float(upper)],
        "unadjusted_one_sided_p_value": float((1 + np.count_nonzero(effects <= 0)) / (path_count + 1)),
        "seed": int(seed),
        "path_count": int(path_count),
        "mean_block_length": int(mean_block_length),
        "block_rule": "stationary_bootstrap_geometric_blocks",
        "common_date_count": int(len(common)),
        "common_start_date": common["date"].min().date().isoformat(),
        "common_end_date": common["date"].max().date().isoformat(),
        "index_matrix_sha256": bootstrap_index_hash(indices),
    }


def holm_adjust(raw_p_values: Mapping[str, float], *, alpha: float = HOLM_ALPHA) -> dict[str, dict[str, Any]]:
    ordered = sorted(
        ((str(key), float(value)) for key, value in raw_p_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[str, dict[str, Any]] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, (key, raw) in enumerate(ordered, start=1):
        running = max(running, min(1.0, raw * (family_size - rank + 1)))
        adjusted[key] = {
            "raw_p_value": raw,
            "adjusted_p_value": running,
            "holm_rank": rank,
            "pass": bool(running <= alpha),
        }
    return adjusted


def _restricted_curve(
    comparison: PhaseAComparisonResult,
    panel: PricePanel,
    strategy_key: str,
    asset_group: str,
) -> pd.DataFrame:
    lifecycles = comparison.lifecycles
    if lifecycles.empty:
        selected = lifecycles
    else:
        selected = lifecycles.loc[
            lifecycles["strategy_key"].astype(str).eq(strategy_key)
            & lifecycles["asset_group"].fillna("unknown").astype(str).eq(asset_group)
        ]
    curve = simulate_canonical_portfolio(
        selected, panel, round_trip_cost=ROUND_TRIP_COST
    )
    curve = curve.copy()
    curve["strategy_key"] = strategy_key
    return curve


def build_asset_group_concentration(
    comparison: PhaseAComparisonResult,
    features: pd.DataFrame,
    score_key: str,
    comparator_key: str,
) -> dict[str, Any]:
    lifecycle = comparison.lifecycles
    relevant = lifecycle.loc[
        lifecycle["strategy_key"].astype(str).isin([score_key, comparator_key])
    ] if not lifecycle.empty else lifecycle
    groups = sorted(relevant["asset_group"].fillna("unknown").astype(str).unique().tolist())
    panel = build_price_panel(features)
    effects: list[dict[str, Any]] = []
    for group in groups:
        score_curve = _restricted_curve(comparison, panel, score_key, group)
        comparator_curve = _restricted_curve(comparison, panel, comparator_key, group)
        curves = pd.concat([score_curve, comparator_curve], ignore_index=True)
        common = paired_daily_returns(curves, score_key, comparator_key)
        effects.append(
            {
                "asset_group": group,
                "annualized_mean_daily_return_effect": annualized_paired_effect(common),
            }
        )
    score_lifecycles = lifecycle.loc[
        lifecycle["strategy_key"].astype(str).eq(score_key)
    ] if not lifecycle.empty else lifecycle
    group_values = score_lifecycles.get("asset_group", pd.Series(dtype=str))
    return summarize_asset_group_effects(effects, group_values, comparator_key)


def summarize_asset_group_effects(
    effects: Sequence[Mapping[str, Any]],
    group_values: pd.Series,
    comparator_key: str,
) -> dict[str, Any]:
    normalized_effects = [dict(row) for row in effects]
    positive = [row for row in normalized_effects if row["annualized_mean_daily_return_effect"] > 0]
    positive_total = float(sum(row["annualized_mean_daily_return_effect"] for row in positive))
    dominant = max(
        positive,
        key=lambda row: (row["annualized_mean_daily_return_effect"], row["asset_group"]),
        default=None,
    )
    dominant_share = (
        float(dominant["annualized_mean_daily_return_effect"] / positive_total)
        if dominant is not None and positive_total > 0
        else 0.0
    )
    normalized_groups = group_values.fillna("unknown").astype(str).str.lower()
    missing_share = float(normalized_groups.isin(UNCLASSIFIED_GROUPS).mean()) if len(normalized_groups) else 0.0
    return {
        "measure": "non_additive_group_restricted_portfolio_annualized_mean_daily_return_effect",
        "comparator": comparator_key,
        "effect_by_asset_group": normalized_effects,
        "positive_effect_total": positive_total,
        "dominant_group": None if dominant is None else dominant["asset_group"],
        "dominant_effect_share": dominant_share,
        "missing_or_unclassified_group_share": missing_share,
        "additivity_claimed": False,
    }


def build_robustness_evidence(
    comparison: PhaseAComparisonResult,
    features: pd.DataFrame,
    *,
    bootstrap_paths: int = BOOTSTRAP_PATHS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    score_keys = [f"score_breakout_l{lookback}" for lookback in SCORE_LOOKBACK_GRID]
    dates = _strategy_returns(comparison.portfolio_daily_returns, "trend_filter_only")["date"]
    years = complete_calendar_years(dates)
    walk_forward = build_walk_forward_evidence(
        comparison.portfolio_daily_returns, score_keys, years
    )
    if any(value["fold_count"] < 3 for value in walk_forward.values()):
        raise ValueError("Phase A2 requires at least three complete walk-forward folds")
    loyo = build_loyo_evidence(comparison.portfolio_daily_returns, score_keys, years)

    bootstrap: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    comparators: dict[str, str] = {}
    for position, score_key in enumerate(score_keys):
        comparator = select_stronger_deterministic_comparator(
            comparison.portfolio_daily_returns
        )
        comparators[score_key] = comparator
        common = paired_daily_returns(
            comparison.portfolio_daily_returns, score_key, comparator
        )
        bootstrap[score_key] = paired_stationary_bootstrap(
            common,
            path_count=bootstrap_paths,
            seed=bootstrap_seed + position,
        )
        raw_p[score_key] = bootstrap[score_key]["unadjusted_one_sided_p_value"]
    correction = holm_adjust(raw_p)

    metrics = comparison.portfolio_metrics
    candidate_evidence: dict[str, dict[str, Any]] = {}
    for score_key in score_keys:
        score_row = metrics.loc[metrics["strategy_key"].astype(str).eq(score_key)].iloc[0]
        placebo = metrics.loc[metrics["parent_signal_key"].astype(str).eq(score_key)]
        executable_match = bool(
            len(placebo)
            and pd.to_numeric(placebo["executable_trigger_count"], errors="coerce")
            .eq(int(score_row["executable_trigger_count"]))
            .all()
        )
        counts_available = bool(
            len(placebo)
            and placebo[
                ["raw_boolean_signal_count", "executable_trigger_count", "completed_lifecycle_count"]
            ].notna().all().all()
        )
        concentration = build_asset_group_concentration(
            comparison, features, score_key, comparators[score_key]
        )
        candidate_evidence[score_key] = {
            "walk_forward_fold_count": walk_forward[score_key]["fold_count"],
            "walk_forward_improvement_ratio": walk_forward[score_key]["improvement_ratio"],
            "leave_one_year_out_result_count": loyo[score_key]["case_count"],
            "leave_one_year_out_stability_ratio": loyo[score_key]["stability_ratio"],
            "date_block_bootstrap_paired_effect_confidence_interval": bootstrap[score_key]["confidence_interval_95"],
            "multiple_testing_adjusted_p_value": correction[score_key]["adjusted_p_value"],
            "corrected_evidence_pass": correction[score_key]["pass"],
            "asset_group_concentration_diagnostics": concentration,
            "event_count_comparability": counts_available,
            "executable_trigger_count_comparability": executable_match,
        }
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "complete_calendar_years": years,
        "walk_forward": walk_forward,
        "leave_one_year_out": loyo,
        "date_block_bootstrap": bootstrap,
        "multiple_testing": {
            "method": "Holm",
            "adjustment_family": score_keys,
            "rejection_threshold": HOLM_ALPHA,
            "results": correction,
        },
        "stronger_deterministic_comparator": comparators,
        "candidate_evidence": candidate_evidence,
        "placebo": {
            "method": "circular_eligible_session_index",
            "requested_offset": PLACEBO_SHIFT_BARS,
            "per_symbol_executable_trigger_counts_preserved": bool(
                all(value["executable_trigger_count_comparability"] for value in candidate_evidence.values())
            ),
        },
    }


def select_representative_score(
    portfolio_metrics: pd.DataFrame, robustness: Mapping[str, Any]
) -> str:
    candidates = []
    for lookback in SCORE_LOOKBACK_GRID:
        key = f"score_breakout_l{lookback}"
        metric = portfolio_metrics.loc[
            portfolio_metrics["strategy_key"].astype(str).eq(key)
        ].iloc[0]
        correction = robustness["multiple_testing"]["results"][key]
        effect = robustness["date_block_bootstrap"][key]["paired_annualized_effect_estimate"]
        candidates.append(
            (
                float(correction["adjusted_p_value"]),
                -float(effect),
                -float(metric["strategy_cagr"]),
                int(lookback),
                key,
            )
        )
    return sorted(candidates)[0][-1]


def transition_to_phase_b1(
    classification: Mapping[str, Any],
    portfolio_metrics: pd.DataFrame,
    robustness: Mapping[str, Any],
) -> dict[str, Any]:
    class_name = str(classification["classification"])
    best_simple = select_stronger_deterministic_comparator_from_metrics(portfolio_metrics)
    if class_name == "Retain":
        score_key = str(classification["strategy_key"])
        signals = [score_key, best_simple]
        status = "retained"
    elif class_name == "Reject":
        signals = [best_simple]
        status = "removed_from_phase_b"
    elif class_name == "Inconclusive":
        score_key = select_representative_score(portfolio_metrics, robustness)
        signals = [score_key, best_simple]
        status = "exploratory_and_unresolved"
    else:
        raise ValueError(f"unsupported Phase A2 classification: {class_name}")
    return {
        "next_phase": "Phase B1 -- entry-family screening",
        "score_breakout_status": status,
        "signal_candidates": list(dict.fromkeys(signals)),
        "maximum_signal_candidates": 2,
    }


def select_stronger_deterministic_comparator_from_metrics(
    portfolio_metrics: pd.DataFrame,
) -> str:
    rows = portfolio_metrics.loc[
        portfolio_metrics["variant"].isin(["trend_filter_only", "prior_price_high"])
    ].copy()
    rows["strategy_cagr"] = pd.to_numeric(rows["strategy_cagr"], errors="coerce")
    rows = rows.sort_values(
        ["strategy_cagr", "strategy_key"], ascending=[False, True], kind="mergesort"
    )
    if rows.empty:
        raise ValueError("deterministic comparator metrics are missing")
    return str(rows.iloc[0]["strategy_key"])


def _metric_record(metrics: pd.DataFrame, key: str) -> dict[str, Any]:
    row = metrics.loc[metrics["strategy_key"].astype(str).eq(key)]
    if row.empty:
        raise ValueError(f"missing metric row for {key}")
    return _json_safe(row.iloc[0].to_dict())


def render_result_markdown(summary: Mapping[str, Any], robustness: Mapping[str, Any]) -> str:
    classification = summary["classification"]
    metrics = summary["selected_metrics"]
    score_key = summary["representative_score_key"]
    score = metrics[score_key]
    trend = metrics["trend_filter_only"]
    price = metrics["prior_price_high_l20"]
    evidence = robustness["candidate_evidence"][score_key]
    boot = robustness["date_block_bootstrap"][score_key]
    correction = robustness["multiple_testing"]["results"][score_key]
    concentration = evidence["asset_group_concentration_diagnostics"]

    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{100 * float(value):.2f}%"

    lines = [
        "# Trend Strategy v2 Phase A2 Result",
        "",
        f"## Empirical classification: {classification['classification']}",
        "",
        str(classification["reason"]),
        (
            "Inconclusive means score breakout remains exploratory and unresolved: the evidence "
            "does not justify either retaining it as a validated incremental rule or removing it "
            "as rejected. The transition rule therefore proceeds to Phase B1 with one representative "
            "score lookback and the best simple deterministic comparator."
        ),
        "This is provisional in-sample empirical research, not production approval and not genuine OOS evidence.",
        "",
        "## Frozen input",
        "",
        f"- Economic dates: {summary['snapshot_date_range']['minimum']} through {summary['snapshot_date_range']['maximum']}",
        f"- Snapshot SHA-256: `{summary['complete_snapshot_sha256']}`",
        f"- Collector source commit: `{summary['snapshot_source_code_commit']}`",
        f"- Empirical analysis source commit: `{summary['analysis_code_commit']}`",
        f"- Retained/requested symbols: {summary['data_coverage']['retained_symbol_count']}/{summary['data_coverage']['requested_symbol_count']}",
        f"- Failed or empty symbols: {', '.join(summary['data_coverage']['failed_or_empty_symbols']) or 'none'}",
        "",
        "## Portfolio comparison",
        "",
        "| Signal | CAGR | CAGR / SPY | MDD | abs(MDD) / SPY | CDaR95 | abs(CDaR95) / SPY | Calmar | Calmar / SPY | Recovery days | Turnover | Costs | Raw signals | Executable triggers | Completed trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    portfolio_keys = [
        "score_breakout_l10",
        "score_breakout_l20",
        "score_breakout_l40",
        "trend_filter_only",
        "prior_price_high_l20",
        "signal_surge_v0",
    ]
    for key in portfolio_keys:
        row = metrics[key]
        lines.append(
            f"| {key} | {pct(row['strategy_cagr'])} | {float(row['strategy_cagr_spy_ratio']):.3f} | "
            f"{pct(row['strategy_maximum_drawdown'])} | {float(row['maximum_drawdown_spy_ratio']):.3f} | "
            f"{pct(row['strategy_cdar95'])} | {float(row['cdar95_spy_ratio']):.3f} | "
            f"{float(row['strategy_calmar']):.3f} | {float(row['calmar_spy_ratio']):.3f} | "
            f"{int(row['strategy_recovery_duration_days'])} | "
            f"{float(row['annual_turnover']):.3f} | {float(row['total_transaction_cost']):.2f} | "
            f"{int(row['raw_boolean_signal_count'])} | {int(row['executable_trigger_count'])} | "
            f"{int(row['completed_lifecycle_count'])} |"
        )
    lines.extend([
        "",
        (
            f"The representative {score_key} produced a {pct(score['strategy_cagr'])} CAGR versus "
            f"{pct(trend['strategy_cagr'])} for trend-only and {pct(price['strategy_cagr'])} for "
            "the prior-price-high comparator. It improved return over both, but its drawdown, "
            "CDaR95, Calmar, and recovery were worse than prior-price-high, and its CAGR retained "
            f"only {float(score['strategy_cagr_spy_ratio']):.3f} of SPY rather than the provisional 0.80 objective."
        ),
        "The legacy signal is shown only as a historical diagnostic baseline.",
        "",
        "## Robustness evidence",
        "",
        "| Score | Stronger comparator | Walk-forward improved | LOYO stable | Reversing years | Annualized paired effect | 95% CI | Raw one-sided p | Holm-adjusted p | Pass | Dominant group/share | Missing/unclassified share |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---|---|---:|",
    ])
    for lookback in SCORE_LOOKBACK_GRID:
        key = f"score_breakout_l{lookback}"
        candidate = robustness["candidate_evidence"][key]
        candidate_boot = robustness["date_block_bootstrap"][key]
        candidate_loyo = robustness["leave_one_year_out"][key]
        candidate_correction = robustness["multiple_testing"]["results"][key]
        candidate_concentration = candidate["asset_group_concentration_diagnostics"]
        lines.append(
            f"| {key} | {robustness['stronger_deterministic_comparator'][key]} | "
            f"{robustness['walk_forward'][key]['improving_fold_count']}/{candidate['walk_forward_fold_count']} "
            f"({candidate['walk_forward_improvement_ratio']:.3f}) | "
            f"{candidate_loyo['passing_case_count']}/{candidate['leave_one_year_out_result_count']} "
            f"({candidate['leave_one_year_out_stability_ratio']:.3f}) | "
            f"{candidate_loyo['reversing_omitted_years']} | "
            f"{pct(candidate_boot['paired_annualized_effect_estimate'])} | "
            f"[{pct(candidate_boot['confidence_interval_95'][0])}, {pct(candidate_boot['confidence_interval_95'][1])}] | "
            f"{candidate_boot['unadjusted_one_sided_p_value']:.6f} | "
            f"{candidate_correction['adjusted_p_value']:.6f} | "
            f"{str(candidate_correction['pass']).lower()} | "
            f"{candidate_concentration['dominant_group']} / {candidate_concentration['dominant_effect_share']:.3f} | "
            f"{candidate_concentration['missing_or_unclassified_group_share']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Bootstrap uses 5,000 stationary-bootstrap paths with mean block length 20. The representative seed is {boot['seed']} and index hash is `{boot['index_matrix_sha256']}`; every candidate's seed and index hash are recorded in `robustness_evidence.json`.",
            f"Holm correction covers all {len(SCORE_LOOKBACK_GRID)} tested score lookbacks at alpha {HOLM_ALPHA:.2f}; no unadjusted result is used as final evidence.",
            f"Asset-group concentration is `{concentration['measure']}`. Group-restricted effects are a non-additive concentration diagnostic, not additive contributions.",
            f"The circular eligible-session-index placebo uses requested offset {PLACEBO_SHIFT_BARS}; per-symbol executable-trigger counts are preserved={str(evidence['executable_trigger_count_comparability']).lower()}.",
            "Every walk-forward fold and LOYO omission is reported separately in `robustness_evidence.json`.",
            "",
            "## Phase B1 transition",
            "",
            f"Exact signal set: {', '.join(summary['phase_b1_transition']['signal_candidates'])}.",
            f"Score breakout status: {summary['phase_b1_transition']['score_breakout_status']}.",
            "",
            "## Limitations",
            "",
            "The historical panel uses the current frozen AUM ranking and present-day product availability. It therefore has survivorship and current-universe bias. The result is suitable only for provisional Phase A2 comparison and does not establish production readiness or genuine out-of-sample performance.",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_phase_a2_reports(root: Path, snapshot_dir: Path) -> dict[str, Any]:
    """Regenerate summary prose from committed empirical CSV/JSON evidence."""
    summary_path = snapshot_dir / "phase_a2_summary.json"
    robustness_path = snapshot_dir / "robustness_evidence.json"
    metrics_path = snapshot_dir / "portfolio_metrics.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    manifest = json.loads((snapshot_dir / "input_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    summary.setdefault("snapshot_source_code_commit", manifest["source_code_commit"])
    summary.setdefault("analysis_code_commit", _git_head(root.resolve()))
    selected_keys = [
        "score_breakout_l10",
        "score_breakout_l20",
        "score_breakout_l40",
        "trend_filter_only",
        "prior_price_high_l20",
        "signal_surge_v0",
    ]
    summary["selected_metrics"] = {
        key: _metric_record(metrics, key) for key in selected_keys
    }
    _write_json(summary_path, summary)
    (snapshot_dir / "RESULT.md").write_text(
        render_result_markdown(summary, robustness), encoding="utf-8", newline="\n"
    )
    (root / "docs" / "research" / "trend_v2" / "CURRENT_STATE.md").write_text(
        render_current_state(summary), encoding="utf-8", newline="\n"
    )
    return summary


def render_current_state(summary: Mapping[str, Any]) -> str:
    classification = summary["classification"]
    transition = summary["phase_b1_transition"]
    signals = ", ".join(f"`{key}`" for key in transition["signal_candidates"])
    return f"""# Trend Strategy v2 Current State

## Current status

- PR #18 froze a v1 OOS candidate as baseline evidence.
- PR #24 defined its maturity protocol.
- The v1 OOS collector was not implemented or activated; no OOS collection is active.
- Phase A1 implemented the comparison architecture.
- Phase A2 froze adjusted daily OHLCV and completed the empirical comparison and robustness analysis.
- Legacy v1 scanner and backtest behavior remain unchanged.

## Current phase

Phase A2 complete -- empirical score-breakout classification: **{classification['classification']}**

## Phase A2 result

{classification['reason']}

- Frozen economic dates: {summary['snapshot_date_range']['minimum']} through {summary['snapshot_date_range']['maximum']}.
- Frozen snapshot SHA-256: `{summary['complete_snapshot_sha256']}`.
- Score breakout status for Phase B: `{transition['score_breakout_status']}`.
- This is provisional in-sample research, not production approval or genuine OOS evidence.

## Exact next task

{transition['next_phase']} using exactly: {signals}.

Maximum Phase B1 signal candidates: {transition['maximum_signal_candidates']}.

## Blocking limitations

- The historical universe uses present-day AUM and present-day product availability, creating current-universe and survivorship bias.
- Final performance conclusions still require point-in-time universe reconstruction or a survivorship-sensitivity analysis.

## Explicitly deferred

- broad exit-family expansion beyond the Phase B sequence;
- position-sizing optimization;
- point-in-time universe reconstruction;
- new OOS cohort creation;
- OOS collector implementation or activation;
- scanner UI redesign.
"""


def run_empirical_phase_a2(
    root: Path,
    snapshot_dir: Path,
    *,
    bootstrap_paths: int = BOOTSTRAP_PATHS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run Phase A2 strictly from validated frozen files."""
    frozen = load_frozen_inputs(snapshot_dir)
    cfg = dict(frozen.manifest["universe_parameters"])
    features = build_historical_features(frozen.prices, frozen.universe, cfg)
    comparison = run_phase_a_signal_comparison(
        features,
        score_lookbacks=SCORE_LOOKBACK_GRID,
        random_seeds=FREQUENCY_MATCHED_RANDOM_SEEDS,
        round_trip_cost=ROUND_TRIP_COST,
    )
    robustness = build_robustness_evidence(
        comparison,
        features,
        bootstrap_paths=bootstrap_paths,
        bootstrap_seed=bootstrap_seed,
    )
    classification = classify_score_breakout(
        comparison.portfolio_metrics,
        comparison.signal_diagnostic_summary,
        robustness_evidence=robustness["candidate_evidence"],
    )
    representative = (
        str(classification["strategy_key"])
        if classification["classification"] == "Retain"
        else select_representative_score(comparison.portfolio_metrics, robustness)
    )
    transition = transition_to_phase_b1(
        classification, comparison.portfolio_metrics, robustness
    )
    requested = list(frozen.manifest["requested_symbols"])
    retained = list(frozen.manifest["retained_symbols"])
    selected_keys = [
        "score_breakout_l10",
        "score_breakout_l20",
        "score_breakout_l40",
        "trend_filter_only",
        "prior_price_high_l20",
        "signal_surge_v0",
    ]
    summary: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "classification": classification,
        "snapshot_source_code_commit": frozen.manifest["source_code_commit"],
        "analysis_code_commit": _git_head(root.resolve()),
        "representative_score_key": representative,
        "complete_snapshot_sha256": frozen.manifest["complete_snapshot_sha256"],
        "snapshot_date_range": {
            "minimum": frozen.manifest["retained_minimum_economic_date"],
            "maximum": frozen.manifest["retained_maximum_economic_date"],
        },
        "data_coverage": {
            "requested_symbol_count": len(requested),
            "retained_symbol_count": len(retained),
            "failed_or_empty_symbols": frozen.manifest["failed_or_empty_symbols"],
            "row_count": frozen.manifest["row_count"],
        },
        "selected_metrics": {
            key: _metric_record(comparison.portfolio_metrics, key) for key in selected_keys
        },
        "phase_b1_transition": transition,
        "limitations": frozen.manifest["known_limitations"],
    }

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    comparison.signal_diagnostic_summary.sort_values("signal_key").to_csv(
        snapshot_dir / "signal_diagnostic_summary.csv", index=False, lineterminator="\n"
    )
    comparison.portfolio_metrics.sort_values("strategy_key").to_csv(
        snapshot_dir / "portfolio_metrics.csv", index=False, lineterminator="\n"
    )
    _write_json(snapshot_dir / "robustness_evidence.json", robustness)
    _write_json(snapshot_dir / "phase_a2_summary.json", summary)
    (snapshot_dir / "RESULT.md").write_text(
        render_result_markdown(summary, robustness), encoding="utf-8", newline="\n"
    )
    current_state = root / "docs" / "research" / "trend_v2" / "CURRENT_STATE.md"
    current_state.write_text(
        render_current_state(summary), encoding="utf-8", newline="\n"
    )
    return summary
