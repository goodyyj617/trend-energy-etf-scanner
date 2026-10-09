"""Overfitting controls that need memory across sessions (stored under lab_results/).

trials.csv    every distinct strategy that was backtested (single runs, grid cells,
              robustness checks). The count N and the spread of their Sharpe ratios feed
              the Deflated Sharpe Ratio. Re-running an identical setup is not a new trial.
trial_returns.npz  monthly returns of each trial (month-end x trial key). Their average
              pairwise correlation rho turns N into an effective number of independent
              trials, N_eff = rho + (1 - rho) * N: neighbouring grid cells are near-copies of
              each other and should not count as fully separate attempts.
holdout.json  the reserved out-of-sample period: its start date, every change to it, and
              every time a strategy was evaluated on it.

All paths are derived from RESULTS_DIR when used, so pointing RESULTS_DIR elsewhere (tests)
redirects every file. The returns file is plain numpy arrays, readable by any pandas version.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import daily_sharpe
from .store import RESULTS_DIR

DEFAULT_HOLDOUT_START = "2024-01-01"
TRIAL_COLUMNS = ["time", "kind", "key", "start", "end", "daily_sharpe", "n_days", "description"]
MIN_OVERLAP_MONTHS = 24


def _trials_path() -> Path:
    return RESULTS_DIR / "trials.csv"


def _returns_path() -> Path:
    return RESULTS_DIR / "trial_returns.npz"


def _holdout_path() -> Path:
    return RESULTS_DIR / "holdout.json"


def trial_key(cfg, universe: list[str]) -> str:
    payload = json.dumps({"cfg": cfg.to_dict(), "universe": sorted(universe)}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def log_trials(rows: list[tuple]) -> None:
    """rows: (kind, cfg, universe, equity). Appends only keys not seen before.
    Monthly returns are stored for every key that does not have them yet (also old keys)."""
    if not rows:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    seen = set(load_trials()["key"]) if _trials_path().exists() else set()
    returns = load_trial_returns()
    new_returns = {}
    out = []
    for kind, cfg, universe, equity in rows:
        key = trial_key(cfg, universe)
        if key not in returns.columns and key not in new_returns:
            new_returns[key] = equity.dropna().resample("ME").last().pct_change(fill_method=None).dropna()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "time": datetime.now().isoformat(timespec="seconds"), "kind": kind, "key": key,
            "start": str(equity.index[0].date()), "end": str(equity.index[-1].date()),
            "daily_sharpe": daily_sharpe(equity), "n_days": len(equity), "description": cfg.describe(),
        })
    if out:
        pd.DataFrame(out, columns=TRIAL_COLUMNS).to_csv(
            _trials_path(), mode="a", header=not _trials_path().exists(), index=False, encoding="utf-8")
    if new_returns:
        _save_returns(pd.concat([returns, pd.DataFrame(new_returns)], axis=1))


def _save_returns(df: pd.DataFrame) -> None:
    df = df.sort_index()
    np.savez(_returns_path(), dates=df.index.to_numpy(dtype="datetime64[ns]").astype("int64"),
             keys=np.array([str(k) for k in df.columns]), values=df.to_numpy(dtype=float))


def load_trial_returns() -> pd.DataFrame:
    path = _returns_path()
    legacy = RESULTS_DIR / "trial_returns.pkl"
    if not path.exists() and legacy.exists():  # one-time migration from the first (pickle) format
        old = pd.read_pickle(legacy)
        keep = [k for k in old.columns if k in set(load_trials()["key"])]  # drops entries not in the trial log
        _save_returns(old[keep])
        legacy.unlink()
    if not path.exists():
        return pd.DataFrame()
    with np.load(path, allow_pickle=False) as z:
        return pd.DataFrame(z["values"], index=pd.to_datetime(z["dates"]), columns=list(z["keys"]))


_EFF_CACHE: dict = {}


def mean_pairwise_corr(x: np.ndarray, min_overlap: int = MIN_OVERLAP_MONTHS) -> float:
    """Mean Pearson correlation over all column pairs, each pair using its overlapping rows
    (same result as DataFrame.corr(min_periods=...) averaged off the diagonal).

    Trials share a few date ranges, so columns are grouped by their non-missing pattern and
    each pair of groups is handled as one matrix product on the rows both groups have.
    """
    valid = ~np.isnan(x)
    groups: dict[bytes, list[int]] = {}
    for j in range(x.shape[1]):
        groups.setdefault(valid[:, j].tobytes(), []).append(j)
    keys = list(groups)
    total, count = 0.0, 0
    for a in range(len(keys)):
        for b in range(a, len(keys)):
            ca, cb = groups[keys[a]], groups[keys[b]]
            rows = valid[:, ca[0]] & valid[:, cb[0]]
            m = int(rows.sum())
            if m < min_overlap:
                continue

            def z(cols):
                block = x[np.ix_(rows, cols)]
                sd = block.std(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return (block - block.mean(axis=0)) / sd, sd > 0

            za, oka = z(ca)
            zb, okb = (za, oka) if a == b else z(cb)
            corr = (za[:, oka].T @ zb[:, okb]) / m
            if a == b:  # same group: every unordered pair once, no diagonal
                iu = np.triu_indices(corr.shape[0], k=1)
                total += corr[iu].sum() * 2
                count += len(iu[0]) * 2
            else:
                total += corr.sum() * 2
                count += corr.size * 2
    return total / count if count else 0.0


def effective_trials() -> tuple[float, float, int]:
    """(N_eff, average pairwise correlation rho, number of trials with stored returns).

    rho is the mean of all pairwise correlations of monthly trial returns (pairs overlapping
    at least MIN_OVERLAP_MONTHS). Without returns, N_eff falls back to N (rho = 0).
    """
    n = len(load_trials())
    r = load_trial_returns()
    if r.empty:
        return float(n), 0.0, 0
    stamp = (str(_returns_path()), _returns_path().stat().st_mtime, n)
    if _EFF_CACHE.get("stamp") != stamp:
        _EFF_CACHE.update(stamp=stamp, value=(mean_pairwise_corr(r.to_numpy(dtype=float)), r.shape[1]))
    rho, covered = _EFF_CACHE["value"]
    rho = min(max(rho, 0.0), 1.0)
    return rho + (1 - rho) * n, rho, covered


def load_trials() -> pd.DataFrame:
    if not _trials_path().exists():
        return pd.DataFrame(columns=TRIAL_COLUMNS)
    return pd.read_csv(_trials_path(), encoding="utf-8")


def trial_stats() -> tuple[int, float]:
    """(number of distinct trials, variance of their daily Sharpe ratios)."""
    t = load_trials()
    sr = pd.to_numeric(t["daily_sharpe"], errors="coerce").dropna()
    return len(t), float(sr.var()) if len(sr) > 1 else float("nan")


# --------------------------------------------------------------------------- holdout

def load_holdout() -> dict:
    if _holdout_path().exists():
        try:
            return json.loads(_holdout_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"enabled": True, "start": DEFAULT_HOLDOUT_START, "changes": [], "evaluations": []}


def _save_holdout(state: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    _holdout_path().write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_holdout(enabled: bool, start: str) -> dict:
    state = load_holdout()
    if state["enabled"] != enabled or state["start"] != start:
        state["changes"].append({"time": datetime.now().isoformat(timespec="seconds"),
                                 "from": {"enabled": state["enabled"], "start": state["start"]},
                                 "to": {"enabled": enabled, "start": start}})
        state["enabled"], state["start"] = enabled, start
        _save_holdout(state)
    return state


def record_holdout_evaluation(name: str, cfg, metrics: dict, period: tuple[str, str]) -> dict:
    state = load_holdout()
    state["evaluations"].append({
        "time": datetime.now().isoformat(timespec="seconds"), "name": name, "description": cfg.describe(),
        "fingerprint": cfg.fingerprint(), "strategy_fingerprint": cfg.strategy_fingerprint(), "period": list(period),
        "metrics": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v)
                    for k, v in metrics.items() if not k.startswith("spy_") or k in ("spy_cagr", "spy_mdd")},
    })
    _save_holdout(state)
    return state
