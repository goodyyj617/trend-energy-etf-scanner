"""Overfitting controls that need memory across sessions (stored under lab_results/).

trials.csv    every distinct strategy that was backtested (single runs, grid cells,
              robustness checks). The count N and the spread of their Sharpe ratios feed
              the Deflated Sharpe Ratio. Re-running an identical setup is not a new trial.
holdout.json  the reserved out-of-sample period: its start date, every change to it, and
              every time a strategy was evaluated on it.
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

TRIALS_PATH = RESULTS_DIR / "trials.csv"
HOLDOUT_PATH = RESULTS_DIR / "holdout.json"
DEFAULT_HOLDOUT_START = "2024-01-01"
TRIAL_COLUMNS = ["time", "kind", "key", "start", "end", "daily_sharpe", "n_days", "description"]


def trial_key(cfg, universe: list[str]) -> str:
    payload = json.dumps({"cfg": cfg.to_dict(), "universe": sorted(universe)}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def log_trials(rows: list[tuple]) -> None:
    """rows: (kind, cfg, universe, equity). Appends only keys not seen before."""
    if not rows:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    seen = set(load_trials()["key"]) if TRIALS_PATH.exists() else set()
    out = []
    for kind, cfg, universe, equity in rows:
        key = trial_key(cfg, universe)
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
            TRIALS_PATH, mode="a", header=not TRIALS_PATH.exists(), index=False, encoding="utf-8")


def load_trials() -> pd.DataFrame:
    if not TRIALS_PATH.exists():
        return pd.DataFrame(columns=TRIAL_COLUMNS)
    return pd.read_csv(TRIALS_PATH, encoding="utf-8")


def trial_stats() -> tuple[int, float]:
    """(number of distinct trials, variance of their daily Sharpe ratios)."""
    t = load_trials()
    sr = pd.to_numeric(t["daily_sharpe"], errors="coerce").dropna()
    return len(t), float(sr.var()) if len(sr) > 1 else float("nan")


# --------------------------------------------------------------------------- holdout

def load_holdout() -> dict:
    if HOLDOUT_PATH.exists():
        try:
            return json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"enabled": True, "start": DEFAULT_HOLDOUT_START, "changes": [], "evaluations": []}


def _save_holdout(state: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    HOLDOUT_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


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
        "fingerprint": cfg.fingerprint(), "period": list(period),
        "metrics": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v)
                    for k, v in metrics.items() if not k.startswith("spy_") or k in ("spy_cagr", "spy_mdd")},
    })
    _save_holdout(state)
    return state
