"""Save and reload allocation strategies under lab_results/ (kind "alloc", one folder each).

meta.json holds the configuration (an AllocConfig, or a MixConfig under "mix"), the period and
the headline metrics; equity.csv holds the strategy, SPY and exposure; weights.csv the daily
weights. A saved strategy is re-run on the current data when it is compared, so the scorecard
always uses one common period.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .aa_engine import AllocConfig, AllocResult, MixConfig
from .store import RESULTS_DIR

KIND = "alloc"


def _folder(name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:40] or KIND
    path = RESULTS_DIR / f"{stamp}_{KIND}_{safe}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_result(res: AllocResult, metrics: dict, name: str, data_source: str, mix: MixConfig | None = None) -> Path:
    path = _folder(name)
    start, end = res.period
    meta = {
        "kind": KIND, "name": name, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": data_source, "period": [str(start.date()), str(end.date())],
        "config": res.config.to_dict(), "description": (mix.describe() if mix else res.config.describe()),
        "metrics": {k: (None if v is None or (isinstance(v, float) and v != v) else v) for k, v in metrics.items()},
    }
    if mix is not None:
        meta["mix"] = mix.to_dict()
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame({"strategy": res.equity, "SPY": res.benchmark, "exposure": res.exposure}).to_csv(path / "equity.csv")
    res.weights.to_csv(path / "weights.csv")
    return path


def list_saved() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    out = []
    for meta_path in sorted(RESULTS_DIR.glob(f"*_{KIND}_*/meta.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("kind") != KIND:
            continue
        meta["path"] = meta_path.parent
        out.append(meta)
    return out


def config_of(meta: dict) -> AllocConfig | MixConfig:
    if "mix" in meta:
        return MixConfig.from_dict(meta["mix"])
    return AllocConfig.from_dict(meta["config"])


def load_equity(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "equity.csv", index_col=0, parse_dates=True)


def load_weights(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "weights.csv", index_col=0, parse_dates=True)
