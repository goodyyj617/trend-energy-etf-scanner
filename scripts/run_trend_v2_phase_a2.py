from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trend_v2_phase_a2 import (  # noqa: E402
    BOOTSTRAP_PATHS,
    BOOTSTRAP_SEED,
    collect_and_freeze_snapshot,
    refresh_phase_a2_reports,
    run_empirical_phase_a2,
)


DEFAULT_SNAPSHOT_DIR = ROOT / "docs" / "research" / "trend_v2" / "phase_a2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trend Strategy v2 Phase A2")
    parser.add_argument("action", choices=("collect", "analyze", "report"))
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--bootstrap-paths", type=int, default=BOOTSTRAP_PATHS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot_dir.resolve()
    if args.action == "collect":
        result = collect_and_freeze_snapshot(ROOT, snapshot_dir)
    elif args.action == "analyze":
        if args.bootstrap_paths < BOOTSTRAP_PATHS:
            raise ValueError(
                f"final Phase A2 analysis requires at least {BOOTSTRAP_PATHS} bootstrap paths"
            )
        result = run_empirical_phase_a2(
            ROOT,
            snapshot_dir,
            bootstrap_paths=args.bootstrap_paths,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        result = refresh_phase_a2_reports(ROOT, snapshot_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
