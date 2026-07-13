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
RECENT_TRADE_LIMIT = 250

# Score breakout robustness grid.
# The grid is intentionally compact enough for the daily GitHub Actions job,
# but structured so that more parameters can be added later for statistical tests.
SCORE_LOOKBACK_GRID = [10, 20, 40]
R20_MIN_GRID = [-0.02, 0.00, 0.02]
ER20_MIN_GRID = [0.05, 0.10, 0.15]


@dataclass(frozen=True