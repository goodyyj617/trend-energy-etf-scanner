from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .features import add_signal_surge_v0, compute_symbol_features

SIGNAL_COL = "signal_surge_v0"
ROUND_TRIP_COST = 0.002
MAX_HOLDING_DAYS = 63


@dataclass(frozen=True)
class EntryRule:
    key: str
    label: str
    description: str
    indices_fn: Callable[[pd.DataFrame], list[int]]


@dataclass(frozen=True)
class ExitRule:
    key: str
    label: str
    description: str
    stop_fn: Optional[Callable[[pd.Series], float]] = None
    use_trailing_max: