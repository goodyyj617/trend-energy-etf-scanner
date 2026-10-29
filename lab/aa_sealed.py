"""The A42 sealed window (allocation DECISIONS A42 §0-5, stage 5): 1990-01-02 – 2000-08-31.

The series for this window live only on Drive (`lab_exchange/a42_sealed/`), written unread by
scripts/aa_a42_fetch.py with their SHA-256 in docs/research/allocation/2026-10-29_a42_stage0/sealed_hashes.csv.
Research code must not read them, and must not start a backtest before 2000-09-01, until stage 5 opens the window
once with `--unseal-a42` (`unseal=True` here). Every read is checked against the recorded hash.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

SEAL_START, SEAL_END = "1990-01-02", "2000-08-31"
RESEARCH_START = "2000-09-01"
SEALED_DIR = Path(os.environ.get("A42_SEALED_DIR", r"H:\내 드라이브\lab_exchange\a42_sealed"))
HASHES = Path(__file__).resolve().parents[1] / "docs" / "research" / "allocation" / "2026-10-29_a42_stage0" / "sealed_hashes.csv"


class SealedError(RuntimeError):
    """Raised when code reaches into the sealed window without --unseal-a42."""


def guard(start: str | pd.Timestamp, unseal: bool = False) -> None:
    """Refuse a start date inside or before the sealed window unless unsealed."""
    if pd.Timestamp(start) < pd.Timestamp(RESEARCH_START) and not unseal:
        raise SealedError(f"{RESEARCH_START} 이전은 A42 봉인 구간입니다. 5단계에서 --unseal-a42로만 엽니다 (요청: {pd.Timestamp(start).date()}).")


def read(name: str, unseal: bool = False) -> pd.DataFrame:
    """One sealed file, after checking its hash. Raises SealedError without unseal=True."""
    if not unseal:
        raise SealedError("봉인 구간 자료는 --unseal-a42 없이 읽지 않습니다.")
    data = (SEALED_DIR / name).read_bytes()
    want = pd.read_csv(HASHES).set_index("file")["sha256"].get(name)
    if want is None or hashlib.sha256(data).hexdigest() != want:
        raise SealedError(f"{name}의 해시가 기록과 다릅니다.")
    from io import BytesIO
    return pd.read_csv(BytesIO(data))


__all__ = ["SEAL_START", "SEAL_END", "RESEARCH_START", "SEALED_DIR", "SealedError", "guard", "read"]
