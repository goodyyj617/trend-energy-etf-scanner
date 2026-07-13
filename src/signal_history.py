from __future__ import annotations

from pathlib import Path

import pandas as pd


SIGNAL_COL = "signal_surge_v0"


def _empty_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal_streak_start_date"] = None
    out["signal_streak_trading_days"] = 0
    out["signal_streak_calendar_days"] = 0
    out["is_first_signal_today"] = False
    return out


def _load_history(history_dir: Path) -> pd.DataFrame:
    if not history_dir.exists():
        return pd.DataFrame()

    frames = []
    for path in sorted(history_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[signal_history] failed to read {path}: {exc}")
            continue

        if "symbol" not in df.columns or "date" not in df.columns:
            continue

        if SIGNAL_COL not in df.columns:
            continue

        frames.append(df[["symbol", "date", SIGNAL_COL]].copy())

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["symbol", "date"])
    hist[SIGNAL_COL] = hist[SIGNAL_COL].fillna(False).astype(bool)
    return hist


def _compute_one_symbol_streak(g: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    g = g.sort_values("date").drop_duplicates(subset=["date"], keep="last").copy()

    current_rows = g[g["date"].eq(as_of)]
    if current_rows.empty:
        return {
            "signal_streak_start_date": None,
            "signal_streak_trading_days": 0,
            "signal_streak_calendar_days": 0,
            "is_first_signal_today": False,
        }

    is_current_signal = bool(current_rows.iloc[-1][SIGNAL_COL])
    if not is_current_signal:
        return {
            "signal_streak_start_date": None,
            "signal_streak_trading_days": 0,
            "signal_streak_calendar_days": 0,
            "is_first_signal_today": False,
        }

    streak_dates = []
    for _, row in g.sort_values("date", ascending=False).iterrows():
        if not bool(row[SIGNAL_COL]):
            break
        streak_dates.append(row["date"])

    if not streak_dates:
        return {
            "signal_streak_start_date": None,
            "signal_streak_trading_days": 0,
            "signal_streak_calendar_days": 0,
            "is_first_signal_today": False,
        }

    start_date = min(streak_dates)
    trading_days = len(streak_dates)
    calendar_days = int((as_of.normalize() - start_date.normalize()).days)
    is_first_signal_today = trading_days == 1

    return {
        "signal_streak_start_date": start_date.date().isoformat(),
        "signal_streak_trading_days": int(trading_days),
        "signal_streak_calendar_days": int(calendar_days),
        "is_first_signal_today": bool(is_first_signal_today),
    }


def add_signal_history(latest: pd.DataFrame, history_dir: str | Path, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """
    Add current signal streak information.

    Definitions:
    - signal_streak_start_date:
        first date of the current consecutive TRUE streak
    - signal_streak_trading_days:
        number of trading rows in the current consecutive TRUE streak
    - signal_streak_calendar_days:
        calendar days from streak start to as_of
    - is_first_signal_today:
        TRUE if today is the first TRUE day of the current streak

    Only current TRUE signals receive streak metadata.
    Current FALSE rows get blank / zero values.
    """
    if latest.empty:
        return _empty_signal_columns(latest)

    if "symbol" not in latest.columns or "date" not in latest.columns or SIGNAL_COL not in latest.columns:
        return _empty_signal_columns(latest)

    out = _empty_signal_columns(latest)

    as_of_ts = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(as_of_ts):
        as_of_ts = pd.to_datetime(out["date"], errors="coerce").max()

    if pd.isna(as_of_ts):
        return out

    hist = _load_history(Path(history_dir))

    current = out[["symbol", "date", SIGNAL_COL]].copy()
    current["date"] = pd.to_datetime(current["date"], errors="coerce")
    current = current.dropna(subset=["symbol", "date"])
    current[SIGNAL_COL] = current[SIGNAL_COL].fillna(False).astype(bool)

    combined = pd.concat([hist, current], ignore_index=True)
    if combined.empty:
        return out

    combined = combined.dropna(subset=["symbol", "date"])
    combined = combined.sort_values(["symbol", "date"])
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")

    records = {}
    for symbol, g in combined.groupby("symbol", sort=False):
        records[symbol] = _compute_one_symbol_streak(g, as_of_ts)

    for idx, row in out.iterrows():
        symbol = row.get("symbol")
        record = records.get(symbol)
        if not record:
            continue

        out.at[idx, "signal_streak_start_date"] = record["signal_streak_start_date"]
        out.at[idx, "signal_streak_trading_days"] = record["signal_streak_trading_days"]
        out.at[idx, "signal_streak_calendar_days"] = record["signal_streak_calendar_days"]
        out.at[idx, "is_first_signal_today"] = record["is_first_signal_today"]

    return out
