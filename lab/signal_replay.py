"""Replay a slot-limited account from a saved per-signal table, and look up screens at each signal.

A Stage C cell's per-signal table (engine run_backtest(outcomes=True), saved by lab/stage_c_study.py)
already holds, for every entry signal, its next-open entry and the close on which the rule's exit
fires. The account of the engine (max_positions slots, 1 / slots of equity at purchase, no
rebalancing, held symbols skipped, signals beyond the free slots dropped, cost on both sides, cash
at the T-bill rate) then needs only open and close prices: `replay` rebuilds it without running
the blocks or the exits again. With `keep` (a mask on the signals, e.g. from a pre-buy screen) a
screen x entry x exit study becomes a filter on the table plus a cheap replay.

`signal_values` evaluates any screen or score of lab/screens.py on the panel and reads its value on
each signal's (symbol, signal day): data up to that day's close only, the same moment the engine
decides the purchase, so there is no lookahead.
"""
from __future__ import annotations

from operator import mul

import numpy as np
import pandas as pd

from .data import Panel
from .engine import CANCELLED, TRADE_COLUMNS, SignalCache, StrategyConfig


def _window(panel: Panel, cfg: StrategyConfig) -> pd.DatetimeIndex:
    idx = panel.close.index
    return idx[(idx >= pd.Timestamp(cfg.start)) & (idx <= pd.Timestamp(cfg.end))]


def replay(panel: Panel, cfg: StrategyConfig, signals: pd.DataFrame, keep=None,
           cache: SignalCache | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """(equity, trades) of the engine's account on the signals of `signals` (rows where keep is True).

    signals: symbol, signal_date, exit_signal_date (NaT = no exit in the window), exit_reason
    (CANCELLED = no price at the next open). Older tables without exit_signal_date fall back to the
    session before the sale (exact unless the open after the exit had no price).
    Buying order is the engine's: panel.rank (smaller first) if set, else the cfg.liquidity_days
    dollar volume (larger first); ties by column order.
    """
    dates = _window(panel, cfg)
    n = len(dates)
    cols = panel.close.columns
    sig = signals if keep is None else signals[np.asarray(keep, dtype=bool)]
    si = dates.get_indexer(pd.DatetimeIndex(sig["signal_date"]))
    sj = cols.get_indexer(sig["symbol"].astype(str))
    if "exit_signal_date" in sig:
        xi = dates.get_indexer(pd.DatetimeIndex(sig["exit_signal_date"]))
    else:  # sale index - 1
        xi = np.where(sig["closed"].to_numpy(dtype=bool), si + 1 + sig["holding_days"].to_numpy() - 1, -1)
    ok = (si >= 0) & (si < n - 1) & (sj >= 0)
    # exit signal day of a position bought at the open after signal day i (the engine's exit_at)
    exit_of: dict[tuple[int, int], int | None] = {}
    by_day: dict[int, list[int]] = {}
    for i, j, x in zip(si[ok].tolist(), sj[ok].tolist(), xi[ok].tolist()):
        exit_of[(j, i + 1)] = x if x >= 0 else None
        by_day.setdefault(i, []).append(j)
    cache = cache or SignalCache(panel)
    if panel.rank is not None:
        key = np.nan_to_num(-panel.rank.loc[dates].to_numpy(dtype=float), nan=-1e18)
    else:
        key = np.nan_to_num(cache.dollar_volume(int(cfg.liquidity_days)).loc[dates].to_numpy(dtype=float), nan=-1.0)
    ordered = {i: sorted(sorted(set(js)), key=lambda j, ki=key[i]: -ki[j]) for i, js in by_day.items()}

    O = panel.open.loc[dates].to_numpy(dtype=float)
    LC = panel.close.loc[dates].ffill().to_numpy(dtype=float)
    cash_r = (panel.cash.reindex(dates).fillna(0.0).to_numpy().tolist() if (cfg.cash_yield and panel.cash is not None)
              else [0.0] * n)
    c = cfg.cost_bps / 10_000.0
    K = int(cfg.max_positions)
    cash = 1.0
    last_close = np.full(len(cols), np.nan)
    pos: dict[int, dict] = {}
    pending_entry: list[int] = []
    pending_exit: dict[int, bool] = {}
    exits_on: dict[int, list[int]] = {}
    equity = [0.0] * n
    trades = []
    for i in range(n):
        Oi = O[i]
        for j in list(pending_exit):
            px = Oi[j]
            if px != px:
                continue
            p = pos.pop(j)
            cash += p["shares"] * px * (1 - c)
            trades.append((cols[j], dates[p["entry_i"]], dates[i], p["entry_price"], px,
                           (px * (1 - c)) / (p["entry_price"] * (1 + c)) - 1.0, i - p["entry_i"], ""))
            del pending_exit[j]
        if pending_entry:
            held_value = sum(p["shares"] * float(Oi[j] if Oi[j] == Oi[j] else last_close[j]) for j, p in pos.items())
            slot = (cash + held_value) / K
            for j in pending_entry:
                if len(pos) >= K or cash <= 1e-9:
                    break
                px = Oi[j]
                if px != px or px <= 0:
                    continue
                px = float(px)
                alloc = min(slot, cash)
                pos[j] = {"shares": alloc / (px * (1 + c)), "entry_price": px, "entry_i": i}
                cash -= alloc
                x = exit_of.get((j, i))
                if x is not None:
                    exits_on.setdefault(x, []).append(j)
            pending_entry = []
        cash *= 1.0 + cash_r[i]
        last_close = LC[i]
        held = list(pos)
        equity[i] = cash + sum(map(mul, [pos[j]["shares"] for j in held], map(float, map(last_close.__getitem__, held))))
        for j in exits_on.pop(i, ()):
            pending_exit[j] = True
        free = K - (len(pos) - len(pending_exit))
        if free > 0 and i in ordered:
            pending_entry = [j for j in ordered[i] if j not in pos][:free]
    return pd.Series(equity, index=dates, name="strategy", dtype=float), pd.DataFrame(trades, columns=TRADE_COLUMNS)


def signal_values(panel: Panel, signals: pd.DataFrame, key: str, value: float | None = None,
                  frame: pd.DataFrame | None = None) -> pd.Series:
    """The value of a lab/screens.py screen or score on each signal's (symbol, signal day), aligned to
    the signals' index. Exclusion and hygiene screens give 1.0 = 'would be excluded'. Values use data
    up to the signal-day close (the decision moment), so there is no lookahead. `frame` reuses a
    precomputed screen frame (dates x symbols)."""
    if frame is None:
        from . import screens

        frame = screens.ALL_SCREENS[key].compute(panel, value)
    frame = frame.astype("float64")
    pos = frame.index.get_indexer(pd.DatetimeIndex(signals["signal_date"]))
    cols = frame.columns.get_indexer(signals["symbol"].astype(str))
    ok = (pos >= 0) & (cols >= 0)
    out = np.full(len(signals), np.nan)
    out[ok] = frame.to_numpy()[pos[ok], cols[ok]]
    return pd.Series(out, index=signals.index, name=key)


def keep_mask(panel: Panel, signals: pd.DataFrame, exclude: dict | None = None) -> np.ndarray:
    """True for signals that pass every exclusion screen in `exclude` ({key: value}) and were filled
    (not CANCELLED). Selection by score is a ranking among that day's names: build it with
    stage_c_study.daily_eligible and look it up with signal_values(frame=...)."""
    # np.array copies: under pandas copy-on-write .to_numpy() can be a read-only view
    keep = np.array(signals["exit_reason"] != CANCELLED, dtype=bool) if "exit_reason" in signals else np.ones(len(signals), bool)
    for key, value in (exclude or {}).items():
        keep &= ~(signal_values(panel, signals, key, value).fillna(0.0).to_numpy() > 0.5)
    return keep
