"""Dynamic layers on a static allocation (allocation DECISIONS A42 §4: A38 and A40 together).

A setting = signal × action × confirmation × tranches × volatility target × cooldown, on top of a target allocation
(quarterly rebalance, 10 bp), simulated with `aa_canary.DynamicSim` (the engine's arithmetic: decide at a close, trade
at the next close). Pieces:

  ``confirm(st, n)``        a new state counts only after n days in a row (0 = as read)
  ``delay(st, k)``          the state read k days late (one tranche of a split decision)
  ``action_tables``         target weights per state: the A40 actions (`aa_canary.tables`) and "riskcd" (A38 type:
                            every equity product → the CD-rate ETF)
  ``vol_levels``            month-end exposure k = min(1, target / realised 60-day volatility of the static mix),
                            rounded to 0.05; the rest goes to the CD-rate ETF; traded at the next close
  ``paths``                 equity paths of one setting (one per tranche) for one or many state series (placebos)

A tranche split averages the equity of n sub-portfolios whose signal is read 0, 5, 10, 15 days late (each holds 1/n
of the money, no rebalancing between them).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import aa_canary as K
from .aa_assets import CASH, PRODUCTS
from .aa_engine import TRADING_DAYS, decision_rows

CD = "459580"
CONFIRM_DAYS = 10
TRANCHE_LAGS = {1: (0,), 2: (0, 5), 4: (0, 5, 10, 15)}
VOL_WINDOW = 60
K_STEP = 0.05
K_LEVELS = np.round(np.arange(K_STEP, 1.0 + 1e-9, K_STEP), 2)  # 0.05 … 1.00
SIGNALS = ("none", "k2", "k3", "k4", "k6", "k8", "d1")
ACTIONS = ("defensive", "safe", "riskcd", "tilt_nd")
ACTION_KO = {"defensive": "나스닥 → 방어 주식", "safe": "나스닥 → 안전 묶음", "riskcd": "모든 위험자산 → CD (A38형)",
             "tilt_nd": "나스닥 ↔ 방어 주식 10%p", "none": "—"}


@dataclass(frozen=True)
class Setting:
    signal: str = "none"
    action: str = "none"
    confirm: int = 0
    tranches: int = 1
    vol: float = 0.0  # annual target, 0 = off
    cooldown: int = 0

    def label(self) -> str:
        if self.signal == "none":
            sig = "신호 없음"
        else:
            sig = f"{K.RULES[self.signal].label} · {ACTION_KO[self.action]} · 확인 {self.confirm}일 · 쿨다운 {self.cooldown}일"
        return f"{sig} · 트랑슈 {self.tranches} · 변동성 목표 {'없음' if not self.vol else f'{self.vol:.0%}'}"


def all_settings() -> list[Setting]:
    """6 signals × 4 actions × 2 × 3 × 3 × 2 = 864, plus 'no signal' × 3 tranches × 3 volatility targets = 9."""
    out = [Setting("none", "none", 0, n, v, 0) for v in (0.0, 0.08, 0.10) for n in (1, 2, 4)]
    for s in SIGNALS[1:]:
        for a in ACTIONS:
            for c in (0, CONFIRM_DAYS):
                for n in (1, 2, 4):
                    for v in (0.0, 0.08, 0.10):
                        for cd in (0, K.COOLDOWN):
                            out.append(Setting(s, a, c, n, v, cd))
    return out


def confirm(st: np.ndarray, n: int) -> np.ndarray:
    """A change of state is taken only when the raw state has been the new value for n days in a row."""
    st = np.asarray(st, dtype=int)
    if n <= 1:
        return st.copy()
    out = st.copy()
    held = st[0]
    run, last = 0, st[0]
    for t in range(len(st)):
        run = run + 1 if st[t] == last else 1
        last = st[t]
        if st[t] != held and run >= n:
            held = st[t]
        out[t] = held
    return out


def confirm_many(S: np.ndarray, n: int) -> np.ndarray:
    """`confirm` for each row of S (paths × days), vectorised over the rows."""
    S = np.asarray(S, dtype=int)
    if n <= 1:
        return S.copy()
    P, T = S.shape
    out = np.empty_like(S)
    held = S[:, 0].copy()
    last = S[:, 0].copy()
    run = np.zeros(P, dtype=int)
    for t in range(T):
        same = S[:, t] == last
        run = np.where(same, run + 1, 1)
        last = S[:, t]
        flip = (S[:, t] != held) & (run >= n)
        held = np.where(flip, S[:, t], held)
        out[:, t] = held
    return out


def delay(S: np.ndarray, k: int) -> np.ndarray:
    """The state read k days late (the first k days keep the first state)."""
    S = np.asarray(S)
    if k <= 0:
        return S.copy()
    out = np.empty_like(S)
    out[..., k:] = S[..., :-k]
    out[..., :k] = S[..., :1]
    return out


def equity_products(columns: list[str]) -> list[str]:
    return [c for c in columns if c in PRODUCTS and PRODUCTS[c].asset_class == "주식"]


def action_tables(base: dict[str, float], signal: str, action: str, columns: list[str]) -> np.ndarray:
    """Target weights (K × columns, fractions) per state. 'none' → one row; tilt_nd keeps states 0 and 1."""
    tot = sum(base.values())
    b = np.zeros(len(columns))
    for c, x in base.items():
        b[columns.index(c)] += x / tot
    if signal == "none" or action == "none":
        return b[None, :]
    if action == "riskcd":
        off = b.copy()
        for c in equity_products(columns):
            i = columns.index(c)
            off[columns.index(CD)] += off[i]
            off[i] = 0.0
        return np.vstack([b, off])
    if base.get(K.NASDAQ, 0) <= 0:  # nothing to move (F2 / F3 hold no nasdaq-100): the action changes nothing
        return np.vstack([b, b])
    tab = K.tables(base, signal, action, columns)
    return tab[:2]


def full_move(base: dict[str, float], signal: str, action: str, columns: list[str]) -> np.ndarray:
    """The risk-off target an equal-cost static moves toward: the action's state-1 row; no signal: all to CD."""
    if signal == "none":
        v = np.zeros(len(columns))
        v[columns.index(CD)] = 1.0
        return v
    return action_tables(base, signal, action, columns)[1]


def scaled_tables(tab: np.ndarray, columns: list[str], use_vol: bool) -> np.ndarray:
    """K × columns → (K × L) × columns: each state's weights times k, the rest in the CD-rate ETF (L = 1 without
    the volatility target)."""
    if not use_vol:
        return tab
    cd = columns.index(CD)
    out = []
    for row in tab:
        for k in K_LEVELS:
            v = row * k
            v[cd] += 1.0 - k
            out.append(v)
    return np.vstack(out)


def vol_levels(daily_ret: np.ndarray, dates: pd.DatetimeIndex, target: float) -> np.ndarray:
    """Index into K_LEVELS held on each row: set at month-end closes (k = min(1, target / σ60)), from the next row."""
    r = pd.Series(daily_ret, index=dates)
    vol = r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().to_numpy() * np.sqrt(TRADING_DAYS)
    T = len(dates)
    lev = np.full(T, len(K_LEVELS) - 1)
    cur = len(K_LEVELS) - 1
    month_end = set(decision_rows(dates, "M").tolist())
    for t in range(T):
        lev[t] = cur
        if t in month_end and np.isfinite(vol[t]) and vol[t] > 0:
            k = min(1.0, target / vol[t])
            cur = int(np.clip(np.round(k / K_STEP) - 1, 0, len(K_LEVELS) - 1))
    return lev


def compose(S: np.ndarray, qrows: np.ndarray, cooldown: int, lev: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """decide / choice / first for DynamicSim from state rows S (P × T) and the exposure levels."""
    S = np.atleast_2d(S)
    P, T = S.shape
    L = len(K_LEVELS) if lev is not None else 1
    decide = np.zeros((P, T), dtype=bool)
    choice = np.zeros((P, T), dtype=int)
    first = np.zeros(P, dtype=int)
    if lev is not None:
        lchange = np.zeros(T, dtype=bool)
        lchange[1:] = lev[1:] != lev[:-1]
        lchange_dec = np.zeros(T, dtype=bool)
        lchange_dec[:-1] = lchange[1:]  # the level set at a month-end close trades at the next close: decide there
        lchange_dec[T - 1] = False
    for p in range(P):
        sc = K.schedule(S[p], qrows, cooldown)
        if lev is None:
            decide[p], choice[p], first[p] = sc.decide, sc.choice, sc.first
        else:
            lv_next = np.concatenate([lev[1:], lev[-1:]])  # the level decided at row t is the one held from t+1
            decide[p] = sc.decide | lchange_dec
            choice[p] = sc.choice * L + lv_next
            first[p] = sc.first * L + lev[0]
    return decide, choice, first


def paths(sim: K.DynamicSim, tab: np.ndarray, S: np.ndarray, setting: Setting, qrows: np.ndarray,
          lev: np.ndarray | None) -> np.ndarray:
    """Equity paths (P × T) of a setting for state rows S (P × T): tranches averaged, each starting at 1."""
    S = np.atleast_2d(S)
    S = confirm_many(S, setting.confirm) if setting.confirm else S
    tabs_one = scaled_tables(tab, sim.columns, lev is not None)
    out = np.zeros((S.shape[0], sim.n_days))
    lags = TRANCHE_LAGS[setting.tranches]
    for k in lags:
        Sk = delay(S, k)
        decide, choice, first = compose(Sk, qrows, setting.cooldown, lev)
        tabs = np.broadcast_to(tabs_one, (len(Sk),) + tabs_one.shape)
        p, _, _ = sim.run(tabs, decide, choice, first)
        out += p / p[:, :1]
    return out / len(lags)


def cagr(P: np.ndarray) -> np.ndarray:
    P = np.atleast_2d(P)
    return (P[:, -1] / P[:, 0]) ** (TRADING_DAYS / max(P.shape[1] - 1, 1)) - 1.0


def mdd(P: np.ndarray) -> np.ndarray:
    P = np.atleast_2d(P)
    return (P / np.maximum.accumulate(P, axis=1) - 1.0).min(axis=1)


def mix(core: np.ndarray, sat: np.ndarray | None, share: float) -> np.ndarray:
    """Daily fixed-ratio mix of the core paths with one satellite path (returns mixed, no trading cost); the
    satellite counts as the core before it starts (NaN)."""
    if share <= 0 or sat is None:
        return core
    core = np.atleast_2d(core)
    rc = core[:, 1:] / core[:, :-1] - 1.0
    rs = sat[1:] / sat[:-1] - 1.0
    rs = np.where(np.isfinite(rs), rs, np.nan)
    r = np.where(np.isfinite(rs)[None, :], (1 - share) * rc + share * rs[None, :], rc)
    return np.concatenate([np.ones((len(core), 1)), np.cumprod(1 + r, axis=1)], axis=1)


__all__ = ["Setting", "all_settings", "confirm", "confirm_many", "delay", "action_tables", "full_move", "scaled_tables",
           "vol_levels", "compose", "paths", "cagr", "mdd", "mix", "SIGNALS", "ACTIONS", "ACTION_KO", "CD", "K_LEVELS"]
