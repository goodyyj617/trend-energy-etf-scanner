"""Stage H: picking big winners walk-forward, with a 1%-rule account (DECISIONS 86), resumable.

    python -m lab.pick_study --panel DIR --eligible FILE --v2 DIR --events DIR --out DIR [--workers N] [--dry]

1. H1 signal tables (E1-E9 and V1-V6 canonical entries x 25% trailing or ATR k stop, k 1.5 / 2 / 3),
   built like `stage_c_study --signals-only`. H9 = Stage C X9 tables (v2 for E, G-2 for V).
2. Per setting (exit H1 / H9 x k): the pooled stock-day signals, their 1%-rule weight
   w = min(1% / stop%, 10%), contribution w x return, labels (main: contribution >= +5%; B50; SPY
   excess > 0) and the 40 G-1 features + 15 entry flags (computed once, temp folder only).
3. Walk-forward 2000-2023 (first fit 1993-1999, refit each January on trades closed before it):
   L2 logistic and a training-direction rank average, with and without ATR%, for each label.
   Placebo: training labels shuffled by stock blocks, 100 times (main variant).
4. Accounts: 10-stock 1%-rule account (score order vs dollar volume vs 100 random orders) and no
   stock limit (each signal at its weight; calendar-time portfolio scaled to 100% exposure; model as
   a top-30% filter vs all). Capped and uncapped trades are reported apart.
Output (aggregates only in the repo): settings/<x>_<k>/*.csv, summary.csv, verdicts.csv,
summary.json, figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import event_study as ev, figures, stage_c_study as sc, tail_study as ts
from .crsp_study import SAMPLES, load_panel_with_cash
from .data import BENCHMARK
from .engine import CANCELLED
from .metrics import deflated_sharpe

# ---- registered (DECISIONS 86): fixed before any result
K_VALUES = [1.5, 2.0, 3.0]
EXITS_H = ("H1", "H9")
H1_TRAIL = 25
RISK = 0.01
CAP = 0.10
SLOTS = 10
BAN = 20  # sessions without re-entry after any exit
COST_BPS = 20.0
CHECK_COST_BPS = 40.0
BIG_CONTRIB = 0.05
TOP_SHARE = 0.30
LAMBDA = 1.0
RANK_MIN = 0.02
FIRST_TEST, LAST_TEST = 2000, 2023
HALVES = {"2000-2011": (2000, 2011), "2012-2023": (2012, 2023)}
AUC_MIN = 0.55
YEAR_SHARE = 0.60
N_PLACEBO = 100
N_RANDOM = 100
PLACEBO_ROWS = 200_000  # each placebo fit uses this many training rows (computation limit)
SEED = 86
TARGETS = ("main", "b50", "spy")
MODELS = ("logit", "rank")
MAIN = ("logit", True, "main")  # model, with ATR%, target
N_TRIALS_BEFORE = 63_400  # Stages B-G logged trials (approx.), DECISIONS 85
N_NEW = 48
_C_ENTRIES = dict(sc.ENTRIES)
_C_CANON = dict(sc.ENTRY_CANON)
ENTRY_KEYS = [f"E{i}" for i in range(1, 10)] + list(ev.EVENTS)
FEATURES = ts.FEATURES + [f"is_{e}" for e in ENTRY_KEYS]
_W: dict = {}


def settings() -> list[tuple[str, int]]:
    return [(x, ki) for x in EXITS_H for ki in range(len(K_VALUES))]


def tag(x: str, ki: int) -> str:
    return f"{x}_k{K_VALUES[ki]:g}"


def log(m: str) -> None:
    ts.log(m)


# --------------------------------------------------------------------------- 1. H1 signal tables

def use_h1_menu(dry: bool = False) -> None:
    keys = ENTRY_KEYS if not dry else ["E1", "E9"]
    sc.ENTRIES.clear()
    for e in keys:
        key, vals = _C_ENTRIES[e] if e in _C_ENTRIES else ev.EVENTS[e]
        sc.ENTRIES[e] = (key, [vals[_C_CANON.get(e, 1)]])
        sc.ENTRY_CANON[e] = 0
    sc.EXITS.clear()
    sc.EXITS["H1"] = ("atr_stop", list(K_VALUES))
    sc.EXIT_EXTRA["H1"] = {"trailing_pct": H1_TRAIL}
    sc.EXIT_CANON["H1"] = 1
    sc.SAMPLES = {"full": SAMPLES["full"]}


def _h1_init(panel_dir: str, eligible: str, dry: bool):
    use_h1_menu(dry)
    sc._setup(load_panel_with_cash(Path(panel_dir)), pd.read_parquet(eligible))


# --------------------------------------------------------------------------- 2. signals, weights, labels

def table_path(dirs: dict, x: str, e: str) -> tuple[Path, int]:
    """(signal table, entry cell) of entry e under exit x."""
    if x == "H1":
        return dirs["h1"] / "signals" / "full" / f"{e}__H1_signals.parquet", 0
    root = dirs["v2"] if e.startswith("E") else dirs["events"]
    return root / "signals" / "full" / f"{e}__X9_signals.parquet", _C_CANON.get(e, 1)


def pooled(dirs: dict, x: str, ki: int, keys: list[str]) -> pd.DataFrame:
    parts = []
    for e in keys:
        f, xi = table_path(dirs, x, e)
        s = pd.read_parquet(f)
        s = s[(s["xi"] == xi) & (s["yi"] == ki) & (s["exit_reason"] != CANCELLED)]
        parts.append(s.assign(entry=e))
    s = pd.concat(parts, ignore_index=True)
    flags = pd.crosstab([s["symbol"], s["signal_date"]], s["entry"]).clip(upper=1)
    flags.columns = [f"is_{c}" for c in flags.columns]
    s = s.drop_duplicates(["symbol", "signal_date"]).set_index(["symbol", "signal_date"]).join(flags).reset_index()
    for e in keys:
        if f"is_{e}" not in s:
            s[f"is_{e}"] = 0
    return s


def weights(px: ts.Prices, m: pd.DataFrame, k: float) -> pd.DataFrame:
    """1%-rule target weight, contribution and labels of each signal."""
    atr = px.ATR[m["si"].to_numpy(), m["sj"].to_numpy()].astype(float)
    entry = px.O[m["ei"].to_numpy(), m["sj"].to_numpy()].astype(float)
    stop_pct = k * atr / entry
    raw = np.where(stop_pct > 0, RISK / stop_pct, np.nan)
    w = np.minimum(raw, CAP)
    out = pd.DataFrame({"stop_pct": stop_pct, "w": w, "capped": raw >= CAP})
    out["contrib"] = w * m["ret"].to_numpy()
    out["y_main"] = (out["contrib"] >= BIG_CONTRIB).astype(float)
    out["y_b50"] = (m["ret"].to_numpy() >= 0.5).astype(float)
    out["y_spy"] = (m["x_spy"].to_numpy() > 0).astype(float)
    bad = ~np.isfinite(w)
    out.loc[bad, ["y_main", "y_b50", "y_spy"]] = np.nan
    return out


def build_setting(px: ts.Prices, dirs: dict, x: str, ki: int, keys: list[str]) -> pd.DataFrame:
    s = pooled(dirs, x, ki, keys)
    m = px.measures(s, SAMPLES["full"][1])
    s = pd.concat([s[["symbol", "signal_date", "entry_date", "exit_signal_date", "return", "holding_days", "closed", "exit_reason"]
                     + [f"is_{e}" for e in keys]].reset_index(drop=True),
                   m[["si", "ei", "sj", "sale", "x_spy", "x_ew"]]], axis=1)
    return pd.concat([s, weights(px, m.assign(ret=s["return"].to_numpy()), K_VALUES[ki])], axis=1)


# --------------------------------------------------------------------------- 3. models

def to_pct(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile of x within the finite training values (0-1); NaN -> 0.5."""
    t = np.sort(train[np.isfinite(train)])
    if not len(t):
        return np.full(len(x), 0.5)
    out = np.searchsorted(t, x, side="right") / len(t)
    return np.where(np.isfinite(x), out, 0.5)


def fit_logit(X: np.ndarray, y: np.ndarray, lam: float = LAMBDA, iters: int = 50) -> np.ndarray:
    """L2 logistic regression by Newton steps; X already centred; intercept not penalised."""
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    pen = np.full(p + 1, lam)
    pen[0] = 0.0
    for _ in range(iters):
        z = np.clip(Xb @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (y - mu) - pen * beta
        H = (Xb * (mu * (1 - mu))[:, None]).T @ Xb + np.diag(pen)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def hist_auc(x: np.ndarray, y: np.ndarray, bins: int = 1000) -> float:
    """AUC of x (0-1) for binary y from a 1,000-bin histogram (ties within a bin count half)."""
    b = np.minimum((x * bins).astype(int), bins - 1)
    pos = np.bincount(b[y == 1], minlength=bins).astype(float)
    neg = np.bincount(b[y == 0], minlength=bins).astype(float)
    P, N = pos.sum(), neg.sum()
    if not P or not N:
        return 0.5
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    return float((pos * (below + 0.5 * neg)).sum() / (P * N))


@dataclass
class Fitted:
    kind: str
    cols: list
    coef: np.ndarray

    def score(self, Xc: np.ndarray) -> np.ndarray:
        if self.kind == "logit":
            return self.coef[0] + Xc @ self.coef[1:]
        return Xc @ self.coef


def fit(kind: str, Xc: np.ndarray, y: np.ndarray, cols: list) -> Fitted:
    if kind == "logit":
        return Fitted(kind, cols, fit_logit(Xc, y))
    a = np.array([hist_auc(Xc[:, j] + 0.5, y) - 0.5 for j in range(Xc.shape[1])])
    return Fitted(kind, cols, np.where(np.abs(a) >= RANK_MIN, a, 0.0))


def design(s: pd.DataFrame, train: np.ndarray, rows: np.ndarray, cols: list) -> np.ndarray:
    """Centred training-percentile matrix of `rows` (entry flags stay 0/1 - 0.5)."""
    X = np.empty((len(rows), len(cols)))
    for j, c in enumerate(cols):
        v = s[c].to_numpy(dtype=float)
        X[:, j] = (v[rows] if c.startswith("is_") else to_pct(v[train], v[rows])) - 0.5
    return X


def walk_forward(s: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator, n_placebo: int, only=None):
    """Scores of every test-year signal for each variant, training thresholds, and placebo AUCs.
    `only` (a set of (model, with ATR%, target)) limits the variants fitted (e.g. {MAIN})."""
    sd = pd.DatetimeIndex(s["signal_date"])
    year = sd.year.to_numpy()
    sale_date = dates[np.minimum(s["sale"].to_numpy(), len(dates) - 1)]
    closed = s["closed"].to_numpy(dtype=bool)
    sym = s["symbol"].astype(str).to_numpy()
    scores = {}
    thr = []
    placebo = []
    for Y in range(FIRST_TEST, LAST_TEST + 1):
        cut = pd.Timestamp(f"{Y}-01-01")
        train = np.flatnonzero(closed & (sale_date < cut) & (sd < cut))
        test = np.flatnonzero(year == Y)
        if not len(test) or not len(train):
            continue
        for with_atr in (True, False):
            cols = [c for c in FEATURES if with_atr or c != "atr_pct"]
            Xtr = design(s, train, train, cols)
            Xte = design(s, train, test, cols)
            for target in TARGETS:
                y = s[f"y_{target}"].to_numpy()[train]
                ok = np.isfinite(y)
                for kind in MODELS:
                    if only is not None and (kind, with_atr, target) not in only:
                        continue
                    f = fit(kind, Xtr[ok], y[ok], cols)
                    key = (kind, with_atr, target)
                    scores.setdefault(key, np.full(len(s), np.nan))[test] = f.score(Xte)
                    thr.append({"year": Y, "model": kind, "atr": with_atr, "target": target,
                                "threshold": float(np.quantile(f.score(Xtr[ok]), 1 - TOP_SHARE))})
                    if (kind, with_atr, target) == MAIN and n_placebo:
                        placebo.extend(_placebo_year(Xtr[ok], y[ok], sym[train][ok], Xte, test, Y, cols, rng, n_placebo))
    return scores, pd.DataFrame(thr), pd.DataFrame(placebo)


def _placebo_year(Xtr, y, sym, Xte, test, Y, cols, rng, n):
    """Training labels shuffled by stock blocks: every signal of stock A takes a label drawn from
    the signals of the stock A is mapped to (a permutation of the training stocks)."""
    out = []
    names, inv = np.unique(sym, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(names)))
    sizes = np.bincount(inv, minlength=len(names))
    for b in range(n):
        perm = rng.permutation(len(names))
        src = perm[inv]
        pick = starts[src] + (rng.random(len(y)) * sizes[src]).astype(int)
        yp = y[order[pick]]
        rows = rng.choice(len(yp), size=min(PLACEBO_ROWS, len(yp)), replace=False)
        f = Fitted("logit", cols, fit_logit(Xtr[rows], yp[rows]))
        out.append({"draw": b, "year": Y, "test": test, "score": f.score(Xte)})
    return out


# --------------------------------------------------------------------------- 4. accounts

def replay_risk(si, ei, sj, sale, closed, w, key, O, LC, cash_r, cost, slots=SLOTS, ban=BAN, log=None):
    """10-stock 1%-rule account on the window arrays: signals of day i are bought at the open of
    ei (their next session) in `key` order (larger first) while a slot and cash are free; target
    value = w x equity at that open (capped by cash); no re-entry for `ban` sessions after an exit.
    Returns (equity, trades: [j, entry_i, exit_i, ret, weight, row]). With `log` (a dict) it is
    filled per day: invested (stock share of equity at the close), positions, left (signals of the
    day left unbought because slots or cash ran out), idle (free slots and no buyable signal),
    banned (signals skipped by the re-entry rule) and held (symbols held at the close)."""
    n = O.shape[0]
    by_day: dict[int, list[int]] = {}
    for r in np.argsort(-key, kind="stable"):
        if 0 <= ei[r] < n:
            by_day.setdefault(int(ei[r]), []).append(int(r))
    sells: dict[int, list[int]] = {}
    pos: dict[int, dict] = {}
    last_exit: dict[int, int] = {}
    cash = 1.0
    eq = np.empty(n)
    trades = []
    for i in range(n):
        Oi = O[i]
        for j in sells.pop(i, ()):
            p = pos.get(j)
            if p is None or p["sale"] != i:
                continue
            px = Oi[j]
            if px != px:
                continue
            cash += p["shares"] * px * (1 - cost)
            trades.append((j, p["entry_i"], i, px * (1 - cost) / (p["entry_px"] * (1 + cost)) - 1.0, p["weight"], p["r"]))
            del pos[j]
            last_exit[j] = i
        left = banned = bought = 0
        if i in by_day and len(pos) < slots:
            held = sum(p["shares"] * (Oi[j] if Oi[j] == Oi[j] else LC[i - 1, j]) for j, p in pos.items())
            eq_now = cash + held
            cand = by_day[i]
            for q, r in enumerate(cand):
                if len(pos) >= slots or cash <= 1e-9:
                    left = len(cand) - q
                    break
                j = int(sj[r])
                if j in pos or (j in last_exit and i - last_exit[j] < ban):
                    banned += j not in pos
                    continue
                px = Oi[j]
                if px != px or px <= 0 or not np.isfinite(w[r]):
                    continue
                alloc = min(w[r] * eq_now, cash)
                pos[j] = {"shares": alloc / (px * (1 + cost)), "entry_px": float(px), "entry_i": i,
                          "sale": int(sale[r]) if closed[r] else -1, "weight": alloc / eq_now, "r": int(r)}
                cash -= alloc
                bought += 1
                if closed[r] and sale[r] < n:
                    sells.setdefault(int(sale[r]), []).append(j)
        elif i in by_day:
            left = len(by_day[i])
        cash *= 1.0 + cash_r[i]
        eq[i] = cash + sum(p["shares"] * LC[i, j] for j, p in pos.items())
        if log is not None:
            log.setdefault("invested", np.zeros(n))[i] = 1.0 - cash / eq[i] if eq[i] > 0 else 0.0
            log.setdefault("positions", np.zeros(n, dtype=int))[i] = len(pos)
            log.setdefault("left", np.zeros(n, dtype=int))[i] = left
            log.setdefault("idle", np.zeros(n, dtype=bool))[i] = len(pos) < slots and bought == 0 and left == 0
            log.setdefault("banned", np.zeros(n, dtype=int))[i] = banned
            log.setdefault("held", [None] * n)[i] = list(pos)
    for j, p in pos.items():  # still open at the end: valued at the last close
        trades.append((j, p["entry_i"], n - 1, LC[n - 1, j] / (p["entry_px"] * (1 + cost)) - 1.0, p["weight"], p["r"]))
    return eq, pd.DataFrame(trades, columns=["j", "entry_i", "exit_i", "ret", "weight", "row"])


def calendar_weighted(ei, sale, closed, sj, w, O, C, cash, cost):
    """No stock limit: every position at its own weight w; on days the weights add up to more than
    100% they are scaled down to 100%; the rest earns T-bills. Returns (equity, gross exposure)."""
    n = len(cash)
    if not len(ei):
        return np.cumprod(1.0 + cash), np.zeros(n)
    cols, local = np.unique(sj, return_inverse=True)
    o, c = O[:, cols].astype(float), C[:, cols].astype(float)
    lc = pd.DataFrame(c).ffill().to_numpy()
    prev = np.vstack([np.full((1, len(cols)), np.nan), lc[:-1]])
    cc = np.where(np.isfinite(c) & np.isfinite(prev), c / prev - 1.0, 0.0)
    end = np.where(closed, sale, n)
    num = np.zeros(n)
    gross = np.zeros(n)
    for k in np.unique(local):  # per symbol: held days e+1 .. end-1 earn close-to-close at weight w
        m = local == k
        held = np.zeros(n + 1)
        np.add.at(held, ei[m] + 1, w[m])
        np.add.at(held, np.minimum(end[m], n), -w[m])
        h = np.cumsum(held)[:n]
        num += h * cc[:, k]
        gross += h
    r_in = np.where(np.isfinite(c[ei, local]) & (o[ei, local] > 0), c[ei, local] / (o[ei, local] * (1 + cost)) - 1.0, 0.0)
    np.add.at(num, ei, w * r_in)
    np.add.at(gross, ei, w)
    ex = closed & (sale < n)
    si_, sl = sale[ex], local[ex]
    r_out = np.where(np.isfinite(o[si_, sl]) & np.isfinite(prev[si_, sl]), o[si_, sl] * (1 - cost) / prev[si_, sl] - 1.0, 0.0)
    np.add.at(num, si_, w[ex] * r_out)
    np.add.at(gross, si_, w[ex])
    scale = np.maximum(gross, 1.0)
    day = num / scale + cash * (1.0 - np.minimum(gross, 1.0))
    return np.cumprod(1.0 + day), gross


def curve_stats(eq: pd.Series) -> dict:
    v = eq.to_numpy(dtype=float)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {"cagr": float(v[-1] / v[0]) ** (1 / years) - 1 if years > 0 else np.nan,
            "mdd": float((v / np.maximum.accumulate(v) - 1).min())}


def period_stats(eq: pd.Series) -> dict:
    out = {}
    for name, (a, b) in {"all": (FIRST_TEST, LAST_TEST), **HALVES, "2017-2023": (2017, 2023)}.items():
        e = eq.loc[f"{a}":f"{b}"]
        st = curve_stats(e / e.iloc[0])
        out[f"cagr_{name}"], out[f"mdd_{name}"] = st["cagr"], st["mdd"]
    return out


def yearly(eq: pd.Series) -> pd.Series:
    y = eq.groupby(eq.index.year).last()
    first = eq.iloc[0]
    return y / pd.concat([pd.Series([first], index=[y.index[0] - 1]), y.iloc[:-1]]).to_numpy() - 1


# --------------------------------------------------------------------------- per-setting worker

def _init(panel_dir: str, eligible: str, out: str):
    panel = load_panel_with_cash(Path(panel_dir))
    elig = pd.read_parquet(eligible)
    _W["px"] = ts.Prices(panel, elig)
    idx = panel.close.index
    win = (idx >= pd.Timestamp(f"{FIRST_TEST}-01-01")) & (idx <= pd.Timestamp(SAMPLES["full"][1]))
    _W["w0"] = int(np.flatnonzero(win)[0])
    _W["wdates"] = idx[win]
    _W["cash"] = panel.cash.reindex(idx[win]).fillna(0.0).to_numpy()
    _W["dv"] = panel.close.mul(panel.volume).rolling(20, min_periods=10).mean().to_numpy(dtype="float32")
    _W["out"] = Path(out)


def run_setting(job):
    x, ki, n_placebo, n_random = job
    px: ts.Prices = _W["px"]
    out = _W["out"] / "settings" / tag(x, ki)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(_W["out"] / "work" / f"{tag(x, ki)}.parquet")
    rng = np.random.default_rng(SEED + 10 * ki + (0 if x == "H1" else 5))
    t0 = time.time()
    scores, thr, plc = walk_forward(s, px.dates, rng, n_placebo)
    thr.to_csv(out / "thresholds.csv", index=False)
    pd.DataFrame({f"{k}_{'atr' if a else 'noatr'}_{t}": v for (k, a, t), v in scores.items()}).to_parquet(
        _W["out"] / "work" / f"{tag(x, ki)}_scores.parquet")  # per signal: temp folder only
    yr = pd.DatetimeIndex(s["signal_date"]).year.to_numpy()
    test = (yr >= FIRST_TEST) & (yr <= LAST_TEST)
    # selection power (AUC) by half, every variant and label
    aucs = []
    for (kind, atr, target), sc_ in scores.items():
        for target_eval in TARGETS:
            y = s[f"y_{target_eval}"].to_numpy()
            row = {"model": kind, "atr": atr, "target": target, "eval": target_eval}
            for h, (a, b) in HALVES.items():
                m = test & (yr >= a) & (yr <= b) & np.isfinite(sc_) & np.isfinite(y)
                row[f"auc_{h}"] = ts.auc(sc_[m], y[m])
            aucs.append(row)
    aucs = pd.DataFrame(aucs)
    aucs.to_csv(out / "auc.csv", index=False)
    if len(plc):
        main_y = s["y_main"].to_numpy()
        rows = []
        for b, g in plc.groupby("draw"):
            sc_p = np.full(len(s), np.nan)
            for t, v in zip(g["test"], g["score"]):
                sc_p[t] = v
            r = {"draw": b}
            for h, (a, bb) in HALVES.items():
                m = test & (yr >= a) & (yr <= bb) & np.isfinite(sc_p) & np.isfinite(main_y)
                r[f"auc_{h}"] = ts.auc(sc_p[m], main_y[m])
            rows.append(r)
        pd.DataFrame(rows).to_csv(out / "placebo_auc.csv", index=False)
    log(f"  {tag(x, ki)} 걷기식 · 위약 {time.time() - t0:.0f}s")
    # accounts on the test window
    w0, n = _W["w0"], len(_W["wdates"])
    O = px.O[w0:w0 + n].astype(float)
    LC = px.LC[w0:w0 + n].astype(float)
    C = LC
    sel = np.flatnonzero(test & (s["ei"].to_numpy() >= w0) & (s["ei"].to_numpy() < w0 + n))
    ss = s.iloc[sel].reset_index(drop=True)
    ei = ss["ei"].to_numpy() - w0
    sale = ss["sale"].to_numpy() - w0
    closed = ss["closed"].to_numpy(dtype=bool) & (sale < n)
    sj = ss["sj"].to_numpy()
    w = ss["w"].to_numpy()
    cost = COST_BPS / 10_000
    accounts, curves, trade_rows = [], {}, []

    def account(name, key, cost_=cost):
        eq, tr = replay_risk(ei - 1, ei, sj, sale, closed, w, key, O, LC, _W["cash"], cost_)
        e = pd.Series(eq, index=_W["wdates"])
        st = {"account": name, "trades": len(tr), **period_stats(e)}
        tr = tr.assign(contrib=tr["weight"] * tr["ret"], capped=ss["capped"].to_numpy()[tr["row"].to_numpy()],
                       x_spy=ss["x_spy"].to_numpy()[tr["row"].to_numpy()])
        return e, st, tr

    dv = _W["dv"][ss["si"].to_numpy(), sj]
    e, st, tr = account("dollar_volume", np.nan_to_num(dv, nan=-1.0))
    accounts.append(st)
    curves["dollar_volume"] = e
    trade_rows.append(tr.assign(account="dollar_volume"))
    rand_years = []
    for b in range(n_random):
        e, st, _ = account(f"random_{b}", rng.random(len(ss)))
        accounts.append(st)
        rand_years.append(yearly(e).rename(b))
    for (kind, atr, target), sc_ in scores.items():
        if target != "main":
            continue
        name = f"{kind}_{'atr' if atr else 'noatr'}"
        e, st, tr = account(name, np.nan_to_num(sc_[sel], nan=-1e18))
        accounts.append(st)
        curves[name] = e
        trade_rows.append(tr.assign(account=name))
    acc = pd.DataFrame(accounts)
    acc.to_csv(out / "accounts.csv", index=False)
    pd.DataFrame(curves).astype("float32").to_parquet(out / "curves.parquet")
    ry = pd.concat(rand_years, axis=1)
    yearly_tab = pd.DataFrame({k: yearly(v) for k, v in curves.items()}).assign(random_median=ry.median(axis=1))
    yearly_tab.to_csv(out / "yearly.csv")
    trades = pd.concat(trade_rows, ignore_index=True)
    trades.groupby(["account", "capped"]).agg(trades=("ret", "size"), mean_contrib=("contrib", "mean"), win=("ret", lambda r: float((r > 0).mean())),
                                               big=("contrib", lambda c: float((c >= BIG_CONTRIB).mean())),
                                               mean_x_spy=("x_spy", "mean")).reset_index().to_csv(out / "trades_capped.csv", index=False)
    # no stock limit: every signal at its own weight; model as a top-30% filter
    unl, cal = [], {}
    syr = pd.DatetimeIndex(ss["signal_date"]).year.to_numpy()
    for name, sc_ in (("all", None),) + tuple((f"top30_{k}_{'atr' if a else 'noatr'}", v[sel]) for (k, a, t), v in scores.items() if t == "main"):
        if sc_ is None:
            keep = np.ones(len(ss), bool)
        else:
            th = thr[(thr["model"] == name.split("_")[1]) & (thr["atr"] == name.endswith("_atr")) & (thr["target"] == "main")]
            th = th.set_index("year")["threshold"]
            keep = np.isfinite(sc_) & (sc_ >= th.reindex(syr).to_numpy())
        eq, gross = calendar_weighted(ei[keep], sale[keep], closed[keep], sj[keep], w[keep], O, C, _W["cash"], cost)
        e = pd.Series(eq, index=_W["wdates"])
        cal[name] = e
        g = ss[keep]
        row = {"view": name, "signals": int(keep.sum()), **period_stats(e),
               "gross_median": float(np.median(gross[gross > 0])) if (gross > 0).any() else 0.0,
               "gross_over_100": float((gross > 1).mean())}
        for h, (a, b) in HALVES.items():
            m = (syr[keep] >= a) & (syr[keep] <= b)
            row[f"mean_contrib_{h}"] = float(g["contrib"].to_numpy()[m].mean()) if m.any() else np.nan
        for cap in (False, True):
            gg = g[g["capped"] == cap]
            sfx = "capped" if cap else "uncapped"
            row.update({f"{sfx}_n": len(gg), f"{sfx}_mean_contrib": float(gg["contrib"].mean()) if len(gg) else np.nan,
                        f"{sfx}_win": float((gg["return"] > 0).mean()) if len(gg) else np.nan,
                        f"{sfx}_big": float((gg["contrib"] >= BIG_CONTRIB).mean()) if len(gg) else np.nan})
        row["mean_x_spy"] = float(g["x_spy"].mean())
        unl.append(row)
    pd.DataFrame(unl).to_csv(out / "unlimited.csv", index=False)
    pd.DataFrame(cal).astype("float32").to_parquet(out / "calendar.parquet")
    log(f"  {tag(x, ki)} 완료 {time.time() - t0:.0f}s")
    return x, ki


def check_cost(out: Path, x: str, ki: int) -> pd.Series:
    """The main model's 10-stock account of one setting replayed at CHECK_COST_BPS (needs _init)."""
    px = _W["px"]
    s = pd.read_parquet(out / "work" / f"{tag(x, ki)}.parquet")
    sc_ = pd.read_parquet(out / "work" / f"{tag(x, ki)}_scores.parquet")["logit_atr_main"].to_numpy()
    w0, n = _W["w0"], len(_W["wdates"])
    yr = pd.DatetimeIndex(s["signal_date"]).year.to_numpy()
    sel = np.flatnonzero((yr >= FIRST_TEST) & (yr <= LAST_TEST) & (s["ei"].to_numpy() >= w0) & (s["ei"].to_numpy() < w0 + n))
    ss = s.iloc[sel]
    ei = ss["ei"].to_numpy() - w0
    sale = ss["sale"].to_numpy() - w0
    closed = ss["closed"].to_numpy(dtype=bool) & (sale < n)
    eq, _ = replay_risk(ei - 1, ei, ss["sj"].to_numpy(), sale, closed, ss["w"].to_numpy(), np.nan_to_num(sc_[sel], nan=-1e18),
                        px.O[w0:w0 + n].astype(float), px.LC[w0:w0 + n].astype(float), _W["cash"], CHECK_COST_BPS / 10_000)
    return pd.Series(eq, index=_W["wdates"])


def write_figures(out: Path, v: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    made = []
    base = pd.read_csv(out / "baselines.csv").set_index("baseline") if (out / "baselines.csv").exists() else None
    spy_all = float(base.loc["SPY", "cagr_all"]) if base is not None else None
    labs = [f"{r.exit} k{r.k:g}" for r in v.itertuples()]
    made.append(figures.paired_bars(labs, {"2000–2011": v["auc_2000-2011"].to_numpy() - 0.5, "2012–2023": v["auc_2012-2023"].to_numpy() - 0.5},
                                    "고르는 힘: 평가 구간 AUC − 0.5 (주 모형, 기준 +0.05)", "AUC − 0.5", out / "figures" / "auc.png",
                                    pct=False, ref=AUC_MIN - 0.5, ref_label="기준 +0.05", legend_out=True))
    made.append(figures.paired_bars(labs, {"모형 순": v["model_cagr_all"].to_numpy(), "거래대금 순": v["dv_cagr_all"].to_numpy(),
                                           "무작위 중앙값": v["random_median_cagr_all"].to_numpy()},
                                    "10종목 1% 룰 계좌 CAGR, 2000–2023 (점선 = SPY)", "CAGR (%)", out / "figures" / "account10_cagr.png",
                                    legend_out=True, ref=spy_all, ref_label="SPY"))
    for h in HALVES:
        series = {"모형 순": [], "거래대금 순": [], "무작위 중앙값": []}
        for x, ki in settings():
            acc = pd.read_csv(out / "settings" / tag(x, ki) / "accounts.csv").set_index("account")
            series["모형 순"].append(acc.loc["logit_atr", f"cagr_{h}"])
            series["거래대금 순"].append(acc.loc["dollar_volume", f"cagr_{h}"])
            series["무작위 중앙값"].append(acc[acc.index.str.startswith("random_")][f"cagr_{h}"].median())
        spy_h = float(base.loc["SPY", f"cagr_{h}"]) if base is not None else None
        made.append(figures.paired_bars(labs, {k: np.asarray(vv) for k, vv in series.items()},
                                        f"10종목 계좌 CAGR, {h.replace('-', '–')} (점선 = SPY)", "CAGR (%)",
                                        out / "figures" / f"account10_{h.replace('-', '_')}.png", legend_out=True, ref=spy_h,
                                        ref_label="SPY"))
    un = pd.concat([pd.read_csv(out / "settings" / tag(x, ki) / "unlimited.csv").assign(setting=f"{x} k{K_VALUES[ki]:g}")
                    for x, ki in settings()])
    m = un[un["view"].isin(["all", "top30_logit_atr"])].pivot(index="setting", columns="view", values="cagr_all").reindex(labs)
    made.append(figures.paired_bars(labs, {"전부": m["all"].to_numpy(), "상위 30%": m["top30_logit_atr"].to_numpy()},
                                    "종목 수 제한 없음(달력 시간 포트폴리오) CAGR, 2000–2023", "CAGR (%)", out / "figures" / "unlimited_cagr.png",
                                    legend_out=True))
    return made


# --------------------------------------------------------------------------- verdicts and main

def baselines(px: ts.Prices, wdates: pd.DatetimeIndex, w0: int, cash: np.ndarray) -> pd.DataFrame:
    n = len(wdates)
    spy = pd.Series(px.spy_c[w0:w0 + n], index=wdates)
    r = spy.pct_change(fill_method=None).fillna(0.0)
    full_spy = pd.Series(px.spy_c, index=px.dates)
    on = (full_spy > full_spy.rolling(200).mean()).shift(1, fill_value=False).to_numpy()[w0:w0 + n]
    ma = pd.Series(np.cumprod(1 + np.where(on, r.to_numpy(), cash)), index=wdates)
    ew = pd.Series(px.ew[w0:w0 + n] / px.ew[w0], index=wdates)
    return pd.DataFrame([{"baseline": k, **period_stats(v)} for k, v in
                         (("SPY", spy / spy.iloc[0]), ("SPY 200일선", ma), ("동일비중 유니버스", ew))])


def verdicts(out: Path, base: pd.DataFrame) -> pd.DataFrame:
    spy = base.set_index("baseline").loc["SPY"]
    rows = []
    for x, ki in settings():
        d = out / "settings" / tag(x, ki)
        auc = pd.read_csv(d / "auc.csv")
        a = auc[(auc["model"] == MAIN[0]) & (auc["atr"] == MAIN[1]) & (auc["target"] == MAIN[2]) & (auc["eval"] == "main")].iloc[0]
        pl = pd.read_csv(d / "placebo_auc.csv") if (d / "placebo_auc.csv").exists() else None
        v1 = all(a[f"auc_{h}"] >= AUC_MIN and (pl is None or a[f"auc_{h}"] > pl[f"auc_{h}"].quantile(0.95)) for h in HALVES)
        acc = pd.read_csv(d / "accounts.csv").set_index("account")
        rnd = acc[acc.index.str.startswith("random_")]
        model = acc.loc["logit_atr"]
        dv = acc.loc["dollar_volume"]
        yt = pd.read_csv(d / "yearly.csv", index_col=0)
        share_dv = float((yt["logit_atr"] > yt["dollar_volume"]).mean())
        share_rd = float((yt["logit_atr"] > yt["random_median"]).mean())
        v2a = all(model[f"cagr_{h}"] > dv[f"cagr_{h}"] and model[f"cagr_{h}"] > rnd[f"cagr_{h}"].median() for h in HALVES) \
            and share_dv >= YEAR_SHARE and share_rd >= YEAR_SHARE
        un = pd.read_csv(d / "unlimited.csv").set_index("view")
        v2b = all(un.loc["top30_logit_atr", f"mean_contrib_{h}"] > un.loc["all", f"mean_contrib_{h}"] for h in HALVES)
        passes = all(model[f"cagr_{p}"] > spy[f"cagr_{p}"] and abs(model[f"mdd_{p}"]) < abs(spy[f"mdd_{p}"]) for p in ("all", "2017-2023")) \
            and model["trades"] >= 30
        rows.append({"exit": x, "k": K_VALUES[ki], "auc_2000-2011": a["auc_2000-2011"], "auc_2012-2023": a["auc_2012-2023"],
                     "placebo95_2000-2011": pl["auc_2000-2011"].quantile(0.95) if pl is not None else np.nan,
                     "placebo95_2012-2023": pl["auc_2012-2023"].quantile(0.95) if pl is not None else np.nan,
                     "v1_select": v1, "years_beat_dv": share_dv, "years_beat_random": share_rd, "v2_account10": v2a,
                     "v2_unlimited": v2b, "market_pass": passes,
                     **{f"model_{k}": model[k] for k in ("cagr_all", "mdd_all", "cagr_2017-2023", "mdd_2017-2023", "trades")},
                     "dv_cagr_all": dv["cagr_all"], "random_median_cagr_all": rnd["cagr_all"].median()})
    v = pd.DataFrame(rows)
    v["neighbours_pass"] = v.groupby("exit")["market_pass"].transform("sum")
    v["candidate_20bp"] = v["market_pass"] & (v["neighbours_pass"] >= 2)
    return v


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--v2", required=True, help="stagec_signals_v2 (E × X9 신호표)")
    ap.add_argument("--events", required=True, help="G-2 결과 폴더 (V × X9 신호표)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="E1·E9만, k 2만, 위약 3번, 무작위 3번")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["E1", "E9"] if a.dry else ENTRY_KEYS
    n_placebo, n_random = (3, 3) if a.dry else (N_PLACEBO, N_RANDOM)
    # 1. H1 tables
    use_h1_menu(a.dry)
    with ProcessPoolExecutor(max_workers=max(1, min(a.workers, 4)), initializer=_h1_init, initargs=(a.panel, a.eligible, a.dry)) as ex:
        sc.run_signal_tables(ex, out / "h1", log)
    # 2. per-setting tables and features (temp folder only)
    panel = load_panel_with_cash(Path(a.panel))
    eligible = pd.read_parquet(a.eligible)
    panel.eligible = eligible
    px = ts.Prices(panel, eligible)
    dirs = {"h1": out / "h1", "v2": Path(a.v2), "events": Path(a.events)}
    work = out / "work"
    work.mkdir(exist_ok=True)
    todo = settings() if not a.dry else [("H1", 1), ("H9", 1)]
    if not all((work / f"{tag(x, ki)}.parquet").exists() for x, ki in todo):
        sets = {tag(x, ki): build_setting(px, dirs, x, ki, keys) for x, ki in todo}
        base = pd.concat([v[["si", "sj"]] for v in sets.values()]).drop_duplicates().reset_index(drop=True)
        log(f"신호(종목·날짜): {len(base):,}개, 특징 계산")
        ts.attach_features(panel, eligible, {"base": base})
        for t, v in sets.items():
            v = v.merge(base, on=["si", "sj"], how="left")
            for e in ENTRY_KEYS:
                if f"is_{e}" not in v:
                    v[f"is_{e}"] = 0
            v.to_parquet(work / f"{t}.parquet")
        del sets, base
    del panel
    # 3-4. per setting in parallel
    jobs = [(x, ki, n_placebo, n_random) for x, ki in todo if not (out / "settings" / tag(x, ki) / "unlimited.csv").exists()]
    log(f"설정 {len(jobs)}개: 걷기식 · 위약 · 계좌")
    with ProcessPoolExecutor(max_workers=max(1, min(a.workers, len(jobs) or 1)), initializer=_init,
                             initargs=(a.panel, a.eligible, str(out))) as ex:
        for x, ki in ex.map(run_setting, jobs):
            log(f"  끝: {tag(x, ki)}")
    _init(a.panel, a.eligible, str(out))
    base = baselines(_W["px"], _W["wdates"], _W["w0"], _W["cash"])
    base.to_csv(out / "baselines.csv", index=False)
    if a.dry:
        log("점검 실행 완료")
        return
    v = verdicts(out, base)
    v["candidate_40bp"] = False
    v["dsr_raw"] = v["dsr_new"] = np.nan
    if v["candidate_20bp"].any():
        spy = base.set_index("baseline").loc["SPY"]
        srs = []
        for x, ki in settings():
            cur = pd.read_parquet(out / "settings" / tag(x, ki) / "curves.parquet")
            r = cur.astype(float).pct_change(fill_method=None)
            srs.extend((r.mean() / r.std()).dropna().tolist())
        sr_var = float(np.var(srs))
        for i in np.flatnonzero(v["candidate_20bp"].to_numpy()):
            x, k = v.loc[i, "exit"], v.loc[i, "k"]
            eq = check_cost(out, x, K_VALUES.index(k))
            st = period_stats(eq)
            v.loc[i, "candidate_40bp"] = all(st[f"cagr_{p}"] > spy[f"cagr_{p}"] and abs(st[f"mdd_{p}"]) < abs(spy[f"mdd_{p}"])
                                             for p in ("all", "2017-2023"))
            main_eq = pd.read_parquet(out / "settings" / tag(x, K_VALUES.index(k)) / "curves.parquet")["logit_atr"].astype(float)
            v.loc[i, "dsr_raw"] = deflated_sharpe(main_eq, N_TRIALS_BEFORE + N_NEW, sr_var)
            v.loc[i, "dsr_new"] = deflated_sharpe(main_eq, N_NEW, sr_var)
    v.to_csv(out / "verdicts.csv", index=False)
    write_figures(out, v)
    log(f"완료 · 판정 1 통과 {int(v['v1_select'].sum())}/6 · 10종목 {int(v['v2_account10'].sum())}/6 · "
        f"제한 없음 {int(v['v2_unlimited'].sum())}/6 · 후보(20bp) {int(v['candidate_20bp'].sum())}")


if __name__ == "__main__":
    main(sys.argv[1:])
