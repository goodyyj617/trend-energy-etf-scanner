"""Canary signals on B2–B5 (allocation DECISIONS A40; design https://claude.ai/artifact/7AyPtvA6372TwXaSJFYxPq).

The weights stay those of B2–B5. A rule reads assets that are never bought (US ETFs in USD, FRED series)
or the held products' own KRW prices and gives a state each day; the state picks one of a few target
weights built from the base allocation. Trades follow the signal, not the calendar:

  - the state is read at the close of day t (FRED series one day later, ``lab.macro.on_dates``) and a change
    trades at the next close (t+1), as every decision in the engine;
  - the quarter-end rebalance of B2–B5 stays and brings the held state's target back;
  - with ``cooldown`` = 21 a signal trade blocks the next ones for 21 rows; when the block ends and the
    signal differs from what is held, it trades then.

Pieces:
  ``CANARY_RULES`` / ``HELD_RULES``  the 11 canaries (K1–K11) and the 3 held-product rules (D1–D3)
  ``Inputs`` + ``states(rule, inputs, param, dates)``  integer state per date
  ``tables(base, rule, action)``     target weights per state (K × columns, fractions)
  ``schedule(states, quarter_rows, cooldown)``  which rows decide and to which state
  ``DynamicSim``                     many weight paths at once, equal to ``aa_engine.simulate`` (tests)
  ``placebo_states``                 the same state series turned around the calendar (circular shift)
  ``equal_cost_static``              the static allocation that moves x %p for good, x matching a CAGR
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import aa_local as loc
from . import macro
from .aa_assets import CASH
from .aa_engine import TRADING_DAYS, decision_rows
from .data import Panel

COOLDOWN = 21  # trading days after a signal trade during which the signal is not followed (the user's "1개월")
MONTH = {1: 21, 3: 63, 6: 126, 9: 189, 12: 252, 15: 315}  # months → rows, for the daily form of monthly rules

NASDAQ = loc.NASDAQ
STAPLES, HEALTH = "453630", "453640"
DEFENSIVE = (STAPLES, HEALTH)  # where action "defensive" sends nasdaq (B3 · B5 hold neither: half each)
OTHER_EQUITY = ("251350", loc.KODEX200)  # never switched

# --------------------------------------------------------------------------- canary data (never bought)

SECTORS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
# older series that carry a late ETF back, newest first; each one is scaled to meet the next on its first day
CHAINS: dict[str, tuple[str, ...]] = {
    "VWO": ("VWO", "VEIEX"),  # Vanguard Emerging Markets Index (1994)
    "BND": ("BND", "AGG", "VBMFX"),  # Vanguard Total Bond Market Index (1986)
    "AGG": ("AGG", "VBMFX"),
    "EEM": ("EEM", "VEIEX"),
    # Fidelity International Index (EAFE, 1997; Yahoo lists it as FSPSX, formerly FSIIX), then Vanguard Total Intl
    "EFA": ("EFA", "FSPSX", "FSIIX", "VGTSX"),
    "TIP": ("TIP", "VIPSX"),  # Vanguard Inflation-Protected Securities (2000-06)
}
CANARY_TICKERS = sorted({"SPY", *SECTORS, *[s for v in CHAINS.values() for s in v]})


def extend_back(series: pd.Series, older: pd.Series) -> pd.Series:
    """``series`` with its days before the first valid one filled by ``older``'s path, scaled to meet it."""
    first = series.first_valid_index()
    if first is None:
        return older.copy()
    o = older.ffill()
    if first not in o.index or pd.isna(o.loc[first]):
        return series
    return series.where(series.index >= first, o * (series.loc[first] / o.loc[first]))


def chained(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """USD closes with each ``CHAINS`` series carried back; returns (closes, one row per chain link)."""
    out = close.copy()
    rows = []
    for name, links in CHAINS.items():
        have = [s for s in links if s in close.columns and close[s].notna().any()]
        if not have:
            continue
        s = close[have[-1]].copy()
        for newer in reversed(have[:-1]):
            s = extend_back(close[newer], s)
        out[name] = s
        rows.append({"신호 자산": name, "이어 붙인 순서": " ← ".join(have), "시작": s.first_valid_index(),
                     "없음": ", ".join(x for x in links if x not in have)})
    return out, pd.DataFrame(rows)


@dataclass
class Inputs:
    """Everything a rule reads. ``usd`` = chained USD closes (canaries) on their own dates, ``usd_cash`` the
    USD cash return, ``fx`` KRW per USD, ``krw`` the KRW panel (held products), ``macro_df`` the FRED
    snapshot (None = ``lab.macro.data()``)."""
    usd: pd.DataFrame
    usd_cash: pd.Series
    fx: pd.Series
    krw: Panel
    macro_df: pd.DataFrame | None = None
    currency: str = "usd"  # "krw": price canaries converted at the same day's USD/KRW (the A40 currency check)

    def closes(self, cols) -> pd.DataFrame:
        px = self.usd[list(cols)]
        if self.currency == "krw":
            fx = self.fx.reindex(self.fx.index.union(px.index)).ffill().reindex(px.index)
            px = px.mul(fx, axis=0)
        return px

    def macro(self, sid: str, index: pd.DatetimeIndex, sma: int | None = None) -> tuple[pd.Series, pd.Series | None]:
        raw = (self.macro_df if self.macro_df is not None else macro.data())[sid].dropna()
        avg = raw.rolling(sma, min_periods=sma).mean() if sma else None
        return macro.on_dates(raw, index), (macro.on_dates(avg, index) if avg is not None else None)


# --------------------------------------------------------------------------- rules

@dataclass(frozen=True)
class Rule:
    key: str
    label: str  # short name for tables
    source: str  # who published it / where the definition comes from
    what: str  # computation and risk-off condition, in the user's words
    params: tuple  # (centre, neighbour, neighbour)
    kind: str = "canary"  # canary | held
    signal_start: str = ""  # first date the rule can read (filled in by the study from the data)
    actions: tuple[str, ...] = ("defensive", "safe")


ACTIONS = {"defensive": "나스닥 → 방어 주식", "safe": "나스닥 → 안전 묶음", "tilt_nd": "나스닥 ↔ 방어 주식 10 %p",
           "tilt_es": "주식 ↔ 안전 묶음 10 %p"}

CANARY_RULES: dict[str, Rule] = {r.key: r for r in [
    Rule("k1", "K1 DAA", "Keller & Keuning 2018 (DAA)", "VWO·BND의 13612W 모멘텀. 하나가 음수면 절반, 둘 다면 전부",
         ("13612W", "12m", "6m")),
    Rule("k2", "K2 VAA·BAA", "Keller & Keuning 2017 (VAA), Keller 2022 (BAA)", "SPY·EFA·EEM·AGG의 13612W 중 하나라도 음수",
         ("13612W", "12m", "6m")),
    Rule("k3", "K3 HAA", "Keller & Keuning 2023 (HAA)", "TIP의 13612U(1·3·6·12개월 평균)가 음수", ("13612U", "12m", "6m")),
    Rule("k4", "K4 S&P500 200일선", "Faber 2007 (10개월선의 일간판)", "SPY 종가가 N일 평균 아래", (200, 150, 250)),
    Rule("k5", "K5 절대 모멘텀", "Antonacci 2014", "SPY N개월 총수익이 단기국채 N개월 수익보다 낮음", (12, 9, 15)),
    Rule("k6", "K6 신용 스프레드", "추세 연구실 블록 credit_calm", "Baa − 10년 국채(BAA10Y)가 N관측일 평균 위", (200, 150, 250)),
    Rule("k7", "K7 장단기 금리차", "Estrella & Mishkin 1998", "10년 − 3개월 국채(T10Y3M)가 X %p 아래", (0.0, -0.25, 0.25)),
    Rule("k8", "K8 VIX", "추세 연구실 블록 vix_calm", "VIX(VIXCLS)가 X 이상", (25.0, 20.0, 30.0)),
    Rule("k9", "K9 섹터 폭", "Keller & Keuning 2016 (PAA)의 폭 개념", "미국 9개 섹터 ETF 중 200일선 아래가 N개 이상", (5, 4, 6)),
    Rule("k10", "K10 금융 불안 지수", "Kritzman & Li 2010", "SPY·EFA·EEM·AGG 일간 수익률의 마할라노비스 거리(직전 3년), 10일 평균이 직전 3년 Q분위 위",
         (0.80, 0.75, 0.85)),
    Rule("k11", "K11 원/달러 추세", "한국 신호", "원/달러(DEXKOUS)가 N일 평균 위(원화 약세)", (200, 150, 250)),
]}
HELD_RULES: dict[str, Rule] = {r.key: r for r in [
    Rule("d1", "D1 나스닥 자기 추세", "보유 상품", "나스닥100 원화 종가가 N일 평균 아래", (200, 150, 250), kind="held"),
    Rule("d2", "D2 나스닥 vs 방어 주식", "보유 상품", "두 묶음의 N개월 원화 수익률, 약한 쪽에서 강한 쪽으로 10 %p", (12, 9, 15),
         kind="held", actions=("tilt_nd",)),
    Rule("d3", "D3 주식 vs 안전 묶음", "보유 상품", "두 묶음의 N개월 원화 수익률, 약한 쪽에서 강한 쪽으로 10 %p", (12, 9, 15),
         kind="held", actions=("tilt_es",)),
]}
RULES = {**CANARY_RULES, **HELD_RULES}
CURRENCY_RULES = ("k1", "k3", "k4")  # also computed on KRW prices (the answer to "why USD?")
TURB_WINDOW, TURB_SMOOTH = 756, 10


def _ret(px: pd.DataFrame | pd.Series, n: int):
    return px / px.shift(n) - 1.0


def momentum(px: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Daily form of the Keller momenta on closes ``px`` (rows = trading days)."""
    r1, r3, r6, r12 = (_ret(px, MONTH[m]) for m in (1, 3, 6, 12))
    if kind == "13612W":
        return (12 * r1 + 4 * r3 + 2 * r6 + r12) / 4.0
    if kind == "13612U":
        return (r1 + r3 + r6 + r12) / 4.0
    if kind == "12m":
        return r12
    if kind == "6m":
        return r6
    raise ValueError(kind)


def sma(px, n: int):
    return px.rolling(int(n), min_periods=int(n)).mean()


def turbulence(rets: pd.DataFrame, window: int = TURB_WINDOW, smooth: int = TURB_SMOOTH) -> pd.Series:
    """Kritzman & Li's turbulence: (r_t − μ)ᵀ Σ⁻¹ (r_t − μ) with μ, Σ from the ``window`` rows before t,
    then its ``smooth``-day mean. NaN until a full window of complete rows exists."""
    R = rets.to_numpy(dtype=float)
    ok = np.isfinite(R).all(axis=1)
    d = np.full(len(R), np.nan)
    for t in range(window, len(R)):
        if not ok[t]:
            continue
        X = R[t - window:t][ok[t - window:t]]
        if len(X) < 0.9 * window:
            continue
        mu = X.mean(axis=0)
        inv = np.linalg.pinv(np.cov(X, rowvar=False))
        x = R[t] - mu
        d[t] = float(x @ inv @ x)
    return pd.Series(d, index=rets.index).rolling(smooth, min_periods=smooth).mean()


def _on(x: pd.Series | pd.DataFrame, dates: pd.DatetimeIndex):
    """A series on its own (US) dates, read on ``dates``: the last value on or before each date."""
    return x.reindex(x.index.union(dates)).ffill().reindex(dates)


def sleeve_index(close: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily-rebalanced index of a sleeve at fixed internal ratios (KRW closes)."""
    tot = sum(weights.values())
    r = sum(close[s].ffill().pct_change(fill_method=None) * (w / tot) for s, w in weights.items())
    return (1.0 + r.fillna(0.0)).cumprod().where(sum(close[s].notna() for s in weights) == len(weights))


def sleeves(base: dict[str, float]) -> dict[str, dict[str, float]]:
    """The user's sleeves of an allocation: N nasdaq, D defensive equity (half-half when none is held),
    O other equity, S everything else (bonds, gold, cash, dollar)."""
    d = {s: base[s] for s in DEFENSIVE if base.get(s, 0) > 0} or {s: 1.0 for s in DEFENSIVE}
    o = {s: base[s] for s in OTHER_EQUITY if base.get(s, 0) > 0}
    s = {k: v for k, v in base.items() if k != NASDAQ and k not in DEFENSIVE and k not in OTHER_EQUITY and v > 0}
    return {"N": {NASDAQ: base.get(NASDAQ, 0.0)}, "D": d, "O": o, "S": s}


def raw_states(rule: str, inp: Inputs, param, dates: pd.DatetimeIndex, base: dict[str, float] | None = None) -> pd.Series:
    """State of ``rule`` on ``dates`` as floats, NaN where its inputs do not exist yet (see ``states``)."""
    if rule in ("k1", "k2", "k3"):
        cols = {"k1": ("VWO", "BND"), "k2": ("SPY", "EFA", "EEM", "AGG"), "k3": ("TIP",)}[rule]
        mom = momentum(inp.closes(cols), param)
        bad = (mom < 0).sum(axis=1)
        s = bad if rule == "k1" else (bad > 0).astype(int)
        return _on(s.astype(float).where(mom.notna().all(axis=1)), dates)
    if rule in ("k4", "d1"):
        px = inp.closes(["SPY"])["SPY"] if rule == "k4" else inp.krw.close[NASDAQ].ffill()
        m = sma(px, param)
        return _on((px < m).astype(float).where(m.notna()), dates)
    if rule == "k5":
        n = MONTH[int(param)]
        px = inp.usd["SPY"]
        cash = (1.0 + inp.usd_cash.reindex(px.index).fillna(0.0)).cumprod()
        r, c = _ret(px, n), _ret(cash, n)
        return _on((r < c).astype(float).where(r.notna() & c.notna()), dates)
    if rule in ("k6", "k7", "k8"):
        sid = {"k6": "BAA10Y", "k7": "T10Y3M", "k8": "VIXCLS"}[rule]
        v, a = inp.macro(sid, dates, int(param) if rule == "k6" else None)
        if rule == "k6":
            return (v > a).astype(float).where(v.notna() & a.notna())
        return (v < param if rule == "k7" else v >= param).astype(float).where(v.notna())
    if rule == "k9":
        px = inp.closes(SECTORS)
        m = sma(px, 200)
        below = (px < m).sum(axis=1)
        return _on((below >= int(param)).astype(float).where(m.notna().all(axis=1)), dates)
    if rule == "k10":
        rets = inp.usd[["SPY", "EFA", "EEM", "AGG"]].pct_change(fill_method=None)
        tb = turbulence(rets)
        thr = tb.rolling(TURB_WINDOW, min_periods=TURB_WINDOW).quantile(float(param)).shift(1)
        return _on((tb > thr).astype(float).where(tb.notna() & thr.notna()), dates)
    if rule == "k11":
        raw = inp.fx.dropna()
        v, a = macro.on_dates(raw, dates), macro.on_dates(sma(raw, int(param)), dates)
        return (v > a).astype(float).where(v.notna() & a.notna())
    if rule in ("d2", "d3"):
        if base is None:
            raise ValueError("D2·D3에는 기본 비중이 필요합니다.")
        sl = sleeves(base)
        close = inp.krw.close
        if rule == "d2":
            a, b = sleeve_index(close, sl["N"]), sleeve_index(close, sl["D"])
        else:
            a = sleeve_index(close, {**sl["N"], **({k: v for k, v in sl["D"].items() if base.get(k, 0) > 0}), **sl["O"]})
            b = sleeve_index(close, sl["S"])
        n = MONTH[int(param)]
        ra, rb = _ret(a, n), _ret(b, n)
        s = pd.Series(0.0, index=ra.index).mask(rb > ra, 1.0).mask(ra > rb, 2.0)
        return _on(s.where(ra.notna() & rb.notna()), dates)
    raise KeyError(rule)


def states(rule: str, inp: Inputs, param, dates: pd.DatetimeIndex, base: dict[str, float] | None = None) -> pd.Series:
    """Integer state of ``rule`` on ``dates`` (0 = base allocation). Canary states are 0/1, K1 0/1/2 (number
    of bad canaries), D2/D3 0 (equal), 1 (move toward D / S) or 2 (move toward N / equity). A rule with
    no data yet on a date (or a stale FRED value) is 0 there: the base allocation is held."""
    return raw_states(rule, inp, param, dates, base).fillna(0).astype(int)


def signal_start(raw: pd.Series) -> pd.Timestamp | None:
    """First date a rule could read its inputs (``raw_states`` not NaN)."""
    return raw.first_valid_index()


# --------------------------------------------------------------------------- target weights per state

def tables(base: dict[str, float], rule: str, action: str, columns: list[str], step: float = 10.0) -> np.ndarray:
    """Target weights (fractions over ``columns``, which end with CASH) for each state of ``rule``.

    defensive / safe: state f of nasdaq moves to the defensive pair (held ratio, else half each) or to the safe
    sleeve (its own ratios); K1 has states 0, 1, 2 = none, half, all. tilt_nd / tilt_es (D2 / D3): state 1 moves
    ``step`` %p toward D (from N) or toward S (from N + D + O, pro rata); state 2 the other way, never more than
    the weight that is there."""
    sl = sleeves(base)
    tot = sum(base.values())
    col = {c: i for i, c in enumerate(columns)}

    def vec(w: dict[str, float]) -> np.ndarray:
        v = np.zeros(len(columns))
        for k, x in w.items():
            v[col[k]] += x
        return v / tot

    b = vec(base)

    def ratio(d: dict[str, float]) -> np.ndarray:
        s = sum(d.values())
        return vec({k: x / s * tot for k, x in d.items()}) if s > 0 else np.zeros(len(columns))

    n = base.get(NASDAQ, 0.0) / tot
    e_n = vec({NASDAQ: tot})
    if action in ("defensive", "safe"):
        dest = ratio(sl["D"] if action == "defensive" else sl["S"])
        fracs = (0.0, 0.5, 1.0) if rule == "k1" else (0.0, 1.0)
        return np.vstack([b - f * n * e_n + f * n * dest for f in fracs])
    st = step / 100.0
    if action == "tilt_nd":
        d_held = {k: v for k, v in sl["D"].items() if base.get(k, 0) > 0}
        back = min(st, sum(d_held.values()) / tot)
        return np.vstack([b, b - min(st, n) * e_n + min(st, n) * ratio(sl["D"]),
                          b + back * e_n - back * ratio(d_held) if d_held else b])
    if action == "tilt_es":
        eq = {**sl["N"], **{k: v for k, v in sl["D"].items() if base.get(k, 0) > 0}, **sl["O"]}
        e_tot, s_tot = sum(eq.values()) / tot, sum(sl["S"].values()) / tot
        a, c = min(st, e_tot), min(st, s_tot)
        return np.vstack([b, b - a * ratio(eq) + a * ratio(sl["S"]), b + c * ratio(eq) - c * ratio(sl["S"])])
    raise KeyError(action)


# --------------------------------------------------------------------------- when to trade

@dataclass
class Schedule:
    decide: np.ndarray  # bool per row: a decision at this close (traded at the next row)
    choice: np.ndarray  # int per row: the state whose target is decided
    first: int  # state held from the first row
    signal_rows: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def n_signal(self) -> int:
        return len(self.signal_rows)

    def whipsaws(self, within: int = COOLDOWN) -> int:
        """Signal trades followed by another signal trade within ``within`` rows."""
        return int((np.diff(self.signal_rows) < within).sum()) if self.n_signal > 1 else 0


def schedule(st: np.ndarray, quarter_rows: np.ndarray, cooldown: int = 0) -> Schedule:
    """Decisions for a state series. A row decides when the signal leaves the held state (and, with a
    cooldown, at least ``cooldown`` rows have passed since the last signal decision) or when it is a
    quarter-end row; the decided target is the held state's. The last row never decides (no next close)."""
    st = np.asarray(st, dtype=int)
    T = len(st)
    quarter = np.zeros(T, dtype=bool)
    q = np.asarray(quarter_rows, dtype=int)
    quarter[q[(q >= 0) & (q < T)]] = True
    if cooldown <= 0:  # every change is followed, so the held state is the signal itself
        change = np.zeros(T, dtype=bool)
        change[1:T - 1] = st[1:T - 1] != st[:T - 2]
        decide = change | quarter
        decide[T - 1] = False
        choice = st.copy()
        if T > 1:
            choice[T - 1] = st[T - 2]
        return Schedule(decide, choice, int(st[0]), np.flatnonzero(change))
    decide = np.zeros(T, dtype=bool)
    choice = np.zeros(T, dtype=int)
    s, qr = st.tolist(), quarter.tolist()
    held, last = s[0], -(10 ** 9)
    rows = []
    for t in range(T - 1):
        if s[t] != held and t - last >= cooldown:
            held, last = s[t], t
            rows.append(t)
            decide[t] = True
        elif qr[t]:
            decide[t] = True
        choice[t] = held
    choice[T - 1] = held
    return Schedule(decide, choice, s[0], np.asarray(rows, dtype=int))


def quarter_rows(dates: pd.DatetimeIndex, offset: int = 0) -> np.ndarray:
    rows = decision_rows(dates, "Q") + int(offset)
    return rows[rows < len(dates) - 1]


def placebo_states(st: np.ndarray, n: int, seed: int, min_shift: int = TRADING_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """``n`` copies of ``st`` turned around the calendar by a random shift of at least ``min_shift`` rows:
    the same number and length of risk-off spells, at other times. Returns (states n × T, shifts)."""
    st = np.asarray(st)
    T = len(st)
    if T <= 2 * min_shift:
        raise ValueError("기간이 너무 짧습니다.")
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, T - min_shift, size=n)
    return np.vstack([np.roll(st, -int(k)) for k in shifts]), shifts


# --------------------------------------------------------------------------- simulator

class DynamicSim:
    """Daily simulation of many weight paths over the same dates; the same arithmetic as
    ``aa_engine.simulate`` with band 0 and no leverage (a decision at row t trades at the close of t+1,
    one-way turnover × cost, cash in the last column)."""

    def __init__(self, panel: Panel, syms: list[str], start: str, end: str, cost_bps: float = 10.0):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        close_all = panel.close[list(syms)]
        first_valid = close_all.apply(lambda s: s.first_valid_index())
        late = [s for s in syms if pd.isna(first_valid[s]) or first_valid[s] > start_ts]
        if late:
            raise ValueError("시작일 이후에 데이터가 시작되는 자산: " + ", ".join(late))
        idx = panel.close.index
        dates = idx[(idx >= start_ts) & (idx <= end_ts)]
        close = close_all.loc[:dates[-1]].ffill()
        rets = close.pct_change(fill_method=None).loc[dates]
        cash = (panel.cash if panel.cash is not None else pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0).loc[dates]
        R = np.nan_to_num(rets.to_numpy(dtype=float), nan=0.0)
        self.growth = np.concatenate([1.0 + R, (1.0 + cash.to_numpy(dtype=float))[:, None]], axis=1)
        self.syms, self.columns = list(syms), list(syms) + [CASH]
        self.dates, self.cash, self.cost = dates, cash, cost_bps / 1e4
        self.n_days = len(dates)

    def run(self, tabs: np.ndarray, decide: np.ndarray, choice: np.ndarray, first: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``tabs`` P × K × columns (fractions), ``decide`` / ``choice`` P × days, ``first`` P.
        Returns (equity paths P × days, annual one-way turnover P, trades P)."""
        tabs = np.asarray(tabs, dtype=float)
        P, T = tabs.shape[0], self.n_days
        decide, choice = np.asarray(decide, dtype=bool), np.asarray(choice, dtype=int)
        if decide.shape != (P, T) or choice.shape != (P, T):
            raise ValueError("decide·choice는 (경로 수, 거래일 수)여야 합니다.")
        rows = np.arange(P)
        hold = tabs[rows, np.asarray(first, dtype=int)].copy()
        prev = np.ones(P)
        pend = np.full(P, -1)
        paths = np.empty((P, T))
        turn, trades = np.zeros(P), np.zeros(P, dtype=int)
        for t in range(T):
            hold *= self.growth[t]
            value = hold.sum(axis=1)
            has = pend >= 0
            if has.any():
                h = np.flatnonzero(has)
                w = tabs[h, pend[h]]
                tr = np.abs(w * value[h, None] - hold[h]).sum(axis=1) / 2.0
                value[h] = value[h] - tr * self.cost
                hold[h] = w * value[h, None]
                turn[h] += tr / prev[h]
                trades[h] += 1
                pend[h] = -1
            paths[:, t] = value
            prev = value
            d = decide[:, t]
            pend[d] = choice[d, t]
        return paths, turn / (max(T, 1) / TRADING_DAYS), trades


def static_rows(base: np.ndarray, sim: DynamicSim, offset: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``run`` arguments for static allocations (P × columns) rebalanced at quarter ends."""
    base = np.atleast_2d(base)
    P, T = len(base), sim.n_days
    decide = np.zeros((P, T), dtype=bool)
    decide[:, quarter_rows(sim.dates, offset)] = True
    return base[:, None, :], decide, np.zeros((P, T), dtype=int), np.zeros(P, dtype=int)


def equal_cost_static(base: dict[str, float], rule: str, action: str, columns: list[str], target_cagr: float,
                      cagr_of, n_grid: int = 41) -> tuple[float, np.ndarray]:
    """The static allocation that moves x of nasdaq (or of the tilt) to the action's destination for good,
    with x chosen so its CAGR matches ``target_cagr``. ``cagr_of(W)`` returns CAGRs of static rows.
    x runs over a grid from 0 to the full move of the rule's strongest state; the crossing is interpolated.
    Returns (x as a share of the full move, weights)."""
    tab = tables(base, rule, action, columns)
    full = tab[-1] if action in ("defensive", "safe") else tab[1]  # the risk-off target
    xs = np.linspace(0.0, 1.0, n_grid)
    W = np.vstack([tab[0] + x * (full - tab[0]) for x in xs])
    c = np.asarray(cagr_of(W), dtype=float)
    if target_cagr >= c[0]:
        x = 0.0
    elif target_cagr <= c.min():
        x = float(xs[int(np.argmin(c))])
    else:
        k = int(np.flatnonzero(c <= target_cagr)[0])
        x = float(xs[k - 1] + (xs[k] - xs[k - 1]) * (c[k - 1] - target_cagr) / (c[k - 1] - c[k]))
    return x, tab[0] + x * (full - tab[0])


__all__ = ["COOLDOWN", "CANARY_RULES", "HELD_RULES", "RULES", "ACTIONS", "CURRENCY_RULES", "CHAINS", "CANARY_TICKERS",
           "SECTORS", "Inputs", "Rule", "chained", "extend_back", "momentum", "turbulence", "sleeves", "sleeve_index",
           "raw_states", "states", "signal_start", "tables", "schedule", "Schedule", "quarter_rows", "placebo_states", "DynamicSim", "static_rows",
           "equal_cost_static"]
