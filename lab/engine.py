"""Portfolio backtest engine.

Rules (also shown in the UI as RULES_KO):
- Signals use data up to the day-t close; orders execute at the day t+1 open.
- Entry: every selected entry block is true (AND) and the symbol passes the price and
  liquidity filters.
- Exit: any selected exit block is true (OR).
- Up to `max_positions` holdings. Each new position receives min(equity / max_positions, cash).
  Positions are not rebalanced afterwards.
- Uninvested cash earns the panel's cash return (T-bill ETF) when `cash_yield` is on.
- When more symbols signal than free slots, larger average dollar volume goes first.
- `cost_bps` is charged on each side (buy and sell) and should include slippage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from operator import mul

import numpy as np
import pandas as pd

from .blocks import ALL_BLOCKS, ENTRY_BLOCKS, EXIT_BLOCKS, atr
from .data import BENCHMARK, Panel

RULES_KO = """- 신호는 **t일 종가까지의 데이터**로 계산하고, 주문은 **t+1일 시가**에 체결합니다 (미래 데이터 사용 없음).
- 진입: 선택한 진입 조건이 **모두** 참이고, 최소 주가·최소 거래대금 조건을 통과한 종목.
- 청산: 선택한 청산 조건 중 **하나라도** 참이면 다음 날 시가에 전량 매도.
- 최대 보유 종목 수만큼 슬롯이 있고, 새 종목에는 min(목표 금액, 남은 현금)을 배정합니다. 이후 리밸런싱은 하지 않습니다.
  - **동일 금액**: 목표 금액 = 총자산 ÷ 최대 보유 수.
  - **변동성 역가중**: 목표 금액 = (총자산 ÷ 최대 보유 수) × min(유니버스 중앙 변동성 ÷ 그 종목 변동성, 2). 변동성 = 신호일까지 60거래일 일간 수익률의 표준편차.
    변동성이 낮은 종목은 더 많이(최대 2배), 높은 종목은 적게 사서 종목마다 위험을 비슷하게 맞춥니다.
- 빈 슬롯보다 신호가 많으면 **평균 거래대금이 큰 종목**부터 매수합니다 (성과와 무관한 유동성 기준).
- 남는 현금은 **단기국채 ETF(BIL) 수익률**을 받습니다 (BIL 상장 전 2007년 이전은 13주 국채금리 ÷ 252). 끄면 0%.
  현금 대용 ETF를 사고파는 비용은 무시합니다 (호가 차이가 매우 작음).
- 거래 비용(bp)은 매수와 매도 **각각**에 부과합니다. 수수료와 슬리피지를 합친 값으로 넣으세요.
- 다음 날 시가가 없으면(거래 정지 등) 매도는 다음 거래일로 미루고, 매수는 취소합니다.
- 가격은 배당·분할 조정 가격(yfinance auto_adjust)이므로 배당 재투자 효과가 포함됩니다. 세금은 반영하지 않습니다.
"""


TRADE_COLUMNS = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return", "holding_days", "exit_reason"]
VOL_DAYS = 60  # volatility window for inverse-volatility sizing
SIZE_CAP = 2.0  # a position is at most twice the equal slot
SIZING_LABELS = {"equal": "동일 금액", "inverse_vol": "변동성 역가중"}


@dataclass(frozen=True)
class StrategyConfig:
    entries: dict[str, float | None]
    exits: dict[str, float | None]
    start: str
    end: str
    max_positions: int = 10
    cost_bps: float = 10.0
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0
    liquidity_days: int = 20
    cash_yield: bool = True
    sizing: str = "equal"  # "equal" | "inverse_vol"
    # strategy-only test (lab/single.py, DECISIONS 67): every entry signal is taken whatever the
    # cash, so each symbol's trades match trading it alone; the engine's equity is then not used
    single: bool = False

    def __post_init__(self):
        # numpy numbers (e.g. values read back from a grid table) become plain Python numbers,
        # so configs serialise to JSON and hash identically however they were built
        def native(v):
            return v.item() if hasattr(v, "item") else v

        object.__setattr__(self, "entries", {k: native(v) for k, v in self.entries.items()})
        object.__setattr__(self, "exits", {k: native(v) for k, v in self.exits.items()})

    def with_param(self, key: str, value: float) -> "StrategyConfig":
        if key in self.entries:
            return replace(self, entries={**self.entries, key: value})
        if key in self.exits:
            return replace(self, exits={**self.exits, key: value})
        if key == "max_positions":
            return replace(self, max_positions=int(value))
        if key == "cost_bps":
            return replace(self, cost_bps=float(value))
        raise KeyError(key)

    def param_value(self, key: str) -> float:
        if key in self.entries:
            return self.entries[key]
        if key in self.exits:
            return self.exits[key]
        return getattr(self, key)

    def tunable(self) -> list[str]:
        """Selected blocks that have a numeric parameter (fixed-rule blocks are excluded)."""
        return [k for k in [*self.entries, *self.exits] if ALL_BLOCKS[k].param is not None]

    def describe(self) -> str:
        def fmt(blocks, chosen):
            return " + ".join(blocks[k].label + ("" if v is None else f"({_num(v)})") for k, v in chosen.items()) or "없음"

        size = "" if self.sizing == "equal" else f" | {SIZING_LABELS[self.sizing]}"
        slots = "종목별 단독 매매" if self.single else f"최대 {self.max_positions}종목{size}"
        return f"진입: {fmt(ENTRY_BLOCKS, self.entries)} | 청산: {fmt(EXIT_BLOCKS, self.exits)} | {slots}"

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["sizing"] == "equal":  # fields added later are omitted at their default: earlier keys stay valid
            del d["sizing"]
        if not d["single"]:
            del d["single"]
        return d

    @staticmethod
    def from_dict(d: dict) -> "StrategyConfig":
        known = {k: v for k, v in d.items() if k in StrategyConfig.__dataclass_fields__}
        return StrategyConfig(**known)

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]

    def strategy_fingerprint(self) -> str:
        """Identity of the rules alone (dates excluded): the same strategy on another period matches."""
        d = {k: v for k, v in self.to_dict().items() if k not in ("start", "end")}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


@dataclass
class BacktestResult:
    config: StrategyConfig
    equity: pd.Series  # portfolio value at each close, starts at 1.0
    exposure: pd.Series  # invested value / equity at each close
    benchmark: pd.Series  # SPY close normalised to 1.0 on the first day
    trades: pd.DataFrame  # closed trades
    open_positions: pd.DataFrame
    universe_size: int
    warnings: list[str] = field(default_factory=list)
    # orders decided on the last close, for the next open: {"buy": [symbols], "sell": {symbol: exit block key}}
    pending_orders: dict = field(default_factory=dict)

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.equity.index[0], self.equity.index[-1]


class SignalCache:
    """Reuses block outputs across backtests on the same panel (important for grids)."""

    def __init__(self, panel: Panel):
        self.panel = panel
        self._frames: dict[tuple[str, float | None], pd.DataFrame] = {}
        self._dv: dict[int, pd.DataFrame] = {}
        self._atr: pd.DataFrame | None = None
        self._memo: dict[tuple, object] = {}

    def memo(self, key: tuple, make):
        """Per-window derived data (plain lists for the daily loop). Grids reuse one window."""
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def frame(self, key: str, value: float | None) -> pd.DataFrame:
        k = (key, None if value is None else float(value))
        if k not in self._frames:
            block = ALL_BLOCKS[key]
            self._frames[k] = block.compute(self.panel, value).fillna(False).astype(bool)
        return self._frames[k]

    def vol_rows(self, dates: pd.DatetimeIndex) -> tuple[list, list]:
        """60-day volatility of each symbol and its cross-sectional median over the tradable universe."""
        rets = self.panel.close.pct_change(fill_method=None)
        vol = rets.rolling(VOL_DAYS, min_periods=VOL_DAYS).std()
        tradable = vol[self.panel.tradable_symbols]
        ref = tradable.median(axis=1, skipna=True)
        return vol.loc[dates].to_numpy(dtype=float).tolist(), ref.loc[dates].to_numpy(dtype=float).tolist()

    def dollar_volume(self, days: int) -> pd.DataFrame:
        if days not in self._dv:
            self._dv[days] = (self.panel.close * self.panel.volume).rolling(days, min_periods=days).mean()
        return self._dv[days]

    @property
    def atr14(self) -> pd.DataFrame:
        if self._atr is None:
            self._atr = atr(self.panel, 14)
        return self._atr


def run_backtest(panel: Panel, cfg: StrategyConfig, cache: SignalCache | None = None) -> BacktestResult:
    if not cfg.entries:
        raise ValueError("진입 조건을 하나 이상 선택하세요.")
    if not cfg.exits:
        raise ValueError("청산 조건을 하나 이상 선택하세요.")
    if BENCHMARK not in panel.symbols:
        raise ValueError(f"벤치마크 {BENCHMARK} 가격이 데이터에 없습니다.")
    cache = cache or SignalCache(panel)

    idx = panel.close.index
    in_period = (idx >= pd.Timestamp(cfg.start)) & (idx <= pd.Timestamp(cfg.end))
    if in_period.sum() < 2:
        raise ValueError("선택한 기간에 거래일이 2일 미만입니다.")
    dates = idx[in_period]

    w = (dates[0], dates[-1])
    liq = int(cfg.liquidity_days)
    dv = cache.dollar_volume(liq)
    tradable = cache.memo(("tradable",), lambda: panel.close.columns.isin(panel.tradable_symbols))

    def make_candidates():
        entry = None
        for key, value in cfg.entries.items():
            f = cache.frame(key, value)
            entry = f if entry is None else (entry & f)
        eligible = (panel.close >= cfg.min_price) & (dv >= cfg.min_dollar_volume)
        mask = (entry & eligible).loc[dates].to_numpy() & tradable
        rows, cols = np.nonzero(mask)
        bounds = np.searchsorted(rows, np.arange(len(dates) + 1)).tolist()
        cols = cols.tolist()
        return [cols[bounds[i]:bounds[i + 1]] for i in range(len(dates))]

    candidates = cache.memo(("entry", w, tuple(cfg.entries.items()), cfg.min_price, cfg.min_dollar_volume, liq),
                            make_candidates)
    # Exit checks run once per position, at entry, on whole columns (symbol x day arrays): an
    # exit depends only on that position's entry price and the closes after it, so the first
    # exit day can be found with array operations instead of checking every holding every day.
    signal_exits = {k: cache.memo(("exit", w, k, cfg.exits[k]),
                                  lambda k=k: np.ascontiguousarray(cache.frame(k, cfg.exits[k]).loc[dates].to_numpy(dtype=bool).T))
                    for k in cfg.exits if EXIT_BLOCKS[k].compute is not None}
    trail = cfg.exits.get("trailing_pct")
    stop = cfg.exits.get("stop_loss_pct")
    atr_k = cfg.exits.get("atr_trail")
    AT = cache.memo(("atr", w), lambda: np.ascontiguousarray(cache.atr14.loc[dates].to_numpy(dtype=float).T)) if atr_k is not None else None
    CT = cache.memo(("close", w), lambda: np.ascontiguousarray(panel.close.loc[dates].to_numpy(dtype=float).T))
    # Plain Python lists: the daily loop touches single values, and Python floats are far
    # faster than numpy scalars for that. Results are bit-for-bit identical (both are doubles).
    O = cache.memo(("open", w), lambda: panel.open.loc[dates].to_numpy(dtype=float).tolist())
    LC = cache.memo(("last_close", w),  # last known close of each symbol, within the window
                    lambda: panel.close.loc[dates].ffill().to_numpy(dtype=float).tolist())
    rank = cache.memo(("rank", w, liq), lambda: np.nan_to_num(dv.loc[dates].to_numpy(dtype=float), nan=-1.0).tolist())
    # each day's candidates already in buying order (larger dollar volume first). The sort is
    # stable, so dropping held symbols afterwards leaves the same order as sorting after the drop.
    ordered = cache.memo(("ordered", w, tuple(cfg.entries.items()), cfg.min_price, cfg.min_dollar_volume, liq),
                         lambda: [sorted(cs, key=lambda j, ri=ri: -ri[j]) for cs, ri in zip(candidates, rank)])
    inv_vol = cfg.sizing == "inverse_vol"
    single = cfg.single
    if inv_vol:
        V, VREF = cache.memo(("vol", w), lambda: cache.vol_rows(dates))
    cash_r = (panel.cash.reindex(dates).fillna(0.0).to_numpy().tolist() if (cfg.cash_yield and panel.cash is not None)
              else [0.0] * len(dates))
    symbols = panel.symbols
    c = cfg.cost_bps / 10_000.0
    K = int(cfg.max_positions)
    trail_mult = None if trail is None else 1 - trail / 100.0
    stop_mult = None if stop is None else 1 - stop / 100.0
    exit_keys = list(cfg.exits)
    n = len(dates)
    path_exits = [k for k in exit_keys if k not in signal_exits]  # depend on the entry price

    def make_next_signal():
        """next_sig[j, d]: first day >= d on which a signal exit of symbol j fires (n = never)."""
        hit = CT == CT
        hit &= np.logical_or.reduce([signal_exits[k] for k in signal_exits])
        days = np.where(hit, np.arange(n), n)
        return np.ascontiguousarray(np.minimum.accumulate(days[:, ::-1], axis=1)[:, ::-1])

    next_sig = (cache.memo(("next_signal", w, tuple((k, cfg.exits[k]) for k in signal_exits)), make_next_signal)
                if signal_exits else None)

    def first_exit(j: int, e: int, entry_price: float) -> tuple[int, str] | None:
        """(day, exit key) of the first close from day e on that triggers an exit of a position
        in symbol j bought at the day-e open, or None. Same rule as checking day by day: days
        without a close are skipped; the peak includes the entry price and every close so far
        (today's included); the first selected exit in cfg.exits order gives the reason.
        Scans in growing chunks so a short holding does not read the whole period."""
        stop_at = n if next_sig is None else int(next_sig[j, e])  # signal exits need no scan
        if not path_exits:
            return None if stop_at >= n else (stop_at, next(k for k in exit_keys if signal_exits[k][j, stop_at]))
        stop_at = min(n, stop_at + 1)  # the signal day itself is scanned for the reason order
        peak, start, size = entry_price, e, 16
        while start < stop_at:
            end = min(stop_at, start + size)
            cj = CT[j, start:end]
            valid = cj == cj
            pk = np.maximum.accumulate(np.where(valid, cj, -np.inf))
            np.maximum(pk, peak, out=pk)
            hits = []
            for k in exit_keys:
                if k in signal_exits:
                    hits.append(signal_exits[k][j, start:end])
                elif k == "trailing_pct":
                    hits.append(cj < pk * trail_mult)
                elif k == "atr_trail":
                    a = AT[j, start:end]
                    hits.append((a == a) & (cj < pk - atr_k * a))
                else:  # stop_loss_pct
                    hits.append(cj < entry_price * stop_mult)
            hit = valid & np.logical_or.reduce(hits)
            if hit.any():
                d = int(hit.argmax())
                return start + d, next(k for k, h in zip(exit_keys, hits) if h[d])
            peak = float(pk[-1])
            start, size = end, size * 2
        return None

    # an exit depends only on (symbol, entry day) once the exit rules are fixed, so the answers
    # are shared by every backtest with these exits on this window (all entry rules of a grid)
    exit_at = cache.memo(("exit_at", w, tuple(cfg.exits.items())), dict)

    nan_ = float("nan")
    day = cache.memo(("days", w), lambda: list(dates))  # indexing a DatetimeIndex one item at a time is slow
    cash = 1.0
    last_close = [float("nan")] * len(symbols)  # yesterday's LC row (all NaN before the first day)
    pos: dict[int, dict] = {}
    held_j: list[int] = []  # pos keys and shares in pos order, rebuilt only when pos changes
    held_sh: list[float] = []
    pending_entry: list[int] = []
    pending_exit: dict[int, str] = {}
    exits_on: dict[int, list[tuple[int, str]]] = {}  # day -> [(symbol, exit key)] in holding order
    equity = [0.0] * n
    exposure = [0.0] * n
    trades: list[tuple] = []

    for i in range(n):
        Oi = O[i]
        # 1) execute yesterday's orders at today's open
        for j in list(pending_exit):
            px = Oi[j]
            if px != px:
                continue  # no bar today; try again tomorrow
            p = pos.pop(j)
            cash += p["shares"] * px * (1 - c)
            trades.append((  # columns of TRADE_COLUMNS (tuples build the frame faster than dicts)
                symbols[j], day[p["entry_i"]], day[i], p["entry_price"], px,
                (px * (1 - c)) / (p["entry_price"] * (1 + c)) - 1.0, i - p["entry_i"],
                EXIT_BLOCKS[pending_exit[j]].label,
            ))
            del pending_exit[j]
            held_j = list(pos)
            held_sh = [q["shares"] for q in pos.values()]
        if pending_entry:
            held_value = sum(p["shares"] * (Oi[j] if Oi[j] == Oi[j] else last_close[j]) for j, p in pos.items())
            slot = (cash + held_value) / K
            for j in pending_entry:
                if len(pos) >= K or (cash <= 1e-9 and not single):
                    break
                px = Oi[j]
                if px != px or px <= 0:
                    continue
                target = slot
                if inv_vol:  # signal day = previous session
                    v, ref = V[i - 1][j] if i else nan_, VREF[i - 1] if i else nan_
                    if v == v and v > 0 and ref == ref:
                        target = slot * min(ref / v, SIZE_CAP)
                alloc = target if single else min(target, cash)
                pos[j] = {"shares": alloc / (px * (1 + c)), "entry_price": px, "entry_i": i}
                cash -= alloc
                x = exit_at.get((j, i), exit_at)
                if x is exit_at:
                    x = exit_at[(j, i)] = first_exit(j, i, px)
                if x is not None:
                    exits_on.setdefault(x[0], []).append((j, x[1]))
            pending_entry = []
            held_j = list(pos)
            held_sh = [q["shares"] for q in pos.values()]

        # 2) cash earns today's T-bill return, then mark to market at today's close
        cash *= 1.0 + cash_r[i]
        last_close = LC[i]
        invested = sum(map(mul, held_sh, map(last_close.__getitem__, held_j)))  # same terms, same order
        equity[i] = cash + invested
        exposure[i] = invested / equity[i] if equity[i] > 0 else 0.0

        # steps 3-4 also run on the last day: they only queue orders for the next open, which
        # the backtest never reaches, and those orders are reported as pending_orders

        # 3) exit checks on today's close (found at entry by first_exit). Holdings are listed in
        # entry order, which is also the order of pos, so sells queue in the same order.
        for j, k in exits_on.pop(i, ()):
            pending_exit[j] = k

        # 4) entry candidates on today's close
        free = K - (len(pos) - len(pending_exit))
        if free > 0:
            cand = [j for j in ordered[i] if j not in pos]
            pending_entry = cand[:free]

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    open_df = pd.DataFrame([
        {"symbol": symbols[j], "entry_date": day[p["entry_i"]], "entry_price": p["entry_price"],
         "last_close": last_close[j], "unrealized_return": last_close[j] * (1 - c) / (p["entry_price"] * (1 + c)) - 1.0}
        for j, p in pos.items()
    ], columns=["symbol", "entry_date", "entry_price", "last_close", "unrealized_return"])

    bench = panel.close[BENCHMARK].loc[dates].ffill()
    warnings = []
    if bench.isna().any():
        warnings.append("선택 기간 초반에 SPY 가격이 없어 벤치마크 비교가 일부 날짜에서 빠졌습니다.")
    bench = bench / bench.dropna().iloc[0]
    return BacktestResult(
        config=cfg,
        equity=pd.Series(equity, index=dates, name="strategy", dtype=float),
        exposure=pd.Series(exposure, index=dates, name="exposure", dtype=float),
        benchmark=bench.rename(BENCHMARK),
        trades=trades_df,
        open_positions=open_df,
        universe_size=int(tradable.sum()),
        warnings=warnings,
        pending_orders={"buy": [symbols[j] for j in pending_entry],
                        "sell": {symbols[j]: k for j, k in pending_exit.items()}},
    )


# --------------------------------------------------------------------------- baselines

BASELINE_LABELS = {
    "spy": "SPY 보유",
    "spy_ma200": "SPY 200일선",
    "equal_weight": "유니버스 동일비중",
}
BASELINE_HELP = {
    "spy": "SPY를 기간 내내 보유 (배당 재투자).",
    "spy_ma200": "SPY 종가 > 200일 이동평균이면 보유, 아래로 내려가면 다음 날 시가에 매도하고 현금(단기국채) 보유. "
                 "규칙 하나짜리 가장 단순한 추세추종. 같은 거래 비용·현금 수익률 적용.",
    "equal_weight": "거래 대상 유니버스의 모든 종목을 같은 비중으로 보유하고 매일 비중을 다시 맞춤 (신호 없음). "
                    "‘종목 선택 없이 이 유니버스를 그냥 샀다면’의 기준.",
}


def baselines(panel: Panel, cfg: StrategyConfig, dates: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Simple reference strategies on the same dates, costs and cash assumption."""
    out: dict[str, pd.Series] = {}
    spy = panel.close[BENCHMARK].reindex(dates).ffill()
    out["spy"] = spy / spy.dropna().iloc[0]
    ma_cfg = replace(cfg, entries={"above_ma": 200}, exits={"below_ma": 200}, max_positions=1,
                     min_price=0.0, min_dollar_volume=0.0)
    out["spy_ma200"] = run_backtest(panel.subset([BENCHMARK]), ma_cfg).equity.reindex(dates)
    rets = panel.close[panel.tradable_symbols].reindex(dates).pct_change(fill_method=None)
    ew = rets.mean(axis=1, skipna=True).fillna(0.0)
    ew.iloc[0] = 0.0
    out["equal_weight"] = (1 + ew).cumprod()
    return out
