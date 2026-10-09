"""Portfolio backtest engine.

Rules (also shown in the UI):
- Signals use data up to the day-t close; orders execute at the day t+1 open.
- Entry: every selected entry block is true (AND) and the symbol passes the liquidity filter.
- Exit: any selected exit block is true (OR).
- Up to `max_positions` holdings. Each new position receives min(equity / max_positions, cash).
  Positions are not rebalanced afterwards. Uninvested cash earns 0%.
- When more symbols signal than free slots, larger 20-day average dollar volume goes first.
- `cost_bps` is charged on each side (buy and sell) and should include slippage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from .blocks import ENTRY_BLOCKS, EXIT_BLOCKS
from .data import BENCHMARK, Panel

RULES_KO = """- 신호는 **t일 종가까지의 데이터**로 계산하고, 주문은 **t+1일 시가**에 체결합니다 (미래 데이터 사용 없음).
- 진입: 선택한 진입 조건이 **모두** 참이고, 최소 주가·최소 거래대금 조건을 통과한 종목.
- 청산: 선택한 청산 조건 중 **하나라도** 참이면 다음 날 시가에 전량 매도.
- 최대 보유 종목 수만큼 슬롯이 있고, 새 종목에는 min(총자산 ÷ 최대 보유 수, 남은 현금)을 배정합니다. 이후 리밸런싱은 하지 않습니다.
- 빈 슬롯보다 신호가 많으면 **20일 평균 거래대금이 큰 종목**부터 매수합니다 (성과와 무관한 유동성 기준).
- 남는 현금의 이자는 0%로 가정합니다.
- 거래 비용(bp)은 매수와 매도 **각각**에 부과합니다. 슬리피지를 포함한 값으로 넣으세요.
- 다음 날 시가가 없으면(거래 정지 등) 매도는 다음 거래일로 미루고, 매수는 취소합니다.
- 가격은 배당·분할 조정 가격(yfinance auto_adjust)이므로 배당 재투자 효과가 포함됩니다.
"""


@dataclass(frozen=True)
class StrategyConfig:
    entries: dict[str, float]
    exits: dict[str, float]
    start: str
    end: str
    max_positions: int = 10
    cost_bps: float = 10.0
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0

    def with_param(self, key: str, value: float) -> "StrategyConfig":
        if key in self.entries:
            return replace(self, entries={**self.entries, key: value})
        if key in self.exits:
            return replace(self, exits={**self.exits, key: value})
        if key == "max_positions":
            return replace(self, max_positions=int(value))
        raise KeyError(key)

    def param_value(self, key: str) -> float:
        if key in self.entries:
            return self.entries[key]
        if key in self.exits:
            return self.exits[key]
        return getattr(self, key)

    def describe(self) -> str:
        def fmt(blocks, chosen):
            return " + ".join(f"{blocks[k].label}({_num(v)})" for k, v in chosen.items()) or "없음"

        return f"진입: {fmt(ENTRY_BLOCKS, self.entries)} | 청산: {fmt(EXIT_BLOCKS, self.exits)} | 최대 {self.max_positions}종목"

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


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

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.equity.index[0], self.equity.index[-1]


class SignalCache:
    """Reuses block outputs across backtests on the same panel (important for grids)."""

    def __init__(self, panel: Panel):
        self.panel = panel
        self._frames: dict[tuple[str, float], pd.DataFrame] = {}
        dollar_volume = panel.close * panel.volume
        self.dv20 = dollar_volume.rolling(20, min_periods=20).mean()

    def frame(self, key: str, value: float) -> pd.DataFrame:
        k = (key, float(value))
        if k not in self._frames:
            block = ENTRY_BLOCKS.get(key) or EXIT_BLOCKS[key]
            self._frames[k] = block.compute(self.panel, value).fillna(False).astype(bool)
        return self._frames[k]


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

    entry = None
    for key, value in cfg.entries.items():
        f = cache.frame(key, value)
        entry = f if entry is None else (entry & f)
    eligible = (panel.close >= cfg.min_price) & (cache.dv20 >= cfg.min_dollar_volume)
    tradable = panel.close.columns.isin(panel.tradable_symbols)
    entry = (entry & eligible).loc[dates].to_numpy() & tradable

    signal_exits = [k for k in cfg.exits if EXIT_BLOCKS[k].compute is not None]
    exit_frames = {k: cache.frame(k, cfg.exits[k]).loc[dates].to_numpy() for k in signal_exits}
    trail = cfg.exits.get("trailing_pct")
    stop = cfg.exits.get("stop_loss_pct")

    O = panel.open.loc[dates].to_numpy(dtype=float)
    C = panel.close.loc[dates].to_numpy(dtype=float)
    rank = np.nan_to_num(cache.dv20.loc[dates].to_numpy(dtype=float), nan=-1.0)
    symbols = panel.symbols
    c = cfg.cost_bps / 10_000.0
    K = int(cfg.max_positions)

    n = len(dates)
    cash = 1.0
    last_close = np.full(len(symbols), np.nan)
    pos: dict[int, dict] = {}
    pending_entry: list[int] = []
    pending_exit: dict[int, str] = {}
    equity = np.empty(n)
    exposure = np.empty(n)
    trades: list[dict] = []

    for i in range(n):
        # 1) execute yesterday's orders at today's open
        for j in list(pending_exit):
            px = O[i, j]
            if np.isnan(px):
                continue  # no bar today; try again tomorrow
            p = pos.pop(j)
            cash += p["shares"] * px * (1 - c)
            trades.append({
                "symbol": symbols[j],
                "entry_date": dates[p["entry_i"]],
                "exit_date": dates[i],
                "entry_price": p["entry_price"],
                "exit_price": px,
                "return": (px * (1 - c)) / (p["entry_price"] * (1 + c)) - 1.0,
                "holding_days": i - p["entry_i"],
                "exit_reason": EXIT_BLOCKS[pending_exit[j]].label,
            })
            del pending_exit[j]
        if pending_entry:
            held_value = sum(p["shares"] * (O[i, j] if not np.isnan(O[i, j]) else last_close[j]) for j, p in pos.items())
            target = (cash + held_value) / K
            for j in pending_entry:
                if len(pos) >= K or cash <= 1e-9:
                    break
                px = O[i, j]
                if np.isnan(px) or px <= 0:
                    continue
                alloc = min(target, cash)
                pos[j] = {"shares": alloc / (px * (1 + c)), "entry_price": px, "entry_i": i, "peak": px}
                cash -= alloc
            pending_entry = []

        # 2) mark to market at today's close
        today = C[i]
        has = ~np.isnan(today)
        last_close[has] = today[has]
        invested = sum(p["shares"] * last_close[j] for j, p in pos.items())
        equity[i] = cash + invested
        exposure[i] = invested / equity[i] if equity[i] > 0 else 0.0

        if i == n - 1:
            break

        # 3) exit checks on today's close
        for j, p in pos.items():
            if j in pending_exit or np.isnan(today[j]):
                continue
            p["peak"] = max(p["peak"], today[j])
            reason = None
            for k in cfg.exits:
                if k in exit_frames:
                    hit = exit_frames[k][i, j]
                elif k == "trailing_pct":
                    hit = today[j] < p["peak"] * (1 - trail / 100.0)
                else:  # stop_loss_pct
                    hit = today[j] < p["entry_price"] * (1 - stop / 100.0)
                if hit:
                    reason = k
                    break
            if reason:
                pending_exit[j] = reason

        # 4) entry candidates on today's close
        free = K - (len(pos) - len(pending_exit))
        if free > 0:
            cand = np.flatnonzero(entry[i])
            cand = [j for j in cand if j not in pos]
            cand.sort(key=lambda j: -rank[i, j])
            pending_entry = cand[:free]

    trades_df = pd.DataFrame(trades, columns=[
        "symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return", "holding_days", "exit_reason",
    ])
    open_df = pd.DataFrame([
        {"symbol": symbols[j], "entry_date": dates[p["entry_i"]], "entry_price": p["entry_price"],
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
        equity=pd.Series(equity, index=dates, name="strategy"),
        exposure=pd.Series(exposure, index=dates, name="exposure"),
        benchmark=bench.rename(BENCHMARK),
        trades=trades_df,
        open_positions=open_df,
        universe_size=int(tradable.sum()),
        warnings=warnings,
    )
