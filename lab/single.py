"""Strategy-only test: each ETF traded on its own (DECISIONS 67).

The rule is applied to every symbol separately: 100% in the symbol while a position is open,
otherwise in cash (T-bills). There are no slots, no position sizing and no choice between
signals, so the result measures the rule itself.

One engine run gives every symbol's trades at once: with one slot per symbol and the engine's
`single` flag (entries are not limited by cash) every signal is taken, and each symbol's trade sequence (entries, trailing stops, exits) is the
same as when that symbol is traded alone, because position state is kept per symbol. The daily
return of each symbol's stand-alone account is rebuilt from those trades:
  entry day   close / (open x (1 + cost)) - 1        (cash until the open is ignored)
  held days   close / previous close - 1
  exit day    open x (1 - cost) / previous close - 1 (cash after the open is ignored)
  otherwise   the cash return
The "sleeve portfolio" is the daily average of these accounts over the symbols with data that
day (equal weight, rebalanced daily): the portfolio implied by the trade records.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .data import BENCHMARK, Panel
from .engine import SignalCache, StrategyConfig, run_backtest
from .target import DD_BUDGET, _cagr, _mdd, blend, fit_weight, target_report

MIN_YEARS = 3.0  # a symbol needs this much data in the period to be judged


def single_config(cfg: StrategyConfig, panel: Panel) -> StrategyConfig:
    return replace(cfg, max_positions=max(1, len(panel.tradable_symbols)), sizing="equal", single=True)


def symbol_returns(panel: Panel, cfg: StrategyConfig, result, with_held: bool = False):
    """Daily returns (dates x symbols) of each symbol traded alone, over the backtest dates.
    with_held=True also returns the bool frame of days on which the symbol was held."""
    dates = result.equity.index
    syms = panel.tradable_symbols
    close = panel.close[syms].reindex(dates)
    opn = panel.open[syms].reindex(dates)
    c = cfg.cost_bps / 10_000.0
    cash = (panel.cash.reindex(dates).fillna(0.0) if (cfg.cash_yield and panel.cash is not None)
            else pd.Series(0.0, index=dates)).to_numpy()
    out = np.repeat(cash[:, None], len(syms), axis=1)
    pos = {s: k for k, s in enumerate(syms)}
    day = {d: i for i, d in enumerate(dates)}
    cl, op = close.to_numpy(dtype=float), opn.to_numpy(dtype=float)
    last_cl = close.ffill().to_numpy(dtype=float)  # latest close on or before each day
    prev_cl = np.vstack([np.full((1, len(syms)), np.nan), last_cl[:-1]])
    cc = np.where(np.isnan(cl) | np.isnan(prev_cl), 0.0, cl / prev_cl - 1.0)  # close-to-close, 0 on gaps
    held = np.zeros(out.shape, dtype=bool)

    def fill(j, a, b_exit):
        """Entry at the open of day a; exit at the open of day b_exit (None = still open)."""
        last = len(dates) - 1 if b_exit is None else b_exit - 1
        out[a, j] = cl[a, j] / (op[a, j] * (1 + c)) - 1.0 if cl[a, j] == cl[a, j] else 0.0
        out[a + 1:last + 1, j] = cc[a + 1:last + 1, j]
        held[a:last + 1, j] = True
        if b_exit is not None:
            prev = last_cl[b_exit - 1, j]
            out[b_exit, j] = op[b_exit, j] * (1 - c) / prev - 1.0 if prev == prev and op[b_exit, j] == op[b_exit, j] else 0.0

    for t in result.trades.itertuples():
        fill(pos[t.symbol], day[t.entry_date], day[t.exit_date])
    for t in result.open_positions.itertuples():
        fill(pos[t.symbol], day[t.entry_date], None)
    frame = pd.DataFrame(out, index=dates, columns=syms)
    has_data = close.notna().cumsum() > 0  # from the symbol's first close in the period
    frame = frame.where(has_data)
    return (frame, pd.DataFrame(held, index=dates, columns=syms)) if with_held else frame


_BH: dict = {}  # buy-and-hold weight per (panel, symbol, first date, last date): same for every rule


def evaluate(panel: Panel, cfg: StrategyConfig, cache: SignalCache | None = None) -> dict:
    """Per-symbol and sleeve-portfolio results of one rule (DECISIONS 67)."""
    scfg = single_config(cfg, panel)
    res = run_backtest(panel, scfg, cache)
    rets, held = symbol_returns(panel, scfg, res, with_held=True)
    dates = rets.index
    cash = (panel.cash.reindex(dates).fillna(0.0) if panel.cash is not None else pd.Series(0.0, index=dates)).to_numpy()
    close = panel.close[rets.columns].reindex(dates)
    trades_per = res.trades["symbol"].value_counts()
    rows = []
    for s in rets.columns:
        r = rets[s]
        ok = r.notna().to_numpy()
        if ok.sum() < MIN_YEARS * 252:
            continue
        rs, cs = r.to_numpy()[ok], cash[ok]
        bh = close[s].pct_change(fill_method=None).to_numpy()[ok]
        bh = np.where(np.isnan(bh), cs, bh)  # after a delisting (or on a gap) the holder sits in cash
        bh[0] = 0.0
        key = (id(panel), s, dates[ok][0], dates[ok][-1])
        if key not in _BH:
            _BH[key] = fit_weight(bh, cs)
        w_s, w_b = fit_weight(rs, cs), _BH[key]
        eq_raw, eq_s, eq_b = blend(rs, cs, 1.0), blend(rs, cs, w_s), blend(bh, cs, w_b)
        rows.append({"symbol": s, "years": ok.sum() / 252, "trades": int(trades_per.get(s, 0)),
                     "cagr": _cagr(eq_raw), "mdd": _mdd(eq_raw), "bh_cagr": _cagr(blend(bh, cs, 1.0)),
                     "bh_mdd": _mdd(blend(bh, cs, 1.0)), "dd_weight": w_s, "dd_cagr": _cagr(eq_s),
                     "bh_dd_cagr": _cagr(eq_b), "excess": _cagr(eq_s) - _cagr(eq_b)})
    per = pd.DataFrame(rows)
    # sleeve = equal weight over the accounts "in play" that day: symbols that may be bought (the
    # point-in-time universe, or every symbol with data) plus symbols still held after dropping out
    in_play = (panel.eligible[rets.columns].reindex(dates).fillna(False).astype(bool) if panel.eligible is not None
               else rets.notna()) | held
    weights = in_play.astype(float)
    sleeve_ret = (rets.fillna(0.0) * weights).sum(axis=1) / weights.sum(axis=1).replace(0.0, np.nan)
    sleeve_ret = sleeve_ret.fillna(pd.Series(cash, index=dates))
    sleeve = (1.0 + sleeve_ret).cumprod()
    sleeve.iloc[0] = 1.0
    port = target_report(sleeve, panel.close[BENCHMARK], panel.cash if panel.cash is not None else pd.Series(0.0, index=dates),
                         n_trades=len(res.trades))
    return {
        "per_symbol": per,
        "symbols": len(per),
        "symbol_pass_share": float((per["excess"] >= 0).mean()) if len(per) else np.nan,
        "symbol_median_excess": float(per["excess"].median()) if len(per) else np.nan,
        "symbol_median_mdd": float(per["mdd"].median()) if len(per) else np.nan,
        "n_trades": len(res.trades),
        "sleeve_cagr": _cagr(sleeve.to_numpy()), "sleeve_mdd": _mdd(sleeve.to_numpy()),
        "sleeve": sleeve,
        **{f"sleeve_{k}": v for k, v in port.items()},
    }
