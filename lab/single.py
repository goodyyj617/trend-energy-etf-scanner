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
from .engine import PARTIAL_FRACTION, SignalCache, StrategyConfig, run_backtest
from .target import DD_BUDGET, _cagr, _mdd, blend, blend_columns, fit_weight, fit_weights, target_report

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

    def fill(j, a, b_exit, p=None):
        """Entry at the open of day a; exit at the open of day b_exit (None = still open); partial sale
        of PARTIAL_FRACTION at the open of day p (None = none)."""
        last = len(dates) - 1 if b_exit is None else b_exit - 1
        out[a, j] = cl[a, j] / (op[a, j] * (1 + c)) - 1.0 if cl[a, j] == cl[a, j] else 0.0
        out[a + 1:last + 1, j] = cc[a + 1:last + 1, j]
        held[a:last + 1, j] = True
        if p is not None and (b_exit is None or p < b_exit):  # from day p the account is part stock, part cash
            f = PARTIAL_FRACTION
            prev = last_cl[p - 1, j]
            stock = (1 - f) * (1 + cc[p, j])
            kept = f * op[p, j] * (1 - c) / prev
            out[p, j] = stock + kept - 1.0
            end = last + 1
            g_s = np.cumprod(1 + cc[p + 1:end, j])
            g_k = np.cumprod(1 + cash[p + 1:end])
            total = np.concatenate([[stock + kept], stock * g_s + kept * g_k])
            out[p + 1:end, j] = total[1:] / total[:-1] - 1.0
            if b_exit is not None:
                s_prev, k_prev = (stock * g_s[-1], kept * g_k[-1]) if end > p + 1 else (stock, kept)
                prev_x = last_cl[b_exit - 1, j]
                sale = s_prev * (op[b_exit, j] * (1 - c) / prev_x) if prev_x == prev_x and op[b_exit, j] == op[b_exit, j] else s_prev
                out[b_exit, j] = (sale + k_prev * (1 + cash[b_exit])) / (s_prev + k_prev) - 1.0
            return
        if b_exit is not None:
            prev = last_cl[b_exit - 1, j]
            out[b_exit, j] = op[b_exit, j] * (1 - c) / prev - 1.0 if prev == prev and op[b_exit, j] == op[b_exit, j] else 0.0

    part = {(t.symbol, t.entry_date): day[t.date] for t in result.partial_exits.itertuples()}
    for t in result.trades.itertuples():
        fill(pos[t.symbol], day[t.entry_date], day[t.exit_date], part.get((t.symbol, t.entry_date)))
    for t in result.open_positions.itertuples():
        fill(pos[t.symbol], day[t.entry_date], None, part.get((t.symbol, t.entry_date)))
    frame = pd.DataFrame(out, index=dates, columns=syms)
    has_data = close.notna().cumsum() > 0  # from the symbol's first close in the period
    frame = frame.where(has_data)
    return (frame, pd.DataFrame(held, index=dates, columns=syms)) if with_held else frame


_BH: dict = {}  # buy-and-hold columns per (panel, first date, last date): the same for every rule


def per_symbol(panel: Panel, rets: pd.DataFrame, cash: np.ndarray, trades_per: pd.Series) -> pd.DataFrame:
    """One row per judged symbol: the strategy's stand-alone account against holding the symbol, both
    raw and scaled to the 15% drawdown budget. All symbols at once (columns of one matrix); a symbol's
    account starts on its first close in the period and runs to the end of the period."""
    dates = rets.index
    R = rets.to_numpy(dtype=float)
    ok = ~np.isnan(R)
    ndays = ok.sum(axis=0)
    judged = ndays >= MIN_YEARS * 252
    if not judged.any():
        return pd.DataFrame(columns=["symbol", "years", "trades", "cagr", "mdd", "bh_cagr", "bh_mdd", "dd_weight",
                                     "dd_cagr", "bh_dd_cagr", "excess"])
    syms = rets.columns[judged]
    R, ok, ndays = R[:, judged], ok[:, judged], ndays[judged]
    C = np.where(ok, cash[:, None], 0.0)
    Rz = np.where(ok, R, 0.0)
    key = (id(panel), dates[0], dates[-1])
    if key not in _BH:
        _BH[key] = {}
    cached = _BH[key]
    need = [s for s in syms if s not in cached]
    if need:
        cl = panel.close[need].reindex(dates).to_numpy(dtype=float)
        pct = np.vstack([np.full((1, len(need)), np.nan), cl[1:] / cl[:-1] - 1.0])
        idx = [syms.get_loc(s) for s in need]
        okn, cn = ok[:, idx], C[:, idx]
        bh = np.where(np.isnan(pct), cn, pct)  # after a delisting (or on a gap) the holder sits in cash
        first = okn.argmax(axis=0)
        bh[first, np.arange(len(need))] = 0.0  # no return on the first day
        bh = np.where(okn, bh, 0.0)
        w_b = fit_weights(bh, cn)
        raw_b, sc_b = blend_columns(bh, cn, 1.0), blend_columns(bh, cn, w_b)
        nd = okn.sum(axis=0)
        for k, s in enumerate(need):
            cached[s] = (_cagr_n(raw_b[-1, k], nd[k]), _mdd(raw_b[:, k]), _cagr_n(sc_b[-1, k], nd[k]))
    bh_cagr, bh_mdd, bh_dd = (np.array([cached[s][i] for s in syms]) for i in range(3))
    w_s = fit_weights(Rz, C)
    raw, sc = blend_columns(Rz, C, 1.0), blend_columns(Rz, C, w_s)
    cagr = np.array([_cagr_n(raw[-1, k], ndays[k]) for k in range(len(syms))])
    dd_cagr = np.array([_cagr_n(sc[-1, k], ndays[k]) for k in range(len(syms))])
    mdd = (raw / np.maximum.accumulate(raw, axis=0) - 1.0).min(axis=0)
    return pd.DataFrame({"symbol": syms, "years": ndays / 252, "trades": [int(trades_per.get(s, 0)) for s in syms],
                         "cagr": cagr, "mdd": mdd, "bh_cagr": bh_cagr, "bh_mdd": bh_mdd, "dd_weight": w_s,
                         "dd_cagr": dd_cagr, "bh_dd_cagr": bh_dd, "excess": dd_cagr - bh_dd})


def _cagr_n(final: float, days: int) -> float:
    """CAGR of an equity path that starts at 1 and ends at `final` after `days` returns (as target._cagr)."""
    return float(final ** (252 / max(days, 1)) - 1.0) if final > 0 else -1.0


def evaluate(panel: Panel, cfg: StrategyConfig, cache: SignalCache | None = None) -> dict:
    """Per-symbol and sleeve-portfolio results of one rule (DECISIONS 67)."""
    scfg = single_config(cfg, panel)
    res = run_backtest(panel, scfg, cache)
    rets, held = symbol_returns(panel, scfg, res, with_held=True)
    dates = rets.index
    cash = (panel.cash.reindex(dates).fillna(0.0) if panel.cash is not None else pd.Series(0.0, index=dates)).to_numpy()
    per = per_symbol(panel, rets, cash, res.trades["symbol"].value_counts())
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
