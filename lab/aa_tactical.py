"""Published tactical allocation rules (Keller family) for the allocation lab (DECISIONS A24–A25).

Signals are always computed on US ETF total-return prices, the way the papers define them and the way
a Korean investor would follow them; holdings are either those US ETFs (USD run, to reproduce the
published numbers) or the Korean-listed substitutes in `KRW_MAP` (KRW run, what an ISA can buy).

Rules as published (monthly; "price" = the close on the sampling day of each month):
- HAA (Keller & Keuning 2023, "Dual and Canary Momentum with Rising Yields/Inflation: Hybrid Asset
  Allocation", SSRN 4346906). Momentum 13612U = mean of the 1, 3, 6 and 12-month returns.
  Canary TIP: if its 13612U > 0, hold the top 4 of SPY IWM VEA VWO VNQ DBC IEF TLT at 25% each, and
  any of the four with 13612U <= 0 goes to the better of IEF and BIL instead. Otherwise 100% the
  better of IEF and BIL.
- BAA aggressive, "G4" (Keller 2022, "Relative and Absolute Momentum in Times of Rising/Low Yields:
  Bold Asset Allocation", SSRN 4166845). Canary SPY VWO VEA BND with 13612W = (12·r1 + 4·r3 + 2·r6
  + r12) / 4: if any is negative, go defensive. Offensive: the single best of QQQ VWO VEA BND by
  SMA(12) momentum = price / mean of the last 13 monthly prices − 1. Defensive: the best 3 of
  TIP DBC BIL IEF TLT LQD BND by the same momentum at 1/3 each; any of the three below BIL's goes
  to BIL.
- VAA aggressive, "G4" (Keller & Keuning 2017, SSRN 3002624). 13612W on SPY EFA EEM AGG: if all are
  positive, 100% the best of them; otherwise 100% the best of LQD IEF SHY.

History: VEA (2007-07), BND (2007-04) and BIL (2007-05) start late, so earlier months use EFA, AGG and
the panel's cash index (T-bill), rescaled to meet the real series (`spliced_close`). DBC starts
2006-02, so the first full 12-month signal is in 2007-02.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .aa_assets import CASH
from .aa_engine import AllocConfig, AllocResult, TRADING_DAYS, simulate

STRATEGIES = {
    "haa": "HAA (Keller 2023)",
    "baa_g4": "BAA 공격형 G4 (Keller 2022)",
    "vaa_g4": "VAA 공격형 G4 (Keller 2017)",
}
STRATEGY_TEXT = {
    "haa": "카나리아 TIP의 13612U 모멘텀이 양수면 SPY·IWM·VEA·VWO·VNQ·DBC·IEF·TLT 중 상위 4개를 25%씩(음수인 것은 IEF·BIL 중 나은 쪽으로), 음수면 IEF·BIL 중 나은 쪽 100%.",
    "baa_g4": "카나리아 SPY·VWO·VEA·BND 중 하나라도 13612W가 음수면 방어(TIP·DBC·BIL·IEF·TLT·LQD·BND 중 SMA12 모멘텀 상위 3개, BIL보다 약한 것은 BIL), 아니면 QQQ·VWO·VEA·BND 중 SMA12 모멘텀 1위 100%.",
    "vaa_g4": "SPY·EFA·EEM·AGG의 13612W가 모두 양수면 그중 1위 100%, 하나라도 음수면 LQD·IEF·SHY 중 1위 100%.",
}
SIGNAL_SYMBOLS = {
    "haa": ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT", "TIP", "BIL"],
    "baa_g4": ["SPY", "VWO", "VEA", "BND", "QQQ", "TIP", "DBC", "BIL", "IEF", "TLT", "LQD"],
    "vaa_g4": ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"],
}
SPLICE = {"VEA": "EFA", "BND": "AGG", "VWO": "EEM"}
ALL_SYMBOLS = sorted({s for v in SIGNAL_SYMBOLS.values() for s in v} | set(SPLICE.values()))
US_CLASS = {"SPY": "미국 주식", "QQQ": "미국 주식", "IWM": "미국 주식", "VEA": "해외·신흥 주식", "EFA": "해외·신흥 주식",
            "VWO": "해외·신흥 주식", "EEM": "해외·신흥 주식", "VNQ": "리츠", "DBC": "원자재", "GLD": "금",
            "IEF": "채권", "TLT": "채권", "TIP": "채권", "LQD": "채권", "BND": "채권", "AGG": "채권",
            "SHY": "현금성", "BIL": "현금성"}
CLASS_ORDER_US = ["미국 주식", "해외·신흥 주식", "리츠", "원자재", "금", "채권", "현금성"]  # late-starting ETF -> older sibling for earlier history

# US ticker -> (Korean product code or None, quality, note). Quality: 정확 = same index and currency
# exposure; 근사 = a related product (hedged where the US ETF is not, a wider index, ...); 없음 = no
# Korean-listed product; the KRW run then holds `missing_to` instead.
KRW_MAP = {
    "SPY": ("379800", "정확", "KODEX 미국S&P500 (환노출)"),
    "QQQ": ("379810", "정확", "KODEX 미국나스닥100 (환노출)"),
    "IWM": ("280930", "근사", "KODEX 미국러셀2000(H): 환헤지라 원화 약세 때의 완충이 없음"),
    "VEA": ("251350", "근사", "MSCI 선진국(World)은 미국이 약 70%라 '미국 외 선진국'과 다름. 국내에 미국 외 선진국 ETF 없음"),
    "EFA": ("251350", "근사", "MSCI 선진국(World)은 미국이 약 70%라 '미국 외 선진국'과 다름"),
    "VWO": ("291890", "근사", "KODEX MSCI EM선물(H): 환헤지"),
    "EEM": ("291890", "근사", "KODEX MSCI EM선물(H): 환헤지"),
    "VNQ": ("352560", "근사", "KODEX 미국부동산리츠(H): 환헤지"),
    "DBC": (None, "없음", "광범위 원자재 지수 ETF가 국내에 없음 (원유·농산물 개별 선물 ETF뿐)"),
    "IEF": ("0091C0", "근사", "KODEX 미국10년국채액티브(H): 환헤지"),
    "TLT": ("484790", "근사", "KODEX 미국30년국채액티브(H): 환헤지"),
    "TIP": ("468370", "정확", "KODEX iShares미국인플레이션국채액티브 (환노출)"),
    "LQD": ("468630", "정확", "KODEX iShares미국투자등급회사채액티브 (환노출)"),
    "BND": ("0091C0", "근사", "미국 종합채권 대신 미국10년국채(H)"),
    "AGG": ("0091C0", "근사", "미국 종합채권 대신 미국10년국채(H)"),
    "SHY": ("459580", "근사", "미국 단기국채 대신 CD금리(원화 현금)"),
    "BIL": ("459580", "근사", "달러 현금 대신 CD금리(원화 현금)"),
    "GLD": ("411060", "정확", "ACE KRX금현물 (원화 금값)"),
}


@dataclass(frozen=True)
class TacticalSpec:
    """A tactical strategy or a fixed-weight blend of several, decided once a month.

    parts: ((strategy key, percent), ...); blending happens on the target weights, so one portfolio
    trades the netted difference (same as monthly-rebalanced sleeves without double trading).
    offset: trading days after the month's last trading day on which prices are sampled and the
    decision is made (0 = month end, 1 = first trading day of the next month, 10 ≈ mid-month).
    delay: decisions are executed this many months late (the portfolio holds the older target).
    same_close: trade at the decision day's own close (the usual monthly-backtest convention, used
    only to reproduce published numbers); False trades at the next close, like the rest of the lab.
    """
    name: str
    parts: tuple = (("haa", 50.0), ("baa_g4", 50.0))
    start: str = "2008-01-01"
    end: str = "2026-12-31"
    cost_bps: float = 10.0
    offset: int = 0
    delay: int = 0
    same_close: bool = False
    missing_to: str = "459580"  # KRW run: where the weight of a US ETF without a Korean product goes
    skip_months: tuple = field(default=())  # decision months (period strings "YYYY-MM") left untraded

    @property
    def kinds(self) -> list[str]:
        return [k for k, _ in self.parts]

    def describe(self) -> str:
        parts = " · ".join(f"{STRATEGIES[k].split(' (')[0]} {w:g}%" for k, w in self.parts)
        timing = "월말" if self.offset == 0 else f"월말+{self.offset}거래일"
        extra = f", {self.delay}개월 지연" if self.delay else ""
        return f"{parts} | 매월 {timing}{extra}"

    def parameter_count(self) -> int:
        # the published rules count as fixed; the blend weights and the timing are the free choices
        return (1 if len(self.parts) > 1 else 0) + (1 if self.offset else 0)


# --------------------------------------------------------------------------- price preparation

def spliced_close(close: pd.DataFrame, cash: pd.Series | None) -> pd.DataFrame:
    """USD closes with late-starting ETFs extended backwards by their older sibling's returns, and
    BIL backed by the cash index before it trades."""
    out = close.copy()
    for new, old in SPLICE.items():
        if new in out.columns and old in out.columns:
            out[new] = _extend_back(out[new], out[old])
        elif old in out.columns:
            out[new] = out[old]
    if cash is not None:
        cash_idx = (1.0 + cash.reindex(out.index).fillna(0.0)).cumprod()
        out["BIL"] = _extend_back(out["BIL"], cash_idx) if "BIL" in out.columns else cash_idx
    return out


def _extend_back(series: pd.Series, older: pd.Series) -> pd.Series:
    first = series.first_valid_index()
    if first is None:
        return older.copy()
    o = older.ffill()
    if pd.isna(o.loc[first]):
        return series
    scaled = o * (series.loc[first] / o.loc[first])
    return series.where(series.index >= first, scaled)


def sampling_rows(index: pd.DatetimeIndex, offset: int = 0) -> np.ndarray:
    """Row positions of the month's last trading day shifted by `offset` rows (kept inside the index)."""
    period = index.to_period("M")
    last = pd.Series(np.arange(len(index)), index=index).groupby(period).last().to_numpy(dtype=int)
    rows = last + int(offset)
    return rows[(rows >= 0) & (rows < len(index))]


# --------------------------------------------------------------------------- momentum and rules

def _ret(m: pd.DataFrame, i: int, k: int) -> pd.Series:
    return m.iloc[i] / m.iloc[i - k] - 1.0


def mom_13612u(m: pd.DataFrame, i: int) -> pd.Series:
    return (_ret(m, i, 1) + _ret(m, i, 3) + _ret(m, i, 6) + _ret(m, i, 12)) / 4.0


def mom_13612w(m: pd.DataFrame, i: int) -> pd.Series:
    return (12 * _ret(m, i, 1) + 4 * _ret(m, i, 3) + 2 * _ret(m, i, 6) + _ret(m, i, 12)) / 4.0


def mom_sma12(m: pd.DataFrame, i: int) -> pd.Series:
    return m.iloc[i] / m.iloc[i - 12:i + 1].mean() - 1.0


def _add(w: dict, s: str, x: float) -> None:
    w[s] = w.get(s, 0.0) + x


def rule_weights(kind: str, m: pd.DataFrame, i: int) -> dict[str, float] | None:
    """Target weights (US tickers, sum 1) of one rule at monthly row i; None if history is too short."""
    if i < 12:
        return None
    syms = SIGNAL_SYMBOLS[kind]
    sub = m[syms]
    if sub.iloc[i - 12:i + 1].isna().any().any():
        return None
    w: dict[str, float] = {}
    if kind == "haa":
        mom = mom_13612u(sub, i)
        safe = "IEF" if mom["IEF"] > mom["BIL"] else "BIL"
        if mom["TIP"] <= 0:
            return {safe: 1.0}
        off = ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"]
        for s in mom[off].sort_values(ascending=False).index[:4]:
            _add(w, s if mom[s] > 0 else safe, 0.25)
        return w
    if kind == "baa_g4":
        canary = mom_13612w(sub, i)[["SPY", "VWO", "VEA", "BND"]]
        rel = mom_sma12(sub, i)
        if (canary > 0).all():
            return {rel[["QQQ", "VWO", "VEA", "BND"]].idxmax(): 1.0}
        dfn = ["TIP", "DBC", "BIL", "IEF", "TLT", "LQD", "BND"]
        for s in rel[dfn].sort_values(ascending=False).index[:3]:
            _add(w, s if rel[s] >= rel["BIL"] else "BIL", 1.0 / 3.0)
        return w
    if kind == "vaa_g4":
        mom = mom_13612w(sub, i)
        off = ["SPY", "EFA", "EEM", "AGG"]
        if (mom[off] > 0).all():
            return {mom[off].idxmax(): 1.0}
        return {mom[["LQD", "IEF", "SHY"]].idxmax(): 1.0}
    raise ValueError(f"unknown strategy {kind}")


def signal_table(sig_close: pd.DataFrame, spec: TacticalSpec) -> pd.DataFrame:
    """Monthly target weights in US tickers (rows: decision dates, columns: tickers), blended by spec.parts.
    Months where any part lacks 12 months of history are dropped."""
    needed = sorted({s for k in spec.kinds for s in SIGNAL_SYMBOLS[k]})
    missing = [s for s in needed if s not in sig_close.columns]
    if missing:
        raise ValueError(f"신호에 필요한 미국 ETF 가격이 없습니다: {', '.join(missing)}")
    rows = sampling_rows(sig_close.index, spec.offset)
    m = sig_close[needed].ffill().iloc[rows]
    total = sum(w for _, w in spec.parts)
    out = {}
    for i in range(len(m)):
        blend: dict[str, float] = {}
        ok = True
        for kind, pct in spec.parts:
            w = rule_weights(kind, m, i)
            if w is None:
                ok = False
                break
            for s, x in w.items():
                _add(blend, s, x * pct / total)
        if ok:
            out[m.index[i]] = blend
    table = pd.DataFrame(out).T.fillna(0.0).sort_index()
    return table[sorted(table.columns)] if len(table) else table


def map_weights(w: dict[str, float], holdings: str, columns: list[str], missing_to: str) -> dict[str, float]:
    """US-ticker weights -> weights over the holding panel's columns ("usd": identity, "krw": KRW_MAP)."""
    out: dict[str, float] = {}
    for s, x in w.items():
        if x == 0:
            continue
        if holdings == "usd":
            tgt = s
        else:
            code, _, _ = KRW_MAP.get(s, (None, "없음", ""))
            tgt = code if code is not None else missing_to
        if tgt not in columns and tgt != CASH:
            raise ValueError(f"보유 데이터에 {tgt}({s} 대체) 가격이 없습니다.")
        _add(out, tgt, x)
    return out


def class_table(table: pd.DataFrame) -> pd.DataFrame:
    """Monthly target weights summed by asset class (columns in CLASS_ORDER_US)."""
    cls = table.T.groupby(lambda s: US_CLASS.get(s, "기타")).sum().T
    return cls.reindex(columns=[c for c in CLASS_ORDER_US if c in cls.columns])


def krw_coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Share of each month's target weight that has an exact / approximate / no Korean product."""
    rows = {}
    for d, w in table.iterrows():
        acc = {"정확": 0.0, "근사": 0.0, "없음": 0.0}
        for s, x in w.items():
            acc[KRW_MAP.get(s, (None, "없음", ""))[1]] += float(x)
        rows[d] = acc
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- run

def run_tactical(hold_panel, sig_close: pd.DataFrame, spec: TacticalSpec, holdings: str = "usd",
                 sig_cash: pd.Series | None = None) -> AllocResult:
    """Simulate spec on hold_panel (USD ETF panel when holdings="usd", KRW product panel when "krw").

    sig_close: US ETF closes for the signals (raw; they are spliced here with sig_cash for BIL)."""
    from .aa_engine import benchmark_of

    sig = spliced_close(sig_close, sig_cash)
    table = signal_table(sig, spec)
    if spec.delay:
        table = table.shift(spec.delay).dropna(how="all")
    if spec.skip_months:
        skip = set(spec.skip_months)
        table = table[[str(d.to_period("M")) not in skip for d in table.index]]
    close_all = hold_panel.close.ffill()
    idx = close_all.index
    start = max(pd.Timestamp(spec.start), table.index[0] if len(table) else idx[-1])
    dates = idx[(idx >= start) & (idx <= pd.Timestamp(spec.end))]
    if len(dates) < 30:
        raise ValueError("시뮬레이션 기간이 30거래일 미만입니다.")
    mapped = {d: map_weights(w[w > 0].to_dict(), holdings, list(close_all.columns), spec.missing_to)
              for d, w in table.iterrows()}
    cols = sorted({s for w in mapped.values() for s in w if s != CASH})
    rets = close_all[cols].pct_change(fill_method=None).loc[dates]
    cash_full = hold_panel.cash.reindex(idx).fillna(0.0) if hold_panel.cash is not None else pd.Series(0.0, index=idx)
    cash = cash_full.loc[dates]
    targets: dict[int, dict[str, float]] = {}
    first = None
    for d, w in mapped.items():
        if d < dates[0]:
            first = w  # the latest decision before the window: held from day one
            continue
        if d > dates[-1]:
            break
        # the signal panel may hold a day the holding panel lacks (a holiday bar): act on the last day <= d
        i = int(dates.searchsorted(d, side="right")) - 1
        key = i - 1 if spec.same_close else i
        if key < 0:
            first = w
            continue
        if key < len(dates) - 1:
            targets[key] = w
    if first is None:
        first = targets[min(targets)] if targets else {CASH: 1.0}
    eq, weights, turnover, trades = simulate(rets, cash, targets, spec.cost_bps, 0.0, None, 0.0, 0.0, first)
    bench_sym = benchmark_of(hold_panel)
    bench = close_all[bench_sym].loc[dates]
    bench = bench / bench.iloc[0]
    avg = weights.mean()
    cfg = AllocConfig(spec.name, {s: round(float(v) * 100, 4) for s, v in avg.items() if v > 1e-6} or {CASH: 100},
                      str(dates[0].date()), str(dates[-1].date()), rebalance="M", cost_bps=spec.cost_bps)
    res = AllocResult(cfg, eq, bench, cash, weights, pd.Series(1.0, index=dates), turnover, trades)
    res.signals = table  # US-ticker targets, for the holdings chart and the coverage table
    res.spec = spec
    return res


def monthly_sharpe(equity: pd.Series) -> float:
    """Sharpe of monthly returns with a zero risk-free rate (the convention of the friend's tool)."""
    me = equity.groupby(equity.index.to_period("M")).last()
    r = me.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(12)) if len(r) > 2 and r.std() > 0 else float("nan")


__all__ = ["STRATEGIES", "STRATEGY_TEXT", "KRW_MAP", "ALL_SYMBOLS", "US_CLASS", "TacticalSpec", "run_tactical",
           "signal_table", "spliced_close", "class_table", "krw_coverage", "monthly_sharpe", "TRADING_DAYS"]
