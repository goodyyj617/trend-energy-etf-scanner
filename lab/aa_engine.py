"""Asset-allocation backtest engine (weights, rebalancing, tactical overlays, leverage, mixes).

Rules (shown in the app as RULES_KO):
- Target weights are decided on the close of a decision day (month, quarter or year end) using
  data up to that close, and the portfolio is traded to them at the NEXT trading day's close.
- Between decisions the weights drift with prices. With a band, the trade is skipped when every
  weight is within `band` of its target.
- Overlays change the targets each month from the base weights: a trend filter (asset below its
  N-month moving average -> safe asset), absolute momentum (N-month return below cash -> safe),
  dual momentum (only the strongest asset, or safe), risk parity (inverse volatility) and a
  volatility target (scale the whole mix so its recent volatility equals the target).
- Leverage is daily-reset, like a leveraged ETF on the whole mix: exposure k means
  r = k * r_mix + (1 - k) * cash - max(k - 1, 0) * spread - ter.  Per-asset leverage (e.g. a 2x
  S&P 500 sleeve) applies the same formula to that asset's own daily return.
- `cost_bps` is charged on turnover (one side). Uninvested cash earns the panel's cash return.
- Prices are total-return indices. The app feeds a KRW panel of Korean-listed ETFs
  (lab/aa_korea.py: FX and hedging are inside the product returns, cash is the KRW base rate);
  the engine itself is currency-agnostic. The benchmark is `panel.benchmark` when the panel has
  one (KODEX 미국S&P500 for the KRW panel), otherwise SPY. No tax.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from .aa_assets import CASH, Preset, equity_codes, short
from .data import BENCHMARK, Panel

TRADING_DAYS = 252
RISK_PARITY_MIN_VOL = 0.01  # annualised; below this an asset is cash-like and keeps its base weight in risk parity
REBALANCE_LABELS = {"M": "매월", "Q": "매분기", "A": "매년", "none": "안 함 (매수 후 보유)"}
OVERLAY_LABELS = {
    "none": "없음 (정적 배분)",
    "sma": "추세 필터 · N개월 이동평균 아래면 안전자산",
    "absmom": "절대 모멘텀 · N개월 수익률이 현금보다 낮으면 안전자산",
    "dualmom": "듀얼 모멘텀 · 가장 강한 자산 하나만 (현금보다 약하면 안전자산)",
    "riskparity": "리스크 패리티 · 최근 N일 변동성의 역수로 비중",
    "voltarget": "변동성 목표 · 최근 63일 변동성을 목표(%)에 맞춰 노출 조절",
}
OVERLAY_PARAM = {"none": None, "sma": ("개월 수", 10, 1, 24), "absmom": ("개월 수", 12, 1, 24),
                 "dualmom": ("개월 수", 12, 1, 24), "riskparity": ("변동성 계산 일수", 63, 20, 252),
                 "voltarget": ("목표 변동성 (%/년)", 10.0, 2.0, 40.0)}
RULES_KO = """- 목표 비중은 **결정일(월·분기·연말) 종가**까지의 데이터로 정하고, **다음 거래일 종가**에 그 비중으로 맞춥니다 (미래 데이터 사용 없음).
- 결정일 사이에는 가격대로 비중이 흘러갑니다. 밴드를 주면 모든 자산이 목표 ± 밴드 안에 있을 때 거래를 건너뜁니다.
- overlay는 기본 비중을 **매월** 고쳐 씁니다.
  - 추세 필터: 자산의 월말 종가가 N개월 이동평균 아래면 그 비중을 안전자산으로.
  - 절대 모멘텀: 자산의 N개월 수익률이 현금의 N개월 수익률보다 낮으면 안전자산으로.
  - 듀얼 모멘텀: 선택한 자산 중 N개월 수익률이 가장 높은 하나만 보유. 그것도 현금보다 약하면 안전자산.
  - 리스크 패리티: 최근 N일 일간 변동성의 역수에 비례해 비중을 다시 매깁니다 (기본 비중은 자산 목록으로만 쓰임).
  - 변동성 목표: 기본 배분의 최근 63일 변동성이 목표보다 높으면 노출을 줄여 현금으로, 낮으면 레버리지 상한까지 늘립니다.
- 레버리지는 **일일 리셋**입니다 (전체 배분에 거는 레버리지 ETF와 같음). 노출 k일 때
  수익률 = k × 배분 수익률 + (1 − k) × 현금 수익률 − max(k − 1, 0) × 조달 스프레드 − 레버리지 상품 보수.
  자산별 배수(예: S&P500 2배)는 그 자산의 일간 수익률에 같은 식을 적용합니다. 변동성 손실(volatility decay)은 이 계산에 자연히 포함됩니다.
- 거래 비용(bp)은 회전율(한쪽)에 부과합니다. 남는 현금은 **원화 단기금리(한국은행 기준금리)**를 받습니다.
- 국내 ETF의 수익률은 상장 전 기간까지 미국 ETF(배당 재투자) + 원/달러 환율 + 두 나라 단기금리로 만듭니다.
  환노출 상품은 환율 변동을 그대로 받고, 환헤지(H) 상품은 환율 대신 ‘원화 금리 − 달러 금리 − 헤지 추가 비용’을 받습니다. 각 상품의 총보수도 뺍니다.
- **세금은 반영하지 않습니다** (ISA: 손익 통산 후 200만원(서민형 400만원) 비과세, 초과분 9.9% 분리과세).
"""


@dataclass(frozen=True)
class Overlay:
    kind: str = "none"
    param: float | None = None
    safe: str = CASH  # where the weight of an "off" asset goes
    # trend filter / absolute momentum: switch only these assets (empty = every risky asset; A38 filters equities only)
    only: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "only", tuple(self.only))  # a list read back from JSON

    def describe(self) -> str:
        if self.kind == "none":
            return "정적"
        p = OVERLAY_PARAM[self.kind]
        v = self.param if self.param is not None else p[1]
        name = OVERLAY_LABELS[self.kind].split(" · ")[0]
        safe = "" if self.safe == CASH else f", 안전자산 {short(self.safe)}"
        only = f", {'·'.join(short(s) for s in self.only)}에만" if self.only else ""
        return f"{name} {v:g}{only}{safe}"


@dataclass(frozen=True)
class Leverage:
    factor: float = 1.0  # exposure of the whole mix (1 = none). With a vol target this is the cap.
    spread_bps: float = 100.0  # borrowing cost above the cash rate, per year
    ter_bps: float = 60.0  # extra fee of leveraged products, per year, charged while k > 1
    per_asset: dict[str, float] = field(default_factory=dict)  # symbol -> daily-reset multiple

    def describe(self) -> str:
        parts = []
        if self.factor != 1.0:
            parts.append(f"전체 {self.factor:g}배")
        parts += [f"{short(s)} {m:g}배" for s, m in self.per_asset.items() if m != 1.0]
        return ", ".join(parts) if parts else "없음"

    @property
    def any(self) -> bool:
        return self.factor != 1.0 or any(m != 1.0 for m in self.per_asset.values())


@dataclass(frozen=True)
class AllocConfig:
    name: str
    weights: dict[str, float]  # symbol (or CASH) -> percent; normalised to sum 100 when run
    start: str
    end: str
    rebalance: str = "Q"  # "M" | "Q" | "A" | "none"
    band: float = 0.0  # percent points; 0 = trade at every decision
    cost_bps: float = 10.0
    overlay: Overlay = Overlay()
    leverage: Leverage = Leverage()
    cash_yield: bool = True
    rebalance_offset: int = 0  # trade this many trading days after the period's last day (timing-luck check, A27)

    def __post_init__(self):
        def native(v):
            return v.item() if hasattr(v, "item") else v

        object.__setattr__(self, "weights", {k: float(native(v)) for k, v in self.weights.items() if native(v) != 0})
        if isinstance(self.overlay, dict):
            object.__setattr__(self, "overlay", Overlay(**self.overlay))
        if isinstance(self.leverage, dict):
            object.__setattr__(self, "leverage", Leverage(**self.leverage))

    @property
    def assets(self) -> list[str]:
        """Symbols the simulation needs prices for (CASH excluded)."""
        syms = [s for s in self.weights if s != CASH]
        if self.overlay.safe != CASH and self.overlay.safe not in syms:
            syms.append(self.overlay.safe)
        return syms

    def describe(self) -> str:
        w = " · ".join(f"{short(s)} {v:g}" for s, v in self.weights.items())
        ov = self.overlay.describe()
        lev = self.leverage.describe()
        extra = [REBALANCE_LABELS[self.rebalance] + (f" 밴드 {self.band:g}%p" if self.band else "")]
        if ov != "정적":
            extra.append(ov)
        if lev != "없음":
            extra.append("레버리지 " + lev)
        if self.rebalance_offset:
            extra.append(f"기간 끝 +{self.rebalance_offset}거래일에 매매")
        return f"{w} | " + " | ".join(extra)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overlay"] = asdict(self.overlay)
        if not d["overlay"]["only"]:  # keep fingerprints of configs saved before the field existed
            d["overlay"].pop("only")
        else:
            d["overlay"]["only"] = list(d["overlay"]["only"])
        d["leverage"] = asdict(self.leverage)
        if not d["rebalance_offset"]:  # keep fingerprints of configs saved before the field existed
            d.pop("rebalance_offset")
        return d

    @staticmethod
    def from_dict(d: dict) -> "AllocConfig":
        known = {k: v for k, v in d.items() if k in AllocConfig.__dataclass_fields__}
        return AllocConfig(**known)

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]

    def parameter_count(self) -> int:
        """Free choices a researcher could have tuned (used by the simplicity axis)."""
        n = 1 if len(self.weights) > 1 else 0  # the weight vector itself
        n += 1 if self.rebalance != "Q" else 0
        n += 1 if self.band else 0
        if self.overlay.kind != "none":
            n += 1 + (1 if self.overlay.safe != CASH else 0) + (1 if self.overlay.only else 0)
        n += 1 if self.leverage.factor != 1.0 else 0
        n += sum(1 for m in self.leverage.per_asset.values() if m != 1.0)
        return n


@dataclass
class AllocResult:
    config: AllocConfig
    equity: pd.Series  # starts at 1.0
    benchmark: pd.Series  # SPY, normalised to 1.0 on the first day
    cash: pd.Series  # daily cash return on the same dates
    weights: pd.DataFrame  # effective weights at each close (columns: assets + CASH), before leverage
    exposure: pd.Series  # k, the daily exposure of the mix (1 = unlevered, fully invested)
    turnover: float  # one-way turnover per year, as a fraction of equity
    n_trades: int  # rebalancing events that traded
    sleeves: dict | None = None  # for a mix: {name: equity Series}
    warnings: list[str] = field(default_factory=list)

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.equity.index[0], self.equity.index[-1]


# --------------------------------------------------------------------------- helpers

def decision_rows(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Positions of the last row of each month / quarter / year within index."""
    if freq == "none":
        return np.array([], dtype=int)
    period = index.to_period({"M": "M", "Q": "Q", "A": "Y"}[freq])
    last = pd.Series(np.arange(len(index)), index=index).groupby(period).last()
    return last.to_numpy(dtype=int)


def _month_end_closes(close: pd.DataFrame) -> pd.DataFrame:
    rows = decision_rows(close.index, "M")
    return close.iloc[rows]


def levered_returns(r: pd.Series, k: float, cash: pd.Series, spread_bps: float, ter_bps: float) -> pd.Series:
    """Daily-reset leverage k on a return series: k*r + (1-k)*cash - max(k-1,0)*spread - ter(k>1)."""
    if k == 1.0:
        return r
    fin = max(k - 1.0, 0.0) * spread_bps / 1e4 / TRADING_DAYS
    ter = ter_bps / 1e4 / TRADING_DAYS if k > 1.0 else 0.0
    return k * r + (1.0 - k) * cash - fin - ter


def _normalise(w: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in w.values())
    if total <= 0:
        raise ValueError("비중의 합이 0입니다.")
    return {s: v / total for s, v in w.items()}


# --------------------------------------------------------------------------- target weights

class _Signals:
    """Month-end statistics an overlay needs, computed once per run."""

    def __init__(self, close: pd.DataFrame, cash: pd.Series):
        self.close = close
        self.me = _month_end_closes(close)
        cash_idx = (1.0 + cash.fillna(0.0)).cumprod()
        self.cash_me = cash_idx.iloc[decision_rows(close.index, "M")]
        self.rets = close.pct_change(fill_method=None)

    def above_sma(self, d: pd.Timestamp, months: int) -> pd.Series:
        hist = self.me.loc[:d]
        if len(hist) < months:
            return pd.Series(True, index=self.close.columns)  # not enough history: stay invested
        return hist.iloc[-1] > hist.iloc[-months:].mean()

    def momentum(self, d: pd.Timestamp, months: int) -> tuple[pd.Series, float]:
        hist = self.me.loc[:d]
        if len(hist) <= months:
            return pd.Series(np.nan, index=self.close.columns), float("nan")
        cash_hist = self.cash_me.loc[:d]
        return hist.iloc[-1] / hist.iloc[-1 - months] - 1.0, float(cash_hist.iloc[-1] / cash_hist.iloc[-1 - months] - 1.0)

    def volatility(self, d: pd.Timestamp, days: int) -> pd.Series:
        hist = self.rets.loc[:d].iloc[-days:]
        return hist.std() * np.sqrt(TRADING_DAYS)


def target_weights(base: dict[str, float], overlay: Overlay, sig: _Signals, d: pd.Timestamp) -> dict[str, float]:
    """Weights (fractions, incl. CASH) the overlay wants on decision day d."""
    w = dict(base)
    kind = overlay.kind
    if kind in ("none", "voltarget"):
        return w
    risky = [s for s in w if s != CASH]
    p = overlay.param if overlay.param is not None else OVERLAY_PARAM[kind][1]
    off: list[str] = []
    if kind == "sma":
        ok = sig.above_sma(d, int(p))
        off = [s for s in risky if not bool(ok[s])]
    elif kind == "absmom":
        mom, cash_mom = sig.momentum(d, int(p))
        if not np.isnan(cash_mom):
            off = [s for s in risky if mom[s] < cash_mom]
    elif kind == "dualmom":
        mom, cash_mom = sig.momentum(d, int(p))
        if not np.isnan(cash_mom):
            best = mom[risky].idxmax()
            keep = {best: 1.0} if mom[best] > cash_mom else {overlay.safe: 1.0}
            return keep
        return {s: 1.0 / len(risky) for s in risky}  # no history yet: equal weight
    elif kind == "riskparity":
        vol = sig.volatility(d, int(p))[risky]
        # cash-like assets (a CD-rate product: volatility ~0) keep their base weight; otherwise the
        # inverse of a near-zero volatility would put the whole portfolio into them
        cashlike = [s for s in risky if vol[s] == vol[s] and vol[s] < RISK_PARITY_MIN_VOL]
        volatile = [s for s in risky if s not in cashlike]
        v = vol[volatile]
        if volatile and v.notna().all() and (v > 0).all():
            inv = 1.0 / v
            scale = 1.0 - w.get(CASH, 0.0) - sum(w[s] for s in cashlike)
            w = {**{s: w[s] for s in cashlike}, **{s: float(inv[s] / inv.sum()) * scale for s in volatile}}
            if base.get(CASH):
                w[CASH] = base[CASH]
        return w
    if overlay.only:
        off = [s for s in off if s in overlay.only]
    for s in off:
        if s == overlay.safe:  # the safe asset is where weight goes; it is never switched off itself
            continue
        w[overlay.safe] = w.get(overlay.safe, 0.0) + w.pop(s)
    return w


# --------------------------------------------------------------------------- simulation core

def simulate(rets: pd.DataFrame, cash: pd.Series, targets: dict[int, dict[str, float]], cost_bps: float,
             band: float, exposure: pd.Series | None = None, spread_bps: float = 0.0, ter_bps: float = 0.0,
             first: dict[str, float] | None = None) -> tuple[pd.Series, pd.DataFrame, float, int]:
    """Daily simulation of a weight path.

    rets: daily returns of the risky columns (no CASH column); cash: daily cash return.
    targets: {row position of the decision day: weights incl. CASH}; traded at the next row's close.
    first: weights held from the first row (defaults to the first target, or all cash).
    exposure: daily k for leverage (None = 1).
    Returns (equity, weights DataFrame incl. CASH, annual one-way turnover, trades).
    """
    cols = list(rets.columns)
    R = rets.to_numpy(dtype=float)
    R = np.nan_to_num(R, nan=0.0)
    C = cash.to_numpy(dtype=float)
    n, m = R.shape
    k_arr = np.ones(n) if exposure is None else exposure.to_numpy(dtype=float)
    cost = cost_bps / 1e4
    fin = spread_bps / 1e4 / TRADING_DAYS
    ter = ter_bps / 1e4 / TRADING_DAYS

    def vec(w: dict[str, float]) -> np.ndarray:
        v = np.zeros(m + 1)
        for s, x in w.items():
            v[cols.index(s) if s != CASH else m] = x
        return v

    start_w = vec(first) if first is not None else (vec(targets[min(targets)]) if targets else vec({CASH: 1.0}))
    hold = start_w.copy()  # fractions of unlevered equity, incl. cash in the last slot
    unlev = 1.0  # the unlevered mix (what the weights describe)
    equity = 1.0
    eq_out = np.empty(n)
    w_out = np.empty((n, m + 1))
    pending: np.ndarray | None = None
    turnover = 0.0
    trades = 0
    for t in range(n):
        growth = np.append(1.0 + R[t], 1.0 + C[t])
        hold = hold * growth
        value = hold.sum()
        r_mix = value / unlev - 1.0 if unlev > 0 else 0.0
        if pending is not None:
            tgt = pending * value
            tr = np.abs(tgt - hold).sum() / 2.0  # one-way turnover of the trade
            if band <= 0 or (np.abs(hold / value - pending).max() > band / 100.0):
                value -= tr * cost
                r_mix = value / unlev - 1.0 if unlev > 0 else 0.0
                hold = pending * value
                turnover += tr / unlev if unlev > 0 else 0.0
                trades += 1
            pending = None
        unlev = value
        k = k_arr[t]
        r = k * r_mix + (1.0 - k) * C[t] - max(k - 1.0, 0.0) * fin - (ter if k > 1.0 else 0.0)
        equity *= 1.0 + r
        eq_out[t] = equity
        w_out[t] = hold / value if value > 0 else hold
        if t in targets:
            pending = vec(targets[t])
    years = max(n, 1) / TRADING_DAYS
    eq = pd.Series(eq_out, index=rets.index)
    weights = pd.DataFrame(w_out, index=rets.index, columns=cols + [CASH])
    return eq, weights, turnover / years, trades


# --------------------------------------------------------------------------- public API

def _common_window(panel: Panel, syms: list[str], start: str, end: str) -> tuple[pd.DatetimeIndex, list[str]]:
    close = panel.close[syms]
    first_valid = close.apply(lambda s: s.first_valid_index())
    warnings = []
    data_start = first_valid.max()
    if pd.isna(data_start):
        raise ValueError("선택한 자산의 가격 데이터가 없습니다.")
    begin = max(pd.Timestamp(start), data_start)
    if begin > pd.Timestamp(start):
        late = first_valid[first_valid == data_start].index.tolist()
        warnings.append(f"{', '.join(short(s) for s in late)}의 데이터가 {data_start.date()}부터 있어 시작일을 그날로 옮겼습니다.")
    idx = panel.close.index
    mask = (idx >= begin) & (idx <= pd.Timestamp(end))
    dates = idx[mask]
    if len(dates) < 30:
        raise ValueError("선택한 기간에 거래일이 30일 미만입니다.")
    return dates, warnings


def benchmark_of(panel: Panel) -> str:
    return getattr(panel, "benchmark", None) or BENCHMARK


def run_allocation(panel: Panel, cfg: AllocConfig) -> AllocResult:
    bench_sym = benchmark_of(panel)
    if bench_sym not in panel.symbols:
        raise ValueError(f"벤치마크 {bench_sym} 가격이 데이터에 없습니다.")
    syms = cfg.assets
    missing = [s for s in syms if s not in panel.symbols]
    if missing:
        raise ValueError(f"가격 데이터가 없는 자산: {', '.join(short(m) for m in missing)}")
    base = _normalise(cfg.weights)
    dates, warns = _common_window(panel, syms or [bench_sym], cfg.start, cfg.end)
    close = panel.close.loc[:dates[-1], syms].ffill()
    cash_full = (panel.cash if (panel.cash is not None and cfg.cash_yield) else pd.Series(0.0, index=panel.close.index))
    cash_full = cash_full.reindex(panel.close.index).fillna(0.0)
    rets_full = close.pct_change(fill_method=None)
    for s, mult in cfg.leverage.per_asset.items():
        if s in rets_full.columns and mult != 1.0:
            rets_full[s] = levered_returns(rets_full[s], float(mult), cash_full.loc[rets_full.index],
                                           cfg.leverage.spread_bps, cfg.leverage.ter_bps)
    rets = rets_full.loc[dates]
    cash = cash_full.loc[dates]

    # decision days: the overlay decides monthly, a static mix at its own frequency
    freq = "M" if cfg.overlay.kind != "none" else cfg.rebalance
    rows = decision_rows(dates, freq) + int(cfg.rebalance_offset)
    rows = rows[rows < len(dates) - 1]  # a decision on the last day has no next close
    sig = _Signals(close, cash_full.loc[close.index]) if cfg.overlay.kind != "none" else None
    targets: dict[int, dict[str, float]] = {}
    for i in rows:
        d = dates[i]
        targets[int(i)] = target_weights(base, cfg.overlay, sig, d) if sig is not None else dict(base)
    first = target_weights(base, cfg.overlay, sig, dates[0]) if sig is not None else dict(base)

    exposure = None
    lev = cfg.leverage
    if cfg.overlay.kind == "voltarget":
        exposure = _vol_target_exposure(rets, cash, base, dates, float(cfg.overlay.param or OVERLAY_PARAM["voltarget"][1]) / 100.0,
                                        lev.factor)
    elif lev.factor != 1.0:
        exposure = pd.Series(lev.factor, index=dates)

    eq, weights, turnover, trades = simulate(rets, cash, targets, cfg.cost_bps, cfg.band, exposure,
                                             lev.spread_bps, lev.ter_bps, first)
    bench = panel.close[bench_sym].loc[dates].ffill()
    bench = bench / bench.iloc[0]
    exp_series = exposure if exposure is not None else pd.Series(1.0, index=dates)
    return AllocResult(cfg, eq, bench, cash, weights, exp_series, turnover, trades, warnings=warns)


def preset_config(preset: Preset, start: str, end: str, **kw) -> AllocConfig:
    """The AllocConfig of a registered preset. A static preset is what it always was (quarterly, no overlay, the
    AllocConfig defaults); a preset with a rule (A38) carries its overlay and decides monthly. `kw` sets other fields."""
    if preset.rule is None:
        return AllocConfig(preset.name, preset.weights, start, end, **kw)
    r = preset.rule
    only = equity_codes(preset.weights) if r.equity_only else ()
    fields = {"rebalance": "M", "overlay": Overlay(r.kind, float(r.param), r.safe, only=only), **kw}
    return AllocConfig(preset.name, preset.weights, start, end, **fields)


def current_targets(panel: Panel, cfg: AllocConfig) -> tuple[dict[str, float], str]:
    """Target weights (fractions, may include CASH) the strategy holds after the latest month end,
    for the rebalancing calculator. Returns (weights, explanation)."""
    base = _normalise(cfg.weights)
    if cfg.overlay.kind == "none":
        return base, "정적 배분: 목표 비중 그대로"
    res = run_allocation(panel, replace(cfg, end=str(panel.last_date.date())))
    dates = res.equity.index
    d = dates[decision_rows(dates, "M")[-1]]
    if cfg.overlay.kind == "voltarget":
        k = float(res.exposure.iloc[-1])
        if k >= 1.0:
            return base, f"변동성 목표: 최근 노출 {k:.2f} (1 이상은 레버리지 상품 없이는 못 따라가므로 기본 비중으로 표시)"
        w = {s: v * k for s, v in base.items()}
        w[CASH] = w.get(CASH, 0.0) + (1.0 - k)
        return w, f"변동성 목표: {d.date()} 기준 노출 {k:.2f} → 나머지는 현금"
    syms = cfg.assets
    close = panel.close.loc[:dates[-1], syms].ffill()
    sig = _Signals(close, panel.cash.reindex(close.index).fillna(0.0) if panel.cash is not None else pd.Series(0.0, index=close.index))
    return target_weights(base, cfg.overlay, sig, d), f"{cfg.overlay.describe()}: {d.date()} 월말 신호 기준"


def _vol_target_exposure(rets: pd.DataFrame, cash: pd.Series, base: dict[str, float], dates: pd.DatetimeIndex,
                         target: float, cap: float, lookback: int = 63) -> pd.Series:
    """Monthly exposure k = min(cap, target / realised vol of the base mix), applied from the next day."""
    w = pd.Series({s: v for s, v in base.items() if s != CASH})
    mix = (rets[w.index] * w).sum(axis=1) + base.get(CASH, 0.0) * cash
    vol = mix.rolling(lookback, min_periods=lookback).std() * np.sqrt(TRADING_DAYS)
    k = pd.Series(1.0, index=dates)
    cap = max(cap, 1.0)
    rows = set(decision_rows(dates, "M").tolist())
    current = 1.0
    for i in range(len(dates)):
        k.iloc[i] = current
        if i in rows:
            v = vol.iloc[i]
            current = float(min(cap, target / v)) if np.isfinite(v) and v > 0 else 1.0
    return k


# --------------------------------------------------------------------------- mixes of strategies

@dataclass(frozen=True)
class MixConfig:
    name: str
    sleeves: tuple  # ((AllocConfig, weight percent), ...)
    rebalance: str = "M"
    cost_bps: float = 0.0  # sleeve-level costs are already inside each sleeve
    rebalance_offset: int = 0  # as AllocConfig.rebalance_offset, for the sleeve re-weighting days

    def describe(self) -> str:
        return (" + ".join(f"{c.name} {w:g}%" for c, w in self.sleeves) + f" | 슬리브 재조정 {REBALANCE_LABELS[self.rebalance]}"
                + (f" | 기간 끝 +{self.rebalance_offset}거래일에 매매" if self.rebalance_offset else ""))

    def to_dict(self) -> dict:
        d = {"name": self.name, "rebalance": self.rebalance, "cost_bps": self.cost_bps,
             "sleeves": [{"weight": w, "config": c.to_dict()} for c, w in self.sleeves]}
        if self.rebalance_offset:
            d["rebalance_offset"] = self.rebalance_offset
        return d

    @staticmethod
    def from_dict(d: dict) -> "MixConfig":
        return MixConfig(d["name"], tuple((AllocConfig.from_dict(s["config"]), float(s["weight"])) for s in d["sleeves"]),
                         d.get("rebalance", "M"), float(d.get("cost_bps", 0.0)), int(d.get("rebalance_offset", 0)))


def run_mix(panel: Panel, mix: MixConfig) -> AllocResult:
    """Combine strategies: each sleeve runs on its own, then the sleeves are held at fixed weights
    and rebalanced among themselves at `mix.rebalance`."""
    results = [run_allocation(panel, c) for c, _ in mix.sleeves]
    idx = results[0].equity.index
    for r in results[1:]:
        idx = idx.intersection(r.equity.index)
    if len(idx) < 30:
        raise ValueError("슬리브들의 공통 기간이 30거래일 미만입니다.")
    names = []
    for c, _ in mix.sleeves:  # unique column names
        n = c.name
        while n in names:
            n += " "
        names.append(n)
    sleeve_eq = pd.DataFrame({n: r.equity.loc[idx] for n, r in zip(names, results)})
    rets = sleeve_eq.pct_change(fill_method=None)
    cash = results[0].cash.loc[idx]
    w = _normalise({n: w for n, (_, w) in zip(names, mix.sleeves)})
    rows = decision_rows(idx, mix.rebalance) + int(mix.rebalance_offset)
    rows = rows[rows < len(idx) - 1]
    targets = {int(i): dict(w) for i in rows}
    eq, weights, turnover, trades = simulate(rets, cash, targets, mix.cost_bps, 0.0, None, 0.0, 0.0, dict(w))
    bench = results[0].benchmark.loc[idx]
    bench = bench / bench.iloc[0]
    cfg = AllocConfig(mix.name, {n: v * 100 for n, v in w.items()}, str(idx[0].date()), str(idx[-1].date()),
                      rebalance=mix.rebalance, cost_bps=mix.cost_bps)
    warns = [x for r in results for x in r.warnings]
    return AllocResult(cfg, eq, bench, cash, weights, pd.Series(1.0, index=idx), turnover, trades,
                       sleeves={n: r.equity.loc[idx] / r.equity.loc[idx].iloc[0] for n, r in zip(names, results)},
                       warnings=warns)


def mix_parameter_count(mix: MixConfig) -> int:
    return sum(c.parameter_count() for c, _ in mix.sleeves) + (1 if len(mix.sleeves) > 1 else 0)


def with_rebalance_offset(obj, offset: int):
    """The same strategy trading `offset` trading days after each period end (a mix shifts its
    sleeves and its own re-weighting days together). Used by the 날짜 견고 part of the final gate."""
    if isinstance(obj, MixConfig):
        return replace(obj, sleeves=tuple((replace(c, rebalance_offset=int(offset)), w) for c, w in obj.sleeves),
                       rebalance_offset=int(offset))
    return replace(obj, rebalance_offset=int(offset))
