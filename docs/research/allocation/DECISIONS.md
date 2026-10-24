# Asset-Allocation Lab Decisions

Numbered A1, A2, … so they do not collide with `docs/research/trend_v2/DECISIONS.md`. All Accepted.

### Scope and engine (2026-10-24)

A1. The asset-allocation lab is a separate Streamlit app (`lab/aa_app.py`, port 8511, bookmark scheme
    `aalab://`) that reuses the trend lab's data layer (`lab/data.py`) and metric formulas
    (`lab/metrics.py`). It does not touch `lab/app.py`, `lab/engine.py` or the trend lab's trial log,
    so allocation runs do not inflate the trend lab's DSR trial count.
A2. Asset classes are represented by US-listed ETFs with the longest history (SPY, QQQ, IWM, DIA, MDY,
    EFA, EEM, TLT, IEF, AGG, LQD, TIP, GLD, IYR, XLE, XLU) in USD total return. Korean-listed products
    are listed as information only; FX, hedging cost, fees and tax are not modelled.
A3. Execution: targets are decided on the close of the decision day (month/quarter/year end) and traded
    at the next trading day's close. Weights drift between decisions. A band (percent points) skips a
    trade when every weight is within the band. Costs are charged on one-way turnover. Cash earns the
    panel's cash return (BIL, ^IRX/252 before 2007).
A4. Overlays decide monthly from the base weights: trend filter (asset below its N-month moving average
    of month-end closes -> safe asset), absolute momentum (N-month return below cash -> safe), dual
    momentum (hold only the strongest asset, or the safe asset when it trails cash), risk parity
    (inverse N-day volatility), volatility target (scale the whole mix to a target volatility, cash for
    the rest, the leverage factor as the cap).
A5. Leverage is daily-reset: with exposure k, r = k*r_mix + (1-k)*cash - max(k-1,0)*spread - ter(k>1).
    Defaults: spread 100 bp/yr over cash, extra fee 60 bp/yr while levered. Per-asset multiples apply the
    same formula to that asset's daily return. Volatility decay is therefore part of the result.
A6. A mix runs each sleeve on its own and holds the sleeves at fixed weights, restored at the mix's
    rebalance frequency. Mixes of mixes are not supported.

### Scorecard (2026-10-24, registered before the first scorecard was run)

A7. Nine axes, each scored 0-100 as the percentile rank within the compared set (average ranks for
    ties): 단순 (free parameters), 일관 (share of rolling 36-month windows beating cash), 위기 (mean
    of strategy return minus SPY return over the registered crisis windows), 꼬리 (CVaR 5% of monthly
    returns), 비용 (annual turnover x cost + leverage financing), 샤프 전반 / 후반 (excess-over-cash
    Sharpe in each half of the period), 최대낙폭, 칼마. Raw values are always shown next to the scores.
A8. Crisis windows (SPY peak to trough): 2000-03-24 → 2002-10-09, 2007-10-09 → 2009-03-09, 2011-04-29 →
    2011-10-03, 2018-09-20 → 2018-12-24, 2020-02-19 → 2020-03-23, 2022-01-03 → 2022-10-12. Only windows
    inside the compared period count.
A9. The composite "견고" is the weighted mean of the axis scores (default equal weights, user-adjustable)
    and is labelled reference only. Verdicts use gates, consistent with trend-lab decision 24: the
    exploration gate (Calmar >= SPY Calmar and |MDD| <= 0.75 x |SPY MDD|) and the final gate (CAGR >= 0.80
    x SPY CAGR and |MDD| <= 0.75 x |SPY MDD|). No trade-count gate: a static allocation has few trades by
    design.
A10. Presets registered now: the user's AAA (2026-06 weights, the 2025-09 weights, a gold-20 variant),
    SPY, 60/40, Dalio all-weather, permanent portfolio, golden butterfly (IWM for small-cap value, cash
    for short Treasuries), GTAA 5. MSCI World is looked through as 0.7 SPY + 0.3 EFA. Presets are not
    changed after results are seen; a new preset is a new registration.
A11. Notes: `docs/research/allocation/NOTES.md` holds three sections (실무에 바로 쓰는 발견, 정직한 경고,
    여기까지 오게 된 캠페인) as dated bullets. The app appends to it; other threads may edit it directly.
    Every app run is logged to `lab_results/aa_runs.csv` (not committed).
A12. Results of the 2026-10-24 review that motivated the lab (USD, 2005-01 – 2026-10): no static mix
    (8), dynamic overlay (7) or leverage variant (16) met CAGR > SPY and MDD < SPY at once. Record:
    docs/research/allocation/2026-10-24_all_weather_review.md and NOTES.md.
