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

### Korean-listed products in KRW (2026-10-24, user request: "ISA 계좌라 한국 상장 ETF만 살 수 있다")

A13. The lab's building blocks are Korean-listed products (`lab/aa_assets.py` `PRODUCTS`, 20 items: the
    user's six holdings, alternatives per asset class, cash/dollar products and six leveraged ETFs) and
    every result is a KRW return. This supersedes A2. Before listing (most products listed after 2015)
    each product is backed by a US proxy converted the way the product earns its return
    (`lab/aa_korea.py`): unhedged = USD return × USD/KRW change; hedged (H) = USD excess return over USD
    cash + KRW cash − extra hedge cost; KOSPI products = EWY converted to KRW; CD-rate ETF = KRW cash.
    Leveraged products are a daily-reset multiple of the excess return over the funding currency's
    cash rate. Every product carries its total fee (총보수) as a daily drag. The period choice
    "실제 상장일 이후만" starts at the latest listing date among the chosen products, i.e. without
    proxy data.
A14. Presets are re-registered as Korean products with the same intent as A10 (A10's USD mapping is
    retired, not edited): the user's AAA uses its actual six products; MSCI World is KODEX MSCI선진국
    (251350, proxy 0.7 SPY + 0.3 EFA as in A10); GTAA 5's EAFE leg becomes 251350; golden butterfly's
    small-cap leg is KODEX 미국러셀2000(H) (280930); short Treasuries / cash become KODEX CD금리액티브
    (459580); Dalio's commodity 7.5 stays folded into gold. Registered before any KRW result was seen.
A15. Benchmark = KODEX 미국S&P500 (379800, unhedged) in KRW; it replaces SPY in every gate, delta and
    crisis table (gate formulas unchanged, A9). Uninvested cash earns the Bank of Korea base rate.
    Consequence recorded in NOTES: the KRW benchmark's MDD (−29.6%, 2005–2026) is about half of USD
    SPY's, so the MDD gate is stricter than in the USD version.
A16. Data and model constants: USD/KRW = FRED DEXKOUS (noon buying rate in New York, same clock as
    US closes) committed as `lab/aa_usdkrw.csv.gz`, newer days spliced from Yahoo KRW=X when online;
    KRW cash = Bank of Korea base-rate step schedule `lab/aa_bok_rate.csv` (rows before 2018-11 marked
    "memory" until checked against ECOS); hedge cost = rate gap + 20 bp/yr (sidebar slider). Known gaps
    entered as product constants, not fitted: TIGER 미국S&P500레버리지 tracks the price index, so
    −3 %/yr for missing dividends ×2; 452250 tracks Ultra T-bond futures but is proxied by 2×TLT
    (duration understated); the KRX gold premium and tracking error are not modelled.
A17. Every research run is logged with its full configuration (`aa_runs.csv` gains a `config` JSON
    column; older files are migrated in place). The 전략 비교 page re-runs every logged and saved
    configuration once (duplicates by configuration, ignoring name and dates), optionally with presets,
    on one period (common-period toggle), and shows one table (CAGR, MDD, Sharpe, Calmar, volatility,
    worst 12 months, both gates), a CAGR–MDD map with gate lines, and overlaid equity/drawdown curves.
    Runs from the USD version are skipped because their assets are not in the KRW panel. Viewing the
    page is not logged again.
A18. Rebalancing calculator (`lab/aa_rebalance.py`): targets come from a preset, a saved strategy or
    the current research strategy (overlays: the target at the last month end). Modes: buy_only (no
    sells; new money to underweight products in proportion to their shortfall, any rest by target
    weight) and full (trade to target; within ±band for every product it falls back to buy_only).
    Share counts are truncated to whole shares at the entered price. Holdings are saved only to
    `lab_results/aa_holdings.json` (local, not committed). It is a calculator, not advice; tax and
    the account's cash buffer are the user's call.
A19. Leverage in the UI: real exposure is built by checking leveraged products (A13) and weighting
    them. The whole-mix virtual multiple (A5) stays in an expander labelled research-only, because no
    Korean product offers it. Per-asset virtual multiples are removed from the UI (the engine keeps
    the field so old saves still load).
