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

### Lower-drawdown candidates (2026-10-24, registered before any of these were run)

A20. User request: "MDD를 줄이고 안정적인 수익" for the ISA allocation. Candidates, all Korean products,
    quarterly rebalancing, 10 bp cost, KRW, hedge extra 20 bp:
    - Equity share × defensive style grid. Equity sleeve keeps the held split S&P500 : MSCI선진국 :
      EM선물(H) ≈ 6 : 3 : 2. 현재형 defensive = 30y(H) : 10y(H) : 금현물 ≈ 10 : 25 : 10; 분산형 defensive
      = bonds : 금현물 : CD금리 ≈ 45 : 33 : 22 with 30y about a quarter of the bonds.
      E55-현재 (= 내 AAA) 30/15/10 | 30y 10, 10y 25, 금 10; E55-분산 30/15/10 | 30y 5, 10y 15, 금 15, CD 10;
      E45-현재형 25/12/8 | 12, 30, 13; E45-분산 25/12/8 | 7, 18, 금 18, CD 12;
      E35-현재형 19/10/6 | 15, 35, 15; E35-분산 19/10/6 | 8, 22, 금 20, CD 15.
    - E55-달러: 내 AAA with 10y 25 → 15 and KODEX 미국달러선물 10.
    - Rules on 내 AAA: risk parity (63 days), volatility target 8% (cap 1.0), trend filter 10 months →
      CD금리, quarterly check with a 5 %p band.
    - Reference presets: S&P500, 60/40, 올웨더, 영구, 골든 버터플라이, AAA 금 20, AAA 구배분.
    Periods: 2005-01-03 to the last date, and the halves 2005–2014 and 2015–end.
    Judging (fixed now): (1) return floor, full-period CAGR ≥ 내 AAA CAGR − 1.0 %p; (2) among those,
    lexicographic order by the worse of the two half-period MDDs, then full-period Calmar, then the
    longest underwater stretch. Gaps under 1 %p of MDD or 0.3 %p of CAGR are read as ties. Crisis-window
    returns, worst 12 months and rolling 36-month wins over cash are reported, not ranked. In-sample, no
    holdout; the result is a recommendation, not an approval.
A21. A20 result (2026-10-24, data to 2026-10-08; memo 2026-10-24_lower_drawdown.md): the rule ranks
    E35-분산 first (8.2% / −11.1%) and E45-분산 second (8.9% / −13.0%) against 내 AAA (9.1% / −16.2%). A
    check added after the run (201 rolling 5-year windows) shows E35-분산 giving up 1.4 %p a year in the
    median window, more than the 1 %p floor, so the recommendation to the user is E45-분산 (median −0.6 %p,
    MDD lower in 76% of windows), with E35-분산 as the option for a larger drawdown cut. Both are registered
    as presets `stable45`, `stable35` after the results were seen (selection bias noted in the note text).
    Overlays (risk parity, volatility target, trend filter) failed the return floor; the dollar-futures
    variant's gain came from 2008 alone. Script: `scripts/aa_lower_drawdown_study.py`.

### Broad search for a better low-drawdown allocation (2026-10-24, registered before running)

A22. User request after A21: look wider (more products, sector diversification, overlays, mixes, leverage
    products), judge on two things (drawdown inside S&P500's big crash windows, and a return not too far
    behind), and remove only clearly bad candidates rather than apply a strict gate.
    Products added to the catalog (facts checked 2026-10-24 on etfshopping.com / KODEX product list):
    KODEX 미국S&P500유틸리티 463640 (XLU), 필수소비재 453630 (XLP), 헬스케어 453640 (XLV), 에너지(합성)
    218420 (XLE), TIGER 미국다우존스30 245340 (DIA), TIGER 미국MSCI리츠(합성 H) 182480 (IYR), KODEX iShares
    미국인플레이션국채액티브 468370 (TIP), KODEX iShares미국투자등급회사채액티브 468630 (LQD). The KODEX
    sector series is taken as unhedged (the tech product's sheet says "KRW index, no hedge"; the others
    are not stated). XLV and XLP have no long cached history here, so they enter only the 2016+ check.
    Candidate families, all KRW, quarterly rebalance, 10 bp, hedge extra 20 bp, 2005-01-03 – 2026-10-08
    plus 2015-01-02 – end:
    F1 product swaps on 안정형 45 and 안정형 35: 5 and 10 %p of the nearest class moved into each new
       product (TIP, LQD from 10y(H); XLU, XLE, DIA from S&P500; REIT(H) from gold).
    F2 equity sleeve variants: S&P500 split with 유틸리티 / 다우 / 에너지 at 1/3 and 1/2 of the sleeve.
    F3 overlays on 내 AAA, 안정형 45, 안정형 35: trend filter {6, 10, 12} months × safe {CD, 10y(H), 금};
       absolute momentum {6, 12} × safe {CD, 금}; risk parity {63, 126} days; volatility target
       {6, 8, 10}% with cap 1.0 and, research-only, cap 1.5.
    F4 mixes (run_mix, monthly sleeve rebalance): 안정형 35 or 45 as the core (50–90 %) with one satellite
       (10–50 %) from: dual momentum S&P500/선진국/EM 12m → CD; S&P500 trend filter 10m → CD; 영구;
       S&P500 2× product 225040 held static; gold; and a few three-sleeve mixes.
    F5 return stacking as static weights: in 안정형 35/45 replace half of the S&P500 weight with
       TIGER 미국S&P500레버리지(합성 H) 225040 and give the freed weight to 금 or CD.
    Measures per candidate: CAGR, MDD, 위기 MDD = worst drawdown inside S&P500's crash windows
    (2007-10-09→2009-03-09, 2020-02-19→2020-03-23, 2022-01-03→2022-10-12, each measured from the
    strategy's level at the window start), CAGR gap to S&P500(원화), 2015+ CAGR and MDD, turnover.
    Lenient rule: a candidate is dropped only if some other candidate beats it on both full-period CAGR
    by ≥ 0.3 %p and 위기 MDD by ≥ 1 %p without a worse full-period MDD ("clearly dominated"). Survivors are
    shown sorted by 위기 MDD with the Pareto frontier marked; the 2015+ numbers are reported, not used
    to drop. Secondary check (not part of the ranking): the 2016-08 – 2026-07 ETF snapshot in KRW for
    sector/factor sleeves (XLV, XLP, XLU, VIG, USMV, XLK, XLF, XLI) inside 안정형 45, over 2018Q4, 2020
    and 2022 only. Still in-sample; presets registered from the result are marked as such.

A23. Result of A22 (2026-10-24, memo 2026-10-24_broad_search.md; script scripts/aa_broad_search.py).
    169 candidates ran on the KRW panel 2005-01-03 – 2026-10-08. The lenient rule dropped 78, 91 survived,
    30 sit on the CAGR × 위기-MDD frontier. The frontier is close to a straight line from 안정형 35
    (8.2 % / 위기 −11.1 %) through 안정형 45 (8.9 % / −13.0 %) to 내 AAA · 금 20 (10.0 % / −15.8 %):
    about 0.35–0.5 %p of CAGR per 1 %p of crisis drawdown. No candidate from any family left that line by
    more than ~0.2 %p, so nothing found here raises return at an equal crisis drawdown.
    Family findings: (F1) none of the eight new products moves CAGR by more than 0.1 %p; TIP/LQD/에너지
    swaps worsen MDD 0.3–1.5 %p, 유틸리티/다우 trims 0.2–0.3 %p at 안정형 35, 리츠(H) is clearly worse.
    (F2, and the 2016+ snapshot) defensive sectors behave like "less equity": CAGR and MDD fall together at
    a rate no better than lowering the equity weight; 테크 raises CAGR only because the sample is a tech
    bull market; 금융·산업재 are worse on both. (F3) every trend filter → CD/10y(H) is dominated (turnover
    100–200 %/yr); risk parity is dominated; 절대모멘텀 12m → CD has the smallest crisis drawdown of all
    (−9.1 %) but 95 %/yr turnover and 2015+ CAGR 6.8 %; 절대모멘텀 → 금 reaches 11 % CAGR with a 2008
    drawdown of −21 to −23 %; 변동성 목표 8–10 % on 안정형 35 is the only overlay with modest turnover
    (14.5 %) and cuts the 2008 drawdown to −7 to −9 % at a 0.4–0.6 %p CAGR cost, without helping 2020.
    (F4) the 영구 satellite on 안정형 45 lowers crisis drawdown at almost no return cost (70/30: 8.7 % /
    −11.8 %); the gold satellite raises CAGR with the 2008 drawdown growing 1.6 %p per 10 % of gold; the
    dual-momentum satellite is fine at 10 %; trend and 2× satellites are dominated. (F5) return stacking is
    no better than simply holding 안정형 45: the 2× product is hedged, so it also loses the KRW cushion
    that unhedged S&P500 gets in a crisis.
    Decision: register two presets from the result, both marked "결과를 본 뒤 등록": 안정형 40 (the static
    weights of 안정형 45 70 + 영구 30: S&P500 25 · 선진국 8 · EM 6 · 30년(H) 12 · 10년(H) 13 · 금 20 · CD 16,
    8.7 % / −11.8 %, 2015+ 8.6 % / −11.7 %) as the balance point, and 안정형 45 · 금 26 변형 (the static
    weights of 안정형 45 90 + 금 10: 9.3 % / −13.7 %, 2015+ 9.6 % / −12.4 %) as the higher-return option,
    with the standing warning that gold above 20 % leans on the post-2015 gold regime. Overlays stay a
    research option, not a recommendation. Everything remains in-sample; differences under 0.3 %p CAGR /
    1 %p MDD are treated as ties (A20).
    Engine fixes found by this run (both covered by tests): an overlay's safe asset that is also in the
    base weights was being switched off and re-added (weights summed to 2.0); risk parity gave the whole
    portfolio to a zero-volatility cash product. Now the safe asset is never switched off, and assets with
    annualised volatility below RISK_PARITY_MIN_VOL (1 %) keep their base weight.
