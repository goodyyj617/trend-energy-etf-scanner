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

### Gate reset, tactical strategies and validation (2026-10-25, user request in the thread "자산배분 연구실 Gate 재설정과 검증 기능 확장")

A24. User request (2026-10-24 23:42 KST): replicate the important parts of a friend's allocation tool (12 screenshots:
    HAA+BAA mixes, real daily drawdown, cost deduction, rebalance-date and lazy-rebalance risk, a 15% holdout,
    252-split honest-vs-equal cross-validation, bootstrap / jackknife / DSR, holdings over time, ISA/CMA fill
    helper), reset the final gate around HAA+BAA aggressive 50:50, and check whether the MDD comes only from
    crises that a signal could avoid. Registered before the first run:
    - `lab/aa_tactical.py`: HAA (Keller & Keuning 2023), BAA aggressive G4 (Keller 2022), VAA aggressive G4
      (Keller & Keuning 2017), implemented from memory of the papers (rules in the module docstring) and
      blends of them on the target weights. Signals always use US ETF total-return closes sampled on the
      decision day of each month; VEA, BND and BIL are extended back with EFA, AGG and the T-bill index.
      Holdings are the US ETFs (USD run, for reproduction) or Korean products via `KRW_MAP` with a quality
      label (정확 / 근사 / 없음); DBC has no Korean product and goes to CD금리 (option: 금현물).
    - Execution as in A3 (decide at the close, trade at the next close). `same_close` (trade at the deciding
      close) exists only to reproduce published monthly backtests and is labelled 재현용 in the app.
    - Study script `scripts/aa_gate_reset_study.py`; KRW window 2008-01 – 2026-10 (DBC needs 12 months of
      history), plus 2005-01 for the static presets.
A25. Result of A24's reproduction (memo 2026-10-25_gate_reset_and_validation.md). USD 2008-01 – 2026-08:
    HAA 50 + BAA 50 9.7 % / −15.5 % / monthly Sharpe (rf 0) 1.07 with same-close trading vs the friend's
    10.3 % / −15.5 % / 1.09; with next-close trading 8.8 % / −15.9 % / 0.97. KRW with Korean products:
    9.6 % / −17.1 % (DBC → CD) and 10.2 % / −17.1 % (DBC → gold); HAA alone 10.2 % / −13.2 %; BAA alone
    8.8 % / −22.9 %. Of the blend's monthly target weight, 25 % has an exact Korean product, 68 % an
    approximate one (hedged, or MSCI World for VEA) and 7 % none (DBC).
A26. Final gate reset (user: "CAGR 기준을 조금 낮추고, MDD 기준은 −15% 선으로", "HAA+BAA와 비슷한 기준"):
    최종 Gate = CAGR >= 0.70 x S&P500(원화) CAGR over the same period AND MDD >= −15 % (absolute, daily).
    This replaces the final gate of A9 (CAGR >= 0.80 x, |MDD| <= 0.75 x |benchmark MDD|, about −22 % in KRW).
    The exploration gate is unchanged. A timing check ("날짜 견고": worst MDD over rebalance dates month end,
    +1, +5, +10 and +15 trading days >= −20 %) was first shown next to the gate as a reference; A30 moved it
    inside. `AllocGates(min_cagr_ratio=0.70, max_mdd_abs=0.15)`; `max_mdd_abs=None` restores the old MDD rule.
    On 2005-01 – 2026-10: 안정형 40 (8.6 % / −11.8 %) and 안정형 45 (8.8 % / −13.0 %) pass, 안정형 35 misses the
    CAGR bar (8.2 % vs 8.5 %), 내 AAA misses on MDD (−16.2 %); HAA (from 2007-03) passes but is timing-fragile
    (worst −34.3 %), HAA+BAA fails on MDD (−17.1 %). The CAGR bar moves with the window (S&P500 KRW 12.1 % from
    2005, 13.4 % from 2008), so a pass should be read with the period.
A27. Validation tools (`lab/aa_validate.py`, app page "전술 전략 · 검증", bookmark aalab://tactical), settings
    fixed before the first run: rebalance-date offsets {0, 1, 5, 10, 15} trading days after the period end
    (new `AllocConfig.rebalance_offset`, omitted from `to_dict` when 0 so saved fingerprints do not change) and a
    one-month delay for tactical rules; lazy rebalancing with the next trade 1 / 2 / 3 months later at 60 / 30 /
    10 %, 50 draws, seed 7; holdout = last 15 % of trading days, never used to choose; CSCV with 10 blocks
    (C(10,5) = 252 splits), Sharpe as the selection score, blends on a 10 % grid of up to 3 sleeves as
    daily-weighted averages, PBO = share of splits where the in-sample best lands in the bottom half out of
    sample; stationary block bootstrap with mean block 21 days, 1000 draws; leave-one-out over blend parts.
    Also: monthly-return heatmap on the research page, target asset-class chart and Korean-substitute table
    for tactical runs.
A28. Result of A27 (KRW, memo): static presets move by ±0.3 %p CAGR and ±0.5 %p MDD across rebalance dates;
    tactical rules swing widely (HAA+BAA MDD −16.0 % to −32.7 %, HAA −13.2 % to −34.3 %, worst at +10 days),
    matching the friend's own "실전 이탈 리스크" table. Lazy rebalancing: HAA+BAA CAGR median 9.1 %, MDD median
    −17.2 %, 10th percentile −19.8 %. Holdout 2023-11 – 2026-10: HAA+BAA Sharpe 0.77 → 0.47, 안정형 40 0.57 →
    1.19. Bootstrap 90 % intervals of HAA+BAA and 안정형 40 overlap (Sharpe 0.43–1.03 vs 0.40–0.98). CSCV over
    HAA, BAA, VAA, 안정형 40, 내 AAA, 영구, 60/40, S&P500 (2,276 blends): tuned beats equal weight in 48 % of
    splits, PBO 50 %, out-of-sample Sharpe medians 0.68 vs 0.67. Weight tuning is noise here; equal weights of
    chosen sleeves are the honest default.
A29. Crisis-avoidance study (user: "MDD가 특정 상황에만 생기는지, 미리 감지하고 피할 수 있는지, 피하면 더 좋은지";
    `lab/aa_signals.py`, `scripts/aa_crisis_signal_study.py`). Registered before the run: bases 안정형 40, 안정형 45,
    내 AAA (KRW, 2005-01 – 2026-10, quarterly); signals S&P500 below its 10-month average, below its 200-day
    average, TIP 13612U <= 0, any of SPY/VWO/VEA/BND 13612W < 0, VIX > 25, Baa−10y spread > its 12-month low
    + 0.5 %p, 10y−3m < 0, and the hindsight oracle (on inside the three crash windows) as the upper bound; when
    a signal is on at a month end, every equity product moves to CD금리 (variant: 미국달러선물) at the next close.
    Result: the three deepest drawdowns of every preset are 2008, 2020 and 2022, but the next ones are −9 to
    −10 % (including the drawdown in progress since 2026-06), so even the oracle only takes 안정형 40 from −11.8 %
    to −9.4 % (내 AAA −16.2 % → −10.2 %). Trend, VIX, credit and breadth-canary switches cut the 2008 drawdown
    (−11.8 % → −7.3 %) but cost 1.0–1.7 %p CAGR (about 2 %p since 2015) and leave the full-period MDD the same
    or worse through false alarms. The TIP canary is the only signal that raised CAGR and Sharpe (안정형 40
    8.6 % → 8.8 %, 0.67 → 0.82; 2022 drawdown −9.3 % → −4.7 %) but it does not touch 2020, so the MDD is
    unchanged; it is timing-sensitive (+10 days: 8.3 % / −14.1 %) and the rule was published after 2022.
    Dollar futures as the refuge were worse than CD. Decision: no switch is added to the presets; lowering
    the equity weight remains the reliable way to lower drawdown; the TIP canary stays an out-of-sample
    candidate to re-check later.
A30. 날짜 견고 joins the final gate (user's choice on 2026-10-25 00:31 KST: "넣기: 최악 MDD −20% 이내").
    최종 Gate = A26 AND the worst MDD over the five rebalance dates of A27 (month end, +1, +5, +10, +15 trading
    days; same period, no one-month delay) >= −20 %. Metric `timing_mdd_worst` from
    `lab/aa_validate.timing_worst_mdd`; a mix shifts its sleeves and its own re-weighting days together
    (`aa_engine.with_rebalance_offset`, new `MixConfig.rebalance_offset`, omitted from `to_dict` when 0). A
    result without the number (not computed) does not pass. The app computes it on the research page, the
    comparison page (cached per strategy and period), the scorecard and tactical KRW runs; comparisons trimmed
    to a common period judge it on that period. `AllocGates(max_timing_mdd_abs=None)` gives A26 alone.
    Why −20 %: wider than the −15 % MDD line so that static mixes (which move ±0.5 %p) are not failed by
    noise, tight enough to fail rules whose low MDD depends on the decision day (HAA −13.2 % at month end,
    −34.3 % at +10 days). Result (scripts/aa_gate_reset_study.py [8]–[9]): on 2005-01 – 2026-10 안정형 40
    (worst −12.0 %) and 안정형 45 (−13.1 %) still pass; HAA (from 2007-03), which passed A26, now fails; the
    others failed already. On 2008-01 – 2026-10 nothing passes (CAGR bar 9.4 %). Correction: the A26 table in
    the study script compared against `AllocGates()` after its defaults had changed, so its "기존" column
    showed the new rule; with the old rule HAA fails (10.2 % < 0.8 × 13.4 %). No other row changed.

### Global static search with MDD as the only cut (2026-10-25, registered before running)

A31. User request (2026-10-25 01:20–01:39 KST, thread "더 나은 자산배분 전략 찾기"): the lab's searches so far were local
    perturbations of 안정형 35/45 (audit: docs/research/allocation/2026-10-25_wider_search_design.md §1); search a much
    wider space, keep **full-period MDD ≥ −15 % as the only mandatory cut**, do not drop candidates on anything else
    (the user judges CAGR against MDD themselves), and show every other check as a column or a ⚠ flag.
    Correction recorded at the same time: under the A30 gate 안정형 45·금 26 (9.2 % / −13.7 %) and 골든 버터플라이
    (8.6 % / −14.7 %) also pass; the A30 note listed only the six strategies its script had run.
    Universe (30): all 28 products of A13/A22 plus two synthetic products, 미국10년 환노출 (IEF × USD/KRW; real products
    TIGER 미국채10년선물 305080 / KODEX 미국채10년선물 308620 to be verified before any preset) and 미국30년 환노출
    (TLT × USD/KRW; real product to be verified). Classes: 미국 주식 (379800 449180 379810 245340 463640 453630 453640
    218420), 그 밖 주식 (251350 291890 069500 280930), 환헤지 채권 (0091C0 484790), 환노출 채권 (468370 468630 + the two
    synthetic), 금·실물 (411060 132030 352560 182480), 현금·달러 (459580 261240), 레버리지 (225040 409820 418660 452250
    225130 122630, class total ≤ 20 %).
    Sampling (seed 2026): (1) class weights: 20,000 Sobol points; the leverage share is 0 in half of them and uniform
    on (0, 20 %] in the other half, the remaining share is Dirichlet(1) over the six other classes (uniform on the
    simplex, no other cap); (2) products: per class 1 / 2 / 3 products with probability 50 / 35 / 15 % (3 only in the
    two equity classes, never more than the class has), uniform picks, Dirichlet(1) split; 10 product draws per class
    point → 200,000 candidates; weights rounded to whole percent, parts under 3 % dropped, renormalised to 100 (largest
    remainder); (3) check sample: 50,000 portfolios of 3–10 products drawn from all 30 without class structure, same
    rounding; (4) refinement: the top 300 by CAGR among MDD-passing candidates are improved by moving 5 %p from one
    product to another (all pairs) until no neighbour is better, at most 3 rounds; refined candidates are flagged
    "다듬음" and listed apart from raw samples.
    Simulation: a static-only vectorised simulator (`lab/aa_search.StaticSim`) reproducing `run_allocation` (decide at
    the period's last close, trade at the next close, one-way cost 10 bp, drift between trades) within 1e-9 on synthetic
    panels (test) and within 1e-6 on the registered presets (run-time self-check); quarterly, 2005-01-03 – 2026-10-16,
    hedge extra 20 bp, product fees. Finalists are re-run with the engine.
    Columns per candidate (none drops a candidate): CAGR, MDD, excess Sharpe, Calmar, turnover, product count, class
    shares; 2005–2014 and 2015–2026 CAGR/MDD (⚠ when a half breaks −15 %); drawdowns inside the 2008 / 2020 / 2022
    windows of A22; CAGR/MDD from 2008-01 and 2010-01; holdout (last 15 % of days) CAGR/MDD; for the top 2,000: share of
    the 5-year rolling windows with MDD within −15 % and the median window CAGR (⚠ below 80 %); for the top 300: worst MDD
    over all ±5 %p neighbours (⚠ when it breaks −15 %) and the A30 timing check (offsets 0/1/5/10/15).
    Campaign-level dials (reported, not used to drop): overlap between the top lists on the full period and on
    2005-01 – 2023-07 (holdout removed) with the holdout CAGR/MDD of the latter; CSCV over monthly returns (10 blocks,
    252 splits, rule = best CAGR among candidates within −15 % MDD in the training half) giving PBO and, per candidate,
    the share of splits where it stays within −15 % MDD out of sample.
    Outputs: top 10 per MDD band (−10 … −15 %, 1 %p bands; ties within 0.3 %p CAGR shown together), a CAGR × MDD scatter
    (colour = equity share, marker = leverage), the full candidate CSV and a top-300 CSV, the registered presets placed
    in the same table. The user chooses; recommendations are one line per band and do not alter the tables.
    Stages 2–4 as in the design memo (finalist sensitivity; HAA/BAA with 4-tranche staggering, Korean-product signals
    and core/satellite mixes; real Korean ETF prices and 국고채 via the lab PC) follow in later entries.
    Run 2026-10-25 02:15 KST (seed 2026, 249,934 candidates, 40,992 within −15 %): memo 2026-10-25_global_search.md and
    folder 2026-10-25_global_search/. No preset was registered from it; the refined (다듬음) list is not a candidate list.

A32. Stage 2 of A31 (user's list, 2026-10-25 23:35 KST: "MDD −15%, CAGR 13.2%로 나오는 것도 포함, 나스닥 포함"):
    candidates R (best refined, 13.2 / −14.9), A (best raw sample), B1–B5 (best per band without leverage or synthetic
    products), plus the user's current 내 AAA and 안정형 40 for comparison. Each is re-run, one change at a time, with
    the engine (`run_allocation`, band 0 / 3 / 5 %p), monthly / annual rebalancing, 20 bp cost, the A30 date offsets,
    hedge cost 40 bp, starts 2008 / 2010 / 2015, every ±2 %p and ±5 %p neighbour (share within −15 %, worst MDD, CAGR
    median / min), and, where a synthetic unhedged treasury is held, with it replaced by the hedged product and by
    달러선물. Nothing is a cut; the memo reads the table. Script `scripts/aa_stage2.py`, outputs
    `2026-10-25_stage2/` (stage2.csv, neighbours.csv.gz), memo `2026-10-25_stage2.md`.
    Result: R and B1 break −15 % under almost every change (neighbours 27–56 % outside, band, monthly, date offsets;
    R also under product substitution) — edge overfits, not candidates. A holds under ±2 %p (91 %) and substitution
    but exceeds −15 % by 0.1–0.2 %p with a 3 %p band or date offsets. B2 (10.9 / −13.2) and B4 (9.5 / −11.7) stay
    within −15 % under every test. 내 AAA is outside the line in every test (−16.3 %, 3 % of ±2 %p neighbours inside).
    No preset registered; the user chooses.
