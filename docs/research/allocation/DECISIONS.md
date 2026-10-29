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

### Real Korean ETF prices after listing (2026-10-26)

A33. User (2026-10-26 00:37 KST, on learning that every allocation backtest so far was the US-proxy model): "우선 데이터
    받아와서 하자". Real daily prices of the 28 products and of 305080 / 308620 / 453850 (stand-ins for the two synthetic
    unhedged treasuries) were downloaded on the lab PC (Naver closes + Yahoo distributions, to 2026-10-23) and compared
    with the model; memo `2026-10-26_real_prices.md`, tables `2026-10-26_real_prices/` (products.csv, candidates.csv).
    Rules (fixed before the comparison was read): real total return r = (close + distribution on its ex-date) / previous
    close − 1; a product with a US underlying maps US date t to the first Seoul close strictly after t, 069500 / 122630 /
    459580 to the last Seoul close on or before t; the hybrid panel switches each product to real prices from the US date
    mapped to its 6th Seoul trading day (5 days after the first real close), scaled to meet the model there;
    SYN_IEF_U → 305080, SYN_TLT_U → 453850; 2005-01-03 – 2026-10-16 (USD/KRW end), quarterly, 10 bp.
    Result: candidates move CAGR ±0.25 %p and MDD 0–0.45 %p; R (−14.94 → −15.37) and B1 (−14.86 → −15.30) leave the
    −15 % line; A (−14.68), B2 (−13.23), B4 (−11.65), B3, B5, U2–U4, 안정형 40 and 안정형 45 · 금 26 stay inside; 내 AAA is
    outside on both (−16.25). Product monthly correlations 0.94–0.98 (gold spot 0.87); real single-product MDDs are often
    1–7 %p deeper than the model; real unhedged 10-year treasuries returned 1.2–1.9 %p a year less than the synthetic one
    with the same MDD. Most real windows are 1–5 years and contain no 2008 / 2020 crisis, so the crisis drawdowns are
    still the model's.
    Lab change: `lab/aa_real.py` (hybrid panel, `python -m lab.aa_real fetch` → `lab_data/kr/`, gitignored) and a sidebar
    choice '가격 기준' = 실제 가격 우선 / 모형만, applied to every page and to the search page's engine re-run; per-product
    real-price start dates in the sidebar and on '용어 · 상품 설명'; the stage-2 table on '전역 탐색 결과' gains
    '실제 가격 반영 CAGR / MDD' and the date from which every holding is real (columns, not tests). The default stays
    '모형만' (`DEFAULT_SOURCE`) until the user answers whether the hybrid becomes the default; if it does, stages 1–2
    are re-run on it on the lab PC. The recorded A31 / A32 numbers remain model numbers.

A34. User's choice on the decision card (2026-10-26 12:34 KST, "바꾸기"): the hybrid panel of A33 becomes the lab default
    (`lab/aa_real.DEFAULT_SOURCE = "real"`); '모형만' stays available in the sidebar. The app downloads the price files
    by itself when they are missing or older than 20 hours (`aa_real.ensure_files`; a download that covers fewer than 90 %
    of the codes never replaces the files; offline it keeps the old files or falls back to the model and says so).
    `scripts/aa_global_search.py` and `scripts/aa_stage2.py` take `--prices real|model` (default model, the registered
    A31 / A32 runs); stages 1–2 are re-run with `--prices real` into new folders, same seed, rules and candidates, and the
    A31 / A32 records are kept as the model reference.

A35. The A34 re-runs, done on the lab PC on 2026-10-26 with the A33 price files (same SHA-256), same seed 2026, sample
    sizes, period, rules and universe; only `--prices real` differs. Outputs `2026-10-26_global_search_real/` (stage 1 +
    representatives), `2026-10-26_stage2_real/` (stage 2 on the A32 candidates), `2026-10-26_stage2_real_newreps/` (stage 2
    on this run's own representatives, read as optimistic: picked and tested on the same data), memo
    `2026-10-26_real_reruns.md`; the 250k-row table stays off the repo (Drive lab_exchange/aa_real_20261026/). The
    stage-2 base rows reproduce A33 to two decimals. The page '전역 탐색 결과' shows this run by default
    (`aa_search_results.RESULT_SETS`, radio '기록된 결과'); the A31 / A32 model records stay selectable and unchanged.
    `scripts/aa_global_search_reps.py` gains `--prices` (the run patched `load` to the same effect). Nothing is promoted:
    the frontier moved 0–0.2 %p, B2 / B4 pass every stage-2 MDD test as before, A breaks the same three by 0.1–0.3 %p,
    R and B1 are out from the base setting on. New representatives lean on KODEX 200, whose real total return beats the
    model by 3.7 %p/yr (A33); no preset from them until that gap is understood.

A36. Three search presets, registered after the results at the user's request (2026-10-26 18:00 KST, "B2, B4, A 모두 등록"):
    `gs_b2` 탐색 B2 (= stage-2 B (−14,−13]), `gs_b4` 탐색 B4 (= B (−12,−11]) and `gs_a` 탐색 A, whose synthetic
    '미국10년 환노출' 8 % is merged into 미국10년국채액티브(H) 0091C0 (the stage-2 '가상→(H)' substitution) so every
    holding is buyable in the ISA. Before registering, the three ran through the same stage-2 tests on the real-price
    panel (lab PC, `2026-10-26_stage2_presets/`): B2 and B4 stay within −15 % in all 12 MDD tests (B4 also for every ±5 %p
    neighbour); the substituted A is 12.06 % / −14.48 % but breaks the line at band 3 %p (−15.61 %) and keeps only 53 % of
    its ±5 %p neighbours inside, so its preset note says it is the least robust of the three. Presets are frozen records.
    Same PR: the page '실험 기록' (`lab/aa_experiments.py`, bookmark `aalab://experiments`) per the user's rule of the same
    day that every experiment shows why it was run, its key results and graphs; it covers A31, A32, A33, A35 and A36, and
    each later experiment adds an entry in its own PR.

A37. Structure check around B2 · B3 · B4, asked by the user (2026-10-26 23:22 KST: raise the trust in their shared
    structure, "a finer search near B2, B3, B4"); scope "①–④ 모두 (닷컴 포함)" picked on the decision card. Picking a
    better point nearby was not done (A31's PBO 0.87 says that is the over-fitting step); the structure was tested
    instead, with reading rules fixed in the design doc before the run. Universe = the A31 B subset minus KODEX 200
    (21 products; its real−model gap is unexplained), whole percents, ≥ 3 %, ≤ 10 products, quarterly, 10 bp, real-price
    first, 2005-01-03 – 2026-10-16, seed 2027; code `lab/aa_local.py`, `scripts/aa_local_study.py`, tests
    `tests/test_aa_local.py`; results `2026-10-27_local_study/`, memo `2026-10-27_local_study.md`.
    ① neighbour map (5,000 random L1 neighbours per centre up to 30, plus same-role swaps); rule: ≥ 80 % of L1 ≤ 10
    neighbours within −15 %. Result 100 % for all three; passed. Four same-role swaps break the line (B2 gold futures(H)
    → spot −17.10, staples → utilities −15.13; B3 CD → 10-yr(H) −15.83; B4 CD → dollar futures −15.52).
    ② structure rules R1 equity 35–50 % with nasdaq ≥ 40 % of it, R2 gold 10–35 %, R3 30-yr(H) ≤ 6 % and no REITs;
    20,000 family members vs 20,000 two-level random allocations of the same products; paired ablations X1 nasdaq →
    S&P500, X2 gold → other safe holdings, X3 CD + 10-yr(H) → 30-yr(H), X4 defensive sectors → S&P500. Rule: family
    share within −15 % ≥ 2 × random; a feature whose removal moves the share < 5 %p is 'coincidental'. Result 32.6 % vs
    29.7 % (1.10 ×): failed. The family buys return (median CAGR 10.9 vs 7.4 %), not safety; X3 (40.3 → 9.0 %) and X4
    (50.3 → 37.0 %) carry the safety, nasdaq lowers it (+4.8 %p when removed) and gold is coincidental (−1.8 %p); both
    add CAGR (+1.6 / +1.7 %p).
    ③ in-family CSCV PBO (rule < 0.5) 0.40 vs random 0.61; 2005–15 vs 2016–26 CAGR rank correlation (rule ≥ 0.5) 0.58:
    passed. Shown beside it: MDD rank correlation 0.28; 252-day circular block bootstrap (1,000 draws) P(MDD < −15 %)
    B2 17.8 %, B3 46.5 %, B4 10.7 %, 안정형 40 9.1 %, 내 AAA 85.4 %.
    ④ dot-com: proxies that start after 2000-09 are spliced (close only) with the series fixed before the run, chosen by
    the best monthly correlation over the proxy's first 36 months (IEF←VFITX, TLT←VUSTX, LQD←VFICX, TIP←VIPSX,
    GLD←GC=F, EFA←VGTSX, EEM←VEIEX; FSIIX / VDMIX unavailable). Rule: B2 / B3 / B4 MDD from 2000-09 ≥ −15 %, plus the
    family share; ④ weighs most. Result B2 −16.4 %, B3 −30.8 %, B4 −25.6 % (trough 2002-07-23); family 0.4 % within vs
    random 24 %: failed. After the first run the comparison list was widened to every registered preset (information,
    not part of the rule); within −15 % from 2000-09 are only 안정형 35 (−11.7), 영구 (−12.0), 안정형 40 (−13.7) and
    골든 버터플라이 (−14.8); 탐색 A is −21.6. The re-run re-downloaded the mutual-fund histories, which moved the
    drawdown / CAGR columns by ~1e-7; counts and shares are identical.
    Consequence fixed by the rule: the notes of presets `gs_b2`, `gs_b4` and `gs_a` now say "2005년 이후에만 확인됨" with
    their dot-com MDD (weights unchanged; presets stay frozen records). Nothing is promoted or demoted beyond that; the
    next step (a pre-registered trend overlay, a nasdaq cap, or accepting the label) is the user's call. Caveats: all
    of 2000–2004 is model (proxies × KRW, BOK 'memory' rates for the hedges), the bond substitutes ran 1.2–2.6 %p/yr
    below their ETFs in the overlap (so bond-heavy rows are if anything penalised), Mar–Aug 2000 is missing, and every
    number is in-sample. Same PR: experiment entry 'a37' on '실험 기록' with five charts.

A38. 10-month trend filter on the equities of B2 · B3 · B4, picked by the user on the next-direction card after A37
    ("추세 필터를 얹어 시험") and run as fixed in the design doc (https://claude.ai/artifact/3PkTmzQJdQESFNhWJvmPne,
    approved "이대로 실행"). Weights unchanged. Filter: at each month end an equity product (nasdaq-100, S&P500 /
    S&P500(H), Dow, the four sectors, MSCI World, EM futures(H), Russell 2000(H)) whose KRW month-end close is below the
    mean of its last 10 month-end closes moves its base weight to KODEX CD금리액티브 (459580) for the month; gold, bonds,
    REITs, CD and dollar futures are never switched; fewer than 10 month ends = invested; the signal reads history
    before the start. Engine: `Overlay.only` (empty = every risky product, as before; saved runs read unchanged);
    with an overlay the engine re-weights monthly, so '정적 월' is the like-for-like reference. Settings 정적 분기 /
    정적 월 / 필터 8·10·12 (equities only) / Faber 원형 10 (every risky product, reference); periods 2005-01-03 and
    2000-09-01 (A37 ④ panel, model before 2005) to 2026-10-16, real-price first, 10 bp, rebalance-date offsets as A30;
    samples = the first 2,000 of A37 ②'s family and random draws. Code `scripts/aa_trend_filter_study.py`, tests
    `tests/test_aa_trend_filter.py`; results `2026-10-27_trend_filter/`, memo `2026-10-27_trend_filter.md`.
    Rules: point = B2 / B3 / B4 with 필터 10 inside −15 % in both periods; robust = also with 8 and 12; structure =
    ≥ 50 % of the family sample inside in both periods with 필터 10; allocations passing point and robust are proposed
    as '필터 판' presets, registration is the user's call, the static presets stay. CAGR shown, never judged.
    Results: point and robust passed for all three. From 2000-09 MDD B2 −16.4 → −13.0, B3 −30.8 → −13.4, B4 −25.6 →
    −11.4 % (dot-com window −4.7 / −5.3 / −3.7 %); with the filter the deepest drawdown is 2008 in both periods. 8 and
    12 months stay inside (worst −13.4 %), date-offset worst −13.1 / −14.1 / −11.5 %. From 2005 MDD hardly moves
    (−13.2 / −12.9 / −11.7 → −13.0 / −13.4 / −11.4); no help in 2008 (B3 deeper), 1.5–2.1 %p shallower in 2020.
    Cost: CAGR 2005–2026 10.7 → 8.9, 10.0 → 8.2, 9.7 → 8.0 %; 2016–2026 −2.1 to −2.5 %p; 2000-09–2026 −0.6 / −0.1 /
    −0.1 %p. About 20 % of the equity weight sat in CD on average; the switched-off set changed 95 / 58 / 68 times in
    21 years; turnover 0.08–0.09 → 0.73–0.87 a year. Structure failed: family 45.85 % inside in both periods (static
    0.3 %), random + filter 50.25 % (static 22.7 %); the family still buys CAGR (median 9.2 vs 6.3 % from 2005). After
    the cost the filtered B's sit beside 안정형 40 static (8.7 / −11.8 from 2005, 8.0 / −13.7 from 2000-09, 9.3 % from
    2016) rather than above it. S&P500 alone with the filter: −45.4 → −30.7 % from 2000-09 but −29.6 → −30.7 % from
    2005, CAGR 12.2 → 7.3 %, date-offset worst −39.4 %. Consequence: B2 · B3 · B4 '필터 판' proposed as preset
    candidates per the rule; the user decided on 2026-10-27 not to register them (static presets unchanged).
    Caveats: 10 months is Faber's value from a sample that contains 2000–02, so the dot-com pass is not an independent test of the filter length; 2000–2004 is all model and Mar–Aug 2000 is
    missing; the filtered versions have not had an A32-style stage-2 shake (band, cost, start year, neighbours);
    in-sample. Same PR: experiment entry 'a38' on '실험 기록' with four charts and PNGs in `2026-10-27_trend_filter/figures/`.

### B2–B5 composition and canary signals (2026-10-27, user request 10:04 KST; design https://claude.ai/artifact/7AyPtvA6372TwXaSJFYxPq, approved "둘 다 이대로 진행" 11:09 KST)

A39. Is the B2–B5 composition itself sound? User: "가장 보고 싶은 건, 현재 B2~B5 의 종목 구성이 정말로 괜찮은 것인가야.
    나스닥은 다 들어있는데 나머지는 꽤 달라서 그래." B2–B4 as A37 (B5 = stage-2 row `B 레버·가상 없음 (-11,-10]`: nasdaq-100
    32 · KODEX 200 4 · gold futures(H) 17 · CD 35 · dollar futures 12); static, quarterly, 10 bp, real-price first,
    2005-01-03 – 2026-10-16, ①–③ also from 2000-09-01 on the A37 ④ proxies. Code `lab/aa_composition.py`, runner
    `scripts/aa_composition_study.py`, tests `tests/test_aa_composition.py`; results `2026-10-27_composition/`, memo
    `2026-10-27_composition.md`. Rules fixed before the run: ① leave one out (pro rata / to CD; summary = pro rata),
    per half 2005–15 / 2016–26: 제 몫 = both halves MDD ≥ 0.5 %p deeper or CAGR ≥ 0.3 %p lower, 있으나 마나 = both
    move less, 빼면 나아짐 = no half worse by those and one better, else 기간 따라 다름; ② swap the rest (row nasdaq
    weight + column's other products, 4×4): 바꿔 써도 됨 = all 12 off-diagonal cells inside −15 % and none > 0.5 %p
    below its row (2005–2026; 2000-09 information); ③ B 평균 (mean of the four, whole percents by largest remainder)
    and a skeleton grid nasdaq-100 {20,25,30,35} × KRX gold {10,…,35} × CD rest: 같은 일을 한다 = inside −15 % and CAGR
    ≥ min of B2–B5 (9.28 %); if B 평균 passes it is proposed as a preset candidate and the user is asked whether to add
    it to A40; ④ the A31 real-price sample without leverage / synthetic (71,031) re-scored per half, top 100 by CAGR
    inside −15 % on one half: 다시 뽑힘 = median nasdaq and gold shares within the B2–B5 span (19–33, 12–35); ⑤ A33
    products table: 모형 경고 = |real − model| ≥ 1 %p/yr or monthly corr < 0.95.
    Results: ① nasdaq-100 is 제 몫 in all four (out → CAGR −1.6 / −3.6 / −3.4 / −3.8 %p); 빼면 나아짐 only for small
    slices (B2 health care 6: CAGR −0.17, MDD unchanged; B3 TIPS 7 and B4 IG corporates 8: CAGR +0.3–0.4, MDD
    0.35–0.4 %p deeper); B2's gold futures 35 is 기간 따라 다름 in 2005–2026 but out of it the 2000-09 MDD goes −16.4 →
    −32.0 %. ② 나머지에 의미 있음: 0 of 12 swaps outside −15 % but 5 lose > 0.5 %p; read by column the rest is a dial
    (B2's rest 11.6–11.9 % / −14.4 to −14.9 %, B5's rest −8.2 to −11.2 % with the lowest CAGR); from 2000-09 11 of 12
    are outside. ③ B 평균 = nasdaq-100 29 · staples 8 · health 1 · MSCI World 3 · KODEX 200 1 · 10y(H) 5 · 30y(H) 1 ·
    TIPS 2 · IG 3 · KRX gold 7 · gold futures(H) 13 · CD 23 · dollar 4 → 10.01 / −11.91 (2000-09: 8.23 / −24.2):
    같은 일을 한다; 8 of 24 skeletons too (nasdaq 25–35 · gold 20–35, e.g. 30 · 30 · 40 = 10.00 / −12.75, 35 · 30 · 35 =
    10.72 / −13.48); from 2000-09 every skeleton with nasdaq ≥ 25 is outside (−17.5 to −29.8), the nasdaq-20 ones inside
    (−12.8 to −14.0) but at 6.6–9.0 % from 2005. ④ chosen on 2005–15: 다른 구성 (nasdaq 0, defensive 46.5, gold 6; on
    2016–26 MDD median −18.8 %, 5 % inside); chosen on 2016–26: 다시 뽑힘 (nasdaq 20.5, gold 32; on 2005–15 24 % inside);
    B2 ranks 146 / 25,151 and 279 / 16,873, B3–B5 5,916–11,828 on 2005–15 but 114–282 on 2016–26. ⑤ 11 of 13 모형 경고
    (bonds real > model 3.0–4.4 %p/yr over 1.1–3.0 years, sectors +1.3–2.0, KODEX 200 +3.7, gold futures −1.4, KRX gold
    corr 0.87, CD and dollar futures by correlation); only nasdaq-100 (+0.48) and MSCI World (+0.14) 괜찮음.
    Consequence: B 평균 proposed as a preset candidate per the rule and the user asked about A40; on 2026-10-27 the user chose '등록 없이 기록만' (not registered, not added to A40);
    skeletons are information only; presets `gs_b2` / `gs_b4` unchanged. Caveats: most of 2005–listing is model (⑤),
    2000–2004 all model with Mar–Aug 2000 missing, the ③ threshold is set by search-picked points, in-sample. Engine
    check 3.4e-14. Same PR: experiment entry 'a39' on '실험 기록' and PNGs in `2026-10-27_composition/figures/`.

A40. Canary signals on B2–B5, weights unchanged. User: "카나리아 자산군 도입으로 동적 자산배분 … 최소 8가지 이상", and "매매는
    정기 리밸런싱에 더하지 말고, 신호가 바뀐 순간 기준으로 … 추후에 다시 1개월 재매매 금지 규칙". Sleeves: N nasdaq-100,
    D staples · health care (half each when neither is held), O MSCI World / KODEX 200 (never moved), S everything
    else (bonds, gold, CD, dollar). Actions: nasdaq → D in its held ratio ('defensive'), nasdaq → S pro rata ('safe');
    K1 moves half / all for one / both canaries negative; D2 / D3 tilt 10 %p from the weaker to the stronger sleeve.
    Signals (centre, two neighbours): K1 DAA (VWO · BND 13612W), K2 VAA · BAA (SPY · EFA · EEM · AGG 13612W), K3 HAA
    (TIP 13612U), K4 SPY < 200-day mean, K5 SPY 12-month total return < T-bill, K6 BAA10Y > 200-obs mean, K7 T10Y3M <
    0, K8 VIX ≥ 25, K9 ≥ 5 of 9 US sectors below 200-day, K10 Kritzman–Li turbulence of SPY · EFA · EEM · AGG > 3-year
    80th pct, K11 USD/KRW > 200-day; D1 nasdaq-100 KRW < 200-day, D2 nasdaq vs D 12-month KRW, D3 equity vs S 12-month
    KRW. Canaries in USD (US ETFs, FRED one day late via `macro.on_dates`); late ETFs chained to same-index funds
    (VWO / EEM ← VEIEX, BND / AGG ← VBMFX, EFA ← FSPSX ← VGTSX, TIP ← VIPSX; FSIIX failed to download). Trading: state
    at the close of t, a change trades at the t+1 close (바로 매매), or with no new signal trade for 21 rows after one
    (1개월 금지); the quarter-end rebalance stays and goes to the held state's target (reading of "정기 리밸런싱에 더하지
    말고" = do not wait for the quarter; stated in the design). Periods 2005-01-03 and 2000-09-01 to 2026-10-16,
    real-price first, 10 bp. Operational readings: D rules read the 2000-09 KRW panel in both periods; MDD is
    measured from the signal start when it is later than the period start (K3 2001-06-29, K10 2002-01-29, D3
    2001-08-30); placebos (200 circular shifts ≥ 252 rows) follow 바로 매매 and also judge the 1개월 금지 rows; the
    equal-cost static moves x of the full action for good with x matching the row's 2005 CAGR; date offsets
    0/1/5/10/15 run for rows passing 1 and 2, worst over both periods. Code `lab/aa_canary.py`, runner
    `scripts/aa_canary_study.py`, tests `tests/test_aa_canary.py`; results `2026-10-27_canary/`, memo
    `2026-10-27_canary.md`. Verdicts fixed before the run: 1 2005 MDD ≥ −15 %; 2 2000-09 MDD ≥ −15 %; 3 both
    neighbours pass 1 · 2 and offset worst ≥ −20 %; 4 2000-09 MDD ≥ 1 %p shallower than the equal-cost static; 5
    shallower than ≥ 90 % of placebos. Rows passing 1–5 are proposed as preset candidates; registration is the user's.
    Results (208 rows = 4 allocations × 26 signal-actions × 2 versions): passes 200 / 53 / 39 / 60 / 66, all five 27,
    all on B3 · B4 · B5. 23 are US-equity trend → safe sleeve (K2 · K4 · K5 · D1 × B3–B5 × both versions, less B3 · D1
    바로 매매), 2 B5 → defensive (D1, K5 바로 매매), 2 B5 D3 (late start, weak on verdict 2). Candidates: MDD −7.9 to
    −13.3 % from 2005 and −8.2 to −13.6 % from 2000-09 (static B3 / B4 / B5 −30.8 / −25.6 / −26.2); 2000-09 CAGR vs
    static −0.6 to +0.4 %p; 2005 CAGR −0.8 to −2.0 %p (median −1.2, D3 excluded), 2016–26 median −1.3 %p with MDD 1.3
    %p shallower, crises 2008 / 2020 / 2022 shallower by a median 1.4 / 3.3 / 1.7 %p; equal-cost static 1.4–11.1 %p
    deeper (median 5.7); placebos 97–100 % deeper; vs A38 same allocation CAGR +0.4 %p median from 2005, MDD 1.1 %p
    shallower; vs 안정형 40 static CAGR −0.2 / −0.3 %p median, MDD 2.5 %p shallower from 2000-09. E.g. B4 · K5 → safe ·
    1개월 금지 8.90 / −10.25 (2005), 7.91 / −11.72 (2000-09), 1.4 signal trades a year. B2: none (the equal-cost static
    matches or beats the signals, placebo shares 86–94 %). K1, K3, K6–K11 and D2: none (K8 deeper than most placebos;
    K1 31 trades a year 바로 매매). 1개월 금지 cuts signal trades to about a third (K2 24.0 → 7.7, K4 6.2 → 2.6, K5 4.5 →
    1.4 a year) with 13 vs 14 candidates. 20 bp: CAGR −0.05 to −0.86 %p, MDD within −13.7 %. KRW-read K4: 2000-09 MDD
    ~5 %p deeper on average (0.4–8.9), supporting USD canaries. Engine check 3.7e-14; look-ahead checked by truncated
    inputs (tests). Consequence: the 27 rows proposed as preset candidates; on 2026-10-27 the user chose '등록 없이 기록만' (nothing registered, no stage-2 shake of a signal row); registering one would need
    canary support in `aa_engine` / the app first, and an A32-style stage-2 shake of the chosen row should come before.
    Caveats: 208 combinations (candidates cluster in one idea and its neighbours pass); the trend rules were published
    on data containing 2000–02 and 2008, so the dot-com pass is not independent; most of 2005–2026 is model prices
    (A39 ⑤) and 2000–2004 is all model; post-2005 crisis help is 1–3 %p; in-sample. Same PR: experiment entry 'a40' on
    '실험 기록' and PNGs in `2026-10-27_canary/figures/`.

A41. Principle-first map of a 13-product universe, backtests used only to check (run 2026-10-28; memo `2026-10-28_a41.md`). User,
    2026-10-27 20:48 / 21:17 and 2026-10-28 11:12 KST: keep MDD −15 % as the goal but do not drop an allocation
    just for crossing it (so −15 % is a reference line, not a cut); keep 2025–2026 out of the backtest and look at it
    once at the end; every growth and defensive equity product on its own; gold priced as the KRX spot account;
    unhedged US treasuries only as real KRX-listed ETFs buyable in ISA (no synthetic products), earlier ETFs or US
    proxies before listing; no weight rule fixed in advance, cover the space as widely as possible and read it as a
    multivariate function; cost / spread / tax set up but not applied; '권고대로': drop TIPS, IG corporates, MSCI World
    and KODEX 200, and build the 'gold by new deposits only' version as a setting without running it. Design fixed
    before results in `2026-10-28_a41_design.md` (user-reviewed plan: Claude Doc '자산배분 재설계안'): research window
    2000-09-01 – 2024-12-31 (A37 `extend_history` before 2005; info sub-window from 2005; halves split at 2012-12),
    hold-out 2025-01-02 onward used once after the candidates are shown; ISA + gold account rebalanced as one
    portfolio; quarterly, next-close, 10 bp. Products (13): 379810, 379800, 453630, 453640, 463640, KRX spot gold
    (2014-03-24; fallback 411060 with fee added back; before GLD / GC=F × FX), 459580, one dollar-cash axis 0046A0
    (chained 261240 real 2016-12 – 2025-04, BIL / T-bill × FX before), 329750 (KIS 0–1-yr treasuries; pre-listing proxy = USD cash, not SHY),
    305080, 0091C0, 476760, 484790 (chained 453850); 261240 / 455030 / 308620 are not separate axes. Stage 1: 2^20
    scrambled Sobol points, per product an inclusion coordinate (p 0.5) and a size coordinate, 2 % floor then uniform
    Dirichlet, integer %; metrics per point include CAGR / MDD on the window, halves and 2005+, crisis windows,
    quarter-end start-date CAGR p10 (= the rank) and quarter-end 10-year windows; zones 안 (≥ −15) / 경계 (−15 to −17)
    / 밖, all points kept. Stage 2: no gradient ascent (MDD is non-smooth and A32's refined R walked to the line);
    Scheffé quadratic surrogate (fit on even points, R² on odd) for slope maps only, product-level Sobol indices on
    real simulations (Saltelli N = 2^14, Jansen, bootstrap 200), 1 / 5 %p exchange rates by finite differences,
    leave-one-product-out pairs on 50,000 points, all subsets summarised. Selection: (가) points with MDD ≥ −15 %,
    every quarter-end 10-year window ≥ −15 % and ±5 %p neighbours ≥ 80 % inside, plateau = rank within 0.3 %p of the
    best, grouped by the set of products ≥ 10 %, a centre (mean weights, < 2 % dropped, integer %) for each group
    with ≥ 5 % of the plateau, up to 5; (나) the same with the line at −17 %, only if its best rank beats (가)'s. The
    same selection on each half, judged on the other half, and on rule weights (equal, inverse vol with cash 30 %,
    equity risk share ≤ 50 %); PBO by CSCV (10 blocks, 20,000 points) for A31's rule and a plateau rule. Stage 3
    (engine) on every centre, rule weights and comparators (B2, B4, 안정형 40, two A39 skeletons, S&P500): ±5 / ±10 %p
    neighbours and leave-one-out; 528 rebalancing versions (monthly / quarterly / half-yearly / yearly × month phase
    × 0–20 trading-day offsets × bands 0 / 3 / 5 / 10, 10 bp, plus 20 bp at offset 0); monthly starts 2000-09 –
    2015-12 and 172 monthly 10-year windows. Verdicts, pass / fail only: 1 MDD ≥ −15 %, 2 ±5 %p neighbours ≥ 80 %
    inside, 3 ≥ 90 % of rebalancing versions inside, 4 every monthly 10-year window inside; passers ranked by the
    monthly start-date CAGR p10; misses kept with their size; rebalancing worst < −20 % and start-date p10 vs 안정형
    40 are info columns. Stage 4 hold-out recorded only. Stage 0 PR (no results) also registers the new products,
    fixes `aa_real.SYN_REAL` (the unhedged 30-yr stood on 453850 = ACE 미국30년국채액티브(H), hedged; A35 samples
    holding the synthetic unhedged 30-yr used hedged returns after 2023-03; B2–B5 and 안정형 presets unaffected) to
    476760, and adds `lab/aa_frictions.py` (off, bit-identical test). Nothing is registered by the study.
    Results (2000-09 – 2024-12, KRW, quarterly, 10 bp; engine check 4.4e-14): 2^20 points, 11.2 % inside −15 %, best
    CAGR inside 9.15 %. Sobol total index on MDD: nasdaq-100 0.36, CD 0.18, 10-yr(H) 0.18, 30-yr(H) 0.15; adding CD or
    10-yr(H) raises the inside share by 12.6 / 10.2 %p (hedged beats unhedged at the same maturity). Flat region: 10
    points (CAGR 8.6–9.0, MDD −13.3 to −14.3); the grouping rule split them, so no full-window centre; half-window
    centres (5) failed the other half (full MDD −19 to −42 %, 0 % of 528 rebalancing versions inside). PBO 0.91 (A31
    rule) / 0.63 (flat-region rule). Among comparators only 안정형 40 kept every rebalancing version and 10-year window
    inside. Post hoc (after the results; the pre-registered outcome stays "no full-window centre"): each flat region's
    mean became one centre (labelled 사후). (가) nasdaq-100 18 · S&P500 3 · health care 8 · KRX gold 30 · 10-yr(H) 30 ·
    30-yr(H) 11 → 8.84 / −13.71 %, neighbours ±5 94.7 % inside, all 10-year windows inside, rebalancing 81.3 % inside
    (monthly 100, quarterly 91.7, half-yearly 86.1, annual 74.7 %; 5 %p band 65 %) → verdict 3 missed. (나) nasdaq-100
    17 · S&P500 6 · staples 2 · health care 8 · KRX gold 35 · 10-yr(H) 13 · 30-yr(H) 19 → 9.40 / −15.73 %; against
    −17 % it passes 1, 2, 4 and misses 3 (80.9 %). Hold-out 2025-01-02 – 2026-10-27, each target once (the user chose
    to run it after seeing the post-hoc centre; (나) and the three rule weights were missed in the first run and added
    the same day): (가) +23.0 % (MDD −11.3), (나) +24.8 (−13.2), 안정형 40 +21.5 (−9.1), B2 +26.4 (−10.0), S&P500
    +25.4 (−19.2), rule weights +9.9 to +11.7; 11 of 16 inside −15 %. Recorded only. Registration (user's choice, 2026-10-28,
    after the hold-out): (가) became preset `a41_ga` "A41 (가) 평균 중심 (사후, 월·분기 리밸런싱만)"; its name and note
    carry the post-hoc label, the monthly/quarterly-only condition and the model-price caveats. (나) and the half
    centres are not registered. Caveats: 10-yr(H) has no real price
    before 2025-08 (all model in the research window); 2000–2004 all model; sampling was PCG64 pseudo-random, not
    Sobol; in-sample. Same PR: experiment entry 'a41', the 'A41 연구 구간' period on 'B2–B5 시험 모음', figures ①–⑧.

A42. Verify the rule that picks weights, not the picked weights (design fixed 2026-10-29 before any run; memo
    `2026-10-29_a42_design.md`). User asked for an audit of A1–A41 and a plan "testing as many combinations as
    possible, time is fine"; the audit doc listed 36 gaps (data 10, method 12, operations 7, reading 7) and the user
    chose the full scope, stages 0→5 in order (2026-10-29 10:39 KST). Why: A31–A41 looked at millions of weights on
    the same 2000-09 – 2024-12 data; a picked weight cannot be re-verified there, the 2025–26 hold-out is spent, and
    안정형 40's pass is a re-check (registered after seeing 2005–2026). Four axes fixed now: (1) walk-forward of eight
    selection rules R1–R8 (yearly re-pick on expanding and 10-year windows, hold next year, test 2006–2024; fixed
    comparators F1–F5 labelled not out-of-sample); (2) model uncertainty as an axis, 24 price sets = prices
    {model, adjusted, real-first} × hedge cost {0, 20, 40, 60 bp} × KRW cash {BOK base, CD91}, pass = every set; the
    representative set is real-first · h* (grid value nearest the stage-0 estimate from real 2017–18 hedged UST ETFs)
    · CD91; (3) six-factor exposures (EQ, GROWTH, UST, KTB, FX, GOLD) and eight fixed scenarios S1–S8, pass = at most
    two worse than −25 %; (4) sealed data: 1990-01 – 2000-08 proxies built in stage 0 and stored only on Drive (hash in
    repo, code refuses to read without `--unseal-a42`), opened once in stage 5 with a −20 % line; forward tracking of
    real prices from 2026-11 by a monthly Action, first look 2027-11. Stages: 0 data (CD91, hedge-cost estimate,
    KTB 3/10/30-yr and US-dividend axes, quirk rules `aa_real.QUIRKS`, adjusted model, FX timing check, 24-set
    re-run of B2/B4/안정형 40/(가)/holdings/S&P500); 1 `aa_exposure.py`; 2 static search v2 over 17 products (A41 13
    + KTB 3 + US dividend; leverage arm adds 409820 and 225040, sum ≤ 30 %), samples T1 two-level Sobol / T2 free
    (inclusion 0.3/0.5/0.7) / T3 k-uniform / TL leverage, 2^20 each, seed 2042; second grid on 50k inside + 10k
    border + 20k outside points × 1,152 settings (24 price sets × 4 periods × bands 0/3/5/10 × 3 phases, band added
    to `StaticSim`); point filter = MDD ≥ −17 % in all 24 sets, rules compared by walk-forward CAGR / MDD / longest
    recovery Pareto, rule pass = walk-forward MDD ≥ −17 % in all sets; 3 operating policies (324 = account single /
    ISA + gold-by-new-money × deposits × execution × costs × tax × ISA limits; implement `DepositRule`/`TaxRule`;
    the user declares the policy they will use, default quarterly · no band · two accounts · 10M + 1M KRW a month ·
    general ISA; rebalancing verdicts count only that declared family); 4 dynamic layer (8,730 settings = 10 targets
    × signals none/K2/K3/K4/K6/K8/D1 × 4 actions × confirm 0/10 days × tranches 1/2/4 × vol target none/8/10 % ×
    cooldown 0/21, plus HAA50+BAA50 satellite 0/25/50 %; 200 placebos and equal-cost static; A40 verdicts); 5 final
    gates F1–F5 (representative MDD ≥ −15 % with 경계 listed apart, all sets ≥ −17 %, ≥ 90 % of the declared policy
    family inside, scenarios, rule walk-forward), sealed-window opening, campaign ledger `trials_ledger.csv` (A31–A41
    back-filled) for deflated Sharpe, nothing registered by the study. Criteria are not lowered if no candidate passes.
    Corrections to the audit doc (signal keys, KTB codes, setting count) are listed in the memo.
