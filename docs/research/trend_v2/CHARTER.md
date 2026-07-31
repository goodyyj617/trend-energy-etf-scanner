# Trend Strategy v2 Charter

## Project objective

Trend Strategy v2 seeks ETF trend-following filters, signals, and trading
rules that can retain approximately 80% or more of SPY CAGR while materially
reducing maximum drawdown, CDaR, and recovery duration.

## Fixed principles

1. The strategy must tolerate unexpected large market declines better than SPY.
2. It should retain broadly comparable long-term return rather than maximize
   isolated trade statistics.
3. Losses should be bounded while profitable trends are allowed to continue.
4. Fixed holding-period exits are prohibited.
5. Fixed profit targets are prohibited.
6. Portfolio-level return and downside risk are primary.
7. Trade-level metrics are diagnostics.
8. Simpler rules are preferred unless added complexity demonstrates
   incremental and robust value.
9. Current v1 OOS artifacts remain historical baseline evidence only.
10. No existing OOS cohort is to be activated while Trend Strategy v2 research
    is underway.

## Provisional research objectives

- CAGR / SPY CAGR >= 0.80
- absolute strategy MDD / absolute SPY MDD <= 0.75
- strategy CDaR95 / SPY CDaR95 <= 0.80
- strategy Calmar >= SPY Calmar

These exact thresholds remain provisional research gates and are not
production approval criteria.
