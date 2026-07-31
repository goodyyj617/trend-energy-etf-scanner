# Trend Strategy v2 Decisions

## Settled decisions

All decisions in this section have status `Accepted`.

1. Existing collector implementation and activation work is paused.
2. PR #18 and PR #24 remain as baseline preregistration records and are not deleted.
3. The existing v1 cohort will not be activated during Trend Strategy v2 research.
4. The existing scanner Boolean is considered too entangled to serve as the canonical v2 architecture.
5. The trend-energy score, score-breakout trigger, and score-lookback grid are retired from the primary Trend v2 research path.
6. R20 and ER20 threshold grids will not be retained as independent v2 grids.
7. Fixed holding-period exits are prohibited in v2.
8. Complete Cartesian parameter search is prohibited.
9. SPY-relative portfolio return and downside risk are primary research objectives.
10. Codex must use repository context files instead of relying on long chat history.
11. The primary deliverable is a reusable web backtest and strategy-comparison tool, not a one-off preferred strategy result.
12. The user must be able to configure signals and backtest rules from the web UI.
13. Strategy execution results and evaluation criteria must be separate so unchanged backtests can be re-evaluated without rerunning.
14. The default comparison method is non-compensatory: configurable gates, Pareto selection, epsilon tolerance, robustness vetoes, and lexicographic tie-breaking.
15. The UI may provide user-adjustable metric weights only as a separately labeled exploratory comparison mode; weighted rankings must not override mandatory gates or the default Pareto/robustness result.
16. The visible web interface is Korean-first.
17. Every acronym and metric must have a dedicated explanation with formulas, variable definitions, numerical examples, interpretation, assumptions, and limitations.
18. Strategy runs, evaluation profiles, and evaluation runs must be versioned, hashed, and preserved in history to expose threshold or weight changes made after observing results.

## Open decisions

- final trend-filter definition;
- first non-score signal families exposed in the web strategy builder;
- Phase B entry families;
- Phase B initial-stop families;
- Phase B trailing-exit families;
- exact research and final evaluation-profile defaults;
- result-retention limits and external artifact storage;
- position-sizing and concentration-control design;
- point-in-time universe remediation.
