# Provisional Trend-Following Gate Production Implementation

## Scope

This implementation converts the approved bounded analysis into the production Backtest Only qualification contract. It changes gate calculation, bounded output fields, deterministic ranking, tests, and the static Backtest dashboard.

It does not change signal definitions or parameter grids, entry rules, exit rules, stops, the 63-day maximum holding period, transaction costs, universe construction, backtest date-window semantics, completed-trade simulation, strategy count, or raw-trade publication policy. No full ten-year backtest was run during development.

The production analysis schema is version 3 and the gate configuration is `provisional-trend-following-v1`.

## Gate definitions

### Sample Gate

Pass when overall `completed_trades >= 100`.

### Edge Gate

Pass when overall `profit_factor >= 1.2` and overall `avg_trade_return > 0`.

Profit Factor must be a published finite value meeting the threshold. A missing zero-loss Profit Factor is not silently treated as infinity or as a pass.

### Eligible entry year and annual joint-positive year

The bounded strategy-year aggregate remains assigned by trade entry year. An annual row is eligible only when:

- `is_full_calendar_year = true`; and
- annual `completed_trades >= 100`.

Partial years remain in entry-year details but never enter the primary Time Gate numerator, denominator, or LOYO folds.

An eligible year is joint-positive only when annual `avg_trade_return > 0` and annual `profit_factor > 1.0`. Annual median trade return is retained as a diagnostic and does not affect the result.

`joint_positive_year_ratio = joint_positive_years / eligible_years`.

### Time Gate and LOYO reconstruction

The Time Gate passes only when:

- `eligible_years >= 5`;
- `joint_positive_year_ratio >= 0.60`; and
- `loyo_pass_ratio >= 0.80`.

For every eligible year, LOYO subtracts that year's annual numerators and denominator from the eligible-year totals. It reconstructs:

- pooled completed trades;
- pooled arithmetic sum of trade returns;
- pooled average trade return;
- pooled gross profit;
- pooled absolute gross loss; and
- pooled Profit Factor.

A fold passes when pooled completed trades are at least 100, pooled average return is positive, and pooled Profit Factor is at least 1.2. No raw trade reconstruction or additional simulation pass is used for LOYO.

### Parameter Gate

Direct neighbors hold entry and exit fixed and change exactly one Score Breakout parameter to an adjacent grid value. A raw eligible neighbor has at least 100 overall completed trades. Its independent edge passes when overall Profit Factor is at least 1.2 and overall average return is positive.

The effective outcome signature follows the approved validation methodology:

- overall completed trades, average return, median return, trade win rate, and Profit Factor; plus
- the complete ordered entry-year vector of year, completed trades, average return, median return, and Profit Factor.

A neighbor identical to the candidate is removed. Remaining neighbors with identical signatures are counted once. The Parameter Gate passes with at least two effective eligible neighbors and `effective_neighbor_edge_pass_ratio >= 0.60`.

Time is deliberately excluded from the neighbor edge definition. Raw eligible-neighbor count and raw edge-pass ratio remain bounded diagnostics.

### Final qualification

`qualification_tier` is `Qualified` only when Sample, Edge, Time, and effective Parameter gates all pass. Otherwise it is `Not qualified`. Median trade return, win rate, bootstrap metrics, and risk diagnostics are not mandatory gates.

## Output contract

The strategy summary includes the four component pass fields, `mandatory_gates_pass`, `qualification_tier`, annual robustness fields, LOYO diagnostics, and raw/effective neighbor diagnostics requested by the approved design. Compatibility aliases such as `positive_year_ratio`, `time_stability_status`, `parameter_stability_status`, `robustness_tier`, `eligible_neighbors`, and `neighbor_pass_ratio` remain available, but their production meaning follows the new contract.

The top-level JSON records:

- schema and serialized thresholds;
- qualification data availability and reason;
- `strategy_year_aggregation_sec`; and
- separate `gate_analysis_sec` timing.

Backtest Only continues to own and publish the bounded qualification outputs. Daily Scan continues to restore those files rather than overwrite or delete them. Raw event-level trade, diagnostic-event, and skipped-event files remain prohibited.

## Unavailable-data behavior

The annual aggregate is validated for required columns, unique `strategy_key × entry_year` rows, entry-year basis, numeric integrity, nonnegative completed counts, and consistent full/partial flags.

If the aggregate is unavailable or invalid:

- the Time Gate is `Not available` with a reason;
- effective Parameter evidence is `Not available` because behavioral deduplication requires the annual outcome vector;
- `mandatory_gates_pass` is false;
- `qualification_tier` is `Not qualified`; and
- the dashboard renders unavailable metrics safely.

There is no fallback to the former annual median-return Time Gate, and missing values are not treated as zero or as a pass.

## Deterministic ranking

Qualification and ranking remain separate. No weighted score is introduced. Strategy summaries use this lexicographic order:

1. `qualification_tier`, Qualified first;
2. `time_gate_pass`;
3. `parameter_gate_pass`;
4. `loyo_pass_ratio`, descending;
5. `joint_positive_year_ratio`, descending;
6. `effective_neighbor_edge_pass_ratio`, descending;
7. overall `profit_factor`, descending;
8. overall `avg_trade_return`, descending;
9. `completed_trades`, descending; and
10. `strategy_key`, ascending.

Applied to the currently committed bounded aggregate, 42 strategies qualify. The production ranking selects `score_bo_l40_rm002_erp010__signal_3d_confirm__ma50`. This differs from the analysis note's primary representative because the approved production ranking explicitly places effective-neighbor edge ratio before Profit Factor and does not use bootstrap probability. The R20 `-0.02` / `0.00` behavioral tie is resolved only by the final strategy-key ordering.

## Dashboard changes

The Candidate Snapshot shows qualification, all four gates, completed trades, average and diagnostic median returns, Profit Factor, eligible full years, joint-positive-year ratio, LOYO ratio, and effective neighbor edge ratio.

The leaderboard exposes the requested sortable fields and visibly labels each row `Qualified` or `Not qualified`; text is always present, so status does not rely on color. Separate diagnostic leaders report highest Profit Factor, average return, joint-positive-year ratio, LOYO ratio, effective parameter robustness, and diagnostic median return among Qualified rows.

Entry-Year Stability now displays full/partial coverage, trade count, average and diagnostic median returns, Profit Factor, arithmetic sum of returns, Time eligibility, and joint-positive status. Metric definitions and the exact serialized thresholds state the provisional in-sample limitations.

Committed schema-v2 data is intentionally not reinterpreted as the new gate model. Until a schema-v3 Backtest Only run is published, the dashboard shows new qualification fields as unavailable and does not label an old row Qualified.

## Pre-merge bounded audit

The final pre-merge audit used the committed 540-row overall summary, the 5,400-row entry-year aggregate, and the PR #12 scenario and candidate outputs. It did not rerun the full backtest. The approved PR #12 definition was independently reconstructed and compared with the PR #13 production functions using `rtol = 1e-12` and `atol = 1e-10` for floating-point fields; integer and boolean fields required exact equality.

### Qualified-set and field equivalence

- Analysis-qualified count: **42**.
- Production-qualified count: **42**.
- Keys only in analysis: **none**.
- Keys only in production: **none**.
- Exact set match: **yes**.
- Field mismatches across all 42 Qualified strategies: **none**.

The field comparison covered completed trades, average return, Profit Factor, eligible and joint-positive year counts and ratio, LOYO fold and pass counts and ratio, effective-neighbor count and edge-pass ratio, and all four component gate booleans. As an additional cross-check, all corresponding fields in the 20 PR #12 rows published for scenario C at 0.60 matched the independent reconstruction within the same tolerances. No difference was hidden by display rounding.

### Top-10 production ranking audit

All values below are the native ranking values before display formatting. `Q`, `T`, and `P` are Qualification, Time Gate, and Parameter Gate respectively.

| Rank | Strategy key | Q | T | P | LOYO ratio | Joint-positive ratio | Effective-neighbor ratio | Profit Factor | Avg trade return | Completed trades |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `score_bo_l40_rm002_erp010__signal_3d_confirm__ma50` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.439782991074818 | 0.0071428364334109 | 3,720 |
| 2 | `score_bo_l40_rp000_erp010__signal_3d_confirm__ma50` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.439782991074818 | 0.0071428364334109 | 3,720 |
| 3 | `score_bo_l40_rm002_erp005__signal_3d_confirm__ma50` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.4343332978879957 | 0.0069266706646455 | 3,860 |
| 4 | `score_bo_l40_rp000_erp005__signal_3d_confirm__ma50` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.4343332978879957 | 0.0069266706646455 | 3,860 |
| 5 | `score_bo_l40_rm002_erp015__signal_3d_confirm__ma50` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.4329288394308104 | 0.007103908620082 | 3,527 |
| 6 | `score_bo_l40_rp000_erp015__signal_3d_confirm__ma50` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.4329288394308104 | 0.007103908620082 | 3,527 |
| 7 | `score_bo_l40_rp002_erp010__signal_3d_confirm__ma50` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.4121794073577374 | 0.0080258240376001 | 2,913 |
| 8 | `score_bo_l40_rm002_erp015__signal_3d_confirm__low20` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.4075971268617544 | 0.0066244243259158 | 3,573 |
| 9 | `score_bo_l40_rp000_erp015__signal_3d_confirm__low20` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.4075971268617544 | 0.0066244243259158 | 3,573 |
| 10 | `score_bo_l40_rp002_erp015__signal_3d_confirm__ma50` | Qualified | true | true | 1.0 | 0.625 | 1.0 | 1.4013676109207516 | 0.0078160383315016 | 2,834 |

The leader and its `r20_min = 0.00` twin are identical through completed-trade count; `strategy_key` is their first differing ranking field and selects the `-0.02` key. Against the PR #12 primary `score_bo_l20_rm002_erp010__first_signal__low10` (production rank 20), the first differing field is effective-neighbor edge-pass ratio, `1.0` versus `0.75`. Against that primary's `r20_min = 0.00` twin (rank 18), it is again the effective ratio, `1.0` versus `0.80`. The older What-if leader `score_bo_l10_rp002_erp005__first_signal__low20` and previous dashboard leader `score_bo_l10_rm002_erp005__breakout_5d_after_signal__low20_minus_0_5atr` are both Not qualified under the approved gate, so Qualification Tier is the first differing field.

The new leader does not contradict the earlier family-level finding that L40 and 3D confirm were weak on average. Across all 540 strategies in the current bounded summary, L40 has mean Profit Factor `1.2447954744049858` versus `1.4145272943115317` for L20, and 3D confirm has mean Profit Factor `1.2195644886568138` versus `1.4376750414606922` for First signal. Those are family averages. The revised full-year/LOYO definition nevertheless finds a concentrated strong L40/3D subset: 27 of the 42 Qualified rows use each family. The individual leader passes every mandatory gate, has perfect LOYO and effective-neighbor ratios, and then has the highest Profit Factor among the remaining tied rows. Family weakness describes the average member; individual qualification decides eligibility; the lexicographic tuple ranks only the eligible individual rows.

### Sorting safety

The backend now explicitly coerces every numeric ranking column to numeric before its stable sort. Qualification, Time, and Parameter booleans sort pass-first; all requested numeric fields sort descending; missing numeric values sort last; and `strategy_key` is the final ascending tie-breaker. Shuffling the 540 input rows does not change the result.

The audit found and corrected one bounded display-sorting defect: the dashboard comparator previously ignored an infinite comparison produced when a valid number was compared with the missing-value sentinel. The corrected typed comparator keeps missing values last in both ascending and descending interactive sorts and never compares formatted percentage or count strings. Deterministic Python and JavaScript tests cover numeric values such as `10` versus `2`, missing values, boolean direction, both numeric directions, final key ordering, and shuffled input.

### Gate-contamination checks

All checks passed:

- partial 2017 and 2026 rows are visible but excluded from eligible years and LOYO;
- annual and overall median trade returns are diagnostic only;
- bootstrap fields affect neither qualification nor production ranking;
- neighbor edge pass uses only overall completed trades, Profit Factor, and average return, never the Time Gate;
- missing or invalid annual aggregates cannot become Qualified; and
- candidate-identical outcomes are removed and remaining identical neighbor signatures are deduplicated before the effective count and ratio are calculated.

**Pre-merge conclusion: PR #13 is safe to merge after the sorting-safety correction and bounded checks described here.** One full Backtest Only run remains required after merge; it was not run during this audit.

## Development validation

Development uses deterministic unit fixtures and the committed bounded annual/overall summaries. It covers gate boundaries, partial-year exclusion, annual eligibility, joint-positive semantics, LOYO numerator reconstruction, effective-neighbor count and ratio boundaries, all-gates qualification, negative-median qualification, unavailable annual data, input-order-independent ranking, UI fallback text, workflow ownership, raw-output protection, and existing simulation equivalence.

The committed bounded data reproduces 42 Qualified strategies under the production contract without rerunning the backtest. Python compilation, all targeted unit/integration tests, JavaScript syntax checking, and `git diff --check` are required before publication.

## Validation after merge

One full Backtest Only run is required after merge to regenerate schema-v3 production outputs. Reviewers should then verify:

1. strategy count and completed-trade totals are unchanged;
2. raw and skipped output behavior is unchanged;
3. annual reconstruction identities still pass documented tolerances;
4. gate-analysis runtime is reported separately;
5. the all-period summary contains the required fields and 42 Qualified strategies for the current data snapshot, allowing for upstream market-data revisions;
6. partial years remain visible but excluded;
7. the primary row follows the documented lexicographic ranking; and
8. Daily Scan does not alter Backtest Only outputs.

These gates remain provisional and in-sample. A Qualified label does not establish out-of-sample profitability or portfolio-level performance.
