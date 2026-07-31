# Repository Instructions

## Scope

These instructions apply to the entire repository.

For tasks explicitly related to Trend Strategy v2, also read:

- `docs/research/trend_v2/CHARTER.md`
- `docs/research/trend_v2/DECISIONS.md`
- `docs/research/trend_v2/SEARCH_PROTOCOL.md`
- `docs/research/trend_v2/CURRENT_STATE.md`

Treat those repository files, rather than prior chat history, as the
persistent project context.

## Work discipline

- Implement the requested phase before performing a broad audit.
- Do not revisit merged or settled work unless the task explicitly asks for it.
- Do not reopen decisions recorded in `DECISIONS.md` without an explicit user request.
- Do not perform repeated validation or review loops.
- Run targeted tests while implementing code.
- Run the complete bounded suite at most once, after implementation is complete.
- Perform at most one final audit pass.
- Report blocking defects separately from non-blocking follow-up ideas.
- Do not expand the requested scope merely because adjacent improvements are possible.
- Ask a question only when a genuinely blocking ambiguity cannot be resolved from repository files.
- Do not claim that a design, strategy, or research result is production-approved unless the repository explicitly records that approval.

## Documentation-only changes

For a documentation-only task:

- do not run the complete pytest suite;
- run `git diff --check`;
- verify the exact changed-file list;
- verify that no generated data, workflow, source, config, or dependency file changed.

## Trend Strategy v2 principles

For Trend Strategy v2 tasks:

- Preserve the core trend-following principle: limit losses and allow profitable trends to continue.
- Do not introduce fixed holding-period exits.
- Do not introduce fixed profit targets.
- Keep universe eligibility, trend filter, signal, entry, initial stop,
  trailing exit, position sizing, and portfolio construction as separately
  identifiable components.
- Limit numeric parameters to one per component family and normally no more
  than two per complete candidate.
- Do not construct an unrestricted Cartesian product.
- Use sequential screening, Pareto filtering, behavior deduplication, and
  successive halving to reduce candidates.
- Treat portfolio return and downside risk as primary objectives.
- Treat Profit Factor, win rate, median trade return, and other trade-level
  statistics as diagnostics rather than primary objectives.
- Compare strategy results with SPY on exact common economic dates.
- Do not implement or activate an OOS collector unless the task explicitly requests it.
- Preserve legacy v1 behavior unless a task explicitly authorizes a migration.

## Generated data

- Do not modify files under `docs/data` unless the task explicitly requires a generated-data run.
- Do not manually edit generated Backtest or Daily Scan artifacts.
- Do not run Backtest Only merely to validate source or documentation changes.

## Git and pull requests

- Use a focused branch and one pull request per defined phase.
- Do not create an additional pull request when updating an existing one.
- Do not rebase or force-push unless explicitly instructed.
- Keep PR descriptions factual and limited to implemented behavior,
  validation performed, and unresolved blocking limitations.
- Do not describe planned future work as already implemented.

## Completion report

Report only:

- changed files;
- implemented behavior;
- tests or checks actually run;
- blocking limitations;
- exact next phase recorded in `CURRENT_STATE.md`.
