# Repository Instructions

## Scope

These instructions apply to the entire repository.

For tasks explicitly related to Trend Strategy v2, these are the relevant
reference documents. Open them only as directed by the default context policy
below or when a specific dependency requires them:

- `docs/research/trend_v2/CHARTER.md`
- `docs/research/trend_v2/DECISIONS.md`
- `docs/research/trend_v2/SEARCH_PROTOCOL.md`
- `docs/research/trend_v2/PRODUCT_REQUIREMENTS.md`
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

## Product direction

- The primary deliverable is a reusable web backtest and strategy-comparison tool, not a one-off preferred strategy run.
- The web UI must allow the user to configure signals and economic backtest rules without source-code edits.
- Strategy execution, metric calculation, robustness validation, and candidate selection must be separate layers.
- Stored strategy results must be reusable under new evaluation profiles without rerunning an unchanged economic path.
- Non-compensatory gates, Pareto selection, epsilon tolerance, and robustness vetoes are the default comparison mode.
- User-adjustable metric weights may be offered only as a clearly labeled exploratory comparison view and must not silently replace the default decision mode.
- The visible web interface must be Korean-first.
- Every acronym and metric shown in the UI must link to a dedicated explanation containing the exact formula, variable definitions, a numerical example, interpretation, assumptions, and limitations.
- Read `PRODUCT_REQUIREMENTS.md` before changing the web UI, research architecture, result persistence, metric engine, or selection logic.

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

## Codex context and credit discipline

Use a narrow, evidence-driven inspection strategy for every task in this repository.

### Default context policy

Read first only:

- this AGENTS.md
- docs/research/trend_v2/CURRENT_STATE.md when the task concerns trend_v2
- the immediately preceding task or Foundation document
- files directly named by the current task

Do not preload all historical Foundation documents.

Do not recursively inspect entire directories.

Do not read every test file or every source file by default.

Use targeted symbol, filename, and reference searches before opening files.

Treat completed decisions summarized in CURRENT_STATE.md as authoritative unless a concrete contradiction is found.

Open additional files only when a specific dependency requires them.

Do not perform broad repository audits unless the user explicitly requests one.

Do not reopen or revalidate merged historical work without concrete evidence of a regression.

### Execution policy

Use the smallest sufficient implementation scope.

Do not create unrelated cleanup, refactoring, documentation, or formatting work.

Do not spawn subagents unless parallel work is clearly necessary.

Prefer one implementation path over multiple speculative alternatives.

Do not repeatedly inspect git status, diffs, manifests, or unchanged files.

Do not reread files whose relevant contracts have already been established in the current session.

### Validation policy

Run directly relevant targeted tests during implementation.

Run the bounded full suite at most once, after implementation is complete, unless a concrete failure requires another run.

Do not repeat successful test suites.

Do not run all supported dependency-version suites unless the task changes compatibility-sensitive behavior or the user explicitly requires them.

Use existing CI for redundant platform validation where appropriate.

### Publishing policy

After implementation and validation are complete:

- publish the existing commit as-is
- do not reread implementation files
- do not rerun successful tests
- do not amend or regenerate the commit
- use only the minimum commands needed to push and create the Draft PR

If authentication or push authorization blocks publishing, report the exact branch and SHA and stop.

### Output policy

Keep progress updates minimal.

Do not restate the full task specification.

Keep the final report limited to:

- PR URL or publishing blocker
- branch
- head SHA
- changed files
- implemented behavior
- validation results
- remaining limitations
- exact next task

### Reasoning policy

Use medium reasoning effort by default for well-specified implementation work.

Escalate to high reasoning only when there is a concrete architectural contradiction, difficult test failure, statistical ambiguity, security concern, or data-integrity risk.

Do not use high reasoning for publishing, status checks, file moves, minor documentation, or mechanical edits.
