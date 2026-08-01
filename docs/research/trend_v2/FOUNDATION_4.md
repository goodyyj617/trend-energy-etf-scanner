# Trend Strategy v2 Foundation 4

## Scope

Foundation 4 adds a Korean-first, local, read-only web interface over the
Foundation 3 saved-run API. It lets a user inspect stored `StrategyRun`,
`EvaluationProfile`, `EvaluationRun`, artifact, robustness, behavior, retention,
integrity, provenance, and `ExecutionAttempt` evidence. It does not add an
economic-backtest call, strategy form, write route, queue command, retry,
cancellation, worker control, market-data download, or remote fetch.

The UI is evidence exploration, not a production-approval dashboard. Incomplete
or missing evidence is displayed explicitly and never promoted to a favorable
status.

## UI architecture and technology choice

The implementation uses packaged semantic HTML, CSS, and dependency-free
JavaScript served by the Python standard library on the same loopback origin as
the API. `TrendWebApplication` serves only four fixed routes (`/`,
`/index.html`, `/assets/app.js`, and `/assets/style.css`) and delegates
`/api/v1/*` to `ReadOnlyTrendApi`. `build_web_server` uses
`ThreadingHTTPServer`; the browser never receives a filesystem path.

This is the smallest maintainable choice for the repository because Foundation
3 already uses the standard-library HTTP server and the repository has no
frontend package manager or build pipeline. It adds no framework, runtime
dependency, generated bundle, or remote CDN. The existing `web/` Daily ETF Scan
page and `docs/backtest_dashboard.*` Backtest Only output remain separate and
unchanged.

`scripts/run_trend_v2_web.py --store <existing-result-store>` opens a persisted
store by its recorded retention policy and starts the combined loopback server.
`scripts/example_trend_v2_foundation_4.py` builds synthetic evidence only; its
default mode starts an ephemeral server, verifies UI/API reads, prints the
demonstration record, and shuts down. `--serve` keeps the synthetic UI available
for manual review.

## Page and component inventory

The main navigation has nine Korean-first areas:

1. `개요`: stored-object counts, artifact/attempt state counts, version
   boundaries, registry identity, issues, and the evidence-quality warning.
2. `저장된 전략 실행`: allow-listed filters, deterministic sort, cursor
   pagination, saved-run selection, terminal result, provenance, artifact
   summaries, integrity, and retention.
3. `평가 결과`: same-StrategyRun profile comparison and separate mandatory
   gate, epsilon-Pareto, robustness-veto, lexicographic, behavior-deduplication,
   and exploratory weighted stages.
4. `성과 및 위험`: bounded portfolio, benchmark, drawdown, rolling, yearly,
   exposure, cash, position-count, turnover, and transaction-cost charts.
5. `강건성`: stored walk-forward, LOYO, paired block-bootstrap, confidence
   interval, p-value/correction, cost-stress, concentration, DSR, and PBO fields
   when present, plus exact unavailable errors and profile veto results.
6. `행동 유사도`: bounded pair selection with separate return correlation,
   active/entry/exit-date Jaccard, normalized path distance, cluster,
   representative, and simplicity metadata.
7. `실행 이력`: append-only operational attempts, timestamps, retry parent,
   stage, progress, failure, artifacts, commit, engine, and worker metadata,
   explicitly separate from immutable StrategyRun outcomes.
8. `설명`: searchable centralized Korean definitions.
9. `시스템 정보`: health, API/registry/engine versions, response bounds, CORS,
   and security boundary.

Reusable components are the status badge, evidence empty/error state, filter
toolbar, cursor pager, semantic data table, key/value provenance view, metric
definition link, evaluation-pipeline stage, accessible SVG chart, and chart
summary table.

## API endpoint mapping

| UI area | Read-only endpoints |
|---|---|
| Startup and system | `/health`, `/metadata`, `/overview`, `/terminology` |
| Saved-run list | `/runs` |
| Run identity | `/runs/{id}`, `/manifest`, `/specification`, `/provenance`, `/status`, `/artifacts` |
| Bounded charts | `/curve`, `/benchmark-curve`, `/yearly-metrics`, `/rolling-metrics`, `/derived-metrics` |
| Stored robustness and behavior | `/robustness-summary`, `/behavior-summary`, bounded EvaluationRun `/behavior` |
| Profiles and evaluations | `/evaluation-profiles`, `/evaluation-profiles/{id}`, `/evaluation-runs`, `/evaluation-runs/{id}`, `/outputs`, `/behavior` |
| Operational attempts | `/execution-attempts`, `/execution-attempts/{id}` |

Foundation 4 adds four read-only representations without changing any economic
calculation: `/overview` aggregates bounded registry counts, `/terminology`
serves the centralized safe terminology mapping, and `/benchmark-curve` applies
the existing bounded artifact read contract to the stored benchmark path.
`/behavior-summary` returns only fingerprints, source hashes, and input counts;
the browser does not receive the full behavior comparison paths on run detail.

## State model

The browser owns only transient presentation state: active route, selected run
or evaluation, allow-listed filter values, bounded cursor history, terminology
cache, and one `AbortController`. Route changes cancel outstanding requests.
No browser state is persisted as economic evidence.

Domain state remains separate:

- `StrategyRun` is an immutable terminal economic result (`succeeded`, `failed`,
  or `partial`).
- `EvaluationProfile` is a versioned and hashed decision definition.
- `EvaluationRun` is an immutable application of one profile to stored runs.
- `ResultStore` and its reconstructed registry supply stored evidence.
- `ExecutionAttempt` is an operational lifecycle and may be pending, queued,
  running, cancelling, cancelled, failed, or completed. It never changes the
  StrategyRun terminal result shown beside it.

## Artifact availability model

The UI preserves all Foundation 3 classifications:

- `available` / 사용 가능;
- `missing` / 누락;
- `pruned` / 보존 정책에 따라 제거됨;
- `corrupt` / 손상됨;
- `never_generated` / 생성된 적 없음;
- `unsupported_schema` / 지원하지 않는 스키마;
- `integrity_failed` / 출처 무결성 실패.

Status badges include text and a visible symbol, not color alone. Artifact
tables also show retention, integrity, row count, stored size, and content hash.
The API's Korean error and stable machine code are retained for missing,
corrupt, pruned, unsupported, provenance-invalid, and robustness-evidence
failures. The UI does not convert these conditions to a single unavailable
state.

## Chart behavior

Initial daily and rolling reads request at most 250 rows. Yearly reads request
at most 100 rows. The browser does not request an unbounded curve by default;
Foundation 3 also enforces a hard 1,000-row time-series maximum.

Portfolio and benchmark values share a chart only on exact matching economic
dates within the selected bounded response. Stored benchmark common-date
provenance is displayed beside the chart when present. Drawdown is labeled as
the displayed-page high-water-mark calculation so it is not presented as an
unbounded full-history value. Economic dates use their actual time distance;
points separated by more than five calendar days are not connected by a line.
No artificial observation is inserted.

Every chart has a title, unit, date range, labeled endpoint axis, legend when
multiple series are present, an accessible image label, a metric explanation
link where applicable, and an accompanying first/latest/min/max table. Missing
or incomplete evidence produces a Korean state block rather than an invented
series.

## Evaluation-pipeline presentation

The default non-compensatory stages are numbered and displayed independently:

1. mandatory gates show metric, direction, operator, threshold, observed value,
   pass/fail, reason, profile identity, and exact profile hash;
2. epsilon-Pareto shows membership, dominators, objectives, and epsilon values;
3. robustness veto shows passed, failed, evidence-missing, not-enabled, or
   unavailable reason without fabricating evidence;
4. lexicographic tie-break shows configured order and the candidate's eligible
   order;
5. behavior deduplication shows stored cluster/representative metadata.

The optional weighted section is visually separated and uses only the exact
field name `exploratory_weighted_value`. It explicitly says that the value does
not override a gate failure, Pareto result, or robustness veto. The UI does not
use legacy strategy-score wording.

Multiple EvaluationRuns that contain the same StrategyRun are shown together.
Their profile identities, hashes, metrics, gates, Pareto objectives, epsilon,
vetoes, lexicographic order, normalization, weights, and resulting labels are
compared. The page states that profile-only changes reuse the economic path and
do not rerun a backtest.

## Explanation schema and terminology source

`config/trend_v2/terminology_ko.json` remains the single source for API status
labels and metric/concept explanations. `/terminology` exposes a safe canonical
copy; components do not carry competing Korean metric definitions.

Each entry contains Korean and English names, abbreviation, exact formula text,
variable definitions, a worked numerical example, interpretation, unit,
annualization convention, assumptions, limitations, misleading cases, and
applicable mandatory-gate/Pareto/exploratory/robustness use. Coverage includes
all Foundation 4 required metrics and concepts: CAGR and SPY-relative ratios,
volatility, conventional/HAC Sharpe, Sortino, drawdown/CDaR/Calmar, recovery and
time under water, rolling return, expected shortfall, downside deviation,
turnover/cost drag/exposure/cash/positions, walk-forward, LOYO, paired block
bootstrap, confidence intervals, raw/adjusted p-values, Holm, White Reality
Check, Hansen SPA, DSR, PBO, cost stress, concentration, Pareto/epsilon,
mandatory gate, robustness veto, lexicographic tie-break, behavior
deduplication, and `exploratory_weighted_value`.

## Accessibility and usability

- document language, headings, landmarks, skip link, labeled controls, table
  headers, and keyboard-operable links/buttons are semantic;
- visible `:focus-visible` treatment is present;
- text/background combinations use high-contrast dark ink and light surfaces;
- pass/fail and artifact conditions use symbols plus Korean text;
- chart SVGs have text alternatives and accompanying tables;
- loading and error regions are live regions;
- layout reflows below ordinary desktop widths and navigation remains usable;
- filters reset explicitly and invalid allow-list queries display the API's
  Korean validation message rather than being ignored.

## Security boundaries

The default bind remains `127.0.0.1`. A non-loopback host still requires the
Foundation 3 explicit override. The application serves a fixed asset map and
does not accept a browser filesystem path. Double-decoded traversal, backslash,
and null-byte requests fail closed. The Content Security Policy permits scripts,
styles, images, and connections only from the same origin; there is no remote
URL fetch, CDN, inline script, or form submission.

API response redaction, secret-like field removal, stack-trace suppression,
absolute-path suppression, response bounds, CORS defaults, identifier
validation, and write-method rejection remain active. The frontend renders
untrusted values through text escaping or `textContent` and does not display a
server diagnostic stack.

## Test strategy

`tests/test_trend_v2_foundation_4.py` uses only synthetic local ResultStore
fixtures. It covers Korean navigation/headings, fixed same-origin assets,
security headers, overview metadata, saved-run sorting/pagination, artifact
summary, Korean invalid-query errors, run detail, 250-row curve/benchmark
bounds, two profiles for one run, profile hashes, separated evaluation stages,
mandatory-gate fields, exact robustness error state, all artifact
classifications, execution-attempt separation, no absolute path leakage,
explanation completeness and safe formula rendering, keyboard accessibility,
non-color status cues, legacy asset separation, no economic-backtest call,
JavaScript syntax, and a real ephemeral loopback HTTP start/read/shutdown.

Foundation 1–3 tests remain the domain regression boundary. Existing Daily ETF
Scan, Backtest Only, supported pandas-version, FutureWarning-as-error, Python
compilation, and bounded full-suite checks remain final validation inputs.

## Known limitations and deferred execution functionality

- The charts show a bounded page, not an automatic unbounded history. A later
  phase may add explicit range selection and bounded page stitching.
- No exchange-calendar service is introduced; the chart preserves stored dates
  and does not assert that every absent weekday is a missing market observation.
- Robustness simulations are not recomputed. Enabled vetoes fail closed when
  required stored evidence is absent.
- Authentication, remote binding, TLS, cloud storage, distributed calculation,
  and production deployment remain deferred.
- Strategy construction, candidate-count estimation, confirmation, execution,
  start/cancel/retry APIs, workers, queues, signal search, and parameter
  optimization remain deferred to later foundations.

The exact next task is Foundation 5: controlled strategy-construction and
execution contracts, including explicit candidate-count estimation and user
confirmation, without yet performing unrestricted large parameter searches.
