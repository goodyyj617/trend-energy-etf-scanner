# Trend Strategy v2 Foundation 3

## Scope and architecture

Foundation 3 exposes saved local results without adding an economic backtest
endpoint, background worker, or final web interface. It adds three layers above
the Foundation 1/2 contracts:

1. `ExecutionAttempt` records mutable operational progress separately from an
   immutable economic `StrategyRunManifest`.
2. `SavedRunRegistryBuilder` reconstructs a canonical discovery index from
   `LocalResultStore` content and append-only execution-attempt events.
3. `ReadOnlyTrendApi` exposes bounded typed reads. `build_http_server` is a
   small standard-library `ThreadingHTTPServer` adapter; no web framework or new
   dependency is required.

The `ResultStore` objects, immutable manifests, evaluation files, retention
events, and execution-attempt events remain the source of truth. The persisted
registry is a disposable acceleration artifact and is never the sole record of
an economic result or operational attempt.

## Domain boundaries

`StrategyRun` identifies one requested economic specification and its immutable
completed manifest. Its manifest continues to accept only `succeeded`, `failed`,
or `partial`.

`EvaluationProfile` identifies the versioned comparison rules. Its hash is not
part of derived-metric calculation identity.

`EvaluationRun` records one immutable application of a profile to stored runs.
Foundation 3 additionally preserves pairwise behavior diagnostics and
simplicity metadata in the evaluation result so the API does not need to
reconstruct those UI diagnostics.

`ExecutionAttempt` is an operational record, not a `StrategyRun`. It may be
`pending`, `queued`, `running`, `cancelling`, `cancelled`, `failed`, or
`completed`. A retry receives a different deterministic attempt identity while
retaining the same intended `strategy_run_id`.

The API does not call the economic backtest engine. It only reads stored
artifacts. Derived artifacts are loaded from Foundation 2 content-addressed
caches; generation remains available only through
`calculate_and_evaluate_saved_runs`.

## Execution-attempt contract

Schema version: `execution_attempt_v1`.

The immutable identity content is:

- intended `strategy_run_id` and the complete requested specification;
- attempt number and optional retry-parent attempt ID;
- created timestamp;
- source commit;
- engine version.

The mutable operational snapshot includes start/completion timestamps,
operational status, terminal outcome, failure code/message, progress summary,
current stage, created artifact references, and worker/process metadata.

The file repository version is `file_execution_attempt_repository_v1`. It saves
canonical full snapshots as append-only numbered events. A repeated identical
save is idempotent. Terminal snapshots cannot be changed.

Allowed forward transitions are:

- `pending` to `queued`, `running`, `cancelled`, or `failed`;
- `queued` to `running`, `cancelling`, `cancelled`, or `failed`;
- `running` to `cancelling`, `completed`, or `failed`;
- `cancelling` to `cancelled`, `completed`, or `failed`.

Timestamp order and status/timestamp/outcome combinations are validated. A
`completed` attempt has `succeeded` or `partial` outcome, a `failed` attempt has
failure details, and a `cancelled` attempt has `cancelled` outcome. These are
operational outcomes; they do not add pending/running values to an immutable
`StrategyRunManifest`.

## Registry schema and identity

Registry schema: `saved_run_registry_v1`.

Rebuild algorithm version: `result_store_registry_rebuild_v1`.

The registry contains canonical ordered entries for:

- StrategyRun manifest/specification identities, snapshot, engine, source
  commit, terminal status, date range, warnings, and limitations;
- direct and derived artifacts with exact content hash, owner, kind, sizes, row
  count, schema, availability, retention state, and integrity state;
- derived metric manifests, calculation settings and versions, benchmark
  identity/hash, and source artifact hashes;
- EvaluationProfiles and EvaluationRuns, including exact profile hashes;
- execution attempts associated with their intended StrategyRun;
- orphan object hashes and rebuild issues.

`registry_id` is the SHA-256 deterministic ID of the registry schema/rebuild
versions, source fingerprint, ordered entries, orphan hashes, and ordered issue
records. The source fingerprint hashes the relative store location, size, and
bytes of every source-of-truth file while excluding the registry itself and
temporary files. Rebuilds over unchanged content therefore produce equivalent
canonical JSON and the same registry ID.

### Refresh and stale-index behavior

`load_or_rebuild` recomputes the source fingerprint. It loads the persisted
registry only when the schema, canonical identity, and fingerprint validate.
Missing, corrupt, unsupported, or stale registry files are rebuilt and replaced
atomically. Deleting the registry loses no economic or operational record.

### Duplicate and orphan behavior

Discovery scans all persisted manifest/profile/evaluation locations and groups
records by deterministic identity. Equivalent duplicates collapse to one entry
and create a `duplicate_*_equivalent` issue. Conflicting duplicates select the
lexicographically first relative record, create a `duplicate_*_conflict` issue,
and do not silently merge content. A derived manifest whose StrategyRun is
absent is recorded as orphaned. Unreferenced content-addressed objects remain in
the store and appear as ordered orphan hashes; rebuild never deletes them.

### Artifact integrity and retention states

Every referenced object is checked for stored size, gzip validity, logical
size, SHA-256, JSON validity, row count, and its known Foundation 2 schema. A
derived artifact is marked `integrity_failed` if any source evidence referenced
by its derived manifest is missing, pruned, or hash-invalid.

Availability values are:

- `available`;
- `missing` for a referenced object that is absent without a retention event;
- `corrupt` for invalid bytes, hash, row count, or known schema content;
- `pruned` for an exact immutable retention marker;
- `never_generated` when no manifest ever referenced that supported artifact;
- `unsupported_schema` for a known artifact with an unsupported schema version.

`mark_artifact_pruned` records an immutable `artifact_retention_event_v1`
marker. It deliberately does not perform destructive deletion. The registry and
API fail closed on a pruned marker even if bytes remain temporarily on disk.

## API version and server boundary

API version: `trend_v2_local_read_api_v1`.

Path prefix: `/api/v1`.

The production surface accepts only `GET` and `HEAD`. Any write method receives
`method_not_allowed`. The server binds to `127.0.0.1:8765` by default; the port
is configurable. A non-loopback bind requires an explicit configuration
override. CORS is disabled by default, and configured origins must be explicit
localhost HTTP(S) origins.

The API accepts identifiers, filters, dates, pagination cursors, and fixed
artifact selectors only. It accepts no filesystem path, remote URL, code, or
shell parameter. Traversal segments, backslashes, and invalid identifiers fail
before lookup. No endpoint downloads market data or executes Python dynamically.
Fields with secret-like names are redacted from response trees. Normal errors
contain neither stack traces nor absolute local paths.

Authentication is intentionally deferred. The trust model is one local user on
a loopback-only machine reading that user's own ResultStore. Any future remote
binding must add authentication, authorization, transport security, and a new
threat review rather than relying on this local trust boundary.

## Endpoint inventory

### Health and versions

- `GET /api/v1/health`
- `GET /api/v1/metadata`

Metadata reports supported artifact schemas, registry/result-store versions,
metric registry version, calculation/definition engines, behavior engine,
execution repository, pagination limits, CORS origins, and stable error codes.

### Saved StrategyRuns

- `GET /api/v1/runs`
- `GET /api/v1/runs/{strategy_run_id}`
- `GET /api/v1/runs/{strategy_run_id}/manifest`
- `GET /api/v1/runs/{strategy_run_id}/specification`
- `GET /api/v1/runs/{strategy_run_id}/provenance`
- `GET /api/v1/runs/{strategy_run_id}/status`
- `GET /api/v1/runs/{strategy_run_id}/artifacts`

### Curves and stored derived artifacts

- `GET /api/v1/runs/{strategy_run_id}/curve`
- `GET /api/v1/runs/{strategy_run_id}/yearly-metrics`
- `GET /api/v1/runs/{strategy_run_id}/rolling-metrics`
- `GET /api/v1/runs/{strategy_run_id}/derived-metrics`
- `GET /api/v1/runs/{strategy_run_id}/robustness-summary`
- `GET /api/v1/runs/{strategy_run_id}/behavior`

Row artifacts accept `start_date`, `end_date`, `page_size`, and `cursor`.
Rolling metrics also accept `window_sessions`. A specific cached calculation may
be selected with `derived_metric_id`. The API never returns all time-series rows
by default.

### Evaluation profiles and results

- `GET /api/v1/evaluation-profiles`
- `GET /api/v1/evaluation-profiles/{evaluation_profile_id}`
- `GET /api/v1/evaluation-runs`
- `GET /api/v1/evaluation-runs/{evaluation_run_id}`
- `GET /api/v1/evaluation-runs/{evaluation_run_id}/outputs`
- `GET /api/v1/evaluation-runs/{evaluation_run_id}/behavior`

The outputs endpoint separates mandatory gates, Pareto membership/dominators,
robustness vetoes, lexicographic tie-break order, exploratory weighted output,
decision labels, and cluster metadata. The behavior endpoint exposes stored
return correlation, active/entry/exit-date Jaccard values, normalized path
distance, cluster/representative identity, and simplicity metadata.

### Execution attempts

- `GET /api/v1/execution-attempts`
- `GET /api/v1/execution-attempts/{execution_attempt_id}`

There are no start, cancel, retry, queue, or worker endpoints.

## Filtering, sorting, and pagination

Unknown query fields, repeated fields, invalid enum values, unsupported sort
keys, invalid dates, oversized pages, malformed cursors, and stale cursors are
rejected rather than ignored.

StrategyRun filters are terminal status, profile ID, data snapshot ID, engine
version, source commit, overlapping economic date range, artifact key and
availability, integrity status, and retention status. Sort fields are creation
time, StrategyRun ID, terminal status, and engine version.

EvaluationRun filters are profile ID, StrategyRun ID, completed status,
integrity status, and creation-date range. Execution-attempt filters are
intended StrategyRun ID, operational status, source commit, engine version, and
created-date range.

The default list sort is descending creation timestamp. The deterministic
identity is always the secondary ordering key. Default list page size is 50 and
maximum is 200. Default time-series page size is 250 and maximum is 1,000.
Cursors encode the registry ID, resource, normalized query signature, and
offset; they cannot be reused after the registry or query changes.

## Error contract

Every API error has:

- machine-readable `code`;
- Korean-first `message_ko`;
- optional English `diagnostic_en`;
- request ID;
- optional object identity;
- `recoverable` flag;
- optional Korean next action.

Stable codes include `not_found`, `invalid_identifier`, `invalid_query`,
`artifact_missing`, `artifact_corrupt`, `schema_unsupported`,
`integrity_validation_failed`, `retention_pruned_artifact`,
`benchmark_coverage_failure`, `robustness_evidence_missing`,
`method_not_allowed`, and `internal_error`. Missing, corrupt, and pruned
artifacts use HTTP 404, 409, and 410 respectively.

## Korean terminology behavior

Machine fields remain stable English identifiers. Korean labels and error
messages come from the single `config/trend_v2/terminology_ko.json` source under
its `api` section. Run, attempt, artifact, integrity, and validation labels are
not redefined in individual routes. Existing metric terminology entries remain
the source for the later explanation interface.

## Example

`scripts/example_trend_v2_foundation_3.py` creates synthetic local artifacts in
a temporary directory and demonstrates deterministic reconstruction, saved-run
listing, provenance, two EvaluationRuns over the same StrategyRun, derived
cache reuse with zero economic-backtest calls, separate attempt/run status, and
distinct missing/corrupt/pruned API errors. It downloads no data and writes no
repository artifact.

## Known limitations and deferred work

- The standard-library server is a local development boundary, not a remote or
  production deployment stack.
- Authentication, authorization, TLS, remote access, remote object storage,
  distributed queues, workers, schedulers, and cancellation execution are
  deferred.
- Retention events classify pruning but Foundation 3 does not implement an
  automatic destructive retention executor.
- Foundation 2 still conservatively treats first/last calendar years as
  incomplete and does not recompute missing robustness simulations.
- Present-day universe and survivorship limitations remain unresolved.
- No signal search, rule optimization, economic backtest execution endpoint,
  final UI, or OOS collector activation is included.

The exact next task is Foundation 4: Korean-first web UI and explanation tab
consuming the local API, without yet expanding into unrestricted strategy
execution.
