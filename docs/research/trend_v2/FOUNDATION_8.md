# Trend Strategy v2 Foundation 8

## Scope

Foundation 8 adds `trend_v2_workflow_v1`, a thin persisted coordinator over
the established construction, execution, robustness, and evaluation contracts.
It stores immutable workflow intent and append-only reference events; it does
not duplicate economic paths, robustness evidence, or evaluation artifacts.

## Workflow contract

Each workflow has a deterministic identity, Korean label, construction,
provenance, integrity hash, and append-only events for normalization, estimate,
confirmation, economic request/start, robustness plan/start, and evaluation.
The display stage is derived from these persisted references, so browser refresh
and local API restart reconstruct state without creating new requests.

The coordinator delegates to Foundation 5 `ControlledExecutionService` for
normalization, exact estimates, confirmation, immutable `ExecutionRequest`,
`ExecutionAttempt`, cache reuse, cancellation and idempotency. It delegates to
Foundation 7 `RobustnessExecutionService` for plans, attempts, evidence, and
restart-safe scenario recovery. It applies an `EvaluationProfile` only through
the established stored-run evaluation path; profile changes do not rerun an
economic path.

## Limits

No remote execution, market-data download, arbitrary code, unrestricted search,
new signal research, optimizer, cloud storage, or production approval is added.
Cost stress remains subject to the registered canonical economic runner; missing
or provenance-invalid evidence remains explicit and fails closed.
