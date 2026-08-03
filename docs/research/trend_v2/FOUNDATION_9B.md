# Trend Strategy v2 Foundation 9B

## Local operability contract

Foundation 9B adds one local launcher with `preflight`, `start`, and `status`
modes. Preflight is deterministic and local-only. Each ordered check has a
stable code, pass/warning/blocking status, Korean user message, short technical
diagnostic, suggested action, and component. Blocking checks prevent startup;
warnings remain visible. It checks supported Python and imports, configuration,
persisted ResultStore schema and access, workflow-state directories, frozen
snapshot members, loopback port, and registered canonical runners.

## Startup, shutdown, and recovery

`scripts/run_trend_v2_web.py start --store ...` is authoritative. It binds only
to loopback, reports the exact URL and recovery summary, and does not start a
second server when the port is unavailable. Ctrl+C stops intake, asks active
economic attempts to cancel through their established lifecycle, waits for the
local executor, and preserves append-only state.

On startup, persisted Foundation 6 manager records, ExecutionAttempts,
robustness attempts, and workflow records are reconstructed. No process absence
is treated as proof of completion. Lost local ownership becomes explicit stale,
interrupted, or blocked state. Completed and valid economic/robustness evidence
is reusable; corrupt records fail closed. Reconciliation is state-idempotent:
the same recovered state reuses the same compact recovery record rather than
creating work or duplicate requests.

## Resume and UI

Existing Foundation 6 and robustness resume contracts remain authoritative:
only pending/failed/cancelled/blocked incomplete units are requeued, while
completed/reused units are retained. Confirmation and artifact validation still
guard execution and corrupt dependencies are not reused. The Korean workflow
view now distinguishes persisted stage from live service state and displays the
last restart-recovery result, stale/blocked counts, resume availability, reuse,
and corrupt dependency guidance without redesigning the workflow UI.

## Recovery evidence and limitations

`trend_v2_recovery_record_v1` is a compact, content-hashed, append-only local
record containing recovery identity/time/source commit, scan counts,
transitions, stale classifications, reused/resumed counts, blocked/corrupt
items, and warnings. It contains no copied StrategyRuns or robustness evidence
and is not a production approval.

Known limits remain local-only single-host operation, cooperative cancellation,
no automatic retry loop, no remote workers/storage, and no market-data download.
The exact next recommended task is Foundation 10 product workflow evolution
only after a separately approved scope; Foundation 9B does not begin it.
