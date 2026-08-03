# Trend Strategy v2 Foundation 9C

`python scripts/run_trend_v2_web.py init --store <path>` is the explicit,
idempotent first-run ResultStore bootstrap. It invokes `LocalResultStore` with
bounded deterministic defaults and creates the canonical policy plus workflow
state directories. Existing non-empty uninitialized, incompatible, or corrupt
stores are refused without overwrite. Init and preflight make no network calls.

Missing and empty stores block preflight and direct users to init. The launcher
defers application imports so help, init, and preflight do not fail with an
optional runtime package traceback. Snapshot verification uses Git blob bytes
when available (otherwise working-tree bytes), preventing Windows CRLF false
failures without weakening real corruption detection.

Remaining limits: local-only storage, no remote deployment, and no market-data
download. Exact next task: Foundation 10 -- separately scoped product workflow
evolution, without unrestricted optimization.
