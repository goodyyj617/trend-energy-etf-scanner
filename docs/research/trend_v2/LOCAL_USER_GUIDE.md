# Local strategy workflow guide

## Start and stop

Use Python with the repository dependencies installed:

```powershell
python scripts/run_trend_v2_web.py --store <local-result-store>
```

Open the loopback URL printed by the command (normally `http://127.0.0.1:8765/`).
Press `Ctrl+C` in that terminal to stop the service. Results, workflow records,
execution attempts, and robustness records stay under the local result-store;
back up that directory before changing or cleaning it.

## Run a workflow

Choose controlled strategy components and dates, inspect the exact candidate and
workload estimate, and confirm when the policy requires it. Start economic work
only after confirmation. Configure the bounded robustness methods, then select
an `EvaluationProfile` to apply mandatory gates, Pareto selection, robustness
vetoes, tie-breaks, and behavior deduplication.

Changing a profile or decision threshold creates a new evaluation result and
does not rerun an unchanged economic path or valid robustness evidence.

## Recovery and safe handling

Reopen the same local URL after a browser refresh or API restart. The workflow
view reconstructs persisted references and labels interrupted work explicitly.
Use resume only where the displayed recoverability permits it. `missing`,
`incomplete`, `corrupt`, and `interrupted` are not successful results; corrupt
or provenance-invalid records are not reused. To clean up safely, first stop
the service, copy the complete store to a backup location, then remove only an
entire disposable local store.

## Known limitations

This is a loopback-only local tool. It has no authentication, cloud deployment,
remote worker, remote storage, or market-data download support. Historical
universe survivorship limitations and the absence of active OOS collection
remain unchanged.

## Canonical cost stress

Cost stress reruns the same local economic strategy with only the selected
transaction-cost and slippage assumptions increased by an allow-listed plan
multiplier. It is not a return haircut. The result lists the stressed cost,
slippage, round-trip assumption, deltas from the base run, worst scenario, and
survival ratio. Matching valid scenario evidence may be reused; missing,
incomplete, corrupt, or provenance-invalid evidence is unavailable rather than
treated as a pass. The next recommended task is local acceptance and
startup/recovery hardening.
