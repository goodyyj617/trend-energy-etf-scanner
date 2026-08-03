# Trend Strategy v2 Foundation 9E

## Candidate-estimate response rendering

The construction form reached the candidate-estimate request in a real browser,
then failed while rendering the response with `Cannot read properties of
undefined (reading 'fold_count')`.

The candidate-estimate endpoint had two implementation paths. The original
construction path returned an envelope containing `normalized_construction`
and `candidate_estimate`. The Foundation 6 catalog path returned its estimate
object directly, with `normalized_construction` nested inside that estimate.
The UI assumed the former shape and attempted to read a walk-forward object
from the Foundation 6 normalized catalog selection, where that field is not
part of the contract.

The canonical endpoint response is now one envelope:

```json
{
  "normalized_construction": { "...": "normalized request" },
  "candidate_estimate": { "...": "versioned estimate fields" },
  "strategy_run_candidate_ids": ["..."]
}
```

The frontend reads that envelope, maps only versioned candidate-count names
for presentation, and treats omitted walk-forward and robustness objects as
disabled/zero only for the catalog estimate contract that does not model those
workloads. Missing required estimate fields now produce an actionable Korean
contract message in the preview area; the strategy-construction form remains
visible and the selected `research_default` profile is preserved.

Validation covered the canonical endpoint envelope, valid legacy and catalog
preview rendering, candidate counts, zero optional workloads, malformed
response handling, first-run profile behavior, init/preflight/start focused
regressions, Python compilation, and JavaScript syntax.

Next recommended task: Foundation 10 -- separately scoped product workflow
evolution, without unrestricted optimization.
