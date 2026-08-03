# Trend Strategy v2 Foundation 9D

`init --store` seeds the three repository default EvaluationProfiles through
the canonical ResultStore save path. Reruns reuse identical profiles, seed only
missing defaults, and block a same-name profile with different content. Existing
compatible stores therefore need not be recreated.

The construction form defaults to the research profile, shows Korean labels,
and disables submission with an actionable Korean message when profiles are
unavailable. First run: install requirements, run `init --store`, then
`preflight --store`, then `start --store`.

Remaining limitations are local-only storage and no market-data download. Next
recommended task: Foundation 10, separately scoped product workflow evolution.
