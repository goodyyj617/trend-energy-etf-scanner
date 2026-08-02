# Trend Strategy v2 Foundation 9A

Foundation 9A completes canonical cost-stress execution. Each bounded multiplier
creates a new immutable stressed `StrategyRunSpec` from the completed base
specification. Only `transaction_costs.parameters.bps` and
`slippage.parameters.bps` change; universe, dates, signal, exits, sizing,
portfolio constraints, benchmark, engine, and source snapshot remain identical.

The adapter invokes the registered canonical Phase A portfolio runner and stores
scenario outputs below the existing robustness execution root. Scenario identity
binds the base run, normalized stressed assumptions, engine, snapshot, and
adapter contract version. Valid matching artifacts are reused; corrupt,
incomplete, mismatched, or provenance-invalid artifacts are not reused.

Cost-stress evidence reports each scenario's assumptions, curves' artifact
hashes, core and benchmark metrics, deltas from the base run, survival result,
reason, reuse flag, and provenance. Aggregate evidence reports completed,
reused, failed, incomplete, worst scenario, survival ratio, and the explicit
all-completed-scenarios survival rule. No combined robustness score is added.

The existing Foundation 7 plan estimate and confirmation hash already include
cost-stress scenario counts. Foundation 8 workflow start delegates to that
existing plan/attempt lifecycle, so restart and threshold-only evaluation changes
do not create another cost rerun.
