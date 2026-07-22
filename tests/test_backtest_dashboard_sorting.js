const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync("docs/backtest_dashboard.js", "utf8");
const context = {
  console,
  fmt: (_key, value) => String(value ?? ""),
  renderBacktest: () => {},
  document: {
    getElementById: () => null,
    querySelectorAll: () => []
  }
};
vm.createContext(context);
vm.runInContext(`${source}
globalThis.auditCandidateComparator = candidateComparator;
globalThis.auditSort = (rows, key, direction) => {
  backtestSortKey = key;
  backtestSortDir = direction;
  return sortCompletedTradeRows(rows);
};`, context);

const base = {
  qualification_tier: "Qualified",
  time_gate_pass: true,
  parameter_gate_pass: true,
  loyo_pass_ratio: 1,
  joint_positive_year_ratio: 0.625,
  effective_neighbor_edge_pass_ratio: 1,
  profit_factor: 2,
  avg_trade_return: 0.01,
  completed_trades: 100
};

function orderedFirst(winnerChanges, loserChanges) {
  const winner = { ...base, strategy_key: "winner", ...winnerChanges };
  const loser = { ...base, strategy_key: "loser", ...loserChanges };
  assert.equal([loser, winner].sort(context.auditCandidateComparator)[0].strategy_key, "winner");
}

orderedFirst({}, { qualification_tier: "Not qualified" });
orderedFirst({}, { time_gate_pass: false });
orderedFirst({}, { parameter_gate_pass: false });
for (const field of [
  "loyo_pass_ratio", "joint_positive_year_ratio", "effective_neighbor_edge_pass_ratio",
  "profit_factor", "avg_trade_return", "completed_trades"
]) {
  orderedFirst({ [field]: "10" }, { [field]: "2" });
  orderedFirst({ [field]: 2 }, { [field]: null });
}

assert.deepEqual(
  [{ ...base, strategy_key: "b" }, { ...base, strategy_key: "a" }]
    .sort(context.auditCandidateComparator).map(row => row.strategy_key),
  ["a", "b"]
);

const numericRows = [
  { strategy_key: "missing", profit_factor: null },
  { strategy_key: "ten", profit_factor: "10" },
  { strategy_key: "two", profit_factor: "2" }
];
assert.equal(context.auditSort(numericRows, "profit_factor", -1).map(row => row.strategy_key).join(","), "ten,two,missing");
assert.equal(context.auditSort(numericRows, "profit_factor", 1).map(row => row.strategy_key).join(","), "two,ten,missing");

console.log("backtest_dashboard_sorting=PASS");
