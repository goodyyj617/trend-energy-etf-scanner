const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const elements = new Map([
  ["portfolioUnavailable", { textContent: "" }],
  ["portfolioMetrics", { innerHTML: "" }],
  ["portfolioChart", { innerHTML: "" }]
]);
const context = {
  console,
  fmt: (_key, value) => String(value ?? ""),
  renderBacktest: () => {},
  backtestPayload: null,
  portfolioManifest: { status: "Not available", reason: "Full run required." },
  portfolioBenchmark: null,
  fetch: async () => ({ ok: false }),
  document: {
    getElementById: id => elements.get(id) || null,
    querySelectorAll: () => []
  }
};
vm.createContext(context);
const source = fs.readFileSync("docs/backtest_dashboard.js", "utf8");
vm.runInContext(`${source}
globalThis.rangeMetrics = portfolioRangeMetrics;
globalThis.renderPortfolio = renderPortfolioForCandidate;`, context);

const metrics = context.rangeMetrics([
  { date: "2024-01-01", portfolio_equity: 500 },
  { date: "2024-01-02", portfolio_equity: 550 },
  { date: "2024-01-03", portfolio_equity: 495 }
]);
assert.equal(metrics.starting_equity, 1000);
assert.equal(metrics.ending_equity, 990);
assert.ok(Math.abs(metrics.maximum_drawdown + .1) < 1e-12);

Promise.resolve(context.renderPortfolio({ strategy_key: "missing" })).then(() => {
  assert.equal(elements.get("portfolioUnavailable").textContent, "Full run required.");
  assert.equal(elements.get("portfolioMetrics").innerHTML, "");
  assert.equal(elements.get("portfolioChart").innerHTML, "");
  console.log("portfolio_dashboard=PASS");
});
