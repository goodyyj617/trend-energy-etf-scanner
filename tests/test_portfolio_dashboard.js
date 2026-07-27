const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const cdarFixtures = JSON.parse(
  fs.readFileSync(path.join(__dirname, "fixtures", "portfolio_cdar_fixtures.json"), "utf8")
);

function fakeElement() {
  return {
    textContent: "", innerHTML: "", value: "", min: "", max: "",
    disabled: false, checked: false,
    addEventListener: () => {}
  };
}

const elementIds = [
  "portfolioUnavailable", "portfolioMetrics", "portfolioChart",
  "portfolioStartDate", "portfolioEndDate", "portfolioApplyRange",
  "portfolioResetRange", "portfolioLogScale"
];
const elements = new Map(elementIds.map(id => [id, fakeElement()]));
const context = {
  console,
  AbortController,
  fmt: (_key, value) => String(value ?? ""),
  renderBacktest: () => {},
  backtestPayload: null,
  portfolioManifest: null,
  portfolioBenchmark: null,
  fetch: async () => ({ ok: false, status: 500, json: async () => null }),
  document: {
    getElementById: id => elements.get(id) || null,
    querySelectorAll: () => []
  }
};
vm.createContext(context);
const source = fs.readFileSync("docs/backtest_dashboard.js", "utf8");
vm.runInContext(`${source}
globalThis.testRangeMetrics = portfolioRangeMetrics;
globalThis.testCDaR = conditionalDrawdownAtRisk;
globalThis.testRenderPortfolio = renderPortfolioForCandidate;
globalThis.testSelectedKey = () => selectedPortfolioCurve?.strategy_key || null;
globalThis.testResetPortfolioState = () => {
  portfolioCurveCache = new Map();
  selectedPortfolioCurve = null;
  portfolioSelectionToken = 0;
  if (portfolioAbortController) portfolioAbortController.abort();
  portfolioAbortController = null;
};`, context);

function curvePayload(key, ending = 900) {
  return {
    status: "Available",
    curve_schema_version: 2,
    strategy_key: key,
    series: [
      {
        date: "2024-01-01T23:59:59.999999Z", observation_type: "initialization",
        portfolio_equity: 1000, cash_value: 1000, invested_value: 0,
        gross_exposure: 0, active_position_count: 0, daily_portfolio_return: 0,
        cumulative_return: 0, running_peak_equity: 1000, drawdown: 0,
        transaction_cost_paid: 0, turnover: 0
      },
      {
        date: "2024-01-02", observation_type: "trading_session",
        portfolio_equity: 1000, daily_portfolio_return: 0
      },
      {
        date: "2024-01-03", observation_type: "trading_session",
        portfolio_equity: ending, daily_portfolio_return: ending / 1000 - 1
      }
    ]
  };
}

function manifest(keys = ["A", "B"]) {
  return {
    status: "Available",
    curve_schema_version: 2,
    strategies: keys.map(key => ({
      strategy_key: key,
      file: `${key}.json`,
      summary: {}
    }))
  };
}

function response(payload, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => payload };
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function reset(keys = ["A", "B"]) {
  context.testResetPortfolioState();
  context.portfolioManifest = manifest(keys);
  context.portfolioBenchmark = {
    status: "Available",
    series: [
      { date: "2024-01-01T23:59:59.999999Z", observation_type: "initialization", benchmark_equity: 1000 },
      { date: "2024-01-02", observation_type: "trading_session", benchmark_equity: 1000 },
      { date: "2024-01-03", observation_type: "trading_session", benchmark_equity: 950 }
    ]
  };
  for (const element of elements.values()) Object.assign(element, fakeElement());
}

function assertCleared(message) {
  assert.equal(elements.get("portfolioUnavailable").textContent, message);
  assert.equal(elements.get("portfolioMetrics").innerHTML, "");
  assert.equal(elements.get("portfolioChart").innerHTML, "");
  assert.equal(elements.get("portfolioStartDate").value, "");
  assert.equal(elements.get("portfolioEndDate").value, "");
  assert.equal(elements.get("portfolioApplyRange").disabled, true);
}

async function run() {
  const metrics = context.testRangeMetrics(curvePayload("A").series);
  assert.equal(metrics.starting_equity, 1000);
  assert.equal(metrics.ending_equity, 900);
  assert.ok(Math.abs(metrics.maximum_drawdown + .1) < 1e-12);
  assert.ok(Math.abs(metrics.conditional_drawdown_at_risk_95 + .1) < 1e-12);

  for (const fixture of cdarFixtures) {
    const values = [...fixture.values, ...Array(fixture.repeat_zeros).fill(0)];
    assert.ok(
      Math.abs(context.testCDaR(values, .95) - fixture.expected) < 1e-12,
      fixture.name
    );
  }
  assert.ok(Math.abs(context.testCDaR([NaN, Infinity, -Infinity, -.2, 0], .95) + .2) < 1e-12);
  assert.throws(() => context.testCDaR([.01, 0], .95));
  assert.equal(context.testCDaR([5e-10, 0], .95), 0);

  // Published -> unpublished clears every selected-strategy value immediately.
  reset();
  context.fetch = async url => response(curvePayload(url.includes("A.json") ? "A" : "B"));
  await context.testRenderPortfolio({ strategy_key: "A" });
  assert.equal(context.testSelectedKey(), "A");
  assert.notEqual(elements.get("portfolioChart").innerHTML, "");
  await context.testRenderPortfolio({ strategy_key: "U" });
  assertCleared("Portfolio curve not published under the bounded publication policy.");

  // Unpublished -> published restores only the newly selected strategy.
  await context.testRenderPortfolio({ strategy_key: "B" });
  assert.equal(context.testSelectedKey(), "B");
  assert.equal(elements.get("portfolioApplyRange").disabled, false);

  // Published A -> published B, with A resolving last, cannot repaint A.
  reset();
  const slowA = deferred();
  const fastB = deferred();
  context.fetch = url => url.includes("A.json") ? slowA.promise : fastB.promise;
  const renderA = context.testRenderPortfolio({ strategy_key: "A" });
  const renderB = context.testRenderPortfolio({ strategy_key: "B" });
  fastB.resolve(response(curvePayload("B", 1100)));
  await renderB;
  slowA.resolve(response(curvePayload("A", 800)));
  await renderA;
  assert.equal(context.testSelectedKey(), "B");
  assert.match(elements.get("portfolioMetrics").innerHTML, /1100\.00/);

  // Published A -> unpublished B, with A resolving last, remains cleared.
  reset();
  const lateA = deferred();
  context.fetch = () => lateA.promise;
  const lateRender = context.testRenderPortfolio({ strategy_key: "A" });
  await context.testRenderPortfolio({ strategy_key: "U" });
  lateA.resolve(response(curvePayload("A")));
  await lateRender;
  assertCleared("Portfolio curve not published under the bounded publication policy.");

  // Repeated rapid switching deterministically leaves the final selection.
  reset();
  const requests = [];
  context.fetch = url => {
    const item = deferred();
    requests.push({ url, ...item });
    return item.promise;
  };
  const rapidA1 = context.testRenderPortfolio({ strategy_key: "A" });
  const rapidB = context.testRenderPortfolio({ strategy_key: "B" });
  const rapidA2 = context.testRenderPortfolio({ strategy_key: "A" });
  requests[2].resolve(response(curvePayload("A", 1200)));
  await rapidA2;
  requests[1].resolve(response(curvePayload("B", 700)));
  requests[0].resolve(response(curvePayload("A", 800)));
  await Promise.all([rapidA1, rapidB]);
  assert.equal(context.testSelectedKey(), "A");
  assert.match(elements.get("portfolioMetrics").innerHTML, /1200\.00/);

  // Missing benchmark remains explicit while the strategy still renders.
  reset(["A"]);
  context.portfolioBenchmark = { status: "Not available", reason: "missing", series: [] };
  context.fetch = async () => response(curvePayload("A"));
  await context.testRenderPortfolio({ strategy_key: "A" });
  assert.match(elements.get("portfolioUnavailable").textContent, /SPY benchmark unavailable/);
  assert.notEqual(elements.get("portfolioChart").innerHTML, "");

  // Fetch, missing-file, malformed, and empty states are distinct and clear.
  for (const [status, payload, expected] of [
    [500, null, "Portfolio curve load failed."],
    [404, null, "Portfolio curve file missing unexpectedly."],
    [200, { status: "Available", curve_schema_version: 2, strategy_key: "A", series: [{}] }, "Portfolio curve response is malformed."],
    [200, { status: "Available", curve_schema_version: 2, strategy_key: "A", series: [] }, "Portfolio curve response is empty."]
  ]) {
    reset(["A"]);
    context.fetch = async () => response(payload, status);
    await context.testRenderPortfolio({ strategy_key: "A" });
    assertCleared(expected);
  }

  const visible = [...elements.values()].map(item => `${item.textContent}${item.innerHTML}`).join(" ");
  assert.equal(/NaN|Infinity|undefined/.test(visible), false);
  console.log("portfolio_dashboard=PASS");
}

run().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
