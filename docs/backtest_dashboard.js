const backtestPercentFields = new Set([
  "trade_win_rate", "avg_trade_return", "median_trade_return", "sum_trade_returns",
  "trade_sequence_drawdown", "worst_trade_return", "tail_return_10", "stop_hit_rate",
  "max_hold_exit_rate", "net_return", "gross_return", "neighbor_pass_ratio",
  "positive_year_ratio", "win_rate", "avg_return", "median_return", "max_drawdown"
]);

const baseFmtForBacktest = fmt;
fmt = function enhancedFmt(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  if (backtestPercentFields.has(key)) {
    const num = Number(value);
    if (!Number.isFinite(num)) return "";
    return `${(num * 100).toFixed(1)}%`;
  }
  return baseFmtForBacktest(key, value);
};

let selectedBacktestStrategyKey = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function parseSignalParams(row) {
  try {
    if (!row?.signal_params) return {};
    return typeof row.signal_params === "string" ? JSON.parse(row.signal_params) : row.signal_params;
  } catch (_err) {
    return {};
  }
}

function finiteNumber(value, fallback = null) {
  if (value === null || value === undefined || value === "") return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function pct(value) {
  const number = finiteNumber(value);
  return number === null ? "-" : `${(number * 100).toFixed(1)}%`;
}

function numFmt(value, digits = 2) {
  const number = finiteNumber(value);
  return number === null ? "-" : number.toFixed(digits);
}

function integerFmt(value) {
  const number = finiteNumber(value);
  return number === null ? "-" : Math.round(number).toLocaleString("en-US");
}

function dateRange(start, end) {
  if (!start && !end) return "Not available";
  return `${start || "open"} to ${end || "open"}`;
}

function normalizeCompletedTradeRow(row) {
  const params = parseSignalParams(row);
  return {
    ...row,
    completed_trades: row.completed_trades ?? row.trades ?? 0,
    avg_trade_return: row.avg_trade_return ?? row.avg_return,
    median_trade_return: row.median_trade_return ?? row.median_return,
    trade_win_rate: row.trade_win_rate ?? row.win_rate,
    // Never relabel legacy compounded total_return as an arithmetic sum.
    sum_trade_returns: row.sum_trade_returns ?? null,
    trade_sequence_drawdown: row.trade_sequence_drawdown ?? row.max_drawdown,
    avg_days: row.avg_days ?? row.avg_holding_days,
    score_lookback: params.score_lookback,
    r20_min: params.r20_min,
    er20_min: params.er20_min,
    robustness_tier: row.robustness_tier || "Awaiting regenerated data",
    robustness_tier_rank: finiteNumber(row.robustness_tier_rank, -1),
    mandatory_gates_passed: finiteNumber(row.mandatory_gates_passed, 0),
    parameter_stability_status: row.parameter_stability_status || "Not available",
    time_stability_status: row.time_stability_status || "Not available",
    regime_stability_status: row.regime_stability_status || "Not available"
  };
}

function availablePeriods() {
  const periods = backtestPayload?.period_analysis;
  if (Array.isArray(periods) && periods.length) return periods;
  return [{
    key: "all",
    label: "All available (regenerate data for Phase 1 analysis)",
    filter_mode: "legacy_summary_only",
    requested_entry_start: null,
    requested_entry_end: backtestPayload?.as_of || null,
    included_entry_start: null,
    included_entry_end: null,
    realized_exit_start: null,
    realized_exit_end: null,
    included_completed_trades: backtestPayload?.trade_count_total ?? null,
    summary: backtestPayload?.summary || [],
    yearly_details: []
  }];
}

function syncPeriodOptions(periods) {
  const select = document.getElementById("entryPeriodPreset");
  if (!select) return;
  const current = select.value || "all";
  select.innerHTML = "";
  for (const period of periods) {
    const option = document.createElement("option");
    option.value = period.key;
    option.textContent = period.label;
    select.appendChild(option);
  }
  select.value = periods.some(period => period.key === current) ? current : periods[0].key;
}

function selectedPeriod() {
  const periods = availablePeriods();
  syncPeriodOptions(periods);
  const key = document.getElementById("entryPeriodPreset")?.value || periods[0].key;
  return periods.find(period => period.key === key) || periods[0];
}

function candidateComparator(a, b) {
  const comparisons = [
    finiteNumber(b.robustness_tier_rank, -1) - finiteNumber(a.robustness_tier_rank, -1),
    finiteNumber(b.mandatory_gates_passed, 0) - finiteNumber(a.mandatory_gates_passed, 0),
    finiteNumber(b.median_trade_return, -Infinity) - finiteNumber(a.median_trade_return, -Infinity),
    finiteNumber(b.profit_factor, -Infinity) - finiteNumber(a.profit_factor, -Infinity),
    Math.abs(finiteNumber(a.trade_sequence_drawdown, Infinity)) - Math.abs(finiteNumber(b.trade_sequence_drawdown, Infinity)),
    finiteNumber(b.completed_trades, 0) - finiteNumber(a.completed_trades, 0)
  ];
  return comparisons.find(value => value !== 0 && Number.isFinite(value)) || String(a.strategy_key).localeCompare(String(b.strategy_key));
}

function gateBadge(status) {
  const normalized = String(status || "Not available").toLowerCase().replaceAll(" ", "-");
  return `<span class="gate-badge gate-${escapeHtml(normalized)}">${escapeHtml(status || "Not available")}</span>`;
}

function renderPeriodStatus(period) {
  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
  };
  setText("requestedEntryPeriod", dateRange(period.requested_entry_start, period.requested_entry_end));
  setText("includedEntryPeriod", dateRange(period.included_entry_start, period.included_entry_end));
  setText("realizedExitPeriod", dateRange(period.realized_exit_start, period.realized_exit_end));
  setText("includedCompletedTrades", integerFmt(period.included_completed_trades));
}

function snapshotCard(label, value, detail = "") {
  return `<div class="snapshot-item"><span>${escapeHtml(label)}</span><b>${value}</b>${detail ? `<small>${escapeHtml(detail)}</small>` : ""}</div>`;
}

function renderCandidateSnapshot(candidate, period) {
  const container = document.getElementById("candidateSnapshot");
  if (!container) return;
  if (!candidate) {
    container.innerHTML = '<div class="empty-state">No completed-trade candidate is available for this entry period.</div>';
    return;
  }
  const params = `L${candidate.score_lookback ?? "-"} / R20 ${numFmt(candidate.r20_min, 2)} / ER20 ${numFmt(candidate.er20_min, 2)}`;
  container.innerHTML = [
    snapshotCard("Robustness Tier", escapeHtml(candidate.robustness_tier), `${candidate.mandatory_gates_passed}/3 mandatory gates passed`),
    snapshotCard("Signal Parameters", escapeHtml(params)),
    snapshotCard("Entry / Exit", `${escapeHtml(candidate.entry_label)}<br>${escapeHtml(candidate.exit_label)}`),
    snapshotCard("Completed Trades", integerFmt(candidate.completed_trades), period.label),
    snapshotCard("Avg Trade Ret", pct(candidate.avg_trade_return)),
    snapshotCard("Median Trade Ret", pct(candidate.median_trade_return)),
    snapshotCard("Trade Win Rate", pct(candidate.trade_win_rate)),
    snapshotCard("Sum of Trade Returns", pct(candidate.sum_trade_returns), "Arithmetic event-level sum"),
    snapshotCard("Trade-Sequence DD", pct(candidate.trade_sequence_drawdown), "Not portfolio drawdown"),
    snapshotCard("Profit Factor", numFmt(candidate.profit_factor, 2)),
    snapshotCard("Avg Days", numFmt(candidate.avg_days, 1)),
    snapshotCard("Included Period", escapeHtml(dateRange(period.included_entry_start, period.included_entry_end)), `Realized exits: ${dateRange(period.realized_exit_start, period.realized_exit_end)}`)
  ].join("");
}

function componentCard(title, status, lines) {
  return `<div class="component-card"><div class="component-title"><b>${escapeHtml(title)}</b>${gateBadge(status)}</div>${lines.map(line => `<span>${line}</span>`).join("")}</div>`;
}

function renderComponentSummary(candidate) {
  const container = document.getElementById("componentSummary");
  if (!container) return;
  if (!candidate) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = [
    componentCard("Performance", candidate.mandatory_gate_status, [
      `Sample ${gateBadge(candidate.gate_sample)}`,
      `Median ${gateBadge(candidate.gate_median_trade_return)}`,
      `Profit Factor ${gateBadge(candidate.gate_profit_factor)}`
    ]),
    componentCard("Downside", "Descriptive", [
      `Trade-Sequence DD: <b>${pct(candidate.trade_sequence_drawdown)}</b>`,
      `Worst Trade Ret: <b>${pct(candidate.worst_trade_return)}</b>`,
      `10th Pctl Trade Ret: <b>${pct(candidate.tail_return_10)}</b>`
    ]),
    componentCard("Parameter Stability", candidate.parameter_stability_status, [
      `Eligible neighbors: <b>${integerFmt(candidate.eligible_neighbors)}</b>`,
      `Neighbor pass ratio: <b>${pct(candidate.neighbor_pass_ratio)}</b>`
    ]),
    componentCard("Time Stability", candidate.time_stability_status, [
      `Eligible entry years: <b>${integerFmt(candidate.eligible_years)}</b>`,
      `Positive year ratio: <b>${pct(candidate.positive_year_ratio)}</b>`
    ]),
    componentCard("Regime Stability", candidate.regime_stability_status, [
      "SPY history is not present in the current static pipeline; deferred without adding a dependency."
    ])
  ].join("");
}

function candidateCells(row, compact = false) {
  const parameterLabel = `L${row.score_lookback ?? "-"} / R20 ${numFmt(row.r20_min, 2)} / ER20 ${numFmt(row.er20_min, 2)}`;
  if (compact) {
    return `
      <td>${escapeHtml(row.robustness_tier)}</td><td class="numeric">${escapeHtml(row.score_lookback ?? "")}</td>
      <td class="numeric">${numFmt(row.r20_min, 2)}</td><td class="numeric">${numFmt(row.er20_min, 2)}</td>
      <td>${escapeHtml(row.entry_label)}</td><td>${escapeHtml(row.exit_label)}</td><td class="numeric">${integerFmt(row.completed_trades)}</td>
      <td class="numeric">${pct(row.avg_trade_return)}</td><td class="numeric">${pct(row.median_trade_return)}</td>
      <td class="numeric">${pct(row.trade_win_rate)}</td><td class="numeric">${pct(row.sum_trade_returns)}</td>
      <td class="numeric">${pct(row.trade_sequence_drawdown)}</td><td class="numeric">${numFmt(row.profit_factor, 2)}</td><td class="numeric">${numFmt(row.avg_days, 1)}</td>`;
  }
  return `
    <td>${escapeHtml(row.robustness_tier)}</td><td>${row.mandatory_gates_passed}/3</td>
    <td title="${escapeHtml(row.signal_label)}">${escapeHtml(parameterLabel)}</td><td>${escapeHtml(row.entry_label)}</td><td>${escapeHtml(row.exit_label)}</td>
    <td class="numeric">${integerFmt(row.completed_trades)}</td><td class="numeric">${pct(row.avg_trade_return)}</td>
    <td class="numeric">${pct(row.median_trade_return)}</td><td class="numeric">${pct(row.trade_win_rate)}</td>
    <td class="numeric">${pct(row.sum_trade_returns)}</td><td class="numeric">${pct(row.trade_sequence_drawdown)}</td>
    <td class="numeric">${pct(row.worst_trade_return)}</td><td class="numeric">${pct(row.tail_return_10)}</td>
    <td class="numeric">${numFmt(row.profit_factor, 2)}</td><td class="numeric">${pct(row.stop_hit_rate)}</td>
    <td class="numeric">${numFmt(row.avg_days, 1)}</td><td>${gateBadge(row.parameter_stability_status)}</td>
    <td>${gateBadge(row.time_stability_status)}</td><td>${gateBadge(row.regime_stability_status)}</td>`;
}

function renderCandidateTable(selector, rows, compact = false, limit = null) {
  const body = document.querySelector(`${selector} tbody`);
  if (!body) return;
  body.innerHTML = "";
  const visible = limit === null ? rows : rows.slice(0, limit);
  for (const row of visible) {
    const tr = document.createElement("tr");
    tr.className = "candidate-row";
    tr.dataset.strategyKey = row.strategy_key;
    if (row.strategy_key === selectedBacktestStrategyKey) tr.classList.add("selected-candidate");
    tr.innerHTML = candidateCells(row, compact);
    tr.addEventListener("click", () => {
      selectedBacktestStrategyKey = row.strategy_key;
      renderExitAppliedBacktest();
    });
    body.appendChild(tr);
  }
}

function renderParameterDetails(rows) {
  const body = document.querySelector("#parameterDetailsTable tbody");
  if (!body) return;
  body.innerHTML = rows.slice(0, 100).map(row => `
    <tr>
      <td>${escapeHtml(row.signal_label)} / ${escapeHtml(row.entry_label)} / ${escapeHtml(row.exit_label)}</td>
      <td>${gateBadge(row.parameter_stability_status)}</td><td class="numeric">${integerFmt(row.available_neighbors)}</td>
      <td class="numeric">${integerFmt(row.eligible_neighbors)}</td><td class="numeric">${pct(row.neighbor_pass_ratio)}</td>
      <td class="numeric">${pct(row.neighbor_median_return_min)} to ${pct(row.neighbor_median_return_max)}</td>
      <td class="numeric">${pct(row.neighbor_drawdown_min)} to ${pct(row.neighbor_drawdown_max)}</td>
    </tr>`).join("");
}

function renderYearlyDetails(period, candidate) {
  const body = document.querySelector("#yearlyDetailsTable tbody");
  if (!body) return;
  const rows = (period.yearly_details || []).filter(row => row.strategy_key === candidate?.strategy_key);
  body.innerHTML = rows.length ? rows.map(row => `
    <tr><td>${escapeHtml(row.year)}</td><td>${gateBadge(row.status)}</td><td class="numeric">${integerFmt(row.completed_trades)}</td>
    <td class="numeric">${pct(row.avg_trade_return)}</td><td class="numeric">${pct(row.median_trade_return)}</td>
    <td class="numeric">${pct(row.trade_win_rate)}</td><td class="numeric">${numFmt(row.profit_factor, 2)}</td>
    <td class="numeric">${pct(row.trade_sequence_drawdown)}</td></tr>`).join("")
    : '<tr><td colspan="8">No eligible entry-year details are available for this candidate.</td></tr>';
}

function renderThresholds() {
  const container = document.getElementById("activeThresholds");
  if (!container) return;
  const config = backtestPayload?.gate_config;
  if (!config) {
    container.textContent = "Active thresholds are unavailable until backtest data is regenerated with analysis schema v2.";
    return;
  }
  const labels = {
    min_completed_trades: "Minimum completed trades",
    min_bucket_trades: "Minimum trades per year bucket",
    min_eligible_neighbors: "Minimum eligible direct neighbors",
    min_profit_factor: "Minimum Profit Factor",
    min_median_trade_return: "Minimum Median Trade Ret",
    min_neighbor_pass_ratio: "Minimum neighbor pass ratio",
    min_positive_year_ratio: "Minimum positive year ratio",
    min_positive_regime_ratio: "Minimum positive regime ratio"
  };
  const percentKeys = new Set(["min_median_trade_return", "min_neighbor_pass_ratio", "min_positive_year_ratio", "min_positive_regime_ratio"]);
  container.innerHTML = `<strong>Active gate configuration: ${escapeHtml(config.version)}</strong>` + Object.entries(labels)
    .map(([key, label]) => `<span><b>${escapeHtml(label)}:</b> ${percentKeys.has(key) ? pct(config[key]) : escapeHtml(config[key])}</span>`)
    .join("") + '<span>These are transparent, revisable research decisions—not statistical proof.</span>';
}

function renderRecentTradesForPeriod(period) {
  const start = period.requested_entry_start ? new Date(`${period.requested_entry_start}T00:00:00`) : null;
  const end = period.requested_entry_end ? new Date(`${period.requested_entry_end}T23:59:59`) : null;
  const recent = (backtestPayload?.recent_trades || []).filter(row => {
    const entry = new Date(`${row.entry_date}T00:00:00`);
    return (!start || entry >= start) && (!end || entry <= end);
  });
  const keys = ["strategy_label", "signal_label", "entry_label", "exit_label", "symbol", "name", "asset_group", "entry_signal_date", "entry_date", "entry_price", "exit_date", "exit_price", "net_return", "holding_days", "exit_reason"];
  renderTable("#backtestTradesTable", recent, keys);
}

function renderExitAppliedBacktest() {
  if (!backtestPayload) return;
  const period = selectedPeriod();
  const rows = (period.summary || []).map(normalizeCompletedTradeRow).sort(candidateComparator);
  if (!rows.some(row => row.strategy_key === selectedBacktestStrategyKey)) {
    selectedBacktestStrategyKey = rows[0]?.strategy_key || null;
  }
  const candidate = rows.find(row => row.strategy_key === selectedBacktestStrategyKey) || rows[0] || null;

  renderPeriodStatus(period);
  renderCandidateSnapshot(candidate, period);
  renderComponentSummary(candidate);
  renderCandidateTable("#leaderboardTopTable", rows, true, 10);
  renderCandidateTable("#backtestSummaryTable", rows, false);
  renderParameterDetails(rows);
  renderYearlyDetails(period, candidate);
  renderThresholds();
  renderRecentTradesForPeriod(period);

  const meta = document.getElementById("backtestMeta");
  if (meta) {
    meta.textContent = `As of ${backtestPayload.as_of} | ${period.label} | cost: ${(backtestPayload.round_trip_cost * 100).toFixed(2)}% round trip | max hold: ${backtestPayload.max_holding_days} trading days | signal FALSE exit: disabled`;
  }
  const globalMeta = document.getElementById("meta");
  if (globalMeta) globalMeta.textContent = `Exit-applied analysis | ${rows.length} strategies | ${period.label} | selected: ${candidate?.strategy_label || "none"}`;
}

const baseRenderBacktest = renderBacktest;
renderBacktest = function completedTradeRenderBacktest() {
  renderExitAppliedBacktest();
};

document.getElementById("entryPeriodPreset")?.addEventListener("change", () => {
  selectedBacktestStrategyKey = null;
  renderExitAppliedBacktest();
});
