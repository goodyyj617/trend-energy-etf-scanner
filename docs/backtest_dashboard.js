const backtestPercentFields = new Set([
  "trade_win_rate", "avg_trade_return", "median_trade_return", "sum_trade_returns",
  "trade_sequence_drawdown", "worst_trade_return", "tail_return_10", "stop_hit_rate",
  "max_hold_exit_rate", "net_return", "gross_return", "neighbor_pass_ratio",
  "raw_neighbor_edge_pass_ratio", "effective_neighbor_edge_pass_ratio",
  "positive_year_ratio", "joint_positive_year_ratio", "loyo_pass_ratio",
  "win_rate", "avg_return", "median_return", "max_drawdown"
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
let backtestSortKey = null;
let backtestSortDir = -1;

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

function compareNumbers(leftValue, rightValue, direction = -1) {
  const left = finiteNumber(leftValue);
  const right = finiteNumber(rightValue);
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return (left - right) * direction;
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
  const hasProductionGates = row.sample_gate_pass !== undefined
    && row.edge_gate_pass !== undefined
    && row.time_gate_pass !== undefined
    && row.parameter_gate_pass !== undefined;
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
    qualification_tier: hasProductionGates ? (row.qualification_tier || "Not qualified") : "Not qualified",
    robustness_tier: hasProductionGates ? (row.qualification_tier || row.robustness_tier || "Not qualified") : "Not qualified",
    robustness_tier_rank: hasProductionGates ? finiteNumber(row.robustness_tier_rank, 0) : 0,
    mandatory_gates_passed: finiteNumber(row.mandatory_gates_passed, 0),
    mandatory_gates_pass: hasProductionGates ? row.mandatory_gates_pass === true : false,
    sample_gate_status: hasProductionGates ? (row.sample_gate_status || "Not available") : "Not available",
    edge_gate_status: hasProductionGates ? (row.edge_gate_status || "Not available") : "Not available",
    parameter_gate_status: hasProductionGates ? (row.parameter_gate_status || "Not available") : "Not available",
    time_gate_status: hasProductionGates ? (row.time_gate_status || "Not available") : "Not available",
    parameter_stability_status: hasProductionGates ? (row.parameter_gate_status || row.parameter_stability_status || "Not available") : "Not available",
    time_stability_status: hasProductionGates ? (row.time_gate_status || row.time_stability_status || "Not available") : "Not available",
    qualification_reason: hasProductionGates
      ? (row.qualification_reason || "Qualification reason unavailable.")
      : "Regenerate Backtest Only outputs with analysis schema v3; annual robustness gates are unavailable.",
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
    Number(b.qualification_tier === "Qualified") - Number(a.qualification_tier === "Qualified"),
    Number(b.time_gate_pass === true) - Number(a.time_gate_pass === true),
    Number(b.parameter_gate_pass === true) - Number(a.parameter_gate_pass === true),
    compareNumbers(a.loyo_pass_ratio, b.loyo_pass_ratio),
    compareNumbers(a.joint_positive_year_ratio, b.joint_positive_year_ratio),
    compareNumbers(a.effective_neighbor_edge_pass_ratio, b.effective_neighbor_edge_pass_ratio),
    compareNumbers(a.profit_factor, b.profit_factor),
    compareNumbers(a.avg_trade_return, b.avg_trade_return),
    compareNumbers(a.completed_trades, b.completed_trades)
  ];
  return comparisons.find(value => value !== 0) || String(a.strategy_key).localeCompare(String(b.strategy_key));
}

const backtestNumericSortFields = new Set([
  "completed_trades", "avg_trade_return", "median_trade_return", "profit_factor", "eligible_years",
  "joint_positive_year_ratio", "loyo_pass_ratio", "effective_neighbor_edge_pass_ratio"
]);

function sortCompletedTradeRows(rows) {
  if (!backtestSortKey) return [...rows].sort(candidateComparator);
  return [...rows].sort((a, b) => {
    let comparison;
    if (backtestNumericSortFields.has(backtestSortKey)) {
      comparison = compareNumbers(a[backtestSortKey], b[backtestSortKey], backtestSortDir);
    } else {
      comparison = String(a[backtestSortKey] ?? "").localeCompare(String(b[backtestSortKey] ?? ""));
      comparison *= backtestSortDir;
    }
    return comparison || String(a.strategy_key).localeCompare(String(b.strategy_key));
  });
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
    snapshotCard("Qualification Tier", gateBadge(candidate.qualification_tier), candidate.qualification_reason),
    snapshotCard("Sample Gate", gateBadge(candidate.sample_gate_status)),
    snapshotCard("Edge Gate", gateBadge(candidate.edge_gate_status)),
    snapshotCard("Time Gate", gateBadge(candidate.time_gate_status)),
    snapshotCard("Parameter Gate", gateBadge(candidate.parameter_gate_status)),
    snapshotCard("Signal Parameters", escapeHtml(params)),
    snapshotCard("Entry / Exit", `${escapeHtml(candidate.entry_label)}<br>${escapeHtml(candidate.exit_label)}`),
    snapshotCard("Completed Trades", integerFmt(candidate.completed_trades), period.label),
    snapshotCard("Avg Trade Ret", pct(candidate.avg_trade_return)),
    snapshotCard("Median Trade Ret", pct(candidate.median_trade_return), "Diagnostic only"),
    snapshotCard("Profit Factor", numFmt(candidate.profit_factor, 2)),
    snapshotCard("Joint-Positive-Year Ratio", pct(candidate.joint_positive_year_ratio)),
    snapshotCard("Eligible Full Years", integerFmt(candidate.eligible_years)),
    snapshotCard("LOYO Pass Ratio", pct(candidate.loyo_pass_ratio)),
    snapshotCard("Effective Neighbor Edge Pass Ratio", pct(candidate.effective_neighbor_edge_pass_ratio)),
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
    componentCard("Qualification", candidate.qualification_tier, [
      `Sample ${gateBadge(candidate.sample_gate_status)}`,
      `Edge ${gateBadge(candidate.edge_gate_status)}`,
      `Median Trade Ret: <b>${pct(candidate.median_trade_return)}</b> (diagnostic)`
    ]),
    componentCard("Downside", "Descriptive", [
      `Trade-Sequence DD: <b>${pct(candidate.trade_sequence_drawdown)}</b>`,
      `Worst Trade Ret: <b>${pct(candidate.worst_trade_return)}</b>`,
      `10th Pctl Trade Ret: <b>${pct(candidate.tail_return_10)}</b>`
    ]),
    componentCard("Parameter Stability", candidate.parameter_gate_status, [
      `Raw eligible / edge ratio: <b>${integerFmt(candidate.raw_eligible_neighbors)} / ${pct(candidate.raw_neighbor_edge_pass_ratio)}</b>`,
      `Effective eligible / edge ratio: <b>${integerFmt(candidate.effective_eligible_neighbors)} / ${pct(candidate.effective_neighbor_edge_pass_ratio)}</b>`
    ]),
    componentCard("Time Stability", candidate.time_gate_status, [
      `Eligible full entry years: <b>${integerFmt(candidate.eligible_years)}</b>`,
      `Joint-positive years / ratio: <b>${integerFmt(candidate.joint_positive_years)} / ${pct(candidate.joint_positive_year_ratio)}</b>`,
      `LOYO pass count / ratio: <b>${integerFmt(candidate.loyo_pass_count)} / ${pct(candidate.loyo_pass_ratio)}</b>`
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
      <td>${gateBadge(row.qualification_tier)}</td><td class="numeric">${escapeHtml(row.score_lookback ?? "")}</td>
      <td class="numeric">${numFmt(row.r20_min, 2)}</td><td class="numeric">${numFmt(row.er20_min, 2)}</td>
      <td>${escapeHtml(row.entry_label)}</td><td>${escapeHtml(row.exit_label)}</td><td class="numeric">${integerFmt(row.completed_trades)}</td>
      <td class="numeric">${pct(row.avg_trade_return)}</td><td class="numeric">${pct(row.median_trade_return)}</td>
      <td class="numeric">${numFmt(row.profit_factor, 2)}</td><td class="numeric">${integerFmt(row.eligible_years)}</td>
      <td class="numeric">${pct(row.joint_positive_year_ratio)}</td><td class="numeric">${pct(row.loyo_pass_ratio)}</td>
      <td class="numeric">${pct(row.effective_neighbor_edge_pass_ratio)}</td>`;
  }
  return `
    <td>${gateBadge(row.qualification_tier)}</td><td>${row.mandatory_gates_passed}/4</td>
    <td title="${escapeHtml(row.signal_label)}">${escapeHtml(parameterLabel)}</td><td>${escapeHtml(row.entry_label)}</td><td>${escapeHtml(row.exit_label)}</td>
    <td class="numeric">${integerFmt(row.completed_trades)}</td><td class="numeric">${pct(row.avg_trade_return)}</td>
    <td class="numeric">${pct(row.median_trade_return)}</td><td class="numeric">${numFmt(row.profit_factor, 2)}</td>
    <td class="numeric">${integerFmt(row.eligible_years)}</td><td class="numeric">${pct(row.joint_positive_year_ratio)}</td>
    <td class="numeric">${pct(row.loyo_pass_ratio)}</td><td class="numeric">${pct(row.effective_neighbor_edge_pass_ratio)}</td>
    <td class="numeric">${pct(row.trade_win_rate)}</td><td class="numeric">${pct(row.sum_trade_returns)}</td>
    <td class="numeric">${pct(row.trade_sequence_drawdown)}</td><td class="numeric">${pct(row.worst_trade_return)}</td>`;
}

function renderCandidateTable(selector, rows, compact = false, limit = null) {
  const body = document.querySelector(`${selector} tbody`);
  if (!body) return;
  body.innerHTML = "";
  const visible = limit === null ? rows : rows.slice(0, limit);
  for (const row of visible) {
    const tr = document.createElement("tr");
    tr.className = "candidate-row";
    tr.classList.add(row.qualification_tier === "Qualified" ? "qualified-candidate" : "not-qualified-candidate");
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
      <td>${gateBadge(row.parameter_gate_status)}</td><td class="numeric">${integerFmt(row.available_neighbors)}</td>
      <td class="numeric">${integerFmt(row.raw_eligible_neighbors)}</td><td class="numeric">${pct(row.raw_neighbor_edge_pass_ratio)}</td>
      <td class="numeric">${integerFmt(row.effective_eligible_neighbors)}</td><td class="numeric">${pct(row.effective_neighbor_edge_pass_ratio)}</td>
      <td class="numeric">${pct(row.neighbor_median_return_min)} to ${pct(row.neighbor_median_return_max)}</td>
      <td class="numeric">${pct(row.neighbor_drawdown_min)} to ${pct(row.neighbor_drawdown_max)}</td>
    </tr>`).join("");
}

function renderYearlyDetails(period, candidate) {
  const body = document.querySelector("#yearlyDetailsTable tbody");
  if (!body) return;
  const rows = (period.yearly_details || []).filter(row => row.strategy_key === candidate?.strategy_key);
  body.innerHTML = rows.length ? rows.map(row => `
    <tr><td>${escapeHtml(row.year)}</td><td>${row.is_partial_year === true ? "Partial" : (row.is_full_calendar_year === true ? "Full" : "Not available")}</td><td class="numeric">${integerFmt(row.completed_trades)}</td>
    <td class="numeric">${pct(row.avg_trade_return)}</td><td class="numeric">${pct(row.median_trade_return)}</td>
    <td class="numeric">${numFmt(row.profit_factor, 2)}</td><td class="numeric">${pct(row.sum_trade_returns)}</td>
    <td>${row.eligible_for_time_gate === true ? "Yes" : (row.eligible_for_time_gate === false ? "No" : "Not available")}</td>
    <td>${escapeHtml(row.joint_positive_status || "Not available")}</td></tr>`).join("")
    : '<tr><td colspan="9">Entry-year details are not available for this candidate. Time Gate data is unavailable and cannot qualify.</td></tr>';
}

function renderThresholds() {
  const container = document.getElementById("activeThresholds");
  if (!container) return;
  const config = backtestPayload?.gate_config;
  if (!config || finiteNumber(backtestPayload?.analysis_schema_version, 0) < 3) {
    container.textContent = "Production robustness thresholds are unavailable until Backtest Only data is regenerated with analysis schema v3. Existing data is not reinterpreted under the new gate model.";
    return;
  }
  const labels = {
    min_completed_trades: "Minimum completed trades",
    min_profit_factor: "Minimum overall Profit Factor",
    min_avg_trade_return: "Minimum overall average return (exclusive)",
    min_eligible_years: "Minimum eligible full entry years",
    min_annual_completed_trades: "Minimum completed trades per full year",
    min_annual_profit_factor: "Minimum annual Profit Factor (exclusive)",
    min_joint_positive_year_ratio: "Minimum joint-positive-year ratio",
    min_loyo_pass_ratio: "Minimum LOYO pass ratio",
    min_eligible_neighbors: "Minimum effective eligible neighbors",
    min_effective_neighbor_pass_ratio: "Minimum effective neighbor edge pass ratio"
  };
  const percentKeys = new Set([
    "min_avg_trade_return", "min_joint_positive_year_ratio", "min_loyo_pass_ratio",
    "min_effective_neighbor_pass_ratio"
  ]);
  container.innerHTML = `<strong>Active gate configuration: ${escapeHtml(config.version)}</strong>` + Object.entries(labels)
    .filter(([key]) => config[key] !== undefined && config[key] !== null)
    .map(([key, label]) => `<span><b>${escapeHtml(label)}:</b> ${percentKeys.has(key) ? pct(config[key]) : escapeHtml(config[key])}</span>`)
    .join("") + '<span>These are provisional in-sample robustness gates, not statistical proof or evidence of out-of-sample profitability.</span>';
}

function diagnosticLeader(rows, field) {
  const available = rows.filter(row => finiteNumber(row[field]) !== null);
  if (!available.length) return null;
  return [...available].sort((a, b) => (
    finiteNumber(b[field], -Infinity) - finiteNumber(a[field], -Infinity)
    || String(a.strategy_key).localeCompare(String(b.strategy_key))
  ))[0];
}

function renderDiagnosticLeaders(rows) {
  const container = document.getElementById("diagnosticLeaders");
  if (!container) return;
  const qualified = rows.filter(row => row.qualification_tier === "Qualified");
  if (!qualified.length) {
    container.innerHTML = '<div class="empty-state">No Qualified strategies are available for separate diagnostic leaders.</div>';
    return;
  }
  const definitions = [
    ["Highest Profit Factor", "profit_factor", value => numFmt(value, 2)],
    ["Highest Avg Trade Ret", "avg_trade_return", pct],
    ["Best Joint-Positive-Year Ratio", "joint_positive_year_ratio", pct],
    ["Best LOYO Pass Ratio", "loyo_pass_ratio", pct],
    ["Best Effective Parameter Robustness", "effective_neighbor_edge_pass_ratio", pct],
    ["Best Median Trade Ret (diagnostic)", "median_trade_return", pct]
  ];
  container.innerHTML = definitions.map(([label, field, formatter]) => {
    const leader = diagnosticLeader(qualified, field);
    return snapshotCard(label, leader ? escapeHtml(leader.strategy_label) : "-", leader ? formatter(leader[field]) : "Not available");
  }).join("");
}

function updateBacktestHeaderIndicators() {
  document.querySelectorAll("th[data-backtest-key]").forEach(th => {
    if (!th.dataset.label) th.dataset.label = th.textContent.trim();
    const active = th.dataset.backtestKey === backtestSortKey;
    th.textContent = active ? `${th.dataset.label} ${backtestSortDir === -1 ? "▼" : "▲"}` : th.dataset.label;
    th.classList.toggle("sorted", active);
  });
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
  const rows = sortCompletedTradeRows((period.summary || []).map(normalizeCompletedTradeRow));
  if (!rows.some(row => row.strategy_key === selectedBacktestStrategyKey)) {
    selectedBacktestStrategyKey = rows[0]?.strategy_key || null;
  }
  const candidate = rows.find(row => row.strategy_key === selectedBacktestStrategyKey) || rows[0] || null;

  renderPeriodStatus(period);
  renderCandidateSnapshot(candidate, period);
  renderComponentSummary(candidate);
  renderCandidateTable("#leaderboardTopTable", rows, true, 10);
  renderCandidateTable("#backtestSummaryTable", rows, false);
  renderDiagnosticLeaders(rows);
  renderParameterDetails(rows);
  renderYearlyDetails(period, candidate);
  renderThresholds();
  renderRecentTradesForPeriod(period);
  updateBacktestHeaderIndicators();

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
  backtestSortKey = null;
  renderExitAppliedBacktest();
});

document.querySelectorAll("th[data-backtest-key]").forEach(th => {
  th.dataset.label = th.textContent.trim();
  th.addEventListener("click", () => {
    const key = th.dataset.backtestKey;
    if (backtestSortKey === key) backtestSortDir *= -1;
    else {
      backtestSortKey = key;
      backtestSortDir = key === "qualification_tier" ? -1 : (backtestNumericSortFields.has(key) ? -1 : 1);
    }
    renderExitAppliedBacktest();
  });
});
