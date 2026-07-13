let payload = null;
let backtestPayload = null;
let rows = [];
let activeTab = "scanner";
let sortKey = "score";
let sortDir = -1;

const numberFields = new Set([
  "aum", "dollar_vol_rank", "close", "entry_price", "exit_price",
  "r63", "r126", "er63", "te63", "te126", "score", "surge_ratio", "atr20_pct",
  "low10", "low20", "suggested_stop", "stop_distance_pct",
  "signal_streak_trading_days", "signal_streak_calendar_days",
  "trades", "signals", "skipped_stop_broken", "win_rate", "avg_return", "median_return", "total_return", "max_drawdown",
  "avg_holding_days", "profit_factor", "stop_hit_rate", "max_hold_exit_rate",
  "net_return", "gross_return", "holding_days",
  "avg_fwd_1d", "avg_fwd_3d", "avg_fwd_5d", "avg_fwd_10d", "avg_fwd_20d",
  "median_fwd_1d", "median_fwd_3d", "median_fwd_5d", "median_fwd_10d", "median_fwd_20d",
  "win_rate_1d", "win_rate_3d", "win_rate_5d", "win_rate_10d", "win_rate_20d",
  "avg_mfe_20d", "median_mfe_20d", "avg_mae_20d", "median_mae_20d"
]);

const heatmapFields = new Set([
  "aum", "dollar_vol_rank", "r63", "r126", "er63", "te63", "te126", "score",
  "surge_ratio", "atr20_pct", "suggested_stop", "stop_distance_pct", "signal_streak_trading_days",
  "trades", "signals", "skipped_stop_broken", "win_rate", "avg_return", "median_return", "total_return",
  "max_drawdown", "profit_factor", "stop_hit_rate", "net_return",
  "avg_fwd_1d", "avg_fwd_3d", "avg_fwd_5d", "avg_fwd_10d", "avg_fwd_20d",
  "median_fwd_20d", "win_rate_20d", "avg_mfe_20d", "avg_mae_20d"
]);

const lowerIsBetterFields = new Set([
  "dollar_vol_rank", "atr20_pct", "stop_distance_pct", "max_drawdown", "stop_hit_rate", "skipped_stop_broken", "avg_mae_20d"
]);

const displayNames = {
  symbol: "Symbol",
  name: "Name",
  asset_group: "Group",
  group: "Group",
  Group: "Group",
  aum: "AUM",
  dollar_vol_rank: "DV Rank",
  close: "Close",
  entry_price: "Entry",
  exit_price: "Exit",
  r63: "R63",
  r126: "R126",
  er63: "ER63",
  te63: "TE63",
  te126: "TE126",
  score: "Score",
  surge_ratio: "Surge",
  atr20_pct: "ATR20%",
  signal_surge_v0: "Signal",
  signal_streak_start_date: "First Signal",
  signal_streak_trading_days: "Signal Days",
  signal_streak_calendar_days: "Calendar Days",
  is_first_signal_today: "New?",
  suggested_stop: "Suggested Stop",
  stop_distance_pct: "Stop Dist.",
  strategy_label: "Strategy",
  signal_label: "Signal",
  entry_label: "Entry",
  exit_label: "Exit",
  trades: "Trades",
  signals: "Signals",
  skipped_stop_broken: "Skipped",
  win_rate: "Win Rate",
  avg_return: "Avg Ret",
  median_return: "Median Ret",
  total_return: "Total Ret",
  max_drawdown: "Max DD",
  avg_holding_days: "Avg Days",
  profit_factor: "Profit Factor",
  stop_hit_rate: "Stop Hit",
  max_hold_exit_rate: "Max Hold Exit",
  entry_signal_date: "Signal Date",
  entry_date: "Entry Date",
  exit_date: "Exit Date",
  exit_reason: "Exit Reason",
  net_return: "Net Ret",
  gross_return: "Gross Ret",
  holding_days: "Days",
  avg_fwd_1d: "Avg Fwd 1D",
  avg_fwd_3d: "Avg Fwd 3D",
  avg_fwd_5d: "Avg Fwd 5D",
  avg_fwd_10d: "Avg Fwd 10D",
  avg_fwd_20d: "Avg Fwd 20D",
  median_fwd_20d: "Median Fwd 20D",
  win_rate_20d: "Win 20D",
  avg_mfe_20d: "Avg MFE20",
  avg_mae_20d: "Avg MAE20"
};

const presets = {
  exploratory: { label: "탐색형", minR63: 0.00, minER63: 0.15, minSurge: 1.00, maxATR: 0.08, maxDollarRank: 500 },
  basic: { label: "기본형", minR63: 0.03, minER63: 0.20, minSurge: 1.10, maxATR: 0.06, maxDollarRank: 500 },
  strict: { label: "엄격형", minR63: 0.05, minER63: 0.25, minSurge: 1.25, maxATR: 0.05, maxDollarRank: 500 }
};

function getGroup(row) {
  return row.group || row.Group || row.asset_group || row.assetGroup || row.AssetGroup || row["Asset Group"] || "unknown";
}

function normalizeGroupName(value) {
  return String(value || "unknown").trim().toLowerCase();
}

function buildGroupFilterOptions() {
  const select = document.getElementById("groupFilter");
  if (!select) return;
  const currentValue = select.value || "All";
  const groups = [...new Set(rows.map(r => getGroup(r)).filter(Boolean))]
    .sort((a, b) => String(a).localeCompare(String(b)));
  select.innerHTML = "";
  const allOption = document.createElement("option");
  allOption.value = "All";
  allOption.textContent = "All";
  select.appendChild(allOption);
  for (const group of groups) {
    const opt = document.createElement("option");
    opt.value = group;
    opt.textContent = group;
    select.appendChild(opt);
  }
  select.value = groups.includes(currentValue) ? currentValue : "All";
}

function setInputValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value;
}

function applyPreset(presetKey) {
  const preset = presets[presetKey];
  if (!preset) return;
  setInputValue("minR63", preset.minR63.toFixed(2));
  setInputValue("minER63", preset.minER63.toFixed(2));
  setInputValue("minSurge", preset.minSurge.toFixed(2));
  setInputValue("maxATR", preset.maxATR.toFixed(2));
  setInputValue("maxDollarRank", preset.maxDollarRank);
}

function markCustomPreset() {
  const presetSelect = document.getElementById("presetFilter");
  if (presetSelect) presetSelect.value = "custom";
}

function fmt(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";

  if (key === "aum") {
    return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
  }

  if ([
    "r63", "r126", "er63", "te63", "te126", "score", "surge_ratio", "atr20_pct",
    "stop_distance_pct", "win_rate", "avg_return", "median_return", "total_return", "max_drawdown",
    "stop_hit_rate", "max_hold_exit_rate", "net_return", "gross_return",
    "avg_fwd_1d", "avg_fwd_3d", "avg_fwd_5d", "avg_fwd_10d", "avg_fwd_20d",
    "median_fwd_1d", "median_fwd_3d", "median_fwd_5d", "median_fwd_10d", "median_fwd_20d",
    "win_rate_1d", "win_rate_3d", "win_rate_5d", "win_rate_10d", "win_rate_20d",
    "avg_mfe_20d", "median_mfe_20d", "avg_mae_20d", "median_mae_20d"
  ].includes(key)) {
    return Number(value).toFixed(3);
  }

  if (["profit_factor"].includes(key)) return Number(value).toFixed(2);
  if (["close", "low10", "low20", "suggested_stop", "entry_price", "exit_price"].includes(key)) return Number(value).toFixed(2);
  if (["dollar_vol_rank", "signal_streak_trading_days", "signal_streak_calendar_days", "trades", "signals", "skipped_stop_broken", "holding_days"].includes(key)) return Math.round(Number(value));
  if (["avg_holding_days"].includes(key)) return Number(value).toFixed(1);
  if (key === "signal_surge_v0") return value ? "TRUE" : "";
  if (key === "is_first_signal_today") return value ? "NEW" : "";
  return value;
}

function getFilters() {
  return {
    eligibleOnly: document.getElementById("eligibleOnly").checked,
    signalOnly: document.getElementById("signalOnly").checked,
    groupFilter: document.getElementById("groupFilter")?.value || "All",
    minR63: Number(document.getElementById("minR63").value),
    minER63: Number(document.getElementById("minER63").value),
    minSurge: Number(document.getElementById("minSurge").value),
    maxATR: Number(document.getElementById("maxATR").value),
    maxDollarRank: Number(document.getElementById("maxDollarRank").value)
  };
}

function applyFilters(data) {
  const f = getFilters();
  const selectedGroup = normalizeGroupName(f.groupFilter);
  return data.filter(r => {
    const rowGroup = normalizeGroupName(getGroup(r));
    if (selectedGroup !== "all" && rowGroup !== selectedGroup) return false;
    if (f.eligibleOnly && !r.eligible_universe) return false;
    if (f.signalOnly && !r.signal_surge_v0) return false;
    if ((r.r63 ?? -999) < f.minR63) return false;
    if ((r.er63 ?? -999) < f.minER63) return false;
    if ((r.surge_ratio ?? -999) < f.minSurge) return false;
    if ((r.atr20_pct ?? 999) > f.maxATR) return false;
    if ((r.dollar_vol_rank ?? 999999) > f.maxDollarRank) return false;
    return true;
  });
}

function getActiveSignals(data) {
  return data.filter(r => r.signal_surge_v0);
}

function getSortValue(row, key) {
  if (key === "asset_group" || key === "group" || key === "Group") return getGroup(row);
  return row[key];
}

function sortRows(data) {
  return [...data].sort((a, b) => {
    const av = getSortValue(a, sortKey);
    const bv = getSortValue(b, sortKey);
    if (numberFields.has(sortKey)) return ((av ?? -Infinity) - (bv ?? -Infinity)) * sortDir;
    if (sortKey === "signal_surge_v0" || sortKey === "is_first_signal_today") return ((av ? 1 : 0) - (bv ? 1 : 0)) * sortDir;
    return String(av ?? "").localeCompare(String(bv ?? "")) * sortDir;
  });
}

function getSortLabel() {
  const name = displayNames[sortKey] || sortKey;
  const arrow = sortDir === -1 ? "↓" : "↑";
  return `${name} ${arrow}`;
}

function updateHeaderSortIndicators() {
  document.querySelectorAll("th[data-key]").forEach(th => {
    if (!th.dataset.label) th.dataset.label = th.textContent.trim();
    const key = th.dataset.key;
    const baseLabel = th.dataset.label;
    if (key === sortKey) {
      th.textContent = `${baseLabel} ${sortDir === -1 ? "↓" : "↑"}`;
      th.classList.add("sorted");
    } else {
      th.textContent = baseLabel;
      th.classList.remove("sorted");
    }
  });
}

function getFiniteValues(data, key) {
  return data.map(row => Number(getSortValue(row, key))).filter(value => Number.isFinite(value));
}

function buildHeatmapStats(data, keys) {
  const stats = {};
  for (const key of keys) {
    if (!heatmapFields.has(key)) continue;
    const values = getFiniteValues(data, key);
    if (!values.length) continue;
    stats[key] = { min: Math.min(...values), max: Math.max(...values) };
  }
  return stats;
}

function getHeatIntensity(value, key, stats) {
  if (!stats[key]) return 0;
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  const { min, max } = stats[key];
  if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) return 0.25;
  let normalized = (num - min) / (max - min);
  normalized = Math.max(0, Math.min(1, normalized));
  if (lowerIsBetterFields.has(key)) normalized = 1 - normalized;
  return normalized;
}

function applyHeatmapStyle(td, key, value, stats) {
  if (!heatmapFields.has(key)) return;
  const intensity = getHeatIntensity(value, key, stats);
  td.classList.add("heatmap-cell");
  td.style.setProperty("--heat", intensity.toFixed(3));
}

function addCell(tr, row, key, heatmapStats) {
  const td = document.createElement("td");
  td.textContent = key === "asset_group" ? getGroup(row) : fmt(key, row[key]);
  if (key === "symbol") td.classList.add("ticker");
  if (key === "signal_surge_v0" && row.signal_surge_v0) td.classList.add("signal-badge");
  if (key === "is_first_signal_today" && row.is_first_signal_today) td.classList.add("new-badge");
  if (key === "exit_reason") td.classList.add(`exit-${String(row[key] || "").replaceAll("_", "-")}`);
  if (numberFields.has(key)) {
    td.classList.add("numeric");
    applyHeatmapStyle(td, key, row[key], heatmapStats);
  }
  tr.appendChild(td);
}

function addLinksCell(tr, row) {
  const linkTd = document.createElement("td");
  linkTd.classList.add("links");
  linkTd.innerHTML = `
    <a target="_blank" rel="noopener" href="https://finance.yahoo.com/quote/${row.symbol}">Yahoo</a>
    <a target="_blank" rel="noopener" href="https://www.tradingview.com/symbols/AMEX-${row.symbol}/">TV</a>
  `;
  tr.appendChild(linkTd);
}

function renderTable(tableSelector, data, keys, options = {}) {
  const tbody = document.querySelector(`${tableSelector} tbody`);
  if (!tbody) return [];
  const sorted = sortRows(data);
  const heatmapStats = buildHeatmapStats(sorted, keys);
  tbody.innerHTML = "";
  for (const row of sorted) {
    const tr = document.createElement("tr");
    if (row.signal_surge_v0) tr.classList.add("signal");
    for (const key of keys) addCell(tr, row, key, heatmapStats);
    if (options.links) addLinksCell(tr, row);
    tbody.appendChild(tr);
  }
  return sorted;
}

function renderScanner() {
  const keys = ["symbol", "name", "asset_group", "aum", "dollar_vol_rank", "close", "r63", "r126", "er63", "te63", "te126", "score", "surge_ratio", "atr20_pct", "signal_surge_v0"];
  const filtered = applyFilters(rows);
  const rendered = renderTable("#scannerTable", filtered, keys, { links: true }) || [];
  const filteredSignals = rendered.filter(r => r.signal_surge_v0).length;
  document.getElementById("meta").textContent =
    `As of ${payload.as_of} | rows ${payload.row_count} | eligible ${payload.eligible_count} | filtered ${rendered.length} | signals ${filteredSignals} | sorted by ${getSortLabel()}`;
}

function renderActiveSignals() {
  const keys = ["symbol", "name", "asset_group", "signal_streak_start_date", "signal_streak_trading_days", "signal_streak_calendar_days", "is_first_signal_today", "close", "suggested_stop", "stop_distance_pct", "r63", "er63", "score", "surge_ratio"];
  const activeSignals = getActiveSignals(rows);
  const rendered = renderTable("#activeSignalsTable", activeSignals, keys, { links: true }) || [];
  const newSignals = rendered.filter(r => r.is_first_signal_today).length;
  document.getElementById("meta").textContent =
    `As of ${payload.as_of} | active signals ${rendered.length} | new today ${newSignals} | sorted by ${getSortLabel()}`;
}

function renderDiagnosticDefinitions() {
  const box = document.getElementById("diagnosticDefs");
  if (!box || !backtestPayload?.diagnostic_definitions) return;
  const d = backtestPayload.diagnostic_definitions;
  box.innerHTML = `
    <strong>Diagnostic definitions:</strong>
    <span><b>Fwd Nd</b> = ${d.fwd_Nd || "N-day forward return from entry."}</span>
    <span><b>MFE20</b> = ${d.mfe_20d || "Maximum favorable excursion over 20 trading days."}</span>
    <span><b>MAE20</b> = ${d.mae_20d || "Maximum adverse excursion over 20 trading days."}</span>
    <span><b>Skipped</b> = ${d.skipped_stop_broken || "Skipped trades where entry open was already below the initial stop."}</span>
  `;
}

function renderBacktest() {
  const summaryKeys = ["strategy_label", "signal_label", "entry_label", "exit_label", "trades", "skipped_stop_broken", "win_rate", "avg_return", "median_return", "total_return", "max_drawdown", "avg_holding_days", "profit_factor", "stop_hit_rate", "max_hold_exit_rate"];
  const diagnosticKeys = ["signal_label", "entry_label", "signals", "avg_fwd_1d", "avg_fwd_3d", "avg_fwd_5d", "avg_fwd_10d", "avg_fwd_20d", "median_fwd_20d", "win_rate_20d", "avg_mfe_20d", "avg_mae_20d"];
  const tradeKeys = ["strategy_label", "signal_label", "entry_label", "exit_label", "symbol", "name", "asset_group", "entry_signal_date", "entry_date", "entry_price", "exit_date", "exit_price", "net_return", "holding_days", "exit_reason"];

  const summary = backtestPayload?.summary || [];
  const diagnostics = backtestPayload?.diagnostic_summary || [];
  const recentTrades = backtestPayload?.recent_trades || [];
  renderTable("#backtestSummaryTable", summary, summaryKeys);
  renderTable("#signalDiagnosticsTable", diagnostics, diagnosticKeys);
  renderTable("#backtestTradesTable", recentTrades, tradeKeys);
  renderDiagnosticDefinitions();

  const meta = document.getElementById("backtestMeta");
  if (meta && backtestPayload) {
    meta.textContent = `As of ${backtestPayload.as_of} | cost: ${(backtestPayload.round_trip_cost * 100).toFixed(2)}% round trip | max hold: ${backtestPayload.max_holding_days} trading days | signal FALSE exit: disabled`;
  }

  document.getElementById("meta").textContent =
    `Backtest as of ${backtestPayload?.as_of || payload.as_of} | strategies ${summary.length} | diagnostics ${diagnostics.length} | recent trades ${recentTrades.length} | sorted by ${getSortLabel()}`;
}

function render() {
  if (!payload) return;
  if (activeTab === "activeSignals") renderActiveSignals();
  else if (activeTab === "backtest") renderBacktest();
  else renderScanner();
  updateHeaderSortIndicators();
}

function setActiveTab(tabName) {
  activeTab = tabName;
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === tabName));
  document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));
  if (tabName === "activeSignals") {
    document.getElementById("activeSignalsPanel").classList.add("active");
    sortKey = "signal_streak_trading_days";
    sortDir = -1;
  } else if (tabName === "backtest") {
    document.getElementById("backtestPanel").classList.add("active");
    sortKey = "avg_return";
    sortDir = -1;
  } else {
    document.getElementById("scannerPanel").classList.add("active");
    sortKey = "score";
    sortDir = -1;
  }
  render();
}

async function init() {
  const res = await fetch("data/latest.json", { cache: "no-store" });
  payload = await res.json();
  rows = payload.rows || [];
  try {
    const backtestRes = await fetch("data/backtest_summary.json", { cache: "no-store" });
    if (backtestRes.ok) backtestPayload = await backtestRes.json();
  } catch (err) {
    backtestPayload = { as_of: payload.as_of, summary: [], diagnostic_summary: [], recent_trades: [] };
  }
  buildGroupFilterOptions();
  applyPreset("basic");
  render();
}

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
});

document.getElementById("presetFilter").addEventListener("change", event => {
  const presetKey = event.target.value;
  if (presetKey !== "custom") applyPreset(presetKey);
  render();
});

document.querySelectorAll("input[type='number']").forEach(el => {
  el.addEventListener("input", () => { markCustomPreset(); render(); });
  el.addEventListener("change", () => { markCustomPreset(); render(); });
});

document.querySelectorAll("input[type='checkbox'], select:not(#presetFilter)").forEach(el => {
  el.addEventListener("input", render);
  el.addEventListener("change", render);
});

document.getElementById("resetBtn").addEventListener("click", () => {
  document.getElementById("eligibleOnly").checked = true;
  document.getElementById("signalOnly").checked = false;
  const presetFilter = document.getElementById("presetFilter");
  if (presetFilter) presetFilter.value = "basic";
  const groupFilter = document.getElementById("groupFilter");
  if (groupFilter) groupFilter.value = "All";
  applyPreset("basic");
  if (activeTab === "activeSignals") sortKey = "signal_streak_trading_days";
  else if (activeTab === "backtest") sortKey = "avg_return";
  else sortKey = "score";
  sortDir = -1;
  render();
});

document.querySelectorAll("th[data-key]").forEach(th => {
  th.dataset.label = th.textContent.trim();
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (sortKey === key) sortDir *= -1;
    else {
      sortKey = key;
      sortDir = numberFields.has(key) || key === "signal_surge_v0" || key === "is_first_signal_today" ? -1 : 1;
    }
    render();
  });
});

init().catch(err => {
  document.getElementById("meta").textContent = `Failed to load data/latest.json: ${err}`;
});
