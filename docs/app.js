let payload = null;
let rows = [];
let sortKey = "score";
let sortDir = -1;

const numberFields = new Set([
  "aum",
  "dollar_vol_rank",
  "close",
  "r63",
  "r126",
  "er63",
  "te63",
  "te126",
  "score",
  "surge_ratio",
  "atr20_pct"
]);

const heatmapFields = new Set([
  "aum",
  "dollar_vol_rank",
  "r63",
  "r126",
  "er63",
  "te63",
  "te126",
  "score",
  "surge_ratio",
  "atr20_pct"
]);

const lowerIsBetterFields = new Set([
  "dollar_vol_rank",
  "atr20_pct"
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
  r63: "R63",
  r126: "R126",
  er63: "ER63",
  te63: "TE63",
  te126: "TE126",
  score: "Score",
  surge_ratio: "Surge",
  atr20_pct: "ATR20%",
  signal_surge_v0: "Signal"
};

const presets = {
  exploratory: {
    label: "탐색형",
    minR63: 0.00,
    minER63: 0.15,
    minSurge: 1.00,
    maxATR: 0.08,
    maxDollarRank: 500
  },
  basic: {
    label: "기본형",
    minR63: 0.03,
    minER63: 0.20,
    minSurge: 1.10,
    maxATR: 0.06,
    maxDollarRank: 500
  },
  strict: {
    label: "엄격형",
    minR63: 0.05,
    minER63: 0.25,
    minSurge: 1.25,
    maxATR: 0.05,
    maxDollarRank: 500
  }
};

function getGroup(row) {
  return (
    row.group ||
    row.Group ||
    row.asset_group ||
    row.assetGroup ||
    row.AssetGroup ||
    row["Asset Group"] ||
    "unknown"
  );
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
    return Intl.NumberFormat("en", {
      notation: "compact",
      maximumFractionDigits: 1
    }).format(value);
  }

  if ([
    "r63",
    "r126",
    "er63",
    "te63",
    "te126",
    "score",
    "surge_ratio",
    "atr20_pct"
  ].includes(key)) {
    return Number(value).toFixed(3);
  }

  if (key === "close") {
    return Number(value).toFixed(2);
  }

  if (key === "dollar_vol_rank") {
    return Math.round(Number(value));
  }

  if (key === "signal_surge_v0") {
    return value ? "TRUE" : "";
  }

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

function getSortValue(row, key) {
  if (key === "asset_group" || key === "group" || key === "Group") {
    return getGroup(row);
  }

  return row[key];
}

function sortRows(data) {
  return [...data].sort((a, b) => {
    const av = getSortValue(a, sortKey);
    const bv = getSortValue(b, sortKey);

    if (numberFields.has(sortKey)) {
      return ((av ?? -Infinity) - (bv ?? -Infinity)) * sortDir;
    }

    if (sortKey === "signal_surge_v0") {
      return ((av ? 1 : 0) - (bv ? 1 : 0)) * sortDir;
    }

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
    if (!th.dataset.label) {
      th.dataset.label = th.textContent.trim();
    }

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
  return data
    .map(row => Number(getSortValue(row, key)))
    .filter(value => Number.isFinite(value));
}

function buildHeatmapStats(data) {
  const stats = {};

  for (const key of heatmapFields) {
    const values = getFiniteValues(data, key);
    if (!values.length) continue;

    const min = Math.min(...values);
    const max = Math.max(...values);

    stats[key] = { min, max };
  }

  return stats;
}

function getHeatIntensity(value, key, stats) {
  if (!stats[key]) return 0;

  const num = Number(value);
  if (!Number.isFinite(num)) return 0;

  const { min, max } = stats[key];
  if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) {
    return 0.25;
  }

  let normalized = (num - min) / (max - min);
  normalized = Math.max(0, Math.min(1, normalized));

  if (lowerIsBetterFields.has(key)) {
    normalized = 1 - normalized;
  }

  return normalized;
}

function applyHeatmapStyle(td, key, value, stats) {
  if (!heatmapFields.has(key)) return;

  const intensity = getHeatIntensity(value, key, stats);

  td.classList.add("heatmap-cell");
  td.style.setProperty("--heat", intensity.toFixed(3));
}

function render() {
  const tbody = document.querySelector("#scannerTable tbody");
  if (!tbody || !payload) return;

  const filtered = sortRows(applyFilters(rows));
  const heatmapStats = buildHeatmapStats(filtered);

  tbody.innerHTML = "";

  const keys = [
    "symbol",
    "name",
    "asset_group",
    "aum",
    "dollar_vol_rank",
    "close",
    "r63",
    "r126",
    "er63",
    "te63",
    "te126",
    "score",
    "surge_ratio",
    "atr20_pct",
    "signal_surge_v0"
  ];

  for (const r of filtered) {
    const tr = document.createElement("tr");

    if (r.signal_surge_v0) {
      tr.classList.add("signal");
    }

    for (const key of keys) {
      const td = document.createElement("td");

      if (key === "asset_group") {
        td.textContent = getGroup(r);
      } else {
        td.textContent = fmt(key, r[key]);
      }

      if (key === "symbol") {
        td.classList.add("ticker");
      }

      if (key === "signal_surge_v0" && r.signal_surge_v0) {
        td.classList.add("signal-badge");
      }

      if (numberFields.has(key)) {
        td.classList.add("numeric");
        applyHeatmapStyle(td, key, r[key], heatmapStats);
      }

      tr.appendChild(td);
    }

    const linkTd = document.createElement("td");
    linkTd.classList.add("links");
    linkTd.innerHTML = `
      <a target="_blank" rel="noopener" href="https://finance.yahoo.com/quote/${r.symbol}">Yahoo</a>
      <a target="_blank" rel="noopener" href="https://www.tradingview.com/symbols/AMEX-${r.symbol}/">TV</a>
    `;
    tr.appendChild(linkTd);

    tbody.appendChild(tr);
  }

  const filteredSignals = filtered.filter(r => r.signal_surge_v0).length;

  document.getElementById("meta").textContent =
    `As of ${payload.as_of} | rows ${payload.row_count} | eligible ${payload.eligible_count} | filtered ${filtered.length} | signals ${filteredSignals} | sorted by ${getSortLabel()}`;

  updateHeaderSortIndicators();
}

async function init() {
  const res = await fetch("data/latest.json", { cache: "no-store" });
  payload = await res.json();
  rows = payload.rows || [];

  buildGroupFilterOptions();
  applyPreset("basic");
  render();
}

document.getElementById("presetFilter").addEventListener("change", event => {
  const presetKey = event.target.value;
  if (presetKey !== "custom") {
    applyPreset(presetKey);
  }
  render();
});

document.querySelectorAll("input[type='number']").forEach(el => {
  el.addEventListener("input", () => {
    markCustomPreset();
    render();
  });
  el.addEventListener("change", () => {
    markCustomPreset();
    render();
  });
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

  sortKey = "score";
  sortDir = -1;

  render();
});

document.querySelectorAll("th[data-key]").forEach(th => {
  th.dataset.label = th.textContent.trim();

  th.addEventListener("click", () => {
    const key = th.dataset.key;

    if (sortKey === key) {
      sortDir *= -1;
    } else {
      sortKey = key;
      sortDir = numberFields.has(key) || key === "signal_surge_v0" ? -1 : 1;
    }

    render();
  });
});

init().catch(err => {
  document.getElementById("meta").textContent =
    `Failed to load data/latest.json: ${err}`;
});
