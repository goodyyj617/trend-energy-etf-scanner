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

    return String(av ?? "").localeCompare(String(bv ?? "")) * sortDir;
  });
}

function render() {
  const tbody = document.querySelector("#scannerTable tbody");
  if (!tbody || !payload) return;

  const filtered = sortRows(applyFilters(rows));
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

      if (numberFields.has(key) && Number(r[key]) > 0) {
        td.classList.add("positive");
      }

      if (numberFields.has(key) && Number(r[key]) < 0) {
        td.classList.add("negative");
      }

      tr.appendChild(td);
    }

    const linkTd = document.createElement("td");
    linkTd.innerHTML = `
      <a target="_blank" rel="noopener" href="https://finance.yahoo.com/quote/${r.symbol}">Yahoo</a>
      <a target="_blank" rel="noopener" href="https://www.tradingview.com/symbols/AMEX-${r.symbol}/">TV</a>
    `;
    tr.appendChild(linkTd);

    tbody.appendChild(tr);
  }

  const filteredSignals = filtered.filter(r => r.signal_surge_v0).length;

  document.getElementById("meta").textContent =
    `As of ${payload.as_of} | rows ${payload.row_count} | eligible ${payload.eligible_count} | filtered ${filtered.length} | signals ${filteredSignals}`;
}

async function init() {
  const res = await fetch("data/latest.json", { cache: "no-store" });
  payload = await res.json();
  rows = payload.rows || [];

  buildGroupFilterOptions();
  render();
}

document.querySelectorAll("input, select").forEach(el => {
  el.addEventListener("input", render);
  el.addEventListener("change", render);
});

document.getElementById("resetBtn").addEventListener("click", () => {
  document.getElementById("eligibleOnly").checked = true;
  document.getElementById("signalOnly").checked = false;

  const groupFilter = document.getElementById("groupFilter");
  if (groupFilter) groupFilter.value = "All";

  document.getElementById("minR63").value = 0.03;
  document.getElementById("minER63").value = 0.20;
  document.getElementById("minSurge").value = 1.25;
  document.getElementById("maxATR").value = 0.06;
  document.getElementById("maxDollarRank").value = 500;

  render();
});

document.querySelectorAll("th[data-key]").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;

    if (sortKey === key) {
      sortDir *= -1;
    } else {
      sortKey = key;
      sortDir = numberFields.has(key) ? -1 : 1;
    }

    render();
  });
});

init().catch(err => {
  document.getElementById("meta").textContent =
    `Failed to load data/latest.json: ${err}`;
});
