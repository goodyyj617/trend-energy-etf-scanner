let payload = null;
let rows = [];
let sortKey = "score";
let sortDir = -1;

const numberFields = new Set([
  "aum", "dollar_vol_rank", "close", "r63", "r126", "er63", "te63", "te126",
  "score", "surge_ratio", "atr20_pct"
]);

function fmt(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  if (key === "aum") return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
  if (["r63", "r126", "er63", "te63", "te126", "score", "surge_ratio", "atr20_pct"].includes(key)) {
    return Number(value).toFixed(3);
  }
  if (key === "close") return Number(value).toFixed(2);
  if (key === "dollar_vol_rank") return Math.round(Number(value));
  return value;
}

function getFilters() {
  return {
    eligibleOnly: document.getElementById("eligibleOnly").checked,
    signalOnly: document.getElementById("signalOnly").checked,
    minR63: Number(document.getElementById("minR63").value),
    minER63: Number(document.getElementById("minER63").value),
    minSurge: Number(document.getElementById("minSurge").value),
    maxATR: Number(document.getElementById("maxATR").value),
    maxDollarRank: Number(document.getElementById("maxDollarRank").value),
  };
}

function applyFilters(data) {
  const f = getFilters();
  return data.filter(r => {
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

function sortRows(data) {
  return [...data].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (numberFields.has(sortKey)) return ((av ?? -Infinity) - (bv ?? -Infinity)) * sortDir;
    return String(av ?? "").localeCompare(String(bv ?? "")) * sortDir;
  });
}

function render() {
  const tbody = document.querySelector("#scannerTable tbody");
  const filtered = sortRows(applyFilters(rows));
  tbody.innerHTML = "";

  for (const r of filtered) {
    const tr = document.createElement("tr");
    if (r.signal_surge_v0) tr.classList.add("signal");
    const keys = ["symbol", "name", "asset_group", "aum", "dollar_vol_rank", "close", "r63", "r126", "er63", "te63", "te126", "score", "surge_ratio", "atr20_pct", "signal_surge_v0"];
    for (const key of keys) {
      const td = document.createElement("td");
      td.textContent = fmt(key, r[key]);
      if (key === "symbol") td.classList.add("ticker");
      if (numberFields.has(key) && Number(r[key]) > 0) td.classList.add("positive");
      if (numberFields.has(key) && Number(r[key]) < 0) td.classList.add("negative");
      tr.appendChild(td);
    }
    const linkTd = document.createElement("td");
    linkTd.innerHTML = `<a target="_blank" rel="noopener" href="https://finance.yahoo.com/quote/${r.symbol}">Yahoo</a> <a target="_blank" rel="noopener" href="https://www.tradingview.com/symbols/AMEX-${r.symbol}/">TV</a>`;
    tr.appendChild(linkTd);
    tbody.appendChild(tr);
  }

  document.getElementById("meta").textContent = `As of ${payload.as_of} | rows ${payload.row_count} | eligible ${payload.eligible_count} | filtered ${filtered.length} | signals ${payload.signal_count}`;
}

async function init() {
  const res = await fetch("data/latest.json", { cache: "no-store" });
  payload = await res.json();
  rows = payload.rows;
  render();
}

document.querySelectorAll("input").forEach(el => el.addEventListener("input", render));
document.getElementById("resetBtn").addEventListener("click", () => {
  document.getElementById("eligibleOnly").checked = true;
  document.getElementById("signalOnly").checked = false;
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
    if (sortKey === key) sortDir *= -1;
    else { sortKey = key; sortDir = numberFields.has(key) ? -1 : 1; }
    render();
  });
});

init().catch(err => {
  document.getElementById("meta").textContent = `Failed to load data/latest.json: ${err}`;
});
