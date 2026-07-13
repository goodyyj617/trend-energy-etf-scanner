function parseSignalParams(row) {
  try {
    if (!row.signal_params) return {};
    return typeof row.signal_params === "string" ? JSON.parse(row.signal_params) : row.signal_params;
  } catch (err) {
    return {};
  }
}

function pct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "";
  return `${(num * 100).toFixed(2)}%`;
}

function numFmt(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "";
  return num.toFixed(digits);
}

function enrichedDiagnostics() {
  const diagnostics = backtestPayload?.diagnostic_summary || [];
  return diagnostics.map(row => {
    const params = parseSignalParams(row);
    const mfe = Number(row.avg_mfe_20d);
    const mae = Math.abs(Number(row.avg_mae_20d));
    const mfeMaeRatio = Number.isFinite(mfe) && Number.isFinite(mae) && mae > 0 ? mfe / mae : null;
    const robustScore =
      100 * Number(row.median_fwd_20d || 0) +
      50 * Number(row.avg_fwd_20d || 0) +
      5 * (Number(row.win_rate_20d || 0) - 0.5) +
      0.5 * ((mfeMaeRatio || 1) - 1);
    return {
      ...row,
      score_lookback: params.score_lookback,
      r20_min: params.r20_min,
      er20_min: params.er20_min,
      mfe_mae_ratio: mfeMaeRatio,
      robust_score: robustScore
    };
  });
}

function populateRobustEntryOptions(data) {
  const select = document.getElementById("robustEntryFilter");
  if (!select) return;
  const current = select.value || "All";
  const entries = [...new Set(data.map(r => r.entry_label).filter(Boolean))].sort();
  select.innerHTML = '<option value="All">All</option>';
  for (const entry of entries) {
    const opt = document.createElement("option");
    opt.value = entry;
    opt.textContent = entry;
    select.appendChild(opt);
  }
  select.value = entries.includes(current) ? current : "All";
}

function robustMetricLabel(metric) {
  return {
    median_fwd_20d: "Median Fwd20",
    avg_fwd_20d: "Avg Fwd20",
    win_rate_20d: "Win20",
    mfe_mae_ratio: "MFE/|MAE|",
    robust_score: "Robust Score"
  }[metric] || metric;
}

function addKpiCard(parent, label, row, metric) {
  const div = document.createElement("div");
  div.className = "kpi-card";
  if (!row) {
    div.innerHTML = `<div class="kpi-label">${label}</div><div class="kpi-value">-</div>`;
  } else {
    div.innerHTML = `
      <div class="kpi-label">${label}</div>
      <div class="kpi-value">${metric === "mfe_mae_ratio" || metric === "robust_score" ? numFmt(row[metric], 2) : pct(row[metric])}</div>
      <div class="kpi-sub">${row.signal_label} / ${row.entry_label}</div>
      <div class="kpi-sub">n=${row.signals} | L${row.score_lookback} | R20 ${numFmt(row.r20_min, 2)} | ER20 ${numFmt(row.er20_min, 2)}</div>
    `;
  }
  parent.appendChild(div);
}

function renderRobustnessTable(rows) {
  const tbody = document.querySelector("#robustnessTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.signal_label || ""}</td>
      <td>${row.entry_label || ""}</td>
      <td class="numeric">${row.score_lookback ?? ""}</td>
      <td class="numeric">${numFmt(row.r20_min, 2)}</td>
      <td class="numeric">${numFmt(row.er20_min, 2)}</td>
      <td class="numeric">${Math.round(Number(row.signals || 0))}</td>
      <td class="numeric heatmap-cell" style="--heat:${Math.max(0, Math.min(1, Number(row.avg_fwd_20d || 0) / 0.05)).toFixed(3)}">${pct(row.avg_fwd_20d)}</td>
      <td class="numeric heatmap-cell" style="--heat:${Math.max(0, Math.min(1, Number(row.median_fwd_20d || 0) / 0.04)).toFixed(3)}">${pct(row.median_fwd_20d)}</td>
      <td class="numeric">${pct(row.win_rate_20d)}</td>
      <td class="numeric">${pct(row.avg_mfe_20d)}</td>
      <td class="numeric">${pct(row.avg_mae_20d)}</td>
      <td class="numeric">${numFmt(row.mfe_mae_ratio, 2)}</td>
      <td class="numeric">${numFmt(row.robust_score, 2)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderRobustDashboard() {
  const data = enrichedDiagnostics();
  if (!data.length) return;
  populateRobustEntryOptions(data);

  const entryFilter = document.getElementById("robustEntryFilter")?.value || "All";
  const metric = document.getElementById("robustMetric")?.value || "median_fwd_20d";
  const minSignals = Number(document.getElementById("robustMinSignals")?.value || 300);

  const filtered = data
    .filter(row => entryFilter === "All" || row.entry_label === entryFilter)
    .filter(row => Number(row.signals || 0) >= minSignals);

  const ranked = [...filtered].sort((a, b) => Number(b[metric] || -Infinity) - Number(a[metric] || -Infinity));
  const topByMedian = [...filtered].sort((a, b) => Number(b.median_fwd_20d || -Infinity) - Number(a.median_fwd_20d || -Infinity))[0];
  const topByRobust = [...filtered].sort((a, b) => Number(b.robust_score || -Infinity) - Number(a.robust_score || -Infinity))[0];
  const topByRatio = [...filtered].sort((a, b) => Number(b.mfe_mae_ratio || -Infinity) - Number(a.mfe_mae_ratio || -Infinity))[0];

  const cards = document.getElementById("robustCards");
  if (cards) {
    cards.innerHTML = "";
    addKpiCard(cards, "Best Median Fwd20", topByMedian, "median_fwd_20d");
    addKpiCard(cards, "Best Robust Score", topByRobust, "robust_score");
    addKpiCard(cards, "Best MFE / |MAE|", topByRatio, "mfe_mae_ratio");
    const info = document.createElement("div");
    info.className = "kpi-card muted-card";
    info.innerHTML = `
      <div class="kpi-label">Current View</div>
      <div class="kpi-value">${ranked.length}</div>
      <div class="kpi-sub">metric: ${robustMetricLabel(metric)}</div>
      <div class="kpi-sub">entry: ${entryFilter} | min signals: ${minSignals}</div>
    `;
    cards.appendChild(info);
  }

  renderRobustnessTable(ranked.slice(0, 30));
}

const baseRenderBacktest = renderBacktest;
renderBacktest = function enhancedRenderBacktest() {
  baseRenderBacktest();
  renderRobustDashboard();
};

["robustEntryFilter", "robustMetric", "robustMinSignals"].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("input", render);
  el.addEventListener("change", render);
});
