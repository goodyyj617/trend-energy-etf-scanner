"use strict";

const API = "/api/v1";
const CURVE_PAGE_SIZE = 250;
const LIST_PAGE_SIZE = 20;
const routes = {
  workflow: "워크플로",
  construction: "전략 구성",
  profiles: "평가 프로필 스튜디오",
  requests: "실행 요청",
  overview: "개요",
  runs: "저장된 전략 실행",
  evaluations: "평가 결과",
  performance: "성과 및 위험",
  robustness: "강건성",
  behavior: "행동 유사도",
  attempts: "실행 이력",
  explanations: "설명",
  system: "시스템 정보",
};

routes["decision-reports"] = "결정 보고서";

const state = {
  controller: null,
  terminology: {},
  statusLabels: {},
  runCursor: null,
  runCursorHistory: [],
  selectedPerformanceRun: null,
  selectedRobustnessRun: null,
  selectedEvaluationRun: null,
  selectedBehaviorEvaluation: null,
  selectedDecisionReport: null,
  constructionDraft: null,
  constructionEstimate: null,
  confirmationId: null,
  lastExecutionRequestId: null,
  studioSourceProfileId: null,
};

const view = document.getElementById("view");
const pageTitle = document.getElementById("page-title");
const loading = document.getElementById("loading-status");
const message = document.getElementById("global-message");
const connection = document.getElementById("connection-status");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    return new Intl.NumberFormat("ko-KR", { maximumFractionDigits: digits }).format(value);
  }
  return String(value);
}

function numericValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function shortId(value, length = 18) {
  const text = String(value ?? "");
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

function jsonText(value) {
  return escapeHtml(JSON.stringify(value, null, 2));
}

function setLoading(text = "") {
  loading.textContent = text;
}

function setMessage(text = "") {
  message.textContent = text;
}

function statusBadge(code, label) {
  const good = new Set(["available", "valid", "succeeded", "completed", "passed", "selected"]);
  const bad = new Set(["corrupt", "failed", "integrity_failed", "missing", "vetoed"]);
  const warn = new Set(["pruned", "partial", "never_generated", "unsupported_schema", "running", "queued"]);
  const css = good.has(code) ? "good" : bad.has(code) ? "bad" : warn.has(code) ? "warn" : "neutral";
  const icon = good.has(code) ? "✓" : bad.has(code) ? "✕" : "—";
  return `<span class="badge ${css}"><span aria-hidden="true">${icon}</span>${escapeHtml(label || code || "알 수 없음")}</span>`;
}

function availabilityLabel(code) {
  return state.statusLabels[code] || code || "알 수 없음";
}

async function api(path, { optional = false, method = "GET", body = null, idempotencyKey = null } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== null) headers["Content-Type"] = "application/json";
  if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
  const response = await fetch(`${API}${path}`, {
    method,
    signal: state.controller?.signal,
    headers,
    body: body === null ? undefined : JSON.stringify(body),
  });
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error("API 응답 형식을 확인할 수 없습니다.");
  }
  if (!response.ok) {
    const detail = payload?.error || {};
    if (optional) return { __error: detail, __status: response.status };
    const next = detail.next_action_ko ? ` ${detail.next_action_ko}` : "";
    throw new Error(`${detail.message_ko || "저장 결과를 불러오지 못했습니다."}${next}`);
  }
  return payload;
}

function idempotencyKey(operation) {
  return `${operation}:${crypto.randomUUID()}`;
}

function parameterSpace(text) {
  const value = String(text).trim();
  const ranged = value.match(/^(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)\s+step\s+(-?\d+(?:\.\d+)?)$/i);
  if (ranged) return { kind: "decimal_range", start: ranged[1], end: ranged[2], step: ranged[3] };
  if (value.includes(",")) return { kind: "list", values: value.split(",").map((item) => item.trim()) };
  return { kind: "fixed", value };
}

async function ensureTerminology() {
  if (Object.keys(state.terminology).length) return;
  const data = await api("/terminology");
  state.terminology = data.entries || {};
  state.statusLabels = data.status_labels || {};
}

function explanationButton(key, label) {
  if (!state.terminology[key]) return escapeHtml(label || key);
  return `<button type="button" class="definition-link" data-term="${escapeHtml(key)}">${escapeHtml(label || state.terminology[key].korean_term)}</button>`;
}

function bindExplanationLinks() {
  document.querySelectorAll("[data-term]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.term;
      location.hash = `#explanations/${encodeURIComponent(key)}`;
    });
  });
}

function emptyState(title, body) {
  return `<div class="card empty"><strong>${escapeHtml(title)}</strong>${escapeHtml(body)}</div>`;
}

function keyValues(entries) {
  return `<dl class="key-values">${entries.map(([label, value, isId = false]) => `
    <div><dt>${escapeHtml(label)}</dt><dd class="${isId ? "id" : ""}">${escapeHtml(value ?? "—")}</dd></div>`).join("")}</dl>`;
}

function dateRange(rows, dateKey = "economic_date") {
  if (!rows.length) return "표시할 날짜 없음";
  return `${rows[0][dateKey] || "—"} ~ ${rows[rows.length - 1][dateKey] || "—"}`;
}

function chartSummaryTable(rows, series) {
  const cells = series.map((item) => {
    const values = rows.map((row) => numericValue(row[item.key])).filter((value) => value !== null);
    if (!values.length) return `<tr><th scope="row">${escapeHtml(item.label)}</th><td colspan="4">사용 가능한 값 없음</td></tr>`;
    return `<tr><th scope="row">${escapeHtml(item.label)}</th><td>${fmt(values[0])}</td><td>${fmt(values.at(-1))}</td><td>${fmt(Math.min(...values))}</td><td>${fmt(Math.max(...values))}</td></tr>`;
  }).join("");
  return `<details><summary>차트 요약 표</summary><div class="table-wrap"><table>
    <thead><tr><th scope="col">계열</th><th scope="col">첫 값</th><th scope="col">마지막 값</th><th scope="col">최솟값</th><th scope="col">최댓값</th></tr></thead>
    <tbody>${cells}</tbody></table></div></details>`;
}

function lineChart({ title, rows, series, unit, dateKey = "economic_date", termKey = null }) {
  const available = rows.filter((row) => series.some((item) => numericValue(row[item.key]) !== null));
  if (!available.length) return `<article class="chart-card"><h3>${escapeHtml(title)}</h3>${emptyState("자료 없음", "선택한 범위에 저장된 값이 없습니다.")}</article>`;
  const width = 640, height = 260, left = 54, right = 16, top = 20, bottom = 42;
  const timestamps = available.map((row) => Date.parse(row[dateKey]));
  const values = available.flatMap((row) => series.map((item) => numericValue(row[item.key])).filter((value) => value !== null));
  let minX = Math.min(...timestamps), maxX = Math.max(...timestamps);
  let minY = Math.min(...values), maxY = Math.max(...values);
  if (minX === maxX) maxX += 86_400_000;
  if (minY === maxY) {
    const constantPadding = Math.max(Math.abs(minY) * .05, .01);
    minY -= constantPadding;
    maxY += constantPadding;
  }
  const pad = (maxY - minY) * .06;
  minY -= pad; maxY += pad;
  const x = (date) => left + (Date.parse(date) - minX) / (maxX - minX) * (width - left - right);
  const y = (value) => top + (maxY - Number(value)) / (maxY - minY) * (height - top - bottom);
  const grid = [0, .5, 1].map((p) => {
    const py = top + p * (height - top - bottom);
    const label = maxY - p * (maxY - minY);
    return `<line class="grid" x1="${left}" y1="${py}" x2="${width - right}" y2="${py}"/><text x="2" y="${py + 4}">${escapeHtml(fmt(label, 2))}</text>`;
  }).join("");
  const drawn = series.map((item, index) => {
    const className = index === 0 ? "series-a" : "series-b";
    const points = [];
    const segments = [];
    let previous = null;
    for (const row of available) {
      const value = numericValue(row[item.key]);
      if (value === null) { previous = null; continue; }
      const current = { x: x(row[dateKey]), y: y(value), date: Date.parse(row[dateKey]) };
      points.push(`<circle class="point-a" cx="${current.x}" cy="${current.y}" r="1.8"/>`);
      if (previous && current.date - previous.date <= 5 * 86_400_000) {
        segments.push(`<line class="${className}" x1="${previous.x}" y1="${previous.y}" x2="${current.x}" y2="${current.y}"/>`);
      }
      previous = current;
    }
    return segments.join("") + points.join("");
  }).join("");
  const firstDate = available[0][dateKey], lastDate = available.at(-1)[dateKey];
  const legend = `<div class="legend">${series.map((item) => `<span>${escapeHtml(item.label)}</span>`).join("")}</div>`;
  return `<article class="chart-card">
    <h3>${termKey ? explanationButton(termKey, title) : escapeHtml(title)}</h3>
    <p class="chart-meta">단위: ${escapeHtml(unit)} · ${escapeHtml(firstDate)} ~ ${escapeHtml(lastDate)} · 점이 없는 긴 간격은 선으로 연결하지 않음</p>
    <svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(`${title}, ${firstDate}부터 ${lastDate}까지, 단위 ${unit}`)}">
      ${grid}<line class="axis" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"/>
      <text x="${left}" y="${height - 14}">${escapeHtml(firstDate)}</text><text text-anchor="end" x="${width - right}" y="${height - 14}">${escapeHtml(lastDate)}</text>
      ${drawn}
    </svg>${legend}${chartSummaryTable(available, series)}
  </article>`;
}

function yearlyChart(rows) {
  const available = rows.filter((row) => numericValue(row.annual_return) !== null);
  if (!available.length) return `<article class="chart-card"><h3>연도별 수익률</h3>${emptyState("자료 없음", "저장된 연도별 수익률이 없습니다.")}</article>`;
  const width = 640, height = 260, left = 50, right = 12, top = 20, bottom = 44;
  const values = available.map((row) => numericValue(row.annual_return));
  const minY = Math.min(0, ...values), maxY = Math.max(0, ...values);
  const span = maxY - minY || 1;
  const y = (value) => top + (maxY - value) / span * (height - top - bottom);
  const zero = y(0), barWidth = (width - left - right) / available.length * .62;
  const bars = available.map((row, index) => {
    const value = numericValue(row.annual_return), cx = left + (index + .5) * (width - left - right) / available.length;
    const py = y(value), h = Math.abs(zero - py);
    return `<rect class="${value >= 0 ? "bar-pos" : "bar-neg"}" x="${cx - barWidth / 2}" y="${Math.min(zero, py)}" width="${barWidth}" height="${Math.max(1, h)}"><title>${escapeHtml(`${row.calendar_year}: ${fmt(value)}`)}</title></rect><text text-anchor="middle" x="${cx}" y="${height - 16}">${escapeHtml(row.calendar_year)}</text>`;
  }).join("");
  return `<article class="chart-card"><h3>${explanationButton("rolling_returns", "연도별 수익률")}</h3><p class="chart-meta">단위: 비율 · 녹색은 0 이상, 붉은색은 0 미만</p><svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="연도별 수익률 막대 차트"><line class="axis" x1="${left}" y1="${zero}" x2="${width-right}" y2="${zero}"/>${bars}</svg>${chartSummaryTable(available, [{key:"annual_return", label:"연도별 수익률"}])}</article>`;
}

function buildChartRows(curveRows, benchmarkRows) {
  const benchmarkByDate = new Map((benchmarkRows || []).map((row) => [row.economic_date, row]));
  return curveRows.map((row) => ({
    ...row,
    benchmark_value: benchmarkByDate.get(row.economic_date)?.portfolio_value ?? null,
  }));
}

function drawdownRows(rows) {
  let high = null;
  return rows.map((row) => {
    const value = Number(row.portfolio_value);
    high = high === null ? value : Math.max(high, value);
    return { economic_date: row.economic_date, drawdown: high ? value / high - 1 : null };
  });
}

function renderCharts(curve, benchmark, yearly, rolling, derived) {
  const curveRows = curve?.items || [];
  const commonRows = buildChartRows(curveRows, benchmark?.items || []);
  const rollingRows = rolling?.items || [];
  const yearlyRows = yearly?.items || [];
  const commonCount = commonRows.filter((row) => row.benchmark_value !== null).length;
  const provenance = derived?.payload?.benchmark_provenance || derived?.metadata?.benchmark_provenance || {};
  return `<div class="notice">차트는 한 요청당 최대 ${CURVE_PAGE_SIZE}개 행만 읽습니다. 벤치마크 선은 현재 범위의 정확히 같은 경제일 ${commonCount}개만 사용합니다. 저장된 공통일 출처: ${escapeHtml(provenance.common_observation_count ?? "확인 가능한 경우 파생 지표에 기록")}</div>
  <div class="chart-grid">
    ${lineChart({ title:"포트폴리오와 벤치마크 가치", rows:commonRows, series:[{key:"portfolio_value",label:"포트폴리오"},{key:"benchmark_value",label:"벤치마크(공통 경제일)"}], unit:"통화 가치" })}
    ${lineChart({ title:"낙폭", rows:drawdownRows(curveRows), series:[{key:"drawdown",label:"표시 범위 고점 대비 낙폭"}], unit:"비율", termKey:"maximum_drawdown" })}
    ${lineChart({ title:"롤링 수익률", rows:rollingRows, series:[{key:"rolling_return",label:"누적 수익률"},{key:"rolling_annualized_return",label:"연환산 수익률"}], unit:"비율", dateKey:"economic_date", termKey:"rolling_returns" })}
    ${yearlyChart(yearlyRows)}
    ${lineChart({ title:"총·순 익스포저", rows:curveRows, series:[{key:"gross_exposure",label:"총 익스포저"},{key:"net_exposure",label:"순 익스포저"}], unit:"비율", termKey:"gross_exposure" })}
    ${lineChart({ title:"현금 비중", rows:curveRows, series:[{key:"cash_weight",label:"현금 비중"}], unit:"비율", termKey:"cash_weight" })}
    ${lineChart({ title:"보유 종목 수", rows:curveRows, series:[{key:"position_count",label:"보유 종목 수"}], unit:"개", termKey:"position_count" })}
    ${lineChart({ title:"일별 회전율", rows:curveRows, series:[{key:"daily_turnover",label:"일별 회전율"}], unit:"비율", termKey:"turnover" })}
    ${lineChart({ title:"일별 거래비용", rows:curveRows, series:[{key:"transaction_cost",label:"거래비용"}], unit:"통화 가치", termKey:"transaction_cost_drag" })}
  </div>`;
}

async function renderOverview() {
  const data = await api("/overview");
  const artifacts = data.artifact_availability_counts || {};
  const attempts = data.execution_attempt_status_counts || {};
  const versions = data.versions || {};
  view.innerHTML = `
    <p class="lede">저장된 StrategyRun, 평가, 아티팩트와 운영 실행 시도의 현재 인덱스를 한눈에 봅니다.</p>
    <div class="notice">${escapeHtml(data.evidence_quality_note_ko)}</div>
    <section class="card-grid" aria-label="저장 결과 요약">
      ${[["저장된 StrategyRun",data.strategy_run_count],["EvaluationProfile",data.evaluation_profile_count],["EvaluationRun",data.evaluation_run_count],["레지스트리 이슈",data.registry_issue_count]].map(([label,value]) => `<article class="card metric-card"><small>${escapeHtml(label)}</small><strong>${fmt(value,0)}</strong></article>`).join("")}
    </section>
    <div class="two-column">
      <section class="card"><h2>아티팩트 상태</h2><div class="status-list">${Object.entries(artifacts).map(([key,value]) => `${statusBadge(key, availabilityLabel(key))}<span>${fmt(value,0)}개</span>`).join("")}</div></section>
      <section class="card"><h2>실행 시도 운영 상태</h2><div class="status-list">${Object.entries(attempts).map(([key,value]) => `${statusBadge(key, availabilityLabel(key))}<span>${fmt(value,0)}개</span>`).join("")}</div></section>
    </div>
    <section class="card section"><h2>버전과 재구축 식별자</h2>${keyValues([
      ["API",versions.api_version,true],["레지스트리",versions.registry_schema_version,true],["지표 레지스트리",versions.metric_registry_version,true],
      ["계산 엔진",(versions.calculation_engine_versions||[]).join(", "),true],["행동 엔진",(versions.behavior_engine_versions||[]).join(", "),true],
      ["최근 레지스트리 ID",data.last_registry_rebuild_identity?.registry_id,true],["소스 지문",data.last_registry_rebuild_identity?.source_fingerprint,true]
    ])}</section>`;
}

function runFilterForm() {
  return `<form id="run-filters" class="toolbar" novalidate>
    <div class="field"><label for="run-status">종료 상태</label><select id="run-status" name="status"><option value="">전체</option><option value="succeeded">성공</option><option value="partial">부분 완료</option><option value="failed">실패</option></select></div>
    <div class="field"><label for="artifact-state">아티팩트 상태</label><select id="artifact-state" name="artifact_availability"><option value="">전체</option>${["available","missing","pruned","corrupt","never_generated","unsupported_schema"].map((v)=>`<option value="${v}">${availabilityLabel(v)}</option>`).join("")}</select></div>
    <div class="field"><label for="integrity-state">무결성 상태</label><select id="integrity-state" name="integrity_status"><option value="">전체</option>${["valid","missing","corrupt","pruned","unsupported_schema","integrity_failed","not_checked"].map((v)=>`<option value="${v}">${availabilityLabel(v)}</option>`).join("")}</select></div>
    <div class="field"><label for="run-start">시작일(겹치는 범위)</label><input id="run-start" name="start_date" type="date"></div>
    <div class="field"><label for="run-end">종료일(겹치는 범위)</label><input id="run-end" name="end_date" type="date"></div>
    <div class="field"><label for="run-sort">정렬</label><select id="run-sort" name="sort"><option value="-creation_time">생성일 최신순</option><option value="creation_time">생성일 오래된순</option><option value="strategy_run_id">ID 오름차순</option><option value="-engine_version">엔진 버전 내림차순</option></select></div>
    <button class="button" type="submit">조회</button><button class="button secondary" type="reset">초기화</button>
  </form>`;
}

async function renderRuns() {
  view.innerHTML = `<p class="lede">허용된 조건과 정렬만 사용해 저장된 불변 경제 실행 결과를 탐색합니다.</p>${runFilterForm()}<div id="run-results"></div>`;
  const form = document.getElementById("run-filters");
  form.addEventListener("submit", async (event) => { event.preventDefault(); state.runCursorHistory=[]; await loadRuns(); });
  form.addEventListener("reset", () => { setTimeout(() => { state.runCursorHistory=[]; loadRuns(); }, 0); });
  await loadRuns();
}

async function loadRuns(cursor = null) {
  const form = document.getElementById("run-filters");
  const params = new URLSearchParams({ page_size: String(LIST_PAGE_SIZE) });
  for (const [key, value] of new FormData(form).entries()) if (value) params.set(key, value);
  if (cursor) params.set("cursor", cursor);
  const target = document.getElementById("run-results");
  try {
    setMessage(""); setLoading("저장된 StrategyRun을 조회하는 중입니다…");
    const data = await api(`/runs?${params.toString()}`);
    state.runCursor = data.page.next_cursor;
    if (!data.items.length) { target.innerHTML = emptyState("조건에 맞는 실행 없음", "필터를 줄이거나 날짜 범위를 확인해 주세요."); return; }
    target.innerHTML = `<div class="table-wrap"><table><thead><tr><th scope="col">선택</th><th scope="col">StrategyRun</th><th scope="col">종료 상태</th><th scope="col">경제 기간</th><th scope="col">엔진·커밋</th><th scope="col">아티팩트</th><th scope="col">무결성·보존</th></tr></thead><tbody>
      ${data.items.map((run) => `<tr><td><button class="button secondary open-run" data-run-id="${escapeHtml(run.strategy_run_id)}">열기</button></td><td class="id" title="${escapeHtml(run.strategy_run_id)}">${escapeHtml(shortId(run.strategy_run_id))}</td><td>${statusBadge(run.terminal_status, run.terminal_status_ko || availabilityLabel(run.terminal_status))}</td><td>${escapeHtml(run.economic_date_range.start)}<br>${escapeHtml(run.economic_date_range.end)}</td><td><span class="id">${escapeHtml(run.engine_version)}</span><br><span class="id">${escapeHtml(shortId(run.source_commit,12))}</span></td><td>${escapeHtml((run.available_artifact_keys||[]).join(", ") || "사용 가능 항목 없음")}</td><td>${statusBadge(run.integrity_status,run.integrity_status_ko||availabilityLabel(run.integrity_status))}<br>${statusBadge(run.retention_status,availabilityLabel(run.retention_status))}</td></tr>`).join("")}
      </tbody></table></div><div class="pagination"><span>${data.page.total}개 중 ${data.page.returned}개 표시</span><button id="runs-prev" class="button secondary" ${state.runCursorHistory.length ? "" : "disabled"}>이전</button><button id="runs-next" class="button secondary" ${data.page.next_cursor ? "" : "disabled"}>다음</button></div>`;
    document.querySelectorAll(".open-run").forEach((button)=>button.addEventListener("click",()=>{location.hash=`#runs/${encodeURIComponent(button.dataset.runId)}`;}));
    document.getElementById("runs-next")?.addEventListener("click",()=>{state.runCursorHistory.push(cursor); loadRuns(data.page.next_cursor);});
    document.getElementById("runs-prev")?.addEventListener("click",()=>{const previous=state.runCursorHistory.pop(); loadRuns(previous);});
  } catch (error) { setMessage(error.message); target.innerHTML = emptyState("조회 조건 오류", "표시된 한국어 오류를 확인해 주세요."); }
  finally { setLoading(""); }
}

function artifactTable(artifacts) {
  return `<div class="table-wrap"><table><thead><tr><th scope="col">아티팩트</th><th scope="col">가용성</th><th scope="col">무결성</th><th scope="col">보존</th><th scope="col">행·크기</th><th scope="col">해시</th></tr></thead><tbody>${artifacts.map((item)=>`<tr><th scope="row">${escapeHtml(item.artifact_key)}</th><td>${statusBadge(item.availability,availabilityLabel(item.availability))}</td><td>${statusBadge(item.integrity_status,availabilityLabel(item.integrity_status))}</td><td>${statusBadge(item.retention_state,availabilityLabel(item.retention_state))}</td><td>${fmt(item.row_count,0)}행<br>${fmt(item.stored_bytes,0)} bytes</td><td class="id">${escapeHtml(item.content_hash || "—")}</td></tr>`).join("")}</tbody></table></div>`;
}

function optionalState(payload, label) {
  if (!payload?.__error) return "";
  return emptyState(`${label}: ${payload.__error.message_ko || "사용 불가"}`, `상태 코드: ${payload.__error.code || "unknown"}`);
}

async function renderRunDetail(runId) {
  await ensureTerminology();
  const encoded = encodeURIComponent(runId);
  const [detail, manifest, specification, provenance, status, artifacts, curve, benchmark, yearly, rolling, derived, robustness, behavior, evaluations, attempts] = await Promise.all([
    api(`/runs/${encoded}`), api(`/runs/${encoded}/manifest`), api(`/runs/${encoded}/specification`), api(`/runs/${encoded}/provenance`), api(`/runs/${encoded}/status`), api(`/runs/${encoded}/artifacts`),
    api(`/runs/${encoded}/curve?page_size=${CURVE_PAGE_SIZE}`,{optional:true}), api(`/runs/${encoded}/benchmark-curve?page_size=${CURVE_PAGE_SIZE}`,{optional:true}), api(`/runs/${encoded}/yearly-metrics?page_size=100`,{optional:true}), api(`/runs/${encoded}/rolling-metrics?window_sessions=63&page_size=${CURVE_PAGE_SIZE}`,{optional:true}), api(`/runs/${encoded}/derived-metrics`,{optional:true}), api(`/runs/${encoded}/robustness-summary`,{optional:true}), api(`/runs/${encoded}/behavior-summary`,{optional:true}),
    api(`/evaluation-runs?strategy_run_id=${encoded}&page_size=50`), api(`/execution-attempts?intended_strategy_run_id=${encoded}&page_size=50`)
  ]);
  const charts = curve.__error ? optionalState(curve,"일별 포트폴리오 곡선") : renderCharts(curve, benchmark.__error?{}:benchmark, yearly.__error?{}:yearly, rolling.__error?{}:rolling, derived.__error?{}:derived);
  view.innerHTML = `<p><a href="#runs">← 저장된 전략 실행으로 돌아가기</a></p>
    <p class="lede">불변 StrategyRun 결과와 별도의 평가·운영 실행 시도를 함께 확인합니다.</p>
    <section class="card-grid"><article class="card metric-card"><small>종료 상태</small><strong>${statusBadge(status.terminal_status,status.terminal_status_ko)}</strong></article><article class="card metric-card"><small>무결성</small><strong>${statusBadge(status.integrity_status,status.integrity_status_ko)}</strong></article><article class="card metric-card"><small>보존 상태</small><strong>${statusBadge(status.retention_status,availabilityLabel(status.retention_status))}</strong></article><article class="card metric-card"><small>연결된 평가 / 실행 시도</small><strong>${evaluations.page.total} / ${attempts.page.total}</strong></article></section>
    <section class="card section"><h2>실행 식별과 출처</h2>${keyValues([["strategy_run_id",runId,true],["소스 커밋",detail.source_commit,true],["엔진 버전",detail.engine_version,true],["데이터 스냅샷",detail.source_data_snapshot_id,true],["경제 기간",`${detail.economic_date_range.start} ~ ${detail.economic_date_range.end}`],["벤치마크",JSON.stringify(detail.canonical_specification?.benchmark||{})],["명세 해시",detail.specification_hash,true],["매니페스트 해시",detail.manifest_hash,true]])}</section>
    <section class="section"><div class="section-header"><div><h2>성과·위험 차트</h2><p>초기 요청은 항상 제한된 행 수만 읽습니다.</p></div></div>${charts}</section>
    <section class="section"><h2>아티팩트 가용성·보존·무결성</h2>${artifactTable(artifacts.items)}</section>
    <div class="two-column section"><section class="card"><h2>파생 지표</h2>${derived.__error?optionalState(derived,"파생 지표"):`<pre>${jsonText(derived.payload)}</pre>`}</section><section class="card"><h2>강건성 근거</h2>${robustness.__error?optionalState(robustness,"강건성 근거"):`<pre>${jsonText(robustness.payload)}</pre>`}</section></div>
    <div class="two-column section"><section class="card"><h2>행동 메타데이터</h2>${behavior.__error?optionalState(behavior,"행동 메타데이터"):`<pre>${jsonText(behavior.payload)}</pre>`}</section><section class="card"><h2>계산 출처</h2><pre>${jsonText(provenance.calculation_provenance)}</pre></section></div>
    <section class="card section"><h2>연결된 EvaluationRun</h2>${evaluations.items.length?evaluations.items.map((item)=>`<p><a href="#evaluations">${escapeHtml(item.evaluation_run_id)}</a> · profile hash <span class="id">${escapeHtml(item.profile_hash)}</span></p>`).join(""):emptyState("평가 없음","이 StrategyRun에 연결된 저장 평가가 없습니다.")}</section>
    <section class="card section"><h2>연결된 ExecutionAttempt</h2><p class="notice">아래 운영 상태는 StrategyRun의 불변 종료 결과와 별개입니다.</p>${attempts.items.length?attempts.items.map((item)=>`<p><span class="id">${escapeHtml(item.execution_attempt_id)}</span> ${statusBadge(item.operational_status,item.operational_status_ko)}</p>`).join(""):emptyState("실행 시도 없음","연결된 운영 실행 기록이 없습니다.")}</section>
    <section class="card section"><h2>원본 계약</h2><details><summary>매니페스트</summary><pre>${jsonText(manifest)}</pre></details><details><summary>전략 명세</summary><pre>${jsonText(specification)}</pre></details><details><summary>아티팩트 해시와 출처</summary><pre>${jsonText(provenance)}</pre></details></section>`;
  bindExplanationLinks();
}

async function loadRunChoices() {
  const data = await api("/runs?page_size=200&sort=-creation_time");
  return data.items;
}

function runOptions(runs, selected) {
  return runs.map((run)=>`<option value="${escapeHtml(run.strategy_run_id)}" ${run.strategy_run_id===selected?"selected":""}>${escapeHtml(shortId(run.strategy_run_id,28))} · ${escapeHtml(run.economic_date_range.start)}~${escapeHtml(run.economic_date_range.end)}</option>`).join("");
}

async function renderPerformance() {
  await ensureTerminology();
  const runs = await loadRunChoices();
  if (!runs.length) { view.innerHTML=emptyState("저장 실행 없음","차트로 표시할 StrategyRun이 없습니다."); return; }
  const runId = state.selectedPerformanceRun && runs.some((r)=>r.strategy_run_id===state.selectedPerformanceRun) ? state.selectedPerformanceRun : runs[0].strategy_run_id;
  state.selectedPerformanceRun = runId;
  const encoded=encodeURIComponent(runId);
  const [curve,benchmark,yearly,rolling,derived] = await Promise.all([api(`/runs/${encoded}/curve?page_size=${CURVE_PAGE_SIZE}`,{optional:true}),api(`/runs/${encoded}/benchmark-curve?page_size=${CURVE_PAGE_SIZE}`,{optional:true}),api(`/runs/${encoded}/yearly-metrics?page_size=100`,{optional:true}),api(`/runs/${encoded}/rolling-metrics?window_sessions=63&page_size=${CURVE_PAGE_SIZE}`,{optional:true}),api(`/runs/${encoded}/derived-metrics`,{optional:true})]);
  view.innerHTML=`<p class="lede">저장된 시계열의 첫 번째 제한 페이지를 사용해 성과와 위험을 함께 봅니다.</p><div class="toolbar"><div class="field"><label for="performance-run">StrategyRun 선택</label><select id="performance-run">${runOptions(runs,runId)}</select></div><a class="button secondary" href="#runs/${encodeURIComponent(runId)}">상세 열기</a></div>${curve.__error?optionalState(curve,"일별 포트폴리오 곡선"):renderCharts(curve,benchmark.__error?{}:benchmark,yearly.__error?{}:yearly,rolling.__error?{}:rolling,derived.__error?{}:derived)}`;
  document.getElementById("performance-run").addEventListener("change",(event)=>{state.selectedPerformanceRun=event.target.value; renderPerformance().catch(showFatal);}); bindExplanationLinks();
}

function gateRows(results, profile) {
  if (!results?.length) return `<tr><td colspan="7">설정된 필수 관문 없음</td></tr>`;
  return results.map((gate)=>`<tr><th scope="row">${explanationButton(gate.metric_key,gate.metric_key)}</th><td>${escapeHtml(profile.metric_directions?.[gate.metric_key]||"—")}</td><td>${escapeHtml(gate.operator)}</td><td>${fmt(gate.threshold)}</td><td>${fmt(gate.value)}</td><td>${statusBadge(gate.passed?"passed":"failed",gate.passed?"통과":"실패")}</td><td>${escapeHtml(gate.reason)}</td></tr>`).join("");
}

function robustnessGateRows(results, profile) {
  if (!results?.length) return `<tr><td colspan="7">해당 없음</td></tr>`;
  return results.map((gate) => {
    let code = "failed", label = "실패";
    if (gate.passed) { code = "passed"; label = "통과"; }
    else if (gate.reason === "not_applicable") { code = "neutral"; label = "해당 없음"; }
    else if (gate.value === null || /missing|unavailable|not_supplied|evidence/i.test(gate.reason || "")) {
      code = "warn"; label = "근거 누락";
    }
    return `<tr><th scope="row">${explanationButton(gate.metric_key,gate.metric_key)}</th><td>${escapeHtml(profile.metric_directions?.[gate.metric_key]||"—")}</td><td>${escapeHtml(gate.operator)}</td><td>${fmt(gate.threshold)}</td><td>${fmt(gate.value)}</td><td>${statusBadge(code,label)}</td><td>${escapeHtml(gate.reason)}</td></tr>`;
  }).join("");
}

function pipelineFor(output, profile, evaluation) {
  const robustnessConfigured = (profile.robustness_vetoes||[]).length > 0;
  const robustnessRows = output.robustness_vetoes?.length ? robustnessGateRows(output.robustness_vetoes,profile) : `<tr><td colspan="7">${robustnessConfigured?"해당 없음":"사용 설정되지 않음"}</td></tr>`;
  const weighted = output.exploratory_weighted;
  return `<article class="card section"><h2>${escapeHtml(shortId(evaluation.evaluation_run_id,30))}</h2>${keyValues([["EvaluationProfile",evaluation.evaluation_profile_id,true],["profile hash",evaluation.profile_hash,true],["비교 모드",evaluation.comparison_mode],["StrategyRun",output.strategy_run_id,true]])}
    <div class="pipeline">
      <section class="card pipeline-stage"><div class="pipeline-index">1 · 필수 관문</div><h3>mandatory gates</h3><div class="table-wrap"><table><thead><tr><th scope="col">지표</th><th scope="col">방향</th><th scope="col">조건</th><th scope="col">기준값</th><th scope="col">관측값</th><th scope="col">판정</th><th scope="col">이유</th></tr></thead><tbody>${gateRows(output.mandatory_gates,profile)}</tbody></table></div></section>
      <section class="card pipeline-stage"><div class="pipeline-index">2 · 엡실론 파레토</div><h3>epsilon-Pareto</h3><p>${statusBadge(output.pareto.member?"passed":"failed",output.pareto.member?"파레토 구성원":"지배됨")}</p><p>지배한 실행: ${escapeHtml((output.pareto.dominated_by||[]).join(", ")||"없음")}</p><pre>${jsonText(profile.pareto_objectives||[])}</pre></section>
      <section class="card pipeline-stage"><div class="pipeline-index">3 · 강건성 거부권</div><h3>robustness veto</h3><div class="table-wrap"><table><thead><tr><th scope="col">지표</th><th scope="col">방향</th><th scope="col">조건</th><th scope="col">기준값</th><th scope="col">관측값</th><th scope="col">판정</th><th scope="col">이유</th></tr></thead><tbody>${robustnessRows}</tbody></table></div></section>
      <section class="card pipeline-stage"><div class="pipeline-index">4 · 사전식 동률 해소</div><h3>lexicographic tie-break</h3><p>선택 가능 후보 내 순서: ${fmt(output.tie_break_order,0)}</p><pre>${jsonText(profile.lexicographic_tie_break||[])}</pre></section>
      <section class="card pipeline-stage"><div class="pipeline-index">5 · 행동 중복 제거</div><h3>behavior deduplication</h3><pre>${jsonText(output.behavior)}</pre></section>
      <section class="card pipeline-stage weighted"><div class="pipeline-index">별도 탐색 보기</div><h3>${explanationButton("exploratory_weighted_value","exploratory_weighted_value")}</h3>${weighted?`<p>값: <strong>${fmt(weighted.exploratory_weighted_value)}</strong> · 탐색 순위: ${fmt(weighted.rank,0)}</p><p class="notice">필수 관문 실패나 파레토·강건성 판정을 덮어쓰지 않습니다.</p><pre>${jsonText(weighted)}</pre>`:`<p>이 프로필에는 탐색 가중 출력이 없습니다.</p>`}</section>
    </div></article>`;
}

async function renderEvaluations() {
  await ensureTerminology();
  const list = await api("/evaluation-runs?page_size=200&sort=-creation_time");
  if (!list.items.length) { view.innerHTML=emptyState("저장 평가 없음","비교할 EvaluationRun이 없습니다."); return; }
  const runIds=[...new Set(list.items.flatMap((item)=>item.strategy_run_ids))].sort();
  const selected=state.selectedEvaluationRun&&runIds.includes(state.selectedEvaluationRun)?state.selectedEvaluationRun:runIds[0]; state.selectedEvaluationRun=selected;
  const applicable=list.items.filter((item)=>item.strategy_run_ids.includes(selected));
  const bundles=await Promise.all(applicable.map(async(item)=>({evaluation:item,profile:await api(`/evaluation-profiles/${encodeURIComponent(item.evaluation_profile_id)}`),outputs:await api(`/evaluation-runs/${encodeURIComponent(item.evaluation_run_id)}/outputs?page_size=200`)})));
  view.innerHTML=`<p class="lede">하나의 저장된 경제 경로에 적용된 여러 EvaluationProfile을 비교합니다. 프로필만 바꾸면 경제 백테스트는 다시 실행되지 않습니다.</p><div class="notice">필수 관문 → 엡실론 파레토 → 강건성 거부권 → 사전식 동률 해소 → 행동 중복 제거가 기본 흐름이며, 탐색 가중 출력은 별도입니다.</div><div class="toolbar"><div class="field"><label for="evaluation-run-choice">StrategyRun 선택</label><select id="evaluation-run-choice">${runIds.map((id)=>`<option value="${escapeHtml(id)}" ${id===selected?"selected":""}>${escapeHtml(shortId(id,34))}</option>`).join("")}</select></div></div>
    <section class="section"><h2>프로필 비교</h2><div class="table-wrap"><table><thead><tr><th scope="col">프로필 ID / 해시</th><th scope="col">사용 지표</th><th scope="col">필수 관문</th><th scope="col">파레토·ε</th><th scope="col">강건성</th><th scope="col">정규화·가중치</th><th scope="col">평가 결과</th></tr></thead><tbody>${bundles.map(({evaluation,profile,outputs})=>{const out=outputs.items.find((x)=>x.strategy_run_id===selected);return `<tr><td class="id">${escapeHtml(profile.evaluation_profile_id)}<br>${escapeHtml(evaluation.profile_hash)}</td><td>${escapeHtml((profile.enabled_metrics||[]).join(", "))}</td><td>${(profile.mandatory_gates||[]).length}개</td><td>${(profile.pareto_objectives||[]).map((x)=>`${escapeHtml(x.metric_key)} ε=${fmt(x.epsilon)}`).join("<br>")}</td><td>${(profile.robustness_vetoes||[]).length?`${profile.robustness_vetoes.length}개 설정` : "사용 설정되지 않음"}</td><td>${escapeHtml(profile.normalization_method||"없음")}<br>${escapeHtml(JSON.stringify(profile.exploratory_metric_weights||{}))}</td><td>${(out?.decision_labels||[]).map((x)=>statusBadge(x.includes("selected")?"selected":"neutral",x)).join(" ")}</td></tr>`;}).join("")}</tbody></table></div></section>
    ${bundles.map(({evaluation,profile,outputs})=>{const out=outputs.items.find((x)=>x.strategy_run_id===selected);return out?pipelineFor(out,profile,evaluation):emptyState("평가 결과 없음","선택한 실행 결과가 이 평가에 없습니다.");}).join("")}`;
  document.getElementById("evaluation-run-choice").addEventListener("change",(event)=>{state.selectedEvaluationRun=event.target.value;renderEvaluations().catch(showFatal);}); bindExplanationLinks();
}

function robustnessFields(payload) {
  const fields = [
    ["워크포워드 폴드 수","walk_forward_fold_count","walk_forward"],["워크포워드 통과 비율","walk_forward_pass_ratio","walk_forward"],["워크포워드 최악 폴드","walk_forward_worst_fold","walk_forward"],
    ["LOYO 사례 수","loyo_case_count","loyo"],["LOYO 안정성 비율","loyo_stability_ratio","loyo"],["LOYO 반전 연도","loyo_reversing_years","loyo"],
    ["블록 부트스트랩 효과","block_bootstrap_effect","paired_block_bootstrap"],["신뢰구간","bootstrap_confidence_interval","confidence_interval"],["원시 p값","raw_p_value","raw_p_value"],["보정 p값","adjusted_p_value","adjusted_p_value"],["다중검정 보정","multiple_testing_method","multiple_testing_correction"],
    ["비용 스트레스 생존","transaction_cost_stress_survival","cost_stress"],["지배 자산군","dominant_asset_group","asset_group_concentration"],["지배 자산군 비중","dominant_group_share","asset_group_concentration"],["Deflated Sharpe Ratio","deflated_sharpe_ratio","deflated_sharpe_ratio"],["PBO","pbo","pbo"]
  ];
  const present=fields.filter(([,key])=>payload[key]!==undefined&&payload[key]!==null);
  if (!present.length) return emptyState("저장된 강건성 값 없음","없는 계산을 화면에서 만들지 않습니다.");
  return `<div class="table-wrap"><table><thead><tr><th scope="col">근거</th><th scope="col">저장 값</th></tr></thead><tbody>${present.map(([label,key,term])=>`<tr><th scope="row">${explanationButton(term,label)}</th><td>${escapeHtml(typeof payload[key]==="object"?JSON.stringify(payload[key]):fmt(payload[key]))}</td></tr>`).join("")}</tbody></table></div>`;
}

async function renderRobustness() {
  await ensureTerminology(); const runs=await loadRunChoices();
  if(!runs.length){view.innerHTML=emptyState("저장 실행 없음","강건성 근거를 조회할 StrategyRun이 없습니다.");return;}
  const runId=state.selectedRobustnessRun&&runs.some((r)=>r.strategy_run_id===state.selectedRobustnessRun)?state.selectedRobustnessRun:runs[0].strategy_run_id;state.selectedRobustnessRun=runId;
  const evidence=await api(`/runs/${encodeURIComponent(runId)}/robustness-summary`,{optional:true});
  const evals=await api(`/evaluation-runs?strategy_run_id=${encodeURIComponent(runId)}&page_size=50`);
  const vetoBundles=await Promise.all(evals.items.map(async(e)=>({e,profile:await api(`/evaluation-profiles/${encodeURIComponent(e.evaluation_profile_id)}`),outputs:await api(`/evaluation-runs/${encodeURIComponent(e.evaluation_run_id)}/outputs?page_size=200`)})));
  view.innerHTML=`<p class="lede">저장소에 실제로 존재하는 검증 근거만 표시합니다. 결측 계산을 추정하거나 대체하지 않습니다.</p><div class="toolbar"><div class="field"><label for="robustness-run">StrategyRun 선택</label><select id="robustness-run">${runOptions(runs,runId)}</select></div></div><section class="card section"><h2>저장 강건성 근거</h2>${evidence.__error?`<div class="notice"><strong>근거 누락 또는 사용 불가</strong><br>${escapeHtml(evidence.__error.message_ko)}<br>정확한 코드: ${escapeHtml(evidence.__error.code)}</div>`:robustnessFields(evidence.payload)}</section><section class="card section"><h2>프로필별 강건성 거부권</h2>${vetoBundles.length?vetoBundles.map(({e,profile,outputs})=>{const out=outputs.items.find((x)=>x.strategy_run_id===runId);if(!(profile.robustness_vetoes||[]).length)return `<p><span class="id">${escapeHtml(e.profile_hash)}</span> ${statusBadge("neutral","사용 설정되지 않음")}</p>`;return `<div><p><span class="id">${escapeHtml(e.profile_hash)}</span></p><div class="table-wrap"><table><thead><tr><th>지표</th><th>방향</th><th>조건</th><th>기준값</th><th>관측값</th><th>판정</th><th>이유</th></tr></thead><tbody>${robustnessGateRows(out?.robustness_vetoes||[],profile)}</tbody></table></div></div>`;}).join(""):emptyState("연결된 평가 없음","저장된 강건성 판정이 없습니다.")}</section>`;
  document.getElementById("robustness-run").addEventListener("change",(event)=>{state.selectedRobustnessRun=event.target.value;renderRobustness().catch(showFatal);});bindExplanationLinks();
}

async function renderBehavior() {
  await ensureTerminology(); const evaluations=await api("/evaluation-runs?page_size=200&sort=-creation_time");
  if(!evaluations.items.length){view.innerHTML=emptyState("행동 비교 없음","행동 진단이 연결된 EvaluationRun이 없습니다.");return;}
  const selected=state.selectedBehaviorEvaluation&&evaluations.items.some((e)=>e.evaluation_run_id===state.selectedBehaviorEvaluation)?state.selectedBehaviorEvaluation:evaluations.items[0].evaluation_run_id;state.selectedBehaviorEvaluation=selected;
  const data=await api(`/evaluation-runs/${encodeURIComponent(selected)}/behavior?page_size=50`);
  view.innerHTML=`<p class="lede">수익률 상관, 활성·진입·청산일 Jaccard, 경로 거리는 서로 다른 진단입니다. 하나의 숨은 결합값으로 합치지 않습니다.</p><div class="toolbar"><div class="field"><label for="behavior-evaluation">EvaluationRun 선택</label><select id="behavior-evaluation">${evaluations.items.map((e)=>`<option value="${escapeHtml(e.evaluation_run_id)}" ${e.evaluation_run_id===selected?"selected":""}>${escapeHtml(shortId(e.evaluation_run_id,34))}</option>`).join("")}</select></div></div>
  <section class="section"><h2>쌍별 진단 (${data.page.returned}/${data.page.total})</h2>${data.pairwise_diagnostics.length?`<div class="table-wrap"><table><thead><tr><th scope="col">비교 쌍</th><th scope="col">수익률 상관</th><th scope="col">활성일 Jaccard</th><th scope="col">진입일 Jaccard</th><th scope="col">청산일 Jaccard</th><th scope="col">정규화 경로 거리</th></tr></thead><tbody>${data.pairwise_diagnostics.map((pair)=>{const d=pair.diagnostics||{};return `<tr><th scope="row" class="id">${escapeHtml(pair.pair_id)}</th><td>${fmt(d.daily_return_correlation)}</td><td>${fmt(d.active_date_jaccard)}</td><td>${fmt(d.entry_date_jaccard)}</td><td>${fmt(d.exit_date_jaccard)}</td><td>${fmt(d.normalized_path_distance)}</td></tr>`;}).join("")}</tbody></table></div>`:emptyState("비교 쌍 없음","이 평가에 저장된 쌍별 진단이 없습니다.")}</section>
  <div class="two-column"><section class="card"><h2>행동 군집·대표 StrategyRun</h2><pre>${jsonText(data.candidate_clusters)}</pre></section><section class="card"><h2>단순성 메타데이터</h2><pre>${jsonText(data.simplicity_metadata)}</pre></section></div>`;
  document.getElementById("behavior-evaluation").addEventListener("change",(event)=>{state.selectedBehaviorEvaluation=event.target.value;renderBehavior().catch(showFatal);});
}

function componentField(id, label, value, help) {
  return `<div class="field"><label for="${id}">${escapeHtml(label)} <abbr title="${escapeHtml(help)}">설명</abbr></label><input id="${id}" value="${escapeHtml(value)}" readonly></div>`;
}

function buildConstruction(profileIds) {
  return {
    schema_version: "strategy_construction_request_v1",
    data_snapshot: "phase_a2_frozen_2026_07_30",
    backtest_start_date: document.getElementById("construction-start").value,
    backtest_end_date: document.getElementById("construction-end").value,
    universe: { option_id: "phase_a2_historical_eligible_v1", parameters: {} },
    benchmark: { option_id: "spy_adjusted_close_v1", parameters: {} },
    trend_filter: { option_id: "price_above_rising_ma200_v0", parameters: {} },
    signal: { option_id: "prior_price_high_l20_v1", parameters: { lookback: { kind: "fixed", value: 20 } } },
    entry_rule: { option_id: "first_event_next_open_v1", parameters: {} },
    initial_stop: { option_id: "signal_day_low20_v1", parameters: {} },
    trailing_exit: { option_id: "ratcheting_low20_v1", parameters: {} },
    position_sizing: { option_id: "canonical_equal_weight_active_v1", parameters: {} },
    portfolio_constraints: { option_id: "long_only_cash_constrained_v1", parameters: {} },
    transaction_cost: { option_id: "round_trip_bps_v1", parameters: { bps: parameterSpace(document.getElementById("construction-cost").value) } },
    slippage: { option_id: "round_trip_slippage_bps_v1", parameters: { bps: parameterSpace(document.getElementById("construction-slippage").value) } },
    walk_forward: { enabled: Number(document.getElementById("construction-folds").value) > 0, fold_count: Number(document.getElementById("construction-folds").value) },
    robustness: { scenario_count: Number(document.getElementById("construction-robustness").value) },
    evaluation_profile_ids: profileIds,
  };
}

function thresholdRows(estimate) {
  return estimate.threshold_results.map((item) => `<tr><th scope="row">${escapeHtml(item.threshold)}</th><td>${fmt(item.observed, 0)}</td><td>${fmt(item.limit, 0)}</td><td>${statusBadge(item.triggered ? "failed" : "passed", item.triggered ? `초과/도달 (${item.severity})` : "미도달")}</td></tr>`).join("");
}

function foundation6CatalogPanel(catalog) {
  if (!catalog) return "";
  const rows = Object.entries(catalog.categories || {}).flatMap(([family, options]) => options.map((option) => `<tr><th scope="row">${escapeHtml(option.name_ko)}</th><td>${escapeHtml(option.name_en)}</td><td>${escapeHtml(family)}</td><td>${escapeHtml(option.engine_adapter_support)}</td><td>${escapeHtml((option.compatibility || []).join(", ") || "-")}</td><td><pre>${jsonText(option.parameters || {})}</pre></td></tr>`)).join("");
  return `<section class="card section"><h2>Foundation 6 허용 전략 라이브러리</h2><p class="notice">카탈로그 ${escapeHtml(catalog.catalog_version)} · ${escapeHtml(catalog.catalog_hash)}. 옵션 정의, 단위, 범위, 호환성 및 엔진 상태는 서버의 단일 버전 카탈로그에서 읽습니다.</p><div class="table-wrap"><table><thead><tr><th>한국어</th><th>English</th><th>범주</th><th>엔진 상태</th><th>호환성</th><th>매개변수·단위</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
}

function candidateEstimatePresentation(data) {
  if (!data || typeof data !== "object" || !data.candidate_estimate || typeof data.candidate_estimate !== "object"
      || !data.normalized_construction || typeof data.normalized_construction !== "object") {
    throw new Error("후보 수 계산 응답에서 정규화된 전략 구성 또는 후보 추정 정보를 확인할 수 없습니다. API 응답 계약을 확인하세요.");
  }
  const source = data.candidate_estimate;
  const foundation6 = source.schema_version === "candidate_space_estimate_v2";
  const required = foundation6
    ? ["raw_cartesian_combinations", "valid_unique_economic_candidates", "evaluation_only_applications", "total_estimated_work"]
    : ["raw_cartesian_candidate_count", "economic_strategy_run_candidate_count", "evaluation_profile_application_count", "estimated_total_execution_units"];
  if (required.some((field) => !Number.isFinite(Number(source[field])))) {
    throw new Error("후보 수 계산 응답의 필수 추정값이 올바르지 않습니다. API 응답 계약을 확인하세요.");
  }
  const normalized = data.normalized_construction;
  const walkForward = normalized.walk_forward;
  const robustness = normalized.robustness;
  const foldCount = walkForward && typeof walkForward === "object" ? Number(walkForward.fold_count) : 0;
  const scenarioCount = robustness && typeof robustness === "object" ? Number(robustness.scenario_count) : 0;
  if (!Number.isFinite(foldCount) || !Number.isFinite(scenarioCount)) {
    throw new Error("후보 수 계산 응답의 검증 작업 값이 올바르지 않습니다. API 응답 계약을 확인하세요.");
  }
  const estimate = foundation6 ? {
    ...source,
    raw_cartesian_candidate_count: source.raw_cartesian_combinations,
    economic_strategy_run_candidate_count: source.valid_unique_economic_candidates,
    evaluation_profile_application_count: source.evaluation_only_applications,
    robustness_scenario_count: source.robustness_workload,
    estimated_total_execution_units: source.total_estimated_work,
    estimated_reuse_count: source.reusable_completed_candidates,
    estimated_new_backtest_count: source.new_candidates_requiring_execution,
    threshold_results: [],
    confirmation_required: false,
    hard_limit_exceeded: false,
  } : source;
  return {
    estimate,
    normalized: {
      ...normalized,
      walk_forward: walkForward || { enabled: false, fold_count: 0 },
      robustness: robustness || { scenario_count: 0 },
    },
  };
}

function estimateSummary(data) {
  const { estimate, normalized } = candidateEstimatePresentation(data);
  const unsupportedExecution = normalized.walk_forward.fold_count > 0
    || normalized.robustness.scenario_count > 0;
  return `<section class="card section"><h2>정규화 및 후보 미리보기</h2>
    <div class="card-grid">
      <article class="card metric-card"><small>원시 Cartesian 후보</small><strong>${fmt(estimate.raw_cartesian_candidate_count, 0)}</strong></article>
      <article class="card metric-card"><small>중복 제거 경제 후보</small><strong>${fmt(estimate.economic_strategy_run_candidate_count, 0)}</strong></article>
      <article class="card metric-card"><small>평가 프로필 적용</small><strong>${fmt(estimate.evaluation_profile_application_count, 0)}</strong></article>
      <article class="card metric-card"><small>강건성 작업</small><strong>${fmt(estimate.robustness_scenario_count, 0)}</strong></article>
      <article class="card metric-card"><small>총 실행 단위</small><strong>${fmt(estimate.estimated_total_execution_units, 0)}</strong></article>
      <article class="card metric-card"><small>재사용 / 신규 백테스트</small><strong>${fmt(estimate.estimated_reuse_count, 0)} / ${fmt(estimate.estimated_new_backtest_count, 0)}</strong></article>
    </div>
    <div class="table-wrap"><table><thead><tr><th>정책 임계값</th><th>관측값</th><th>한도</th><th>판정</th></tr></thead><tbody>${thresholdRows(estimate)}</tbody></table></div>
    ${estimate.hard_limit_exceeded ? `<div class="notice danger">⛔ 하드 한도 위반: 확인으로 우회할 수 없습니다.</div>` : estimate.confirmation_required ? `<div class="notice">⚠ 대규모 요청 확인이 필요합니다. 요청·추정·정책 해시에 묶인 일회성 확인만 허용됩니다.</div>` : `<div class="notice">✓ 명시적 대규모 확인 없이 실행 가능한 범위입니다.</div>`}
    ${unsupportedExecution ? `<div class="notice danger">⛔ 워크포워드 fold와 강건성 시나리오는 작업량 미리보기만 지원합니다. 현재 실행 어댑터로 요청하려면 두 값을 0으로 설정하세요.</div>` : ""}
    <details open><summary>정확한 정규화 요청</summary><pre>${jsonText(normalized)}</pre></details>
    <details><summary>결정적 StrategyRun 후보 순서</summary><pre>${jsonText(data.strategy_run_candidate_ids)}</pre></details>
    <div class="actions">
      ${estimate.confirmation_required && !estimate.hard_limit_exceeded && !unsupportedExecution ? `<button type="button" id="confirm-construction">이 정확한 요청을 명시적으로 확인</button>` : ""}
      ${!estimate.hard_limit_exceeded && !unsupportedExecution ? `<button type="button" id="create-execution-request">불변 실행 요청 만들기</button>` : ""}
    </div></section>`;
}

async function renderConstruction() {
  const [options, profiles] = await Promise.all([api("/construction/options"), api("/evaluation-profiles?page_size=200")]);
  profiles.items.sort((left,right)=>(left.name === "research_default" ? -1 : right.name === "research_default" ? 1 : left.name.localeCompare(right.name)));
  const profileOptions = profiles.items.map((profile, index) => `<option value="${escapeHtml(profile.evaluation_profile_id)}" ${index === 0 ? "selected" : ""}>${escapeHtml(profile.name)} · ${escapeHtml(shortId(profile.evaluation_profile_id, 28))}</option>`).join("");
  view.innerHTML = `<p class="lede">허용 목록에 있는 규칙만 조합합니다. 임의 코드, 동적 Python, 원격 URL, 무제한 범위는 입력할 수 없습니다.</p>
    <form id="construction-form" class="card section">
      <h2>통제된 전략 구성</h2>
      <div class="form-grid">
        <div class="field"><label for="construction-start">백테스트 시작일 <abbr title="동결 스냅샷 안의 경제 시작일">설명</abbr></label><input id="construction-start" type="date" value="2024-01-02" required></div>
        <div class="field"><label for="construction-end">백테스트 종료일 <abbr title="동결 스냅샷 안의 경제 종료일">설명</abbr></label><input id="construction-end" type="date" value="2024-12-31" required></div>
        ${componentField("snapshot", "데이터 스냅샷", "phase_a2_frozen_2026_07_30", "해시 검증된 로컬 동결 데이터")}
        ${componentField("universe", "유니버스", "phase_a2_historical_eligible_v1", "동결된 적격 상태를 그대로 사용")}
        ${componentField("benchmark", "벤치마크", "SPY", "정확히 겹치는 경제 날짜로 비교")}
        ${componentField("trend-filter", "추세 필터", "price_above_rising_ma200_v0", "종가와 상승 중인 MA200 기반 가격 전용 필터")}
        ${componentField("signal-family", "신호", "prior_price_high_l20_v1", "폐기된 내부 합성 지표 없이 직전 20일 가격 고점 돌파")}
        ${componentField("entry-rule", "진입 규칙", "first_event_next_open_v1", "첫 이벤트 다음 유효 시가")}
        ${componentField("initial-stop", "초기 손절", "signal_day_low20_v1", "신호일 Low20")}
        ${componentField("trailing-exit", "추적 청산", "ratcheting_low20_v1", "Low20 손절선을 위로만 이동")}
        ${componentField("position-sizing", "포지션 크기", "canonical_equal_weight_active_v1", "활성 종목 동일 비중")}
        ${componentField("portfolio-constraints", "포트폴리오 제약", "long_only_cash_constrained_v1", "롱 전용, 현금 제약, 차입 없음")}
        <div class="field"><label for="construction-cost">거래비용 bp <abbr title="고정값, 쉼표 목록, 또는 0..10 step 5 형식">설명</abbr></label><input id="construction-cost" value="5"></div>
        <div class="field"><label for="construction-slippage">슬리피지 bp <abbr title="고정값, 유한 목록, 끝점에 정확히 도달하는 범위">설명</abbr></label><input id="construction-slippage" value="2"></div>
        <div class="field"><label for="construction-folds">워크포워드 fold 수 <abbr title="0은 비활성; 현재 어댑터는 작업량 미리보기만 지원하며 0이 아닌 실행 요청은 거부">설명</abbr></label><input id="construction-folds" type="number" min="0" max="20" value="0"></div>
        <div class="field"><label for="construction-robustness">강건성 시나리오 수 <abbr title="현재 어댑터는 작업량 미리보기만 지원하며 0이 아닌 실행 요청은 거부">설명</abbr></label><input id="construction-robustness" type="number" min="0" max="20" value="0"></div>
        <div class="field"><label for="construction-profile">평가 프로필 <abbr title="여러 프로필을 선택할 수 있으며 평가 설정은 경제 StrategyRun 식별자에 포함되지 않음">설명</abbr></label><select id="construction-profile" multiple size="${Math.min(Math.max(profiles.items.length, 2), 5)}" required>${profileOptions}</select></div>
      </div>
      <button type="submit">정규화하고 정확한 후보 수 계산</button>
    </form>
    <section class="card section"><h2>현재 로컬 정책</h2><pre>${jsonText(options.execution_policy)}</pre></section>
    ${foundation6CatalogPanel(options.foundation_6_catalog)}
    <div id="construction-preview">${state.constructionEstimate ? estimateSummary(state.constructionEstimate) : ""}</div>`;
  const profileSelect = document.getElementById("construction-profile");
  const profileLabels = {research_default:"연구용 기본 평가",final_eligibility_default:"최종 적격성 평가",exploratory_weighted_example:"탐색용 가중 평가 예시"};
  profileSelect.multiple = false; profileSelect.size = 1;
  [...profileSelect.options].forEach((option)=>{const profile=profiles.items.find((item)=>item.evaluation_profile_id===option.value); option.textContent=profileLabels[profile?.name]||profile?.name||option.value;});
  if (!profiles.items.length) { profileSelect.disabled=true; const button=document.querySelector("#construction-form button[type=submit]"); button.disabled=true; button.insertAdjacentHTML("beforebegin",'<p class="notice">사용 가능한 평가 프로필이 없습니다. ResultStore 기본 프로필을 초기화하세요.</p>'); }
  document.getElementById("construction-form").addEventListener("submit", async (event) => {
    event.preventDefault(); setMessage("");
    try {
      const profileIds = [...document.getElementById("construction-profile").selectedOptions].map((option) => option.value);
      state.constructionDraft = buildConstruction(profileIds);
      state.confirmationId = null;
      const estimate = await api("/construction/estimate", { method: "POST", body: state.constructionDraft });
      const preview = estimateSummary(estimate);
      state.constructionEstimate = estimate;
      document.getElementById("construction-preview").innerHTML = preview;
      bindConstructionActions();
    } catch (error) {
      state.constructionEstimate = null;
      document.getElementById("construction-preview").innerHTML = `<p class="notice danger">${escapeHtml(error?.message || "후보 수 계산 결과를 표시하지 못했습니다. 다시 시도하세요.")}</p>`;
    }
  });
  if (state.constructionEstimate) bindConstructionActions();
}

function bindConstructionActions() {
  document.getElementById("confirm-construction")?.addEventListener("click", async () => {
    try {
      const confirmation = await api("/construction/confirm", { method: "POST", body: state.constructionDraft, idempotencyKey: idempotencyKey("confirm") });
      state.confirmationId = confirmation.confirmation_id;
      setMessage(`확인 완료: ${confirmation.confirmation_id} · 만료 ${confirmation.expires_timestamp}`);
    } catch (error) {
      showFatal(error);
    }
  });
  document.getElementById("create-execution-request")?.addEventListener("click", async () => {
    try {
      const estimate = state.constructionEstimate.candidate_estimate;
      if (estimate.confirmation_required && !state.confirmationId) throw new Error("먼저 이 정확한 대규모 요청을 확인해 주세요.");
      const request = await api("/execution-requests", { method: "POST", body: { construction: state.constructionDraft, confirmation_id: state.confirmationId }, idempotencyKey: idempotencyKey("request") });
      state.lastExecutionRequestId = request.execution_request_id;
      await api(`/execution-requests/${encodeURIComponent(request.execution_request_id)}/start`, { method: "POST", body: {}, idempotencyKey: idempotencyKey("start") });
      location.hash = "#requests";
    } catch (error) {
      showFatal(error);
    }
  });
}

async function renderRequests() {
  let persistedManager = null;
  try { persistedManager = await api("/execution-manager"); } catch (_) { persistedManager = null; }
  const requestId = state.lastExecutionRequestId;
  view.innerHTML = `<p class="lede">불변 ExecutionRequest와 운영 ExecutionAttempt를 분리해 표시합니다. 대기·실행 중 상태는 성공한 StrategyRun이 아닙니다.</p>
    <div class="toolbar"><div class="field"><label for="request-id-input">실행 요청 ID</label><input id="request-id-input" value="${escapeHtml(requestId || "")}" placeholder="execution_request_..."></div><button type="button" id="load-request">상태 읽기</button></div>
    <div id="request-progress">${requestId ? "상태를 읽는 중입니다." : emptyState("실행 요청 선택", "전략 구성에서 요청을 만들거나 ID를 입력하세요.")}</div>
    ${persistedManager ? `<section class="card section"><h2>지속 실행 관리자</h2><p class="notice">${escapeHtml(persistedManager.source_of_truth)} · 요청 ${fmt(persistedManager.request_count,0)}개. live, stale, interrupted, recovered, reconciled 상태는 append-only 복구 이력과 분리하여 표시됩니다.</p><details><summary>작업자와 복구 이력</summary><pre>${jsonText({workers:persistedManager.workers,recovery_history:persistedManager.recovery_history})}</pre></details></section>` : ""}`;
  const load = async () => {
    const identity = document.getElementById("request-id-input").value.trim();
    if (!identity) return;
    state.lastExecutionRequestId = identity;
    const data = await api(`/execution-requests/${encodeURIComponent(identity)}`);
    const attempts = data.attempts || [];
    document.getElementById("request-progress").innerHTML = `<section class="card section"><h2>실행 요청</h2>${keyValues([["ExecutionRequest", data.execution_request_id, true],["정규화 해시", data.normalized_construction_hash, true],["추정 해시", data.candidate_estimate_hash, true],["스냅샷", data.data_snapshot_identity, true],["정책", `${data.execution_policy_version} · ${data.execution_policy_hash}`, true]])}<pre>${jsonText(data.candidate_estimate)}</pre></section>
      <section class="section"><h2>후보 진행 (${attempts.length})</h2>${attempts.length ? `<div class="table-wrap"><table><thead><tr><th>ExecutionAttempt</th><th>운영 상태</th><th>후보 진행</th><th>현재 단계·시간</th><th>실패 요약</th><th>아티팩트</th><th>동작</th></tr></thead><tbody>${attempts.map((attempt) => `<tr><th class="id">${escapeHtml(attempt.execution_attempt_id)}</th><td>${statusBadge(attempt.operational_status, attempt.operational_status)}</td><td>${escapeHtml(`${attempt.progress_summary?.candidate_ordinal || "?"}/${attempt.progress_summary?.candidate_total || "?"}`)}<pre>${jsonText(attempt.progress_summary)}</pre></td><td>${escapeHtml(attempt.current_stage || "-")}<br>${escapeHtml(attempt.started_timestamp || attempt.created_timestamp)}<br>${escapeHtml(attempt.completed_timestamp || "진행 중")}</td><td>${escapeHtml(attempt.failure_code || "-")}<br>${escapeHtml(attempt.failure_message || "-")}</td><td><pre>${jsonText(attempt.artifact_references || [])}</pre></td><td>${["queued","running"].includes(attempt.operational_status) ? `<button data-cancel="${escapeHtml(attempt.execution_attempt_id)}">취소</button>` : ""}${["failed","cancelled"].includes(attempt.operational_status) ? `<button data-retry="${escapeHtml(attempt.execution_attempt_id)}">새 시도로 재시도</button>` : ""}</td></tr>`).join("")}</tbody></table></div>` : emptyState("시작 전", "아직 생성된 ExecutionAttempt가 없습니다.")}</section>`;
    document.querySelectorAll("[data-cancel]").forEach((button) => button.addEventListener("click", async () => { await api(`/execution-attempts/${encodeURIComponent(button.dataset.cancel)}/cancel`, { method: "POST", body: {}, idempotencyKey: idempotencyKey("cancel") }); await load(); }));
    document.querySelectorAll("[data-retry]").forEach((button) => button.addEventListener("click", async () => { await api(`/execution-attempts/${encodeURIComponent(button.dataset.retry)}/retry`, { method: "POST", body: {}, idempotencyKey: idempotencyKey("retry") }); await load(); }));
  };
  document.getElementById("load-request").addEventListener("click", () => load().catch(showFatal));
  if (requestId) await load();
}

async function renderAttempts() {
  const data=await api("/execution-attempts?page_size=200&sort=-created_timestamp");
  view.innerHTML=`<p class="lede">ExecutionAttempt의 운영 진행과 불변 StrategyRun 종료 결과를 분리해 표시합니다. 시작·취소·재시도·작업자 제어 기능은 없습니다.</p><div class="notice">운영 상태가 실행 중이거나 실패여도, 그것을 StrategyRun의 불변 경제 결과로 바꾸어 표시하지 않습니다.</div>${data.items.length?`<div class="table-wrap section"><table><thead><tr><th scope="col">실행 시도</th><th scope="col">의도한 StrategyRun</th><th scope="col">재시도 관계</th><th scope="col">시간</th><th scope="col">운영 상태 / 종료 결과</th><th scope="col">단계·진행</th><th scope="col">실패</th><th scope="col">출처·작업자</th></tr></thead><tbody>${data.items.map((a)=>`<tr><th scope="row" class="id">${escapeHtml(a.execution_attempt_id)}</th><td class="id">${escapeHtml(a.intended_strategy_run_id)}</td><td>${fmt(a.attempt_number,0)}차<br><span class="id">${escapeHtml(a.retry_parent_attempt_id||"—")}</span></td><td>생성 ${escapeHtml(a.created_timestamp)}<br>시작 ${escapeHtml(a.started_timestamp||"—")}<br>완료 ${escapeHtml(a.completed_timestamp||"—")}</td><td>${statusBadge(a.operational_status,a.operational_status_ko)}<br>${statusBadge(a.terminal_outcome||"neutral",availabilityLabel(a.terminal_outcome||"미정"))}</td><td>${escapeHtml(a.current_stage||"—")}<pre>${jsonText(a.progress_summary||{})}</pre></td><td>${escapeHtml(a.failure_code||"—")}<br>${escapeHtml(a.failure_message||"—")}</td><td><span class="id">${escapeHtml(shortId(a.source_commit,12))}</span><br>${escapeHtml(a.engine_version)}<details><summary>작업자·아티팩트</summary><pre>${jsonText({worker_metadata:a.worker_metadata,artifact_references:a.artifact_references})}</pre></details></td></tr>`).join("")}</tbody></table></div>`:emptyState("실행 이력 없음","저장된 ExecutionAttempt가 없습니다.")}`;
}

function definitionCard(key, entry) {
  const decision=(entry.applicable_decision_modes||[]).join(", ");
  return `<article class="card definition-card" id="term-${escapeHtml(key)}" tabindex="-1"><p class="eyebrow">${escapeHtml(entry.abbreviation||key)}</p><h2>${escapeHtml(entry.korean_term)} <small>(${escapeHtml(entry.english_term)})</small></h2><div class="formula">${escapeHtml(entry.formula_text)}</div><dl><dt>변수 정의</dt><dd>${escapeHtml((entry.variable_definitions||[]).join(" · "))}</dd><dt>숫자 예시</dt><dd>${escapeHtml(entry.worked_numerical_example)}</dd><dt>해석</dt><dd>${escapeHtml(entry.interpretation)}</dd><dt>단위</dt><dd>${escapeHtml(entry.unit)}</dd><dt>연환산</dt><dd>${escapeHtml(entry.annualization_convention)}</dd><dt>가정</dt><dd>${escapeHtml((entry.assumptions||[]).join(" · "))}</dd><dt>한계</dt><dd>${escapeHtml((entry.limitations||[]).join(" · "))}</dd><dt>오해하기 쉬운 상황</dt><dd>${escapeHtml((entry.misleading_cases||[]).join(" · "))}</dd><dt>의사결정 사용</dt><dd>${escapeHtml(decision)}</dd></dl></article>`;
}

async function renderExplanations(termKey=null) {
  await ensureTerminology(); const entries=Object.entries(state.terminology).sort((a,b)=>a[1].korean_term.localeCompare(b[1].korean_term,"ko"));
  view.innerHTML=`<p class="lede">중앙 용어집 하나에서 공식, 변수, 숫자 예시, 해석, 단위, 연환산, 가정, 한계와 의사결정 사용 위치를 제공합니다.</p><div class="toolbar"><div class="field"><label for="term-search">설명 검색</label><input id="term-search" type="search" placeholder="예: CAGR, 최대낙폭, 파레토"></div><span>${entries.length}개 항목</span></div><div id="definition-grid" class="definition-grid">${entries.map(([key,entry])=>definitionCard(key,entry)).join("")}</div>`;
  document.getElementById("term-search").addEventListener("input",(event)=>{const query=event.target.value.trim().toLocaleLowerCase("ko");document.querySelectorAll(".definition-card").forEach((card)=>{card.hidden=query&&!card.textContent.toLocaleLowerCase("ko").includes(query);});});
  if(termKey){const target=document.getElementById(`term-${termKey}`);if(target){target.scrollIntoView({block:"start"});target.focus();}}
}

function workspaceKey(workflowId, operation) {
  const storageKey = `trend-v2-workspace-key:${workflowId}:${operation}`;
  let value = localStorage.getItem(storageKey);
  if (!value) { value = idempotencyKey(`workspace-${operation}`); localStorage.setItem(storageKey, value); }
  return value;
}

function workspaceConstruction(profileId) {
  return {
    schema_version: "strategy_construction_request_v1",
    data_snapshot: "phase_a2_frozen_2026_07_30",
    backtest_start_date: document.getElementById("workspace-start").value,
    backtest_end_date: document.getElementById("workspace-end").value,
    universe: { option_id: "phase_a2_historical_eligible_v1", parameters: {} },
    benchmark: { option_id: "spy_adjusted_close_v1", parameters: {} },
    trend_filter: { option_id: document.getElementById("workspace-trend").value, parameters: {} },
    signal: { option_id: "prior_price_high_v2", parameters: { lookback: { kind: "fixed", value: Number(document.getElementById("workspace-lookback").value) } } },
    entry_rule: { option_id: "first_event_next_open_v1", parameters: {} },
    initial_stop: { option_id: "signal_day_low20_v1", parameters: {} },
    trailing_exit: { option_id: "ratcheting_low20_v1", parameters: {} },
    position_sizing: { option_id: "canonical_equal_weight_active_v1", parameters: {} },
    portfolio_constraints: { option_id: "long_only_cash_constrained_v1", parameters: {} },
    transaction_cost: { option_id: "round_trip_bps_v1", parameters: { bps: parameterSpace(document.getElementById("workspace-cost").value) } },
    slippage: { option_id: "round_trip_slippage_bps_v1", parameters: { bps: parameterSpace(document.getElementById("workspace-slippage").value) } },
    walk_forward: { enabled: false, fold_count: 0 }, robustness: { scenario_count: 0 },
    evaluation_profile_ids: [profileId],
  };
}

function workspaceCard(number, title, status, body) {
  return `<section class="card section"><p class="eyebrow">${number}단계</p><h2>${title} ${statusBadge(status || "not_started", status || "not_started")}</h2>${body}</section>`;
}

async function renderResearchWorkspace() {
  const workflows = await api("/workflows");
  let workflowId = localStorage.getItem("trend-v2-workflow-id");
  if (!workflowId || !workflows.items.some((item) => item.workflow_id === workflowId)) workflowId = workflows.items[0]?.workflow_id || null;
  if (!workflowId) {
    const [profiles, options] = await Promise.all([api("/evaluation-profiles?page_size=200"), api("/construction/options")]);
    const profileOptions = profiles.items.map((item) => `<option value="${escapeHtml(item.evaluation_profile_id)}">${escapeHtml(item.name)}</option>`).join("");
    const trendOptions = (options.foundation_6_catalog?.categories?.trend_filter || []).filter((item) => item.engine_adapter_support === "supported").map((item) => `<option value="${escapeHtml(item.option_id)}" ${item.option_id === "price_above_rising_ma200_v0" ? "selected" : ""}>${escapeHtml(item.name_ko)}</option>`).join("");
    view.innerHTML = `<p class="lede">저장된 연구 절차를 한 화면에서 이어갑니다. 각 결과는 참조로만 연결되며 경제 백테스트 결과를 복사하지 않습니다.</p>${workspaceCard("1", "전략 구성", "needs_action", `<form id="workspace-create"><div class="form-grid"><div class="field"><label>워크플로우 이름</label><input id="workspace-label" maxlength="120" required value="새 연구 워크플로우"></div><div class="field"><label>시작일</label><input id="workspace-start" type="date" value="2024-01-02" required></div><div class="field"><label>종료일</label><input id="workspace-end" type="date" value="2024-12-31" required></div><div class="field"><label>추세 필터</label><select id="workspace-trend">${trendOptions}</select></div><div class="field"><label>고점 lookback</label><input id="workspace-lookback" type="number" min="20" max="55" value="20"></div><div class="field"><label>거래비용 bp</label><input id="workspace-cost" value="5"></div><div class="field"><label>슬리피지 bp</label><input id="workspace-slippage" value="2"></div><div class="field"><label>EvaluationProfile</label><select id="workspace-profile">${profileOptions}</select></div></div><button type="submit" ${profiles.items.length ? "" : "disabled"}>워크플로우 만들기</button><a class="button-link" href="#profiles">프로필 스튜디오</a></form>`)}`;
    document.getElementById("workspace-create")?.addEventListener("submit", async (event) => { event.preventDefault(); try { const key = workspaceKey("new", "create"); const profileId = document.getElementById("workspace-profile").value; const saved = await api("/workflows", {method:"POST", body:{label_ko:document.getElementById("workspace-label").value, construction:workspaceConstruction(profileId)}, idempotencyKey:key}); localStorage.setItem("trend-v2-workflow-id", saved.workflow_id); localStorage.removeItem("trend-v2-workspace-key:new:create"); await renderResearchWorkspace(); } catch (error) { document.getElementById("workspace-create").insertAdjacentHTML("beforeend", `<p class="notice danger">${escapeHtml(error.message)}</p>`); } });
    return;
  }
  localStorage.setItem("trend-v2-workflow-id", workflowId);
  const [data, profiles] = await Promise.all([api(`/workflows/${encodeURIComponent(workflowId)}`), api("/evaluation-profiles?page_size=200")]);
  const refs = data.references || {}; const economic = refs.economic_progress || {}; const robustness = refs.robustness_progress || {};
  const action = (id, label, disabled=false) => `<button id="workspace-${id}" ${disabled ? "disabled" : ""}>${label}</button>`;
  view.innerHTML = `<div class="toolbar"><div><p class="eyebrow">Research Workspace</p><h2>${escapeHtml(data.label_ko)}</h2><p class="id">${escapeHtml(data.workflow_id)}</p></div><select id="workspace-select">${workflows.items.map((item)=>`<option value="${escapeHtml(item.workflow_id)}" ${item.workflow_id===workflowId?"selected":""}>${escapeHtml(item.label_ko)} · ${escapeHtml(item.stage)}</option>`).join("")}</select><button id="workspace-new">새 워크플로우</button></div>${workspaceCard("1", "전략 구성", refs.normalized_construction ? "completed" : "needs_action", `${refs.normalized_construction ? `<p class="id">${escapeHtml(refs.normalized_construction.construction_hash || "정규화됨")}</p>` : action("normalize", "구성 정규화")}`)}${workspaceCard("2", "호환성 및 후보 수", refs.candidate_estimate ? "completed" : (refs.normalized_construction ? "needs_action" : "not_started"), refs.candidate_estimate ? `<pre>${jsonText(refs.candidate_estimate)}</pre>` : action("estimate", "후보 수 계산", !refs.normalized_construction))}${workspaceCard("3", "경제 실행 및 후보 진행", economic.status || (refs.candidate_estimate ? "needs_action" : "not_started"), economic.attempts ? `<p>상태: ${escapeHtml(economic.status)} · 후보 ${economic.attempts.length}개</p><a class="button-link" href="#requests">실행 요청 상세</a><pre>${jsonText(economic.attempts.map((item)=>({execution_attempt_id:item.execution_attempt_id,status:item.operational_status,stage:item.current_stage})))}</pre>` : action("start", "실행 요청 및 시작", !refs.candidate_estimate))}${workspaceCard("4", "강건성", robustness.status || (economic.status === "completed" ? "needs_action" : "not_started"), robustness.attempt ? `<p>상태: ${escapeHtml(robustness.status)}</p><a class="button-link" href="#robustness">강건성 전문 화면</a>` : `<p class="notice">완료된 StrategyRun의 강건성 계획과 실행은 기존 전문 화면에서 구성합니다.</p><a class="button-link" href="#robustness">강건성 열기</a>`) }${workspaceCard("5", "EvaluationProfile 적용", refs.evaluation ? "completed" : (economic.status === "completed" ? "needs_action" : "not_started"), refs.evaluation ? `<p class="id">${escapeHtml(refs.evaluation.evaluation_run_id)}</p><a class="button-link" href="#evaluations">평가 결과 상세</a>` : `<div class="field"><label>저장된 EvaluationProfile</label><select id="workspace-evaluation-profile">${profiles.items.map((item)=>`<option value="${escapeHtml(item.evaluation_profile_id)}">${escapeHtml(item.name)}</option>`).join("")}</select></div>${action("evaluate", "저장된 결과에 프로필 적용", economic.status !== "completed")}`)}${workspaceCard("6", "결과 확인", refs.evaluation ? "completed" : "not_started", refs.evaluation ? `<a class="button-link" href="#evaluations">EvaluationRun 보기</a> <a class="button-link" href="#runs">StrategyRun 보기</a>` : "<p>평가 완료 후 기존 결과 화면으로 이동할 수 있습니다.</p>")}`;
  if (refs.evaluation) {
    const reportRuns = refs.evaluation.strategy_run_ids || economic.strategy_run_ids || [];
    const attachedPlan = robustness.status === "completed" ? refs.robustness_plan?.robustness_plan_id : null;
    const existingReport = refs.decision_report;
    view.insertAdjacentHTML("beforeend", workspaceCard("7", "결정 보고서", existingReport ? "completed" : "needs_action", existingReport
      ? `<p class="id">${escapeHtml(existingReport.decision_report_id)}</p><a class="button-link" href="#decision-reports">결정 보고서 보기</a>`
      : `<div class="field"><label>StrategyRun</label><select id="workspace-decision-report-run">${reportRuns.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(shortId(id, 36))}</option>`).join("")}</select></div><button id="workspace-decision-report" ${reportRuns.length ? "" : "disabled"}>결정 보고서 생성/복원</button>`));
    document.getElementById("workspace-decision-report")?.addEventListener("click", async () => {
      try {
        const response = await api(`/workflows/${encodeURIComponent(workflowId)}/decision-report`, {
          method: "POST",
          body: {
            strategy_run_id: document.getElementById("workspace-decision-report-run").value,
            evaluation_run_id: refs.evaluation.evaluation_run_id,
            ...(attachedPlan ? { robustness_plan_id: attachedPlan } : {}),
          },
          idempotencyKey: workspaceKey(workflowId, "decision-report"),
        });
        state.selectedDecisionReport = response.decision_report.decision_report_id;
        localStorage.setItem("trend-v2-decision-report-id", state.selectedDecisionReport);
        location.hash = "#decision-reports";
      } catch (error) { setMessage(error.message); }
    });
  }
  document.getElementById("workspace-select").addEventListener("change", (event)=>{ localStorage.setItem("trend-v2-workflow-id", event.target.value); renderResearchWorkspace().catch(showFatal); });
  document.getElementById("workspace-new").addEventListener("click", ()=>{ localStorage.removeItem("trend-v2-workflow-id"); renderResearchWorkspace().catch(showFatal); });
  const bind = (id, endpoint, body={}) => document.getElementById(`workspace-${id}`)?.addEventListener("click", async()=>{ try { await api(`/workflows/${encodeURIComponent(workflowId)}${endpoint}`, {method:"POST", body, idempotencyKey:workspaceKey(workflowId,id)}); await renderResearchWorkspace(); } catch(error) { setMessage(error.message); }});
  bind("normalize", "/normalize"); bind("estimate", "/estimate"); bind("start", "/start-economic");
  document.getElementById("workspace-evaluate")?.addEventListener("click", async()=>{ try { await api(`/workflows/${encodeURIComponent(workflowId)}/evaluate`, {method:"POST", body:{evaluation_profile_id:document.getElementById("workspace-evaluation-profile").value}, idempotencyKey:workspaceKey(workflowId,"evaluate")}); await renderResearchWorkspace(); } catch(error) { setMessage(error.message); }});
}

async function renderDecisionReports() {
  const reports = await api("/decision-reports?page_size=200");
  if (!reports.items.length) {
    view.innerHTML = emptyState("결정 보고서 없음", "완료된 EvaluationRun에서 StrategyRun을 선택해 결정 보고서를 생성하거나 복원하세요.");
    return;
  }
  const restored = state.selectedDecisionReport || localStorage.getItem("trend-v2-decision-report-id");
  const selected = restored && reports.items.some((item) => item.decision_report_id === restored)
    ? restored : reports.items[0].decision_report_id;
  state.selectedDecisionReport = selected;
  localStorage.setItem("trend-v2-decision-report-id", selected);
  const detail = await api(`/decision-reports/${encodeURIComponent(selected)}`);
  const report = detail.decision_report;
  const metrics = detail.economic_metrics || {};
  const evidenceLinks = detail.evidence_links || {};
  const issues = detail.issues || [];
  const metricRows = Object.keys(metrics).length
    ? Object.entries(metrics).map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`).join("")
    : '<tr><td colspan="2">저장된 후보 raw metrics에 표시 가능한 경제 지표가 없습니다.</td></tr>';
  const links = [
    ["StrategyRun", evidenceLinks.strategy_run, report.strategy_run_id],
    ["EvaluationRun", evidenceLinks.evaluation_run, report.evaluation_run_id],
    ["EvaluationProfile", evidenceLinks.profile, report.evaluation_profile_id],
    ["행동 비교", evidenceLinks.behavior, report.evidence_references.evaluation.behavior_comparison_hash],
    ["강건성 근거", evidenceLinks.robustness, report.evidence_references.robustness.evidence_hash || "첨부되지 않음"],
  ].map(([label, href, value]) => `<li><a href="${href ? (label === "EvaluationRun" ? "#evaluations" : label === "StrategyRun" ? "#runs" : label === "행동 비교" ? "#behavior" : label === "강건성 근거" ? "#robustness" : "#profiles") : "#decision-reports"}">${escapeHtml(label)}</a><span class="id">${escapeHtml(value)}</span></li>`).join("");
  view.innerHTML = `<div class="toolbar"><div><p class="eyebrow">Decision Report</p><h2>결정 보고서</h2></div><select id="decision-report-select">${reports.items.map((item) => `<option value="${escapeHtml(item.decision_report_id)}" ${item.decision_report_id === selected ? "selected" : ""}>${escapeHtml(shortId(item.decision_report_id, 24))} · ${escapeHtml(item.decision_state)} / ${escapeHtml(item.evidence_state)}</option>`).join("")}</select></div>
  <section class="card section"><h2>평가 규칙상 결론 ${statusBadge(detail.decision_state, detail.decision_state)}</h2><p class="notice">개인화된 투자 조언이 아닙니다.</p><p>근거 상태 ${statusBadge(detail.evidence_state, detail.evidence_state)}</p></section>
  <section class="card section"><h2>주요 경제 결과</h2><div class="table-wrap"><table><tbody>${metricRows}</tbody></table></div></section>
  <section class="card section"><h2>프로필 판정</h2><pre>${jsonText(detail.evaluation)}</pre></section>
  <section class="card section"><h2>근거 상태와 경고</h2>${issues.length ? `<pre>${jsonText(issues)}</pre>` : "<p>필수 참조가 현재 검증되었습니다.</p>"}</section>
  <section class="card section"><h2>근거 추적</h2><ul>${links}</ul><pre>${jsonText({template_version:report.template_version, profile_hash:report.profile_hash, evidence_references:report.evidence_references})}</pre></section>`;
  document.getElementById("decision-report-select").addEventListener("change", (event) => {
    state.selectedDecisionReport = event.target.value;
    renderDecisionReports().catch(showFatal);
  });
}

async function renderWorkflow() {
  const workflowId = localStorage.getItem("trend-v2-workflow-id");
  if (!workflowId) {
    view.innerHTML = `<section class="card section"><h2>안내형 전략 워크플로</h2><p class="lede">구성, 경제 실행, 강건성, 평가의 참조와 재사용 여부를 단계별로 표시합니다.</p><p class="notice">워크플로 ID를 만든 뒤 이 브라우저에서 다시 열면 새 요청을 만들지 않고 저장된 상태를 복구합니다.</p></section>`;
    return;
  }
  const [data, local] = await Promise.all([api(`/workflows/${encodeURIComponent(workflowId)}`, {optional:true}), api("/local-status", {optional:true})]);
  if (data.__error) { localStorage.removeItem("trend-v2-workflow-id"); view.innerHTML=emptyState("워크플로 없음", "저장된 워크플로를 복구할 수 없습니다."); return; }
  const refs=data.references||{};
  const recovery=local.__error?null:local.last_reconciliation;
  const resumed=data.recoverability?.resumable;
  const resumeText=resumed ? "재개 가능: 완료된 경제 결과와 유효한 강건성 근거는 재사용하고, 중단·차단 단위만 명시적으로 재개하세요." : "재개 불가: 시작된 유효 작업 참조가 없거나 의존성이 손상되었습니다.";
  const resumeButton=resumed ? `<button id="resume-workflow">중단된 단위 재개</button>` : "";
  const recoveryNotice=recovery ? `<section class="card section"><h2>서비스 재시작 복구</h2><p class="notice">저장 상태 기준 · 복구 ID ${escapeHtml(shortId(recovery.recovery_id,12))} · 차단 ${fmt((recovery.blocked_items||[]).length,0)}건 · 손상 ${fmt((recovery.corrupt_items||[]).length,0)}건</p><p>${escapeHtml(resumeText)}</p>${resumeButton}<details><summary>마지막 복구 결과</summary><pre>${jsonText(recovery)}</pre></details></section>` : "";
  view.innerHTML=`<section class="card section"><h2>${escapeHtml(data.label_ko)}</h2>${keyValues([["워크플로",data.workflow_id,true],["저장된 단계",data.stage],["실시간 서비스 상태",local.__error?"확인 불가":(local.active_attempts||[]).length?"작업 있음":"활성 작업 없음"],["복구 가능",resumed?"예":"아니오"],["생성",data.created_timestamp]])}</section>${recoveryNotice}<section class="card section"><h2>단계별 참조</h2>${keyValues([["정규화",refs.normalized_construction?.construction_hash||"—",true],["후보 추정",refs.candidate_estimate?.candidate_estimate_hash||"—",true],["ExecutionRequest",refs.economic?.execution_request_id||"—",true],["강건성 계획",refs.robustness_plan?.robustness_plan_id||"—",true],["EvaluationRun",refs.evaluation?.evaluation_run_id||"—",true]])}<p class="notice">재사용된 완료 결과와 강건성 근거는 다시 실행하지 않습니다. 손상되었거나 누락된 의존성은 재개 대상이 아닙니다.</p><details><summary>저장된 워크플로 이력</summary><pre>${jsonText({recoverability:data.recoverability,events:data.events})}</pre></details></section>`;
  document.getElementById("resume-workflow")?.addEventListener("click", async () => { await api(`/workflows/${encodeURIComponent(workflowId)}/resume`, { method: "POST", body: {}, idempotencyKey: idempotencyKey("workflow-resume") }); await renderWorkflow(); });
}

async function renderSystem() {
  const [health,metadata,overview]=await Promise.all([api("/health"),api("/metadata"),api("/overview")]);
  view.innerHTML=`<p class="lede">저장 근거 읽기와 통제된 로컬 쓰기 경계 및 현재 버전을 확인합니다.</p><section class="card-grid"><article class="card metric-card"><small>API 상태</small><strong>${statusBadge(health.status,health.status_ko)}</strong></article><article class="card metric-card"><small>읽기 전용</small><strong>${health.read_only?"예":"통제 쓰기 활성"}</strong></article><article class="card metric-card"><small>최대 목록 페이지</small><strong>${fmt(metadata.maximum_page_size,0)}</strong></article><article class="card metric-card"><small>최대 시계열 페이지</small><strong>${fmt(metadata.maximum_time_series_page_size,0)}</strong></article></section><div class="two-column"><section class="card"><h2>버전</h2><pre>${jsonText(metadata)}</pre></section><section class="card"><h2>레지스트리 재구축 식별</h2><pre>${jsonText(overview.last_registry_rebuild_identity)}</pre><h3>보안 경계</h3><ul><li>기본 루프백 호스트만 사용</li><li>브라우저에서 임의 경로·원격 URL 입력 없음</li><li>허용 목록 밖 쓰기·셸·동적 Python·시장 데이터 요청 없음</li><li>CORS 비활성 또는 명시적 로컬 출처만 허용</li><li>오류 화면에 스택 추적이나 로컬 절대 경로를 표시하지 않음</li></ul></section></div>`;
}

function studioDraft(source) {
  const draft = JSON.parse(JSON.stringify(source));
  for (const key of ["name", "approval_status", "schema_version", "lineage"]) delete draft[key];
  return draft;
}

function studioJsonField(id, label, value) {
  return `<div class="field studio-json"><label for="${id}">${escapeHtml(label)}</label><textarea id="${id}" rows="5" spellcheck="false">${escapeHtml(JSON.stringify(value, null, 2))}</textarea></div>`;
}

function studioParse(id, label) {
  try { return JSON.parse(document.getElementById(id).value); }
  catch (_error) { throw new Error(`${label} 입력은 JSON 형식이어야 합니다.`); }
}

function studioPayload(sourceId, source) {
  const profile = studioDraft(source);
  profile.description = document.getElementById("studio-description").value;
  profile.enabled_metrics = studioParse("studio-enabled-metrics", "사용 지표");
  profile.metric_directions = studioParse("studio-directions", "지표 방향");
  profile.metric_modes = studioParse("studio-modes", "지표 모드");
  profile.mandatory_gates = studioParse("studio-gates", "필수 관문");
  profile.pareto_objectives = studioParse("studio-pareto", "파레토 목표");
  profile.robustness_vetoes = studioParse("studio-vetoes", "강건성 거부권");
  profile.lexicographic_tie_break = studioParse("studio-tiebreak", "사전식 동률 해소");
  profile.behavior_deduplication = studioParse("studio-behavior", "행동 중복 제거");
  profile.exploratory_metric_weights = studioParse("studio-weights", "탐색 가중치");
  profile.normalization_method = document.getElementById("studio-normalization").value || null;
  profile.ranking_sensitivity_delta = numericValue(document.getElementById("studio-sensitivity").value);
  profile.high_weighted_rank_cutoff = numericValue(document.getElementById("studio-rank-cutoff").value);
  return { source_profile_id: sourceId, change_summary_ko: document.getElementById("studio-change-summary").value, profile };
}

async function renderProfileStudioEditor(sourceId, source, options) {
  const draft = studioDraft(source);
  const metricList = options.metrics.map((item) => `${item.metric_key} · ${item.korean_name}`).join("\n");
  view.innerHTML = `<p class="lede">기존 프로필은 수정하지 않습니다. 검증된 새 버전만 저장하며, 탐색 가중 결과는 기본 비보상형 판단을 바꾸지 않습니다.</p>
  <section class="card section"><h2>복제 원본</h2>${keyValues([["원본",sourceId,true],["원본 해시",source.profile_hash||"저장된 프로필",true],["승인 상태",source.approval_status]])}</section>
  <form id="profile-studio-form" class="section"><section class="card"><h2>버전 설명</h2><div class="field"><label for="studio-change-summary">변경 요약 (한국어)</label><input id="studio-change-summary" required maxlength="500" placeholder="예: 회복기간 기준을 보수적으로 조정"></div><div class="field"><label for="studio-description">설명</label><textarea id="studio-description" rows="3" maxlength="2000">${escapeHtml(draft.description||"")}</textarea></div></section>
  <section class="card"><h2>Metric Registry</h2><p class="notice">아래 등록 지표와 용도만 사용할 수 있습니다. 임의 수식·코드·경로는 저장되지 않습니다.</p><pre>${escapeHtml(metricList)}</pre>${studioJsonField("studio-enabled-metrics","사용 지표",draft.enabled_metrics)}${studioJsonField("studio-directions","지표 방향",draft.metric_directions)}${studioJsonField("studio-modes","지표 모드",draft.metric_modes)}</section>
  <section class="card"><h2>기본 비보상형 판단</h2>${studioJsonField("studio-gates","필수 관문",draft.mandatory_gates)}${studioJsonField("studio-pareto","파레토 목표와 ε",draft.pareto_objectives)}${studioJsonField("studio-vetoes","강건성 거부권",draft.robustness_vetoes)}${studioJsonField("studio-tiebreak","사전식 동률 해소",draft.lexicographic_tie_break)}</section>
  <section class="card"><h2>행동 중복 제거</h2><p class="notice">기존 상관·Jaccard·경로 거리 진단만 사용합니다.</p>${studioJsonField("studio-behavior","고정 진단 조건과 대표 순서",draft.behavior_deduplication)}</section>
  <section class="card weighted"><h2>별도 탐색 가중 보기</h2><p class="notice">필수 관문, 파레토, 강건성, 동률 해소 또는 행동 중복 제거 결과를 덮어쓰지 않습니다.</p>${studioJsonField("studio-weights","탐색 가중치",draft.exploratory_metric_weights)}<div class="field"><label for="studio-normalization">정규화</label><select id="studio-normalization"><option value="">없음</option>${options.normalization_methods.map((item)=>`<option value="${item}" ${draft.normalization_method===item?"selected":""}>${item}</option>`).join("")}</select></div><div class="two-column"><div class="field"><label for="studio-sensitivity">민감도 δ</label><input id="studio-sensitivity" type="number" min="0.001" max="1" step="0.01" value="${escapeHtml(draft.ranking_sensitivity_delta)}"></div><div class="field"><label for="studio-rank-cutoff">높은 탐색 순위 기준</label><input id="studio-rank-cutoff" type="number" min="1" step="1" value="${escapeHtml(draft.high_weighted_rank_cutoff)}"></div></div></section>
  <div class="toolbar"><button type="button" id="studio-validate">프로필 검증</button><button type="button" id="studio-save" disabled>새 불변 버전 저장</button><button type="button" id="studio-cancel">목록으로</button></div><div id="studio-validation" aria-live="polite"></div></form>`;
  let validated = null;
  document.getElementById("studio-cancel").addEventListener("click",()=>renderProfileStudio().catch(showFatal));
  document.getElementById("studio-validate").addEventListener("click",async()=>{try { const payload=studioPayload(sourceId,source); const result=await api("/evaluation-profiles/validate",{method:"POST",body:payload}); validated=result.valid?{payload,draftHash:result.draft_hash}:null; document.getElementById("studio-save").disabled=!validated; document.getElementById("studio-validation").innerHTML=result.valid?'<p class="notice">검증 완료: 현재 초안만 저장할 수 있습니다.</p>':`<div class="notice">${(result.errors||[]).map((item)=>escapeHtml(item.message_ko||item.code)).join("<br>")}</div>`; } catch(error) { validated=null; document.getElementById("studio-save").disabled=true; document.getElementById("studio-validation").textContent=error.message; }});
  document.getElementById("studio-save").addEventListener("click",async()=>{if(!validated)return; const current=studioPayload(sourceId,source); const response=await api("/evaluation-profiles",{method:"POST",body:{...current,validated_draft_hash:validated.draftHash},idempotencyKey:idempotencyKey("profile-save")}); await renderProfileStudioSaved(response).catch(showFatal);});
}

async function renderProfileStudioSaved(saved) {
  const [history,runs] = await Promise.all([api(`/evaluation-profiles/${encodeURIComponent(saved.evaluation_profile_id)}/lineage`),api("/runs?page_size=200")]);
  view.innerHTML=`<section class="card section"><h2>새 불변 EvaluationProfile 저장됨</h2>${keyValues([["프로필",saved.evaluation_profile_id,true],["해시",saved.profile_hash,true],["변경 요약",saved.lineage.change_summary_ko],["버전",saved.lineage.revision]])}</section><section class="card section"><h2>계보와 변경 이력</h2><div class="table-wrap"><table><thead><tr><th>버전</th><th>프로필</th><th>부모</th><th>변경</th></tr></thead><tbody>${history.items.map((item)=>`<tr><td>${fmt(item.lineage.revision,0)}</td><td class="id">${escapeHtml(item.evaluation_profile_id)}</td><td class="id">${escapeHtml(item.lineage.parent_profile_id||"—")}</td><td>${escapeHtml(item.lineage.change_summary_ko)}</td></tr>`).join("")}</tbody></table></div></section><section class="card section"><h2>저장된 StrategyRun에 명시적으로 적용</h2><p class="notice">경제 백테스트를 다시 실행하지 않고 저장된 근거만 사용합니다.</p><div class="field"><label for="studio-apply-run">StrategyRun</label><select id="studio-apply-run">${runs.items.map((item)=>`<option value="${escapeHtml(item.strategy_run_id)}">${escapeHtml(shortId(item.strategy_run_id,40))}</option>`).join("")}</select></div><button id="studio-apply" ${runs.items.length?"":"disabled"}>이 프로필 적용</button><div id="studio-apply-result" aria-live="polite"></div></section>`;
  document.getElementById("studio-apply")?.addEventListener("click",async()=>{const runId=document.getElementById("studio-apply-run").value;const result=await api(`/evaluation-profiles/${encodeURIComponent(saved.evaluation_profile_id)}/apply`,{method:"POST",body:{strategy_run_id:runId},idempotencyKey:idempotencyKey("profile-apply")});document.getElementById("studio-apply-result").innerHTML=`<p class="notice">EvaluationRun 생성: ${escapeHtml(result.evaluation_run.evaluation_run_id)} · 경제 백테스트 시작: 아니오</p>`;});
}

async function renderProfileStudio() {
  const [profiles,options]=await Promise.all([api("/evaluation-profiles?page_size=200&sort=name"),api("/evaluation-profile-studio/options")]);
  profiles.items.sort((a,b)=>a.name==="research_default"?-1:b.name==="research_default"?1:a.name.localeCompare(b.name,"ko"));
  view.innerHTML=`<p class="lede">저장된 EvaluationProfile을 복제해 새 불변 버전을 만듭니다. 기본 비교는 비보상형이며 가중 결과는 탐색용입니다.</p><section class="card section"><h2>저장된 프로필</h2><div class="table-wrap"><table><thead><tr><th>이름</th><th>모드</th><th>승인 상태</th><th>해시</th><th>작업</th></tr></thead><tbody>${profiles.items.map((item)=>`<tr><th>${escapeHtml(item.name)}</th><td>${escapeHtml(item.comparison_mode)}</td><td>${escapeHtml(item.approval_status)}</td><td class="id">${escapeHtml(shortId(item.profile_hash,18))}</td><td><button data-studio-clone="${escapeHtml(item.evaluation_profile_id)}">복제</button></td></tr>`).join("")}</tbody></table></div></section>`;
  document.querySelectorAll("[data-studio-clone]").forEach((button)=>button.addEventListener("click",async()=>{const id=button.dataset.studioClone;const source=await api(`/evaluation-profiles/${encodeURIComponent(id)}`);await renderProfileStudioEditor(id,source,options);}));
}

function showFatal(error) {
  if (error?.name === "AbortError") return;
  setLoading(""); setMessage(error?.message || "화면을 불러오지 못했습니다.");
  view.innerHTML = emptyState("화면 로딩 실패", "위 오류 메시지를 확인한 뒤 다시 시도해 주세요.");
}

async function navigate() {
  state.controller?.abort(); state.controller=new AbortController(); setMessage(""); setLoading("화면을 불러오는 중입니다…");
  const raw=location.hash.replace(/^#/,"")||"overview"; const [routeRaw,detailRaw]=raw.split("/"); const route=routes[routeRaw]?routeRaw:"overview"; const detail=detailRaw?decodeURIComponent(detailRaw):null;
  pageTitle.textContent=route==="runs"&&detail?"StrategyRun 상세":routes[route];
  document.querySelectorAll("#primary-nav a").forEach((link)=>{if(link.dataset.route===route)link.setAttribute("aria-current","page");else link.removeAttribute("aria-current");});
  try {
    await ensureTerminology();
    if(route==="workflow")await renderResearchWorkspace();
    else if(route==="construction")await renderConstruction();
    else if(route==="requests")await renderRequests();
    else if(route==="overview")await renderOverview();
    else if(route==="runs"&&detail)await renderRunDetail(detail);
    else if(route==="runs")await renderRuns();
    else if(route==="profiles")await renderProfileStudio();
    else if(route==="evaluations")await renderEvaluations();
    else if(route==="decision-reports")await renderDecisionReports();
    else if(route==="performance")await renderPerformance();
    else if(route==="robustness")await renderRobustness();
    else if(route==="behavior")await renderBehavior();
    else if(route==="attempts")await renderAttempts();
    else if(route==="explanations")await renderExplanations(detail);
    else if(route==="system")await renderSystem();
    connection.className="connection ok"; connection.innerHTML='<span class="connection-dot" aria-hidden="true"></span>API 연결됨';
  } catch(error){showFatal(error);connection.className="connection error";connection.innerHTML='<span class="connection-dot" aria-hidden="true"></span>API 오류';}
  finally{
    setLoading("");
    const explanationTarget = route === "explanations" && detail
      ? document.getElementById(`term-${detail}`)
      : null;
    (explanationTarget || document.getElementById("main-content")).focus({preventScroll:true});
  }
}

window.addEventListener("hashchange",navigate);
window.addEventListener("DOMContentLoaded",()=>{
  const navigation=document.getElementById("primary-nav");
  if(navigation&&!navigation.querySelector('[data-route="workflow"]')) navigation.insertAdjacentHTML("afterbegin",'<a href="#workflow" data-route="workflow"><span aria-hidden="true">◎</span> 연구 워크스페이스</a>');
  if(navigation&&!navigation.querySelector('[data-route="decision-reports"]')) navigation.insertAdjacentHTML("afterbegin",'<a href="#decision-reports" data-route="decision-reports"><span aria-hidden="true">▣</span> 결정 보고서</a>');
  navigate();
});

// Stable helpers exposed only for synthetic, dependency-free UI contract tests.
globalThis.TrendV2Ui = Object.freeze({ escapeHtml, availabilityLabel, numericValue, drawdownRows, buildChartRows, candidateEstimatePresentation, estimateSummary });
