"use strict";

const advisorForm = document.querySelector("#advisor-analysis-form");
const advisorDescription = document.querySelector("#advisor-model-description");
const advisorLoading = document.querySelector("#advisor-loading");
const advisorError = document.querySelector("#advisor-error");
const advisorOutput = document.querySelector("#advisor-results");
const lastAnalysisOffer = document.querySelector("#last-analysis-offer");
const useLastAnalysis = document.querySelector("#use-last-analysis");
let advisorChart;
const savedReportId = sessionStorage.getItem("report_id");
if (savedReportId) lastAnalysisOffer.hidden = false;

function asPercent(value) {
  return `${(Number(value) * 100).toFixed(2).replace(/\.00$/, "")}%`;
}

function roundLongDecimals(value) {
  return String(value).replace(/-?\d+\.\d{3,}/g, (number) => Number(number).toFixed(2));
}

function makeText(tag, value, className) {
  const node = document.createElement(tag);
  node.textContent = roundLongDecimals(value);
  if (className) node.className = className;
  return node;
}

function showError(message) {
  advisorError.textContent = roundLongDecimals(message);
  advisorError.hidden = false;
  advisorOutput.hidden = true;
}

function evidenceRow(text) {
  const row = document.createElement("div");
  row.className = "evidence-row";
  row.append(document.createTextNode("Evidence"));
  const numbers = text.match(/-?\d+(?:\.\d+)?\s*(?:%|pp|percentage points|samples|correct)?/gi) || [];
  for (const number of numbers) row.append(makeText("span", number.trim(), "evidence-chip"));
  if (!numbers.length) row.append(makeText("span", text));
  return row;
}

function renderSuggestions(report, advice) {
  const summary = document.querySelector("#advisor-summary");
  summary.replaceChildren();
  summary.append(makeText("p", "REPORT SUMMARY", "eyebrow"));
  summary.append(makeText("p", advice.summary));
  summary.append(makeText("p", `Baseline ${asPercent(report.baseline.accuracy)} · mean perturbed ${asPercent(report.mean_perturbed_accuracy)} · robustness score ${asPercent(report.robustness_score)} · weakest: ${report.worst_perturbation?.name || "not available"}`, "advisor-summary-metrics"));

  const entries = Object.entries(report.perturbations).sort((a, b) => a[1].accuracy - b[1].accuracy);
  const valueList = document.querySelector("#advisor-perturbation-values");
  valueList.replaceChildren();
  entries.forEach(([name, item]) => {
    const drop = (report.baseline.accuracy - item.accuracy) * 100;
    const row = document.createElement("div");
    row.className = "advisor-value-row";
    row.append(makeText("span", name), makeText("span", asPercent(item.accuracy)), makeText("span", `${drop.toFixed(2)} pp`));
    valueList.append(row);
  });

  if (advisorChart) advisorChart.destroy();
  advisorChart = new Chart(document.querySelector("#advisor-chart"), {
    type: "bar",
    data: { labels: entries.map(([name]) => name), datasets: [
      { label: "Baseline", data: entries.map(() => report.baseline.accuracy * 100), backgroundColor: "rgba(185,166,255,.25)", borderRadius: 5 },
      { label: "Perturbed accuracy", data: entries.map(([, item]) => item.accuracy * 100), backgroundColor: "#8b70f5", borderRadius: 5 },
    ] },
    options: { indexAxis: "y", maintainAspectRatio: false, scales: { x: { min: 0, max: 100, ticks: { color: "#aaa2bf", callback: (value) => `${Number(value).toFixed(2).replace(/\.00$/, "")}%` }, grid: { color: "rgba(189,177,235,.12)" } }, y: { ticks: { color: "#e3def1" }, grid: { display: false } } }, plugins: { legend: { labels: { color: "#e3def1" } }, tooltip: { callbacks: { label: (context) => { const item = entries[context.dataIndex][1]; const acc = asPercent(context.raw / 100); return context.datasetIndex === 0 ? `Baseline: ${acc}` : `Accuracy: ${acc} · drop ${((report.baseline.accuracy - item.accuracy) * 100).toFixed(2)} pp`; } } } } },
  });

  const cards = document.querySelector("#suggestion-cards");
  cards.replaceChildren();
  const priorityOrder = { high: 0, medium: 1, low: 2 };
  [...advice.suggestions].sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]).forEach((suggestion) => {
    const card = document.createElement("article");
    card.className = "suggestion-card";
    const titleRow = document.createElement("div");
    titleRow.className = "suggestion-title-row";
    titleRow.append(makeText("span", suggestion.priority, `priority-badge priority-${suggestion.priority}`), makeText("h3", suggestion.title));
    card.append(titleRow, evidenceRow(suggestion.evidence));
    for (const [label, field] of [["What to change", "what_to_change"], ["Expected effect", "expected_effect"]]) {
      const details = document.createElement("details");
      details.append(makeText("summary", label), makeText("p", suggestion[field]));
      card.append(details);
    }
    cards.append(card);
  });
  const caveats = document.querySelector("#advisor-caveats");
  caveats.replaceChildren();
  advice.caveats.forEach((item) => caveats.append(makeText("li", item)));
  advisorError.hidden = true;
  advisorOutput.hidden = false;
}

async function loadAdvice(reportId, report = null) {
  advisorError.hidden = true;
  advisorOutput.hidden = true;
  advisorLoading.hidden = false;
  useLastAnalysis.disabled = true;
  try {
    let serverReport = report;
    if (!serverReport) {
      const reportResponse = await fetch(`/report/${encodeURIComponent(reportId)}`);
      serverReport = await reportResponse.json();
      if (reportResponse.status === 404) {
        sessionStorage.removeItem("report_id");
        lastAnalysisOffer.hidden = true;
        throw new Error("Please run the analysis again.");
      }
      if (!reportResponse.ok || serverReport.error) throw new Error(serverReport.error || "Could not load the report.");
    }
    const response = await fetch("/advise", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ report_id: reportId, model_description: advisorDescription.value.slice(0, 3000) }) });
    const advice = await response.json();
    if (response.status === 404) {
      sessionStorage.removeItem("report_id");
      lastAnalysisOffer.hidden = true;
      throw new Error("Please run the analysis again.");
    }
    if (!response.ok || advice.error) throw new Error(advice.error || "Could not generate suggestions.");
    renderSuggestions(serverReport, advice);
  } catch (error) {
    showError(error.message || "Could not generate suggestions.");
  } finally {
    advisorLoading.hidden = true;
    useLastAnalysis.disabled = false;
  }
}

useLastAnalysis.addEventListener("click", () => loadAdvice(savedReportId));
advisorForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  advisorError.hidden = true;
  advisorOutput.hidden = true;
  advisorLoading.hidden = false;
  const submit = advisorForm.querySelector("button[type=submit]");
  submit.disabled = true;
  try {
    const response = await fetch("/analyze", { method: "POST", body: new FormData(advisorForm) });
    const report = await response.json();
    if (!response.ok || report.error) throw new Error(report.error || "Analysis failed.");
    sessionStorage.setItem("report_id", report.report_id);
    lastAnalysisOffer.hidden = false;
    await loadAdvice(report.report_id, report);
  } catch (error) {
    showError(error.message || "Analysis failed.");
  } finally {
    advisorLoading.hidden = true;
    submit.disabled = false;
  }
});
