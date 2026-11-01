"use strict";

const form = document.querySelector("#analysis-form");
const loading = document.querySelector("#loading");
const errorPanel = document.querySelector("#error-panel");
const results = document.querySelector("#results");
const downloadButton = document.querySelector("#download-card");
let chart;

function disableDownloadButton() {
  downloadButton.removeAttribute("href");
  downloadButton.classList.add("disabled");
  downloadButton.setAttribute("aria-disabled", "true");
  downloadButton.setAttribute("tabindex", "-1");
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showError(message) {
  errorPanel.textContent = message;
  errorPanel.hidden = false;
  results.hidden = true;
}

function renderCards(data) {
  const cards = document.querySelector("#summary-cards");
  cards.replaceChildren();
  const worstClass = data.worst_class;
  const values = [
    ["Baseline accuracy", percent(data.baseline.accuracy), `${data.baseline.correct} / ${data.baseline.total} correct`],
    ["Mean perturbed accuracy", percent(data.mean_perturbed_accuracy), "Across all registered perturbations"],
    ["Robustness score", data.robustness_score === null ? "N/A" : percent(data.robustness_score), "Mean perturbed accuracy ÷ baseline"],
    ["Worst perturbation", data.worst_perturbation?.name ?? "N/A", data.worst_perturbation ? percent(data.worst_perturbation.accuracy) : ""],
    ["Worst class", worstClass ? `Class ${worstClass.class_id}` : "N/A", worstClass ? percent(worstClass.mean_perturbed_accuracy) : ""],
  ];
  for (const [title, value, detail] of values) {
    const card = element("article", "", "summary-card");
    card.append(element("p", title), element("strong", value), element("small", detail));
    cards.append(card);
  }
}

function renderPerturbations(data) {
  const entries = Object.entries(data.perturbations).sort((a, b) => a[1].accuracy - b[1].accuracy);
  const rows = document.querySelector("#perturbation-rows");
  rows.replaceChildren();
  for (const [name, item] of entries) {
    const row = document.createElement("tr");
    const drop = (data.baseline.accuracy - item.accuracy) * 100;
    for (const value of [name, percent(item.accuracy), `${item.correct} / ${item.total}`, `${drop.toFixed(2)} pp`]) {
      row.append(element("td", value));
    }
    rows.append(row);
  }

  if (chart) chart.destroy();
  chart = new Chart(document.querySelector("#perturbation-chart"), {
    type: "bar",
    data: {
      labels: entries.map(([name]) => name),
      datasets: [{
        label: "Accuracy",
        data: entries.map(([, item]) => item.accuracy * 100),
        backgroundColor: entries.map((_, index) => index === 0 ? "#bd5b4d" : "#39866a"),
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { min: 0, max: 100, ticks: { callback: (value) => `${value}%` } } },
    },
  });
}

function renderClasses(data) {
  const rows = document.querySelector("#class-rows");
  rows.replaceChildren();
  const classes = Object.entries(data.per_class).map(([classId, item]) => {
    const perturbations = Object.entries(item.perturbations);
    const worst = perturbations.sort((a, b) => a[1] - b[1])[0];
    return { classId, item, worst, rank: worst ? worst[1] : item.clean_accuracy };
  }).sort((a, b) => a.rank - b.rank).slice(0, 10);

  for (const entry of classes) {
    const row = document.createElement("tr");
    const cells = [
      `Class ${entry.classId}`,
      String(entry.item.n),
      percent(entry.item.clean_accuracy),
      entry.worst ? entry.worst[0] : "—",
      percent(entry.rank),
    ];
    for (const value of cells) row.append(element("td", value));
    rows.append(row);
  }
}

function renderReport(data) {
  const banner = document.querySelector("#baseline-banner");
  banner.className = `notice ${data.baseline_ok ? "success" : "failure"}`;
  banner.textContent = data.baseline_ok
    ? (data.baseline.matches === undefined
      ? "Baseline computed. No claimed accuracy was supplied for comparison."
      : "Baseline check passed. The model and test data are consistent with the claimed accuracy.")
    : (data.message || "The model or data does not match the claimed accuracy.");
  renderCards(data);
  renderPerturbations(data);
  renderClasses(data);
  if (data.report_id) {
    downloadButton.href = `/card/${encodeURIComponent(data.report_id)}`;
    downloadButton.classList.remove("disabled");
    downloadButton.removeAttribute("aria-disabled");
    downloadButton.removeAttribute("tabindex");
  } else {
    disableDownloadButton();
  }
  errorPanel.hidden = true;
  results.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorPanel.hidden = true;
  results.hidden = true;
  disableDownloadButton();
  loading.hidden = false;
  const submit = form.querySelector("button[type=submit]");
  submit.disabled = true;
  try {
    const response = await fetch("/analyze", { method: "POST", body: new FormData(form) });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || "Analysis failed.");
    renderReport(data);
  } catch (error) {
    showError(error.message || "Analysis failed.");
  } finally {
    loading.hidden = true;
    submit.disabled = false;
  }
});
