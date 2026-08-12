const state = {
  dashboard: null,
  lastPositionPrediction: null,
  lastDatasetResult: null,
};
console.log('app.js loaded');

const FIELD_LABELS = {
  Nation: "Nation",
  Squad: "Squad",
  Comp: "League",
  Pos: "Position",
  season: "Season",
  Age: "Age",
  MP: "Matches Played",
  Starts: "Starts",
  Min: "Minutes",
  "90s": "90s",
  Gls: "Goals",
  Ast: "Assists",
  PK: "Penalty Goals",
  PKatt: "Penalty Attempts",
  CrdY: "Yellow Cards",
  CrdR: "Red Cards",
  xG: "xG",
  npxG: "npxG",
  xAG: "xAG",
  PrgC: "Progressive Carries",
  PrgP: "Progressive Passes",
  PrgR: "Progressive Receptions",
};

const POSITION_FORM_GROUPS = [
  {
    title: "Profile & Context",
    fields: ["Nation", "Squad", "Comp", "season"],
  },
  {
    title: "Playing Time",
    fields: ["Age", "MP", "Starts", "Min", "90s"],
  },
  {
    title: "Role Signals",
    fields: ["Gls", "Ast", "PK", "PKatt", "CrdY", "CrdR", "xG", "npxG", "xAG", "PrgC", "PrgP", "PrgR"],
  },
];

const GOALS_FORM_GROUPS = [
  {
    title: "Profile & Context",
    fields: ["Nation", "Squad", "Comp", "Pos", "season"],
  },
  {
    title: "Playing Time",
    fields: ["Age", "MP", "Starts", "Min", "90s"],
  },
  {
    title: "Creation & Progression",
    fields: ["Ast", "PK", "PKatt", "CrdY", "CrdR", "xG", "npxG", "xAG", "PrgC", "PrgP", "PrgR"],
  },
];

document.addEventListener("DOMContentLoaded", () => {
  initializeDashboard()
    .catch((error) => {
      renderLoadError(error);
    })
    .finally(() => {
      attachDatasetFormHandler();
      initializeSimilaritySystem();
      initRecommendationSystem();
      initPlayerToolsSystem();
      initRadarSystem();
      initPlayerIntelligenceSystem();
      initClusteringSystem();
      initMarketValueSystem();
      initTransferReplaceSystem();
      initBestXISystem();
      initInjuryRiskSystem();
    });
});

function attachDatasetFormHandler() {
  const form = document.getElementById("dataset-form");
  if (!form) {
    console.error("Dataset form not found - cannot attach handler");
    return;
  }

  form.onsubmit = (event) => {
    console.log("=== FORM SUBMIT EVENT FIRED ===");
    event.preventDefault();
    event.stopPropagation();
    console.log("Default prevented, calling handleDatasetSubmit");
    handleDatasetSubmit(event);
    return false;
  };
  
  console.log("Dataset form onsubmit handler attached successfully");
}

async function initializeDashboard() {
  const dashboard = await fetchJSON("/api/dashboard");
  state.dashboard = dashboard;

  renderHero(dashboard.hero);
  renderKpis(dashboard.kpis);
  renderClassificationChart(dashboard.classificationComparison);
  renderRegressionChart(dashboard.regressionComparison, dashboard.baselineComparison);
  renderModelExamples(dashboard.modelExamples);
  renderFindings(dashboard.findings);
  renderClassBreakdown(dashboard.classBreakdown);
  renderMatrices(dashboard.confusionMatrices);
  renderFeatureHighlights(dashboard.featureHighlights);
  renderPredictor(dashboard.predictor);
  renderAwardRadar(dashboard.awardRadar);
  renderDatasetLab(dashboard.predictor.schema);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed.");
  }
  return payload;
}

function renderLoadError(error) {
  const message = `
    <div class="loading-card">
      <strong>Dashboard failed to load.</strong>
      <p>${error.message}</p>
    </div>
  `;
  ["kpi-grid", "classification-chart", "regression-chart", "findings-grid"].forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.innerHTML = message;
    }
  });

  ["dataset-summary", "dataset-top10", "dataset-league-top5"].forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.innerHTML = message;
    }
  });
}

function renderHero(hero) {
  document.getElementById("hero-title").textContent = hero.title;
  document.getElementById("hero-subtitle").textContent = hero.subtitle;

  const heroMeta = document.getElementById("hero-meta");
  const metaRows = [
    { label: "Rows in shared base", value: hero.meta.combined_rows },
    { label: "Rows in detailed 2024/25", value: hero.meta.detailed_rows },
    { label: "Inputs for live classifier", value: hero.meta.combined_features },
    { label: "Inputs in detailed comparison model", value: hero.meta.detailed_features },
  ];
  heroMeta.innerHTML = metaRows
    .map(
      (item) => `
        <div class="hero-meta-item">
          <span>${item.label}</span>
          <strong>${item.value}</strong>
        </div>
      `
    )
    .join("");
}

function renderKpis(kpis) {
  const container = document.getElementById("kpi-grid");
  container.innerHTML = kpis
    .map(
      (item) => `
        <article class="kpi-card ${item.tone}">
          <div class="card-kicker">${item.label}</div>
          <div class="kpi-value">${item.value}${item.suffix}</div>
          <div class="kpi-detail">${item.detail}</div>
        </article>
      `
    )
    .join("");
}

function renderClassificationChart(rows) {
  const container = document.getElementById("classification-chart");
  const groups = groupBy(rows, "experimentLabel");

  container.innerHTML = Object.entries(groups)
    .map(([experimentLabel, items]) => {
      const bars = items
        .map(
          (item) => `
            <div class="bar-row">
              <div class="bar-label">${item.model}</div>
              <div>
                <div class="bar-track">
                  <div class="bar-fill" style="width: ${item.accuracy}%"></div>
                </div>
                <div class="metrics-inline">
                  <span class="metric-chip">Accuracy ${item.accuracy}%</span>
                  <span class="metric-chip">Macro-F1 ${item.macroF1}%</span>
                  <span class="metric-chip">MF recall ${item.midfielderRecall}%</span>
                </div>
              </div>
              <div class="bar-value">${item.accuracy}%</div>
            </div>
          `
        )
        .join("");

      return `
        <div class="chart-group">
          <div class="chart-group-title">${experimentLabel}</div>
          ${bars}
        </div>
      `;
    })
    .join("");
}

function renderRegressionChart(rows, baseline) {
  const container = document.getElementById("regression-chart");
  const maxR2 = Math.max(...rows.map((row) => row.r2), 1);

  container.innerHTML = rows
    .map((item, index) => {
      const width = Math.max((item.r2 / maxR2) * 100, 0);
      return `
        <div class="chart-group">
          <div class="bar-row">
            <div class="bar-label">${item.model}</div>
            <div>
              <div class="bar-track">
                <div class="bar-fill ${index === 0 ? "" : "alt"}" style="width: ${width}%"></div>
              </div>
              <div class="metrics-inline">
                <span class="metric-chip">R2 ${item.r2}</span>
                <span class="metric-chip">MAE ${item.mae}</span>
                <span class="metric-chip">RMSE ${item.rmse}</span>
                <span class="metric-chip">MSE ${item.mse}</span>
              </div>
            </div>
            <div class="bar-value">${item.r2}</div>
          </div>
        </div>
      `;
    })
    .join("");

  document.getElementById("baseline-callout").innerHTML = `
    <strong>Baseline vs refresh.</strong>
    Notebook RF holdout accuracy was ${baseline.classification_holdout_baseline}%. The shared-base winner adds ${baseline.combined_improvement} pts, and the detailed winner adds ${baseline.detailed_improvement} pts.
    The notebook linear regression baseline landed at R2 ${baseline.regression_baseline_r2} with MSE ${baseline.regression_baseline_mse}; the exported best regressor trims MSE by ${Math.abs(baseline.regression_mse_delta).toFixed(3)}.
  `;
}

function renderModelExamples(modelExamples) {
  document.getElementById("position-examples").innerHTML = `
    <div class="example-section">
      <div class="note-chip">Examples use ${modelExamples.modelsUsed.position}</div>
      <div class="example-title">Examples the position model nailed</div>
      <div class="example-grid">
        ${modelExamples.position.accurate.map(renderPositionExampleCard).join("")}
      </div>
    </div>
    <div class="example-section">
      <div class="example-title">Examples where roles overlapped</div>
      <div class="example-grid">
        ${modelExamples.position.misses.map(renderPositionExampleCard).join("")}
      </div>
    </div>
  `;

  document.getElementById("goal-examples").innerHTML = `
    <div class="example-section">
      <div class="note-chip">Examples use ${modelExamples.modelsUsed.goals}</div>
      <div class="example-title">Closest goal predictions</div>
      <div class="example-grid">
        ${modelExamples.goals.accurate.map(renderGoalExampleCard).join("")}
      </div>
    </div>
    <div class="example-section">
      <div class="example-title">Biggest goal misses</div>
      <div class="example-grid">
        ${modelExamples.goals.misses.map(renderGoalExampleCard).join("")}
      </div>
    </div>
  `;
}

function renderPositionExampleCard(example) {
  const correct = example.actual === example.predicted;
  return `
    <article class="example-card">
      <div class="example-top">
        <div>
          <strong>${example.player}</strong>
          <span>${example.squad}</span>
        </div>
        <span class="example-badge ${correct ? "good" : "warn"}">
          ${example.actual} → ${example.predicted}
        </span>
      </div>
      <div class="metrics-inline">
        <span class="metric-chip">Confidence ${example.confidence}%</span>
        <span class="metric-chip">G+A ${example.goalContrib}</span>
        <span class="metric-chip">PrgP ${example.progressivePasses}</span>
        <span class="metric-chip">PrgR ${example.progressiveReceptions}</span>
      </div>
    </article>
  `;
}

function renderGoalExampleCard(example) {
  return `
    <article class="example-card">
      <div class="example-top">
        <div>
          <strong>${example.player}</strong>
          <span>${example.squad}</span>
        </div>
        <span class="example-badge">
          ${example.position} / ${example.predictedPosition}
        </span>
      </div>
      <div class="metrics-inline">
        <span class="metric-chip">Actual ${example.actualGoals}</span>
        <span class="metric-chip">Pred ${example.predictedGoals}</span>
        <span class="metric-chip">Error ${example.goalError}</span>
        <span class="metric-chip">Conf ${example.positionConfidence}%</span>
      </div>
    </article>
  `;
}

function renderFindings(findings) {
  const container = document.getElementById("findings-grid");
  container.innerHTML = findings
    .map(
      (finding) => `
        <article class="finding-card">
          <div class="card-kicker">${finding.eyebrow}</div>
          <div class="finding-value">${finding.value}</div>
          <h3>${finding.title}</h3>
          <p>${finding.body}</p>
        </article>
      `
    )
    .join("");
}

function renderAwardRadar(awardRadar) {
  document.getElementById("award-subtitle").textContent = awardRadar.subtitle;
  document.getElementById("award-methodology").textContent = awardRadar.methodology;
  document.getElementById("award-model-stack").innerHTML = `
    <span class="note-chip">${awardRadar.modelsUsed.position}</span>
    <span class="note-chip">${awardRadar.modelsUsed.goals}</span>
  `;
  document.getElementById("award-how").innerHTML = awardRadar.howItWorks
    .map((item) => `<div class="how-item">${item}</div>`)
    .join("");

  document.getElementById("award-insights").innerHTML = awardRadar.insights
    .map(
      (insight) => `
        <article class="award-insight-card">
          <div class="card-kicker">${insight.label}</div>
          <strong>${insight.value}</strong>
          <p>${insight.detail}</p>
        </article>
      `
    )
    .join("");

  document.getElementById("award-candidates").innerHTML = awardRadar.candidates
    .map(
      (candidate, index) => `
        <article class="candidate-card ${index === 0 ? "featured" : ""}">
          <div class="candidate-rank">#${index + 1}</div>
          <div class="candidate-main">
            <div>
              <h3>${candidate.player}</h3>
              <p>${candidate.squad} · ${candidate.league} · ${candidate.positionLabel}</p>
            </div>
            <div class="candidate-score">${candidate.score}</div>
          </div>
          <div class="metrics-inline">
            <span class="metric-chip">Goals ${candidate.goals}</span>
            <span class="metric-chip">Assists ${candidate.assists}</span>
            <span class="metric-chip">G+A ${candidate.goalContrib}</span>
            <span class="metric-chip">Pred goals ${candidate.predictedGoals}</span>
            <span class="metric-chip">Delta ${candidate.goalDelta}</span>
            <span class="metric-chip">Role conf ${candidate.positionConfidence}%</span>
          </div>
        </article>
      `
    )
    .join("");

  document.getElementById("award-leaders").innerHTML = awardRadar.leagueLeaders
    .map(
      (leader) => `
        <div class="leader-card">
          <span>${leader.league}</span>
          <strong>${leader.player}</strong>
          <small>${leader.squad}</small>
          <b>${leader.score}</b>
        </div>
      `
    )
    .join("");
}

function renderClassBreakdown(breakdown) {
  document.getElementById("combined-classes").innerHTML = renderClassCards(breakdown.combined);
  document.getElementById("detailed-classes").innerHTML = renderClassCards(breakdown.detailed);
}

function renderClassCards(cards) {
  return cards
    .map(
      (card) => `
        <div class="class-card">
          <div class="class-card-top">
            <div>
              <div class="class-card-name">${card.label}</div>
              <div>${card.fullLabel}</div>
            </div>
            <div class="note-chip">${card.support} rows</div>
          </div>
          <div class="class-stats">
            <span class="metric-chip">Precision ${card.precision}%</span>
            <span class="metric-chip">Recall ${card.recall}%</span>
            <span class="metric-chip">F1 ${card.f1}%</span>
          </div>
        </div>
      `
    )
    .join("");
}

function renderMatrices(matrices) {
  document.getElementById("combined-matrix").innerHTML = renderMatrix(matrices.combined);
  document.getElementById("detailed-matrix").innerHTML = renderMatrix(matrices.detailed);
}

function renderMatrix(matrixData) {
  const maxValue = Math.max(...matrixData.matrix.flat(), 1);
  const header = `
    <div class="matrix-header">
      <div></div>
      ${matrixData.labels.map((label) => `<div>${label}</div>`).join("")}
    </div>
  `;

  const rows = matrixData.matrix
    .map((row, rowIndex) => {
      const cells = row
        .map((value) => {
          const opacity = 0.12 + (value / maxValue) * 0.88;
          return `
            <div class="matrix-cell" style="background: rgba(159, 232, 112, ${opacity})">
              ${value}
            </div>
          `;
        })
        .join("");
      return `
        <div class="matrix-row">
          <div class="matrix-row-label">${matrixData.labels[rowIndex]}</div>
          ${cells}
        </div>
      `;
    })
    .join("");

  return `<div class="matrix-card">${header}${rows}</div>`;
}

function renderFeatureHighlights(features) {
  document.getElementById("classification-features").innerHTML = renderFeatureList(features.classification);
  document.getElementById("detailed-classification-features").innerHTML = renderFeatureList(features.detailedClassification);
  document.getElementById("regression-features").innerHTML = renderFeatureList(features.regression, true);
}

function renderFeatureList(rows, showDirection = false) {
  return `
    <div class="feature-list">
      ${rows
        .map(
          (row) => `
            <div class="feature-row">
              <strong>${row.label}</strong>
              <span class="feature-value ${showDirection ? row.direction : ""}">
                ${row.value}
              </span>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderPredictor(predictor) {
  document.getElementById("predictor-note").textContent = predictor.note;

  const positionForm = document.getElementById("position-form");
  const goalsForm = document.getElementById("goals-form");

  buildModelForm(positionForm, POSITION_FORM_GROUPS, predictor.schema, "classification");
  buildModelForm(goalsForm, GOALS_FORM_GROUPS, predictor.schema, "goals");

  positionForm.addEventListener("submit", handlePositionSubmit);
  goalsForm.addEventListener("submit", handleGoalsSubmit);

  updateModelDescriptions();

  const positionSelect = document.getElementById("position-model");
  const goalsSelect = document.getElementById("goals-model");
  positionSelect.addEventListener("change", updateModelDescriptions);
  goalsSelect.addEventListener("change", updateModelDescriptions);
}

function renderDatasetLab(schema) {
  const form = document.getElementById("dataset-form");
  const positionOptions = schema.models.position;
  const goalsOptions = schema.models.goals;

  form.innerHTML = `
    <section class="form-group">
      <h4>Dataset Upload</h4>
      <div class="field file-field">
        <label for="dataset-file">Choose CSV file</label>
        <input id="dataset-file" name="dataset-file" type="file" accept=".csv,text/csv" required />
      </div>
    </section>

    <section class="form-group">
      <h4>Models In Use</h4>
      <div class="field-grid">
        <div class="field">
          <label for="dataset-position-model">Position model</label>
          <select id="dataset-position-model" name="positionModel">
            ${positionOptions
              .map(
                (option) => `
                  <option value="${option.id}" ${option.id === schema.models.defaultPositionModel ? "selected" : ""}>
                    ${option.label}
                  </option>
                `
              )
              .join("")}
          </select>
        </div>
        <div class="field">
          <label for="dataset-goals-model">Goals model</label>
          <select id="dataset-goals-model" name="goalsModel">
            ${goalsOptions
              .map(
                (option) => `
                  <option value="${option.id}" ${option.id === schema.models.defaultGoalsModel ? "selected" : ""}>
                    ${option.label}
                  </option>
                `
              )
              .join("")}
          </select>
        </div>
      </div>
      <label class="sync-toggle">
        <input id="dataset-sync-position" type="checkbox" checked />
        Use the predicted position for the goals model on every row
      </label>
    </section>

    <p class="form-note">
      Recommended columns: Player, Nation, Squad, Comp, Pos, season, Age, Born, MP, Starts, Min, 90s,
      Gls, Ast, PK, PKatt, CrdY, CrdR, xG, npxG, xAG, PrgC, PrgP, PrgR. Missing fields will fall back
      to the training defaults.
    </p>
    <button class="submit-button" type="submit">Run Dataset Predictions</button>
  `;

  attachDatasetFormHandler();
}

function buildModelForm(form, groups, schema, mode) {
  const prefix = mode === "classification" ? "position" : "goals";
  const modelKey = mode === "classification" ? "position" : "goals";
  const modelOptions = schema.models[modelKey];
  const defaultModel = mode === "classification" ? schema.models.defaultPositionModel : schema.models.defaultGoalsModel;
  const formBody = groups
    .map(
      (group) => `
        <section class="form-group">
          <h4>${group.title}</h4>
          <div class="field-grid">
            ${group.fields.map((field) => renderField(field, schema, prefix)).join("")}
          </div>
        </section>
      `
    )
    .join("");

  const note = mode === "classification" ? schema.inputNotes.classification : schema.inputNotes.goals;
  const buttonLabel = mode === "classification" ? "Predict Position" : "Predict Goals";

  form.innerHTML = `
    <section class="form-group">
      <h4>Model In Use</h4>
      <div class="field">
        <label for="${prefix}-model">Choose model</label>
        <select name="model" id="${prefix}-model">
          ${modelOptions
            .map(
              (option) => `
                <option value="${option.id}" ${option.id === defaultModel ? "selected" : ""}>
                  ${option.label}
                </option>
              `
            )
            .join("")}
        </select>
      </div>
      <p id="${prefix}-model-description" class="form-note"></p>
    </section>
    ${formBody}
    <p class="form-note">${note}</p>
    <button class="submit-button" type="submit">${buttonLabel}</button>
  `;
}

function renderField(field, schema, prefix) {
  const label = FIELD_LABELS[field] || field;
  const options = schema.options[field];
  const defaultValue = schema.defaults[field] ?? "";
  const inputType = typeof defaultValue === "number" ? "number" : "text";
  const inputId = `${prefix}-${field}`;

  if (field === "season" || field === "Pos") {
    const selectOptions = options
      .map(
        (option) => `
          <option value="${option}" ${String(defaultValue) === String(option) ? "selected" : ""}>
            ${option}
          </option>
        `
      )
      .join("");
    return `
      <div class="field">
        <label for="${inputId}">${label}</label>
        <select name="${field}" id="${inputId}">
          ${selectOptions}
        </select>
      </div>
    `;
  }

  if (Array.isArray(options)) {
    const datalistId = `${prefix}-${field}-options`;
    return `
      <div class="field">
        <label for="${inputId}">${label}</label>
        <input
          id="${inputId}"
          name="${field}"
          type="text"
          list="${datalistId}"
          value="${defaultValue}"
          autocomplete="off"
        />
        <datalist id="${datalistId}">
          ${options.map((option) => `<option value="${option}"></option>`).join("")}
        </datalist>
      </div>
    `;
  }

  return `
    <div class="field">
      <label for="${inputId}">${label}</label>
      <input
        id="${inputId}"
        name="${field}"
        type="${inputType}"
        inputmode="decimal"
        step="any"
        value="${defaultValue}"
      />
    </div>
  `;
}

async function handlePositionSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  setButtonBusy(button, true, "Running...");

  try {
    const payload = formToObject(form);
    const result = await fetchJSON("/api/predict/position", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.lastPositionPrediction = result.prediction;
    if (document.getElementById("sync-position-toggle").checked) {
      const goalsPos = document.querySelector("#goals-form [name='Pos']");
      if (goalsPos) {
        goalsPos.value = result.prediction;
      }
    }
    renderPositionResult(result);
  } catch (error) {
    renderErrorPanel("position-result", error.message);
  } finally {
    setButtonBusy(button, false, "Predict Position");
  }
}

async function handleGoalsSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  setButtonBusy(button, true, "Running...");

  try {
    const payload = formToObject(form);
    if (document.getElementById("sync-position-toggle").checked && state.lastPositionPrediction) {
      payload.Pos = state.lastPositionPrediction;
      const goalsPos = form.querySelector("[name='Pos']");
      if (goalsPos) {
        goalsPos.value = state.lastPositionPrediction;
      }
    }

    const result = await fetchJSON("/api/predict/goals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderGoalsResult(result);
  } catch (error) {
    renderErrorPanel("goals-result", error.message);
  } finally {
    setButtonBusy(button, false, "Predict Goals");
  }
}

async function handleDatasetSubmit(event) {
  event.preventDefault();
  console.log("Dataset form submitted");
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  const fileInput = document.getElementById("dataset-file");
  const file = fileInput?.files?.[0];

  if (!file) {
    console.warn("No file selected");
    renderDatasetError("Choose a CSV file first.");
    return;
  }

  console.log("Processing file:", file.name);
  setButtonBusy(button, true, "Running...");

  try {
    const csvText = await file.text();
    console.log("CSV loaded, sending to /api/predict/dataset");
    const payload = {
      filename: file.name,
      csvText,
      positionModel: document.getElementById("dataset-position-model").value,
      goalsModel: document.getElementById("dataset-goals-model").value,
      usePredictedPosition: document.getElementById("dataset-sync-position").checked,
    };

    const result = await fetchJSON("/api/predict/dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    state.lastDatasetResult = result;
    renderDatasetBatchResult(result);
  } catch (error) {
    console.error("Dataset submission error:", error);
    renderDatasetError(error.message);
  } finally {
    setButtonBusy(button, false, "Run Dataset Predictions");
  }
}

function renderPositionResult(result) {
  document.getElementById("position-result").classList.remove("empty-state");
  document.getElementById("position-result").innerHTML = `
    <div class="card-kicker">Predicted Position</div>
    <p class="result-headline">${result.predictionLabel}</p>
    <div class="note-chip">Model: ${result.model}</div>
    <p class="result-copy">${result.modelDescription}</p>
    <div class="metrics-inline">
      <span class="metric-chip">Top probability ${result.topProbabilityPct}%</span>
      <span class="metric-chip">Code ${result.prediction}</span>
    </div>
    <div class="probability-stack">
      ${result.probabilities
        .map(
          (item, index) => `
            <div class="probability-row">
              <div class="probability-label">${item.label}</div>
              <div class="bar-track">
                <div class="bar-fill ${index === 0 ? "" : "muted"}" style="width: ${item.probabilityPct}%"></div>
              </div>
              <div class="bar-value">${item.probabilityPct}%</div>
            </div>
          `
        )
        .join("")}
    </div>
    <div class="derived-grid">
      ${renderDerivedItem("G+A", result.derived.goalsPlusAssists)}
      ${renderDerivedItem("G-PK", result.derived.nonPenaltyGoals)}
      ${renderDerivedItem("npxG+xAG", result.derived.npxGPlusxAG)}
      ${renderDerivedItem("Goals /90", result.derived.goalsPer90)}
      ${renderDerivedItem("xG /90", result.derived.xGPer90)}
      ${renderDerivedItem("Start rate", result.derived.startRate)}
    </div>
    ${renderWarnings(result.warnings)}
  `;
}

function renderGoalsResult(result) {
  document.getElementById("goals-result").classList.remove("empty-state");
  document.getElementById("goals-result").innerHTML = `
    <div class="card-kicker">Predicted Goals</div>
    <p class="result-headline">${result.predictedGoals}</p>
    <div class="note-chip">Model: ${result.model}</div>
    <p class="result-copy">${result.modelDescription}</p>
    <div class="metrics-inline">
      <span class="metric-chip">Training percentile ${result.percentile}%</span>
      <span class="metric-chip">Raw model output ${result.rawPrediction}</span>
    </div>
    <div class="derived-grid">
      ${renderDerivedItem("xG /90", result.derived.xGPer90)}
      ${renderDerivedItem("xAG /90", result.derived.xAGPer90)}
      ${renderDerivedItem("Prog passes /90", result.derived.progressivePassesPer90)}
      ${renderDerivedItem("Start rate", result.derived.startRate)}
    </div>
    ${renderWarnings(result.warnings)}
  `;
}

function renderDatasetBatchResult(result) {
  const summary = document.getElementById("dataset-summary");
  summary.classList.remove("empty-state");
  summary.innerHTML = `
    <div class="card-kicker">${result.filename}</div>
    <p class="result-headline">${result.rowsProcessed}</p>
    <p class="result-copy">
      Processed rows with ${result.positionModel} for classification and ${result.goalsModel} for regression.
    </p>
    <div class="metrics-inline">
      <span class="metric-chip">Avg predicted goals ${result.summary.averagePredictedGoals}</span>
      <span class="metric-chip">Peak predicted goals ${result.summary.maxPredictedGoals}</span>
      <span class="metric-chip">Mean role confidence ${result.summary.meanPositionConfidence}%</span>
      <span class="metric-chip">${result.usePredictedPosition ? "Goals model uses predicted positions" : "Goals model uses dataset positions"}</span>
    </div>
    <div class="dataset-breakdown">
      ${result.summary.positionBreakdown
        .map((item) => `<span class="note-chip">${item.label} ${item.count}</span>`)
        .join("")}
    </div>
    <div class="metrics-inline">
      <span class="metric-chip">Detected columns ${result.columnsDetected}</span>
      <span class="metric-chip">Defaulted columns ${result.filledColumns.length}</span>
      <span class="metric-chip">Low-minute rows ${result.lowMinuteRows}</span>
    </div>
    ${renderWarnings(result.warnings)}
  `;

  const top10 = document.getElementById("dataset-top10");
  top10.innerHTML = result.ballonCandidates
    .map(
      (player, index) => `
        <article class="dataset-player-card ${index === 0 ? "featured" : ""}">
          <div class="dataset-player-head">
            <div>
              <h4>${player.player}</h4>
              <p>${player.squad} / ${player.league} / ${player.predictedPositionLabel}</p>
            </div>
            <div class="dataset-rank">#${player.rank}</div>
          </div>
          <div class="metrics-inline">
            <span class="metric-chip">Ballon score ${player.ballonScore}</span>
            <span class="metric-chip">Pred goals ${player.predictedGoals}</span>
            <span class="metric-chip">Pos conf ${player.positionConfidencePct}%</span>
            <span class="metric-chip">Actual goals ${player.actualGoals}</span>
          </div>
        </article>
      `
    )
    .join("");

  renderLeagueTop5(result.topLeaguePlayers);

}

function renderDatasetError(message) {
  renderErrorPanel("dataset-summary", message);
  document.getElementById("dataset-top10").innerHTML = `
    <div class="warning-stack">
      <div class="warning-item">${message}</div>
    </div>
  `;
  document.getElementById("dataset-league-top5").innerHTML = `
    <div class="warning-stack">
      <div class="warning-item">${message}</div>
    </div>
  `;
  document.getElementById("dataset-preview").classList.remove("empty-state");
  document.getElementById("dataset-preview").innerHTML = `
    <div class="warning-stack">
      <div class="warning-item">${message}</div>
    </div>
  `;
}

function renderDerivedItem(label, value) {
  return `
    <div class="derived-item">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `;
}

function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    return "";
  }
  return `
    <div class="warning-stack">
      ${warnings.map((warning) => `<div class="warning-item">${warning}</div>`).join("")}
    </div>
  `;
}

function renderLeagueTop5(leagues) {
  const leagueGrid = document.getElementById("dataset-league-top5");
  leagueGrid.innerHTML = leagues
    .map(
      (league) => `
        <section class="league-card">
          <div class="league-title">${league.league}</div>
          <div class="league-player-list">
            ${league.players
              .map(
                (player) => `
                  <div class="league-player-row">
                    <strong>#${player.rank} ${player.player}</strong>
                    <span>${player.squad}</span>
                    <span>${player.predictedPositionLabel}</span>
                    <span>Goals ${player.predictedGoals}</span>
                    <span>Conf ${player.positionConfidencePct}%</span>
                  </div>
                `
              )
              .join("")}
          </div>
        </section>
      `
    )
    .join("");
}

function renderErrorPanel(targetId, message) {
  const target = document.getElementById(targetId);
  target.classList.remove("empty-state");
  target.innerHTML = `
    <div class="warning-stack">
      <div class="warning-item">${message}</div>
    </div>
  `;
}

function updateModelDescriptions() {
  const schema = state.dashboard.predictor.schema;
  const positionModelId = document.getElementById("position-model").value;
  const goalsModelId = document.getElementById("goals-model").value;
  const positionOption = schema.models.position.find((option) => option.id === positionModelId);
  const goalsOption = schema.models.goals.find((option) => option.id === goalsModelId);

  document.getElementById("position-model-description").textContent = `${positionOption.dataset}: ${positionOption.description}`;
  document.getElementById("goals-model-description").textContent = `${goalsOption.dataset}: ${goalsOption.description}`;
}

function formToObject(form) {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function setButtonBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = label;
}

function groupBy(items, key) {
  return items.reduce((accumulator, item) => {
    const value = item[key];
    if (!accumulator[value]) {
      accumulator[value] = [];
    }
    accumulator[value].push(item);
    return accumulator;
  }, {});
}

// ========== PLAYER SIMILARITY SYSTEM ==========

function initializeSimilaritySystem() {
  const searchBtn = document.getElementById("similarity-search-btn");
  const playerInput = document.getElementById("similarity-player-input");
  const topKSelect = document.getElementById("similarity-top-k");

  if (!searchBtn || !playerInput) return;

  let autocompleteCache = [];

  // Autocomplete functionality
  playerInput.addEventListener("input", debounce(async (event) => {
    const query = event.target.value.trim();
    if (query.length < 2) {
      document.getElementById("similarity-autocomplete").innerHTML = "";
      return;
    }

    try {
      const result = await fetchJSON(`/api/player-search?q=${encodeURIComponent(query)}&limit=8`);
      renderAutocomplete(result.results);
      autocompleteCache = result.results;
    } catch (error) {
      console.error("Autocomplete error:", error);
    }
  }, 300));

  // Search button handler
  searchBtn.addEventListener("click", async () => {
    const playerName = playerInput.value.trim();
    if (!playerName) {
      document.getElementById("similarity-status").textContent = "Please enter a player name.";
      return;
    }

    setButtonBusy(searchBtn, true, "Searching...");
    document.getElementById("similarity-status").textContent = "";

    try {
      const topK = parseInt(topKSelect.value, 10) || 5;
      const metric = document.getElementById('similarity-metric')?.value || 'cosine';
      const result = await fetchJSON(`/api/player-similarity?player=${encodeURIComponent(playerName)}&top_k=${topK}&metric=${metric}`);
      
      if (result.error) {
        document.getElementById("similarity-status").textContent = result.error;
        if (result.searchResults && result.searchResults.length > 0) {
          renderAutocompleteResults(result.searchResults);
        }
        renderSimilarResults([]);
        renderSubjectProfile(null);
      } else {
        renderSubjectProfile(result.subject);
        renderSimilarResults(result.similarPlayers);
        document.getElementById("similarity-status").textContent = `Found ${result.similarPlayers.length} similar players.`;
      }
    } catch (error) {
      document.getElementById("similarity-status").textContent = `Error: ${error.message}`;
    } finally {
      setButtonBusy(searchBtn, false, "Find Similar Players");
    }
  });

  // Allow Enter key to trigger search
  playerInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      searchBtn.click();
    }
  });
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function renderAutocomplete(results) {
  const dropdown = document.getElementById("similarity-autocomplete");
  if (!results || results.length === 0) {
    dropdown.innerHTML = "";
    return;
  }

  dropdown.innerHTML = results
    .map(
      (player) => `
        <div class="autocomplete-item" onclick="selectAutocompletedPlayer('${player.name.replace(/'/g, "\\'")}')">
          <strong>${player.name}</strong>
          <span>${player.squad}</span>
          <small>${player.position} • ${player.matches_90s}×90</small>
        </div>
      `
    )
    .join("");
}

function renderAutocompleteResults(results) {
  const dropdown = document.getElementById("similarity-autocomplete");
  dropdown.innerHTML = `
    <div class="autocomplete-header">Did you mean:</div>
    ${results
      .map(
        (player) => `
          <div class="autocomplete-item" onclick="selectAutocompletedPlayer('${player.name.replace(/'/g, "\\'")}')">
            <strong>${player.name}</strong>
            <span>${player.squad}</span>
            <small>${player.position}</small>
          </div>
        `
      )
      .join("")}
  `;
}

function selectAutocompletedPlayer(playerName) {
  document.getElementById("similarity-player-input").value = playerName;
  document.getElementById("similarity-autocomplete").innerHTML = "";
  document.getElementById("similarity-search-btn").click();
}

function renderSubjectProfile(profile) {
  const container = document.getElementById("similarity-subject-profile");
  
  if (!profile) {
    container.classList.add("empty-state");
    container.innerHTML = "Select a player to see their profile.";
    return;
  }

  container.classList.remove("empty-state");
  container.innerHTML = `
    <div class="profile-header">
      <h3>${profile.name}</h3>
      <div class="profile-meta">
        <span class="position-badge">${profile.position}</span>
        <span>${profile.squad}</span>
        <span>${profile.nation}</span>
      </div>
    </div>
    <div class="profile-stats">
      <div class="stat-item">
        <span>Age</span>
        <strong>${profile.age || "—"}</strong>
      </div>
      <div class="stat-item">
        <span>Matches (90s)</span>
        <strong>${profile.matches_played} (${profile.matches_90s})</strong>
      </div>
      <div class="stat-item">
        <span>Goals</span>
        <strong>${profile.goals}</strong>
      </div>
      <div class="stat-item">
        <span>Assists</span>
        <strong>${profile.assists}</strong>
      </div>
      <div class="stat-item">
        <span>xG</span>
        <strong>${profile.xg !== null ? profile.xg : "—"}</strong>
      </div>
      <div class="stat-item">
        <span>xA</span>
        <strong>${profile.xa !== null ? profile.xa : "—"}</strong>
      </div>
    </div>
    <div class="profile-dataset">
      <small>Data from: <strong>${profile.dataset}</strong></small>
    </div>
  `;
}

function renderSimilarResults(players) {
  const container = document.getElementById("similarity-results");
  
  if (!players || players.length === 0) {
    container.innerHTML = '<div class="empty-state-message">No similar players found. Try another search.</div>';
    return;
  }

  container.innerHTML = players
    .map(
      (player, index) => `
        <article class="similar-player-card ${index === 0 ? "top-match" : ""}">
          <div class="card-header">
            <div class="rank-badge">#${player.rank}</div>
            <div class="similarity-score">${player.similarity}%</div>
          </div>
          <div class="player-info">
            <h4>${player.name}</h4>
            <p>${player.squad}</p>
            <div class="position-badge-small">${player.position}</div>
          </div>
          <div class="player-stats">
            <span class="stat-chip">Age ${player.age || "—"}</span>
            <span class="stat-chip">Goals ${player.goals}</span>
            <span class="stat-chip">Assists ${player.assists}</span>
            <span class="stat-chip">90s ${player.matches_90s}</span>
            <span class="stat-chip">xG ${player.xg !== null ? player.xg : "—"}</span>
          </div>
          <div class="footer-info">
            <small>${player.nation}</small>
          </div>
        </article>
      `
    )
    .join("");
}

// ============================================================
// GENERIC AUTOCOMPLETE HELPER
// ============================================================
function setupAutocomplete(inputId, dropdownId, onSelect) {
  const input = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);
  if (!input || !dropdown) return;

  input.addEventListener('input', debounce(async (e) => {
    const q = e.target.value.trim();
    if (q.length < 2) { dropdown.innerHTML = ''; return; }
    try {
      const res = await fetchJSON(`/api/player-search?q=${encodeURIComponent(q)}&limit=8`);
      dropdown.innerHTML = (res.results || []).map(p =>
        `<div class="autocomplete-item" data-name="${escHtml(p.name)}">
          <strong>${escHtml(p.name)}</strong>
          <span>${escHtml(p.squad)}</span>
          <small>${p.position} · ${p.matches_90s}×90</small>
        </div>`
      ).join('');
      dropdown.querySelectorAll('.autocomplete-item').forEach(el => {
        el.addEventListener('click', () => {
          input.value = el.dataset.name;
          dropdown.innerHTML = '';
          onSelect(el.dataset.name);
        });
      });
    } catch {}
  }, 300));

  input.addEventListener('keypress', e => { if (e.key === 'Enter') { dropdown.innerHTML = ''; onSelect(input.value.trim()); } });
  document.addEventListener('click', e => { if (!input.contains(e.target) && !dropdown.contains(e.target)) dropdown.innerHTML = ''; });
}

function escHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ============================================================
// RECOMMENDATIONS, COMPARE, SCOUT REPORT, ROLE DETECTION
// ============================================================
function initRecommendationSystem() {
  const btn = document.getElementById('rec-run-btn');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    setButtonBusy(btn, true, 'Searching...');
    const params = new URLSearchParams();
    const pos = document.getElementById('rec-position').value;
    const maxAge = document.getElementById('rec-max-age').value;
    const min90s = document.getElementById('rec-min-90s').value || '8';
    if (pos) params.set('position', pos);
    if (maxAge) params.set('max_age', maxAge);
    params.set('min_90s', min90s);
    params.set('top_k', '10');
    try {
      const res = await fetchJSON(`/api/player-recommendations?${params.toString()}`);
      renderRecommendations(res.recommendations || []);
      document.getElementById('rec-status').textContent = `Found ${(res.recommendations || []).length} recommendations.`;
    } catch (err) {
      document.getElementById('rec-status').textContent = `Error: ${err.message}`;
    } finally {
      setButtonBusy(btn, false, 'Find Recommendations');
    }
  });
}

function renderRecommendations(players) {
  const el = document.getElementById('rec-results');
  el.classList.remove('empty-state');
  if (!players.length) {
    el.innerHTML = '<div class="empty-state">No players matched those filters.</div>';
    return;
  }
  el.innerHTML = players.map(p => `
    <div class="tool-row">
      <div><strong>#${p.rank} ${escHtml(p.name)}</strong><span>${escHtml(p.squad)} · ${p.position} · Age ${p.age ?? '—'}</span></div>
      <b>${p.score}</b>
      <small>${escHtml(p.rationale || '')}</small>
    </div>`).join('');
}

function initPlayerToolsSystem() {
  setupAutocomplete('cmp-player-a', 'cmp-autocomplete-a', () => {});
  setupAutocomplete('cmp-player-b', 'cmp-autocomplete-b', () => {});
  setupAutocomplete('scout-player', 'scout-autocomplete', () => {});
  setupAutocomplete('role-player', 'role-autocomplete', () => {});

  document.getElementById('cmp-run-btn')?.addEventListener('click', async () => {
    const a = document.getElementById('cmp-player-a').value.trim();
    const b = document.getElementById('cmp-player-b').value.trim();
    if (!a || !b) return;
    const res = await fetchJSON(`/api/player-compare?player_a=${encodeURIComponent(a)}&player_b=${encodeURIComponent(b)}`);
    renderCompareTool(res);
  });

  document.getElementById('scout-run-btn')?.addEventListener('click', async () => {
    const name = document.getElementById('scout-player').value.trim();
    if (!name) return;
    const res = await fetchJSON(`/api/scout-report?player=${encodeURIComponent(name)}`);
    renderScoutReport(res);
  });

  document.getElementById('role-run-btn')?.addEventListener('click', async () => {
    const name = document.getElementById('role-player').value.trim();
    if (!name) return;
    const res = await fetchJSON(`/api/player-role?player=${encodeURIComponent(name)}`);
    renderRoleDetection(res);
  });
}

function renderCompareTool(data) {
  const el = document.getElementById('cmp-results');
  el.classList.remove('empty-state');
  if (data.error) { el.innerHTML = `<div class="empty-state">${escHtml(data.error)}</div>`; return; }
  el.innerHTML = `
    <div class="tool-summary"><strong>${escHtml(data.playerA.name)}</strong> vs <strong>${escHtml(data.playerB.name)}</strong></div>
    ${(data.metrics || []).map(m => `
      <div class="mini-compare-row">
        <span>${escHtml(m.metric)}</span>
        <b>${m.playerA}</b>
        <b>${m.playerB}</b>
        <small>${escHtml(m.leader)} +${m.delta}</small>
      </div>`).join('')}`;
}

function renderScoutReport(data) {
  const el = document.getElementById('scout-results');
  el.classList.remove('empty-state');
  if (data.error) { el.innerHTML = `<div class="empty-state">${escHtml(data.error)}</div>`; return; }
  const strengths = (data.strengths || []).map(s => escHtml(s.label || s.stat || s.feature)).join(', ');
  const similar = (data.similarPlayers || []).map(p => escHtml(p.name)).join(', ');
  el.innerHTML = `
    <div class="tool-summary"><strong>${escHtml(data.profile.name)}</strong><span>${escHtml(data.profile.squad)} · ${data.profile.position}</span></div>
    <p>${escHtml(data.summary)}</p>
    <div class="tool-chip-list">
      <span class="metric-chip">Role: ${escHtml(data.role.primaryRole || '—')}</span>
      <span class="metric-chip">Potential: ${data.potential.potentialScore ?? '—'}</span>
    </div>
    <p><strong>Strengths:</strong> ${strengths || '—'}</p>
    <p><strong>Similar players:</strong> ${similar || '—'}</p>
    <small>${escHtml(data.caution || '')}</small>`;
}

function renderRoleDetection(data) {
  const el = document.getElementById('role-results');
  el.classList.remove('empty-state');
  if (data.error) { el.innerHTML = `<div class="empty-state">${escHtml(data.error)}</div>`; return; }
  el.innerHTML = `
    <div class="tool-summary"><strong>${escHtml(data.player)}</strong><span>${escHtml(data.squad)} · ${data.position}</span></div>
    <div class="role-primary">${escHtml(data.primaryRole)} <span>${data.confidence}%</span></div>
    ${(data.roleScores || []).map(r => `
      <div class="shap-bar-row">
        <span class="shap-label">${escHtml(r.role)}</span>
        <div class="shap-track"><div class="shap-fill positive" style="width:${r.score}%"></div></div>
        <span class="shap-value positive">${r.score}</span>
      </div>`).join('')}
    <p class="shap-method-note">${escHtml(data.explanation || '')}</p>`;
}

// ============================================================
// FEATURE 4 — RADAR CHART
// ============================================================
const RADAR_COLORS = ['#9fe870', '#f43f5e'];

function initRadarSystem() {
  const btn = document.getElementById('radar-search-btn');
  if (!btn) return;

  setupAutocomplete('radar-player1', 'radar-autocomplete1', () => {});
  setupAutocomplete('radar-player2', 'radar-autocomplete2', () => {});

  btn.addEventListener('click', async () => {
    const p1 = document.getElementById('radar-player1').value.trim();
    const p2 = document.getElementById('radar-player2').value.trim();
    if (!p1) { document.getElementById('radar-status').textContent = 'Enter at least Player 1.'; return; }

    setButtonBusy(btn, true, 'Loading...');
    document.getElementById('radar-status').textContent = '';
    try {
      const params = `player1=${encodeURIComponent(p1)}${p2 ? '&player2=' + encodeURIComponent(p2) : ''}`;
      const res = await fetchJSON(`/api/player-radar?${params}`);
      if (res.error) { document.getElementById('radar-status').textContent = res.error; return; }
      renderRadarChart(res);
    } catch (err) {
      document.getElementById('radar-status').textContent = `Error: ${err.message}`;
    } finally {
      setButtonBusy(btn, false, 'Generate Radar');
    }
  });
}

function renderRadarChart(data) {
  const container = document.getElementById('radar-container');
  const legendEl = document.getElementById('radar-legend');
  const metrics = data.metrics;
  const players = data.players;
  const N = metrics.length;
  const cx = 160, cy = 160, R = 130;
  const angleStep = (2 * Math.PI) / N;

  function polar(index, radius) {
    const angle = angleStep * index - Math.PI / 2;
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  }

  // Grid rings
  let gridLines = '';
  [0.25, 0.5, 0.75, 1].forEach(frac => {
    const pts = metrics.map((_, i) => polar(i, R * frac));
    gridLines += `<polygon points="${pts.map(p => `${p.x},${p.y}`).join(' ')}" fill="none" stroke="rgba(14,15,12,0.1)" stroke-width="1"/>`;
  });

  // Axis lines + labels
  let axes = '';
  metrics.forEach((m, i) => {
    const outer = polar(i, R);
    axes += `<line x1="${cx}" y1="${cy}" x2="${outer.x}" y2="${outer.y}" stroke="rgba(14,15,12,0.15)" stroke-width="1"/>`;
    const lp = polar(i, R + 22);
    const anchor = lp.x < cx - 5 ? 'end' : lp.x > cx + 5 ? 'start' : 'middle';
    axes += `<text x="${lp.x}" y="${lp.y}" text-anchor="${anchor}" dominant-baseline="middle" font-size="10" font-weight="600" fill="#4f554d">${m}</text>`;
  });

  // Player polygons
  let polys = '';
  players.forEach((p, pi) => {
    const color = RADAR_COLORS[pi % RADAR_COLORS.length];
    const pts = p.values.map((v, i) => polar(i, R * (v / 100)));
    const pointStr = pts.map(pt => `${pt.x},${pt.y}`).join(' ');
    polys += `<polygon points="${pointStr}" fill="${color}" fill-opacity="0.18" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
    pts.forEach(pt => {
      polys += `<circle cx="${pt.x}" cy="${pt.y}" r="4" fill="${color}" stroke="#fff" stroke-width="1.5"/>`;
    });
  });

  container.innerHTML = `<svg width="320" height="320" viewBox="0 0 320 320">
    <g>${gridLines}</g>
    <g>${axes}</g>
    <g>${polys}</g>
  </svg>`;

  legendEl.innerHTML = players.map((p, pi) => `
    <div class="radar-legend-item">
      <div class="radar-legend-dot" style="background:${RADAR_COLORS[pi % RADAR_COLORS.length]}"></div>
      <span>${escHtml(p.name)}</span>
    </div>`).join('');
}

// ============================================================
// FEATURE 5, 6, 8 — PLAYER INTELLIGENCE
// ============================================================
function initPlayerIntelligenceSystem() {
  const btn = document.getElementById('intel-search-btn');
  if (!btn) return;
  setupAutocomplete('intel-player-input', 'intel-autocomplete', () => {});

  btn.addEventListener('click', async () => {
    const name = document.getElementById('intel-player-input').value.trim();
    if (!name) { document.getElementById('intel-status').textContent = 'Enter a player name.'; return; }

    setButtonBusy(btn, true, 'Analysing...');
    document.getElementById('intel-status').textContent = '';
    try {
      const [swRes, potRes, shapRes] = await Promise.all([
        fetchJSON(`/api/player-analysis?player=${encodeURIComponent(name)}`),
        fetchJSON(`/api/player-potential?player=${encodeURIComponent(name)}`),
        fetchJSON(`/api/player-explain?player=${encodeURIComponent(name)}`),
      ]);
      if (swRes.error) { document.getElementById('intel-status').textContent = swRes.error; return; }
      renderStrengthsWeaknesses(swRes);
      renderPotentialScore(potRes);
      renderShapPanel(shapRes);
    } catch (err) {
      document.getElementById('intel-status').textContent = `Error: ${err.message}`;
    } finally {
      setButtonBusy(btn, false, 'Analyse Player');
    }
  });
}

function renderStrengthsWeaknesses(data) {
  const panel = document.getElementById('sw-panel');
  panel.classList.remove('empty-state');
  const maxPct = 100;

  function statRows(list, cls) {
    return list.map(s => `
      <div class="sw-stat-row">
        <span class="sw-stat-label">${escHtml(s.label || s.stat || s.feature || 'Metric')}</span>
        <div class="sw-bar-track"><div class="sw-bar-fill ${cls}" style="width:${Math.min(s.percentile, maxPct)}%"></div></div>
        <span class="sw-pct">${s.percentile}p</span>
      </div>`).join('');
  }

  panel.innerHTML = `
    <div class="sw-section">
      <div class="sw-section-title strength-title">🌟 Strengths (top percentiles)</div>
      ${statRows(data.strengths || [], 'strength')}
    </div>
    <div class="sw-section">
      <div class="sw-section-title weakness-title">📉 Weaknesses (low percentiles)</div>
      ${statRows(data.weaknesses || [], 'weakness')}
    </div>
    <p class="sw-peer-note">Percentiles vs all players in the same position (${data.position || '—'}).</p>`;
}

function renderPotentialScore(data) {
  const panel = document.getElementById('potential-panel');
  if (data.error) { panel.innerHTML = `<div class="empty-state">${data.error}</div>`; return; }
  panel.classList.remove('empty-state');

  const score = data.potential_score ?? 0;
  const maxScore = 100;
  const R = 50, circ = 2 * Math.PI * R;
  const dashOffset = circ * (1 - score / maxScore);

  const tierColors = { Elite: '#f43f5e', WorldClass: '#f59e0b', Good: '#6366f1', Average: '#10b981', Developing: '#64748b' };
  const tierKey = (data.tier || '').replace(' ', '');
  const color = tierColors[tierKey] || '#9fe870';

  panel.innerHTML = `
    <div class="potential-ring-wrap">
      <svg class="potential-ring-svg" width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="${R}" fill="none" stroke="rgba(14,15,12,0.1)" stroke-width="10"/>
        <circle cx="60" cy="60" r="${R}" fill="none" stroke="${color}" stroke-width="10"
          stroke-dasharray="${circ}" stroke-dashoffset="${dashOffset}" stroke-linecap="round"
          style="transition:stroke-dashoffset 1s ease"/>
      </svg>
      <div class="potential-ring-label">
        <div class="potential-score-value">${score}</div>
        <span class="potential-tier-badge" style="background:${color}">${data.tier || '—'}</span>
      </div>
    </div>
    <div class="potential-breakdown">
      <div class="potential-breakdown-item"><span>Age</span><strong>${data.age ?? '—'}</strong></div>
      <div class="potential-breakdown-item"><span>Position</span><strong>${data.position ?? '—'}</strong></div>
      <div class="potential-breakdown-item"><span>G+A / 90</span><strong>${data.contributions_per90 ?? '—'}</strong></div>
      <div class="potential-breakdown-item"><span>xG+xA / 90</span><strong>${data.xg_per90 ?? '—'}</strong></div>
    </div>
    <p class="sw-peer-note">${data.rationale || ''}</p>`;
}

function renderShapPanel(data) {
  const panel = document.getElementById('shap-panel');
  if (data.error) { panel.innerHTML = `<div class="empty-state">${data.error}</div>`; return; }
  panel.classList.remove('empty-state');

  const features = data.shap_values || [];
  const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_value)), 0.001);

  const rows = features.map(f => {
    const pct = Math.min(Math.abs(f.shap_value) / maxAbs * 100, 100);
    const dir = f.shap_value >= 0 ? 'positive' : 'negative';
    return `
      <div class="shap-bar-row">
        <span class="shap-label">${escHtml(f.feature)}</span>
        <div class="shap-track"><div class="shap-fill ${dir}" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="shap-value ${dir}">${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(3)}</span>
      </div>`;
  }).join('');

  panel.innerHTML = `
    <div class="shap-prediction-badge">
      <strong>Predicted: ${escHtml(data.predicted_position || '—')}</strong>
      <span class="metric-chip">conf ${data.confidence || '—'}%</span>
    </div>
    ${rows}
    <p class="shap-method-note">${data.method_note || 'Approximated via permutation importance.'}</p>`;
}

// ============================================================
// FEATURE 2 — CLUSTERING
// ============================================================
const CLUSTER_META = {
  Finisher:    { icon: '⚽', color: '#f43f5e', desc: 'Goal-hungry forwards who excel at finishing and shots on target.' },
  Playmaker:   { icon: '🎯', color: '#6366f1', desc: 'Creative midfielders who drive progressive passes and assists.' },
  Defensive:   { icon: '🛡️', color: '#0ea5e9', desc: 'Disciplined defenders focused on interceptions and tackles.' },
  'Box-to-Box': { icon: '⚡', color: '#f59e0b', desc: 'Complete midfielders who contribute in attack and defence.' },
};

async function initClusteringSystem() {
  const statusEl = document.getElementById('cluster-status');
  const summaryEl = document.getElementById('cluster-summary');
  const playersEl = document.getElementById('cluster-players');
  const filterEl = document.getElementById('cluster-filter');
  if (!statusEl) return;

  statusEl.textContent = 'Loading cluster data…';
  try {
    const algorithm = document.getElementById('cluster-algorithm')?.value || 'kmeans';
    const res = await fetchJSON(`/api/player-clusters?algorithm=${encodeURIComponent(algorithm)}`);
    if (res.error) { statusEl.textContent = res.error; return; }
    statusEl.textContent = `${res.total_players} players assigned to ${res.n_clusters} clusters.`;

    // Summary cards
    summaryEl.innerHTML = (res.cluster_summary || []).map(c => {
      const meta = CLUSTER_META[c.name] || { icon: '👤', color: '#9fe870', desc: '' };
      return `
        <div class="cluster-card">
          <div class="cluster-icon">${meta.icon}</div>
          <div class="cluster-name" style="color:${meta.color}">${escHtml(c.name)}</div>
          <div class="cluster-desc">${meta.desc}</div>
          <div class="cluster-stats">
            <div class="cluster-stat"><strong>${c.count}</strong> players</div>
            <div class="cluster-stat"><strong>${c.avg_goals?.toFixed(1) ?? '—'}</strong> avg Gls</div>
          </div>
        </div>`;
    }).join('');

    // Player chips (store globally for filtering)
    window._clusterPlayers = res.players || [];
    renderClusterPlayers(window._clusterPlayers);

    filterEl?.addEventListener('change', () => {
      const val = filterEl.value;
      const filtered = val === 'all' ? window._clusterPlayers : window._clusterPlayers.filter(p => p.cluster === val);
      renderClusterPlayers(filtered);
    });
    document.getElementById('cluster-algorithm')?.addEventListener('change', initClusteringSystem, { once: true });
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
}

function renderClusterPlayers(players) {
  const el = document.getElementById('cluster-players');
  if (!el) return;
  el.innerHTML = (players || []).slice(0, 200).map(p => {
    const meta = CLUSTER_META[p.cluster] || { color: '#9fe870' };
    return `
      <div class="cluster-player-chip" style="--chip-color:${meta.color}">
        <strong>${escHtml(p.name)}</strong>
        <span>${escHtml(p.squad || '—')}</span>
        <span class="cluster-badge" style="background:${meta.color}">${escHtml(p.cluster)}</span>
      </div>`;
  }).join('');
}

// ============================================================
// FEATURE 3 — MARKET VALUE
// ============================================================
function initMarketValueSystem() {
  const form = document.getElementById('mv-form');
  if (!form) return;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = form.querySelector('button[type="submit"]');
    setButtonBusy(btn, true, 'Estimating…');
    try {
      const payload = {
        Gls:  parseFloat(document.getElementById('mv-gls').value) || 0,
        Ast:  parseFloat(document.getElementById('mv-ast').value) || 0,
        xG:   parseFloat(document.getElementById('mv-xg').value) || 0,
        xAG:  parseFloat(document.getElementById('mv-xag').value) || 0,
        PrgC: parseFloat(document.getElementById('mv-prgc').value) || 0,
        PrgP: parseFloat(document.getElementById('mv-prgp').value) || 0,
        PrgR: parseFloat(document.getElementById('mv-prgr').value) || 0,
        '90s': parseFloat(document.getElementById('mv-90s').value) || 0,
      };
      const res = await fetchJSON('/api/predict/market-value', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      renderMarketValue(res);
    } catch (err) {
      document.getElementById('mv-result').innerHTML = `<div class="empty-state">${err.message}</div>`;
    } finally {
      setButtonBusy(btn, false, 'Estimate Market Value');
    }
  });
}

function renderMarketValue(data) {
  const panel = document.getElementById('mv-result');
  panel.classList.remove('empty-state');
  if (data.predictionAvailable === false || data.available === false) {
    panel.innerHTML = renderReadinessNotice(data);
    return;
  }
  const pct = data.percentile_rank ?? 50;
  const tierColors = { Elite: '#f43f5e', WorldClass: '#f59e0b', Good: '#6366f1', Average: '#10b981', BelowAverage: '#64748b' };
  const tKey = (data.tier || '').replace(' ', '');
  const color = tierColors[tKey] || '#9fe870';
  const contribs = data.contributions || {};
  const maxC = Math.max(...Object.values(contribs).map(Math.abs), 0.001);

  const contribRows = Object.entries(contribs).map(([k, v]) => {
    const p = Math.min(Math.abs(v) / maxC * 100, 100);
    const cls = v >= 0 ? 'pos' : 'neg';
    return `
      <div class="mv-contrib-row">
        <span class="mv-contrib-label">${escHtml(k)}</span>
        <div class="mv-contrib-track"><div class="mv-contrib-fill ${cls}" style="width:${p.toFixed(1)}%"></div></div>
        <span class="mv-contrib-value">${v >= 0 ? '+' : ''}${v.toFixed(1)}</span>
      </div>`;
  }).join('');

  panel.innerHTML = `
    <div class="mv-value-display">
      <div class="mv-value-number">€${data.estimated_value_m ?? '—'}M</div>
      <div class="mv-tier-badge" style="background:${color}">${data.tier || '—'}</div>
      <div class="mv-percentile">Top ${100 - pct}% of all players · ${pct}th percentile</div>
      <div class="mv-gauge-track"><div class="mv-gauge-fill" style="width:${pct}%"></div></div>
    </div>
    <div class="mv-contributions">${contribRows}</div>
    <p class="mv-note">${data.note || 'Performance-based index — not an official transfer fee.'}</p>`;
}

function renderReadinessNotice(data) {
  const required = (data.requiredData || []).map(item => `<li>${escHtml(item)}</li>`).join('');
  return `
    <div class="readiness-card">
      <strong>${escHtml(data.feature || 'Prediction')} unavailable</strong>
      <p>${escHtml(data.message || 'The current dataset does not contain enough labelled data for this prediction.')}</p>
      <ul>${required}</ul>
    </div>`;
}

function initInjuryRiskSystem() {
  const btn = document.getElementById('injury-check-btn');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    setButtonBusy(btn, true, 'Checking...');
    try {
      const res = await fetchJSON('/api/predict/injury-risk');
      const panel = document.getElementById('injury-result');
      panel.classList.remove('empty-state');
      panel.innerHTML = renderReadinessNotice(res);
    } catch (err) {
      document.getElementById('injury-result').innerHTML = `<div class="empty-state">${escHtml(err.message)}</div>`;
    } finally {
      setButtonBusy(btn, false, 'Check Injury Data');
    }
  });
}

// ============================================================
// FEATURE 7 — TRANSFER REPLACEMENT
// ============================================================
function initTransferReplaceSystem() {
  const btn = document.getElementById('tr-search-btn');
  if (!btn) return;
  setupAutocomplete('tr-player-input', 'tr-autocomplete', () => {});

  btn.addEventListener('click', async () => {
    const name = document.getElementById('tr-player-input').value.trim();
    if (!name) { document.getElementById('tr-status').textContent = 'Enter a player name.'; return; }

    setButtonBusy(btn, true, 'Searching…');
    document.getElementById('tr-status').textContent = '';
    try {
      const topK = parseInt(document.getElementById('tr-top-k').value, 10) || 5;
      const ageRange = parseInt(document.getElementById('tr-age-range').value, 10) || 5;
      const excludeSquad = document.getElementById('tr-exclude-squad').checked;
      const res = await fetchJSON(
        `/api/transfer-replace?player=${encodeURIComponent(name)}&top_k=${topK}&age_range=${ageRange}&exclude_squad=${excludeSquad}`
      );
      if (res.error) { document.getElementById('tr-status').textContent = res.error; return; }
      renderTransferSubject(res.subject);
      renderTransferResults(res.replacements || []);
      document.getElementById('tr-status').textContent = `Found ${(res.replacements || []).length} replacement candidates.`;
    } catch (err) {
      document.getElementById('tr-status').textContent = `Error: ${err.message}`;
    } finally {
      setButtonBusy(btn, false, 'Find Replacements');
    }
  });
}

function renderTransferSubject(subject) {
  const panel = document.getElementById('tr-subject-panel');
  if (!subject) return;
  panel.classList.remove('empty-state');
  panel.innerHTML = `
    <div class="transfer-subject-card">
      <div class="transfer-subject-name">${escHtml(subject.name)}</div>
      <div class="profile-meta">
        <span class="position-badge">${subject.position || '—'}</span>
        <span>${escHtml(subject.squad || '—')}</span>
        <span>Age ${subject.age ?? '—'}</span>
      </div>
    </div>
    <div class="profile-stats">
      <div class="stat-item"><span>Goals</span><strong>${subject.goals ?? 0}</strong></div>
      <div class="stat-item"><span>Assists</span><strong>${subject.assists ?? 0}</strong></div>
      <div class="stat-item"><span>xG</span><strong>${subject.xg ?? '—'}</strong></div>
      <div class="stat-item"><span>xAG</span><strong>${subject.xa ?? '—'}</strong></div>
      <div class="stat-item"><span>PrgC</span><strong>${subject.prgc ?? '—'}</strong></div>
      <div class="stat-item"><span>PrgP</span><strong>${subject.prgp ?? '—'}</strong></div>
    </div>`;
}

function renderTransferResults(players) {
  const container = document.getElementById('tr-results-panel');
  if (!players.length) { container.innerHTML = ''; return; }
  container.innerHTML = players.map((player, index) => `
    <article class="similar-player-card ${index === 0 ? 'top-match' : ''}">
      <div class="card-header">
        <div class="rank-badge">#${index + 1}</div>
        <div class="similarity-score">${player.similarity}%</div>
      </div>
      <div class="player-info">
        <h4>${escHtml(player.name)}</h4>
        <p>${escHtml(player.squad)}</p>
        <div class="position-badge-small">${player.position}</div>
      </div>
      <div class="player-stats">
        <span class="stat-chip">Age ${player.age || '—'}</span>
        <span class="stat-chip">Goals ${player.goals}</span>
        <span class="stat-chip">Assists ${player.assists}</span>
        <span class="stat-chip">xG ${player.xg ?? '—'}</span>
      </div>
    </article>`).join('');
}

// ============================================================
// FEATURE 9 — BEST XI GENERATOR
// ============================================================
const FORMATION_LAYOUTS = {
  '4-3-3': {
    GK:  [{ x: 50, y: 90 }],
    LB:  [{ x: 15, y: 72 }], CB: [{ x: 35, y: 72 }, { x: 65, y: 72 }], RB: [{ x: 85, y: 72 }],
    LCM: [{ x: 25, y: 52 }], CM: [{ x: 50, y: 52 }], RCM: [{ x: 75, y: 52 }],
    LW:  [{ x: 15, y: 22 }], ST: [{ x: 50, y: 15 }], RW: [{ x: 85, y: 22 }],
  },
  '4-4-2': {
    GK:  [{ x: 50, y: 90 }],
    LB:  [{ x: 15, y: 72 }], CB: [{ x: 35, y: 72 }, { x: 65, y: 72 }], RB: [{ x: 85, y: 72 }],
    LM:  [{ x: 15, y: 50 }], LCM: [{ x: 38, y: 50 }], RCM: [{ x: 62, y: 50 }], RM: [{ x: 85, y: 50 }],
    LST: [{ x: 35, y: 18 }], RST: [{ x: 65, y: 18 }],
  },
  '3-5-2': {
    GK:  [{ x: 50, y: 90 }],
    LCB: [{ x: 22, y: 75 }], CB: [{ x: 50, y: 75 }], RCB: [{ x: 78, y: 75 }],
    LWB: [{ x: 10, y: 55 }], LCM: [{ x: 30, y: 52 }], CM: [{ x: 50, y: 50 }], RCM: [{ x: 70, y: 52 }], RWB: [{ x: 90, y: 55 }],
    LST: [{ x: 35, y: 18 }], RST: [{ x: 65, y: 18 }],
  },
  '4-2-3-1': {
    GK:  [{ x: 50, y: 90 }],
    LB:  [{ x: 15, y: 74 }], CB: [{ x: 35, y: 74 }, { x: 65, y: 74 }], RB: [{ x: 85, y: 74 }],
    LDM: [{ x: 35, y: 57 }], RDM: [{ x: 65, y: 57 }],
    LAM: [{ x: 18, y: 35 }], CAM: [{ x: 50, y: 33 }], RAM: [{ x: 82, y: 35 }],
    ST:  [{ x: 50, y: 14 }],
  },
};

function initBestXISystem() {
  const btn = document.getElementById('xi-generate-btn');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    const formation = document.getElementById('xi-formation').value;
    setButtonBusy(btn, true, 'Building XI…');
    document.getElementById('xi-status').textContent = '';
    try {
      const res = await fetchJSON(`/api/best-xi?formation=${encodeURIComponent(formation)}`);
      if (res.error) { document.getElementById('xi-status').textContent = res.error; return; }
      renderBestXI(res, formation);
    } catch (err) {
      document.getElementById('xi-status').textContent = `Error: ${err.message}`;
    } finally {
      setButtonBusy(btn, false, 'Generate Best XI');
    }
  });
}

function posClass(pos) {
  if (!pos) return 'pos-mf';
  const p = pos.toUpperCase();
  if (p.includes('GK')) return 'pos-gk';
  if (p.includes('DF') || p.includes('CB') || p.includes('LB') || p.includes('RB') || p.includes('WB')) return 'pos-df';
  if (p.includes('FW') || p.includes('ST') || p.includes('LW') || p.includes('RW')) return 'pos-fw';
  return 'pos-mf';
}

function renderBestXI(data, formation) {
  const pitch = document.getElementById('xi-pitch');
  const sidebar = document.getElementById('xi-sidebar');
  const players = data.players || [];
  const layout = FORMATION_LAYOUTS[formation] || FORMATION_LAYOUTS['4-3-3'];

  // Flatten layout slots in order
  const slots = [];
  Object.entries(layout).forEach(([slotKey, positions]) => {
    positions.forEach(pos => slots.push({ slotKey, ...pos }));
  });

  // Build pitch nodes
  pitch.innerHTML = '';
  players.forEach((p, i) => {
    const slotPos = slots[i] || { x: 50, y: 50 };
    const node = document.createElement('div');
    node.className = 'xi-player-node';
    node.style.left = `${slotPos.x}%`;
    node.style.top = `${slotPos.y}%`;
    const initials = (p.name || '?').split(' ').map(w => w[0]).join('').slice(0, 3).toUpperCase();
    const pc = posClass(p.slot || p.position);
    node.innerHTML = `
      <div class="xi-player-avatar ${pc}">${initials}</div>
      <div class="xi-player-name">${escHtml((p.name || '').split(' ').pop())}</div>
      <div class="xi-player-slot">${escHtml(p.slot || p.position || '—')}</div>`;
    pitch.appendChild(node);
  });

  // Sidebar: stats summary + player list
  const totalGoals = players.reduce((s, p) => s + (p.goals || 0), 0);
  const totalAssists = players.reduce((s, p) => s + (p.assists || 0), 0);
  const avgScore = players.length ? (players.reduce((s, p) => s + (p.score || 0), 0) / players.length).toFixed(1) : '—';

  const listRows = players.map(p => `
    <div class="xi-list-row">
      <div class="xi-list-avatar ${posClass(p.slot || p.position)}">${(p.name || '?').split(' ').map(w=>w[0]).join('').slice(0,2).toUpperCase()}</div>
      <div class="xi-list-info">
        <div class="xi-list-name">${escHtml(p.name || '—')}</div>
        <div class="xi-list-meta">${escHtml(p.squad || '—')} · ${p.slot || p.position || '—'}</div>
      </div>
      <div class="xi-list-score">${(p.score || 0).toFixed(1)}</div>
    </div>`).join('');

  sidebar.innerHTML = `
    <div class="xi-summary-card">
      <div class="card-kicker">Team Stats</div>
      <div class="xi-summary-grid">
        <div class="xi-summary-stat"><strong>${totalGoals}</strong><span>Total Goals</span></div>
        <div class="xi-summary-stat"><strong>${totalAssists}</strong><span>Total Assists</span></div>
        <div class="xi-summary-stat"><strong>${avgScore}</strong><span>Avg Score</span></div>
        <div class="xi-summary-stat"><strong>${formation}</strong><span>Formation</span></div>
      </div>
    </div>
    <div class="surface-card">
      <div class="card-kicker">Player List</div>
      <div class="xi-player-list">${listRows}</div>
    </div>`;
}
