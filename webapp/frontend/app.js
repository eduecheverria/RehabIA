/* =========================================================
   RehabIA — app.js (clinical, dark mode, patient data)
   Preserves: IDs, fetch endpoints, callbacks, 4-step order.
   ========================================================= */

const $ = (id) => document.getElementById(id);

const state = { columns: [], srate: null };

/* Live preview of burst detection — activates after first Analyze */
let previewEnabled = false;
let previewTimer = null;
let previewAbort = null;

/* ---------- Plot theming (reads CSS vars so dark/light work) ---------- */

function plotColors() {
  const cs = getComputedStyle(document.documentElement);
  return {
    bg:     cs.getPropertyValue("--plot-bg").trim() || "#fff",
    grid:   cs.getPropertyValue("--plot-grid").trim() || "#eef1f0",
    zero:   cs.getPropertyValue("--plot-zero").trim() || "#cfd5d3",
    fg:     cs.getPropertyValue("--fg").trim() || "#1b2826",
    muted:  cs.getPropertyValue("--muted").trim() || "#6a7876",
    border: cs.getPropertyValue("--border-strong").trim() || "#cfd5d3",
    surface:cs.getPropertyValue("--surface").trim() || "#fff",
  };
}

function plotLayout() {
  const c = plotColors();
  return {
    paper_bgcolor: c.bg,
    plot_bgcolor: c.bg,
    font: { color: c.fg, family: "Inter, system-ui, sans-serif", size: 12 },
    margin: { l: 54, r: 24, t: 36, b: 44 },
    xaxis: { gridcolor: c.grid, zerolinecolor: c.zero, linecolor: c.zero, tickfont: { size: 11, color: c.muted } },
    yaxis: { gridcolor: c.grid, zerolinecolor: c.zero, linecolor: c.zero, tickfont: { size: 11, color: c.muted } },
    hoverlabel: { bgcolor: c.surface, bordercolor: c.border, font: { family: "JetBrains Mono, monospace", size: 11, color: c.fg } },
  };
}

const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function themeAccents() {
  const cs = getComputedStyle(document.documentElement);
  return {
    ACCENT: cs.getPropertyValue("--accent").trim() || "#2b8d96",
    SIGNAL: cs.getPropertyValue("--signal").trim() || "#4fa578",
    DANGER: cs.getPropertyValue("--danger").trim() || "#c94a4a",
  };
}

let GROUP_COLORS = [];
function refreshGroupColors() {
  const a = themeAccents();
  GROUP_COLORS = [a.ACCENT, a.SIGNAL, "#d9a14a", a.DANGER, "#9b7ecf", "#4fb2b8"];
}

/* ---------- Small helpers ---------- */

function setInfo(el, msg, kind = "") {
  if (!el) return;
  el.textContent = msg || "";
  el.className = "info " + kind;
}

function setStep(n, statusText, stateName) {
  const step = document.querySelector(`.step[data-step="${n}"]`);
  const st = $(`st-${n}`);
  if (!step) return;
  step.classList.remove("is-active", "is-done", "is-error", "is-locked");
  if (stateName === "active") step.classList.add("is-active");
  else if (stateName === "done") step.classList.add("is-done");
  else if (stateName === "error") step.classList.add("is-error");
  else if (stateName === "locked") step.classList.add("is-locked");
  if (st && statusText) st.textContent = statusText;
}

function unlockCard(id) { const c = $(id); if (c) c.classList.remove("is-locked"); }
function lockCard(id)   { const c = $(id); if (c) c.classList.add("is-locked"); }

function showError(panelId, detail) {
  const p = $(panelId);
  if (!p) return;
  const body = p.querySelector("[data-err-msg]");
  if (body) body.textContent = detail || "Error desconocido";
  p.classList.add("is-visible");
}
function clearError(panelId) { const p = $(panelId); if (p) p.classList.remove("is-visible"); }

function setButtonLoading(btn, loading) {
  if (!btn) return;
  if (loading) { btn.classList.add("is-loading"); btn.disabled = true; }
  else { btn.classList.remove("is-loading"); btn.disabled = false; }
}

function setPlotLoading(plotEl, loading) {
  const wrap = plotEl.closest(".plot-wrap");
  if (wrap) wrap.classList.toggle("is-loading", !!loading);
}

function hideEmpty(id) { const e = $(id); if (e) e.classList.add("hidden"); }
function showEmpty(id) { const e = $(id); if (e) e.classList.remove("hidden"); }

function formatNumber(n) { try { return n.toLocaleString("es-419"); } catch { return String(n); } }

function setMeta(id, value) {
  const el = $(id);
  if (!el) return;
  el.textContent = value;
  el.classList.remove("placeholder");
}

/* ---------- Theme toggle ---------- */

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("rehabia.theme", theme); } catch (e) {}
  $("theme-light")?.classList.toggle("is-on", theme === "light");
  $("theme-dark")?.classList.toggle("is-on", theme === "dark");
  refreshGroupColors();
  // Re-style any already-rendered plots
  ["emg-plot", "erp-plot", "reorder-plot"].forEach(id => {
    const el = document.getElementById(id);
    if (el && el.data && el.data.length) {
      Plotly.relayout(el, plotLayout());
    }
  });
}

function initTheme() {
  const saved = (() => { try { return localStorage.getItem("rehabia.theme"); } catch { return null; } })();
  applyTheme(saved || "light");
  document.querySelectorAll("[data-theme-set]").forEach(btn => {
    btn.addEventListener("click", () => applyTheme(btn.dataset.themeSet));
  });
}

/* ---------- Patient data (persisted) ---------- */

const PATIENT_KEY = "rehabia.patient";
const PATIENT_FIELDS = ["name", "id", "date", "session", "dx", "therapist", "notes"];

function loadPatient() {
  try { return JSON.parse(localStorage.getItem(PATIENT_KEY) || "{}") || {}; }
  catch { return {}; }
}
function savePatient(data) {
  try { localStorage.setItem(PATIENT_KEY, JSON.stringify(data)); } catch {}
}

function initials(name) {
  if (!name) return "??";
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map(p => p[0] || "").join("").toUpperCase() || "??";
}

function formatDateCo(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso + "T00:00:00");
    return d.toLocaleDateString("es-419", { day: "2-digit", month: "short", year: "numeric" });
  } catch { return iso; }
}

function renderPatient() {
  const p = loadPatient();
  const hasAny = PATIENT_FIELDS.some(k => (p[k] || "").trim());

  const nameEl = $("p-name");
  if (p.name) { nameEl.textContent = p.name; nameEl.classList.remove("placeholder"); }
  else { nameEl.textContent = "Sin datos del paciente"; nameEl.classList.add("placeholder"); }

  $("p-avatar").textContent = initials(p.name);

  const idEl = $("p-id");
  if (p.id) { idEl.textContent = p.id; idEl.hidden = false; } else { idEl.hidden = true; }

  $("p-date").textContent = formatDateCo(p.date);
  $("p-session").textContent = p.session || "—";
  $("p-dx").textContent = p.dx || "—";
  $("p-therapist").textContent = p.therapist || "—";

  const notesWrap = $("p-notes-wrap");
  if (p.notes && p.notes.trim()) {
    $("p-notes").textContent = p.notes;
    notesWrap.hidden = false;
  } else {
    notesWrap.hidden = true;
  }

  // Meta row visibility: show a hint to edit if totally empty
  $("p-meta").style.display = hasAny ? "" : "none";
}

function openPatientEdit() {
  const p = loadPatient();
  $("p-in-name").value = p.name || "";
  $("p-in-id").value = p.id || "";
  $("p-in-date").value = p.date || new Date().toISOString().slice(0, 10);
  $("p-in-session").value = p.session || "";
  $("p-in-dx").value = p.dx || "";
  $("p-in-therapist").value = p.therapist || "";
  $("p-in-notes").value = p.notes || "";
  $("patient-view").classList.add("is-editing");
  setTimeout(() => $("p-in-name").focus(), 50);
}
function closePatientEdit() {
  $("patient-view").classList.remove("is-editing");
}

function initPatient() {
  renderPatient();
  $("patient-edit-btn").addEventListener("click", openPatientEdit);
  $("patient-cancel-btn").addEventListener("click", closePatientEdit);
  $("patient-save-btn").addEventListener("click", () => {
    const data = {
      name: $("p-in-name").value.trim(),
      id: $("p-in-id").value.trim(),
      date: $("p-in-date").value,
      session: $("p-in-session").value.trim(),
      dx: $("p-in-dx").value.trim(),
      therapist: $("p-in-therapist").value.trim(),
      notes: $("p-in-notes").value.trim(),
    };
    savePatient(data);
    renderPatient();
    closePatientEdit();
  });
}

/* ---------- Channels ---------- */

function populateChannelSelectors(columns) {
  const emg = $("emg-channel");
  const eeg = $("eeg-channel");
  emg.innerHTML = "";
  eeg.innerHTML = "";
  for (const c of columns) {
    if (c.startsWith("EMG")) emg.appendChild(new Option(c, c));
    if (c.startsWith("EEG")) eeg.appendChild(new Option(c, c));
  }
}

/* ---------- 1. Upload ---------- */

async function uploadFile() {
  const btn = $("upload-btn");
  const file = $("file-input").files[0];
  if (!file) {
    setInfo($("upload-info"), "Debe seleccionar un archivo primero.", "err");
    setStep(1, "Sin archivo", "error");
    return;
  }

  setInfo($("upload-info"), "Cargando...", "work");
  setStep(1, "Cargando…", "active");
  setButtonLoading(btn, true);

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    state.columns = data.columns;
    state.srate = data.srate;

    populateChannelSelectors(data.columns);

    setMeta("m-file", data.filename);
    setMeta("m-samples", formatNumber(data.n_samples));
    setMeta("m-duration", `${data.duration.toFixed(1)} s`);
    setMeta("m-srate", `${data.srate} Hz`);

    setInfo($("upload-info"),
      `${data.filename} — ${formatNumber(data.n_samples)} muestras · ${data.duration.toFixed(1)}s @ ${data.srate} Hz`, "ok");

    $("analyze-section").hidden = false;
    unlockCard("card-2");
    setStep(1, "Listo", "done");
    setStep(2, "Esperando análisis", "active");
  } catch (e) {
    setInfo($("upload-info"), "Error: " + e.message, "err");
    setStep(1, "Error de carga", "error");
  } finally {
    setButtonLoading(btn, false);
  }
}

/* ---------- 2. Analyze ---------- */

function buildAnalyzeBody() {
  return {
    filters: {
      highpass: parseFloat($("f-hp").value) || null,
      lowpass: parseFloat($("f-lp").value) || null,
      notch: parseFloat($("f-notch").value) || null,
    },
    burst: {
      threshold: parseFloat($("b-threshold").value),
      time_before: parseFloat($("b-time-before").value),
      time_after: parseFloat($("b-time-after").value),
      before_a: parseFloat($("b-before-a").value),
      after_a: parseFloat($("b-after-a").value),
      duration: parseFloat($("b-duration").value),
    },
    emg_channel: $("emg-channel").value,
    eeg_channels: Array.from($("eeg-channel").options).map((o) => o.value),
  };
}

async function runAnalyze() {
  const btn = $("analyze-btn");
  const body = buildAnalyzeBody();

  clearError("analyze-error");
  setInfo($("markers-info"), "Procesando…", "work");
  setStep(2, "Procesando…", "active");
  setButtonLoading(btn, true);
  setPlotLoading($("emg-plot"), true);
  hideEmpty("emg-empty");

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    setInfo($("markers-info"), `${data.n_markers} contracciones detectadas`, "ok");
    $("analyze-chips").hidden = false;
    $("chip-markers").textContent = data.n_markers;
    $("chip-threshold").textContent = (data.threshold ?? parseFloat($("b-threshold").value)).toFixed(2);
    $("export-btn").disabled = data.n_markers === 0;

    plotEmg(data);

    lockCard("card-4");
    setStep(4, "Requiere paso 3", "locked");
    $("reorder-chips").hidden = true;

    if (data.n_markers > 0) {
      unlockCard("card-3");
      unlockCard("card-5");
      setStep(2, `${data.n_markers} contracciones`, "done");
      setStep(3, "Listo para calcular", "active");
      setStep(5, "Listo para comparar", "active");
    } else {
      lockCard("card-3");
      lockCard("card-5");
      setStep(2, "Sin contracciones", "error");
      setStep(3, "Requiere paso 2", "locked");
      setStep(5, "Requiere paso 2", "locked");
    }

    previewEnabled = true;
  } catch (e) {
    showError("analyze-error", e.message);
    setInfo($("markers-info"), "Error de análisis", "err");
    setStep(2, "Error", "error");
    showEmpty("emg-empty");
  } finally {
    setButtonLoading(btn, false);
    setPlotLoading($("emg-plot"), false);
  }
}

function plotEmg(data) {
  const { ACCENT, SIGNAL, DANGER } = themeAccents();
  const traces = [{
    x: data.time, y: data.emg_scaled,
    type: "scattergl", mode: "lines",
    name: "EMG",
    line: { color: ACCENT, width: 1 },
    hovertemplate: "t=%{x:.3f}s · %{y:.3f}<extra></extra>",
  }];
  const shapes = [
    { type: "line", xref: "paper", x0: 0, x1: 1, y0: data.threshold, y1: data.threshold,
      line: { color: DANGER, width: 1, dash: "dot" } },
    ...data.marker_times.map((t) => ({
      type: "line", x0: t, x1: t, yref: "paper", y0: 0, y1: 1,
      line: { color: SIGNAL, width: 1 }, opacity: 0.7,
    })),
  ];
  Plotly.newPlot("emg-plot", traces, {
    ...plotLayout(),
    title: { text: "Señal muscular y contracciones detectadas", font: { size: 12.5 }, x: 0.01 },
    xaxis: { ...plotLayout().xaxis, title: { text: "Tiempo (s)", font: { size: 11 } } },
    yaxis: { ...plotLayout().yaxis, title: { text: "Amplitud", font: { size: 11 } } },
    shapes, showlegend: false,
  }, PLOT_CONFIG);
}

/* ---------- 2b. Live preview of burst detection ---------- */

function schedulePreview() {
  if (!previewEnabled) return;
  clearTimeout(previewTimer);
  previewTimer = setTimeout(runPreview, 300);
}

function updateMarkersOnly(data) {
  const { SIGNAL, DANGER } = themeAccents();
  const shapes = [
    {
      type: "line", xref: "paper",
      x0: 0, x1: 1, y0: data.threshold, y1: data.threshold,
      line: { color: DANGER, width: 1, dash: "dot" },
    },
    ...data.marker_times.map((t) => ({
      type: "line", x0: t, x1: t, yref: "paper", y0: 0, y1: 1,
      line: { color: SIGNAL, width: 1 }, opacity: 0.7,
    })),
  ];
  Plotly.relayout("emg-plot", { shapes });
}

async function runPreview() {
  if (previewAbort) previewAbort.abort();
  previewAbort = new AbortController();

  try {
    const res = await fetch("/api/burst_preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildAnalyzeBody()),
      signal: previewAbort.signal,
    });
    if (!res.ok) return;
    const data = await res.json();

    if (data.signal_changed === false) {
      updateMarkersOnly(data);
    } else {
      plotEmg(data);
    }

    $("chip-markers").textContent = data.n_markers;
    $("chip-threshold").textContent = (data.threshold ?? parseFloat($("b-threshold").value)).toFixed(2);
    setInfo($("markers-info"), `${data.n_markers} contracciones (vista previa — apretá Analizar para confirmar)`, "work");
  } catch (e) {
    /* silent: aborted requests are expected when typing fast */
  }
}

/* ---------- 3. Segment / ERP ---------- */

async function runSegment() {
  const btn = $("segment-btn");
  const body = {
    eeg_channel: $("eeg-channel").value,
    window: parseFloat($("s-window").value),
    onset: parseFloat($("s-onset").value),
    baseline: parseFloat($("s-baseline").value),
  };

  clearError("segment-error");
  setInfo($("segment-info"), "Procesando…", "work");
  setStep(3, "Procesando…", "active");
  setButtonLoading(btn, true);
  setPlotLoading($("erp-plot"), true);
  hideEmpty("erp-empty");

  try {
    const res = await fetch("/api/segment", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    setInfo($("segment-info"), `Respuesta promediada sobre ${data.n_trials} repeticiones`, "ok");
    $("segment-chips").hidden = false;
    $("chip-trials").textContent = data.n_trials;
    plotErp(data);

    unlockCard("card-4");
    setStep(3, `${data.n_trials} repeticiones`, "done");
    setStep(4, "Listo para comparar", "active");
  } catch (e) {
    showError("segment-error", e.message);
    setInfo($("segment-info"), "Error al calcular", "err");
    setStep(3, "Error", "error");
    showEmpty("erp-empty");
  } finally {
    setButtonLoading(btn, false);
    setPlotLoading($("erp-plot"), false);
  }
}

function plotErp(data) {
  const { ACCENT, SIGNAL } = themeAccents();
  const traces = [
    { x: data.t, y: data.eeg_avg, type: "scatter", mode: "lines",
      name: "Cerebro (promedio)", line: { color: ACCENT, width: 2 }, yaxis: "y1",
      hovertemplate: "t=%{x:.3f}s · %{y:.4f} µV<extra>Cerebro</extra>" },
    { x: data.t, y: data.emg_avg, type: "scatter", mode: "lines",
      name: "Músculo (promedio)", line: { color: SIGNAL, width: 1.2 }, yaxis: "y2",
      hovertemplate: "t=%{x:.3f}s · %{y:.4f}<extra>Músculo</extra>" },
  ];
  const L = plotLayout();
  Plotly.newPlot("erp-plot", traces, {
    ...L,
    title: { text: `Respuesta promedio · ${data.n_trials} repeticiones`, font: { size: 12.5 }, x: 0.01 },
    xaxis: { ...L.xaxis, title: { text: "Tiempo relativo al movimiento (s)", font: { size: 11 } }, zeroline: true, zerolinecolor: SIGNAL, zerolinewidth: 1 },
    yaxis: { ...L.yaxis, title: { text: "Cerebro (µV)", font: { size: 11 } }, side: "left" },
    yaxis2: { title: { text: "Músculo", font: { size: 11 } }, overlaying: "y", side: "right", showgrid: false, tickfont: { size: 11, color: L.xaxis.tickfont.color }, linecolor: L.xaxis.linecolor },
    legend: { x: 0, y: 1.12, orientation: "h", font: { size: 11 } },
  }, PLOT_CONFIG);
}

/* ---------- 4. Reorder ---------- */

async function runReorder() {
  const btn = $("reorder-btn");
  const body = { n_groups: parseInt($("r-n-groups").value, 10) };

  clearError("reorder-error");
  setInfo($("reorder-info"), "Procesando…", "work");
  setStep(4, "Procesando…", "active");
  setButtonLoading(btn, true);
  setPlotLoading($("reorder-plot"), true);
  hideEmpty("reorder-empty");

  try {
    const res = await fetch("/api/reorder_split", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    const sizes = data.groups.map((g) => g.n_trials).join(" / ");
    setInfo($("reorder-info"), `${data.n_total_trials} repeticiones → ${data.groups.length} grupos (${sizes})`, "ok");
    $("reorder-chips").hidden = false;
    $("chip-groups").textContent = data.groups.length;
    $("chip-split").textContent = sizes;
    plotReorder(data);
    setStep(4, `${data.groups.length} grupos`, "done");
  } catch (e) {
    showError("reorder-error", e.message);
    setInfo($("reorder-info"), "Error al comparar", "err");
    setStep(4, "Error", "error");
    showEmpty("reorder-empty");
  } finally {
    setButtonLoading(btn, false);
    setPlotLoading($("reorder-plot"), false);
  }
}

function plotReorder(data) {
  const { SIGNAL } = themeAccents();
  refreshGroupColors();
  const traces = data.groups.map((g, i) => ({
    x: data.t, y: g.avg, type: "scatter", mode: "lines",
    name: `Grupo ${i + 1} (n=${g.n_trials})`,
    line: { color: GROUP_COLORS[i % GROUP_COLORS.length], width: 1.8 },
    hovertemplate: `t=%{x:.3f}s · %{y:.4f}<extra>Grupo ${i + 1}</extra>`,
  }));
  const L = plotLayout();
  Plotly.newPlot("reorder-plot", traces, {
    ...L,
    title: { text: "Respuesta promedio por grupo", font: { size: 12.5 }, x: 0.01 },
    xaxis: { ...L.xaxis, title: { text: "Tiempo relativo al movimiento (s)", font: { size: 11 } }, zeroline: true, zerolinecolor: SIGNAL, zerolinewidth: 1 },
    yaxis: { ...L.yaxis, title: { text: "Cerebro (µV)", font: { size: 11 } } },
    legend: { x: 0, y: 1.12, orientation: "h", font: { size: 11 } },
  }, PLOT_CONFIG);
}

/* ---------- 5. Compare methods (clustering) ---------- */

const CLUSTER_COLORS = { burst: "#2ca06f", rest: "#c06a4a" };

async function initClusterFeatures() {
  try {
    const res = await fetch("/api/cluster_features");
    if (!res.ok) return;
    const { features } = await res.json();
    const selX = $("cl-feat-x");
    const selY = $("cl-feat-y");
    selX.innerHTML = "";
    selY.innerHTML = "";
    for (const f of features) {
      selX.appendChild(new Option(f, f));
      selY.appendChild(new Option(f, f));
    }
    selX.value = "RMS";
    selY.value = "BandPow_20-150";
  } catch (e) { /* silent */ }
}

async function runCluster() {
  const btn = $("cluster-btn");
  const body = {
    feature_x: $("cl-feat-x").value,
    feature_y: $("cl-feat-y").value,
    win_s: parseFloat($("cl-win").value),
    hop_s: parseFloat($("cl-hop").value),
    tolerance_s: parseFloat($("cl-tol").value),
  };

  clearError("cluster-error");
  setInfo($("cluster-info"), "Procesando… (la primera vez extrae features, ~unos segundos)", "work");
  setStep(5, "Procesando…", "active");
  setButtonLoading(btn, true);
  setPlotLoading($("cluster-scatter"), true);
  hideEmpty("cluster-empty");

  try {
    const res = await fetch("/api/cluster_compare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    const m = data.metrics;

    setInfo(
      $("cluster-info"),
      `BacAv: ${m.n_bacav} · K-Means: ${m.n_kmeans} · coincidencia ${m.precision}% / ${m.recall}%`,
      "ok"
    );
    $("cluster-chips").hidden = false;
    $("chip-prec").textContent = `${m.matched_bacav}/${m.n_bacav} (${m.precision}%)`;
    $("chip-recall").textContent = `${m.matched_kmeans}/${m.n_kmeans} (${m.recall}%)`;

    plotClusterScatter(data);
    plotClusterTimeline(data);
    setStep(5, `${m.precision}% / ${m.recall}%`, "done");
  } catch (e) {
    showError("cluster-error", e.message);
    setInfo($("cluster-info"), "Error al comparar", "err");
    setStep(5, "Error", "error");
    showEmpty("cluster-empty");
  } finally {
    setButtonLoading(btn, false);
    setPlotLoading($("cluster-scatter"), false);
  }
}

function plotClusterScatter(data) {
  const burst = data.burst_cluster;
  const sc = data.scatter;
  const groups = { burst: { x: [], y: [] }, rest: { x: [], y: [] } };
  for (let i = 0; i < sc.x.length; i++) {
    const g = sc.labels[i] === burst ? "burst" : "rest";
    groups[g].x.push(sc.x[i]);
    groups[g].y.push(sc.y[i]);
  }

  const traces = [
    {
      x: groups.rest.x, y: groups.rest.y, type: "scattergl", mode: "markers",
      name: "Reposo", marker: { color: CLUSTER_COLORS.rest, size: 3, opacity: 0.4 },
    },
    {
      x: groups.burst.x, y: groups.burst.y, type: "scattergl", mode: "markers",
      name: "Burst", marker: { color: CLUSTER_COLORS.burst, size: 3, opacity: 0.5 },
    },
    {
      x: data.centroids.map((c) => c[0]), y: data.centroids.map((c) => c[1]),
      type: "scattergl", mode: "markers", name: "Centroides",
      marker: { color: "#000", size: 12, symbol: "x", line: { width: 1 } },
    },
  ];

  const L = plotLayout();
  Plotly.newPlot("cluster-scatter", traces, {
    ...L,
    title: { text: "Espacio de features — K-Means (2 grupos)", font: { size: 12.5 }, x: 0.01 },
    xaxis: { ...L.xaxis, title: { text: data.feature_x, font: { size: 11 } } },
    yaxis: { ...L.yaxis, title: { text: data.feature_y, font: { size: 11 } } },
    legend: { x: 0, y: 1.12, orientation: "h", font: { size: 11 } },
  }, PLOT_CONFIG);
}

function plotClusterTimeline(data) {
  const { DANGER, SIGNAL } = themeAccents();
  const traces = [
    {
      x: data.bacav_times, y: data.bacav_times.map(() => 1),
      type: "scattergl", mode: "markers", name: `BacAv (${data.bacav_times.length})`,
      marker: { color: DANGER, symbol: "line-ns-open", size: 14, line: { width: 1.4 } },
    },
    {
      x: data.kmeans_times, y: data.kmeans_times.map(() => 0),
      type: "scattergl", mode: "markers", name: `K-Means (${data.kmeans_times.length})`,
      marker: { color: SIGNAL, symbol: "line-ns-open", size: 14, line: { width: 1.4 } },
    },
  ];
  const L = plotLayout();
  Plotly.newPlot("cluster-timeline", traces, {
    ...L,
    margin: { l: 70, r: 24, t: 28, b: 36 },
    title: { text: "Marcadores en el tiempo — BacAv vs K-Means", font: { size: 12.5 }, x: 0.01 },
    xaxis: { ...L.xaxis, title: { text: "Tiempo (s)", font: { size: 11 } } },
    yaxis: { ...L.yaxis, tickvals: [0, 1], ticktext: ["K-Means", "BacAv"], range: [-0.5, 1.5] },
    showlegend: false,
  }, PLOT_CONFIG);
}

/* ---------- Export ---------- */

function downloadMarkers() { window.location.href = "/api/export/markers"; }

/* ---------- Event wiring ---------- */

initTheme();
refreshGroupColors();
initPatient();
initClusterFeatures();

$("upload-btn").addEventListener("click", uploadFile);
$("analyze-btn").addEventListener("click", runAnalyze);
$("segment-btn").addEventListener("click", runSegment);
$("export-btn").addEventListener("click", downloadMarkers);
$("reorder-btn").addEventListener("click", runReorder);
$("cluster-btn").addEventListener("click", runCluster);

document.querySelectorAll(".step").forEach((s) => {
  s.addEventListener("click", (e) => {
    if (s.classList.contains("is-locked")) { e.preventDefault(); return; }
    const n = s.dataset.step;
    const target = $(`card-${n}`);
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: "smooth", block: "start" }); }
  });
});

$("file-input").addEventListener("change", () => {
  const f = $("file-input").files[0];
  if (f) setInfo($("upload-info"), `${f.name} · listo para cargar`, "");
});

/* Live-preview listeners — update markers as user tweaks params */
const PREVIEW_INPUTS = [
  "f-hp", "f-lp", "f-notch",
  "b-threshold", "b-duration", "b-time-before", "b-time-after", "b-before-a", "b-after-a",
  "emg-channel",
];
PREVIEW_INPUTS.forEach((id) => {
  const el = $(id);
  if (!el) return;
  el.addEventListener("input", schedulePreview);
  el.addEventListener("change", schedulePreview);
});
