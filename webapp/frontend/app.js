const $ = (id) => document.getElementById(id);

const state = {
  columns: [],
  srate: null,
};

const PLOT_LAYOUT = {
  paper_bgcolor: "#171a21",
  plot_bgcolor: "#0f1319",
  font: { color: "#e6e8eb", size: 12 },
  margin: { l: 50, r: 20, t: 30, b: 40 },
};

const PLOT_CONFIG = { responsive: true, displaylogo: false };

function setInfo(el, msg, kind = "") {
  el.textContent = msg;
  el.className = "info " + kind;
}

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

async function uploadFile() {
  const file = $("file-input").files[0];
  if (!file) {
    setInfo($("upload-info"), "Elegí un archivo primero.", "err");
    return;
  }
  setInfo($("upload-info"), "Cargando...");

  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    setInfo($("upload-info"), "Error: " + err.detail, "err");
    return;
  }

  const data = await res.json();
  state.columns = data.columns;
  state.srate = data.srate;

  populateChannelSelectors(data.columns);

  setInfo(
    $("upload-info"),
    `${data.filename} — ${data.n_samples.toLocaleString()} muestras, ${data.duration.toFixed(1)}s @ ${data.srate} Hz`,
    "ok"
  );

  $("analyze-section").hidden = false;
}

async function runAnalyze() {
  const body = {
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

  setInfo($("markers-info"), "Procesando...");
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    setInfo($("markers-info"), "Error: " + err.detail, "err");
    return;
  }

  const data = await res.json();
  setInfo($("markers-info"), `${data.n_markers} marcadores detectados`, "ok");
  $("export-btn").disabled = data.n_markers === 0;

  plotEmg(data);

  $("reorder-section").hidden = true;
  if (data.n_markers > 0) {
    $("segment-section").hidden = false;
  }
}

function plotEmg(data) {
  const traces = [
    {
      x: data.time,
      y: data.emg_scaled,
      type: "scattergl",
      mode: "lines",
      name: "EMG rectificado (escalado)",
      line: { color: "#4f8cff", width: 1 },
    },
  ];

  const shapes = [
    {
      type: "line",
      xref: "paper",
      x0: 0, x1: 1,
      y0: data.threshold, y1: data.threshold,
      line: { color: "#ff5d5d", width: 1, dash: "dot" },
    },
    ...data.marker_times.map((t) => ({
      type: "line",
      x0: t, x1: t,
      yref: "paper",
      y0: 0, y1: 1,
      line: { color: "#5cd98c", width: 1 },
    })),
  ];

  Plotly.newPlot("emg-plot", traces, {
    ...PLOT_LAYOUT,
    title: { text: "EMG + marcadores", font: { size: 13 } },
    xaxis: { title: "Tiempo (s)" },
    yaxis: { title: "Amplitud escalada" },
    shapes,
  }, PLOT_CONFIG);
}

async function runSegment() {
  const body = {
    eeg_channel: $("eeg-channel").value,
    window: parseFloat($("s-window").value),
    onset: parseFloat($("s-onset").value),
    baseline: parseFloat($("s-baseline").value),
  };

  setInfo($("segment-info"), "Procesando...");
  const res = await fetch("/api/segment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    setInfo($("segment-info"), "Error: " + err.detail, "err");
    return;
  }

  const data = await res.json();
  setInfo($("segment-info"), `ERP promediado sobre ${data.n_trials} trials`, "ok");
  plotErp(data);
  $("reorder-section").hidden = false;
}

function plotErp(data) {
  const traces = [
    {
      x: data.t, y: data.eeg_avg,
      type: "scatter", mode: "lines",
      name: "EEG promedio",
      line: { color: "#4f8cff", width: 2 },
      yaxis: "y1",
    },
    {
      x: data.t, y: data.emg_avg,
      type: "scatter", mode: "lines",
      name: "EMG promedio",
      line: { color: "#5cd98c", width: 1 },
      yaxis: "y2",
    },
  ];

  Plotly.newPlot("erp-plot", traces, {
    ...PLOT_LAYOUT,
    title: { text: `ERP (${data.n_trials} trials)`, font: { size: 13 } },
    xaxis: { title: "Tiempo relativo al marcador (s)", zeroline: true, zerolinecolor: "#5cd98c" },
    yaxis: { title: "EEG (µV)", side: "left" },
    yaxis2: { title: "EMG", overlaying: "y", side: "right", showgrid: false },
    legend: { x: 0, y: 1.1, orientation: "h" },
  }, PLOT_CONFIG);
}

async function runReorder() {
  const body = { n_groups: parseInt($("r-n-groups").value, 10) };

  setInfo($("reorder-info"), "Procesando...");
  const res = await fetch("/api/reorder_split", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    setInfo($("reorder-info"), "Error: " + err.detail, "err");
    return;
  }

  const data = await res.json();
  const sizes = data.groups.map((g) => g.n_trials).join(" / ");
  setInfo(
    $("reorder-info"),
    `${data.n_total_trials} trials → ${data.groups.length} grupos (${sizes})`,
    "ok"
  );
  plotReorder(data);
}

const GROUP_COLORS = ["#4f8cff", "#5cd98c", "#ffb84d", "#ff5d5d", "#b88dff", "#5cd9d9"];

function plotReorder(data) {
  const traces = data.groups.map((g, i) => ({
    x: data.t,
    y: g.avg,
    type: "scatter",
    mode: "lines",
    name: `Grupo ${i + 1} (n=${g.n_trials})`,
    line: { color: GROUP_COLORS[i % GROUP_COLORS.length], width: 1.8 },
  }));

  Plotly.newPlot("reorder-plot", traces, {
    ...PLOT_LAYOUT,
    title: { text: "EEG promedio por grupo (reorden aleatorio)", font: { size: 13 } },
    xaxis: { title: "Tiempo relativo al marcador (s)", zeroline: true, zerolinecolor: "#5cd98c" },
    yaxis: { title: "EEG (µV)" },
    legend: { x: 0, y: 1.1, orientation: "h" },
  }, PLOT_CONFIG);
}

function downloadMarkers() {
  window.location.href = "/api/export/markers";
}

$("upload-btn").addEventListener("click", uploadFile);
$("analyze-btn").addEventListener("click", runAnalyze);
$("segment-btn").addEventListener("click", runSegment);
$("export-btn").addEventListener("click", downloadMarkers);
$("reorder-btn").addEventListener("click", runReorder);
