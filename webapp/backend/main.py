import io
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import processing


COLUMN_NAMES = ["Tiempo_s", "EEG_1", "EEG_2", "EMG_1", "EMG_2", "EMG_3", "EMG_4", "EMG_5", "EMG_6"]

STATE: dict = {
    "df": None,
    "srate": None,
    "eeg_filtered": {},
    "emg_filtered": None,
    "emg_scaled": None,
    "markers": None,
    "marker_times": None,
    "emg_channel": None,
    "eeg_epochs_corrected": None,
    "segment_t": None,
}


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="RehabIA Web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FilterParams(BaseModel):
    highpass: Optional[float] = 1.0
    lowpass: Optional[float] = 100.0
    notch: Optional[float] = 50.0


class BurstParams(BaseModel):
    threshold: float = 0.2
    time_before: float = 0.02
    time_after: float = 0.02
    before_a: float = 0.20
    after_a: float = 0.15
    duration: float = 0.4


class AnalyzeRequest(BaseModel):
    filters: FilterParams
    burst: BurstParams
    emg_channel: str
    eeg_channels: List[str] = ["EEG_1", "EEG_2"]


class SegmentRequest(BaseModel):
    eeg_channel: str = "EEG_1"
    window: float = 2.0
    onset: float = 1.0
    baseline: float = 0.1


class ReorderRequest(BaseModel):
    n_groups: int = 2
    seed: Optional[int] = None


def _parse_file(name: str, content: bytes) -> pd.DataFrame:
    bio = io.BytesIO(content)
    suffix = Path(name).suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(bio)
    else:
        df = pd.read_csv(bio, sep=r"\s+", header=None, engine="python")
        if df.shape[1] == len(COLUMN_NAMES):
            df.columns = COLUMN_NAMES
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Archivo con {df.shape[1]} columnas, se esperaban {len(COLUMN_NAMES)}.",
            )

    if "Tiempo_s" not in df.columns:
        raise HTTPException(status_code=400, detail="Falta la columna 'Tiempo_s'.")

    return df


def _infer_srate(df: pd.DataFrame) -> float:
    t = df["Tiempo_s"].values
    if len(t) < 2:
        raise HTTPException(status_code=400, detail="Archivo con muy pocas muestras.")
    dt = float(np.median(np.diff(t[:1000])))
    if dt <= 0:
        raise HTTPException(status_code=400, detail="No se pudo inferir la frecuencia de muestreo.")
    return round(1.0 / dt)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    df = _parse_file(file.filename, content)
    srate = _infer_srate(df)

    STATE["df"] = df
    STATE["srate"] = srate
    STATE["eeg_filtered"] = {}
    STATE["emg_filtered"] = None
    STATE["emg_scaled"] = None
    STATE["markers"] = None
    STATE["marker_times"] = None
    STATE["emg_channel"] = None
    STATE["eeg_epochs_corrected"] = None
    STATE["segment_t"] = None

    duration = float(df["Tiempo_s"].iloc[-1] - df["Tiempo_s"].iloc[0])

    return {
        "filename": file.filename,
        "n_samples": len(df),
        "srate": srate,
        "duration": duration,
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


def _downsample(x: np.ndarray, max_points: int = 8000):
    if len(x) <= max_points:
        return x.tolist(), np.arange(len(x))
    step = max(1, len(x) // max_points)
    idx = np.arange(0, len(x), step)
    return x[idx].tolist(), idx


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    df = STATE["df"]
    if df is None:
        raise HTTPException(status_code=400, detail="Primero cargá un archivo.")

    srate = STATE["srate"]

    if req.emg_channel not in df.columns:
        raise HTTPException(status_code=400, detail=f"Canal EMG '{req.emg_channel}' no existe.")

    emg_raw = df[req.emg_channel].values.astype(float)
    emg_filt = processing.apply_filters(
        emg_raw, srate,
        highpass=req.filters.highpass,
        lowpass=req.filters.lowpass,
        notch=req.filters.notch,
    )

    eeg_filtered = {}
    for ch in req.eeg_channels:
        if ch in df.columns:
            eeg_filtered[ch] = processing.apply_filters(
                df[ch].values.astype(float), srate,
                highpass=req.filters.highpass,
                lowpass=req.filters.lowpass,
                notch=req.filters.notch,
            )

    markers, emg_scaled = processing.detect_markers(
        emg_filt, srate,
        threshold=req.burst.threshold,
        time_after=req.burst.time_after,
        time_before=req.burst.time_before,
        after_a=req.burst.after_a,
        before_a=req.burst.before_a,
        duration=req.burst.duration,
    )

    STATE["emg_filtered"] = emg_filt
    STATE["emg_scaled"] = emg_scaled
    STATE["eeg_filtered"] = eeg_filtered
    STATE["markers"] = markers
    STATE["emg_channel"] = req.emg_channel
    STATE["eeg_epochs_corrected"] = None
    STATE["segment_t"] = None

    time_full = df["Tiempo_s"].values
    marker_times = time_full[markers].tolist() if len(markers) else []
    STATE["marker_times"] = marker_times

    time_ds, idx = _downsample(time_full)
    emg_scaled_ds = emg_scaled[idx].tolist()
    emg_filt_ds = emg_filt[idx].tolist()

    return {
        "time": time_ds,
        "emg_filtered": emg_filt_ds,
        "emg_scaled": emg_scaled_ds,
        "marker_times": marker_times,
        "n_markers": int(len(markers)),
        "threshold": req.burst.threshold,
    }


@app.post("/api/segment")
def segment(req: SegmentRequest):
    if STATE["markers"] is None or STATE["emg_filtered"] is None:
        raise HTTPException(status_code=400, detail="Primero corré /api/analyze.")

    if req.eeg_channel not in STATE["eeg_filtered"]:
        raise HTTPException(status_code=400, detail=f"EEG '{req.eeg_channel}' no está filtrado.")

    srate = STATE["srate"]
    eeg = STATE["eeg_filtered"][req.eeg_channel]
    emg = STATE["emg_filtered"]
    markers = STATE["markers"]

    eeg_epochs, emg_epochs = processing.segment_data(
        eeg, emg, markers, window=req.window, onset=req.onset, srate=srate,
    )

    if eeg_epochs.size == 0:
        raise HTTPException(status_code=400, detail="No hay épocas válidas (ventana excede los bordes).")

    eeg_avg, emg_avg, eeg_corrected = processing.epoch_and_average(
        eeg_epochs, emg_epochs, srate, baseline=req.baseline
    )

    win_samples = eeg_epochs.shape[1]
    t = (np.arange(win_samples) / srate) - req.onset

    STATE["eeg_epochs_corrected"] = eeg_corrected
    STATE["segment_t"] = t

    return {
        "t": t.tolist(),
        "eeg_avg": eeg_avg.tolist(),
        "emg_avg": emg_avg.tolist(),
        "n_trials": int(eeg_epochs.shape[0]),
    }


@app.post("/api/reorder_split")
def reorder_split(req: ReorderRequest):
    epochs = STATE["eeg_epochs_corrected"]
    t = STATE["segment_t"]
    if epochs is None or t is None:
        raise HTTPException(status_code=400, detail="Primero corré /api/segment.")

    if req.n_groups < 2:
        raise HTTPException(status_code=400, detail="n_groups debe ser >= 2.")

    rng = np.random.default_rng(req.seed) if req.seed is not None else np.random.default_rng()
    groups = processing.reorder_and_split(epochs, n_groups=req.n_groups, rng=rng)

    if not groups:
        raise HTTPException(status_code=400, detail="No hay épocas para partir.")

    return {
        "t": t.tolist(),
        "n_total_trials": int(epochs.shape[0]),
        "groups": [
            {"n_trials": int(n), "avg": avg.tolist()}
            for avg, n in groups
        ],
    }


@app.get("/api/export/markers")
def export_markers():
    if STATE["marker_times"] is None or not STATE["marker_times"]:
        raise HTTPException(status_code=400, detail="No hay marcadores para exportar.")

    df_out = pd.DataFrame({
        "marker_index": STATE["markers"],
        "marker_time_s": STATE["marker_times"],
        "emg_channel": STATE["emg_channel"],
    })

    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=markers.csv"},
    )


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
