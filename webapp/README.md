# webapp — Interfaz HTML alternativa

Versión sin Streamlit: backend FastAPI + frontend HTML/JS con Plotly. Reutiliza la lógica DSP del proyecto en un módulo independiente.

---

## Estructura

```
webapp/
├── backend/
│   ├── main.py            # FastAPI — endpoints /api/upload, /api/analyze, /api/segment, /api/export/markers
│   └── processing.py      # Funciones DSP (filtros, detección de bursts, segmentación, ERP)
└── frontend/
    ├── index.html         # Single-page UI
    ├── app.js             # Fetch + Plotly.js
    └── style.css
```

---

## Cómo correr

Desde la raíz del repo:

```bash
# Instalar dependencias (si no lo hiciste)
pip install -r requirements.txt

# Levantar servidor
uvicorn webapp.backend.main:app --reload --port 8000
```

Abrir http://localhost:8000 en el navegador.

---

## Flujo

1. **Cargar archivo** (`.txt` whitespace-separated de 9 columnas o `.csv` con encabezados).
2. **Filtros + canales + bursts** → clic en **Analizar**. Se grafica EMG escalado con líneas verdes en cada marcador y una línea roja horizontal con el umbral.
3. **Segmentación** (aparece al detectar ≥1 marcador) → clic en **Segmentar**. Se grafica ERP promedio (EEG a la izquierda, EMG a la derecha).
4. **Exportar marcadores CSV** descarga `markers.csv` con índices y tiempos.

---

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/upload` | Carga archivo, infiere srate, guarda en memoria |
| POST | `/api/analyze` | Aplica filtros a EMG + EEG, detecta bursts |
| POST | `/api/segment` | Segmenta EEG/EMG alrededor de marcadores, promedia |
| GET  | `/api/export/markers` | Descarga CSV de marcadores |
| GET  | `/` | Sirve `index.html` |
| GET  | `/static/*` | Assets del frontend |

---

## Estado

Estado guardado en memoria en un dict global (`STATE` en `main.py`). Pensado para uso local single-user — no persistir entre reinicios, no compartir entre sesiones.

---

## Qué NO incluye la v1

- Validación visual interactiva de parámetros de burst (sliders con preview en tiempo real como en la app de Streamlit).
- Análisis espectral (PSD, espectrograma).
- Detección de artefactos.
- Análisis wavelet.

Pensado para iterarse encima.
