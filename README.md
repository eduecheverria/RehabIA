# RehabIA — Analizador de Señales EEG/EMG

Herramienta de análisis de señales cerebrales (EEG) y musculares (EMG) orientada a rehabilitación motora. Permite detectar activación muscular (bursts EMG), segmentar el EEG alrededor de esos eventos y analizar fenómenos pre-movimiento como el **Bereitschaftspotential (BP)** y la **desincronización en banda Beta (13–30 Hz)**.

---

## Estructura del proyecto

```
RehabIA/
│
├── UI/                         # Aplicación Streamlit (interfaz de usuario)
│   ├── app.py                  # App principal — carga datos, filtra, detecta bursts, analiza ERP
│   ├── app_edu.py              # Versión educativa — misma lógica, UI más guiada y comentada
│   └── processing.py           # Funciones de procesamiento de señal (filtros, features, detección)
│
├── data/                       # Datos crudos y scripts de preprocesamiento
│   ├── Data.txt                # Dataset original (9 col: Tiempo + 2 EEG + 6 EMG, 1000 Hz)
│   ├── convert_to_csv.py       # Convierte Data.txt a CSV con nombres de columna intuitivos
│   └── datos_eeg_emg.csv       # CSV generado (no versionado en git — correr convert_to_csv.py)
│
├── laboratory/                 # Exploración y análisis experimental
│   ├── exploracion_inicial.ipynb   # Notebook de exploración: estadísticas, PSD, ranking de canales
│   └── wavelets_maps.png           # Referencia visual de análisis wavelet
│
├── nota-reunion-09-04.txt      # Notas de reunión clínica (contexto del proyecto)
├── requirements.txt            # Dependencias Python
└── .gitignore
```

---

## Formato de los datos

El archivo `Data.txt` tiene **9 columnas separadas por espacios**, sin encabezado, a **1000 Hz**:

| Columna | Nombre | Tipo | Descripción |
|---------|--------|------|-------------|
| 0 | `Tiempo_s` | float | Vector de tiempo (s), paso de 0.001 s |
| 1 | `EEG_1` | float | Canal EEG principal (~1–2 µV) |
| 2 | `EEG_2` | float | Canal EEG secundario (~1–2 µV) |
| 3–8 | `EMG_1..EMG_6` | float | Canales EMG musculares (~0.02 µV) |

El dataset actual tiene **304 000 muestras** (~5 minutos de grabación).

---

## Instalación

```bash
# Clonar el repo
git clone <url>
cd RehabIA

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### 1. Convertir datos a CSV
```bash
python data/convert_to_csv.py
# Genera: data/datos_eeg_emg.csv
```

### 2. Correr la app
```bash
streamlit run UI/app.py
```

### 3. Exploración de datos
```bash
jupyter notebook laboratory/exploracion_inicial.ipynb
```

---

## Pipeline de análisis

```
Data.txt
   │
   ▼
[Cargar y filtrar]          → highpass 1 Hz, lowpass 100 Hz, notch 50 Hz
   │
   ▼
[Detectar bursts EMG]       → umbralización con validación de amplitud antes/después
   │
   ▼
[Segmentar EEG]             → ventanas alrededor de cada marcador
   │
   ▼
[ERP promedio]              → Bereitschaftspotential (onda lenta pre-movimiento)
   │
   ▼
[ERDS espectral]            → caída de potencia en banda Beta (13–30 Hz) pre-movimiento
```

---

## Componentes principales

### `UI/app.py`
App Streamlit completa. Incluye:
- Carga y preview del archivo `.txt`/`.csv`
- Filtros configurables (pasa-alto, pasa-bajo, notch)
- Configuración visual interactiva de parámetros de burst (sliders + preview en tiempo real)
- Detección de marcadores EMG con análisis de calidad por marcador
- Segmentación de EEG y cálculo de ERP promedio
- Exportación de marcadores a CSV

### `UI/processing.py`
Librería de funciones de DSP:

| Función | Descripción |
|---------|-------------|
| `apply_filters` | Butterworth highpass + lowpass + IIR notch |
| `detect_markers` | Umbralización + validación amplitud antes/después + refractory period |
| `calculate_features` | RMS, media, std, potencia por banda EEG |
| `spectral_analysis` | PSD (Welch) + espectrograma |
| `segment_data` | Segmenta EEG/EMG en ventanas alrededor de marcadores |
| `epoch_and_average` | Baseline correction + promedio de épocas |
| `detect_artifacts` | Detección por amplitud, gradiente o estadística |
| `com_sin`, `com_gau` | Sinusoides complejas y gaussianas para análisis wavelet |

### `UI/app_edu.py`
Versión simplificada ideal para demostración. Usa nombres de columna en vez de índices numéricos y tiene comentarios más detallados.

---

## Contexto clínico

> **Notas de reunión (09-04):** el objetivo es detectar activación muscular voluntaria mediante EMG y usarla para analizar la preparación motora en EEG. La banda Beta (13–30 Hz) cae antes de movimientos voluntarios (desincronización). En Parkinson esta banda está aumentada. El pipeline busca detectar ese patrón para cerrar el lazo en dispositivos de rehabilitación.

---

## Dependencias principales

| Paquete | Uso |
|---------|-----|
| `streamlit` | Interfaz web interactiva |
| `numpy` / `scipy` | Procesamiento de señal, filtros, FFT |
| `pandas` | Manejo de datos tabulares |
| `plotly` | Gráficos interactivos en la app |
| `matplotlib` | Gráficos en el notebook |
