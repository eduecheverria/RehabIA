# UI — Interfaz de usuario (Streamlit)

Contiene la aplicación web interactiva para análisis de señales EEG/EMG y la librería de procesamiento de señal que la sostiene.

---

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `app.py` | App principal. Flujo completo: carga de datos → filtrado → configuración de bursts → detección de marcadores EMG → segmentación EEG → ERP y análisis espectral. |
| `app_edu.py` | Versión educativa de la app. Misma lógica que `app.py` pero con selector de columna por nombre (no por índice) y manejo de errores más explícito. Ideal para demos o primeras pruebas. |
| `processing.py` | Librería de funciones DSP. No tiene interfaz gráfica: es importada por ambas apps. |

---

## Funciones en `processing.py`

| Función | Entrada | Salida | Descripción |
|---------|---------|--------|-------------|
| `apply_filters(data, srate, highpass, lowpass, notch)` | array + parámetros de corte | array filtrado | Butterworth orden 4 (high/lowpass) + IIR notch (Q=30). |
| `detect_markers(emg, srate, threshold, ...)` | EMG + 6 parámetros de burst | array de índices | Rectifica, normaliza, umbraliza, valida amplitud antes/después, aplica refractory period. |
| `calculate_features(data, srate)` | array | dict | RMS, media, std, varianza, kurtosis, skewness, potencia total, frecuencia media/mediana, potencia por banda (delta/theta/alpha/beta/gamma). |
| `spectral_analysis(data, srate)` | array | dict | PSD por método de Welch + espectrograma (scipy). Devuelve `freqs`, `psd`, `f_spec`, `t_spec`, `spectrogram`, `peak_freq`. |
| `segment_data(eeg, emg, markers, window, onset, srate)` | señales + marcadores | (eeg_epochs, emg_epochs) | Corta ventanas de `window` segundos alrededor de cada marcador, con `onset` como tiempo cero. |
| `epoch_and_average(eeg_epochs, emg_epochs, srate, baseline)` | epochs arrays | (eeg_avg, emg_avg) | Baseline correction en EEG (resta media de los primeros `baseline` s) y promedio de todas las épocas. |
| `reorder_and_split(eeg_epochs, n_groups)` | epochs array | lista de promedios | Reordena aleatoriamente los trials y calcula promedio por grupo (para validar consistencia). |
| `detect_artifacts(data, srate, method)` | array + método | array de índices | Detecta artefactos por amplitud extrema, gradiente abrupto o estadística por ventana (kurtosis). |
| `create_emg_timeseries_with_markers(emg, markers, srate)` | EMG + marcadores | DataFrame | Arma DataFrame con columnas: `Tiempo_s`, `EMG_Filtrado`, `EMG_Escalado`, `Marcadores`, `Contexto_Marcador`, `Numero_Marcador`. |
| `create_synced_controls(param_name, ...)` | parámetros UI | valor sincronizado | Control Streamlit con slider + input numérico sincronizados via `session_state`. |
| `com_sin(wtime, freq)` | tiempo + frecuencias | array complejo | Sinusoides complejas para análisis wavelet (Morlet). |
| `com_gau(wtime, s)` | tiempo + escalas | array real | Ventanas gaussianas para convolución wavelet. |

---

## Cómo usar

```bash
# Desde la raíz del proyecto
streamlit run UI/app.py

# Versión educativa
streamlit run UI/app_edu.py
```

La app espera un archivo `.txt` o `.csv` con columnas en el orden:
`Tiempo_s | EEG_1 | EEG_2 | EMG_1 | EMG_2 | EMG_3 | EMG_4 | EMG_5 | EMG_6`

Usar el script `data/convert_to_csv.py` para generar el CSV desde `Data.txt`.

---

## Dependencias

- `processing.py` debe estar en el mismo directorio que `app.py` y `app_edu.py` (import relativo).
- Paquetes: `streamlit`, `pandas`, `numpy`, `scipy`, `plotly`.
