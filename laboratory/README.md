# laboratory — Exploración y análisis experimental

Espacio de trabajo para explorar los datos libremente, probar parámetros y entender la señal antes de llevarla a la app de producción.

---

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `exploracion_inicial.ipynb` | Notebook de exploración completa del dataset. Ver secciones abajo. |
| `wavelets_maps.png` | Imagen de referencia de mapas espectrales por análisis wavelet (Morlet). |

---

## Secciones del notebook `exploracion_inicial.ipynb`

| # | Sección | Qué hace |
|---|---------|----------|
| 1 | Carga y descripción general | Lee el CSV, imprime shape, duración, frecuencia de muestreo y tabla `describe()`. |
| 2 | Vista completa de todos los canales | Gráfico de los 8 canales (2 EEG + 6 EMG) a lo largo de todo el registro. Útil para detectar artefactos o canales saturados. |
| 3 | Estadísticas por canal | Tabla con RMS, std, rango, kurtosis y skewness. Gráficos de barras comparativos. |
| 4 | PSD — Densidad espectral de potencia | PSD de EEG (con bandas delta/theta/alpha/beta/gamma coloreadas) y PSD de EMG. Escala logarítmica. |
| 5 | Zoom: primeros 10 segundos | Morfología a nivel de muestra para detectar ruido de línea (50 Hz) y artefactos. |
| 6 | Ranking de canales EMG | Calcula kurtosis y energía en banda 20–150 Hz por canal EMG. Devuelve el mejor candidato para burst detection y lo visualiza rectificado y normalizado [0,1]. |
| 7 | EEG filtrado por bandas | EEG_1 y EEG_2 filtrados en Alpha (8–13 Hz) y Beta (13–30 Hz) — las bandas de interés clínico. |
| 8 | Espectrograma EEG | Vista tiempo-frecuencia (0–80 Hz) de ambos canales EEG. Permite ver si hay modulación espectral a lo largo del tiempo. |
| 9 | Resumen y próximos pasos | Imprime resumen del dataset y el canal EMG recomendado para el siguiente paso del pipeline. |

---

## Cómo usar

```bash
# Asegurarse de tener el CSV generado
python data/convert_to_csv.py

# Abrir el notebook
jupyter notebook laboratory/exploracion_inicial.ipynb
```

El notebook lee el CSV desde `../data/datos_eeg_emg.csv` (ruta relativa desde `laboratory/`).

---

## Dependencias

- Requiere que `data/datos_eeg_emg.csv` exista (generado por `data/convert_to_csv.py`).
- Paquetes: `numpy`, `pandas`, `matplotlib`, `scipy`.
