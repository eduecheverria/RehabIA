# data — Datos crudos y scripts de preprocesamiento

Contiene el dataset original de señales EEG/EMG y el script para convertirlo a un formato más manejable.

---

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `Data.txt` | Dataset original. 9 columnas separadas por espacios, sin encabezado, ~304 000 filas (~5 min a 1000 Hz). |
| `convert_to_csv.py` | Script de conversión: lee `Data.txt`, asigna nombres de columna intuitivos y exporta a CSV. |
| `datos_eeg_emg.csv` | CSV generado por el script (no versionado en git). Regenerar con `python convert_to_csv.py`. |

---

## Formato de `Data.txt`

```
0.000  1.14105  1.23566  -0.01038  -0.02441  -0.01526  -0.02014  -0.02014  -0.02319
0.001  1.23898  1.30197  -0.01022  -0.02185  -0.01526  -0.01996  -0.02100  -0.02563
...
```

| Col | Nombre CSV | Escala típica | Descripción |
|-----|-----------|---------------|-------------|
| 0 | `Tiempo_s` | 0 – 304 s | Vector de tiempo, paso 0.001 s |
| 1 | `EEG_1` | ~1–2 µV | Canal EEG principal |
| 2 | `EEG_2` | ~1–2 µV | Canal EEG secundario |
| 3 | `EMG_1` | ~0.01–0.03 µV | Canal EMG 1 |
| 4 | `EMG_2` | ~0.01–0.03 µV | Canal EMG 2 |
| 5 | `EMG_3` | ~0.01–0.03 µV | Canal EMG 3 |
| 6 | `EMG_4` | ~0.01–0.03 µV | Canal EMG 4 |
| 7 | `EMG_5` | ~0.01–0.03 µV | Canal EMG 5 |
| 8 | `EMG_6` | ~0.01–0.03 µV | Canal EMG 6 |

---

## Cómo usar `convert_to_csv.py`

```bash
# Uso básico (lee Data.txt, genera datos_eeg_emg.csv en la misma carpeta)
python data/convert_to_csv.py

# Especificar archivos
python data/convert_to_csv.py --input Data.txt --output mi_salida.csv
```

El script imprime un resumen al correr:
```
Leyendo Data.txt ...
  Muestras  : 304,011
  Duración  : 304.0 s  (5.1 min)
  Canales   : ['EEG_1', 'EEG_2', 'EMG_1', ..., 'EMG_6']
  Frec.     : 1000 Hz

Guardado en datos_eeg_emg.csv  (22.5 MB)
```

---

## Dependencias

- `convert_to_csv.py` requiere solo `pandas`.
- El CSV generado es consumido por `laboratory/exploracion_inicial.ipynb` y puede cargarse directamente en `UI/app.py`.
