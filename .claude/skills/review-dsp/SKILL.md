---
name: review-dsp
description: Revisa código de procesamiento de señal (DSP) del proyecto RehabIA. Verifica parámetros de filtros Butterworth, supuestos de sampling rate, lógica de detección de bursts EMG, correctitud de operaciones EEG/EMG, y posibles bugs en segmentación o análisis ERP/ERDS.
when_to_use: '"revisa este código", "está bien el filtro", "chequea processing.py", "hay algo malo en la detección", "review del DSP", "es correcto este análisis"'
---

# Revisión de código DSP — RehabIA

Contexto del proyecto: `fs = 1000 Hz`, señales EEG (~1–2 µV) y EMG (~0.02 µV), 9 columnas (Tiempo, EEG_1, EEG_2, EMG_1…EMG_6).

Lee el código indicado completo antes de empezar la revisión.

---

## Lista de verificación

### Filtros
- [ ] Frecuencias de corte < Nyquist (`fs/2 = 500 Hz`)
- [ ] Orden Butterworth ≤ 6 (órdenes mayores son numéricamente inestables)
- [ ] Notch a **50 Hz** (red europea), no 60 Hz
- [ ] ERP/ERP: usar `filtfilt` (zero-phase), no `lfilter` (introduce retardo de fase)
- [ ] El filtrado ocurre **antes** de la detección de bursts, no después

### Detección de bursts EMG
- [ ] Umbral calculado sobre señal rectificada o envolvente RMS, no la cruda
- [ ] Período refractario en **muestras**: `refractory_samples = refractory_ms * fs / 1000`
- [ ] Validación de amplitud: ventana pre-burst < ventana post-burst (criterio de activación voluntaria)
- [ ] Los índices de marcadores son `int`, no `float`

### Segmentación y épocas
- [ ] Ventanas no salen del array: `idx - pre >= 0` y `idx + post < len(signal)`
- [ ] Baseline calculado **solo** en la ventana pre-estímulo
- [ ] Épocas con shape consistente `(n_epochs, n_samples)` antes de promediar
- [ ] Baseline correction: restar la media del período baseline por época, no la media global

### Análisis espectral (ERDS / Beta desync)
- [ ] Welch: `nperseg` ≤ longitud de la señal
- [ ] Banda Beta definida como **13–30 Hz**
- [ ] Bereitschaftspotential: ventana de análisis ≥ 1 s antes del marcador EMG
- [ ] ERDS normalizado respecto al baseline espectral, no respecto al pico

### General
- [ ] Shapes de arrays verificables en puntos críticos (usar `.shape` o asserts)
- [ ] No hay división por cero en normalización o baseline (ventana de baseline ≥ 1 muestra)
- [ ] `fs` pasado como parámetro explícito, no hardcodeado dentro de funciones
- [ ] Funciones puras sin efectos secundarios en la lógica de cálculo

---

## Cómo reportar

Para cada problema encontrado:

1. Archivo y número de línea
2. El problema concreto (qué está mal y por qué)
3. La corrección sugerida con código

Si el código es correcto, confirma explícitamente qué puntos de la lista pasaron sin problemas.
