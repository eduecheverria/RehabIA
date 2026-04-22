---
name: generate-docs
description: Genera y actualiza documentación del proyecto RehabIA: READMEs de carpeta, docstrings de módulos Python, y el README principal. Úsalo cuando el usuario pida documentar una carpeta, archivo, función o el proyecto completo, o cuando se añada código nuevo sin documentación.
when_to_use: '"documenta esta carpeta", "añade docstrings", "actualiza el README", "genera la documentación", "documenta este archivo"'
allowed-tools: Read Glob
---

# Generación de Documentación

## Alcance

Elige el flujo según el pedido:

| Pedido | Flujo |
|--------|-------|
| "documenta esta carpeta" / ruta de carpeta | → **Folder README** |
| "documenta este archivo / función / clase" | → **Module docstrings** |
| "actualiza el README principal" | → **Project README** |

Si el pedido es ambiguo, pregunta qué nivel antes de continuar.

---

## Flujo: Folder README

1. **Lista** todos los archivos de la carpeta (usa Glob). Ignora: `__pycache__/`, `*.pyc`, `.git/`, `node_modules/`.

2. **Lee** cada archivo relevante:
   - `.py` → completo; identifica funciones/clases públicas y su propósito real
   - `.ipynb` → celdas; identifica qué analiza o produce cada sección
   - `.md` existente → léelo para preservar info ya documentada
   - `.csv` / `.txt` → primeras 20 líneas para inferir formato y columnas
   - `.json` / `.yaml` → completo para entender configuración

3. **Detecta el contexto del proyecto**: revisa el README raíz para entender cómo encaja esta carpeta en el pipeline de análisis.

4. **Escribe `README.md`** usando la plantilla en [templates.md](templates.md). Guárdalo dentro de la carpeta documentada.

---

## Flujo: Module docstrings

1. **Lee** el archivo `.py` completo.

2. Para cada función o clase pública sin docstring (o con uno desactualizado):
   - Identifica qué hace realmente (desde el código, no solo el nombre)
   - Identifica parámetros, tipos de retorno, y efectos secundarios
   - Nota detalles del dominio (unidades de señal, supuestos de sampling rate)

3. **Escribe docstrings** en estilo Google. Ver [templates.md](templates.md) para el formato exacto.

4. **No** añadas docstrings a helpers privados (`_nombre`) salvo que tengan lógica no obvia.

---

## Flujo: Project README

1. **Lee** el `README.md` actual para preservar secciones válidas.

2. **Lee** los READMEs de subcarpetas (UI/, data/, laboratory/) para contenido actualizado.

3. **Actualiza** usando la plantilla de Project README en [templates.md](templates.md).

4. Preserva el diagrama del pipeline si ya existe; actualiza solo las secciones cuyo contenido cambió.

---

## Reglas

- Escribe en **español**. Usa inglés técnico para código, nombres de funciones y términos de procesamiento de señal (EEG, EMG, ERP, PSD, wavelet, Butterworth, Bereitschaftspotential).
- No copies docstrings literalmente en los READMEs — sintetiza con tus propias palabras.
- No inventes comportamiento: si no puedes leer un archivo, dilo explícitamente.
- Si ya existe un README, actualiza solo las secciones que cambiaron; no borres información válida.
- Mantén los READMEs bajo 150 líneas. Si el contenido es mayor, mueve el material de referencia a un archivo separado.
