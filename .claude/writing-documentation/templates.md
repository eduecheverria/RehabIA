# Documentation Templates

## Folder README template

```markdown
# <folder-name> — <one-line purpose>

<2-3 lines describing the role of this folder within the RehabIA pipeline>

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `file.py` | What it does and its main public functions |
| `data.csv` | Format, columns, and what the data represents |

## Cómo usar

```bash
# Concrete, copy-paste commands for the most important operations
python UI/app.py
```

## Dependencias

<Which other folders, files, or modules this folder requires>
```

---

## Docstring template

Google style. Adapt length to complexity — a one-liner is fine for obvious functions.

```python
def function_name(param1: type, param2: type) -> return_type:
    """One-line summary of what the function does.

    Longer explanation only if the behavior is non-obvious (algorithm,
    domain constraint, known limitation).

    Args:
        param1: Description. Include units for signal data (e.g., "in Hz").
        param2: Description.

    Returns:
        Description of what is returned and its shape/type if array.

    Raises:
        ValueError: When and why this is raised.
    """
```

**Domain-specific notes for RehabIA:**
- Always document sampling rate assumptions (default: 1000 Hz)
- For filter functions: document cutoff frequencies and filter order
- For segmentation functions: document window size in samples AND milliseconds
- For ERP functions: document the baseline period used

---

## Project README template

```markdown
# RehabIA

<One paragraph: what the tool does and who it is for>

## Pipeline

<Diagram or numbered list of the full analysis pipeline>

## Estructura

| Carpeta | Función |
|---------|---------|
| `UI/` | ... |
| `data/` | ... |
| `laboratory/` | ... |

## Instalación

```bash
pip install -r requirements.txt
streamlit run UI/app.py
```

## Formato de datos

<Table with columns, units, and expected sampling rate>

## Referencias

<Key papers or methods implemented (Bereitschaftspotential, Beta desync, etc.)>
```
