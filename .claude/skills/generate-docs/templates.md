# Plantillas de Documentación — RehabIA

## Folder README

```markdown
# <nombre-carpeta> — <propósito en una línea>

<2-3 líneas describiendo el rol de esta carpeta dentro del pipeline de RehabIA>

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `archivo.py` | Qué hace y sus funciones principales |
| `datos.csv` | Formato, columnas y qué representan los datos |

## Cómo usar

```bash
# Comandos concretos y copiables para las operaciones más importantes
python UI/app.py
```

## Dependencias

<Qué otras carpetas, archivos o módulos del proyecto necesita esta carpeta>
```

---

## Docstring (Google style)

Adapta la longitud a la complejidad — una sola línea es válida para funciones obvias.

```python
def nombre_funcion(param1: tipo, param2: tipo) -> tipo_retorno:
    """Resumen en una línea de qué hace la función.

    Explicación más larga solo si el comportamiento no es obvio (algoritmo,
    restricción del dominio, limitación conocida).

    Args:
        param1: Descripción. Incluye unidades para datos de señal (ej. "en Hz").
        param2: Descripción.

    Returns:
        Descripción de qué se retorna y su shape/tipo si es array.

    Raises:
        ValueError: Cuándo y por qué se lanza esta excepción.
    """
```

**Notas específicas del dominio RehabIA:**
- Documenta siempre el supuesto de sampling rate (por defecto: 1000 Hz)
- Para funciones de filtrado: documenta frecuencias de corte y orden del filtro
- Para funciones de segmentación: documenta el tamaño de ventana en muestras Y milisegundos
- Para funciones ERP: documenta el período de baseline usado

---

## Project README

```markdown
# RehabIA

<Un párrafo: qué hace la herramienta y para quién es>

## Pipeline de análisis

<Diagrama o lista numerada del pipeline completo>

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

<Tabla con columnas, unidades y sampling rate esperado>

## Referencias

<Papers o métodos clave implementados (Bereitschaftspotential, desincronización Beta, etc.)>
```
