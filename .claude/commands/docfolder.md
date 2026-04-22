Lee todos los archivos de la carpeta que se pasa como argumento (o la carpeta actual si no hay argumento) y genera o actualiza un `README.md` dentro de esa carpeta.

## Pasos que debes seguir

1. **Listar** todos los archivos de la carpeta (usa Glob o Bash `ls`). Ignorar: `__pycache__/`, `*.pyc`, `.git/`, `node_modules/`, archivos binarios no textuales salvo imágenes relevantes.

2. **Leer** cada archivo relevante. Para cada tipo:
   - `.py` → leer completo; identificar funciones/clases públicas y su propósito real (no el nombre solo).
   - `.ipynb` → leer celdas; identificar secciones del notebook y qué analiza/produce cada una.
   - `.md` existente → leer para no perder información ya documentada.
   - `.txt` / `.csv` → leer las primeras líneas para inferir formato y columnas.
   - `.json` / `.yaml` → leer para entender configuración.
   - Imágenes → mencionar que existen y su propósito inferido del nombre.

3. **Detectar el contexto del proyecto**: busca en el README raíz o en imports si hay un proyecto mayor al que pertenece esta carpeta, y explica la relación.

4. **Escribir el README.md** con esta estructura exacta:

```
# <nombre de la carpeta> — <propósito en una línea>

<párrafo corto (2-3 líneas) con el rol de esta carpeta dentro del proyecto>

## Contenido

<tabla con columnas: Archivo | Descripción>
Para .py: incluir las funciones principales con firma y qué hacen.
Para .ipynb: listar las secciones.
Para datos: describir columnas/formato.

## Cómo usar

<comandos concretos y copiables para ejecutar lo más importante>

## Dependencias

<qué otros módulos, archivos o carpetas del proyecto necesita esta carpeta>
```

## Reglas

- Escribe en el mismo idioma que el proyecto (detecta por comentarios/variables). Si hay mezcla, usa español.
- No copies docstrings literalmente: sintetiza en lenguaje propio.
- No inventes comportamiento: si no puedes leer un archivo, dilo explícitamente.
- Si ya existe un `README.md`, actualiza solo las secciones que cambiaron; no borres información válida.
- Guarda el resultado como `README.md` dentro de la carpeta documentada.
