---
name: commit
description: Stagea y commitea los cambios actuales con un mensaje descriptivo siguiendo las convenciones del proyecto RehabIA.
disable-model-invocation: true
allowed-tools: Bash(git status) Bash(git diff *) Bash(git add *) Bash(git commit *) Bash(git log *)
---

# Commit

Sigue estos pasos en orden:

## 1. Revisa el estado

```bash
git status
git diff
git diff --staged
```

## 2. Analiza qué cambió

Identifica el propósito real del cambio. Si hay múltiples cambios no relacionados, commitea por separado.

## 3. Stagea los archivos relevantes

Nunca uses `git add .` a ciegas. Agrega solo los archivos que forman parte de este cambio:

```bash
git add <archivos específicos>
```

**Nunca commitees:**
- Archivos de datos (`*.csv`, `Data.txt`) — están en .gitignore
- `__pycache__/`, `*.pyc`
- `.env` o credenciales

## 4. Escribe el mensaje

Formato: `<tipo>: <descripción en ≤72 caracteres>`

Tipos válidos:

| Tipo | Cuándo usarlo |
|------|---------------|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `refactor` | Mejora sin cambiar comportamiento |
| `docs` | Documentación |
| `test` | Tests |
| `chore` | Configuración, dependencias |

Ejemplos de buenos mensajes:
- `feat: detección de bursts con período refractario configurable`
- `fix: baseline correction cuando ventana pre-burst es cero`
- `docs: docstrings en processing.py para funciones de filtrado`
- `refactor: separar lógica de filtrado en processing.py`

## 5. Ejecuta el commit

```bash
git commit -m "<tipo>: <descripción>"
```

Confirma con `git log --oneline -3` que el commit quedó registrado.
