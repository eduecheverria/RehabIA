---
name: writing-documentation
description: Generates and updates documentation for the RehabIA project: folder READMEs, Python module docstrings, and the main project README. Use when the user asks to document a folder, file, function, or the whole project, or when new code is added without documentation.
---

# Writing Documentation

## Scope

This skill handles three documentation levels. Choose based on the request:

| Request | Action |
|---|---|
| "documenta esta carpeta" / folder path | → Follow **Folder README** workflow |
| "documenta este archivo / función / clase" | → Follow **Module Docstrings** workflow |
| "actualiza el README principal" | → Follow **Project README** workflow |

If unclear, ask which level before proceeding.

## Folder README workflow

1. **List** all files in the folder (use Glob). Skip: `__pycache__/`, `*.pyc`, `.git/`, `node_modules/`.

2. **Read** each relevant file:
   - `.py` → full read; identify public functions/classes and their real purpose
   - `.ipynb` → read cells; identify what each section analyzes or produces
   - `.md` existing → read to preserve already-documented info
   - `.csv` / `.txt` → first 20 lines to infer format and columns
   - `.json` / `.yaml` → full read to understand configuration

3. **Detect project context**: check root README.md to understand how this folder fits the larger pipeline.

4. **Write `README.md`** using the template in [templates.md](templates.md#folder-readme-template). Save inside the documented folder.

## Module docstrings workflow

1. **Read** the target `.py` file completely.

2. For each public function or class without a docstring (or with an outdated one):
   - Identify what it actually does (from the code, not just the name)
   - Identify parameters, return types, and any side effects
   - Note domain-specific details (e.g., signal units, sampling rate assumptions)

3. **Write docstrings** using Google style. See [templates.md](templates.md#docstring-template) for format.

4. **Do not** add docstrings to private helpers (`_name`) unless they contain non-obvious logic.

## Project README workflow

1. **Read** current `README.md` to preserve valid sections.

2. **Read** all subfolder READMEs (UI/, data/, laboratory/) for updated content.

3. **Update** the README using the template in [templates.md](templates.md#project-readme-template).

4. Preserve the existing pipeline diagram if present; update only sections whose content changed.

## Rules

- Write in Spanish (project convention). Use technical English for code, function names, and signal-processing terms (EEG, EMG, ERP, PSD, wavelet).
- Do not copy docstrings literally into READMEs — synthesize in your own words.
- Do not invent behavior: if you cannot read a file, say so explicitly.
- If a README already exists, update only changed sections; do not delete valid information.
- Keep READMEs under 150 lines. Move extended reference to a separate file if needed.
