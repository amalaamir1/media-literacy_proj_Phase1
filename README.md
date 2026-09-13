# Media Literacy Project — Phase 1

This repository contains the initial FIGNEWS ingestion and consensus pipeline
for the Social Media Literacy project.

## Repository layout

```text
data/raw/fignews/       Original FIGNEWS release files (never modified)
data/processed/fignews/ Generated modeling tables (kept local)
artifacts/              Generated profiles and Excel exports (kept local)
src/data/fignews/       Reusable FIGNEWS loading and consensus code
scripts/fignews/        Commands that produce processed data and exports
tests/                  Unit tests for the data contract
```

## Setup

```powershell
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` for standard Windows Python,
or `.\.venv\bin\Activate.ps1` when using MSYS2 Python. Then install:

```powershell
python -m pip install -r requirements.txt
```

## Validate the code

```powershell
python -m pytest
```

## Build the processed modeling table

Run this command from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m scripts.fignews.build_processed
```

This produces:

- `data/processed/fignews/modeling_table.parquet`
- `artifacts/profiles/fignews_modeling_table_profile.json`

These files are reproducible and intentionally excluded from Git.

## Create a Unicode-safe Excel copy

```powershell
python -m scripts.fignews.convert_tsv_to_xlsx
```

The export is written to
`artifacts/exports/fignews/FIGNEWS-2024-TEXT-MAIN-UNICODE.xlsx` and is intended
only for human inspection. The pipeline always reads the original TSV files.
