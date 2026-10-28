# Media Literacy Project — Phase 1

This repository implements the FIGNEWS portion of Phase 1: validated source
loading, annotation consensus, post-level train/validation/test splits, and
multilingual TF-IDF plus logistic-regression baselines for bias and propaganda.

The classifier outputs are specialist signals. They do not establish factual
truth, author intent, or a universal propaganda verdict.

## Repository layout

```text
data/raw/fignews/       Original FIGNEWS release files
data/processed/fignews/ Generated modeling and split tables (ignored)
artifacts/              Generated profiles, models, metrics, and exports (ignored)
src/data/fignews/       FIGNEWS schemas, loading, consensus, and splitting
src/modeling/fignews/   Baseline training and evaluation
scripts/fignews/        Parameterized command-line entry points
tests/                  Unit and data-contract tests
```

## Setup

Python 3.11–3.13 is supported. Create and activate a virtual environment, then
install the project and development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pyproject.toml` is the dependency source of truth. `requirements.txt` installs
the project in editable mode with its test dependencies, so no manual
`PYTHONPATH` setting is required.

## Validate the code

```powershell
python -m pytest
```

## Run the FIGNEWS baseline pipeline

Run commands from the repository root.

### 1. Build the modeling table

```powershell
python -m scripts.fignews.build_processed
```

Defaults:

- input votes: `data/raw/fignews/FIGNEWS-2024-ALL-CLEAN-FLAT.tsv`
- MAIN text: `data/raw/fignews/FIGNEWS-2024-TEXT-MAIN.tsv`
- IAA text: `data/raw/fignews/FIGNEWS-2024-TEXT-IAA.tsv`
- output: `data/processed/fignews/modeling_table.parquet`
- profile: `artifacts/profiles/fignews_modeling_table_profile.json`
- consensus: minimum agreement `0.60`, minimum votes `1`

Every path and consensus setting can be overridden:

```powershell
python -m scripts.fignews.build_processed `
  --flat-votes <flat-vote-file> `
  --main-text <main-text-file> `
  --iaa-text <iaa-text-file> `
  --output <modeling-table.parquet> `
  --profile <profile.json> `
  --min-agreement 0.60 `
  --min-votes 1
```

The profile records source paths, source SHA-256 values, schema version, and
the consensus policy. Existing outputs are protected unless `--overwrite` is
explicitly supplied.

### 2. Create post-level splits

```powershell
python -m scripts.fignews.create_splits
```

MAIN posts are deterministically divided into train and validation. IAA posts
are reserved for test. Both task rows for a post always receive the same split.
The split profile records the input hash and a content-stable manifest hash.

Paths and split settings are configurable through `--input`, `--output-dir`,
`--profile`, `--validation-fraction`, `--seed`, and `--min-stratum-size`.
Existing split outputs are protected unless `--overwrite` is supplied.

### 3. Train both baselines

```powershell
python -m scripts.fignews.train_baseline --task both
```

The default policy trains on accepted `bronze_single`, `silver_consensus`, and
`gold_iaa` rows. This reproduces the established baseline. To exclude
single-vote labels in a new policy run:

```powershell
python -m scripts.fignews.train_baseline `
  --task both `
  --eligible-evidence-tiers silver_consensus gold_iaa `
  --output-root <new-output-directory>
```

Existing task output directories are not overwritten unless `--overwrite` is
explicitly supplied. Prefer a new output root for policy experiments.

## Current reference results

| Task | Validation macro F1 | IAA-test macro F1 |
|---|---:|---:|
| Bias, five classes | 0.433 | 0.362 |
| Propaganda, binary | 0.661 | 0.679 |

Bias test results require caution: two taxonomy classes have zero accepted IAA
examples. Evaluation artifacts therefore include both taxonomy-wide macro F1
and observed-support macro F1. Saved maximum class probabilities are
uncalibrated and must not be presented as user confidence.

## Data contracts

- The FIGNEWS 2024 release headers are checked exactly.
- Source identity fields must be non-empty and cannot contain the reserved
  `::` separator.
- `post_uid` is `source_language::id` and must map one-to-one to the legacy
  `post_key` throughout this release.
- Partitions are limited to MAIN and IAA; source languages are limited to the
  five languages in this release.
- Long annotation labels must belong to the corresponding task taxonomy.
- Vote and source metadata must agree before they are joined.
- Export-only annotation columns from source-text files are not included in the
  modeling table.
- Split manifests must match the modeling-table post keys exactly.
- Train/validation rows must come from MAIN and test rows from IAA.

`post_key` remains the compatibility identifier
`batch::source_language::id::type`. It is retained so existing splits,
predictions, and artifacts remain linkable. New outputs also carry the simpler
`post_uid` (`source_language::id`). In Databricks, a deterministic GUID can be
added alongside `post_uid` using the dataset release, language, and source ID;
the GUID should not replace these lineage fields.

## Optional Unicode-safe Excel copies

Excel exports are for human inspection only; the pipeline always reads the
original TSV files.

```powershell
python -m scripts.fignews.convert_tsv_to_xlsx
```

Use `--raw-dir`, `--output-dir`, and `--files` to override locations or select
files. Existing exports are protected unless `--overwrite` is supplied.
