# FIGNEWS baseline implementation

These files are staged outside the repository for review. They do not include
or modify any dataset files.

## Copy locations

```text
src/modeling/__init__.py                 -> src/modeling/__init__.py
src/modeling/fignews/__init__.py         -> src/modeling/fignews/__init__.py
src/modeling/fignews/baseline.py         -> src/modeling/fignews/baseline.py
scripts/fignews/train_baseline.py        -> scripts/fignews/train_baseline.py
tests/test_fignews_baseline.py           -> tests/test_fignews_baseline.py
requirements.txt                         -> requirements.txt
pyproject.toml                           -> pyproject.toml
```

Review the dependency-file replacements before copying them because they add
`scikit-learn` and `joblib` to the existing project dependencies.

## Data contract

The trainer expects:

```text
data/processed/fignews/modeling_table_with_splits.parquet
```

Required fields are `post_key`, `subtask`, `consensus_status`,
`consensus_label`, `source_language`, `split`, and `text`. Translation
experiments additionally use `english_mt` or `arabic_mt`.

Only rows with `consensus_status == accepted` enter an experiment. Training
uses only `split == train`; hyperparameters are selected on `validation`; the
IAA `test` rows are evaluated only after selection.

## Run sequence

From the repository root:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m scripts.fignews.create_splits
python -m pytest
python -m scripts.fignews.train_baseline --task both --text-representation original
```

Language comparisons use the same split and seed:

```powershell
python -m scripts.fignews.train_baseline --task both --text-representation english_translation
python -m scripts.fignews.train_baseline --task both --text-representation arabic_translation
python -m scripts.fignews.train_baseline --task both --text-representation original_plus_english
```

## Generated artifacts

Each run writes to:

```text
artifacts/models/fignews/<task>/<text_representation>/
```

The directory contains `model.joblib`, `run_config.json`, validation and test
metrics, predictions, confusion matrices, and all candidate-selection scores.
The configuration records the input SHA-256, library versions, random seed,
selected hyperparameters, labels, and split counts.

## Interpretation boundary

This is a classical baseline, not the final specialized agent. Bias and
propaganda remain separate classifiers. ArAIEval persuasion-technique evidence
will be integrated later at the agent layer rather than merged into FIGNEWS
ground-truth labels.
