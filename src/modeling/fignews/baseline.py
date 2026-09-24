"""Leakage-safe TF-IDF and logistic-regression baselines for FIGNEWS."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


TaskName = Literal["bias", "propaganda"]
TextRepresentation = Literal[
    "original",
    "english_translation",
    "arabic_translation",
    "original_plus_english",
]

REQUIRED_COLUMNS = frozenset(
    {
        "post_key",
        "subtask",
        "consensus_status",
        "consensus_label",
        "source_language",
        "split",
        "text",
    }
)
EXPECTED_SPLITS = frozenset({"train", "validation", "test"})


class BaselineDataError(ValueError):
    """Raised when the modeling table violates the baseline data contract."""


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration that fully describes one reproducible baseline run."""

    task: TaskName
    text_representation: TextRepresentation = "original"
    random_seed: int = 42
    max_features: int = 50_000
    min_df: int = 2
    max_iter: int = 2_000
    c_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    ngram_ranges: tuple[tuple[int, int], ...] = ((1, 1), (1, 2))


@dataclass(frozen=True)
class BaselineRunResult:
    """Paths and headline scores produced by a completed experiment."""

    task: str
    text_representation: str
    selected_c: float
    selected_ngram_range: tuple[int, int]
    validation_macro_f1: float
    test_macro_f1: float
    output_dir: Path


def _clean_text(series: pd.Series) -> pd.Series:
    result = series.fillna("").astype(str).str.strip()
    return result.mask(result.str.lower().isin({"nan", "none", "null"}), "")


def _build_model_text(
    frame: pd.DataFrame,
    representation: TextRepresentation,
) -> pd.Series:
    if representation == "original":
        return _clean_text(frame["text"])

    if representation == "english_translation":
        if "english_mt" not in frame.columns:
            raise BaselineDataError(
                "english_translation requires the english_mt column"
            )
        return _clean_text(frame["english_mt"])

    if representation == "arabic_translation":
        if "arabic_mt" not in frame.columns:
            raise BaselineDataError(
                "arabic_translation requires the arabic_mt column"
            )
        return _clean_text(frame["arabic_mt"])

    if representation == "original_plus_english":
        if "english_mt" not in frame.columns:
            raise BaselineDataError(
                "original_plus_english requires the english_mt column"
            )
        original = _clean_text(frame["text"])
        english = _clean_text(frame["english_mt"])
        combined = original.copy()
        both_present = original.ne("") & english.ne("")
        combined.loc[both_present] = (
            original.loc[both_present] + " [SEP] " + english.loc[both_present]
        )
        combined.loc[original.eq("")] = english.loc[original.eq("")]
        return combined

    raise BaselineDataError(f"Unsupported text representation: {representation}")


def prepare_task_frame(
    modeling_table: pd.DataFrame,
    config: BaselineConfig,
) -> pd.DataFrame:
    """Filter one task to accepted labels and construct its model input text."""

    missing = sorted(REQUIRED_COLUMNS - set(modeling_table.columns))
    if missing:
        raise BaselineDataError(f"Modeling table is missing columns: {missing}")

    frame = modeling_table.loc[
        modeling_table["subtask"].astype(str).str.lower().eq(config.task)
        & modeling_table["consensus_status"]
        .astype(str)
        .str.lower()
        .eq("accepted")
    ].copy()

    frame["consensus_label"] = (
        frame["consensus_label"].fillna("").astype(str).str.strip().str.lower()
    )
    frame = frame.loc[frame["consensus_label"].ne("")].copy()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["model_text"] = _build_model_text(frame, config.text_representation)
    frame = frame.loc[frame["model_text"].ne("")].copy()

    if frame.empty:
        raise BaselineDataError(
            f"No accepted {config.task} rows contain usable model text"
        )

    unexpected_splits = sorted(set(frame["split"]) - EXPECTED_SPLITS)
    if unexpected_splits:
        raise BaselineDataError(f"Unexpected split values: {unexpected_splits}")

    missing_splits = sorted(EXPECTED_SPLITS - set(frame["split"]))
    if missing_splits:
        raise BaselineDataError(
            f"Accepted {config.task} rows are missing splits: {missing_splits}"
        )

    duplicated = frame["post_key"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, "post_key"].head(5).tolist()
        raise BaselineDataError(
            "A task must contain at most one row per post_key. "
            f"Examples: {examples}"
        )

    split_sets = {
        split: set(frame.loc[frame["split"].eq(split), "post_key"])
        for split in EXPECTED_SPLITS
    }
    if (
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    ):
        raise BaselineDataError("post_key overlap detected between dataset splits")

    train_labels = set(frame.loc[frame["split"].eq("train"), "consensus_label"])
    if len(train_labels) < 2:
        raise BaselineDataError(
            f"Training data needs at least two labels; found {sorted(train_labels)}"
        )

    return frame.sort_values(["split", "post_key"]).reset_index(drop=True)


def _make_pipeline(
    config: BaselineConfig,
    *,
    c_value: float,
    ngram_range: tuple[int, int],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=ngram_range,
                    min_df=config.min_df,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=config.max_iter,
                    random_state=config.random_seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _prediction_table(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(frame["model_text"])
    probabilities = model.predict_proba(frame["model_text"])
    classes = model.named_steps["classifier"].classes_

    result = frame[
        [
            "post_key",
            "source_language",
            "split",
            "consensus_label",
            "model_text",
        ]
    ].copy()
    result = result.rename(columns={"consensus_label": "true_label"})
    result["predicted_label"] = predictions
    result["prediction_confidence"] = probabilities.max(axis=1)

    for index, label in enumerate(classes):
        safe_label = str(label).replace(" ", "_").replace("/", "_")
        result[f"probability__{safe_label}"] = probabilities[:, index]

    return result


def _evaluate(
    predictions: pd.DataFrame,
    label_order: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    true = predictions["true_label"]
    predicted = predictions["predicted_label"]
    report = classification_report(
        true,
        predicted,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true, predicted, labels=label_order)
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"true__{label}" for label in label_order],
        columns=[f"predicted__{label}" for label in label_order],
    )

    metrics: dict[str, object] = {
        "rows": int(len(predictions)),
        "macro_f1": float(
            f1_score(true, predicted, labels=label_order, average="macro", zero_division=0)
        ),
        "accuracy": float((true == predicted).mean()),
        "classification_report": report,
        "rows_by_language": {
            str(key): int(value)
            for key, value in predictions["source_language"].value_counts().items()
        },
    }
    return metrics, matrix_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_baseline_experiment(
    *,
    input_path: str | Path,
    output_root: str | Path,
    config: BaselineConfig,
) -> BaselineRunResult:
    """Select on validation, evaluate once on IAA test, and save all artifacts."""

    input_path = Path(input_path)
    output_dir = Path(output_root) / config.task / config.text_representation
    if not input_path.exists():
        raise FileNotFoundError(f"Modeling table not found: {input_path}")

    table = pd.read_parquet(input_path)
    frame = prepare_task_frame(table, config)
    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("validation")].copy()
    test = frame.loc[frame["split"].eq("test")].copy()
    label_order = sorted(frame["consensus_label"].unique().tolist())

    selection_rows: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_score = -1.0
    best_key: tuple[float, tuple[int, int]] | None = None

    for ngram_range in config.ngram_ranges:
        for c_value in config.c_values:
            model = _make_pipeline(
                config,
                c_value=c_value,
                ngram_range=ngram_range,
            )
            model.fit(train["model_text"], train["consensus_label"])
            validation_predicted = model.predict(validation["model_text"])
            score = float(
                f1_score(
                    validation["consensus_label"],
                    validation_predicted,
                    labels=label_order,
                    average="macro",
                    zero_division=0,
                )
            )
            selection_rows.append(
                {
                    "c": c_value,
                    "ngram_min": ngram_range[0],
                    "ngram_max": ngram_range[1],
                    "validation_macro_f1": score,
                }
            )
            candidate_key = (c_value, ngram_range)
            if score > best_score or (
                score == best_score and (best_key is None or candidate_key < best_key)
            ):
                best_score = score
                best_model = model
                best_key = candidate_key

    if best_model is None or best_key is None:
        raise RuntimeError("Model selection produced no candidate model")

    validation_predictions = _prediction_table(best_model, validation)
    validation_metrics, validation_matrix = _evaluate(
        validation_predictions,
        label_order,
    )

    # Test data is touched only after validation has selected the final candidate.
    test_predictions = _prediction_table(best_model, test)
    test_metrics, test_matrix = _evaluate(test_predictions, label_order)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_dir / "model.joblib")
    pd.DataFrame(selection_rows).sort_values(
        ["validation_macro_f1", "ngram_max", "c"],
        ascending=[False, True, True],
    ).to_csv(output_dir / "selection_results.csv", index=False)
    validation_predictions.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_matrix.to_csv(output_dir / "validation_confusion_matrix.csv")
    test_matrix.to_csv(output_dir / "test_confusion_matrix.csv")
    _write_json(output_dir / "validation_metrics.json", validation_metrics)
    _write_json(output_dir / "test_metrics.json", test_metrics)

    selected_c, selected_ngram = best_key
    run_config = {
        **asdict(config),
        "ngram_ranges": [list(value) for value in config.ngram_ranges],
        "selected_c": selected_c,
        "selected_ngram_range": list(selected_ngram),
        "input_path": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "rows_by_split": {
            str(key): int(value)
            for key, value in frame["split"].value_counts().items()
        },
        "labels": label_order,
        "training_rule": "accepted consensus labels; train split only",
        "selection_rule": "highest validation macro F1; deterministic tie-break",
        "test_rule": "IAA test evaluated once after model selection",
    }
    _write_json(output_dir / "run_config.json", run_config)

    return BaselineRunResult(
        task=config.task,
        text_representation=config.text_representation,
        selected_c=selected_c,
        selected_ngram_range=selected_ngram,
        validation_macro_f1=float(validation_metrics["macro_f1"]),
        test_macro_f1=float(test_metrics["macro_f1"]),
        output_dir=output_dir,
    )
