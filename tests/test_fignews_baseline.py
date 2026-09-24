from __future__ import annotations

import pandas as pd
import pytest

from modeling.fignews.baseline import (
    BaselineConfig,
    BaselineDataError,
    prepare_task_frame,
    run_baseline_experiment,
)


def sample_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    assignments = [
        ("train", "unbiased", "not propaganda"),
        ("train", "biased against israel", "propaganda"),
        ("train", "unbiased", "not propaganda"),
        ("train", "biased against israel", "propaganda"),
        ("validation", "unbiased", "not propaganda"),
        ("validation", "biased against israel", "propaganda"),
        ("test", "unbiased", "not propaganda"),
        ("test", "biased against israel", "propaganda"),
    ]

    for index, (split, bias_label, propaganda_label) in enumerate(assignments):
        for task, label in (
            ("bias", bias_label),
            ("propaganda", propaganda_label),
        ):
            rows.append(
                {
                    "post_key": f"post-{index}",
                    "subtask": task,
                    "consensus_status": "accepted",
                    "consensus_label": label,
                    "source_language": "Arabic" if index % 2 else "English",
                    "split": split,
                    "text": f"original example {index}",
                    "english_mt": f"english example {index}",
                    "arabic_mt": f"arabic example {index}",
                }
            )

    rows.append(
        {
            "post_key": "excluded",
            "subtask": "bias",
            "consensus_status": "tie",
            "consensus_label": None,
            "source_language": "English",
            "split": "train",
            "text": "do not train on this",
            "english_mt": "do not train on this",
            "arabic_mt": "do not train on this",
        }
    )
    return pd.DataFrame(rows)


def test_prepare_filters_task_and_accepted_labels() -> None:
    result = prepare_task_frame(sample_table(), BaselineConfig(task="bias"))

    assert len(result) == 8
    assert set(result["subtask"]) == {"bias"}
    assert set(result["consensus_status"]) == {"accepted"}
    assert "excluded" not in set(result["post_key"])


def test_english_representation_uses_translation() -> None:
    config = BaselineConfig(task="propaganda", text_representation="english_translation")
    result = prepare_task_frame(sample_table(), config)

    assert result["model_text"].str.startswith("english example").all()


def test_missing_required_column_is_rejected() -> None:
    frame = sample_table().drop(columns="split")

    with pytest.raises(BaselineDataError, match="split"):
        prepare_task_frame(frame, BaselineConfig(task="bias"))


def test_missing_partition_is_rejected() -> None:
    frame = sample_table().loc[lambda value: value["split"].ne("test")]

    with pytest.raises(BaselineDataError, match="missing splits"):
        prepare_task_frame(frame, BaselineConfig(task="bias"))


def test_end_to_end_run_saves_reproducible_artifacts(tmp_path) -> None:
    input_path = tmp_path / "modeling_table_with_splits.parquet"
    sample_table().to_parquet(input_path, index=False)
    config = BaselineConfig(
        task="propaganda",
        min_df=1,
        c_values=(1.0,),
        ngram_ranges=((1, 1),),
    )

    result = run_baseline_experiment(
        input_path=input_path,
        output_root=tmp_path / "artifacts",
        config=config,
    )

    assert result.task == "propaganda"
    assert (result.output_dir / "model.joblib").exists()
    assert (result.output_dir / "run_config.json").exists()
    assert (result.output_dir / "validation_metrics.json").exists()
    assert (result.output_dir / "test_metrics.json").exists()
    assert (result.output_dir / "test_predictions.csv").exists()
