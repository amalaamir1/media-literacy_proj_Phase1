import pandas as pd
import pytest

from data.fignews import FignewsSchemaError
from data.fignews.splits import attach_split_manifest, build_split_manifest


def sample_modeling_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for partition, count in (("MAIN", 40), ("IAA", 4)):
        for index in range(count):
            language = "Arabic" if index % 2 == 0 else "English"
            post_key = f"B01::{language}::{index}::{partition}"

            for subtask in ("bias", "propaganda"):
                rows.append(
                    {
                        "post_key": post_key,
                        "type": partition,
                        "source_language": language,
                        "subtask": subtask,
                        "consensus_status": "accepted",
                        "consensus_label": (
                            "unbiased" if subtask == "bias" else "not propaganda"
                        ),
                    }
                )

    return pd.DataFrame(rows)


def test_manifest_is_post_level_and_reserves_iaa_for_test() -> None:
    manifest = build_split_manifest(sample_modeling_table(), seed=42)

    assert len(manifest) == 44
    assert manifest["post_key"].is_unique
    assert manifest["split"].value_counts().to_dict() == {
        "train": 32,
        "validation": 8,
        "test": 4,
    }
    assert set(manifest.loc[manifest["type"].eq("IAA"), "split"]) == {"test"}


def test_manifest_is_reproducible() -> None:
    frame = sample_modeling_table()
    first = build_split_manifest(frame, seed=42)
    second = build_split_manifest(frame, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_both_subtasks_receive_the_same_post_split() -> None:
    frame = sample_modeling_table()
    manifest = build_split_manifest(frame)
    result = attach_split_manifest(frame, manifest)

    splits_per_post = result.groupby("post_key")["split"].nunique()
    assert splits_per_post.eq(1).all()


def test_duplicate_post_subtask_rows_are_rejected() -> None:
    frame = sample_modeling_table()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(FignewsSchemaError, match="Duplicate post/subtask"):
        build_split_manifest(frame)
