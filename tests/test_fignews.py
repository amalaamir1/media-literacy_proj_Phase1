import pandas as pd
import pytest

from data.fignews import (
    FignewsSchemaError,
    build_consensus_from_long,
    build_post_key,
    read_source_texts,
    resolve_consensus,
)


def sample_long_annotations() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "batch": ["B01"] * 4,
            "source_language": ["Arabic"] * 4,
            "id": ["10", "10", "10", "11"],
            "type": ["MAIN"] * 4,
            "subtask": ["propaganda"] * 4,
            "label_normalized": [
                "propaganda",
                "propaganda",
                "not propaganda",
                "unclear",
            ],
            "text": [
                "example one",
                "example one",
                "example one",
                "example two",
            ],
        }
    )

    frame["post_key"] = build_post_key(frame)
    return frame


def test_post_key_groups_annotations() -> None:
    frame = sample_long_annotations()

    assert frame.loc[:2, "post_key"].nunique() == 1
    assert frame["post_key"].nunique() == 2


def test_unclear_wins_instead_of_being_removed() -> None:
    result = resolve_consensus(
        {
            "unclear": 5,
            "propaganda": 1,
        },
        partition="MAIN",
    )

    assert result["total_vote_count"] == 6
    assert result["top_label"] == "unclear"
    assert result["top_share"] == pytest.approx(5 / 6)
    assert result["consensus_label"] is None
    assert result["consensus_status"] == "uncertain"


def test_not_applicable_is_not_a_negative_label() -> None:
    result = resolve_consensus(
        {
            "not applicable": 3,
            "not propaganda": 1,
        },
        partition="MAIN",
    )

    assert result["top_label"] == "not applicable"
    assert result["consensus_label"] is None
    assert result["consensus_status"] == "not_applicable"


def test_accepted_consensus_uses_all_votes() -> None:
    result = resolve_consensus(
        {
            "propaganda": 2,
            "not propaganda": 1,
        },
        partition="MAIN",
        min_agreement=0.60,
    )

    assert result["top_share"] == pytest.approx(2 / 3)
    assert result["consensus_label"] == "propaganda"
    assert result["consensus_status"] == "accepted"


def test_long_consensus_creates_one_row_per_post_and_task() -> None:
    result = build_consensus_from_long(
        sample_long_annotations(),
        min_agreement=0.60,
    )

    accepted = result[
        result["post_key"].str.contains("::10::")
    ].iloc[0]

    uncertain = result[
        result["post_key"].str.contains("::11::")
    ].iloc[0]

    assert accepted["consensus_label"] == "propaganda"
    assert accepted["top_share"] == pytest.approx(2 / 3)
    assert uncertain["consensus_status"] == "uncertain"


def test_missing_consensus_columns_raise_error() -> None:
    frame = sample_long_annotations().drop(
        columns=["source_language"]
    )

    with pytest.raises(
        FignewsSchemaError,
        match="source_language",
    ):
        build_consensus_from_long(frame)


def test_source_reader_treats_quotes_as_text(tmp_path) -> None:
    source = tmp_path / "FIGNEWS-2024-TEXT-MAIN.tsv"
    source.write_text(
        "Batch\tSource Language\tID\tType\tText\tEnglish MT\tArabic MT\n"
        'B01\tArabic\t1\tMAIN\t"نص عربي\tArabic text\tنص عربي\n',
        encoding="utf-8",
    )

    frame = read_source_texts(source)

    assert len(frame) == 1
    assert frame.iloc[0]["text"] == '"نص عربي'
    assert frame.iloc[0]["english_mt"] == "Arabic text"
