import pandas as pd
import pytest

from data.fignews import (
    FignewsSchemaError,
    build_consensus_from_long,
    build_post_key,
    build_post_uid,
    read_long_annotations,
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


def test_post_uid_uses_only_language_and_source_id() -> None:
    frame = sample_long_annotations()
    frame["post_uid"] = build_post_uid(frame)

    assert frame.loc[0, "post_uid"] == "Arabic::10"
    assert frame["post_uid"].nunique() == 2


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
        "Batch\tSource Language\tID\tType\tText\tEnglish MT\tArabic MT\t"
        "Annotator ID\tBias\tPropaganda\tComments\n"
        'B01\tArabic\t1\tMAIN\t"نص عربي\tArabic text\tنص عربي\t\t\t\t\n',
        encoding="utf-8",
    )

    frame = read_source_texts(source)

    assert len(frame) == 1
    assert frame.iloc[0]["text"] == '"نص عربي'
    assert frame.iloc[0]["english_mt"] == "Arabic text"
    assert "bias" not in frame.columns
    assert "propaganda" not in frame.columns


def test_long_reader_matches_real_release_schema(tmp_path) -> None:
    source = tmp_path / "FIGNEWS-2024-ALL-CLEAN.tsv"
    source.write_text(
        "batch\tsource_language\tid\ttype\tteam_name\tsheet_type\t"
        "annotator_id\ttask\tlabel\tNotes\n"
        "B01\tArabic\t1\tMAIN\tteam\tmain\ta1\t"
        "Propaganda Detection\tPropaganda\tchecked\n",
        encoding="utf-8",
    )

    frame = read_long_annotations(source)

    assert frame.iloc[0]["subtask"] == "propaganda"
    assert frame.iloc[0]["label_normalized"] == "propaganda"
    assert frame.iloc[0]["post_key"] == "B01::Arabic::1::MAIN"
    assert frame.iloc[0]["post_uid"] == "Arabic::1"


def test_long_reader_rejects_label_outside_task_taxonomy(tmp_path) -> None:
    source = tmp_path / "FIGNEWS-2024-ALL-CLEAN.tsv"
    source.write_text(
        "batch\tsource_language\tid\ttype\tteam_name\tsheet_type\t"
        "annotator_id\ttask\tlabel\tNotes\n"
        "B01\tArabic\t1\tMAIN\tteam\tmain\ta1\t"
        "Propaganda Detection\tUnbiased\tchecked\n",
        encoding="utf-8",
    )

    with pytest.raises(FignewsSchemaError, match="outside the task taxonomy"):
        read_long_annotations(source)


def test_source_reader_rejects_schema_drift(tmp_path) -> None:
    source = tmp_path / "FIGNEWS-2024-TEXT-MAIN.tsv"
    source.write_text(
        "Batch\tSource Language\tID\tType\tText\tEnglish MT\tArabic MT\t"
        "Annotator ID\tBias\tPropaganda\tComments\tUnexpected\n"
        "B01\tArabic\t1\tMAIN\ttext\tEnglish\tعربي\t\t\t\t\textra\n",
        encoding="utf-8",
    )

    with pytest.raises(FignewsSchemaError, match="schema mismatch"):
        read_source_texts(source)
