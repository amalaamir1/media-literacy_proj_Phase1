from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import FignewsSchemaError, require_columns
from .constants import POST_KEY_COLUMNS
from .consensus import build_consensus_from_flat
from .loaders import read_flat_votes, read_source_corpus


def join_consensus_with_text(
    consensus: pd.DataFrame,
    source_posts: pd.DataFrame,
) -> pd.DataFrame:
    """Attach source text to each post/subtask consensus record."""

    consensus_required = ["post_key", "post_uid", "subtask", *POST_KEY_COLUMNS]
    source_required = [
        "post_key",
        "post_uid",
        *POST_KEY_COLUMNS,
        "text",
        "english_mt",
        "arabic_mt",
        "text_sha256",
    ]
    require_columns(
        consensus,
        consensus_required,
        table_name="FIGNEWS consensus",
    )
    require_columns(
        source_posts,
        source_required,
        table_name="FIGNEWS source corpus",
    )

    if source_posts["post_key"].duplicated().any():
        raise FignewsSchemaError(
            "Source corpus must contain one row per post key"
        )

    result = consensus.merge(
        source_posts,
        on="post_key",
        how="left",
        suffixes=("_consensus", ""),
        validate="many_to_one",
        indicator=True,
    )

    missing = result["_merge"].ne("both")

    if missing.any():
        examples = result.loc[missing, "post_key"].head(5).tolist()

        raise FignewsSchemaError(
            f"{int(missing.sum())} consensus rows have no source text. "
            f"Examples: {examples}"
        )

    for column in POST_KEY_COLUMNS:
        consensus_column = f"{column}_consensus"
        mismatched = result[consensus_column].astype(str).ne(
            result[column].astype(str)
        )
        if mismatched.any():
            examples = result.loc[
                mismatched,
                ["post_key", consensus_column, column],
            ].head(5).to_dict(orient="records")
            raise FignewsSchemaError(
                f"Consensus/source metadata mismatch in {column}. "
                f"Examples: {examples}"
            )

    mismatched_uid = result["post_uid_consensus"].astype(str).ne(
        result["post_uid"].astype(str)
    )
    if mismatched_uid.any():
        examples = result.loc[
            mismatched_uid,
            ["post_key", "post_uid_consensus", "post_uid"],
        ].head(5).to_dict(orient="records")
        raise FignewsSchemaError(
            f"Consensus/source post_uid mismatch. Examples: {examples}"
        )

    redundant_columns = [f"{column}_consensus" for column in POST_KEY_COLUMNS]
    return result.drop(
        columns=["_merge", "post_uid_consensus", *redundant_columns]
    )


def build_fignews_modeling_table(
    *,
    flat_votes_path: str | Path,
    main_text_path: str | Path,
    iaa_text_path: str | Path,
    min_agreement: float = 0.60,
    min_votes: int = 1,
) -> pd.DataFrame:
    """
    Construct the auditable post/subtask modeling table.

    This function does not split, translate, vectorize, or train. Those
    steps belong to later pipeline stages.
    """

    votes = read_flat_votes(flat_votes_path)
    posts = read_source_corpus(main_text_path, iaa_text_path)

    consensus = build_consensus_from_flat(
        votes,
        min_agreement=min_agreement,
        min_votes=min_votes,
    )

    return join_consensus_with_text(consensus, posts)
