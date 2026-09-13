from __future__ import annotations

import pandas as pd

from .common import FignewsSchemaError
from .consensus import build_consensus_from_flat
from .loaders import read_flat_votes, read_source_corpus


def join_consensus_with_text(
    consensus: pd.DataFrame,
    source_posts: pd.DataFrame,
) -> pd.DataFrame:
    """Attach source text to each post/subtask consensus record."""

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

    return result.drop(columns="_merge")


def build_fignews_modeling_table(
    *,
    flat_votes_path: str,
    main_text_path: str,
    iaa_text_path: str,
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