from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pandas as pd

from .common import (
    FignewsSchemaError,
    normalize_label,
    normalize_subtask,
)
from .constants import (
    NOT_APPLICABLE_LABELS,
    UNCERTAIN_LABELS,
)


def _vote_entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())

    if total == 0:
        return 0.0

    nonzero = [count for count in counts.values() if count > 0]

    if len(nonzero) <= 1:
        return 0.0

    raw_entropy = -sum(
        (count / total) * math.log(count / total)
        for count in nonzero
    )

    # Normalized to 0–1 across the labels that received votes.
    return raw_entropy / math.log(len(nonzero))


def _evidence_tier(
    *,
    partition: str,
    status: str,
    total_votes: int,
) -> str:
    if status == "not_applicable":
        return "not_applicable"

    if status in {"uncertain", "tie", "low_agreement"}:
        return "uncertain"

    if total_votes <= 1:
        return "bronze_single"

    if status == "accepted" and partition.upper() == "IAA":
        return "gold_iaa"

    if status == "accepted":
        return "silver_consensus"

    return "silver_soft"


def resolve_consensus(
    vote_counts: Mapping[str, int],
    *,
    partition: str,
    min_agreement: float = 0.60,
    min_votes: int = 1,
) -> dict[str, object]:
    """
    Resolve one post/subtask vote distribution.

    Unclear and Not Applicable remain in the denominator and are allowed
    to win the vote. They are never converted into negative labels.
    """

    if not 0.0 <= min_agreement <= 1.0:
        raise ValueError("min_agreement must be between 0 and 1")

    if min_votes < 1:
        raise ValueError("min_votes must be at least 1")

    counts = {
        normalize_label(label): int(count)
        for label, count in vote_counts.items()
    }

    if any(count < 0 for count in counts.values()):
        raise ValueError("Vote counts cannot be negative")

    total_votes = sum(counts.values())
    top_count = max(counts.values(), default=0)
    leaders = sorted(
        label for label, count in counts.items()
        if count == top_count and count > 0
    )

    top_label = leaders[0] if len(leaders) == 1 else None
    top_share = top_count / total_votes if total_votes else 0.0

    sorted_counts = sorted(counts.values(), reverse=True)
    second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
    second_share = second_count / total_votes if total_votes else 0.0

    consensus_label: str | None = None

    if total_votes == 0:
        status = "no_votes"
    elif len(leaders) > 1:
        status = "tie"
    elif top_label in UNCERTAIN_LABELS:
        status = "uncertain"
    elif top_label in NOT_APPLICABLE_LABELS:
        status = "not_applicable"
    elif total_votes < min_votes:
        status = "insufficient_coverage"
    elif top_share < min_agreement:
        status = "low_agreement"
    else:
        status = "accepted"
        consensus_label = top_label

    return {
        "total_vote_count": total_votes,
        "top_vote_count": top_count,
        "top_label": top_label,
        "top_share": top_share,
        "second_share": second_share,
        "vote_margin": top_share - second_share,
        "vote_entropy": _vote_entropy(counts),
        "consensus_label": consensus_label,
        "consensus_status": status,
        "evidence_tier": _evidence_tier(
            partition=partition,
            status=status,
            total_votes=total_votes,
        ),
        "vote_distribution": json.dumps(
            counts,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def build_consensus_from_long(
    frame: pd.DataFrame,
    *,
    min_agreement: float = 0.60,
    min_votes: int = 1,
) -> pd.DataFrame:
    required = {
        "post_key",
        "subtask",
        "label_normalized",
        "source_language",
        "type",
    }

    missing = sorted(required - set(frame.columns))

    if missing:
        raise FignewsSchemaError(
            f"Long consensus input is missing columns: {missing}"
        )

    records: list[dict[str, object]] = []

    for (post_key, subtask), group in frame.groupby(
        ["post_key", "subtask"],
        sort=True,
    ):
        first = group.iloc[0]
        vote_counts = (
            group["label_normalized"]
            .value_counts(dropna=False)
            .to_dict()
        )

        result = resolve_consensus(
            vote_counts,
            partition=str(first["type"]),
            min_agreement=min_agreement,
            min_votes=min_votes,
        )

        records.append(
            {
                "post_key": post_key,
                "subtask": normalize_subtask(subtask),
                "batch": first.get("batch"),
                "source_language": first["source_language"],
                "id": first.get("id"),
                "type": first["type"],
                "annotation_count": int(group.shape[0]),
                **result,
            }
        )

    return pd.DataFrame.from_records(records)


def build_consensus_from_flat(
    frame: pd.DataFrame,
    *,
    min_agreement: float = 0.60,
    min_votes: int = 1,
) -> pd.DataFrame:
    required = {
        "post_key",
        "batch",
        "source_language",
        "id",
        "type",
    }

    missing = sorted(required - set(frame.columns))

    if missing:
        raise FignewsSchemaError(
            f"Flat consensus input is missing columns: {missing}"
        )

    records: list[dict[str, object]] = []

    for row in frame.to_dict(orient="records"):
        for subtask in ("bias", "propaganda"):
            prefix = f"{subtask}__"

            vote_counts = {
                column.removeprefix(prefix).replace("_", " "): int(value)
                for column, value in row.items()
                if column.startswith(prefix)
            }

            result = resolve_consensus(
                vote_counts,
                partition=str(row["type"]),
                min_agreement=min_agreement,
                min_votes=min_votes,
            )

            records.append(
                {
                    "post_key": row["post_key"],
                    "subtask": subtask,
                    "batch": row["batch"],
                    "source_language": row["source_language"],
                    "id": row["id"],
                    "type": row["type"],
                    **result,
                }
            )

    return pd.DataFrame.from_records(records)