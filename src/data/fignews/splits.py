from __future__ import annotations

import hashlib
import math

import pandas as pd

from .common import FignewsSchemaError


REQUIRED_MODELING_COLUMNS = frozenset(
    {
        "post_key",
        "type",
        "source_language",
        "subtask",
        "consensus_status",
        "consensus_label",
    }
)
EXPECTED_SUBTASKS = frozenset({"bias", "propaganda"})


def _stable_score(post_key: str, seed: int) -> str:
    value = f"{seed}:{post_key}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _target_value(row: pd.Series) -> str:
    status = str(row["consensus_status"]).strip().lower()

    if status == "accepted":
        label = str(row["consensus_label"]).strip().lower()
        if not label or label in {"nan", "none"}:
            raise FignewsSchemaError(
                "An accepted row is missing its consensus label"
            )
        return label

    return f"__{status or 'missing_status'}__"


def _post_level_view(
    modeling_table: pd.DataFrame,
    *,
    min_stratum_size: int,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_MODELING_COLUMNS - set(modeling_table.columns))
    if missing:
        raise FignewsSchemaError(
            f"Modeling table is missing split columns: {missing}"
        )

    duplicated = modeling_table.duplicated(["post_key", "subtask"], keep=False)
    if duplicated.any():
        examples = (
            modeling_table.loc[duplicated, ["post_key", "subtask"]]
            .head(5)
            .to_dict(orient="records")
        )
        raise FignewsSchemaError(
            f"Duplicate post/subtask rows found. Examples: {examples}"
        )

    task_sets = modeling_table.groupby("post_key")["subtask"].agg(frozenset)
    invalid_tasks = task_sets[task_sets != EXPECTED_SUBTASKS]
    if not invalid_tasks.empty:
        raise FignewsSchemaError(
            "Every post must have exactly one bias row and one propaganda row. "
            f"Invalid post examples: {invalid_tasks.index[:5].tolist()}"
        )

    metadata_counts = modeling_table.groupby("post_key")[
        ["type", "source_language"]
    ].nunique(dropna=False)
    if (metadata_counts > 1).any(axis=None):
        raise FignewsSchemaError(
            "Partition and source language must be constant within each post_key"
        )

    working = modeling_table.copy()
    working["split_target"] = working.apply(_target_value, axis=1)

    targets = working.pivot(
        index="post_key",
        columns="subtask",
        values="split_target",
    ).rename_axis(columns=None)
    targets = targets.rename(
        columns={
            "bias": "bias_target",
            "propaganda": "propaganda_target",
        }
    )

    metadata = (
        working[["post_key", "type", "source_language"]]
        .drop_duplicates("post_key")
        .set_index("post_key")
    )
    posts = metadata.join(targets, how="inner").reset_index()
    posts["type"] = posts["type"].astype(str).str.strip().str.upper()
    posts["source_language"] = posts["source_language"].astype(str).str.strip()

    unexpected_partitions = sorted(set(posts["type"]) - {"MAIN", "IAA"})
    if unexpected_partitions:
        raise FignewsSchemaError(
            f"Unexpected FIGNEWS partitions: {unexpected_partitions}"
        )

    posts["full_stratum"] = (
        posts["source_language"]
        + "|bias="
        + posts["bias_target"]
        + "|propaganda="
        + posts["propaganda_target"]
    )

    main_mask = posts["type"].eq("MAIN")
    stratum_counts = posts.loc[main_mask, "full_stratum"].value_counts()
    common_strata = set(stratum_counts[stratum_counts >= min_stratum_size].index)

    posts["stratification_key"] = posts["full_stratum"]
    rare_main = main_mask & ~posts["full_stratum"].isin(common_strata)
    posts.loc[rare_main, "stratification_key"] = (
        posts.loc[rare_main, "source_language"] + "|__rare_target_combination__"
    )
    posts.loc[~main_mask, "stratification_key"] = "IAA|held_out"

    return posts


def _validation_allocations(
    main_posts: pd.DataFrame,
    *,
    validation_fraction: float,
) -> dict[str, int]:
    sizes = main_posts["stratification_key"].value_counts().sort_index()
    target_total = round(len(main_posts) * validation_fraction)

    allocation: dict[str, int] = {}
    remainders: dict[str, float] = {}

    for stratum, size_value in sizes.items():
        size = int(size_value)
        exact = size * validation_fraction
        count = min(math.floor(exact), max(size - 1, 0))
        allocation[stratum] = count
        remainders[stratum] = exact - math.floor(exact)

    while sum(allocation.values()) < target_total:
        candidates = [
            stratum
            for stratum, size in sizes.items()
            if allocation[stratum] < int(size) - 1
        ]
        if not candidates:
            raise FignewsSchemaError(
                "Unable to allocate the requested validation fraction"
            )
        chosen = sorted(
            candidates,
            key=lambda item: (-remainders[item], item),
        )[0]
        allocation[chosen] += 1
        remainders[chosen] = -1.0

    while sum(allocation.values()) > target_total:
        candidates = [stratum for stratum, count in allocation.items() if count > 0]
        chosen = sorted(
            candidates,
            key=lambda item: (remainders[item], item),
        )[0]
        allocation[chosen] -= 1
        remainders[chosen] = 2.0

    return allocation


def build_split_manifest(
    modeling_table: pd.DataFrame,
    *,
    validation_fraction: float = 0.20,
    seed: int = 42,
    min_stratum_size: int = 5,
) -> pd.DataFrame:
    """Assign each FIGNEWS post to train, validation, or held-out test."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if min_stratum_size < 2:
        raise ValueError("min_stratum_size must be at least 2")

    posts = _post_level_view(
        modeling_table,
        min_stratum_size=min_stratum_size,
    )
    posts["split"] = "test"

    main_mask = posts["type"].eq("MAIN")
    main = posts.loc[main_mask].copy()
    if main.empty:
        raise FignewsSchemaError("No MAIN posts are available for train/validation")

    allocations = _validation_allocations(
        main,
        validation_fraction=validation_fraction,
    )
    main["stable_score"] = main["post_key"].map(
        lambda key: _stable_score(str(key), seed)
    )
    main["split"] = "train"

    for stratum, group in main.groupby("stratification_key", sort=True):
        validation_count = allocations[stratum]
        validation_indices = group.sort_values(
            ["stable_score", "post_key"]
        ).index[:validation_count]
        main.loc[validation_indices, "split"] = "validation"

    posts.loc[main.index, "split"] = main["split"]
    posts["random_seed"] = seed
    posts["validation_fraction"] = validation_fraction
    posts["split_strategy"] = (
        "post_key; MAIN=train_validation; IAA=test; "
        "stratified=language+bias_target+propaganda_target"
    )

    manifest_columns = [
        "post_key",
        "type",
        "source_language",
        "bias_target",
        "propaganda_target",
        "stratification_key",
        "split",
        "random_seed",
        "validation_fraction",
        "split_strategy",
    ]
    manifest = posts[manifest_columns].copy()

    split_order = pd.CategoricalDtype(
        ["train", "validation", "test"],
        ordered=True,
    )
    manifest["split"] = manifest["split"].astype(split_order)
    manifest = manifest.sort_values(
        ["split", "source_language", "post_key"]
    ).reset_index(drop=True)
    manifest["split"] = manifest["split"].astype(str)

    if manifest["post_key"].duplicated().any():
        raise FignewsSchemaError("Split manifest contains duplicate post keys")

    return manifest


def attach_split_manifest(
    modeling_table: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one post-level split assignment to both modeling task rows."""

    if manifest["post_key"].duplicated().any():
        raise FignewsSchemaError("Split manifest must have one row per post_key")

    split_columns = ["post_key", "split"]
    result = modeling_table.merge(
        manifest[split_columns],
        on="post_key",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    missing = result["_merge"].ne("both")
    if missing.any():
        examples = result.loc[missing, "post_key"].head(5).tolist()
        raise FignewsSchemaError(
            f"{int(missing.sum())} modeling rows lack a split. Examples: {examples}"
        )

    return result.drop(columns="_merge")
