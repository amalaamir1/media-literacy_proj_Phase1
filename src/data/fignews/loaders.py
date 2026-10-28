from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .common import (
    FignewsSchemaError,
    build_post_key,
    build_post_uid,
    normalize_column_name,
    normalize_label,
    normalize_subtask,
    read_delimited,
    require_exact_columns,
    text_sha256,
    validate_post_identity,
)
from .constants import (
    BIAS_LABELS,
    FLAT_HEADER_COLUMNS,
    FLAT_POSITIONAL_COLUMNS,
    LONG_REQUIRED_COLUMNS,
    PROPAGANDA_LABELS,
    SOURCE_MODEL_COLUMNS,
    SOURCE_RELEASE_COLUMNS,
)


def read_long_annotations(path: str | Path) -> pd.DataFrame:
    """Load FIGNEWS-2024-ALL-CLEAN.tsv annotation rows."""

    frame = read_delimited(path)
    frame.columns = [
        normalize_column_name(column) for column in frame.columns
    ]

    require_exact_columns(
        frame,
        list(LONG_REQUIRED_COLUMNS),
        table_name="FIGNEWS long annotations",
    )

    for column in LONG_REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()

    frame["type"] = frame["type"].str.upper()
    validate_post_identity(frame, table_name="FIGNEWS long annotations")
    frame["subtask"] = frame["task"].map(normalize_subtask)
    frame["label_normalized"] = frame["label"].map(normalize_label)

    unexpected_subtasks = sorted(set(frame["subtask"]) - {"bias", "propaganda"})
    if unexpected_subtasks:
        raise FignewsSchemaError(
            "FIGNEWS long annotations contain unexpected tasks: "
            f"{unexpected_subtasks}"
        )

    allowed_labels = {
        "bias": set(BIAS_LABELS),
        "propaganda": set(PROPAGANDA_LABELS),
    }
    invalid_labels = frame.loc[
        ~frame.apply(
            lambda row: row["label_normalized"]
            in allowed_labels[str(row["subtask"])],
            axis=1,
        ),
        ["subtask", "label"],
    ]
    if not invalid_labels.empty:
        examples = invalid_labels.head(5).to_dict(orient="records")
        raise FignewsSchemaError(
            "FIGNEWS long annotations contain labels outside the task taxonomy. "
            f"Examples: {examples}"
        )

    frame["post_key"] = build_post_key(frame)
    frame["post_uid"] = build_post_uid(frame)

    uid_mappings = frame.groupby("post_uid")["post_key"].nunique()
    if uid_mappings.gt(1).any():
        examples = uid_mappings[uid_mappings.gt(1)].index[:5].tolist()
        raise FignewsSchemaError(
            "Long annotations map a post_uid to multiple source records. "
            f"Examples: {examples}"
        )

    return frame


def infer_partition(path: str | Path) -> str:
    filename = Path(path).name.upper()

    if "IAA" in filename:
        return "IAA"

    if "MAIN" in filename:
        return "MAIN"

    raise FignewsSchemaError(
        "Could not infer MAIN or IAA from the source filename. "
        "Pass partition explicitly."
    )


def read_source_texts(
    path: str | Path,
    *,
    partition: str | None = None,
) -> pd.DataFrame:
    """Load a FIGNEWS MAIN or IAA source-text file."""

    frame = read_delimited(path)
    frame.columns = [
        normalize_column_name(column) for column in frame.columns
    ]

    partition = (partition or infer_partition(path)).upper()

    if partition not in {"MAIN", "IAA"}:
        raise ValueError("partition must be MAIN or IAA")

    require_exact_columns(
        frame,
        list(SOURCE_RELEASE_COLUMNS),
        table_name=f"FIGNEWS {partition} source texts",
    )

    for column in SOURCE_RELEASE_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()

    frame["type"] = frame["type"].str.upper()
    validate_post_identity(
        frame,
        table_name=f"FIGNEWS {partition} source texts",
    )
    unexpected_partition_rows = frame["type"].ne(partition)
    if unexpected_partition_rows.any():
        examples = frame.loc[
            unexpected_partition_rows,
            ["batch", "source_language", "id", "type"],
        ].head(5).to_dict(orient="records")
        raise FignewsSchemaError(
            f"FIGNEWS {partition} source file contains rows from another "
            f"partition. Examples: {examples}"
        )

    for column in ("text", "english_mt", "arabic_mt"):
        if frame[column].eq("").any():
            raise FignewsSchemaError(
                f"FIGNEWS {partition} source texts contain empty {column} values"
            )

    # Annotation/export-only columns are intentionally excluded from the
    # modeling table so future feature code cannot accidentally consume them.
    frame = frame.loc[:, SOURCE_MODEL_COLUMNS].copy()
    frame["post_key"] = build_post_key(frame)
    frame["post_uid"] = build_post_uid(frame)
    frame["text_sha256"] = frame["text"].map(text_sha256)

    if frame["post_key"].duplicated().any():
        duplicate_count = int(frame["post_key"].duplicated().sum())
        raise FignewsSchemaError(
            f"{partition} source file contains "
            f"{duplicate_count} duplicate post keys"
        )
    if frame["post_uid"].duplicated().any():
        duplicate_count = int(frame["post_uid"].duplicated().sum())
        raise FignewsSchemaError(
            f"{partition} source file contains "
            f"{duplicate_count} duplicate post_uid values"
        )

    return frame


def read_source_corpus(
    main_path: str | Path,
    iaa_path: str | Path,
) -> pd.DataFrame:
    """Combine MAIN and IAA source posts without mixing their roles."""

    main = read_source_texts(main_path, partition="MAIN")
    iaa = read_source_texts(iaa_path, partition="IAA")

    frame = pd.concat([main, iaa], ignore_index=True, sort=False)

    if frame["post_key"].duplicated().any():
        duplicates = frame.loc[
            frame["post_key"].duplicated(keep=False),
            "post_key",
        ].unique()

        raise FignewsSchemaError(
            "Post keys overlap between MAIN and IAA: "
            f"{duplicates[:5].tolist()}"
        )
    if frame["post_uid"].duplicated().any():
        duplicates = frame.loc[
            frame["post_uid"].duplicated(keep=False),
            "post_uid",
        ].unique()
        raise FignewsSchemaError(
            "Post UIDs overlap between MAIN and IAA: "
            f"{duplicates[:5].tolist()}"
        )

    return frame


def read_flat_votes(path: str | Path) -> pd.DataFrame:
    """
    Load FIGNEWS-2024-ALL-CLEAN-FLAT.tsv positionally.

    Positional parsing is intentional because the official file contains
    duplicate headings and blank separator columns.
    """

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"FIGNEWS flat file not found: {source.resolve()}"
        )

    raw = pd.read_csv(
        source,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )

    expected_columns = len(FLAT_POSITIONAL_COLUMNS)

    if raw.shape[1] != expected_columns:
        raise FignewsSchemaError(
            f"Flat vote file has {raw.shape[1]} columns; "
            f"expected exactly {expected_columns}"
        )

    actual_header = tuple(normalize_label(value) for value in raw.iloc[0].tolist())
    if actual_header != FLAT_HEADER_COLUMNS:
        raise FignewsSchemaError(
            "Flat vote file header does not match the FIGNEWS 2024 schema. "
            f"Expected: {list(FLAT_HEADER_COLUMNS)}; "
            f"actual: {list(actual_header)}"
        )

    frame = raw.iloc[1:, :expected_columns].copy()
    frame.columns = FLAT_POSITIONAL_COLUMNS
    frame = frame.reset_index(drop=True)

    frame = frame.drop(
        columns=["_bias_separator", "_propaganda_separator"]
    )

    vote_columns = [
        column
        for column in frame.columns
        if column.startswith(("bias__", "propaganda__"))
    ]

    for column in vote_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")

        if numeric.isna().any():
            raise FignewsSchemaError(
                f"Non-numeric values found in vote column {column}"
            )

        if (numeric < 0).any():
            raise FignewsSchemaError(
                f"Negative counts found in vote column {column}"
            )

        frame[column] = numeric.astype("int64")

    for column in ("batch", "source_language", "id", "type"):
        frame[column] = frame[column].astype(str).str.strip()

    frame["type"] = frame["type"].str.upper()
    validate_post_identity(frame, table_name="FIGNEWS flat votes")
    frame["post_key"] = build_post_key(frame)
    frame["post_uid"] = build_post_uid(frame)

    if frame["post_key"].duplicated().any():
        raise FignewsSchemaError(
            "Flat vote file contains duplicate post keys"
        )
    if frame["post_uid"].duplicated().any():
        raise FignewsSchemaError(
            "Flat vote file contains duplicate post_uid values"
        )

    return frame
