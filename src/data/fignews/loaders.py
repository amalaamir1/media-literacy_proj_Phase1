from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .common import (
    FignewsSchemaError,
    build_post_key,
    normalize_column_name,
    normalize_label,
    normalize_subtask,
    read_delimited,
    require_columns,
    text_sha256,
)
from .constants import (
    FLAT_POSITIONAL_COLUMNS,
    LONG_REQUIRED_COLUMNS,
    SOURCE_REQUIRED_COLUMNS,
)


def read_long_annotations(path: str | Path) -> pd.DataFrame:
    """Load FIGNEWS-2024-ALL-CLEAN.tsv annotation rows."""

    frame = read_delimited(path)
    frame.columns = [
        normalize_column_name(column) for column in frame.columns
    ]

    # The official metrics script says `task`; the dataset uses `subtask`.
    if "task" in frame.columns and "subtask" not in frame.columns:
        frame = frame.rename(columns={"task": "subtask"})

    require_columns(
        frame,
        list(LONG_REQUIRED_COLUMNS),
        table_name="FIGNEWS long annotations",
    )

    for column in LONG_REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()

    frame["subtask"] = frame["subtask"].map(normalize_subtask)
    frame["label_normalized"] = frame["label"].map(normalize_label)
    frame["post_key"] = build_post_key(frame)
    frame["text_sha256"] = frame["text"].map(text_sha256)

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

    if "type" not in frame.columns:
        frame["type"] = partition

    frame["type"] = frame["type"].replace("", partition)

    require_columns(
        frame,
        list(SOURCE_REQUIRED_COLUMNS),
        table_name=f"FIGNEWS {partition} source texts",
    )

    for column in SOURCE_REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()

    frame["type"] = frame["type"].str.upper()
    frame["post_key"] = build_post_key(frame)
    frame["text_sha256"] = frame["text"].map(text_sha256)

    if frame["post_key"].duplicated().any():
        duplicate_count = int(frame["post_key"].duplicated().sum())
        raise FignewsSchemaError(
            f"{partition} source file contains "
            f"{duplicate_count} duplicate post keys"
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

    if raw.shape[1] < expected_columns:
        raise FignewsSchemaError(
            f"Flat vote file has {raw.shape[1]} columns; "
            f"expected at least {expected_columns}"
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
    frame["post_key"] = build_post_key(frame)

    if frame["post_key"].duplicated().any():
        raise FignewsSchemaError(
            "Flat vote file contains duplicate post keys"
        )

    return frame
