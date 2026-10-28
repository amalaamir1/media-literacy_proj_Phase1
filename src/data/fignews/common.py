from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd

from .constants import (
    EXPECTED_PARTITIONS,
    EXPECTED_SOURCE_LANGUAGES,
    FIGNEWS_CONTEXT_COLUMNS,
    POST_KEY_COLUMNS,
    POST_KEY_SEPARATOR,
    POST_UID_COLUMNS,
)


class FignewsSchemaError(ValueError):
    """Raised when a FIGNEWS file violates the expected data contract."""


def normalize_column_name(value: object) -> str:
    name = str(value).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def normalize_label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_subtask(value: object) -> str:
    name = normalize_label(value)

    aliases = {
        "bias detection": "bias",
        "bias_detection": "bias",
        "propaganda detection": "propaganda",
        "propaganda_detection": "propaganda",
    }

    return aliases.get(name, name)


def require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...] | list[str],
    *,
    table_name: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))

    if missing:
        raise FignewsSchemaError(
            f"{table_name} is missing required columns: {missing}"
        )


def require_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...] | list[str],
    *,
    table_name: str,
) -> None:
    expected_list = list(expected)
    actual_list = list(frame.columns)
    if actual_list != expected_list:
        missing = sorted(set(expected_list) - set(actual_list))
        unexpected = sorted(set(actual_list) - set(expected_list))
        raise FignewsSchemaError(
            f"{table_name} schema mismatch. Missing: {missing}; "
            f"unexpected: {unexpected}; expected order: {expected_list}; "
            f"actual order: {actual_list}"
        )


def validate_post_identity(
    frame: pd.DataFrame,
    *,
    table_name: str,
) -> None:
    require_columns(frame, list(FIGNEWS_CONTEXT_COLUMNS), table_name=table_name)

    for column in FIGNEWS_CONTEXT_COLUMNS:
        values = frame[column].astype(str).str.strip()
        if values.eq("").any():
            raise FignewsSchemaError(
                f"{table_name} contains empty values in identity column {column}"
            )
        if values.str.contains(POST_KEY_SEPARATOR, regex=False).any():
            raise FignewsSchemaError(
                f"{table_name} contains the reserved separator "
                f"{POST_KEY_SEPARATOR!r} in identity column {column}"
            )

    partitions = set(frame["type"].astype(str).str.strip().str.upper())
    unexpected_partitions = sorted(partitions - EXPECTED_PARTITIONS)
    if unexpected_partitions:
        raise FignewsSchemaError(
            f"{table_name} contains unexpected partitions: "
            f"{unexpected_partitions}"
        )

    languages = set(frame["source_language"].astype(str).str.strip())
    unexpected_languages = sorted(languages - EXPECTED_SOURCE_LANGUAGES)
    if unexpected_languages:
        raise FignewsSchemaError(
            f"{table_name} contains unexpected source languages: "
            f"{unexpected_languages}"
        )


def build_post_key(frame: pd.DataFrame) -> pd.Series:
    validate_post_identity(frame, table_name="FIGNEWS table")

    return frame.loc[:, POST_KEY_COLUMNS].astype(str).apply(
        lambda row: POST_KEY_SEPARATOR.join(value.strip() for value in row),
        axis=1,
    )


def build_post_uid(frame: pd.DataFrame) -> pd.Series:
    """Build the compact deterministic FIGNEWS identifier used downstream."""

    validate_post_identity(frame, table_name="FIGNEWS table")
    return frame.loc[:, POST_UID_COLUMNS].astype(str).apply(
        lambda row: POST_KEY_SEPARATOR.join(value.strip() for value in row),
        axis=1,
    )


def text_sha256(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def read_delimited(path: str | Path) -> pd.DataFrame:
    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"FIGNEWS file not found: {source.resolve()}"
        )

    separator = "\t" if source.suffix.lower() in {".tsv", ".txt"} else ","

    return pd.read_csv(
        source,
        sep=separator,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
