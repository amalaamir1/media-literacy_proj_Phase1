from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd

from .constants import POST_KEY_COLUMNS


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


def build_post_key(frame: pd.DataFrame) -> pd.Series:
    require_columns(
        frame,
        list(POST_KEY_COLUMNS),
        table_name="FIGNEWS table",
    )

    return frame.loc[:, POST_KEY_COLUMNS].astype(str).apply(
        lambda row: "::".join(value.strip() for value in row),
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
