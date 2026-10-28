from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "fignews"
OUTPUT_DIR = ROOT / "artifacts" / "exports" / "fignews"

FILES = (
    "FIGNEWS-2024-TEXT-MAIN.tsv",
    "FIGNEWS-2024-TEXT-IAA.tsv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Unicode-safe Excel inspection copies of FIGNEWS text files."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--files", nargs="+", default=list(FILES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert_file(
    filename: str,
    *,
    raw_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    source = raw_dir / filename
    output = output_dir / filename.replace(
        ".tsv",
        "-UNICODE.xlsx",
    )
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing export: {output}. "
            "Use --overwrite explicitly."
        )

    df = pd.read_csv(
        source,
        sep="\t",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="error",
        quoting=csv.QUOTE_NONE,
    )

    all_text = " ".join(df.astype(str).to_numpy().ravel())

    arabic_count = len(
        re.findall(r"[\u0600-\u06FF]", all_text)
    )
    hebrew_count = len(
        re.findall(r"[\u0590-\u05FF]", all_text)
    )
    corrupted_sequences = len(
        re.findall(r"\?{3,}", all_text)
    )

    print(f"\nSource: {source.name}")
    print(f"Rows: {len(df):,}")
    print(f"Arabic characters: {arabic_count:,}")
    print(f"Hebrew characters: {hebrew_count:,}")
    print(f"Suspicious ??? sequences: {corrupted_sequences:,}")

    if arabic_count == 0 or hebrew_count == 0:
        raise RuntimeError(
            f"{source.name} does not contain detectable Arabic "
            "and Hebrew characters. It may already be corrupted."
        )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name="FIGNEWS",
            index=False,
        )

        worksheet = writer.book["FIGNEWS"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column in worksheet.columns:
            column_letter = column[0].column_letter
            worksheet.column_dimensions[column_letter].width = 22

    print(f"Created: {output}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in args.files:
        convert_file(
            filename,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
