from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "fignews" / "FIGNEWS-2024-TEXT-MAIN.tsv"
OUTPUT = (
    ROOT
    / "artifacts"
    / "exports"
    / "fignews"
    / "FIGNEWS-2024-TEXT-MAIN-UNICODE.xlsx"
)


def main() -> None:
    df = pd.read_csv(
        SOURCE,
        sep="\t",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="error",
        quoting=csv.QUOTE_NONE,
    )

    all_text = " ".join(df.astype(str).to_numpy().ravel())
    arabic_count = len(re.findall(r"[\u0600-\u06FF]", all_text))
    hebrew_count = len(re.findall(r"[\u0590-\u05FF]", all_text))
    corrupted_sequences = len(re.findall(r"\?{3,}", all_text))

    print(f"Rows: {len(df):,}")
    print(f"Arabic characters: {arabic_count:,}")
    print(f"Hebrew characters: {hebrew_count:,}")
    print(f"Suspicious ??? sequences: {corrupted_sequences:,}")

    if arabic_count == 0 or hebrew_count == 0:
        raise RuntimeError(
            "The input TSV does not contain detectable Arabic and Hebrew. "
            "It may already be corrupted."
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="FIGNEWS", index=False)
        worksheet = writer.book["FIGNEWS"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column in worksheet.columns:
            column_letter = column[0].column_letter
            worksheet.column_dimensions[column_letter].width = 22

    print(f"Created: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
