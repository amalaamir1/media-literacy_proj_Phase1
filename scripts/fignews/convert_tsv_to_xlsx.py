# convert_fignews_to_excel.py

from pathlib import Path
import re

import pandas as pd


SOURCE = Path("FIGNEWS-2024-TEXT-MAIN.tsv")
OUTPUT = Path("FIGNEWS-2024-TEXT-MAIN-UNICODE.xlsx")


df = pd.read_csv(
    SOURCE,
    sep="\t",
    encoding="utf-8-sig",
    dtype=str,
    keep_default_na=False,
    on_bad_lines="error",
)

# Confirm the source still contains real Arabic/Hebrew characters.
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

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    df.to_excel(
        writer,
        sheet_name="FIGNEWS",
        index=False,
    )

    worksheet = writer.book["FIGNEWS"]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    # Make text columns readable.
    for column in worksheet.columns:
        column_letter = column[0].column_letter
        worksheet.column_dimensions[column_letter].width = 22

print(f"Created: {OUTPUT.resolve()}")