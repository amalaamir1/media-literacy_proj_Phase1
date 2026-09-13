from __future__ import annotations

import json
from pathlib import Path

from data.fignews.pipeline import build_fignews_modeling_table


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "fignews"
PROCESSED_DIR = ROOT / "data" / "processed" / "fignews"
PROFILE_DIR = ROOT / "artifacts" / "profiles"

MODELING_TABLE_PATH = PROCESSED_DIR / "modeling_table.parquet"
PROFILE_PATH = PROFILE_DIR / "fignews_modeling_table_profile.json"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    table = build_fignews_modeling_table(
        flat_votes_path=str(
            RAW_DIR / "FIGNEWS-2024-ALL-CLEAN-FLAT.tsv"
        ),
        main_text_path=str(
            RAW_DIR / "FIGNEWS-2024-TEXT-MAIN.tsv"
        ),
        iaa_text_path=str(
            RAW_DIR / "FIGNEWS-2024-TEXT-IAA.tsv"
        ),
        min_agreement=0.60,
        min_votes=1,
    )

    table.to_parquet(MODELING_TABLE_PATH, index=False)

    profile = {
        "rows": int(len(table)),
        "unique_posts": int(table["post_key"].nunique()),
        "rows_by_subtask": (
            table["subtask"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "rows_by_partition": (
            table["type"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "consensus_status": (
            table["consensus_status"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "evidence_tiers": (
            table["evidence_tier"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "missing_text_rows": int(
            table["text"].astype(str).str.strip().eq("").sum()
        ),
    }

    PROFILE_PATH.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Modeling table: {MODELING_TABLE_PATH}")
    print(f"Profile: {PROFILE_PATH}")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
