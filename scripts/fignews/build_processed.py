from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.fignews.pipeline import build_fignews_modeling_table


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "fignews"
PROCESSED_DIR = ROOT / "data" / "processed" / "fignews"
PROFILE_DIR = ROOT / "artifacts" / "profiles"

MODELING_TABLE_PATH = PROCESSED_DIR / "modeling_table.parquet"
PROFILE_PATH = PROFILE_DIR / "fignews_modeling_table_profile.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the validated FIGNEWS modeling table."
    )
    parser.add_argument(
        "--flat-votes",
        type=Path,
        default=RAW_DIR / "FIGNEWS-2024-ALL-CLEAN-FLAT.tsv",
    )
    parser.add_argument(
        "--main-text",
        type=Path,
        default=RAW_DIR / "FIGNEWS-2024-TEXT-MAIN.tsv",
    )
    parser.add_argument(
        "--iaa-text",
        type=Path,
        default=RAW_DIR / "FIGNEWS-2024-TEXT-IAA.tsv",
    )
    parser.add_argument("--output", type=Path, default=MODELING_TABLE_PATH)
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--min-agreement", type=float, default=0.60)
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    existing = [path for path in (args.output, args.profile) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            f"{[str(path) for path in existing]}. Use --overwrite explicitly."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.profile.parent.mkdir(parents=True, exist_ok=True)

    table = build_fignews_modeling_table(
        flat_votes_path=args.flat_votes,
        main_text_path=args.main_text,
        iaa_text_path=args.iaa_text,
        min_agreement=args.min_agreement,
        min_votes=args.min_votes,
    )

    table.to_parquet(args.output, index=False)

    profile = {
        "schema_version": 1,
        "consensus_policy": {
            "min_agreement": args.min_agreement,
            "min_votes": args.min_votes,
        },
        "sources": {
            "flat_votes": {
                "path": str(args.flat_votes),
                "sha256": _sha256(args.flat_votes),
            },
            "main_text": {
                "path": str(args.main_text),
                "sha256": _sha256(args.main_text),
            },
            "iaa_text": {
                "path": str(args.iaa_text),
                "sha256": _sha256(args.iaa_text),
            },
        },
        "rows": int(len(table)),
        "unique_posts": int(table["post_key"].nunique()),
        "unique_post_uids": int(table["post_uid"].nunique()),
        "post_key_to_uid_is_one_to_one": bool(
            table[["post_key", "post_uid"]]
            .drop_duplicates()["post_uid"]
            .is_unique
        ),
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

    args.profile.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Modeling table: {args.output}")
    print(f"Profile: {args.profile}")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
