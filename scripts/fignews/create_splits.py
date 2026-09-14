from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.fignews.splits import attach_split_manifest, build_split_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "fignews" / "modeling_table.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "fignews"
DEFAULT_PROFILE = ROOT / "artifacts" / "profiles" / "fignews_split_profile.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe FIGNEWS train/validation/test splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-stratum-size", type=int, default=5)
    return parser.parse_args()


def _nested_counts(frame: pd.DataFrame, first: str, second: str) -> dict[str, object]:
    counts = frame.groupby([first, second]).size()
    result: dict[str, dict[str, int]] = {}
    for (first_value, second_value), count in counts.items():
        result.setdefault(str(first_value), {})[str(second_value)] = int(count)
    return result


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Modeling table not found: {args.input}. "
            "Run `python -m scripts.fignews.build_processed` first."
        )

    modeling_table = pd.read_parquet(args.input)
    manifest = build_split_manifest(
        modeling_table,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        min_stratum_size=args.min_stratum_size,
    )
    modeling_with_splits = attach_split_manifest(modeling_table, manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.profile.parent.mkdir(parents=True, exist_ok=True)

    manifest_parquet = args.output_dir / "split_manifest.parquet"
    manifest_csv = args.output_dir / "split_manifest.csv"
    modeling_output = args.output_dir / "modeling_table_with_splits.parquet"

    manifest.to_parquet(manifest_parquet, index=False)
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    modeling_with_splits.to_parquet(modeling_output, index=False)

    profile = {
        "unique_posts": int(len(manifest)),
        "modeling_rows": int(len(modeling_with_splits)),
        "posts_by_split": manifest["split"].value_counts().to_dict(),
        "modeling_rows_by_split": (
            modeling_with_splits["split"].value_counts().to_dict()
        ),
        "posts_by_split_and_language": _nested_counts(
            manifest,
            "split",
            "source_language",
        ),
        "bias_targets_by_split": _nested_counts(
            manifest,
            "split",
            "bias_target",
        ),
        "propaganda_targets_by_split": _nested_counts(
            manifest,
            "split",
            "propaganda_target",
        ),
        "partition_by_split": _nested_counts(manifest, "split", "type"),
        "random_seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "post_key_overlap_between_splits": False,
    }
    args.profile.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Manifest (Parquet): {manifest_parquet}")
    print(f"Manifest (CSV): {manifest_csv}")
    print(f"Modeling table with splits: {modeling_output}")
    print(f"Profile: {args.profile}")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
