from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_fingerprint(manifest: pd.DataFrame) -> str:
    canonical = manifest.sort_values("post_key").to_csv(
        index=False,
        lineterminator="\n",
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

    output_paths = [
        manifest_parquet,
        manifest_csv,
        modeling_output,
        args.profile,
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            f"{[str(path) for path in existing]}. Use --overwrite explicitly."
        )

    manifest.to_parquet(manifest_parquet, index=False)
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    modeling_with_splits.to_parquet(modeling_output, index=False)

    split_post_sets = {
        split: set(manifest.loc[manifest["split"].eq(split), "post_key"])
        for split in ("train", "validation", "test")
    }
    has_overlap = bool(
        split_post_sets["train"] & split_post_sets["validation"]
        or split_post_sets["train"] & split_post_sets["test"]
        or split_post_sets["validation"] & split_post_sets["test"]
    )

    profile = {
        "schema_version": 1,
        "input_path": str(args.input),
        "input_sha256": _sha256(args.input),
        "manifest_content_sha256": _manifest_fingerprint(manifest),
        "unique_posts": int(len(manifest)),
        "unique_post_uids": int(manifest["post_uid"].nunique()),
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
        "min_stratum_size": args.min_stratum_size,
        "post_key_overlap_between_splits": has_overlap,
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
