"""Command-line entry point for FIGNEWS baseline experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from modeling.fignews.baseline import BaselineConfig, run_baseline_experiment


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "fignews"
    / "modeling_table_with_splits.parquet"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "models" / "fignews"
TEXT_CHOICES = (
    "original",
    "english_translation",
    "arabic_translation",
    "original_plus_english",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe FIGNEWS TF-IDF baselines."
    )
    parser.add_argument(
        "--task",
        choices=("bias", "propaganda", "both"),
        required=True,
    )
    parser.add_argument(
        "--text-representation",
        choices=TEXT_CHOICES,
        default="original",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = ("bias", "propaganda") if args.task == "both" else (args.task,)

    for task in tasks:
        config = BaselineConfig(
            task=task,
            text_representation=args.text_representation,
            random_seed=args.seed,
            max_features=args.max_features,
            min_df=args.min_df,
        )
        result = run_baseline_experiment(
            input_path=args.input,
            output_root=args.output_root,
            config=config,
        )
        printable = asdict(result)
        printable["output_dir"] = str(result.output_dir.resolve())
        print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
