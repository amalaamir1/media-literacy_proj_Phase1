"""Compare the reference and consensus-only FIGNEWS training policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modeling.fignews.baseline import BaselineConfig, run_baseline_experiment


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "fignews"
    / "modeling_table_with_splits.parquet"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "experiments"
    / "fignews"
    / "evidence_policy_comparison"
)
POLICIES = {
    "reference_all_accepted": (
        "bronze_single",
        "silver_consensus",
        "gold_iaa",
    ),
    "strict_consensus_only": (
        "silver_consensus",
        "gold_iaa",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reference and strict FIGNEWS evidence policies and save a "
            "side-by-side comparison."
        )
    )
    parser.add_argument(
        "--task",
        choices=("bias", "propaganda", "both"),
        default="both",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument(
        "--calibration-method",
        choices=("none", "sigmoid", "isotonic"),
        default="sigmoid",
    )
    parser.add_argument("--calibration-cv", type=int, default=5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing policy-run artifacts.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    tasks = ("bias", "propaganda") if args.task == "both" else (args.task,)
    rows: list[dict[str, object]] = []

    for policy_name, evidence_tiers in POLICIES.items():
        for task in tasks:
            result = run_baseline_experiment(
                input_path=args.input,
                output_root=args.output_root / policy_name,
                config=BaselineConfig(
                    task=task,
                    random_seed=args.seed,
                    max_features=args.max_features,
                    min_df=args.min_df,
                    eligible_evidence_tiers=evidence_tiers,
                    calibration_method=args.calibration_method,
                    calibration_cv=args.calibration_cv,
                ),
                overwrite=args.overwrite,
            )
            run_config = _read_json(result.output_dir / "run_config.json")
            validation_metrics = _read_json(
                result.output_dir / "validation_metrics.json"
            )
            test_metrics = _read_json(result.output_dir / "test_metrics.json")
            rows.append(
                {
                    "task": task,
                    "policy": policy_name,
                    "eligible_evidence_tiers": ",".join(evidence_tiers),
                    "calibration_method": args.calibration_method,
                    "train_rows": run_config["rows_by_split"].get("train", 0),
                    "validation_rows": run_config["rows_by_split"].get(
                        "validation", 0
                    ),
                    "test_rows": run_config["rows_by_split"].get("test", 0),
                    "selected_c": result.selected_c,
                    "selected_ngram_range": str(result.selected_ngram_range),
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_log_loss": validation_metrics["log_loss"],
                    "validation_expected_calibration_error": validation_metrics[
                        "expected_calibration_error"
                    ],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_observed_support_macro_f1": test_metrics[
                        "observed_support_macro_f1"
                    ],
                    "test_log_loss": test_metrics["log_loss"],
                    "test_expected_calibration_error": test_metrics[
                        "expected_calibration_error"
                    ],
                    "output_dir": str(result.output_dir.resolve()),
                }
            )

    comparison = pd.DataFrame(rows)
    reference = comparison.loc[
        comparison["policy"].eq("reference_all_accepted"),
        ["task", "validation_macro_f1", "test_macro_f1"],
    ].rename(
        columns={
            "validation_macro_f1": "reference_validation_macro_f1",
            "test_macro_f1": "reference_test_macro_f1",
        }
    )
    comparison = comparison.merge(reference, on="task", validate="many_to_one")
    comparison["validation_macro_f1_delta_vs_reference"] = (
        comparison["validation_macro_f1"]
        - comparison["reference_validation_macro_f1"]
    )
    comparison["test_macro_f1_delta_vs_reference"] = (
        comparison["test_macro_f1"] - comparison["reference_test_macro_f1"]
    )
    comparison = comparison.drop(
        columns=["reference_validation_macro_f1", "reference_test_macro_f1"]
    ).sort_values(["task", "policy"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_root / "policy_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison: {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
