from __future__ import annotations

import argparse
from pathlib import Path

from src.data.build_manifest import build_manifest, write_quality_summary, write_split_distributions
from src.models.knn_pipeline import run_knn_submission
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single entry point for the KNN-only submission pipeline.")
    parser.add_argument("--config", default="configs/knn_submission.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/reports/model_evaluation")
    parser.add_argument("--data-root", default="biodcase_development_set")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--quality-report", default="outputs/reports/quality/data_quality_report.csv")
    parser.add_argument("--quality-summary", default="outputs/reports/quality/data_quality_summary.csv")
    parser.add_argument("--split-distribution-dir", default="outputs/reports/manifest")
    return parser.parse_args()


def maybe_rebuild_manifest(args: argparse.Namespace, config: dict) -> None:
    if not args.rebuild_manifest:
        return
    manifest, issues = build_manifest(
        Path(args.data_root),
        [config.get("train_split", "train"), config.get("test_split", config.get("val_split", "validation"))],
        float(config.get("quality", {}).get("min_valid_seconds", 0.5)),
    )
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    quality_path = Path(args.quality_report)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(quality_path, index=False)
    write_quality_summary(issues, Path(args.quality_summary))
    write_split_distributions(
        manifest,
        [config.get("train_split", "train"), config.get("test_split", config.get("val_split", "validation"))],
        Path(args.split_distribution_dir),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    maybe_rebuild_manifest(args, config)
    output_dir = run_knn_submission(config, Path(args.manifest), Path(args.output_dir))
    print(f"Submission outputs written to {output_dir}")


if __name__ == "__main__":
    main()
