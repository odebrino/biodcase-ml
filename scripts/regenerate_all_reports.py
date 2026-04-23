from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.imbalance_audit import choose_run
from src.data.build_manifest import (
    MANIFEST_SCHEMA_DESCRIPTION,
    build_manifest,
    class_distribution_table,
    dataset_distribution_table,
    split_distribution_table,
    write_quality_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate canonical manifest-derived reports.")
    parser.add_argument("--data-root", default="biodcase_development_set")
    parser.add_argument("--manifest-out", default="data_manifest.csv")
    parser.add_argument("--reports-root", default="outputs/reports")
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--quality-summary", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--audit-config", default="configs/nitro4060.yaml")
    parser.add_argument("--audit-report", default="IMBALANCE_AUDIT.md")
    parser.add_argument("--min-valid-seconds", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_out = Path(args.manifest_out)
    reports_root = Path(args.reports_root)
    manifest_reports = reports_root / "manifest"
    quality_reports = reports_root / "quality"
    audit_reports = reports_root / "audit"
    provenance_reports = reports_root / "provenance"

    for path in (manifest_reports, quality_reports, audit_reports, provenance_reports):
        path.mkdir(parents=True, exist_ok=True)

    manifest, issues = build_manifest(data_root, args.splits, args.min_valid_seconds)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_out, index=False)

    quality_report = Path(args.quality_report) if args.quality_report else quality_reports / "data_quality_report.csv"
    quality_summary = Path(args.quality_summary) if args.quality_summary else quality_reports / "data_quality_summary.csv"
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_summary.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(quality_report, index=False)
    write_quality_summary(issues, quality_summary)

    class_counts = class_distribution_table(manifest)
    split_counts = split_distribution_table(manifest)
    dataset_counts = dataset_distribution_table(manifest)
    class_totals = (
        manifest.groupby("label", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("label")
    )

    for frame, path in (
        (class_counts, manifest_reports / "class_distribution_by_split.csv"),
        (split_counts, manifest_reports / "split_distribution.csv"),
        (dataset_counts, manifest_reports / "dataset_distribution.csv"),
        (class_totals, manifest_reports / "class_distribution_overall.csv"),
    ):
        frame.to_csv(path, index=False)

    (manifest_reports / "manifest_schema.txt").write_text(MANIFEST_SCHEMA_DESCRIPTION + "\n", encoding="utf-8")

    run_dir = Path(args.run_dir) if args.run_dir else choose_run(Path("outputs/runs"))
    if run_dir is None:
        raise SystemExit("No historical run found for imbalance audit regeneration.")

    audit_json = audit_reports / "imbalance_audit_summary.json"
    audit_md = audit_reports / "imbalance_audit_summary.md"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.imbalance_audit",
            "--config",
            args.audit_config,
            "--manifest",
            str(manifest_out),
            "--run-dir",
            str(run_dir),
            "--distribution-out",
            str(manifest_reports / "class_distribution_by_split_with_all.csv"),
            "--out-json",
            str(audit_json),
            "--out-md",
            str(audit_md),
            "--report",
            args.audit_report,
        ],
        check=True,
    )

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "manifest_path": str(manifest_out),
        "manifest_sha256": sha256_file(manifest_out),
        "manifest_rows": int(len(manifest)),
        "quality_issue_rows": int(len(issues)),
        "splits": list(args.splits),
        "reports_root": str(reports_root),
        "historical_audit_run_dir": str(run_dir),
        "commands": [
            (
                f"{sys.executable} scripts/regenerate_all_reports.py --data-root {data_root} "
                f"--manifest-out {manifest_out} --reports-root {reports_root}"
            ),
            (
                f"{sys.executable} -m src.analysis.imbalance_audit --config {args.audit_config} "
                f"--manifest {manifest_out} --run-dir {run_dir} --distribution-out "
                f"{manifest_reports / 'class_distribution_by_split_with_all.csv'} --out-json {audit_json} "
                f"--out-md {audit_md} --report {args.audit_report}"
            ),
        ],
        "artifacts": {
            "manifest": str(manifest_out),
            "quality_report": str(quality_report),
            "quality_summary": str(quality_summary),
            "class_distribution_by_split": str(manifest_reports / "class_distribution_by_split.csv"),
            "split_distribution": str(manifest_reports / "split_distribution.csv"),
            "dataset_distribution": str(manifest_reports / "dataset_distribution.csv"),
            "class_distribution_overall": str(manifest_reports / "class_distribution_overall.csv"),
            "audit_class_distribution_with_all": str(manifest_reports / "class_distribution_by_split_with_all.csv"),
            "imbalance_audit_json": str(audit_json),
            "imbalance_audit_markdown": str(audit_md),
            "manifest_schema": str(manifest_reports / "manifest_schema.txt"),
        },
    }
    (provenance_reports / "report_regeneration_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Regenerated manifest with {len(manifest)} rows at {manifest_out}")
    print(f"Regenerated {len(issues)} quality issues under {quality_reports}")
    print(f"Regenerated canonical reports under {reports_root}")


if __name__ == "__main__":
    main()
