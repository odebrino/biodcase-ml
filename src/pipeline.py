from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src.data.build_manifest import (
    build_manifest as _build_manifest,
    write_split_distributions,
    write_quality_summary,
)
from src.data.cache_tools import cache_files, cache_size
from src.analysis.inspect_errors import export_errors
from src.training.evaluate import evaluate_checkpoint
from src.training.predict import predict_audio, predict_image
from src.training.train import train as _train
from src.utils.config import load_config


def build_manifest(
    data_root: str | Path = "biodcase_development_set",
    out: str | Path = "data_manifest.csv",
    quality_report: str | Path = "outputs/reports/quality/data_quality_report.csv",
    quality_summary: str | Path = "outputs/reports/quality/data_quality_summary.csv",
    split_distribution_dir: str | Path = "outputs/reports/manifest",
    min_valid_seconds: float = 0.5,
    splits: Sequence[str] = ("train", "validation"),
):
    manifest, issues = _build_manifest(Path(data_root), list(splits), min_valid_seconds)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    quality_path = Path(quality_report)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(quality_path, index=False)
    write_quality_summary(issues, Path(quality_summary))

    write_split_distributions(manifest, list(splits), Path(split_distribution_dir))

    print(f"Wrote {len(manifest)} manifest rows to {out_path}")
    print(f"Wrote {len(issues)} data quality issues to {quality_path}")
    return manifest, issues


def train(
    config_path: str | Path = "configs/nitro4060.yaml",
    manifest: str | Path = "data_manifest.csv",
    processed_manifest: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    config["processed_manifest"] = str(processed_manifest or manifest)
    run_dir = _train(config)
    print(f"Run written to {run_dir}")
    return run_dir


def evaluate(
    checkpoint: str | Path,
    config_path: str | Path = "configs/nitro4060.yaml",
    manifest: str | Path = "data_manifest.csv",
    processed_manifest: str | Path | None = None,
    output_dir: str | Path = "outputs/evaluation",
    split: str | None = None,
    num_workers: int | None = None,
) -> dict:
    config = load_config(config_path)
    config["processed_manifest"] = str(processed_manifest or manifest)
    metrics = evaluate_checkpoint(
        checkpoint_path=Path(checkpoint),
        config=config,
        output_dir=Path(output_dir),
        split=split,
        num_workers=num_workers,
    )
    print(json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}, indent=2))
    return metrics


def predict(
    checkpoint: str | Path,
    config_path: str | Path = "configs/nitro4060.yaml",
    image: str | Path | None = None,
    audio: str | Path | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    low_frequency: float = 0.0,
    high_frequency: float = 125.0,
) -> dict:
    config = load_config(config_path)
    if image:
        result = predict_image(Path(checkpoint), config, Path(image))
    elif audio and start_seconds is not None and end_seconds is not None:
        result = predict_audio(
            Path(checkpoint),
            config,
            Path(audio),
            start_seconds,
            end_seconds,
            low_frequency,
            high_frequency,
        )
    else:
        raise ValueError("Use either image or audio with start_seconds and end_seconds.")
    print(json.dumps(result, indent=2))
    return result


def cache_summary(root: str | Path = "processed_cache") -> dict:
    root_path = Path(root)
    files = cache_files(root_path)
    summary = {
        "root": str(root_path),
        "files": len(files),
        "size_gb": cache_size(files) / 1024**3,
    }
    print(f"root: {summary['root']}")
    print(f"files: {summary['files']}")
    print(f"size_gb: {summary['size_gb']:.2f}")
    return summary


def inspect_errors(
    report: str | Path | None = None,
    config_path: str | Path = "configs/nitro4060.yaml",
    out: str | Path = "outputs/error_samples",
    limit: int = 40,
    split: str | None = None,
) -> int:
    config = load_config(config_path)
    report_path = Path(report) if report is not None else None
    if report_path is None:
        from src.analysis.inspect_errors import discover_default_report

        report_path = discover_default_report(Path("outputs/runs"))
    count = export_errors(report_path, config, Path(out), limit, split)
    print(f"exported {count} images")
    return count


def imbalance_audit(
    config_path: str | Path = "configs/nitro4060.yaml",
    manifest: str | Path = "data_manifest.csv",
    run_dir: str | Path | None = None,
    out_json: str | Path = "outputs/reports/audit/imbalance_audit_summary.json",
    out_md: str | Path = "outputs/reports/audit/imbalance_audit_summary.md",
    report: str | Path = "IMBALANCE_AUDIT.md",
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "src.analysis.imbalance_audit",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--report",
        str(report),
    ]
    if run_dir is not None:
        command.extend(["--run-dir", str(run_dir)])
    return _run(command)


def run_tests(args: Sequence[str] = ("-q",)) -> subprocess.CompletedProcess[str]:
    return _run([sys.executable, "-m", "pytest", *args])


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result
