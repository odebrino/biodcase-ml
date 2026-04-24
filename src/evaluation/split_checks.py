from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "split",
    "dataset",
    "filename",
    "audio_path",
    "label",
    "source_row",
    "clip_start_seconds",
    "clip_end_seconds",
    "low_frequency",
    "high_frequency",
}


def _examples(frame: pd.DataFrame, columns: list[str], limit: int = 5) -> list[dict]:
    if frame.empty:
        return []
    return frame[columns].head(limit).to_dict("records")


def _duplicates_between_splits(train: pd.DataFrame, test: pd.DataFrame, key_columns: list[str]) -> tuple[int, list[dict]]:
    if not set(key_columns).issubset(train.columns) or not set(key_columns).issubset(test.columns):
        return 0, []
    train_keys = train[key_columns].fillna("").astype(str)
    test_keys = test[key_columns].fillna("").astype(str)
    merged = train_keys.merge(test_keys.drop_duplicates(), on=key_columns, how="inner")
    return int(len(merged)), _examples(merged, key_columns)


def build_split_integrity_report(manifest: pd.DataFrame, train_split: str = "train", test_split: str = "validation") -> dict:
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns))
    train = manifest[manifest["split"] == train_split].reset_index(drop=True)
    test = manifest[manifest["split"] == test_split].reset_index(drop=True)
    candidate_keys = {
        "dataset_filename": ["dataset", "filename"],
        "audio_path": ["audio_path"],
        "event_window": ["dataset", "filename", "clip_start_seconds", "clip_end_seconds", "low_frequency", "high_frequency"],
        "source_row_per_dataset": ["dataset", "source_row"],
    }
    optional_group_columns = [column for column in ["recording_id", "session_id", "site", "domain", "audio_id"] if column in manifest.columns]
    for column in optional_group_columns:
        candidate_keys[f"optional_{column}"] = [column]

    checks = {}
    leakage_detected = False
    for name, columns in candidate_keys.items():
        count, examples = _duplicates_between_splits(train, test, columns)
        checks[name] = {"columns": columns, "duplicate_count": count, "examples": examples}
        leakage_detected = leakage_detected or count > 0

    report = {
        "required_columns_present": not missing,
        "missing_required_columns": missing,
        "train_split": train_split,
        "test_split": test_split,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "optional_group_columns": optional_group_columns,
        "checks": checks,
        "leakage_detected": bool(leakage_detected),
    }
    return report


def write_split_integrity_report(report: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    lines = [
        "# Split Integrity",
        "",
        f"- train split: `{report['train_split']}`",
        f"- test split: `{report['test_split']}`",
        f"- train rows: `{report['train_rows']}`",
        f"- test rows: `{report['test_rows']}`",
        f"- leakage detected: `{report['leakage_detected']}`",
        "",
    ]
    if report["missing_required_columns"]:
        lines.append(f"- missing required columns: `{report['missing_required_columns']}`")
        lines.append("")
    for name, payload in report["checks"].items():
        lines.append(f"## {name}")
        lines.append(f"- columns: `{payload['columns']}`")
        lines.append(f"- duplicate_count: `{payload['duplicate_count']}`")
        if payload["examples"]:
            lines.append("- examples:")
            for example in payload["examples"]:
                lines.append(f"  - `{example}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def assert_split_integrity(report: dict) -> None:
    if report["missing_required_columns"]:
        raise ValueError(f"Manifest missing required columns for split integrity checks: {report['missing_required_columns']}")
    if report["leakage_detected"]:
        raise ValueError("Potential train/validation leakage detected. See outputs/reports/quality/split_integrity.*")
