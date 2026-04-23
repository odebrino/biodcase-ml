import argparse
import warnings
from pathlib import Path

import pandas as pd

from src.data.labels import label_display_name, normalize_label
from src.data.spectrogram import audio_duration
from src.utils.audio import annotation_offsets_seconds

REQUIRED_COLUMNS = {
    "dataset",
    "filename",
    "annotation",
    "annotator",
    "low_frequency",
    "high_frequency",
    "start_datetime",
    "end_datetime",
}


def _annotation_files(data_root: Path, split: str) -> list[Path]:
    return sorted((data_root / split / "annotations").glob("*.csv"))


def build_manifest(
    data_root: Path,
    splits: list[str],
    min_valid_seconds: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    issues: list[dict] = []
    durations: dict[Path, float] = {}
    seen_events: set[tuple] = set()

    for split in splits:
        for annotation_path in _annotation_files(data_root, split):
            frame = pd.read_csv(annotation_path)
            missing = REQUIRED_COLUMNS.difference(frame.columns)
            if missing:
                issues.append(
                    {
                        "split": split,
                        "source": str(annotation_path),
                        "row": "",
                        "issue": f"missing_columns:{','.join(sorted(missing))}",
                    }
                )
                continue

            for row_number, row in frame.iterrows():
                dataset = str(row["dataset"])
                filename = str(row["filename"])
                raw_label = str(row["annotation"]).strip()
                audio_path = data_root / split / "audio" / dataset / filename
                issue_prefix = {
                    "split": split,
                    "dataset": dataset,
                    "filename": filename,
                    "label_raw": raw_label,
                    "source": str(annotation_path),
                    "row": int(row_number),
                }

                try:
                    label = normalize_label(raw_label)
                except ValueError as exc:
                    issues.append({**issue_prefix, "label": "", "issue": f"unknown_label_alias:{exc}"})
                    continue
                issue_prefix["label"] = label

                if not audio_path.exists():
                    issues.append({**issue_prefix, "issue": "missing_audio"})
                    continue

                event_key = (
                    split,
                    dataset,
                    filename,
                    label,
                    str(row["start_datetime"]),
                    str(row["end_datetime"]),
                )
                if event_key in seen_events:
                    issues.append({**issue_prefix, "issue": "duplicate_event"})
                    continue
                seen_events.add(event_key)

                if audio_path not in durations:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            durations[audio_path] = audio_duration(audio_path)
                    except Exception as exc:
                        issues.append({**issue_prefix, "issue": f"audio_duration_error:{exc}"})
                        continue
                audio_duration_s = durations[audio_path]

                try:
                    start_s, end_s, duration_s = annotation_offsets_seconds(
                        filename,
                        str(row["start_datetime"]),
                        str(row["end_datetime"]),
                    )
                except ValueError as exc:
                    issues.append({**issue_prefix, "issue": f"datetime_parse_error:{exc}"})
                    continue

                if duration_s <= 0:
                    issues.append({**issue_prefix, "issue": "non_positive_duration"})
                    continue

                clip_start_s = max(0.0, start_s)
                clip_end_s = min(end_s, audio_duration_s)
                real_duration_s = clip_end_s - clip_start_s

                if start_s >= audio_duration_s:
                    issues.append(
                        {
                            **issue_prefix,
                            "issue": "start_after_audio_end",
                            "start_seconds": start_s,
                            "end_seconds": end_s,
                            "audio_duration_seconds": audio_duration_s,
                        }
                    )
                    continue
                if real_duration_s < min_valid_seconds:
                    issues.append(
                        {
                            **issue_prefix,
                            "issue": "too_little_audio",
                            "start_seconds": start_s,
                            "end_seconds": end_s,
                            "audio_duration_seconds": audio_duration_s,
                            "real_duration_seconds": real_duration_s,
                        }
                    )
                    continue

                rows.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "filename": filename,
                        "audio_path": str(audio_path),
                        "label": label,
                        "label_raw": raw_label,
                        "label_display": label_display_name(label),
                        "annotator": str(row["annotator"]),
                        "low_frequency": float(row["low_frequency"]),
                        "high_frequency": float(row["high_frequency"]),
                        "start_datetime": str(row["start_datetime"]),
                        "end_datetime": str(row["end_datetime"]),
                        "start_seconds": start_s,
                        "end_seconds": end_s,
                        "clip_start_seconds": clip_start_s,
                        "clip_end_seconds": clip_end_s,
                        "duration_seconds": duration_s,
                        "real_duration_seconds": real_duration_s,
                        "audio_duration_seconds": audio_duration_s,
                        "valid_event": True,
                        "quality_status": "ok" if end_s <= audio_duration_s else "clipped_end",
                        "source_annotation": str(annotation_path),
                        "source_row": int(row_number),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(issues)


def write_distribution(manifest: pd.DataFrame, split: str, out_path: Path) -> None:
    subset = manifest[manifest["split"] == split]
    counts = subset["label"].value_counts().sort_index()
    total = int(counts.sum())
    distribution = pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "fraction": [count / total if total else 0 for count in counts.values],
        }
    )
    distribution.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified annotation manifest.")
    parser.add_argument("--data-root", default="biodcase_development_set")
    parser.add_argument("--out", default="data_manifest.csv")
    parser.add_argument("--quality-report", default="outputs/data_quality_report.csv")
    parser.add_argument("--quality-summary", default="outputs/data_quality_summary.csv")
    parser.add_argument("--min-valid-seconds", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest, issues = build_manifest(data_root, args.splits, args.min_valid_seconds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    quality_path = Path(args.quality_report)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(quality_path, index=False)
    write_quality_summary(issues, Path(args.quality_summary))

    for split in args.splits:
        write_distribution(manifest, split, Path(f"{split}_class_distribution.csv"))

    print(f"Wrote {len(manifest)} manifest rows to {out_path}")
    print(f"Wrote {len(issues)} data quality issues to {quality_path}")


def write_quality_summary(issues: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if issues.empty:
        pd.DataFrame(columns=["issue", "split", "dataset", "label", "count"]).to_csv(out_path, index=False)
        return

    summary = (
        issues.groupby(["issue", "split", "dataset", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["issue", "split", "dataset", "label"])
    )
    summary.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
