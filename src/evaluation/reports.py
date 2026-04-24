from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import compute_metrics, normalized_confusion_matrix


def plot_confusion_matrix(confusion_matrix, class_names: list[str], out_path: Path, normalized: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2f" if normalized else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion matrix row-normalized by true class" if normalized else "Confusion matrix counts")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_dataset_metrics(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> pd.DataFrame:
    rows = []
    for dataset, group in predictions.groupby("dataset"):
        metrics = compute_metrics(group["y_true_idx"].tolist(), group["y_pred_idx"].tolist(), class_names)
        rows.append(
            {
                "dataset": dataset,
                "support": len(group),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_f1_present_classes": metrics["macro_f1_present_classes"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("dataset")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def write_dataset_class_metrics(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> pd.DataFrame:
    rows = []
    for dataset, group in predictions.groupby("dataset"):
        metrics = compute_metrics(group["y_true_idx"].tolist(), group["y_pred_idx"].tolist(), class_names)
        report = metrics["classification_report"]
        for label in class_names:
            row = report[label]
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1-score"],
                    "support": row["support"],
                }
            )
    frame = pd.DataFrame(rows).sort_values(["dataset", "label"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def write_baseline_metrics(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> pd.DataFrame:
    y_true = predictions["y_true_idx"].tolist()
    supports = predictions["y_true_idx"].value_counts().to_dict()
    majority_idx = max(range(len(class_names)), key=lambda idx: supports.get(idx, 0))
    baselines = {
        "majority_class": [majority_idx] * len(y_true),
        "stratified_random_seed0": pd.Series(y_true).sample(frac=1.0, random_state=0).tolist(),
    }
    rows = []
    for name, y_pred in baselines.items():
        metrics = compute_metrics(y_true, y_pred, class_names)
        report = metrics["classification_report"]
        rows.append(
            {
                "baseline": name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "min_class_recall": min(float(report[label]["recall"]) for label in class_names),
            }
        )
    frame = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def write_class_confidence_analysis(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> pd.DataFrame:
    rows = []
    for label in class_names:
        group = predictions[predictions["y_true_label"] == label]
        rows.append(
            {
                "label": label,
                "support": len(group),
                "accuracy": float((group["y_true_label"] == group["y_pred_label"]).mean()) if len(group) else 0.0,
                "mean_true_probability": float(group["true_probability"].mean()) if "true_probability" in group and len(group) else 0.0,
                "median_true_probability": float(group["true_probability"].median()) if "true_probability" in group and len(group) else 0.0,
                "mean_pred_confidence": float(group["pred_confidence"].mean()) if "pred_confidence" in group and len(group) else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def write_error_analysis(predictions: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    errors = predictions[predictions["y_true_label"] != predictions["y_pred_label"]]
    if errors.empty:
        frame = pd.DataFrame(columns=["y_true_label", "y_pred_label", "count"])
    else:
        frame = (
            errors.groupby(["y_true_label", "y_pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def write_bpd_error_report(predictions: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    cols = [
        "dataset",
        "filename",
        "source_row",
        "y_true_label",
        "y_pred_label",
        "pred_confidence",
        "true_probability",
        "low_frequency",
        "high_frequency",
        "duration_seconds",
        "real_duration_seconds",
        "clip_start_seconds",
        "clip_end_seconds",
        "audio_path",
    ]
    mask = (
        ((predictions["y_true_label"] == "bpd") & (predictions["y_pred_label"] == "bmd"))
        | ((predictions["y_true_label"] == "bmd") & (predictions["y_pred_label"] == "bpd"))
    )
    frame = predictions.loc[mask, [col for col in cols if col in predictions.columns]].sort_values(
        "pred_confidence",
        ascending=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def write_bmb_bmz_error_report(predictions: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    cols = [
        "dataset",
        "filename",
        "source_row",
        "y_true_label",
        "y_pred_label",
        "pred_confidence",
        "true_probability",
        "low_frequency",
        "high_frequency",
        "duration_seconds",
        "real_duration_seconds",
        "clip_start_seconds",
        "clip_end_seconds",
        "audio_path",
    ]
    mask = (predictions["y_true_label"] == "bmb") & (predictions["y_pred_label"] == "bmz")
    frame = predictions.loc[mask, [col for col in cols if col in predictions.columns]].sort_values(
        ["dataset", "pred_confidence"],
        ascending=[True, False],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def write_submission_overview(lines: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
