from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    supports = np.bincount(y_true, minlength=len(class_names))
    present_labels = [idx for idx, support in enumerate(supports) if support > 0]
    present_recalls = [cm[idx][idx] / supports[idx] for idx in present_labels]
    present_macro_f1 = (
        f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
        if present_labels
        else 0.0
    )
    return {
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(np.mean(present_recalls)) if present_recalls else 0.0,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_f1_present_classes": float(present_macro_f1),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def metrics_from_label_names(y_true: list[str], y_pred: list[str], class_names: list[str]) -> dict:
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    true_idx = [class_to_idx[label] for label in y_true]
    pred_idx = [class_to_idx[label] for label in y_pred]
    return compute_metrics(true_idx, pred_idx, class_names)


def normalized_confusion_matrix(confusion: list[list[int]]) -> list[list[float]]:
    rows = []
    for row in confusion:
        total = sum(row)
        rows.append([float(value / total) if total else 0.0 for value in row])
    return rows


def write_metrics_json(metrics: dict, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def write_classification_report(report: dict, out_path: str | Path) -> None:
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            rows.append(row)
        else:
            rows.append({"label": label, "score": values})
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_confusion_matrix_csv(confusion: list[list[int]] | list[list[float]], class_names: list[str], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion, index=class_names, columns=class_names).rename_axis("true_label").to_csv(path)


def per_class_metrics_table(metrics: dict, out_path: str | Path) -> None:
    rows = []
    for label, values in metrics["classification_report"].items():
        if isinstance(values, dict) and label not in {"macro avg", "weighted avg"}:
            rows.append(
                {
                    "label": label,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1-score"],
                    "support": values["support"],
                }
            )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def metrics_by_dataset(
    predictions: pd.DataFrame,
    true_col: str,
    pred_col: str,
    class_names: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or ["dataset"]
    rows = []
    for group_key, group in predictions.groupby(group_cols):
        metrics = metrics_from_label_names(group[true_col].tolist(), group[pred_col].tolist(), class_names)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        payload = dict(zip(group_cols, group_key))
        rows.append(
            {
                **payload,
                "support": len(group),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_f1_present_classes": metrics["macro_f1_present_classes"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
