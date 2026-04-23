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


def write_classification_report(report: dict, out_path: str | Path) -> None:
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            rows.append(row)
        else:
            rows.append({"label": label, "score": values})
    pd.DataFrame(rows).to_csv(out_path, index=False)
