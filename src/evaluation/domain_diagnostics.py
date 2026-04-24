from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


DOMAIN_COLUMNS = ["dataset", "domain", "site", "recording_id", "session_id", "audio_id", "annotator"]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def _class_distribution(frame: pd.DataFrame, group_columns: list[str]) -> list[dict]:
    columns = [column for column in group_columns if column in frame.columns]
    if not columns:
        return []
    return (
        frame.groupby(columns + ["label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(columns + ["label"])
        .to_dict("records")
    )


def _prediction_distribution(predictions: pd.DataFrame, class_names: list[str]) -> list[dict]:
    true_counts = predictions["y_true_label"].value_counts().reindex(class_names, fill_value=0)
    pred_counts = predictions["y_pred_label"].value_counts().reindex(class_names, fill_value=0)
    return [
        {
            "label": label,
            "true_count": int(true_counts[label]),
            "predicted_count": int(pred_counts[label]),
            "predicted_minus_true": int(pred_counts[label] - true_counts[label]),
        }
        for label in class_names
    ]


def _per_group_metrics(predictions: pd.DataFrame, group_column: str, class_names: list[str]) -> list[dict]:
    if group_column not in predictions.columns:
        return []
    rows = []
    for group_value, group in predictions.groupby(group_column, dropna=False):
        metrics = compute_metrics(group["y_true_idx"].tolist(), group["y_pred_idx"].tolist(), class_names)
        rows.append(
            {
                group_column: group_value,
                "support": int(len(group)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    return sorted(rows, key=lambda row: str(row[group_column]))


def _top_confusions(predictions: pd.DataFrame) -> list[dict]:
    errors = predictions[predictions["y_true_label"] != predictions["y_pred_label"]]
    if errors.empty:
        return []
    return (
        errors.groupby(["y_true_label", "y_pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict("records")
    )


def _neighbor_label(metadata: pd.DataFrame, idx: int, column: str) -> str:
    if column not in metadata.columns or idx >= len(metadata):
        return ""
    return str(metadata.iloc[idx].get(column, ""))


def per_sample_knn_neighbors(
    pipeline,
    X_test: np.ndarray,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    n_neighbors: int = 5,
) -> pd.DataFrame:
    transformed = pipeline[:-1].transform(X_test)
    model = pipeline.named_steps["model"]
    k = max(1, min(int(n_neighbors), int(getattr(model, "n_samples_fit_", len(train_metadata)))))
    distances, indices = model.kneighbors(transformed, n_neighbors=k, return_distance=True)
    rows = []
    for row_idx in range(len(test_metadata)):
        neighbor_indices = indices[row_idx].tolist()
        neighbor_distances = distances[row_idx].tolist()
        nearest_idx = int(neighbor_indices[0])
        payload = {
            "sample_index": row_idx,
            "dataset": test_metadata.iloc[row_idx].get("dataset", ""),
            "filename": test_metadata.iloc[row_idx].get("filename", ""),
            "source_row": test_metadata.iloc[row_idx].get("source_row", ""),
            "y_true_label": class_names[int(y_true[row_idx])],
            "y_pred_label": class_names[int(y_pred[row_idx])],
            "correct": bool(y_true[row_idx] == y_pred[row_idx]),
            "nearest_distance": float(neighbor_distances[0]),
            "mean_neighbor_distance": float(np.mean(neighbor_distances)),
            "nearest_train_label": _neighbor_label(train_metadata, nearest_idx, "label"),
            "nearest_train_dataset": _neighbor_label(train_metadata, nearest_idx, "dataset"),
            "nearest_train_filename": _neighbor_label(train_metadata, nearest_idx, "filename"),
            "neighbor_indices": " ".join(str(int(idx)) for idx in neighbor_indices),
            "neighbor_distances": " ".join(f"{float(distance):.8g}" for distance in neighbor_distances),
            "neighbor_labels": " ".join(_neighbor_label(train_metadata, int(idx), "label") for idx in neighbor_indices),
            "neighbor_datasets": " ".join(_neighbor_label(train_metadata, int(idx), "dataset") for idx in neighbor_indices),
        }
        rows.append(payload)
    return pd.DataFrame(rows)


def _neighbor_stats(neighbors: pd.DataFrame) -> dict:
    rows = {}
    for correct, group in neighbors.groupby("correct"):
        key = "correct" if bool(correct) else "incorrect"
        rows[key] = {
            "count": int(len(group)),
            "nearest_distance_mean": float(group["nearest_distance"].mean()) if len(group) else 0.0,
            "nearest_distance_median": float(group["nearest_distance"].median()) if len(group) else 0.0,
            "nearest_distance_p90": float(group["nearest_distance"].quantile(0.90)) if len(group) else 0.0,
            "mean_neighbor_distance_mean": float(group["mean_neighbor_distance"].mean()) if len(group) else 0.0,
        }
    return rows


def write_domain_diagnostics(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: dict,
    neighbors: pd.DataFrame,
    class_names: list[str],
    output_dir: Path,
) -> dict:
    available_domain_columns = [column for column in DOMAIN_COLUMNS if column in pd.concat([train_frame, test_frame], ignore_index=True).columns]
    split_distribution = _class_distribution(pd.concat([train_frame, test_frame], ignore_index=True), ["split"])
    grouped_distributions = {
        column: _class_distribution(pd.concat([train_frame, test_frame], ignore_index=True), ["split", column])
        for column in available_domain_columns
    }
    per_domain_metrics = {
        column: _per_group_metrics(predictions, column, class_names)
        for column in available_domain_columns
        if column in predictions.columns
    }
    diagnostics = {
        "available_domain_columns": available_domain_columns,
        "class_distribution_by_split": split_distribution,
        "class_distribution_by_domain": grouped_distributions,
        "official_held_out_per_domain_metrics": per_domain_metrics,
        "per_class_metrics": [
            {"label": label, **metrics["classification_report"][label]}
            for label in class_names
        ],
        "confusion_matrix": {
            "class_names": class_names,
            "values": metrics["confusion_matrix"],
        },
        "top_confused_class_pairs": _top_confusions(predictions),
        "prediction_distribution_vs_true": _prediction_distribution(predictions, class_names),
        "nearest_neighbor_distance_stats": _neighbor_stats(neighbors),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "domain_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, default=_json_default)

    lines = [
        "# Domain Diagnostics",
        "",
        f"- available domain columns: `{available_domain_columns}`",
        f"- official held-out accuracy: `{metrics['accuracy']:.4f}`",
        f"- official held-out macro-F1: `{metrics['macro_f1']:.4f}`",
        "",
        "## Prediction Distribution",
        "",
    ]
    for row in diagnostics["prediction_distribution_vs_true"]:
        lines.append(
            f"- `{row['label']}`: true={row['true_count']}, predicted={row['predicted_count']}, delta={row['predicted_minus_true']}"
        )
    lines.extend(["", "## Top Confusions", ""])
    for row in diagnostics["top_confused_class_pairs"][:15]:
        lines.append(f"- `{row['y_true_label']} -> {row['y_pred_label']}`: {int(row['count'])}")
    lines.extend(["", "## Nearest Neighbor Distances", ""])
    for key, row in diagnostics["nearest_neighbor_distance_stats"].items():
        lines.append(
            f"- `{key}`: count={row['count']}, nearest_mean={row['nearest_distance_mean']:.4f}, nearest_median={row['nearest_distance_median']:.4f}, nearest_p90={row['nearest_distance_p90']:.4f}"
        )
    for column, rows in per_domain_metrics.items():
        lines.extend(["", f"## Held-Out Metrics By {column}", ""])
        for row in rows:
            lines.append(
                f"- `{row[column]}`: support={row['support']}, accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}"
            )
    (output_dir / "domain_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return diagnostics
