import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.models.resnet import create_model
from src.training.common import create_loader, save_predictions
from src.utils.config import load_config
from src.utils.metrics import compute_metrics, write_classification_report


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Evaluation requires PyTorch. Install the base dependencies with "
            "`pip install -r requirements.txt`, then install the PyTorch stack with "
            "`pip install -r requirements-cu124.txt`."
        ) from exc
    return torch


def plot_confusion_matrix(confusion_matrix, class_names: list[str], out_path: Path, normalized: bool = False) -> None:
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


def normalized_confusion_matrix(confusion_matrix: list[list[int]]) -> list[list[float]]:
    rows = []
    for row in confusion_matrix:
        total = sum(row)
        rows.append([float(value / total) if total else 0.0 for value in row])
    return rows


def write_confusion_matrix_csv(confusion_matrix, class_names: list[str], out_path: Path) -> None:
    pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).rename_axis("true_label").to_csv(out_path)


def write_error_analysis(predictions: pd.DataFrame, out_path: Path) -> None:
    errors = predictions[predictions["y_true_label"] != predictions["y_pred_label"]]
    if errors.empty:
        pd.DataFrame(columns=["y_true_label", "y_pred_label", "count"]).to_csv(out_path, index=False)
        return
    summary = (
        errors.groupby(["y_true_label", "y_pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary.to_csv(out_path, index=False)


def write_class_confidence_analysis(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
    rows = []
    for label in class_names:
        group = predictions[predictions["y_true_label"] == label]
        rows.append(
            {
                "label": label,
                "support": len(group),
                "accuracy": float((group["y_true_label"] == group["y_pred_label"]).mean()) if len(group) else 0.0,
                "mean_true_probability": float(group["true_probability"].mean()) if len(group) else 0.0,
                "median_true_probability": float(group["true_probability"].median()) if len(group) else 0.0,
                "mean_pred_confidence": float(group["pred_confidence"].mean()) if len(group) else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_pr_curves(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
    rows = []
    for idx, label in enumerate(class_names):
        prob_col = f"prob_{label}"
        if prob_col not in predictions:
            continue
        y_true = (predictions["y_true_idx"] == idx).astype(int).to_numpy()
        scores = predictions[prob_col].to_numpy()
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        average_precision = average_precision_score(y_true, scores) if y_true.sum() else 0.0
        for point_idx, (prec, rec) in enumerate(zip(precision, recall)):
            rows.append(
                {
                    "label": label,
                    "average_precision": float(average_precision),
                    "recall": float(rec),
                    "precision": float(prec),
                    "threshold": float(thresholds[point_idx]) if point_idx < len(thresholds) else "",
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_baseline_metrics(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
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
    pd.DataFrame(rows).to_csv(out_path, index=False)


def split_role(config: dict, split_name: str) -> str:
    if split_name == config.get("test_split"):
        return "official_held_out_test"
    if split_name == config.get("selection_split"):
        return "inner_validation"
    if split_name == config.get("train_split", "train"):
        return "official_train"
    return "custom"


def write_split_metadata(config: dict, split_name: str, output_dir: Path) -> None:
    metadata = {
        "evaluated_split": split_name,
        "split_role": split_role(config, split_name),
        "train_split": config.get("train_split", "train"),
        "selection_split": config.get("selection_split"),
        "test_split": config.get("test_split", config.get("val_split", "validation")),
        "legacy_val_split": config.get("val_split"),
        "held_out_test_domains": config.get("held_out_test_domains", []),
        "note": (
            "The BIODCASE held-out site-year domains are the official test split. "
            "The repository may still contain the legacy split value 'validation' because that is how earlier manifests were written."
        ),
    }
    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def write_dataset_metrics(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
    rows = []
    for dataset, group in predictions.groupby("dataset"):
        metrics = compute_metrics(group["y_true_idx"].tolist(), group["y_pred_idx"].tolist(), class_names)
        rows.append(
            {
                "dataset": dataset,
                "support": len(group),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_f1_present_classes": metrics["macro_f1_present_classes"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    pd.DataFrame(rows).sort_values("dataset").to_csv(out_path, index=False)


def write_dataset_class_metrics(predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
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
    pd.DataFrame(rows).sort_values(["dataset", "label"]).to_csv(out_path, index=False)


def write_bpd_error_report(predictions: pd.DataFrame, out_path: Path) -> None:
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
    predictions.loc[mask, cols].sort_values("pred_confidence", ascending=False).to_csv(out_path, index=False)


def write_bmb_bmz_error_report(predictions: pd.DataFrame, out_path: Path) -> None:
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
    predictions.loc[mask, cols].sort_values(
        ["dataset", "pred_confidence"],
        ascending=[True, False],
    ).to_csv(out_path, index=False)


def meta_value(metadata: dict, key: str, idx: int, default=""):
    values = metadata.get(key)
    if values is None:
        return default
    value = values[idx]
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: dict,
    output_dir: Path,
    split: str | None = None,
    num_workers: int | None = None,
) -> dict:
    torch = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("args") or config
    if "processed_manifest" in config:
        checkpoint_config["processed_manifest"] = config["processed_manifest"]
    for key in ("train_split", "selection_split", "test_split", "val_split", "held_out_test_domains"):
        if key in config and key not in checkpoint_config:
            checkpoint_config[key] = config[key]
    if num_workers is not None:
        checkpoint_config.setdefault("training", {})["num_workers"] = num_workers
        checkpoint_config["training"]["persistent_workers"] = False
    class_names = checkpoint.get("class_names", checkpoint_config["classes"])
    split_name = split or checkpoint_config.get("test_split", checkpoint_config.get("val_split", "validation"))

    model = create_model(
        name=checkpoint_config["model"]["name"],
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = create_loader(
        manifest_path=checkpoint_config.get("processed_manifest", "data_manifest.csv"),
        split=split_name,
        class_names=class_names,
        train=False,
        config=checkpoint_config,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict] = []

    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu().tolist()
            labels_list = labels.tolist()
            y_true.extend(labels_list)
            y_pred.extend(preds)

            for idx, pred_idx in enumerate(preds):
                true_idx = labels_list[idx]
                probs = probabilities[idx]
                prediction_rows.append(
                    {
                        "image_path": metadata.get("image_path", [""] * len(preds))[idx],
                        "audio_path": meta_value(metadata, "audio_path", idx),
                        "dataset": meta_value(metadata, "dataset", idx),
                        "filename": meta_value(metadata, "filename", idx),
                        "source_row": meta_value(metadata, "source_row", idx),
                        "low_frequency": meta_value(metadata, "low_frequency", idx),
                        "high_frequency": meta_value(metadata, "high_frequency", idx),
                        "duration_seconds": meta_value(metadata, "duration_seconds", idx),
                        "real_duration_seconds": meta_value(metadata, "real_duration_seconds", idx),
                        "clip_start_seconds": meta_value(metadata, "clip_start_seconds", idx),
                        "clip_end_seconds": meta_value(metadata, "clip_end_seconds", idx),
                        "y_true_idx": true_idx,
                        "y_pred_idx": pred_idx,
                        "y_true_label": class_names[true_idx],
                        "y_pred_label": class_names[pred_idx],
                        "pred_confidence": float(probs[pred_idx]),
                        "true_probability": float(probs[true_idx]),
                        **{f"prob_{label}": float(probs[label_idx]) for label_idx, label in enumerate(class_names)},
                    }
                )

    metrics = compute_metrics(y_true, y_pred, class_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_classification_report(metrics["classification_report"], output_dir / "classification_report.csv")
    save_predictions(prediction_rows, output_dir / "val_predictions.csv")
    if split_role(checkpoint_config, split_name) == "official_held_out_test":
        save_predictions(prediction_rows, output_dir / "test_predictions.csv")
    predictions = pd.DataFrame(prediction_rows)
    write_split_metadata(checkpoint_config, split_name, output_dir)
    write_dataset_metrics(predictions, class_names, output_dir / "metrics_by_dataset.csv")
    write_dataset_class_metrics(predictions, class_names, output_dir / "metrics_by_dataset_class.csv")
    write_error_analysis(predictions, output_dir / "error_analysis.csv")
    write_error_analysis(predictions, output_dir / "top_confusion_pairs.csv")
    write_bpd_error_report(predictions, output_dir / "bpd_error_report.csv")
    write_bmb_bmz_error_report(predictions, output_dir / "bmb_bmz_error_report.csv")
    write_class_confidence_analysis(predictions, class_names, output_dir / "class_confidence_analysis.csv")
    write_pr_curves(predictions, class_names, output_dir / "pr_curves.csv")
    write_baseline_metrics(predictions, class_names, output_dir / "baseline_metrics.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.png")
    write_confusion_matrix_csv(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.csv")
    normalized = normalized_confusion_matrix(metrics["confusion_matrix"])
    plot_confusion_matrix(normalized, class_names, output_dir / "confusion_matrix_normalized.png", normalized=True)
    write_confusion_matrix_csv(normalized, class_names, output_dir / "confusion_matrix_normalized.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/nitro4060.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--processed-manifest", default=None)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["processed_manifest"] = args.processed_manifest or args.manifest
    metrics = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        config=config,
        output_dir=Path(args.output_dir),
        split=args.split,
        num_workers=args.num_workers,
    )
    print(json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}, indent=2))


if __name__ == "__main__":
    main()
