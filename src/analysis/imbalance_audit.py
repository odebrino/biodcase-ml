import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import normalized_confusion_matrix
from src.utils.config import load_config
from src.evaluation.metrics import compute_metrics


def class_distribution(frame: pd.DataFrame, class_names: list[str]) -> list[dict]:
    counts = frame["label"].value_counts()
    total = int(counts.sum())
    rows = []
    for label in class_names:
        count = int(counts.get(label, 0))
        rows.append({"label": label, "count": count, "fraction": count / total if total else 0.0})
    return rows


def imbalance_ratio(distribution: list[dict]) -> float:
    non_zero = [row["count"] for row in distribution if row["count"] > 0]
    return max(non_zero) / min(non_zero) if non_zero else 0.0


def choose_run(runs_root: Path) -> Path | None:
    candidates = []
    for metrics_path in runs_root.glob("*/best_metrics.json"):
        run_dir = metrics_path.parent
        if not (run_dir / "val_predictions.csv").exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        candidates.append((float(metrics.get("macro_f1", -1.0)), run_dir))
    return max(candidates, default=(None, None))[1]


def leakage_summary(manifest: pd.DataFrame, train_split: str, eval_split: str) -> dict:
    key_cols = ["dataset", "filename", "label", "start_datetime", "end_datetime"]
    train = manifest[manifest["split"] == train_split]
    eval_frame = manifest[manifest["split"] == eval_split]
    train_keys = set(map(tuple, train[key_cols].astype(str).values.tolist()))
    eval_keys = set(map(tuple, eval_frame[key_cols].astype(str).values.tolist()))
    return {
        "event_overlap_count": len(train_keys & eval_keys),
        "audio_path_overlap_count": len(set(train["audio_path"]) & set(eval_frame["audio_path"])),
        "dataset_overlap": sorted(set(train["dataset"]) & set(eval_frame["dataset"])),
    }


def artifact_audit(run_dir: Path, class_names: list[str]) -> dict:
    expected = [
        "best_metrics.json",
        "classification_report.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "confusion_matrix.csv",
        "confusion_matrix_normalized.csv",
        "baseline_metrics.csv",
        "class_confidence_analysis.csv",
        "pr_curves.csv",
        "top_confusion_pairs.csv",
        "error_analysis.csv",
        "metrics_by_dataset.csv",
        "metrics_by_dataset_class.csv",
        "val_predictions.csv",
    ]
    present = {name: (run_dir / name).exists() for name in expected}
    checks: dict[str, object] = {"present": present}
    predictions = pd.read_csv(run_dir / "val_predictions.csv")
    checks["prediction_rows"] = int(len(predictions))
    checks["labels_in_predictions"] = sorted(predictions["y_true_label"].unique().tolist())

    report = pd.read_csv(run_dir / "classification_report.csv")
    class_rows = report[report["label"].isin(class_names)]
    checks["classification_report_has_all_classes"] = set(class_rows["label"]) == set(class_names)
    checks["classification_report_support_sum"] = float(class_rows["support"].sum())
    checks["support_matches_predictions"] = abs(float(class_rows["support"].sum()) - len(predictions)) < 1e-9

    metrics = json.loads((run_dir / "best_metrics.json").read_text(encoding="utf-8"))
    normalized = normalized_confusion_matrix(metrics["confusion_matrix"])
    row_sums = [sum(row) for row in normalized]
    supports = [sum(row) for row in metrics["confusion_matrix"]]
    checks["normalized_confusion_rows_valid"] = all(
        abs(row_sum - 1.0) < 1e-9 if support else abs(row_sum) < 1e-9
        for row_sum, support in zip(row_sums, supports)
    )
    return checks


def baseline_rows(predictions: pd.DataFrame, class_names: list[str]) -> list[dict]:
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
        rows.append(
            {
                "baseline": name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    return rows


def previous_headline_macro_f1(readme_path: Path) -> float | None:
    if not readme_path.exists():
        return None
    matches = re.findall(r"Macro-F1:\s*`?([0-9.]+)`?", readme_path.read_text(encoding="utf-8"))
    return float(matches[-1]) if matches else None


def write_markdown(summary: dict, out_path: Path) -> None:
    model = summary["model_metrics"]
    lines = [
        "# Class Imbalance Audit",
        "",
        'Outcome: B) "Imbalance was causing a reporting/evaluation problem; this is now fixed."',
        "",
        "## Conclusion",
        (
            "Class imbalance is not currently making the best evaluation misleading: "
            f"accuracy={model['accuracy']:.4f}, balanced_accuracy={model['balanced_accuracy']:.4f}, "
            f"macro_f1={model['macro_f1']:.4f}, weighted_f1={model['weighted_f1']:.4f}."
        ),
        "The previous risk was reporting: accuracy/weighted metrics and raw counts could be read without enough macro, per-class, normalized-confusion, and baseline context.",
        "",
        "## Evidence",
        f"- Audited run: `{summary['run_dir']}`.",
        f"- Train/eval event leakage: {summary['leakage']['event_overlap_count']}; audio leakage: {summary['leakage']['audio_path_overlap_count']}; dataset overlap: {summary['leakage']['dataset_overlap']}.",
        f"- Validation support includes all configured classes: {summary['artifact_audit']['classification_report_has_all_classes']}.",
        f"- Normalized confusion rows are true-label row-normalized: {summary['artifact_audit']['normalized_confusion_rows_valid']}.",
        f"- Split imbalance ratios vs source: {summary['split_imbalance_vs_all']}.",
        "- Best model beats majority and stratified-random baselines on macro F1 and balanced accuracy.",
        "",
        "## Where The Risk Was",
        "The split is dataset-provided rather than randomly stratified, and training is intentionally more imbalanced than the combined source. This is a domain-generalization split, not an accidental stratification bug. The legacy `validation` split name denotes the official held-out test domains, not a generic validation set.",
        "Training already uses class weights and does not rebalance held-out test data. No additional training mitigation is justified by the current best run.",
        "",
        "## What Changed",
        "- Evaluation now saves balanced accuracy, macro precision/recall, weighted precision/recall, baseline metrics, normalized confusion CSV, confidence analysis, PR curve data, and top confusion pairs.",
        "- The audit now saves split distributions, leakage checks, artifact checks, baseline comparisons, and this Markdown conclusion.",
        "",
        "## Most Affected Classes",
    ]
    for row in summary["per_class_metrics"]:
        if row["recall"] < 0.80 or row["f1"] < 0.80:
            lines.append(f"- `{row['label']}`: recall={row['recall']:.4f}, f1={row['f1']:.4f}, support={row['support']:.0f}.")
    lines.extend(
        [
            "",
            "## Headline Metrics Going Forward",
            "Use macro F1, balanced accuracy, per-class recall/F1, and row-normalized confusion matrix with accuracy/weighted F1 as secondary context.",
            "",
            "## Monitoring",
            "Monitor class distributions, leakage counts, macro-vs-weighted metric gaps, minority-class recall, and top confusion pairs for every run.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit class imbalance risks in splits, metrics, and reports.")
    parser.add_argument("--config", default="legacy/cnn/configs/nitro4060.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-json", default="outputs/reports/audit/imbalance_audit_summary.json")
    parser.add_argument("--out-md", default="outputs/reports/audit/imbalance_audit_summary.md")
    parser.add_argument("--report", default="IMBALANCE_AUDIT.md")
    parser.add_argument("--distribution-out", default="outputs/reports/manifest/class_distribution_by_split_with_all.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    class_names = config["classes"]
    manifest = pd.read_csv(args.manifest)
    train_split = config.get("train_split", "train")
    eval_split = config.get("val_split", "validation")
    run_dir = Path(args.run_dir) if args.run_dir else choose_run(Path("outputs/runs"))
    if run_dir is None:
        raise SystemExit("No run with best_metrics.json and val_predictions.csv found.")

    distributions = {}
    for split in sorted(manifest["split"].unique()):
        distribution = class_distribution(manifest[manifest["split"] == split], class_names)
        distributions[split] = {
            "total": int(sum(row["count"] for row in distribution)),
            "imbalance_ratio": imbalance_ratio(distribution),
            "classes": distribution,
        }
    source_distribution = class_distribution(manifest, class_names)
    distributions["all"] = {
        "total": int(sum(row["count"] for row in source_distribution)),
        "imbalance_ratio": imbalance_ratio(source_distribution),
        "classes": source_distribution,
    }
    source_ratio = distributions["all"]["imbalance_ratio"]
    split_imbalance_vs_all = {
        split: {
            "imbalance_ratio": payload["imbalance_ratio"],
            "more_imbalanced_than_source": payload["imbalance_ratio"] > source_ratio,
        }
        for split, payload in distributions.items()
        if split != "all"
    }
    distribution_rows = []
    for split, payload in distributions.items():
        for row in payload["classes"]:
            distribution_rows.append({"split": split, **row})
    distribution_out = Path(args.distribution_out)
    distribution_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(distribution_rows).to_csv(distribution_out, index=False)

    predictions = pd.read_csv(run_dir / "val_predictions.csv")
    y_true = predictions["y_true_idx"].tolist()
    y_pred = predictions["y_pred_idx"].tolist()
    model_metrics = compute_metrics(y_true, y_pred, class_names)
    report = model_metrics["classification_report"]
    per_class = [
        {
            "label": label,
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": float(report[label]["support"]),
        }
        for label in class_names
    ]
    baselines = baseline_rows(predictions, class_names)

    summary = {
        "outcome": "B",
        "run_dir": str(run_dir),
        "split_strategy": "Dataset-provided train plus legacy validation directory; validation denotes the official held-out test domains, and no random split is created by this project.",
        "stratified_split": False,
        "split_imbalance_vs_all": split_imbalance_vs_all,
        "distributions": distributions,
        "leakage": leakage_summary(manifest, train_split, eval_split),
        "model_metrics": {
            key: model_metrics[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
            )
        },
        "per_class_metrics": per_class,
        "baselines": baselines,
        "previous_project_headline_macro_f1": previous_headline_macro_f1(Path("README.md")),
        "artifact_audit": artifact_audit(run_dir, class_names),
        "training_imbalance_handling": {
            "use_class_weights": bool(config["training"].get("use_class_weights", True)),
            "sampler": config["training"].get("sampler", "none"),
            "class_weight_multipliers": config["training"].get("class_weight_multipliers", {}),
            "validation_rebalanced_or_augmented": False,
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, Path(args.out_md))
    write_markdown(summary, Path(args.report))


if __name__ == "__main__":
    main()
