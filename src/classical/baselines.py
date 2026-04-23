from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.representations import representation_vector, valid_manifest_rows
from src.utils.config import load_config, save_config
from src.utils.metrics import compute_metrics, write_classification_report

FAMILY_LABELS = ["ABZ", "DDswp", "20Hz20Plus"]
FAMILY_MAPPING = {
    "bma": "ABZ",
    "bmb": "ABZ",
    "bmz": "ABZ",
    "bmd": "DDswp",
    "bpd": "DDswp",
    "bp20": "20Hz20Plus",
    "bp20plus": "20Hz20Plus",
}
KNOWN_AMBIGUITIES = [
    ("ABZ", ["bma", "bmb", "bmz"], "Bm-A / Bm-B / Bm-Z can transition smoothly and should not be framed as purely model defects."),
    ("DDswp", ["bmd", "bpd"], "Bm-D vs Bp-40Down may reflect intrinsic annotation ambiguity, not only classifier failure."),
    ("20Hz20Plus", ["bp20", "bp20plus"], "Bp-20 vs Bp-20Plus can be ambiguous when overtone energy is faint."),
]


MODEL_NAMES = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "knn",
    "gaussian_nb",
    "random_forest",
    "gradient_boosted_trees",
    "mlp",
]


def model_factory(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    if name == "linear_svm":
        return SVC(kernel="linear", class_weight="balanced", probability=False, random_state=seed)
    if name == "rbf_svm":
        return SVC(kernel="rbf", class_weight="balanced", C=3.0, gamma="scale", probability=False, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "gaussian_nb":
        return GaussianNB()
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "gradient_boosted_trees":
        return GradientBoostingClassifier(random_state=seed)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def needs_scaling(model_name: str) -> bool:
    return model_name in {
        "logistic_regression",
        "linear_svm",
        "rbf_svm",
        "knn",
        "gaussian_nb",
        "mlp",
    }


def make_pipeline(model_name: str, seed: int, pca_components: int | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling(model_name):
        steps.append(("scaler", StandardScaler()))
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, random_state=seed)))
    steps.append(("model", model_factory(model_name, seed)))
    return Pipeline(steps)


def encode_labels(labels: list[str], class_names: list[str]) -> np.ndarray:
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    unknown = sorted(set(labels) - set(class_to_idx))
    if unknown:
        raise ValueError(f"Labels not declared in config: {unknown}")
    return np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)


def build_feature_matrix(frame: pd.DataFrame, audio_cfg: dict, family: str, img_size: int) -> tuple[np.ndarray, list[dict]]:
    vectors = []
    metadata = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        vectors.append(representation_vector(row_dict, audio_cfg, family, img_size))
        metadata.append(
            {
                "split": row_dict.get("split", ""),
                "dataset": row_dict.get("dataset", ""),
                "filename": row_dict.get("filename", ""),
                "source_row": row_dict.get("source_row", ""),
                "label": row_dict.get("label", ""),
                "label_raw": row_dict.get("label_raw", ""),
            }
        )
    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
    return matrix, metadata


def inner_selection_split(
    train_frame: pd.DataFrame,
    class_names: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    indices = np.arange(len(train_frame))
    labels = train_frame["label"].astype(str).tolist()
    y = encode_labels(labels, class_names)
    groups = train_frame["dataset"].astype(str).to_numpy() if "dataset" in train_frame else np.asarray(["all"] * len(train_frame))
    strategy = {
        "selection_source": "official training split only",
        "validation_fraction": validation_fraction,
        "group_column": "dataset",
        "method": "",
    }

    if len(np.unique(groups)) >= 2 and len(train_frame) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        fit_idx, selection_idx = next(splitter.split(indices, y, groups))
        if len(set(y[fit_idx])) >= 2 and len(set(y[selection_idx])) >= 2:
            strategy["method"] = "GroupShuffleSplit by dataset"
            return fit_idx, selection_idx, strategy

    class_counts = np.bincount(y, minlength=len(class_names))
    present_counts = class_counts[class_counts > 0]
    if len(present_counts) >= 2 and int(present_counts.min()) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        fit_idx, selection_idx = next(splitter.split(indices, y))
        strategy["method"] = "StratifiedShuffleSplit fallback within train"
        strategy["fallback_reason"] = "Not enough dataset groups for useful group split."
        return fit_idx, selection_idx, strategy

    fit_idx, selection_idx = train_test_split(indices, test_size=validation_fraction, random_state=seed)
    strategy["method"] = "random train-only fallback"
    strategy["fallback_reason"] = "Too few groups/classes for grouped or stratified split."
    return np.asarray(fit_idx), np.asarray(selection_idx), strategy


def pca_for_family(cfg: dict, family: str) -> int | None:
    pca_cfg = cfg.get("pca", {})
    if not bool(pca_cfg.get("enabled", False)):
        return None
    value = pca_cfg.get("components_by_family", {}).get(family, pca_cfg.get("components"))
    return int(value) if value else None


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    return compute_metrics(y_true.tolist(), y_pred.tolist(), class_names)


def regroup_labels(labels: list[str]) -> list[str]:
    return [FAMILY_MAPPING.get(label, label) for label in labels]


def metrics_from_label_names(y_true: list[str], y_pred: list[str], class_names: list[str]) -> dict:
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    true_idx = [class_to_idx[label] for label in y_true]
    pred_idx = [class_to_idx[label] for label in y_pred]
    return compute_metrics(true_idx, pred_idx, class_names)


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


def per_class_metrics_table(metrics: dict, out_path: Path) -> None:
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
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_confusion_csv(metrics: dict, class_names: list[str], out_path: Path) -> None:
    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names).rename_axis("true_label").to_csv(out_path)


def ambiguity_report(predictions: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Official Test Ambiguity Report",
        "",
        "This report treats several confusions as potentially intrinsic label ambiguity, not only model defects.",
        "",
    ]
    for family_name, labels, explanation in KNOWN_AMBIGUITIES:
        family_set = set(labels)
        mask = predictions["y_true_label"].isin(family_set) & predictions["y_pred_label"].isin(family_set)
        within_family = predictions.loc[mask & (predictions["y_true_label"] != predictions["y_pred_label"])]
        lines.append(f"## {family_name}")
        lines.append(explanation)
        if within_family.empty:
            lines.append("")
            lines.append("No within-family misclassifications were observed in the selected official-test report.")
            lines.append("")
            continue
        summary = (
            within_family.groupby(["y_true_label", "y_pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        lines.append("")
        lines.append("Observed official-test confusions:")
        for _, row in summary.iterrows():
            lines.append(f"- `{row['y_true_label']} -> {row['y_pred_label']}`: {int(row['count'])}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def comparison_tables(results: pd.DataFrame, run_dir: Path) -> None:
    macro_table = results.pivot(index="representation", columns="model", values="official_test_macro_f1")
    acc_table = results.pivot(index="representation", columns="model", values="official_test_accuracy")
    macro_table.to_csv(run_dir / "official_test_macro_f1_table.csv")
    acc_table.to_csv(run_dir / "official_test_accuracy_table.csv")


def best_model_summary(
    best_family: str,
    best_model: str,
    best_dir: Path,
    official_test_metrics: dict,
    grouped_metrics: dict,
    dataset_metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# Official Test Best-Model Summary",
        "",
        f"- representation: `{best_family}`",
        f"- model: `{best_model}`",
        f"- official test accuracy: `{official_test_metrics['accuracy']:.4f}`",
        f"- official test balanced accuracy: `{official_test_metrics['balanced_accuracy']:.4f}`",
        f"- official test macro-F1: `{official_test_metrics['macro_f1']:.4f}`",
        f"- official test grouped-family macro-F1: `{grouped_metrics['macro_f1']:.4f}`",
        "",
        "Known ambiguity-aware interpretation:",
        "- `Bm-A / Bm-B / Bm-Z` confusions may reflect smooth transitions inside the `ABZ` family.",
        "- `Bm-D` vs `Bp-40Down` may reflect intrinsic `DDswp` ambiguity, not only model failure.",
        "- `Bp-20` vs `Bp-20Plus` may reflect faint overtone ambiguity inside `20Hz20Plus`.",
        "",
        "Per-dataset held-out test summary:",
        "",
    ]
    for _, row in dataset_metrics.iterrows():
        lines.append(
            f"- `{row['dataset']}`: support={int(row['support'])}, "
            f"accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}"
        )
    lines.append("")
    lines.append(f"Detailed ambiguity report: `{best_dir / 'official_test_ambiguity_report.md'}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_predictions(
    rows: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    model_name: str,
    representation: str,
    split_role: str,
) -> None:
    enriched = []
    for idx, row in enumerate(rows):
        enriched.append(
            {
                **row,
                "model": model_name,
                "representation": representation,
                "split_role": split_role,
                "y_true_idx": int(y_true[idx]),
                "y_pred_idx": int(y_pred[idx]),
                "y_true_label": class_names[int(y_true[idx])],
                "y_pred_label": class_names[int(y_pred[idx])],
            }
        )
    pd.DataFrame(enriched).to_csv(out_path, index=False)


def run_classical_baselines(config: dict, manifest_path: Path, output_dir: Path) -> Path:
    seed = int(config.get("seed", 42))
    class_names = list(config["classes"])
    train_split = config.get("train_split", "train")
    test_split = config.get("test_split", config.get("val_split", "validation"))
    validation_fraction = float(config.get("classical", {}).get("validation_fraction", 0.2))
    families = list(config.get("classical", {}).get("representations", ["handcrafted", "patch", "hybrid"]))
    models = list(config.get("classical", {}).get("models", MODEL_NAMES))
    img_size = int(config.get("classical", {}).get("img_size", 32))
    limit_per_split = config.get("classical", {}).get("limit_per_split")
    if limit_per_split is not None:
        limit_per_split = int(limit_per_split)

    run_dir = output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    manifest = valid_manifest_rows(pd.read_csv(manifest_path), splits=[train_split, test_split])
    train_frame = manifest[manifest["split"] == train_split].reset_index(drop=True)
    test_frame = manifest[manifest["split"] == test_split].reset_index(drop=True)
    if limit_per_split:
        train_frame = train_frame.groupby("label", group_keys=False).head(limit_per_split).reset_index(drop=True)
        test_frame = test_frame.groupby("label", group_keys=False).head(limit_per_split).reset_index(drop=True)
    if train_frame.empty:
        raise ValueError(f"No rows found for train_split={train_split!r}")
    if test_frame.empty:
        raise ValueError(f"No rows found for test_split={test_split!r}")

    fit_idx, selection_idx, split_strategy = inner_selection_split(train_frame, class_names, validation_fraction, seed)
    y_train_all = encode_labels(train_frame["label"].astype(str).tolist(), class_names)
    y_test = encode_labels(test_frame["label"].astype(str).tolist(), class_names)

    with (run_dir / "split_strategy.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **split_strategy,
                "train_split": train_split,
                "test_split": test_split,
                "held_out_test_domains": config.get("held_out_test_domains", []),
                "note": "Official held-out test rows are never used to fit scalers, PCA, estimators, or model selection.",
            },
            handle,
            indent=2,
        )

    summary_rows = []
    all_predictions = []
    best_selection = None
    for family in families:
        X_train_all, train_metadata = build_feature_matrix(train_frame, config["audio"], family, img_size)
        X_test, test_metadata = build_feature_matrix(test_frame, config["audio"], family, img_size)
        X_fit = X_train_all[fit_idx]
        y_fit = y_train_all[fit_idx]
        X_selection = X_train_all[selection_idx]
        y_selection = y_train_all[selection_idx]
        pca_components = pca_for_family(config.get("classical", {}), family)
        if pca_components:
            max_components = min(X_fit.shape[0], X_fit.shape[1])
            pca_components = min(pca_components, max_components)

        for model_name in models:
            pipeline = make_pipeline(model_name, seed, pca_components)
            pipeline.fit(X_fit, y_fit)

            selection_pred = pipeline.predict(X_selection)
            test_pred = pipeline.predict(X_test)
            selection_metrics = evaluate_predictions(y_selection, selection_pred, class_names)
            test_metrics = evaluate_predictions(y_test, test_pred, class_names)
            grouped_selection_metrics = metrics_from_label_names(
                regroup_labels([class_names[idx] for idx in y_selection]),
                regroup_labels([class_names[idx] for idx in selection_pred]),
                FAMILY_LABELS,
            )
            grouped_test_metrics = metrics_from_label_names(
                regroup_labels(test_frame["label"].astype(str).tolist()),
                regroup_labels([class_names[idx] for idx in test_pred]),
                FAMILY_LABELS,
            )

            model_dir = run_dir / family / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            with (model_dir / "selection_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(selection_metrics, handle, indent=2)
            with (model_dir / "official_test_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(test_metrics, handle, indent=2)
            with (model_dir / "selection_grouped_family_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(grouped_selection_metrics, handle, indent=2)
            with (model_dir / "official_test_grouped_family_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(grouped_test_metrics, handle, indent=2)
            write_classification_report(selection_metrics["classification_report"], model_dir / "selection_classification_report.csv")
            write_classification_report(test_metrics["classification_report"], model_dir / "official_test_classification_report.csv")
            write_classification_report(
                grouped_test_metrics["classification_report"],
                model_dir / "official_test_grouped_family_classification_report.csv",
            )
            per_class_metrics_table(test_metrics, model_dir / "official_test_per_class_metrics.csv")
            per_class_metrics_table(grouped_test_metrics, model_dir / "official_test_grouped_family_per_class_metrics.csv")
            write_confusion_csv(test_metrics, class_names, model_dir / "official_test_confusion_matrix.csv")
            write_confusion_csv(grouped_test_metrics, FAMILY_LABELS, model_dir / "official_test_grouped_family_confusion_matrix.csv")

            selection_rows = [train_metadata[int(idx)] for idx in selection_idx]
            write_predictions(
                selection_rows,
                y_selection,
                selection_pred,
                class_names,
                model_dir / "selection_predictions.csv",
                model_name,
                family,
                "inner_validation",
            )
            write_predictions(
                test_metadata,
                y_test,
                test_pred,
                class_names,
                model_dir / "official_test_predictions.csv",
                model_name,
                family,
                "official_held_out_test",
            )
            test_predictions = pd.read_csv(model_dir / "official_test_predictions.csv")
            test_predictions["y_true_family"] = regroup_labels(test_predictions["y_true_label"].tolist())
            test_predictions["y_pred_family"] = regroup_labels(test_predictions["y_pred_label"].tolist())
            test_predictions.to_csv(model_dir / "official_test_predictions.csv", index=False)
            metrics_by_dataset(test_predictions, "y_true_label", "y_pred_label", class_names).to_csv(
                model_dir / "official_test_metrics_by_dataset.csv",
                index=False,
            )
            metrics_by_dataset(test_predictions, "y_true_family", "y_pred_family", FAMILY_LABELS).to_csv(
                model_dir / "official_test_grouped_family_metrics_by_dataset.csv",
                index=False,
            )
            ambiguity_report(test_predictions, model_dir / "official_test_ambiguity_report.md")

            summary = {
                "representation": family,
                "model": model_name,
                "features_before_pipeline": int(X_train_all.shape[1]),
                "pca_components": pca_components,
                "fit_rows": int(len(fit_idx)),
                "selection_rows": int(len(selection_idx)),
                "official_test_rows": int(len(test_frame)),
                "selection_macro_f1": selection_metrics["macro_f1"],
                "selection_balanced_accuracy": selection_metrics["balanced_accuracy"],
                "selection_accuracy": selection_metrics["accuracy"],
                "official_test_macro_f1": test_metrics["macro_f1"],
                "official_test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "official_test_accuracy": test_metrics["accuracy"],
                "official_test_grouped_family_macro_f1": grouped_test_metrics["macro_f1"],
            }
            summary_rows.append(summary)
            candidate = (selection_metrics["macro_f1"], test_metrics["macro_f1"], family, model_name)
            if best_selection is None or candidate > best_selection[:4]:
                best_selection = (
                    selection_metrics["macro_f1"],
                    test_metrics["macro_f1"],
                    family,
                    model_name,
                    model_dir,
                )
            all_predictions.extend(
                {
                    **row,
                    "model": model_name,
                    "representation": family,
                    "split_role": "official_held_out_test",
                    "y_true_idx": int(y_test[idx]),
                    "y_pred_idx": int(test_pred[idx]),
                    "y_true_label": class_names[int(y_test[idx])],
                    "y_pred_label": class_names[int(test_pred[idx])],
                    "y_true_family": regroup_labels([class_names[int(y_test[idx])]])[0],
                    "y_pred_family": regroup_labels([class_names[int(test_pred[idx])]])[0],
                }
                for idx, row in enumerate(test_metadata)
            )

    results = pd.DataFrame(summary_rows).sort_values(
        ["selection_macro_f1", "official_test_macro_f1"],
        ascending=[False, False],
    )
    results.to_csv(run_dir / "official_test_results.csv", index=False)
    results.to_csv(run_dir / "results.csv", index=False)
    comparison_tables(results, run_dir)
    all_predictions_frame = pd.DataFrame(all_predictions)
    all_predictions_frame.to_csv(run_dir / "all_official_test_predictions.csv", index=False)
    metrics_by_dataset(
        all_predictions_frame,
        "y_true_label",
        "y_pred_label",
        class_names,
        group_cols=["representation", "model", "dataset"],
    ).to_csv(
        run_dir / "all_official_test_metrics_by_dataset.csv",
        index=False,
    )
    metrics_by_dataset(
        all_predictions_frame,
        "y_true_family",
        "y_pred_family",
        FAMILY_LABELS,
        group_cols=["representation", "model", "dataset"],
    ).to_csv(
        run_dir / "all_official_test_grouped_family_metrics_by_dataset.csv",
        index=False,
    )
    family_mapping = {
        "ABZ": ["bma", "bmb", "bmz"],
        "DDswp": ["bmd", "bpd"],
        "20Hz20Plus": ["bp20", "bp20plus"],
    }
    with (run_dir / "grouped_family_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(family_mapping, handle, indent=2)
    if best_selection is not None:
        _, _, best_family, best_model, best_dir = best_selection
        (run_dir / "official_test_best_model.txt").write_text(
            f"representation={best_family}\nmodel={best_model}\npath={best_dir}\n",
            encoding="utf-8",
        )
        best_metrics = json.loads((best_dir / "official_test_metrics.json").read_text(encoding="utf-8"))
        best_grouped_metrics = json.loads((best_dir / "official_test_grouped_family_metrics.json").read_text(encoding="utf-8"))
        best_dataset_metrics = pd.read_csv(best_dir / "official_test_metrics_by_dataset.csv")
        best_model_summary(
            best_family,
            best_model,
            best_dir,
            best_metrics,
            best_grouped_metrics,
            best_dataset_metrics,
            run_dir / "official_test_best_summary.md",
        )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe classical ML baselines.")
    parser.add_argument("--config", default="configs/classical_baselines.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/classical")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = run_classical_baselines(config, Path(args.manifest), Path(args.output_dir))
    print(f"Classical baseline run written to {run_dir}")


if __name__ == "__main__":
    main()
