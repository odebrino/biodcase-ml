from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.representations import valid_manifest_rows
from src.features.feature_sets import build_feature_matrix
from src.models.knn_pipeline import apply_label_aggregation, encode_labels, feature_cache_config, load_manifest_and_validate
from src.utils.config import load_config, save_config


NOTEBOOK_DATASETS = [
    "casey2014",
    "kerguelen2005",
    "maudrise2014",
    "elephantisland2013",
    "rosssea2014",
    "elephantisland2014",
    "ballenyislands2015",
    "greenwich2015",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Projet.ipynb strict classical KNN experiments.")
    parser.add_argument("--config", default="configs/knn_notebook_reproduction.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/reports/notebook_reproduction")
    return parser.parse_args()


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _scoring() -> dict:
    return {
        "accuracy": "accuracy",
        "macro_precision": make_scorer(precision_score, average="macro", zero_division=0),
        "macro_recall": make_scorer(recall_score, average="macro", zero_division=0),
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
    }


def _cv_metric_dict(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, seed: int, n_splits: int) -> dict:
    min_class_count = int(np.bincount(y).min())
    effective_splits = max(2, min(int(n_splits), min_class_count))
    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    cv = cross_validate(pipeline, X, y, cv=splitter, scoring=_scoring(), n_jobs=1, error_score="raise")
    metrics: dict[str, float | int] = {"n_splits": int(effective_splits)}
    for key in _scoring():
        values = cv[f"test_{key}"]
        metrics[f"{key}_mean"] = float(np.mean(values))
        metrics[f"{key}_std"] = float(np.std(values))
    return metrics


def _can_evaluate(frame: pd.DataFrame, classes: list[str]) -> tuple[bool, str]:
    counts = frame["label"].astype(str).value_counts()
    missing = [label for label in classes if int(counts.get(label, 0)) < 2]
    if missing:
        return False, f"not enough samples per class for stratified evaluation: {missing}"
    return True, ""


def _feature_matrix(
    frame: pd.DataFrame,
    manifest_path: Path,
    split_name: str,
    config: dict,
    feature_set: str,
    output_label: str,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    X, metadata, names, _ = build_feature_matrix(
        frame,
        manifest_path=manifest_path,
        split_name=split_name,
        audio_cfg=config["audio"],
        feature_name=feature_set,
        cache_cfg=feature_cache_config(config),
        show_progress=True,
        desc=f"{output_label} {feature_set}",
    )
    return X, metadata, names


def notebook_exact_leaky_audit(
    frame: pd.DataFrame,
    manifest_path: Path,
    config: dict,
    classes: list[str],
    feature_set: str,
    dataset_name: str,
) -> dict:
    ok, reason = _can_evaluate(frame, classes)
    base = {
        "mode": "notebook_exact_leaky_audit",
        "dataset": dataset_name,
        "feature_set": feature_set,
        "task_mode": "3class_notebook_cv",
        "diagnostic_only": True,
        "not_strict_split_safe": True,
        "not_for_model_selection": True,
        "scaler_fit_scope": "full_selected_dataset_before_train_test_split",
        "rows": int(len(frame)),
        "status": "ok" if ok else "skipped",
        "skip_reason": reason,
    }
    if not ok:
        return base
    X, _, names = _feature_matrix(frame, manifest_path, f"train_{dataset_name}_leaky_audit", config, feature_set, dataset_name)
    y = encode_labels(frame["label"].astype(str).tolist(), classes)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=float(config.get("notebook_reproduction", {}).get("test_size", 0.2)),
        random_state=int(config.get("seed", 42)),
        stratify=y,
    )
    effective_neighbors = min(5, max(1, len(X_train)))
    model = KNeighborsClassifier(n_neighbors=effective_neighbors, weights="uniform", metric="minkowski", p=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        **base,
        **_metric_dict(y_test, y_pred),
        "feature_dimension": int(len(names)),
        "knn_params": json.dumps(
            {
                "requested_n_neighbors": 5,
                "effective_n_neighbors": int(effective_neighbors),
                "weights": "uniform",
                "metric": "minkowski",
                "p": 2,
            }
        ),
    }


def notebook_exact_split_safe(
    frame: pd.DataFrame,
    manifest_path: Path,
    config: dict,
    classes: list[str],
    feature_set: str,
    dataset_name: str,
    task_mode: str,
) -> dict:
    ok, reason = _can_evaluate(frame, classes)
    base = {
        "mode": "notebook_exact_split_safe",
        "dataset": dataset_name,
        "feature_set": feature_set,
        "task_mode": task_mode,
        "diagnostic_only": False,
        "not_strict_split_safe": False,
        "not_for_model_selection": False,
        "scaler_fit_scope": "inside_sklearn_pipeline_cv_train_folds_only",
        "rows": int(len(frame)),
        "status": "ok" if ok else "skipped",
        "skip_reason": reason,
    }
    if not ok:
        return base
    X, _, names = _feature_matrix(frame, manifest_path, f"train_{dataset_name}_split_safe", config, feature_set, dataset_name)
    y = encode_labels(frame["label"].astype(str).tolist(), classes)
    requested_splits = int(config.get("notebook_reproduction", {}).get("n_splits", 5))
    effective_splits = max(2, min(requested_splits, int(frame["label"].value_counts().min())))
    min_train_rows = len(frame) - int(np.ceil(len(frame) / effective_splits))
    effective_neighbors = min(5, max(1, min_train_rows))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=effective_neighbors, weights="uniform", metric="minkowski", p=2)),
        ]
    )
    metrics = _cv_metric_dict(
        pipeline,
        X,
        y,
        seed=int(config.get("seed", 42)),
        n_splits=requested_splits,
    )
    return {
        **base,
        **metrics,
        "feature_dimension": int(len(names)),
        "knn_params": json.dumps(
            {
                "requested_n_neighbors": 5,
                "effective_n_neighbors": int(effective_neighbors),
                "weights": "uniform",
                "metric": "minkowski",
                "p": 2,
            }
        ),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _prepare_train_frames(manifest: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = valid_manifest_rows(manifest, splits=[config.get("train_split", "train")])
    train_7 = train.copy().reset_index(drop=True)
    train_3 = apply_label_aggregation(train.copy(), config).reset_index(drop=True)
    return train_7, train_3


def run_notebook_reproduction(config: dict, manifest_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config_used.yaml")
    manifest = load_manifest_and_validate(manifest_path, config)
    train_7, train_3 = _prepare_train_frames(manifest, config)
    rep_cfg = config.get("notebook_reproduction", {})
    feature_set = str(rep_cfg.get("feature_set", "notebook_exact_44"))
    target_dataset = str(rep_cfg.get("default_dataset", "elephantisland2014"))
    datasets = list(rep_cfg.get("datasets", NOTEBOOK_DATASETS))
    classes_3 = list(rep_cfg.get("classes_3", ["ABZ", "DDswp", "20Hz20Plus"]))
    classes_7 = list(rep_cfg.get("classes_7", ["bma", "bmb", "bmd", "bmz", "bp20", "bp20plus", "bpd"]))

    per_dataset_rows = []
    for dataset in datasets:
        selected = train_3[train_3["dataset"].astype(str) == dataset].reset_index(drop=True)
        if selected.empty:
            continue
        per_dataset_rows.append(notebook_exact_leaky_audit(selected, manifest_path, config, classes_3, feature_set, dataset))
        per_dataset_rows.append(notebook_exact_split_safe(selected, manifest_path, config, classes_3, feature_set, dataset, "3class_notebook_cv"))
    per_dataset = pd.DataFrame(per_dataset_rows)
    per_dataset.to_csv(output_dir / "per_dataset_results.csv", index=False)

    combined_rows = []
    combined_rows.append(
        notebook_exact_split_safe(
            train_3,
            manifest_path,
            config,
            classes_3,
            feature_set,
            "all_train_datasets",
            "3class_notebook_cv",
        )
    )
    combined_rows.append(
        notebook_exact_split_safe(
            train_7,
            manifest_path,
            {**deepcopy(config), "label_aggregation": {"enabled": False}},
            classes_7,
            feature_set,
            "all_train_datasets",
            "7class_strict_cv",
        )
    )
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(output_dir / "combined_train_results.csv", index=False)

    target_frame = train_3[train_3["dataset"].astype(str) == target_dataset].reset_index(drop=True)
    leaky = notebook_exact_leaky_audit(target_frame, manifest_path, config, classes_3, feature_set, target_dataset)
    split_safe = notebook_exact_split_safe(target_frame, manifest_path, config, classes_3, feature_set, target_dataset, "3class_notebook_cv")
    summary_payload = {
        "notebook_exact_leaky_audit": leaky,
        "notebook_exact_split_safe_default_dataset": split_safe,
        "combined_train_results": combined.to_dict("records"),
        "per_dataset_results_path": str(output_dir / "per_dataset_results.csv"),
        "combined_train_results_path": str(output_dir / "combined_train_results.csv"),
        "official_validation_used": False,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    per_dataset_ok = per_dataset[per_dataset["status"] == "ok"].copy() if not per_dataset.empty else pd.DataFrame()
    best_split_safe = per_dataset_ok[per_dataset_ok["mode"] == "notebook_exact_split_safe"].copy()
    if not best_split_safe.empty:
        best_split_safe = best_split_safe.sort_values(["weighted_f1_mean", "accuracy_mean"], ascending=[False, False])

    lines = [
        "# Notebook Reproduction Summary",
        "",
        "This report uses only the official `train` split. The on-disk `validation/` official held-out split is not used.",
        "",
        "## Default Elephant Island 2014 Audit",
        "",
        f"- leaky diagnostic accuracy: `{leaky.get('accuracy', float('nan')):.4f}`",
        f"- leaky diagnostic macro-F1: `{leaky.get('macro_f1', float('nan')):.4f}`",
        f"- leaky diagnostic weighted-F1: `{leaky.get('weighted_f1', float('nan')):.4f}`",
        f"- split-safe CV accuracy: `{split_safe.get('accuracy_mean', float('nan')):.4f}`",
        f"- split-safe CV macro-F1: `{split_safe.get('macro_f1_mean', float('nan')):.4f}`",
        f"- split-safe CV weighted-F1: `{split_safe.get('weighted_f1_mean', float('nan')):.4f}`",
        "",
        "The leaky audit intentionally follows the notebook-like scaler-before-split pattern and is diagnostic only.",
        "",
        "## Per-Dataset Results",
        "",
    ]
    if per_dataset_ok.empty:
        lines.append("No per-dataset result could be evaluated.")
    else:
        columns = [
            "dataset",
            "mode",
            "rows",
            "accuracy",
            "accuracy_mean",
            "macro_f1",
            "macro_f1_mean",
            "weighted_f1",
            "weighted_f1_mean",
            "status",
        ]
        lines.extend(_markdown_table(per_dataset_ok[columns], columns))
    lines.extend(["", "## Combined Train Results", ""])
    columns = [
        "task_mode",
        "dataset",
        "rows",
        "accuracy_mean",
        "macro_precision_mean",
        "macro_f1_mean",
        "weighted_f1_mean",
        "feature_dimension",
        "status",
    ]
    lines.extend(_markdown_table(combined[columns], columns))
    if not best_split_safe.empty:
        best = best_split_safe.iloc[0]
        lines.extend(
            [
                "",
                "## Best Per-Dataset Split-Safe Result",
                "",
                f"- dataset: `{best['dataset']}`",
                f"- accuracy: `{best['accuracy_mean']:.4f}`",
                f"- weighted-F1: `{best['weighted_f1_mean']:.4f}`",
                f"- macro-F1: `{best['macro_f1_mean']:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Notebook 0.89 Interpretation",
            "",
            "A reproduced 0.89-like score here should be interpreted as an internal train-split/CV result, not as official held-out performance.",
            "Any scaler-before-split result is marked diagnostic-only and is not eligible for model selection.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "per_dataset_results.md").write_text("\n".join(lines[lines.index("## Per-Dataset Results") :]) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = run_notebook_reproduction(config, Path(args.manifest), Path(args.output_dir))
    print(f"Notebook reproduction outputs written to {output_dir}")


if __name__ == "__main__":
    main()
