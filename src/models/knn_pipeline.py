from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from src.data.representations import valid_manifest_rows
from src.evaluation.domain_diagnostics import per_sample_knn_neighbors, write_domain_diagnostics
from src.evaluation.label_mapping_audit import write_label_mapping_audit
from src.evaluation.metrics import (
    compute_metrics,
    metrics_from_label_names,
    normalized_confusion_matrix,
    per_class_metrics_table,
    write_classification_report,
    write_confusion_matrix_csv,
    write_metrics_json,
)
from src.evaluation.reports import (
    plot_confusion_matrix,
    write_baseline_metrics,
    write_bmb_bmz_error_report,
    write_bpd_error_report,
    write_class_confidence_analysis,
    write_dataset_class_metrics,
    write_dataset_metrics,
    write_error_analysis,
    write_submission_overview,
)
from src.evaluation.split_checks import assert_split_integrity, build_split_integrity_report, write_split_integrity_report
from src.features.feature_sets import build_feature_matrix, feature_spec_from_name
from src.utils.config import load_config, save_config


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


class FiniteValueTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 64, random_state: int = 0):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        values = np.asarray(X)
        effective_k = max(1, min(int(self.k), values.shape[1]))
        self.selector_ = SelectKBest(score_func=mutual_info_classif, k=effective_k)
        self.selector_.fit(values, y)
        self.effective_k_ = effective_k
        return self

    def transform(self, X):
        return self.selector_.transform(X)


class SafePCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 64, random_state: int = 0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        values = np.asarray(X)
        max_components = max(1, min(values.shape[0], values.shape[1]))
        effective_components = max(1, min(int(self.n_components), max_components))
        self.pca_ = PCA(n_components=effective_components, random_state=self.random_state)
        self.pca_.fit(values)
        self.effective_components_ = effective_components
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def encode_labels(labels: list[str], class_names: list[str]) -> np.ndarray:
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    unknown = sorted(set(labels) - set(class_to_idx))
    if unknown:
        raise ValueError(f"Labels not declared in config: {unknown}")
    return np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)


def apply_label_aggregation(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    aggregation_cfg = config.get("label_aggregation", {}) or {}
    if not bool(aggregation_cfg.get("enabled", False)):
        return frame
    mapping = {str(key): str(value) for key, value in dict(aggregation_cfg.get("mapping", {})).items()}
    if not mapping:
        raise ValueError("label_aggregation.enabled is true, but no mapping was provided.")
    updated = frame.copy()
    if "label_original" not in updated.columns:
        updated["label_original"] = updated["label"].astype(str)
    updated["label"] = updated["label"].astype(str).map(mapping).fillna(updated["label"].astype(str))
    return updated


def regroup_labels(labels: list[str]) -> list[str]:
    return [FAMILY_MAPPING.get(label, label) for label in labels]


def selected_feature_set(config: dict) -> str:
    submission_cfg = config.get("submission", {})
    return str(
        submission_cfg.get("feature_set")
        or submission_cfg.get("representation")
        or config.get("feature_set")
        or "handcrafted_stats"
    )


def feature_cache_config(config: dict) -> dict:
    cache_cfg = deepcopy(config.get("feature_cache", {}))
    if not cache_cfg:
        cache_cfg = {"enabled": True, "root": "outputs/cache/features"}
    cache_cfg.setdefault("enabled", True)
    cache_cfg.setdefault("root", "outputs/cache/features")
    return cache_cfg


def preprocessing_config(config: dict) -> dict:
    preprocessing = deepcopy(config.get("preprocessing", {}))
    knn_cfg = config.get("knn", {})
    if "reducer" not in preprocessing and knn_cfg.get("pca_components") is not None:
        preprocessing["reducer"] = {"type": "pca_components", "value": int(knn_cfg["pca_components"])}
    preprocessing.setdefault("imputer", "median")
    preprocessing.setdefault("scaler", "standard")
    preprocessing.setdefault("reducer", {"type": "none"})
    return preprocessing


def selected_knn_params(config: dict) -> dict:
    knn_cfg = deepcopy(config.get("knn", {}))
    knn_cfg.setdefault("n_neighbors", int(knn_cfg.get("candidate_n_neighbors", [5])[0] if knn_cfg.get("candidate_n_neighbors") else 5))
    knn_cfg.setdefault("weights", "distance")
    knn_cfg.setdefault("metric", "euclidean")
    knn_cfg.setdefault("algorithm", "auto")
    knn_cfg.setdefault("leaf_size", 30)
    knn_cfg.setdefault("p", 2)
    knn_cfg.setdefault("limit_per_split", None)
    return knn_cfg


def scaler_from_name(name: str, seed: int = 42):
    name = str(name or "none").lower()
    if name in {"none", "passthrough"}:
        return "passthrough"
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name in {"normalizer", "l2"}:
        return Normalizer()
    if name == "standard_l2":
        return Pipeline([("standard", StandardScaler()), ("normalize", Normalizer())])
    if name == "robust_l2":
        return Pipeline([("robust", RobustScaler()), ("normalize", Normalizer())])
    if name == "quantile_uniform":
        return QuantileTransformer(output_distribution="uniform", random_state=seed, subsample=10000)
    if name == "quantile_normal":
        return QuantileTransformer(output_distribution="normal", random_state=seed, subsample=10000)
    if name == "power":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    raise ValueError(f"Unknown scaler: {name}")


def reducer_from_config(reducer_cfg: dict, seed: int):
    reducer_cfg = deepcopy(reducer_cfg or {"type": "none"})
    reducer_type = str(reducer_cfg.get("type", "none")).lower()
    if reducer_type == "none":
        return "passthrough", {"type": "none"}
    if reducer_type == "variance_threshold":
        threshold = float(reducer_cfg.get("threshold", 0.0))
        return VarianceThreshold(threshold=threshold), {"type": reducer_type, "threshold": threshold}
    if reducer_type == "select_k_best":
        k = int(reducer_cfg.get("value", reducer_cfg.get("k", 64)))
        return SafeSelectKBest(k=k, random_state=seed), {"type": reducer_type, "k": k}
    if reducer_type == "pca_variance":
        value = float(reducer_cfg.get("value", 0.95))
        return PCA(n_components=value, random_state=seed), {"type": reducer_type, "value": value}
    if reducer_type == "pca_components":
        value = int(reducer_cfg.get("value", reducer_cfg.get("components", 64)))
        return SafePCA(n_components=value, random_state=seed), {"type": reducer_type, "value": value}
    if reducer_type == "nca":
        value = reducer_cfg.get("value")
        kwargs = {"random_state": seed, "max_iter": int(reducer_cfg.get("max_iter", 100))}
        if value is not None:
            kwargs["n_components"] = int(value)
        return NeighborhoodComponentsAnalysis(**kwargs), {"type": reducer_type, "value": value, "max_iter": kwargs["max_iter"]}
    raise ValueError(f"Unknown reducer type: {reducer_type}")


def normalized_knn_params(knn_cfg: dict, fit_rows: int) -> dict:
    params = deepcopy(knn_cfg)
    params["n_neighbors"] = max(1, min(int(params.get("n_neighbors", 5)), int(fit_rows)))
    metric = str(params.get("metric", "euclidean")).lower()
    algorithm = str(params.get("algorithm", "auto")).lower()
    if metric == "cosine" and algorithm in {"ball_tree", "kd_tree"}:
        algorithm = "brute"
    params["metric"] = metric
    params["algorithm"] = algorithm
    params["weights"] = str(params.get("weights", "distance")).lower()
    params["leaf_size"] = int(params.get("leaf_size", 30))
    params["p"] = int(params.get("p", 2))
    return params


def build_model_pipeline(preprocessing: dict, knn_cfg: dict, seed: int, fit_rows: int) -> Pipeline:
    reducer, reducer_metadata = reducer_from_config(preprocessing.get("reducer", {"type": "none"}), seed=seed)
    params = normalized_knn_params(knn_cfg, fit_rows=fit_rows)
    model_kwargs: dict[str, Any] = {
        "n_neighbors": params["n_neighbors"],
        "weights": params["weights"],
        "metric": params["metric"],
        "algorithm": params["algorithm"],
        "leaf_size": params["leaf_size"],
    }
    if params["metric"] == "minkowski":
        model_kwargs["p"] = params["p"]
    steps: list[tuple[str, Any]] = [
        ("finite", FiniteValueTransformer()),
        ("imputer", SimpleImputer(strategy=str(preprocessing.get("imputer", "median")))),
        ("scaler", scaler_from_name(str(preprocessing.get("scaler", "none")), seed=seed)),
        ("reducer", reducer),
        ("model", KNeighborsClassifier(**model_kwargs)),
    ]
    pipeline = Pipeline(steps)
    pipeline.reducer_metadata = reducer_metadata  # type: ignore[attr-defined]
    pipeline.knn_metadata = params  # type: ignore[attr-defined]
    return pipeline


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
            lines.extend(["", "No within-family misclassifications were observed in the selected official-test report.", ""])
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_submission_config_from_candidate(base_config: dict, candidate: dict, cv_summary: dict | None = None) -> dict:
    config = deepcopy(base_config)
    config.setdefault("submission", {})
    config["submission"]["feature_set"] = candidate["feature_set"]
    config["preprocessing"] = {
        "imputer": candidate["imputer"],
        "scaler": candidate["scaler"],
        "reducer": deepcopy(candidate["reducer"]),
    }
    config["knn"] = {
        **deepcopy(config.get("knn", {})),
        "n_neighbors": int(candidate["n_neighbors"]),
        "weights": candidate["weights"],
        "metric": candidate["metric"],
        "algorithm": candidate["algorithm"],
        "leaf_size": int(candidate["leaf_size"]),
        "p": int(candidate.get("p", 2)),
    }
    if cv_summary is not None:
        config["search_metadata"] = cv_summary
    return config


def _output_subdir(feature_set_name: str) -> str:
    return feature_set_name.replace("/", "_")


def _prediction_frame(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], probabilities: np.ndarray | None) -> pd.DataFrame:
    frame = metadata.copy()
    frame["y_true_idx"] = y_true
    frame["y_pred_idx"] = y_pred
    frame["y_true_label"] = [class_names[int(idx)] for idx in y_true]
    frame["y_pred_label"] = [class_names[int(idx)] for idx in y_pred]
    if probabilities is not None:
        frame["pred_confidence"] = probabilities.max(axis=1)
        frame["true_probability"] = [float(probabilities[idx, label]) for idx, label in enumerate(y_true)]
        for class_idx, class_name in enumerate(class_names):
            frame[f"prob_{class_name}"] = probabilities[:, class_idx]
    return frame


def _top_level_summary_row(feature_set_name: str, feature_count: int, train_rows: int, test_rows: int, metrics: dict, grouped_metrics: dict, pipeline: Pipeline) -> dict:
    knn_meta = getattr(pipeline, "knn_metadata", {})
    reducer_meta = getattr(pipeline, "reducer_metadata", {})
    scaler = pipeline.named_steps["scaler"]
    scaler_name = "none" if scaler == "passthrough" else getattr(scaler, "__class__", type(scaler)).__name__
    return {
        "feature_set": feature_set_name,
        "model": "knn",
        "features_before_pipeline": int(feature_count),
        "fit_rows": int(train_rows),
        "official_test_rows": int(test_rows),
        "imputer": pipeline.named_steps["imputer"].strategy,
        "scaler": scaler_name,
        "reducer_type": reducer_meta.get("type", "none"),
        "n_neighbors": knn_meta.get("n_neighbors"),
        "weights": knn_meta.get("weights"),
        "metric": knn_meta.get("metric"),
        "algorithm": knn_meta.get("algorithm"),
        "leaf_size": knn_meta.get("leaf_size"),
        "p": knn_meta.get("p"),
        "official_test_accuracy": metrics["accuracy"],
        "official_test_balanced_accuracy": metrics["balanced_accuracy"],
        "official_test_macro_f1": metrics["macro_f1"],
        "official_test_grouped_family_macro_f1": grouped_metrics["macro_f1"],
    }


def _overview_lines(feature_set_name: str, metrics: dict, grouped_metrics: dict, dataset_metrics: pd.DataFrame, pipeline: Pipeline) -> list[str]:
    knn_meta = getattr(pipeline, "knn_metadata", {})
    reducer_meta = getattr(pipeline, "reducer_metadata", {})
    scaler = pipeline.named_steps["scaler"]
    scaler_name = "none" if scaler == "passthrough" else scaler.__class__.__name__
    lines = [
        "# KNN Submission Overview",
        "",
        f"- feature_set: `{feature_set_name}`",
        "- model: `knn`",
        f"- imputer: `{pipeline.named_steps['imputer'].strategy}`",
        f"- scaler: `{scaler_name}`",
        f"- reducer: `{reducer_meta}`",
        f"- n_neighbors: `{knn_meta.get('n_neighbors')}`",
        f"- weights: `{knn_meta.get('weights')}`",
        f"- metric: `{knn_meta.get('metric')}`",
        f"- official held-out test accuracy: `{metrics['accuracy']:.4f}`",
        f"- official held-out test balanced accuracy: `{metrics['balanced_accuracy']:.4f}`",
        f"- official held-out test macro-F1: `{metrics['macro_f1']:.4f}`",
        f"- official held-out grouped-family macro-F1: `{grouped_metrics['macro_f1']:.4f}`",
        "",
        "Per-dataset held-out test summary:",
        "",
    ]
    for _, row in dataset_metrics.iterrows():
        lines.append(
            f"- `{row['dataset']}`: support={int(row['support'])}, accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}"
        )
    lines.extend(
        [
            "",
            "The legacy on-disk split name `validation` is the official held-out test split in this repository.",
            "All transforms in this run were fit on the official training split only.",
        ]
    )
    return lines


def load_manifest_and_validate(manifest_path: Path, config: dict) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    report = build_split_integrity_report(
        manifest,
        train_split=config.get("train_split", "train"),
        test_split=config.get("test_split", config.get("val_split", "validation")),
    )
    write_split_integrity_report(
        report,
        Path("outputs/reports/quality/split_integrity.json"),
        Path("outputs/reports/quality/split_integrity.md"),
    )
    assert_split_integrity(report)
    return manifest


def run_knn_submission(config: dict, manifest_path: Path, output_dir: Path) -> Path:
    seed = int(config.get("seed", 42))
    class_names = list(config["classes"])
    train_split = config.get("train_split", "train")
    test_split = config.get("test_split", config.get("val_split", "validation"))
    feature_set_name = selected_feature_set(config)
    preprocessing = preprocessing_config(config)
    knn_cfg = selected_knn_params(config)
    limit_per_split = knn_cfg.get("limit_per_split")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    save_config(config, output_dir / "selected_knn_config.yaml")

    manifest = load_manifest_and_validate(manifest_path, config)
    write_label_mapping_audit(manifest, config, Path("outputs/reports/quality"))
    manifest = valid_manifest_rows(manifest, splits=[train_split, test_split])
    manifest = apply_label_aggregation(manifest, config)
    train_frame = manifest[manifest["split"] == train_split].reset_index(drop=True)
    test_frame = manifest[manifest["split"] == test_split].reset_index(drop=True)
    if limit_per_split is not None:
        limit = int(limit_per_split)
        train_frame = train_frame.groupby("label", group_keys=False).head(limit).reset_index(drop=True)
        test_frame = test_frame.groupby("label", group_keys=False).head(limit).reset_index(drop=True)
    if train_frame.empty:
        raise ValueError(f"No rows found for train split '{train_split}'")
    if test_frame.empty:
        raise ValueError(f"No rows found for test split '{test_split}'")

    cache_cfg = feature_cache_config(config)
    X_train, train_metadata, _, train_cache_path = build_feature_matrix(
        train_frame,
        manifest_path=manifest_path,
        split_name=train_split,
        audio_cfg=config["audio"],
        feature_name=feature_set_name,
        cache_cfg=cache_cfg,
        show_progress=True,
        desc=f"{feature_set_name} train features",
    )
    X_test, test_metadata, _, test_cache_path = build_feature_matrix(
        test_frame,
        manifest_path=manifest_path,
        split_name=test_split,
        audio_cfg=config["audio"],
        feature_name=feature_set_name,
        cache_cfg=cache_cfg,
        show_progress=True,
        desc=f"{feature_set_name} official test features",
    )

    y_train = encode_labels(train_frame["label"].astype(str).tolist(), class_names)
    y_test = encode_labels(test_frame["label"].astype(str).tolist(), class_names)
    pipeline = build_model_pipeline(preprocessing, knn_cfg, seed=seed, fit_rows=len(X_train))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    metrics = compute_metrics(y_test.tolist(), y_pred.tolist(), class_names)
    grouped_metrics = metrics_from_label_names(
        regroup_labels(test_frame["label"].astype(str).tolist()),
        regroup_labels([class_names[int(idx)] for idx in y_pred]),
        FAMILY_LABELS,
    )
    prediction_frame = _prediction_frame(test_metadata, y_test, y_pred, class_names, probabilities)
    prediction_frame["y_true_family"] = regroup_labels(prediction_frame["y_true_label"].tolist())
    prediction_frame["y_pred_family"] = regroup_labels(prediction_frame["y_pred_label"].tolist())
    neighbor_frame = per_sample_knn_neighbors(
        pipeline,
        X_test,
        train_metadata,
        test_metadata,
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        n_neighbors=int(config.get("diagnostics", {}).get("n_neighbors", 5)),
    )

    feature_dir = output_dir / _output_subdir(feature_set_name) / "knn"
    feature_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_dir / "official_test_predictions.csv", index=False)
    prediction_frame.to_csv(feature_dir / "official_test_predictions.csv", index=False)
    neighbor_frame.to_csv(output_dir / "per_sample_knn_neighbors.csv", index=False)
    neighbor_frame.to_csv(feature_dir / "per_sample_knn_neighbors.csv", index=False)
    write_metrics_json(metrics, output_dir / "official_test_metrics.json")
    write_metrics_json(metrics, feature_dir / "official_test_metrics.json")
    write_metrics_json(grouped_metrics, output_dir / "official_test_grouped_family_metrics.json")
    write_metrics_json(grouped_metrics, feature_dir / "official_test_grouped_family_metrics.json")
    write_classification_report(metrics["classification_report"], output_dir / "official_test_classification_report.csv")
    write_classification_report(metrics["classification_report"], feature_dir / "official_test_classification_report.csv")
    write_classification_report(grouped_metrics["classification_report"], feature_dir / "official_test_grouped_family_classification_report.csv")
    per_class_metrics_table(metrics, output_dir / "official_test_per_class_metrics.csv")
    per_class_metrics_table(metrics, feature_dir / "official_test_per_class_metrics.csv")
    per_class_metrics_table(grouped_metrics, feature_dir / "official_test_grouped_family_per_class_metrics.csv")
    write_confusion_matrix_csv(metrics["confusion_matrix"], class_names, output_dir / "official_test_confusion_matrix.csv")
    write_confusion_matrix_csv(metrics["confusion_matrix"], class_names, feature_dir / "official_test_confusion_matrix.csv")
    write_confusion_matrix_csv(normalized_confusion_matrix(metrics["confusion_matrix"]), class_names, output_dir / "official_test_confusion_matrix_normalized.csv")
    write_confusion_matrix_csv(normalized_confusion_matrix(metrics["confusion_matrix"]), class_names, feature_dir / "official_test_confusion_matrix_normalized.csv")
    write_confusion_matrix_csv(grouped_metrics["confusion_matrix"], FAMILY_LABELS, feature_dir / "official_test_grouped_family_confusion_matrix.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, output_dir / "official_test_confusion_matrix.png")
    plot_confusion_matrix(normalized_confusion_matrix(metrics["confusion_matrix"]), class_names, output_dir / "official_test_confusion_matrix_normalized.png", normalized=True)

    dataset_metrics = write_dataset_metrics(prediction_frame, class_names, output_dir / "official_test_metrics_by_dataset.csv")
    write_dataset_metrics(prediction_frame, class_names, feature_dir / "official_test_metrics_by_dataset.csv")
    write_dataset_class_metrics(prediction_frame, class_names, output_dir / "official_test_per_class_metrics_by_dataset.csv")
    write_dataset_class_metrics(prediction_frame, class_names, feature_dir / "official_test_per_class_metrics_by_dataset.csv")
    write_baseline_metrics(prediction_frame, class_names, output_dir / "baseline_metrics.csv")
    write_class_confidence_analysis(prediction_frame, class_names, output_dir / "class_confidence_analysis.csv")
    write_error_analysis(prediction_frame, output_dir / "top_confusions.csv")
    write_bpd_error_report(prediction_frame, output_dir / "bpd_error_report.csv")
    write_bmb_bmz_error_report(prediction_frame, output_dir / "bmb_bmz_error_report.csv")
    ambiguity_report(prediction_frame, output_dir / "official_test_ambiguity_report.md")
    ambiguity_report(prediction_frame, feature_dir / "official_test_ambiguity_report.md")
    write_domain_diagnostics(
        train_frame=train_frame,
        test_frame=test_frame,
        predictions=prediction_frame,
        metrics=metrics,
        neighbors=neighbor_frame,
        class_names=class_names,
        output_dir=output_dir,
    )

    summary_row = _top_level_summary_row(feature_set_name, X_train.shape[1], len(train_frame), len(test_frame), metrics, grouped_metrics, pipeline)
    summary = pd.DataFrame([summary_row])
    summary.to_csv(output_dir / "official_test_results.csv", index=False)
    summary.to_csv(output_dir / "official_test_accuracy_table.csv", index=False)
    summary.to_csv(output_dir / "official_test_macro_f1_table.csv", index=False)
    write_metrics_json(
        {
            "feature_set": feature_set_name,
            "official_test_accuracy": metrics["accuracy"],
            "official_test_balanced_accuracy": metrics["balanced_accuracy"],
            "official_test_macro_f1": metrics["macro_f1"],
            "official_test_grouped_family_macro_f1": grouped_metrics["macro_f1"],
            "train_cache_path": str(train_cache_path) if train_cache_path else None,
            "test_cache_path": str(test_cache_path) if test_cache_path else None,
        },
        output_dir / "main_experiment_metrics.json",
    )

    overview_lines = _overview_lines(feature_set_name, metrics, grouped_metrics, dataset_metrics, pipeline)
    write_submission_overview(overview_lines, output_dir / "main_experiment_overview.md")

    split_strategy = {
        "selection_source": "search-selected KNN configuration",
        "train_split": train_split,
        "test_split": test_split,
        "held_out_test_domains": config.get("held_out_test_domains", []),
        "note": "Official held-out test rows are never used to fit scalers, imputers, reducers, feature selectors, PCA, metric learning, or the final KNN estimator.",
    }
    write_metrics_json(split_strategy, output_dir / "split_strategy.json")
    write_metrics_json(
        {
            "ABZ": ["bma", "bmb", "bmz"],
            "DDswp": ["bmd", "bpd"],
            "20Hz20Plus": ["bp20", "bp20plus"],
        },
        output_dir / "grouped_family_mapping.json",
    )
    joblib.dump(
        {
            "pipeline": pipeline,
            "class_names": class_names,
            "feature_set": feature_set_name,
            "preprocessing": preprocessing,
            "knn": getattr(pipeline, "knn_metadata", {}),
        },
        output_dir / "knn_pipeline.joblib",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the KNN-only submission pipeline.")
    parser.add_argument("--config", default="configs/knn_submission.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/reports/model_evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = run_knn_submission(config, Path(args.manifest), Path(args.output_dir))
    print(f"KNN submission outputs written to {output_dir}")


if __name__ == "__main__":
    main()
