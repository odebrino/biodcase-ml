from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.representations import build_representation_matrix, valid_manifest_rows
from src.utils.config import load_config


def stratified_sample(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    if per_class <= 0:
        return frame.reset_index(drop=True)
    parts = []
    for _, group in frame.groupby("label", sort=False):
        parts.append(group.sample(min(len(group), per_class), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def cluster_models(seed: int, n_clusters: int) -> dict[str, object]:
    return {
        "kmeans": KMeans(n_clusters=n_clusters, n_init=20, random_state=seed),
        "dbscan": DBSCAN(eps=1.5, min_samples=10),
        "mean_shift": MeanShift(bin_seeding=True),
    }


def fit_predict_clusters(name: str, estimator, X: np.ndarray) -> np.ndarray:
    if name == "mean_shift":
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=min(len(X), 1000))
        if bandwidth and bandwidth > 0:
            estimator.set_params(bandwidth=bandwidth)
    return estimator.fit_predict(X)


def clustering_metrics(y_true: np.ndarray, cluster_labels: np.ndarray, X: np.ndarray) -> dict:
    assigned_mask = cluster_labels != -1
    assigned_count = int(assigned_mask.sum())
    n_clusters = len(set(cluster_labels.tolist()) - {-1})
    metrics = {
        "assigned_fraction": float(assigned_count / len(cluster_labels)) if len(cluster_labels) else 0.0,
        "noise_fraction": float((cluster_labels == -1).sum() / len(cluster_labels)) if len(cluster_labels) else 0.0,
        "clusters_found": int(n_clusters),
        "adjusted_rand_index": float(adjusted_rand_score(y_true, cluster_labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(y_true, cluster_labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, cluster_labels)),
        "homogeneity": float(homogeneity_score(y_true, cluster_labels)),
        "completeness": float(completeness_score(y_true, cluster_labels)),
        "silhouette_assigned": None,
    }
    if assigned_count >= 2 and n_clusters >= 2:
        metrics["silhouette_assigned"] = float(silhouette_score(X[assigned_mask], cluster_labels[assigned_mask]))
    return metrics


def cluster_composition_table(frame: pd.DataFrame, out_path: Path) -> None:
    composition = (
        frame.groupby(["cluster_label", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster_label", "count"], ascending=[True, False])
    )
    composition.to_csv(out_path, index=False)


def run_clustering(config: dict, manifest_path: Path, output_dir: Path) -> Path:
    seed = int(config.get("seed", 42))
    class_names = list(config["classes"])
    test_split = config.get("test_split", config.get("val_split", "validation"))
    family = config.get("clustering", {}).get("representation", "handcrafted")
    img_size = int(config.get("clustering", {}).get("img_size", config.get("classical", {}).get("img_size", 32)))
    per_class = int(config.get("clustering", {}).get("per_class_limit", 500))

    manifest = valid_manifest_rows(pd.read_csv(manifest_path), splits=[test_split])
    manifest = stratified_sample(manifest, per_class=per_class, seed=seed)
    X, metadata = build_representation_matrix(
        manifest,
        config["audio"],
        family,
        img_size,
        show_progress=True,
        desc=f"{family} clustering features",
    )
    labels = manifest["label"].astype(str).tolist()
    y_true = np.asarray([class_names.index(label) for label in labels], dtype=np.int64)

    run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    scaled = Pipeline([("scaler", StandardScaler())]).fit_transform(X)
    summary_rows = []
    for name, estimator in cluster_models(seed, len(class_names)).items():
        cluster_labels = fit_predict_clusters(name, estimator, scaled)
        metrics = clustering_metrics(y_true, cluster_labels, scaled)
        model_dir = run_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        predictions = pd.DataFrame(metadata)
        predictions["label"] = labels
        predictions["cluster_label"] = cluster_labels
        predictions.to_csv(model_dir / "cluster_assignments.csv", index=False)
        cluster_composition_table(predictions, model_dir / "cluster_label_composition.csv")
        with (model_dir / "clustering_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        summary_rows.append({"method": name, **metrics})

    summary = pd.DataFrame(summary_rows).sort_values(
        ["adjusted_mutual_info", "adjusted_rand_index"],
        ascending=[False, False],
    )
    summary.to_csv(run_dir / "exploratory_clustering_summary.csv", index=False)
    lines = [
        "# Exploratory Clustering",
        "",
        f"- representation: `{family}`",
        f"- split analysed: `{test_split}`",
        f"- sampled events per class limit: `{per_class}`",
        "",
        "This analysis is exploratory and unsupervised. It is included as feature-space evidence, not as a replacement for the supervised evaluation.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['method']}`: clusters_found={int(row['clusters_found'])}, "
            f"AMI={row['adjusted_mutual_info']:.4f}, ARI={row['adjusted_rand_index']:.4f}, "
            f"homogeneity={row['homogeneity']:.4f}, completeness={row['completeness']:.4f}, "
            f"noise_fraction={row['noise_fraction']:.4f}"
        )
    (run_dir / "exploratory_clustering_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory clustering on classical features.")
    parser.add_argument("--config", default="configs/classical_baselines.yaml")
    parser.add_argument("--manifest", default="data_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/reports/exploratory_clustering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = run_clustering(config, Path(args.manifest), Path(args.output_dir))
    print(f"Exploratory clustering written to {run_dir}")


if __name__ == "__main__":
    main()
