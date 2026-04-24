import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.feature_sets import build_feature_matrix
from src.models.knn_pipeline import build_model_pipeline, run_knn_submission
from tests.knn_test_utils import make_small_manifest, tiny_knn_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_feature_extraction_shapes_and_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = make_small_manifest(tmp_path / "data")
    train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
    config = tiny_knn_config()
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    for feature_name in [
        "handcrafted_stats",
        "waveform_stats",
        "spectral_stats",
        "band_energy_linear_8",
        "notebook_lowfreq_band_features",
        "relative_lowfreq_shape_features",
        "temporal_lowfreq_shape_features",
        "lowfreq_relative_temporal",
        "mfcc_13",
        "logmel_stats_32",
        "patch_32x32",
        "classical_texture_32x32",
    ]:
        X, metadata, feature_names, _ = build_feature_matrix(
            train_frame,
            manifest_path=manifest_path,
            split_name="train",
            audio_cfg=config["audio"],
            feature_name=feature_name,
            cache_cfg=config["feature_cache"],
            show_progress=False,
        )
        assert X.shape[0] == len(train_frame)
        assert X.shape[1] == len(feature_names)
        assert len(metadata) == len(train_frame)
        assert np.isfinite(X).all()


def test_lowfreq_per_sample_normalization_is_independent_per_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = make_small_manifest(tmp_path / "data")
    train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    config = tiny_knn_config()
    feature_spec = {
        "name": "relative_lowfreq_scaled_test",
        "family": "relative_lowfreq",
        "grid_points": 20,
        "window_seconds": 2.0,
        "total_energy_normalize": True,
        "median_subtract": True,
        "frequency_profile_normalize": True,
        "db_clip": [-80.0, 0.0],
    }

    X_full, _, names_full, _ = build_feature_matrix(
        train_frame,
        manifest_path=manifest_path,
        split_name="train",
        audio_cfg=config["audio"],
        feature_name=feature_spec,
        cache_cfg={"enabled": False},
        show_progress=False,
    )
    X_single, _, names_single, _ = build_feature_matrix(
        train_frame.iloc[[0]].reset_index(drop=True),
        manifest_path=manifest_path,
        split_name="train",
        audio_cfg=config["audio"],
        feature_name=feature_spec,
        cache_cfg={"enabled": False},
        show_progress=False,
    )
    assert names_full == names_single
    assert np.allclose(X_full[0], X_single[0])
    assert np.isfinite(X_full).all()


def test_preprocessing_fits_only_on_training_values_and_removes_nan_inf():
    pipeline = build_model_pipeline(
        preprocessing={"imputer": "mean", "scaler": "standard", "reducer": {"type": "none"}},
        knn_cfg={"n_neighbors": 1, "weights": "uniform", "metric": "euclidean", "algorithm": "auto", "leaf_size": 30, "p": 2},
        seed=0,
        fit_rows=3,
    )
    X_train = np.asarray([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]], dtype=np.float32)
    y_train = np.asarray([0, 1, 0], dtype=np.int64)
    pipeline.fit(X_train, y_train)

    finite = pipeline.named_steps["finite"].transform(np.asarray([[1000.0, np.inf]], dtype=np.float32))
    imputed = pipeline.named_steps["imputer"].transform(finite)
    scaled = pipeline.named_steps["scaler"].transform(imputed)

    assert pipeline.named_steps["imputer"].statistics_[0] == 3.0
    assert pipeline.named_steps["imputer"].statistics_[1] == 3.0
    assert imputed[0, 1] == 3.0
    assert np.isfinite(scaled).all()


def test_knn_submission_driver_writes_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = make_small_manifest(tmp_path / "data")
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "outputs" / "reports" / "model_evaluation"
    run_dir = run_knn_submission(tiny_knn_config(), manifest_path, output_dir)

    results = pd.read_csv(run_dir / "official_test_results.csv")
    strategy = (run_dir / "split_strategy.json").read_text(encoding="utf-8")
    assert set(results["model"]) == {"knn"}
    assert set(results["feature_set"]) == {"handcrafted_stats"}
    assert "Official held-out test rows are never used" in strategy
    assert (run_dir / "official_test_metrics.json").exists()
    assert (run_dir / "official_test_predictions.csv").exists()
    assert (run_dir / "official_test_confusion_matrix.csv").exists()
    assert (run_dir / "domain_diagnostics.json").exists()
    assert (run_dir / "domain_diagnostics.md").exists()
    assert (tmp_path / "outputs" / "reports" / "quality" / "label_mapping_audit.json").exists()
    assert (tmp_path / "outputs" / "reports" / "quality" / "label_mapping_audit.md").exists()
    assert (run_dir / "per_sample_knn_neighbors.csv").exists()
    assert (run_dir / "main_experiment_overview.md").exists()
    assert (run_dir / "selected_knn_config.yaml").exists()
    assert (run_dir / "knn_pipeline.joblib").exists()
    assert (run_dir / "handcrafted_stats" / "knn" / "official_test_metrics.json").exists()
    assert (tmp_path / "outputs" / "reports" / "quality" / "split_integrity.json").exists()
    neighbors = pd.read_csv(run_dir / "per_sample_knn_neighbors.csv")
    assert {"nearest_distance", "mean_neighbor_distance", "nearest_train_label"}.issubset(neighbors.columns)


def test_run_submission_cli_supports_single_entry_point(tmp_path):
    manifest = make_small_manifest(tmp_path / "data")
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    config_path = tmp_path / "knn.yaml"
    config_path.write_text(yaml.safe_dump(tiny_knn_config(), sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "submission_outputs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.run_submission",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "main_experiment_overview.md").exists()
    assert (output_dir / "official_test_confusion_matrix.csv").exists()
    assert (output_dir / "domain_diagnostics.json").exists()


def test_knn_submission_path_does_not_depend_on_legacy_cnn():
    import src.models.knn_pipeline as knn_pipeline
    import src.models.knn_search as knn_search

    assert "legacy.cnn" not in inspect.getsource(knn_pipeline)
    assert "legacy.cnn" not in inspect.getsource(knn_search)
