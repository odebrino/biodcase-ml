from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiments.notebook_reproduction import notebook_exact_leaky_audit, notebook_exact_split_safe
from src.features.feature_sets import build_feature_matrix, feature_names, feature_spec_from_name
from src.features.notebook_exact_features import (
    NOTEBOOK_P_REF,
    notebook_exact_event_signal,
    notebook_exact_spectrogram,
)
from src.models.knn_pipeline import apply_label_aggregation, build_model_pipeline
from tests.knn_test_utils import manifest_row, tiny_audio_config, tiny_knn_config


def _three_class_manifest(root: Path) -> pd.DataFrame:
    specs = [
        ("bma", 25.0),
        ("bmd", 45.0),
        ("bp20", 95.0),
        ("bma", 26.0),
        ("bmd", 50.0),
        ("bp20", 100.0),
    ]
    rows = [manifest_row(root, "train", "elephantisland2014", label, freq, idx) for idx, (label, freq) in enumerate(specs)]
    return pd.DataFrame(rows)


def test_notebook_exact_feature_dimensions_and_finite_values(tmp_path):
    manifest = _three_class_manifest(tmp_path / "data")
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    config = tiny_knn_config()
    config["feature_cache"] = {"enabled": False}

    X26, _, names26, _ = build_feature_matrix(
        manifest,
        manifest_path=manifest_path,
        split_name="train",
        audio_cfg=config["audio"],
        feature_name="notebook_exact_26",
        cache_cfg={"enabled": False},
    )
    X44, _, names44, _ = build_feature_matrix(
        manifest,
        manifest_path=manifest_path,
        split_name="train",
        audio_cfg=config["audio"],
        feature_name="notebook_exact_44",
        cache_cfg={"enabled": False},
    )

    assert X26.shape == (6, 26)
    assert len(names26) == 26
    assert X44.shape == (6, 44)
    assert len(names44) == 44
    assert np.isfinite(X26).all()
    assert np.isfinite(X44).all()


def test_notebook_exact_uses_full_event_segment_not_fixed_two_second_crop():
    sample_rate = 250
    waveform = torch.zeros(1, sample_rate * 3)
    row = {
        "clip_start_seconds": 0.25,
        "clip_end_seconds": 1.25,
        "start_seconds": 0.25,
        "end_seconds": 1.25,
        "duration_seconds": 1.0,
    }

    signal, returned_rate = notebook_exact_event_signal(row, waveform, sample_rate, tiny_audio_config())

    assert returned_rate == sample_rate
    assert len(signal) == sample_rate
    assert len(signal) != sample_rate * 2


def test_notebook_exact_uses_scipy_spectrogram_and_pref_scaling():
    source = inspect.getsource(notebook_exact_spectrogram)

    assert "scipy_signal.spectrogram" in source
    assert NOTEBOOK_P_REF == 2e-5
    assert "NOTEBOOK_P_REF**2" in source


def test_notebook_exact_plus_metadata_feature_names():
    assert len(feature_names(feature_spec_from_name("notebook_exact_44_plus_duration"))) == 45
    assert len(feature_names(feature_spec_from_name("notebook_exact_44_plus_bbox"))) == 48


def test_notebook_reproduction_flags_and_split_safe_pipeline(tmp_path):
    manifest = _three_class_manifest(tmp_path / "data")
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    config = tiny_knn_config("notebook_exact_44")
    config["classes"] = ["ABZ", "DDswp", "20Hz20Plus"]
    config["label_aggregation"]["enabled"] = True
    config["feature_cache"] = {"enabled": False}
    config["notebook_reproduction"] = {"test_size": 0.5, "n_splits": 2}
    frame = apply_label_aggregation(manifest, config)

    leaky = notebook_exact_leaky_audit(frame, manifest_path, config, config["classes"], "notebook_exact_44", "elephantisland2014")
    safe = notebook_exact_split_safe(frame, manifest_path, config, config["classes"], "notebook_exact_44", "elephantisland2014", "3class_notebook_cv")

    assert leaky["diagnostic_only"] is True
    assert leaky["not_strict_split_safe"] is True
    assert leaky["not_for_model_selection"] is True
    assert safe["diagnostic_only"] is False
    assert safe["not_strict_split_safe"] is False
    assert safe["scaler_fit_scope"] == "inside_sklearn_pipeline_cv_train_folds_only"


def test_3class_mapping_and_7class_default_modes():
    frame = pd.DataFrame({"label": ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]})
    config = tiny_knn_config()
    unchanged = apply_label_aggregation(frame, config)
    assert unchanged["label"].tolist() == frame["label"].tolist()

    config["label_aggregation"]["enabled"] = True
    mapped = apply_label_aggregation(frame, config)
    assert mapped["label"].tolist() == ["ABZ", "ABZ", "ABZ", "DDswp", "DDswp", "20Hz20Plus", "20Hz20Plus"]


def test_best_pipeline_estimator_is_knn():
    pipeline = build_model_pipeline(
        {"imputer": "median", "scaler": "standard", "reducer": {"type": "pca_components", "value": 1024}},
        {"n_neighbors": 3, "weights": "uniform", "metric": "minkowski", "algorithm": "auto", "leaf_size": 30, "p": 2},
        seed=0,
        fit_rows=10,
    )

    assert pipeline.named_steps["model"].__class__.__name__ == "KNeighborsClassifier"
