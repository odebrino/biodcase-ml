from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from src.classical.baselines import MODEL_NAMES, make_pipeline, run_classical_baselines


def write_sine(path: Path, frequency: float, seconds: float = 2.0, sample_rate: int = 250) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    wavfile.write(path, sample_rate, (0.5 * np.sin(2 * np.pi * frequency * t)).astype("float32"))


def manifest_row(root: Path, split: str, dataset: str, label: str, frequency: float, idx: int) -> dict:
    audio_path = root / split / "audio" / dataset / f"event_{idx}.wav"
    write_sine(audio_path, frequency=frequency)
    return {
        "split": split,
        "dataset": dataset,
        "filename": audio_path.name,
        "audio_path": str(audio_path),
        "label": label,
        "label_raw": label,
        "source_row": idx,
        "start_seconds": 0.25,
        "end_seconds": 1.25,
        "clip_start_seconds": 0.25,
        "clip_end_seconds": 1.25,
        "duration_seconds": 1.0,
        "real_duration_seconds": 1.0,
        "low_frequency": 15.0 if label == "bma" else 35.0,
        "high_frequency": 35.0 if label == "bma" else 55.0,
        "valid_event": True,
    }


def tiny_config() -> dict:
    return {
        "classes": ["bma", "bmb"],
        "train_split": "train",
        "test_split": "validation",
        "val_split": "validation",
        "held_out_test_domains": ["heldout"],
        "seed": 0,
        "audio": {
            "sample_rate": 250,
            "margin_seconds": 0.0,
            "n_fft": 64,
            "win_length": 64,
            "hop_length": 16,
            "n_mels": 32,
            "f_min": 0.0,
            "f_max": 125.0,
            "frequency_scale": "linear",
            "normalization": "sample",
        },
        "classical": {
            "representations": ["handcrafted"],
            "models": ["gaussian_nb", "random_forest"],
            "img_size": 16,
            "validation_fraction": 0.5,
            "limit_per_split": None,
            "pca": {"enabled": False},
        },
    }


def test_required_classical_model_registry_is_available():
    expected = {
        "logistic_regression",
        "linear_svm",
        "rbf_svm",
        "knn",
        "gaussian_nb",
        "random_forest",
        "gradient_boosted_trees",
        "mlp",
    }

    assert set(MODEL_NAMES) == expected
    for model_name in MODEL_NAMES:
        pipeline = make_pipeline(model_name, seed=0, pca_components=None)
        assert pipeline.steps[-1][0] == "model"


def test_classical_baseline_driver_writes_leakage_safe_artifacts(tmp_path):
    data_root = tmp_path / "data"
    rows = [
        manifest_row(data_root, "train", "train_a", "bma", 25.0, 0),
        manifest_row(data_root, "train", "train_a", "bmb", 45.0, 1),
        manifest_row(data_root, "train", "train_b", "bma", 25.0, 2),
        manifest_row(data_root, "train", "train_b", "bmb", 45.0, 3),
        manifest_row(data_root, "validation", "heldout", "bma", 25.0, 4),
        manifest_row(data_root, "validation", "heldout", "bmb", 45.0, 5),
    ]
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    run_dir = run_classical_baselines(tiny_config(), manifest_path, tmp_path / "outputs")

    results = pd.read_csv(run_dir / "results.csv")
    strategy = (run_dir / "split_strategy.json").read_text(encoding="utf-8")
    assert set(results["model"]) == {"gaussian_nb", "random_forest"}
    assert set(results["representation"]) == {"handcrafted"}
    assert "Official held-out test rows are never used" in strategy
    assert (run_dir / "handcrafted" / "gaussian_nb" / "selection_predictions.csv").exists()
    assert (run_dir / "handcrafted" / "gaussian_nb" / "test_predictions.csv").exists()
    assert (run_dir / "all_test_predictions.csv").exists()
