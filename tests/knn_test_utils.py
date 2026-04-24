from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


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
        "low_frequency": 15.0 if label in {"bma", "zzz"} else 35.0,
        "high_frequency": 35.0 if label in {"bma", "zzz"} else 55.0,
        "valid_event": True,
    }


def tiny_audio_config() -> dict:
    return {
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
    }


def tiny_knn_config(feature_set: str = "handcrafted_stats") -> dict:
    return {
        "classes": ["bma", "bmb"],
        "label_aggregation": {
            "enabled": False,
            "name": "notebook_3class_abz_ddswp_20hz20plus",
            "mapping": {
                "bma": "ABZ",
                "bmb": "ABZ",
                "bmz": "ABZ",
                "bmd": "DDswp",
                "bpd": "DDswp",
                "bp20": "20Hz20Plus",
                "bp20plus": "20Hz20Plus",
            },
        },
        "train_split": "train",
        "test_split": "validation",
        "val_split": "validation",
        "held_out_test_domains": ["heldout"],
        "seed": 0,
        "audio": tiny_audio_config(),
        "submission": {"feature_set": feature_set},
        "feature_cache": {
            "enabled": True,
            "root": "outputs/cache/features",
        },
        "preprocessing": {
            "imputer": "median",
            "scaler": "standard",
            "reducer": {"type": "none"},
        },
        "knn": {
            "n_neighbors": 1,
            "weights": "distance",
            "metric": "euclidean",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "limit_per_split": None,
        },
    }


def tiny_search_config() -> dict:
    config = tiny_knn_config()
    config["search"] = {
        "cv": {
            "strategy": "auto",
            "primary": "domain_aware",
            "group_column": "dataset",
            "n_splits": 2,
            "max_leave_one_group_splits": 8,
        },
        "domain_cv_top_n": 2,
        "notebook_ablation": {
            "feature_sets": [
                "waveform_spectral_stats",
                "handcrafted_stats",
                "notebook_lowfreq_band_features",
                "relative_lowfreq_shape_features",
                "temporal_lowfreq_shape_features",
                "lowfreq_relative_temporal",
            ],
            "imputer": "median",
            "scaler": "robust_l2",
            "reducer": {"type": "none"},
            "n_neighbors": 1,
            "weights": "uniform",
            "metric": "cosine",
            "algorithm": "brute",
            "leaf_size": 30,
            "p": 2,
        },
        "stages": [
            {
                "name": "broad",
                "feature_sets": [
                    "handcrafted_stats",
                    "waveform_stats",
                    "spectral_stats",
                    "band_energy_linear_8",
                    "notebook_lowfreq_band_features",
                    "relative_lowfreq_shape_features",
                    "temporal_lowfreq_shape_features",
                    "lowfreq_relative_temporal",
                ],
                "imputers": ["median", "mean"],
                "scalers": ["standard", "standard_l2"],
                "reducers": [{"type": "none"}, {"type": "pca_components", "value": 4}],
                "n_neighbors": [1, 3],
                "weights": ["uniform", "distance"],
                "metrics": ["euclidean", "cosine"],
                "algorithms": ["auto", "brute"],
                "leaf_sizes": [20, 30],
                "p_values": [1, 2],
                "max_candidates": 8,
                "ensure_feature_coverage": True,
                "top_feature_sets": 2,
            }
        ],
    }
    return config


def make_small_manifest(root: Path, include_unknown_validation_label: bool = False) -> pd.DataFrame:
    rows = [
        manifest_row(root, "train", "train_a", "bma", 25.0, 0),
        manifest_row(root, "train", "train_a", "bmb", 45.0, 1),
        manifest_row(root, "train", "train_b", "bma", 25.0, 2),
        manifest_row(root, "train", "train_b", "bmb", 45.0, 3),
        manifest_row(root, "train", "train_c", "bma", 25.0, 4),
        manifest_row(root, "train", "train_c", "bmb", 45.0, 5),
        manifest_row(root, "train", "train_d", "bma", 25.0, 6),
        manifest_row(root, "train", "train_d", "bmb", 45.0, 7),
        manifest_row(root, "validation", "heldout", "bma", 25.0, 8),
        manifest_row(root, "validation", "heldout", "zzz" if include_unknown_validation_label else "bmb", 45.0, 9),
    ]
    return pd.DataFrame(rows)
