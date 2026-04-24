from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.representations import iter_rows_with_waveforms
from src.features.audio_features import (
    band_energy_feature_names,
    band_energy_from_waveform,
    handcrafted_audio_feature_names,
    handcrafted_audio_features_from_waveform,
    spectral_stats_feature_names,
    spectral_stats_from_waveform,
    waveform_stats_feature_names,
    waveform_stats_from_waveform,
)
from src.features.cache import feature_cache_key, load_feature_cache, save_feature_cache
from src.features.spectrogram_features import (
    hog_feature_names,
    hog_features_from_waveform,
    gradient_hist_feature_names,
    gradient_hist_features_from_waveform,
    logmel_statistics_feature_names,
    logmel_statistics_from_waveform,
    mfcc_feature_names,
    mfcc_features_from_waveform,
    patch_feature_names,
    patch_features_from_waveform,
)
from src.features.lowfreq_features import (
    notebook_lowfreq_feature_names,
    notebook_lowfreq_features_from_waveform,
    relative_lowfreq_feature_names,
    relative_lowfreq_features_from_waveform,
    temporal_lowfreq_feature_names,
    temporal_lowfreq_features_from_waveform,
)
from src.features.notebook_exact_features import (
    class_region_lowfreq_feature_names,
    class_region_lowfreq_features_from_waveform,
    notebook_exact_feature_names,
    notebook_exact_features_from_waveform,
)


AVAILABLE_FEATURE_SETS = {
    "handcrafted_stats": {"family": "handcrafted_stats"},
    "waveform_stats": {"family": "waveform_stats"},
    "spectral_stats": {"family": "spectral_stats"},
    "waveform_spectral_stats": {"family": "hybrid", "parts": ["waveform_stats", "spectral_stats"]},
    "band_energy_linear_8": {"family": "band_energy", "n_bands": 8, "scale": "linear"},
    "band_energy_linear_16": {"family": "band_energy", "n_bands": 16, "scale": "linear"},
    "band_energy_linear_32": {"family": "band_energy", "n_bands": 32, "scale": "linear"},
    "band_energy_linear_64": {"family": "band_energy", "n_bands": 64, "scale": "linear"},
    "band_energy_mel_8": {"family": "band_energy", "n_bands": 8, "scale": "mel"},
    "band_energy_mel_16": {"family": "band_energy", "n_bands": 16, "scale": "mel"},
    "band_energy_mel_32": {"family": "band_energy", "n_bands": 32, "scale": "mel"},
    "band_energy_mel_64": {"family": "band_energy", "n_bands": 64, "scale": "mel"},
    "mfcc_13": {"family": "mfcc", "n_mfcc": 13, "n_mels": 64},
    "mfcc_20": {"family": "mfcc", "n_mfcc": 20, "n_mels": 64},
    "mfcc_30": {"family": "mfcc", "n_mfcc": 30, "n_mels": 96},
    "mfcc_40": {"family": "mfcc", "n_mfcc": 40, "n_mels": 128},
    "mfcc_60": {"family": "mfcc", "n_mfcc": 60, "n_mels": 128},
    "logmel_stats_32": {"family": "logmel_stats", "n_mels": 32},
    "logmel_stats_64": {"family": "logmel_stats", "n_mels": 64},
    "logmel_stats_96": {"family": "logmel_stats", "n_mels": 96},
    "logmel_stats_128": {"family": "logmel_stats", "n_mels": 128},
    "patch_32x32": {"family": "patch", "width": 32, "height": 32},
    "patch_48x48": {"family": "patch", "width": 48, "height": 48},
    "patch_64x64": {"family": "patch", "width": 64, "height": 64},
    "patch_96x64": {"family": "patch", "width": 64, "height": 96},
    "patch_128x64": {"family": "patch", "width": 64, "height": 128},
    "classical_texture_32x32": {"family": "gradient_hist", "width": 32, "height": 32, "bins": 8},
    "classical_texture_64x64": {"family": "gradient_hist", "width": 64, "height": 64, "bins": 12},
    "notebook_lowfreq_band_features": {
        "family": "notebook_lowfreq",
        "grid_points": 32,
        "include_mean": True,
        "include_max": True,
        "include_global": True,
        "window": "gaussian",
        "window_seconds": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "db_clip": [-80.0, 0.0],
    },
    "notebook_lowfreq_mean_20": {
        "family": "notebook_lowfreq",
        "grid_points": 20,
        "include_mean": True,
        "include_max": False,
        "include_global": True,
        "window": "gaussian",
        "window_seconds": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "db_clip": [-80.0, 0.0],
    },
    "notebook_lowfreq_max_20": {
        "family": "notebook_lowfreq",
        "grid_points": 20,
        "include_mean": False,
        "include_max": True,
        "include_global": True,
        "window": "gaussian",
        "window_seconds": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "db_clip": [-80.0, 0.0],
    },
    "notebook_lowfreq_meanmax_64": {
        "family": "notebook_lowfreq",
        "grid_points": 64,
        "include_mean": True,
        "include_max": True,
        "include_global": False,
        "window": "gaussian",
        "window_seconds": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "db_clip": [-80.0, 0.0],
    },
    "relative_lowfreq_shape_features": {
        "family": "relative_lowfreq",
        "grid_points": 32,
        "window": "gaussian",
        "window_seconds": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "median_subtract": True,
        "db_clip": [-80.0, 0.0],
    },
    "temporal_lowfreq_shape_features": {
        "family": "temporal_lowfreq",
        "window": "gaussian",
        "window_seconds": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "total_energy_normalize": True,
    },
    "lowfreq_relative_temporal": {
        "family": "hybrid",
        "parts": ["relative_lowfreq_shape_features", "temporal_lowfreq_shape_features"],
    },
    "lowfreq_all": {
        "family": "hybrid",
        "parts": ["notebook_lowfreq_band_features", "relative_lowfreq_shape_features", "temporal_lowfreq_shape_features"],
    },
    "lowfreq_all_plus_mfcc": {
        "family": "hybrid",
        "parts": ["lowfreq_all", "mfcc_20"],
    },
    "lowfreq_all_plus_logmel": {
        "family": "hybrid",
        "parts": ["lowfreq_all", "logmel_stats_32"],
    },
    "lowfreq_all_plus_waveform_spectral": {
        "family": "hybrid",
        "parts": ["lowfreq_all", "waveform_spectral_stats"],
    },
    "notebook_exact_26": {
        "family": "notebook_exact",
        "variant": "26",
        "window_dur": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
    },
    "notebook_exact_44": {
        "family": "notebook_exact",
        "variant": "44",
        "window_dur": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
    },
    "notebook_exact_44_noclip": {
        "family": "notebook_exact",
        "variant": "44",
        "window_dur": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
    },
    "notebook_exact_44_dynrange": {
        "family": "notebook_exact",
        "variant": "44",
        "window_dur": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "dynamic_range_clip": True,
        "dynamic_range_db": 80.0,
    },
    "notebook_exact_44_plus_duration": {
        "family": "notebook_exact",
        "variant": "44",
        "window_dur": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "include_duration": True,
    },
    "notebook_exact_44_plus_bbox": {
        "family": "notebook_exact",
        "variant": "44",
        "window_dur": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
        "include_bbox": True,
        "rule_dependent": True,
    },
    "class_region_lowfreq_features": {
        "family": "class_region_lowfreq",
        "window_dur": 2.0,
        "f_min": 10.0,
        "f_max": 120.0,
    },
    "notebook_exact_44_plus_regions": {
        "family": "hybrid",
        "parts": ["notebook_exact_44", "class_region_lowfreq_features"],
    },
    "lowfreq_all_plus_regions": {
        "family": "hybrid",
        "parts": ["lowfreq_all", "class_region_lowfreq_features"],
    },
    "hybrid_spectral_band16": {
        "family": "hybrid",
        "parts": ["spectral_stats", "band_energy_linear_16"],
    },
    "hybrid_mfcc20_logmel64": {
        "family": "hybrid",
        "parts": ["mfcc_20", "logmel_stats_64"],
    },
    "hybrid_stats_mfcc20_logmel64": {
        "family": "hybrid",
        "parts": ["handcrafted_stats", "mfcc_20", "logmel_stats_64"],
    },
    "hybrid_stats_patch32": {
        "family": "hybrid",
        "parts": ["handcrafted_stats", "patch_32x32"],
    },
    "all_stats": {
        "family": "hybrid",
        "parts": ["waveform_stats", "spectral_stats", "band_energy_linear_16", "mfcc_20", "logmel_stats_64"],
    },
}


def list_feature_set_names() -> list[str]:
    return sorted(AVAILABLE_FEATURE_SETS)


def feature_spec_from_name(name: str | dict) -> dict:
    if isinstance(name, dict):
        return deepcopy(name)
    if name not in AVAILABLE_FEATURE_SETS:
        known = ", ".join(list_feature_set_names())
        raise ValueError(f"Unknown feature set '{name}'. Known feature sets: {known}")
    spec = deepcopy(AVAILABLE_FEATURE_SETS[name])
    spec["name"] = name
    return spec


def feature_names(spec: dict) -> list[str]:
    family = spec["family"]
    if family == "handcrafted_stats":
        return handcrafted_audio_feature_names()
    if family == "waveform_stats":
        return waveform_stats_feature_names()
    if family == "spectral_stats":
        return spectral_stats_feature_names()
    if family == "band_energy":
        return band_energy_feature_names(int(spec["n_bands"]), str(spec.get("scale", "linear")))
    if family == "mfcc":
        return mfcc_feature_names(int(spec["n_mfcc"]))
    if family == "logmel_stats":
        return logmel_statistics_feature_names(int(spec["n_mels"]))
    if family == "patch":
        return patch_feature_names(int(spec["width"]), int(spec["height"]))
    if family == "hog":
        return []
    if family == "gradient_hist":
        return gradient_hist_feature_names(int(spec["width"]), int(spec["height"]), int(spec.get("bins", 8)))
    if family == "notebook_lowfreq":
        return notebook_lowfreq_feature_names(spec)
    if family == "relative_lowfreq":
        return relative_lowfreq_feature_names(spec)
    if family == "temporal_lowfreq":
        return temporal_lowfreq_feature_names(spec)
    if family == "notebook_exact":
        return notebook_exact_feature_names(spec)
    if family == "class_region_lowfreq":
        return class_region_lowfreq_feature_names(spec)
    if family == "hybrid":
        names: list[str] = []
        for part in spec["parts"]:
            names.extend(feature_names(feature_spec_from_name(part)))
        return names
    raise ValueError(f"Unknown feature family: {family}")


def _fmin_fmax(audio_cfg: dict) -> tuple[float, float]:
    return float(audio_cfg.get("f_min", 0.0)), float(audio_cfg.get("f_max", 125.0))


def feature_vector_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, spec: dict) -> np.ndarray:
    family = spec["family"]
    effective_audio_cfg = deepcopy(audio_cfg)
    effective_audio_cfg.update(deepcopy(spec.get("audio", {})))
    if family == "handcrafted_stats":
        return handcrafted_audio_features_from_waveform(row, waveform, sample_rate, effective_audio_cfg)
    if family == "waveform_stats":
        return waveform_stats_from_waveform(row, waveform, sample_rate, effective_audio_cfg)
    if family == "spectral_stats":
        return spectral_stats_from_waveform(row, waveform, sample_rate, effective_audio_cfg)
    if family == "band_energy":
        return band_energy_from_waveform(
            row,
            waveform,
            sample_rate,
            effective_audio_cfg,
            n_bands=int(spec["n_bands"]),
            scale=str(spec.get("scale", "linear")),
        )
    if family == "mfcc":
        f_min, f_max = _fmin_fmax(effective_audio_cfg)
        return mfcc_features_from_waveform(
            row,
            waveform,
            sample_rate,
            effective_audio_cfg,
            n_mfcc=int(spec["n_mfcc"]),
            n_mels=int(spec.get("n_mels", max(64, int(spec["n_mfcc"]) * 2))),
            f_min=float(spec.get("f_min", f_min)),
            f_max=float(spec.get("f_max", f_max)),
        )
    if family == "logmel_stats":
        f_min, f_max = _fmin_fmax(effective_audio_cfg)
        return logmel_statistics_from_waveform(
            row,
            waveform,
            sample_rate,
            effective_audio_cfg,
            n_mels=int(spec["n_mels"]),
            f_min=float(spec.get("f_min", f_min)),
            f_max=float(spec.get("f_max", f_max)),
        )
    if family == "patch":
        return patch_features_from_waveform(
            row,
            waveform,
            sample_rate,
            effective_audio_cfg,
            width=int(spec["width"]),
            height=int(spec["height"]),
        )
    if family == "hog":
        return hog_features_from_waveform(
            row,
            waveform,
            sample_rate,
            effective_audio_cfg,
            width=int(spec["width"]),
            height=int(spec["height"]),
        )
    if family == "gradient_hist":
        return gradient_hist_features_from_waveform(
            row,
            waveform,
            sample_rate,
            effective_audio_cfg,
            width=int(spec["width"]),
            height=int(spec["height"]),
            bins=int(spec.get("bins", 8)),
        )
    if family == "notebook_lowfreq":
        return notebook_lowfreq_features_from_waveform(row, waveform, sample_rate, effective_audio_cfg, spec)
    if family == "relative_lowfreq":
        return relative_lowfreq_features_from_waveform(row, waveform, sample_rate, effective_audio_cfg, spec)
    if family == "temporal_lowfreq":
        return temporal_lowfreq_features_from_waveform(row, waveform, sample_rate, effective_audio_cfg, spec)
    if family == "notebook_exact":
        return notebook_exact_features_from_waveform(row, waveform, sample_rate, effective_audio_cfg, spec)
    if family == "class_region_lowfreq":
        return class_region_lowfreq_features_from_waveform(row, waveform, sample_rate, effective_audio_cfg, spec)
    if family == "hybrid":
        parts = [feature_vector_from_waveform(row, waveform, sample_rate, effective_audio_cfg, feature_spec_from_name(part)) for part in spec["parts"]]
        return np.concatenate(parts).astype(np.float32)
    raise ValueError(f"Unknown feature family: {family}")


def build_feature_matrix(
    frame: pd.DataFrame,
    manifest_path: str | Path,
    split_name: str,
    audio_cfg: dict,
    feature_name: str | dict,
    cache_cfg: dict | None = None,
    show_progress: bool = False,
    desc: str = "features",
) -> tuple[np.ndarray, pd.DataFrame, list[str], Path | None]:
    spec = feature_spec_from_name(feature_name)
    cache_cfg = cache_cfg or {}
    cache_enabled = bool(cache_cfg.get("enabled", False))
    cache_root = cache_cfg.get("root", "outputs/cache/features")
    row_identity = (
        frame[["dataset", "filename", "source_row", "label", "split"]]
        .fillna("")
        .astype(str)
        .to_dict("records")
    )
    cache_key = feature_cache_key(manifest_path, split_name, spec, audio_cfg, row_identity)
    if cache_enabled:
        cached = load_feature_cache(cache_root, cache_key)
        if cached is not None:
            return cached["X"], pd.DataFrame(cached["metadata"]), cached["feature_names"], Path(cached["cache_path"])

    vectors = []
    metadata = []
    for row_dict, waveform, sample_rate in iter_rows_with_waveforms(frame, show_progress=show_progress, desc=desc):
        vector = feature_vector_from_waveform(row_dict, waveform, sample_rate, audio_cfg, spec)
        if not np.all(np.isfinite(vector)):
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        vectors.append(vector.astype(np.float32))
        metadata_row = {
            "split": row_dict.get("split", ""),
            "dataset": row_dict.get("dataset", ""),
            "filename": row_dict.get("filename", ""),
            "source_row": row_dict.get("source_row", ""),
            "label": row_dict.get("label", ""),
            "label_raw": row_dict.get("label_raw", ""),
            "audio_path": row_dict.get("audio_path", ""),
            "low_frequency": row_dict.get("low_frequency", ""),
            "high_frequency": row_dict.get("high_frequency", ""),
            "duration_seconds": row_dict.get("duration_seconds", ""),
            "real_duration_seconds": row_dict.get("real_duration_seconds", ""),
            "clip_start_seconds": row_dict.get("clip_start_seconds", ""),
            "clip_end_seconds": row_dict.get("clip_end_seconds", ""),
        }
        for optional_column in ["recording_id", "session_id", "site", "domain", "audio_id", "annotator"]:
            if optional_column in row_dict:
                metadata_row[optional_column] = row_dict.get(optional_column, "")
        metadata.append(metadata_row)
    X = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
    metadata_frame = pd.DataFrame(metadata)
    feature_name_list = feature_names(spec) if X.size else []
    cache_path = None
    if cache_enabled:
        cache_path = save_feature_cache(
            cache_root,
            cache_key,
            {
                "X": X,
                "metadata": metadata,
                "feature_names": feature_name_list,
                "cache_path": str(Path(cache_root) / cache_key[:2] / f"{cache_key}.joblib"),
            },
        )
    return X, metadata_frame, feature_name_list, cache_path
