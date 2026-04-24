from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.fftpack import dct

from src.data.spectrogram import (
    crop_event,
    literal_time_frequency_crop_from_waveform,
    prepare_waveform,
    require_torch,
    spectrogram_frame,
)


def _resize_values(values, height: int, width: int) -> np.ndarray:
    torch = require_torch()
    tensor = torch.as_tensor(values, dtype=torch.float32)
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def _logmel_crop_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    mel_cfg = deepcopy(audio_cfg)
    mel_cfg["frequency_scale"] = "mel"
    mel_cfg["n_mels"] = int(n_mels)
    mel_cfg["f_min"] = float(f_min)
    mel_cfg["f_max"] = float(f_max)
    waveform, sample_rate = prepare_waveform(waveform, sample_rate, mel_cfg)
    start_s = float(row.get("clip_start_seconds", row.get("start_seconds", 0.0)))
    end_s = float(row.get("clip_end_seconds", row.get("end_seconds", start_s)))
    segment = crop_event(waveform, sample_rate, start_s, end_s, 0.0)
    frame = spectrogram_frame(segment, sample_rate, mel_cfg, start_offset_s=start_s)
    return frame.values.detach().cpu().numpy().astype(np.float64)


def logmel_statistics_feature_names(n_mels: int) -> list[str]:
    names = []
    for idx in range(n_mels):
        names.extend(
            [
                f"logmel_band_{idx}_mean",
                f"logmel_band_{idx}_std",
                f"logmel_band_{idx}_min",
                f"logmel_band_{idx}_max",
                f"logmel_band_{idx}_q10",
                f"logmel_band_{idx}_q50",
                f"logmel_band_{idx}_q90",
            ]
        )
    pooled_names = []
    for axis_name in ["time", "frequency"]:
        for statistic in ["mean", "std", "min", "max"]:
            pooled_names.append(f"{axis_name}_pooled_{statistic}")
    return names + pooled_names


def logmel_statistics_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    values = _logmel_crop_from_waveform(row, waveform, sample_rate, audio_cfg, n_mels=n_mels, f_min=f_min, f_max=f_max)
    if values.size == 0:
        return np.zeros(len(logmel_statistics_feature_names(n_mels)), dtype=np.float32)
    band_means = values.mean(axis=1)
    band_stds = values.std(axis=1)
    band_mins = values.min(axis=1)
    band_maxs = values.max(axis=1)
    band_q10 = np.percentile(values, 10, axis=1)
    band_q50 = np.percentile(values, 50, axis=1)
    band_q90 = np.percentile(values, 90, axis=1)
    pooled = np.asarray(
        [
            float(values.mean(axis=0).mean()),
            float(values.mean(axis=0).std()),
            float(values.mean(axis=0).min()),
            float(values.mean(axis=0).max()),
            float(values.mean(axis=1).mean()),
            float(values.mean(axis=1).std()),
            float(values.mean(axis=1).min()),
            float(values.mean(axis=1).max()),
        ],
        dtype=np.float32,
    )
    stacked = np.column_stack([band_means, band_stds, band_mins, band_maxs, band_q10, band_q50, band_q90]).reshape(-1)
    return np.concatenate([stacked.astype(np.float32), pooled]).astype(np.float32)


def mfcc_feature_names(n_mfcc: int) -> list[str]:
    names = []
    for prefix, stats in [
        ("mfcc", ("mean", "std", "min", "max")),
        ("delta_mfcc", ("mean", "std")),
        ("delta2_mfcc", ("mean", "std")),
    ]:
        for idx in range(n_mfcc):
            for stat in stats:
                names.append(f"{prefix}_{idx}_{stat}")
    return names


def mfcc_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, n_mfcc: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    values = _logmel_crop_from_waveform(row, waveform, sample_rate, audio_cfg, n_mels=n_mels, f_min=f_min, f_max=f_max)
    if values.size == 0:
        return np.zeros(len(mfcc_feature_names(n_mfcc)), dtype=np.float32)
    coeffs = dct(values, axis=0, norm="ortho")[:n_mfcc]
    delta = np.diff(coeffs, axis=1, prepend=coeffs[:, :1])
    delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
    base = np.column_stack([coeffs.mean(axis=1), coeffs.std(axis=1), coeffs.min(axis=1), coeffs.max(axis=1)]).reshape(-1)
    delta_stats = np.column_stack([delta.mean(axis=1), delta.std(axis=1)]).reshape(-1)
    delta2_stats = np.column_stack([delta2.mean(axis=1), delta2.std(axis=1)]).reshape(-1)
    return np.concatenate([base, delta_stats, delta2_stats]).astype(np.float32)


def patch_feature_names(width: int, height: int) -> list[str]:
    return [f"patch_{height}x{width}_{idx}" for idx in range(width * height)]


def patch_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, width: int, height: int) -> np.ndarray:
    crop = literal_time_frequency_crop_from_waveform(waveform, sample_rate, row, audio_cfg)
    values = crop.values.detach().cpu().numpy().astype(np.float32)
    patch = _resize_values(values, height=height, width=width)
    return patch.reshape(-1).astype(np.float32)


def hog_feature_names(width: int, height: int, size: int) -> list[str]:
    return [f"hog_{height}x{width}_{idx}" for idx in range(size)]


def hog_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, width: int, height: int) -> np.ndarray:
    try:
        from skimage.feature import hog
    except ImportError as exc:
        raise RuntimeError("HOG features require scikit-image, which is not installed.") from exc
    patch = patch_features_from_waveform(row, waveform, sample_rate, audio_cfg, width=width, height=height).reshape(height, width)
    features = hog(
        patch,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(features, dtype=np.float32)


def gradient_hist_feature_names(width: int, height: int, bins: int = 8) -> list[str]:
    return [f"gradient_hist_{height}x{width}_bin_{idx}" for idx in range(bins)] + [
        f"gradient_hist_{height}x{width}_magnitude_mean",
        f"gradient_hist_{height}x{width}_magnitude_std",
    ]


def gradient_hist_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, width: int, height: int, bins: int = 8) -> np.ndarray:
    patch = patch_features_from_waveform(row, waveform, sample_rate, audio_cfg, width=width, height=height).reshape(height, width)
    grad_y, grad_x = np.gradient(patch.astype(np.float64))
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = (np.arctan2(grad_y, grad_x) + np.pi) / (2.0 * np.pi)
    hist, _ = np.histogram(orientation, bins=bins, range=(0.0, 1.0), weights=magnitude)
    hist = hist.astype(np.float64)
    hist = hist / max(float(hist.sum()), 1e-12)
    return np.concatenate(
        [
            hist,
            np.asarray([float(magnitude.mean()), float(magnitude.std())], dtype=np.float64),
        ]
    ).astype(np.float32)
