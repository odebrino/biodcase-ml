from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.signal import find_peaks, stft, windows

from src.data.spectrogram import crop_event, prepare_waveform


LOWFREQ_DEFAULT_RANGE = (10.0, 120.0)
LOWFREQ_GRID_OPTIONS = (20, 32, 64)


def _event_signal(row: dict, waveform, sample_rate: int, audio_cfg: dict, spec: dict) -> tuple[np.ndarray, int]:
    cfg = deepcopy(audio_cfg)
    cfg.update(deepcopy(spec.get("audio", {})))
    waveform, sample_rate = prepare_waveform(waveform, sample_rate, cfg)
    start_s = float(row.get("clip_start_seconds", row.get("start_seconds", 0.0)))
    end_s = float(row.get("clip_end_seconds", row.get("end_seconds", start_s)))
    window_seconds = spec.get("window_seconds", cfg.get("lowfreq_window_seconds", 2.0))
    if window_seconds:
        window = float(window_seconds)
        center = 0.5 * (start_s + end_s)
        start_s = max(0.0, center - 0.5 * window)
        end_s = start_s + window
    else:
        margin = float(cfg.get("crop_padding_seconds", 0.0))
        start_s = max(0.0, start_s - margin)
        end_s = end_s + margin
    segment = crop_event(waveform, sample_rate, start_s, end_s, 0.0)
    signal = segment.squeeze(0).detach().cpu().numpy().astype(np.float64)
    return signal, int(sample_rate)


def _lowfreq_spectrogram(row: dict, waveform, sample_rate: int, audio_cfg: dict, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal, sample_rate = _event_signal(row, waveform, sample_rate, audio_cfg, spec)
    if signal.size == 0:
        signal = np.zeros(1, dtype=np.float64)

    n_fft = int(spec.get("n_fft", audio_cfg.get("n_fft", 512)))
    win_length = int(spec.get("win_length", audio_cfg.get("win_length", min(n_fft, max(16, signal.size)))))
    win_length = max(16, min(win_length, max(16, signal.size)))
    n_fft = max(n_fft, win_length)
    hop_length = int(spec.get("hop_length", audio_cfg.get("hop_length", max(1, win_length // 10))))
    hop_length = max(1, min(hop_length, win_length))
    if signal.size < win_length:
        signal = np.pad(signal, (0, win_length - signal.size), mode="constant")

    window_name = str(spec.get("window", "gaussian")).lower()
    if window_name == "gaussian":
        std = float(spec.get("gaussian_std", max(1.0, win_length / 6.0)))
        window = windows.gaussian(win_length, std=std, sym=True)
    else:
        window = window_name

    noverlap = max(0, min(win_length - 1, win_length - hop_length))
    freqs, _, spectrum = stft(
        signal,
        fs=sample_rate,
        window=window,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum).astype(np.float64) ** 2
    f_min = float(spec.get("f_min", LOWFREQ_DEFAULT_RANGE[0]))
    f_max = min(float(spec.get("f_max", LOWFREQ_DEFAULT_RANGE[1])), sample_rate / 2.0)
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        freqs = np.linspace(f_min, f_max, max(2, int(spec.get("grid_points", 32))), dtype=np.float64)
        power = np.zeros((len(freqs), 1), dtype=np.float64)
    else:
        freqs = freqs[mask].astype(np.float64)
        power = power[mask].astype(np.float64)
    power = np.maximum(power, 1e-20)

    if bool(spec.get("total_energy_normalize", False)):
        power = power / max(float(np.sum(power)), 1e-20)

    compression = str(spec.get("compression", "db")).lower()
    if compression == "log1p":
        values = np.log1p(power)
    elif compression == "log":
        values = np.log(power)
    else:
        values = 10.0 * np.log10(power)

    clip_range = spec.get("db_clip", audio_cfg.get("lowfreq_db_clip"))
    if clip_range is not None and len(clip_range) == 2:
        values = np.clip(values, float(clip_range[0]), float(clip_range[1]))

    percentile_clip = spec.get("percentile_clip")
    if percentile_clip is not None and len(percentile_clip) == 2:
        low, high = np.percentile(values, [float(percentile_clip[0]), float(percentile_clip[1])])
        values = np.clip(values, low, high)

    if bool(spec.get("median_subtract", False)):
        values = values - float(np.median(values))

    if bool(spec.get("frequency_profile_normalize", False)):
        profile_median = np.median(values, axis=1, keepdims=True)
        profile_iqr = np.percentile(values, 75, axis=1, keepdims=True) - np.percentile(values, 25, axis=1, keepdims=True)
        values = (values - profile_median) / np.maximum(profile_iqr, 1e-6)

    return freqs.astype(np.float64), power.astype(np.float64), values.astype(np.float64)


def _grid(spec: dict) -> np.ndarray:
    grid_points = int(spec.get("grid_points", 32))
    return np.linspace(
        float(spec.get("f_min", LOWFREQ_DEFAULT_RANGE[0])),
        float(spec.get("f_max", LOWFREQ_DEFAULT_RANGE[1])),
        grid_points,
        dtype=np.float64,
    )


def _interp_profile(freqs: np.ndarray, profile: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if freqs.size == 0 or profile.size == 0:
        return np.zeros(len(grid), dtype=np.float64)
    return np.interp(grid, freqs, profile, left=float(profile[0]), right=float(profile[-1])).astype(np.float64)


def notebook_lowfreq_feature_names(spec: dict) -> list[str]:
    grid_points = int(spec.get("grid_points", 32))
    include_global = bool(spec.get("include_global", True))
    include_mean = bool(spec.get("include_mean", True))
    include_max = bool(spec.get("include_max", True))
    names: list[str] = []
    if include_global:
        names.extend(["lowfreq_global_mean", "lowfreq_global_std", "lowfreq_global_min", "lowfreq_global_max"])
    if include_mean:
        names.extend([f"lowfreq_grid_{idx}_mean" for idx in range(grid_points)])
    if include_max:
        names.extend([f"lowfreq_grid_{idx}_max" for idx in range(grid_points)])
    return names


def notebook_lowfreq_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, spec: dict) -> np.ndarray:
    freqs, _, values = _lowfreq_spectrogram(row, waveform, sample_rate, audio_cfg, spec)
    grid = _grid(spec)
    include_global = bool(spec.get("include_global", True))
    include_mean = bool(spec.get("include_mean", True))
    include_max = bool(spec.get("include_max", True))
    features: list[float] = []
    if include_global:
        features.extend([float(values.mean()), float(values.std()), float(values.min()), float(values.max())])
    if include_mean:
        features.extend(_interp_profile(freqs, values.mean(axis=1), grid).tolist())
    if include_max:
        features.extend(_interp_profile(freqs, values.max(axis=1), grid).tolist())
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def relative_lowfreq_feature_names(spec: dict) -> list[str]:
    grid_points = int(spec.get("grid_points", 32))
    names: list[str] = []
    names.extend([f"lowfreq_grid_{idx}_energy_fraction" for idx in range(grid_points)])
    names.extend([f"lowfreq_grid_{idx}_median_centered_mean" for idx in range(grid_points)])
    names.extend([f"lowfreq_grid_{idx}_median_centered_max" for idx in range(grid_points)])
    names.extend(
        [
            "lowfreq_ratio_10_30_to_30_60",
            "lowfreq_ratio_10_30_to_60_120",
            "lowfreq_ratio_30_60_to_60_120",
            "lowfreq_fraction_20_30",
            "lowfreq_fraction_30_60",
            "lowfreq_fraction_60_120",
            "lowfreq_peak_frequency_hz",
            "lowfreq_peak_energy_relative_to_median",
            "lowfreq_centroid_hz",
            "lowfreq_bandwidth_hz",
            "lowfreq_entropy",
        ]
    )
    return names


def _band_energy(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(power[mask].sum()) if np.any(mask) else 0.0


def relative_lowfreq_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, spec: dict) -> np.ndarray:
    freqs, power, values = _lowfreq_spectrogram(row, waveform, sample_rate, audio_cfg, spec)
    grid = _grid(spec)
    mean_power = power.mean(axis=1)
    max_power = power.max(axis=1)
    mean_values = values.mean(axis=1)
    max_values = values.max(axis=1)
    profile = _interp_profile(freqs, mean_power, grid)
    profile_max = _interp_profile(freqs, max_power, grid)
    value_profile = _interp_profile(freqs, mean_values, grid)
    value_max_profile = _interp_profile(freqs, max_values, grid)
    energy_fraction = profile / max(float(profile.sum()), 1e-20)
    centered_mean = value_profile - float(np.median(value_profile))
    centered_max = value_max_profile - float(np.median(value_max_profile))

    total_energy = float(power.sum()) + 1e-20
    e_10_30 = _band_energy(freqs, power, 10.0, 30.0)
    e_30_60 = _band_energy(freqs, power, 30.0, 60.0)
    e_60_120 = _band_energy(freqs, power, 60.0, 120.0)
    e_20_30 = _band_energy(freqs, power, 20.0, 30.0)
    peak_idx = int(np.argmax(mean_power)) if mean_power.size else 0
    normalized_profile = mean_power / max(float(mean_power.sum()), 1e-20)
    centroid = float(np.sum(freqs * normalized_profile))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * normalized_profile)))
    entropy = float(-np.sum(normalized_profile * np.log2(normalized_profile + 1e-20)) / np.log2(max(2, len(normalized_profile))))
    extras = [
        e_10_30 / max(e_30_60, 1e-20),
        e_10_30 / max(e_60_120, 1e-20),
        e_30_60 / max(e_60_120, 1e-20),
        e_20_30 / total_energy,
        e_30_60 / total_energy,
        e_60_120 / total_energy,
        float(freqs[peak_idx]) if freqs.size else 0.0,
        float(mean_power[peak_idx] / max(float(np.median(mean_power)), 1e-20)) if mean_power.size else 0.0,
        centroid,
        bandwidth,
        entropy,
    ]
    features = np.concatenate([energy_fraction, centered_mean, centered_max, np.asarray(extras, dtype=np.float64)])
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def temporal_lowfreq_feature_names(spec: dict) -> list[str]:
    names: list[str] = []
    for chunks in (3, 5, 10):
        for idx in range(chunks):
            names.append(f"lowfreq_time_{chunks}_chunk_{idx}_mean_energy")
        for idx in range(chunks):
            names.append(f"lowfreq_time_{chunks}_chunk_{idx}_max_energy")
    names.extend(
        [
            "lowfreq_temporal_energy_slope",
            "lowfreq_temporal_max_index_fraction",
            "lowfreq_temporal_max_to_total",
            "lowfreq_temporal_top10_fraction",
            "lowfreq_temporal_smoothed_peak_count",
        ]
    )
    return names


def _chunk_stats(values: np.ndarray, chunks: int) -> tuple[list[float], list[float]]:
    parts = np.array_split(values, chunks)
    means = [float(part.mean()) if part.size else 0.0 for part in parts]
    maxes = [float(part.max()) if part.size else 0.0 for part in parts]
    return means, maxes


def temporal_lowfreq_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, spec: dict) -> np.ndarray:
    _, power, _ = _lowfreq_spectrogram(row, waveform, sample_rate, audio_cfg, spec)
    temporal = power.sum(axis=0).astype(np.float64)
    if temporal.size == 0:
        temporal = np.zeros(1, dtype=np.float64)
    temporal = temporal / max(float(temporal.sum()), 1e-20)
    features: list[float] = []
    for chunks in (3, 5, 10):
        means, maxes = _chunk_stats(temporal, chunks)
        features.extend(means)
        features.extend(maxes)
    x = np.linspace(0.0, 1.0, len(temporal), dtype=np.float64)
    slope = float(np.polyfit(x, temporal, 1)[0]) if len(temporal) >= 2 else 0.0
    max_idx = int(np.argmax(temporal))
    top_count = max(1, int(np.ceil(0.10 * len(temporal))))
    smoothed = np.convolve(temporal, np.ones(3, dtype=np.float64) / 3.0, mode="same") if len(temporal) >= 3 else temporal
    peaks, _ = find_peaks(smoothed, height=float(np.mean(smoothed) + np.std(smoothed)))
    features.extend(
        [
            slope,
            max_idx / max(1, len(temporal) - 1),
            float(temporal[max_idx] / max(float(temporal.sum()), 1e-20)),
            float(np.sort(temporal)[-top_count:].sum() / max(float(temporal.sum()), 1e-20)),
            float(len(peaks)),
        ]
    )
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
