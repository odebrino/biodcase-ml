from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks, windows

from src.data.spectrogram import crop_event, prepare_waveform


NOTEBOOK_FREQ_GRID = np.linspace(10.0, 120.0, 20, dtype=np.float64)
NOTEBOOK_P_REF = 2e-5


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def notebook_exact_event_signal(row: dict, waveform, sample_rate: int, audio_cfg: dict) -> tuple[np.ndarray, int]:
    """Return the full annotated event segment, not a fixed-size centered crop."""

    cfg = deepcopy(audio_cfg)
    waveform, sample_rate = prepare_waveform(waveform, sample_rate, cfg)
    start_s = _as_float(row.get("clip_start_seconds", row.get("start_seconds", 0.0)))
    end_s = _as_float(row.get("clip_end_seconds", row.get("end_seconds", start_s)))
    if end_s <= start_s:
        end_s = start_s + max(_as_float(row.get("duration_seconds", 0.0)), 1.0 / max(1, sample_rate))
    segment = crop_event(waveform, sample_rate, start_s, end_s, 0.0)
    signal = segment.squeeze(0).detach().cpu().numpy().astype(np.float64)
    if signal.size < 3:
        signal = np.pad(signal, (0, 3 - signal.size), mode="constant")
    return signal, int(sample_rate)


def notebook_exact_spectrogram(
    row: dict,
    waveform,
    sample_rate: int,
    audio_cfg: dict,
    spec: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = spec or {}
    signal, sample_rate = notebook_exact_event_signal(row, waveform, sample_rate, audio_cfg)
    window_dur = float(spec.get("window_dur", spec.get("window_seconds", 2.0)))
    target_window = max(32, int(window_dur * sample_rate * 2))
    window_nsamp = min(signal.size - 1, target_window)
    window_nsamp = max(2, window_nsamp)
    step_nsamp = max(1, int(window_nsamp * 0.25))
    noverlap = max(0, min(window_nsamp - 1, window_nsamp - step_nsamp))
    gaussian_std = float(spec.get("gaussian_std", max(1.0, window_nsamp / 6.0)))
    window = windows.gaussian(window_nsamp, std=gaussian_std, sym=True)

    freqs, times, power = scipy_signal.spectrogram(
        signal,
        fs=sample_rate,
        window=window,
        nperseg=window_nsamp,
        noverlap=noverlap,
        nfft=window_nsamp,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    f_min = float(spec.get("f_min", 10.0))
    f_max = min(float(spec.get("f_max", 120.0)), sample_rate / 2.0)
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        freqs = NOTEBOOK_FREQ_GRID.copy()
        power = np.zeros((len(freqs), 1), dtype=np.float64)
        times = np.zeros(1, dtype=np.float64)
    else:
        freqs = freqs[mask].astype(np.float64)
        power = power[mask].astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(power / (NOTEBOOK_P_REF**2))
    db = np.nan_to_num(db, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    if bool(spec.get("dynamic_range_clip", False)) and db.size:
        peak = float(np.max(db))
        floor = peak - float(spec.get("dynamic_range_db", 80.0))
        db = np.clip(db, floor, peak)

    return freqs.astype(np.float64), times.astype(np.float64), db.astype(np.float64)


def _interp_profile(freqs: np.ndarray, profile: np.ndarray, grid: np.ndarray = NOTEBOOK_FREQ_GRID) -> np.ndarray:
    if freqs.size == 0 or profile.size == 0:
        return np.zeros(len(grid), dtype=np.float64)
    return np.interp(grid, freqs, profile, left=float(profile[0]), right=float(profile[-1])).astype(np.float64)


def _band_stat(freqs: np.ndarray, values: np.ndarray, low: float, high: float, reducer: str) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    selected = values[mask]
    if reducer == "max":
        return float(np.max(selected))
    return float(np.mean(selected))


def notebook_exact_feature_names(spec: dict) -> list[str]:
    variant = str(spec.get("variant", "44"))
    names = ["notebook_db_mean", "notebook_db_std", "notebook_db_min", "notebook_db_max"]
    names.extend([f"notebook_grid_{idx:02d}_mean_db" for idx in range(20)])
    if variant == "26":
        names.extend(["notebook_bma_20_30_mean_db", "notebook_bma_20_30_max_db"])
    else:
        names.extend([f"notebook_grid_{idx:02d}_max_db" for idx in range(20)])
    if bool(spec.get("include_duration", False)):
        names.append("event_duration_seconds")
    if bool(spec.get("include_bbox", False)):
        names.extend(["bbox_low_frequency", "bbox_high_frequency", "bbox_bandwidth", "bbox_center_frequency"])
    return names


def notebook_exact_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, spec: dict) -> np.ndarray:
    freqs, _, db = notebook_exact_spectrogram(row, waveform, sample_rate, audio_cfg, spec)
    mean_profile = db.mean(axis=1) if db.size else np.zeros_like(freqs)
    max_profile = db.max(axis=1) if db.size else np.zeros_like(freqs)
    features: list[float] = [
        float(db.mean()) if db.size else 0.0,
        float(db.std()) if db.size else 0.0,
        float(db.min()) if db.size else 0.0,
        float(db.max()) if db.size else 0.0,
    ]
    features.extend(_interp_profile(freqs, mean_profile).tolist())
    if str(spec.get("variant", "44")) == "26":
        features.append(_band_stat(freqs, mean_profile, 20.0, 30.0, "mean"))
        features.append(_band_stat(freqs, max_profile, 20.0, 30.0, "max"))
    else:
        features.extend(_interp_profile(freqs, max_profile).tolist())

    if bool(spec.get("include_duration", False)):
        features.append(_as_float(row.get("duration_seconds", row.get("real_duration_seconds", 0.0))))
    if bool(spec.get("include_bbox", False)):
        low = _as_float(row.get("low_frequency", 0.0))
        high = _as_float(row.get("high_frequency", low))
        features.extend([low, high, max(0.0, high - low), 0.5 * (low + high)])
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


REGION_BANDS = [
    ("bma_16_27", 16.0, 27.0),
    ("bma_20_30", 20.0, 30.0),
    ("dd_20_120", 20.0, 120.0),
    ("dd_30_90", 30.0, 90.0),
    ("bp_15_30", 15.0, 30.0),
    ("bp_overtone_80_120", 80.0, 120.0),
]


def class_region_lowfreq_feature_names(spec: dict) -> list[str]:
    names: list[str] = []
    for name, _, _ in REGION_BANDS:
        names.extend([f"{name}_mean_db", f"{name}_max_db", f"{name}_energy_fraction"])
    names.extend(
        [
            "ratio_16_30_to_30_60",
            "ratio_20_30_to_80_120",
            "ratio_30_90_to_10_120",
            "ratio_80_120_to_15_30",
            "event_duration_seconds",
            "temporal_energy_slope",
            "temporal_max_index_fraction",
            "temporal_concentration_max_to_total",
            "temporal_top10_fraction",
            "temporal_peak_count",
            "dominant_frequency_mean",
            "dominant_frequency_std",
            "dominant_frequency_min",
            "dominant_frequency_max",
            "dominant_frequency_slope",
            "frequency_centroid_hz",
            "frequency_bandwidth_hz",
            "frequency_entropy",
            "frequency_peak_count",
            "overtone_80_120_to_20_30",
        ]
    )
    return names


def _band_energy_from_db(freqs: np.ndarray, db: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    power_like = np.power(10.0, db[mask] / 10.0)
    return float(np.sum(power_like))


def class_region_lowfreq_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, spec: dict) -> np.ndarray:
    freqs, _, db = notebook_exact_spectrogram(row, waveform, sample_rate, audio_cfg, spec)
    if db.size == 0:
        return np.zeros(len(class_region_lowfreq_feature_names(spec)), dtype=np.float32)

    mean_profile = db.mean(axis=1)
    max_profile = db.max(axis=1)
    total_energy = max(_band_energy_from_db(freqs, db, 10.0, 120.0), 1e-20)
    features: list[float] = []
    for _, low, high in REGION_BANDS:
        features.append(_band_stat(freqs, mean_profile, low, high, "mean"))
        features.append(_band_stat(freqs, max_profile, low, high, "max"))
        features.append(_band_energy_from_db(freqs, db, low, high) / total_energy)

    e_16_30 = _band_energy_from_db(freqs, db, 16.0, 30.0)
    e_30_60 = _band_energy_from_db(freqs, db, 30.0, 60.0)
    e_20_30 = _band_energy_from_db(freqs, db, 20.0, 30.0)
    e_80_120 = _band_energy_from_db(freqs, db, 80.0, 120.0)
    e_30_90 = _band_energy_from_db(freqs, db, 30.0, 90.0)
    e_15_30 = _band_energy_from_db(freqs, db, 15.0, 30.0)
    features.extend(
        [
            e_16_30 / max(e_30_60, 1e-20),
            e_20_30 / max(e_80_120, 1e-20),
            e_30_90 / total_energy,
            e_80_120 / max(e_15_30, 1e-20),
            _as_float(row.get("duration_seconds", row.get("real_duration_seconds", 0.0))),
        ]
    )

    temporal = np.power(10.0, db / 10.0).sum(axis=0)
    temporal = temporal / max(float(temporal.sum()), 1e-20)
    x_time = np.linspace(0.0, 1.0, len(temporal), dtype=np.float64)
    slope = float(np.polyfit(x_time, temporal, 1)[0]) if len(temporal) >= 2 else 0.0
    max_idx = int(np.argmax(temporal)) if temporal.size else 0
    top_count = max(1, int(np.ceil(0.10 * max(1, len(temporal)))))
    smoothed = np.convolve(temporal, np.ones(3, dtype=np.float64) / 3.0, mode="same") if len(temporal) >= 3 else temporal
    peaks, _ = find_peaks(smoothed, height=float(np.mean(smoothed) + np.std(smoothed))) if smoothed.size else ([], {})
    features.extend(
        [
            slope,
            max_idx / max(1, len(temporal) - 1),
            float(temporal[max_idx] / max(float(temporal.sum()), 1e-20)) if temporal.size else 0.0,
            float(np.sort(temporal)[-top_count:].sum() / max(float(temporal.sum()), 1e-20)) if temporal.size else 0.0,
            float(len(peaks)),
        ]
    )

    power_like = np.power(10.0, db / 10.0)
    dominant_idx = np.argmax(power_like, axis=0)
    dominant_freqs = freqs[dominant_idx] if freqs.size else np.zeros(1, dtype=np.float64)
    x_dom = np.linspace(0.0, 1.0, len(dominant_freqs), dtype=np.float64)
    dom_slope = float(np.polyfit(x_dom, dominant_freqs, 1)[0]) if len(dominant_freqs) >= 2 else 0.0
    freq_profile = power_like.mean(axis=1)
    normalized = freq_profile / max(float(freq_profile.sum()), 1e-20)
    centroid = float(np.sum(freqs * normalized)) if freqs.size else 0.0
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * normalized))) if freqs.size else 0.0
    entropy = float(-np.sum(normalized * np.log2(normalized + 1e-20)) / np.log2(max(2, len(normalized))))
    freq_peaks, _ = find_peaks(freq_profile, height=float(np.mean(freq_profile) + np.std(freq_profile))) if freq_profile.size else ([], {})
    features.extend(
        [
            float(np.mean(dominant_freqs)),
            float(np.std(dominant_freqs)),
            float(np.min(dominant_freqs)),
            float(np.max(dominant_freqs)),
            dom_slope,
            centroid,
            bandwidth,
            entropy,
            float(len(freq_peaks)),
            e_80_120 / max(e_20_30, 1e-20),
        ]
    )
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
