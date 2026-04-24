from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, hilbert, stft

from src.data.spectrogram import crop_event, prepare_waveform


BAND_RANGES_HZ = (
    (0.0, 15.0),
    (15.0, 30.0),
    (30.0, 45.0),
    (45.0, 60.0),
    (60.0, 90.0),
    (90.0, 125.0),
)


def _summary(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    ]


def _summary_percentiles(values: np.ndarray, percentiles: tuple[float, ...] = (10.0, 50.0, 90.0)) -> list[float]:
    if values.size == 0:
        return [0.0 for _ in percentiles]
    return [float(np.percentile(values, percentile)) for percentile in percentiles]


def _framed_view(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((1, frame_size), dtype=np.float32)
    if signal.size < frame_size:
        padded = np.pad(signal, (0, frame_size - signal.size), mode="constant")
        return padded.reshape(1, frame_size)
    starts = range(0, signal.size - frame_size + 1, max(1, hop_size))
    return np.stack([signal[start : start + frame_size] for start in starts]).astype(np.float32)


def cropped_waveform_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict) -> tuple[np.ndarray, int]:
    waveform, sample_rate = prepare_waveform(waveform, sample_rate, audio_cfg)
    start_s = float(row.get("clip_start_seconds", row.get("start_seconds", 0.0)))
    end_s = float(row.get("clip_end_seconds", row.get("end_seconds", start_s)))
    crop_padding = float(audio_cfg.get("crop_padding_seconds", 0.0))
    cropped = crop_event(waveform, sample_rate, start_s, end_s, crop_padding)
    values = cropped.squeeze(0).detach().cpu().numpy().astype(np.float64)
    return values, int(sample_rate)


def _frame_settings(signal: np.ndarray, audio_cfg: dict) -> tuple[int, int]:
    frame_size = max(16, min(int(audio_cfg.get("win_length", audio_cfg.get("n_fft", 128))), signal.size or 16))
    hop_size = max(4, min(int(audio_cfg.get("hop_length", 16)), frame_size))
    return frame_size, hop_size


def _spectral_arrays(signal: np.ndarray, sample_rate: int, audio_cfg: dict):
    frame_size, hop_size = _frame_settings(signal, audio_cfg)
    nperseg = max(16, min(int(audio_cfg.get("n_fft", 128)), max(frame_size, signal.size or 16)))
    noverlap = max(0, min(nperseg - 1, nperseg - hop_size))
    if signal.size < nperseg:
        signal = np.pad(signal, (0, nperseg - signal.size), mode="constant")
    freqs, _, spectrum = stft(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, boundary=None)
    magnitude = np.abs(spectrum).astype(np.float64) + 1e-12
    power = magnitude**2
    total_power = np.sum(power, axis=0) + 1e-12
    return freqs.astype(np.float64), magnitude, power, total_power


def _spectral_descriptors(freqs: np.ndarray, magnitude: np.ndarray, power: np.ndarray, total_power: np.ndarray) -> dict[str, np.ndarray]:
    centroid = np.sum(freqs[:, None] * power, axis=0) / total_power
    bandwidth = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :]) ** 2) * power, axis=0) / total_power)
    cumulative_power = np.cumsum(power, axis=0)
    rolloff_threshold = 0.85 * cumulative_power[-1, :]
    rolloff = np.asarray(
        [freqs[min(int(np.searchsorted(cumulative_power[:, idx], rolloff_threshold[idx], side="left")), len(freqs) - 1)] for idx in range(power.shape[1])],
        dtype=np.float64,
    )
    flatness = np.exp(np.mean(np.log(magnitude), axis=0)) / (np.mean(magnitude, axis=0) + 1e-12)
    contrast = np.percentile(magnitude, 90, axis=0) - np.percentile(magnitude, 10, axis=0)
    dominant = freqs[np.argmax(power, axis=0)]
    normalized_power = power / total_power[None, :]
    entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-12), axis=0) / np.log2(max(2, power.shape[0]))
    slopes = []
    centered_freqs = freqs - float(np.mean(freqs))
    denom = float(np.sum(centered_freqs**2)) + 1e-12
    log_mag = np.log(magnitude)
    for column in range(log_mag.shape[1]):
        centered_mag = log_mag[:, column] - float(np.mean(log_mag[:, column]))
        slopes.append(float(np.sum(centered_freqs * centered_mag) / denom))
    return {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff_85_hz": rolloff,
        "spectral_flatness": flatness,
        "spectral_contrast": contrast,
        "dominant_frequency_hz": dominant,
        "spectral_entropy": entropy,
        "spectral_slope": np.asarray(slopes, dtype=np.float64),
    }


def waveform_stats_feature_names() -> list[str]:
    names = ["duration_seconds", "real_duration_seconds", "low_frequency", "high_frequency", "frequency_span_hz"]
    names.extend(["rms_mean", "rms_std", "rms_min", "rms_max", "rms_p10", "rms_p50", "rms_p90"])
    names.extend(["zcr_mean", "zcr_std"])
    names.extend(["envelope_mean", "envelope_std", "envelope_min", "envelope_max", "envelope_p10", "envelope_p50", "envelope_p90"])
    names.extend(
        [
            "envelope_peak_count",
            "envelope_peak_rate",
            "crest_factor",
            "attack_time_fraction",
            "decay_time_fraction",
            "attack_slope",
            "decay_slope",
        ]
    )
    return names


def waveform_stats_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict) -> np.ndarray:
    signal, sample_rate = cropped_waveform_from_waveform(row, waveform, sample_rate, audio_cfg)
    frame_size, hop_size = _frame_settings(signal, audio_cfg)
    frames = _framed_view(signal, frame_size=frame_size, hop_size=hop_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    envelope = np.abs(hilbert(signal)) if signal.size else np.zeros(1, dtype=np.float64)
    peaks, _ = find_peaks(envelope, height=float(np.mean(envelope) + np.std(envelope)))
    crest_factor = float(np.max(np.abs(signal)) / (np.sqrt(np.mean(np.square(signal))) + 1e-12)) if signal.size else 0.0
    peak_idx = int(np.argmax(envelope)) if envelope.size else 0
    attack_fraction = peak_idx / max(1, envelope.size - 1)
    decay_fraction = 1.0 - attack_fraction
    attack_slope = (float(envelope[peak_idx]) - float(envelope[0])) / max(1, peak_idx) if envelope.size else 0.0
    decay_slope = (float(envelope[-1]) - float(envelope[peak_idx])) / max(1, envelope.size - peak_idx - 1) if envelope.size else 0.0
    duration = float(row.get("duration_seconds", row.get("real_duration_seconds", 0.0)))
    real_duration = float(row.get("real_duration_seconds", duration))
    low_frequency = float(row.get("low_frequency", 0.0))
    high_frequency = float(row.get("high_frequency", 0.0))
    features = [
        duration,
        real_duration,
        low_frequency,
        high_frequency,
        max(0.0, high_frequency - low_frequency),
        *_summary(rms),
        *_summary_percentiles(rms),
        float(np.mean(zcr)) if zcr.size else 0.0,
        float(np.std(zcr)) if zcr.size else 0.0,
        *_summary(envelope),
        *_summary_percentiles(envelope),
        float(len(peaks)),
        float(len(peaks) / max(real_duration, 1e-12)),
        crest_factor,
        float(attack_fraction),
        float(decay_fraction),
        float(attack_slope),
        float(decay_slope),
    ]
    return np.asarray(features, dtype=np.float32)


def spectral_stats_feature_names() -> list[str]:
    names = []
    for prefix in [
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_85_hz",
        "spectral_flatness",
        "spectral_contrast",
        "dominant_frequency_hz",
        "spectral_entropy",
        "spectral_slope",
    ]:
        names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max"])
    names.append("frequency_of_max_energy_hz")
    return names


def spectral_stats_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict) -> np.ndarray:
    signal, sample_rate = cropped_waveform_from_waveform(row, waveform, sample_rate, audio_cfg)
    freqs, magnitude, power, total_power = _spectral_arrays(signal, sample_rate, audio_cfg)
    descriptors = _spectral_descriptors(freqs, magnitude, power, total_power)
    frequency_of_max_energy = float(freqs[int(np.argmax(np.sum(power, axis=1)))]) if power.size else 0.0
    features = []
    for key in [
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_85_hz",
        "spectral_flatness",
        "spectral_contrast",
        "dominant_frequency_hz",
        "spectral_entropy",
        "spectral_slope",
    ]:
        features.extend(_summary(descriptors[key]))
    features.append(frequency_of_max_energy)
    return np.asarray(features, dtype=np.float32)


def _mel_space_hz(f_min: float, f_max: float, n_edges: int) -> np.ndarray:
    def hz_to_mel(values):
        return 2595.0 * np.log10(1.0 + values / 700.0)

    def mel_to_hz(values):
        return 700.0 * (10.0 ** (values / 2595.0) - 1.0)

    return mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_edges))


def band_energy_feature_names(n_bands: int, scale: str = "linear") -> list[str]:
    names = []
    for idx in range(n_bands):
        names.extend(
            [
                f"{scale}_band_{idx}_log_energy_mean",
                f"{scale}_band_{idx}_log_energy_std",
                f"{scale}_band_{idx}_log_energy_max",
                f"{scale}_band_{idx}_energy_ratio",
            ]
        )
    for idx in range(max(0, n_bands - 1)):
        names.append(f"{scale}_band_ratio_{idx}_{idx + 1}")
    names.append(f"{scale}_band_energy_entropy")
    return names


def band_energy_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict, n_bands: int, scale: str = "linear") -> np.ndarray:
    signal, sample_rate = cropped_waveform_from_waveform(row, waveform, sample_rate, audio_cfg)
    freqs, _, power, _ = _spectral_arrays(signal, sample_rate, audio_cfg)
    f_min = float(audio_cfg.get("f_min", 0.0))
    f_max = min(float(audio_cfg.get("f_max", sample_rate / 2.0)), float(sample_rate) / 2.0)
    edges = _mel_space_hz(f_min, f_max, n_bands + 1) if scale == "mel" else np.linspace(f_min, f_max, n_bands + 1)
    band_energies = []
    for idx in range(n_bands):
        high_cmp = freqs <= edges[idx + 1] if idx == n_bands - 1 else freqs < edges[idx + 1]
        mask = (freqs >= edges[idx]) & high_cmp
        if not np.any(mask):
            energy = np.zeros(power.shape[1], dtype=np.float64)
        else:
            energy = np.sum(power[mask], axis=0)
        band_energies.append(energy)
    band_matrix = np.vstack(band_energies) if band_energies else np.zeros((0, power.shape[1]), dtype=np.float64)
    total = float(np.sum(band_matrix)) + 1e-12
    features = []
    ratios = []
    for idx in range(n_bands):
        energy = band_matrix[idx]
        log_energy = np.log1p(energy)
        ratio = float(np.sum(energy) / total)
        ratios.append(ratio)
        features.extend([float(np.mean(log_energy)), float(np.std(log_energy)), float(np.max(log_energy)), ratio])
    for idx in range(max(0, n_bands - 1)):
        features.append(float(ratios[idx] / max(ratios[idx + 1], 1e-12)))
    ratio_array = np.asarray(ratios, dtype=np.float64)
    features.append(float(-np.sum(ratio_array * np.log2(ratio_array + 1e-12)) / np.log2(max(2, n_bands))))
    return np.asarray(features, dtype=np.float32)


def handcrafted_audio_feature_names() -> list[str]:
    names = ["duration_seconds", "real_duration_seconds", "low_frequency", "high_frequency", "frequency_span_hz"]
    for prefix in [
        "rms",
        "zcr",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_85_hz",
        "spectral_flatness",
        "spectral_contrast",
        "dominant_frequency_hz",
    ]:
        names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max"])
    names.extend(
        [
            "envelope_mean",
            "envelope_std",
            "envelope_max",
            "envelope_peak_count",
            "crest_factor",
            "snr_like_ratio",
        ]
    )
    for low_hz, high_hz in BAND_RANGES_HZ:
        names.append(f"band_ratio_{int(low_hz)}_{int(high_hz)}hz")
    return names


def handcrafted_audio_features_from_waveform(row: dict, waveform, sample_rate: int, audio_cfg: dict) -> np.ndarray:
    signal, sample_rate = cropped_waveform_from_waveform(row, waveform, sample_rate, audio_cfg)
    frame_size, hop_size = _frame_settings(signal, audio_cfg)
    frames = _framed_view(signal, frame_size=frame_size, hop_size=hop_size)

    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    freqs, magnitude, power, total_power = _spectral_arrays(signal, sample_rate, audio_cfg)
    descriptors = _spectral_descriptors(freqs, magnitude, power, total_power)

    analytic = np.abs(hilbert(signal)) if signal.size else np.zeros(1, dtype=np.float64)
    peaks, _ = find_peaks(analytic, height=float(np.mean(analytic) + np.std(analytic)))
    crest_factor = float(np.max(np.abs(signal)) / (np.sqrt(np.mean(np.square(signal))) + 1e-12)) if signal.size else 0.0
    noise_floor = float(np.percentile(np.abs(signal), 50)) if signal.size else 0.0
    signal_peak = float(np.percentile(np.abs(signal), 95)) if signal.size else 0.0
    snr_like_ratio = signal_peak / max(1e-12, noise_floor + 1e-12)

    freq_limit = min(float(sample_rate) / 2.0, 125.0)
    band_ratios = []
    total_band_energy = float(power[freqs <= freq_limit].sum()) if power.size else 0.0
    for low_hz, high_hz in BAND_RANGES_HZ:
        mask = (freqs >= low_hz) & (freqs < min(high_hz, freq_limit))
        band_energy = float(power[mask].sum()) if np.any(mask) else 0.0
        band_ratios.append(band_energy / max(total_band_energy, 1e-12))

    duration = float(row.get("duration_seconds", row.get("real_duration_seconds", 0.0)))
    real_duration = float(row.get("real_duration_seconds", duration))
    low_frequency = float(row.get("low_frequency", 0.0))
    high_frequency = float(row.get("high_frequency", 0.0))

    features = [
        duration,
        real_duration,
        low_frequency,
        high_frequency,
        max(0.0, high_frequency - low_frequency),
        *_summary(rms),
        *_summary(zcr),
        *_summary(descriptors["spectral_centroid_hz"]),
        *_summary(descriptors["spectral_bandwidth_hz"]),
        *_summary(descriptors["spectral_rolloff_85_hz"]),
        *_summary(descriptors["spectral_flatness"]),
        *_summary(descriptors["spectral_contrast"]),
        *_summary(descriptors["dominant_frequency_hz"]),
        float(np.mean(analytic)) if analytic.size else 0.0,
        float(np.std(analytic)) if analytic.size else 0.0,
        float(np.max(analytic)) if analytic.size else 0.0,
        float(len(peaks)),
        crest_factor,
        snr_like_ratio,
        *band_ratios,
    ]
    return np.asarray(features, dtype=np.float32)
