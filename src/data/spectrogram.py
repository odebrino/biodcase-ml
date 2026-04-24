import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

from src.data.spectrogram_presets import resolve_spectrogram_preset


SPECTROGRAM_TRANSFORM_VERSION = 4


@dataclass(frozen=True)
class SpectrogramFrame:
    values: object
    frequencies_hz: object
    times_seconds: object
    sample_rate: int
    config: dict


def require_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "Install PyTorch before running this command. "
            "See the README for the supported setup."
        ) from exc
    return torch


def optional_torchaudio():
    try:
        import torchaudio
    except Exception as exc:
        return None, exc
    return torchaudio, None


def require_audio_stack():
    torch = require_torch()
    torchaudio, exc = optional_torchaudio()
    if torchaudio is None:
        raise SystemExit(
            "Torchaudio is unavailable or broken in this environment. "
            "Install a torchaudio build that matches PyTorch/CUDA, or use the scipy fallback paths where supported. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return torch, torchaudio


def read_waveform(path: str | Path):
    torch = require_torch()
    torchaudio, _ = optional_torchaudio()
    if torchaudio is not None:
        try:
            return torchaudio.load(path)
        except Exception:
            pass
    return _read_waveform_scipy(torch, path)


def _read_waveform_scipy(torch, path: str | Path):
    try:
        from scipy.io import wavfile
    except ImportError as exc:
        raise SystemExit(
            "Audio loading requires either torchaudio or scipy.io.wavfile. "
            "Torchaudio is unavailable and scipy is not installed."
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, data = wavfile.read(path)
    waveform = torch.as_tensor(data, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)
    return waveform, int(sample_rate)


def audio_duration(path: str | Path) -> float:
    from scipy.io import wavfile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate, data = wavfile.read(path)
    return float(len(data) / sample_rate)


def crop_event(waveform, sample_rate: int, start_s: float, end_s: float, margin_s: float):
    torch = require_torch()
    start = max(0, int(round((start_s - margin_s) * sample_rate)))
    end = max(start + 1, int(round((end_s + margin_s) * sample_rate)))
    if end <= waveform.shape[-1]:
        return waveform[:, start:end]
    return torch.nn.functional.pad(waveform[:, start:], (0, max(0, end - waveform.shape[-1])))


def prepare_waveform(waveform, sample_rate: int, audio_cfg: dict):
    torch = require_torch()
    torchaudio, _ = optional_torchaudio()
    audio_cfg = resolve_spectrogram_preset(audio_cfg)
    target_rate = int(audio_cfg["sample_rate"])
    if sample_rate != target_rate:
        if torchaudio is not None:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        else:
            waveform = _resample_waveform_scipy(torch, waveform, sample_rate, target_rate)
        sample_rate = target_rate

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.float(), sample_rate


def normalize_db(db, audio_cfg: dict):
    audio_cfg = resolve_spectrogram_preset(audio_cfg)
    if audio_cfg.get("normalization", "sample") == "global":
        db_min = float(audio_cfg.get("db_min", -80.0))
        db_max = float(audio_cfg.get("db_max", 0.0))
        return (db.clamp(db_min, db_max) - db_min) / max(1e-6, db_max - db_min)

    db = db - db.min()
    if float(db.max()) > 0:
        db = db / db.max()
    return db


def spectrogram_frame(waveform, sample_rate: int, audio_cfg: dict, start_offset_s: float = 0.0) -> SpectrogramFrame:
    torch = require_torch()
    torchaudio, _ = optional_torchaudio()
    audio_cfg = resolve_spectrogram_preset(audio_cfg)
    waveform, sample_rate = prepare_waveform(waveform, sample_rate, audio_cfg)
    n_fft = int(audio_cfg["n_fft"])
    win_length = int(audio_cfg.get("win_length", n_fft))
    hop_length = int(audio_cfg["hop_length"])
    f_min = float(audio_cfg.get("f_min", 0.0))
    f_max = float(audio_cfg.get("f_max", sample_rate / 2.0))
    frequency_scale = str(audio_cfg.get("frequency_scale", "mel")).lower()
    min_samples = max(n_fft, win_length)
    if waveform.shape[-1] < min_samples:
        waveform = torch.nn.functional.pad(waveform, (0, min_samples - waveform.shape[-1]))

    if frequency_scale == "linear":
        window = torch.hann_window(win_length, device=waveform.device)
        spectrum = torch.stft(
            waveform.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        power = spectrum.abs().pow(2.0).clamp_min(1e-10)
        db = 10.0 * torch.log10(power)
        freqs = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, device=db.device)
        freq_mask = (freqs >= f_min) & (freqs <= f_max)
        db = db[freq_mask]
        freqs = freqs[freq_mask]
    elif frequency_scale == "mel":
        if torchaudio is not None:
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=int(audio_cfg["n_mels"]),
                f_min=f_min,
                f_max=f_max,
                power=2.0,
            )(waveform)
            db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel).squeeze(0)
        else:
            db = _fallback_mel_db(torch, waveform, sample_rate, audio_cfg)
        freqs = mel_bin_centers(torch, int(audio_cfg["n_mels"]), f_min, f_max, db.device)
    else:
        raise ValueError(f"Unknown frequency_scale: {frequency_scale}")

    times = start_offset_s + torch.arange(db.shape[1], device=db.device, dtype=torch.float32) * (hop_length / sample_rate)
    return SpectrogramFrame(
        values=normalize_db(db, audio_cfg),
        frequencies_hz=freqs.float(),
        times_seconds=times,
        sample_rate=sample_rate,
        config=audio_cfg,
    )


def mel_bin_centers(torch, n_mels: int, f_min: float, f_max: float, device):
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    return mel_to_hz((mel_points[:-2] + mel_points[1:-1] + mel_points[2:]) / 3.0)


def hz_to_mel(value: float) -> float:
    return 2595.0 * math.log10(1.0 + value / 700.0)


def mel_to_hz(value):
    return 700.0 * (10.0 ** (value / 2595.0) - 1.0)


def literal_time_frequency_crop_from_waveform(waveform, sample_rate: int, row: dict, audio_cfg: dict):
    torch = require_torch()
    audio_cfg = resolve_spectrogram_preset(audio_cfg)
    start_s = float(row["clip_start_seconds"] if "clip_start_seconds" in row else row["start_seconds"])
    end_s = float(row["clip_end_seconds"] if "clip_end_seconds" in row else row["end_seconds"])
    low_hz = float(row["low_frequency"])
    high_hz = float(row["high_frequency"])

    waveform, sample_rate = prepare_waveform(waveform, sample_rate, audio_cfg)
    segment = crop_event(waveform, sample_rate, start_s, end_s, 0.0)
    frame = spectrogram_frame(segment, sample_rate, audio_cfg, start_offset_s=start_s)
    low, high = sorted((low_hz, high_hz))
    freq_mask = (frame.frequencies_hz >= low) & (frame.frequencies_hz <= high)
    if not bool(freq_mask.any()):
        center = torch.argmin((frame.frequencies_hz - ((low + high) / 2.0)).abs())
        freq_mask[center] = True
    cropped = frame.values[freq_mask, :]
    return SpectrogramFrame(
        values=cropped,
        frequencies_hz=frame.frequencies_hz[freq_mask],
        times_seconds=frame.times_seconds,
        sample_rate=frame.sample_rate,
        config=frame.config,
    )


def literal_time_frequency_crop(row: dict, audio_cfg: dict):
    waveform, sample_rate = read_waveform(row["audio_path"])
    return literal_time_frequency_crop_from_waveform(waveform, sample_rate, row, audio_cfg)


def resize_spectrogram(values, img_size: int):
    torch = require_torch()
    return torch.nn.functional.interpolate(
        values.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def event_cache_key(row: dict, audio_cfg: dict, img_size: int, cache_dtype: str = "float32") -> str:
    audio_cfg = resolve_spectrogram_preset(audio_cfg)
    audio_path = Path(row["audio_path"]) if row.get("audio_path") else None
    audio_stat = audio_path.stat() if audio_path and audio_path.exists() else None
    payload = {
        "version": SPECTROGRAM_TRANSFORM_VERSION,
        "cache_dtype": cache_dtype,
        "split": row.get("split"),
        "dataset": row.get("dataset"),
        "filename": row.get("filename"),
        "audio_path": str(audio_path) if audio_path else None,
        "audio_size": audio_stat.st_size if audio_stat else None,
        "audio_mtime_ns": audio_stat.st_mtime_ns if audio_stat else None,
        "source_row": row.get("source_row"),
        "clip_start_seconds": row.get("clip_start_seconds", row.get("start_seconds")),
        "clip_end_seconds": row.get("clip_end_seconds", row.get("end_seconds")),
        "low_frequency": row.get("low_frequency"),
        "high_frequency": row.get("high_frequency"),
        "audio": {
            "sample_rate": audio_cfg["sample_rate"],
            "margin_seconds": audio_cfg["margin_seconds"],
            "preset": audio_cfg.get("preset"),
            "n_fft": audio_cfg["n_fft"],
            "win_length": audio_cfg.get("win_length"),
            "hop_length": audio_cfg["hop_length"],
            "n_mels": audio_cfg["n_mels"],
            "f_min": audio_cfg["f_min"],
            "f_max": audio_cfg["f_max"],
            "frequency_scale": audio_cfg.get("frequency_scale", "mel"),
            "overlap_percent": audio_cfg.get("overlap_percent"),
            "crop_frequency_semantics": audio_cfg.get("crop_frequency_semantics", "time_crop_with_frequency_band_mask"),
            "normalization": audio_cfg.get("normalization", "sample"),
            "db_min": audio_cfg.get("db_min", -80.0),
            "db_max": audio_cfg.get("db_max", 0.0),
            "img_size": img_size,
        },
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def event_tensor_from_waveform(waveform, sample_rate: int, row: dict, audio_cfg: dict, img_size: int):
    torch = require_torch()
    torchaudio, _ = optional_torchaudio()
    audio_cfg = resolve_spectrogram_preset(audio_cfg)
    waveform, sample_rate = prepare_waveform(waveform, sample_rate, audio_cfg)

    start_s = float(row["clip_start_seconds"] if "clip_start_seconds" in row else row["start_seconds"])
    end_s = float(row["clip_end_seconds"] if "clip_end_seconds" in row else row["end_seconds"])
    cropped = crop_event(waveform, sample_rate, start_s, end_s, float(audio_cfg["margin_seconds"]))

    if torchaudio is not None:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(audio_cfg["n_fft"]),
            win_length=int(audio_cfg.get("win_length", audio_cfg["n_fft"])),
            hop_length=int(audio_cfg["hop_length"]),
            n_mels=int(audio_cfg["n_mels"]),
            f_min=float(audio_cfg["f_min"]),
            f_max=float(audio_cfg["f_max"]),
            power=2.0,
        )(cropped)
        db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel).squeeze(0)
    else:
        db = _fallback_mel_db(torch, cropped, sample_rate, audio_cfg)
    db = normalize_db(db, audio_cfg)
    db = db.flip(0)

    full = torch.nn.functional.interpolate(
        db.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    mask = frequency_mask(
        img_size=img_size,
        low_frequency=float(row["low_frequency"]),
        high_frequency=float(row["high_frequency"]),
        f_min=float(audio_cfg["f_min"]),
        f_max=float(audio_cfg["f_max"]),
        device=full.device,
    )
    highlighted = full * (0.35 + 0.65 * mask)
    return torch.stack([full, mask, highlighted]).float()


def _resample_waveform_scipy(torch, waveform, sample_rate: int, target_rate: int):
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise SystemExit(
            "Resampling requires torchaudio or scipy.signal.resample_poly. "
            "Torchaudio is unavailable and scipy is not installed."
        ) from exc

    divisor = math.gcd(sample_rate, target_rate)
    data = waveform.detach().cpu().numpy()
    resampled = resample_poly(data, target_rate // divisor, sample_rate // divisor, axis=-1)
    return torch.as_tensor(resampled, dtype=torch.float32, device=waveform.device)


def _fallback_mel_db(torch, waveform, sample_rate: int, audio_cfg: dict):
    audio_cfg = resolve_spectrogram_preset(audio_cfg)
    n_fft = int(audio_cfg["n_fft"])
    win_length = int(audio_cfg.get("win_length", n_fft))
    hop_length = int(audio_cfg["hop_length"])
    n_mels = int(audio_cfg["n_mels"])
    f_min = float(audio_cfg["f_min"])
    f_max = float(audio_cfg["f_max"])
    device = waveform.device

    window = torch.hann_window(win_length, device=device)
    spectrum = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    power = spectrum.abs().pow(2.0)
    mel_filter = _mel_filter_bank(torch, sample_rate, n_fft, n_mels, f_min, f_max, device)
    mel = torch.matmul(mel_filter, power).clamp_min(1e-10)
    return 10.0 * torch.log10(mel)


def _mel_filter_bank(torch, sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float, device):
    def hz_to_mel(value: float) -> float:
        return 2595.0 * math.log10(1.0 + value / 700.0)

    def mel_to_hz(value):
        return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, device=device)
    filters = []
    for idx in range(n_mels):
        lower = hz_points[idx]
        center = hz_points[idx + 1]
        upper = hz_points[idx + 2]
        left = (fft_freqs - lower) / (center - lower).clamp_min(1e-12)
        right = (upper - fft_freqs) / (upper - center).clamp_min(1e-12)
        filters.append(torch.minimum(left, right).clamp_min(0.0))
    return torch.stack(filters).float()


def event_tensor(row: dict, audio_cfg: dict, img_size: int):
    waveform, sample_rate = read_waveform(row["audio_path"])
    return event_tensor_from_waveform(waveform, sample_rate, row, audio_cfg, img_size)


def cached_event_tensor(row: dict, audio_cfg: dict, img_size: int, cache_cfg):
    torch = require_torch()
    if not cache_cfg:
        return event_tensor(row, audio_cfg, img_size)

    if isinstance(cache_cfg, (str, Path)):
        cache_cfg = {"root": cache_cfg}

    cache_root = Path(cache_cfg.get("root", "processed_cache"))
    cache_dtype = str(cache_cfg.get("dtype", "float16"))
    rebuild = bool(cache_cfg.get("rebuild", False))
    cache_path = cache_root / str(row["split"]) / str(row["label"]) / f"{event_cache_key(row, audio_cfg, img_size, cache_dtype)}.pt"
    if cache_path.exists() and not rebuild:
        return torch.load(cache_path, map_location="cpu").float()

    tensor = event_tensor(row, audio_cfg, img_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = tensor.half() if cache_dtype == "float16" else tensor.float()
    torch.save(to_save.cpu(), cache_path)
    return tensor


def frequency_mask(img_size: int, low_frequency: float, high_frequency: float, f_min: float, f_max: float, device):
    torch = require_torch()
    def hz_to_mel(value: float) -> float:
        return 2595.0 * math.log10(1.0 + value / 700.0)

    low = max(f_min, min(f_max, low_frequency))
    high = max(f_min, min(f_max, high_frequency))
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    span = max(1e-6, mel_max - mel_min)
    low_pos = (hz_to_mel(low) - mel_min) / span
    high_pos = (hz_to_mel(high) - mel_min) / span

    y = torch.linspace(1.0, 0.0, img_size, device=device).unsqueeze(1).expand(img_size, img_size)
    return ((y >= low_pos) & (y <= high_pos)).float()


def augment_spectrogram(tensor, cfg: dict):
    torch = require_torch()
    out = tensor.clone()

    gain = float(cfg.get("gain", 0.0))
    if gain > 0:
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * gain
        out = out * factor

    noise = float(cfg.get("noise_std", 0.0))
    if noise > 0:
        out = out + torch.randn_like(out) * noise

    time_mask = int(cfg.get("time_mask", 0))
    if time_mask > 0:
        width = int(torch.randint(0, time_mask + 1, (1,)).item())
        if width > 0 and width < out.shape[-1]:
            start = int(torch.randint(0, out.shape[-1] - width + 1, (1,)).item())
            out[:, :, start : start + width] = 0

    freq_mask = int(cfg.get("frequency_mask", 0))
    if freq_mask > 0:
        height = int(torch.randint(0, freq_mask + 1, (1,)).item())
        if height > 0 and height < out.shape[-2]:
            start = int(torch.randint(0, out.shape[-2] - height + 1, (1,)).item())
            out[:, start : start + height, :] = 0

    return out.clamp(0, 1)
