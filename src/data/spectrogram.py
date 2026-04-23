import hashlib
import json
import math
import warnings
from pathlib import Path


SPECTROGRAM_TRANSFORM_VERSION = 4


def require_audio_stack():
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise SystemExit(
            "Install the audio stack before running this command. "
            "See the README for the CUDA setup used on the Nitro V15."
        ) from exc
    return torch, torchaudio


def read_waveform(path: str | Path):
    torch, torchaudio = require_audio_stack()
    try:
        return torchaudio.load(path)
    except ImportError:
        from scipy.io import wavfile

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
    torch, _ = require_audio_stack()
    start = max(0, int(round((start_s - margin_s) * sample_rate)))
    end = max(start + 1, int(round((end_s + margin_s) * sample_rate)))
    if end <= waveform.shape[-1]:
        return waveform[:, start:end]
    return torch.nn.functional.pad(waveform[:, start:], (0, max(0, end - waveform.shape[-1])))


def event_cache_key(row: dict, audio_cfg: dict, img_size: int, cache_dtype: str = "float32") -> str:
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
            "n_fft": audio_cfg["n_fft"],
            "hop_length": audio_cfg["hop_length"],
            "n_mels": audio_cfg["n_mels"],
            "f_min": audio_cfg["f_min"],
            "f_max": audio_cfg["f_max"],
            "normalization": audio_cfg.get("normalization", "sample"),
            "db_min": audio_cfg.get("db_min", -80.0),
            "db_max": audio_cfg.get("db_max", 0.0),
            "img_size": img_size,
        },
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def event_tensor_from_waveform(waveform, sample_rate: int, row: dict, audio_cfg: dict, img_size: int):
    torch, torchaudio = require_audio_stack()
    target_rate = int(audio_cfg["sample_rate"])
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        sample_rate = target_rate

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    start_s = float(row["clip_start_seconds"] if "clip_start_seconds" in row else row["start_seconds"])
    end_s = float(row["clip_end_seconds"] if "clip_end_seconds" in row else row["end_seconds"])
    cropped = crop_event(waveform, sample_rate, start_s, end_s, float(audio_cfg["margin_seconds"]))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=int(audio_cfg["n_fft"]),
        hop_length=int(audio_cfg["hop_length"]),
        n_mels=int(audio_cfg["n_mels"]),
        f_min=float(audio_cfg["f_min"]),
        f_max=float(audio_cfg["f_max"]),
        power=2.0,
    )(cropped)
    db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel).squeeze(0)
    if audio_cfg.get("normalization", "sample") == "global":
        db_min = float(audio_cfg.get("db_min", -80.0))
        db_max = float(audio_cfg.get("db_max", 0.0))
        db = ((db.clamp(db_min, db_max) - db_min) / max(1e-6, db_max - db_min))
    else:
        db = db - db.min()
        if float(db.max()) > 0:
            db = db / db.max()
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


def event_tensor(row: dict, audio_cfg: dict, img_size: int):
    waveform, sample_rate = read_waveform(row["audio_path"])
    return event_tensor_from_waveform(waveform, sample_rate, row, audio_cfg, img_size)


def cached_event_tensor(row: dict, audio_cfg: dict, img_size: int, cache_cfg):
    torch, _ = require_audio_stack()
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
    torch, _ = require_audio_stack()
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
    torch, _ = require_audio_stack()
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
