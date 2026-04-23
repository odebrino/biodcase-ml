from pathlib import Path

import torch
from scipy.io import wavfile

from src.data.spectrogram import cached_event_tensor, event_cache_key


def test_spectrogram_tensor_uses_three_channels_and_cache(tmp_path):
    audio_path = tmp_path / "event.wav"
    wavfile.write(audio_path, 250, torch.zeros(500).numpy().astype("float32"))
    audio_cfg = {
        "sample_rate": 250,
        "margin_seconds": 0.0,
        "n_fft": 64,
        "hop_length": 16,
        "n_mels": 32,
        "f_min": 0.0,
        "f_max": 125.0,
    }
    row = {
        "split": "train",
        "dataset": "tiny",
        "filename": "event.wav",
        "audio_path": str(audio_path),
        "label": "bma",
        "source_row": 0,
        "start_seconds": 0.1,
        "end_seconds": 1.0,
        "clip_start_seconds": 0.1,
        "clip_end_seconds": 1.0,
        "low_frequency": 10.0,
        "high_frequency": 30.0,
    }
    cache_root = tmp_path / "cache"
    cache_cfg = {"root": cache_root, "dtype": "float16", "rebuild": False}

    first = cached_event_tensor(row, audio_cfg, img_size=64, cache_cfg=cache_cfg)
    second = cached_event_tensor(row, audio_cfg, img_size=64, cache_cfg=cache_cfg)

    assert first.shape == (3, 64, 64)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    cache_path = cache_root / "train" / "bma" / f"{event_cache_key(row, audio_cfg, 64, 'float16')}.pt"
    assert cache_path.exists()
    assert torch.load(cache_path).dtype == torch.float16

    first_key = event_cache_key(row, audio_cfg, 64, "float16")
    wavfile.write(audio_path, 250, torch.ones(750).numpy().astype("float32"))
    second_key = event_cache_key(row, audio_cfg, 64, "float16")
    assert second_key != first_key
