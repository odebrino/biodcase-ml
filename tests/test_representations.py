from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from src.data.export_crop_verification import export_crop_verification
from src.data.representations import (
    HANDCRAFTED_FEATURE_NAMES,
    export_representations,
    handcrafted_descriptor_vector,
    literal_patch_vector,
)
from src.data.spectrogram import literal_time_frequency_crop


def write_sine(path: Path, seconds: float = 2.0, sample_rate: int = 250, frequency: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    wavfile.write(path, sample_rate, (0.5 * np.sin(2 * np.pi * frequency * t)).astype("float32"))


def row_for(audio_path: Path) -> dict:
    return {
        "split": "train",
        "dataset": "tiny",
        "filename": audio_path.name,
        "audio_path": str(audio_path),
        "label": "bma",
        "label_raw": "Bm-A",
        "source_row": 0,
        "start_seconds": 0.4,
        "end_seconds": 1.4,
        "clip_start_seconds": 0.4,
        "clip_end_seconds": 1.4,
        "duration_seconds": 1.0,
        "real_duration_seconds": 1.0,
        "low_frequency": 20.0,
        "high_frequency": 40.0,
    }


def linear_audio_cfg() -> dict:
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


def test_literal_time_frequency_crop_respects_annotation_bounds(tmp_path):
    audio_path = tmp_path / "event.wav"
    write_sine(audio_path)

    crop = literal_time_frequency_crop(row_for(audio_path), linear_audio_cfg())

    assert crop.values.ndim == 2
    assert crop.values.shape[0] > 0
    assert crop.values.shape[1] > 0
    assert float(crop.frequencies_hz.min()) >= 20.0
    assert float(crop.frequencies_hz.max()) <= 40.0
    assert float(crop.times_seconds.min()) >= 0.4
    assert float(crop.times_seconds.max()) <= 1.4


def test_patch_and_handcrafted_representations_are_classical_ml_ready(tmp_path):
    audio_path = tmp_path / "event.wav"
    write_sine(audio_path)
    row = row_for(audio_path)

    patch = literal_patch_vector(row, linear_audio_cfg(), img_size=16)
    descriptors = handcrafted_descriptor_vector(row, linear_audio_cfg())

    assert patch.shape == (16 * 16,)
    assert patch.dtype == np.float32
    assert descriptors.shape == (len(HANDCRAFTED_FEATURE_NAMES),)
    assert np.isfinite(descriptors).all()


def test_export_representations_and_crop_verification_artifacts(tmp_path):
    audio_path = tmp_path / "data" / "train" / "audio" / "tiny" / "event.wav"
    write_sine(audio_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([row_for(audio_path)]).to_csv(manifest_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
audio:
  sample_rate: 250
  margin_seconds: 0.0
  n_fft: 64
  win_length: 64
  hop_length: 16
  n_mels: 32
  f_min: 0.0
  f_max: 125.0
  frequency_scale: linear
  normalization: sample
""".strip(),
        encoding="utf-8",
    )

    out_path = tmp_path / "representations" / "handcrafted.npz"
    summary = export_representations(
        manifest_path=manifest_path,
        config_path=config_path,
        out_path=out_path,
        family="handcrafted",
        img_size=16,
    )
    data = np.load(out_path, allow_pickle=True)

    assert summary["rows"] == 1
    assert summary["features"] == len(HANDCRAFTED_FEATURE_NAMES)
    assert data["X"].shape == (1, len(HANDCRAFTED_FEATURE_NAMES))
    assert Path(summary["manifest"]).exists()

    verify_dir = tmp_path / "verify"
    verify = export_crop_verification(
        manifest_path=manifest_path,
        config_path=config_path,
        out_dir=verify_dir,
        img_size=32,
        per_class=1,
    )

    assert len(verify) == 1
    assert Path(verify.iloc[0]["image_path"]).exists()
    assert (verify_dir / "crop_verification_manifest.csv").exists()
    assert verify.iloc[0]["crop_bins_frequency"] > 0
