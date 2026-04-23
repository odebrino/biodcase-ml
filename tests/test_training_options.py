import pandas as pd
import torch
from scipy.io import wavfile

from src.training.common import create_loader
from src.training.losses import FocalLoss
from src.training.train import split_semantics


def test_focal_loss_is_finite():
    loss = FocalLoss(gamma=2.0)
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.5]])
    targets = torch.tensor([0, 1])
    assert torch.isfinite(loss(logits, targets))


def test_focal_loss_applies_weight_after_focal_factor():
    weights = torch.tensor([1.0, 5.0])
    loss = FocalLoss(gamma=2.0, weight=weights)
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.5]])
    targets = torch.tensor([0, 1])

    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    expected = (((1 - pt) ** 2.0) * ce * weights[targets]).mean()

    assert torch.allclose(loss(logits, targets), expected)


def test_weighted_sampler_is_enabled(tmp_path):
    audio_path = tmp_path / "tiny.wav"
    wavfile.write(audio_path, 250, torch.zeros(500).numpy().astype("float32"))
    rows = []
    for idx, label in enumerate(["bma", "bma", "bpd"]):
        rows.append(
            {
                "split": "train",
                "dataset": "tiny",
                "filename": "tiny.wav",
                "audio_path": str(audio_path),
                "label": label,
                "source_row": idx,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "clip_start_seconds": 0.0,
                "clip_end_seconds": 1.0,
                "duration_seconds": 1.0,
                "real_duration_seconds": 1.0,
                "valid_event": True,
                "low_frequency": 10.0,
                "high_frequency": 20.0,
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = {
        "classes": ["bma", "bpd"],
        "model": {"img_size": 32},
        "audio": {
            "sample_rate": 250,
            "margin_seconds": 0.0,
            "n_fft": 64,
            "hop_length": 16,
            "n_mels": 16,
            "f_min": 0.0,
            "f_max": 125.0,
        },
        "dataset": {"mode": "spectrogram"},
        "cache": {"enabled": False},
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "sampler": "weighted",
            "class_weight_multipliers": {"bpd": 3.0},
        },
    }
    loader = create_loader(manifest, "train", config["classes"], train=True, config=config)
    assert loader.sampler is not None
    sample_weights = loader.sampler.weights.tolist()
    assert sample_weights[2] / sample_weights[0] > 5.9


def test_cnn_training_requires_explicit_inner_selection_split():
    config = {
        "train_split": "train",
        "val_split": "validation",
        "official_test_split": "validation",
        "inner_selection_split": None,
        "training": {"allow_official_test_for_selection": False},
    }

    try:
        split_semantics(config)
    except ValueError as exc:
        assert "requires an explicit train-only inner_selection_split" in str(exc)
    else:
        raise AssertionError("Expected split_semantics to reject silent official-test selection fallback.")


def test_cnn_training_can_opt_in_to_historical_official_test_selection():
    config = {
        "train_split": "train",
        "val_split": "validation",
        "official_test_split": "validation",
        "inner_selection_split": None,
        "selection_strategy": "explicit_opt_in_official_test",
        "training": {"allow_official_test_for_selection": True},
    }

    semantics = split_semantics(config)
    assert semantics["effective_selection_split"] == "validation"
    assert semantics["used_official_test_for_selection"] is True
