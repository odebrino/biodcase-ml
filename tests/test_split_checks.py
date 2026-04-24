import pytest

from src.evaluation.split_checks import assert_split_integrity, build_split_integrity_report


def test_split_integrity_detects_duplicate_event_ids_across_train_and_validation():
    manifest = [
        {
            "split": "train",
            "dataset": "tiny",
            "filename": "event.wav",
            "audio_path": "/tmp/train.wav",
            "label": "bma",
            "source_row": 10,
            "clip_start_seconds": 0.0,
            "clip_end_seconds": 1.0,
            "low_frequency": 10.0,
            "high_frequency": 20.0,
        },
        {
            "split": "validation",
            "dataset": "tiny",
            "filename": "event.wav",
            "audio_path": "/tmp/train.wav",
            "label": "bma",
            "source_row": 10,
            "clip_start_seconds": 0.0,
            "clip_end_seconds": 1.0,
            "low_frequency": 10.0,
            "high_frequency": 20.0,
        },
    ]

    import pandas as pd

    report = build_split_integrity_report(pd.DataFrame(manifest))
    assert report["leakage_detected"] is True
    assert report["checks"]["audio_path"]["duplicate_count"] == 1
    with pytest.raises(ValueError, match="leakage"):
        assert_split_integrity(report)
