from pathlib import Path

import pandas as pd
from scipy.io import wavfile

from src import pipeline


def test_pipeline_build_manifest_writes_cli_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data"
    audio_path = data_root / "train" / "audio" / "tiny" / "2020-01-01T00-00-00_000.wav"
    annotation_path = data_root / "train" / "annotations" / "tiny.csv"
    audio_path.parent.mkdir(parents=True)
    annotation_path.parent.mkdir(parents=True)
    wavfile.write(audio_path, 250, pd.Series([0.0] * 500).astype("float32").to_numpy())
    pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "filename": audio_path.name,
                "annotation": "bma",
                "annotator": "test",
                "low_frequency": 10.0,
                "high_frequency": 20.0,
                "start_datetime": "2020-01-01T00:00:00.500000+00:00",
                "end_datetime": "2020-01-01T00:00:01.500000+00:00",
            }
        ]
    ).to_csv(annotation_path, index=False)

    manifest, issues = pipeline.build_manifest(
        data_root=data_root,
        out="manifest.csv",
        quality_report="outputs/quality.csv",
        quality_summary="outputs/summary.csv",
        splits=("train",),
    )

    assert len(manifest) == 1
    assert issues.empty
    assert Path("manifest.csv").exists()
    assert Path("outputs/quality.csv").exists()
    assert Path("outputs/summary.csv").exists()
    assert Path("outputs/reports/manifest/train_class_distribution.csv").exists()


def test_pipeline_cache_summary(tmp_path):
    cache_root = tmp_path / "cache"
    (cache_root / "train" / "bma").mkdir(parents=True)
    (cache_root / "train" / "bma" / "item.pt").write_bytes(b"1234")

    summary = pipeline.cache_summary(cache_root)

    assert summary["files"] == 1
    assert summary["size_gb"] > 0
