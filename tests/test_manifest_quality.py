from pathlib import Path

import pandas as pd
from scipy.io import wavfile

from src.data.labels import normalize_label
from src.data.build_manifest import build_manifest


def write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 250) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, (pd.Series([0.0] * int(seconds * sample_rate))).astype("float32").to_numpy())


def write_annotations(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": "tiny",
            "filename": "2020-01-01T00-00-00_000.wav",
            "annotation": "bma",
            "annotator": "test",
            "low_frequency": 10.0,
            "high_frequency": 20.0,
            "start_datetime": "2020-01-01T00:00:00.500000+00:00",
            "end_datetime": "2020-01-01T00:00:01.500000+00:00",
        },
        {
            "dataset": "tiny",
            "filename": "2020-01-01T00-00-00_000.wav",
            "annotation": "bma",
            "annotator": "test",
            "low_frequency": 10.0,
            "high_frequency": 20.0,
            "start_datetime": "2020-01-01T00:00:00.500000+00:00",
            "end_datetime": "2020-01-01T00:00:01.500000+00:00",
        },
        {
            "dataset": "tiny",
            "filename": "2020-01-01T00-00-00_000.wav",
            "annotation": "bmb",
            "annotator": "test",
            "low_frequency": 10.0,
            "high_frequency": 20.0,
            "start_datetime": "2020-01-01T00:00:03.000000+00:00",
            "end_datetime": "2020-01-01T00:00:04.000000+00:00",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_manifest_discards_duplicates_and_out_of_bounds(tmp_path):
    root = tmp_path / "data"
    write_wav(root / "train" / "audio" / "tiny" / "2020-01-01T00-00-00_000.wav")
    write_annotations(root / "train" / "annotations" / "tiny.csv")

    manifest, issues = build_manifest(root, ["train"], min_valid_seconds=0.5)

    assert len(manifest) == 1
    assert bool(manifest.iloc[0]["valid_event"]) is True
    assert set(issues["issue"]) == {"duplicate_event", "start_after_audio_end"}
    assert manifest.iloc[0]["audio_duration_seconds"] == 2.0


def test_label_aliases_are_normalized_at_manifest_ingestion(tmp_path):
    root = tmp_path / "data"
    audio = root / "train" / "audio" / "tiny" / "2020-01-01T00-00-00_000.wav"
    write_wav(audio)
    rows = [
        {
            "dataset": "tiny",
            "filename": audio.name,
            "annotation": "Bm-A",
            "annotator": "test",
            "low_frequency": 10.0,
            "high_frequency": 20.0,
            "start_datetime": "2020-01-01T00:00:00.100000+00:00",
            "end_datetime": "2020-01-01T00:00:00.600000+00:00",
        },
        {
            "dataset": "tiny",
            "filename": audio.name,
            "annotation": "Bp-40Down",
            "annotator": "test",
            "low_frequency": 30.0,
            "high_frequency": 90.0,
            "start_datetime": "2020-01-01T00:00:00.700000+00:00",
            "end_datetime": "2020-01-01T00:00:01.300000+00:00",
        },
    ]
    annotation_path = root / "train" / "annotations" / "tiny.csv"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(annotation_path, index=False)

    manifest, issues = build_manifest(root, ["train"], min_valid_seconds=0.1)

    assert issues.empty
    assert manifest["label"].tolist() == ["bma", "bpd"]
    assert manifest["label_raw"].tolist() == ["Bm-A", "Bp-40Down"]
    assert manifest["label_display"].tolist() == ["Bm-A", "Bp-40Down"]


def test_all_required_label_alias_variants_normalize():
    aliases = {
        "bma": "bma",
        "BmA": "bma",
        "Bm-A": "bma",
        "bmb": "bmb",
        "BmB": "bmb",
        "Bm-B": "bmb",
        "bmz": "bmz",
        "BmZ": "bmz",
        "Bm-Z": "bmz",
        "bmd": "bmd",
        "BmD": "bmd",
        "Bm-D": "bmd",
        "bp20": "bp20",
        "Bp20": "bp20",
        "Bp-20": "bp20",
        "bp20plus": "bp20plus",
        "Bp20plus": "bp20plus",
        "Bp-20Plus": "bp20plus",
        "bpd": "bpd",
        "BpD": "bpd",
        "Bp-40Down": "bpd",
    }

    assert {alias: normalize_label(alias) for alias in aliases} == aliases
