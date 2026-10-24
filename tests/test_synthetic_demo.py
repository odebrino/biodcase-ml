import json

import pandas as pd

from examples.synthetic_demo import generate_audio, run_demo


def test_demo_runs_real_pipeline_with_isolated_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = run_demo(tmp_path / "demo", seed=42)
    summary = json.loads((output / "summary.json").read_text())
    predictions = pd.read_csv(output / "reports/official_test_predictions.csv")
    split = json.loads((output / "outputs/reports/quality/split_integrity.json").read_text())
    assert summary["data_kind"] == "synthetic_sine_waves"
    assert summary["train_rows"] == 24
    assert summary["holdout_rows"] == len(predictions) == 8
    assert set(predictions["dataset"]) == {"heldout_demo"}
    assert summary["accuracy"] >= 0.75
    assert split["leakage_detected"] is False
    assert len(summary["audio_sha256"]) == len(set(summary["audio_sha256"].values())) == 32
    assert not (tmp_path / "outputs").exists()


def test_generated_signals_repeat_with_same_seed(tmp_path):
    a = generate_audio(tmp_path / "a", 42)
    b = generate_audio(tmp_path / "b", 42)
    from pathlib import Path
    assert all(Path(left).read_bytes() == Path(right).read_bytes()
               for left, right in zip(a["audio_path"], b["audio_path"]))
