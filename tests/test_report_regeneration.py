import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from scipy.io import wavfile
import numpy as np


def test_manifest_reports_do_not_require_a_historical_cnn_run(tmp_path):
    root = tmp_path / "data"
    filename = "2020-01-01T00-00-00_000.wav"
    audio = root / "train/audio/demo" / filename
    audio.parent.mkdir(parents=True)
    wavfile.write(audio, 250, np.zeros(500, dtype="float32"))
    annotations = root / "train/annotations/demo.csv"
    annotations.parent.mkdir(parents=True)
    pd.DataFrame([{
        "dataset": "demo", "filename": filename, "annotation": "bma", "annotator": "synthetic",
        "low_frequency": 10.0, "high_frequency": 40.0,
        "start_datetime": "2020-01-01T00:00:00.25+00:00",
        "end_datetime": "2020-01-01T00:00:01.75+00:00",
    }]).to_csv(annotations, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts/regenerate_all_reports.py"
    result = subprocess.run([
        sys.executable, str(script), "--data-root", str(root), "--manifest-out", "manifest.csv",
        "--reports-root", "reports", "--splits", "train",
    ], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    provenance = json.loads((tmp_path / "reports/provenance/report_regeneration_provenance.json").read_text())
    assert provenance["manifest_rows"] == 1
    assert provenance["manifest_sha256"] == hashlib.sha256((tmp_path / "manifest.csv").read_bytes()).hexdigest()
    assert provenance["historical_audit_run_dir"] is None
    assert provenance["legacy_audit_included"] is False
    assert "imbalance_audit_json" not in provenance["artifacts"]
    assert (tmp_path / "reports/manifest/class_distribution_by_split.csv").exists()
