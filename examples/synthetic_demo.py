"""Exercise the maintained KNN workflow without external recordings.

The signals are generated sine waves, not whale calls. Their metrics validate
software execution only. Run from the repository root with:
    python -m examples.synthetic_demo --output-dir outputs/demo
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.io import wavfile
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_audio(output_dir: Path, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_rate = 250
    time = np.arange(2 * sample_rate) / sample_rate
    rows = []
    domains = [("train", "train_a"), ("train", "train_b"),
               ("train", "train_c"), ("validation", "heldout_demo")]
    for split, domain in domains:
        for label, center_frequency in [("bma", 25.0), ("bmd", 55.0)]:
            for repeat in range(4):
                frequency = center_frequency + rng.uniform(-1.0, 1.0)
                signal = (0.5 * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi))
                          + rng.normal(0, 0.015, len(time))).astype("float32")
                audio = output_dir / "audio" / domain / f"{label}_{repeat}.wav"
                audio.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(audio, sample_rate, signal)
                rows.append({
                    "split": split, "dataset": domain, "filename": audio.name,
                    "audio_path": str(audio), "label": label, "label_raw": label,
                    "source_row": len(rows), "start_seconds": 0.25, "end_seconds": 1.75,
                    "clip_start_seconds": 0.25, "clip_end_seconds": 1.75,
                    "duration_seconds": 1.5, "real_duration_seconds": 1.5,
                    "low_frequency": 10.0, "high_frequency": 80.0, "valid_event": True,
                })
    return pd.DataFrame(rows)


def run_demo(output_dir: Path, seed: int = 42) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_audio(output_dir, seed)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    config = {
        "classes": ["bma", "bmd"], "seed": seed,
        "train_split": "train", "test_split": "validation",
        "held_out_test_domains": ["heldout_demo"],
        "submission": {"feature_set": "waveform_spectral_stats"},
        "audio": {"sample_rate": 250, "margin_seconds": 0.0, "n_fft": 64,
                  "win_length": 64, "hop_length": 16, "n_mels": 16,
                  "f_min": 0.0, "f_max": 125.0, "frequency_scale": "linear",
                  "normalization": "sample"},
        "preprocessing": {"imputer": "median", "scaler": "standard", "reducer": {"type": "none"}},
        "knn": {"n_neighbors": 3, "weights": "uniform", "metric": "euclidean"},
        "feature_cache": {"enabled": False},
    }
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    reports = output_dir / "reports"
    command = [sys.executable, "-m", "src.run_submission", "--config", str(config_path),
               "--manifest", str(manifest_path), "--output-dir", str(reports)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    # Running in the demo directory also isolates legacy relative quality-report paths.
    subprocess.run(command, cwd=output_dir, env=env, check=True)
    metrics = json.loads((reports / "official_test_metrics.json").read_text())
    summary = {
        "data_kind": "synthetic_sine_waves", "seed": seed,
        "purpose": "Software smoke check; not a whale-call or challenge result",
        "train_rows": int((manifest["split"] == "train").sum()),
        "holdout_rows": int((manifest["split"] == "validation").sum()),
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "audio_sha256": {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((output_dir / "audio").rglob("*.wav"))},
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in ["numpy", "scipy", "scikit-learn", "torch"]},
        "run_command": command,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Synthetic pipeline demonstration\n\n"
        "Generated sine waves exercise manifest ingestion, features, training, and evaluation. "
        "The bma/bmd labels are placeholders; the audio contains no real whale recordings.\n\n"
        f"Seed: {seed}. Training rows: {summary['train_rows']}; holdout rows: {summary['holdout_rows']}.\n\n"
        "See `summary.json` for package versions, audio hashes, and the command; "
        "`reports/` for predictions, metrics, and confusion plots; "
        "`outputs/reports/quality/` for split checks. Filenames containing `official_test` "
        "are legacy pipeline names; this run is synthetic.\n",
        encoding="utf-8",
    )
    print(f"Synthetic demo complete: {output_dir / 'summary.json'}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_demo(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
