import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.models.knn_search import run_knn_search
from tests.knn_test_utils import make_small_manifest, tiny_search_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_knn_search_uses_train_only_and_writes_ranked_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = make_small_manifest(tmp_path / "data", include_unknown_validation_label=True)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "outputs" / "reports" / "knn_search"
    run_knn_search(
        tiny_search_config(),
        manifest_path=manifest_path,
        output_dir=output_dir,
        quick=True,
        max_candidates=4,
        n_jobs=1,
        cache_features_override=True,
    )

    results = pd.read_csv(output_dir / "search_results.csv")
    strategy = (output_dir / "split_strategy.json").read_text(encoding="utf-8")
    best_config = yaml.safe_load((output_dir / "best_knn_config.yaml").read_text(encoding="utf-8"))
    assert not results.empty
    assert "validation split is excluded" in strategy
    assert best_config["submission"]["feature_set"] in {
        "handcrafted_stats",
        "waveform_stats",
        "spectral_stats",
        "band_energy_linear_8",
        "notebook_lowfreq_band_features",
        "relative_lowfreq_shape_features",
        "temporal_lowfreq_shape_features",
        "lowfreq_relative_temporal",
        "waveform_spectral_stats",
    }
    assert "model" not in best_config
    assert "training" not in best_config
    assert "domain_cv_macro_f1_mean" in results.columns
    assert "random_cv_macro_f1_mean" in results.columns
    assert "random_cv_weighted_f1_mean" in results.columns
    assert "random_cv_macro_precision_mean" in results.columns
    assert "worst_domain_macro_f1" in results.columns
    assert (output_dir / "best_cv_metrics.json").exists()
    assert (output_dir / "search_summary.md").exists()
    assert (output_dir / "domain_cv_results.csv").exists()
    assert (output_dir / "feature_family_comparison.csv").exists()
    assert (output_dir / "notebook_feature_ablation.csv").exists()
    assert (output_dir / "notebook_feature_ablation.md").exists()
    assert (tmp_path / "outputs" / "reports" / "quality" / "label_mapping_audit.json").exists()
    assert (tmp_path / "outputs" / "reports" / "quality" / "label_mapping_audit.md").exists()


def test_knn_search_cli_smoke_run(tmp_path):
    manifest = make_small_manifest(tmp_path / "data")
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    config_path = tmp_path / "knn_search.yaml"
    config_path.write_text(yaml.safe_dump(tiny_search_config(), sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root())
    output_dir = tmp_path / "search_outputs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.knn_search",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--quick",
            "--max-candidates",
            "4",
            "--n-jobs",
            "1",
            "--cache-features",
            "--stage",
            "full",
            "--cv-focused",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "search_results.csv").exists()
    assert (output_dir / "cv_focused_search_results.csv").exists()
    assert (output_dir / "best_knn_config.yaml").exists()
    assert (output_dir / "domain_cv_results.csv").exists()
    assert (output_dir / "notebook_feature_ablation.csv").exists()


def test_search_results_are_ranked_by_domain_aware_score(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = make_small_manifest(tmp_path / "data")
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "outputs" / "reports" / "knn_search"
    config = tiny_search_config()
    config["search"]["stages"][0]["feature_sets"] = [
        "waveform_spectral_stats",
        "notebook_lowfreq_band_features",
        "relative_lowfreq_shape_features",
    ]
    config["search"]["stages"][0]["max_candidates"] = 6
    config["search"]["stages"][0]["ensure_feature_coverage"] = True
    run_knn_search(
        config,
        manifest_path=manifest_path,
        output_dir=output_dir,
        quick=False,
        max_candidates=6,
        n_jobs=1,
        cache_features_override=True,
    )

    results = pd.read_csv(output_dir / "search_results.csv")
    sortable = results[results["status"] == "ok"].copy()
    expected = sortable.sort_values(
        ["domain_cv_macro_f1_mean", "domain_cv_accuracy_mean", "random_cv_macro_f1_mean", "domain_cv_macro_f1_std"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    assert sortable.reset_index(drop=True)["feature_set"].tolist() == expected["feature_set"].tolist()


def test_cv_focused_search_results_are_ranked_by_stratified_score(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = make_small_manifest(tmp_path / "data")
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "outputs" / "reports" / "knn_search"
    config = tiny_search_config()
    config["search"]["stages"][0]["feature_sets"] = [
        "waveform_spectral_stats",
        "notebook_lowfreq_band_features",
        "relative_lowfreq_shape_features",
    ]
    config["search"]["stages"][0]["max_candidates"] = 6
    config["search"]["stages"][0]["ensure_feature_coverage"] = True
    run_knn_search(
        config,
        manifest_path=manifest_path,
        output_dir=output_dir,
        quick=False,
        max_candidates=6,
        n_jobs=1,
        cache_features_override=True,
        cv_focused=True,
    )

    results = pd.read_csv(output_dir / "cv_focused_search_results.csv")
    sortable = results[results["status"] == "ok"].copy()
    expected = sortable.sort_values(
        [
            "random_cv_weighted_f1_mean",
            "random_cv_accuracy_mean",
            "random_cv_macro_f1_mean",
            "random_cv_macro_precision_mean",
            "random_cv_weighted_f1_std",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    assert sortable.reset_index(drop=True)["feature_set"].tolist() == expected["feature_set"].tolist()
    assert (output_dir / "cv_focused_search_summary.md").exists()
