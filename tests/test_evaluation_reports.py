import pandas as pd

from src.evaluation.metrics import normalized_confusion_matrix
from src.evaluation.reports import (
    write_baseline_metrics,
    write_bmb_bmz_error_report,
    write_bpd_error_report,
    write_class_confidence_analysis,
    write_dataset_class_metrics,
    write_dataset_metrics,
)


def test_dataset_class_and_bpd_reports(tmp_path):
    predictions = pd.DataFrame(
        [
            {
                "dataset": "casey2017",
                "filename": "a.wav",
                "source_row": 1,
                "y_true_idx": 0,
                "y_pred_idx": 1,
                "y_true_label": "bpd",
                "y_pred_label": "bmd",
                "pred_confidence": 0.9,
                "true_probability": 0.1,
                "low_frequency": 10.0,
                "high_frequency": 20.0,
                "duration_seconds": 1.0,
                "real_duration_seconds": 1.0,
                "clip_start_seconds": 0.0,
                "clip_end_seconds": 1.0,
                "audio_path": "a.wav",
            },
            {
                "dataset": "casey2017",
                "filename": "b.wav",
                "source_row": 2,
                "y_true_idx": 1,
                "y_pred_idx": 1,
                "y_true_label": "bmd",
                "y_pred_label": "bmd",
                "pred_confidence": 0.8,
                "true_probability": 0.8,
                "low_frequency": 10.0,
                "high_frequency": 20.0,
                "duration_seconds": 1.0,
                "real_duration_seconds": 1.0,
                "clip_start_seconds": 0.0,
                "clip_end_seconds": 1.0,
                "audio_path": "b.wav",
            },
            {
                "dataset": "kerguelen2014",
                "filename": "c.wav",
                "source_row": 3,
                "y_true_idx": 2,
                "y_pred_idx": 3,
                "y_true_label": "bmb",
                "y_pred_label": "bmz",
                "pred_confidence": 0.95,
                "true_probability": 0.02,
                "low_frequency": 15.0,
                "high_frequency": 35.0,
                "duration_seconds": 2.0,
                "real_duration_seconds": 2.0,
                "clip_start_seconds": 1.0,
                "clip_end_seconds": 3.0,
                "audio_path": "c.wav",
            },
        ]
    )
    dataset_class_path = tmp_path / "metrics_by_dataset_class.csv"
    dataset_metrics_path = tmp_path / "metrics_by_dataset.csv"
    bpd_path = tmp_path / "bpd_error_report.csv"
    bmb_bmz_path = tmp_path / "bmb_bmz_error_report.csv"

    write_dataset_class_metrics(predictions, ["bpd", "bmd", "bmb", "bmz"], dataset_class_path)
    write_dataset_metrics(predictions, ["bpd", "bmd", "bmb", "bmz", "absent"], dataset_metrics_path)
    write_bpd_error_report(predictions, bpd_path)
    write_bmb_bmz_error_report(predictions, bmb_bmz_path)

    dataset_class = pd.read_csv(dataset_class_path)
    dataset_metrics = pd.read_csv(dataset_metrics_path)
    bpd = pd.read_csv(bpd_path)
    bmb_bmz = pd.read_csv(bmb_bmz_path)
    assert set(dataset_class["label"]) == {"bpd", "bmd", "bmb", "bmz"}
    assert "macro_f1_present_classes" in dataset_metrics.columns
    assert dataset_metrics.iloc[0]["macro_f1_present_classes"] > dataset_metrics.iloc[0]["macro_f1"]
    assert len(bpd) == 1
    assert bpd.iloc[0]["y_true_label"] == "bpd"
    assert bpd.iloc[0]["y_pred_label"] == "bmd"
    assert len(bmb_bmz) == 1
    assert bmb_bmz.iloc[0]["y_true_label"] == "bmb"
    assert bmb_bmz.iloc[0]["y_pred_label"] == "bmz"


def test_normalized_confusion_matrix_rows_sum_for_non_empty_classes():
    normalized = normalized_confusion_matrix([[2, 1], [0, 0]])
    assert abs(sum(normalized[0]) - 1.0) < 1e-9
    assert sum(normalized[1]) == 0.0


def test_baseline_and_confidence_reports_include_all_classes(tmp_path):
    predictions = pd.DataFrame(
        [
            {
                "y_true_idx": 0,
                "y_pred_idx": 0,
                "y_true_label": "a",
                "y_pred_label": "a",
                "pred_confidence": 0.9,
                "true_probability": 0.9,
            },
            {
                "y_true_idx": 1,
                "y_pred_idx": 0,
                "y_true_label": "b",
                "y_pred_label": "a",
                "pred_confidence": 0.8,
                "true_probability": 0.2,
            },
        ]
    )
    baseline_path = tmp_path / "baseline_metrics.csv"
    confidence_path = tmp_path / "class_confidence_analysis.csv"
    write_baseline_metrics(predictions, ["a", "b"], baseline_path)
    write_class_confidence_analysis(predictions, ["a", "b"], confidence_path)

    baselines = pd.read_csv(baseline_path)
    confidence = pd.read_csv(confidence_path)
    assert {"majority_class", "stratified_random_seed0"} == set(baselines["baseline"])
    assert set(confidence["label"]) == {"a", "b"}
    assert confidence["support"].sum() == len(predictions)
