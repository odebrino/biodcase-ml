from src.evaluation.metrics import compute_metrics


def test_compute_metrics_handles_absent_class():
    metrics = compute_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        class_names=["a", "b", "absent"],
    )
    assert metrics["confusion_matrix"][2] == [0, 0, 0]
    assert metrics["macro_f1"] < metrics["macro_f1_present_classes"]
    assert "balanced_accuracy" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "weighted_precision" in metrics
    assert "weighted_recall" in metrics
