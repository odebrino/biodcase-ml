"""Evaluation helpers for the KNN submission path."""

from .metrics import (
    compute_metrics,
    metrics_by_dataset,
    metrics_from_label_names,
    normalized_confusion_matrix,
    per_class_metrics_table,
    write_classification_report,
    write_confusion_matrix_csv,
    write_metrics_json,
)
from .domain_diagnostics import per_sample_knn_neighbors, write_domain_diagnostics
from .label_mapping_audit import NOTEBOOK_3CLASS_MAPPING, write_label_mapping_audit
from .reports import (
    plot_confusion_matrix,
    write_baseline_metrics,
    write_bmb_bmz_error_report,
    write_bpd_error_report,
    write_class_confidence_analysis,
    write_dataset_class_metrics,
    write_dataset_metrics,
    write_error_analysis,
    write_json,
    write_submission_overview,
)
from .split_checks import (
    REQUIRED_MANIFEST_COLUMNS,
    assert_split_integrity,
    build_split_integrity_report,
    write_split_integrity_report,
)

__all__ = [
    "REQUIRED_MANIFEST_COLUMNS",
    "NOTEBOOK_3CLASS_MAPPING",
    "assert_split_integrity",
    "build_split_integrity_report",
    "compute_metrics",
    "metrics_by_dataset",
    "metrics_from_label_names",
    "normalized_confusion_matrix",
    "per_class_metrics_table",
    "plot_confusion_matrix",
    "per_sample_knn_neighbors",
    "write_baseline_metrics",
    "write_bmb_bmz_error_report",
    "write_bpd_error_report",
    "write_class_confidence_analysis",
    "write_classification_report",
    "write_confusion_matrix_csv",
    "write_dataset_class_metrics",
    "write_dataset_metrics",
    "write_domain_diagnostics",
    "write_error_analysis",
    "write_json",
    "write_label_mapping_audit",
    "write_metrics_json",
    "write_split_integrity_report",
    "write_submission_overview",
]
