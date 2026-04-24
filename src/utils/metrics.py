"""Backward-compatible metrics re-export.

The submission-oriented source of truth now lives in ``src.evaluation.metrics``.
"""

from src.evaluation.metrics import compute_metrics, write_classification_report

__all__ = ["compute_metrics", "write_classification_report"]
