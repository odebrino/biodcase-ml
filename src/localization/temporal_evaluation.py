"""
Temporal Localization Evaluation Metrics (Phase 6 Bonus)

Implements IoU-based metrics for evaluating temporal event detection.

Metrics:
- Intersection over Union (IoU): overlap between predicted and ground truth boxes
- Precision/Recall at IoU thresholds (similar to COCO detection evaluation)
- Event-level accuracy: proportion of correctly localized events
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class IoUMetrics:
    """Metrics for temporal event localization."""
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    
    def to_dict(self) -> Dict:
        return {
            "iou_threshold": self.iou_threshold,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_iou": self.mean_iou,
        }


def compute_iou(pred_box: Tuple[float, float], gt_box: Tuple[float, float]) -> float:
    """
    Compute Intersection over Union (IoU) for temporal boxes.
    
    Args:
        pred_box: (start, end) predicted temporal bounds
        gt_box: (start, end) ground truth temporal bounds
    
    Returns:
        IoU score in [0, 1]
    """
    pred_start, pred_end = pred_box
    gt_start, gt_end = gt_box
    
    # Intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter_duration = max(0, inter_end - inter_start)
    
    # Union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union_duration = union_end - union_start
    
    if union_duration == 0:
        return 0.0
    
    return inter_duration / union_duration


def match_detections_to_ground_truth(
    predictions: List,
    ground_truth: List,
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, float]:
    """
    Match predicted events to ground truth events using greedy IoU matching.
    
    Args:
        predictions: List of TemporalEvent (detected)
        ground_truth: List of TemporalEvent (ground truth)
        iou_threshold: Minimum IoU to count as match
    
    Returns:
        (true_positives, false_positives, false_negatives, mean_iou)
    """
    if len(ground_truth) == 0 and len(predictions) == 0:
        return 0, 0, 0, 1.0
    
    if len(ground_truth) == 0:
        # All predictions are false positives
        return 0, len(predictions), 0, 0.0
    
    if len(predictions) == 0:
        # All ground truth are false negatives
        return 0, 0, len(ground_truth), 0.0
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(predictions), len(ground_truth)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            iou_matrix[i, j] = compute_iou(
                (pred.start_time, pred.end_time),
                (gt.start_time, gt.end_time)
            )
    
    # Greedy matching
    matched_gt = set()
    matched_pred = set()
    tp = 0
    ious = []
    
    for i in range(len(predictions)):
        best_j = np.argmax(iou_matrix[i, :])
        best_iou = iou_matrix[i, best_j]
        
        if best_iou >= iou_threshold and best_j not in matched_gt:
            tp += 1
            matched_gt.add(best_j)
            matched_pred.add(i)
            ious.append(best_iou)
    
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    mean_iou = np.mean(ious) if ious else 0.0
    
    return tp, fp, fn, mean_iou


def evaluate_temporal_detection(
    detection_results: List,
    iou_thresholds: List[float] = [0.5, 0.75, 0.9],
) -> Dict:
    """
    Evaluate temporal detection across multiple IoU thresholds.
    
    Args:
        detection_results: List of DetectionResult objects
        iou_thresholds: List of IoU thresholds for evaluation
    
    Returns:
        Dict with metrics for each threshold
    """
    results_by_threshold = {}
    
    for iou_thresh in iou_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_iou = 0
        count = 0
        
        for result in detection_results:
            tp, fp, fn, mean_iou = match_detections_to_ground_truth(
                result.detected_events,
                result.ground_truth_events,
                iou_threshold=iou_thresh,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_iou += mean_iou
            count += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = total_iou / count if count > 0 else 0.0
        
        metrics = IoUMetrics(
            iou_threshold=iou_thresh,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=mean_iou,
        )
        
        results_by_threshold[iou_thresh] = metrics
    
    return results_by_threshold


def generate_temporal_detection_report(
    detection_results: List,
    output_path: str = "temporal_detection_report.json",
) -> Dict:
    """
    Generate a comprehensive report of temporal detection evaluation.
    
    Includes:
    - Per-threshold metrics
    - Per-file summary
    - Class-level breakdown (if multiclass)
    
    Args:
        detection_results: List of DetectionResult objects
        output_path: Where to save JSON report
    
    Returns:
        Report dict
    """
    # Evaluate at multiple thresholds
    threshold_results = evaluate_temporal_detection(
        detection_results,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95]
    )
    
    # Aggregate stats
    report = {
        "evaluation_type": "temporal_localization",
        "num_files": len(detection_results),
        "metrics_by_iou_threshold": {
            str(k): v.to_dict() for k, v in threshold_results.items()
        },
        "file_summaries": [],
    }
    
    # Per-file summaries
    for result in detection_results:
        file_summary = {
            "audio_path": result.audio_path,
            "duration_seconds": result.duration_seconds,
            "ground_truth_events": len(result.ground_truth_events),
            "detected_events": len(result.detected_events),
        }
        
        # IoU at 0.5 threshold for quick reference
        tp, fp, fn, _ = match_detections_to_ground_truth(
            result.detected_events,
            result.ground_truth_events,
            iou_threshold=0.5,
        )
        file_summary["tp_at_iou50"] = tp
        file_summary["fp_at_iou50"] = fp
        file_summary["fn_at_iou50"] = fn
        
        report["file_summaries"].append(file_summary)
    
    # Save report
    if output_path:
        import json
        with open(output_path, 'w') as f:
            # Convert np types to native Python for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(report, f, indent=2, default=convert)
        print(f"Report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage (demonstration)
    print("Temporal Localization Evaluation Module")
    print("Supports: IoU, Precision/Recall, F1 at multiple thresholds")
    print("See PHASE_6_BONUS_TEMPORAL_LOCALIZATION.md for usage examples.")
