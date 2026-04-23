"""
Temporal Detector: Frame-level event detection for multi-event localization.

This module implements a simple sliding window detector that:
1. Receives weak supervision from manifest time annotations
2. Creates frame-level labels for training (event vs. no-event)
3. Trains a classifier on spectrograms or features
4. Outputs temporal bounding boxes via post-processing

DESIGN PRINCIPLE:
- Separate from crop-level classification
- Uses time/frequency annotations as weak labels
- Does NOT require perfect event alignment
- Frames with any event portion → labeled as event
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TemporalEvent:
    """Represents a detected or ground-truth event."""
    start_time: float          # seconds
    end_time: float            # seconds
    label: str                 # class label
    confidence: float = 1.0    # detection confidence [0, 1]
    source: str = "ground_truth"  # "ground_truth" or "detected"


@dataclass
class DetectionResult:
    """Result of temporal detection on one audio file."""
    audio_path: str
    duration_seconds: float
    ground_truth_events: List[TemporalEvent]
    detected_events: List[TemporalEvent]
    frame_labels: np.ndarray  # (n_frames,) binary or multi-class
    frame_predictions: np.ndarray  # (n_frames,) predictions
    frame_confidences: np.ndarray  # (n_frames, n_classes) or (n_frames,)


class TemporalDetector:
    """
    Frame-level detector for temporal event localization.
    
    Approach:
    1. Manifest time annotations → frame-level labels (event/no-event)
    2. Sliding window over spectrogram time axis
    3. Per-frame classification (binary or multi-class)
    4. Post-processing: connected components → bounding boxes
    5. Evaluation: IoU metrics against ground truth
    
    HONEST LIMITATION:
    This requires raw WAV files to extract spectrograms at runtime.
    Currently, the framework is implemented; execution requires external data.
    For testing, we provide manifest-based offline evaluation.
    """
    
    def __init__(
        self,
        frame_duration_ms: float = 512.0,  # overlap for dense frames
        min_event_duration_sec: float = 0.5,
        confidence_threshold: float = 0.5,
        use_multiclass: bool = False,
        classes: Optional[List[str]] = None,
    ):
        """
        Args:
            frame_duration_ms: Duration of each frame in milliseconds
            min_event_duration_sec: Minimum duration to report as event
            confidence_threshold: Post-processing threshold
            use_multiclass: If True, predict class per frame; else binary
            classes: List of class labels (if multiclass)
        """
        self.frame_duration_ms = frame_duration_ms
        self.min_event_duration_sec = min_event_duration_sec
        self.confidence_threshold = confidence_threshold
        self.use_multiclass = use_multiclass
        self.classes = classes or []
        self.model = None
    
    def create_frame_labels_from_manifest(
        self,
        manifest_df: pd.DataFrame,
        audio_duration_seconds: float,
        fps: float = 4.0,  # frames per second (matches frame_duration_ms)
    ) -> Tuple[np.ndarray, np.ndarray, List[TemporalEvent]]:
        """
        Convert manifest annotations to frame-level labels.
        
        Args:
            manifest_df: Rows from data_manifest.csv for one audio file
            audio_duration_seconds: Total duration of the audio
            fps: Frames per second (default 4.0 = 250ms per frame)
        
        Returns:
            frame_labels: (n_frames,) binary labels (1 = event, 0 = silence)
            frame_class_labels: (n_frames,) per-frame class indices
            events: List of TemporalEvent objects from manifest
        """
        n_frames = int(np.ceil(audio_duration_seconds * fps))
        frame_labels = np.zeros(n_frames, dtype=np.int32)
        frame_class_labels = np.zeros(n_frames, dtype=np.int32)
        events = []
        
        class_to_idx = {c: i + 1 for i, c in enumerate(self.classes)} if self.use_multiclass else {}
        
        for _, row in manifest_df.iterrows():
            start_sec = row['start_seconds']
            end_sec = row['end_seconds']
            label = row['label']
            
            # Create event object
            event = TemporalEvent(
                start_time=start_sec,
                end_time=end_sec,
                label=label,
                source="ground_truth"
            )
            events.append(event)
            
            # Convert to frame indices
            start_frame = int(start_sec * fps)
            end_frame = int(np.ceil(end_sec * fps))
            end_frame = min(end_frame, n_frames)
            
            # Mark frames as event
            frame_labels[start_frame:end_frame] = 1
            
            if self.use_multiclass and label in class_to_idx:
                frame_class_labels[start_frame:end_frame] = class_to_idx[label]
        
        return frame_labels, frame_class_labels, events
    
    def post_process_frame_predictions(
        self,
        frame_predictions: np.ndarray,
        frame_confidences: np.ndarray,
        fps: float = 4.0,
    ) -> List[TemporalEvent]:
        """
        Convert frame-level predictions to temporal event boxes.
        
        Args:
            frame_predictions: (n_frames,) predicted class indices
            frame_confidences: (n_frames,) or (n_frames, n_classes) confidence scores
            fps: Frames per second
        
        Returns:
            List of TemporalEvent objects representing detected events
        """
        if frame_confidences.ndim == 2:
            # Multi-class confidences: take max per frame
            confidences = np.max(frame_confidences, axis=1)
        else:
            confidences = frame_confidences
        
        # Extract connected components above threshold
        binary_pred = (frame_predictions > 0) & (confidences >= self.confidence_threshold)
        
        # Find transitions
        diff = np.diff(binary_pred.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if binary_pred[0]:
            starts = np.concatenate([[0], starts])
        if binary_pred[-1]:
            ends = np.concatenate([ends, [len(binary_pred)]])
        
        events = []
        for start_frame, end_frame in zip(starts, ends):
            duration_sec = (end_frame - start_frame) / fps
            
            # Filter by minimum duration
            if duration_sec >= self.min_event_duration_sec:
                # Determine class if multiclass
                class_preds = frame_predictions[start_frame:end_frame]
                if self.use_multiclass and np.any(class_preds > 0):
                    class_idx = np.argmax(np.bincount(class_preds[class_preds > 0]))
                    label = self.classes[class_idx - 1] if class_idx > 0 else "unknown"
                else:
                    label = "event"
                
                # Average confidence
                avg_confidence = np.mean(confidences[start_frame:end_frame])
                
                event = TemporalEvent(
                    start_time=start_frame / fps,
                    end_time=end_frame / fps,
                    label=label,
                    confidence=avg_confidence,
                    source="detected"
                )
                events.append(event)
        
        return events
    
    def create_dummy_predictions(
        self,
        frame_labels: np.ndarray,
        fps: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dummy predictions for testing (without real model).
        
        This is for demonstration: predictions = labels with slight noise.
        In production, replace with actual model inference.
        """
        # Add small random noise to labels for realism
        noise = np.random.binomial(1, 0.1, size=frame_labels.shape)
        predictions = np.clip(frame_labels + noise, 0, 1)
        
        # Confidences: high for correct frames, low otherwise
        confidences = np.ones_like(predictions, dtype=np.float32) * 0.6
        confidences[predictions == frame_labels] = 0.9
        
        return predictions, confidences
    
    def detect_on_manifest_row(
        self,
        manifest_row: pd.Series,
        manifest_df: pd.DataFrame,
        fps: float = 4.0,
    ) -> DetectionResult:
        """
        Offline detection using manifest annotations as labels.
        
        This demonstrates the pipeline without requiring actual WAV files.
        For real execution, replace the dummy predictions with model inference.
        
        Args:
            manifest_row: One row representing an audio file
            manifest_df: All rows (to get events from this audio)
            fps: Frames per second
        
        Returns:
            DetectionResult with ground truth and detections
        """
        audio_duration = manifest_row['audio_duration_seconds']
        audio_path = manifest_row['audio_path']
        
        # Create frame labels from manifest annotations
        frame_labels, frame_class_labels, ground_truth_events = self.create_frame_labels_from_manifest(
            manifest_df,
            audio_duration,
            fps=fps
        )
        
        # Get predictions (in real scenario, run inference on spectrogram)
        frame_predictions, frame_confidences = self.create_dummy_predictions(frame_labels, fps=fps)
        
        # Post-process to get temporal boxes
        detected_events = self.post_process_frame_predictions(
            frame_predictions, frame_confidences, fps=fps
        )
        
        result = DetectionResult(
            audio_path=audio_path,
            duration_seconds=audio_duration,
            ground_truth_events=ground_truth_events,
            detected_events=detected_events,
            frame_labels=frame_labels,
            frame_predictions=frame_predictions,
            frame_confidences=frame_confidences,
        )
        
        return result
    
    def report_summary(self, results: List[DetectionResult]) -> Dict:
        """
        Generate summary statistics for a list of detection results.
        
        Returns dict with total events detected, ground truth, avg IoU, etc.
        """
        total_gt = sum(len(r.ground_truth_events) for r in results)
        total_det = sum(len(r.detected_events) for r in results)
        
        return {
            "num_files": len(results),
            "total_ground_truth_events": total_gt,
            "total_detected_events": total_det,
            "frame_predictions_shape": results[0].frame_predictions.shape if results else None,
            "status": "DEMO / REQUIRES_RAW_WAV_FOR_REAL_INFERENCE",
        }


if __name__ == "__main__":
    # Example usage (demonstration only)
    detector = TemporalDetector(
        classes=["bma", "bmb", "bmd", "bmz", "bp20", "bp20plus", "bpd"],
        use_multiclass=True,
    )
    print("Temporal Detector initialized.")
    print(f"Classes: {detector.classes}")
    print(f"This detector requires rawWAV files for actual inference.")
    print("See PHASE_6_BONUS_TEMPORAL_LOCALIZATION.md for usage.")
