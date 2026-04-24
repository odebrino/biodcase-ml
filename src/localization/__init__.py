"""
Temporal Localization Module (Phase 6 Bonus)

This module provides experimental multi-event temporal detection/localization
using audio-level annotations as supervision signals.

NOTE: This is a BONUS experimental feature, separate from the main classification pipeline.
It requires RAW WAV FILES to execute. See PHASE_6_BONUS_TEMPORAL_LOCALIZATION.md for details.
It is not part of the KNN-only submission path.

The approach:
1. Uses manifest time annotations as weak labels for events
2. Implements sliding window detector over spectrogram time axis
3. Frame-level classification: event vs. no-event (or per-class)
4. Post-processing to extract temporal bounding boxes
5. Evaluation via IoU (Intersection over Union) metrics
"""

__all__ = ["TemporalDetector", "temporal_evaluation"]
