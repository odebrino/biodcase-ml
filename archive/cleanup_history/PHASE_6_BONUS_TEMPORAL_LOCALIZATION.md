# Phase 6: Optional Bonus — Temporal Event Detection & Localization

**Status**: EXPERIMENTAL / BONUS  
**Updated**: 2026-04-23  
**Scope**: Separate from main classification pipeline  
**Data requirement**: RAW WAV FILES (external) for real execution  

---

## Overview

This phase adds an **optional bonus experimental feature** for multi-event temporal localization: given an audio recording, detect *when* and *where* in time each bioacoustic event occurs.

**Not part of the core task.** The main pipeline (classical baselines) is for classifying pre-extracted crop events. This feature is for audio-level detection/boundary localization.

---

## Design Principles

### 1. Keep Separate from Main Pipeline
- Core code in new module: `src/localization/`
- Separate config: `configs/temporal_localization.yaml`
- Does NOT modify `src.classical.baselines` or `src.training.train`
- Independent evaluation metrics (IoU-based, not crop-level metrics)

### 2. Use Manifest Annotations as Supervision
- Time boundaries from `data_manifest.csv`: `start_seconds`, `end_seconds`
- Frequency boundaries: `low_frequency`, `high_frequency` (for reference)
- Class labels: `label` (7 canonical classes)
- These annotations become **weak labels** for frame-level training

### 3. Sliding Window Detector
- Input: Spectrogram (or extracted features) over time
- Frame-level classification: 
  - **Binary mode**: event vs. no-event
  - **Multi-class mode**: {no-event, bma, bmb, bmd, bmz, bp20, bp20plus, bpd}
- Output: Temporal bounding boxes `[start_time, end_time, class, confidence]`

### 4. Honest Limitations
- ✗ WAV files NOT in repository → cannot run real inference
- ✓ Framework IS implemented and tested
- ✓ Can work in **offline mode** using manifest labels as both training and evaluation supervision
- ⚠️ Real-world performance unknown (no labeled test set outside manifest)

---

## Architecture

### Module Structure

```
src/localization/
├── __init__.py                      # Module exports
├── temporal_detector.py              # Frame-level detector
└── temporal_evaluation.py            # IoU metrics & evaluation
```

### Key Classes

#### `TemporalDetector`
**Framework for multi-event temporal detection.**

```python
from src.localization.temporal_detector import TemporalDetector

detector = TemporalDetector(
    frame_duration_ms=512.0,
    min_event_duration_sec=0.5,
    confidence_threshold=0.5,
    use_multiclass=True,
    classes=["bma", "bmb", "bmd", "bmz", "bp20", "bp20plus", "bpd"]
)
```

**Key Methods:**
- `create_frame_labels_from_manifest()` → Convert time boxes to frame-level binary/multi-class labels
- `post_process_frame_predictions()` → Extract boxes from frame predictions via connected components
- `detect_on_manifest_row()` → Offline demonstration mode (uses manifest as both labels and predictions)
- `create_dummy_predictions()` → Generate synthetic predictions for testing (without real model)

#### `IoUMetrics` (Evaluation)
**Temporal event localization quality metrics.**

```python
from src.localization.temporal_evaluation import (
    compute_iou,
    match_detections_to_ground_truth,
    evaluate_temporal_detection,
    generate_temporal_detection_report,
)
```

**Metrics computed:**
- **Intersection over Union (IoU)**: Overlap between predicted and ground-truth boxes
- **Precision**: TP / (TP + FP) at IoU threshold
- **Recall**: TP / (TP + FN) at IoU threshold
- **F1**: Harmonic mean of precision and recall
- **Mean IoU**: Average IoU across all matched detections

**Multi-threshold evaluation:**
- IoU thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95]
- Per-threshold: TP, FP, FN, Precision, Recall, F1

---

## How It Works

### Approach Overview

```
Manifest Annotations
    ↓
Frame-Level Labels (event/no-event or per-class)
    ↓
Sliding Window Features (from spectrogram or pre-extracted)
    ↓
Frame-Level Classifier (binary or multi-class)
    ↓
Frame Predictions + Confidences
    ↓
Post-Processing (connected components, min duration filter)
    ↓
Temporal Event Boxes [start, end, class, conf]
    ↓
IoU Matching to Ground Truth
    ↓
Metrics: Precision, Recall, F1, Mean IoU
```

### Step 1: Convert Annotations to Frame Labels

**Input**: Manifest CSV with `start_seconds`, `end_seconds` for each event  
**Output**: Frame-level binary or multi-class labels

```python
import pandas as pd
from src.localization.temporal_detector import TemporalDetector

manifest = pd.read_csv('data_manifest.csv')

detector = TemporalDetector(
    classes=["bma", "bmb", "bmd", "bmz", "bp20", "bp20plus", "bpd"],
    use_multiclass=True,
)

# For one audio file:
audio_events = manifest[manifest['audio_path'] == 'some/audio.wav']
audio_duration = audio_events.iloc[0]['audio_duration_seconds']

frame_labels, frame_class_labels, events = detector.create_frame_labels_from_manifest(
    audio_events,
    audio_duration,
    fps=4.0  # 4 frames per second = 250ms per frame
)

# frame_labels: (n_frames,) binary mask where 1 = event, 0 = silence
# frame_class_labels: (n_frames,) multi-class indices
# events: list of TemporalEvent objects
```

### Step 2: Frame-Level Classification

**In practice**: Apply a model (CNN, classifier) to spectrogram windows (or pre-extracted features)  
**For now**: Use dummy predictions or manifest labels themselves for testing

```python
# Offline demonstration: predictions = labels + noise
frame_predictions, frame_confidences = detector.create_dummy_predictions(frame_labels)

# For real inference: train a model on (spectrogram_patches, frame_labels)
# then call model.predict() on the spectrogram
```

### Step 3: Post-Process to Event Boxes

Extract connected regions of predicted events and convert to temporal coordinates.

```python
detected_events = detector.post_process_frame_predictions(
    frame_predictions=frame_predictions,
    frame_confidences=frame_confidences,
    fps=4.0  # must match the fps used when creating frame labels
)

# detected_events: List of TemporalEvent
# Each event: start_time, end_time, label, confidence, source='detected'
```

### Step 4: Evaluate via IoU Matching

Compare detected events to ground-truth (manifest) events.

```python
from src.localization.temporal_evaluation import (
    match_detections_to_ground_truth,
    evaluate_temporal_detection
)

# Single file:
tp, fp, fn, mean_iou = match_detections_to_ground_truth(
    detected_events,
    ground_truth_events,
    iou_threshold=0.5
)

# Multiple files / thresholds:
detection_results = [...]  # List of DetectionResult objects
metrics_by_threshold = evaluate_temporal_detection(
    detection_results,
    iou_thresholds=[0.5, 0.75, 0.9]
)

for thresh, metrics in metrics_by_threshold.items():
    print(f"IoU≥{thresh}: Precision={metrics.precision:.3f}, "
          f"Recall={metrics.recall:.3f}, F1={metrics.f1:.3f}")
```

---

## Current Status

### ✅ Implemented

- [x] `TemporalDetector` class with frame-level labeling
- [x] Post-processing to extract temporal boxes (connected components)
- [x] IoU-based evaluation metrics
- [x] Multi-threshold evaluation framework
- [x] Dummy prediction generation for testing
- [x] Offline evaluation using manifest annotations
- [x] JSON report generation

### ❌ NOT Implemented (Requires External Data)

- [ ] **Real model inference** — Needs trained detector model checkpoint
- [ ] **Spectrogram extraction at runtime** — Needs WAV audio files
- [ ] **Cross-validation on frame-level task** — Needs to split audio into train/val
- [ ] **Temporal data augmentation** — Would boost generalization
- [ ] **Multi-scale frame rates** — Could improve small event detection

### ⚠️ Honest Limitations

| Issue | Why | Resolution |
|---|---|---|
| No WAV files in repo | Outside repository scope; external data | User provides raw audio + manifest |
| Framework only, no model weights | Not trained; out of scope for Phase 6 | User trains detector or provides checkpoint |
| Offline evaluation only | No real inference → no true test performance | Evaluate on provided audio via framework |
| No cross-validation within audio | Temporal splitting is complex; out of scope | Future: implement temporal train/val splits |
| Frame rate is fixed (4 fps) | Simplifies API; may miss very short events | Config allows customization |

---

## Running the Detector (Demonstration)

### Offline Mode: Using Manifest as Labels

```bash
python -c "
import pandas as pd
from src.localization.temporal_detector import TemporalDetector
from src.localization.temporal_evaluation import evaluate_temporal_detection

# Load manifest
manifest = pd.read_csv('data_manifest.csv')

# Initialize detector
detector = TemporalDetector(
    classes=['bma', 'bmb', 'bmd', 'bmz', 'bp20', 'bp20plus', 'bpd'],
    use_multiclass=True,
    min_event_duration_sec=0.5,
    confidence_threshold=0.5,
)

# Process each unique audio file (demonstration on first 10)
detection_results = []
for audio_path in manifest['audio_path'].unique()[:10]:
    audio_events = manifest[manifest['audio_path'] == audio_path]
    if len(audio_events) == 0:
        continue
    
    # Get unique audio metadata from first row
    audio_row = audio_events.iloc[0]
    
    # Demonstration: detect on this audio (offline mode using manifest)
    result = detector.detect_on_manifest_row(audio_row, audio_events)
    detection_results.append(result)
    
    print(f'Processed {audio_path}: {len(result.ground_truth_events)} GT, '
          f'{len(result.detected_events)} Detections')

# Evaluate
if detection_results:
    metrics = evaluate_temporal_detection(detection_results, iou_thresholds=[0.5, 0.75, 0.9])
    for thresh, m in metrics.items():
        print(f'IoU≥{thresh:.2f}: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}')
"
```

### Real Mode: With Actual WAV Files (When Available)

```python
# Pseudo-code for future real usage:
from src.data.representations import build_spectrogram_frames
from src.localization.temporal_detector import TemporalDetector

# 1. Extract frames from actual WAV file
frames = build_spectrogram_frames(
    audio_path='biodcase_development_set/train/audio/.../example.wav',
    config=config,  # spectrogram parameters
    fps=4.0,
)

# 2. Run inference on frames (assuming trainer model available)
model.eval()
frame_predictions, frame_confidences = model.predict(frames)

# 3. Post-process to boxes
detector = TemporalDetector(...)
detected_events = detector.post_process_frame_predictions(
    frame_predictions,
    frame_confidences,
    fps=4.0,
)

# 4. Evaluate against manifest annotations
result = DetectionResult(
    audio_path='...',
    duration_seconds=duration,
    ground_truth_events=gt_events,
    detected_events=detected_events,
    frame_labels=...,
    frame_predictions=...,
    frame_confidences=...,
)
```

---

## Configuration

See `configs/temporal_localization.yaml`:

```yaml
temporal_detector:
  frame_duration_ms: 512.0       # Spectrogram frame size
  fps: 4.0                        # Frames per second
  min_event_duration_sec: 0.5     # Filter very short detections
  confidence_threshold: 0.5       # Post-processing threshold
  use_multiclass: true
  classes: [bma, bmb, bmd, bmz, bp20, bp20plus, bpd]

evaluation:
  iou_thresholds: [0.5, 0.75, 0.9]
  report_path: "temporal_detection_report.json"

data:
  manifest_path: "data_manifest.csv"
  output_dir: "outputs/temporal_localization"
  splits: [train, validation]
```

---

## Tests & Verification

### Unit Tests

Test frame labeling, IoU computation, connected component extraction:

```bash
python -m pytest tests/test_temporal_localization.py -v
```

**Tested:**
- ✓ Frame label creation from time boxes
- ✓ IoU computation (identity, perfect overlap, no overlap, partial overlap cases)
- ✓ Greedy matching algorithm
- ✓ Post-processing (connected components, minimum duration)
- ✓ Edge cases (empty predictions, zero duration, etc.)

### Integration Test

End-to-end on manifest (offline mode):

```bash
python src/localization/__init__.py  # or import and test interactively
```

Current status: **WORKING** (offline/demonstration mode)  
Real execution status: **BLOCKED** (requires WAV files)

---

## Data Requirements for Full Implementation

### What's Available
- ✓ Time annotations (start/end in seconds)
- ✓ Class labels
- ✓ Audio file paths
- ✓ Duration metadata

### What's Missing
- ✗ **Raw WAV audio files** — Critical for spectrogram extraction
- ✗ **Trained detector model** — Would need to train on labeled frame sequences
- ✗ **Separate temporal train/val splits** — Currently uses manifest splits (train/validation are class labels, not temporal splits)
- ✗ **Multi-audio compilation** — Some detectors work on full hour-long recordings; we have individual event crops

### To Enable Real Execution

**Provide:**
1. Raw WAV files under `biodcase_development_set/train/audio/` and `biodcase_development_set/validation/audio/`
2. Config for spectrogram extraction matching our presets
3. Optional: pre-trained detector model checkpoint

**Then run:**
```bash
python -c "
import yaml
import pandas as pd
from src.localization.temporal_detector import TemporalDetector

with open('configs/temporal_localization.yaml') as f:
    config = yaml.safe_load(f)

manifest = pd.read_csv(config['data']['manifest_path'])
detector = TemporalDetector(**config['temporal_detector'])

# Process all audio files (now with real VAW files)
detection_results = []
for audio_path in manifest['audio_path'].unique():
    audio_events = manifest[manifest['audio_path'] == audio_path]
    # [... real spectrogram extraction + inference ...]
    
# Evaluate
from src.localization.temporal_evaluation import generate_temporal_detection_report
report = generate_temporal_detection_report(
    detection_results,
    output_path=config['evaluation']['report_path']
)
"
```

---

## Relationship to Main Pipeline

| Property | Main Task (Classification) | Bonus Task (Localization) |
|---|---|---|
| **Input** | Pre-extracted crop events | Full audio recording |
| **Output** | Class label for crop | Temporal event boundaries for entire audio |
| **Labels** | Event-level classes | Frame-level labels (event vs. silence) |
| **Evaluation** | Per-crop accuracy, macro-F1 | IoU, Precision/Recall at IoU thresholds |
| **Data dependency** | Manifest only | Manifest + WAV files |
| **Production status** | Ready (classical baselines code done) | Experimental/Bonus |
| **Required for brief** | YES | NO (optional) |

**They do NOT interfere** because:
- Different modules (`src.classical` vs. `src.localization`)
- Different configs
- Different evaluation metrics
- Different data requirements
- Separate feature extraction pipelines

---

## Future Extensions

If WAV files become available:

1. **Train a real frame-level detector**
   - Spectrogram → Conv layers → frame predictions
   - Loss: binary cross-entropy (or multi-class) per frame
   - Data augmentation: SpecAugment, time shifting

2. **Temporal data splits**
   - Cannot use manifest splits (they're class labels)
   - Implement: split by date ranges or audio file IDs
   - Ensures no temporal leakage

3. **Multi-scale detector**
   - Different frame rates for different event durations
   - Ensemble: combine predictions from 2, 4, 8 fps

4. **End-to-end evaluation**
   - Run detector on full recordings
   - Compare with manual boundary annotations (if available)
   - Compute false alarm rates, missed detection rates

5. **Frequency-domain localization**
   - Extend to 2D boxes: (time_start, time_end, freq_low, freq_high)
   - Use low_frequency, high_frequency from manifest
   - Evaluate 2D IoU instead of 1D

---

## Documentation & Honesty

### What We Claim
- ✓ Framework for frame-level temporal detection is implemented
- ✓ IoU-based evaluation is correct
- ✓ Offline demonstration mode works

### What We DON'T Claim
- ✗ Real detection accuracy (no test results, no model trained)
- ✗ COCO-style detection metrics (we simplified to IoU only)
- ✗ Temporal data leakage prevention (we use manifest splits, not temporal splits)
- ✗ Production readiness (still experimental, bonus feature)

### Blockers & External Dependencies

| Item | Status | Why |
|---|---|---|
| WAV files | ✗ MISSING | Outside repository scope |
| Trained model | ✗ MISSING | Too expensive to train/store |
| TensorFlow/PyTorch integration | ✗ NOT IMPLEMENTED | Framework-agnostic design for now |
| Temporal data augmentation | ✗ NOT IMPLEMENTED | Low priority (bonus feature) |
| Multi-modal fusion (freq+time) | ✗ NOT IMPLEMENTED | Could be future extension |

---

## Summary

**Phase 6 provides a clean, separate experimental framework for multi-event temporal localization.**

- Framework: ✓ Complete
- Code quality: ✓ Tested (unit tests)
- Integration: ✓ Does NOT break main task
- Honesty: ✓ Clear about limitations
- Optional status: ✓ Clearly marked

**Real execution requires external data (WAV files) and model training, which are out of scope but feasible with the provided framework.**

---

**End of Phase 6**
