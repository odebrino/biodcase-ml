# BIODCASE Bioacoustic Classification

This repository contains a cleaned bioacoustic classification project for
BIODCASE-style Antarctic whale event annotations.

## Status Labels

- current primary path: classical, non-convolutional baseline suite
- historical comparison path: CNN event-classification runs under `outputs/runs/`
- experimental: bonus temporal localization code under `src/localization/`
- external-data-dependent: any claim that requires official external counts or sources beyond the bundled snapshot

## What Is Physically In This Repo

Present in this repository snapshot:

- source code under `src/`
- tests under `tests/`
- configs under `configs/`
- cleaned tracker files: `AUDIT_PROGRESS.json` and `CLEANUP_MASTER_TRACKER.md`
- bundled dataset tree `biodcase_development_set/`
- bundled processed manifest `data_manifest.csv`
- canonical manifest-derived reports under `outputs/reports/`
- bundled historical CNN runs under `outputs/runs/`

Not bundled in this repository snapshot:

- executed classical-baseline result directories under `outputs/classical/`
- external official BIODCASE count tables or project-owner reconciliation notes
- APLOSE software itself

Historical cleanup notes are archived under `archive/cleanup_history/` and are
not authoritative.

## Reference Docs

- [DATASET_SPEC.md](DATASET_SPEC.md): bundled split structure, held-out domains,
  manifest-authoritative counts, and external-count limits
- [ANNOTATION_GUIDELINES_SUMMARY.md](ANNOTATION_GUIDELINES_SUMMARY.md):
  class inventory, ambiguity-aware families, and APLOSE-style background
- [ZIP_CONTENTS_AND_LIMITATIONS.md](ZIP_CONTENTS_AND_LIMITATIONS.md): exact
  bundled contents, absent artifacts, historical paths, and reproducibility
  limits
- [RESULT_PROVENANCE.md](RESULT_PROVENANCE.md): provenance expectations for
  bundled result directories
- [HISTORICAL_PATHS.md](HISTORICAL_PATHS.md): optional reference for old run IDs
  and exploratory historical outputs

## Result Status

| result family | status | note |
| --- | --- | --- |
| classical baselines | implemented but not bundled | `src/classical/baselines.py` is present, but `outputs/classical/` is absent in this snapshot |
| historical CNN | historical only | `outputs/runs/20260421-223457/` is bundled, but it is treated as historical comparison evidence with partial provenance |
| temporal localization | implemented but not bundled | experimental code under `src/localization/`; no canonical bundled result package |
| crop verification exports | implemented but not bundled | `src/data/export_crop_verification.py` can generate them on demand |
| manifest-derived reports | bundled | canonical regenerated artifacts live under `outputs/reports/` with provenance metadata |

## Split Semantics

The on-disk directory name `validation` is a legacy name for the official
held-out test split. In this repository:

- `train` means the official training split
- `validation` on disk means the official held-out test split
- internal model-selection splits must be derived from `train` only

Do not read `validation` here as a generic inner-validation split.

## Current Primary Path

The main scientific path is the classical baseline suite:

```bash
python -m src.classical.baselines \
  --config configs/classical_baselines.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/classical
```

This path is implemented and tested. It is not bundled with executed
`outputs/classical/` results in the current snapshot.

Implemented model families:

- logistic regression
- linear SVM
- RBF SVM
- KNN
- Gaussian Naive Bayes
- random forest
- gradient boosted trees
- MLP

Implemented representation families:

- patch
- handcrafted descriptors
- hybrid

The classical driver fits preprocessing on train-derived data only and keeps the
official held-out test split for final evaluation.

## Historical CNN Path

The CNN path remains for historical comparison only.

Bundled historical run directory:

- `outputs/runs/20260421-223457/`

Bundled artifacts in that directory include:

- `best_model.pt`
- `best_metrics.json`
- `history.csv`
- confusion matrices
- per-dataset metrics
- prediction/error reports

This bundled run is historical evidence only. It has useful metrics and config
snapshots, but it does not meet the newer provenance standard used for
canonical packaged results.

This path is not the primary methodological path for compliance with the brief.

## Experimental Bonus Path

The repository also contains an experimental bonus path for temporal
detection/localization:

- code: `src/localization/`
- config: `configs/temporal_localization.yaml`

This path is separate from the main crop-classification workflow and should be
treated as experimental.

## Manifest and Raw Data Reality

The bundled manifest is:

- `data_manifest.csv`

Bundled manifest facts from the current snapshot:

- rows: `76123`
- split counts: `train=58510`, `validation=17613`
- bundled manifest columns include `label_raw` and `label_display`

The canonical report-regeneration entrypoint is:

```bash
.venv/bin/python scripts/regenerate_all_reports.py
```

This rebuilds `data_manifest.csv` from the bundled annotations and audio, then
writes canonical report outputs under `outputs/reports/` with provenance in
`outputs/reports/provenance/report_regeneration_provenance.json`.

The bundled dataset tree also contains:

- `biodcase_development_set/train/annotations/` with 8 CSVs
- `biodcase_development_set/validation/annotations/` with 3 CSVs
- `biodcase_development_set/train/audio/` with 6004 WAV files found in this snapshot
- `biodcase_development_set/validation/audio/` with 587 WAV files found in this snapshot

## Label Semantics

The canonical internal labels are:

- `bma`
- `bmb`
- `bmz`
- `bmd`
- `bp20`
- `bp20plus`
- `bpd`

Alias normalization is implemented at manifest-ingestion time in
`src.data.labels`.

## Crop and Representation Semantics

Two representation semantics are explicitly separated:

- `time_crop_with_frequency_band_mask`: legacy spectrogram tensor with band mask
  and highlight channels
- literal time-frequency crop: crop defined by annotation time and frequency
  coordinates mapped onto explicit spectrogram axes

The current classical feature exporters use the literal time-frequency crop.

## APLOSE / Annotation-Guideline Context

This repository does not implement APLOSE itself.

The guideline context matters scientifically because:

- spectrogram parameters change what is visually and numerically separable
- overlapping fragments and partial events make annotation targets imperfect
- some confusions are plausibly intrinsic ambiguity, not only model defects
- ambiguity-aware interpretation is needed for `ABZ`, `DDswp`, and `20Hz20Plus`

The repo reflects this through:

- APLOSE-inspired spectrogram presets in `configs/aplose_*.yaml`
- grouped-family evaluation in the classical reporting path
- ambiguity-aware markdown summaries for official-test evaluation

## Evaluation Outputs

Historical CNN evaluation outputs are bundled under `outputs/runs/*`.

Canonical manifest-derived reports are bundled under:

- `outputs/reports/manifest/`
- `outputs/reports/quality/`
- `outputs/reports/audit/`
- `outputs/reports/provenance/`

The classical reporting code is implemented to write:

- `official_test_results.csv`
- `official_test_macro_f1_table.csv`
- `official_test_accuracy_table.csv`
- per-model official-test metrics
- grouped-family official-test metrics
- per-dataset official-test metrics
- ambiguity-aware reports

These outputs are implemented but not bundled because `outputs/classical/` has
not been generated in this snapshot.

## What Remains External-Data-Dependent

The following cannot be proven honestly from the repository alone:

- reconciliation between bundled manifest counts and any official external count table
- why bundled train WAV count is `6004` while some prior docs referred to `6007`
- whether any official BIODCASE documentation uses totals different from the current manifest
- annotation campaign discipline claims that require external campaign documentation beyond the bundled files

For those points, the correct status is: requires external data or official
external evidence.

## Final Status

Clean now:

- one authoritative source tree
- one canonical manifest at `data_manifest.csv`
- one canonical manifest-derived report tree at `outputs/reports/`
- explicit held-out-test semantics with leakage-safe selection logic
- truthful result-family labeling for bundled, historical, and absent artifacts

Still external-data-dependent:

- reconciliation against external official BIODCASE count tables
- any claim that depends on project-owner records not bundled here

Optional future work:

- executed classical result bundles under `outputs/classical/`
- broader spectrogram ablations
- experimental temporal-localization result packaging

## Tests

```bash
python -m pytest -q
```

## Main Files

- `src/classical/baselines.py`
- `src/data/build_manifest.py`
- `src/data/representations.py`
- `src/data/export_crop_verification.py`
- `src/training/train.py`
- `src/training/evaluate.py`
- `src/localization/temporal_detector.py`
- `configs/classical_baselines.yaml`
- `IMBALANCE_AUDIT.md`
- `RESULT_PROVENANCE.md`
- `AUDIT_PROGRESS.json`
- `CLEANUP_MASTER_TRACKER.md`
