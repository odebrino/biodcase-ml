# Phase 5: Documentation Reconciliation and Honest Blockers

**Updated**: 2026-04-23  
**Objective**: Make the repository scientifically honest and audit-complete by documenting what is implemented, what is deferred, and what cannot be verified without external evidence.

---

## 1. Method-Compliance Summary

This table maps brief requirements to repository implementation status:

| Brief Requirement | Repository Status | Notes |
|---|---|---|
| **Classical baselines** (logistic regression, SVM, KNN, Naive Bayes, Random Forest, gradient boosting, MLP) | **IMPLEMENTED** | All models available in `src/classical/baselines`. Representation families (patch, handcrafted, hybrid) ready. Classical driver at `src.classical.baselines`. |
| **Non-convolutional primary path** | **IMPLEMENTED** | `configs/classical_baselines.yaml` is now the default. CNN remains only as historical comparison in `outputs/runs/`. |
| **Label normalization** (7 canonical classes with alias handling) | **IMPLEMENTED** | `src.data.labels` normalizes all brief-specified aliases; `label_raw` preserves originals. |
| **APLOSE-inspired spectrogram presets** | **PARTIALLY IMPLEMENTED** | Presets `aplose_512_98.yaml` and `aplose_256_90.yaml` expose nfft/winsize/overlap. **NOTE**: Repository does not implement full APLOSE workflow; these are inspirations only. |
| **Time-frequency crop extraction** | **IMPLEMENTED** | `literal_time_frequency_crop` maps annotation coordinates to spectrogram axes. Legacy mask/highlight representation remains separate. |
| **Feature representation families** | **IMPLEMENTED** | Patch, handcrafted (duration, centroid, bandwidth, rolloff, energy, contrast), and hybrid (concatenated) families exported as npz with feature names. |
| **PCA-based representation** | **IMPLEMENTED** | Optional PCA reduction available for patch and hybrid families. |
| **Merged-category evaluation** (ABZ, DDswp, 20Hz20Plus) | **IMPLEMENTED** | Classical evaluation writes both 7-class and regrouped-family metrics with documented mapping. |
| **Family-level ambiguity-aware reporting** | **IMPLEMENTED** | Official-test reports include grouped-family metrics, per-dataset views, normalized confusion matrices, and per-class recall/F1. |
| **Imbalance audit** | **IMPLEMENTED** | Documented in `IMBALANCE_AUDIT.md`. Imbalance exists by design (domain-generalization split); not a bug. Mitigation: class weights during training, macro F1 as primary metric. |
| **Metric standards** | **IMPLEMENTED** | Macro-F1, balanced accuracy, per-class recall/F1, row-normalized confusion matrix. Accuracy and weighted-F1 preserved for reference. |
| **Bonus: Temporal localization** | **NOT IMPLEMENTED** | Optional; deferred. Repository focuses on event classification, not multi-event temporal bounding. |
| **Bonus: Clustering methods** | **NOT IMPLEMENTED** | Optional; deferred. KMeans, DBSCAN, HDBSCAN, Mean Shift not implemented. |
| **Bonus: Pyramid Match Kernel** | **NOT IMPLEMENTED** | Optional; deferred. Variable-size image strategy not implemented. |

---

## 2. Implementation Details by Phase

### Phase 1 & 2: Data Preparation & Representations
- Label normalization from raw annotations ✓
- Manifest generation with quality audit ✓
- Time-frequency crop extraction ✓
- Spectrogram presets (APLOSE-inspired) ✓
- Feature families: patch, handcrafted, hybrid ✓
- PCA integration ✓

### Phase 3: Classical Baselines
- 8 model families (Logistic, SVM-Linear, SVM-RBF, KNN, Naive Bayes, Random Forest, Gradient Boosting, MLP) ✓
- Representation family combinations ✓
- Cross-validation on train split ✓
- Metrics standardization ✓
- **ARTIFACT MISSING**: `outputs/classical/` directory does not exist; classical baselines have not been run yet.

### Phase 4: Evaluation & Reporting
- Per-class metrics with normalized confusion matrix ✓
- Family-level grouped evaluation ✓
- Per-dataset metrics (casey2017, kerguelen2014, kerguelen2015) ✓
- Confidence analysis and PR curves ✓
- Imbalance audit and leakage checks ✓

### Phase 5: Documentation & Reconciliation (This Phase)
- Honest implementation summary ✓
- Blockers and external dependencies documented ✓
- APLOSE context explanation ✓
- Missing run artifacts identified ✓
- Method-compliance mapping ✓

---

## 3. APLOSE / Annotation Guideline Context

### Why These Guidelines Matter Scientifically

#### Spectrogram Parameter Sensitivity
Bioacoustic event classification depends critically on time-frequency representation parameters:
- **nfft** (window length) controls frequency resolution; smaller values (256) preserve temporal detail but reduce frequency precision
- **window overlap %** affects temporal granularity; 98% overlap creates dense frame sequences but increases data volume
- **Frequency scale** (mel vs. linear) affects perceptual vs. parametric accuracy

The brief's APLOSE recommendations constrain this space to reproducible, ecologically-grounded choices.

#### Annotation Ambiguity & Uncertainty
Whale vocalizations are inherently variable:
- **Frequency bandwidth**: Multiple shallow or deep frequencies may occur together; a single freq box is a simplification
- **Temporal overlap**: Events may fragment or overlap; annotation discipline requires single-annotation-per-event to avoid leakage
- **Spectrogram artifacts**: Aliasing, spectral leakage, and time-frequency trade-offs introduce interpretation uncertainty

Repository strategy: explicit time/frequency coordinate mapping + separate legacy mask representation = transparency about uncertainty.

#### Overlapping Fragments & Single-Annotation Discipline
The BIODCASE dataset applies a **single-annotation-per-event** rule:
- Each row is one event; no multi-fragment rows
- Overlapping events in the recording are separate rows
- This discipline prevents label corruption from multi-class events

**Repository evidence**: All events in `data_manifest.csv` pass `valid_event == True` after quality audit. Single-fragment constraint is verified during manifest generation.

#### Ambiguity-Aware Interpretation
This repository uses:
- **Per-class metrics** (recall, F1, confusion) rather than accuracy alone
- **Confidence thresholds** for downstream decisions
- **Error analysis by family** (ABZ confusions, DDswp confusions, 20Hz20Plus confusions)
- **Macro-F1 as the primary metric** to account for class imbalance and ambiguous boundaries

This acknowledges that some misclassifications may be scientifically reasonable given annotation uncertainty.

---

## 4. Data Manifest Verification

### Manifest Statistics (from `data_manifest.csv`)

| Split | Total Events | bma | bmb | bmd | bmz | bp20 | bp20plus | bpd |
|---|---|---|---|---|---|---|---|---|
| **train** | 58,510 | 18,092 | 4,622 | 13,141 | 1,596 | 10,380 | 5,003 | 5,676 |
| **validation** (official test) | 17,613 | 6,268 | 2,277 | 2,168 | 918 | 2,547 | 2,757 | 678 |
| **Total** | **76,123** | **24,360** | **6,899** | **15,309** | **2,514** | **12,927** | **7,760** | **6,354** |

### Official Counts Mismatch

**Status**: `NEEDS_RAW_DATA_OR_EXTERNAL_EVIDENCE`

The brief likely specifies official dataset totals. Without access to:
- Original annotation CSV files from the campaign
- Authoritative documentation from BIODCASE project leads
- The raw WAV files and their original annotations

**We cannot verify the exact mismatch.**

Current manifest represents events that passed quality audit:
- Events with duration ≥ 0.5 seconds
- Events occurring within audio file boundaries
- No duplicate rows, no zero-duration events

**Next steps to resolve**: Consult raw data or external authoritative sources.

---

## 5. Missing Run Artifacts

### CNN Best Run: `outputs/runs/20260421-223457/`

**Status**: ✓ **PRESENT**

This run (config: `configs/nitro4060_bpd.yaml`) contains:
- `best_model.pt` – best checkpoint (43.8 MB)
- `best_metrics.json` – best-epoch metrics
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `metrics_by_dataset.csv` – per-official-dataset metrics
- `classification_report.csv`, `error_analysis.csv`
- `history.csv` – training curve

Results (official test split):
- Accuracy: 0.9301
- Macro-F1: 0.8866
- F1(bpd): 0.7725
- Macro-F1(casey2017): 0.7642

**NOTE**: This run is CNN-based and thus **not the methodological primary path** (which is now classical baselines). It is preserved as historical comparison.

### Classical Baselines Output: `outputs/classical/`

**Status**: ✗ **ABSENT**

The README documents the classical baseline command:
```bash
python -m src.classical.baselines \
  --config configs/classical_baselines.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/classical
```

This command is ready to run but has **NOT been executed yet**. No timestamp-dated classical result directories exist.

**What this means**:
- Classical baseline code is implemented and tested
- Example outputs are generated in unit tests
- But no production run has been captured in the repository

**Action taken**: Documented in code; updated README to clarify this is the *intended* primary path, not yet the *executed* path.

### Raw Audio Data: `biodcase_development_set/`

**Status**: ✗ **ABSENT** (metadata only)

The directory structure exists:
```
biodcase_development_set/
  train/
    annotations/
    audio/
  validation/
    annotations/
    audio/
```

But **WAV and annotation CSV files are not present in this repository snapshot**. The manifest CSV was generated from them and survives in `data_manifest.csv`, but the originals are external.

---

## 6. Documentation Changes Summary

### What Changed

1. **PHASE_5_RECONCILIATION.md** (this file)
   - Honest method-compliance mapping
   - Blocker identification
   - Data verification notes
   - APLOSE context explanation

2. **README.md** (updated)
   - Clarified that classical baselines are the *intended* primary path, not yet executed
   - Added explicit statement that CNN runs are for historical comparison only
   - Documented what cannot be verified without raw data
   - Added reference to PHASE_5_RECONCILIATION.md

3. **EXTERNAL_DATA_REQUIREMENTS.md** (new)
   - Lists what raw data is needed to verify claims
   - Specifies what evidence would resolve blockers

---

## 7. Blockers & External Dependencies

| Blocker | Category | Resolution |
|---|---|---|
| Official event count verification | `NEEDS_RAW_DATA` | Requires original annotation CSVs or authoritative project documentation |
| Full APLOSE workflow reproduction | `NEEDS_RAW_DATA` | Requires original WAV + annotation; current repo has manifest only |
| Annotation campaign discipline verification | `NEEDS_RAW_DATA` | Requires checking original annotations for multi-fragments or overlaps |
| Classical baselines execution results | `REPOSITORY` | Not yet run; code is ready; user can execute with one command |
| Pyramid Match Kernel implementation | `OPTIONAL` | Deferred as per brief (optional bonus); not a core blocker |
| Clustering methods implementation | `OPTIONAL` | Deferred as per brief (optional bonus); not a core blocker |
| Temporal localization task | `OPTIONAL` | Deferred as per brief (optional bonus); not a core blocker |

---

## 8. Verification Checklist

- [x] Classical baselines code verified in `src.classical.baselines`
- [x] All 8 model families implemented
- [x] Representation families (patch, handcrafted, hybrid) available
- [x] Label normalization for all 7 classes working
- [x] Time-frequency crop extraction working
- [x] APLOSE-inspired presets available
- [x] Imbalance audit completed and documented
- [x] Per-class, per-family, per-dataset metrics implemented
- [x] Best CNN run preserved for historical comparison
- [x] Manifest generation and quality audit working
- [x] No data leakage; cross-validation on train only
- [ ] Classical baselines executed and results captured (ready; not yet run)
- [ ] Official counts matched with external evidence (blocker: needs raw data)
- [ ] Raw audio files preserved (outside scope; expected as external data)

---

## 9. How to Use This Document

**For auditors/reviewers**:
- Use the method-compliance table above as the authoritative status record
- Refer to IMBALANCE_AUDIT.md for class imbalance findings
- See EXTERNAL_DATA_REQUIREMENTS.md for what data would resolve blockers

**For future development**:
- Classical baselines are ready to run; execute `python -m src.classical.baselines ...` to generate results
- To verify official counts: obtain raw annotation CSVs and compare with manifest
- To extend APLOSE implementation: use presets in `configs/aplose_*.yaml` as starting point

**For reproducibility**:
- CNN best run is in `outputs/runs/20260421-223457/` (historical reference only)
- Manifest generation command is fully documented in README.md
- All config files are version-controlled; no hidden hyperparameters

---

## 10. Final Compliance Statement

**This repository honestly implements:**
- ✓ Classical baseline comparison space
- ✓ Non-convolutional primary scientific path (ready; not yet executed)
- ✓ Label normalization, time-frequency crop, representation families
- ✓ APLOSE-inspired spectrogram presets
- ✓ Imbalance audit and per-class reporting
- ✓ Ambiguity-aware evaluation framework
- ✓ Code quality, test coverage, reproducibility

**This repository does NOT claim:**
- Fabricated classical baseline results (code ready; not executed yet)
- Exact precision on official counts (needs external verification)
- Full APLOSE texture/acoustic reproduction (guided by spirit, not literal)
- Optional bonus implementations (clustering, temporal localization, PMK)

**What is deferred to external data or future work:**
- Execution and capture of classical baseline results
- Resolution of official counts via raw data verification
- Full APLOSE compliance verification
- Optional bonus features

---

**End of Phase 5**
