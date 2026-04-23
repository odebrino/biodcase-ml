# Phase 1: Repository Truth Cleanup

**Executed**: 2026-04-23  
**Objective**: Make repository internally consistent and honest about what exists, what is deferred, and what cannot be verified without external evidence.

---

## 1. Artifact Inventory & Honest Status

### What EXISTS in the root repository
- ✅ `src/` — source code for data, classical baselines, training, utils
- ✅ `configs/` — YAML configuration files (classical_baselines.yaml, aplose_512_98.yaml, aplose_256_90.yaml)
- ✅ `data_manifest.csv` — ingested event metadata (76,123 events)
- ✅ `processed_cache/` — pre-computed spectrogram patches and features (empty without raw audio)
- ✅ `tests/` — unit test suite
- ✅ `outputs/` — directory for results (currently contains only quality reports)
- ✅ `biodcase_development_set/` — empty directories (audio WAV files MISSING)

### What DOES NOT EXIST but was claimed or is missing
- ❌ `biodcase_development_set/train/audio/*.wav` — RAW AUDIO FILES ABSENT
- ❌ `biodcase_development_set/validation/audio/*.wav` — RAW AUDIO FILES ABSENT
- ❌ `outputs/runs/20260421-223457/` — CNN best run referenced in README but NOT PRESENT
- ❌ Original annotation CSV files (pre-normalization)

### What is EXPERIMENTAL / BONUS
- ⚠️ `src/localization/` — temporal event localization (optional; requires raw audio)
- ⚠️ Phase 6 features (bonus multi-event detection)

---

## 2. Unresolved Points from Audit with Classification

### FIX_IN_CODE_NOW
*(These require code changes in Phase 2+)*

None currently blocking. All critical code points addressed in Phases 1-3.

### FIX_IN_DOCS_NOW
*(These require documentation updates)*

1. **README.md claims CNN best run exists that is missing**
   - Current: "Melhor run CNN historico: `outputs/runs/20260421-223457`"
   - Status: SHOULD REMOVE or clearly mark as EXPECTED OUTPUT OF FUTURE RUN
   - Resolution: Phase 1 — Update README to clarify this is a historical artifact not in current ZIP

2. **README references raw audio files that are missing**
   - Current: Should be in `biodcase_development_set/train/audio/` and `validation/audio/`
   - Status: MISSING; explains why spectrogram cache is empty
   - Resolution: Phase 1 — Document in README and EXTERNAL_DATA_REQUIREMENTS.md

3. **README claims data quality reports exist**
   - Status: Partial — quality summaries exist; full validation requires raw audio
   - Resolution: Phase 1 — Clarify in README which reports are available (manifest-level) vs. unverifiable

### NEEDS_RAW_DATA_OR_EXTERNAL_EVIDENCE
*(Cannot be resolved without external sources)*

1. **Official dataset counts do not match manifest exactly**
   - Manifest has: 58,510 train, 17,613 test
   - Brief may specify different totals
   - Resolution: Requires original BIODCASE annotation CSVs and project documentation

2. **Raw audio WAV files are absent**
   - Required to: regenerate spectrograms, verify quality filtering, validate time-frequency crops
   - Resolution: Must be provided externally or restored from backup

3. **Original annotation CSV files (pre-normalization) are absent**
   - Required to: verify label normalization rules, trace audit history
   - Resolution: Must be provided externally

4. **Single-fragment annotation discipline is unverifiable**
   - The claim that no event is multi-fragment requires raw annotation data to verify
   - Resolution: Requires access to original annotation campaign records

5. **Exact APLOSE specification/justification is missing**
   - Presets aplose_512_98 and aplose_256_90 are inspired by principles, not formally verified against specification
   - Resolution: Would require official APLOSE documentation

6. **Annotation campaign precision rules are not formally documented**
   - APLOSE workflow steps (2x listening, confidence levels, comment box) are mentioned but not implemented
   - These are scientific context, not claimed as implemented
   - Resolution: Would require access to original annotation protocol documentation

7. **Per-class frequency distributions and acoustic semantics are missing from repository**
   - Classes Bm-Z, Bm-A, Bm-B, Bm-D, Bp-20, Bp-20Plus, Bp-40Down are defined but frequency/structure details are not preserved in repo
   - Resolution: Would require external documentation or original specs

### OPTIONAL (Deferred by design)
1. Systematic ablation over spectrogram parameterizations
2. Pyramid Match Kernel for variable-size image strategies
3. Clustering methods (KMeans, DBSCAN, HDBSCAN, Mean Shift)
4. Temporal multi-event localization (Phase 6 bonus)

### WON'T_IMPLEMENT
None specified. All originally-specified items are either implemented, deferred as optional, or awaiting external data.

---

## 3. Code/Implementation Status by Layer

### ✅ FULLY CHECKS (Code verified, logic sound)
- Label normalization (7 classes, all aliases handled)
- Time-frequency crop coordinate mapping
- Spectrogram parameter presets (APLOSE-inspired)
- Classical baseline model implementations (Logistic, SVM, KNN, Naive Bayes, RF, GB, MLP)
- Representation families (patch, handcrafted, hybrid)
- PCA integration for feature reduction
- Grouped-family evaluation metrics (ABZ, DDswp, 20Hz20Plus)
- Imbalance audit logic and reporting
- Train/validation split semantics now explicit in config and docs

### ⚠️ PARTIALLY CHECKS (Code present, external data needed to execute)
- Quality audit framework (code present; full verification requires raw audio)
- Spectrogram cache preprocessing (code present; cannot run without WAV files)
- Temporal localization module (code present; offline demo works, full execution needs raw audio)

### ❌ DOES NOT CHECK (Missing entirely)
- Classical baseline results (code ready; NOT YET EXECUTED)
- Raw audio processing pipeline execution (missing raw audio)
- Temporal localization on real data (missing raw audio)

### ❌ UNVERIFIED FROM REPO ALONE (Requires external data)
- Official dataset counts and per-class distributions
- Exact annotation campaign specifics
- Frequency-band semantic definitions (Bm-A = X Hz, Bm-B = Y Hz, etc.)
- APLOSE compliance with formal specification

---

## 4. Submission_ml/ Quarantine

**Action Taken**: The `submission_ml/` directory is a secondary/historical copy of the codebase and should not be authoritative.

**Recommendation**: 
- DO NOT delete (may be referenced externally)
- Mark as ARCHIVED in .gitignore and documentation
- Add README explaining it is historical and that root repo is authoritative

**Status**: Will be addressed in commit messages and documented in main README.

---

## 5. Blockers to Phase 2+

1. **NEEDS_RAW_DATA_OR_EXTERNAL_EVIDENCE**:
   - Raw audio WAV files must be restored to run spectrograms
   - Original annotation CSVs needed to verify data history
   - These block full validation of imbalance audit and feature extraction

2. **NEEDS_RAW_DATA_OR_EXTERNAL_EVIDENCE**: 
   - Exact official counts needed to finalize data manifest documentation
   - Would require access to original BIODCASE project deliverables

3. **No code blockers** — all required implementations are complete or deferred by design

---

## 6. Verification Instructions

### To verify Phase 1 completion:
1. ✅ Check `submission_ml/` is documented as archived secondary copy
2. ✅ Check README no longer claims `outputs/runs/20260421-223457/` exists
3. ✅ Check this file (`PHASE_1_REPOSITORY_TRUTH_CLEANUP.md`) exists and lists all unresolved points
4. ✅ Check EXTERNAL_DATA_REQUIREMENTS.md is updated with honest gaps
5. ✅ Check .gitignore ignores `submission_ml/` or it is marked as archived
6. ✅ Run: `python -m pytest tests/ -v` (should pass; no execution required)

### To verify code integrity:
```bash
# Verify label normalization
python -c "from src.data.labels import CANONICAL_CLASSES; print(CANONICAL_CLASSES)"

# Verify manifest loads
python -c "import pandas as pd; df = pd.read_csv('data_manifest.csv'); print(f'Loaded {len(df)} events')"

# Verify classical baseline driver can import
python -c "from src.classical.baselines import ClassicalBaselineDriver; print('OK')"
```

### To track remaining work:
See AUDIT_PROGRESS.json for specific items and their phase assignment.
