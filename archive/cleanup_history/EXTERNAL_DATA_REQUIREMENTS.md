# External Data Requirements

This document specifies what data or external evidence is needed to **verify or complete** aspects of this repository that cannot be proven from the repository alone.

---

## 1. Official Dataset Counts & Manifest Reconciliation

### What We Have
`data_manifest.csv` contains **76,123 events** structured as follows:

| Split | Training | Official Test (called "validation") |
|---|---|---|
| **Count** | 58,510 | 17,613 |
| **Classes** | 7 (all present) | 6 (bpd absent from validation) |

### What We Need to Verify
- **Official BIODCASE project counts** from the annotation campaign
- **Per-class breakdowns** from authoritative project documentation
- **Explanation of any mismatch** between official totals and manifest totals

### Why This Matters
If the brief or official documentation specifies exact counts that differ from `data_manifest.csv`, we need to:
1. Understand why the manifest differs (events filtered? re-annotations? data quality decisions?)
2. Document the delta explicitly
3. Explain which counts are authoritative for future reference

### How to Resolve
1. Obtain the **original annotation CSV files** from BIODCASE project leads
2. Cross-reference with manifest row counts
3. If differences exist, document in `PHASE_5_RECONCILIATION.md` under "Data Manifest Verification"

### Current Status
**UNRESOLVED** — awaiting external data source

---

## 2. Raw Audio Files & Original Annotations

### What We Have
- **Manifest CSV** containing event metadata and class labels
- **Config files** for processing audio at specified parameters
- **Quality audit report** showing which events passed/failed quality checks

### What We Don't Have
- `.wav` audio files from `biodcase_development_set/train/audio/`
- `.wav` audio files from `biodcase_development_set/validation/audio/`
- Original annotation CSV files (before normalization and quality filtering)

### Why This Matters
Without raw audio and annotations, we cannot independently verify:
- Event duration and time-frequency boundaries
- Quality filtering decisions (why events were marked "invalid")
- Single-fragment discipline (no multi-fragments per row)
- Original label spellings before normalization
- Annotation campaign consistency and precision rules

### How to Resolve
Restore or provide:
1. Directory `biodcase_development_set/train/audio/` with all `.wav` files
2. Directory `biodcase_development_set/validation/audio/` with all `.wav` files
3. Original annotation CSVs (before `src.data.build_manifest` processing)

### Current Status
**NOT AVAILABLE IN REPOSITORY** — expected as external data

---

## 3. APLOSE Compliance & Spectrogram Parameter Justification

### What We Have
- **Presets** `configs/aplose_512_98.yaml` and `configs/aplose_256_90.yaml`
- **Documentation** in README explaining nfft, winsize, overlap parameters
- **Reference to APLOSE principles** in code comments

### What We Don't Have
- **Official APLOSE specification document** (if it exists as external guidance)
- **Detailed acoustic ecology rationale** for why these parameters are optimal for whale calls
- **Peer-reviewed publications** justifying the specific presets

### Why This Matters
The brief emphasizes APLOSE guidelines for spectrogram parameter sensitivity. These presets should not be claimed as "APLOSE-compliant" without evidence that they follow official guidance, not just inspiration.

### How to Resolve
1. Provide the **official APLOSE specification** (if publicly available)
2. Cross-reference our presets against official values
3. Document any deliberate deviations and why they were chosen
4. Update comments in config files with citations or justification

### Current Status
**PARTIALLY RESOLVABLE** — code implements the spirit; external spec could confirm precision

---

## 4. Annotation Campaign Precision Rules

### What We Have
- **`valid_event` flag** in manifest indicating which events passed quality checks
- **Quality audit reports** (`outputs/data_quality_report.csv`, `outputs/data_quality_summary.csv`)
- **Single-fragment discipline** implied by manifest structure

### What We Don't Have
- **Official annotation campaign guidelines** (e.g., "annotators must mark one event per row, no overlaps")
- **Annotator inter-rater reliability scores** (if calibration was done)
- **Original notes or conflict resolution records** from the campaign
- **Precision thresholds** (e.g., "time precision ±0.5s, frequency precision ±1 Hz")

### Why This Matters
To claim scientific integrity, we need to verify:
- Were single-fragment rules enforced consistently?
- Were overlapping events always split into separate rows?
- What precision standard was used for time/frequency boundaries?
- How were ambiguous calls resolved (e.g., borderline species)?

### How to Resolve
1. Obtain **annotation campaign documentation** from BIODCASE project leads
2. Review **annotator guidelines** or training materials used
3. Document any **formal inter-rater agreement studies** that were conducted
4. Update `README.md` with specific precision rules and disciplines enforced

### Current Status
**UNRESOLVED** — awaiting campaign documentation

---

## 5. Biological & Acoustic Semantic Definitions

### What We Have
- **Class labels** (bma, bmb, bmd, bmz, bp20, bp20plus, bpd)
- **Implied distinctions** from feature distributions and confusion matrices
- **References to brief** mentioning Baleen whale, Blue whale groups

### What We Don't Have
- **Scientific species/subspecies mappings** (e.g., which Baleen whale species is "bma"?)
- **Acoustic call type distinctions** (e.g., why separate "bmb" from "bmd"?)
- **Ecological or behavioral context** (e.g., are these calls mating, feeding, migration signals?)
- **Published taxonomies or nomenclature** for these call types

### Why This Matters
For reproducibility and scientific credibility, future researchers need to understand:
- What biological phenomenon does each label represent?
- How were these categories justified (ecological? acoustic? morphological?)
- Are these standard in marine bioacoustics literature?

### How to Resolve
1. Consult **BIODCASE project publications** or white papers
2. Provide **mapping from labels to biological entities** (species, call types, behaviors)
3. Add a reference section to `README.md` with relevant bioacoustics literature
4. Document any **ambiguities or phylogenetic overlaps** (e.g., if two species sound similar)

### Current Status
**DEFERRED TO EXTERNAL DOCUMENTATION** — not within scope of code repository

---

## 6. Best CNN Run Reproducibility

### What We Have
- **Best run checkpoint** at `outputs/runs/20260421-223457/best_model.pt` (43.8 MB)
- **Metrics and confusion matrices** in the same directory
- **Config file** specifying hyperparameters and data processing

### What We Don't Have
- **Random seeds used during training** (for true reproducibility)
- **Full training history** with per-batch loss values (only `history.csv` with epochs)
- **Test set predictions** in easily parsed format (only in metrics)

### Why This Matters
For historical comparison and debugging, we may need to:
- Reproduce this exact run and verify metrics match
- Understand sensitivity to random initialization
- Inspect misclassified examples in detail

### How to Resolve
1. Document the **exact seeds** used (check `outputs/runs/20260421-223457/run_metadata.json`)
2. Include **reproducibility section** in README with seed values
3. Verify run by re-training with same config and seeds

### Current Status
**PARTIALLY AVAILABLE** — run exists; metadata should be checked for full reproducibility

---

## 7. Classical Baselines Execution Status

### What We Have
- **Complete classical baseline code** in `src/classical/baselines.py`
- **Config files** for different representation families
- **Test coverage** showing baselines work correctly
- **Example outputs** from unit tests

### What We Don't Have
- **Executed classical baselines results** in `outputs/classical/` (directory does not exist)
- **Comparison table** showing which representation family + model combo performs best
- **Timing benchmarks** for training/inference on this dataset

### Why This Matters
The brief requires classical baselines as the *primary* evaluation path. Without executed results, we cannot:
- Claim the repository meets the brief's requirement
- Compare classical vs. CNN performance
- Justify why CNN is only historical

### How to Resolve
1. **Execute** `python -m src.classical.baselines --config configs/classical_baselines.yaml --manifest data_manifest.csv --output-dir outputs/classical`
2. Capture results in `outputs/classical/<timestamp>/`
3. Document top-performing combination in `EXPERIMENTS.md`
4. Update README with classical baseline results

### Current Status
**ACTION ITEM** — ready to execute; awaiting user decision

---

## 8. Deferred Optional Features

The brief lists optional bonus features that are **explicitly deferred**:

### Temporal Localization (Multi-Event Bounding)
- **Current status**: Not implemented
- **Why deferred**: Brief indicates this is optional bonus; repository focuses on event classification
- **How to implement**: Extend labels to include (class, temporal_start, temporal_end) tuples; retrain model to output bounding boxes

### Clustering Methods (KMeans, DBSCAN, HDBSCAN, Mean Shift)
- **Current status**: Not implemented
- **Why deferred**: Listed as optional bonus; classical baselines do not require clustering for classification
- **How to implement**: Add clustering driver in `src/analysis/clustering.py`; compare cluster assignments with ground-truth labels; use for data exploration

### Pyramid Match Kernel (Variable-Size Image Strategy)
- **Current status**: Not implemented
- **Why deferred**: Listed as optional bonus; fixed-size representations sufficient for current baselines
- **How to implement**: Implement PMK in `src/models/` as an alternative similarity metric for SVM

### Systematic Ablation Over Spectrogram Parameters
- **Current status**: Presets (aplose_512_98, aplose_256_90) available but not systematically ablated
- **Why deferred**: Listed as optional; full parameter sweep could be expensive
- **How to implement**: Create job list sweeping nfft in {256, 512, 1024}, overlap in {90, 95, 98}, etc.; run representations and classical baselines for each combo

---

## 9. Summary Table

| Item | Have | Need | Urgency |
|---|---|---|---|
| Official dataset counts | ✗ | Official docs / raw annotations | HIGH |
| Raw audio files | ✗ | WAV files from BIODCASE | MEDIUM (external data) |
| Original annotations | ✗ | CSV files before processing | MEDIUM (external data) |
| APLOSE specification | ✗ | Official doc (if public) | MEDIUM |
| Annotation campaign guidelines | ✗ | Campaign docs from leads | MEDIUM |
| Biological definitions | ✗ | BIODCASE publications | LOW (reference doc) |
| CNN run reproducibility | ✓ | Verify seeds; document | LOW |
| Classical baselines results | ✗ | Execute & capture | HIGH (for brief compliance) |
| Clustering methods | ✗ | Optional bonus; defer | LOW |
| Temporal localization | ✗ | Optional bonus; defer | LOW |
| PMK implementation | ✗ | Optional bonus; defer | LOW |

---

## How to Use This Document

1. **For immediate reproducibility**: Focus on item 7 (classical baselines execution)
2. **For scientific audit**: Address items 1-6 with external data/documentation
3. **For optional extensions**: Items 8+ can be tackled in future phases
4. **For future labs**: Use this as a checklist of external resources to gather

---

**Last Updated**: 2026-04-23  
**Author**: Phase 5 Reconciliation  
**Next Review**: After external data is obtained or after classical baselines are executed
