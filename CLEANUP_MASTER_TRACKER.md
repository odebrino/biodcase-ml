# Cleanup Master Tracker

## Cleanup Principles

- root repo is authoritative
- every artifact must be classified as bundled / implemented but not bundled / historical / requires external data / removed
- one source of truth per topic
- docs must never overstate what is physically present
- official held-out test must never be silently reused for model selection

| issue_id | severity | title | repository_evidence | resolution_strategy | external_data_needed | files_to_touch | commands_to_run | validation_check | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLN-001 | high | stale duplicate subtree if present | `submission_ml/` duplicated root modules, configs, tests, and docs before cleanup | remove `submission_ml/` and keep root as the only authoritative tree | no | `submission_ml/`, `README.md`, `CLEANUP_MASTER_TRACKER.md` | `find . -maxdepth 3 -type d | sort`, `rg -n "submission_ml" .` | duplicate subtree either removed or explicitly archived with clear status | fixed | none |
| CLN-002 | high | phantom result artifact references in docs | docs previously mixed real and missing result artifacts | classify each result family as bundled / implemented but not bundled / historical / external-data-dependent and add provenance expectations | no | `README.md`, `EXPERIMENTS.md`, `RESULT_PROVENANCE.md`, `CLEANUP_MASTER_TRACKER.md` | `rg -n "outputs/runs|outputs/classical|best run|checkpoint" README.md EXPERIMENTS.md RESULT_PROVENANCE.md` | no doc claims a missing artifact is bundled | fixed | none |
| CLN-003 | high | absent `outputs/classical/` while docs imply bundled results | `outputs/classical/` is still absent in the repository snapshot | downgrade classical-result claims and record the path as implemented but not bundled | no | `README.md`, `EXPERIMENTS.md`, `RESULT_PROVENANCE.md`, `CLEANUP_MASTER_TRACKER.md` | `test ! -e outputs/classical`, `rg -n "outputs/classical" README.md EXPERIMENTS.md RESULT_PROVENANCE.md` | docs match physically bundled classical artifacts | fixed | none |
| CLN-004 | critical | legacy `validation` naming used for official held-out test | on-disk data still uses `validation/`, but repository semantics are now explicit | standardize semantics in docs and code while keeping compatibility aliases only where needed | no | `README.md`, `configs/*.yaml`, `src/`, notebook | `rg -n "official held-out test|validation denotes the official held-out test" README.md DATASET_SPEC.md ZIP_CONTENTS_AND_LIMITATIONS.md src tests` | official held-out test semantics are explicit everywhere that matters for packaging | fixed | none |
| CLN-005 | critical | possible selection leakage in historical CNN flow | historical CNN training previously fell back from missing selection split to held-out `validation` | harden historical code paths and docs so official test cannot be silently used for selection | no | `src/training/train.py`, `src/training/evaluate.py`, `README.md`, `CLEANUP_MASTER_TRACKER.md` | `rg -n "selection_split|val_split|legacy_official_test_used_for_selection" src/training README.md` | no silent fallback from missing inner validation to official test | fixed | none |
| CLN-006 | high | stale helper defaults to missing run IDs | helper defaults no longer point to hard-coded missing runs | discover existing run reports dynamically and keep hard-coded run IDs out of helper defaults | no | `src/analysis/inspect_errors.py`, `src/pipeline.py`, `README.md`, `RESULT_PROVENANCE.md` | `rg -n "20260421-213619" src README.md RESULT_PROVENANCE.md`, `rg -n "discover_default_report|default=None" src/analysis/inspect_errors.py src/pipeline.py` | no helper defaults point at stale or misleading run IDs | fixed | none |
| CLN-007 | high | manifest/schema drift around `label_raw` | bundled manifest originally lacked `label_raw`, while current code and docs expected it | regenerate canonical manifest from bundled raw data and make the regenerated snapshot authoritative | no | `data_manifest.csv`, `src/data/build_manifest.py`, `README.md`, `scripts/regenerate_all_reports.py` | `.venv/bin/python scripts/regenerate_all_reports.py`, `python - <<'PY' ... list(m.columns) ... PY` | bundled manifest schema matches current code expectations | fixed | none |
| CLN-008 | medium | duplicate root-level generated CSVs | root had conflicting generated count CSVs and root-level report duplicates | move canonical generated reports to `outputs/reports/` and delete conflicting root-level copies | no | root CSVs, `outputs/`, reporting scripts, docs | `.venv/bin/python scripts/regenerate_all_reports.py`, `find outputs/reports -maxdepth 3 -type f | sort` | one canonical source of truth for manifest-derived reports | fixed | none |
| CLN-009 | medium | inconsistent docs around class counts or validation-set claims | bundled docs are now reconciled to repository evidence, but exact external official-count reconciliation is still unavailable | document repository-authoritative counts and mark unresolved external mismatches explicitly | yes | `README.md`, `AUDIT_PROGRESS.json`, `DATASET_SPEC.md`, `ZIP_CONTENTS_AND_LIMITATIONS.md` | `rg -n "6007|official external count table|requires external data" README.md DATASET_SPEC.md ZIP_CONTENTS_AND_LIMITATIONS.md AUDIT_PROGRESS.json` | docs do not overclaim unverifiable counts or split semantics | blocked by missing external data | official external count tables or owner reconciliation are not bundled |
| CLN-010 | medium | notebook or JSON audit sidecars may contain stale state | notebook outputs and root audit sidecars were stale before cleanup | clear contradictory notebook outputs, regenerate canonical reports, and keep only truthful sidecars | no | `pipeline_walkthrough.ipynb`, `AUDIT_PROGRESS.json`, `outputs/reports/*`, `IMBALANCE_AUDIT.md` | `rg -n "20260421-213619|outputs/data_quality_report.csv|outputs/imbalance_audit_summary" pipeline_walkthrough.ipynb IMBALANCE_AUDIT.md outputs/reports -g '*.json' -g '*.md'` | sidecars do not contradict repo contents or are clearly marked historical | fixed | none |

## scan_hits

| file | matched_terms | classification | note |
| --- | --- | --- | --- |
| `README.md` | `outputs/runs/`, `outputs/classical/`, `20260421-223457`, `label_raw`, `validation`, `selection_split`, `historical`, `bundled`, `current` | stale doc claim | mixes truthful bundled CNN references with stronger-than-evidence classical claims and legacy validation wording |
| `AUDIT_PROGRESS.json` | `label_raw`, `validation`, `outputs/runs/20260421-223457` | stale artifact dependency | still asserts the best run is absent even though it is bundled |
| `EXPERIMENTS.md` | `outputs/runs/`, `20260421-223457` | stale doc claim | still presents CNN historical run as current experiment reference |
| `submission_ml/README.md` | `outputs/runs/`, `20260421-223457`, `validation` | stale doc claim | duplicate subtree carries stale duplicated docs |
| `outputs/imbalance_audit_summary.json` | `outputs/runs/20260421-223457`, `validation` | stale generated report | generated report reflects legacy split wording and old run selection |
| `outputs/imbalance_audit_summary.md` | `outputs/runs/20260421-223457`, `validation` | stale generated report | same as JSON companion |
| `submission_ml/EXPERIMENTS.md` | `outputs/runs/`, `20260421-223457` | stale doc claim | duplicate subtree experiment log |
| `submission_ml/configs/baseline.yaml` | `validation` | compatibility alias | duplicate subtree config mirrors old naming |
| `submission_ml/src/analysis/inspect_errors.py` | `validation`, `20260421-213619` | dangerous code default | duplicate subtree helper defaults to historical run ID and legacy split |
| `tests/test_representations.py` | `label_raw` | test fixture | expected schema exercise |
| `tests/test_classical_baselines.py` | `label_raw`, `validation`, `validation_fraction` | test fixture | expected inner-validation fixture only |
| `PHASE_1_REPOSITORY_TRUTH_CLEANUP.md` | `validation`, `outputs/runs/20260421-223457`, `submission_ml` | stale doc claim | earlier cleanup note contradicts current bundled repo state |
| `submission_ml/src/analysis/imbalance_audit.py` | `validation`, `outputs/runs/`, `eval_split` | dangerous code default | duplicate subtree keeps old semantics and chooser defaults |
| `IMBALANCE_AUDIT.md` | `outputs/runs/20260421-223457`, `validation` | harmless reference | bundled report points to existing run but still uses legacy split term |
| `configs/classical_baselines.yaml` | `validation_fraction` | harmless reference | internal train-only selection parameter |
| `EXTERNAL_DATA_REQUIREMENTS.md` | `validation`, `outputs/runs/20260421-223457`, `outputs/classical/` | stale doc claim | claims `bpd` absent from validation and asks for external best-run artifact even though bundled |
| `src/analysis/inspect_errors.py` | `validation`, `20260421-213619` | dangerous code default | root helper hard-codes legacy split and historical run ID |
| `configs/baseline.yaml` | `validation`, `selection_split` | compatibility alias | legitimate config compatibility, but semantics need clearer hardening |
| `src/pipeline.py` | `validation`, `20260421-213619` | dangerous code default | helper default report path points to specific historical run |
| `src/analysis/imbalance_audit.py` | `validation`, `outputs/runs/`, `eval_split` | stale artifact dependency | chooser defaults and wording remain tied to legacy run/report outputs |
| `PHASE_5_RECONCILIATION.md` | `outputs/classical/`, `outputs/runs/20260421-223457`, `validation` | stale doc claim | claims classical outputs missing and raw data absent while raw tree and CNN run are bundled |
| `src/classical/baselines.py` | `label_raw`, `validation`, `inner_selection_split` | harmless reference | train-only classical selection logic is explicit and acceptable |
| `src/training/train.py` | `validation`, `selection_split`, `legacy_official_test_used_for_selection` | dangerous code default | explicit leakage-warning path still exists if no inner split is set |
| `src/training/evaluate.py` | `validation`, `selection_split` | compatibility alias | evaluation semantics explicit but still tied to legacy naming |
| `src/data/build_manifest.py` | `label_raw`, `validation` | harmless reference | canonical source for current schema and raw-label capture |
| `src/data/dataset.py` | `label_raw` | harmless reference | runtime backfill for older manifests |
| `src/data/export_crop_verification.py` | `label_raw` | harmless reference | metadata passthrough only |
| `pipeline_walkthrough.ipynb` | `outputs/runs/`, `20260421-223457`, `20260421-213619`, `validation`, `current` | notebook side effect | notebook hardcodes historical runs, split names, and output summaries |
| `configs/temporal_localization.yaml` | `validation` | compatibility alias | bonus path still tied to on-disk split naming |

## artifact_inventory

### source docs
- `README.md`
- `EXPERIMENTS.md`
- `IMBALANCE_AUDIT.md`
- `PHASE_1_REPOSITORY_TRUTH_CLEANUP.md`
- `PHASE_5_RECONCILIATION.md`
- `PHASE_6_BONUS_TEMPORAL_LOCALIZATION.md`
- `EXTERNAL_DATA_REQUIREMENTS.md`
- `codex_prompts_biodcase_cleanup/*.md`

### generated reports
- `outputs/data_quality_report.csv`
- `outputs/data_quality_summary.csv`
- `outputs/class_distribution_by_split.csv`
- `outputs/imbalance_audit_summary.json`
- `outputs/imbalance_audit_summary.md`
- root split-distribution CSVs
- `outputs/runs/*/*` metrics, reports, confusion matrices, predictions, and configs

### stale generated reports
- `outputs/imbalance_audit_summary.json`
- `outputs/imbalance_audit_summary.md`
- `val_class_distribution.csv`

### test data
- `tests/*`
- `biodcase_development_set/train/annotations/*`
- `biodcase_development_set/validation/annotations/*`

### notebook outputs
- `pipeline_walkthrough.ipynb`

### historical artifacts
- `outputs/runs/*`
- `outputs/error_samples/20260421-213619/*`
- `submission_ml/` subtree

## validation_rename_dependency_map

- `biodcase_development_set/validation/` on-disk directory structure
- `configs/baseline.yaml` (`val_split`, `test_split`)
- `src/pipeline.py` default `splits=("train", "validation")`
- `src/data/build_manifest.py` default `--splits train validation`
- `src/training/train.py` selection fallback to `val_split`
- `src/training/evaluate.py` compatibility with `val_split`
- `src/analysis/inspect_errors.py` hard-coded `split = "validation"`
- `src/analysis/imbalance_audit.py` uses `eval_split = val_split`
- `src/localization/temporal_*` configs and docs
- `pipeline_walkthrough.ipynb`
- root-level `validation_class_distribution.csv`
- duplicate `val_class_distribution.csv`

## delete_candidates

1. `submission_ml/` duplicate subtree
2. `val_class_distribution.csv` duplicate/stale generated CSV
3. stale generated summaries under `outputs/imbalance_audit_summary.*` if regenerated from canonical scripts
4. stale sidecar docs `PHASE_1_REPOSITORY_TRUTH_CLEANUP.md`, `PHASE_5_RECONCILIATION.md`, and `PHASE_6_BONUS_TEMPORAL_LOCALIZATION.md` if replaced by truthful canonical docs later

## rename_risks

- renaming on-disk `validation/` directories would break bundled manifest paths and helper scripts
- renaming `val_split` blindly would drift configs/tests that still rely on compatibility aliases
- removing `validation_class_distribution.csv` before choosing a canonical replacement would break doc references
- editing notebook semantics without updating embedded outputs would leave contradictory rendered cells

## code_hardening_needed

- `src/training/train.py`: stop silent fallback from missing `selection_split` to official held-out test
- `src/analysis/inspect_errors.py`: remove stale run ID default and hard-coded split
- `src/pipeline.py`: replace stale report default `outputs/runs/20260421-213619/...`
- `src/analysis/imbalance_audit.py`: avoid silently choosing stale historical runs and legacy wording

## count_authority_candidates

- `data_manifest.csv` — canonical manifest-derived event counts if schema/provenance is reconciled
- `outputs/data_quality_summary.csv` — canonical quality issue counts if regenerated against the canonical manifest
- `outputs/class_distribution_by_split.csv` — preferred canonical split-distribution summary once regenerated
- `train_class_distribution.csv` / `validation_class_distribution.csv` / `val_class_distribution.csv` — currently duplicated derived outputs, not all authoritative
- historical narrative docs in `archive/cleanup_history/` — non-authoritative for counts

## structural_actions

| path | action | reason | dependency_confirmation | replacement_path | validation_check |
| --- | --- | --- | --- | --- | --- |
| `submission_ml/` | deleted | stale duplicate subtree; root repo is authoritative | discovery scan found duplicate docs/configs/tests and no active root tests importing from it | none | `test ! -e submission_ml` |
| `val_class_distribution.csv` | deleted | stale duplicate generated CSV with conflicting counts vs `validation_class_distribution.csv` | search found no live code path depending on this specific file name | later canonical report under `outputs/reports/manifest/` or regenerated root outputs | `test ! -e val_class_distribution.csv` |
| `PHASE_1_REPOSITORY_TRUTH_CLEANUP.md` | archived | stale status doc contradicted bundled repo state | not imported by code/tests; historical only | `archive/cleanup_history/PHASE_1_REPOSITORY_TRUTH_CLEANUP.md` | `test -f archive/cleanup_history/PHASE_1_REPOSITORY_TRUTH_CLEANUP.md` |
| `PHASE_5_RECONCILIATION.md` | archived | stale reconciliation doc overclaimed missing artifacts | historical only; to be replaced by truthful canonical docs | `archive/cleanup_history/PHASE_5_RECONCILIATION.md` | `test -f archive/cleanup_history/PHASE_5_RECONCILIATION.md` |
| `PHASE_6_BONUS_TEMPORAL_LOCALIZATION.md` | archived | stale phase note not authoritative for final packaging | historical only; bonus path remains in code | `archive/cleanup_history/PHASE_6_BONUS_TEMPORAL_LOCALIZATION.md` | `test -f archive/cleanup_history/PHASE_6_BONUS_TEMPORAL_LOCALIZATION.md` |
| `EXTERNAL_DATA_REQUIREMENTS.md` | archived | stale truth doc contained contradicted statements about bundled assets | historical only; to be replaced by canonical limitations doc later | `archive/cleanup_history/EXTERNAL_DATA_REQUIREMENTS.md` | `test -f archive/cleanup_history/EXTERNAL_DATA_REQUIREMENTS.md` |
| `archive/cleanup_history/README_ARCHIVED.md` | added | marks archived status/history docs as non-authoritative | new archive root for historical cleanup traceability | authoritative note inside archive | `test -f archive/cleanup_history/README_ARCHIVED.md` |

## documentation_truth_actions

| file | misleading claim removed | contradiction fixed | status clarified | evidence source used |
| --- | --- | --- | --- | --- |
| `README.md` | removed archived phase-doc links and removed claim that bundled manifest contains `label_raw` | corrected bundled raw-data presence, bundled best CNN run presence, and `outputs/classical/` absence | labeled classical as current primary, CNN as historical, temporal localization as experimental, and external-count reconciliation as external-data-dependent | bundled repo tree, `data_manifest.csv` schema, `find outputs/runs/20260421-223457`, WAV/CSV counts in `biodcase_development_set/` |
| `EXPERIMENTS.md` | removed implication that the historical CNN run is the current primary path | corrected executed-vs-implemented status for classical baselines and temporal localization | labeled CNN as historical, classical as current primary but not bundled, temporal localization as experimental | `outputs/runs/`, absence of `outputs/classical/`, presence of `src/localization/` |
| `AUDIT_PROGRESS.json` | removed stale statements claiming the raw dataset tree and best run are absent | corrected status of bundled raw data, bundled annotations/WAVs, and bundled best CNN run | marked official external-count reconciliation as still requiring outside evidence | bundled repo tree and manifest/schema inspection |

## split_hardening_actions

| file | change | risk addressed | validation |
| --- | --- | --- | --- |
| `configs/baseline.yaml` | added `official_test_split`, `inner_selection_split`, `selection_strategy`, and `training.allow_official_test_for_selection` | makes split roles explicit instead of relying on ambiguous legacy naming | config remains loadable and inherited by downstream configs |
| `src/training/train.py` | introduced explicit split resolution and failure when inner selection is missing without opt-in | removes silent selection-on-held-out fallback in the historical CNN path | targeted tests and full pytest pass |
| `src/training/evaluate.py` | prefers `official_test_split` / `inner_selection_split` metadata over legacy aliases | makes split-role metadata explicit and safer | compile + tests pass |
| `src/analysis/inspect_errors.py` | removed stale hard-coded report default and inferred split more safely from report/audio path | removes stale helper default and avoids unconditional `validation` assumption | compile + tests pass |
| `src/pipeline.py` | removed stale default report path in helper wrapper | removes stale helper default to historical run ID | compile + tests pass |
| `tests/test_training_options.py` | added assertions for fail-loud split semantics and explicit opt-in path | guards against regression of silent held-out-test selection | targeted tests pass |

## manifest_regeneration_actions

| item | outcome | evidence |
| --- | --- | --- |
| regeneration possible | yes | bundled `biodcase_development_set/train/annotations/*.csv`, `biodcase_development_set/validation/annotations/*.csv`, and bundled WAV trees were sufficient to rebuild the manifest |
| canonical manifest | regenerated in place at `data_manifest.csv` | regenerated manifest now has 76123 rows and includes `label_raw` plus `label_display` |
| canonical report entrypoint | added | `scripts/regenerate_all_reports.py` rebuilds the manifest, quality reports, count reports, and imbalance audit from one manifest |
| canonical report locations | added | `outputs/reports/manifest/`, `outputs/reports/quality/`, `outputs/reports/audit/`, `outputs/reports/provenance/` |
| stale duplicates removed | yes | removed `train_class_distribution.csv`, `validation_class_distribution.csv`, `outputs/class_distribution_by_split.csv`, `outputs/data_quality_report.csv`, `outputs/data_quality_summary.csv`, `outputs/imbalance_audit_summary.json`, and `outputs/imbalance_audit_summary.md` |
| claims downgraded | not needed for manifest regeneration itself | regeneration was possible from bundled source data, but external official-count reconciliation still remains external-data-dependent |

## result_family_status

| result family | status | bundled path | provenance note |
| --- | --- | --- | --- |
| classical baselines | implemented but not bundled | none | code and tests are present, but `outputs/classical/` is absent |
| historical CNN | historical only | `outputs/runs/20260421-223457/` | bundled metrics/config are present, but provenance is partial and below the current packaging standard |
| temporal localization | implemented but not bundled | none | experimental code path only |
| crop verification exports | implemented but not bundled | none | generated on demand by script, not pre-bundled |
| manifest-derived reports | bundled | `outputs/reports/` | regenerated with provenance in `outputs/reports/provenance/report_regeneration_provenance.json` |

## final_resolution_summary

| category | items |
| --- | --- |
| fixed | `CLN-001`, `CLN-002`, `CLN-003`, `CLN-004`, `CLN-005`, `CLN-006`, `CLN-007`, `CLN-008`, `CLN-010` |
| blocked by missing external data | `CLN-009` |
| intentionally documented limitation | classical result bundles absent in this snapshot; historical CNN provenance remains partial and is labeled historical only |
| deferred research enhancement | broader spectrogram ablation and optional future experiment bundles |
