# BIODCASE KNN Submission Repository

This repository is maintained around one submission path only: a strict KNN
pipeline in the root `src/` package.

Strict KNN here means no CNN, no legacy CNN checkpoint, no frozen neural
embedding, and no pretrained neural model. The final estimator is always
`sklearn.neighbors.KNeighborsClassifier`.

## Maintained Submission Path

The submission source of truth is:

- config: `configs/knn_submission.yaml`
- search config: `configs/knn_search.yaml`
- search entry point: `python -m src.models.knn_search`
- submission entry point: `python -m src.run_submission`
- KNN pipeline: `src/models/knn_pipeline.py`
- KNN search: `src/models/knn_search.py`
- evaluation: `src/evaluation/`
- feature extraction: `src/features/`
- manifest and spectrogram preparation: `src/data/`

Duplicated maintained code under `submission_professor/src/` is no longer part
of the active source tree. Historical bundle artifacts remain there only as a
snapshot.

## Split Semantics

The on-disk split names are legacy:

- `train` = official training split
- `validation` = official held-out test split

`validation` must never be used for:

- model selection
- scaler fitting
- imputation fitting
- PCA or feature-selection fitting
- metric-learning fitting
- final KNN fitting

All search and transform fitting must happen on `train` only.

Held-out official test domains in the bundled setup:

- `casey2017`
- `kerguelen2014`
- `kerguelen2015`

## Reality Check

Strict KNN snapshots should be read separately:

Preserved professor/bundled KNN snapshot:

- official held-out accuracy: `0.2290`
- official held-out macro-F1: `0.1136`

Previous quick-search KNN snapshot:

- quick train-only CV accuracy: `0.7563`
- quick train-only CV macro-F1: `0.7564`
- official held-out accuracy: `0.2821`
- official held-out macro-F1: `0.1765`

Latest full/domain-aware KNN snapshot:

- domain-aware CV accuracy: `0.1966`
- domain-aware CV macro-F1: `0.1438`
- selected feature set: `waveform_spectral_stats`
- selected KNN: `k=1`, cosine metric, `RobustScaler`
- official held-out accuracy: `0.2612`
- official held-out macro-F1: `0.1812`

Previous CV-focused 7-class snapshot:

- search mode: `--cv-focused`, official `train` split only
- StratifiedKFold CV accuracy: `0.7583`
- StratifiedKFold CV macro-F1: `0.7583`
- selected feature set: `lowfreq_all_plus_waveform_spectral`
- selected KNN: `k=7`, Minkowski metric, `StandardScaler`,
  `SelectKBest(k=1024)`
- secondary domain-aware CV for selected config: accuracy `0.1612`,
  macro-F1 `0.1089`
- official held-out evaluation: skipped for this phase

Current notebook-reproduction/CV-focused snapshots:

- exact notebook leaky audit on `elephantisland2014`, 3-class:
  accuracy `0.6205`, macro-F1 `0.5844`, weighted-F1 `0.6128`
- exact notebook split-safe CV on `elephantisland2014`, 3-class:
  accuracy `0.6291`, macro-F1 `0.5971`, weighted-F1 `0.6225`
- best per-dataset split-safe exact notebook result:
  `maudrise2014`, accuracy `0.9750`, macro-F1 `0.6842`,
  weighted-F1 `0.9741`
- best combined/balanced 3-class strict KNN CV search:
  accuracy `0.8317`, macro-F1 `0.8321`, weighted-F1 `0.8321`
- best combined/balanced 7-class strict KNN CV search:
  accuracy `0.7911`, macro-F1 `0.7910`, weighted-F1 `0.7910`
- official held-out evaluation: skipped for this phase

The only bundled result above 90% accuracy is historical CNN material:

- historical CNN held-out accuracy: `0.9301`
- historical CNN macro-F1: `0.8866`

That CNN path now lives under `legacy/cnn/` and is not the maintained
submission route.

Reorganization alone is not expected to reach 90%. A valid `>90%` claim would
require the real official held-out `validation` split to exceed 0.90 after
fitting the full KNN pipeline on `train` only.

The large gap between random/stratified CV and domain-aware CV is the main
diagnostic finding. For official held-out robustness, domain-aware CV is the
more realistic selection criterion. For `Projet.ipynb`-style development
tables, use the explicit CV-focused mode below, which ranks by train-only
StratifiedKFold CV and reports domain-aware CV as a secondary diagnostic.

`Projet.ipynb`-style 3-class KNN results are not directly comparable to this
strict 7-class official task unless the grading specification explicitly
allows the aggregation `bma,bmb,bmz -> ABZ`, `bmd,bpd -> DDswp`, and
`bp20,bp20plus -> 20Hz20Plus`.

The current exact notebook audit did not reproduce the reported `0.89` on the
suspected `elephantisland2014` setup. A `0.89`-like accuracy appears on some
single-dataset split-safe audits, but the best valid combined/balanced
CV-focused searches reached `0.8317` for the 3-class task and `0.7911` for the
7-class task.

## Evaluation Modes

CV-focused experiments:

- use only the official `train` split
- rank by StratifiedKFold weighted-F1, then accuracy, then macro-F1
- are the closest maintained comparison to `Projet.ipynb`-style internal
  train-split evaluation
- do not run or optimize on the official held-out `validation` split

Official held-out evaluation:

- is an optional final check with `src.run_submission`
- fits all transforms and KNN on official `train` only
- evaluates once on the legacy on-disk `validation` split
- is not the target of the current CV-focused phase

## Install

```bash
pip install -r requirements.txt
```

## Run CV-Focused KNN Search

Run the exact notebook reproduction audit first:

```bash
python -m src.experiments.notebook_reproduction \
  --config configs/knn_notebook_reproduction.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/notebook_reproduction
```

Then run the CV-focused searches. These use official `train` only and do not
run official held-out validation:

```bash
python -m src.models.knn_search \
  --config configs/knn_search_3class_cv_full.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/knn_search_3class_cv_full \
  --stage full \
  --n-jobs 1 \
  --cache-features \
  --cv-focused \
  --max-candidates 200
```

```bash
python -m src.models.knn_search \
  --config configs/knn_search_7class_cv_full.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/knn_search_7class_cv_full \
  --stage full \
  --n-jobs 1 \
  --cache-features \
  --cv-focused \
  --max-candidates 200
```

The 3-class config is not the default official task. It exists only to audit
the notebook's apparent label aggregation and internal-CV setup.

## Run Domain-Aware KNN Search

Search uses only the official `train` split and writes ranked candidates plus a
best submission config. Without `--cv-focused`, the maintained search ranks by
the most realistic available train-only domain-aware CV scenario:

```bash
python -m src.models.knn_search \
  --config configs/knn_search.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/knn_search
```

If the full search is too expensive, use the supported runtime controls:

```bash
python -m src.models.knn_search \
  --config configs/knn_search.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/knn_search \
  --quick \
  --max-candidates 12 \
  --n-jobs 1 \
  --cache-features
```

Inspect after search:

- `outputs/reports/notebook_reproduction/summary.md`
- `outputs/reports/notebook_reproduction/per_dataset_results.csv`
- `outputs/reports/notebook_reproduction/combined_train_results.csv`
- `outputs/reports/knn_search_3class_cv_full/search_results.csv`
- `outputs/reports/knn_search_3class_cv_full/search_summary.md`
- `outputs/reports/knn_search_3class_cv_full/best_cv_metrics.json`
- `outputs/reports/knn_search_3class_cv_full/best_knn_config.yaml`
- `outputs/reports/knn_search_7class_cv_full/search_results.csv`
- `outputs/reports/knn_search_7class_cv_full/search_summary.md`
- `outputs/reports/knn_search_7class_cv_full/best_cv_metrics.json`
- `outputs/reports/knn_search_7class_cv_full/best_knn_config.yaml`
- `outputs/reports/quality/label_mapping_audit.json`
- `outputs/reports/quality/label_mapping_audit.md`

## Run Official KNN Submission Evaluation

This is optional in the current CV-focused phase. Run it only when you want a
final held-out check with either the default config or a selected search config:

```bash
python -m src.run_submission \
  --config configs/knn_submission.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/model_evaluation
```

Or:

```bash
python -m src.run_submission \
  --config outputs/reports/knn_search/best_knn_config.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/model_evaluation
```

Inspect after submission:

- `outputs/reports/model_evaluation/official_test_results.csv`
- `outputs/reports/model_evaluation/official_test_metrics.json`
- `outputs/reports/model_evaluation/official_test_confusion_matrix.csv`
- `outputs/reports/model_evaluation/official_test_predictions.csv`
- `outputs/reports/model_evaluation/main_experiment_overview.md`
- `outputs/reports/model_evaluation/selected_knn_config.yaml`
- `outputs/reports/model_evaluation/domain_diagnostics.json`
- `outputs/reports/model_evaluation/domain_diagnostics.md`
- `outputs/reports/model_evaluation/per_sample_knn_neighbors.csv`
- `outputs/reports/quality/split_integrity.json`
- `outputs/reports/quality/split_integrity.md`

## Legacy Material

Historical CNN provenance now lives under `legacy/cnn/`.

Non-submission material still present for reference:

- `legacy/cnn/`
- `src/localization/`
- `src/classical/clustering.py`

The root maintained submission path remains KNN-only.
