# Experiments

This file tracks the repository's preserved experiment families and the current
maintained KNN workflow.

## Legacy CNN

Historical CNN material was moved to `legacy/cnn/`.

Bundled legacy result snapshot:

| run | config | official_accuracy | official_macro_f1 | f1_bpd |
| --- | --- | ---: | ---: | ---: |
| `20260421-223457` | `legacy/cnn/configs/nitro4060_bpd.yaml` | 0.9301 | 0.8866 | 0.7725 |

This is provenance only. It is not the maintained submission path.

Historical legacy commands:

```bash
python -m legacy.cnn.training.train --config legacy/cnn/configs/nitro4060_bpd.yaml --manifest data_manifest.csv
python -m legacy.cnn.training.evaluate --checkpoint outputs/runs/<run>/best_model.pt --config legacy/cnn/configs/nitro4060_bpd.yaml --manifest data_manifest.csv --output-dir outputs/runs/<run>
```

## CV-Focused KNN Experiments

The current phase targets train-only cross-validation performance, not
official held-out optimization. This is the closest maintained comparison to
`Projet.ipynb`-style internal train-split evaluation.

Notebook reproduction audit:

```bash
python -m src.experiments.notebook_reproduction \
  --config configs/knn_notebook_reproduction.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/notebook_reproduction
```

Current 3-class CV-focused search:

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

Current 7-class CV-focused search:

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

Older general CV-focused command:

```bash
python -m src.models.knn_search \
  --config configs/knn_search.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/knn_search \
  --stage full \
  --n-jobs 1 \
  --cache-features \
  --cv-focused
```

Selection order in this mode:

- StratifiedKFold weighted-F1
- StratifiedKFold accuracy
- StratifiedKFold macro-F1
- macro precision
- lower StratifiedKFold weighted-F1 variance

CV-focused artifacts:

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

Notebook exact reproduction audit:

- default dataset/mode: `elephantisland2014`, 3-class notebook mapping
- feature set: `notebook_exact_44`
- leaky diagnostic audit: accuracy `0.6205`, macro-F1 `0.5844`,
  weighted-F1 `0.6128`
- split-safe CV reproduction: accuracy `0.6291`, macro-F1 `0.5971`,
  weighted-F1 `0.6225`
- best per-dataset split-safe exact notebook result: `maudrise2014`,
  accuracy `0.9750`, macro-F1 `0.6842`, weighted-F1 `0.9741`
- combined train exact notebook 3-class CV: accuracy `0.5786`,
  macro-F1 `0.5553`, weighted-F1 `0.5705`
- combined train exact notebook 7-class CV: accuracy `0.4864`,
  macro-F1 `0.4412`, weighted-F1 `0.4751`

Current best 3-class CV-focused search result:

- search mode: `configs/knn_search_3class_cv_full.yaml --max-candidates 200`
- dataset mode: `balanced_train_cv`
- selected feature set: `lowfreq_all_plus_waveform_spectral`
- selected config: `RobustScaler` + `SelectKBest(k=64)` +
  `KNN(k=7, metric=manhattan, weights=uniform, algorithm=auto)`
- StratifiedKFold CV accuracy: `0.8317`
- StratifiedKFold CV weighted-F1: `0.8321`
- StratifiedKFold CV macro-F1: `0.8321`
- secondary domain-aware CV accuracy for selected config: `0.4202`
- secondary domain-aware CV macro-F1 for selected config: `0.1985`
- official held-out evaluation: skipped

Current best 7-class CV-focused search result:

- search mode: `configs/knn_search_7class_cv_full.yaml --max-candidates 200`
- dataset mode: `balanced_train_cv`
- selected feature set: `lowfreq_all_plus_logmel`
- selected config: `StandardScaler` + `VarianceThreshold` +
  `KNN(k=5, metric=manhattan, weights=distance, algorithm=brute)`
- StratifiedKFold CV accuracy: `0.7911`
- StratifiedKFold CV weighted-F1: `0.7910`
- StratifiedKFold CV macro-F1: `0.7910`
- secondary domain-aware CV accuracy for selected config: `0.1852`
- secondary domain-aware CV macro-F1 for selected config: `0.1232`
- official held-out evaluation: skipped

Previous CV-focused 7-class result:

- search mode: `--stage full --n-jobs 1 --cache-features --cv-focused`
- selected feature set: `lowfreq_all_plus_waveform_spectral`
- selected config: `StandardScaler` + `SelectKBest(k=1024)` +
  `KNN(k=7, metric=minkowski, algorithm=brute)`
- StratifiedKFold CV accuracy: `0.7583`
- StratifiedKFold CV macro-F1: `0.7583`
- secondary domain-aware CV accuracy for selected config: `0.1612`
- secondary domain-aware CV macro-F1 for selected config: `0.1089`
- official held-out evaluation: skipped

Optional 3-class notebook-style CV config:

```bash
python -m src.models.knn_search \
  --config configs/knn_search_3class_cv.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/knn_search_3class_cv \
  --stage full \
  --n-jobs 1 \
  --cache-features \
  --cv-focused
```

The 3-class config is CV-only unless the grading specification explicitly
confirms that `ABZ`, `DDswp`, and `20Hz20Plus` are the target labels.

## Domain-Aware KNN Search

The maintained optimization workflow is:

```bash
python -m src.models.knn_search \
  --config configs/knn_search.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/knn_search \
  --stage full \
  --n-jobs 1 \
  --cache-features
```

This search uses only `train` and excludes the official held-out
`validation` split from feature fitting and model selection.
Without `--cv-focused`, search ranks candidates by the most realistic available
train-only CV scenario, preferring dataset/domain-aware folds over ordinary
random StratifiedKFold when feasible.

Key search artifacts:

- `outputs/reports/knn_search/search_results.csv`
- `outputs/reports/knn_search/search_summary.md`
- `outputs/reports/knn_search/best_cv_metrics.json`
- `outputs/reports/knn_search/best_knn_config.yaml`
- `outputs/reports/knn_search/domain_cv_results.csv`
- `outputs/reports/knn_search/feature_family_comparison.csv`
- `outputs/reports/knn_search/notebook_feature_ablation.csv`
- `outputs/reports/knn_search/notebook_feature_ablation.md`
- `outputs/reports/quality/label_mapping_audit.json`
- `outputs/reports/quality/label_mapping_audit.md`

## Optional Official KNN Evaluation

Official held-out evaluation is not part of the current CV-focused phase. Run
it only as an optional final check with a selected config:

```bash
python -m src.run_submission \
  --config outputs/reports/knn_search/best_knn_config.yaml \
  --manifest data_manifest.csv \
  --output-dir outputs/reports/model_evaluation
```

Strict KNN snapshots:

Preserved professor/bundled KNN snapshot:

- official accuracy: `0.2290`
- official macro-F1: `0.1136`

Previous quick-search KNN snapshot:

- search mode: quick train-only CV
- quick CV accuracy: `0.7563`
- quick CV macro-F1: `0.7564`
- selected config: `handcrafted_stats` + `RobustScaler` + `VarianceThreshold` + `KNN(k=1, metric=manhattan, algorithm=brute)`
- official held-out accuracy: `0.2821`
- official held-out macro-F1: `0.1765`

Latest full/domain-aware KNN snapshot:

- search mode: `--stage full --n-jobs 1 --cache-features`
- ranking criterion: train-only leave-one-dataset/domain CV
- selected feature set: `waveform_spectral_stats`
- selected config: `RobustScaler` + no reducer + `KNN(k=1, metric=cosine, algorithm=brute)`
- domain-aware CV accuracy: `0.1966`
- domain-aware CV macro-F1: `0.1438`
- official held-out accuracy: `0.2612`
- official held-out macro-F1: `0.1812`

For CV-focused experiments, the quick random/stratified score is intentionally
the primary comparison target because it mirrors the notebook-style internal
evaluation better. For official held-out robustness, that same score is
optimistic; domain-aware CV remains the better diagnostic. The official
held-out result remains far below 90% and far below the historical legacy CNN,
which is forbidden for the maintained strict KNN path.

## Projet Notebook Comparison

`Projet.ipynb` reportedly reached about `0.89` KNN accuracy using a 3-class
aggregation:

- `bma`, `bmb`, `bmz` -> `ABZ`
- `bmd`, `bpd` -> `DDswp`
- `bp20`, `bp20plus` -> `20Hz20Plus`

The maintained official path currently uses the original 7 labels. The notebook
result is therefore not comparable unless the grading specification allows that
aggregation.

The exact notebook-style audit did not reproduce `0.89` on the suspected
`elephantisland2014` setup: the leaky diagnostic score was `0.6205` accuracy
and the split-safe CV score was `0.6291` accuracy. A `0.89`-like score did
appear on some single-dataset split-safe audits, especially `maudrise2014`
at `0.9750` accuracy, but its macro-F1 was only `0.6842`, so the headline
accuracy is strongly affected by class distribution.

The best valid combined/balanced CV search reached `0.8317` accuracy for the
3-class task and `0.7911` for the 7-class task. The project therefore has not
reproduced or beaten the notebook's `0.89` as a valid combined split-safe CV
result.

## Classical And Other Material

Other non-submission code still present:

- `src.classical.baselines`
- `src/localization/`

These are not the maintained official submission route.
