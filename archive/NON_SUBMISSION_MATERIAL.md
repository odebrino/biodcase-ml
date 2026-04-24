# Non-Submission Material

The maintained submission path is the strict classical KNN pipeline:

- `configs/knn_submission.yaml`
- `configs/knn_search.yaml`
- `src/run_submission.py`
- `src/models/knn_pipeline.py`
- `src/models/knn_search.py`
- `src/features/`
- `src/evaluation/`

Strict KNN means no CNN, no legacy CNN checkpoint, no frozen neural embedding,
and no pretrained neural model.

Historical CNN material lives under `legacy/cnn/` for provenance only. It is
not imported by the maintained KNN path and must not be used for strict KNN
submission results.

Other non-submission material:

- `src/localization/`: experimental temporal-localization bonus path
- `configs/temporal_localization.yaml`: temporal-localization config
- `src/classical/clustering.py`: exploratory unsupervised analysis
