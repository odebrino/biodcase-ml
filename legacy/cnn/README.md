# Legacy CNN

This directory contains the historical CNN classification path that was moved
out of the root maintained submission tree.

It is preserved for provenance, comparison, and result traceability only.

It is not the maintained BIODCASE submission path.

The active submission path is KNN-only:

- `configs/knn_submission.yaml`
- `configs/knn_search.yaml`
- `src/models/knn_pipeline.py`
- `src/models/knn_search.py`
- `src/run_submission.py`

Historical CNN assets preserved here:

- `legacy/cnn/training/`
- `legacy/cnn/models/resnet.py`
- `legacy/cnn/configs/`

Legacy commands:

```bash
python -m legacy.cnn.training.train --config legacy/cnn/configs/nitro4060_bpd.yaml --manifest data_manifest.csv
python -m legacy.cnn.training.evaluate --checkpoint outputs/runs/<run>/best_model.pt --config legacy/cnn/configs/nitro4060_bpd.yaml --manifest data_manifest.csv --output-dir outputs/runs/<run>
```

Any bundled result above 90% accuracy currently belongs to this legacy CNN
branch, not to the maintained KNN submission pipeline.
