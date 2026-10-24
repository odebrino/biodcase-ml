# Data access and reproduction

The checked-in manifest and summary artifacts accompany an external BioDCASE/ATBFL development dataset. Raw audio is not included in this Git repository.

## Obtain the source data

1. Open the [official BioDCASE Task 2 page](https://biodcase.github.io/challenge2026/task2) and follow its development-data download link to [Zenodo record 18832958](https://zenodo.org/records/18832958).
2. Download the development audio/annotation archives from that record. Record the version and file checksums; newer corrected releases can produce different processed counts.
3. Extract and arrange the split directories under `biodcase_development_set/` in the repository root:

```text
biodcase_development_set/
  train/
    audio/<site-year>/*.wav
    annotations/<site-year>.csv
  validation/
    audio/<site-year>/*.wav
    annotations/<site-year>.csv
```

At documentation time, the record lists `2026_BioDCASE_development_set.zip` (5,591,313,490 bytes), MD5 `1105fa1f707fe9f270ccc3ad05ddb7d0`. Verify the record's current checksum for the version you obtain. The complete source archive was not downloaded during this software revision.

Use the development set for this repository. The challenge evaluation archive is a separate resource. The stored `validation` holdout uses Casey 2017 and Kerguelen 2014/2015. Cite the source record and its creators when using the recordings; data reuse conditions are separate from software terms.

The checked-in manifest uses paths beginning with `biodcase_development_set/`. Keep that layout to use it directly, or rebuild a manifest for the actual location/version of your download. The recorded manifest has 58,510 train and 17,613 holdout events, not a claim about every release's official annotation totals.

## Regenerate manifest-derived reports

From the repository root, with the CPU environment installed:

```bash
python scripts/regenerate_all_reports.py \
  --data-root biodcase_development_set \
  --manifest-out outputs/new-study/manifest.csv \
  --reports-root outputs/new-study/reports
```

This builds the manifest, quality checks, counts, and provenance without requiring a historical CNN run. It does not fit or evaluate a model.

To run the maintained KNN evaluation using the source development holdout:

```bash
python -m src.run_submission \
  --config configs/knn_submission.yaml \
  --manifest outputs/new-study/manifest.csv \
  --output-dir outputs/new-study/evaluation
```

Model selection belongs inside the training split. Generate reports in a new directory and retain their real provenance. Running a default configuration does not reproduce every historical search result.

The historical CNN imbalance audit can be requested separately with `--include-legacy-audit --run-dir <existing-run> --audit-config <existing-config>`. It requires those actual artifacts; it is not part of the maintained KNN/data-report path.

## Run without the full dataset

```bash
python -m examples.synthetic_demo --output-dir outputs/demo
```

This generates 32 short sine-wave WAVs, fits on 24 training examples, and evaluates on eight distinct generated holdout examples. It exercises the real feature/KNN/report code, records seed/environment/audio hashes, and keeps all outputs inside the demo directory. The example uses placeholder bma/bmd labels for schema compatibility. Its accuracy describes artificial signals only.
