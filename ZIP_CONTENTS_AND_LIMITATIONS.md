# ZIP Contents And Limitations

This document describes what the repository snapshot physically contains and
what it does not.

## Physically bundled

Bundled in this snapshot:

- source code under `src/`
- configs under `configs/`
- tests under `tests/`
- bundled dataset tree under `biodcase_development_set/`
- regenerated canonical manifest `data_manifest.csv`
- canonical manifest-derived reports under `outputs/reports/`
- historical CNN run directories under `outputs/runs/`
- cleanup tracking under `AUDIT_PROGRESS.json` and `CLEANUP_MASTER_TRACKER.md`

## Not bundled

Not bundled in this snapshot:

- executed classical baseline result directories under `outputs/classical/`
- a canonical bundled temporal-localization result directory
- APLOSE software itself
- external official BIODCASE count tables or owner reconciliation records

## Historical paths

Historical paths are present and should not be confused with the primary
current methodology:

- `outputs/runs/20260421-223457/`: bundled historical CNN comparison run
- `outputs/error_samples/20260421-213619/`: historical exported error samples
- `archive/cleanup_history/`: archived cleanup notes, intentionally
  non-authoritative

## Experimental paths

Experimental code paths are present under:

- `src/localization/`
- `configs/temporal_localization.yaml`

These are separate from the main crop-classification pipeline and are not
packaged as the primary result family.

## Split naming

The on-disk `validation/` directory is a legacy name for the official held-out
test domains. It is not an inner model-selection split.

Inner model selection, where implemented, must be derived from the official
training split only.

## What can be reproduced from this repository alone

Reproducible from the bundled snapshot:

- manifest regeneration
- manifest-derived count and quality reports
- crop extraction and representation export
- classical-baseline code execution
- historical CNN artifact inspection

## What cannot be reproduced or proven from this repository alone

Not reproducible or not fully provable from the bundled snapshot alone:

- external official count reconciliation
- claims about challenge-owner totals beyond the bundled data
- a bundled classical-results package, because it is absent
- a canonical packaged temporal-localization results bundle, because it is
  absent
- the external APLOSE annotation workflow itself

## External-data requirements

For the cleaned repository as bundled:

- manifest/report regeneration does not require additional raw audio beyond what
  is already bundled
- exact reconciliation against official external counts requires external
  official documents or owner evidence
- reproducing future large experiment bundles may require compute, but not
  additional raw data if the current bundled tree is used
