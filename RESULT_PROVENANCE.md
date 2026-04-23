# Result Provenance

This repository uses the following provenance expectations for bundled result
directories.

## Required for a fully bundled result package

Every result directory that is presented as a bundled, reproducible result
package should include:

- timestamped output directory
- config snapshot
- command used
- seed
- split strategy
- manifest hash when available
- model comparison table when the run is part of a grid
- best-model summary
- per-class metrics
- confusion matrix
- grouped ambiguity report when applicable
- per-dataset metrics when applicable

## Current repository reality

- `outputs/reports/`:
  bundled and regenerated in this snapshot; provenance is recorded in
  `outputs/reports/provenance/report_regeneration_provenance.json`
- `outputs/runs/20260421-223457/`:
  bundled historical CNN run with partial provenance; it includes metrics,
  config, predictions, and run metadata, but it does not include the full
  command line or manifest hash expected for a fully packaged result bundle
- `outputs/classical/`:
  not bundled in this snapshot; classical baselines are implemented in code but
  executed result directories are absent
- temporal localization outputs:
  not bundled in a canonical result directory; the code path remains
  experimental

Treat historical bundles and implemented-but-unbundled paths accordingly in
docs and downstream packaging.
