# Result Provenance

Current-checkout inventory, updated 24 October 2026: stored search and
notebook-reproduction summaries exist; raw audio, the final-evaluation KNN
bundle, and `outputs/runs/20260421-223457/` are absent. See
[Results and evidence](docs/RESULTS.md) for the authoritative inventory.
Earlier claims below about that CNN run being bundled refer to a previous
local snapshot and are superseded by this inventory.

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
  historical CNN run referenced by older documentation; not bundled in this
  checkout. Its metrics are not independently verifiable from a bundled run
- `outputs/classical/`:
  not bundled in this snapshot; classical baselines are implemented in code but
  executed result directories are absent
- temporal localization outputs:
  not bundled in a canonical result directory; the code path remains
  experimental

Treat historical bundles and implemented-but-unbundled paths accordingly in
docs and downstream packaging.
