# Annotation Guidelines Summary

This file summarizes the scientific context reflected by the repository. It is
background guidance, not a claim that the full annotation workflow is
implemented here.

## Seven-class inventory

The canonical class set used by the cleaned codebase is:

- `bma` / `Bm-A`
- `bmb` / `Bm-B`
- `bmz` / `Bm-Z`
- `bmd` / `Bm-D`
- `bp20` / `Bp-20`
- `bp20plus` / `Bp-20Plus`
- `bpd` / `Bp-40Down`

Alias normalization happens during manifest ingestion so raw variants such as
`Bm-A`, `BmA`, `Bp-40Down`, and related forms are mapped into these canonical
labels while preserving `label_raw`.

## Ambiguity-aware merged families

The repository supports optional regrouped reporting for known ambiguity
families:

- `ABZ` = `bma`, `bmb`, `bmz`
- `DDswp` = `bmd`, `bpd`
- `20Hz20Plus` = `bp20`, `bp20plus`

These regroupings are for interpretation and reporting. The main task remains
the original seven-class classification problem.

## APLOSE-inspired spectrogram settings

The repository mirrors guideline-inspired presets through:

- `configs/aplose_512_98.yaml`
- `configs/aplose_256_90.yaml`

Those presets resolve to named settings in `src/data/spectrogram_presets.py`.
They are guidance-aligned spectrogram configurations, not literal APLOSE UI
exports.

## Why the guideline context matters

Several distinctions in this task can be intrinsically ambiguous:

- `Bm-A`, `Bm-B`, and `Bm-Z` can be visually or acoustically close
- `Bm-D` and `Bp-40Down` can overlap in low-frequency structure
- `Bp-20` and `Bp-20Plus` can differ by degree rather than by a perfectly clean
  boundary

Because of that, confusion patterns are not always pure model failure. They can
also reflect label ambiguity, partial fragments, and annotation uncertainty.

## Operator-facing annotation concepts

The repository keeps these ideas as scientific context only:

- confidence judgments may vary between annotators
- fragments can be partial, truncated, or overlapping
- one acoustic event may not have perfectly sharp boundaries
- spectrogram parameter choices influence what looks separable

These concepts help explain why ambiguity-aware analysis is useful.

## What is not implemented here

The repository does not implement the APLOSE web UI or the full external
annotation campaign workflow. It implements ingestion, normalization,
spectrogram generation, crop extraction, representation export, and downstream
evaluation based on already available annotations.
