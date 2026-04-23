# Dataset Spec

This document describes the bundled dataset snapshot as it exists in this
cleaned repository.

## Official split concept

The project follows the challenge-style split that is already laid out on disk:

- `train/`: official training domains
- `validation/`: legacy directory name for the official held-out test domains

No new random train/test split is created by the cleaned repository.

## Site-year structure

The bundled snapshot contains 11 site-year domains.

Training domains:

- `ballenyislands2015`
- `casey2014`
- `elephantisland2013`
- `elephantisland2014`
- `greenwich2015`
- `kerguelen2005`
- `maudrise2014`
- `rosssea2014`

Held-out final-evaluation domains on disk under `validation/`:

- `casey2017`
- `kerguelen2014`
- `kerguelen2015`

## Authoritative numbers in the cleaned repository

The following numbers are authoritative for this cleaned repository snapshot
because they are derived directly from the bundled data and regenerated
manifest:

- `data_manifest.csv` rows: `76123`
- manifest split counts: `train=58510`, `validation=17613`
- train audio WAV files found in the bundled tree: `6004`
- validation audio WAV files found in the bundled tree: `587`
- train annotation CSVs bundled: `8`
- validation annotation CSVs bundled: `3`

Per-domain manifest event counts:

| split | dataset | events |
| --- | --- | ---: |
| train | `ballenyislands2015` | 2222 |
| train | `casey2014` | 6866 |
| train | `elephantisland2013` | 21913 |
| train | `elephantisland2014` | 20957 |
| train | `greenwich2015` | 1128 |
| train | `kerguelen2005` | 2960 |
| train | `maudrise2014` | 2360 |
| train | `rosssea2014` | 104 |
| validation | `casey2017` | 3263 |
| validation | `kerguelen2014` | 8814 |
| validation | `kerguelen2015` | 5536 |

## Published totals versus processed totals

This repository distinguishes between:

- published or externally circulated challenge totals
- the processed manifest totals that can be regenerated from the bundled
  snapshot

Those numbers may differ for several reasons:

- duplicate events are removed during manifest construction
- events with invalid timing are excluded
- events with too little usable audio after clipping are excluded
- labels are normalized from raw aliases into canonical classes
- the bundled snapshot may not be byte-for-byte identical to an external
  challenge distribution or later owner reconciliation

## What cannot be fully verified from this repository alone

The repository cannot, by itself, prove:

- that the regenerated manifest totals match any official external BIODCASE
  count table
- why some older notes referred to train WAV counts such as `6007` instead of
  the `6004` WAVs currently present in the bundled tree
- whether any external challenge release used different inclusion/exclusion
  rules before processing

Those claims require external official sources or project-owner evidence, not
just the cleaned repository contents.
