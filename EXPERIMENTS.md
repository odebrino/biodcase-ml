# Experiments

This file is a factual experiment index for the bundled snapshot.

## Bundled Historical CNN Runs

Bundled run directories exist under `outputs/runs/`.

Historically notable bundled CNN run:

- `outputs/runs/20260421-223457/`

Selected metrics from the bundled artifacts in that run:

| run | config | accuracy | macro_f1 | f1_bpd |
| --- | --- | ---: | ---: | ---: |
| `20260421-223457` | `configs/nitro4060_bpd.yaml` | 0.9301 | 0.8866 | 0.7725 |

This is a historical comparison path only.

Provenance status:

- bundled metrics/configs: yes
- bundled command line: no
- bundled manifest hash: no
- bundled result-family summary metadata: partial only

## Classical Baselines

The classical baseline suite is implemented in `src.classical.baselines`.

Current bundled status:

- code: present
- tests: present
- executed `outputs/classical/` results: not bundled

This means the classical path is implemented but the repository snapshot does
not currently ship a timestamped classical result package.

## Experimental Temporal Localization

The bonus temporal-localization path is experimental.

Current bundled status:

- code: present under `src/localization/`
- executed canonical result directory: not bundled

## Commands

Historical CNN path:

```bash
python -m src.training.train --config configs/nitro4060_bpd.yaml --manifest data_manifest.csv
python -m src.training.evaluate --checkpoint outputs/runs/<run>/best_model.pt --config configs/nitro4060_bpd.yaml --manifest data_manifest.csv --output-dir outputs/runs/<run>
```

Current primary classical path:

```bash
python -m src.classical.baselines --config configs/classical_baselines.yaml --manifest data_manifest.csv --output-dir outputs/classical
```
