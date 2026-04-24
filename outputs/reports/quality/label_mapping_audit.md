# Label Mapping Audit

## Current Strict KNN Labels

- `bma`, `bmb`, `bmd`, `bmz`, `bp20`, `bp20plus`, `bpd`

## Notebook 3-Class Mapping

- `bma` -> `ABZ`
- `bmb` -> `ABZ`
- `bmz` -> `ABZ`
- `bmd` -> `DDswp`
- `bpd` -> `DDswp`
- `bp20` -> `20Hz20Plus`
- `bp20plus` -> `20Hz20Plus`

## Conclusion

The maintained strict KNN config uses the original 7 labels, not the notebook 3-class aggregation. That label granularity mismatch can make the notebook's reported KNN accuracy non-comparable, especially because ABZ, DDswp, and 20Hz20Plus collapse known ambiguous subclasses. The larger issue remains evaluation protocol: same-domain/random splits are optimistic while dataset/domain-aware CV collapses.

## Class Counts By Split

- `train` / `bma`: 18092
- `train` / `bmb`: 4622
- `train` / `bmd`: 13141
- `train` / `bmz`: 1596
- `train` / `bp20`: 10380
- `train` / `bp20plus`: 5003
- `train` / `bpd`: 5676
- `validation` / `bma`: 6268
- `validation` / `bmb`: 2277
- `validation` / `bmd`: 2168
- `validation` / `bmz`: 918
- `validation` / `bp20`: 2547
- `validation` / `bp20plus`: 2757
- `validation` / `bpd`: 678

## Notebook 3-Class Counts By Split

- `train` / `20Hz20Plus`: 15383
- `train` / `ABZ`: 24310
- `train` / `DDswp`: 18817
- `validation` / `20Hz20Plus`: 5304
- `validation` / `ABZ`: 9463
- `validation` / `DDswp`: 2846
