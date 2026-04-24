# Notebook Feature Ablation

This compares Projet.ipynb-inspired low-frequency classical features using train-only CV.
The official held-out validation split is not used for this ranking.
Ranking mode: `StratifiedKFold CV first`.

| feature_set | dim | random_acc | random_weighted_f1 | random_macro_f1 | domain_acc | domain_macro_f1 | worst_domain_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lowfreq_all_plus_waveform_spectral` | 277 | 0.7560 | 0.7564 | 0.7564 | 0.1637 | 0.1125 | 0.0549 |
| `class_region_lowfreq_features` | 38 | 0.7324 | 0.7333 | 0.7333 | 0.1552 | 0.1080 | 0.0666 |
| `notebook_exact_44_plus_regions` | 82 | 0.7277 | 0.7298 | 0.7298 | 0.1527 | 0.1028 | 0.0630 |
| `notebook_lowfreq_band_features` | 68 | 0.7149 | 0.7158 | 0.7158 | 0.1574 | 0.1045 | 0.0585 |
| `notebook_exact_26` | 26 | 0.6871 | 0.6873 | 0.6873 | 0.1567 | 0.0978 | 0.0486 |
| `lowfreq_all` | 216 | 0.6840 | 0.6852 | 0.6852 | 0.1375 | 0.0922 | 0.0504 |
| `notebook_exact_44_plus_duration` | 45 | 0.6716 | 0.6730 | 0.6730 | 0.1467 | 0.0939 | 0.0531 |
| `notebook_exact_44` | 44 | 0.6690 | 0.6704 | 0.6704 | 0.1496 | 0.0941 | 0.0504 |
| `notebook_exact_44_noclip` | 44 | 0.6690 | 0.6704 | 0.6704 | 0.1496 | 0.0941 | 0.0504 |
| `notebook_exact_44_dynrange` | 44 | 0.6683 | 0.6697 | 0.6697 | 0.1494 | 0.0942 | 0.0509 |
| `relative_lowfreq_shape_features` | 107 | 0.6290 | 0.6325 | 0.6325 | 0.0970 | 0.0769 | 0.0486 |
| `lowfreq_relative_temporal` | 148 | 0.6143 | 0.6190 | 0.6190 | 0.0927 | 0.0746 | 0.0437 |
| `temporal_lowfreq_shape_features` | 41 | 0.2179 | 0.2175 | 0.2175 | 0.0339 | 0.0390 | 0.0274 |
