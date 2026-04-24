# Notebook Feature Ablation

This compares Projet.ipynb-inspired low-frequency classical features using train-only CV.
The official held-out validation split is not used for this ranking.
Ranking mode: `StratifiedKFold CV first`.

| feature_set | dim | random_acc | random_weighted_f1 | random_macro_f1 | domain_acc | domain_macro_f1 | worst_domain_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lowfreq_all_plus_waveform_spectral` | 277 | 0.8040 | 0.8055 | 0.8055 | 0.5322 | 0.2232 | 0.0589 |
| `class_region_lowfreq_features` | 38 | 0.7750 | 0.7761 | 0.7761 | 0.2895 | 0.1321 | 0.0589 |
| `notebook_lowfreq_band_features` | 68 | 0.7600 | 0.7625 | 0.7625 | 0.4504 | 0.2314 | 0.0589 |
| `notebook_exact_44_plus_regions` | 82 | 0.7593 | 0.7612 | 0.7612 | 0.3864 | 0.2140 | 0.0589 |
| `lowfreq_all` | 216 | 0.7480 | 0.7506 | 0.7506 | 0.3733 | 0.2018 | 0.0589 |
| `notebook_exact_44_plus_duration` | 45 | 0.7303 | 0.7327 | 0.7327 | 0.4363 | 0.2290 | 0.0589 |
| `notebook_exact_26` | 26 | 0.7253 | 0.7297 | 0.7297 | 0.4573 | 0.2357 | 0.0589 |
| `lowfreq_relative_temporal` | 148 | 0.7247 | 0.7283 | 0.7283 | 0.2389 | 0.1067 | 0.0455 |
| `notebook_exact_44` | 44 | 0.7237 | 0.7260 | 0.7260 | 0.4414 | 0.2293 | 0.0589 |
| `notebook_exact_44_noclip` | 44 | 0.7237 | 0.7260 | 0.7260 | 0.4414 | 0.2293 | 0.0589 |
| `notebook_exact_44_dynrange` | 44 | 0.7237 | 0.7260 | 0.7260 | 0.4414 | 0.2293 | 0.0589 |
| `relative_lowfreq_shape_features` | 107 | 0.7223 | 0.7248 | 0.7248 | 0.2501 | 0.1135 | 0.0589 |
| `temporal_lowfreq_shape_features` | 41 | 0.4023 | 0.4021 | 0.4021 | 0.0832 | 0.0508 | 0.0529 |
