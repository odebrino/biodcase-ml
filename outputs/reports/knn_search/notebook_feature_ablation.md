# Notebook Feature Ablation

This compares Projet.ipynb-inspired low-frequency classical features using train-only CV.
The official held-out validation split is not used for this ranking.
Ranking mode: `StratifiedKFold CV first`.

| feature_set | dim | random_acc | random_macro_f1 | domain_acc | domain_macro_f1 | worst_domain_acc | worst_domain_macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `handcrafted_stats` | 49 | 0.7276 | 0.7277 | 0.1952 | 0.1393 | 0.0999 | 0.1126 |
| `lowfreq_all_plus_waveform_spectral` | 277 | 0.7006 | 0.7004 | 0.2076 | 0.1450 | 0.0918 | 0.1034 |
| `waveform_spectral_stats` | 61 | 0.6976 | 0.6976 | 0.1966 | 0.1438 | 0.0972 | 0.1198 |
| `lowfreq_all_plus_logmel` | 448 | 0.6886 | 0.6875 | 0.1880 | 0.1333 | 0.0950 | 0.1032 |
| `lowfreq_all` | 216 | 0.6074 | 0.6061 | 0.1550 | 0.1105 | 0.0788 | 0.0890 |
| `lowfreq_all_plus_mfcc` | 376 | 0.5957 | 0.5948 | 0.1589 | 0.1180 | 0.0797 | 0.0832 |
| `notebook_lowfreq_band_features` | 68 | 0.6036 | 0.5945 | 0.1565 | 0.1172 | 0.0963 | 0.1025 |
| `relative_lowfreq_shape_features` | 107 | 0.5746 | 0.5741 | 0.1347 | 0.1020 | 0.0801 | 0.0897 |
| `lowfreq_relative_temporal` | 148 | 0.5694 | 0.5696 | 0.1264 | 0.0971 | 0.0761 | 0.0811 |
| `temporal_lowfreq_shape_features` | 41 | 0.2070 | 0.2082 | 0.0484 | 0.0481 | 0.0437 | 0.0380 |
