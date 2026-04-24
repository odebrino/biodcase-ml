# Notebook Reproduction Summary

This report uses only the official `train` split. The on-disk `validation/` official held-out split is not used.

## Default Elephant Island 2014 Audit

- leaky diagnostic accuracy: `0.6205`
- leaky diagnostic macro-F1: `0.5844`
- leaky diagnostic weighted-F1: `0.6128`
- split-safe CV accuracy: `0.6291`
- split-safe CV macro-F1: `0.5971`
- split-safe CV weighted-F1: `0.6225`

The leaky audit intentionally follows the notebook-like scaler-before-split pattern and is diagnostic only.

## Per-Dataset Results

| dataset | mode | rows | accuracy | accuracy_mean | macro_f1 | macro_f1_mean | weighted_f1 | weighted_f1_mean | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| casey2014 | notebook_exact_leaky_audit | 6866 | 0.9272 | nan | 0.4913 | nan | 0.9149 | nan | ok |
| casey2014 | notebook_exact_split_safe | 6866 | nan | 0.9189 | nan | 0.4775 | nan | 0.9066 | ok |
| kerguelen2005 | notebook_exact_leaky_audit | 2960 | 0.6554 | nan | 0.6465 | nan | 0.6515 | nan | ok |
| kerguelen2005 | notebook_exact_split_safe | 2960 | nan | 0.6642 | nan | 0.6535 | nan | 0.6606 | ok |
| maudrise2014 | notebook_exact_leaky_audit | 2360 | 0.9767 | nan | 0.7196 | nan | 0.9767 | nan | ok |
| maudrise2014 | notebook_exact_split_safe | 2360 | nan | 0.9750 | nan | 0.6842 | nan | 0.9741 | ok |
| elephantisland2013 | notebook_exact_leaky_audit | 21913 | 0.6035 | nan | 0.5540 | nan | 0.5933 | nan | ok |
| elephantisland2013 | notebook_exact_split_safe | 21913 | nan | 0.6121 | nan | 0.5662 | nan | 0.6033 | ok |
| elephantisland2014 | notebook_exact_leaky_audit | 20957 | 0.6205 | nan | 0.5844 | nan | 0.6128 | nan | ok |
| elephantisland2014 | notebook_exact_split_safe | 20957 | nan | 0.6291 | nan | 0.5971 | nan | 0.6225 | ok |
| ballenyislands2015 | notebook_exact_leaky_audit | 2222 | 0.6674 | nan | 0.5136 | nan | 0.6596 | nan | ok |
| ballenyislands2015 | notebook_exact_split_safe | 2222 | nan | 0.6498 | nan | 0.5089 | nan | 0.6431 | ok |
| greenwich2015 | notebook_exact_leaky_audit | 1128 | 0.8805 | nan | 0.4042 | nan | 0.8668 | nan | ok |
| greenwich2015 | notebook_exact_split_safe | 1128 | nan | 0.9060 | nan | 0.4455 | nan | 0.8915 | ok |

## Combined Train Results

| task_mode | dataset | rows | accuracy_mean | macro_precision_mean | macro_f1_mean | weighted_f1_mean | feature_dimension | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3class_notebook_cv | all_train_datasets | 58510 | 0.5786 | 0.5711 | 0.5553 | 0.5705 | 44 | ok |
| 7class_strict_cv | all_train_datasets | 58510 | 0.4864 | 0.4704 | 0.4412 | 0.4751 | 44 | ok |

## Best Per-Dataset Split-Safe Result

- dataset: `maudrise2014`
- accuracy: `0.9750`
- weighted-F1: `0.9741`
- macro-F1: `0.6842`

## Notebook 0.89 Interpretation

A reproduced 0.89-like score here should be interpreted as an internal train-split/CV result, not as official held-out performance.
Any scaler-before-split result is marked diagnostic-only and is not eligible for model selection.
