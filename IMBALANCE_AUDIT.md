# Class Imbalance Audit

Outcome: B) "Imbalance was causing a reporting/evaluation problem; this is now fixed."

## Conclusion
Class imbalance is not currently making the best evaluation misleading: accuracy=0.9301, balanced_accuracy=0.9022, macro_f1=0.8866, weighted_f1=0.9309.
The previous risk was reporting: accuracy/weighted metrics and raw counts could be read without enough macro, per-class, normalized-confusion, and baseline context.

## Evidence
- Audited run: `outputs/runs/20260421-223457`.
- Train/eval event leakage: 0; audio leakage: 0; dataset overlap: [].
- Validation support includes all configured classes: True.
- Normalized confusion rows are true-label row-normalized: True.
- Split imbalance ratios vs source: {'train': {'imbalance_ratio': 11.335839598997493, 'more_imbalanced_than_source': True}, 'validation': {'imbalance_ratio': 9.244837758112094, 'more_imbalanced_than_source': False}}.
- Best model beats majority and stratified-random baselines on macro F1 and balanced accuracy.

## Where The Risk Was
The split is dataset-provided rather than randomly stratified, and training is intentionally more imbalanced than the combined source. This is a domain-generalization split, not an accidental stratification bug. The repository's legacy `validation` split name denotes the official held-out test domains (`casey2017`, `kerguelen2014`, `kerguelen2015`), not a generic validation set.
Training already uses class weights and does not rebalance held-out test data. No additional training mitigation is justified by the current best run.

## What Changed
- Evaluation now saves balanced accuracy, macro precision/recall, weighted precision/recall, baseline metrics, normalized confusion CSV, confidence analysis, PR curve data, and top confusion pairs.
- The audit now saves split distributions, leakage checks, artifact checks, baseline comparisons, and this Markdown conclusion.

## Most Affected Classes
- `bmb`: recall=0.7286, f1=0.7891, support=2277.
- `bmz`: recall=0.9444, f1=0.7565, support=918.
- `bpd`: recall=0.7537, f1=0.7725, support=678.

## Headline Metrics Going Forward
Use macro F1, balanced accuracy, per-class recall/F1, and row-normalized confusion matrix with accuracy/weighted F1 as secondary context.

## Monitoring
Monitor class distributions, leakage counts, macro-vs-weighted metric gaps, minority-class recall, and top confusion pairs for every run.
