# Contributions and scope

Maintained by [Rafael Ode Brino](https://github.com/odebrino).

## Documented repository work

The [reproducibility revision](https://github.com/odebrino/biodcase-ml/commit/f07385f19faf5402389845ac06efbec9453e9af2) provides an inspectable record of:

- A pinned CPU environment, maintained/legacy test separation, and Python 3.12/3.14 CI.
- A generated-audio example exercising the real manifest, feature, KNN, and report path.
- Training-based baseline selection and portable manifest-derived report generation.
- Reconciliation of featured results with available artifacts, task/split labels, and domain-shift limitations.
- Corrected source-data access and attribution instructions.

These concrete changes can be reviewed in that commit and reproduced through the README commands. They describe repository and reproducibility work, rather than a statement of exclusive authorship of the original study.

## Scientific implementation and external work

The maintained code implements event preparation, audio features, training-fitted transformations, KNN selection/evaluation, and per-class/domain diagnostics. [Results](RESULTS.md) identifies the stored experiments and their limits; [experiment history](EXPERIMENT_HISTORY.md) distinguishes historical artifacts.

The BioDCASE/ATBFL dataset and upstream numerical/ML methods have their own creators. [Data provenance](DATA_ACCESS.md) and [NOTICE](../NOTICE) distinguish those external materials from the repository's MIT-licensed original software. Git history records code changes; it does not by itself establish the original division of research responsibilities.
