"""Render a comparison from stored CV artifacts; no model is retrained."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/validation-gap.png")
    args = parser.parse_args()
    source = ROOT / "outputs/reports/knn_search/best_cv_metrics.json"
    candidate = json.loads(source.read_text())["best_candidate"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]):
        values = [candidate[f"random_cv_{metric}_mean"], candidate[f"domain_cv_{metric}_mean"]]
        errors = [candidate[f"random_cv_{metric}_std"], candidate[f"domain_cv_{metric}_std"]]
        bars = ax.bar(["Stratified CV", "Domain-aware CV"], values, yerr=errors,
                      color=["#3276a5", "#ce772a"], capsize=5, width=0.55)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.11, f"{value:.3f}",
                    ha="center", fontsize=11)
    axes[0].set_ylabel("Mean fold score")
    fig.suptitle("One stored 7-class KNN configuration, two validation schemes", fontsize=13)
    fig.supxlabel("Development diagnostics · error bars: fold SD · no challenge evaluation score", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Figure generated from {source.relative_to(ROOT)}: {args.output}")


if __name__ == "__main__":
    main()
