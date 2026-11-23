"""Render portfolio figures from archived metrics and curves."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SPLITS = {"random": "Random stratified", "chronological": "Chronological"}
MODELS = {
    "neural": "Neural network", "logistic": "Logistic regression", "knn": "k-NN",
    "calendar_schedule": "Calendar schedule", "majority": "Majority class",
}
COLORS = ("#17658A", "#B14B27")


def load(name):
    return json.loads((DOCS / name).read_text(encoding="utf8"))


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.svg", metadata={"Date": None})
    fig.savefig(directory / f"{name}.png", dpi=250)
    plt.close(fig)


def plot_benchmark(directory):
    result = load("final-benchmark.json")
    if result["status"] != "complete":
        raise ValueError("Expected a completed held-out benchmark.")
    splits = {entry["split"]: entry for entry in result["results"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), layout="constrained")
    row = np.arange(len(MODELS))
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("A  Accuracy", "B  Macro F1")):
        for index, (split, label) in enumerate(SPLITS.items()):
            values = [splits[split]["models"][key]["test"][metric] for key in MODELS]
            bars = ax.barh(row + (-0.18 if index == 0 else 0.18), values,
                           height=0.30, color=COLORS[index], label=label,
                           hatch=None if index == 0 else "//", edgecolor=COLORS[index])
            ax.bar_label(bars, labels=[f"{x:.1%}" if metric == "accuracy" else f"{x:.3f}" for x in values],
                         padding=3, fontsize=8.5)
        ax.set(yticks=row, yticklabels=list(MODELS.values()), xlim=(0, 1.03), title=title)
        ax.invert_yaxis()
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        if metric == "accuracy":
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.12), ncols=2, frameon=False)
    fig.suptitle("Held-out workload classification", fontsize=16, fontweight="bold")
    save(fig, directory, "test-split-comparison")


def plot_ablation(directory):
    short = load("encoding-experiment.json")
    long = load("encoding-extended.json")
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), layout="constrained")
    for row, method in enumerate(SPLITS):
        long_run = next(r for r in long["results"] if r["split"] == method)
        short_run = next(r for r in short["results"]
                         if (r["split"], r["features"], r["model"]) == (method, "electrical", "neural"))
        score_ax, loss_ax = axes[row]
        for index, (arm, label) in enumerate((("A", "A · 15 inputs"), ("C", "C · 12 inputs"))):
            early = short_run["arms"][arm]
            extended = long_run["arms"][arm]
            color, style = COLORS[index], ("-" if index == 0 else "--")
            scores = [[], []]
            curves = []
            for before, after in zip(early, extended, strict=True):
                if before["configuration"]["seed"] != after["configuration"]["seed"]:
                    raise ValueError("Ablation seeds do not match across budgets.")
                values = [fit["validation"]["macro_f1"] for fit in (before, after)]
                for bucket, value in zip(scores, values):
                    bucket.append(value)
                score_ax.plot([600, 2000], values, style, color=color, alpha=0.22, marker="o", markersize=3)
                curve = after["convergence"]["validation_loss"]
                curves.append(curve)
                loss_ax.plot(np.arange(500, len(curve) + 1), curve[499:], style, color=color, alpha=0.22)
            score_ax.plot([600, 2000], [mean(s) for s in scores], style, color=color,
                          marker="o" if index == 0 else "s", linewidth=2, label=label)
            common = min(map(len, curves))
            loss_ax.plot(np.arange(500, common + 1),
                         np.median([curve[499:common] for curve in curves], axis=0),
                         style, color=color, linewidth=2)
        score_ax.set(title=f"{SPLITS[method]} · validation macro F1",
                     xticks=[600, 2000], xlim=(480, 2120), xlabel="Epoch budget", ylabel="Macro F1")
        loss_ax.set(title=f"{SPLITS[method]} · validation loss",
                    xlim=(500, 2000), xlabel="Epoch", ylabel="Cross-entropy")
        loss_ax.axvline(600, linestyle=":", color="#4C5258", linewidth=1)
        for ax in (score_ax, loss_ax):
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)
    axes[0, 0].legend(loc="lower center", bbox_to_anchor=(1.06, 1.12), ncols=2, frameon=False)
    fig.suptitle("Neural one-hot ablation across training budgets", fontsize=16, fontweight="bold")
    save(fig, directory, "encoding-budget-comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DOCS / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "steel-load-portfolio"})
    plot_benchmark(args.output_dir)
    plot_ablation(args.output_dir)


if __name__ == "__main__":
    main()

