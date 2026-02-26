"""
Generate plots and tables from experiment results.

Reads runs/clf/all_results.json and produces:
  - Accuracy vs Data Size line plot (3 scenarios)
  - Training Time vs Data Size plot
  - Summary table printed to console

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results runs/clf/all_results.json --out_dir runs/clf/plots
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIO_STYLE = {
    "real":  {"color": "#2196F3", "marker": "o", "label": "Real only"},
    "synth": {"color": "#FF9800", "marker": "s", "label": "Synth only"},
    "both":  {"color": "#4CAF50", "marker": "^", "label": "Real + Synth"},
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def group_by_scenario(results):
    grouped = {}
    for r in results:
        grouped.setdefault(r["scenario"], []).append(r)
    for v in grouped.values():
        v.sort(key=lambda x: x["n_per_class"] if isinstance(x["n_per_class"], int) else 99999)
    return grouped


def plot_accuracy(grouped, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, runs in grouped.items():
        style = SCENARIO_STYLE[scenario]
        sizes = [r["n_per_class"] for r in runs]
        accs = [r["test_accuracy"] * 100 for r in runs]
        ax.plot(sizes, accs, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.set_xlabel("Training images per class", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy vs Training Data Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([r["n_per_class"] for r in list(grouped.values())[0]])
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_size.png", dpi=150)
    print(f"Saved: {out_dir / 'accuracy_vs_size.png'}")
    plt.close(fig)


def plot_time(grouped, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, runs in grouped.items():
        style = SCENARIO_STYLE[scenario]
        sizes = [r["n_per_class"] for r in runs]
        times = [r["train_time_sec"] for r in runs]
        ax.plot(sizes, times, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.set_xlabel("Training images per class", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Computational Cost vs Training Data Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([r["n_per_class"] for r in list(grouped.values())[0]])
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_size.png", dpi=150)
    print(f"Saved: {out_dir / 'time_vs_size.png'}")
    plt.close(fig)


def plot_per_class_f1(grouped, out_dir):
    """Bar chart: per-class F1 at the largest data size for each scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (scenario, runs) in zip(axes, grouped.items()):
        r = runs[-1]  # largest size
        classes = list(r["per_class"].keys())
        f1s = [r["per_class"][c]["f1"] * 100 for c in classes]
        colors = ["#EF5350", "#FFEE58", "#FFA726"]
        ax.bar(classes, f1s, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{SCENARIO_STYLE[scenario]['label']}\n(n={r['n_per_class']}/class)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_ylabel("F1 Score (%)" if ax == axes[0] else "")
        for i, v in enumerate(f1s):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    fig.suptitle("Per-Class F1 Score (Largest Data Size)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'per_class_f1.png'}")
    plt.close(fig)


def print_table(grouped):
    print("\n" + "=" * 75)
    print(f"{'Size':>6} | {'Real Acc':>9} | {'Synth Acc':>10} | {'Both Acc':>9} | {'Real(s)':>8} | {'Synth(s)':>9} | {'Both(s)':>8}")
    print("-" * 75)

    sizes = sorted(set(r["n_per_class"] for runs in grouped.values() for r in runs))
    lookup = {(r["scenario"], r["n_per_class"]): r for runs in grouped.values() for r in runs}

    for n in sizes:
        vals = []
        for s in ["real", "synth", "both"]:
            r = lookup.get((s, n))
            if r:
                vals.append(f"{r['test_accuracy']*100:>8.2f}%")
            else:
                vals.append("     N/A")
        times = []
        for s in ["real", "synth", "both"]:
            r = lookup.get((s, n))
            if r:
                times.append(f"{r['train_time_sec']:>7.1f}s")
            else:
                times.append("    N/A")
        print(f"{n:>6} | {vals[0]:>9} | {vals[1]:>10} | {vals[2]:>9} | {times[0]:>8} | {times[1]:>9} | {times[2]:>8}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="runs/clf/all_results.json")
    parser.add_argument("--out_dir", type=str, default="runs/clf/plots")
    args = parser.parse_args()

    results = load_results(args.results)
    grouped = group_by_scenario(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy(grouped, out_dir)
    plot_time(grouped, out_dir)
    plot_per_class_f1(grouped, out_dir)
    print_table(grouped)


if __name__ == "__main__":
    main()
