"""Plots for Constella unified evaluation."""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RESULTS_DIR = "results"
LABEL_ORDER = ["extra-small", "small", "medium", "large", "extra-large"]
LABEL_SHORT = {"extra-small": "extra-small", "small": "small", "medium": "medium",
               "large": "large", "extra-large": "extra-large"}

APPROACHES = ["Constella", "Naive", "Traditional"]
DISPLAY_NAMES = {"Constella": "Constella", "Naive": "NB", "Traditional": "TB"}
COLORS = {"Constella": "C0", "Naive": "C3", "Traditional": "C4"}
APPROACH_MARKERS = {"Constella": "o", "Naive": "s", "Traditional": "v"}
SCENARIO_MARKERS = {
    "extra-small": "s", "small": "^", "medium": "o",
    "large": "D", "extra-large": "P",
}

plt.rcParams.update({
    "font.size": 11, "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 150,
})


def load_csv():
    path = os.path.join(RESULTS_DIR, "constella_results.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, name),
                format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {name}")


def plot_cost_success_tradeoff():
    """Scatter with lines: x=cost, y=success_rate, color=approach, shape=scenario.
    Points of the same approach are connected by lines across scenarios."""
    rows = load_csv()
    fig, ax = plt.subplots(figsize=(6, 2.5))

    for approach in APPROACHES:
        subset = sorted(
            [r for r in rows if r["approach"] == approach],
            key=lambda r: LABEL_ORDER.index(r["label"]),
        )
        costs = [float(r["cost"]) for r in subset]
        successes = [float(r["success_rate"]) for r in subset]
        labels = [r["label"] for r in subset]

        ax.plot(costs, successes,
                color=COLORS[approach], linewidth=1.2, linestyle="--",
                alpha=0.5, zorder=1)

        for cost, success, label in zip(costs, successes, labels):
            ax.scatter(cost, success, s=90, zorder=3,
                       color=COLORS[approach],
                       marker=SCENARIO_MARKERS[label],
                       edgecolors="black", linewidths=0.5)

    approach_handles = [
        mlines.Line2D([], [], color=COLORS[a], marker=APPROACH_MARKERS[a],
                      linestyle="--", linewidth=1.2, markersize=6,
                      markeredgecolor="black", markeredgewidth=0.5,
                      label=DISPLAY_NAMES[a])
        for a in APPROACHES
    ]
    scenario_handles = [
        mlines.Line2D([], [], color="gray", marker=SCENARIO_MARKERS[l],
                      linestyle="None", markersize=6,
                      markeredgecolor="black", markeredgewidth=0.5, label=l)
        for l in LABEL_ORDER
    ]

    all_handles = approach_handles + [mlines.Line2D([], [], linestyle="None")] + scenario_handles
    ax.legend(handles=all_handles, loc="center left",
              fontsize=6, borderaxespad=0.5, bbox_to_anchor=(0, 0.35))

    ax.set_xlabel("Cost")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.1, 1.10)
    ax.set_xscale("log")
    fig.tight_layout()
    save(fig, "plot_cost_success_tradeoff.pdf")


def plot_latency():
    """Line plot: x=scenario, y=mean latency, one line per approach."""
    rows = load_csv()
    fig, ax = plt.subplots(figsize=(6, 2.5))

    for approach in APPROACHES:
        subset = sorted(
            [r for r in rows if r["approach"] == approach],
            key=lambda r: LABEL_ORDER.index(r["label"]),
        )
        labels = [LABEL_SHORT[r["label"]] for r in subset]
        latencies = [float(r["mean_latency"]) for r in subset]
        ax.plot(labels, latencies,
                marker=APPROACH_MARKERS[approach], linewidth=1.5,
                linestyle="--", color=COLORS[approach],
                label=DISPLAY_NAMES[approach])

    ax.set_xlabel("Scenario", fontsize=9)
    ax.set_ylabel("Time (s)", fontsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, "plot_latency.pdf")


def plot_energy():
    """Grouped bar chart: x=scenario, y=total energy (Wh), log scale."""
    rows = load_csv()
    fig, ax = plt.subplots(figsize=(6, 2.5))

    n_approaches = len(APPROACHES)
    bar_width = 0.25
    x = np.arange(len(LABEL_ORDER))

    for i, approach in enumerate(APPROACHES):
        subset = sorted(
            [r for r in rows if r["approach"] == approach],
            key=lambda r: LABEL_ORDER.index(r["label"]),
        )
        energies = [float(r["total_energy"]) for r in subset]
        offset = (i - (n_approaches - 1) / 2) * bar_width
        # Replace 0 with None so log scale skips them (Traditional has 0 energy)
        plot_energies = [e if e > 0 else None for e in energies]
        ax.bar(x + offset, [e if e is not None else 0 for e in plot_energies],
               bar_width, color=COLORS[approach], label=DISPLAY_NAMES[approach],
               edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_SHORT[l] for l in LABEL_ORDER], fontsize=9)
    ax.set_xlabel("Scenario", fontsize=9)
    ax.set_ylabel("Energy (Wh)", fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "plot_energy.pdf")


if __name__ == "__main__":
    plot_cost_success_tradeoff()
    plot_latency()
    plot_energy()
