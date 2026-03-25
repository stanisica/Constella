"""Benchmark OCRI and LIA execution times over multiple iterations.

Runs evaluate_constella's Constella approach N times per scenario,
computes average OCRI and LIA simulation times, and plots them.

"""

import argparse
import json
import os
import sys
import time
import statistics

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils import load_config, load_layers, solve_ocri
import simulate as _sim_module
from simulate import simulate

# ---------------------------------------------------------------------------
# Instrument _decide_lia to measure per-call wall-clock time
# ---------------------------------------------------------------------------
_original_decide_lia = _sim_module._decide_lia
_lia_total_time = 0.0
_lia_call_count = 0


def _timed_decide_lia(*args, **kwargs):
    global _lia_total_time, _lia_call_count
    t0 = time.perf_counter()
    result = _original_decide_lia(*args, **kwargs)
    _lia_total_time += time.perf_counter() - t0
    _lia_call_count += 1
    return result


_sim_module._decide_lia = _timed_decide_lia


def reset_lia_timer():
    global _lia_total_time, _lia_call_count
    _lia_total_time = 0.0
    _lia_call_count = 0


def get_lia_avg_time():
    if _lia_call_count == 0:
        return 0.0
    return _lia_total_time / _lia_call_count


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(iterations):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cfg = load_config(base_dir)

    scenario_path = os.path.join(base_dir, "scenarios", "scenario_constella.json")
    with open(scenario_path) as f:
        scenario = json.load(f)

    # Warmup MIP solver
    print("Warming up MIP solver...")
    warmup_layers = load_layers(base_dir, scenario["configs"][0]["model"])
    solve_ocri(warmup_layers, 1.0, 1, 1, cfg)
    print("Warmup done.\n")

    labels = []
    ocri_avgs = []
    lia_avgs = []
    lia_decision_avgs = []

    for config in scenario["configs"]:
        label = config["label"]
        model_name = config["model"]
        I_total = float(config["I_total"])
        X_total = int(config["X_total"])
        Y_total = int(config["Y_total"])
        layers = load_layers(base_dir, model_name)

        print(f"{'=' * 60}")
        print(f"  {label} ({model_name}): I={I_total:.0f}, X={X_total}, Y={Y_total}")
        print(f"  Running {iterations} iterations...")
        print(f"{'=' * 60}")

        ocri_times = []
        lia_times = []
        lia_per_decision = []

        for i in range(iterations):
            # OCRI timing
            t0 = time.perf_counter()
            X, Y, l = solve_ocri(layers, I_total, X_total, Y_total, cfg)
            ocri_elapsed = time.perf_counter() - t0

            W, D = layers[l - 1]

            # LIA timing: use total accumulated _decide_lia time (0 if Y=0)
            reset_lia_timer()
            simulate(X, Y, l, W, D, I_total, cfg, "LIA", seed=42 + i)
            lia_elapsed = _lia_total_time
            lia_avg_decision = get_lia_avg_time()

            ocri_times.append(ocri_elapsed)
            lia_times.append(lia_elapsed)
            lia_per_decision.append(lia_avg_decision)

            print(f"  [{i+1:>{len(str(iterations))}}/{iterations}] "
                  f"OCRI: {ocri_elapsed*1000:8.2f} ms   "
                  f"LIA total: {lia_elapsed*1000:8.2f} ms   "
                  f"LIA/decision: {lia_avg_decision*1e6:8.2f} µs")

        ocri_mean = statistics.mean(ocri_times)
        lia_mean = statistics.mean(lia_times)
        lia_dec_mean = statistics.mean(lia_per_decision)
        ocri_std = statistics.stdev(ocri_times) if iterations > 1 else 0
        lia_std = statistics.stdev(lia_times) if iterations > 1 else 0
        lia_dec_std = statistics.stdev(lia_per_decision) if iterations > 1 else 0

        labels.append(label)
        ocri_avgs.append(ocri_mean * 1000)  # convert to ms
        lia_avgs.append(lia_mean * 1000)    # convert to ms
        lia_decision_avgs.append(lia_dec_mean)

        print(f"  => OCRI avg: {ocri_mean*1000:.2f} ms (std: {ocri_std*1000:.2f} ms)")
        print(f"  => LIA  avg: {lia_mean*1000:.2f} ms (std: {lia_std*1000:.2f} ms)")
        print(f"  => LIA  avg/decision: {lia_dec_mean*1e6:.2f} µs (std: {lia_dec_std*1e6:.2f} µs)\n")

    # --- Plot 1: total times (linear scale) ---
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 2.4))
    x = range(len(labels))

    total_avgs = [o + l for o, l in zip(ocri_avgs, lia_avgs)]

    ax.plot(x, total_avgs, marker='o', linewidth=1.5, linestyle='--',
            label='Constella', color='C0')
    ax.plot(x, ocri_avgs, marker='s', linewidth=1.5, linestyle='--',
            label='OCRI', color='green')
    ax.plot(x, lia_avgs, marker='^', linewidth=1.5, linestyle='--',
            label='LIA', color='orange')

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Scenario", fontsize=9)
    ax.set_ylabel("Time (ms)", fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "benchmark_timing.pdf")
    fig.savefig(plot_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    lia_decision_ms = [t * 1000 for t in lia_decision_avgs]
    constella_decision_ms = [o + l for o, l in zip(ocri_avgs, lia_decision_ms)]

    fig2, ax2 = plt.subplots(figsize=(6, 2.4))

    ax2.plot(x, constella_decision_ms, marker='o', linewidth=1.5, linestyle='--',
             label='Constella', color='C0')
    ax2.plot(x, ocri_avgs, marker='s', linewidth=1.5, linestyle='--',
             label='OCRI', color='green')
    ax2.plot(x, lia_decision_ms, marker='^', linewidth=1.5, linestyle='--',
             label='LIA', color='orange')

    ax2.set_yscale("log")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("Scenario", fontsize=9)
    ax2.set_ylabel("Time per decision (ms)", fontsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    plot_path2 = os.path.join(out_dir, "benchmark_timing_per_decision.pdf")
    fig2.savefig(plot_path2, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    print(f"Saved plot to {plot_path2}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OCRI and LIA execution times"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=10,
        help="Number of iterations per scenario (default: 10)"
    )
    args = parser.parse_args()
    benchmark(args.iterations)


if __name__ == "__main__":
    main()
