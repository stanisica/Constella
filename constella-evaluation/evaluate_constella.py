"""Constella evaluation: compares Constella, Naive Baseline, and Traditional Baseline."""

import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils import load_config, load_layers, solve_ocri
import simulate as _sim_module
from simulate import simulate

RAW_IMAGE_D = 3 * 224 * 224 * 4  # 3×224×224×float32 in bytes (matching layer D units)

APPROACHES = ["Constella", "Naive", "Traditional"]

CSV_COLUMNS = [
    "label", "model", "approach", "X", "Y", "l", "W", "D", "cost",
    "success_rate", "mean_latency", "median_latency", "max_latency",
    "p95_latency", "mean_proc_energy", "mean_comm_energy", "total_energy",
    "ocri_time", "lia_avg_time",
]

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

def heuristic_config(layers, X_total, Y_total):
    """Pick the middle layer, use all available satellites."""
    l_star = len(layers) // 2 + 1
    return X_total, Y_total, l_star


def run_approach(approach, layers, I_total, X_total, Y_total, cfg):
    """Return (X, Y, l, W, D, metrics, ocri_time, lia_avg_time) for a given approach."""
    ocri_time = 0.0
    lia_avg_time = 0.0

    if approach == "Constella":
        t0 = time.perf_counter()
        X, Y, l = solve_ocri(layers, I_total, X_total, Y_total, cfg)
        ocri_time = time.perf_counter() - t0
        W, D = layers[l - 1]
        reset_lia_timer()
        metrics = simulate(X, Y, l, W, D, I_total, cfg, "LIA")
        lia_avg_time = get_lia_avg_time()

    elif approach == "Naive":
        X, Y, l = heuristic_config(layers, X_total, Y_total)
        W, D = layers[l - 1]
        metrics = simulate(X, Y, l, W, D, I_total, cfg, "Static")

    elif approach == "Traditional":
        X, Y, l = X_total, 0, 0
        W, D = 0, RAW_IMAGE_D
        metrics = simulate(X, Y, l, W, D, I_total, cfg, "DirectOnly")

    return X, Y, l, W, D, metrics, ocri_time, lia_avg_time


def evaluate():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(base_dir)
    alpha, beta = cfg["alpha"], cfg["beta"]

    scenario_path = os.path.join(base_dir, "scenarios", "scenario_constella.json")
    with open(scenario_path) as f:
        scenario = json.load(f)

    # Warmup: pay MIP solver initialization cost before timing
    warmup_layers = load_layers(base_dir, scenario["configs"][0]["model"])
    solve_ocri(warmup_layers, 1.0, 1, 1, cfg)

    results = []

    for config in scenario["configs"]:
        label = config["label"]
        model_name = config["model"]
        I_total = float(config["I_total"])
        X_total = int(config["X_total"])
        Y_total = int(config["Y_total"])
        layers = load_layers(base_dir, model_name)

        print(f"\n{'='*60}")
        print(f"  {label} ({model_name}): I={I_total}, X_total={X_total}, Y_total={Y_total}")
        print(f"{'='*60}")

        for approach in APPROACHES:
            X, Y, l, W, D, metrics, ocri_time, lia_avg_time = run_approach(
                approach, layers, I_total, X_total, Y_total, cfg
            )
            cost = alpha * X + beta * Y

            print(f"  {approach:<16} X={X:>5} Y={Y:>5} l={l:>3} "
                  f"cost={cost:>8.1f} success={metrics.success_rate:.4f} "
                  f"mean_lat={metrics.mean_latency:.2f}")

            results.append({
                "label": label,
                "model": model_name,
                "approach": approach,
                "X": X,
                "Y": Y,
                "l": l,
                "W": W,
                "D": D,
                "cost": f"{cost:.1f}",
                "success_rate": f"{metrics.success_rate:.4f}",
                "mean_latency": f"{metrics.mean_latency:.2f}",
                "median_latency": f"{metrics.median_latency:.2f}",
                "max_latency": f"{metrics.max_latency:.2f}",
                "p95_latency": f"{metrics.p95_latency:.2f}",
                "mean_proc_energy": f"{metrics.mean_proc_energy:.6f}",
                "mean_comm_energy": f"{metrics.mean_comm_energy:.6f}",
                "total_energy": f"{metrics.total_energy:.6f}",
                "ocri_time": f"{ocri_time:.6f}",
                "lia_avg_time": f"{lia_avg_time:.9f}",
            })

    # Sort by scenario label, then approach order
    label_order = ["extra-small", "small", "medium", "large", "extra-large"]
    results.sort(key=lambda r: (
        label_order.index(r["label"]) if r["label"] in label_order else 999,
        APPROACHES.index(r["approach"]),
    ))

    # Print summary table
    print(f"\n{'='*120}")
    col_w = [14, 16, 6, 6, 4, 10, 10, 12, 12, 12]
    header = (
        f"{'label':<{col_w[0]}} {'approach':<{col_w[1]}} "
        f"{'X':>{col_w[2]}} {'Y':>{col_w[3]}} {'l':>{col_w[4]}} "
        f"{'cost':>{col_w[5]}} {'success':>{col_w[6]}} "
        f"{'mean_lat':>{col_w[7]}} {'p95_lat':>{col_w[8]}} "
        f"{'total_E':>{col_w[9]}}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<{col_w[0]}} {r['approach']:<{col_w[1]}} "
            f"{r['X']:>{col_w[2]}} {r['Y']:>{col_w[3]}} {r['l']:>{col_w[4]}} "
            f"{r['cost']:>{col_w[5]}} {r['success_rate']:>{col_w[6]}} "
            f"{r['mean_latency']:>{col_w[7]}} {r['p95_latency']:>{col_w[8]}} "
            f"{r['total_energy']:>{col_w[9]}}"
        )

    # Save CSV
    csv_path = os.path.join(out_dir, "constella_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    evaluate()
