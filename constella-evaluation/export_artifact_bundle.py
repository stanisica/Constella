"""Export a reviewer-facing bundle for the paper's empirical artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict


LABEL_ORDER = ["extra-small", "small", "medium", "large", "extra-large"]
APPROACH_ORDER = ["Constella", "Naive", "Traditional"]
PAPER_MODEL_NAMES = {
    "alexnet": "alexnet",
    "squeezenet1_0": "squeezenet1.0",
    "resnet50": "resnet50",
    "swin_b": "swin-b",
    "efficientnet_b0": "efficientnet-b0",
}
RESULTS_FILES = [
    "constella_results.csv",
    "benchmark_timing_summary.csv",
    "benchmark_timing_raw.csv",
    "plot_cost_success_tradeoff.pdf",
    "plot_latency.pdf",
    "plot_energy.pdf",
    "benchmark_timing.pdf",
    "benchmark_timing_per_decision.pdf",
]


def repo_root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def get_results_dir(base_dir: str) -> str:
    return os.environ.get(
        "CONSTELLA_RESULTS_DIR",
        os.path.join(base_dir, "artifact-output", "paper-results"),
    )


def load_json(path: str):
    with open(path) as handle:
        return json.load(handle)


def load_results_csv(results_dir: str):
    path = os.path.join(results_dir, "constella_results.csv")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def copy_results(results_dir: str, output_dir: str) -> None:
    for filename in RESULTS_FILES:
        src = os.path.join(results_dir, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing generated results file: {src}")
        dst = os.path.abspath(os.path.join(output_dir, filename))
        if os.path.abspath(src) == dst:
            continue
        shutil.copy2(src, dst)


def copy_model_layers(base_dir: str, output_dir: str) -> None:
    source_dir = os.environ.get(
        "CONSTELLA_MODEL_LAYERS_DIR",
        os.path.join(base_dir, "model-layers"),
    )
    target_dir = os.path.join(output_dir, "model-layers")
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if not filename.endswith(".json"):
            continue
        src = os.path.abspath(os.path.join(source_dir, filename))
        dst = os.path.abspath(os.path.join(target_dir, filename))
        if src == dst:
            continue
        shutil.copy2(src, dst)


def get_model_layers_dir(base_dir: str) -> str:
    return os.environ.get(
        "CONSTELLA_MODEL_LAYERS_DIR",
        os.path.join(base_dir, "model-layers"),
    )


def get_layer_count(base_dir: str, model_name: str) -> int:
    path = os.path.join(get_model_layers_dir(base_dir), f"{model_name}.json")
    with open(path) as handle:
        return len(json.load(handle))


def write_table1_csv(base_dir: str, output_dir: str) -> None:
    scenario = load_json(os.path.join(base_dir, "scenarios", "scenario_constella.json"))
    cfg = load_json(os.path.join(base_dir, "scenarios", "config_base.json"))

    path = os.path.join(output_dir, "table1_scenarios_and_parameters.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "paper_model_name",
                "artifact_model_name",
                "layer_count",
                "I_total",
                "X_total",
                "Y_total",
                "R_max_bps",
                "p_Wh_per_FLOP",
                "q_Wh_per_bit",
                "alpha",
                "beta",
                "E_processor_Wh",
                "E_comm_Wh",
                "T_comp_s",
                "T_comm_s",
                "T_idle_s",
                "delta_t_s",
            ]
        )
        for item in scenario["configs"]:
            writer.writerow(
                [
                    item["label"],
                    PAPER_MODEL_NAMES[item["model"]],
                    item["model"],
                    get_layer_count(base_dir, item["model"]),
                    item["I_total"],
                    item["X_total"],
                    item["Y_total"],
                    cfg["R_max"],
                    cfg["p"],
                    cfg["q"],
                    cfg["alpha"],
                    cfg["beta"],
                    cfg["E_processor"],
                    cfg["E_comm"],
                    cfg["T_comp"],
                    cfg["T_comm"],
                    cfg["T_idle"],
                    cfg["delta_t"],
                ]
            )


def write_claims_csv(rows, output_dir: str) -> None:
    by_approach = defaultdict(list)
    for row in rows:
        by_approach[row["approach"]].append(row)

    for approach_rows in by_approach.values():
        approach_rows.sort(key=lambda row: LABEL_ORDER.index(row["label"]))

    constella_rows = by_approach["Constella"]
    naive_rows = by_approach["Naive"]
    traditional_rows = by_approach["Traditional"]

    path = os.path.join(output_dir, "paper_claims_summary.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim", "value", "source"])

        min_success = min(float(row["success_rate"]) for row in constella_rows)
        writer.writerow(
            [
                "Constella minimum success rate across scenarios",
                f"{min_success:.4f}",
                "Section 4.2, Figure 2, constella_results.csv",
            ]
        )

        cost_ratios = []
        latency_ratios = []
        energy_ratios = []
        for idx, row in enumerate(constella_rows):
            c_cost = float(row["cost"])
            c_lat = float(row["mean_latency"])
            c_energy = float(row["total_energy"])
            for baseline in (naive_rows[idx], traditional_rows[idx]):
                cost_ratios.append(float(baseline["cost"]) / c_cost)
                latency_ratios.append(float(baseline["mean_latency"]) / c_lat)
                energy_ratios.append(float(baseline["total_energy"]) / c_energy)

        writer.writerow(
            [
                "Maximum cost reduction factor vs baselines",
                f"{max(cost_ratios):.2f}",
                "Abstract, Section 4.2, Figure 2, constella_results.csv",
            ]
        )
        writer.writerow(
            [
                "Maximum latency reduction factor vs baselines",
                f"{max(latency_ratios):.2f}",
                "Abstract, Section 4.2, Figure 3, constella_results.csv",
            ]
        )
        writer.writerow(
            [
                "Maximum energy reduction factor vs baselines",
                f"{max(energy_ratios):.2f}",
                "Section 4.2, Figure 4, constella_results.csv",
            ]
        )


def write_manifest(output_dir: str) -> None:
    path = os.path.join(output_dir, "MANIFEST.md")
    with open(path, "w") as handle:
        handle.write(
            "# Constella paper artifact bundle\n\n"
            "This directory contains the generated data and figures for the paper's\n"
            "empirical results.\n\n"
            "## Command\n\n"
            "`./reproduce_paper_artifacts.sh [output_dir]`\n\n"
            "## Paper mapping\n\n"
            "- Table 1: `table1_scenarios_and_parameters.csv`\n"
            "- Figure 2: `plot_cost_success_tradeoff.pdf`\n"
            "- Figure 3: `plot_latency.pdf`\n"
            "- Figure 4: `plot_energy.pdf`\n"
            "- Figure 5: `benchmark_timing.pdf`\n"
            "- Supplemental per-decision timing: `benchmark_timing_per_decision.pdf`\n"
            "- Figure 5 source data: `benchmark_timing_summary.csv` and `benchmark_timing_raw.csv`\n"
            "- Generated model-layer profiles: `model-layers/*.json`\n"
            "- Per-scenario raw metrics: `constella_results.csv`\n"
            "- Derived headline claims: `paper_claims_summary.csv`\n"
        )


def write_provenance(output_dir: str) -> None:
    metadata = {
        "source_files": {
            "scenario_config": "scenarios/scenario_constella.json",
            "base_config": "scenarios/config_base.json",
            "model_layer_data": "model-layers/*.json generated during artifact reproduction",
            "model_layer_generator": "constella-evaluation/generate_model_layers.py",
            "evaluation_script": "constella-evaluation/evaluate_constella.py",
            "benchmark_script": "constella-evaluation/benchmark_timing.py",
            "plot_script": "constella-evaluation/plot_constella.py",
        },
        "models": [
            "alexnet",
            "squeezenet1_0",
            "resnet50",
            "swin_b",
            "efficientnet_b0",
        ],
    }
    path = os.path.join(output_dir, "provenance.json")
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Export generated paper artifacts")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_dir = repo_root()
    results_dir = get_results_dir(base_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rows = load_results_csv(results_dir)
    rows.sort(
        key=lambda row: (
            LABEL_ORDER.index(row["label"]),
            APPROACH_ORDER.index(row["approach"]),
        )
    )

    copy_results(results_dir, output_dir)
    copy_model_layers(base_dir, output_dir)
    write_table1_csv(base_dir, output_dir)
    write_claims_csv(rows, output_dir)
    write_manifest(output_dir)
    write_provenance(output_dir)


if __name__ == "__main__":
    main()
