# Constella

This repository is the artifact for the paper "Constella: A Novel
Framework for Cost-Efficient Distributed AI Inference in LEO Space Data
Centers". It reproduces the selected empirical results from the paper: Table 1,
Figures 2-5, and the headline claims on success rate, deployment cost, latency,
energy consumption, and OCRI/LIA execution overhead.

This README is intended to serve as the artifact evaluation overview document
and can be exported to PDF for submission.

## Overview

Constella combines OCRI, an offline mixed-integer resource identifier, with LIA,
an online latency-aware ISL assignment algorithm. The artifact is self-contained
source code and runs as user-level software on a standard Linux environment. It
does not require root access, administrator privileges, GPUs, HPC resources,
proprietary software, external datasets, or pretrained model weights.

The paper states that constellation parameters are derived from the BUPT-1
dataset and that hardware efficiency follows NVIDIA Jetson Orin Nano
measurements. For the efficiency of artifact evaluation, the derived constants are encoded
in `scenarios/config_base.json`, thus there is no need for downloading the raw dataset. DNN layer profiles are regenerated locally from torchvision
architectures with `weights=None`.

## Getting Started Guide

Use Python 3.12 or a compatible Python 3 version. The artifact was validated in
a conda environment with Python 3.12.2 and these package versions:

- `mip==1.17.2`
- `numpy==2.4.2`
- `matplotlib==3.10.8`
- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchinfo==1.8.0`

Create a user-level virtual environment from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If using conda, create or activate a Python 3.12 environment and install the same
requirements:

```bash
conda create -n constella-aep python=3.12
conda activate constella-aep
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import mip, numpy, matplotlib, torch, torchvision, torchinfo; print('OK')"
python constella-evaluation/generate_model_layers.py --models all --check
```

The layer-profile check should report the paper model layer counts: AlexNet 23,
SqueezeNet 1.0 67, ResNet50 182, Swin-B 311, and EfficientNet-B0 329.

## Step-by-Step Reproduction

Run the full paper reproduction pipeline from the repository root:

```bash
./reproduce_paper_artifacts.sh
```

An alternate output directory may be supplied:

```bash
./reproduce_paper_artifacts.sh /tmp/constella-paper-results
```

The script performs the complete paper workflow:

1. Regenerates model-layer profiles for AlexNet, SqueezeNet 1.0, ResNet50,
   Swin-B, and EfficientNet-B0 using torchinfo and input shape `(3, 224, 224)`.
2. Runs the Constella, Naive Baseline (NB), and Traditional Baseline (TB)
   evaluation.
3. Runs the 50-iteration OCRI/LIA timing benchmark.
4. Regenerates the paper figures.
5. Exports the reviewer bundle to `artifact-output/paper-results/`.

For a shorter functional check, run only the main evaluation:

```bash
python constella-evaluation/evaluate_constella.py
```

For optional supplemental robustness checks beyond the paper figures:

```bash
./run_extended_evaluation.sh
```

The extended evaluation keeps the five paper scenario sizes fixed and varies
additional DNN architectures. Its output is written to
`artifact-output/extended-evaluation/` and is not required to reproduce the
paper.

## Output Mapping

| Paper item | Command | Generated output | Validation criterion |
| --- | --- | --- | --- |
| Table 1: scenarios and constellation parameters | `./reproduce_paper_artifacts.sh` | `artifact-output/paper-results/table1_scenarios_and_parameters.csv` | Scenario sizes, model names, `|L|`, and shared parameters match Table 1. |
| Figure 2: deployment cost vs. success rate | `./reproduce_paper_artifacts.sh` | `artifact-output/paper-results/plot_cost_success_tradeoff.pdf` and `constella_results.csv` | Constella reaches minimum success `0.8192` and one to two orders of magnitude lower cost. |
| Figure 3: mean inference latency | `./reproduce_paper_artifacts.sh` | `artifact-output/paper-results/plot_latency.pdf` and `constella_results.csv` | Maximum mean-latency reduction is about `2.7x`. |
| Figure 4: energy consumption per orbit | `./reproduce_paper_artifacts.sh` | `artifact-output/paper-results/plot_energy.pdf` and `constella_results.csv` | Maximum energy reduction is about `74x`; medium is the documented exception where TB uses less energy. |
| Figure 5: mean execution time per orbit | `./reproduce_paper_artifacts.sh` | `artifact-output/paper-results/benchmark_timing.pdf`, `benchmark_timing_summary.csv`, and `benchmark_timing_raw.csv` | OCRI and LIA remain lightweight; exact wall-clock values may vary by machine. |
| Supplemental timing detail | `./reproduce_paper_artifacts.sh` | `artifact-output/paper-results/benchmark_timing_per_decision.pdf` | Per-decision LIA overhead remains small; this plot is not a paper figure. |
| Headline paper claims | `./reproduce_paper_artifacts.sh` | `artifact-output/paper-results/paper_claims_summary.csv` | Values match the expected claim table below. |
| Provenance and manifest | `./reproduce_paper_artifacts.sh` | `artifact-output/paper-results/provenance.json` and `MANIFEST.md` | Files identify source scripts, models, and generated outputs. |

Artifact model identifiers use Python names (`squeezenet1_0`, `swin_b`,
`efficientnet_b0`). The exported Table 1 CSV also includes the paper display
names (`squeezenet1.0`, `swin-b`, `efficientnet-b0`).

## Expected Results and Validation

After reproduction, inspect:

```bash
cat artifact-output/paper-results/paper_claims_summary.csv
```

Expected paper-level values are:

| Claim | Expected value |
| --- | ---: |
| Constella minimum success rate across scenarios | `0.8192` |
| Maximum cost reduction factor vs. baselines | about `204.82x` |
| Maximum latency reduction factor vs. baselines | about `2.68x` |
| Maximum energy reduction factor vs. baselines | about `74.01x` |

Additional checks tied directly to the paper text:

- In the small scenario, NB success is `0.1000`.
- In the medium scenario, mean latency is approximately Constella `1541 s`, NB
  `4129 s`, and TB `3518 s`.
- In the large scenario, total energy is approximately Constella `43.0 Wh`, NB
  `3183.6 Wh`, and TB `1191.0 Wh`.
- In the extra-large scenario, OCRI selects `Y = 0`, so LIA has zero routing
  decisions and zero measured LIA time.

Timing results are wall-clock measurements and should not be compared by exact
equality. The paper reports OCRI runtimes of `3.93-23.35 ms`, cumulative LIA
time of `0.04-4.19 ms` per orbit, and combined Constella overhead under `28 ms`.
While different machines may produce slightly different values, the overal expected behavior
is that both OCRI and LIA remain low-overhead compared to baselines.

## Execution Time and Resources

The full reproduction process and installation steps should complete within one
hour on a recent workstation or cluster compute node with network access for
Python packages. It is noteworthy that no container, VM, root privileges, or GPU access is required.

Reference validation platform:

- Operating system: macOS 26.3.1
- Python environment: conda `py312`
- Python: 3.12.2
- Hardware: standard CPU-only workstation
- Observed full reproduction time: < 1min

## Artifact Size and Packaging

The artifact is self-contained. No additional datasets are downloaded during
execution. At the time of this check, the repository is approximately 2.1 MB and
the generated `artifact-output/` directory is approximately 156 KB. The submission archive should include source code, scripts, `requirements.txt`,
scenario files, model-layer reference files, and this README. Exclude local
environments and caches such as `.venv/`, conda environments, `__pycache__/`,
`.artifact-cache/`, and package-manager caches.

## Artifact Files

- `reproduce_paper_artifacts.sh`: main paper reproduction command.
- `run_experiment.sh`: compatibility wrapper for the main command.
- `mip_solver.py`: OCRI MILP implementation.
- `simulate.py` and `orbital_model.py`: constellation simulation and routing.
- `constella-evaluation/generate_model_layers.py`: torchinfo-based model-layer
  profiler.
- `constella-evaluation/evaluate_constella.py`: paper evaluation metrics.
- `constella-evaluation/benchmark_timing.py`: OCRI/LIA timing benchmark.
- `constella-evaluation/plot_constella.py`: paper plot generation.
- `constella-evaluation/export_artifact_bundle.py`: reviewer bundle exporter.
- `scenarios/*.json`: paper scenarios and simulation parameters.
- `model-layers/*.json`: checked-in reference model-layer profiles.

