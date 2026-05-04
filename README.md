# Constella

This repository contains the artifact for the paper "Constella: A Novel
Framework for Cost-Efficient Distributed AI Inference in LEO Space Data
Centers". It reproduces the selected empirical results from the paper: Table 1,
Figures 2-5, and the headline claims on success rate, deployment cost, latency,
energy consumption, and OCRI/LIA execution overhead.

The artifact is self-contained source code. It runs as user-level software with
Python 3.12 and does not require root access, GPUs, HPC resources, proprietary
software, external datasets, or pretrained model weights.

For the reviewer-facing overview source, see `OVERVIEW.md`.

## What Is Reproduced

The workflow regenerates DNN layer profiles, evaluates Constella against the
Naive Baseline (NB) and Traditional Baseline (TB), benchmarks OCRI/LIA timing,
renders the paper plots, exports a reviewer bundle, and validates the generated
outputs against expected paper-result values.

The paper states that constellation parameters are derived from the BUPT-1
dataset and that hardware efficiency follows NVIDIA Jetson Orin Nano
measurements. For artifact evaluation, the derived constants are encoded in
`scenarios/config_base.json`; no raw dataset download is required. DNN layer
profiles are regenerated locally from torchvision architectures with
`weights=None`.

## Requirements

Supported evaluation path:

- Ubuntu 22.04.5 LTS on `x86_64`
- Python 3.12
- User-level virtual environment created by `scripts/create_env.sh`

Other Python versions may work, but they are not part of the supported artifact
evaluation path.

The pinned Python dependencies are listed in `requirements.txt`:

- `mip==1.17.1`
- `numpy==2.2.4`
- `matplotlib==3.10.0`
- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchinfo==1.8.0`

## Setup

From the repository root, create the repo-local environment:

```bash
./scripts/create_env.sh
```

This creates `.constella-venv`, upgrades `pip`, and installs the pinned
dependencies from `requirements.txt`. If `.constella-venv` already exists, the
script exits without modifying it. To rebuild from scratch:

```bash
./scripts/remove_env.sh
./scripts/create_env.sh
```

Verify the environment:

```bash
./scripts/check_env.sh
```

The check imports the required Python packages and verifies that local
torchvision/torchinfo layer profiling reproduces the checked-in model-layer
profiles:

- AlexNet: 23 layers
- SqueezeNet 1.0: 67 layers
- ResNet50: 182 layers
- Swin-B: 311 layers
- EfficientNet-B0: 329 layers

If Python 3.12 is available through a user-level tool such as micromamba,
activate or expose that Python before running `scripts/create_env.sh`. The
artifact itself does not depend on micromamba.

## Reproduce Paper Artifacts

Run the full paper reproduction pipeline:

```bash
./reproduce_paper_artifacts.sh
```

By default, outputs are written to:

```text
artifact-output/paper-results/
```

An alternate output directory may be supplied:

```bash
./reproduce_paper_artifacts.sh /tmp/constella-paper-results
```

The script performs the complete paper workflow:

1. Regenerates model-layer profiles for AlexNet, SqueezeNet 1.0, ResNet50,
   Swin-B, and EfficientNet-B0 using torchinfo and input shape `(3, 224, 224)`.
2. Runs the Constella, NB, and TB evaluation.
3. Runs the 50-iteration OCRI/LIA timing benchmark.
4. Regenerates the paper figures.
5. Exports the reviewer bundle.

For a shorter functional check after `scripts/check_env.sh`, run only the main
evaluation:

```bash
.constella-venv/bin/python constella-evaluation/evaluate_constella.py
```

## Validate Results

After reproduction, validate the generated bundle:

```bash
./scripts/validate_results.sh
```

For a custom output directory:

```bash
./scripts/validate_results.sh /tmp/constella-paper-results
```

The validator checks required files, CSV schemas, scenario and approach
coverage, deterministic paper metrics with tight tolerances, Table 1 metadata,
model-layer counts and JSON structure, and timing-output invariants. It does
not compare wall-clock timing values for exact equality.

Validation writes a JSON report to:

```text
artifact-output/paper-results/validation_report.json
```

A successful Linux validation run printed:

```text
Status: PASS
Checks: 1117 passed, 0 failed, 1117 total
```

## Output Mapping

The paper-result files below are produced by `./reproduce_paper_artifacts.sh`.
The validation report is produced afterward by `./scripts/validate_results.sh`.

| Paper item | Generated output | Validation criterion |
| --- | --- | --- |
| Table 1: scenarios and constellation parameters | `table1_scenarios_and_parameters.csv` | Scenario sizes, model names, layer counts, and shared parameters match the paper table. |
| Figure 2: deployment cost vs. success rate | `plot_cost_success_tradeoff.pdf`, `constella_results.csv` | Constella reaches minimum success `0.8192` and substantially lower cost. |
| Figure 3: mean inference latency | `plot_latency.pdf`, `constella_results.csv` | Maximum mean-latency reduction is about `2.68x`. |
| Figure 4: energy consumption per orbit | `plot_energy.pdf`, `constella_results.csv` | Maximum energy reduction is about `74.01x`; the medium scenario is the documented exception where TB uses less energy. |
| Figure 5: mean execution time per orbit | `benchmark_timing.pdf`, `benchmark_timing_summary.csv`, `benchmark_timing_raw.csv` | OCRI and LIA remain lightweight; exact wall-clock values may vary by machine. |
| Supplemental timing detail | `benchmark_timing_per_decision.pdf` | Per-decision LIA overhead remains small; this plot is not a paper figure. |
| Headline claims | `paper_claims_summary.csv` | Values match the expected claims encoded in `constella-evaluation/expected_results.json`. |
| Provenance and manifest | `provenance.json`, `MANIFEST.md` | Files identify source scripts, models, and generated outputs. |
| Validation report | `validation_report.json` | Records pass/fail status and all validation checks. |

## Expected Headline Results

The validator checks these headline values:

| Claim | Expected value |
| --- | ---: |
| Constella minimum success rate across scenarios | `0.8192` |
| Maximum cost reduction factor vs. baselines | `204.82x` |
| Maximum latency reduction factor vs. baselines | `2.68x` |
| Maximum energy reduction factor vs. baselines | `74.01x` |

Additional deterministic checks include scenario-level cost, success, latency,
energy, selected split layer, selected `X`/`Y`, Table 1 parameters, and model
layer counts.

Timing results are wall-clock measurements and should not be compared by exact
equality. The validator checks timing file structure, row coverage, nonnegative
timings, and the extra-large `Y = 0` invariant that gives zero LIA routing time.

## Validated Platform

The current Linux validation run used:

- Operating system: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- Architecture: `x86_64`
- Python used to create `.constella-venv`: 3.12.13
- `.constella-venv` Python: 3.12.13
- Observed full reproduction time: 29.560 s
- Observed validation time: 0.029 s
- Generated paper-results directory size: 324 KB
- Validation result: `1117 passed, 0 failed`

To record fresh wall-clock times on another machine, run:

```bash
time ./reproduce_paper_artifacts.sh
time ./scripts/validate_results.sh
```

## Artifact Size and Packaging

The artifact source package should include source code, shell scripts,
`requirements.txt`, scenario files, model-layer reference files, `README.md`,
and `OVERVIEW.md`.

Exclude generated or local-only files and directories:

- `.constella-venv/`
- `.venv/`
- `artifact-output/`
- `__pycache__/`
- `.artifact-cache/`
- package-manager caches
- local paper drafts such as `Constella.pdf`

No additional datasets are downloaded during execution.

## File Guide

- `reproduce_paper_artifacts.sh`: main paper reproduction command.
- `run_experiment.sh`: compatibility wrapper for the main command.
- `scripts/create_env.sh`: creates `.constella-venv`.
- `scripts/check_env.sh`: verifies dependencies and model-layer regeneration.
- `scripts/remove_env.sh`: removes `.constella-venv`.
- `scripts/validate_results.sh`: validates generated results.
- `mip_solver.py`: OCRI MILP implementation.
- `simulate.py` and `orbital_model.py`: constellation simulation and routing.
- `constella-evaluation/generate_model_layers.py`: torchinfo-based model-layer profiler.
- `constella-evaluation/evaluate_constella.py`: paper evaluation metrics.
- `constella-evaluation/benchmark_timing.py`: OCRI/LIA timing benchmark.
- `constella-evaluation/plot_constella.py`: paper plot generation.
- `constella-evaluation/export_artifact_bundle.py`: reviewer bundle exporter.
- `constella-evaluation/validate_results.py`: artifact validation checks.
- `constella-evaluation/expected_results.json`: machine-readable expected values.
- `scenarios/*.json`: paper scenarios and simulation parameters.
- `model-layers/*.json`: checked-in reference model-layer profiles.
