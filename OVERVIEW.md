# Constella Artifact Overview

This document is the source for the Euro-Par 2026 artifact overview document.
It describes how to install, execute, validate, and interpret the Constella
artifact for the paper "Constella: A Novel Framework for Cost-Efficient
Distributed AI Inference in LEO Space Data Centers".

The artifact reproduces selected empirical results from the paper: Table 1,
Figures 2-5, and the headline claims on success rate, deployment cost, latency,
energy consumption, and OCRI/LIA execution overhead.

## 1. Getting Started Guide

### Supported Platform

The supported artifact evaluation path is user-level Python software on a
standard Linux environment. No root access, administrator privileges, GPUs, HPC
resources, proprietary software, external datasets, or pretrained model weights
are required.

Validated Linux platform:

- Operating system: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- Architecture: `x86_64`
- Python used to create the artifact environment: 3.12.13
- Artifact environment Python: 3.12.13
- Generated `artifact-output/paper-results/` size: 324 KB
- Validation result: `1117 passed, 0 failed`

Other Python versions may work, but they are not part of the supported
evaluation path. The artifact should be evaluated with Python 3.12.

### Dependencies

The pinned dependencies are:

- `mip==1.17.1`
- `numpy==2.2.4`
- `matplotlib==3.10.0`
- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchinfo==1.8.0`

These are installed from `requirements.txt` into a repo-local virtual
environment named `.constella-venv`.

### Installation

From the repository root, create the artifact environment:

```bash
./scripts/create_env.sh
```

This script requires `python3.12` to be available on `PATH`. It creates
`.constella-venv`, upgrades `pip`, and installs the pinned dependencies. If the
environment already exists, the script exits without modifying it.

To recreate the environment:

```bash
./scripts/remove_env.sh
./scripts/create_env.sh
```

If Python 3.12 is provided by a user-level tool such as micromamba, activate or
expose that Python before running `scripts/create_env.sh`. Micromamba is not a
dependency of the artifact.

### Environment Verification

Run:

```bash
./scripts/check_env.sh
```

This verifies that required Python packages import correctly and that local
torchvision/torchinfo profiling reproduces the checked-in model-layer profiles.
The expected layer counts are:

| Model | Expected layers |
| --- | ---: |
| AlexNet | 23 |
| SqueezeNet 1.0 | 67 |
| ResNet50 | 182 |
| Swin-B | 311 |
| EfficientNet-B0 | 329 |

The setup and verification steps should complete within one hour on a recent
Linux workstation or compute node with network access for Python packages.

## 2. Step-by-Step Instructions to Reproduce Results

### Full Paper Reproduction

Run:

```bash
./reproduce_paper_artifacts.sh
```

By default, the generated bundle is written to:

```text
artifact-output/paper-results/
```

An alternate output directory may be supplied:

```bash
./reproduce_paper_artifacts.sh /tmp/constella-paper-results
```

The full workflow performs five steps:

1. Regenerates model-layer profiles for AlexNet, SqueezeNet 1.0, ResNet50,
   Swin-B, and EfficientNet-B0 using torchinfo and input shape `(3, 224, 224)`.
2. Runs the Constella, Naive Baseline (NB), and Traditional Baseline (TB)
   evaluation.
3. Runs the 50-iteration OCRI/LIA timing benchmark.
4. Regenerates the paper figures.
5. Exports the reviewer bundle.

To record the exact end-to-end runtime on a reviewer machine, run:

```bash
time ./reproduce_paper_artifacts.sh
```

The Linux validation run completed successfully, but its full wall-clock runtime
was not separately recorded. The generated `artifact-output/paper-results/`
directory was 324 KB.

### Reduced Functional Check

After `scripts/check_env.sh`, a shorter functional check can run the main
evaluation without regenerating all plots and benchmark artifacts:

```bash
.constella-venv/bin/python constella-evaluation/evaluate_constella.py
```

This produces or updates `artifact-output/paper-results/constella_results.csv`
and prints the per-scenario comparison table. It exercises the core OCRI,
simulation, and baseline evaluation code, but it does not replace the full
paper reproduction command.

### Validation

After the full reproduction command, run:

```bash
./scripts/validate_results.sh
```

For a custom output directory:

```bash
./scripts/validate_results.sh /tmp/constella-paper-results
```

The validator compares generated outputs against
`constella-evaluation/expected_results.json` and writes:

```text
artifact-output/paper-results/validation_report.json
```

A successful run prints:

```text
Status: PASS
Checks: 1117 passed, 0 failed, 1117 total
```

The validator checks:

- Required output files and generated model-layer files.
- CSV schemas, row counts, scenario names, model names, and approach names.
- Table 1 scenario metadata and shared simulation parameters.
- Headline paper claims.
- Scenario-level cost, success, latency, energy, split-layer, and `X`/`Y` values.
- Regenerated model-layer counts and JSON structure.
- Timing CSV structure, iteration coverage, nonnegative timings, and the
  extra-large `Y = 0` zero-LIA invariant.

Timing values are wall-clock measurements and are not checked by exact equality.
The validator checks structural and qualitative timing properties that should
hold across machines.

### Output Mapping

The paper-result files below are generated by `./reproduce_paper_artifacts.sh`.
The validation report is generated afterward by `./scripts/validate_results.sh`.

| Paper item | Generated output | How to interpret |
| --- | --- | --- |
| Table 1 | `table1_scenarios_and_parameters.csv` | Scenario sizes, model names, layer counts, and shared simulation parameters should match the paper. |
| Figure 2 | `plot_cost_success_tradeoff.pdf`, `constella_results.csv` | Constella maintains at least `0.8192` success while reducing cost substantially. |
| Figure 3 | `plot_latency.pdf`, `constella_results.csv` | Maximum mean-latency reduction is about `2.68x`. |
| Figure 4 | `plot_energy.pdf`, `constella_results.csv` | Maximum energy reduction is about `74.01x`; the medium scenario is the documented exception where TB uses less energy. |
| Figure 5 | `benchmark_timing.pdf`, `benchmark_timing_summary.csv`, `benchmark_timing_raw.csv` | OCRI and LIA remain lightweight. Exact timing values may differ by machine. |
| Supplemental timing | `benchmark_timing_per_decision.pdf` | Per-decision LIA overhead remains small; this is not a paper figure. |
| Headline claims | `paper_claims_summary.csv` | Contains the derived paper-level claims checked by the validator. |
| Reviewer manifest | `MANIFEST.md` | Lists the generated bundle contents and paper mapping. |
| Provenance | `provenance.json` | Identifies source scripts, models, and generated-output provenance. |
| Validation | `validation_report.json` | Records pass/fail status and all validation checks. |

### Expected Headline Results

| Claim | Expected value |
| --- | ---: |
| Constella minimum success rate across scenarios | `0.8192` |
| Maximum cost reduction factor vs. baselines | `204.82x` |
| Maximum latency reduction factor vs. baselines | `2.68x` |
| Maximum energy reduction factor vs. baselines | `74.01x` |

Additional checks tied to the paper include:

- In the small scenario, NB success is `0.1000`.
- In the medium scenario, mean latency is approximately Constella `1541 s`, NB
  `4129 s`, and TB `3518 s`.
- In the large scenario, total energy is approximately Constella `43.0 Wh`, NB
  `3183.6 Wh`, and TB `1191.0 Wh`.
- In the extra-large scenario, OCRI selects `Y = 0`, so LIA has zero routing
  decisions and zero measured LIA time.

### Artifact Size and Data Limits

The artifact is self-contained. No additional datasets are downloaded during
execution. The submission package should exclude local virtual environments,
generated outputs, caches, and local paper drafts.

Exclude:

- `.constella-venv/`
- `.venv/`
- `artifact-output/`
- `__pycache__/`
- `.artifact-cache/`
- package-manager caches
- `Constella.pdf`

The generated Linux `artifact-output/paper-results/` directory was 324 KB. The
large `du -sh .` value observed on the Linux machine included local environment
and cache directories and is not the artifact package size.

## 3. Artifact Files

- `reproduce_paper_artifacts.sh`: main paper reproduction command.
- `run_experiment.sh`: compatibility wrapper for the main command.
- `scripts/create_env.sh`: creates `.constella-venv`.
- `scripts/check_env.sh`: verifies dependencies and model-layer regeneration.
- `scripts/remove_env.sh`: removes `.constella-venv`.
- `scripts/validate_results.sh`: validates generated results.
- `mip_solver.py`: OCRI MILP implementation.
- `simulate.py` and `orbital_model.py`: constellation simulation and routing.
- `constella-evaluation/generate_model_layers.py`: torchinfo-based profiler.
- `constella-evaluation/evaluate_constella.py`: paper evaluation metrics.
- `constella-evaluation/benchmark_timing.py`: OCRI/LIA timing benchmark.
- `constella-evaluation/plot_constella.py`: paper plot generation.
- `constella-evaluation/export_artifact_bundle.py`: reviewer bundle exporter.
- `constella-evaluation/validate_results.py`: artifact validation checks.
- `constella-evaluation/expected_results.json`: expected paper-result values.
- `scenarios/*.json`: paper scenarios and simulation parameters.
- `model-layers/*.json`: checked-in reference model-layer profiles.
