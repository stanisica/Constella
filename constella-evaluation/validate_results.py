"""Validate generated Constella artifact results against expected outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "artifact-output", "paper-results")
DEFAULT_EXPECTED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "expected_results.json"
)

CONSTELLA_COLUMNS = [
    "label",
    "model",
    "approach",
    "X",
    "Y",
    "l",
    "W",
    "D",
    "cost",
    "success_rate",
    "mean_latency",
    "median_latency",
    "max_latency",
    "p95_latency",
    "mean_proc_energy",
    "mean_comm_energy",
    "total_energy",
    "ocri_time",
    "lia_avg_time",
]

TIMING_SUMMARY_COLUMNS = [
    "label",
    "model",
    "iterations",
    "ocri_mean_ms",
    "ocri_std_ms",
    "lia_total_mean_ms",
    "lia_total_std_ms",
    "lia_per_decision_mean_us",
    "lia_per_decision_std_us",
]

TIMING_RAW_COLUMNS = [
    "label",
    "model",
    "iteration",
    "ocri_ms",
    "lia_total_ms",
    "lia_per_decision_us",
]

CLAIM_COLUMNS = ["claim", "value", "source"]


class Validator:
    def __init__(self, results_dir: str, expected: dict[str, Any]):
        self.results_dir = os.path.abspath(results_dir)
        self.expected = expected
        self.checks: list[dict[str, Any]] = []

    def pass_check(self, group: str, name: str, detail: str = "") -> None:
        self.checks.append(
            {"group": group, "name": name, "status": "pass", "detail": detail}
        )

    def fail_check(self, group: str, name: str, detail: str) -> None:
        self.checks.append(
            {"group": group, "name": name, "status": "fail", "detail": detail}
        )

    def tolerance(self, name: str | None) -> float:
        tolerances = self.expected["tolerances"]
        if not name:
            return float(tolerances["default_abs"])
        return float(tolerances[name])

    def path(self, filename: str) -> str:
        return os.path.join(self.results_dir, filename)

    def read_csv(self, filename: str) -> list[dict[str, str]]:
        with open(self.path(filename), newline="") as handle:
            return list(csv.DictReader(handle))

    def assert_close(
        self,
        group: str,
        name: str,
        actual: Any,
        expected: Any,
        tolerance_name: str | None = None,
    ) -> None:
        tolerance = self.tolerance(tolerance_name)
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (TypeError, ValueError):
            self.fail_check(group, name, f"not numeric: actual={actual!r}")
            return

        if math.isclose(actual_float, expected_float, abs_tol=tolerance, rel_tol=0.0):
            self.pass_check(
                group,
                name,
                f"actual={actual_float:g}, expected={expected_float:g}, tolerance={tolerance:g}",
            )
            return

        self.fail_check(
            group,
            name,
            f"actual={actual_float:g}, expected={expected_float:g}, tolerance={tolerance:g}",
        )

    def assert_equal(self, group: str, name: str, actual: Any, expected: Any) -> None:
        if str(actual) == str(expected):
            self.pass_check(group, name, f"actual={actual}")
            return
        self.fail_check(group, name, f"actual={actual!r}, expected={expected!r}")

    def validate_required_files(self) -> None:
        for filename in self.expected["required_files"]:
            if os.path.exists(self.path(filename)):
                self.pass_check("files", filename, "present")
            else:
                self.fail_check("files", filename, "missing")

        for model_name in self.expected["layer_counts"]:
            filename = os.path.join("model-layers", f"{model_name}.json")
            if os.path.exists(self.path(filename)):
                self.pass_check("files", filename, "present")
            else:
                self.fail_check("files", filename, "missing")

    def validate_constella_results(self) -> None:
        rows = self.read_csv("constella_results.csv")
        expected_rows = self.expected["constella_results"]
        expected_count = len(self.expected["label_order"]) * len(self.expected["approaches"])

        self.assert_equal("structure", "constella_results row count", len(rows), expected_count)
        self.validate_columns("structure", "constella_results columns", rows, CONSTELLA_COLUMNS)

        by_key = {f"{row['label']}|{row['approach']}": row for row in rows}
        self.assert_equal("structure", "constella_results unique keys", len(by_key), len(rows))

        for key, expected_row in expected_rows.items():
            row = by_key.get(key)
            if row is None:
                self.fail_check("results", key, "missing result row")
                continue
            self.pass_check("results", key, "row present")

            for field in ("model", "X", "Y", "l", "W", "D"):
                self.assert_equal("results", f"{key}.{field}", row[field], expected_row[field])

            for field in ("cost", "mean_latency", "p95_latency", "total_energy"):
                self.assert_close(
                    "results",
                    f"{key}.{field}",
                    row[field],
                    expected_row[field],
                    "rounded_abs" if field in ("mean_latency", "p95_latency") else None,
                )

            self.assert_close(
                "results",
                f"{key}.success_rate",
                row["success_rate"],
                expected_row["success_rate"],
                "rate_abs",
            )

    def validate_claims(self) -> None:
        rows = self.read_csv("paper_claims_summary.csv")
        self.validate_columns("structure", "paper_claims_summary columns", rows, CLAIM_COLUMNS)
        by_claim = {row["claim"]: row for row in rows}

        for claim, expected in self.expected["paper_claims"].items():
            row = by_claim.get(claim)
            if row is None:
                self.fail_check("claims", claim, "missing claim row")
                continue
            self.assert_close("claims", claim, row["value"], expected["value"], expected["tolerance"])

    def validate_table1(self) -> None:
        rows = self.read_csv("table1_scenarios_and_parameters.csv")
        expected_rows = self.expected["table1"]
        shared = self.expected["shared_parameters"]
        self.assert_equal("structure", "table1 row count", len(rows), len(expected_rows))
        by_scenario = {row["scenario"]: row for row in rows}

        for expected_row in expected_rows:
            scenario = expected_row["scenario"]
            row = by_scenario.get(scenario)
            if row is None:
                self.fail_check("table1", scenario, "missing scenario row")
                continue
            self.pass_check("table1", scenario, "row present")

            for field, expected_value in expected_row.items():
                self.assert_equal("table1", f"{scenario}.{field}", row[field], expected_value)

            for field, expected_value in shared.items():
                self.assert_close("table1", f"{scenario}.{field}", row[field], expected_value)

    def validate_model_layers(self) -> None:
        for model_name, expected_count in self.expected["layer_counts"].items():
            path = self.path(os.path.join("model-layers", f"{model_name}.json"))
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                layers = json.load(handle)
            self.assert_equal("model-layers", f"{model_name} layer count", len(layers), expected_count)
            malformed = [
                index
                for index, item in enumerate(layers, start=1)
                if not (
                    isinstance(item, list)
                    and len(item) == 2
                    and all(isinstance(value, int) and value >= 0 for value in item)
                )
            ]
            if malformed:
                self.fail_check("model-layers", f"{model_name} structure", f"bad layers: {malformed[:5]}")
            else:
                self.pass_check("model-layers", f"{model_name} structure", "all layers are [W, D] integer pairs")

    def validate_timing(self) -> None:
        summary_rows = self.read_csv("benchmark_timing_summary.csv")
        raw_rows = self.read_csv("benchmark_timing_raw.csv")
        labels = self.expected["label_order"]
        iterations = int(self.expected["timing"]["iterations"])
        zero_lia_labels = set(self.expected["timing"]["zero_lia_labels"])

        self.validate_columns("structure", "benchmark_timing_summary columns", summary_rows, TIMING_SUMMARY_COLUMNS)
        self.validate_columns("structure", "benchmark_timing_raw columns", raw_rows, TIMING_RAW_COLUMNS)
        self.assert_equal("timing", "summary row count", len(summary_rows), len(labels))
        self.assert_equal("timing", "raw row count", len(raw_rows), len(labels) * iterations)

        summary_by_label = {row["label"]: row for row in summary_rows}
        raw_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in raw_rows:
            raw_by_label[row["label"]].append(row)

        for label in labels:
            summary = summary_by_label.get(label)
            if summary is None:
                self.fail_check("timing", f"{label} summary", "missing")
                continue
            self.assert_equal("timing", f"{label}.model", summary["model"], self.expected["models"][label])
            self.assert_equal("timing", f"{label}.iterations", summary["iterations"], iterations)

            for field in TIMING_SUMMARY_COLUMNS[3:]:
                self.assert_nonnegative("timing", f"{label}.{field}", summary[field])

            if label in zero_lia_labels:
                self.assert_close(
                    "timing",
                    f"{label}.lia_total_mean_ms",
                    summary["lia_total_mean_ms"],
                    0.0,
                    "timing_zero_abs",
                )
                self.assert_close(
                    "timing",
                    f"{label}.lia_per_decision_mean_us",
                    summary["lia_per_decision_mean_us"],
                    0.0,
                    "timing_zero_abs",
                )

            raw_items = raw_by_label.get(label, [])
            self.assert_equal("timing", f"{label} raw iteration count", len(raw_items), iterations)
            iteration_numbers = sorted(int(item["iteration"]) for item in raw_items)
            self.assert_equal(
                "timing",
                f"{label} raw iteration sequence",
                ",".join(str(item) for item in iteration_numbers),
                ",".join(str(item) for item in range(1, iterations + 1)),
            )
            for item in raw_items:
                for field in TIMING_RAW_COLUMNS[3:]:
                    self.assert_nonnegative("timing", f"{label}.raw[{item['iteration']}].{field}", item[field])

    def validate_columns(
        self,
        group: str,
        name: str,
        rows: list[dict[str, str]],
        expected_columns: list[str],
    ) -> None:
        if not rows:
            self.fail_check(group, name, "no rows available")
            return
        actual_columns = list(rows[0].keys())
        if actual_columns == expected_columns:
            self.pass_check(group, name, "columns match")
        else:
            self.fail_check(group, name, f"actual={actual_columns}, expected={expected_columns}")

    def assert_nonnegative(self, group: str, name: str, actual: Any) -> None:
        try:
            actual_float = float(actual)
        except (TypeError, ValueError):
            self.fail_check(group, name, f"not numeric: actual={actual!r}")
            return
        if actual_float >= 0.0:
            self.pass_check(group, name, f"actual={actual_float:g}")
        else:
            self.fail_check(group, name, f"negative value: actual={actual_float:g}")

    def validate(self) -> dict[str, Any]:
        self.validate_required_files()

        required_csvs = [
            "constella_results.csv",
            "paper_claims_summary.csv",
            "table1_scenarios_and_parameters.csv",
            "benchmark_timing_summary.csv",
            "benchmark_timing_raw.csv",
        ]
        if any(not os.path.exists(self.path(filename)) for filename in required_csvs):
            return self.report()

        self.validate_constella_results()
        self.validate_claims()
        self.validate_table1()
        self.validate_model_layers()
        self.validate_timing()
        return self.report()

    def report(self) -> dict[str, Any]:
        failed = [check for check in self.checks if check["status"] == "fail"]
        passed = [check for check in self.checks if check["status"] == "pass"]
        return {
            "status": "pass" if not failed else "fail",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results_dir": self.results_dir,
            "summary": {
                "passed": len(passed),
                "failed": len(failed),
                "total": len(self.checks),
            },
            "checks": self.checks,
        }


def load_expected(path: str) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def print_report(report: dict[str, Any], report_path: str) -> None:
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
    for check in report["checks"]:
        by_group[check["group"]][check["status"]] += 1

    passed_claims = [
        check for check in report["checks"]
        if check["group"] == "claims" and check["status"] == "pass"
    ]
    result_rows = [
        check for check in report["checks"]
        if check["group"] == "results"
        and check["status"] == "pass"
        and check["name"].count("|") == 1
        and check["detail"] == "row present"
    ]

    print("Constella artifact validation")
    print(f"Results directory: {report['results_dir']}")
    print(f"Report: {report_path}")
    print(f"Status: {report['status'].upper()}")
    print(
        "Checks: "
        f"{report['summary']['passed']} passed, "
        f"{report['summary']['failed']} failed, "
        f"{report['summary']['total']} total"
    )
    print("")
    print("Validated scope:")
    print(f"- Required output files and generated model-layer files")
    print(f"- {len(result_rows)} scenario/approach result rows in constella_results.csv")
    print(f"- {len(passed_claims)} headline paper claims in paper_claims_summary.csv")
    print("- Table 1 scenario metadata and shared simulation parameters")
    print("- Regenerated model-layer counts and JSON structure")
    print("- Timing CSV structure, iteration coverage, nonnegative timings, and zero-LIA invariant")
    print("")
    print("Check groups:")
    for group in sorted(by_group):
        counts = by_group[group]
        print(f"- {group}: {counts['pass']} passed, {counts['fail']} failed")

    failed = [check for check in report["checks"] if check["status"] == "fail"]
    if failed:
        print("")
        print("Failures:")
        for check in failed:
            print(f"- [{check['group']}] {check['name']}: {check['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated Constella artifact results")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=DEFAULT_RESULTS_DIR,
        help="Generated results directory (default: artifact-output/paper-results)",
    )
    parser.add_argument(
        "--expected",
        default=DEFAULT_EXPECTED_PATH,
        help="Expected-results JSON file",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Validation report JSON path (default: <results_dir>/validation_report.json)",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    report_path = args.report or os.path.join(results_dir, "validation_report.json")
    expected = load_expected(args.expected)
    validator = Validator(results_dir, expected)
    report = validator.validate()

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print_report(report, report_path)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
