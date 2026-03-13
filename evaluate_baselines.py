import csv
import json
import math
import os
import sys
from mip import Model, minimize, INTEGER, BINARY, xsum


def load_layers(base_dir, model_name):
    path = os.path.join(base_dir, "model-layers", f"{model_name}.json")
    with open(path) as f:
        return [(W, D) for W, D in json.load(f)]


def load_config(base_dir):
    path = os.path.join(base_dir, "scenarios", "config_base.json")
    with open(path) as f:
        return json.load(f)


def solve_ocri(layers, I_total, X_total, Y_total, cfg):
    I_max = int(math.floor(cfg["T_comp"] / cfg["delta_t"]))
    m = Model()
    m.verbose = 0

    x = m.add_var(var_type=INTEGER, lb=1, name="x")
    y = m.add_var(var_type=INTEGER, lb=0, name="y")
    z = {l: m.add_var(var_type=BINARY, name=f"z_{l}") for l in range(1, len(layers) + 1)}
    m.add_constr(xsum(z[l] for l in range(1, len(layers) + 1)) == 1)

    W_sel = xsum(layers[l-1][0] * z[l] for l in range(1, len(layers) + 1))
    D_sel = xsum(layers[l-1][1] * z[l] for l in range(1, len(layers) + 1))

    m.add_constr(I_total * D_sel <= (x + y) * cfg["R_max"] * cfg["T_comm"])
    m.add_constr(I_total * W_sel * cfg["p"] <= x * cfg["E_processor"])
    m.add_constr(I_total * (W_sel * cfg["p"] + D_sel * cfg["q"]) <= x * cfg["E_processor"] + y * cfg["E_comm"])
    m.add_constr(I_total <= x * I_max)
    m.add_constr(x >= 1)
    m.add_constr(y >= 0)
    m.add_constr(x <= X_total)
    m.add_constr(y <= Y_total)

    m.objective = minimize(cfg["alpha"] * x + cfg["beta"] * y)
    m.optimize()

    l_opt = [l for l in range(1, len(layers) + 1) if z[l].x >= 0.99][0]
    return int(round(x.x)), int(round(y.x)), l_opt


def compute_success(X, Y, l, layers, I_total, cfg):
    W, D = layers[l - 1]
    p, q = cfg["p"], cfg["q"]
    E_processor = cfg["E_processor"]
    E_comm      = cfg["E_comm"]

    cost_per_image_proc = W * p + D * q
    cost_per_image_comm = D * q

    cap_proc = math.floor(E_processor / cost_per_image_proc) if cost_per_image_proc > 0 else I_total
    cap_comm = math.floor(E_comm      / cost_per_image_comm) if cost_per_image_comm > 0 else I_total

    w                = min(X * cap_proc + Y * cap_comm, I_total)
    success_rate     = w / I_total
    transmitted_bits = w * D
    return success_rate, transmitted_bits


def evaluate(scenario_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg      = load_config(base_dir)

    with open(scenario_path) as f:
        scenario = json.load(f)

    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
    output_path   = os.path.join(base_dir, "results", f"{scenario_name}_baselines.csv")

    alpha = cfg["alpha"]
    beta  = cfg["beta"]

    csv_columns = ["label", "model", "I_total", "approach", "X", "Y", "l", "W_gflops", "D_mbits", "cost", "success_rate", "transmitted_bits"]

    all_rows = []

    for config in scenario["configs"]:
        label      = config["label"]
        model_name = config["model"]
        I_total    = float(config["I_total"])
        X_total    = int(config["X_total"])
        Y_total    = int(config["Y_total"])
        layers     = load_layers(base_dir, model_name)

        N = X_total + Y_total
        X_o, Y_o, l_o = solve_ocri(layers, I_total, X_total, Y_total, cfg)
        X_m, Y_m, l_m = N,                  0,               len(layers)
        X_c, Y_c, l_c = 1,                  0,               1
        X_b, Y_b, l_b = math.floor(N / 2),  math.ceil(N / 2), math.floor(len(layers) / 2) + 1

        approaches = [
            ("OCRI", X_o, Y_o, l_o),
            ("MEA",  X_m, Y_m, l_m),
            ("CA",   X_c, Y_c, l_c),
            ("BA",   X_b, Y_b, l_b),
        ]

        for name, X, Y, l in approaches:
            cost             = alpha * X + beta * Y
            sr, tb           = compute_success(X, Y, l, layers, I_total, cfg)
            W_l, D_l         = layers[l - 1]
            all_rows.append({
                "label":            label,
                "model":            model_name,
                "I_total":          I_total,
                "approach":         name,
                "X":                X,
                "Y":                Y,
                "l":                l,
                "W_gflops":         f"{W_l / 1e9:.4f}",
                "D_mbits":          f"{D_l / 1e6:.4f}",
                "cost":             f"{cost:.1f}",
                "success_rate":     f"{sr:.4f}",
                "transmitted_bits": f"{tb:.0f}",
            })

    label_order = ["extra-small", "small", "medium", "large", "extra-large"]
    all_rows.sort(key=lambda r: label_order.index(r["label"]) if r["label"] in label_order else 999)

    col_w = [14, 8, 6, 6, 6, 12, 10, 10, 13, 20]
    header_line = (
        f"{'label':<{col_w[0]}} {'approach':<{col_w[1]}} "
        f"{'X':>{col_w[2]}} {'Y':>{col_w[3]}} {'l':>{col_w[4]}} "
        f"{'W_gflops':>{col_w[5]}} {'D_mbits':>{col_w[6]}} "
        f"{'cost':>{col_w[7]}} {'success_rate':>{col_w[8]}} {'transmitted_bits':>{col_w[9]}}"
    )
    print(header_line)
    print("-" * len(header_line))
    for r in all_rows:
        print(
            f"{r['label']:<{col_w[0]}} {r['approach']:<{col_w[1]}} "
            f"{r['X']:>{col_w[2]}} {r['Y']:>{col_w[3]}} {r['l']:>{col_w[4]}} "
            f"{r['W_gflops']:>{col_w[5]}} {r['D_mbits']:>{col_w[6]}} "
            f"{r['cost']:>{col_w[7]}} {r['success_rate']:>{col_w[8]}} {r['transmitted_bits']:>{col_w[9]}}"
        )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_baselines.py <scenario_file>")
        sys.exit(1)
    evaluate(sys.argv[1])
