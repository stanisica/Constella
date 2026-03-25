import json
import math
import os
import sys

from mip import Model, minimize, INTEGER, BINARY, xsum


def load_config(base_dir):
    path = os.path.join(base_dir, "scenarios", "config_base.json")
    with open(path) as f:
        return json.load(f)


def load_layers(base_dir, model_name):
    path = os.path.join(base_dir, "model-layers", f"{model_name}.json")
    with open(path) as f:
        return [(W, D) for W, D in json.load(f)]


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


def generate_model_layers(model_name):
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../dnn-analysis'))
    from analyzer.model_loader import ModelLoader
    from analyzer.layer_profiler import LayerProfiler

    model = ModelLoader.load_pretrained(model_name, pretrained=False, device='cpu')
    input_size = ModelLoader.get_input_size(model_name)
    input_tensor = torch.randn(1, *input_size)

    profiler = LayerProfiler(model, device='cpu')
    profiler.profile(input_tensor)
    summary = profiler.get_summary()

    cumulative_flops = 0
    layers = []
    for m in summary['metrics']:
        cumulative_flops += m.flops
        layers.append((cumulative_flops, m.output_size_bytes * 8))

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'model-layers')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_name}.json')

    with open(output_path, 'w') as f:
        json.dump(layers, f)

    print(f"Saved {len(layers)} layers to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <model_name>")
        sys.exit(1)
    generate_model_layers(sys.argv[1])
