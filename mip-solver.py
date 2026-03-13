import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from mip import Model, minimize, INTEGER, BINARY, xsum

@dataclass
class SolverResult:
    layer: int
    W: float
    D: float
    x: int
    y: int
    cost: float
    elapsed: float

def save_result_to_csv(result, model_name, I_total, output_path):
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "I_total", "layer", "x", "y", "cost", "elapsed"])
        writer.writerow([model_name, I_total, result.layer, result.x, result.y,
                         f"{result.cost:.1f}", f"{result.elapsed:.6f}"])

model_name = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
I_total    = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
output_csv = sys.argv[3] if len(sys.argv) > 3 else None
X_total    = int(sys.argv[4])   if len(sys.argv) > 4 else 250
Y_total    = int(sys.argv[5])   if len(sys.argv) > 5 else 250

base = os.path.dirname(__file__)

config_path = os.path.join(base, 'scenarios', 'config_base.json')
with open(config_path) as f:
    cfg = json.load(f)

layers_path = os.path.join(base, 'model-layers', f'{model_name}.json')
with open(layers_path) as f:
    layers = [(W, D) for W, D in json.load(f)]

t_start = time.perf_counter()

m = Model()
m.verbose = 0

# Constants
alpha       = cfg["alpha"]
beta        = cfg["beta"]
R_max       = cfg["R_max"]
T_comm      = cfg["T_comm"]
E_processor = cfg["E_processor"]
E_comm      = cfg["E_comm"]
p           = cfg["p"]
q           = cfg["q"]
T_comp      = cfg["T_comp"]
delta_t     = cfg["delta_t"]
I_max       = int(math.floor(T_comp / delta_t))

# Variables
x = m.add_var(var_type=INTEGER, lb=1, name="x")
y = m.add_var(var_type=INTEGER, lb=0, name="y")
z = {l: m.add_var(var_type=BINARY, name=f"z_{l}") for l in range(1, len(layers)+1)}
m.add_constr(xsum(z[l] for l in range(1, len(layers) + 1)) == 1)

W_selected = xsum(layers[l-1][0] * z[l] for l in range(1, len(layers) + 1))
D_selected = xsum(layers[l-1][1] * z[l] for l in range(1, len(layers) + 1))

# Constraints
m.add_constr(I_total * D_selected <= (x+y) * R_max * T_comm)
m.add_constr(I_total * W_selected * p <= x * E_processor)
m.add_constr(I_total * (W_selected * p + D_selected * q) <= x * E_processor + y * E_comm)
m.add_constr(I_total <= x * I_max)
m.add_constr(x >= 1)
m.add_constr(y >= 0)
m.add_constr(x <= X_total)
m.add_constr(y <= Y_total)

m.objective = minimize(alpha * x + beta * y)

m.optimize()

t_end = time.perf_counter()

selected_layer = [l for l in range(1, len(layers) + 1) if z[l].x >= 0.99][0]
W_opt, D_opt = layers[selected_layer - 1]

result = SolverResult(
    layer=selected_layer,
    W=W_opt,
    D=D_opt,
    x=int(round(x.x)),
    y=int(round(y.x)),
    cost=m.objective_value,
    elapsed=t_end - t_start,
)

if output_csv:
    save_result_to_csv(result, model_name, I_total, output_csv)

print(f"Layer:   {result.layer}")
print(f"W(l):    {result.W}")
print(f"D(l):    {result.D}")
print(f"X:       {result.x}")
print(f"Y:       {result.y}")
print(f"Cost:    {result.cost:.1f}")
print(f"Time:    {result.elapsed:.6f} s")
