import json
import os
import sys

from mip_solver import solve_ocri


def load_config(base_dir):
    path = os.path.join(base_dir, "scenarios", "config_base.json")
    with open(path) as f:
        return json.load(f)


def load_layers(base_dir, model_name):
    layers_dir = os.environ.get(
        "CONSTELLA_MODEL_LAYERS_DIR",
        os.path.join(base_dir, "model-layers"),
    )
    path = os.path.join(layers_dir, f"{model_name}.json")
    with open(path) as f:
        return [(W, D) for W, D in json.load(f)]


def get_results_dir(default_base_dir):
    return os.environ.get(
        "CONSTELLA_RESULTS_DIR",
        os.path.join(default_base_dir, "artifact-output", "paper-results"),
    )


def generate_model_layers(model_name):
    from generate_model_layers import generate_model_layers as generate

    output_dir = os.path.join(os.path.dirname(__file__), "..", "model-layers")
    layers = generate(model_name, output_dir)
    output_path = os.path.join(output_dir, f"{model_name}.json")
    print(f"Saved {len(layers)} layers to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <model_name>")
        sys.exit(1)
    generate_model_layers(sys.argv[1])
