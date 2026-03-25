import json
import os
import sys

from mip_solver import solve_ocri


def load_config(base_dir):
    path = os.path.join(base_dir, "scenarios", "config_base.json")
    with open(path) as f:
        return json.load(f)


def load_layers(base_dir, model_name):
    path = os.path.join(base_dir, "model-layers", f"{model_name}.json")
    with open(path) as f:
        return [(W, D) for W, D in json.load(f)]


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
