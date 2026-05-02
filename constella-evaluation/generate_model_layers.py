"""Generate model-layer profiles used by the Constella evaluation.

The evaluation consumes JSON files in ``model-layers/``. This script regenerates
those files from PyTorch/torchvision models using torchinfo, matching the
profiling procedure used to create the checked-in layer data.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any


PAPER_MODELS = [
    "alexnet",
    "squeezenet1_0",
    "resnet50",
    "swin_b",
    "efficientnet_b0",
]

@dataclass(frozen=True)
class LayerMetrics:
    output_size_bytes: int
    flops: int


def _load_torch_stack():
    try:
        import torch
        import torchvision.models as models
        from torchinfo import summary
    except ImportError as exc:
        raise SystemExit(
            "Layer generation requires torch, torchvision, and torchinfo. "
            "Install those packages, then rerun this command."
        ) from exc
    return torch, models, summary


def _supported_models(models_module: Any) -> dict[str, Any]:
    return {
        "alexnet": models_module.alexnet,
        "efficientnet_b0": models_module.efficientnet_b0,
        "resnet50": models_module.resnet50,
        "squeezenet1_0": models_module.squeezenet1_0,
        "swin_b": models_module.swin_b,
    }


def _load_model(model_name: str, device: str):
    _, models_module, _ = _load_torch_stack()
    supported = _supported_models(models_module)
    if model_name not in supported:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {sorted(supported)}")

    try:
        model = supported[model_name](weights=None)
    except TypeError:
        model = supported[model_name](pretrained=False)
    model = model.to(device)
    model.eval()
    return model


def _shape_num_elements(shape: Any) -> int:
    if not shape or not isinstance(shape, (list, tuple)):
        return 0

    num_elements = 1
    for dim in shape:
        if isinstance(dim, int):
            num_elements *= dim
    return num_elements


def _profile_layers(model_name: str, device: str) -> list[LayerMetrics]:
    torch, _, summary = _load_torch_stack()
    model = _load_model(model_name, device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    stats = summary(
        model,
        input_size=tuple(input_tensor.shape),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        row_settings=["var_names"],
        depth=float("inf"),
        verbose=0,
        device=device,
    )

    model_class_name = model.__class__.__name__
    layers: list[LayerMetrics] = []

    for layer_info in stats.summary_list[1:]:
        if (
            getattr(layer_info, "class_name", None) == model_class_name
            and not getattr(layer_info, "var_name", None)
        ):
            continue

        output_shape = getattr(layer_info, "output_size", ())
        output_size_bytes = _shape_num_elements(output_shape) * 4
        macs = getattr(layer_info, "macs", 0) or 0
        flops = int(macs * 2)
        layers.append(LayerMetrics(output_size_bytes=output_size_bytes, flops=flops))

    return layers


def _layer_pairs(model_name: str, device: str) -> list[list[int]]:
    cumulative_flops = 0
    generated: list[list[int]] = []

    for layer in _profile_layers(model_name, device):
        cumulative_flops += layer.flops
        generated.append([cumulative_flops, layer.output_size_bytes * 8])

    return generated


def generate_model_layers(model_name: str, output_dir: str, device: str = "cpu") -> list[list[int]]:
    generated = _layer_pairs(model_name, device)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.json")
    with open(output_path, "w") as handle:
        json.dump(generated, handle)
        handle.write("\n")

    return generated


def check_model_layers(model_name: str, output_dir: str, device: str = "cpu") -> bool:
    existing_path = os.path.join(output_dir, f"{model_name}.json")
    with open(existing_path) as handle:
        existing = json.load(handle)

    generated = _layer_pairs(model_name, device)

    if generated == existing:
        print(f"{model_name}: OK ({len(generated)} layers)")
        return True

    print(
        f"{model_name}: MISMATCH "
        f"(generated {len(generated)} layers, existing {len(existing)} layers)"
    )
    for index, (new, old) in enumerate(zip(generated, existing), start=1):
        if new != old:
            print(f"  first mismatch at layer {index}: generated={new}, existing={old}")
            break
    return False


def _models_arg(value: str) -> list[str]:
    if value == "all":
        return PAPER_MODELS
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Constella model-layer JSON files")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names, or 'all' for the paper models",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model-layers"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in memory and compare against existing JSON files without writing",
    )
    args = parser.parse_args()

    models_to_process = _models_arg(args.models)
    if args.check:
        ok = all(check_model_layers(model, args.output_dir, args.device) for model in models_to_process)
        raise SystemExit(0 if ok else 1)

    for model in models_to_process:
        layers = generate_model_layers(model, args.output_dir, args.device)
        print(f"{model}: wrote {len(layers)} layers to {args.output_dir}")


if __name__ == "__main__":
    main()
