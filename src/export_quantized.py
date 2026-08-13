"""Export inspectable integer weights and per-group scales from a QAT model.

This artifact is backend-neutral and educational. A deployment runtime still
needs a packed integer GEMM kernel to turn it into a speed or memory win.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig, from_dict
from model import GPT
from quantization import QATLinear, quantized_integer_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    config = from_dict(ModelConfig, checkpoint["model_config"])
    if not config.qat:
        raise ValueError("checkpoint model_config.qat is false; use a QAT config first")
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    tensors: dict[str, object] = {
        "format": "symmetric_grouped_integer_v1",
        "weight_bits": config.qat_weight_bits,
        "group_size": config.qat_group_size,
        "model_config": checkpoint["model_config"],
        "linears": {},
        "float_state": {},
    }
    quantized_linears = tensors["linears"]
    assert isinstance(quantized_linears, dict)
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            integers, scales, padding = quantized_integer_weight(
                module.weight, config.qat_weight_bits, config.qat_group_size
            )
            quantized_linears[name] = {
                "integers": integers,
                "scales": scales,
                "padding": padding,
                "shape": tuple(module.weight.shape),
            }
    float_state = tensors["float_state"]
    assert isinstance(float_state, dict)
    for name, value in model.state_dict().items():
        if not any(name == f"{linear_name}.weight" for linear_name in quantized_linears):
            float_state[name] = value
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, output)
    print(f"exported {len(quantized_linears)} quantized matrices to {output}")


if __name__ == "__main__":
    main()
