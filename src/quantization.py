"""Readable fake quantization primitives for quantization-aware training.

The tensors remain floating point. The forward pass simulates symmetric integer
rounding while the straight-through estimator supplies gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    return original + (quantized - original).detach()


def fake_quantize_per_token(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric dynamic quantization over the last dimension of each token."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    dequantized = (x / scale).round().clamp(-qmax, qmax) * scale
    return _ste(x, dequantized)


def fake_quantize_grouped_weight(
    weight: torch.Tensor, bits: int, group_size: int
) -> torch.Tensor:
    """Symmetric per-group fake quantization for a [out, in] weight matrix."""
    if weight.ndim != 2:
        raise ValueError("grouped weight quantization expects a matrix")
    qmax = 2 ** (bits - 1) - 1
    in_features = weight.shape[1]
    padding = (-in_features) % group_size
    padded = F.pad(weight, (0, padding)) if padding else weight
    grouped = padded.reshape(weight.shape[0], -1, group_size)
    scale = grouped.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    dequantized = (grouped / scale).round().clamp(-qmax, qmax) * scale
    dequantized = dequantized.reshape(weight.shape[0], -1)[:, :in_features]
    return _ste(weight, dequantized)


class QATLinear(nn.Linear):
    """nn.Linear with optional activation and weight fake quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        weight_bits: int = 8,
        activation_bits: int = 8,
        group_size: int = 64,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.group_size = group_size
        self.fake_quant_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)
        x = fake_quantize_per_token(x, self.activation_bits)
        weight = fake_quantize_grouped_weight(
            self.weight, self.weight_bits, self.group_size
        )
        return F.linear(x, weight, self.bias)


def set_qat_enabled(module: nn.Module, enabled: bool) -> None:
    for child in module.modules():
        if isinstance(child, QATLinear):
            child.fake_quant_enabled = enabled


def quantized_integer_weight(
    weight: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Export grouped integer values and scales for inspection/deployment work."""
    qmax = 2 ** (bits - 1) - 1
    padding = (-weight.shape[1]) % group_size
    padded = F.pad(weight.detach().float(), (0, padding)) if padding else weight.detach().float()
    grouped = padded.reshape(weight.shape[0], -1, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    integers = (grouped / scales).round().clamp(-qmax, qmax)
    dtype = torch.int8 if bits <= 8 else torch.int16
    return integers.to(dtype), scales.squeeze(-1), padding
