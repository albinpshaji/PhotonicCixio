"""
Differentiable Finite-Bit DAC Quantization with Straight-Through Estimators (STE).
Mathematical references:
- Esser et al. (2019), ICLR 2020 (Learned Step Size Quantization - LSQ)
- Fang et al. (2019), Optics Express 27(10), 14009-14029
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class STEQuantizeFunction(torch.autograd.Function):
    """
    Custom autograd function implementing uniform B-bit discretization
    with Straight-Through Estimator (STE) gradient approximation.
    """
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        val_min: float,
        val_max: float,
        bits: int
    ) -> torch.Tensor:
        """
        Forward pass: Discretizes x into 2^bits uniform levels in [val_min, val_max].
        """
        n_levels = (1 << bits) - 1  # 2^B - 1
        step_size = (val_max - val_min) / float(n_levels)

        # Normalized coordinates: [0, n_levels]
        x_norm = (x - val_min) / step_size
        x_clamped = torch.clamp(x_norm, 0.0, float(n_levels))
        x_rounded = torch.round(x_clamped)

        # Re-scale back to physical range
        x_quant = x_rounded * step_size + val_min

        # Save mask for gradient clamping in backward pass
        ctx.save_for_backward(x)
        ctx.val_min = val_min
        ctx.val_max = val_max

        return x_quant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: Straight-Through Estimator (STE).
        Passes gradient through unchanged where val_min <= x <= val_max,
        and zeros gradient outside the physical dynamic range.
        """
        (x,) = ctx.saved_tensors
        val_min = ctx.val_min
        val_max = ctx.val_max

        # Indicator mask: 1 within bounds, 0 outside
        in_bounds = (x >= val_min) & (x <= val_max)
        grad_input = grad_output * in_bounds.to(grad_output.dtype)

        # Gradients for non-tensor arguments are None
        return grad_input, None, None, None


class DACQuantizer(nn.Module):
    """
    Differentiable B-bit Digital-to-Analog Converter (DAC) module.
    Discretizes continuous phase commands into finite hardware control levels.
    Models:
    - Finite B-bit uniform quantization with Straight-Through Estimator (STE)
    - Differential Non-Linearity (DNL) and Integral Non-Linearity (INL)
    - High-frequency electrical phase jitter / power-supply voltage ripple
    """
    def __init__(
        self,
        bits: int = 6,
        val_min: float = 0.0,
        val_max: float = math.pi,
        enabled: bool = True,
        dnl_lsb: float = 0.0,
        inl_lsb: float = 0.0,
        phase_jitter_std: float = 0.0,
        enable_phase_jitter: bool = False
    ):
        super().__init__()
        if bits < 1 or bits > 16:
            raise ValueError(f"DAC bits must be between 1 and 16 (got {bits})")
        self.bits = bits
        self.val_min = val_min
        self.val_max = val_max
        self.enabled = enabled
        self.dnl_lsb = dnl_lsb
        self.inl_lsb = inl_lsb
        self.phase_jitter_std = phase_jitter_std
        self.enable_phase_jitter = enable_phase_jitter

        # Precompute static INL table if DNL/INL is configured
        n_levels = 1 << bits
        if (dnl_lsb > 0 or inl_lsb > 0) and enabled:
            torch.manual_seed(42 + bits)
            # DNL error per code step: Gaussian bounded by dnl_lsb
            dnl = torch.randn(n_levels) * max(dnl_lsb, 1e-4)
            dnl = dnl - torch.mean(dnl)  # Zero-mean DNL
            # Cumulative INL
            inl = torch.cumsum(dnl, dim=0)
            if torch.max(torch.abs(inl)) > 0 and inl_lsb > 0:
                inl = (inl / torch.max(torch.abs(inl))) * inl_lsb
            self.register_buffer("inl_table_lsb", inl)
        else:
            self.register_buffer("inl_table_lsb", torch.zeros(n_levels))

    @property
    def step_size(self) -> float:
        """Resolution LSB of the DAC: (val_max - val_min) / (2^B - 1)."""
        n_levels = (1 << self.bits) - 1
        return (self.val_max - self.val_min) / float(n_levels)

    def forward(self, x: torch.Tensor, add_jitter: bool = True) -> torch.Tensor:
        """
        Quantizes input tensor x using Straight-Through Estimator,
        with optional INL distortion and analog phase jitter.
        """
        if not self.enabled:
            return x

        # 1. Base quantization with STE
        x_quant = STEQuantizeFunction.apply(x, self.val_min, self.val_max, self.bits)

        # 2. Apply INL error if configured
        if self.inl_lsb > 0:
            step = self.step_size
            code_idx = torch.clamp(
                torch.round((x - self.val_min) / step).long(),
                0,
                (1 << self.bits) - 1
            )
            inl_table = self.inl_table_lsb.to(device=x.device, dtype=x.dtype)
            inl_err = inl_table[code_idx] * step
            x_quant = x_quant + inl_err

        # 3. Add high-frequency phase jitter (Gaussian analog ripple)
        if self.enable_phase_jitter and add_jitter and self.phase_jitter_std > 0:
            jitter = torch.randn_like(x_quant) * self.phase_jitter_std
            x_quant = x_quant + jitter

        return x_quant
