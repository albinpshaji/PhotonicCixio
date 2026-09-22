"""
Coherent Backreflection and Multi-Cavity Fabry-Perot Interference Model.
Calibrated for standard Silicon-on-Insulator (SOI) integrated circuits.

Governing Physics:
Any discontinuity in waveguide effective index (grating coupler facets, directional coupler
junctions, waveguide crossings) reflects a small fraction of guided optical power backward.
Between pairs of reflective interfaces separated by effective cavity length L_cav, standing-wave
Fabry-Perot interference produces wavelength-dependent transmission ripple:

    H_FP(lambda) = sqrt((1 - R1) * (1 - R2)) / (1 - sqrt(R1 * R2) * exp(i * 2 * phi))
    where phi = 2 * pi * n_eff * L_cav / lambda

Free Spectral Range (FSR):
    FSR = lambda_0^2 / (2 * n_g * L_cav)
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from src.config import PhotonicConfig, PhysicalConstants


class FabryPerotBackreflection(nn.Module):
    """
    Differentiable PyTorch module modeling distributed Fabry-Perot cavity ripple
    induced by interface backreflections (grating couplers, crossings, coupler facets).
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.n_modes = config.n_modes
        self.enabled = config.enable_backreflection and not config.ideal_mode

        # Power reflectivities R = 10^(R_dB / 10)
        self.R_gc = 10.0 ** (config.grating_reflectivity_db / 10.0)
        self.R_cross = 10.0 ** (config.crossing_reflectivity_db / 10.0)
        self.R_dc = 10.0 ** (config.coupler_reflectivity_db / 10.0)

        self.n_eff = PhysicalConstants.n_eff
        self.n_g = PhysicalConstants.n_g
        self.base_cavity_length = config.cavity_length

        # Mode-dependent cavity length variations across the waveguide bus (meters)
        # Outer waveguides have slightly longer routing arcs between stages
        mode_offsets = torch.linspace(0.85, 1.15, steps=self.n_modes) * self.base_cavity_length
        self.register_buffer("cavity_lengths", mode_offsets)

    def compute_transfer_filter(
        self,
        wavelength: Optional[float] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Computes the complex transmission multiplier H_FP for each mode at given wavelength.

        Args:
            wavelength: Optical wavelength in meters.
            device: Target torch device.

        Returns:
            Tensor of shape (n_modes,) with complex transmission coefficients.
        """
        if not self.enabled:
            dev = device if device is not None else self.cavity_lengths.device
            return torch.ones(self.n_modes, dtype=torch.complex64, device=dev)

        lam = wavelength if wavelength is not None else self.config.wavelength
        dev = device if device is not None else self.cavity_lengths.device
        L_cav = self.cavity_lengths.to(dev)

        # Composite interface reflectivities for internal cavity:
        # Front facet: grating coupler + first crossing
        # Rear facet: directional coupler interface
        R1 = self.R_gc
        R2 = self.R_cross + self.R_dc

        # Round-trip phase accumulation: 2 * phi = 4 * pi * n_eff * L_cav / lam
        phi_roundtrip = 4.0 * math.pi * self.n_eff * L_cav / lam

        # Complex round-trip propagation factor: sqrt(R1 * R2) * exp(i * phi_roundtrip)
        r_product = math.sqrt(R1 * R2)
        complex_exp = torch.polar(
            torch.full_like(phi_roundtrip, r_product),
            phi_roundtrip
        )

        # Numerator: sqrt((1 - R1) * (1 - R2))
        numerator = math.sqrt((1.0 - R1) * (1.0 - R2))

        # Denominator: 1 - r_product * exp(i * 2 * phi)
        # Using 3rd-order geometric expansion for numerical precision & fast autograd:
        # 1 / (1 - z) approx 1 + z + z^2 + z^3
        z = complex_exp
        inv_denom = 1.0 + z + (z**2) + (z**3)

        H_fp = numerator * inv_denom
        return H_fp.to(torch.complex64)

    def forward(
        self,
        E_field: torch.Tensor,
        wavelength: Optional[float] = None
    ) -> torch.Tensor:
        """
        Applies Fabry-Perot cavity spectral ripple to the optical wave field.

        Args:
            E_field: Optical field tensor of shape (..., n_modes) or (..., n_modes, 2).
            wavelength: Optical wavelength in meters.

        Returns:
            Modulated optical field tensor of identical shape.
        """
        if not self.enabled:
            return E_field

        H_fp = self.compute_transfer_filter(wavelength, device=E_field.device)

        if E_field.ndim > 1 and E_field.shape[-1] == 2 and E_field.shape[-2] == self.n_modes:
            # Jones vector case: shape (..., n_modes, 2)
            H_fp = H_fp.unsqueeze(-1)

        return E_field * H_fp
