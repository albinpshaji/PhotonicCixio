"""
Spatially Correlated Wafer Map Model for Silicon Photonic Integrated Circuits.
Models 2D correlated lithographic line-edge roughness (LER), chemical-mechanical polishing (CMP)
dishing, and photoresist exposure gradients across the physical die area.

Mathematical references:
- Bogaerts et al. (2020), Nature 586, 207-216
- Gu et al. (2020), IEEE/ACM ASP-DAC 2020, pp. 205-210
- Sunny et al. (2021), ACM/IEEE DAC 2021, pp. 1069-1074
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.config import PhotonicConfig


def build_spatial_covariance_matrix(
    coords: torch.Tensor,
    correlation_length_x: float = 250.0e-6,
    correlation_length_y: float = 200.0e-6,
    total_variance: float = 1.0,
    systematic_ratio: float = 0.75,
    jitter: float = 1e-6
) -> torch.Tensor:
    """
    Constructs the M x M spatial covariance matrix Sigma over 2D MZI coordinates.

    Sigma_ij = sigma_sys^2 * exp( - (x_i - x_j)^2 / (2 * Lx^2) - (y_i - y_j)^2 / (2 * Ly^2) )
             + sigma_rand^2 * delta_ij

    Args:
        coords: Tensor of shape (M, 2) with (x, y) coordinates in meters.
        correlation_length_x: Correlation length in horizontal die direction (Lx).
        correlation_length_y: Correlation length in vertical die direction (Ly).
        total_variance: Overall variance sigma^2.
        systematic_ratio: Fraction of variance that is spatially correlated (0 to 1).
        jitter: Numerical diagonal regularization for strict positive definiteness.

    Returns:
        Sigma: Covariance matrix of shape (M, M).
    """
    M = coords.shape[0]
    x = coords[:, 0]  # (M,)
    y = coords[:, 1]  # (M,)

    dx = x.unsqueeze(1) - x.unsqueeze(0)  # (M, M)
    dy = y.unsqueeze(1) - y.unsqueeze(0)  # (M, M)

    sigma_sys_sq = total_variance * systematic_ratio
    sigma_rand_sq = total_variance * (1.0 - systematic_ratio)

    # Anisotropic Gaussian / Squared Exponential covariance kernel
    spatial_dist_sq = (dx / correlation_length_x) ** 2 + (dy / correlation_length_y) ** 2
    Sigma_sys = sigma_sys_sq * torch.exp(-0.5 * spatial_dist_sq)

    # Add uncorrelated nugget and numerical jitter on the diagonal
    Sigma_rand = (sigma_rand_sq + jitter) * torch.eye(M, dtype=coords.dtype, device=coords.device)

    return Sigma_sys + Sigma_rand


class SpatialWaferMap(nn.Module):
    """
    Generates spatially correlated wafer fabrication maps across the 2D MZI mesh layout.
    Simulates:
    1. Spatially correlated directional coupler split-ratio deviations (epsilon1, epsilon2)
    2. Spatially correlated intrinsic waveguide phase offsets (phi_0) from width/thickness variations
    """
    def __init__(self, coords: torch.Tensor, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.total_mzis = coords.shape[0]
        self.register_buffer("coords", coords)

        if config.enable_coupler_errors and not config.ideal_mode:
            if config.enable_spatial_correlation:
                # Build spatial covariance matrix for coupler error epsilon
                eps_var = config.coupler_error_std ** 2
                Sigma_eps = build_spatial_covariance_matrix(
                    coords=coords,
                    correlation_length_x=config.correlation_length_x,
                    correlation_length_y=config.correlation_length_y,
                    total_variance=eps_var,
                    systematic_ratio=config.systematic_var_ratio
                )
                # Cholesky decomposition: Sigma = L @ L^T
                L_eps = torch.linalg.cholesky(Sigma_eps)
                self.register_buffer("L_eps", L_eps)

                # Build spatial covariance for intrinsic phase variations
                phase_var = config.intrinsic_phase_std ** 2
                Sigma_phase = build_spatial_covariance_matrix(
                    coords=coords,
                    correlation_length_x=config.correlation_length_x,
                    correlation_length_y=config.correlation_length_y,
                    total_variance=phase_var,
                    systematic_ratio=config.systematic_var_ratio
                )
                L_phase = torch.linalg.cholesky(Sigma_phase)
                self.register_buffer("L_phase", L_phase)

                # Sample correlated wafer map using configured seed
                gen = torch.Generator(device=coords.device)
                gen.manual_seed(config.wafer_seed)

                # Coupler deviations
                z1 = torch.randn(self.total_mzis, generator=gen, device=coords.device, dtype=coords.dtype)
                z2 = torch.randn(self.total_mzis, generator=gen, device=coords.device, dtype=coords.dtype)
                eps1 = L_eps @ z1
                eps2 = L_eps @ z2

                # Intrinsic phase bias (radians)
                z_phase = torch.randn(self.total_mzis, generator=gen, device=coords.device, dtype=coords.dtype)
                phi_intrinsic = L_phase @ z_phase

            else:
                # Uncorrelated i.i.d. Gaussian sampling
                gen = torch.Generator(device=coords.device)
                gen.manual_seed(config.wafer_seed)
                eps1 = torch.randn(self.total_mzis, generator=gen, device=coords.device) * config.coupler_error_std
                eps2 = torch.randn(self.total_mzis, generator=gen, device=coords.device) * config.coupler_error_std
                phi_intrinsic = torch.randn(self.total_mzis, generator=gen, device=coords.device) * config.intrinsic_phase_std

            # Physically bound deviations
            eps1 = torch.clamp(eps1, -0.25, 0.25)
            eps2 = torch.clamp(eps2, -0.25, 0.25)
        else:
            eps1 = torch.zeros(self.total_mzis, device=coords.device, dtype=coords.dtype)
            eps2 = torch.zeros(self.total_mzis, device=coords.device, dtype=coords.dtype)
            phi_intrinsic = torch.zeros(self.total_mzis, device=coords.device, dtype=coords.dtype)

        self.register_buffer("coupler_eps1", eps1)
        self.register_buffer("coupler_eps2", eps2)
        self.register_buffer("phi_intrinsic", phi_intrinsic)

    def get_coupler_errors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the static wafer-correlated coupler errors (eps1, eps2)."""
        return self.coupler_eps1, self.coupler_eps2

    def get_intrinsic_phase_bias(self) -> torch.Tensor:
        """Returns the static lithographic intrinsic phase bias (phi_intrinsic)."""
        return self.phi_intrinsic
