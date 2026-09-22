"""
Full Jones Vector Polarization Tracking and Polarization-Dependent Loss (PDL) Model.
Calibrated for 220 nm Silicon-on-Insulator (SOI) waveguides at 1550 nm.

Governing Physics:
1. Dual-Polarization State Representation:
   E_p = [E_p_TE, E_p_TM]^T in C^2 for each spatial waveguide mode p.

2. Strong Geometric Modal Birefringence in 450 nm x 220 nm Strip Core:
   n_eff_TE = 2.445,  n_eff_TM = 1.785  (Delta n_eff = 0.660)
   Propagation phase velocity beta = (2 * pi / lambda) * n_eff

3. Polarization Mode Cross-Coupling at Waveguide Bends:
   Due to vertical waveguide asymmetry, sidewall roughness, and bend curvature:
   [E_TE']   [ cos(theta)  -sin(theta) ] [ E_TE ]
   [E_TM'] = [ sin(theta)   cos(theta) ] [ E_TM ]
   where theta is the bend coupling angle (typically 0.01 - 0.03 rad).

4. Polarization Dependent Loss (PDL):
   TM mode has weaker vertical optical confinement, incurring ~3x higher
   crossing and bend scattering losses than TE mode.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.config import PhotonicConfig, PhysicalConstants


class JonesPolarizationModel(nn.Module):
    """
    PyTorch differentiable module for tracking full 2-polarization Jones vectors
    through the Clements photonic mesh with modal birefringence, bend cross-coupling,
    and polarization-dependent attenuation.
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.n_modes = config.n_modes
        self.enabled = config.enable_polarization and not config.ideal_mode

        self.n_eff_te = PhysicalConstants.n_eff
        self.n_eff_tm = config.n_eff_tm
        self.coupling_rad = config.polarization_coupling_rad
        self.wavelength = config.wavelength

        # Build 2x2 rotation coupling matrix for bend polarization mixing
        theta = self.coupling_rad
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # J_rot = [[cos, -sin], [sin, cos]]
        J_rot = torch.tensor([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ], dtype=torch.float32)
        self.register_buffer("J_rot", J_rot)

        # Differential attenuation per crossing (TE vs TM)
        # TE loss = config.crossing_loss_db (0.025 dB)
        # TM loss = config.tm_crossing_loss_db (0.08 dB)
        te_loss_lin = 10.0 ** (-config.crossing_loss_db / 20.0)
        tm_loss_lin = 10.0 ** (-config.tm_crossing_loss_db / 20.0)
        pdl_vector = torch.tensor([te_loss_lin, tm_loss_lin], dtype=torch.float32)
        self.register_buffer("crossing_pdl", pdl_vector)

    def propagate_polarization(
        self,
        E_jones: torch.Tensor,
        distance: float
    ) -> torch.Tensor:
        """
        Applies modal birefringence phase accumulation over distance L.

        Args:
            E_jones: Tensor of shape (..., n_modes, 2) with complex field components [TE, TM].
            distance: Propagation distance in meters.

        Returns:
            Field tensor of shape (..., n_modes, 2) after birefringent phase drift.
        """
        if not self.enabled:
            return E_jones

        k0 = 2.0 * math.pi / self.wavelength
        phi_te = k0 * self.n_eff_te * distance
        phi_tm = k0 * self.n_eff_tm * distance

        phase_te = torch.polar(torch.tensor(1.0, device=E_jones.device), torch.tensor(phi_te, device=E_jones.device))
        phase_tm = torch.polar(torch.tensor(1.0, device=E_jones.device), torch.tensor(phi_tm, device=E_jones.device))
        biref_phasor = torch.stack([phase_te, phase_tm], dim=-1).to(E_jones.dtype)

        return E_jones * biref_phasor

    def apply_bend_coupling(self, E_jones: torch.Tensor) -> torch.Tensor:
        """
        Applies polarization mode rotation at waveguide bend transitions.

        Args:
            E_jones: Tensor of shape (..., n_modes, 2).

        Returns:
            Rotated Jones field tensor of shape (..., n_modes, 2).
        """
        if not self.enabled:
            return E_jones

        # E_jones has shape (..., 2). Apply 2x2 rotation J_rot on the last dimension
        rot = self.J_rot.to(E_jones.device).to(E_jones.dtype)
        # (..., 2) @ (2, 2)^T = (..., 2)
        return torch.matmul(E_jones, rot.T)

    def apply_crossing_pdl(
        self,
        E_jones: torch.Tensor,
        num_crossings: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies polarization-dependent crossing loss to TE and TM components.

        Args:
            E_jones: Tensor of shape (..., n_modes, 2).
            num_crossings: Tensor of shape (n_modes,) with crossing count per channel.

        Returns:
            PDL-attenuated field tensor.
        """
        if not self.enabled:
            return E_jones

        dev = E_jones.device
        pdl = self.crossing_pdl.to(dev)  # (2,)
        # Attenuation factor = (crossing_pdl) ** num_crossings
        # Shape: (n_modes, 2)
        atten_te = torch.pow(pdl[0], num_crossings.to(dev))
        atten_tm = torch.pow(pdl[1], num_crossings.to(dev))
        atten_matrix = torch.stack([atten_te, atten_tm], dim=-1).to(E_jones.dtype)

        return E_jones * atten_matrix

    def total_optical_power(self, E_jones: torch.Tensor) -> torch.Tensor:
        """
        Calculates total detected optical power: P = |E_TE|^2 + |E_TM|^2.

        Args:
            E_jones: Field tensor of shape (..., n_modes, 2) or scalar (..., n_modes).

        Returns:
            Real optical power tensor of shape (..., n_modes).
        """
        if E_jones.shape[-1] == 2 and E_jones.ndim > 1:
            return torch.sum(torch.abs(E_jones)**2, dim=-1)
        return torch.abs(E_jones)**2
