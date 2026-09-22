"""
Waveguide Bend Radiation Loss and Mode-Mismatch Scattering Physics.
Calibrated for standard Silicon-on-Insulator (SOI) 220 nm strip waveguides.

Governing Physics:
1. Conformal Mapping / Marcatili Radiation Loss per circular bend:
   alpha_bend(R) = C1 * exp(-C2 * R)
   where R is the local bend radius (meters).
   C1 = 0.12 dB, C2 = 0.45e6 m^-1 (0.45 um^-1) at 1550 nm.

2. S-Bend Routing Geometry in Clements Planar Mesh:
   Lateral waveguide displacement dy over inter-stage pitch dx:
   R_min approx (dx^2 + dy^2) / (4 * dy)
   Total arc length and radiation attenuation are computed per spatial channel.

3. Waveguide Width Transition Mode-Mismatch:
   Scattering loss at interfaces between straight routing waveguides (450 nm)
   and coupler access / interaction sections (350 nm - 500 nm).
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.config import PhotonicConfig


def compute_sbend_radius_and_loss(
    dx: float,
    dy: float,
    C1: float = 0.12,
    C2: float = 0.45e6,
    min_bend_radius: float = 5.0e-6
) -> Tuple[float, float]:
    """
    Computes effective minimum bend radius R and total radiation loss (dB) for an S-bend.

    Args:
        dx: Horizontal longitudinal distance (m).
        dy: Vertical lateral offset (m).
        C1: Marcuse bend radiation scaling constant (dB).
        C2: Marcuse bend decay parameter (1/m).
        min_bend_radius: Physical minimum bend radius threshold (m).

    Returns:
        R: Bend radius (m).
        loss_db: Total S-bend radiation loss (dB).
    """
    if abs(dy) < 1e-9:
        return float("inf"), 0.0

    # Geometric S-bend radius: R = (dx^2 + dy^2) / (4 * |dy|)
    R = (dx**2 + dy**2) / (4.0 * abs(dy))
    R = max(R, min_bend_radius)

    # Arc angle for each half of the S-bend: sin(theta) = dx / (2 * R)
    sin_theta = min(dx / (2.0 * R), 1.0)
    theta = math.asin(sin_theta)

    # Radiation loss for two arcs of angle theta:
    # A 90 deg (pi/2 rad) bend produces C1 * exp(-C2 * R) dB
    loss_90deg = C1 * math.exp(-C2 * R)
    loss_sbend_db = 2.0 * (theta / (0.5 * math.pi)) * loss_90deg
    return R, loss_sbend_db


class WaveguideBendLossModel(nn.Module):
    """
    Differentiable PyTorch module computing mode-dependent bend radiation loss
    and mode-mismatch scattering loss across all routing layers of a Clements mesh.
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.n_modes = config.n_modes
        self.enabled = config.enable_bend_loss and not config.ideal_mode

        # Precompute per-mode cumulative bend and transition loss across all N stages
        mode_bend_loss_db = self._compute_cumulative_mesh_bend_loss()
        self.register_buffer("mode_bend_loss_db", mode_bend_loss_db)

        # Field transmission factors: 10^(-loss_db / 20)
        bend_transmission = torch.pow(10.0, -mode_bend_loss_db / 20.0)
        self.register_buffer("bend_transmission", bend_transmission)

    def _compute_cumulative_mesh_bend_loss(self) -> torch.Tensor:
        """
        Calculates cumulative bend radiation + transition scattering loss (dB)
        for each waveguide channel traversing the Clements mesh.
        """
        losses = torch.zeros(self.n_modes, dtype=torch.float32)
        if not self.enabled:
            return losses

        dx = self.config.heater_pitch_x
        dy_base = self.config.heater_pitch_y
        C1 = self.config.bend_loss_coefficient_C1
        C2 = self.config.bend_loss_coefficient_C2
        R_min = self.config.min_bend_radius
        trans_loss = self.config.transition_loss_db

        # In a Clements mesh of depth N (n_modes), waveguides alternate pairing:
        # Even stages pair (2k, 2k+1); odd stages pair (2k+1, 2k+2).
        # Modes at the boundaries (0 and N-1) route straight in alternate stages,
        # while interior modes transition up and down by dy_base.
        for layer in range(self.config.num_layers):
            if layer % 2 == 0:
                # Even layer: pairs (0,1), (2,3), ...
                for k in range(self.n_modes // 2):
                    p, q = 2 * k, 2 * k + 1
                    # Inner bend between straight tracks and coupler arms
                    _, loss_p = compute_sbend_radius_and_loss(dx, 0.5 * dy_base, C1, C2, R_min)
                    _, loss_q = compute_sbend_radius_and_loss(dx, 0.5 * dy_base, C1, C2, R_min)
                    # Mode mismatch at coupler entry & exit (2 transitions)
                    losses[p] += loss_p + 2.0 * trans_loss
                    losses[q] += loss_q + 2.0 * trans_loss
            else:
                # Odd layer: pairs (1,2), (3,4), ...
                # Mode 0 and mode N-1 route without coupler in this stage
                losses[0] += 0.5 * trans_loss
                losses[-1] += 0.5 * trans_loss
                for k in range((self.n_modes - 2) // 2):
                    p, q = 2 * k + 1, 2 * k + 2
                    _, loss_p = compute_sbend_radius_and_loss(dx, 0.5 * dy_base, C1, C2, R_min)
                    _, loss_q = compute_sbend_radius_and_loss(dx, 0.5 * dy_base, C1, C2, R_min)
                    losses[p] += loss_p + 2.0 * trans_loss
                    losses[q] += loss_q + 2.0 * trans_loss

        return losses

    def forward(self, E_field: torch.Tensor) -> torch.Tensor:
        """
        Applies bend radiation and transition loss attenuation to the optical field.

        Args:
            E_field: Optical field tensor of shape (..., n_modes) or (..., n_modes, 2).

        Returns:
            Field tensor attenuated by bend radiation and mode-mismatch scattering.
        """
        if not self.enabled:
            return E_field

        trans = self.bend_transmission.to(E_field.device).to(E_field.dtype)
        if E_field.ndim > 1 and E_field.shape[-1] == 2 and E_field.shape[-2] == self.n_modes:
            trans = trans.unsqueeze(-1)

        return E_field * trans
