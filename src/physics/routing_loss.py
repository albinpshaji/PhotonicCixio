"""
Physical Waveguide Routing, Crossing Loss, and Path-Length Dispersion Model.
Simulates physical 2D planar Clements layout routing constraints:
1. Waveguide crossing insertion losses (~0.025 dB/crossing) and inter-channel crosstalk (~ -40 dB)
2. Mode-dependent physical propagation path lengths, attenuation (dB/cm), and phase skew
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from src.config import PhotonicConfig, PhysicalConstants


class ClementsPhysicalRouting(nn.Module):
    """
    Simulates planar waveguide routing geometry between Clements mesh columns.
    Applies:
    - Waveguide crossing attenuation and coherent optical crosstalk
    - Route-dependent waveguide propagation loss and propagation phase delay
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.n_modes = config.n_modes
        self.enabled = config.enable_physical_routing and not config.ideal_mode

        # Physical constants
        self.n_eff = PhysicalConstants.n_eff
        self.lambda_0 = config.wavelength
        self.crossing_loss_db = config.crossing_loss_db
        self.crossing_xtalk_db = config.crossing_crosstalk_db
        self.prop_loss_per_cm = config.propagation_loss_db_per_cm

        # Precompute crossings per inter-layer routing stage
        self._build_crossing_map()

        # Precompute channel physical route lengths (meters)
        self._build_propagation_lengths()

    def _build_crossing_map(self):
        """
        Calculates waveguide crossing pairs for inter-column routing.
        In planar Clements routing between alternating layers:
        - Transition from even column to odd column crosses adjacent interior modes.
        - Crossing pairs: (1, 2), (3, 4), ..., (N-3, N-2)
        """
        crossing_pairs = []
        for k in range(1, self.n_modes - 1, 2):
            crossing_pairs.append((k, k + 1))

        if crossing_pairs:
            p_idx = torch.tensor([p for p, q in crossing_pairs], dtype=torch.long)
            q_idx = torch.tensor([q for p, q in crossing_pairs], dtype=torch.long)
        else:
            p_idx = torch.zeros(0, dtype=torch.long)
            q_idx = torch.zeros(0, dtype=torch.long)

        self.register_buffer("crossing_p", p_idx)
        self.register_buffer("crossing_q", q_idx)

        # Cumulative crossing count per channel across all layers
        counts = torch.zeros(self.n_modes, dtype=torch.float32)
        num_crossing_layers = self.n_modes // 2
        for k in range(1, self.n_modes - 1, 2):
            counts[k] += num_crossing_layers
            counts[k + 1] += num_crossing_layers
        self.register_buffer("crossing_counts", counts)

        # Crossing field transmission and crosstalk amplitude
        if self.enabled and len(crossing_pairs) > 0:
            a_cross = 10.0 ** (-self.crossing_loss_db / 20.0)
            eta_power = 10.0 ** (self.crossing_xtalk_db / 10.0)
            t_bar = a_cross * math.sqrt(max(1.0 - eta_power, 0.0))
            t_cross = a_cross * math.sqrt(eta_power)
        else:
            t_bar = 1.0
            t_cross = 0.0

        self.register_buffer("t_bar", torch.tensor(t_bar, dtype=torch.float32))
        self.register_buffer("t_cross", torch.tensor(t_cross, dtype=torch.float32))

    def _build_propagation_lengths(self):
        """
        Computes physical waveguide route lengths (meters) for each mode channel.
        Outer modes have slight excess path length to maintain equidistant wavefronts.
        """
        # Base column pitch L_col approx 150 um
        l_col = self.config.heater_pitch_x + 30.0e-6
        total_mesh_length = self.n_modes * l_col  # e.g., 8 * 150 um = 1.2 mm

        lengths = []
        for i in range(self.n_modes):
            # Center modes have minimal routing bends; outer modes have small s-bend excess length
            bend_excess = abs(i - (self.n_modes - 1) / 2.0) * 15.0e-6
            lengths.append(total_mesh_length + bend_excess)

        route_lengths = torch.tensor(lengths, dtype=torch.float32)  # (N,) in meters
        self.register_buffer("route_lengths", route_lengths)

        # Propagation loss in dB: prop_loss_per_cm * (L * 100)
        loss_db = self.prop_loss_per_cm * (route_lengths * 100.0)
        route_attenuations = torch.pow(10.0, -loss_db / 20.0)
        self.register_buffer("route_attenuations", route_attenuations)

    def apply_inter_stage_crossings(self, field: torch.Tensor, col: int) -> torch.Tensor:
        """
        Applies waveguide crossing loss and optical crosstalk between columns col and col+1.
        Active only between even and odd transitions.

        Args:
            field: Complex optical field of shape (..., N).
            col: Current column index.

        Returns:
            Perturbed optical field of shape (..., N).
        """
        if not self.enabled or len(self.crossing_p) == 0:
            return field

        # Crossings occur between even and odd columns
        if col % 2 != 0:
            return field

        p = self.crossing_p.to(field.device)
        q = self.crossing_q.to(field.device)

        t_b = self.t_bar.to(device=field.device, dtype=field.dtype)
        t_c = (1.0j * self.t_cross).to(device=field.device, dtype=field.dtype)

        Ep = field[..., p]
        Eq = field[..., q]

        # Crossing 2x2 mixing: Ep' = t_bar*Ep + i*t_cross*Eq
        Ep_new = t_b * Ep + t_c * Eq
        Eq_new = t_c * Ep + t_b * Eq

        field_out = field.clone()
        field_out[..., p] = Ep_new
        field_out[..., q] = Eq_new
        return field_out

    def apply_channel_propagation(
        self,
        field: torch.Tensor,
        wavelength: Optional[float] = None
    ) -> torch.Tensor:
        """
        Applies route-dependent physical waveguide propagation loss and phase delay.

        Args:
            field: Complex field of shape (..., N).
            wavelength: Operating wavelength in meters.

        Returns:
            Field of shape (..., N) after chip propagation.
        """
        if not self.enabled:
            return field

        lam = wavelength if wavelength is not None else self.lambda_0
        beta = 2.0 * math.pi * self.n_eff / lam

        # Phase delay: exp(i * beta * L)
        phase_delay = beta * self.route_lengths.to(device=field.device)
        phasor = torch.exp(1.0j * phase_delay).to(dtype=field.dtype)

        # Attenuation
        atten = self.route_attenuations.to(device=field.device, dtype=field.dtype)

        return field * atten * phasor
