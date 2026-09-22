"""
Nonlinear Optical Physics for Silicon Photonics Integrated Circuits.
Calibrated for standard 220 nm SOI single-mode waveguides (450 nm x 220 nm) at 1550 nm.

Governing Physics:
1. Two-Photon Absorption (TPA):
   dI/dz = - alpha_lin * I - beta_TPA * I^2
   T_TPA = 1 / (1 + beta_TPA * I * L_eff)

2. Free-Carrier Generation & Absorption (FCA):
   N_c = (beta_TPA * I^2 * tau_c) / (2 * h * nu)
   alpha_FCA = sigma_FCA * N_c
   T_FCA = exp(-alpha_FCA * L)

3. Free-Carrier Dispersion (FCD) & Self-Phase Modulation (SPM):
   delta_phi_SPM = (2 * pi / lambda) * n_2 * I * L_eff
   delta_phi_FCD = - (2 * pi / lambda) * sigma_n * N_c * L

Mathematical references:
- Soref & Bennett, IEEE JQE 23(1), 123-129 (1987)
- Rong et al., Nature 433, 292-294 (2005)
- Jalali & Fathpour, J. Lightwave Technol. 24(12), 4600 (2006)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.config import PhotonicConfig, PhysicalConstants


class SiliconNonlinearOptics(nn.Module):
    """
    PyTorch differentiable module computing intensity-dependent optical nonlinearities:
    TPA field attenuation, FCA losses, Kerr self-phase modulation (SPM), and
    free-carrier dispersion (FCD).
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.enabled = config.enable_nonlinear_optics and not config.ideal_mode

        # Silicon material and waveguide geometry constants
        self.beta_tpa = config.tpa_coefficient       # m/W (0.79e-9 m/W = 0.79 cm/GW)
        self.A_eff = config.effective_mode_area       # m^2 (0.1e-12 m^2 = 0.1 um^2)
        self.tau_c = config.free_carrier_lifetime     # s (1.0e-9 s)
        self.n2 = config.nonlinear_index_n2           # m^2/W (6.0e-18 m^2/W)
        self.wavelength = config.wavelength           # m (1550 nm)

        # Photon energy h * nu
        c = PhysicalConstants.c
        h = PhysicalConstants.h
        self.photon_energy = (h * c) / self.wavelength

        # Drude / Soref-Bennett empirical cross sections at 1550 nm
        # sigma_FCA = 1.45e-21 m^2 (1.45e-17 cm^2)
        self.sigma_fca = 1.45e-21
        # sigma_n = 1.35e-27 m^3 (Drude index shift per carrier density)
        self.sigma_n = 1.35e-27

        # Nominal inter-stage waveguide propagation length per Clements column (meters)
        self.segment_length = config.heater_pitch_x

    def forward(
        self,
        E_field: torch.Tensor,
        segment_length: Optional[float] = None
    ) -> torch.Tensor:
        """
        Applies nonlinear optical propagation corrections to an optical field tensor.

        Args:
            E_field: Complex optical field tensor of shape (..., N) or (..., N, 2)
                     normalized such that sum(|E|^2) represents optical power in Watts.
            segment_length: Waveguide propagation distance in meters (defaults to pitch_x).

        Returns:
            Nonlinearly perturbed complex field tensor of the same shape.
        """
        if not self.enabled:
            return E_field

        L = segment_length if segment_length is not None else self.segment_length

        # Optical power per spatial mode (Watts): P = |E|^2
        power = torch.abs(E_field)**2
        # Optical intensity (W/m^2) inside single-mode core
        intensity = power / self.A_eff

        # 1. Two-Photon Absorption (TPA) field transmission
        # T_TPA = 1 / (1 + beta_TPA * I * L)
        denom_tpa = 1.0 + self.beta_tpa * intensity * L
        field_atten_tpa = 1.0 / torch.sqrt(torch.clamp(denom_tpa, min=1.0))

        # 2. Free-Carrier Absorption (FCA)
        # N_c = (beta_TPA * I^2 * tau_c) / (2 * h * nu)
        N_c = (self.beta_tpa * (intensity**2) * self.tau_c) / (2.0 * self.photon_energy)
        alpha_fca = self.sigma_fca * N_c
        field_atten_fca = torch.exp(-0.5 * alpha_fca * L)

        # 3. Self-Phase Modulation (SPM) and Free-Carrier Dispersion (FCD)
        k0 = 2.0 * math.pi / self.wavelength
        phi_spm = k0 * self.n2 * intensity * L
        phi_fcd = -k0 * self.sigma_n * N_c * L
        phi_total = phi_spm + phi_fcd

        # Composite complex transmission multiplier
        # amp = field_atten_tpa * field_atten_fca
        # phase = exp(i * phi_total)
        net_amplitude = field_atten_tpa * field_atten_fca
        complex_phase = torch.polar(
            torch.ones_like(phi_total),
            phi_total
        )

        transfer_factor = net_amplitude.to(E_field.dtype) * complex_phase.to(E_field.dtype)
        return E_field * transfer_factor
