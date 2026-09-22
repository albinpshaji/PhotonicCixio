"""
Nonlinear Optical Physics for Silicon Photonics Integrated Circuits.
Calibrated for standard 220 nm SOI single-mode waveguides (450 nm x 220 nm) at 1550 nm.

Governing Physics (CW / Steady-State):
1. Two-Photon Absorption (TPA) and Free-Carrier Absorption (FCA):
   dP/dz = - alpha_lin * P - (beta_TPA / A_eff) * P^2 - (sigma_FCA * beta_TPA * tau_c / (2 * h * nu * A_eff^2)) * P^3

2. Self-Phase Modulation (SPM) and Free-Carrier Dispersion (FCD):
   delta_phi(L) = (2 * pi / lambda) * ( (n_2 / A_eff) * <P> - (sigma_n * beta_TPA * tau_c / (2 * h * nu * A_eff^2)) * <P^2> ) * L

Mathematical references:
- Soref & Bennett, IEEE JQE 23(1), 123-129 (1987)
- Rong et al., Nature 433, 292-294 (2005)
- Jalali & Fathpour, J. Lightwave Technol. 24(12), 4600 (2006)
- Blanco-Redondo et al. (2015), arXiv:1505.02517
"""

import math
import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.config import PhotonicConfig, PhysicalConstants

logger = logging.getLogger(__name__)


class SiliconNonlinearOptics(nn.Module):
    """
    PyTorch differentiable module computing power-dependent optical nonlinearities:
    Two-Photon Absorption (TPA), Free-Carrier Absorption (FCA), Kerr Self-Phase Modulation (SPM),
    and Free-Carrier Dispersion (FCD).
    Calibrated in optical power basis (Watts) for CW steady-state propagation.
    """
    def __init__(self, config: PhotonicConfig):
        super().__init__()
        self.config = config
        self.enabled = config.enable_nonlinear_optics and not config.ideal_mode

        # Silicon material and waveguide geometry constants (calibrated 220 nm SOI)
        self.beta_tpa = getattr(config, "tpa_coefficient", 6.5e-12)      # m/W (0.65 cm/GW)
        self.A_eff = getattr(config, "effective_mode_area", 0.055e-12)     # m^2 (0.055 um^2)
        self.tau_c = getattr(config, "free_carrier_lifetime", 2.5e-9)     # s (2.5 ns)
        self.n2 = getattr(config, "nonlinear_index_n2", 4.5e-18)          # m^2/W (4.5e-18 m^2/W)
        self.wavelength = config.wavelength                              # m (1550 nm)

        # Critical power threshold for thermal/FCA runaway warning (Watts)
        self.P_critical = 0.050

        # Photon energy h * nu
        c = PhysicalConstants.c
        h = PhysicalConstants.h
        self.photon_energy = (h * c) / self.wavelength

        # Drude / Soref-Bennett empirical cross sections at 1550 nm
        self.sigma_fca = 1.45e-21  # m^2 (1.45e-17 cm^2)
        self.sigma_n = 1.35e-27    # m^3 (index change per carrier m^-3)

        # Precomputed interaction rates
        self.k_tpa = self.beta_tpa / self.A_eff
        self.k_fca = (self.sigma_fca * self.beta_tpa * self.tau_c) / (2.0 * self.photon_energy * (self.A_eff ** 2))

        k0 = 2.0 * math.pi / self.wavelength
        self.k_spm = k0 * self.n2 / self.A_eff
        self.k_fcd = -k0 * (self.sigma_n * self.beta_tpa * self.tau_c) / (2.0 * self.photon_energy * (self.A_eff ** 2))

        # Nominal waveguide propagation length per Clements column (meters)
        self.segment_length = config.heater_pitch_x

    def forward(
        self,
        E_field: torch.Tensor,
        segment_length: Optional[float] = None,
        include_linear_loss: bool = False
    ) -> torch.Tensor:
        """
        Applies nonlinear optical propagation corrections to an optical field tensor.

        Args:
            E_field: Complex optical field tensor of shape (..., N) or (..., N, 2)
                     calibrated such that sum(|E|^2) represents optical power in Watts.
            segment_length: Waveguide propagation distance in meters (defaults to pitch_x).
            include_linear_loss: If True, integrates linear waveguide loss in ODE.
                                 Default False because linear loss is handled by loss_model.

        Returns:
            Nonlinearly perturbed complex field tensor of the same shape.
        """
        if not self.enabled:
            return E_field

        L = segment_length if segment_length is not None else self.segment_length

        # Optical power per spatial mode (Watts): P = |E|^2
        P_in = torch.real(E_field * torch.conj(E_field))
        if E_field.ndim > 1 and E_field.shape[-1] == 2:
            # Polarization Jones vector: total mode power
            P_mode = torch.sum(P_in, dim=-1, keepdim=True)
        else:
            P_mode = P_in

        # Check power threshold
        max_p = torch.max(P_mode).item()
        if max_p > self.P_critical:
            logger.warning(
                f"Peak optical power {max_p*1e3:.1f} mW exceeds critical threshold "
                f"{self.P_critical*1e3:.1f} mW. Two-photon absorption and free-carrier runaway may occur."
            )

        # 4th-order Runge-Kutta integration of steady-state CW power loss
        # dP/dz = - k_tpa * P^2 - k_fca * P^3
        steps = 4
        dz = L / steps

        alpha_lin = ((self.config.propagation_loss_db_per_cm * 100.0) / 4.343) if include_linear_loss else 0.0

        def dP_dz(P):
            P_nonneg = torch.clamp(P, min=0.0)
            loss = self.k_tpa * (P_nonneg ** 2) + self.k_fca * (P_nonneg ** 3)
            if alpha_lin > 0:
                loss = loss + alpha_lin * P_nonneg
            return -loss

        P = P_mode.clone()
        sum_P = torch.zeros_like(P)
        sum_P2 = torch.zeros_like(P)

        for _ in range(steps):
            k1 = dP_dz(P)
            k2 = dP_dz(torch.clamp(P + 0.5 * dz * k1, min=0.0))
            k3 = dP_dz(torch.clamp(P + 0.5 * dz * k2, min=0.0))
            k4 = dP_dz(torch.clamp(P + dz * k3, min=0.0))
            P_next = torch.clamp(P + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), min=0.0)
            P_mid = 0.5 * (P + P_next)
            sum_P = sum_P + P_mid * dz
            sum_P2 = sum_P2 + (P_mid ** 2) * dz
            P = P_next

        # Amplitude transmission factor: t_amp = sqrt(P(L) / P_0)
        # Clamping min=1e-12 avoids infinite 1/(2*sqrt(x)) gradient singularity at zero power in autograd
        power_ratio = P / torch.clamp(P_mode, min=1e-15)
        t_amp = torch.sqrt(torch.clamp(power_ratio, min=1e-12, max=1.0)).to(dtype=E_field.dtype)

        # Accumulated nonlinear phase shift: Kerr SPM + Free-Carrier Dispersion (FCD)
        phi_total = self.k_spm * sum_P + self.k_fcd * sum_P2
        complex_phase = torch.exp(1.0j * phi_total).to(dtype=E_field.dtype)

        return E_field * t_amp * complex_phase
